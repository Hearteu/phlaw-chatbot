# chatbot/views.py
import logging
from functools import lru_cache

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .chat_engine import chat_with_law_bot
from .serializers import ChatRequestSerializer

logger = logging.getLogger(__name__)

# Process-level LLM cache (persists within the same Django process)
_LLM_INSTANCE = None
_LLM_LOADED = False

def get_cached_llm():
    """Get cached LLM instance using process-level caching"""
    global _LLM_INSTANCE, _LLM_LOADED
    
    if _LLM_INSTANCE is None and not _LLM_LOADED:
        try:
            from .generator import _ensure_llm
            _LLM_INSTANCE = _ensure_llm()
            _LLM_LOADED = True
            logger.info("✅ LLM model cached successfully in process memory")
        except Exception as e:
            logger.error(f"❌ Failed to cache LLM model: {e}")
            _LLM_INSTANCE = None
    else:
        logger.info("🔄 LLM model loaded from process memory cache")
    
    return _LLM_INSTANCE

def clear_llm_cache():
    """Clear the process-level LLM cache."""
    global _LLM_INSTANCE, _LLM_LOADED
    _LLM_INSTANCE = None
    _LLM_LOADED = False
    logger.info("🧹 Cleared LLM cache.")


class ChatView(APIView):
    # Disable auth/CSRF for simple local testing. Remove these in prod.
    authentication_classes: list = []
    permission_classes: list = []

    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({"error": "Missing or invalid 'query'."},
                            status=status.HTTP_400_BAD_REQUEST)

        query = serializer.validated_data["query"]
        history = serializer.validated_data.get("history") or []
        
        # Clear LLM cache to ensure new model is loaded
        clear_llm_cache()
        
        # Pre-load LLM model to cache it
        try:
            get_cached_llm()
        except Exception as e:
            logger.warning(f"LLM pre-loading failed: {e}")
        
        try:
            # Call chat_with_law_bot directly without timeout
            answer = chat_with_law_bot(query, history=history)
            
            # Ensure string response
            if not isinstance(answer, str):
                answer = str(answer)
            return Response({"response": answer}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.exception("chat failed: %s", e)
            # Match your existing contract: generic 500 with error field
            return Response({"error": "Internal server error."},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
