# chatbot/views.py
import logging
from functools import lru_cache

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .chat_engine import chat_with_law_bot
from .debug_logger import debug_capture
from .model_cache import get_cached_llm, get_memory_stats, reload_llm
from .serializers import ChatRequestSerializer

logger = logging.getLogger(__name__)


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
        
        # Use centralized LLM model cache
        try:
            get_cached_llm()
        except Exception as e:
            logger.warning(f"LLM loading failed: {e}")
        
        try:
            # Use proper logging-based debug capture
            with debug_capture(max_messages=50) as debug_logger:
                # Call chat_with_law_bot directly without timeout
                answer = chat_with_law_bot(query, history=history)
                
                # Ensure string response
                if not isinstance(answer, str):
                    answer = str(answer)
                
                # Get captured debug messages
                debug_messages = debug_logger.get_debug_messages()
                
                # Include debug output in response
                response_data = {"response": answer}
                if debug_messages:
                    response_data["debug_info"] = {
                        "debug_output": debug_messages,
                        "debug_count": len(debug_messages)
                    }
                
                # Check response size to prevent broken pipes
                response_size = len(str(response_data))
                if response_size > 100000:  # 100KB limit
                    logger.warning(f"Large response size: {response_size} bytes")
                    # Truncate debug output if too large
                    if debug_messages and len(debug_messages) > 20:
                        response_data["debug_info"]["debug_output"] = debug_messages[:20]
                        response_data["debug_info"]["debug_count"] = len(debug_messages)
                        response_data["debug_info"]["truncated"] = True
                
                return Response(response_data, status=status.HTTP_200_OK)
                
        except Exception as e:
            logger.exception("chat failed: %s", e)
            # Match your existing contract: generic 500 with error field
            return Response({"error": "Internal server error."},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AdminReloadLLMView(APIView):
    """Admin endpoint to reload LLM model using centralized cache"""
    # Disable auth/CSRF for simple local testing. Remove these in prod.
    authentication_classes: list = []
    permission_classes: list = []

    def post(self, request):
        """Reload the LLM model using centralized cache management"""
        try:
            # Use centralized reload function
            llm_instance = reload_llm()
            
            if llm_instance is not None:
                return Response({
                    "message": "LLM model reloaded successfully",
                    "status": "success"
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    "message": "LLM model reload failed - no instance available",
                    "status": "error"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.exception("LLM reload failed: %s", e)
            return Response({
                "message": f"LLM reload failed: {str(e)}",
                "status": "error"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AdminMemoryStatsView(APIView):
    """Admin endpoint to get memory and model statistics"""
    # Disable auth/CSRF for simple local testing. Remove these in prod.
    authentication_classes: list = []
    permission_classes: list = []

    def get(self, request):
        """Get comprehensive memory and model statistics"""
        try:
            stats = get_memory_stats()
            return Response(stats, status=status.HTTP_200_OK)
        except Exception as e:
            logger.exception("Failed to get memory stats: %s", e)
            return Response({
                "message": f"Failed to get memory stats: {str(e)}",
                "status": "error"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)