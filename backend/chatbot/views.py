# chatbot/views.py
import logging
import time
from functools import lru_cache

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .chat_engine import chat_with_law_bot
from .debug_logger import debug_capture
from .model_cache import clear_llm_cache, get_memory_stats, reload_llm
from .retriever import LegalRetriever
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
        
        # Note: Local GGUF LLM is NOT loaded here - chatbot uses TogetherAI for generation
        # The GGUF model is only used for contextual chunking (lazy-loaded when needed)
        
        try:
            # Use proper logging-based debug capture
            with debug_capture(max_messages=50) as debug_logger:
                # Call chat_with_law_bot directly without timeout
                answer = chat_with_law_bot(query, history=history)
                
                # Handle caching response format
                cached_case_data = None
                if isinstance(answer, dict) and "_cached_case_data" in answer:
                    # Extract the actual response and cached data
                    cached_case_data = answer["_cached_case_data"]
                    answer = answer["content"]
                elif not isinstance(answer, str):
                    answer = str(answer)
                
                # Get captured debug messages
                debug_messages = debug_logger.get_debug_messages()
                
                # Include debug output in response
                response_data = {"response": answer}
                
                # Include cached case data for frontend to manage
                if cached_case_data:
                    response_data["_cached_case_data"] = cached_case_data
                
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
            import traceback
            print("="*80)
            print("❌ CHAT ERROR:")
            print(traceback.format_exc())
            print("="*80)
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




class StreamingChatView(APIView):
    """Streaming chat endpoint with real-time progress updates"""
    authentication_classes: list = []
    permission_classes: list = []

    def post(self, request):
        import json

        from django.http import StreamingHttpResponse
        
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({"error": "Missing or invalid 'query'."},
                            status=status.HTTP_400_BAD_REQUEST)

        query = serializer.validated_data["query"]
        history = serializer.validated_data.get("history") or []
        
        def generate_stream():
            """Generator function for streaming responses with progress updates"""
            try:
                # Import here to avoid circular dependencies
                from .chat_engine import chat_with_law_bot_streaming

                # Send initial metadata
                yield f"data: {json.dumps({'type': 'start', 'query': query})}\n\n"
                
                # Generate streaming response with progress updates
                for chunk in chat_with_law_bot_streaming(query, history=history):
                    if isinstance(chunk, dict):
                        # Progress update
                        yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        # Content chunk (string)
                        yield f"data: {json.dumps({'type': 'content', 'chunk': chunk})}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                logger.exception("Streaming chat failed: %s", e)
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        response = StreamingHttpResponse(
            generate_stream(),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response


class ClearCacheView(APIView):
    """API endpoint to clear LLM response caches"""
    authentication_classes: list = []
    permission_classes: list = []

    def post(self, request):
        """Clear specified caches"""
        cache_type = request.data.get('type', 'all')
        
        try:
            cleared_caches = []
            
            
            if cache_type in ['all', 'retrieval']:
                # Clear retrieval cache for both collections
                for collection in ['jurisprudence']:
                    try:
                        retriever = LegalRetriever(collection=collection)
                        retriever.clear_cache()
                    except Exception as e:
                        logger.warning(f"Could not clear retrieval cache for {collection}: {e}")
                cleared_caches.append('retrieval')
            
            if cache_type in ['all', 'model']:
                clear_llm_cache()
                cleared_caches.append('model')
            
            return Response({
                "message": f"Successfully cleared caches: {', '.join(cleared_caches)}",
                "cleared_caches": cleared_caches,
                "status": "success"
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception("Failed to clear caches: %s", e)
            return Response({
                "message": f"Failed to clear caches: {str(e)}",
                "status": "error"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)