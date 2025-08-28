# chatbot/views.py
import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .chat_engine import chat_with_law_bot
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
        try:
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
