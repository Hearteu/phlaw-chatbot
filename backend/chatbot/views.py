from django.shortcuts import render
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework.response import Response
from rest_framework.views import APIView

from .chat_engine import chat_with_law_bot


class ChatView(APIView):
    @swagger_auto_schema(
        operation_description="Ask the PH Law Chatbot a legal question",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['query'],
            properties={
                'query': openapi.Schema(type=openapi.TYPE_STRING, description='The legal question to ask'),
            },
        ),
        responses={
            200: openapi.Response(
                description="Chatbot's answer",
                examples={
                    "application/json": {
                        "answer": "The Constitution of the Philippines states..."
                    }
                }
            ),
            400: "Missing query"
        }
    )
    def post(self, request):
        query = request.data.get("query", "")
        if not query:
            return Response({"error": "Missing query"}, status=400)
        answer = chat_with_law_bot(query)
        return Response({"answer": answer})
