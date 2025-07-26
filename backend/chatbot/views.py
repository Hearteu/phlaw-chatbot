from django.shortcuts import render
from rest_framework.response import Response
# Create your views here.
from rest_framework.views import APIView

from .chat_engine import chat_with_law_bot


class ChatView(APIView):
    def post(self, request):
        query = request.data.get("query", "")
        if not query:
            return Response({"error": "Missing query"}, status=400)
        answer = chat_with_law_bot(query)
        return Response({"answer": answer})
    

