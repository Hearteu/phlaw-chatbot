# chatbot/serializers.py
from rest_framework import serializers


class ChatRequestSerializer(serializers.Serializer):
    query = serializers.CharField(allow_blank=False, trim_whitespace=True)
