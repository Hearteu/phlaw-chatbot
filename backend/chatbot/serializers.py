# chatbot/serializers.py
from rest_framework import serializers


class ChatRequestSerializer(serializers.Serializer):
    query = serializers.CharField(allow_blank=False, trim_whitespace=True)
    # Optional conversation history: list of {role: 'system'|'user'|'assistant', content: str}
    history = serializers.ListField(
        child=serializers.DictField(child=serializers.CharField()),
        required=False,
        allow_empty=True,
    )
