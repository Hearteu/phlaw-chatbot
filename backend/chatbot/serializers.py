# chatbot/serializers.py
from rest_framework import serializers


class MessageSerializer(serializers.Serializer):
    """Serializer for individual chat messages in history"""
    role = serializers.ChoiceField(
        choices=['system', 'user', 'assistant'],
        allow_blank=False
    )
    content = serializers.CharField(allow_blank=True, trim_whitespace=True)
    
    def validate_role(self, value):
        """Validate that role is one of the expected values"""
        if value not in ['system', 'user', 'assistant']:
            raise serializers.ValidationError(
                f"Invalid role '{value}'. Must be one of: system, user, assistant"
            )
        return value


class ChatRequestSerializer(serializers.Serializer):
    query = serializers.CharField(allow_blank=False, trim_whitespace=True)
    # Optional conversation history: list of {role: 'system'|'user'|'assistant', content: str}
    history = serializers.ListField(
        child=MessageSerializer(),
        required=False,
        allow_empty=True,
    )


