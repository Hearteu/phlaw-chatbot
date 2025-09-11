#!/usr/bin/env python3
"""
Test script for Docker model integration with local LLM fallback
"""
import os
import sys

import django

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from chatbot.docker_model_client import (generate_with_fallback,
                                         get_docker_model_client)
from chatbot.generator import (generate_response,
                               generate_response_from_messages)


def test_docker_client():
    """Test Docker model client connection"""
    print("🔍 Testing Docker model client...")
    
    client = get_docker_model_client()
    print(f"✅ Docker client created")
    print(f"📊 Model info: {client.get_model_info()}")
    
    if client.is_available:
        print("✅ Docker model runner is available")
    else:
        print("⚠️ Docker model runner not available, will use local LLM fallback")


def test_generate_with_fallback():
    """Test generation with fallback"""
    print("\n🔍 Testing generation with fallback...")
    
    test_prompt = "System: You are a helpful assistant.\n\nUser: What is 2+2?"
    
    try:
        response = generate_with_fallback(test_prompt, max_tokens=50)
        print(f"✅ Generated response: {response[:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_generator_functions():
    """Test the updated generator functions"""
    print("\n🔍 Testing generator functions...")
    
    # Test generate_response
    try:
        response = generate_response("What is the capital of the Philippines?")
        print(f"✅ generate_response: {response[:100]}...")
    except Exception as e:
        print(f"❌ generate_response error: {e}")
    
    # Test generate_response_from_messages
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 5+5?"}
        ]
        response = generate_response_from_messages(messages)
        print(f"✅ generate_response_from_messages: {response[:100]}...")
    except Exception as e:
        print(f"❌ generate_response_from_messages error: {e}")


def main():
    """Run all tests"""
    print("🚀 Testing Docker Model Integration")
    print("=" * 50)
    
    test_docker_client()
    test_generate_with_fallback()
    test_generator_functions()
    
    print("\n✅ Integration test completed!")
    print("\n📋 Next steps:")
    print("1. Start your Docker model runner: docker run -p 8001:8001 ai/llama3.2")
    print("2. Start Django server: python manage.py runserver")
    print("3. Test the chatbot through the API")


if __name__ == "__main__":
    main()
