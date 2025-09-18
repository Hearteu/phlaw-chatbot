# docker_model_client.py — Docker model runner client with local LLM fallback
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .model_cache import get_fresh_llm


class DockerModelClient:
    """Client for Docker model runner with local LLM fallback"""
    
    def __init__(self, base_url: str = "http://localhost:8001/engines/llama.cpp/v1", 
                 model_name: str = "ai/llama3.2", api_key: str = "dummy_value"):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
        self.is_available = False
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test if Docker model runner is available"""
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            # Test with a simple request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            self.is_available = True
            print("✅ Docker model runner is available")
            return True
        except Exception as e:
            print(f"⚠️ Docker model runner not available: {e}")
            self.is_available = False
            return False
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Docker model runner"""
        if not self.is_available:
            raise RuntimeError("Docker model runner not available")
        
        try:
            # Convert prompt to messages format
            messages = self._prompt_to_messages(prompt)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 10000),
                temperature=kwargs.get('temperature', 0.3),
                top_p=kwargs.get('top_p', 0.85),
                stop=kwargs.get('stop', ["User:", "Human:", "Assistant:", "\n\n\n\n"]),
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Docker model generation failed: {e}")
            raise
    
    def generate_response_from_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from message history using Docker model runner"""
        if not self.is_available:
            raise RuntimeError("Docker model runner not available")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 4096),
                temperature=kwargs.get('temperature', 0.3),
                top_p=kwargs.get('top_p', 0.85),
                stop=kwargs.get('stop', ["User:", "Human:", "Assistant:", "\n\n\n\n"])
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Docker model generation failed: {e}")
            raise
    
    def _prompt_to_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Convert prompt string to messages format"""
        # Simple heuristic: if prompt contains "System:", "User:", "Assistant:" 
        # then split by those markers, otherwise treat as user message
        if "System:" in prompt and "User:" in prompt:
            messages = []
            parts = prompt.split("User:")
            if len(parts) >= 2:
                system_part = parts[0].replace("System:", "").strip()
                if system_part:
                    messages.append({"role": "system", "content": system_part})
                
                user_part = parts[1].split("Assistant:")[0].strip()
                if user_part:
                    messages.append({"role": "user", "content": user_part})
            return messages
        else:
            # Treat entire prompt as user message
            return [{"role": "user", "content": prompt}]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Docker model information"""
        if not self.is_available:
            return {"error": "Docker model runner not available"}
        
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "is_available": self.is_available,
            "type": "docker_model_runner"
        }


# Global instance
_docker_client = DockerModelClient()

def get_docker_model_client() -> DockerModelClient:
    """Get or create Docker model client instance"""
    global _docker_client
    return _docker_client


def generate_with_fallback(prompt: str, **kwargs) -> str:
    """Generate response with Docker model runner, fallback to local LLM"""
    docker_client = get_docker_model_client()
    
    if docker_client.is_available:
        try:
            print("🐳 Using Docker model runner...")
            return docker_client.generate_response(prompt, **kwargs)
        except Exception as e:
            print(f"❌ Docker model failed: {e}")
            raise
    else:
        raise RuntimeError("Docker model runner not available")


def generate_messages_with_fallback(messages: List[Dict[str, str]], **kwargs) -> str:
    """Generate response from messages with Docker model runner, fallback to local LLM"""
    docker_client = get_docker_model_client()
    
    if docker_client.is_available:
        try:
            print("🐳 Using Docker model runner...")
            return docker_client.generate_response_from_messages(messages, **kwargs)
        except Exception as e:
            print(f"❌ Docker model failed: {e}")
            raise
    else:
        raise RuntimeError("Docker model runner not available")
