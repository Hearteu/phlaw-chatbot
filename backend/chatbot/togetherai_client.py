# togetherai_client.py — TogetherAI API client for Llama-3.3-70B-Instruct-Turbo
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class TogetherAIClient:
    """Client for TogetherAI API with Llama-3.3-70B-Instruct-Turbo model"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):
                #  model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"):
                #  model_name: str = "openai/gpt-oss-20b"):
        self.api_key = api_key or os.getenv("TOGETHERAI_API_KEY")
        self.model_name = model_name
        self.base_url = "https://api.together.xyz/v1"
        self.client = None
        self.is_available = False
        
        if self.api_key:
            self._initialize_client()
        else:
            print("❌ TogetherAI API key not found. Please set TOGETHERAI_API_KEY environment variable.")
    
    def _initialize_client(self) -> bool:
        """Initialize TogetherAI client"""
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.is_available = True
            print(f"✅ TogetherAI client initialized with model: {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize TogetherAI client: {e}")
            self.is_available = False
            return False
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using TogetherAI API"""
        if not self.is_available:
            raise RuntimeError("TogetherAI client not available")
        
        try:
            # Convert prompt to messages format
            messages = self._prompt_to_messages(prompt)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 4096),
                temperature=kwargs.get('temperature', 0.3),
                top_p=kwargs.get('top_p', 0.85),
                stop=kwargs.get('stop', ["User:", "Human:", "\n\n\n\n"]),
                stream=kwargs.get('stream', False)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ TogetherAI generation failed: {e}")
            raise
    
    def generate_response_from_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from message history using TogetherAI API"""
        if not self.is_available:
            raise RuntimeError("TogetherAI client not available")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 4096),
                temperature=kwargs.get('temperature', 0.3),
                top_p=kwargs.get('top_p', 0.85),
                stop=kwargs.get('stop', ["User:", "Human:", "\n\n\n\n"]),
                stream=kwargs.get('stream', False)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ TogetherAI generation failed: {e}")
            raise
    
    def _prompt_to_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Convert prompt string to messages format"""
        # Simple heuristic: if prompt contains "System:", "User:" 
        # then split by those markers, otherwise treat as user message
        if "System:" in prompt and "User:" in prompt:
            messages = []
            parts = prompt.split("User:")
            if len(parts) >= 2:
                system_part = parts[0].replace("System:", "").strip()
                if system_part:
                    messages.append({"role": "system", "content": system_part})
                
                user_part = parts[1].strip()
                if user_part:
                    messages.append({"role": "user", "content": user_part})
            return messages
        else:
            # Treat entire prompt as user message
            return [{"role": "user", "content": prompt}]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get TogetherAI model information"""
        if not self.is_available:
            return {"error": "TogetherAI client not available"}
        
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "is_available": self.is_available,
            "type": "togetherai_api",
            "provider": "TogetherAI"
        }
    
    def test_connection(self) -> bool:
        """Test connection to TogetherAI API"""
        if not self.is_available:
            return False
        
        try:
            # Test with a simple request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                max_tokens=10
            )
            print("✅ TogetherAI API connection test successful")
            return True
        except Exception as e:
            print(f"❌ TogetherAI API connection test failed: {e}")
            return False


# Global instance
_togetherai_client = None

def get_togetherai_client() -> TogetherAIClient:
    """Get or create TogetherAI client instance"""
    global _togetherai_client
    if _togetherai_client is None:
        _togetherai_client = TogetherAIClient()
    return _togetherai_client


def generate_with_togetherai(prompt: str, **kwargs) -> str:
    """Generate response with TogetherAI API"""
    client = get_togetherai_client()
    
    if client.is_available:
        try:
            print("🚀 Using TogetherAI API...")
            return client.generate_response(prompt, **kwargs)
        except Exception as e:
            print(f"❌ TogetherAI generation failed: {e}")
            raise
    else:
        raise RuntimeError("TogetherAI client not available")


def generate_messages_with_togetherai(messages: List[Dict[str, str]], **kwargs) -> str:
    """Generate response from messages with TogetherAI API"""
    client = get_togetherai_client()
    
    if client.is_available:
        try:
            print("🚀 Using TogetherAI API...")
            return client.generate_response_from_messages(messages, **kwargs)
        except Exception as e:
            print(f"❌ TogetherAI generation failed: {e}")
            raise
    else:
        raise RuntimeError("TogetherAI client not available")
