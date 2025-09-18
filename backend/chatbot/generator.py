# generator.py — Enhanced Law LLM response generator with centralized model management
import re
from typing import Any, Dict, List, Optional

# Import Docker model client with fallback
from .docker_model_client import (generate_messages_with_fallback,
                                  generate_with_fallback)


def _clean_response_text(text: str) -> str:
    """Clean up response text by removing instruction tokens and formatting artifacts"""
    if not text:
        return ""
    
    # Remove instruction tokens
    text = text.replace("[/INST]", "").replace("[INST]", "")
    
    # Remove other common formatting artifacts
    text = text.replace("[/SYS]", "").replace("[SYS]", "")
    text = text.replace("[/USER]", "").replace("[USER]", "")
    text = text.replace("[/ASSISTANT]", "").replace("[ASSISTANT]", "")
    
    # Remove leading/trailing whitespace and newlines
    text = text.strip()
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text

def _ensure_llm():
    """Docker-only mode: no local LLM."""
    return None

def generate_response(prompt: str, _retry_count: int = 0) -> str:
    """Generate response using Docker model runner with local LLM fallback"""
    try:
        # Use Docker model runner with local LLM fallback
        response = generate_with_fallback(
            prompt,
            max_tokens=2048,
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.1,
            stop=["User:", "Human:", "Assistant:", "\n\n\n\n"]
        )
        
        # Clean the response text
        cleaned_text = _clean_response_text(response)
        return cleaned_text
        
    except Exception as e:
        print(f"❌ Error in LLM generation: {e}")
        return "I apologize, but I encountered a technical error. Please try again later."

def generate_response_from_messages(messages: List[Dict[str, str]], _retry_count: int = 0) -> str:
    """Generate response from message history using Docker model runner with local LLM fallback"""
    try:
        # Use Docker model runner with local LLM fallback
        response = generate_messages_with_fallback(
            messages,
            max_tokens=2048,
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.1,
            stop=["User:", "Human:", "Assistant:", "\n\n\n\n"]
        )
        
        # Clean the response text
        cleaned_text = _clean_response_text(response)
        return cleaned_text
        
    except Exception as e:
        print(f"❌ Error in LLM generation: {e}")
        return "I apologize, but I encountered a technical error. Please try again later."

def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert message history to prompt format"""
    prompt_parts = []
    
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    return "\n\n".join(prompt_parts) + "\n\nAssistant:"

def generate_legal_response(prompt: str, context: str = "", is_case_digest: bool = False) -> str:
    """Generate legal response with optimized prompt construction for simplified two-path logic"""
    
    # For case digests, use the prompt as-is (it already contains the custom format)
    if is_case_digest:
        enhanced_prompt = prompt
    else:
        # For keyword queries, use the system prompt
        legal_system_prompt = """You are a Philippine legal assistant. Analyze the provided sources and answer questions based on that information. 

For keyword queries, present the top 3 most relevant cases in this format and include the case type when available:
"Here are the possible cases:

1. [Case Title] (G.R. No. [number]) — [Case type]
2. [Case Title] (G.R. No. [number]) — [Case type]
3. [Case Title] (G.R. No. [number]) — [Case type]"

If a case type is not available for a case, omit the "— [Case type]" part for that line.

Provide complete, accurate responses with proper citations. If the sources contain relevant information, use it to answer the question. If the sources don't contain relevant information, say 'The sources don't contain information about this topic.'"""
        
        if context:
            # Include context in the prompt
            enhanced_prompt = f"System: {legal_system_prompt}\n\nContext: {context}\n\nUser: {prompt}\n\nAssistant:"
        else:
            enhanced_prompt = f"System: {legal_system_prompt}\n\nUser: {prompt}\n\nAssistant:"
    
    # Use Docker model runner with local LLM fallback
    try:
        response = generate_with_fallback(enhanced_prompt)
        return response.strip() if response else "I apologize, but I was unable to generate a response."
            
    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        return f"I apologize, but I encountered an error: {str(e)}"

def generate_case_digest_response(prompt: str, context: str = "") -> str:
    """Generate case digest response with specialized formatting"""
    return generate_legal_response(prompt, context, is_case_digest=True)

def get_llm_info() -> Dict[str, Any]:
    """Get LLM model information"""
    try:
        llm = _ensure_llm()
        if llm is None:
            return {"error": "LLM not available"}
        
        return {
            "model_path": getattr(llm, "model_path", "Unknown"),
            "context_length": getattr(llm, "n_ctx", "Unknown"),
            "gpu_layers": getattr(llm, "n_gpu_layers", "Unknown"),
            "threads": getattr(llm, "n_threads", "Unknown"),
            "batch_size": getattr(llm, "n_batch", "Unknown"),
        }
    except Exception as e:
        return {"error": str(e)}
