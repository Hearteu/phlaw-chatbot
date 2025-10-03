# generator.py — Enhanced Law LLM response generator with centralized model management
import re
from typing import Any, Dict, List, Optional

from .model_cache import get_cached_llm

# Response cache for deterministic generation
_response_cache = {}


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
    """Ensure local GGUF LLM is loaded and available."""
    return get_cached_llm()

def generate_response(prompt: str, _retry_count: int = 0) -> str:
    """Generate response using local llama.cpp GGUF model."""
    try:
        llm = _ensure_llm()
        if llm is None:
            raise RuntimeError("Local LLM not available")

        generation = llm(
            prompt,
            max_tokens=2048,
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.1,
            stop=["User:", "Human:", "\n\n\n\n"],
        )
        response_text = generation.get("choices", [{}])[0].get("text", "") if isinstance(generation, dict) else str(generation)
        cleaned_text = _clean_response_text(response_text)
        return cleaned_text
    except Exception as e:
        print(f"❌ Error in LLM generation: {e}")
        return "I apologize, but I encountered a technical error. Please try again later."

def generate_response_from_messages(messages: List[Dict[str, str]], _retry_count: int = 0) -> str:
    """Generate response from message history using local llama.cpp by converting to a prompt."""
    try:
        prompt = _messages_to_prompt(messages)
        return generate_response(prompt, _retry_count=_retry_count)
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
    
    return "\n\n".join(prompt_parts)

def generate_legal_response(prompt: str, context: str = "", is_case_digest: bool = False) -> str:
    """Generate legal response with optimized prompt construction for simplified two-path logic"""
    
    # Create cache key for deterministic responses
    cache_key = f"{prompt}_{context}_{is_case_digest}"
    
    # Check cache first
    if cache_key in _response_cache:
        print(f"📋 Response cache hit for prompt: {prompt[:50]}...")
        return _response_cache[cache_key]
    
    # For case digests, use the prompt as-is (it already contains the custom format)
    if is_case_digest:
        enhanced_prompt = prompt
    else:
        # For keyword queries, use the system prompt
        legal_system_prompt = """You are a Philippine legal assistant. Analyze the provided sources and answer questions based on that information. 

For keyword queries, present the top 3 most relevant cases in this format and include the case type when available:
"Here are the possible cases:

1. [Case Title] (G.R. No. [number])
2. [Case Title] (G.R. No. [number])
3. [Case Title] (G.R. No. [number])"

Provide complete, accurate responses with proper citations. If the sources contain relevant information, use it to answer the question. If the sources don't contain relevant information, say 'The sources don't contain information about this topic.'"""
        
        if context:
            # Include context in the prompt
            enhanced_prompt = f"System: {legal_system_prompt}\n\nContext: {context}\n\nUser: {prompt}"
        else:
            enhanced_prompt = f"System: {legal_system_prompt}\n\nUser: {prompt}"
    
    # Use local LLM
    try:
        result = generate_response(enhanced_prompt)
        result = result.strip() if result else "I apologize, but I was unable to generate a response."
        _response_cache[cache_key] = result
        print(f"💾 Cached response for prompt: {prompt[:50]}...")
        return result
    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        return f"I apologize, but I encountered an error: {str(e)}"

def generate_case_digest_response(prompt: str, context: str = "") -> str:
    """Generate case digest response with specialized formatting"""
    return generate_legal_response(prompt, context, is_case_digest=True)

def clear_response_cache():
    """Clear the response cache"""
    global _response_cache
    _response_cache.clear()
    print("🧹 Response cache cleared")

def get_response_cache_stats():
    """Get response cache statistics"""
    return {
        "cache_size": len(_response_cache),
        "cache_keys": list(_response_cache.keys())[:10]  # First 10 keys for debugging
    }

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
