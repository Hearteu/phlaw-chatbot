# generator.py — Enhanced Law LLM response generator using TogetherAI API
import re
from typing import Any, Dict, List, Optional

from .togetherai_client import (generate_messages_with_togetherai,
                                generate_with_togetherai,
                                get_togetherai_client)


def _clean_response_text(text: str) -> str:
    """Clean up response text by removing instruction tokens, formatting artifacts, and meta-commentary"""
    if not text:
        return ""
    
    # Remove instruction tokens
    text = text.replace("[/INST]", "").replace("[INST]", "")
    
    # Remove other common formatting artifacts
    text = text.replace("[/SYS]", "").replace("[SYS]", "")
    text = text.replace("[/USER]", "").replace("[USER]", "")
    text = text.replace("[/ASSISTANT]", "").replace("[ASSISTANT]", "")
    
    # Remove meta-commentary phrases that add unnecessary disclaimers
    # Remove patterns like "Based on my general knowledge" at the start
    text = re.sub(r'^(Based on (my )?(general )?knowledge(,)?|I can provide|However,? please note that,?|I may not have (complete )?details?|Note that I)\.?\s*', 
                  '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove trailing disclaimers
    text = re.sub(r'\s+However,? please note that.*$', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\s+Please note that I may not have.*$', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove leading/trailing whitespace and newlines
    text = text.strip()
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text

def _ensure_togetherai_client():
    """Ensure TogetherAI client is available."""
    return get_togetherai_client()

def generate_response(prompt: str, _retry_count: int = 0) -> str:
    """Generate response using TogetherAI API."""
    try:
        togetherai_client = _ensure_togetherai_client()
        if not togetherai_client.is_available:
            raise RuntimeError("TogetherAI client not available")

        result = generate_with_togetherai(
            prompt,
            max_tokens=4096,
            temperature=0.3,
            top_p=0.85,
            stop=["User:", "Human:", "\n\n\n\n"]
        )
        cleaned_text = _clean_response_text(result)
        return cleaned_text
    except Exception as e:
        print(f"❌ Error in TogetherAI generation: {e}")
        return "I apologize, but I encountered a technical error. Please try again later."

def generate_response_from_messages(messages: List[Dict[str, str]], _retry_count: int = 0) -> str:
    """Generate response from message history using TogetherAI API."""
    try:
        togetherai_client = _ensure_togetherai_client()
        if not togetherai_client.is_available:
            raise RuntimeError("TogetherAI client not available")

        result = generate_messages_with_togetherai(
            messages,
            max_tokens=4096,
            temperature=0.3,
            top_p=0.85,
            stop=["User:", "Human:", "\n\n\n\n"]
        )
        cleaned_text = _clean_response_text(result)
        return cleaned_text
    except Exception as e:
        print(f"❌ Error in TogetherAI generation: {e}")
        return "I apologize, but I encountered a technical error. Please try again later."


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
        return result
    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        return f"I apologize, but I encountered an error: {str(e)}"

def generate_case_digest_response(prompt: str, context: str = "") -> str:
    """Generate case digest response with specialized formatting"""
    return generate_legal_response(prompt, context, is_case_digest=True)


def generate_conversational_response(query: str, history: List[Dict[str, str]] = None, 
                                    context: str = "", is_case_digest: bool = False) -> str:
    """Generate conversational response with full context awareness and Philippine Law focus"""
    togetherai_client = _ensure_togetherai_client()
    if not togetherai_client.is_available:
        raise RuntimeError("TogetherAI client not available")
    
    if history is None:
        history = []
    
    # Build conversational messages
    messages = [
        {
            "role": "system",
            "content": """You are a Philippine Law expert. Answer queries directly and concisely.

CRITICAL: Do NOT include meta-commentary or disclaimers. Answer directly:
- NO phrases like "Based on my knowledge", "I can provide", "However, please note"
- NO disclaimers about your knowledge or limitations
- Simply state the answer directly

Provide accurate information about Philippine jurisprudence and legal principles. 
Remember and reference previous parts of the conversation. Always cite relevant Supreme Court cases and legal provisions when relevant."""
        }
    ]
    
    # Add conversation history (keep last 5 exchanges for context)
    for msg in history[-5:]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    
    # Add current query with context
    if context:
        user_message = f"Based on this legal context:\n\n{context}\n\nUser question: {query}"
    else:
        user_message = query
    
    messages.append({"role": "user", "content": user_message})
    
    # Generate response
    try:
        # Use more tokens for case digests (they're longer)
        max_response_tokens = 4096 if is_case_digest else 2048
        response = togetherai_client.generate_response_from_messages(
            messages,
            max_tokens=max_response_tokens,
            temperature=0.3,
            top_p=0.85
        )
        
        return _clean_response_text(response)
    except Exception as e:
        print(f"❌ Error in conversational generation: {e}")
        return "I apologize, but I encountered a technical error. Please try again later."


def get_llm_info() -> Dict[str, Any]:
    """Get TogetherAI model information"""
    try:
        togetherai_client = _ensure_togetherai_client()
        if not togetherai_client.is_available:
            return {"error": "TogetherAI client not available"}
        
        return togetherai_client.get_model_info()
    except Exception as e:
        return {"error": str(e)}
