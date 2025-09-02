# generator.py — Enhanced Law LLM response generator with optimized performance
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from llama_cpp import Llama

# =============================================================================
# PERFORMANCE OPTIMIZATION CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "law-chat.Q4_K_M.gguf")

# Performance-optimized LLM configuration for Q4_K_M model
LLM_CONFIG = {
    "model_path": MODEL_PATH,
    "n_ctx": 4096,                    # Increased context for legal documents
    "n_gpu_layers": -1,               # Use all available GPU layers
    "n_threads": 8,                   # Optimized thread count
    "n_batch": 256,                   # Optimized batch size
    "use_mmap": True,                 # Use memory mapping
    "use_mlock": False,               # Disable memory locking for speed
}

# Rev21 Labs API configuration
REV21_BASE_URL = os.getenv("REV21_BASE_URL", "https://ai-tools.rev21labs.com/api/v1")
REV21_ENDPOINT_PATH = os.getenv("REV21_ENDPOINT_PATH", "/chat/completions")
REV21_API_KEY = os.getenv("REV21_API_KEY", "")
REV21_ENABLED = (os.getenv("REV21_ENABLED", "true").lower() not in {"0", "false", "no"}) and bool(REV21_API_KEY)

# Global LLM instance for caching
_LLM_INSTANCE = None
_LLM_LOADED = False

def _ensure_llm() -> Llama:
    """Get or create the LLM instance with fallback configurations"""
    global _LLM_INSTANCE, _LLM_LOADED
    
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE
    
    try:
        print("🚀 Loading optimized LLM model...")
        _LLM_INSTANCE = Llama(**LLM_CONFIG)
        print("✅ LLM model loaded successfully with performance optimizations")
        _LLM_LOADED = True
        return _LLM_INSTANCE
        
    except Exception as e:
        print(f"❌ Failed to load LLM model with optimized config: {e}")
        print("🔄 Trying basic configuration...")
        
        try:
            # Fallback to basic configuration
            _LLM_INSTANCE = Llama(
                model_path=MODEL_PATH,
                n_ctx=4096,
                n_gpu_layers=-1,
                n_threads=6,
                use_mmap=True,
                n_batch=128,
            )
            print("✅ LLM model loaded with basic configuration")
            _LLM_LOADED = True
            return _LLM_INSTANCE
            
        except Exception as e2:
            print(f"❌ Basic configuration failed: {e2}")
            print("🔄 Trying minimal configuration...")
            
            try:
                # Ultra-conservative fallback
                _LLM_INSTANCE = Llama(
                    model_path=MODEL_PATH,
                    n_ctx=2048,
                    n_threads=4,
                    use_mmap=True,
                )
                print("✅ LLM model loaded with minimal configuration")
                _LLM_LOADED = True
                return _LLM_INSTANCE
                
            except Exception as e3:
                print(f"❌ All LLM configurations failed: {e3}")
                raise RuntimeError("Failed to load LLM model with any configuration")

def _call_rev21_prompt(prompt: str) -> Optional[str]:
    """Call Rev21 Labs API with prompt"""
    if not REV21_ENABLED:
        return None
    
    url = f"{REV21_BASE_URL}{REV21_ENDPOINT_PATH}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": REV21_API_KEY,
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 900,
        "temperature": 0.4,
        "top_p": 0.9,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                return None
            data = resp.json()
            text = _parse_rev21_response(data)
            if isinstance(text, str) and text.strip():
                text = re.sub(r"\s*\[/?INST\]\s*", " ", text)
                return text.strip()
    except Exception:
        return None
    return None

def _call_rev21_messages(messages: List[Dict[str, str]]) -> Optional[str]:
    """Call Rev21 Labs API with message history"""
    if not REV21_ENABLED:
        return None
    
    url = f"{REV21_BASE_URL}{REV21_ENDPOINT_PATH}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": REV21_API_KEY,
    }
    payload = {
        "messages": messages,
        "max_tokens": 900,
        "temperature": 0.4,
        "top_p": 0.9,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                return None
            data = resp.json()
            text = _parse_rev21_response(data)
            if isinstance(text, str) and text.strip():
                text = re.sub(r"\s*\[/?INST\]\s*", " ", text)
                return text.strip()
    except Exception:
        return None
    return None

def _parse_rev21_response(data: Dict[str, Any]) -> Optional[str]:
    """Parse Rev21 Labs API response"""
    try:
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        return None
    except Exception:
        return None

def generate_response(prompt: str) -> str:
    """Generate response using Rev21 Labs API or local LLM fallback"""
    # Try Rev21 Labs API first
    remote = _call_rev21_prompt(prompt)
    if remote:
        return remote
    
    # Fallback to local LLM
    try:
        llm = _ensure_llm()
        response = llm(
            prompt,
            max_tokens=1024,
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.1,
            stop=["User:", "Human:", "Assistant:", "\n\n\n\n"],
            top_k=20,
        )
        
        if response and "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["text"].strip()
        else:
            return "I apologize, but I was unable to generate a response."
            
    except Exception as e:
        print(f"❌ Local LLM generation failed: {e}")
        return f"I apologize, but I encountered an error: {str(e)}"

def generate_response_from_messages(messages: List[Dict[str, str]]) -> str:
    """Generate response from message history using Rev21 Labs API or local LLM fallback"""
    # Try Rev21 Labs API first
    remote = _call_rev21_messages(messages)
    if remote:
        return remote
    
    # Fallback to local LLM
    try:
        llm = _ensure_llm()
        
        # Convert messages to prompt format
        prompt = _messages_to_prompt(messages)
        
        response = llm(
            prompt,
            max_tokens=1024,
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.1,
            stop=["User:", "Human:", "Assistant:", "\n\n\n\n"],
            top_k=20,
        )
        
        if response and "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["text"].strip()
        else:
            return "I apologize, but I was unable to generate a response."
            
    except Exception as e:
        print(f"❌ Local LLM generation failed: {e}")
        return f"I apologize, but I encountered an error: {str(e)}"

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
    """Generate legal response with optimized prompt construction"""
    # Enhanced legal system prompt with case digest support
    if is_case_digest:
        legal_system_prompt = """You are PHLaw-Chatbot's Case Digest Writer. Produce Philippine Supreme Court case digests that are strictly grounded in the retrieved documents. Match the user's requested scope (full digest or specific section only). Never invent facts. If something isn't in the sources, write: "Not stated in sources."

FORMAT (use these exact section labels and order unless the user asks for a subset)
1) Issue
2) Facts
3) Ruling
4) Discussion
5) Citations (when applicable)
6) Legal Terms Breakdown (only when the user asks)

DETAILED RULES

A. ISSUE
- Frame each issue using the "Whether or not …" convention (start with "Whether or not" exactly).
- If multiple issues, enumerate them (1), (2), (3)… each starting with "Whether or not …"
- Base issues ONLY on the retrieved text (issues section, opening statements, or questions resolved). If unclear, infer the narrowest faithful formulation. If still unclear, write: "Not stated in sources."

B. FACTS
- Prioritize substantive facts first (the story that explains what actually happened), then procedural facts (RTC → CA → SC).
- Storytell in a tight narrative (clear, chronological, concise). No bullet dump of dates unless necessary.
- Do not include facts that are not in the sources. If a key fact is ambiguous or contested, note it.

C. RULING
- Go beyond disposition: include doctrines, legal tests, standards, and the Court's reasoning.
- Explain how the Supreme Court resolved each framed issue (map issue → rule/test → application → conclusion).
- Include lower court outcomes when available:
  • RTC: [result + brief reason]
  • CA:  [result + brief reason]
  • SC:  [final holding + key doctrine]
- When available, quote the WHEREFORE/dispositive clause verbatim (put it in quotation marks) and then briefly explain its effect.

D. DISCUSSION (Opinions)
- Identify and summarize any concurring or dissenting opinions. Explain how they differ from the ponencia and any doctrinal implications.
- If none mentioned, state: "No separate opinions noted in the sources."

E. CITATIONS
- If the user asks about general doctrines/cases (not a single G.R. number), list 2–5 leading cases with full citations (Case v. Case, G.R. No. _____, [Month Day, Year], [Division/En Banc if stated]) and a one-line doctrinal takeaway for each.
- In single-case digests, always include the case header inline at the very top of Facts or Issue if present in sources: Case name; G.R. No.; Date; Ponente; (Division/En Banc if present).
- Only cite what appears in the retrieved text; avoid external sources.

F. LEGAL TERMS BREAKDOWN (On Request)
- If the user explicitly asks to define a legal term, provide:
  • Plain-language definition,
  • How the term is applied in this case (if applicable),
  • The controlling rule or source (if present in the retrieved text).
- Keep it concise and practical for students.

G. STYLE & SAFETY
- Be precise, neutral, and exam-ready.
- Use short paragraphs, smart subheadings, and numbering for multiple issues.
- No speculation. No "as an AI" disclaimers. If information is missing, write "Not stated in sources."
- If sources conflict, prefer Supreme Court final holding; briefly note the conflict."""
    else:
        legal_system_prompt = """You are a Philippine legal assistant. Analyze the provided sources and answer questions based on that information. Provide complete, accurate responses with proper citations. If the sources contain relevant information, use it to answer the question. If the sources don't contain relevant information, say 'The sources don't contain information about this topic.'"""
    
    if context:
        # Include context in the prompt
        enhanced_prompt = f"System: {legal_system_prompt}\n\nContext: {context}\n\nUser: {prompt}\n\nAssistant:"
    else:
        enhanced_prompt = f"System: {legal_system_prompt}\n\nUser: {prompt}\n\nAssistant:"
    
    # Try Rev21 Labs API first
    remote = _call_rev21_prompt(enhanced_prompt)
    if remote:
        return remote
    
    # Fallback to local LLM
    try:
        llm = _ensure_llm()
        response = llm(
            enhanced_prompt,
            max_tokens=1024,
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.1,
            stop=["User:", "Human:", "Assistant:", "\n\n\n\n"],
            top_k=20,
        )
        
        if response and "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["text"].strip()
        else:
            return "I apologize, but I was unable to generate a response."
            
    except Exception as e:
        print(f"❌ Local LLM generation failed: {e}")
        return f"I apologize, but I encountered an error: {str(e)}"

def generate_case_digest_response(prompt: str, context: str = "") -> str:
    """Generate case digest response with specialized formatting"""
    return generate_legal_response(prompt, context, is_case_digest=True)

def get_llm_info() -> Dict[str, Any]:
    """Get LLM model information"""
    try:
        llm = _ensure_llm()
        return {
            "model_path": MODEL_PATH,
            "context_length": getattr(llm, "n_ctx", "Unknown"),
            "gpu_layers": getattr(llm, "n_gpu_layers", "Unknown"),
            "threads": getattr(llm, "n_threads", "Unknown"),
            "batch_size": getattr(llm, "n_batch", "Unknown"),
        }
    except Exception as e:
        return {"error": str(e)}
