# generator.py — Enhanced Law LLM response generator with centralized model management
import re
from typing import Any, Dict, List, Optional

# Import centralized model cache
from .model_cache import get_cached_llm


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
    """Get or create the LLM instance using centralized cache"""
    return get_cached_llm()

def generate_response(prompt: str, _retry_count: int = 0) -> str:
    """Generate response using local LLM with single retry guard"""
    try:
        llm = _ensure_llm()
        
        # Use more conservative generation parameters to avoid CUDA tensor issues
        response = llm(
            prompt,
            max_tokens=2048,  # Increased for better responses
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.1,
            stop=["User:", "Human:", "Assistant:", "\n\n\n\n"],
            top_k=20,
            stream=False,  # Disable streaming to avoid memory issues
            echo=False,    # Don't echo the prompt
            tfs_z=1.0,     # Add tensor fusion parameter for stability
        )
        
        if response and "choices" in response and len(response["choices"]) > 0:
            print("LLM: law-chat (prompt)")
            raw_text = response["choices"][0]["text"]
            cleaned_text = _clean_response_text(raw_text)
            return cleaned_text
        else:
            return "I apologize, but I was unable to generate a response."
            
    except MemoryError as e:
        print(f"❌ Memory error in LLM generation: {e}")
        return "I apologize, but I'm experiencing memory issues. Please try a simpler question."
    except Exception as e:
        print(f"❌ Local LLM generation failed: {e}")
        # Check if this is a persistent error that shouldn't be retried
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ["access violation", "sampler", "ggml_assert", "tensor", "cuda"]):
            print("🔄 Detected persistent CUDA/tensor error, using fallback...")
            return "I apologize, but I'm experiencing technical difficulties. Please try a simpler question or try again later."
        
        # Single retry with guard to prevent infinite recursion
        if _retry_count < 1:
            print("🔄 Retrying generation...")
            return generate_response(prompt, _retry_count + 1)
        else:
            print("❌ Retry limit reached")
            return "I apologize, but I encountered a technical error. Please try again later."

def generate_response_from_messages(messages: List[Dict[str, str]], _retry_count: int = 0) -> str:
    """Generate response from message history using local LLM with single retry guard"""
    try:
        llm = _ensure_llm()
        
        # Convert messages to prompt format
        prompt = _messages_to_prompt(messages)
        
        # Use more conservative generation parameters to avoid CUDA tensor issues
        response = llm(
            prompt,
            max_tokens=2048,  # Increased for better responses
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.1,
            stop=["User:", "Human:", "Assistant:", "\n\n\n\n"],
            top_k=20,
            stream=False,  # Disable streaming to avoid memory issues
            echo=False,    # Don't echo the prompt
            tfs_z=1.0,     # Add tensor fusion parameter for stability
        )
        
        if response and "choices" in response and len(response["choices"]) > 0:
            print("LLM: law-chat (messages)")
            raw_text = response["choices"][0]["text"]
            cleaned_text = _clean_response_text(raw_text)
            return cleaned_text
        else:
            return "I apologize, but I was unable to generate a response."
            
    except MemoryError as e:
        print(f"❌ Memory error in LLM generation: {e}")
        return "I apologize, but I'm experiencing memory issues. Please try a simpler question."
    except Exception as e:
        print(f"❌ Local LLM generation failed: {e}")
        
        # Check if this is a persistent error that shouldn't be retried
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ["access violation", "sampler", "ggml_assert", "tensor", "cuda"]):
            print("🔄 Detected persistent CUDA/tensor error, using fallback...")
            return "I apologize, but I'm experiencing technical difficulties. Please try a simpler question or try again later."
        
        # Single retry with guard to prevent infinite recursion
        if _retry_count < 1:
            print("🔄 Retrying generation...")
            return generate_response_from_messages(messages, _retry_count + 1)
        else:
            print("❌ Retry limit reached")
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
- Output format: bullet list, ONE sentence per bullet, no sub-bullets.
- Prioritize substantive facts first (the story of what happened), then procedural facts (RTC → CA → SC).
- Keep each bullet concise and grounded in the sources only. If unclear, omit or say "Not stated in sources."

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
    
    # Use local LLM
    try:
        llm = _ensure_llm()
        response = llm(
            enhanced_prompt,
            max_tokens=2000,
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
