# chat_engine.py — Simplified Law LLM chat engine with chunking support
import concurrent.futures
import re
from typing import Any, Dict, List, Optional, Tuple

# Import evaluation metrics
try:
    from .evaluation_metrics import (AutomatedContentScoring,
                                     ContentRelevanceMetrics,
                                     EvaluationTracker, LegalAccuracyMetrics)
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("⚠️ Evaluation metrics not available")

# Import TogetherAI client
try:
    from .togetherai_client import generate_messages_with_togetherai
    TOGETHERAI_AVAILABLE = True
except ImportError:
    TOGETHERAI_AVAILABLE = False
    print("⚠️ TogetherAI client not available")

# Import web search functionality
try:
    from .web_search import get_web_searcher
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    print("⚠️ Web search not available")


def _is_follow_up_question(query: str) -> bool:
    """Check if query is asking for more details about a previously discussed case"""
    q = query.strip().lower()
    
    follow_up_patterns = [
        r"delve\s+deeper",
        r"tell\s+me\s+more",
        r"explain\s+more",
        r"give\s+me\s+more\s+details",
        r"what\s+were\s+the\s+(facts|issues|ruling)",
        r"explain\s+the\s+(facts|issues|ruling|principles)",
        r"more\s+about\s+this\s+case",
        r"expand\s+on",
        r"elaborate\s+on",
        r"can\s+you\s+explain\s+(further|more)",
    ]
    
    for pattern in follow_up_patterns:
        if re.search(pattern, q):
            return True
    
    return False

def _is_case_digest_request(query: str) -> bool:
    """Check if query is asking for a case digest of a specific case"""
    q = query.strip().lower()
    
    case_digest_patterns = [
        r"digest\s+the\s+case\s+of",
        r"case\s+digest\s+of",
        r"summarize\s+the\s+case\s+of",
        r"tell\s+me\s+about\s+the\s+case\s+of",
        r"details\s+about\s+the\s+case\s+of",
        r"explain\s+the\s+case\s+of",
        r"what\s+is\s+the\s+case\s+of",
        r"case\s+of\s+[a-zA-Z\s]+\s+vs\.?\s+[a-zA-Z\s]+",
        r"[a-zA-Z\s]+\s+vs\.?\s+[a-zA-Z\s]+",
        r"g\.r\.\s*no\.?\s*\d+",
    ]
    
    for pattern in case_digest_patterns:
        if re.search(pattern, q):
            return True
    
    return False

def _is_ponente_query(query: str) -> bool:
    """Check if query is asking for cases penned by a specific justice"""
    q = query.strip().lower()
    print(f"🔍 Checking ponente query: '{q}'")
    
    ponente_patterns = [
        r".*cases\s+penned\s+by\s+justice",
        r".*decisions\s+by\s+justice",
        r".*opinions\s+by\s+justice",
        r".*justice\s+[a-zA-Z\s]+\s+cases",
        r".*justice\s+[a-zA-Z\s]+\s+decisions",
        r".*justice\s+[a-zA-Z\s]+\s+opinions",
        r".*penned\s+by\s+justice",
        r".*written\s+by\s+justice",
        r".*authored\s+by\s+justice",
    ]
    
    for i, pattern in enumerate(ponente_patterns):
        if re.search(pattern, q):
            print(f"✅ Matched pattern {i+1}: {pattern}")
            return True
    
    print(f"❌ No ponente patterns matched")
    return False

def _extract_ponente_name_from_query(query: str) -> Optional[str]:
    """Extract justice name from a ponente query"""
    q = query.strip()
    print(f"🔍 Extracting justice name from: '{q}'")
    
    # Pattern for "cases penned by Justice [Name]"
    match = re.search(r".*cases\s+penned\s+by\s+justice\s+([a-zA-Z\s\-']+)", q, re.IGNORECASE)
    if match:
        result = match.group(1).strip()
        print(f"✅ Extracted justice name: '{result}'")
        return result
    
    # Pattern for "decisions by Justice [Name]"
    match = re.search(r".*decisions\s+by\s+justice\s+([a-zA-Z\s\-']+)", q, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern for "Justice [Name] cases"
    match = re.search(r".*justice\s+([a-zA-Z\s\-']+)\s+cases", q, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern for "Justice [Name] decisions"
    match = re.search(r".*justice\s+([a-zA-Z\s\-']+)\s+decisions", q, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern for "penned by Justice [Name]"
    match = re.search(r".*penned\s+by\s+justice\s+([a-zA-Z\s\-']+)", q, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None

def _normalize_name_for_match(name: str) -> str:
    """Normalize justice names for robust matching (lowercase, remove punctuation, hyphens -> space)."""
    if not name:
        return ""
    n = name.lower()
    # Replace hyphens with spaces to match hyphenated surnames like Lazaro-Javier
    n = n.replace('-', ' ')
    # Remove periods and commas
    n = re.sub(r"[\.,]", "", n)
    # Collapse multiple spaces
    n = re.sub(r"\s+", " ", n).strip()
    return n

def _extract_surname(name: str) -> str:
    """Extract the surname (last token) from a provided justice name."""
    if not name:
        return ""
    tokens = [t for t in _normalize_name_for_match(name).split(' ') if t]
    return tokens[-1] if tokens else ""

def _ponente_matches(doc_ponente: str, justice_name: Optional[str]) -> bool:
    """Return True if the document ponente matches the requested justice (supports surname-only queries)."""
    if not doc_ponente:
        return False
    if not justice_name:
        return False
    
    norm_doc = _normalize_name_for_match(doc_ponente)
    norm_query = _normalize_name_for_match(justice_name)
    
    # Direct substring match (covers full names)
    if norm_query and norm_query in norm_doc:
        return True
    
    # Surname-only match as fallback (e.g., Lazaro -> Lazaro Javier)
    surname = _extract_surname(justice_name)
    if surname:
        # Check if surname appears as a separate word
        if surname in norm_doc.split(' '):
            return True
        # Check if surname is part of any token (for hyphenated names)
        if any(surname == tok or surname in tok for tok in norm_doc.split(' ')):
            return True
        # Check for common justice title patterns
        justice_titles = ['j', 'jj', 'cj', 'saj']
        for title in justice_titles:
            if f"{surname} {title}" in norm_doc or f"{surname}, {title}" in norm_doc:
                return True
    
    # Handle common name variations
    if justice_name:
        # Check for common variations like "Lazaro" vs "Lazaro-Javier"
        base_name = justice_name.replace('-', ' ').replace('.', '').strip()
        if base_name.lower() in norm_doc:
            return True
        
        # Check for partial matches in multi-word names
        query_parts = [part.strip() for part in base_name.split() if part.strip()]
        if len(query_parts) > 1:
            # Check if all parts of the query name appear in the document ponente
            if all(part.lower() in norm_doc for part in query_parts):
                return True
    
    return False

def _extract_case_name_from_digest_request(query: str) -> Optional[str]:
    """Extract case name from a case digest request"""
    q = query.strip()
    
    # Pattern for "digest the case of X vs Y"
    match = re.search(r"digest\s+the\s+case\s+of\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern for "case digest of X vs Y"
    match = re.search(r"case\s+digest\s+of\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern for "summarize the case of X vs Y"
    match = re.search(r"summarize\s+the\s+case\s+of\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern for "tell me about the case of X vs Y"
    match = re.search(r"tell\s+me\s+about\s+the\s+case\s+of\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern for "X vs Y" or "X v. Y" (including hyphens and periods)
    match = re.search(r"([a-zA-Z\s\-\.]+)\s+vs\.?\s+([a-zA-Z\s\-\.]+)", q, re.IGNORECASE)
    if match:
        return f"{match.group(1).strip()} vs {match.group(2).strip()}"
    
    return None

def _generate_detailed_case_response(full_case: Dict, query: str, history: Optional[List[Dict]] = None) -> str:
    """Generate a detailed response for follow-up questions about a specific case"""
    if not TOGETHERAI_AVAILABLE:
        return _generate_case_summary_from_jsonl(full_case, query, None, history)
    
    try:
        # Extract case metadata
        case_title = full_case.get("case_title", "Unknown Case")
        gr_number = full_case.get("gr_number", "Unknown")
        ponente = full_case.get("ponente", "Unknown")
        date = full_case.get("promulgation_date", "Unknown")
        clean_text = full_case.get("clean_text", "")
        
        # Build a more detailed prompt based on the query
        q = query.lower()
        
        if "facts" in q or "factual" in q:
            focus = "facts and factual background"
            instruction = "Focus specifically on the factual background, events, and circumstances of the case."
        elif "issues" in q or "issue" in q:
            focus = "legal issues and questions"
            instruction = "Focus specifically on the legal issues, questions, and problems raised in the case."
        elif "ruling" in q or "decision" in q or "hold" in q:
            focus = "ruling and decision"
            instruction = "Focus specifically on the court's ruling, decision, and legal reasoning."
        elif "principles" in q or "doctrine" in q or "law" in q:
            focus = "legal principles and doctrines"
            instruction = "Focus specifically on the legal principles, doctrines, and precedents established."
        else:
            focus = "detailed information"
            instruction = "Provide comprehensive details about the case."
        
        detailed_prompt = f"""You are a Philippine Law expert. The user is asking for more details about a specific case they previously discussed.

Case Information:
- Title: {case_title}
- G.R. Number: {gr_number}
- Ponente: {ponente}
- Date: {date}

User Query: "{query}"

Task: Provide detailed information focusing on {focus}.

{instruction}

Context from case text:
{clean_text[:4000]}

Provide a detailed, well-structured response that directly addresses what the user is asking for. Use clear headings and organize the information logically."""

        messages = [
            {"role": "system", "content": "You are a Philippine Law expert providing detailed case analysis."},
            {"role": "user", "content": detailed_prompt}
        ]
        
        print(f"🤖 Generating detailed response for follow-up question...")
        response = generate_messages_with_togetherai(
            messages,
            max_tokens=1024,
            temperature=0.3,
            top_p=0.9
        )
        
        # Add source URL if available
        source_url = full_case.get("source_url")
        if source_url:
            response += f"\n\n---\n📚 **Source Reference:**\nView the complete case text: {source_url}"
        
        print(f"✅ Generated detailed follow-up response")
        return response.strip()
        
    except Exception as e:
        print(f"⚠️ Detailed response generation failed: {e}")
        # Fallback to regular case summary
        return _generate_case_summary_from_jsonl(full_case, query, None, history)

def _handle_web_search_query(query: str, history: Optional[List[Dict]] = None) -> str:
    """Handle web search queries using external search APIs"""
    
    if not WEB_SEARCH_AVAILABLE:
        return "Web search is currently unavailable. Please try searching for specific cases in our database or rephrase your question."
    
    try:
        searcher = get_web_searcher()
        
        if not searcher.is_available():
            return "Web search is not configured. Please check API credentials or try searching for specific cases in our database."
        
        print(f"🌐 Performing web search for: '{query}'")
        
        # Perform web search
        web_results = searcher.search_legal_content(query, num_results=5)
        
        if not web_results:
            return "I couldn't find current information on that topic. Please try rephrasing your question or asking about specific cases in our database."
        
        print(f"✅ Web search found {len(web_results)} results")
        
        # Generate response using web results
        return _generate_web_search_response(query, web_results, history)
        
    except Exception as e:
        print(f"❌ Web search failed: {e}")
        return "I encountered an error while searching for current information. Please try again later or ask about specific cases in our database."

def _generate_web_search_response(query: str, web_results: List[Dict], history: Optional[List[Dict]] = None) -> str:
    """Generate response based on web search results"""
    
    if not TOGETHERAI_AVAILABLE:
        return _generate_simple_web_response(web_results)
    
    try:
        # Prepare web search context
        search_context = ""
        for i, result in enumerate(web_results[:3], 1):  # Use top 3 results
            search_context += f"""
{i}. **{result['title']}**
   Source: {result['source']}
   Content: {result['snippet']}
   URL: {result['url']}
"""
        
        web_search_prompt = f"""You are a Philippine Law expert. The user asked: "{query}"

Based on the following web search results, provide a comprehensive answer:

{search_context}

Task: 
1. Synthesize the information from multiple sources
2. Focus on Philippine legal context
3. Provide accurate, up-to-date information
4. If information conflicts, note the discrepancies
5. Structure your response with clear headings
6. Do NOT include inline references like [1], [2], etc.
7. Do NOT include a "References" or "Sources" section

Provide a well-structured response with clear sections. Focus on delivering comprehensive information without citation formatting."""

        messages = [
            {"role": "system", "content": "You are a Philippine Law expert providing current legal information based on web search results."},
            {"role": "user", "content": web_search_prompt}
        ]
        
        print(f"🤖 Generating web search response with TogetherAI...")
        response = generate_messages_with_togetherai(
            messages,
            max_tokens=1024,
            temperature=0.3,
            top_p=0.9
        )
        
        # Return the response without additional sources section
        print(f"✅ Generated web search response")
        
        return response.strip()
        
    except Exception as e:
        print(f"⚠️ Web search response generation failed: {e}")
        # Fallback to simple response
        return _generate_simple_web_response(web_results)

def _generate_simple_web_response(web_results: List[Dict]) -> str:
    """Generate a simple response when TogetherAI is not available"""
    
    if not web_results:
        return "No current information found on this topic."
    
    response_parts = ["Based on current web sources, here's what I found:\n"]
    
    for i, result in enumerate(web_results[:3], 1):
        response_parts.append(f"**{i}. {result['title']}**")
        response_parts.append(f"Summary: {result['snippet']}")
        response_parts.append("")
    
    return "\n".join(response_parts)

def _extract_case_from_history(history: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Extract the most recent case information from conversation history"""
    if not history:
        return None
    
    # Look for the most recent assistant message that contains case information
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            
            # Extract GR number from the content
            gr_match = re.search(r'G\.R\.\s*No\.\s*([0-9\-]+)', content)
            if gr_match:
                gr_number = gr_match.group(1)
                return {"gr_number": gr_number, "content": content}
            
            # Extract case title (look for bold titles)
            title_match = re.search(r'\*\*([^*]+)\*\*', content)
            if title_match:
                return {"case_title": title_match.group(1), "content": content}
    
    return None

def _classify_and_handle_query(query: str, history: Optional[List[Dict]] = None) -> Optional[str]:
    """
    Use TogetherAI to classify query intent and handle simple queries directly.
    Returns:
        - None if query needs RAG retrieval (case search)
        - String response if query can be handled directly (greetings, general questions, definitions)
    """
    if not TOGETHERAI_AVAILABLE:
        return None
    
    try:
        # Build classification prompt
        classification_system = """You are a Philippine Law Chatbot specializing in Philippine Supreme Court jurisprudence.

Your task is to classify the user's query and respond appropriately:

1. **GREETINGS/CASUAL**: If it's a greeting (hi, hello, etc.) or casual conversation (thanks, bye, how are you):
   - Respond with a friendly welcome message explaining what you can help with
   - Mention: case lookups by G.R. number, legal definitions, and jurisprudence research

2. **LEGAL DEFINITIONS/EXPLANATIONS**: If asking "what is", "define", "explain" about a legal concept, doctrine, or term:
   - Provide a clear, concise explanation/definition using your legal knowledge
   - Include examples if relevant
   - Examples: "what is doctrine of stare decisis", "define res judicata", "explain certiorari"
   - Do NOT search for cases - just explain the concept directly

3. **GENERAL/VAGUE QUESTIONS**: If asking overly broad questions like "what are doctrines", "what is law", "how does court work":
   - Politely ask them to be more specific
   - Give examples of specific queries you can handle

4. **FOLLOW-UP QUESTIONS**: If the query is asking for more details about a previously discussed case:
   - "delve deeper into the facts"
   - "tell me more about the ruling"
   - "what were the issues raised"
   - "explain the legal principles"
   - "give me more details about this case"
   - Any question referring to "this case", "the case", or asking for elaboration
   
   Then respond with EXACTLY: "[SEARCH_CASES]"
   
5. **WEB SEARCH QUERIES**: If the query is asking for:
   - Recent legal developments ("recent Supreme Court decisions", "latest legal updates")
   - Current legal information ("current tax law", "latest labor law changes")
   - Constitutional provisions not in local database
   - Statutory laws, codes, or recent amendments
   - Legal news, current events, or recent court decisions
   - Legal procedures, requirements, or how-to information
   
   Then respond with EXACTLY: "[WEB_SEARCH]"

6. **CASE DIGEST REQUESTS**: If the query is:
   - Asking for a digest of a specific case ("digest the case of X vs Y", "case digest of X vs Y")
   - Requesting a summary of a specific case ("summarize the case of X vs Y")
   - Asking for details about a specific case ("tell me about the case of X vs Y")
   - Mentions specific parties in a case ("Tan-Andal vs Andal", "People vs Sanchez")
   - Looking up a specific case by G.R. number ("G.R. No. 123456")
   
   Then respond with EXACTLY: "[CASE_DIGEST]"

7. **CASE SEARCH QUERIES**: If the query is:
   - Searching for cases on a specific legal topic ("impeachment cases", "illegal dismissal cases")
   - Asking for cases that APPLY or DEMONSTRATE a doctrine/principle
   - Example: "cases about stare decisis", "show me res judicata cases"
   - General case searches without specific case names
   
   Then respond with EXACTLY: "[SEARCH_CASES]"
   
   Do NOT add any other text, just "[SEARCH_CASES]" or "[CASE_DIGEST]"

Examples:
- "hi" → Give greeting
- "what are doctrines" → Ask to be more specific (too vague)
- "what is doctrine of stare decisis" → Explain the doctrine directly (definition)
- "define res judicata" → Explain the concept directly (definition)
- "explain certiorari" → Explain the concept directly (definition)
- "what is negligence in tort law" → Explain the concept directly (definition)
- "digest the case of Tan-Andal vs Andal" → [CASE_DIGEST]
- "case digest of People vs Sanchez" → [CASE_DIGEST]
- "tell me about the case of X vs Y" → [CASE_DIGEST]
- "summarize the case of ABC vs DEF" → [CASE_DIGEST]
- "G.R. No. 123456" → [CASE_DIGEST]
- "impeachment cases" → [SEARCH_CASES] (search for cases)
- "estafa cases" → [SEARCH_CASES] (search for cases)
- "annulment cases" → [SEARCH_CASES] (search for cases)
- "cases applying stare decisis" → [SEARCH_CASES] (search for cases)
- "show me res judicata cases" → [SEARCH_CASES] (search for cases)
- "delve deeper into the facts of this case" → [SEARCH_CASES] (follow-up)
- "tell me more about the ruling" → [SEARCH_CASES] (follow-up)
- "recent Supreme Court decisions on labor law" → [WEB_SEARCH]
- "current tax law provisions" → [WEB_SEARCH]
- "latest legal news" → [WEB_SEARCH]
- "how to file a petition for certiorari" → [WEB_SEARCH]
- "cases about illegal dismissal" → [SEARCH_CASES]

IMPORTANT: Pay attention to conversation history to maintain context. If the user is asking for more details about a previously discussed case, treat it as a follow-up question and respond with [SEARCH_CASES]."""

        # Build messages with conversation history
        messages = [{"role": "system", "content": classification_system}]
        
        # Add conversation history if available
        if history:
            print(f"📜 Including {len(history)} messages from conversation history")
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
        else:
            print("📜 No conversation history available")
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        print(f"🤖 Classifying query with TogetherAI (total messages: {len(messages)})...")
        response = generate_messages_with_togetherai(
            messages,
            max_tokens=4092,
            temperature=0.2,
            top_p=0.9
        )
        
        print(f"🤖 Classification response: '{response[:100]}...'")
        
        # Check if we need to search cases
        if "[SEARCH_CASES]" in response:
            print("🔍 Query requires case search - proceeding with RAG retrieval")
            return None
        
        # Check if this is a case digest request
        if "[CASE_DIGEST]" in response:
            print("📋 Query requires case digest - proceeding with focused case retrieval")
            return None
        
        # Otherwise, return the AI's direct response
        print("✅ Query handled directly by AI")
        return response.strip()
        
    except Exception as e:
        print(f"⚠️ Query classification failed: {e}")
        # On error, default to RAG retrieval
        return None


def _extract_gr_number(query: str) -> str:
    """Extract G.R. number from query, returns numeric part or empty string."""
    if not query:
        return ""
    q = query.strip()
    # Common patterns: "G.R. No. 123456", "G.R. NOS. 151809-12", "GR No. 123456", bare digits with separators
    # Match "G.R. NOS." (plural) or "G.R. No." (singular)
    m = re.search(r"G\.R\.?\s*NOS?\.?\s*([0-9\-]+)", q, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"GR\s*NOS?\.?\s*([0-9\-]+)", q, re.IGNORECASE)
    if m:
        return m.group(1)
    # Fallback: a long number that looks like GR (7+ digits)
    m = re.search(r"\b(\d{5,})\b", q)
    return m.group(1) if m else ""

def _extract_special_number(query: str) -> str:
    """Extract special number from query (A.M., OCA, etc.), returns formatted number or empty string."""
    if not query:
        return ""
    q = query.strip()
    
    # Patterns for different special case types
    special_patterns = [
        (r"A\.M\.\s+No\.?\s*([0-9\-]+[A-Z]*)", "A.M. No. {}"),
        (r"OCA\s+No\.?\s*([0-9\-]+[A-Z]*)", "OCA No. {}"),
        (r"U\.C\.\s+No\.?\s*([0-9\-]+[A-Z]*)", "U.C. No. {}"),
        (r"ADM\s+No\.?\s*([0-9\-]+[A-Z]*)", "ADM No. {}"),
        (r"A\.C\.\s+No\.?\s*([0-9\-]+[A-Z]*)", "A.C. No. {}"),
        (r"AC\s+No\.?\s*([0-9\-]+[A-Z]*)", "AC No. {}"),
        (r"B\.M\.\s+No\.?\s*([0-9\-]+[A-Z]*)", "B.M. No. {}"),
        (r"LRC\s+No\.?\s*([0-9\-]+[A-Z]*)", "LRC No. {}"),
        (r"SP\s+No\.?\s*([0-9\-]+[A-Z]*)", "SP No. {}"),
    ]
    
    for pattern, format_str in special_patterns:
        m = re.search(pattern, q, re.IGNORECASE)
        if m:
            number = m.group(1).strip()
            return format_str.format(number)
    
    return ""

def _to_title_case(text: str) -> str:
    """Convert text to proper title case for better readability"""
    if not text:
        return text
    
    # Handle special legal terms that should stay uppercase
    legal_terms = {
        'VERSUS', 'PETITIONER', 'PETITIONERS', 'RESPONDENT', 'RESPONDENTS',
        'COMPLAINANT', 'COMPLAINANTS', 'PLAINTIFF', 'PLAINTIFFS', 'DEFENDANT', 'DEFENDANTS',
        'G.R.', 'A.M.', 'LRC', 'SP', 'NO.', 'ET', 'AL.', 'ET AL', 'JR.', 'SR.', 'III', 'IV',
        'INC.', 'CORP.', 'CO.', 'LTD.', 'LLC.', 'PHIL.'
    }
    
    # Convert to title case first
    title_text = text.title()
    
    # Restore legal terms to uppercase
    for term in legal_terms:
        title_text = re.sub(r'\b' + re.escape(term.lower()) + r'\b', term, title_text, flags=re.IGNORECASE)
    
    # Make common words lowercase
    lowercase_words = {'Of', 'The', 'And', 'In', 'On', 'At', 'To', 'For', 'With', 'By'}
    for word in lowercase_words:
        # Only replace if not at the beginning of the sentence and not after punctuation
        title_text = re.sub(r'(?<!^)(?<![.!?]\s)\b' + re.escape(word) + r'\b', word.lower(), title_text)
    
    # Handle special cases for names with periods
    title_text = re.sub(r'\b([A-Z])\.\s+([A-Z])\.\s+([A-Z])', r'\1. \2. \3', title_text)  # J. P. SMITH
    title_text = re.sub(r'\b([A-Z])\.\s+([A-Z])', r'\1. \2', title_text)  # J. SMITH
    
    # Make "vs." lowercase LAST (after all other processing to ensure it sticks)
    title_text = re.sub(r'\bVs\.\b', 'vs.', title_text, flags=re.IGNORECASE)
    title_text = re.sub(r'\bV\.\b', 'v.', title_text, flags=re.IGNORECASE)
    title_text = re.sub(r'\bVersus\b', 'versus', title_text, flags=re.IGNORECASE)
    
    # Highlight GR numbers with clickable links (using custom protocol for frontend handling)
    title_text = re.sub(r'\(G\.R\.\s+No\.\s+([0-9\-]+)\)', r'**[G.R. No. \1](gr:\1)**', title_text)
    title_text = re.sub(r'\(A\.M\.\s+No\.\s+([0-9\-]+)\)', r'**[A.M. No. \1](am:\1)**', title_text)
    
    return title_text

def _normalize_gr_display(value: str) -> str:
    """Return a clean 'G.R. No. <num>' or 'Unknown' if not parseable."""
    if not value:
        return "Unknown"
    # if already contains a GR pattern, extract numeric part
    m = re.search(r"G\.R\.?\s*No\.?\s*([0-9\-]+)", value, re.IGNORECASE)
    if not m:
        m = re.search(r"\b(\d{5,})\b", value)
    num = m.group(1) if m else ""
    return f"G.R. No. {num}" if num else "Unknown"

def _display_title(d: Dict) -> str:
    """Prefer proper case titles; otherwise, provide a concise fallback."""
    meta = d.get("metadata", {}) or {}
    
    # Debug: Print what we have
    print(f"🔍 _display_title debug:")
    print(f"   d.get('title'): '{d.get('title', 'None')[:50]}...'")
    print(f"   d.get('case_title'): '{d.get('case_title', 'None')[:50]}...'")
    print(f"   meta.get('title'): '{meta.get('title', 'None')[:50]}...'")
    print(f"   meta.get('case_title'): '{meta.get('case_title', 'None')[:50]}...'")
    
    # Prefer explicit title-like fields (matching actual webscraped data structure)
    title = (
        d.get("case_title")
        or meta.get("case_title")
        or d.get("title")
        or meta.get("title")
        or meta.get("header")
        or ""
    )
    if title:
        t = title.strip()
        print(f"   Found title: '{t[:100]}...'")
        
        # Check if this looks like factual content rather than a case title
        if _is_factual_content(t):
            print(f"   Title looks like factual content, using GR number instead")
            # Use GR number instead of factual content
            gr = d.get("gr_number") or meta.get("gr_number")
            if gr:
                result = f"Case ({_normalize_gr_display(str(gr))})"
                print(f"   Using GR number: '{result}'")
                return result
        
        # Ensure we have a complete case title with both parties
        # Check if title contains both petitioner and respondent
        if "vs." in t.lower() or "v." in t.lower() or "versus" in t.lower():
            # This looks like a proper case title, use it as-is
            if len(t) <= 300:  # Increased limit for longer case titles
                result = t
            else:
                result = t[:297] + "..."
            print(f"   Using complete case title: '{result[:100]}...'")
            return result
        else:
            # Title might be incomplete, try to find complete version in content
            text = d.get("content") or d.get("text") or meta.get("text") or ""
            if text:
                for line in text.split('\n')[:15]:  # Check more lines
                    line = line.strip()
                    if (len(line) >= 30 and 
                        re.search(r"\b(v\.|vs\.|versus)\b", line, re.IGNORECASE) and
                        not _is_factual_content(line)):
                        result = line[:297] + "..." if len(line) > 300 else line
                        print(f"   Using extracted complete title: '{result[:100]}...'")
                        return result
        
        # Fallback to original title processing
        if len(t) <= 300:
            result = t
        else:
            result = t[:297] + "..."
        # Convert to title case for better readability
        result = _to_title_case(result)
        print(f"   Using database title: '{result[:100]}...'")
        return result
    else:
        print(f"   No title found, trying content extraction...")
    
    # Try to infer a title from content lines containing v./vs.
    text = d.get("content") or d.get("text") or meta.get("text") or ""
    if text:
        for line in (text.split('\n')[:10]):  # Only check first 10 lines
            line = line.strip()
            if len(line) >= 20 and re.search(r"\b(v\.|vs\.|versus)\b", line, re.IGNORECASE):
                # Check if this looks like a proper case title (not factual content)
                if not _is_factual_content(line):
                    result = line[:117] + "..." if len(line) > 120 else line
                    result = _to_title_case(result)
                    print(f"   Using extracted title: '{result[:100]}...'")
                    return result
        # Fallback: detect all-caps caption with PETITIONER/RESPONDENT/COMPLAINANT
        for line in (text.split('\n')[:12]):
            line = line.strip().strip('. ')
            if not line:
                continue
            if re.search(r"\b(PETITIONER|RESPONDENT|COMPLAINANT|PLAINTIFF|DEFENDANT)S?\b", line, re.IGNORECASE):
                if 20 <= len(line) <= 160 and not _is_factual_content(line):
                    result = _to_title_case(line)
                    print(f"   Using extracted title: '{result[:100]}...'")
                    return result
    
    # Fallback to generic label to avoid duplicating GR in title and suffix
    # Use GR/special number if available
    gr = d.get("gr_number") or meta.get("gr_number")
    spec = d.get("special_number") or meta.get("special_number")
    if gr:
        return f"Case ({_normalize_gr_display(str(gr))})"
    if spec:
        return f"Case ({spec})"
    return "Untitled case"


def _is_factual_content(text: str) -> bool:
    """Check if text looks like factual content rather than a case title."""
    text_lower = text.lower().strip()
    
    # Patterns that indicate factual content rather than case titles
    factual_patterns = [
        r"the marriage was",
        r"respondent complained that",
        r"their sexual relationship",
        r"the couple talked about",
        r"is a product of a broken family",
        r"her father always subjected",
        r"her mother to physical abuse",
        r"both parents are living",
        r"her auntie took care",
        r"her mother would give her away",
        r"in exchange for money",
        r"the case involve",
        r"the case stem",
        r"the case arise",
        r"this is an",
        r"this case is",
        r"the instant case",
        r"the present case",
        r"complainant claims",
        r"respondent served",
        r"while respondent",
        r"complainant states",
    ]
    
    for pattern in factual_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Check for narrative indicators
    narrative_indicators = [
        "in february", "in march", "in april", "in may", "in june",
        "in july", "in august", "in september", "in october", "in november", "in december",
        "in january", "on january", "on february", "on march",
        "the facts", "the edsa", "revolution toppled",
        "pursuant to this", "pursuant to the mandate",
        "following the", "after the", "was to establish", "was established"
    ]
    
    for indicator in narrative_indicators:
        if indicator in text_lower:
            return True
    
    return False


def _clean_case_title(title: str) -> str:
    """Clean up case title by removing noise and formatting."""
    if not title:
        return ""
    
    # Remove common noise
    noise_patterns = [
        r"^PROMULGATED:\s*",
        r"^DECIDED:\s*",
        r"^RESOLVED:\s*",
        r"\s+D\s+E\s+C\s+I\s+S\s+I\s+O\s+N.*$",
        r"\s+DECISION.*$",
        r"\s+RESOLUTION.*$",
        r"\s+ORDER.*$",
        r"^SUPREME\s+COURT\s+",
        r"^REPUBLIC\s+OF\s+THE\s+PHILIPPINES\s+",
        r"^EN\s+BANC\s+",
        r"^FIRST\s+DIVISION\s+",
        r"^SECOND\s+DIVISION\s+",
        r"^THIRD\s+DIVISION\s+",
        r"^PER\s+CURIAM\s+",
        # Remove promulgation dates and similar metadata
        r"\s+Promulgated:\s+[A-Z]{3}\s+\d{1,2}\s+\d{4}.*$",
        r"\s+Promulgated:\s+\d{1,2}\s+[A-Z]{3}\s+\d{4}.*$",
        r"\s+Promulgated:\s+\d{4}.*$",
        r"\s+Date:\s+[A-Z]{3}\s+\d{1,2}\s+\d{4}.*$",
        r"\s+Date:\s+\d{1,2}\s+[A-Z]{3}\s+\d{4}.*$",
        r"\s+Date:\s+\d{4}.*$",
    ]
    
    cleaned = title.strip()
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    
    return cleaned

def _extract_case_title_components(query: str) -> Dict[str, str]:
    """Extract case title components for smart matching"""
    query_lower = query.lower().strip()
    
    # Normalize vs variants
    query_normalized = re.sub(r'\bversus\b', 'vs', query_lower, flags=re.IGNORECASE)
    query_normalized = re.sub(r'\bvs\.?\b', 'vs', query_normalized, flags=re.IGNORECASE)
    
    # Extract parties (before and after vs)
    vs_match = re.search(r'\bvs\b', query_normalized)
    if vs_match:
        before_vs = query_normalized[:vs_match.start()].strip()
        after_vs = query_normalized[vs_match.end():].strip()
        
        # Extract individual names/entities
        petitioner_parts = [p.strip() for p in re.split(r'[,&\s]+', before_vs) if p.strip()]
        respondent_parts = [p.strip() for p in re.split(r'[,&\s]+', after_vs) if p.strip()]
        
        return {
            'petitioner': ' '.join(petitioner_parts),
            'respondent': ' '.join(respondent_parts),
            'petitioner_parts': petitioner_parts,
            'respondent_parts': respondent_parts,
            'full_query': query_normalized
        }
    else:
        # No vs found, treat as single entity search
        parts = [p.strip() for p in re.split(r'[,&\s]+', query_normalized) if p.strip()]
        return {
            'petitioner': query_normalized,
            'respondent': '',
            'petitioner_parts': parts,
            'respondent_parts': [],
            'full_query': query_normalized
        }

def _calculate_title_similarity(query_components: Dict[str, str], case_title: str) -> float:
    """Calculate similarity score between query components and case title"""
    if not case_title:
        return 0.0
    
    case_title_lower = case_title.lower()
    score = 0.0
    
    # Exact match gets highest score
    if query_components['full_query'] in case_title_lower:
        score += 100.0
    
    # Check for petitioner and respondent matches
    if query_components['petitioner'] and query_components['petitioner'] in case_title_lower:
        score += 50.0
    
    if query_components['respondent'] and query_components['respondent'] in case_title_lower:
        score += 50.0
    
    # Check for individual name parts
    for part in query_components['petitioner_parts']:
        if part and len(part) > 2:  # Skip very short parts
            if part in case_title_lower:
                score += 20.0
            # Partial match
            elif any(part in word for word in case_title_lower.split()):
                score += 10.0
    
    for part in query_components['respondent_parts']:
        if part and len(part) > 2:  # Skip very short parts
            if part in case_title_lower:
                score += 20.0
            # Partial match
            elif any(part in word for word in case_title_lower.split()):
                score += 10.0
    
    return score

def _advanced_retrieve(retriever, query: str, k: int = 8, is_case_digest: bool = False, history: List[Dict] = None):
    """Unified retrieval wrapper with chunking support"""
    results = retriever.retrieve(query, k=k, is_case_digest=is_case_digest, conversation_history=history)
    
    # If results are chunks, create context efficiently
    if results and 'section' in results[0]:
        # Results are chunks - create compact context
        context = retriever._create_context_from_chunks(results, max_tokens=2500)
        return results, context
    
    return results

# --- Stronger dispositive detection & extraction ---
DISPOSITIVE_HDR = r"(?:WHEREFORE|ACCORDINGLY|IN VIEW OF THE FOREGOING|IN VIEW WHEREOF|THUS|HENCE|PREMISES CONSIDERED)"
SO_ORDERED = r"SO\s+ORDERED\.?"
RULING_REGEX = re.compile(
    rf"{DISPOSITIVE_HDR}[\s\S]{{0,10000}}?{SO_ORDERED}",
    re.IGNORECASE,
)
# Some decisions omit SO ORDERED but still have a dispositive paragraph
RULING_NO_SO_FALLBACK = re.compile(
    rf"(?:^{DISPOSITIVE_HDR}[\s\S]{{50,5000}}?$)",
    re.IGNORECASE | re.MULTILINE,
)
# Catch short single-paragraph orders that end with SO ORDERED without header
RULING_SO_ORDERED_FALLBACK = re.compile(
    rf"(?s)(?:^|\n\n).{{0,5000}}?{SO_ORDERED}",
    re.IGNORECASE,
)

# Facts / Issues headings (expanded)
FACTS_HINT_RE = re.compile(
    r"^\s*(?:Factual\s+(?:Antecedents|Background)|Antecedent\s+Facts|"
    r"Facts(?:\s+of\s+the\s+Case)?|Statement\s+of\s+Facts|The\s+Facts)\s*[:\-–]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
ISSUES_HINT_RE = re.compile(
    r"^\s*(?:Issues?(?:\s+for\s+Resolution)?|Questions?\s+Presented|Issue)\s*[:\-–]?\s*$"
    r"|^\s*(?:[IVX]+\.)?\s*Whether\b",
    re.IGNORECASE | re.MULTILINE,
)

# Keep a single retriever instance (lazy-init for reuse)
retriever = None

# Evaluation tracker instance
evaluation_tracker = None

def get_evaluation_tracker():
    """Get or create evaluation tracker instance"""
    global evaluation_tracker
    if evaluation_tracker is None and METRICS_AVAILABLE:
        evaluation_tracker = EvaluationTracker()
    return evaluation_tracker


def evaluate_response(query: str, response: str, reference_text: str, 
                     case_metadata: Optional[Dict] = None,
                     expert_scores: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """
    Evaluate chatbot response using integrated metrics
    
    Args:
        query: User query
        response: Chatbot generated response
        reference_text: Reference legal text from JSONL
        case_metadata: Optional case metadata
        expert_scores: Optional expert evaluation scores
        
    Returns:
        Evaluation results dictionary or None if metrics not available
    """
    if not METRICS_AVAILABLE:
        print("⚠️ Evaluation metrics not available")
        return None
    
    try:
        tracker = get_evaluation_tracker()
        if tracker:
            evaluation_result = tracker.log_evaluation(
                query=query,
                response=response,
                reference=reference_text,
                case_metadata=case_metadata,
                expert_scores=expert_scores
            )
            
            print(f"📊 Evaluation Scores:")
            print(f"   BLEU: {evaluation_result['automated_scores']['bleu']['bleu_avg']:.4f}")
            print(f"   ROUGE-1 F1: {evaluation_result['automated_scores']['rouge']['rouge_1']['f1']:.4f}")
            print(f"   ROUGE-2 F1: {evaluation_result['automated_scores']['rouge']['rouge_2']['f1']:.4f}")
            print(f"   ROUGE-L F1: {evaluation_result['automated_scores']['rouge']['rouge_l']['f1']:.4f}")
            print(f"   Legal Elements: {evaluation_result['automated_scores']['legal_elements']['presence_rate']:.4f}")
            print(f"   Overall Relevance: {evaluation_result['automated_scores']['overall_relevance']['score']:.4f}")
            
            return evaluation_result
    except Exception as e:
        print(f"⚠️ Error during evaluation: {e}")
        return None


def get_evaluation_statistics() -> Optional[Dict[str, Any]]:
    """Get aggregated evaluation statistics for current session"""
    if not METRICS_AVAILABLE:
        return None
    
    tracker = get_evaluation_tracker()
    if tracker:
        return tracker.get_session_statistics()
    return None

# --- Small utils ---
def _normalize_ws(s: str, max_chars: int = 1600) -> str:
    if not s:
        return ""
    s = s.replace("\r", "\n")
    # keep paragraph breaks for readability; collapse intra-paragraph whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s[:max_chars]

def _extract_dispositive(text: str) -> str:
    """Return the shortest verbatim dispositive paragraph(s) if present, else ''."""
    if not text:
        return ""
    # Prefer classic header→SO ORDERED span (greedy across lines)
    m = RULING_REGEX.search(text)
    if m:
        return _normalize_ws(m.group(0), max_chars=10000)
    # Next: single-paragraph ending in SO ORDERED
    m = RULING_SO_ORDERED_FALLBACK.search(text)
    if m:
        return _normalize_ws(m.group(0), max_chars=5000)
    # Last: dispositive without SO ORDERED (rare but happens)
    m = RULING_NO_SO_FALLBACK.search(text)
    if m:
        return _normalize_ws(m.group(0), max_chars=5000)
    return ""

def _generate_case_summary(docs: List[Dict], query: str, retriever=None, history: List[Dict] = None) -> str:
    """Generate a brief summary of the case from retrieved documents"""
    if not docs:
        return "No case information found."
    
    # Get the main case document (highest score)
    main_doc = max(docs, key=lambda x: x.get("score", 0.0))
    
    # Try to find the correct case by looking for exact matches first
    exact_match_doc = None
    query_lower = query.lower().strip()
    query_words = set(query_lower.split())
    
    print(f"[SEARCH] Looking for exact match for query: '{query_lower}'")
    print(f"[SEARCH] Query words: {query_words}")
    
    # First, try to find a document that matches the query exactly
    for i, doc in enumerate(docs):
        title = (doc.get("title", "") or doc.get("metadata", {}).get("title", "")).lower().strip()
        gr = doc.get("gr_number", "") or doc.get("metadata", {}).get("gr_number", "")
        score = doc.get("score", 0.0)
        
        print(f"  Doc {i+1}: '{title[:50]}...' | G.R. {gr} | Score: {score:.3f}")

        if not title:
            continue

        # More strict matching: check for exact phrase match first
        if query_lower in title or title in query_lower:
            exact_match_doc = doc
            print(f"  ✅ Exact phrase match found: '{title[:50]}...'")
            break
        # Then check for significant word overlap (at least 4 words for better precision)
        elif len(query_words.intersection(set(title.split()))) >= 4:
            exact_match_doc = doc
            print(f"  ✅ Significant word overlap found: '{title[:50]}...'")
            break
    
    # Use exact match document if found, otherwise use main document
    source_doc = exact_match_doc if exact_match_doc else main_doc
    
    if exact_match_doc:
        print(f"[SEARCH] Using exact match document: G.R. {exact_match_doc.get('gr_number', 'Unknown')}")
    else:
        print(f"[SEARCH] No exact match found, using main document: G.R. {main_doc.get('gr_number', 'Unknown')}")
    
    # Extract metadata from the chosen document
    case_title = (source_doc.get("case_title", "") or 
                 source_doc.get("metadata", {}).get("case_title", "") or
                 source_doc.get("title", "") or 
                 source_doc.get("metadata", {}).get("title", "") or
                 query)  # Fallback to query if no title found
    
    gr_number = (source_doc.get("metadata", {}).get("gr_number", "") or 
                source_doc.get("gr_number", "") or 
                source_doc.get("metadata", {}).get("case_id", "") or
                "Not available")
    
    special_number = (source_doc.get("metadata", {}).get("special_number", "") or 
                     source_doc.get("special_number", "") or
                     "Not available")
    
    # Clean up G.R. number to avoid duplication
    if gr_number and gr_number != "Not available":
        # Remove "G.R. No." prefix if it exists to avoid duplication
        gr_number = re.sub(r'^G\.R\.\s+No\.\s*', '', gr_number, flags=re.IGNORECASE).strip()
        if gr_number:
            gr_number = f"G.R. No. {gr_number}"
    
    # Use special number if no GR number available
    case_number = gr_number if gr_number != "Not available" else special_number
    
    ponente = (source_doc.get("metadata", {}).get("ponente", "") or 
              source_doc.get("ponente", "") or 
              source_doc.get("metadata", {}).get("justice", "") or
              "")
    
    # If ponente is still empty, try to extract from content
    if not ponente:
        content = source_doc.get("content", "") or source_doc.get("text", "")
        if content:
            # Try to extract ponente from content (e.g., "CALLEJO, SR., J.")
            ponente_match = re.search(r'\b([A-Z][A-Za-z\-\']+(?:\s+[A-Z][A-Za-z\-\']+)*(?:\s+(?:SR\.|JR\.|III|IV))*)\s*,\s*(J\.|JJ\.|CJ|SAJ)\b', content[:2000])
            if ponente_match:
                ponente = f"{ponente_match.group(1)}, {ponente_match.group(2)}"
    
    if not ponente:
        ponente = "Not available"
    
    date = (source_doc.get("metadata", {}).get("promulgation_date", "") or 
           source_doc.get("metadata", {}).get("date", "") or 
           source_doc.get("date", "") or 
           source_doc.get("metadata", {}).get("decision_date", "") or
           "Not available")
    
    case_type = (source_doc.get("metadata", {}).get("case_type", "") or 
                source_doc.get("case_type", "") or 
                source_doc.get("metadata", {}).get("type", "") or
                "Not available")
    
    print(f"[SEARCH] Final metadata - Title: {case_title[:50]}... | Case: {case_number} | Ponente: {ponente}")
    
    # Try to load full case content from JSONL file
    full_case_content = None
    if case_number and case_number != "Not available":
        # Clean the case number for JSONL lookup (remove display formatting)
        if gr_number != "Not available":
            jsonl_case_number = re.sub(r'^G\.R\.\s+No\.\s*', '', gr_number, flags=re.IGNORECASE).strip()
            print(f"[SEARCH] Loading full case content from JSONL for G.R. {jsonl_case_number}...")
        else:
            jsonl_case_number = special_number
            print(f"[SEARCH] Loading full case content from JSONL for Special {jsonl_case_number}...")
        try:
            from .retriever import load_case_from_jsonl
            full_case_content = load_case_from_jsonl(jsonl_case_number)
            if full_case_content:
                print(f"✅ Loaded full case content: {len(str(full_case_content))} characters")
            else:
                print(f"⚠️ No full case content found for {jsonl_case_number}")
        except Exception as e:
            print(f"⚠️ Error loading full case content: {e}")
    
    # If we have full case content, use it for better summary
    if full_case_content:
        print("[SEARCH] Using full case content for summary generation...")
        # Extract content from full case (using actual JSONL fields)
        facts_content = full_case_content.get("clean_text", "") or full_case_content.get("body", "")
        ruling_content = full_case_content.get("ruling", "")
        
        # Use LLM to generate a proper storytelling summary from JSONL content
        print("[SEARCH] Using LLM to generate storytelling summary from JSONL content...")
        
        # Prepare context for LLM
        context_parts = []
        if facts_content and len(facts_content.strip()) > 100:
            context_parts.append(f"FACTS: {facts_content[:2000]}...")
        if ruling_content and len(ruling_content.strip()) > 100:
            context_parts.append(f"RULING: {ruling_content[:2000]}...")
        
        context = "\n\n".join(context_parts) if context_parts else "No detailed content available."
        
        # Create a prompt for storytelling summary
        storytelling_prompt = f"""Create a 5-sentence case summary for this Philippine Supreme Court case:

Case: {case_title}
Case No.: {case_number}
Ponente: {ponente}

Context:
{context}

Write exactly 5 complete sentences that tell the story of this case. Start with who the parties are, then what happened, what the legal question was, how the court decided, and what the outcome was. Do not include any numbered lists or questions - just write the 5 sentences directly."""

        try:
            from .generator import generate_conversational_response
            print(f"[SEARCH] Calling LLM with prompt length: {len(storytelling_prompt)} characters")
            print(f"[SEARCH] Context length: {len(context)} characters")
            raw_summary = generate_conversational_response(storytelling_prompt, history=history, context="", is_case_digest=False)
            print(f"[SEARCH] Raw LLM output: '{raw_summary[:200]}...' (length: {len(raw_summary)})")
            
            # Clean up the LLM output
            summary_text = raw_summary.strip()
            
            # Remove common LLM artifacts
            summary_text = re.sub(r'^\[INST\].*?\[/INST\]\s*', '', summary_text, flags=re.DOTALL)
            summary_text = re.sub(r'^Assistant:\s*', '', summary_text)
            summary_text = re.sub(r'^Response:\s*', '', summary_text)
            summary_text = re.sub(r'^Summary:\s*', '', summary_text)
            summary_text = re.sub(r'^\[INST\].*?$', '', summary_text, flags=re.MULTILINE)
            summary_text = re.sub(r'^\[/INST\].*?$', '', summary_text, flags=re.MULTILINE)
            summary_text = re.sub(r'^\d+\.\s*', '', summary_text, flags=re.MULTILINE)  # Remove numbered list format
            
            # Ensure it ends with a period
            if summary_text and not summary_text.endswith('.'):
                summary_text += '.'
            
            print(f"✅ Generated LLM storytelling summary: {len(summary_text)} characters")
            print(f"✅ Cleaned summary: '{summary_text[:200]}...'")
        except Exception as e:
            print(f"⚠️ LLM generation failed: {e}, using fallback")
            # Fallback to simple extraction
            summary_text = f"This case involves {case_title}. The case was decided by the Supreme Court of the Philippines and involves legal proceedings between the parties."
        
        print(f"✅ Generated storytelling summary from full case content: {len(summary_text)} characters")
        
        # Final fallback: if summary is still empty or too short, create a basic one
        if not summary_text or len(summary_text.strip()) < 50:
            print("⚠️ Summary too short, creating fallback summary from JSONL content...")
            if facts_content and len(facts_content.strip()) > 100:
                # Extract first meaningful sentence from facts
                sentences = facts_content.split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 50 and 'petitioner' in sentence.lower():
                        # Clean up the sentence to remove metadata noise
                        clean_sentence = re.sub(r'^\d+,\s*December\s+\d+,\s+\d+\s*\]\s*', '', sentence)
                        clean_sentence = re.sub(r'^\d+,\s*[A-Za-z]+\s+\d+,\s+\d+\s*\]\s*', '', clean_sentence)
                        clean_sentence = re.sub(r'^\d+\s*\]\s*', '', clean_sentence)
                        clean_sentence = re.sub(r'^[A-Z\s]+\s*$', '', clean_sentence)  # Remove all-caps lines
                        if clean_sentence and len(clean_sentence.strip()) > 30:
                            summary_text = f"This case involves {case_title}. {clean_sentence}."
                            break
            if not summary_text or len(summary_text.strip()) < 50:
                summary_text = f"This case involves {case_title}. The case was decided by the Supreme Court of the Philippines and involves legal proceedings between the parties."
            print(f"✅ Created fallback summary: {len(summary_text)} characters")
    else:
        # No JSONL content available, try to extract from retrieved documents
        print("⚠️ No JSONL content available, trying to extract from retrieved documents...")
        
        # Try to find meaningful content from the retrieved documents
        best_content = ""
        for doc in docs:
            content = doc.get("content", "") or doc.get("text", "")
            if content and len(content.strip()) > len(best_content):
                best_content = content
        
        if best_content and len(best_content.strip()) > 100:
            # Try to extract a meaningful summary from the content
            sentences = best_content.split('.')
            meaningful_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 30 and 
                    not sentence.startswith(('G.R. No.', 'Supreme Court', 'E-Library', '—', 'The relevant', 'On 02 August', '[', 'PANGANIBAN')) and
                    not sentence.endswith(('for short', 'as follows', 'Decision, as follows')) and
                    not sentence.isupper() and
                    any(word in sentence.lower() for word in ['petitioner', 'respondent', 'alleged', 'claimed', 'contended', 'filed', 'sought', 'requested', 'dispute', 'agreement', 'contract', 'breach', 'damages', 'injury', 'loss', 'court', 'held', 'found', 'determined', 'concluded', 'ruled', 'decided'])):
                    meaningful_sentences.append(sentence)
                    if len(meaningful_sentences) >= 3:
                        break
            
            if meaningful_sentences:
                summary_text = '. '.join(meaningful_sentences) + '.'
            else:
                summary_text = f"This case involves {case_title}. The case was decided by the Supreme Court of the Philippines and involves legal proceedings between the parties."
        else:
            summary_text = f"This case involves {case_title}. The case was decided by the Supreme Court of the Philippines and involves legal proceedings between the parties."
    
    return f"**{case_title}**\n\n**G.R. No.:** {gr_number}\n**Ponente:** {ponente}\n**Date:** {date}\n**Case Type:** {case_type}\n\n**Brief Summary:**\n{summary_text}"


def _extract_enhanced_case_type(main_doc: Dict) -> str:
    """Extract case type using Saibo classification metadata"""
    # Get Saibo classification metadata
    case_type_classification = (main_doc.get("metadata", {}).get("case_type_classification", {}) or 
                               main_doc.get("case_type_classification", {}))
    
    if case_type_classification and isinstance(case_type_classification, dict):
        # Get the highest confidence case type from Saibo classification
        if case_type_classification:
            best_case_type = max(case_type_classification.items(), key=lambda x: x[1])
            if best_case_type[1] > 0.5:  # Confidence threshold
                print(f"Using Saibo classification: {best_case_type[0]} (confidence: {best_case_type[1]:.3f})")
                return best_case_type[0].lower()
            else:
                print(f"Saibo classification confidence too low ({best_case_type[1]:.3f}), using default")
                return "civil"
    
    # If no Saibo classification available, default to civil
    print("No Saibo classification metadata found, using default")
    return "civil"


def _extract_legal_concepts(main_doc: Dict) -> List[str]:
    """Extract key legal concepts and terms from the main document"""
    content = (main_doc.get("content", "") or 
              main_doc.get("text", "") or "").lower()
    
    # Legal concept patterns
    legal_concepts = []
    
    # Legal doctrines and principles
    doctrine_patterns = [
        r'\b(doctrine of [a-z\s]+)\b',
        r'\b(principle of [a-z\s]+)\b',
        r'\b(rule of [a-z\s]+)\b',
        r'\b(test of [a-z\s]+)\b',
        r'\b(standard of [a-z\s]+)\b'
    ]
    
    for pattern in doctrine_patterns:
        matches = re.findall(pattern, content)
        legal_concepts.extend(matches)
    
    # Legal terms and phrases
    legal_terms = [
        'due process', 'equal protection', 'freedom of speech', 'right to privacy',
        'separation of powers', 'checks and balances', 'judicial review',
        'burden of proof', 'preponderance of evidence', 'beyond reasonable doubt',
        'statute of limitations', 'res judicata', 'stare decisis', 'precedent',
        'jurisdiction', 'venue', 'standing', 'ripeness', 'mootness',
        'contractual obligation', 'breach of contract', 'specific performance',
        'damages', 'injunction', 'restitution', 'rescission'
    ]
    
    for term in legal_terms:
        if term in content:
            legal_concepts.append(term)
    
    # Extract case citations (G.R. numbers mentioned in content)
    gr_pattern = r'g\.r\.\s*no\.?\s*(\d+)'
    gr_matches = re.findall(gr_pattern, content, re.IGNORECASE)
    legal_concepts.extend([f"G.R. No. {gr}" for gr in gr_matches[:3]])  # Limit to 3 citations
    
    return list(set(legal_concepts))  # Remove duplicates


def _extract_case_relationships(main_doc: Dict) -> Dict[str, List[str]]:
    """Extract case relationships including cited cases, similar parties, and legal areas"""
    content = (main_doc.get("content", "") or 
              main_doc.get("text", "") or "").lower()
    
    relationships = {
        'cited_cases': [],
        'similar_parties': [],
        'legal_areas': [],
        'key_doctrines': []
    }
    
    # Extract cited cases (G.R. numbers mentioned in content)
    gr_pattern = r'g\.r\.\s*no\.?\s*(\d+)'
    gr_matches = re.findall(gr_pattern, content, re.IGNORECASE)
    relationships['cited_cases'] = [f"G.R. No. {gr}" for gr in gr_matches[:5]]  # Limit to 5 citations
    
    # Extract party names (petitioner/respondent patterns)
    party_patterns = [
        r'petitioner[s]?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'respondent[s]?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'plaintiff[s]?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'defendant[s]?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    ]
    
    for pattern in party_patterns:
        matches = re.findall(pattern, content)
        relationships['similar_parties'].extend(matches)
    
    # Extract legal areas from content
    legal_area_keywords = {
        'constitutional': ['constitutional', 'constitution', 'bill of rights', 'fundamental rights'],
        'criminal': ['criminal', 'penal', 'crime', 'offense', 'violation'],
        'civil': ['civil', 'contract', 'obligation', 'damages', 'liability'],
        'administrative': ['administrative', 'government', 'public', 'official'],
        'labor': ['labor', 'employment', 'worker', 'wage', 'salary'],
        'family': ['family', 'marriage', 'divorce', 'custody', 'support'],
        'property': ['property', 'real estate', 'land', 'title', 'ownership'],
        'commercial': ['commercial', 'business', 'corporation', 'partnership'],
        'tort': ['tort', 'negligence', 'injury', 'harm', 'wrongful']
    }
    
    for area, keywords in legal_area_keywords.items():
        if any(keyword in content for keyword in keywords):
            relationships['legal_areas'].append(area)
    
    # Extract key legal doctrines
    doctrine_patterns = [
        r'\b(doctrine of [a-z\s]+)\b',
        r'\b(principle of [a-z\s]+)\b',
        r'\b(rule of [a-z\s]+)\b'
    ]
    
    for pattern in doctrine_patterns:
        matches = re.findall(pattern, content)
        relationships['key_doctrines'].extend(matches)
    
    # Remove duplicates
    for key in relationships:
        relationships[key] = list(set(relationships[key]))
    
    return relationships


def _generate_enhanced_search_queries(case_type: str, legal_concepts: List[str], case_title: str, case_relationships: Dict[str, List[str]]) -> List[str]:
    """Generate diverse search queries for finding related cases using relationships"""
    queries = []
    
    # Case type based queries
    queries.extend([
        f"{case_type} cases",
        f"{case_type} law",
        f"similar {case_type} cases"
    ])
    
    # Legal concept based queries
    for concept in legal_concepts[:3]:  # Use top 3 concepts
        queries.append(f'"{concept}" cases')
        queries.append(f"cases involving {concept}")
    
    # Relationship-based queries
    # Cited cases queries
    for cited_case in case_relationships.get('cited_cases', [])[:2]:  # Use top 2 cited cases
        queries.append(f"cases citing {cited_case}")
        queries.append(f"similar to {cited_case}")
    
    # Legal area queries
    for legal_area in case_relationships.get('legal_areas', [])[:2]:  # Use top 2 legal areas
        queries.append(f"{legal_area} cases")
        queries.append(f"{legal_area} law")
    
    # Doctrine-based queries
    for doctrine in case_relationships.get('key_doctrines', [])[:2]:  # Use top 2 doctrines
        queries.append(f'"{doctrine}" cases')
        queries.append(f"cases applying {doctrine}")
    
    # Title-based queries
    if case_title and len(case_title) > 10:
        # Extract key terms from title
        title_words = [word for word in case_title.split() 
                      if len(word) > 3 and word.lower() not in ['case', 'versus', 'against', 'petitioner', 'respondent']]
        if title_words:
            queries.append(" ".join(title_words[:3]))
    
    # General legal area queries
    queries.extend([
        "Supreme Court decisions",
        "Philippine jurisprudence",
        "legal precedents"
    ])
    
    return queries[:8]  # Increased limit to 8 queries for better coverage


def _filter_and_score_cases(candidates: List[Dict], main_doc: Dict, legal_concepts: List[str], case_relationships: Dict[str, List[str]]) -> List[Dict]:
    """Filter and score candidate cases for relevance using relationships"""
    good_docs = []
    
    # Enhanced bad title patterns
    bad_title_patterns = [
                "docketed as", "against the advice", "ceased reporting", "came back only",
                "administrative case for", "crim. case no", "people v", "unknown", "third division",
                "chico-nazario", "honorable", "presiding judge", "just ceased", "came back only",
                "rommel", "baybay", "presiding judge", "this branch", "starting 6 april",
                "c-63250", "alex sabayan", "chico-nazario", "third division", "senate and the house",
                "congress commenced", "regular session", "commission on appointments", "constituted on",
        "august 2004", "july 2004", "164978", "g.r. no. g.r. no", "g.r. no. unknown",
        "e-library", "supreme court", "decision", "resolution", "order"
    ]
    
    for doc in candidates:
        # Extract and clean title
        title = (doc.get("metadata", {}).get("title", "") or 
                doc.get("title", "") or 
                doc.get("metadata", {}).get("case_title", ""))
        
        # Check for bad titles
        is_bad_title = any(pattern in title.lower() for pattern in bad_title_patterns)
        if is_bad_title or len(title.strip()) < 20:
            continue
        
        # Try to extract better title from content
        content = doc.get("content", "") or doc.get("text", "")
        if content:
            lines = content.split('\n')
            for line in lines[:15]:
                line = line.strip()
                if (len(line) > 30 and 
                    ('v.' in line or 'vs.' in line or 'versus' in line.lower()) and
                    not any(bad_word in line.lower() for bad_word in bad_title_patterns) and
                    not line.startswith(('—', 'Supreme Court', 'E-Library', 'The factual', 'On 02 August'))):
                    if 'metadata' not in doc:
                        doc['metadata'] = {}
                    doc['metadata']['title'] = line
                    doc['title'] = line
                    title = line
                    break
            
        # Calculate enhanced relevance score with relationships
        relevance_score = _calculate_enhanced_relevance_score(doc, main_doc, legal_concepts, case_relationships)
        doc['relevance_score'] = relevance_score
        
        # Only include cases with decent relevance
        if relevance_score > 0.3:
            good_docs.append(doc)
    
    return good_docs


def _calculate_enhanced_relevance_score(candidate_doc: Dict, main_doc: Dict, legal_concepts: List[str], case_relationships: Dict[str, List[str]]) -> float:
    """Calculate enhanced relevance score for a candidate case using relationships"""
    score = 0.0
    
    # Base score from vector similarity
    base_score = candidate_doc.get('score', 0.0)
    score += base_score * 0.3
    
    # Case type similarity
    main_case_type = _extract_enhanced_case_type(main_doc)
    candidate_case_type = _extract_enhanced_case_type(candidate_doc)
    if main_case_type == candidate_case_type:
        score += 0.25
    
    # Legal concept overlap
    candidate_content = (candidate_doc.get("content", "") or 
                        candidate_doc.get("text", "") or "").lower()
    concept_matches = sum(1 for concept in legal_concepts if concept.lower() in candidate_content)
    if concept_matches > 0:
        score += min(0.15, concept_matches * 0.03)
    
    # Relationship-based scoring
    candidate_gr = (candidate_doc.get("metadata", {}).get("gr_number", "") or 
                   candidate_doc.get("gr_number", ""))
    
    # Cited cases relationship (high bonus)
    cited_cases = case_relationships.get('cited_cases', [])
    if candidate_gr in cited_cases:
        score += 0.3  # High bonus for cited cases
    
    # Legal area overlap
    main_legal_areas = case_relationships.get('legal_areas', [])
    candidate_legal_areas = _extract_case_relationships(candidate_doc).get('legal_areas', [])
    area_overlap = len(set(main_legal_areas) & set(candidate_legal_areas))
    if area_overlap > 0:
        score += min(0.2, area_overlap * 0.1)
    
    # Doctrine overlap
    main_doctrines = case_relationships.get('key_doctrines', [])
    candidate_doctrines = _extract_case_relationships(candidate_doc).get('key_doctrines', [])
    doctrine_overlap = len(set(main_doctrines) & set(candidate_doctrines))
    if doctrine_overlap > 0:
        score += min(0.15, doctrine_overlap * 0.05)
    
    # Party name similarity (if available)
    main_parties = case_relationships.get('similar_parties', [])
    candidate_parties = _extract_case_relationships(candidate_doc).get('similar_parties', [])
    party_overlap = len(set(main_parties) & set(candidate_parties))
    if party_overlap > 0:
        score += min(0.1, party_overlap * 0.05)
    
    # Temporal relevance (recent cases get slight boost)
    candidate_year = candidate_doc.get("metadata", {}).get("promulgation_year")
    if candidate_year and isinstance(candidate_year, int):
        if candidate_year >= 2010:  # Recent cases
            score += 0.08
        elif candidate_year >= 2000:  # Moderately recent
            score += 0.04
    
    # Section type bonus (prefer rulings and dispositive sections)
    section_type = candidate_doc.get("section_type", "")
    if section_type in ["ruling", "dispositive", "summary"]:
        score += 0.08
    
    return min(1.0, score)  # Cap at 1.0


def _calculate_relevance_score(candidate_doc: Dict, main_doc: Dict, legal_concepts: List[str]) -> float:
    """Calculate relevance score for a candidate case (legacy function for compatibility)"""
    score = 0.0
    
    # Base score from vector similarity
    base_score = candidate_doc.get('score', 0.0)
    score += base_score * 0.4
    
    # Case type similarity
    main_case_type = _extract_enhanced_case_type(main_doc)
    candidate_case_type = _extract_enhanced_case_type(candidate_doc)
    if main_case_type == candidate_case_type:
        score += 0.3
    
    # Legal concept overlap
    candidate_content = (candidate_doc.get("content", "") or 
                        candidate_doc.get("text", "") or "").lower()
    concept_matches = sum(1 for concept in legal_concepts if concept.lower() in candidate_content)
    if concept_matches > 0:
        score += min(0.2, concept_matches * 0.05)
    
    # Temporal relevance (recent cases get slight boost)
    candidate_year = candidate_doc.get("metadata", {}).get("promulgation_year")
    if candidate_year and isinstance(candidate_year, int):
        if candidate_year >= 2010:  # Recent cases
            score += 0.1
        elif candidate_year >= 2000:  # Moderately recent
            score += 0.05
    
    # Section type bonus (prefer rulings and dispositive sections)
    section_type = candidate_doc.get("section_type", "")
    if section_type in ["ruling", "dispositive", "summary"]:
        score += 0.1
    
    return min(1.0, score)  # Cap at 1.0

def _generate_case_listing_summary(full_case: Dict) -> str:
    """Generate a brief 3-5 sentence summary for case listings"""
    if not TOGETHERAI_AVAILABLE:
        return _generate_simple_case_summary(full_case)
    
    try:
        # Extract key information from the case
        case_title = full_case.get("case_title", "Unknown Case")
        gr_number = full_case.get("gr_number", "")
        ponente = full_case.get("ponente", "")
        date = full_case.get("promulgation_date", "")
        
        # Get case content - use clean_text which actually exists
        clean_text = full_case.get("clean_text", "")
        
        # If no content, return simple summary
        if not clean_text or len(clean_text) < 100:
            return _generate_simple_case_summary(full_case)
        
        # Use first 1000 chars of clean_text as context
        case_content = clean_text[:1000]
        
        summary_prompt = f"""You are a Philippine Law expert. Generate a brief 2 sentence summary of this Supreme Court case for a case listing.

Case Information:
- Title: {case_title}
- G.R. Number: {gr_number}
- Ponente: {ponente}
- Date: {date}

Case Content:
{case_content}

Task: Write a concise 2 sentence summary that:
1. Briefly describes what the case is about
2. Mentions the key legal issue or controversy
3. States the main ruling or outcome
4. Keep it informative but concise for a case listing

Format: Write only the summary sentences, no additional text or formatting."""

        response = generate_messages_with_togetherai(
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=200,
            temperature=0.3
        )
        
        if response and len(response.strip()) > 20:
            return response.strip()
        else:
            return _generate_simple_case_summary(full_case)
            
    except Exception as e:
        print(f"⚠️ Failed to generate AI case summary: {e}")
        return _generate_simple_case_summary(full_case)

def _generate_simple_case_summary(full_case: Dict) -> str:
    """Generate a simple 3 sentence summary from available case data"""
    case_title = full_case.get("case_title", "")
    clean_text = full_case.get("clean_text", "")
    ponente = full_case.get("ponente", "")
    
    summary_parts = []
    
    # Try to extract meaningful sentences from clean_text
    if clean_text and len(clean_text) > 100:
        clean = clean_text.replace('\n', ' ').strip()
        # Get first few sentences
        sentences = [s.strip() for s in clean.split('.') if len(s.strip()) > 50]
        
        # Take first 2-3 meaningful sentences
        for sentence in sentences[:3]:
            # Skip meta sentences
            if not any(skip in sentence.lower() for skip in ['g.r. no.', 'petitioner', 'respondent', 'supreme court']):
                summary_parts.append(f"{sentence}.")
                if len(summary_parts) >= 2:
                    break
    
    # Add ponente if available
    if ponente and len(summary_parts) < 3:
        summary_parts.append(f"The decision was penned by Justice {ponente}.")
    
    # Return summary if we have something meaningful
    if summary_parts:
        return " ".join(summary_parts)
    else:
        # Last resort fallback
        return f"Supreme Court case regarding {case_title.split(',')[0] if ',' in case_title else 'a legal matter'}."

def _generate_single_case_summary(case_data: Tuple[int, Dict]) -> Tuple[int, str]:
    """Generate summary for a single case - used in parallel processing"""
    index, d = case_data
    
    case_summary = ""
    gr_raw = d.get("gr_number") or d.get("metadata", {}).get("gr_number") or ""
    special_raw = d.get("special_number") or d.get("metadata", {}).get("special_number") or ""
    
    if gr_raw or special_raw:
        try:
            from .retriever import load_case_from_jsonl
            lookup_id = str(gr_raw) if gr_raw else str(special_raw)
            full_case = load_case_from_jsonl(lookup_id)
            
            if full_case and isinstance(full_case, dict):
                case_summary = _generate_case_listing_summary(full_case)
        except Exception as e:
            print(f"⚠️ Failed to load case summary for {lookup_id}: {e}")
    
    return index, case_summary

def _generate_case_summaries_parallel(cases_data: List[Dict], max_workers: int = 5) -> List[str]:
    """Generate case summaries in parallel using ThreadPoolExecutor"""
    
    # Prepare case data for parallel processing
    case_data_tuples = [(i, case_data) for i, case_data in enumerate(cases_data)]
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(_generate_single_case_summary, case_tuple): i 
            for i, case_tuple in enumerate(case_data_tuples)
        }
        
        # Collect results in order
        summaries = [""] * len(cases_data)
        for future in concurrent.futures.as_completed(future_to_index):
            try:
                index, summary = future.result()
                summaries[index] = summary
            except Exception as e:
                print(f"⚠️ Error generating case summary: {e}")
                # Keep empty summary for failed cases
    
    return summaries

def _process_cases_with_parallel_summaries(picked: List[Dict]) -> List[str]:
    """Process cases with parallel summary generation"""
    
    # Generate summaries in parallel
    case_summaries = _generate_case_summaries_parallel(picked, max_workers=5)
    
    # Build final items
    items = []
    for i, (d, case_summary) in enumerate(zip(picked, case_summaries), 1):
        title = _display_title(d)
        
        # Get case number (GR or special)
        gr_raw = d.get("gr_number") or d.get("metadata", {}).get("gr_number") or ""
        special_raw = d.get("special_number") or d.get("metadata", {}).get("special_number") or ""
        
        if gr_raw:
            case_number = _normalize_gr_display(str(gr_raw))
        elif special_raw:
            case_number = str(special_raw)
        else:
            case_number = "Unknown"

        # Improve title if needed (from full case data if available)
        if case_summary and (title in {"Untitled case", ""} or title.startswith("Case (")):
            try:
                from .retriever import load_case_from_jsonl
                lookup_id = str(gr_raw) if gr_raw else str(special_raw)
                full_case = load_case_from_jsonl(lookup_id)
                if full_case and isinstance(full_case, dict):
                    t2 = full_case.get("case_title") or full_case.get("title")
                    if t2 and isinstance(t2, str) and len(t2.strip()) >= 10:
                        title = t2.strip()[:120]
            except Exception:
                pass  # Keep original title if improvement fails

        # Format the case number with clickable links
        formatted_case_number = ""
        if case_number != "Unknown":
            # Extract just the number part for the link
            if "G.R. No." in case_number:
                gr_match = re.search(r'G\.R\.\s+No\.\s+([0-9\-]+)', case_number)
                if gr_match:
                    gr_num = gr_match.group(1)
                    formatted_case_number = f"**[{case_number}](gr:{gr_num})**"
                else:
                    formatted_case_number = f"**({case_number})**"
            elif "A.M. No." in case_number:
                am_match = re.search(r'A\.M\.\s+No\.\s+([A-Z0-9\-]+)', case_number)
                if am_match:
                    am_num = am_match.group(1)
                    formatted_case_number = f"**[{case_number}](am:{am_num})**"
                else:
                    formatted_case_number = f"**({case_number})**"
            else:
                formatted_case_number = f"**({case_number})**"
        else:
            formatted_case_number = f"({case_number})"

        # Get date for display
        date = d.get("promulgation_date") or d.get("metadata", {}).get("promulgation_date", "")
        date_display = f" | {date}" if date else ""
        
        # Build the case listing item
        if case_number != "Unknown" and case_number in title:
            # Case number already in title
            item = f"{i}. **{title}{date_display}**"
        else:
            # Separate case number
            item = f"{i}. **{title}{date_display}** {formatted_case_number}"
        
        # Add case summary if available
        if case_summary:
            item += f"\n{case_summary}"
        
        items.append(item)
    
    return items

def _generate_case_summary_from_jsonl(full_case_content: Dict, query: str, retriever, history: List[Dict] = None) -> str:
    """Generate case summary directly from JSONL content using LLM"""
    print("🔍 Generating case summary from JSONL content using LLM...")
    
    # Extract metadata
    case_title = full_case_content.get("case_title", "")
    gr_number = full_case_content.get("gr_number", "")
    
    # Clean up G.R. number to avoid duplication
    if gr_number:
        # Remove "G.R. No." prefix if it exists to avoid duplication
        gr_number = re.sub(r'^G\.R\.\s+No\.\s*', '', gr_number, flags=re.IGNORECASE).strip()
        if gr_number:
            gr_number = f"G.R. No. {gr_number}"
        else:
            gr_number = "Not available"
    else:
        gr_number = "Not available"
    
    ponente = full_case_content.get("ponente", "")
    date = full_case_content.get("promulgation_date", "") or full_case_content.get("date", "")
    case_type = full_case_content.get("case_type", "")
    
    # SIMPLIFIED FULL-TEXT APPROACH: Feed entire clean_text to leverage 128k context length
    clean_text = full_case_content.get("clean_text", "")
    
    # Debug: Check what content we found
    print(f"🔍 Clean text length: {len(clean_text)}")
    print(f"🔍 Available fields: {list(full_case_content.keys())}")
    print(f"🔍 Case title: '{full_case_content.get('case_title', 'NONE')[:100]}...'")
    print(f"🔍 GR number: '{full_case_content.get('gr_number', 'NONE')}'")
    print(f"🔍 Special number: '{full_case_content.get('special_number', 'NONE')}'")
    
    # Build simple context with full case text
    if clean_text and len(clean_text.strip()) > 0:
        # Safety check: ensure we don't exceed 128k tokens (≈512k characters)
        max_chars = 500000  # Leave some buffer for prompt overhead
        if len(clean_text) > max_chars:
            print(f"⚠️ Case text too long ({len(clean_text)} chars), truncating to {max_chars}")
            clean_text = clean_text[:max_chars] + "... [TRUNCATED]"
        
        context = f"""CASE INFORMATION:
- Title: {case_title}
- G.R. Number: {gr_number}
- Date: {date}
- Ponente: {ponente}
- Case Type: {case_type}

FULL CASE TEXT:
{clean_text}"""
        
        print(f"✅ Full case text loaded ({len(clean_text)} characters)")
    else:
        context = "No detailed content available."
        print("❌ No clean_text found in case content")
    
    # Create a streamlined prompt for full-text case digest generation
    case_digest_prompt = f"""
        Create a comprehensive CASE DIGEST for a Philippine Supreme Court case using the FULL CASE TEXT provided below.
        Extract all information directly from the case text. If a detail is missing, write exactly: Not stated in sources.

        **CRITICAL INSTRUCTIONS:**
        - Use ONLY information from the provided case text
        - Quote the DISPOSITIVE verbatim (WHEREFORE/ACCORDINGLY/etc.) - do NOT paraphrase
        - Include ALL section headers exactly as shown below
        - Focus on SUBSTANTIVE FACTS (what actually happened) not procedural steps
        - Extract the complete dispositive portion without truncation

        **CASE TEXT:**
        {context}

        NOW PRODUCE the digest EXACTLY in this format (keep headers bold):
        
        **CRITICAL: You MUST include ALL section headers exactly as shown below. Do NOT skip any headers, especially **ISSUE/S:**.**

        **{case_title}**

        **{gr_number} | {date} | Ponente: {ponente}**

        **Nature:** [Nature of case, e.g., Petition for Review on Certiorari]

        **Topic:** [Legal topic, e.g., Original Document Rule]

        **Case Type:** [e.g., annulment, estafa, administrative matter, etc.]

        **Doctrine:**
        [One concise controlling principle derived from the Court’s reasoning in Context. Focus on the main legal rule resolving the dispute; avoid mere procedural trivia unless dispositive.]

        **Ticker/Summary:**
        [2–4 sentences: parties, core conflict, key procedural posture if shown, and the outcome direction.]

        **Petitioner/s:** [Exact name(s) as in Context] | **Respondent/s:** [Exact name(s) as in Context]

        **Facts:**
        1) [SUBSTANTIVE FACT: What actually happened between the parties - the real events, relationships, and circumstances that led to the legal dispute]
        2) [SUBSTANTIVE FACT: The core legal relationship or transaction that is the subject of the case]
        3) [SUBSTANTIVE FACT: Key events, actions, or circumstances that created the legal controversy]
        4) [Continue with substantive facts only - avoid procedural steps like "filed petition", "court denied", etc.]
        
        **CRITICAL FOR FACTS SECTION:**
        - Focus on WHAT HAPPENED between the parties, not court procedures
        - Include the actual events, relationships, transactions, or circumstances that led to the legal dispute
        - Avoid procedural facts like "filed petition", "court denied", "appealed"
        - Include substantive facts like marriages, contracts, injuries, transactions, relationships
        - Each fact should describe actual events or circumstances, not legal procedures
        
        **EXAMPLES OF GOOD FACTS:**
        - "The parties were married on [date] and lived together for [period]"
        - "The contract was executed on [date] for the sale of [property]"
        - "The accident occurred when [specific circumstances]"
        - "The employment relationship began when [specific events]"
        
        **EXAMPLES OF BAD FACTS (AVOID THESE):**
        - "Petitioner filed a petition" (procedural)
        - "The court denied the motion" (procedural)  
        - "Respondent appealed the decision" (procedural)

        **Petitioner's Contention:**
        1) [What the PETITIONER claims/argues]
        2) [...]
        3) [...]

        **Respondent's Contention:**
        1) [What the RESPONDENT(S) claim/argue]
        2) [...]
        3) [...]

        **RTC:** [e.g., IN FAVOR OF PETITIONER/RESPONDENT or Not stated in sources]
        - [Key reasoning if stated; else Not stated in sources]

        **CA:** [e.g., AFFIRMED/REVERSED or Not stated in sources]
        - [Key reasoning if stated; else Not stated in sources]

        **ISSUE/S:**
        WHETHER OR NOT [State the first legal issue]: [YES/NO]
        WHETHER OR NOT [State the second legal issue]: [YES/NO]
        WHETHER OR NOT [State the third legal issue]: [YES/NO]
        [Continue for all issues found in the case - each issue must start with "WHETHER OR NOT" and end with a clear YES or NO answer]
        
        **🚨 CRITICAL FORMATTING RULE FOR ISSUES - MANDATORY:**
        - You MUST start the issues section with **ISSUE/S:** (with colon)
        - Do NOT write issues without this header
        - Do NOT skip the **ISSUE/S:** header
        - The format must be exactly: **ISSUE/S:** followed by WHETHER OR NOT questions
        - FAILURE TO INCLUDE **ISSUE/S:** HEADER WILL RESULT IN INCORRECT FORMAT
        
        **EXAMPLE OF CORRECT ISSUES FORMAT:**
        **ISSUE/S:**
        WHETHER OR NOT the respondent judge committed grave abuse of discretion: YES
        WHETHER OR NOT the constitutional provision on bail applies to extradition proceedings: NO
        WHETHER OR NOT the right to bail is available in extradition proceedings: NO

        **SC RULING:**
        [3–5 sentences of analysis: what legal tests/rules were applied and why they led to the result. Do NOT restate the dispositive. No invented citations.]

        **DISPOSITIVE:** ["…verbatim final dispositive text from Context…" or Not stated in sources]
        
        **🚨 CRITICAL FOR DISPOSITIVE EXTRACTION:**
        - Search the FULL CASE TEXT for the complete "WHEREFORE...SO ORDERED" pattern
        - Extract the ENTIRE dispositive portion verbatim - do NOT truncate or summarize
        - Look for phrases like "WHEREFORE", "ACCORDINGLY", "IN VIEW OF THE FOREGOING"
        - Include all orders, directions, and specific instructions given by the Court
        - The dispositive typically appears near the end of the case text
        - Extract the complete text from "WHEREFORE" to "SO ORDERED" (or similar ending)
        
        **EXAMPLE OF PROPER DISPOSITIVE FORMAT:**
        **DISPOSITIVE:** "WHEREFORE, we DISMISS the petition. This case is REMANDED to the trial court to determine whether private respondent is entitled to bail on the basis of 'clear and convincing evidence.' If not, the trial court should order the cancellation of his bail bond and his immediate detention; and thereafter, conduct the extradition proceedings with dispatch. SO ORDERED."
        
        **FINAL VALIDATION CHECK:**
        Before submitting your response, verify that:
        1. You have included the **ISSUE/S:** header (with colon) before listing any issues
        2. All facts are substantive legal facts, not procedural steps
        3. You have included the complete **DISPOSITIVE** section with the full "WHEREFORE...SO ORDERED" text
        4. All sections follow the exact format specified above
        """
    try:
        from .generator import generate_conversational_response
        print(f"🔍 Calling LLM with prompt length: {len(case_digest_prompt)}")
        print(f"🔍 Prompt preview: {case_digest_prompt[:200]}...")
        
        raw_digest = generate_conversational_response(case_digest_prompt, history=history, context="", is_case_digest=True)
        print(f"🔍 Raw LLM response length: {len(raw_digest) if raw_digest else 0}")
        print(f"🔍 Raw LLM response: '{raw_digest[:200] if raw_digest else 'None'}...'")
        
        # Clean up the LLM output
        digest_text = raw_digest.strip() if raw_digest else ""
        
        # Remove common LLM artifacts
        digest_text = re.sub(r'^\[INST\].*?\[/INST\]\s*', '', digest_text, flags=re.DOTALL)
        digest_text = re.sub(r'^Assistant:\s*', '', digest_text)
        digest_text = re.sub(r'^Response:\s*', '', digest_text)
        digest_text = re.sub(r'^Summary:\s*', '', digest_text)
        digest_text = re.sub(r'^\[INST\].*?$', '', digest_text, flags=re.MULTILINE)
        digest_text = re.sub(r'^\[/INST\].*?$', '', digest_text, flags=re.MULTILINE)
        
        print(f"✅ Generated LLM case digest: {len(digest_text)} characters")
        print(f"🔍 Digest preview: {digest_text[:500]}...")
    except Exception as e:
        print(f"⚠️ LLM generation failed: {e}, using fallback")
        digest_text = f"**Facts**\nNot stated in sources.\n\n**Issues**\nNot stated in sources.\n\n**Ruling**\nNot stated in sources."
    
    # Filter out sections that are "Not stated in sources"
    filtered_lines = []
    skip_section = False
    current_section_header = None
    
    for line in digest_text.split('\n'):
        # Check if this is a section header (bold text)
        if line.strip().startswith('**') and line.strip().endswith('**'):
            # If we're currently skipping a section, don't include its header
            if skip_section:
                skip_section = False  # Reset for new section
                current_section_header = line  # Store the header but don't add it yet
                continue
            else:
                # This is a new section header
                current_section_header = line
                skip_section = False
                filtered_lines.append(line)
                continue
        
        # Check if the line says "Not stated in sources"
        if 'Not stated in sources' in line or 'not stated in sources' in line:
            skip_section = True
            # Remove the last section header if we just added it
            if filtered_lines and current_section_header and filtered_lines[-1] == current_section_header:
                filtered_lines.pop()
            continue  # Skip this line
        
        # If we're in a skipped section, skip until next header
        if skip_section:
            # Keep skipping until we hit a new section
            continue
        else:
            filtered_lines.append(line)
    
    digest_text = '\n'.join(filtered_lines)
    
    response_parts = [digest_text]
    
    # Add reference links to case sources
    source_url = full_case_content.get("source_url") if full_case_content else None
    response_parts.append("\n---")
    response_parts.append("📚 Source References:")
    if source_url:
        response_parts.append(f"[{source_url}]({source_url})")
    else:
        response_parts.append("[https://elibrary.judiciary.gov.ph/](https://elibrary.judiciary.gov.ph/)")
    response_parts.append("[https://lawphil.net/](https://lawphil.net/)")
    
    # response_parts.append("\nWhat would you like to know about this case?")
    # response_parts.append("• Case Digest - Complete structured summary")
    # response_parts.append("• Ruling - Court's decision and reasoning") 
    # response_parts.append("• Facts - Case background and events")
    # response_parts.append("• Issues - Legal questions raised")
    # response_parts.append("• Arguments - Legal reasoning and doctrines")
    
    final_response = "\n".join(response_parts)
    
    # Evaluate response if metrics are available
    if METRICS_AVAILABLE and full_case_content:
        try:
            reference_text = full_case_content.get("clean_text", "") or full_case_content.get("body", "")
            case_metadata_for_eval = {
                'case_title': case_title,
                'gr_number': gr_number,
                'ponente': ponente,
                'promulgation_date': date,
                'case_type': case_type
            }
            evaluate_response(query, final_response, reference_text, case_metadata_for_eval)
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
    
    return final_response

def _generate_ponente_response(docs: List[Dict], query: str, justice_name: str, retriever, history: List[Dict] = None) -> str:
    """Generate a response for ponente queries showing cases penned by a specific justice"""
    if not docs:
        return f"I couldn't find any cases penned by Justice {justice_name or 'the specified justice'}."
    
    response_parts = []
    
    # Header
    if justice_name:
        response_parts.append(f"## Cases Penned by Justice {justice_name}")
    else:
        response_parts.append("## Cases by Justice")
    
    response_parts.append("")
    
    # Show up to 5 cases with summaries
    max_cases = min(5, len(docs))
    
    for i, doc in enumerate(docs[:max_cases], 1):
        # Extract case information
        case_title = _display_title(doc)
        gr_number = doc.get("gr_number") or doc.get("metadata", {}).get("gr_number", "")
        ponente = doc.get("ponente") or doc.get("metadata", {}).get("ponente", "")
        date = doc.get("promulgation_date") or doc.get("metadata", {}).get("promulgation_date", "")
        
        # Get case content for summary generation
        case_content = doc.get("content", "") or doc.get("text", "")
        case_type = doc.get("case_type") or doc.get("metadata", {}).get("case_type", "")
        
        # Format case entry with basic info
        date_display = f" | {date}" if date else ""
        case_entry = f"**{i}. {case_title}{date_display}**"
        if gr_number:
            case_entry += f"\n- {gr_number}"
        if ponente:
            case_entry += f"\n- Ponente: {ponente}"
        if case_type:
            case_entry += f"\n- Type: {case_type.title()}"
        
        # Generate a brief summary if content is available
        if case_content and len(case_content) > 100:
            try:
                # Import the generator function
                from .generator import generate_conversational_response

                # Create a prompt for case summary
                summary_prompt = f"""
                Create a brief 2 sentence summary of this Philippine Supreme Court case. Focus on:
                1. The main parties involved
                2. The core legal issue or dispute
                3. The outcome or key ruling
                
                Case: {case_title}
                Content: {case_content[:2000]}...
                
                Provide only the summary, no additional formatting.
                """
                
                # Generate summary using the same LLM as other responses
                summary = generate_conversational_response(
                    summary_prompt, 
                    history=history or [], 
                    context="", 
                    is_case_digest=False
                )
                
                # Clean up the summary
                summary = summary.strip()
                if summary and not summary.startswith("I'm sorry") and not summary.startswith("I cannot"):
                    case_entry += f"\n- **Summary:** {summary}"
                    
            except Exception as e:
                print(f"⚠️ Failed to generate summary for case {i}: {e}")
                # Fallback: extract first few sentences from content
                if case_content:
                    sentences = case_content.split('.')[:2]
                    if sentences:
                        fallback_summary = '. '.join(sentences).strip() + '.'
                        if len(fallback_summary) < 200:
                            case_entry += f"\n- **Summary:** {fallback_summary}"
        
        response_parts.append(case_entry)
        response_parts.append("")
    
    # Add note about total cases found
    if len(docs) > max_cases:
        response_parts.append(f"*Note: Showing {max_cases} of {len(docs)} cases found.*")
    else:
        response_parts.append(f"*Found {len(docs)} case{'s' if len(docs) != 1 else ''}.*")
    
    # Add suggestion for more specific queries
    if justice_name:
        response_parts.append("")
        response_parts.append("💡 **Tip:** You can ask for more details about any specific case by mentioning its name or G.R. number.")
    
    return "\n".join(response_parts)

def chat_with_law_bot(query: str, history: List[Dict] = None):
    import time
    query_start_time = time.time()
    global retriever
    
    print(f"[QUERY] '{query}'")

    # Early handling: ponente queries bypass classification and generic reply paths
    if _is_ponente_query(query):
        print(f"[PONENTE] Early ponente handler for: '{query}'")
        justice_name = _extract_ponente_name_from_query(query)
        if justice_name:
            print(f"🎯 Extracted justice name: '{justice_name}'")
        retrieval_start = time.time()
        if retriever is None:
            try:
                from .retriever import LegalRetriever
                base_retriever = LegalRetriever(collection="jurisprudence", use_contextual_rag=True)
                print("✅ Using Contextual RAG Legal Retriever with jurisprudence collection")
                retriever = base_retriever
            except Exception as e:
                print(f"❌ Failed to load Contextual RAG Legal Retriever: {e}")
                retriever = None
        if retriever:
            try:
                surname = _extract_surname(justice_name) if justice_name else ""
                # Use broader search patterns for better coverage
                search_queries = []
                if justice_name and len(justice_name.split()) > 1:
                    # Full name search
                    search_queries.append(f"ponente {justice_name}")
                    search_queries.append(justice_name)
                if surname:
                    # Surname-based searches
                    search_queries.append(f"ponente {surname}")
                    search_queries.append(surname)
                    search_queries.append(f"{surname}, J.")
                    search_queries.append(f"{surname}, JJ.")
                
                # Try multiple search queries and combine results
                all_docs = []
                for sq in search_queries[:4]:  # Increased to 4 queries for better coverage
                    print(f"🔍 Searching with: '{sq}'")
                    docs = retriever.retrieve(sq, k=50, conversation_history=history)  # Increased from 30 to 50
                    if docs:
                        all_docs.extend(docs)
                
                # Remove duplicates based on case title
                seen_titles = set()
                unique_docs = []
                for doc in all_docs:
                    title = _display_title(doc)
                    if title not in seen_titles:
                        seen_titles.add(title)
                        unique_docs.append(doc)
                
                docs = unique_docs
                filtered_docs: List[Dict] = []
                if docs:
                    for doc in docs:
                        doc_ponente = ((doc.get("metadata", {}) or {}).get("ponente", "") or doc.get("ponente", ""))
                        if _ponente_matches(doc_ponente, justice_name):
                            filtered_docs.append(doc)
                if not filtered_docs and justice_name:
                    print(f"❌ No direct matches. Retrying with surname variants...")
                    variants = []
                    if surname:
                        # More comprehensive variants
                        variants = [
                            f"{surname}, J.", f"{surname}, JJ.", f"{surname} J.", 
                            f"ponente {surname}", surname, f"justice {surname}",
                            f"{surname}-Javier", f"{surname}-Javier, J.", f"{surname}-Javier, JJ.",
                            f"ponente {surname}-Javier", f"justice {surname}-Javier"
                        ]
                        # Add common variations
                        if 'lazaro' in surname.lower():
                            variants.extend([f"lazaro javier", f"lazaro-javier", f"ponente lazaro javier"])
                    
                    for vq in variants:
                        print(f"🔁 Retry ponente search with: '{vq}'")
                        docs_retry = retriever.retrieve(vq, k=50, conversation_history=history)  # Increased from 30 to 50
                        if not docs_retry:
                            continue
                        for doc in docs_retry:
                            doc_ponente = ((doc.get("metadata", {}) or {}).get("ponente", "") or doc.get("ponente", ""))
                            if _ponente_matches(doc_ponente, justice_name):
                                filtered_docs.append(doc)
                        if filtered_docs:
                            break
                if filtered_docs:
                    generation_start = time.time()
                    result = _generate_ponente_response(filtered_docs, query, justice_name, retriever, history)
                    generation_time = time.time() - generation_start
                    print(f"⏱️ Ponente response generation took {generation_time:.2f}s")
                    retrieval_time = time.time() - retrieval_start
                    total_time = time.time() - query_start_time
                    print(f"⏱️ Retrieval took {retrieval_time:.2f}s")
                    print(f"⏱️ Total query time: {total_time:.2f}s")
                    return result
                else:
                    print(f"❌ Still no cases found for ponente after retries")
            except Exception as e:
                print(f"⚠️ Ponente early search failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Step 1: Use TogetherAI to classify and potentially handle the query directly
    classification_start = time.time()
    ai_response = _classify_and_handle_query(query, history)
    classification_time = time.time() - classification_start
    print(f"⏱️ Classification took {classification_time:.2f}s")
    
    if ai_response is not None:
        # If this is a ponente query, bypass the generic LLM reply and proceed to ponente handler
        if _is_ponente_query(query):
            print("[PONENTE] Overriding classifier output to run ponente handler")
        # Otherwise, handle classified actions or return the LLM reply
        else:
            # Check if it's a web search request
            if ai_response == "[WEB_SEARCH]":
                print(f"[WEB_SEARCH] Proceeding with web search for: '{query}'")
                response = _handle_web_search_query(query, history)
                total_time = time.time() - query_start_time
                print(f"⏱️ Total query time: {total_time:.2f}s")
                return response
            else:
                # Regular greeting/general response
                total_time = time.time() - query_start_time
                print(f"⏱️ Total query time: {total_time:.2f}s")
                return ai_response
    
        # Step 2: Check if this is a ponente query
        if _is_ponente_query(query):
            print(f"[PONENTE] Proceeding with ponente search for: '{query}'")
            justice_name = _extract_ponente_name_from_query(query)
            if justice_name:
                print(f"🎯 Extracted justice name: '{justice_name}'")
            
            # Initialize timing for ponente search
            retrieval_start = time.time()
            
            # Initialize retriever if needed
            if retriever is None:
                try:
                    from .retriever import LegalRetriever
                    base_retriever = LegalRetriever(collection="jurisprudence", use_contextual_rag=True)
                    print("✅ Using Contextual RAG Legal Retriever with jurisprudence collection")
                    retriever = base_retriever
                except Exception as e:
                    print(f"❌ Failed to load Contextual RAG Legal Retriever: {e}")
                    retriever = None
            
            if retriever:
                try:
                    # For ponente queries, search for cases by the specific justice
                    print(f"🔍 Searching for cases penned by Justice {justice_name or 'Unknown'}")
                    
                    # Use the justice name in the search query; prefer surname when present
                    surname = _extract_surname(justice_name) if justice_name else ""
                    search_query = (
                        f"ponente {justice_name}" if (justice_name and len(justice_name.split()) > 1)
                        else (f"ponente {surname}" if surname else query)
                    )
                    docs = retriever.retrieve(search_query, k=20, conversation_history=history)
                    
                    if docs:
                        print(f"✅ Found {len(docs)} cases penned by Justice {justice_name or 'Unknown'}")
                        
                        # Filter results to only include cases actually penned by this justice
                        filtered_docs = []
                        for doc in docs:
                            doc_ponente = (
                                (doc.get("metadata", {}) or {}).get("ponente", "")
                                or doc.get("ponente", "")
                            )
                            if _ponente_matches(doc_ponente, justice_name):
                                filtered_docs.append(doc)
                        
                        if filtered_docs:
                            print(f"✅ Found {len(filtered_docs)} cases actually penned by Justice {justice_name or 'Unknown'}")
                            # Generate response for ponente query
                            generation_start = time.time()
                            result = _generate_ponente_response(filtered_docs, query, justice_name, retriever, history)
                            generation_time = time.time() - generation_start
                            print(f"⏱️ Ponente response generation took {generation_time:.2f}s")
                            retrieval_time = time.time() - retrieval_start
                            total_time = time.time() - query_start_time
                            print(f"⏱️ Retrieval took {retrieval_time:.2f}s")
                            print(f"⏱️ Total query time: {total_time:.2f}s")
                            return result
                        else:
                            print(f"❌ No cases found penned by Justice {justice_name or 'Unknown'} — retrying with surname variants")
                            # Retry with surname and common justice suffix formats (surname-only metadata scenario)
                            variants = []
                            if justice_name:
                                sname = _extract_surname(justice_name)
                                if sname:
                                    variants = [
                                        f"{sname}, J.",
                                        f"{sname}, JJ.",
                                        f"{sname} J.",
                                        f"ponente {sname}",
                                        sname,
                                    ]
                                    # If surname may have a hyphenated canonical form, try with common hyphen additions
                                    # e.g., Lazaro -> Lazaro-Javier (best effort)
                                    if '-' not in sname:
                                        variants.append(f"{sname}-Javier, J.")
                            retried = False
                            for vq in variants:
                                print(f"🔁 Retry ponente search with: '{vq}'")
                                docs_retry = retriever.retrieve(vq, k=30, conversation_history=history)
                                if not docs_retry:
                                    continue
                                filtered_retry = []
                                for doc in docs_retry:
                                    doc_ponente = ((doc.get("metadata", {}) or {}).get("ponente", "") or doc.get("ponente", ""))
                                    if _ponente_matches(doc_ponente, justice_name):
                                        filtered_retry.append(doc)
                                if filtered_retry:
                                    retried = True
                                    print(f"✅ Retry succeeded with {len(filtered_retry)} matches")
                                    generation_start = time.time()
                                    result = _generate_ponente_response(filtered_retry, query, justice_name, retriever, history)
                                    generation_time = time.time() - generation_start
                                    print(f"⏱️ Ponente response generation took {generation_time:.2f}s")
                                    retrieval_time = time.time() - retrieval_start
                                    total_time = time.time() - query_start_time
                                    print(f"⏱️ Retrieval took {retrieval_time:.2f}s")
                                    print(f"⏱️ Total query time: {total_time:.2f}s")
                                    return result
                            if not retried:
                                print(f"❌ Still no cases found for ponente after retries")
                    else:
                        print(f"❌ No cases found for ponente query")
                        
                except Exception as e:
                    print(f"⚠️ Ponente search failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Step 3: Check if this is a case digest request
        elif _is_case_digest_request(query):
            print(f"[CASE_DIGEST] Proceeding with focused case digest for: '{query}'")
            case_name = _extract_case_name_from_digest_request(query)
            if case_name:
                print(f"🎯 Extracted case name: '{case_name}'")
            
            # Initialize timing for case digest
            retrieval_start = time.time()
            
            # Initialize retriever if needed
            if retriever is None:
                try:
                    from .retriever import LegalRetriever
                    base_retriever = LegalRetriever(collection="jurisprudence", use_contextual_rag=True)
                    print("✅ Using Contextual RAG Legal Retriever with jurisprudence collection")
                    retriever = base_retriever
                except Exception as e:
                    print(f"❌ Failed to load Contextual RAG Legal Retriever: {e}")
                    retriever = None
            
            if retriever:
                try:
                    # For case digest requests, focus on finding the specific case
                    # Use a more targeted search approach
                    print(f"🔍 Searching for specific case: '{case_name or query}'")
                    
                    # Try to find the case using the extracted name or original query
                    search_query = case_name if case_name else query
                    docs = retriever.retrieve(search_query, k=20, is_case_digest=True, conversation_history=history)
                    
                    if docs:
                        print(f"✅ Found {len(docs)} potential matches")
                        
                        # For case digest, we want to focus on the best match
                        # Look for exact title matches first
                        best_match = None
                        for doc in docs:
                            case_title = doc.get("title") or doc.get("metadata", {}).get("title") or ""
                            if case_name and case_name.lower() in case_title.lower():
                                best_match = doc
                                print(f"🎯 Found exact title match: '{case_title[:100]}...'")
                                break
                        
                        # If no exact match, use the highest scored result
                        if not best_match and docs:
                            best_match = docs[0]
                            print(f"🎯 Using highest scored result: '{best_match.get('title', 'Unknown')[:100]}...'")
                        
                        if best_match:
                            # Extract GR number and load full case content
                            gr_number = best_match.get("gr_number") or best_match.get("metadata", {}).get("gr_number")
                            if gr_number:
                                print(f"🔍 Loading full case content for G.R. No. {gr_number}")
                                try:
                                    from .retriever import load_case_from_jsonl
                                    full_case = load_case_from_jsonl(gr_number)
                                    if full_case:
                                        print(f"✅ Full case content loaded for G.R. No. {gr_number}")
                                        generation_start = time.time()
                                        result = _generate_case_summary_from_jsonl(full_case, query, retriever, history)
                                        generation_time = time.time() - generation_start
                                        print(f"⏱️ Case digest generation took {generation_time:.2f}s")
                                        retrieval_time = time.time() - retrieval_start
                                        total_time = time.time() - query_start_time
                                        print(f"⏱️ Retrieval took {retrieval_time:.2f}s")
                                        print(f"⏱️ Total query time: {total_time:.2f}s")
                                        return result
                                    else:
                                        print(f"❌ No full case content found for G.R. No. {gr_number}")
                                except Exception as e:
                                    print(f"⚠️ Failed to load full case content: {e}")
                            else:
                                print(f"❌ No GR number found in best match")
                        else:
                            print(f"❌ No suitable case found for digest request")
                            return "I'm sorry, but I couldn't find a suitable case for your digest request. Please try searching for a specific case by G.R. number or case name, or search for cases by topic."
                    else:
                        print(f"❌ No cases found for digest request")
                        return "I'm sorry, but I couldn't find any cases matching your digest request. Please try searching for a specific case by G.R. number or case name, or search for cases by topic."
                        
                except Exception as e:
                    print(f"⚠️ Case digest retrieval failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return "I encountered an error while searching for the case digest. Please try again or rephrase your request."
    
    # Step 3: Query needs case search - proceed with RAG retrieval
    retrieval_start = time.time()
    print(f"[SEARCH] Proceeding with case retrieval for: '{query}'")
    
    # Check if this is a follow-up question and extract case context
    if _is_follow_up_question(query):
        print(f"🔄 Follow-up question detected: '{query}'")
        case_info = _extract_case_from_history(history)
        if case_info:
            print(f"🎯 Found case context: {case_info}")
            # If we found a GR number, try to get more detailed information
            if "gr_number" in case_info:
                gr_num = case_info["gr_number"]
                print(f"🔍 Loading detailed case for follow-up: G.R. No. {gr_num}")
                try:
                    from .retriever import load_case_from_jsonl
                    full_case = load_case_from_jsonl(gr_num)
                    if full_case:
                        # Generate more detailed response for follow-up question
                        result = _generate_detailed_case_response(full_case, query, history)
                        return result
                except Exception as e:
                    print(f"⚠️ Failed to load case for follow-up: {e}")

    if retriever is None:
            try:
                from .retriever import LegalRetriever

                # Always use Contextual RAG - no fallback to basic retriever
                base_retriever = LegalRetriever(collection="jurisprudence", use_contextual_rag=True)
                print("✅ Using Contextual RAG Legal Retriever with jurisprudence collection")
                retriever = base_retriever
            except Exception as e:
                print(f"❌ Failed to load Contextual RAG Legal Retriever: {e}")
                retriever = None

    try:
            # Simplified routing: GR-number exact search vs special number vs keyword top-3
            gr_num = _extract_gr_number(query)
            special_num = _extract_special_number(query)
            docs = []
            if gr_num:
                # Direct exact GR number metadata search
                print(f"🎯 GR-number path: {gr_num}")
                docs = retriever._retrieve_by_gr_number(gr_num, k=8)
                print(f"🔍 Docs returned: {len(docs) if docs else 0} items")
                if docs:
                    print(f"🔍 First doc keys: {list(docs[0].keys()) if docs else 'None'}")
                wants_digest = True  # enforce digest format for GR path
                
                # For GR number path, always use JSONL for full content
                # Vector DB only has metadata, JSONL has the actual case content
                if docs:
                    print(f"🔄 Found metadata for GR {gr_num}, fetching full content from JSONL")
                    try:
                        from .retriever import load_case_from_jsonl
                        full_case = load_case_from_jsonl(gr_num)
                        print(f"🔍 JSONL load result: {type(full_case)} - {bool(full_case)}")
                        if full_case:
                            print(f"✅ JSONL content loaded for GR {gr_num}")
                            generation_start = time.time()
                            result = _generate_case_summary_from_jsonl(full_case, query, retriever, history)
                            generation_time = time.time() - generation_start
                            print(f"⏱️ Case digest generation took {generation_time:.2f}s")
                            print(f"🔍 Generated summary: {type(result)} - {bool(result)}")
                            retrieval_time = time.time() - retrieval_start
                            total_time = time.time() - query_start_time
                            print(f"⏱️ Retrieval took {retrieval_time:.2f}s")
                            print(f"⏱️ Total query time: {total_time:.2f}s")
                            return result
                        else:
                            print(f"❌ No JSONL data found for GR {gr_num}")
                            return f"I found metadata for G.R. No. {gr_num} but couldn't retrieve the full case content. The case may be incomplete in our database. Please try searching for cases by topic or party names instead."
                    except Exception as e:
                        print(f"⚠️ JSONL loading failed for GR {gr_num}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"❌ No metadata found for GR {gr_num} in vector DB")
                    return f"I'm sorry, but I couldn't find G.R. No. {gr_num} in our database. This case may not be available in our current collection, or the G.R. number might be incorrect. Please verify the G.R. number and try again, or search for cases by topic or party names."
            elif special_num:
                # Direct exact special number metadata search
                print(f"🎯 Special-number path: {special_num}")
                docs = retriever._retrieve_by_special_number(special_num, k=8)
                wants_digest = True  # enforce digest format for special number path
                
                # Debug: Show metadata found
                if docs:
                    first_doc = docs[0]
                    print(f"🔍 Metadata fields: {list(first_doc.keys())}")
                    print(f"🔍 Case title: '{first_doc.get('title', '')[:100]}...'")
                    print(f"🔍 GR number: {first_doc.get('gr_number', '')}")
                    print(f"🔍 Case type: {first_doc.get('case_type', '')}")
                
                # For special number path, try to load from JSONL using special number
                if docs:
                    print(f"🔄 Found metadata for Special {special_num}, fetching full content from JSONL")
                    print(f"[SEARCH] Debug - Special number extracted: '{special_num}'")
                    try:
                        from .retriever import load_case_from_jsonl

                        # Try loading by special number (may need to implement this in retriever)
                        full_case = load_case_from_jsonl(special_num)
                        print(f"[SEARCH] Debug - Case loaded: {full_case is not None}")
                        if full_case:
                            print(f"✅ JSONL content loaded for Special {special_num}")
                            print(f"[SEARCH] Debug - Case title: '{full_case.get('case_title', 'NONE')[:100]}...'")
                            print(f"[SEARCH] Debug - Available fields: {list(full_case.keys())}")
                            generation_start = time.time()
                            result = _generate_case_summary_from_jsonl(full_case, query, retriever, history)
                            generation_time = time.time() - generation_start
                            print(f"⏱️ Case digest generation took {generation_time:.2f}s")
                            retrieval_time = time.time() - retrieval_start
                            total_time = time.time() - query_start_time
                            print(f"⏱️ Retrieval took {retrieval_time:.2f}s")
                            print(f"⏱️ Total query time: {total_time:.2f}s")
                            return result
                        else:
                            print(f"❌ No JSONL data found for Special {special_num}")
                            return f"I found metadata for Special Number {special_num} but couldn't retrieve the full case content. The case may be incomplete in our database. Please try searching for cases by topic or party names instead."
                    except Exception as e:
                        print(f"⚠️ JSONL loading failed for Special {special_num}: {e}")
                else:
                    print(f"❌ No metadata found for Special {special_num} in vector DB")
                    return f"I'm sorry, but I couldn't find Special Number {special_num} in our database. This case may not be available in our current collection, or the special number might be incorrect. Please verify the special number and try again, or search for cases by topic or party names."
            else:
                # Smart keyword path: prioritize exact case title matches, then find related cases
                print(f"🎯 Smart keyword path: '{query}'")
                
                # Extract case title components for smart matching
                query_components = _extract_case_title_components(query)
                print(f"🔍 Query components: {query_components}")
                
                # Retrieve more candidates for better matching
                result = _advanced_retrieve(retriever, query, k=20, is_case_digest=False, history=history)
                hits = result[0] if isinstance(result, tuple) else result
                
                # Score and sort by title similarity + original score
                scored_cases = []
                for d in hits:
                    # Get case title
                    case_title = d.get("title") or d.get("metadata", {}).get("title") or ""
                    
                    # Calculate title similarity score
                    title_score = _calculate_title_similarity(query_components, case_title)
                    
                    # Get original retrieval score
                    original_score = d.get("score", 0.0)
                    
                    # Check if this is from Contextual RAG (high scores indicate Contextual RAG)
                    is_contextual_rag = original_score > 50.0 or d.get('contextual', False)
                    
                    if is_contextual_rag:
                        # For Contextual RAG results, trust the original score more
                        # Only use title similarity as a tiebreaker
                        combined_score = original_score + (title_score * 0.1)
                        print(f"🎯 Contextual RAG case: '{case_title[:30]}...' - Original: {original_score:.2f}, Title: {title_score:.2f}, Combined: {combined_score:.2f}")
                    else:
                        # For regular retrieval, use the original scoring logic
                        combined_score = (title_score * 0.7) + (original_score * 0.3)
                        print(f"📋 Regular case: '{case_title[:30]}...' - Original: {original_score:.2f}, Title: {title_score:.2f}, Combined: {combined_score:.2f}")
                    
                    d['title_similarity_score'] = title_score
                    d['combined_score'] = combined_score
                    d['is_contextual_rag'] = is_contextual_rag
                    scored_cases.append(d)
                
                # Sort by combined score (highest first)
                scored_cases.sort(key=lambda x: x['combined_score'], reverse=True)
                
                # Pick top 3-5 unique cases by GR/title
                picked = []
                seen_keys = set()
                for d in scored_cases:
                    # Use GR number, special number, or title as unique key
                    gr_num = d.get("gr_number") or d.get("metadata", {}).get("gr_number")
                    special_num = d.get("special_number") or d.get("metadata", {}).get("special_number")
                    title = d.get("title") or d.get("metadata", {}).get("title")
                    key = gr_num or special_num or title
                    
                    if key and key not in seen_keys:
                        seen_keys.add(key)
                        picked.append(d)
                        print(f"✅ Picked case: {title[:50]}... (score: {d['combined_score']:.2f})")
                        print(f"   Title source: title='{d.get('title', 'None')[:30]}', case_title='{d.get('case_title', 'None')[:30]}'")
                        if len(picked) >= 5:  # Return top 3-5 cases
                            break
                if picked:
                    print(f"🎯 Found {len(picked)} relevant cases")
                    print(f"🚀 Generating case summaries in parallel...")
                    
                    # Process cases with parallel summary generation
                    items = _process_cases_with_parallel_summaries(picked)
                    
                    result = "Here are the possible cases:\n" + "\n".join(items)
                    
                    print(f"🎯 Returning case list with {len(items)} items")
                    print(f"🎯 First item: {items[0] if items else 'None'}")
                    retrieval_time = time.time() - retrieval_start
                    total_time = time.time() - query_start_time
                    print(f"⏱️ Retrieval took {retrieval_time:.2f}s")
                    print(f"⏱️ Total query time: {total_time:.2f}s")
                    return result
                else:
                    result = "I couldn't find matching cases in the current database. Try a different G.R. number or broader keywords."
                    result += "\n\n---\n"
                    result += "📚 Source References:\n"
                    result += "[https://elibrary.judiciary.gov.ph/](https://elibrary.judiciary.gov.ph/)\n"
                    result += "[https://lawphil.net/](https://lawphil.net/)"
                    total_time = time.time() - query_start_time
                    print(f"⏱️ Total query time: {total_time:.2f}s")
                    return result
    except Exception as e:
        print(f"[WARNING] Jurisprudence retrieval failed: {e}")
        result = "I encountered a retrieval error. Please try again."
        result += "\n\n---\n"
        result += "📚 Source References:\n"
        result += "[https://elibrary.judiciary.gov.ph/](https://elibrary.judiciary.gov.ph/)\n"
        result += "[https://lawphil.net/](https://lawphil.net/)"
        total_time = time.time() - query_start_time
        print(f"⏱️ Total query time: {total_time:.2f}s")
        return result


def chat_with_law_bot_streaming(query: str, history: List[Dict] = None):
    """Streaming version of chat_with_law_bot with progress updates"""
    global retriever
    
    print(f"[STREAMING QUERY] '{query}'")
    
    # Step 1: Yield progress - Query classification
    yield {'type': 'progress', 'stage': 'classifying', 'message': 'Analyzing your query...'}
    
    # Use TogetherAI to classify and potentially handle the query directly
    ai_response = _classify_and_handle_query(query, history)
    if ai_response is not None:
        # Check if it's a web search request
        if ai_response == "[WEB_SEARCH]":
            yield {'type': 'progress', 'stage': 'web_search', 'message': 'Searching the web...'}
            response = _handle_web_search_query(query, history)
            # Stream the response word by word
            for word in response.split():
                yield word + " "
            return
        else:
            # Regular greeting/general response - stream it
            yield {'type': 'progress', 'stage': 'generating', 'message': 'Generating response...'}
            for word in ai_response.split():
                yield word + " "
            return
    
    # Step 2: Query needs case search - proceed with RAG retrieval
    yield {'type': 'progress', 'stage': 'retrieval_start', 'message': 'Searching legal database...'}
    
    # Initialize retriever if needed
    if retriever is None:
        yield {'type': 'progress', 'stage': 'init_retriever', 'message': 'Initializing search engine...'}
        try:
            from .retriever import LegalRetriever
            base_retriever = LegalRetriever(collection="jurisprudence", use_contextual_rag=True)
            print("✅ Using Contextual RAG Legal Retriever")
            retriever = base_retriever
        except Exception as e:
            print(f"❌ Failed to load retriever: {e}")
            yield "I encountered an error initializing the search engine. Please try again."
            return

    try:
        # Extract GR/special numbers
        gr_num = _extract_gr_number(query)
        special_num = _extract_special_number(query)
        
        if gr_num:
            # GR number path
            yield {'type': 'progress', 'stage': 'gr_lookup', 'message': f'Looking up G.R. No. {gr_num}...'}
            docs = retriever._retrieve_by_gr_number(gr_num, k=8)
            
            if docs:
                yield {'type': 'progress', 'stage': 'loading_full_case', 'message': 'Loading complete case details...'}
                try:
                    from .retriever import load_case_from_jsonl
                    full_case = load_case_from_jsonl(gr_num)
                    if full_case:
                        yield {'type': 'progress', 'stage': 'generating_digest', 'message': 'Generating case digest...'}
                        result = _generate_case_summary_from_jsonl(full_case, query, retriever, history)
                        for word in result.split():
                            yield word + " "
                        return
                except Exception as e:
                    print(f"⚠️ JSONL loading failed: {e}")
            
            yield "Case not found in database."
            return
            
        elif special_num:
            # Special number path
            yield {'type': 'progress', 'stage': 'special_lookup', 'message': f'Looking up {special_num}...'}
            docs = retriever._retrieve_by_special_number(special_num, k=8)
            
            if docs:
                yield {'type': 'progress', 'stage': 'loading_full_case', 'message': 'Loading complete case details...'}
                try:
                    from .retriever import load_case_from_jsonl
                    full_case = load_case_from_jsonl(special_num)
                    if full_case:
                        yield {'type': 'progress', 'stage': 'generating_digest', 'message': 'Generating case digest...'}
                        result = _generate_case_summary_from_jsonl(full_case, query, retriever, history)
                        for word in result.split():
                            yield word + " "
                        return
                except Exception as e:
                    print(f"⚠️ JSONL loading failed: {e}")
            
            yield "Case not found in database."
            return
        else:
            # Keyword search path
            yield {'type': 'progress', 'stage': 'semantic_search', 'message': 'Performing semantic search...'}
            
            # Extract case title components
            query_components = _extract_case_title_components(query)
            
            yield {'type': 'progress', 'stage': 'retrieving', 'message': 'Finding relevant cases...'}
            result = _advanced_retrieve(retriever, query, k=20, is_case_digest=False, history=history)
            hits = result[0] if isinstance(result, tuple) else result
            
            if not hits:
                yield "No matching cases found."
                return
            
            yield {'type': 'progress', 'stage': 'ranking', 'message': f'Ranking {len(hits)} candidates...'}
            
            # Score and sort by title similarity
            scored_cases = []
            for d in hits:
                case_title = d.get("title") or d.get("metadata", {}).get("title") or ""
                title_score = _calculate_title_similarity(query_components, case_title)
                original_score = d.get("score", 0.0)
                is_contextual_rag = original_score > 50.0 or d.get('contextual', False)
                
                if is_contextual_rag:
                    combined_score = original_score + (title_score * 0.1)
                else:
                    combined_score = (title_score * 0.7) + (original_score * 0.3)
                
                d['title_similarity_score'] = title_score
                d['combined_score'] = combined_score
                d['is_contextual_rag'] = is_contextual_rag
                scored_cases.append(d)
            
            scored_cases.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Pick top 5 cases
            picked = []
            seen_keys = set()
            for d in scored_cases:
                gr_num = d.get("gr_number") or d.get("metadata", {}).get("gr_number")
                special_num = d.get("special_number") or d.get("metadata", {}).get("special_number")
                title = d.get("title") or d.get("metadata", {}).get("title")
                key = gr_num or special_num or title
                
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    picked.append(d)
                    if len(picked) >= 5:
                        break
            
            if not picked:
                yield "No matching cases found."
                return
            
            yield {'type': 'progress', 'stage': 'formatting', 'message': f'Generating summaries for {len(picked)} cases in parallel...'}
            
            # Process cases with parallel summary generation
            items = _process_cases_with_parallel_summaries(picked)
            
            # Stream the formatted response
            result = "Here are the possible cases:\n" + "\n".join(items)
            for word in result.split():
                yield word + " "
            return
            
    except Exception as e:
        print(f"[ERROR] Streaming retrieval failed: {e}")
        yield f"I encountered an error: {str(e)}"
        return
