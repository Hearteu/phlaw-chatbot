# chat_engine.py — Simplified Law LLM chat engine with chunking support
import concurrent.futures
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Import evaluation metrics (configurable)
ENABLE_EVALUATION_METRICS = os.getenv("ENABLE_EVALUATION_METRICS", "false").lower() == "true"
if ENABLE_EVALUATION_METRICS:
    try:
        from .evaluation_metrics import (AutomatedContentScoring,
                                         ContentRelevanceMetrics,
                                         LegalAccuracyMetrics)
        METRICS_AVAILABLE = True
    except ImportError:
        METRICS_AVAILABLE = False
        print("⚠️ Evaluation metrics not available")
else:
    METRICS_AVAILABLE = False
    print("ℹ️ Evaluation metrics disabled via configuration")

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


def _initialize_retriever():
    """Initialize retriever if not already loaded"""
    global retriever
    if retriever is None:
        try:
            from .retriever import LegalRetriever
            base_retriever = LegalRetriever(collection="jurisprudence")
            print("✅ Using Contextual RAG Legal Retriever with jurisprudence collection")
            retriever = base_retriever
        except Exception as e:
            print(f"❌ Failed to load Contextual RAG Legal Retriever: {e}")
            retriever = None
    return retriever

def _generate_detailed_case_response(full_case: Dict, query: str, history: Optional[List[Dict]] = None, focus_areas: Optional[str] = None, template_mode: str = "full") -> str:
    """Generate a detailed response for follow-up questions about a specific case"""
    if not TOGETHERAI_AVAILABLE:
        return _generate_case_summary_from_jsonl(full_case, query, None, history, template_mode)
    
    try:
        # Extract case metadata
        case_title = full_case.get("case_title", "Unknown Case")
        gr_number = full_case.get("gr_number", "Unknown")
        ponente = full_case.get("ponente", "Unknown")
        date = full_case.get("promulgation_date", "Unknown")
        clean_text = full_case.get("clean_text", "")
        
        # Use focus information from AI classifier
        
        if focus_areas:
            # Use AI-determined focus
            if "facts" in focus_areas:
                focus = "facts and factual background"
                instruction = "Focus specifically on the factual background, events, and circumstances of the case."
            elif "issues" in focus_areas:
                focus = "legal issues and questions"
                instruction = "Focus specifically on the legal issues, questions, and problems raised in the case."
            elif "ruling" in focus_areas:
                focus = "ruling and decision"
                instruction = "Focus specifically on the court's ruling, decision, and legal reasoning."
            elif "principles" in focus_areas:
                focus = "legal principles and doctrines"
                instruction = "Focus specifically on the legal principles, doctrines, and precedents established."
            elif "procedure" in focus_areas:
                focus = "procedural history"
                instruction = "Focus specifically on the procedural history and process of the case."
            elif "," in focus_areas:
                # Multiple focus areas
                areas = [area.strip() for area in focus_areas.split(",")]
                focus = f"multiple aspects: {', '.join(areas)}"
                instruction = f"Provide detailed information covering all requested aspects: {', '.join(areas)}."
            else:
                focus = "comprehensive details"
                instruction = "Provide comprehensive details about the case."
        else:
            # Fallback to general detailed response
            focus = "comprehensive details"
            instruction = "Provide comprehensive details about the case."
        
        # Use simplified format for testing
        if template_mode == "simplified":
            # Detect which sections the query is asking for
            query_lower = query.lower()
            asking_for_facts = "fact" in query_lower
            asking_for_issues = "issue" in query_lower
            asking_for_ruling = "ruling" in query_lower or "decision" in query_lower
            asking_for_digest = "digest" in query_lower or "covering" in query_lower
            
            # Determine which sections to include
            if asking_for_digest:
                # Full digest requested - include all sections
                sections_instruction = """FACTS:
[Provide the factual background]

ISSUES:
[State the legal issues]

RULING:
[Provide the court's ruling]"""
                sections_note = "Provide ONLY these three sections (FACTS, ISSUES, RULING)."
            elif asking_for_facts and not asking_for_issues and not asking_for_ruling:
                # Only facts requested
                sections_instruction = """FACTS:
[Provide the factual background]"""
                sections_note = "Provide ONLY the FACTS section."
            elif asking_for_ruling and not asking_for_facts and not asking_for_issues:
                # Only ruling requested
                sections_instruction = """RULING:
[Provide the court's ruling]"""
                sections_note = "Provide ONLY the RULING section."
            elif asking_for_issues and not asking_for_facts and not asking_for_ruling:
                # Only issues requested
                sections_instruction = """ISSUES:
[State the legal issues]"""
                sections_note = "Provide ONLY the ISSUES section."
            else:
                # Default: include all sections
                sections_instruction = """FACTS:
[Provide the factual background]

ISSUES:
[State the legal issues]

RULING:
[Provide the court's ruling]"""
                sections_note = "Provide ONLY these three sections (FACTS, ISSUES, RULING)."
            
            detailed_prompt = f"""Answer the user's query about this case using ONLY the requested section(s):

{sections_instruction}

Case Information:
- Title: {case_title}
- G.R. Number: {gr_number}
- Ponente: {ponente}
- Date: {date}

User Query: "{query}"

Focus on: {focus}

{instruction}

Context from case text:
{clean_text[:4000]}

{sections_note} NO other sections, headings, or meta-commentary."""
        else:
            detailed_prompt = f"""Answer the user's query about this case directly and concisely. Do NOT include disclaimers or meta-commentary.

Case Information:
- Title: {case_title}
- G.R. Number: {gr_number}
- Ponente: {ponente}
- Date: {date}

User Query: "{query}"

Focus on: {focus}

{instruction}

Context from case text:
{clean_text[:4000]}

Provide a direct, well-structured answer. Use clear headings. NO phrases like "Based on my knowledge" or "I can provide" - just answer directly."""

        messages = [
            {"role": "system", "content": "You are a Philippine Law expert. Answer directly without disclaimers or meta-commentary."},
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
            response += f"\n\n📚 **Source Reference:**\nView the complete case text:\n[{source_url}]({source_url})"
        
        print(f"✅ Generated detailed follow-up response")
        return response.strip()
        
    except Exception as e:
        print(f"⚠️ Detailed response generation failed: {e}")
        # Fallback to regular case summary
        return _generate_case_summary_from_jsonl(full_case, query, None, history, template_mode)

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
        for i, result in enumerate(web_results[:5], 1):  # Use top 3 results
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
8. Do NOT include a "Conclusion" section

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

def _create_response_with_cached_case(response_text: str, full_case_data: Dict) -> Dict[str, Any]:
    """Create a response object with cached case data for follow-up questions"""
    return {
        "content": response_text,
        "_cached_case_data": full_case_data,
        "role": "assistant"
    }

def _extract_case_from_history(history: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Extract the most recent case information from conversation history"""
    if not history:
        return None
    
    # Look through both user and assistant messages (recent first)
    for msg in reversed(history):
        content = msg.get("content", "")
        if not content:
            continue
            
        # First, check if this message has cached case data
        cached_case = msg.get("_cached_case_data")
        if cached_case and isinstance(cached_case, dict):
            print(f"🎯 Found cached case data in history: {cached_case.get('case_title', 'Unknown')[:50]}...")
            return {"full_case": cached_case, "content": content}
            
        # Extract GR number with more specific patterns (fallback)
        gr_patterns = [
            r'G\.R\.\s*No\.?\s*([0-9\-]+)',  # G.R. No. 12345
            r'GR\s*No\.?\s*([0-9\-]+)',      # GR No 12345
            r'G\.R\.\s*([0-9\-]+)',          # G.R. 12345
        ]
        
        for pattern in gr_patterns:
            gr_match = re.search(pattern, content, re.IGNORECASE)
            if gr_match:
                gr_number = gr_match.group(1)
                return {"gr_number": gr_number, "content": content}
        
        # Extract case titles - look for specific legal case patterns (fallback)
        case_title_patterns = [
            r'\*\*([^*]*\s+vs?\.?\s+[^*]+)\*\*',  # "**Party vs Party**" 
            r'\*\*([^*]*\s+v\.?\s+[^*]+)\*\*',    # "**Party v. Party**"
            r'\*\*([A-Z][^*]*(?:vs?\.?|v\.)[^*]*)\*\*',  # Case with vs/v starting with capital
        ]
        
        for pattern in case_title_patterns:
            title_match = re.search(pattern, content, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
                # Validate it looks like a real case title (has vs/v and is reasonable length)
                if ((' vs ' in title.lower() or ' v. ' in title.lower() or ' v ' in title.lower()) 
                    and len(title) > 10 and len(title) < 200):
                    return {"case_title": title, "content": content}
    
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
   - "how about the facts/issues/ruling/principles/doctrine"
   - Any question referring to "this case", "the case", or asking for elaboration
   - Questions referencing a specific case by name/G.R. number and asking about specific aspects:
     * "What was the final ruling or decision in [CASE] ([G.R./A.C. No.])?" → [SEARCH_CASES:ruling]
     * "What are the facts in [CASE] ([G.R./A.C. No.])?" → [SEARCH_CASES:facts]
     * "What were the issues in [CASE] ([G.R./A.C. No.])?" → [SEARCH_CASES:issues]
     * "What is the ruling in [CASE] ([G.R./A.C. No.])?" → [SEARCH_CASES:ruling]
   
   Then respond with EXACTLY: "[SEARCH_CASES:focus_area]" where focus_area indicates what aspect they want:
   - For facts/factual background: "[SEARCH_CASES:facts]"
   - For legal issues/questions: "[SEARCH_CASES:issues]" 
   - For court ruling/decision: "[SEARCH_CASES:ruling]"
   - For legal principles/doctrines: "[SEARCH_CASES:principles]"
   - For procedural history: "[SEARCH_CASES:procedure]"
   - For multiple aspects (e.g. "facts and ruling"): "[SEARCH_CASES:facts,ruling]"
   - For general/comprehensive details: "[SEARCH_CASES:general]"
   
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
   - Asking for a case digest of a specific case by G.R. number ("make a case digest of G.R. No. 123456")
   - Queries that explicitly say "Provide a case digest for [CASE] ([G.R./A.C. No.]), covering the main facts, issues, and ruling"
   - Any query asking for a comprehensive case digest covering facts, issues, and ruling
   
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
- "make a case digest of G.R. No. 123456" → [CASE_DIGEST]
- "tell me about the case of X vs Y" → [CASE_DIGEST]
- "summarize the case of ABC vs DEF" → [CASE_DIGEST]
- "G.R. No. 123456" → [CASE_DIGEST]
- "Provide a case digest for SPOUSES ANTONIO AND TAN v. ATTY. VALLEJO (A.C. No. 11219), covering the main facts, issues, and ruling." → [CASE_DIGEST]
- "Provide a case digest for TAOK v. CONDE (G.R. No. 254248), covering the main facts, issues, and ruling." → [CASE_DIGEST]
- "impeachment cases" → [SEARCH_CASES] (search for cases)
- "estafa cases" → [SEARCH_CASES] (search for cases)
- "annulment cases" → [SEARCH_CASES] (search for cases)
- "cases applying stare decisis" → [SEARCH_CASES] (search for cases)
- "show me res judicata cases" → [SEARCH_CASES] (search for cases)
- "delve deeper into the facts of this case" → [SEARCH_CASES:facts] (follow-up)
- "tell me more about the ruling" → [SEARCH_CASES:ruling] (follow-up)
- "what were the issues and how did they rule?" → [SEARCH_CASES:issues,ruling] (follow-up)
- "explain the legal principles" → [SEARCH_CASES:principles] (follow-up)
- "give me more details about this case" → [SEARCH_CASES:general] (follow-up)
- "What was the final ruling or decision in SPOUSES ANTONIO AND TAN v. ATTY. VALLEJO (A.C. No. 11219)?" → [SEARCH_CASES:ruling] (follow-up)
- "What are the facts in TAOK v. CONDE (G.R. No. 254248)?" → [SEARCH_CASES:facts] (follow-up)
- "What were the issues in COLMENAR v. COLMENAR (G.R. No. 252467)?" → [SEARCH_CASES:issues] (follow-up)
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
        
        # Check if we need to search cases (with focus information)
        search_match = re.search(r'\[SEARCH_CASES(?::([^]]+))?\]', response)
        if search_match:
            focus_info = search_match.group(1) if search_match.group(1) else None
            if focus_info:
                print(f"🔍 Query requires case search with focus: {focus_info} - proceeding with RAG retrieval")
                # Store focus info for use by detailed response function
                return {"action": "search_cases", "focus": focus_info}
            else:
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


def _extract_special_number(query: str) -> Optional[str]:
    """Extract special number from query (A.C., A.M., OCA, etc.), returns formatted number or None"""
    if not query:
        return None
    
    special_patterns = [
        (r"A\.M\.\s+No\.?\s*([A-Z0-9\-]+)", "A.M. No. {}"),
        (r"OCA\s+No\.?\s*([A-Z0-9\-]+)", "OCA No. {}"),
        (r"U\.C\.\s+No\.?\s*([A-Z0-9\-]+)", "U.C. No. {}"),
        (r"ADM\s+No\.?\s*([A-Z0-9\-]+)", "ADM No. {}"),
        (r"A\.C\.\s+No\.?\s*([A-Z0-9\-]+)", "A.C. No. {}"),
        (r"AC\s+No\.?\s*([A-Z0-9\-]+)", "AC No. {}"),
        (r"B\.M\.\s+No\.?\s*([A-Z0-9\-]+)", "B.M. No. {}"),
        (r"LRC\s+No\.?\s*([A-Z0-9\-]+)", "LRC No. {}"),
        (r"SP\s+No\.?\s*([A-Z0-9\-]+)", "SP No. {}"),
    ]
    
    for pattern, format_str in special_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            number = match.group(1).strip()
            return format_str.format(number)
    
    return None

def _extract_gr_number(query: str) -> str:
    """Extract G.R. number from query, returns numeric part or empty string.
    Does NOT match numbers that are part of special case numbers (A.C., A.M., etc.)."""
    if not query:
        return ""
    q = query.strip()
    
    # First, check if there's a special number - if so, don't extract as GR number
    if _extract_special_number(query):
        return ""
    
    # Common patterns: "G.R. No. 123456", "G.R. NOS. 151809-12", "GR No. 123456"
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

# Keep a single retriever instance (lazy-init for reuse)
retriever = None


def evaluate_response(query: str, response: str, reference_text: str, 
                     case_metadata: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """
    Evaluate chatbot response using integrated automated metrics
    
    Args:
        query: User query
        response: Chatbot generated response
        reference_text: Reference legal text from JSONL
        case_metadata: Optional case metadata
        
    Returns:
        Evaluation results dictionary or None if metrics not available
    """
    if not METRICS_AVAILABLE:
        return None
    
    try:
        # Calculate automated scores
        automated_scores = AutomatedContentScoring.score_legal_response(
            response, reference_text
        )
        
        # Calculate legal accuracy assessment
        accuracy_assessment = LegalAccuracyMetrics.assess_legal_information_accuracy(
            response, reference_text, query
        )
        
        # Combine results
        evaluation_results = {
            'automated_scores': automated_scores,
            'accuracy_assessment': accuracy_assessment,
            'case_metadata': case_metadata
        }
        
        # EvaluationTracker removed - no longer logging evaluations
        
        return evaluation_results
        
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _generate_case_listing_summary(full_case: Dict) -> str:
    """Generate a brief 2-sentence summary for case listings"""
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

def _format_case_number_link(case_number: str) -> str:
    """Format case number with appropriate clickable links"""
    if case_number == "Unknown" or not case_number:
        return f"({case_number})"
    
    # Pattern matching for different case types
    case_patterns = [
        (r'G\.R\.\s+No\.\s+([0-9\-]+)', lambda m: f"**[{case_number}](gr:{m.group(1)})**"),
        (r'A\.M\.\s+No\.\s+([A-Z0-9\-]+)', lambda m: f"**[{case_number}](am:{m.group(1)})**"),
    ]
    
    for pattern, formatter in case_patterns:
        match = re.search(pattern, case_number)
        if match:
            return formatter(match)
    
    # Default formatting for other case types
    return f"**({case_number})**"

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

def _generate_single_case_data(case_data: Tuple[int, Dict]) -> Tuple[int, str, Optional[Dict]]:
    """Generate summary and load full case data - used in parallel processing"""
    index, d = case_data
    
    case_summary = ""
    full_case = None
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
            print(f"⚠️ Failed to load case data for {lookup_id}: {e}")
    
    return index, case_summary, full_case

def _generate_case_data_parallel(cases_data: List[Dict], max_workers: int = 5) -> Tuple[List[str], List[Optional[Dict]]]:
    """Generate case summaries and load full case data in parallel using ThreadPoolExecutor"""
    
    # Prepare case data for parallel processing
    case_data_tuples = [(i, case_data) for i, case_data in enumerate(cases_data)]
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(_generate_single_case_data, case_tuple): i 
            for i, case_tuple in enumerate(case_data_tuples)
        }
        
        # Collect results in order
        summaries = [""] * len(cases_data)
        full_cases = [None] * len(cases_data)
        for future in concurrent.futures.as_completed(future_to_index):
            try:
                index, summary, full_case = future.result()
                summaries[index] = summary
                full_cases[index] = full_case
            except Exception as e:
                print(f"⚠️ Error generating case data: {e}")
                # Keep empty data for failed cases
    
    return summaries, full_cases

def _build_case_display_item(case_data: Dict, case_summary: str, full_case: Optional[Dict], index: int) -> str:
    """Build display item for a single case"""
    
    # Get case title
    title = _display_title(case_data)
    
    # Get case number (GR or special)
    gr_raw = case_data.get("gr_number") or case_data.get("metadata", {}).get("gr_number") or ""
    special_raw = case_data.get("special_number") or case_data.get("metadata", {}).get("special_number") or ""
    
    if gr_raw:
        case_number = _normalize_gr_display(str(gr_raw))
    elif special_raw:
        case_number = str(special_raw)
    else:
        case_number = "Unknown"

    # Improve title if needed using already loaded full case data
    if case_summary and full_case and (title in {"Untitled case", ""} or title.startswith("Case (")):
        improved_title = full_case.get("case_title") or full_case.get("title")
        if improved_title and isinstance(improved_title, str) and len(improved_title.strip()) >= 10:
            title = improved_title.strip()[:120]

    # Format the case number with clickable links
    formatted_case_number = _format_case_number_link(case_number)

    # Get date for display
    date = case_data.get("promulgation_date") or case_data.get("metadata", {}).get("promulgation_date", "")
    date_display = f" | {date}" if date else ""
    
    # Build the case listing item
    if case_number != "Unknown" and case_number in title:
        # Case number already in title
        item = f"{index}. **{title}{date_display}**"
    else:
        # Separate case number
        item = f"{index}. **{title}{date_display}** {formatted_case_number}"
    
    # Add case summary if available
    if case_summary:
        item += f"\n{case_summary}"
    
    return item

def _process_cases_with_parallel_summaries(picked: List[Dict]) -> List[str]:
    """Process cases with parallel summary generation and data loading"""
    
    # Generate summaries and load full case data in parallel (single load per case)
    case_summaries, full_cases = _generate_case_data_parallel(picked, max_workers=5)
    
    # Build final items using helper function
    items = []
    for i, (case_data, case_summary, full_case) in enumerate(zip(picked, case_summaries, full_cases), 1):
        item = _build_case_display_item(case_data, case_summary, full_case, i)
        items.append(item)
    
    return items

def _generate_case_summary_from_jsonl(full_case_content: Dict, query: str, retriever, history: List[Dict] = None, template_mode: str = "full") -> str:
    """Generate case summary directly from JSONL content using LLM
    
    Args:
        full_case_content: Full case data from JSONL
        query: User query
        retriever: Retriever instance
        history: Conversation history
        template_mode: "full" for complete template, "simplified" for facts/issues/ruling only
    """
    print(f"🔍 Generating case summary from JSONL content using LLM (template: {template_mode})...")
    
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
- Title: {case_title} (If the case title is not valid, use the clean_text to extract the title.)
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
    # Choose template based on mode
    if template_mode == "simplified":
        # SIMPLIFIED TEMPLATE: Facts, Issues, Ruling only (for testing)
        # Detect which sections the query is asking for
        query_lower = query.lower()
        asking_for_facts = "fact" in query_lower
        asking_for_issues = "issue" in query_lower
        asking_for_ruling = "ruling" in query_lower or "decision" in query_lower
        asking_for_digest = "digest" in query_lower or "covering" in query_lower
        
        # Determine which sections to include
        if asking_for_digest:
            # Full digest requested - include all sections, match ground truth format
            output_format = f"""NOW PRODUCE the digest in this EXACT format:

FACTS:
[Provide essential facts in paragraph form - focus on what happened, the events, relationships, and circumstances that led to the legal dispute. Be concise and direct.]

ISSUES:
[State the legal issues clearly and directly.]

RULING:
[Comprehensive analysis: what legal tests/rules were applied and why they led to the result. Include the court's reasoning and legal basis for each issue. Be clear and concise.]"""
        elif asking_for_facts and not asking_for_issues and not asking_for_ruling:
            # Only facts requested - match ground truth format exactly
            output_format = f"""NOW PRODUCE ONLY the FACTS section in this EXACT format:

FACTS:
[Provide essential facts in paragraph form - focus on what happened, the events, relationships, and circumstances that led to the legal dispute. Be concise and direct.]"""
        elif asking_for_ruling and not asking_for_facts and not asking_for_issues:
            # Only ruling requested - match ground truth format exactly
            output_format = f"""NOW PRODUCE ONLY the RULING section in this EXACT format:

RULING:
[Comprehensive analysis: what legal tests/rules were applied and why they led to the result. Include the court's reasoning and legal basis for each issue. Be clear and concise.]"""
        elif asking_for_issues and not asking_for_facts and not asking_for_ruling:
            # Only issues requested - match ground truth format exactly
            output_format = f"""NOW PRODUCE ONLY the ISSUES section in this EXACT format:

ISSUES:
[State the legal issues clearly and directly. Each issue should be on a new line or bullet point.]"""
        else:
            # Default: include all sections - match ground truth format
            output_format = f"""NOW PRODUCE the digest in this EXACT format:

FACTS:
[Provide essential facts in paragraph form - focus on what happened, the events, relationships, and circumstances that led to the legal dispute. Be concise and direct.]

ISSUES:
[State the legal issues clearly and directly.]

RULING:
[Comprehensive analysis: what legal tests/rules were applied and why they led to the result. Include the court's reasoning and legal basis for each issue. Be clear and concise.]"""
        
        case_digest_prompt = f"""
        Create a CASE DIGEST for a Philippine Supreme Court case using the FULL CASE TEXT provided below.
        Answer directly - do NOT include meta-commentary like "Based on my knowledge" or "I can provide". Just present the digest.
        Extract all information directly from the case text. If a detail is missing, write exactly: Not stated in sources.

        **CRITICAL INSTRUCTIONS:**
        - Use ONLY information from the provided case text
        - Title: If the case title is not valid, use the clean_text to extract the title.
        - Facts: Provide only the essential and relevant facts necessary to understand the context of the case. Avoid unnecessary narrative or procedural detail.
        - Issues: Identify all legal issues presented and resolved in the decision. Each issue must be stated clearly and, when possible, in question form. Ensure no issue is omitted, merged, or combined.
        - Ruling: Provide a detailed explanation of the Supreme Court's ruling or holding for each issue, including the Court's reasoning and legal basis. Present each ruling immediately after its corresponding issue.
        - Legal Citations: Reproduce all legal provisions, case citations, and statutory references in full, exactly as written in the source text. Do not abbreviate, simplify, summarize, or alter any legal citation, article, section, or case name.
        - Fidelity: Do not add, infer, interpret, or omit any information from the text. Harmonize the content of the case all throughout the digest.
        - Focus: Focus on the constitutional aspect of the case.

        **CASE TEXT:**
        {context}

        {output_format}
        
        """
    else:
        # FULL TEMPLATE: All sections (for normal system operation)
        case_digest_prompt = f"""
        Create a comprehensive CASE DIGEST for a Philippine Supreme Court case using the FULL CASE TEXT provided below.
        Answer directly - do NOT include meta-commentary like "Based on my knowledge" or "I can provide". Just present the digest.
        Extract all information directly from the case text. If a detail is missing, write exactly: Not stated in sources.

        **CRITICAL INSTRUCTIONS:**
        - Use ONLY information from the provided case text
        - Title: If the case title is not valid, use the clean_text to extract the title.
        - Facts: Provide only the essential and relevant facts necessary to understand the context of the case. Avoid unnecessary narrative or procedural detail.
        - Issues: Identify all legal issues presented and resolved in the decision. Each issue must be stated clearly and, when possible, in question form. Ensure no issue is omitted, merged, or combined.
        - Ruling: Provide a detailed explanation of the Supreme Court's ruling or holding for each issue, including the Court's reasoning and legal basis. Present each ruling immediately after its corresponding issue.
        - Priority: Give primary importance to the accuracy and completeness of the Issues and Rulings over the factual summary.
        - Legal Citations: Reproduce all legal provisions, case citations, and statutory references in full, exactly as written in the source text. Do not abbreviate, simplify, summarize, or alter any legal citation, article, section, or case name.
        - Fidelity: Do not add, infer, interpret, or omit any information from the text. Harmonize the content of the case all throughout the digest.
        - Focus: Focus on the constitutional aspect of the case.
        
        **CASE TEXT:**
        {context}

        NOW PRODUCE the digest EXACTLY in this format (keep headers bold) UNLESS specified in the query:
        
        **{case_title}**

        **{gr_number} | {date} | Ponente: {ponente}**

        **Nature:** [Nature of case, e.g., Petition for Review on Certiorari]

        **Topic:** [Legal topic, e.g., Original Document Rule]

        **Case Type:** [e.g., annulment, estafa, administrative matter, etc.]

        **Doctrine:**
        [One concise controlling principle derived from the Court's reasoning in Context. Focus on the main legal rule resolving the dispute; avoid mere procedural trivia unless dispositive.]

        **Ticker/Summary:**
        [2–4 sentences: parties, core conflict, key procedural posture if shown, and the outcome direction.]

        **Petitioner/s:** [Exact name(s) as in Context] | **Respondent/s:** [Exact name(s) as in Context]

        **Facts:**
        1) [SUBSTANTIVE FACT: What actually happened between the parties - the real events, relationships, and circumstances that led to the legal dispute]
        2) [SUBSTANTIVE FACT: The core legal relationship or transaction that is the subject of the case]
        3) [SUBSTANTIVE FACT: Key events, actions, or circumstances that created the legal controversy]
        4) [Continue with substantive facts only - avoid procedural steps like "filed petition", "court denied", etc.]

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
        - WHETHER OR NOT [State the first legal issue]: [YES/NO; Legal basis]
        - WHETHER OR NOT [State the second legal issue]: [YES/NO; Legal basis]
        - WHETHER OR NOT [State the third legal issue]: [YES/NO; Legal basis]
        - [Continue for all issues found in the case - each issue must start with "WHETHER OR NOT" and end with a clear YES or NO answer]
        
        **SC RULING:**
        - [Comprehensive depth analysis: what legal tests/rules were applied and why they led to the result. CAPTURE the REQUISITES for the ruling. Do NOT restate the dispositive. No invented citations.]

        **DISPOSITIVE:**
        - ["…verbatim final dispositive text from Context…" or Not stated in sources]
        
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
    
    response_parts = [digest_text]
    
    # Add reference links to case sources
    source_url = full_case_content.get("source_url") if full_case_content else None
    response_parts.append("\n📚 Source References:")
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

def _generate_baseline_response(query: str, history: Optional[List[Dict]] = None) -> str:
    """
    Generate response using LLM only, without RAG retrieval.
    This is used for baseline testing to compare against RAG-enabled system.
    
    Args:
        query: User query
        history: Conversation history
        
    Returns:
        Response string from LLM
    """
    print("[BASELINE] Generating response with LLM only (no RAG)")
    
    try:
        # Build prompt for baseline mode - answer directly without disclaimers
        system_prompt = """You are a Philippine Law expert. Answer queries directly and concisely based on your knowledge of Philippine jurisprudence and legal principles. 

Provide factual information without meta-commentary like "Based on my knowledge" or "I can provide". Simply state the answer directly.

Do NOT include phrases like:
- "Based on my general knowledge"
- "I can provide"
- "However, please note"
- "I may not have complete details"

Just answer the question directly with the information."""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if available
        if history:
            print(f"[BASELINE] Including {len(history)} messages from conversation history")
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        print(f"[BASELINE] Generating response with TogetherAI (no retrieval)...")
        response = generate_messages_with_togetherai(
            messages,
            max_tokens=4092,
            temperature=0.7,  # Slightly higher temperature for baseline
            top_p=0.9
        )
        
        print(f"[BASELINE] Response generated")
        return response.strip()
        
    except Exception as e:
        print(f"⚠️ Baseline response generation failed: {e}")
        import traceback
        traceback.print_exc()
        return f"I apologize, but I encountered an error generating a response. Please try again."

def chat_with_law_bot(query: str, history: List[Dict] = None, use_rag: bool = True, template_mode: str = "full"):
    """
    Main chatbot function that handles queries with optional RAG.
    
    Args:
        query: User query string
        history: Conversation history
        use_rag: If False, bypass RAG and use LLM only (for baseline testing)
        template_mode: "full" for complete template, "simplified" for facts/issues/ruling only (for testing)
    
    Returns:
        Response string
    """
    import time
    query_start_time = time.time()
    global retriever
    
    print(f"[QUERY] '{query}'")
    
    # BASELINE MODE: No RAG - just use LLM directly
    if not use_rag:
        print("[BASELINE MODE] No RAG - using LLM directly")
        return _generate_baseline_response(query, history)

    # Early handling: ponente queries bypass classification and generic reply paths
    if _is_ponente_query(query):
        print(f"[PONENTE] Early ponente handler for: '{query}'")
        justice_name = _extract_ponente_name_from_query(query)
        if justice_name:
            print(f"🎯 Extracted justice name: '{justice_name}'")
        retrieval_start = time.time()
        _initialize_retriever()
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
            # Check if it's a follow-up question with focus information
            if isinstance(ai_response, dict) and ai_response.get("action") == "search_cases":
                focus_info = ai_response.get("focus")
                print(f"[FOLLOW-UP] Proceeding with focused case retrieval: {focus_info}")
                
                # Try to extract case from history for detailed response
                case_info = _extract_case_from_history(history)
                if case_info:
                    print(f"🎯 Found case context from history")
                    
                    # Check if we have cached full case data (preferred)
                    if "full_case" in case_info:
                        full_case = case_info["full_case"]
                        print(f"⚡ Using cached case data - no database query needed!")
                        # Generate focused detailed response using cached data
                        result = _generate_detailed_case_response(full_case, query, history, focus_info, template_mode)
                        total_time = time.time() - query_start_time
                        print(f"⏱️ Total query time: {total_time:.2f}s")
                        return result
                    
                    # Fallback: Load case from database using various identifiers
                    case_id = None
                    case_type = None
                    
                    if "gr_number" in case_info:
                        case_id = case_info["gr_number"]
                        case_type = "G.R. No."
                    elif "case_title" in case_info:
                        case_id = case_info["case_title"] 
                        case_type = "Case Title"
                    elif "special_number" in case_info:
                        case_id = case_info["special_number"]
                        case_type = "Special No."
                    
                    if case_id:
                        print(f"🔍 No cached data found, loading from database: {case_type} {case_id}")
                        try:
                            from .retriever import load_case_from_jsonl
                            full_case = load_case_from_jsonl(case_id)
                            if full_case:
                                # Generate focused detailed response
                                result = _generate_detailed_case_response(full_case, query, history, focus_info, template_mode)
                                total_time = time.time() - query_start_time
                                print(f"⏱️ Total query time: {total_time:.2f}s")
                                return result
                            else:
                                print(f"❌ Case not found in database: {case_type} {case_id}")
                        except Exception as e:
                            print(f"⚠️ Failed to load case for focused follow-up: {e}")
                    else:
                        print(f"⚠️ Could not extract case identifier from history")
                
                # Continue to regular RAG retrieval if case extraction failed
                query_context = {"focus": focus_info, "is_followup": True}
            # Check if it's a web search request
            elif ai_response == "[WEB_SEARCH]":
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
        _initialize_retriever()
        
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
    
    # Step 3: Query needs case search - proceed with RAG retrieval
    retrieval_start = time.time()
    print(f"[SEARCH] Proceeding with case retrieval for: '{query}'")
    

    _initialize_retriever()

    try:
            # Simplified routing: Check special numbers FIRST, then GR-number exact search, then keyword top-3
            special_num = _extract_special_number(query)
            gr_num = _extract_gr_number(query)  # This won't match if special_num is found (due to fix above)
            docs = []
            if special_num:
                # Direct special number lookup using JSONL
                print(f"🎯 Special-number path: {special_num}")
                from .retriever import load_case_from_jsonl
                full_case = load_case_from_jsonl(special_num)
                if full_case:
                    # Convert to list format expected by digest generation
                    docs = [{
                        "gr_number": "",  # Empty for special numbers
                        "special_number": full_case.get("special_number", special_num),
                        "metadata": {
                            "case_title": full_case.get("case_title", ""),
                            "special_number": full_case.get("special_number", special_num),
                            "promulgation_date": full_case.get("promulgation_date", ""),
                            "ponente": full_case.get("ponente", "")
                        }
                    }]
                    # Store full_case for later use in digest generation
                    # The digest generation code will reload from JSONL if needed
                    print(f"🔍 Loaded case from JSONL: {full_case.get('case_title', 'Unknown')[:50]}")
                else:
                    # Case not found - generate appropriate error message
                    case_num_display = special_num
                    result = f"I'm sorry, but I couldn't find {case_num_display} in our database. This case may not be available in our current collection, or the case number might be incorrect. Please verify the case number and try again, or search for cases by topic or party names."
                    result += "\n\n---\n"
                    result += "📚 Source References:\n"
                    result += "[https://elibrary.judiciary.gov.ph/](https://elibrary.judiciary.gov.ph/)\n"
                    result += "[https://lawphil.net/](https://lawphil.net/)"
                    total_time = time.time() - query_start_time
                    print(f"⏱️ Total query time: {total_time:.2f}s")
                    return result
                wants_digest = True  # enforce digest format for special number path
            elif gr_num:
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
                            result = _generate_case_summary_from_jsonl(full_case, query, retriever, history, template_mode=template_mode)
                            generation_time = time.time() - generation_start
                            print(f"⏱️ Case digest generation took {generation_time:.2f}s")
                            print(f"🔍 Generated summary: {type(result)} - {bool(result)}")
                            retrieval_time = time.time() - retrieval_start
                            total_time = time.time() - query_start_time
                            print(f"⏱️ Retrieval took {retrieval_time:.2f}s")
                            print(f"⏱️ Total query time: {total_time:.2f}s")
                            
                            # Return response with cached case data for follow-up questions
                            print(f"💾 Caching case data for future follow-up questions")
                            return _create_response_with_cached_case(result, full_case)
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
                            result = _generate_case_summary_from_jsonl(full_case, query, retriever, history, template_mode=template_mode)
                            generation_time = time.time() - generation_start
                            print(f"⏱️ Case digest generation took {generation_time:.2f}s")
                            retrieval_time = time.time() - retrieval_start
                            total_time = time.time() - query_start_time
                            print(f"⏱️ Retrieval took {retrieval_time:.2f}s")
                            print(f"⏱️ Total query time: {total_time:.2f}s")
                            
                            # Return response with cached case data for follow-up questions
                            print(f"💾 Caching case data for future follow-up questions")
                            return _create_response_with_cached_case(result, full_case)
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

