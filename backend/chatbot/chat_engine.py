# chat_engine.py — Simplified Law LLM chat engine (GR-number path vs keyword path)
import re
from typing import Dict, List

from .debug_logger import log_debug, log_warning


def _extract_gr_number(query: str) -> str:
    """Extract G.R. number from query, returns numeric part or empty string."""
    if not query:
        return ""
    q = query.strip()
    # Common patterns: "G.R. No. 123456", "GR No. 123456", bare digits with separators
    m = re.search(r"G\.R\.?\s*No\.?\s*([0-9\-]+)", q, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"GR\s*No\.?\s*([0-9\-]+)", q, re.IGNORECASE)
    if m:
        return m.group(1)
    # Fallback: a long number that looks like GR (7+ digits)
    m = re.search(r"\b(\d{5,})\b", q)
    return m.group(1) if m else ""

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
    # Prefer explicit title-like fields
    title = (
        d.get("title")
        or meta.get("title")
        or meta.get("case_title")
        or meta.get("header")
        or ""
    )
    if title:
        t = title.strip()
        # If looks like a case title or is reasonably short, use as-is (with trim)
        if re.search(r"\b(v\.|vs\.|versus)\b", t, re.IGNORECASE) or len(t) <= 120:
            return t if len(t) <= 120 else (t[:117] + "...")
    # Try to infer a title from content lines containing v./vs.
    text = d.get("content") or d.get("text") or ""
    if text:
        for line in (text.split('\n')[:10]):
            line = line.strip()
            if len(line) >= 20 and re.search(r"\b(v\.|vs\.|versus)\b", line, re.IGNORECASE):
                return line[:117] + "..." if len(line) > 120 else line
    # Fallback to generic label to avoid duplicating GR in title and suffix
    return "Untitled case"

def _title_from_query(query: str) -> str:
    """Create a readable case-like title from user keywords as a fallback."""
    if not query:
        return "Untitled case"
    q = query.strip()
    # Normalize vs variants
    q = re.sub(r"\bversus\b", "v.", q, flags=re.IGNORECASE)
    q = re.sub(r"\bvs\.?\b", "v.", q, flags=re.IGNORECASE)
    # Title case but keep v. lowercase
    parts = [p for p in re.split(r"\s+", q) if p]
    out = []
    for p in parts:
        if p.lower() in {"v.", "vs.", "vs", "versus"}:
            out.append("v.")
        else:
            out.append(p[:1].upper() + p[1:])
    return " ".join(out)

def _advanced_retrieve(retriever, query: str, k: int = 8, is_case_digest: bool = False, history: List[Dict] = None):
    """Unified retrieval wrapper (uses retriever.retrieve)."""
    return retriever.retrieve(query, k=k, is_case_digest=is_case_digest)

# --- Stronger dispositive detection & extraction ---
DISPOSITIVE_HDR = r"(?:WHEREFORE|ACCORDINGLY|IN VIEW OF THE FOREGOING|THUS|HENCE|PREMISES CONSIDERED)"
SO_ORDERED = r"SO\s+ORDERED\.?"
RULING_REGEX = re.compile(
    rf"{DISPOSITIVE_HDR}[\s\S]{{0,4000}}?{SO_ORDERED}",
    re.IGNORECASE,
)
# Some decisions omit SO ORDERED but still have a dispositive paragraph
RULING_NO_SO_FALLBACK = re.compile(
    rf"(?:^{DISPOSITIVE_HDR}[\s\S]{{50,1500}}?$)",
    re.IGNORECASE | re.MULTILINE,
)
# Catch short single-paragraph orders that end with SO ORDERED without header
RULING_SO_ORDERED_FALLBACK = re.compile(
    rf"(?s)(?:^|\n\n).{{0,1500}}?{SO_ORDERED}",
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
    # Prefer classic header→SO ORDERED span
    m = RULING_REGEX.search(text)
    if m:
        return _normalize_ws(m.group(0), max_chars=2000)
    # Next: single-paragraph ending in SO ORDERED
    m = RULING_SO_ORDERED_FALLBACK.search(text)
    if m:
        return _normalize_ws(m.group(0), max_chars=1200)
    # Last: dispositive without SO ORDERED (rare but happens)
    m = RULING_NO_SO_FALLBACK.search(text)
    if m:
        return _normalize_ws(m.group(0), max_chars=1200)
    return ""

# Removed title/intent heuristics for simplified engine

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
    
    log_debug(f"[SEARCH] Looking for exact match for query: '{query_lower}'")
    log_debug(f"[SEARCH] Query words: {query_words}")
    
    # First, try to find a document that matches the query exactly
    for i, doc in enumerate(docs):
        title = (doc.get("title", "") or doc.get("metadata", {}).get("title", "")).lower().strip()
        gr = doc.get("gr_number", "") or doc.get("metadata", {}).get("gr_number", "")
        score = doc.get("score", 0.0)
        
        log_debug(f"  Doc {i+1}: '{title[:50]}...' | G.R. {gr} | Score: {score:.3f}")

        if not title:
            continue

        # More strict matching: check for exact phrase match first
        if query_lower in title or title in query_lower:
            exact_match_doc = doc
            log_debug(f"  ✅ Exact phrase match found: '{title[:50]}...'")
            break
        # Then check for significant word overlap (at least 4 words for better precision)
        elif len(query_words.intersection(set(title.split()))) >= 4:
            exact_match_doc = doc
            log_debug(f"  ✅ Significant word overlap found: '{title[:50]}...'")
            break
    
    # Use exact match document if found, otherwise use main document
    source_doc = exact_match_doc if exact_match_doc else main_doc
    
    if exact_match_doc:
        log_debug(f"[SEARCH] Using exact match document: G.R. {exact_match_doc.get('gr_number', 'Unknown')}")
    else:
        log_debug(f"[SEARCH] No exact match found, using main document: G.R. {main_doc.get('gr_number', 'Unknown')}")
    
    # Extract metadata from the chosen document
    case_title = (source_doc.get("metadata", {}).get("title", "") or 
                 source_doc.get("title", "") or 
                 source_doc.get("metadata", {}).get("case_title", "") or
                 query)  # Fallback to query if no title found
    
    gr_number = (source_doc.get("metadata", {}).get("gr_number", "") or 
                source_doc.get("gr_number", "") or 
                source_doc.get("metadata", {}).get("case_id", "") or
                "Not available")
    
    # Clean up G.R. number to avoid duplication
    if gr_number and gr_number != "Not available":
        # Remove "G.R. No." prefix if it exists to avoid duplication
        gr_number = re.sub(r'^G\.R\.\s+No\.\s*', '', gr_number, flags=re.IGNORECASE).strip()
        if gr_number:
            gr_number = f"G.R. No. {gr_number}"
        else:
            gr_number = "Not available"
    
    ponente = (source_doc.get("metadata", {}).get("ponente", "") or 
              source_doc.get("ponente", "") or 
              source_doc.get("metadata", {}).get("justice", "") or
              "Not available")
    
    date = (source_doc.get("metadata", {}).get("promulgation_date", "") or 
           source_doc.get("metadata", {}).get("date", "") or 
           source_doc.get("date", "") or 
           source_doc.get("metadata", {}).get("decision_date", "") or
           "Not available")
    
    case_type = (source_doc.get("metadata", {}).get("case_type", "") or 
                source_doc.get("case_type", "") or 
                source_doc.get("metadata", {}).get("type", "") or
                "Not available")
    
    log_debug(f"[SEARCH] Final metadata - Title: {case_title[:50]}... | G.R.: {gr_number} | Ponente: {ponente}")
    
    # Try to load full case content from JSONL file
    full_case_content = None
    if gr_number and gr_number != "Not available":
        # Clean the G.R. number for JSONL lookup (remove display formatting)
        jsonl_gr_number = re.sub(r'^G\.R\.\s+No\.\s*', '', gr_number, flags=re.IGNORECASE).strip()
        log_debug(f"[SEARCH] Loading full case content from JSONL for G.R. {jsonl_gr_number}...")
        try:
            from .retriever import load_case_from_jsonl
            full_case_content = load_case_from_jsonl(jsonl_gr_number)
            if full_case_content:
                log_debug(f"✅ Loaded full case content: {len(str(full_case_content))} characters")
            else:
                log_warning(f"⚠️ No full case content found for G.R. {jsonl_gr_number}")
        except Exception as e:
            log_warning(f"⚠️ Error loading full case content: {e}")
    
    # If we have full case content, use it for better summary
    if full_case_content:
        log_debug("[SEARCH] Using full case content for summary generation...")
        # Extract content from full case (using actual JSONL fields)
        facts_content = full_case_content.get("body", "")  # Use body for facts
        issues_content = full_case_content.get("header", "")  # Use header for issues
        ruling_content = full_case_content.get("ruling", "")
        decision_content = full_case_content.get("body", "")  # Use body for decision content
        
        # Use LLM to generate a proper storytelling summary from JSONL content
        log_debug("[SEARCH] Using LLM to generate storytelling summary from JSONL content...")
        
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
G.R. No.: {gr_number}
Ponente: {ponente}

Context:
{context}

Write exactly 5 complete sentences that tell the story of this case. Start with who the parties are, then what happened, what the legal question was, how the court decided, and what the outcome was. Do not include any numbered lists or questions - just write the 5 sentences directly."""

        try:
            from .generator import generate_legal_response
            print(f"[SEARCH] Calling LLM with prompt length: {len(storytelling_prompt)} characters")
            print(f"[SEARCH] Context length: {len(context)} characters")
            raw_summary = generate_legal_response(storytelling_prompt, context="", is_case_digest=False)
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


def _find_relevant_cases(docs: List[Dict], retriever, history: List[Dict] = None) -> List[Dict]:
    """Find relevant cases with similar case type"""
    if not docs or not retriever:
        return []
    
    # Get case type from main document with better fallbacks
    main_doc = max(docs, key=lambda x: x.get("score", 0.0))
    case_type = (main_doc.get("metadata", {}).get("case_type", "") or 
                main_doc.get("case_type", "") or 
                main_doc.get("metadata", {}).get("type", ""))
    
    if not case_type or case_type == "Not available":
        # Try to extract case type from content or use a generic search
        content = main_doc.get("content", "") or main_doc.get("text", "")
        if "contract" in content.lower():
            case_type = "contract"
        elif "criminal" in content.lower():
            case_type = "criminal"
        elif "administrative" in content.lower():
            case_type = "administrative"
        else:
            case_type = "civil"  # Default fallback
    
    # Search for cases with similar case type and legal concepts
    try:
        # Try a more specific search for similar cases
        case_title = main_doc.get("title", "") or main_doc.get("metadata", {}).get("title", "")
        search_queries = [
            f"case type {case_type}",
            f"similar to {case_title}",
            f"contract cases",
            f"civil cases"
        ]
        
        relevant_docs = []
        for search_query in search_queries:
            docs = _advanced_retrieve(retriever, search_query, k=3, is_case_digest=False, history=history)
            relevant_docs.extend(docs)
            if len(relevant_docs) >= 5:
                break
        # Filter out the current case
        current_gr = (main_doc.get("metadata", {}).get("gr_number", "") or 
                     main_doc.get("gr_number", "") or 
                     main_doc.get("metadata", {}).get("case_id", ""))
        relevant_docs = [d for d in relevant_docs if (d.get("metadata", {}).get("gr_number", "") or d.get("gr_number", "")) != current_gr]
        
        # Filter out bad documents completely before processing
        good_docs = []
        for doc in relevant_docs:
            title = (doc.get("metadata", {}).get("title", "") or 
                    doc.get("title", "") or 
                    doc.get("metadata", {}).get("case_title", ""))
            
            # Comprehensive list of bad title patterns
            bad_title_patterns = [
                "docketed as", "against the advice", "ceased reporting", "came back only",
                "administrative case for", "crim. case no", "people v", "unknown", "third division",
                "chico-nazario", "honorable", "presiding judge", "just ceased", "came back only",
                "rommel", "baybay", "presiding judge", "this branch", "starting 6 april",
                "c-63250", "alex sabayan", "chico-nazario", "third division", "senate and the house",
                "congress commenced", "regular session", "commission on appointments", "constituted on",
                "august 2004", "july 2004", "164978", "g.r. no. g.r. no", "g.r. no. unknown"
            ]
            
            # Check if title is bad
            is_bad_title = any(pattern in title.lower() for pattern in bad_title_patterns)
            
            # Skip if bad title or too short
            if is_bad_title or len(title.strip()) < 20:
                print(f"⚠️ Skipping document with bad title: '{title[:50]}...'")
                continue
            
            # Try to extract better title from content
            content = doc.get("content", "") or doc.get("text", "")
            if content:
                lines = content.split('\n')
                for line in lines[:15]:  # Check first 15 lines
                    line = line.strip()
                    # Look for proper case titles with v. or vs.
                    if (len(line) > 30 and 
                        ('v.' in line or 'vs.' in line or 'versus' in line.lower()) and
                        not any(bad_word in line.lower() for bad_word in bad_title_patterns) and
                        not line.startswith(('—', 'Supreme Court', 'E-Library', 'The factual', 'On 02 August'))):
                        # Update with better title
                        if 'metadata' not in doc:
                            doc['metadata'] = {}
                        doc['metadata']['title'] = line
                        doc['title'] = line
                        break
            
            # Only add if we have a good title now
            final_title = doc.get("title", "") or doc.get("metadata", {}).get("title", "")
            if len(final_title.strip()) >= 20 and not any(pattern in final_title.lower() for pattern in bad_title_patterns):
                good_docs.append(doc)
            else:
                print(f"⚠️ Skipping document after title extraction: '{final_title[:50]}...'")
        
        relevant_docs = good_docs
        
        # Return up to 3 good cases, or fewer if we don't have enough
        return relevant_docs[:3] if relevant_docs else []
    except Exception as e:
        print(f"⚠️ Error finding relevant cases: {e}")
        return []

def _generate_case_summary_from_jsonl(full_case_content: Dict, query: str, retriever, history: List[Dict] = None) -> str:
    """Generate case summary directly from JSONL content using LLM"""
    print("🔍 Generating case summary from JSONL content using LLM...")
    
    # Extract metadata
    case_title = full_case_content.get("title", "") or full_case_content.get("case_title", "")
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
    case_type = full_case_content.get("case_type", "") or "regular"
    
    # Extract content sections - try multiple possible fields
    facts_content = (full_case_content.get("body", "") or 
                    full_case_content.get("clean_text", "") or 
                    full_case_content.get("sections", {}).get("body", "") if isinstance(full_case_content.get("sections"), dict) else "")
    
    ruling_content = (full_case_content.get("ruling", "") or 
                     full_case_content.get("sections", {}).get("ruling", "") if isinstance(full_case_content.get("sections"), dict) else "")
    
    # Fallback: extract dispositive from body/clean_text/sections when explicit 'ruling' is absent
    if not ruling_content or not ruling_content.strip():
        candidate_texts = []
        # Primary long-form fields
        for key in ("body", "clean_text"):
            val = full_case_content.get(key, "")
            if isinstance(val, str) and val.strip():
                candidate_texts.append(val)
        # Sectioned variants
        sections_obj = full_case_content.get("sections")
        if isinstance(sections_obj, dict):
            for key in ("dispositive", "wherefore", "decision", "body", "ruling"):
                val = sections_obj.get(key, "")
                if isinstance(val, str) and val.strip():
                    candidate_texts.append(val)
        # Try regex extraction on candidates
        extracted = ""
        for txt in candidate_texts:
            extracted = _extract_dispositive(txt)
            if extracted:
                break
        if extracted:
            ruling_content = extracted
            print("✅ Extracted dispositive from body/sections as ruling content")

    # Debug: Check what content we found
    print(f"🔍 Facts content length: {len(facts_content)}")
    print(f"🔍 Ruling content length: {len(ruling_content)}")
    print(f"🔍 Available fields: {list(full_case_content.keys())}")
    
    # Prepare context for LLM
    context_parts = []
    if facts_content and len(facts_content.strip()) > 100:
        context_parts.append(f"FACTS: {facts_content[:2000]}...")
    if ruling_content and len(ruling_content.strip()) > 100:
        context_parts.append(f"RULING: {ruling_content[:2000]}...")
    
    context = "\n\n".join(context_parts) if context_parts else "No detailed content available."
    print(f"🔍 Final context length: {len(context)}")
    
    # Create a prompt for case digest format matching the image structure
    case_digest_prompt = f"""Create a case digest for this Philippine Supreme Court case in the EXACT format shown below:

Context:
{context}

You MUST format your response EXACTLY as follows (use these exact headers and structure):

**{case_title}**

**{gr_number} | {date} | Ponente: {ponente}**

**Nature:** [Type of case, e.g., Petition for Review on Certiorari]

**Topic:** [Legal topic, e.g., Original Document Rule]

**Doctrine:**
[Key legal doctrine or principle]

**Ticker/Summary:**
[Brief case summary]

**Petitioner/s:** [Petitioner name] | **Respondent/s:** [Respondent name]

**Facts:**
1) [First factual point]
2) [Second factual point]
3) [Continue with numbered facts]

**Petitioner's Contention:**
1) [First contention]
2) [Second contention]
3) [Continue with numbered contention]

**Respondent's Contention:**
1) [First contention]
2) [Second contention]
3) [Continue with numbered contention]

**RTC:** [RTC decision, e.g., IN FAVOR OF PETITIONER/RESPONDENT]
- [RTC's statement or reasoning; explain the RTC's reasoning]

**CA:** [CA decision, e.g., REVERSED THE RTC'S RULING]
- [CA's statement or reasoning; explain the CA's reasoning]

**ISSUE/S:** [Legal issue/s, the formar should start with "WHETHER OR NOT", e.g., WHETHER OR NOT...]
- [Answer to the issue, YES OR NO]

**SC RULING:**
[Supreme Court's decision and reasoning; explain the Court's reasoning]

**DISPOSITIVE:** [Final disposition, e.g., PETITION is GRANTED]

Use only information from the provided context. If information is not available, write "Not stated in sources."""

    try:
        from .generator import generate_legal_response
        print(f"🔍 Calling LLM with prompt length: {len(case_digest_prompt)}")
        print(f"🔍 Prompt preview: {case_digest_prompt[:200]}...")
        
        raw_digest = generate_legal_response(case_digest_prompt, context="", is_case_digest=True)
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
    
    # Find related cases using retriever (if available)
    relevant_cases = []
    if retriever:
        try:
            # Search for related cases
            related_docs = _advanced_retrieve(retriever, f"{case_title} similar cases", k=3, history=history)
            relevant_cases = _find_relevant_cases(related_docs, retriever, history)
        except Exception as e:
            print(f"⚠️ Error finding related cases: {e}")
    
    # Build response with case digest format
    response_parts = [digest_text]
    
    if relevant_cases:
        response_parts.append("\n**Related Cases:**")
        for i, case in enumerate(relevant_cases, 1):
            case_title = (case.get("title", "") or 
                         case.get("metadata", {}).get("title", "") or 
                         case.get("metadata", {}).get("case_title", ""))
            case_gr = (case.get("gr_number", "") or 
                      case.get("metadata", {}).get("gr_number", "") or 
                      "Unknown")
            case_type = (case.get("case_type", "") or 
                        case.get("metadata", {}).get("case_type", "") or 
                        "regular")
            response_parts.append(f"{i}. {case_title} ({case_gr}) - {case_type}")
    
    # response_parts.append("\nWhat would you like to know about this case?")
    # response_parts.append("• Case Digest - Complete structured summary")
    # response_parts.append("• Ruling - Court's decision and reasoning") 
    # response_parts.append("• Facts - Case background and events")
    # response_parts.append("• Issues - Legal questions raised")
    # response_parts.append("• Arguments - Legal reasoning and doctrines")
    
    return "\n".join(response_parts)

# Removed intent heuristics; simplified flow uses only GR-number vs keyword path
def _format_history(history: List[Dict]) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-8:]:  # cap to last 8 turns for brevity
        role = (msg.get("role") or "user").lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role not in ("system", "user", "assistant"):
            role = "user"
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def chat_with_law_bot(query: str, history: List[Dict] = None):
    global retriever
    
    # Simplified routing: always treat as jurisprudence; branch GR vs keywords
    print(f"[SEARCH] Query: '{query}'")

    if retriever is None:
            try:
                from .retriever import LegalRetriever
                base_retriever = LegalRetriever()
                retriever = base_retriever
                print("✅ Using Basic Legal Retriever")
            except ImportError as e:
                print(f"❌ Failed to load Legal Retriever: {e}")
                retriever = None

    try:
            # Simplified routing: GR-number exact search vs keyword top-3
            gr_num = _extract_gr_number(query)
            docs = []
            if gr_num:
                # Direct exact GR number metadata search
                print(f"🎯 GR-number path: {gr_num}")
                docs = retriever._retrieve_by_gr_number(gr_num, k=8)
                wants_digest = True  # enforce digest format for GR path
                
                # Debug: Show metadata found
                if docs:
                    first_doc = docs[0]
                    print(f"🔍 Metadata fields: {list(first_doc.keys())}")
                    print(f"🔍 Case title: '{first_doc.get('title', '')[:100]}...'")
                    print(f"🔍 GR number: {first_doc.get('gr_number', '')}")
                    print(f"🔍 Case type: {first_doc.get('case_type', '')}")
                
                # For GR number path, always use JSONL for full content
                # Vector DB only has metadata, JSONL has the actual case content
                if docs:
                    print(f"🔄 Found metadata for GR {gr_num}, fetching full content from JSONL")
                    try:
                        from .retriever import load_case_from_jsonl
                        full_case = load_case_from_jsonl(gr_num)
                        if full_case:
                            print(f"✅ JSONL content loaded for GR {gr_num}")
                            return _generate_case_summary_from_jsonl(full_case, query, retriever, history)
                        else:
                            print(f"❌ No JSONL data found for GR {gr_num}")
                    except Exception as e:
                        print(f"⚠️ JSONL loading failed for GR {gr_num}: {e}")
                else:
                    print(f"❌ No metadata found for GR {gr_num} in vector DB")
            else:
                # Keyword path: retrieve and return the top 3 unique cases
                hits = _advanced_retrieve(retriever, query, k=12, is_case_digest=False, history=history)
                # Pick top 3 unique cases by GR/title
                picked = []
                seen_keys = set()
                for d in hits:
                    key = (d.get("gr_number") or d.get("metadata", {}).get("gr_number") or d.get("title") or d.get("metadata", {}).get("title"))
                    if key and key not in seen_keys:
                        seen_keys.add(key)
                        picked.append(d)
                    if len(picked) >= 3:
                        break
                if picked:
                    items = []
                    for i, d in enumerate(picked, 1):
                        title = _display_title(d)
                        if title == "Untitled case":
                            title = _title_from_query(query)
                        gr_raw = d.get("gr_number") or d.get("metadata", {}).get("gr_number") or ""
                        gr = _normalize_gr_display(str(gr_raw))
                        case_type = (d.get("case_type") or d.get("metadata", {}).get("case_type") or "").strip()
                        suffix = f" — {case_type}" if case_type else ""
                        items.append(f"{i}. {title} ({gr}){suffix}")
                    return "Here are the possible cases:\n" + "\n".join(items)
                else:
                    return "I couldn't find matching cases in the current database. Try a different G.R. number or broader keywords."
    except Exception as e:
        print(f"[WARNING] Jurisprudence retrieval failed: {e}")
        return "I encountered a retrieval error. Please try again."
