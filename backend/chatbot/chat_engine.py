# chat_engine.py — Simplified Law LLM chat engine (GR-number path vs keyword path)
import re
from typing import Dict, List, Tuple

from .debug_logger import log_debug, log_error, log_info, log_warning


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

def _first_case_id(doc: dict) -> str:
    """Prefer a stable case identifier for de-dup: (gr_number or title)."""
    gr = (doc.get("gr_number") or "").strip().lower()
    if gr:
        return f"gr:{gr}"
    title = (doc.get("title") or doc.get("filename") or "untitled").strip().lower()
    return f"title:{title}"

def _title(doc: dict) -> str:
    return doc.get("title") or doc.get("gr_number") or doc.get("filename") or "Untitled case"

def _source(doc: dict) -> str:
    return doc.get("source_url") or "N/A"

def _ensure_section(doc: dict) -> str:
    """Heuristic section tag for each chunk."""
    sec = (doc.get("section") or "").strip().lower()
    if sec:
        return sec
    txt = doc.get("text") or ""
    if not txt:
        return "body"

    # 1) Ruling detection: dispositive with/without SO ORDERED
    if RULING_REGEX.search(txt) or RULING_SO_ORDERED_FALLBACK.search(txt) or RULING_NO_SO_FALLBACK.search(txt):
        return "ruling"

    # 2) Facts / Issues by heading position
    m_f = FACTS_HINT_RE.search(txt)
    m_i = ISSUES_HINT_RE.search(txt)
    if m_f and m_i:
        return "facts" if m_f.start() < m_i.start() else "issues"
    if m_f:
        return "facts"
    if m_i:
        return "issues"

    # 3) Default
    return "body"

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

def _has_section(docs: List[Dict], section: str) -> bool:
    return any((d.get("section") or "").lower() == section for d in docs)

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

def _extract_summary_from_chunks(docs: List[Dict], case_title: str) -> str:
    """Extract summary from document chunks (original method)"""
    # Get the main case document (highest score)
    main_doc = max(docs, key=lambda x: x.get("score", 0.0))
    
    # Try to find better content by searching for more specific sections
    # Look for documents with substantial content (not just captions)
    best_content = ""
    best_section = ""
    current_gr = main_doc.get("gr_number", "") or main_doc.get("metadata", {}).get("gr_number", "")
    
    # First, try to find content from the same case with substantial length
    print(f"🔍 Looking for content for G.R. {current_gr}")
    print(f"🔍 Available documents: {len(docs)}")
    
    for i, doc in enumerate(docs):
        doc_gr = doc.get("gr_number", "") or doc.get("metadata", {}).get("gr_number", "")
        doc_content = doc.get("content", "") or doc.get("text", "")
        doc_section = doc.get("section", "unknown")
        
        print(f"  Doc {i+1}: section='{doc_section}', gr='{doc_gr}', content_len={len(doc_content)}")
        if len(doc_content) > 50:
            content_preview = doc_content[:100]
            print(f"    Preview: {content_preview}...")
        
        # Only consider content from the same case
        if doc_gr == current_gr and doc_content and len(doc_content.strip()) > len(best_content):
            # Be more aggressive in selecting content - prefer any substantial content
            if (doc_section.lower() in ['facts', 'issues', 'ruling', 'header', 'body', 'full_text_ref'] or 
                len(doc_content.strip()) > 100):  # Reduced from 200 to 100 chars
                best_content = doc_content
                best_section = doc_section
                print(f"🔍 Selected content from section '{doc_section}', length={len(doc_content)}")
    
    # If no good content found from same case, try any content
    if not best_content:
        print("🔍 No same-case content found, trying any available content...")
        for doc in docs:
            doc_content = doc.get("content", "") or doc.get("text", "")
            doc_section = doc.get("section", "unknown")
            
            if doc_content and len(doc_content.strip()) > len(best_content):
                best_content = doc_content
                best_section = doc_section
                print(f"🔍 Selected fallback content from section '{doc_section}', length={len(doc_content)}")
    
    # Use the best content found
    content = best_content if best_content else (main_doc.get("content", "") or main_doc.get("text", ""))
    
    # Try to extract a better summary from the content
    if content and len(content.strip()) > 20:  # Reduced from 100 to 20
        # Clean up the content first
        clean_content = re.sub(r'\s*—\s*[^—]*\s*—\s*', ' ', content)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        # Different strategies based on section type
        if best_section == 'facts':
            # For facts section, look for the main factual narrative
            summary_text = _extract_facts_summary(clean_content, case_title)
        elif best_section == 'issues':
            # For issues section, extract the legal questions
            summary_text = _extract_issues_summary(clean_content, case_title)
        elif best_section == 'ruling':
            # For ruling section, extract the court's decision
            summary_text = _extract_ruling_summary(clean_content, case_title)
        else:
            # For other sections, use general extraction
            summary_text = _extract_general_summary(clean_content, case_title)
    else:
        # If still no good content, try to combine content from multiple sections
        combined_content = ""
        for doc in docs:
            doc_content = doc.get("content", "") or doc.get("text", "")
            if doc_content and len(doc_content.strip()) > 10:  # Reduced from 50 to 10
                combined_content += " " + doc_content
                if len(combined_content) > 500:
                    break
        
        if combined_content and len(combined_content.strip()) > 20:  # Reduced from 100 to 20
            # Clean and extract meaningful content
            clean_content = re.sub(r'\s*—\s*[^—]*\s*—\s*', ' ', combined_content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            summary_text = _extract_general_summary(clean_content, case_title)
        else:
            # Last resort: try to extract any meaningful content from the case title itself
            if case_title and len(case_title) > 50:
                summary_text = f"This case involves {case_title}. The case was decided by the Supreme Court of the Philippines."
            else:
                summary_text = f"Case involving {case_title}. Details not available in the retrieved documents."
    
    return summary_text

def _extract_meaningful_sentences(content: str, max_sentences: int, keywords: List[str]) -> List[str]:
    """Extract meaningful sentences with specific keywords"""
    if not content or not content.strip():
        return []
    
    # Clean the content first
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Split by sentences more intelligently
    sentences = re.split(r'[.!?]+', content)
    meaningful_sentences = []
    
    # First pass: look for sentences with keywords
    for sentence in sentences:
        sentence = sentence.strip()
        if (len(sentence) > 30 and  # Increased minimum length
            not sentence.startswith(('G.R. No.', 'Supreme Court', 'E-Library', '—', 'The relevant', 'On 02 August', '[', 'PANGANIBAN')) and
            not sentence.endswith(('for short', 'as follows', 'Decision, as follows')) and
            not sentence.isupper() and  # Skip all-caps sentences
            any(word in sentence.lower() for word in keywords)):
            meaningful_sentences.append(sentence)
            if len(meaningful_sentences) >= max_sentences:
                break
    
    # Second pass: if we don't have enough, be more lenient
    if len(meaningful_sentences) < max_sentences:
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 40 and  # Longer sentences for better context
                not sentence.startswith(('G.R. No.', 'Supreme Court', 'E-Library', '—', 'The relevant', 'On 02 August', '[', 'PANGANIBAN')) and
                not sentence.endswith(('for short', 'as follows', 'Decision, as follows')) and
                sentence not in meaningful_sentences and
                not any(bad_word in sentence.lower() for bad_word in ['summarized in', 'challenged decision', 'procedural and factual']) and
                not sentence.isupper() and  # Skip all-caps sentences
                not sentence.startswith(('The Case', 'This case', 'The general'))):  # Skip generic starts
                meaningful_sentences.append(sentence)
                if len(meaningful_sentences) >= max_sentences:
                    break
    
    # Third pass: if still not enough, take any decent sentences
    if len(meaningful_sentences) < max_sentences:
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 50 and  # Even longer sentences
                not sentence.startswith(('G.R. No.', 'Supreme Court', 'E-Library', '—', 'The relevant', 'On 02 August', '[', 'PANGANIBAN')) and
                not sentence.endswith(('for short', 'as follows', 'Decision, as follows')) and
                sentence not in meaningful_sentences and
                not sentence.isupper() and
                not any(bad_word in sentence.lower() for bad_word in ['summarized in', 'challenged decision', 'procedural and factual', 'the case', 'this case', 'the general']) and
                not sentence.startswith(('The Case', 'This case', 'The general'))):
                meaningful_sentences.append(sentence)
                if len(meaningful_sentences) >= max_sentences:
                    break
    
    return meaningful_sentences

def _extract_facts_summary(content: str, case_title: str) -> str:
    """Extract summary from facts section"""
    sentences = content.split('.')
    meaningful_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if (len(sentence) > 20 and 
            not sentence.startswith(tuple(case_title.split()[:3])) and
            not sentence.startswith(('G.R. No.', 'Supreme Court', 'E-Library', '—')) and
            any(word in sentence.lower() for word in ['petitioner', 'respondent', 'alleged', 'claimed', 'contended', 'filed', 'sought', 'requested', 'dispute', 'agreement', 'contract', 'breach', 'damages', 'injury', 'loss'])):
            meaningful_sentences.append(sentence)
            if len(' '.join(meaningful_sentences)) > 400:
                break
    
    if meaningful_sentences:
        return '. '.join(meaningful_sentences) + '.'
    else:
        return content[:300] + "..." if len(content) > 300 else content

def _extract_issues_summary(content: str, case_title: str) -> str:
    """Extract summary from issues section"""
    sentences = content.split('.')
    meaningful_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if (len(sentence) > 20 and
            not sentence.startswith(tuple(case_title.split()[:3])) and
            not sentence.startswith(('G.R. No.', 'Supreme Court', 'E-Library', '—')) and
            any(word in sentence.lower() for word in ['whether', 'issue', 'question', 'problem', 'dispute', 'controversy', 'matter', 'case', 'court', 'decision', 'ruling'])):
            meaningful_sentences.append(sentence)
            if len(' '.join(meaningful_sentences)) > 400:
                break
    
    if meaningful_sentences:
        return '. '.join(meaningful_sentences) + '.'
    else:
        return content[:300] + "..." if len(content) > 300 else content

def _extract_ruling_summary(content: str, case_title: str) -> str:
    """Extract summary from ruling section"""
    sentences = content.split('.')
    meaningful_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if (len(sentence) > 20 and
            not sentence.startswith(tuple(case_title.split()[:3])) and
            not sentence.startswith(('G.R. No.', 'Supreme Court', 'E-Library', '—')) and
            any(word in sentence.lower() for word in ['court', 'held', 'found', 'determined', 'concluded', 'ruled', 'decided', 'therefore', 'accordingly', 'wherefore', 'so ordered'])):
            meaningful_sentences.append(sentence)
            if len(' '.join(meaningful_sentences)) > 400:
                break
    
    if meaningful_sentences:
        return '. '.join(meaningful_sentences) + '.'
    else:
        return content[:300] + "..." if len(content) > 300 else content

def _extract_general_summary(content: str, case_title: str) -> str:
    """Extract summary from general content"""
    sentences = content.split('.')
    meaningful_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if (len(sentence) > 20 and
            not sentence.startswith(tuple(case_title.split()[:3])) and
            not sentence.startswith(('G.R. No.', 'Supreme Court', 'E-Library', '—')) and
            any(word in sentence.lower() for word in ['case', 'court', 'decision', 'ruling', 'facts', 'issues', 'petitioner', 'respondent', 'contract', 'agreement', 'dispute', 'claim', 'alleged', 'held', 'found', 'determined', 'whether', 'problem', 'matter'])):
            meaningful_sentences.append(sentence)
            if len(' '.join(meaningful_sentences)) > 400:
                break
    
    if meaningful_sentences:
        return '. '.join(meaningful_sentences) + '.'
    else:
        return content[:300] + "..." if len(content) > 300 else content

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

Case: {case_title}
G.R. No.: {gr_number}
Ponente: {ponente}
Date: {date}

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

**Respondent's Contention:**
1) [First contention]
2) [Second contention]

**RTC:** [RTC decision, e.g., IN FAVOR OF PETITIONER/RESPONDENT]
- [RTC's statement or reasoning]

**CA:** [CA decision, e.g., REVERSED THE RTC'S RULING]
- [CA's statement or reasoning]

**ISSUE:** [Legal issue, e.g., WHETHER OR NOT...]
- [YES OR NO]

**SC RULING:**
[Supreme Court's decision and reasoning]

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
    response_parts = [f"**{case_title}**\n\n**G.R. No.:** {gr_number}\n**Ponente:** {ponente}\n**Date:** {date}\n**Case Type:** {case_type}\n\n{digest_text}"]
    
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
            response_parts.append(f"{i}. {case_title} (G.R. No. {case_gr}) - {case_type}")
    
    response_parts.append("\nWhat would you like to know about this case?")
    response_parts.append("• Case Digest - Complete structured summary")
    response_parts.append("• Ruling - Court's decision and reasoning") 
    response_parts.append("• Facts - Case background and events")
    response_parts.append("• Issues - Legal questions raised")
    response_parts.append("• Arguments - Legal reasoning and doctrines")
    
    return "\n".join(response_parts)

# ---------- Enhanced context builder for Law LLM optimization ----------
SECTION_PRIORITY = ["ruling", "issues", "facts", "header", "arguments", "body", "keywords"]

def _dedupe_and_rank(docs: List[Dict], wants_ruling: bool, wants_facts: bool, wants_issues: bool, wants_arguments: bool, wants_keywords: bool, wants_digest: bool = False) -> List[Dict]:
    """
    Prefer one high-quality snippet per (case_id, section), keep higher score.
    """
    # Normalize fields
    for d in docs:
        d["section"] = _ensure_section(d)
        d["score"] = float(d.get("score") or 0.0)
        d["_case_id"] = _first_case_id(d)

    # Keep best per (case, section)
    best = {}
    for d in docs:
        key = (d["_case_id"], d["section"])
        if key not in best or d["score"] > best[key]["score"]:
            best[key] = d

    # Bucket by section
    buckets = {sec: [] for sec in SECTION_PRIORITY}
    for d in best.values():
        if d["section"] in buckets:
            buckets[d["section"]].append(d)

    for sec in buckets:
        buckets[sec].sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # Enhanced intent-aware assembly plan for Law LLM optimization
    if wants_digest:
        # For case digests, prioritize all sections equally to get comprehensive coverage
        plan = ["issues", "facts", "ruling", "arguments", "header", "keywords", "body"]
    elif wants_ruling and not (wants_facts or wants_issues or wants_arguments or wants_keywords):
        plan = ["ruling", "issues", "arguments", "facts", "header", "keywords", "body"]
    elif wants_facts and not (wants_ruling or wants_issues or wants_arguments or wants_keywords):
        plan = ["facts", "issues", "arguments", "ruling", "header", "keywords", "body"]
    elif wants_issues and not (wants_ruling or wants_facts or wants_arguments or wants_keywords):
        plan = ["issues", "arguments", "facts", "ruling", "header", "keywords", "body"]
    elif wants_arguments and not (wants_ruling or wants_facts or wants_issues or wants_keywords):
        plan = ["arguments", "issues", "facts", "ruling", "header", "keywords", "body"]
    elif wants_keywords and not (wants_ruling or wants_facts or wants_issues or wants_arguments):
        plan = ["keywords", "ruling", "issues", "arguments", "facts", "header", "body"]
    else:
        plan = SECTION_PRIORITY

    ordered: List[Dict] = []
    seen_cases = set()

    # For digests, allow multiple sections from same case for comprehensive coverage
    if wants_digest:
        # Take all available sections from top cases
        pool = [d for arr in buckets.values() for d in arr]
        pool.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        picked_keys = set()
        for d in pool:
            key = (d["_case_id"], d["section"])
            if key not in picked_keys:
                ordered.append(d)
                picked_keys.add(key)
                if len(ordered) >= 8:
                    break
    else:
        # First pass: take at most one section per *different case* to maximize case diversity
        for sec in plan:
            for d in buckets.get(sec, []):
                cid = d["_case_id"]
                if cid in seen_cases:
                    continue
                ordered.append(d)
                seen_cases.add(cid)
                break  # only one per section initially

        # Second pass: fill up to 8 with next best unique (case, section)
        if len(ordered) < 8:
            pool = [d for arr in buckets.values() for d in arr]
            picked_keys = {(d["_case_id"], d["section"]) for d in ordered}
            pool.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            for d in pool:
                key = (d["_case_id"], d["section"])
                if key in picked_keys:
                    continue
                ordered.append(d)
                picked_keys.add(key)
                if len(ordered) >= 8:
                    break

    return ordered[:8]

def _build_context(docs: List[Dict]) -> str:
    """
    Number refs as [1], [2], ... Title — Section — URL, then a compact snippet.
    If a 'ruling' ref is present, ensure the snippet contains the dispositive text verbatim.
    """
    lines = []
    item_index = 0
    for d in docs:
        # Derive robust title
        title = _title(d)
        if not title or title.strip().lower() in {"n/a", "untitled", "untitled case"}:
            meta = d.get("metadata", {}) or {}
            title = (
                meta.get("title")
                or meta.get("case_title")
                or d.get("title")
                or d.get("case_title")
                or d.get("gr_number")
                or meta.get("gr_number")
                or (d.get("url") or "").rsplit("/", 1)[-1]
                or f"Document {item_index + 1}"
            )
        url   = _source(d)
        sec   = d.get("section", "body")
        text  = d.get("content") or d.get("text") or ""
        
        # Clean up the text content using the same patterns as retriever
        if text:
            # Remove only specific metadata noise patterns, be more conservative
            text = re.sub(r'\s*—\s*N/A\s*', '', text)  # Remove "— N/A" patterns
            # Remove "Supreme Court E-Library" boilerplate
            text = re.sub(r'Supreme Court E-Library[^—]*', '', text, flags=re.IGNORECASE)
            # Clean up multiple spaces and newlines but preserve content
            text = re.sub(r'\s+', ' ', text).strip()
        
        snippet = _normalize_ws(text, max_chars=8000) if text else ""

        # Force dispositive extraction for ruling sections, so generator can quote verbatim safely
        if sec == "ruling" and text:
            extracted = _extract_dispositive(text)
            if extracted:
                snippet = extracted

        # Skip entries that have neither title nor content
        if not (title and title.strip()) and not (snippet and snippet.strip()):
            continue
        
        # Only show section if it's meaningful (not just "body")
        section_display = f" — {sec}" if sec and sec != "body" else ""
        item_index += 1
        header = f"[{item_index}] {title}{section_display} — {url}"
        lines.append(f"{header}\n{snippet}")
    return "\n\n".join(lines)

# --- Output post-validation ---
def _post_validate(ans: str, wants_facts: bool, wants_ruling: bool, wants_arguments: bool, wants_keywords: bool, wants_digest: bool = False) -> str:
    if not ans:
        return "Not stated in sources."
    # Remove placeholder tokens if any slipped
    ans = ans.replace("[n]", "")
    # Trim trailing spaces
    ans = re.sub(r"[ \t]+$", "", ans, flags=re.MULTILINE).strip()
    
    # More concise responses when sections are missing
    missing_sections = []
    if wants_facts and ("Facts:" not in ans and "\n- " not in ans):
        missing_sections.append("Facts")
    if wants_ruling and "Ruling:" in ans and ("Not stated in sources" not in ans) and ("[" not in ans):
        ans += " [1]"
    if wants_arguments and "Arguments:" not in ans and "Reasoning:" not in ans:
        missing_sections.append("Arguments")
    if wants_keywords and "Keywords:" not in ans and "Doctrine:" not in ans:
        missing_sections.append("Keywords")
    
    # If multiple sections are missing, provide a concise summary
    if len(missing_sections) > 1:
        return f"Available information:\n{ans}\n\nMissing: {', '.join(missing_sections)} - Not stated in sources."
    elif len(missing_sections) == 1:
        return f"{ans}\n\n{missing_sections[0]}: Not stated in sources."
    
    return ans

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
            
            # Remove duplicates based on unique identifier (case_id + section)
            print(f"[SEARCH] Before deduplication: {len(docs)} documents")
            # Debug: show what sections we have
            sections = [doc.get('section', 'body') for doc in docs]
            print(f"[SEARCH] Available sections: {sections}")
            print(f"[SEARCH] Using digest deduplication: {wants_digest}")
            if wants_digest:
                # For digests, allow multiple sections from same case but remove exact duplicates
                seen_content = set()
                unique_docs = []
                for doc in docs:
                    content = doc.get('content', '') or doc.get('text', '')
                    # Use a more specific fingerprint: first 200 chars + last 100 chars
                    content_fingerprint = content[:200] + content[-100:] if len(content) > 300 else content
                    content_hash = hash(content_fingerprint)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        unique_docs.append(doc)
                        case_id = doc.get('metadata', {}).get('case_id', '') or doc.get('metadata', {}).get('gr_number', '')
                        section = doc.get('section', 'body')
                        print(f"  ✅ Added: {case_id}_{section}")
                    else:
                        case_id = doc.get('metadata', {}).get('case_id', '') or doc.get('metadata', {}).get('gr_number', '')
                        section = doc.get('section', 'body')
                        print(f"  ❌ Duplicate content: {case_id}_{section}")
            else:
                # For non-digests, use strict deduplication
                seen_keys = set()
                unique_docs = []
                for doc in docs:
                    case_id = doc.get('metadata', {}).get('case_id', '') or doc.get('metadata', {}).get('gr_number', '')
                    section = doc.get('section', 'body')
                    key = f"{case_id}_{section}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        unique_docs.append(doc)
                        print(f"  ✅ Added: {key}")
                    else:
                        print(f"  ❌ Duplicate: {key}")
            docs = unique_docs[:8]  # Limit to 8 results
            print(f"[SEARCH] After deduplication: {len(docs)} documents")
            if docs and len(docs) > 0:
                print(f"[SEARCH] Found {len(docs)} jurisprudence documents for query: '{query}'")
                # We have jurisprudence results, proceed to context building and generation
            else:
                print(f"[WARNING] No jurisprudence results for query: '{query}'")
                # No jurisprudence results, provide helpful response
                return "I couldn't find matching cases in the current database. Try a different G.R. number or broader keywords."
    except Exception as e:
            print(f"[WARNING] Jurisprudence retrieval failed: {e}")
            return "I encountered a retrieval error. Please try again."

    # Build context and generate answer (digest by default for GR path)
    print(f"[SEARCH] Processing {len(docs)} retrieved documents for context building")
    wants_digest = True if gr_num else False
    wants_ruling = False
    wants_facts = False
    wants_issues = False
    wants_arguments = False
    wants_keywords = False
    chosen = _dedupe_and_rank(docs, wants_ruling, wants_facts, wants_issues, wants_arguments, wants_keywords, wants_digest)
    print(f"📋 Selected {len(chosen)} documents after deduplication and ranking")

    # Enhanced section prioritization for Law LLM optimization
    # Ensure at least one 'facts' chunk if user explicitly asked for facts
    if wants_facts and not _has_section(chosen, "facts"):
        facts_pool = [d for d in docs if (_ensure_section(d) == "facts")]
        if facts_pool:
            best_facts = max(facts_pool, key=lambda x: float(x.get("score") or 0.0))
            # replace the lowest-scoring non-facts doc
            if chosen:
                repl_idx = min(range(len(chosen)), key=lambda i: float(chosen[i].get("score") or 0.0))
                chosen[repl_idx] = best_facts
            else:
                chosen = [best_facts]
    
    # Ensure at least one 'arguments' chunk if user explicitly asked for arguments
    if wants_arguments and not _has_section(chosen, "arguments"):
        arguments_pool = [d for d in docs if (_ensure_section(d) == "arguments")]
        if arguments_pool:
            best_arguments = max(arguments_pool, key=lambda x: float(x.get("score") or 0.0))
            if chosen:
                repl_idx = min(range(len(chosen)), key=lambda i: float(chosen[i].get("score") or 0.0))
                chosen[repl_idx] = best_arguments
            else:
                chosen = [best_arguments]

    # Ensure at least one 'ruling' chunk for digests (and generally helpful)
    if (wants_digest or True) and not _has_section(chosen, "ruling"):
        ruling_pool = [d for d in docs if (_ensure_section(d) == "ruling")]
        if ruling_pool:
            best_ruling = max(ruling_pool, key=lambda x: float(x.get("score") or 0.0))
            if chosen:
                repl_idx = min(range(len(chosen)), key=lambda i: float(chosen[i].get("score") or 0.0))
                chosen[repl_idx] = best_ruling
            else:
                chosen = [best_ruling]
    
    # Ensure at least one 'keywords' chunk if user explicitly asked for keywords
    if wants_keywords and not _has_section(chosen, "keywords"):
        keywords_pool = [d for d in docs if (_ensure_section(d) == "keywords")]
        if keywords_pool:
            best_keywords = max(keywords_pool, key=lambda x: float(x.get("score") or 0.0))
            if chosen:
                repl_idx = min(range(len(chosen)), key=lambda i: float(chosen[i].get("score") or 0.0))
                chosen[repl_idx] = best_keywords
            else:
                chosen = [best_keywords]

    context = _build_context(chosen)
    print(f"📝 Built context from {len(chosen)} documents: {len(context)} characters")
    if context:
        print(f"📄 Context preview: {context[:500]}...")
        print(f"📄 Context sections: {[d.get('section', 'unknown') for d in chosen]}")
    else:
        print("⚠️ No context built from retrieved documents")

    # Enhanced missing-section flags to guide safe fallbacks
    must_note_absent_ruling = wants_ruling and (" — ruling — " not in context)
    must_note_absent_facts  = wants_facts  and (" — facts — "  not in context)
    must_note_absent_arguments = wants_arguments and (" — arguments — " not in context)
    must_note_absent_keywords = wants_keywords and (" — keywords — " not in context)

    # -------- Enhanced conditional answer template for Law LLM optimization --------
    if wants_digest:
        # Case digest template following the specified format
        answer_template = (
            "You are PHLaw-Chatbot's Case Digest Writer. Produce a Philippine Supreme Court case digest that is strictly grounded in the retrieved documents. "
            "Use these exact section labels and order:\n\n"
            "Issue\n"
            "1) Whether or not [precise legal question grounded in sources].\n"
            "2) Whether or not [next issue, if any].\n\n"
            "Facts\n"
            "[Make a bullet in a new line per sentence. Lead with the substantive narrative (who did what, when, where, why), then procedural history (RTC → CA → SC). Keep it chronological and concise.]\n\n"
            "Ruling\n"
            "- Doctrine/Rule: [state controlling doctrines/tests derived from sources].\n"
            "- Application: [map facts to elements/tests; explain the Court's reasoning].\n"
            "- Lower Courts:\n"
            "  • RTC: [result + brief reason]\n"
            "  • CA:  [result + brief reason]\n"
            "- Dispositive: \"[quote the WHEREFORE/dispositive clause verbatim if available].\"\n"
            "- Holding: [final holding in plain language].\n\n"
            "Discussion\n"
            "[Concurring/dissenting opinions: summarize positions and doctrinal implications. If none, say: 'No separate opinions noted in the sources.']\n\n"
            "Citations only if applicable\n"
            "- [Case v. Case], G.R. No. _____, [Date] — [one-line doctrinal takeaway].\n"
            "- [Case v. Case], G.R. No. _____, [Date] — [one-line doctrinal takeaway].\n\n"
            "Legal Terms Breakdown (only if asked)\n"
            "- [Term]: [plain definition] — [how it is applied here], [rule/source if in text].\n\n"
            "IMPORTANT RULES:\n"
            "1. Never invent facts. If something isn't in the sources, write: 'Not stated in sources.'\n"
            "2. Frame each issue using the 'Whether or not …' convention. Base issues ONLY on the retrieved text.\n"
            "3. Include case header inline at the very top of Facts or Issue if present in sources: Case name; G.R. No.; Date; Ponente; (Division/En Banc if present).\n"
            "4. CRITICAL: Do NOT include any metadata noise such as:\n"
            "   - '— body — N/A'\n"
            "   - '— N/A'\n"
            "   - '— header —'\n"
            "   - 'Supreme Court E-Library'\n"
            "   - Any patterns with '—' symbols\n"
            "5. Focus ONLY on actual legal content from the sources.\n"
            "6. If you see noise patterns in the source text, ignore them completely.\n"
            "7. Provide a proper structured digest, not a list of case titles."
        )
    elif wants_ruling and not (wants_facts or wants_issues or wants_arguments or wants_keywords):
        answer_template = (
            "Based on the provided sources, the ruling is: <if dispositive present, put the exact WHEREFORE/So Ordered text in double quotes, "
            "then provide a clear explanation with court reasoning; otherwise write: Not stated in sources.>"
        )
    elif wants_facts and not (wants_ruling or wants_issues or wants_arguments or wants_keywords):
        answer_template = (
            "Based on the sources, here are the key facts:\n"
            "- <extract the most relevant fact, ≤30 words, with [n] citation>\n"
            "- <extract another important fact, ≤30 words, with [n] citation>\n"
            "- <extract additional relevant fact, ≤30 words, with [n] citation>\n"
            "(Provide 3-5 key facts. Each must be supported by the sources with proper citations only if applicable.)"
        )
    elif wants_issues and not (wants_ruling or wants_facts or wants_arguments or wants_keywords):
        answer_template = "The legal issues addressed in this case are: <extract and list the main legal questions with [n] citations only if applicable>"
    elif wants_arguments and not (wants_ruling or wants_facts or wants_issues or wants_keywords):
        answer_template = (
            "The legal arguments presented include:\n"
            "- <extract key legal reasoning, ≤35 words, with [n] citation>\n"
            "- <extract additional legal reasoning, ≤35 words, with [n] citation>\n"
            "(Provide 2-4 key legal arguments. Each must be supported by the sources.)"
        )
    elif wants_keywords and not (wants_ruling or wants_facts or wants_issues or wants_arguments):
        answer_template = (
            "Key legal terms and doctrines mentioned:\n"
            "- <extract important legal term/doctrine, ≤25 words, with [n] citation>\n"
            "- <extract another legal term/doctrine, ≤25 words, with [n] citation>\n"
            "(Identify 2-4 key legal concepts. Each must be supported by the sources.)"
        )
    else:
        # Enhanced default: comprehensive digest with better structure
        answer_template = (
            "Based on the provided sources, here is a comprehensive overview:\n\n"
            "Facts:\n- <extract key fact with [n] citation>\n- <extract another key fact with [n] citation>\n\n"
            "Legal Issues:\n- <identify main legal question with [n] citation>\n\n"
            "Arguments:\n- <extract key legal reasoning with [n] citation>\n\n"
            "Ruling:\n<if dispositive present, quote it verbatim in double quotes with [n] citation only if applicable; then provide a clear explanation with court reasoning"
            "otherwise write: Not stated in sources.>"
        )
    # -------- End enhanced conditional --------

    # Simplified, focused prompt for better RAG performance
    history_block = _format_history(history or [])
    
    if wants_digest:
        # Special prompt for case digest requests
        prompt = (
            f"{answer_template}\n\n"
            f"Conversation so far (if any):\n{history_block}\n\n" if history_block else ""
            f"Sources:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
    else:
        prompt = (
            "You are a Philippine legal assistant. Analyze the provided sources and answer the question based ONLY on that information.\n"
            "If the sources don't contain relevant information, say 'The sources don't contain information about this topic.'\n"
            "Provide clear, accurate responses with proper citations [1], [2], etc.\n\n"
            f"Conversation so far (if any):\n{history_block}\n\n" if history_block else ""
            f"Sources:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

    # Use messages-based generator for conversational context if available
    from .generator import generate_response, generate_response_from_messages
    messages = []
    if history_block:
        # Reconstruct messages: keep the same text as in prompt preface but provide roles
        messages.append({"role": "system", "content": "You are a legal assistant focused on Philippine Supreme Court jurisprudence."})
        for line in history_block.splitlines():
            if ": " in line:
                role, content = line.split(": ", 1)
                role_key = role.strip().lower()
                if role_key not in ("system", "user", "assistant"):
                    role_key = "user"
                messages.append({"role": role_key, "content": content})
    # Always append the current question with the structured instructions and Refs
    if wants_digest:
        messages.append({
            "role": "user",
            "content": (
                f"{answer_template}\n\n"
                f"Sources:\n{context}\n\n"
                f"Question: {query}\n"
                f"Answer:"
            )
        })
    else:
        messages.append({
            "role": "user",
            "content": (
                "Analyze the provided sources and answer the question based ONLY on that information.\n"
                "If the sources don't contain relevant information, say 'The sources don't contain information about this topic.'\n"
                "Provide clear, accurate responses with proper citations [1], [2], etc.\n\n"
                f"Sources:\n{context}\n\n"
                f"Question: {query}\n"
                f"Answer:"
            )
        })

    # Prefer messages-based generation when possible
    try:
        raw = generate_response_from_messages(messages)
    except Exception:
        raw = generate_response(prompt)
    return _post_validate(raw, wants_facts, wants_ruling, wants_arguments, wants_keywords, wants_digest)



# Removed rule-based path and intent types in simplified engine

def _chat_without_retrieval(query: str, history: List[Dict]) -> str:
    from .generator import generate_response, generate_response_from_messages

    history_block = _format_history(history)
    sys = (
        "You are a helpful, concise legal assistant for the Philippines."
        " If the user asks for definitive jurisprudential facts or citations,"
        " offer to search jurisprudence. Avoid unauthorized legal advice."
    )
    messages: List[Dict] = []
    messages.append({"role": "system", "content": sys})
    if history_block:
        for line in history_block.splitlines():
            if ": " in line:
                role, content = line.split(": ", 1)
                role_key = role.strip().lower()
                if role_key not in ("system", "user", "assistant"):
                    role_key = "user"
                messages.append({"role": role_key, "content": content})
    messages.append({"role": "user", "content": query})

    try:
        return generate_response_from_messages(messages)
    except Exception:
        prompt = (f"Conversation so far:\n{history_block}\n\nUser: {query}\nAssistant:") if history_block else (f"User: {query}\nAssistant:")
        return generate_response(prompt)