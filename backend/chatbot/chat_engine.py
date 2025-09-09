# chat_engine.py — Enhanced Law LLM chat engine with optimized embeddings
import re
from typing import Dict, List, Tuple

from .rule_based import RuleBasedResponder

rb = RuleBasedResponder(bot_name="PHLaw-Chatbot")

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

def _intent(query: str) -> Tuple[bool, bool, bool, bool, bool, bool]:
    q = (query or "").lower()

    # negation-aware toggles (e.g., "not the facts", "no issues")
    def want(term: str) -> bool:
        if term not in q:
            return False
        # crude negation check within a short window
        return not re.search(rf"(?:no|not|without)\s+{re.escape(term)}", q)

    wants_ruling = any(want(t) for t in ["ruling", "decision", "disposition", "wherefore", "so ordered", "wherefore clause"])
    # for facts/issues we also allow direct words, not only headings
    wants_facts  = want("facts")  or bool(FACTS_HINT_RE.search(q))
    wants_issues = want("issues") or "whether" in q or bool(ISSUES_HINT_RE.search(q))
    wants_arguments = any(want(t) for t in ["argument", "arguments", "reasoning", "legal reasoning", "doctrine"])
    wants_keywords = any(want(t) for t in ["keyword", "keywords", "legal term", "doctrine", "principle"])
    wants_digest = any(want(t) for t in ["digest", "case digest", "full digest", "complete digest", "comprehensive digest"])
    return wants_ruling, wants_facts, wants_issues, wants_arguments, wants_keywords, wants_digest

def _has_section(docs: List[Dict], section: str) -> bool:
    return any((d.get("section") or "").lower() == section for d in docs)

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

    # First pass: take at most one section per *different case* to maximize case diversity
    for sec in plan:
        for d in buckets.get(sec, []):
            cid = d["_case_id"]
            if cid in seen_cases:
                continue
            ordered.append(d)
            seen_cases.add(cid)
            break  # only one per section initially

    # Second pass: fill up to 5 with next best unique (case, section)
    if len(ordered) < 5:
        pool = [d for arr in buckets.values() for d in arr]
        picked_keys = {(d["_case_id"], d["section"]) for d in ordered}
        pool.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        for d in pool:
            key = (d["_case_id"], d["section"])
            if key in picked_keys:
                continue
            ordered.append(d)
            picked_keys.add(key)
            if len(ordered) >= 5:
                break

    return ordered[:5]

def _build_context(docs: List[Dict]) -> str:
    """
    Number refs as [1], [2], ... Title — Section — URL, then a compact snippet.
    If a 'ruling' ref is present, ensure the snippet contains the dispositive text verbatim.
    """
    lines = []
    for i, d in enumerate(docs, 1):
        title = _title(d)
        url   = _source(d)
        sec   = d.get("section", "body")
        text  = d.get("content") or d.get("text") or ""
        snippet = _normalize_ws(text)

        # Force dispositive extraction for ruling sections, so generator can quote verbatim safely
        if sec == "ruling":
            extracted = _extract_dispositive(text)
            if extracted:
                snippet = extracted

        header = f"[{i}] {title} — {sec} — {url}"
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

# ---------- main entry ----------
KEY_TERMS_NEED_JURIS = {
    "g.r.", "gr no", "gr no.", "gr number", "sc gr",
    "people v", "people vs", "vs.", "v.", "en banc", "supreme court",
    "article ", "art. ", "section ", "sec. ", "rule ", "rule of court",
    "civil code", "revised penal code", "rpc", "constitution",
    "ruling", "decision", "wherefore", "so ordered", "facts", "issues",
    "doctrine", "held", "ratio", "precedent", "jurisprudence", "cite",
    "case", "cases", "court", "legal", "law", "supreme", "philippine", "philippines",
    "2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016",
    "title", "parties", "ponente", "promulgation", "resolution", "petition", "respondent"
}


def _should_query_jurisprudence(query: str, history: List[Dict] = None) -> bool:
    q = (query or "").lower()
    if not q:
        return False
    
    # Check for rule-based patterns first (glossary definitions)
    from .rule_based import DEFINE_RE, ELEMENTS_RE
    if DEFINE_RE.search(q) or ELEMENTS_RE.search(q):
        return False  # These should go to rule-based, not jurisprudence
    
    # Enhanced jurisprudence detection for Law LLM optimization
    
    # 1. Direct legal terms (but exclude glossary terms)
    for t in KEY_TERMS_NEED_JURIS:
        if t in q:
            return True
    
    # 2. Question patterns with legal context (but exclude definition queries)
    if re.search(r"\b(what|when|whether|who|how|where|which)\b", q):
        legal_context = any(term in q for term in ["case", "ruling", "decision", "court", "legal", "law", "supreme", "title", "parties"])
        if legal_context:
            return True
    
    # 3. Year patterns (likely asking about cases)
    if re.search(r"\b(20\d{2})\b", q):
        # Years are very likely case-related queries
        return True
    
    # 4. Case citation patterns
    if re.search(r"\b\w+\s+v\.?s?\.?\s+\w+", q):
        return True
    
    # 5. Follow-up questions (likely about previous case context)
    follow_up_indicators = ["that case", "the case", "this case", "the title", "the parties", "the ruling"]
    if any(indicator in q for indicator in follow_up_indicators):
        return True
    
    # 6. Very short follow-up questions (likely about previous context)
    if len(q.strip()) <= 10 and q.strip() in ["where", "when", "who", "what", "how", "why"]:
        return True
    
    # 6. Context-aware detection: if previous messages were about cases, this is likely a follow-up
    if history:
        recent_context = " ".join([msg.get("content", "") for msg in history[-3:] if msg.get("role") != "system"])
        if any(term in recent_context.lower() for term in ["case", "ruling", "decision", "court", "legal", "law", "supreme"]):
            # Previous context was legal, this is likely a follow-up
            return True
    
    return False    
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
    
    # Enhanced jurisprudence detection for Law LLM optimization
    use_juris = _should_query_jurisprudence(query, history)
    print(f"🔍 Query: '{query}' | Jurisprudence: {use_juris}")
    
    # Derive intent flags early (used for retrieval and formatting)
    wants_ruling, wants_facts, wants_issues, wants_arguments, wants_keywords, wants_digest = _intent(query)

    # Prioritize jurisprudence queries over rule-based responses
    if use_juris:
        if retriever is None:
            from .retriever import LegalRetriever
            retriever = LegalRetriever()
        
        try:
            docs = retriever.retrieve(query, k=8, is_case_digest=wants_digest)
            if docs and len(docs) > 0:
                print(f"🔍 Found {len(docs)} jurisprudence documents for query: '{query}'")
                # We have jurisprudence results, proceed with enhanced processing
                pass
            else:
                print(f"⚠️ No jurisprudence results for query: '{query}'")
                # No jurisprudence results, provide helpful response
                return "I couldn't find any 2025 cases in the current database. This could be because:\n• The cases haven't been processed yet\n• The database needs to be updated\n• There might be a configuration issue\n\nTry asking about a different year or a specific case topic."
        except Exception as e:
            print(f"⚠️ Jurisprudence retrieval failed: {e}")
            # Fall back to rule-based
            msg = rb.answer(query)
            if msg is not None:
                return msg
            return _chat_without_retrieval(query, history or [])
    else:
        # Non-jurisprudence query, try rule-based first
        msg = rb.answer(query)
        if msg is not None:
            return msg
        return _chat_without_retrieval(query, history or [])

    # intents already computed above
    chosen = _dedupe_and_rank(docs, wants_ruling, wants_facts, wants_issues, wants_arguments, wants_keywords, wants_digest)

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
            "[2–4 short paragraphs. Lead with the substantive narrative (who did what, when, where, why), then procedural history (RTC → CA → SC). Keep it chronological and concise.]\n\n"
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
            "Citations (when applicable)\n"
            "- [Case v. Case], G.R. No. _____, [Date] — [one-line doctrinal takeaway].\n"
            "- [Case v. Case], G.R. No. _____, [Date] — [one-line doctrinal takeaway].\n\n"
            "Legal Terms Breakdown (only if asked)\n"
            "- [Term]: [plain definition] — [how it is applied here], [rule/source if in text].\n\n"
            "IMPORTANT: Never invent facts. If something isn't in the sources, write: 'Not stated in sources.' "
            "Frame each issue using the 'Whether or not …' convention. Base issues ONLY on the retrieved text. "
            "Include case header inline at the very top of Facts or Issue if present in sources: Case name; G.R. No.; Date; Ponente; (Division/En Banc if present)."
        )
    elif wants_ruling and not (wants_facts or wants_issues or wants_arguments or wants_keywords):
        answer_template = (
            "Based on the provided sources, the ruling is: <if dispositive present, put the exact WHEREFORE/So Ordered text in double quotes, "
            "then provide a clear explanation with [n]; otherwise write: Not stated in sources.>"
        )
    elif wants_facts and not (wants_ruling or wants_issues or wants_arguments or wants_keywords):
        answer_template = (
            "Based on the sources, here are the key facts:\n"
            "- <extract the most relevant fact, ≤30 words, with [n] citation>\n"
            "- <extract another important fact, ≤30 words, with [n] citation>\n"
            "- <extract additional relevant fact, ≤30 words, with [n] citation>\n"
            "(Provide 3-5 key facts. Each must be supported by the sources with proper citations.)"
        )
    elif wants_issues and not (wants_ruling or wants_facts or wants_arguments or wants_keywords):
        answer_template = "The legal issues addressed in this case are: <extract and list the main legal questions with [n] citations>"
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
            "Ruling:\n<if dispositive present, quote it verbatim in double quotes with [n] citation; "
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