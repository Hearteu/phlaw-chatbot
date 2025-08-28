# chat_engine.py — section-aware, de-duplicated context + structured prompt (refined)
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

def _intent(query: str) -> Tuple[bool, bool, bool]:
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
    return wants_ruling, wants_facts, wants_issues

def _has_section(docs: List[Dict], section: str) -> bool:
    return any((d.get("section") or "").lower() == section for d in docs)

# ---------- context builder ----------
SECTION_PRIORITY = ["ruling", "issues", "facts", "header", "body"]

def _dedupe_and_rank(docs: List[Dict], wants_ruling: bool, wants_facts: bool, wants_issues: bool) -> List[Dict]:
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

    # Intent-aware assembly plan
    if wants_ruling and not (wants_facts or wants_issues):
        plan = ["ruling", "issues", "facts", "header", "body"]
    elif wants_facts and not (wants_ruling or wants_issues):
        plan = ["facts", "issues", "ruling", "header", "body"]
    elif wants_issues and not (wants_ruling or wants_facts):
        plan = ["issues", "facts", "ruling", "header", "body"]
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
        text  = d.get("text") or ""
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
def _post_validate(ans: str, wants_facts: bool, wants_ruling: bool) -> str:
    if not ans:
        return "Not stated in sources."
    # Remove placeholder tokens if any slipped
    ans = ans.replace("[n]", "")
    # Trim trailing spaces
    ans = re.sub(r"[ \t]+$", "", ans, flags=re.MULTILINE).strip()
    # Facts asked but no bullets or Facts: section
    if wants_facts and ("Facts:" not in ans and "\n- " not in ans):
        return "Facts: Not stated in sources."
    # Ruling asked but nothing substantive produced
    if wants_ruling and "Ruling:" in ans and ("Not stated in sources" not in ans) and ("[" not in ans):
        ans += " [1]"
    return ans

# ---------- main entry ----------
KEY_TERMS_NEED_JURIS = {
    "g.r.", "gr no", "gr no.", "gr number", "sc gr",
    "people v", "people vs", "vs.", "v.", "en banc", "supreme court",
    "article ", "art. ", "section ", "sec. ", "rule ", "rule of court",
    "civil code", "revised penal code", "rpc", "constitution",
    "ruling", "decision", "wherefore", "so ordered", "facts", "issues",
    "doctrine", "held", "ratio", "precedent", "jurisprudence", "cite",
}


def _should_query_jurisprudence(query: str) -> bool:
    q = (query or "").lower()
    if not q:
        return False
    for t in KEY_TERMS_NEED_JURIS:
        if t in q:
            return True
    if re.search(r"\b(what|when|whether|who|how)\b", q) and re.search(r"\b(ruling|case|jurisprudence|article|section|rule|doctrine)\b", q):
        return True
    if re.search(r"\b\d{4}\b", q) and re.search(r"\b\w+\s+v\.?s?\.?\s+\w+", q):
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
    # Rule-based first
    msg = rb.answer(query)
    if msg is not None:
        return msg

    use_juris = _should_query_jurisprudence(query)
    if use_juris:
        if retriever is None:
            from .retriever import LegalRetriever
            retriever = LegalRetriever()
        docs = retriever.retrieve(query, k=8)
        if not docs:
            return _chat_without_retrieval(query, history or [])
    else:
        return _chat_without_retrieval(query, history or [])

    wants_ruling, wants_facts, wants_issues = _intent(query)
    chosen = _dedupe_and_rank(docs, wants_ruling, wants_facts, wants_issues)

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

    context = _build_context(chosen)

    # Missing-section flags to guide safe fallbacks
    must_note_absent_ruling = wants_ruling and (" — ruling — " not in context)
    must_note_absent_facts  = wants_facts  and (" — facts — "  not in context)

    # -------- Conditional answer template --------
    if wants_ruling and not (wants_facts or wants_issues):
        answer_template = (
            "Ruling: <if dispositive present, put the exact WHEREFORE/So Ordered text in double quotes, "
            "then a one-sentence explanation with [n]; otherwise write: Not stated in sources.>"
        )
    elif wants_facts and not (wants_ruling or wants_issues):
        answer_template = (
            "Facts:\n"
            "- <one fact, ≤25 words, end with a real [1]/[2]/... citation>\n"
            "- <one fact, ≤25 words, end with a real [n]>\n"
            "- <one fact, ≤25 words, end with a real [n]>\n"
            "(Write 3–6 bullets. Do not write any sentence without a bracketed number. Never write the literal '[n]'.)"
        )
    elif wants_issues and not (wants_ruling or wants_facts):
        answer_template = "Issues: <bullet list; each bullet ends with a real [n] citation>"
    else:
        # Default: full digest
        answer_template = (
            "Facts:\n- <short bullet with [n]>\n- <short bullet with [n]>\n"
            "Issues: <bullet list or a short sentence with [n]>\n"
            "Ruling: <if dispositive present, put the exact quote in double quotes, "
            "then a 1-sentence explanation with [n]; otherwise write: Not stated in sources.>"
        )
    # -------- End conditional --------

    # Structured, instruction-rich prompt
    history_block = _format_history(history or [])
    prompt = (
        "You are a legal assistant focused on Philippine Supreme Court jurisprudence.\n"
        "STRICT RULES:\n"
        "1) Use ONLY the information in Refs. Never invent facts.\n"
        "2) If something is not in Refs, write exactly: Not stated in sources.\n"
        "3) Every sentence must end with a bracketed citation like [1], [2]. Never output the literal '[n]'.\n"
        "4) If quoting the dispositive/WHEREFORE portion, quote it VERBATIM in double quotes.\n"
        "5) Keep bullets short (≤25 words each). No run-on bullets.\n"
        f"{'6) If no dispositive text appears in Refs, write: Ruling: Not stated in sources.' if must_note_absent_ruling else ''}\n"
        f"{'6) If no facts appear in Refs, write: Facts: Not stated in sources.' if must_note_absent_facts else ''}\n\n"
        f"Conversation so far (if any):\n{history_block}\n\n" if history_block else ""
        f"Refs:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:\n{answer_template}"
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
    messages.append({
        "role": "user",
        "content": (
            "Follow these instructions strictly.\n"
            "STRICT RULES:\n"
            "1) Use ONLY the information in Refs. Never invent facts.\n"
            "2) If something is not in Refs, write exactly: Not stated in sources.\n"
            "3) Every sentence must end with a bracketed citation like [1], [2]. Never output the literal '[n]'.\n"
            "4) If quoting the dispositive/WHEREFORE portion, quote it VERBATIM in double quotes.\n"
            "5) Keep bullets short (≤25 words each).\n"
            f"{'6) If no dispositive text appears in Refs, write: Ruling: Not stated in sources.' if must_note_absent_ruling else ''}\n"
            f"{'6) If no facts appear in Refs, write: Facts: Not stated in sources.' if must_note_absent_facts else ''}\n\n"
            f"Refs:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:\n{answer_template}"
        )
    })

    # Prefer messages-based generation when possible
    try:
        raw = generate_response_from_messages(messages)
    except Exception:
        raw = generate_response(prompt)
    return _post_validate(raw, wants_facts, wants_ruling)


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
