# chat_engine.py — section-aware, de-duplicated context + structured prompt (refined)
import re
from typing import Dict, List, Tuple

from .rule_based import RuleBasedResponder
rb = RuleBasedResponder(bot_name="PHLaw-Chatbot")

# --- Stronger dispositive detection & extraction ---
# Capture classic dispositive headers and allow common variants before SO ORDERED
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

# ---------- context builder ----------
SECTION_PRIORITY = ["ruling", "issues", "facts", "header", "body"]

def _dedupe_and_rank(docs: List[Dict], wants_ruling: bool, wants_facts: bool, wants_issues: bool) -> List[Dict]:
    """
    Prefer one high-quality snippet per (case_id, section), keep higher score.
    Also merge adjacent tiny ruling chunks from the same case if needed.
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

# ---------- main entry ----------
def chat_with_law_bot(query: str):
    global retriever
    # Rule-based first
    msg = rb.answer(query)
    if msg is not None:
        return msg

    if retriever is None:
        from .retriever import LegalRetriever
        retriever = LegalRetriever()

    # Pull more for diversification; retriever should already rank by semantic+BM25 hybrid if possible
    docs = retriever.retrieve(query, k=8)
    if not docs:
        return "No relevant jurisprudence found."

    wants_ruling, wants_facts, wants_issues = _intent(query)
    chosen = _dedupe_and_rank(docs, wants_ruling, wants_facts, wants_issues)
    context = _build_context(chosen)

    # If the user clearly wants the ruling but none of the chosen refs contains a dispositive, tell the model to say so
    must_note_absent_ruling = wants_ruling and not any(" — ruling — " in line for line in context.splitlines() if line.startswith("["))

    # Structured, instruction-rich prompt with stronger guardrails
    prompt = (
        "You are a legal assistant focused on Philippine Supreme Court jurisprudence.\n"
        "STRICT RULES:\n"
        "1) Use ONLY the information in Refs. Never invent facts.\n"
        "2) If something is not in Refs, write exactly: Not stated in sources.\n"
        "3) Always add bracketed citations [n] immediately after the sentence they support.\n"
        "4) If quoting the dispositive/WHEREFORE portion, quote it VERBATIM in double quotes.\n"
        "5) Do not mix issues into the ruling. The ruling is only the Court's disposition.\n"
        "6) Keep the answer concise (≈180–220 words) unless the user asked for more.\n"
        "7) Organize when applicable: Facts, Issues, Ruling. One bullet per line for facts.\n"
        f"{'8) If no dispositive text appears in Refs, state: Ruling: Not stated in sources.' if must_note_absent_ruling else ''}\n\n"
        f"Refs:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer (concise; cite like [1], [2]):\n"
        "Facts:\n- <short bullet with [n]>\n- <short bullet with [n]>\n"
        "Issues: <bullet list or a short sentence with [n]>\n"
        "Ruling: <if dispositive present, put the exact quote in double quotes, then a 1-sentence explanation with [n]; "
        "otherwise write: Not stated in sources.>"
    )

    from .generator import generate_response
    return generate_response(prompt)
