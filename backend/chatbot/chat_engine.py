# chat_engine.py — section-aware, de-duplicated context + structured prompt
import re
from typing import Dict, List, Tuple

from .rule_based import RuleBasedResponder

rb = RuleBasedResponder(bot_name="PHLaw-Chatbot")


# Strong cue for dispositive section ending in SO ORDERED.
RULING_REGEX = re.compile(
    r"(?:WHEREFORE|ACCORDINGLY|IN VIEW OF THE FOREGOING|THUS|HENCE)[\s\S]*?SO ORDERED\.?",
    re.IGNORECASE | re.DOTALL,
)

# Fallback: grab any paragraph that ends in “SO ORDERED.” (many trial orders use this without WHEREFORE)
RULING_SO_ORDERED_FALLBACK = re.compile(
    r"(?s)(?:^|\n\n).{0,1200}?SO ORDERED\.?",
    re.IGNORECASE,
)

# Common headings for facts / issues (various styles)
FACTS_HINT_RE = re.compile(
    r"^\s*(?:Factual\s+(?:Antecedents|Background)|Antecedent\s+Facts|"
    r"Facts(?:\s+of\s+the\s+Case)?|The\s+Facts)\s*[:\-–]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

ISSUES_HINT_RE = re.compile(
    r"^\s*(?:Issues?(?:\s+for\s+Resolution)?|Questions?\s+Presented|Issue)\s*[:\-–]?\s*$"
    r"|^\s*(?:[IVX]+\.)?\s*Whether\b",
    re.IGNORECASE | re.MULTILINE,
)

# Keep a single retriever instance
retriever = None

# ---------- small utils ----------
def _normalize_ws(s: str, max_chars: int = 1200) -> str:
    if not s:
        return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_chars]

def _ensure_section(doc: dict) -> str:
    sec = (doc.get("section") or "").strip().lower()
    if sec:
        return sec

    txt = doc.get("text") or ""
    if not txt:
        return "body"

    # 1) Ruling detection (strong header or plain “SO ORDERED.” paragraph)
    if RULING_REGEX.search(txt) or RULING_SO_ORDERED_FALLBACK.search(txt):
        return "ruling"

    # 2) Facts / Issues cues — pick whichever appears first in the text
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


def _intent(query: str) -> Tuple[bool, bool, bool]:
    q = (query or "").lower()
    wants_ruling = any(t in q for t in ["ruling", "decision", "disposition", "wherefore", "so ordered"])
    wants_facts  = bool(FACTS_HINT_RE.search(q))
    wants_issues = bool(ISSUES_HINT_RE.search(q))
    return wants_ruling, wants_facts, wants_issues

def _title(doc: dict) -> str:
    return doc.get("title") or doc.get("gr_number") or doc.get("filename") or "Untitled case"

def _source(doc: dict) -> str:
    return doc.get("source_url") or "N/A"

# ---------- context builder ----------
SECTION_PRIORITY = ["ruling", "issues", "facts", "header", "body"]

def _dedupe_and_rank(docs: List[Dict], wants_ruling: bool, wants_facts: bool, wants_issues: bool) -> List[Dict]:
    """Prefer one high-quality snippet per (url, section), keep higher score."""
    # Ensure sections and default scores
    for d in docs:
        d["section"] = _ensure_section(d)
        d["score"] = float(d.get("score") or 0.0)

    # Keep best per (url, section)
    best = {}
    for d in docs:
        key = (_source(d), d["section"])
        if key not in best or d["score"] > best[key]["score"]:
            best[key] = d

    # Spread: try to assemble context as ruling/facts/issues first, then header/body
    buckets = {sec: [] for sec in SECTION_PRIORITY}
    for d in best.values():
        if d["section"] in buckets:
            buckets[d["section"]].append(d)

    # Sort each bucket by score desc
    for sec in buckets:
        buckets[sec].sort(key=lambda x: x.get("score", 0.0), reverse=True)

    plan: List[str]
    if wants_ruling and not (wants_facts or wants_issues):
        plan = ["ruling", "issues", "facts", "header", "body"]
    elif wants_facts and not (wants_ruling or wants_issues):
        plan = ["facts", "issues", "ruling", "header", "body"]
    elif wants_issues and not (wants_ruling or wants_facts):
        plan = ["issues", "facts", "ruling", "header", "body"]
    else:
        plan = SECTION_PRIORITY

    ordered: List[Dict] = []
    seen_urls = set()

    # First pass: take at most one per section (highest score), avoid repeating the same case first
    for sec in plan:
        for d in buckets.get(sec, []):
            url = _source(d)
            if url in seen_urls:
                continue
            ordered.append(d)
            seen_urls.add(url)
            break  # only one per section in this pass

    # Second pass: fill remaining slots with next best regardless of section, but avoid duplicate (url, section)
    if len(ordered) < 4:
        pool = [d for arr in buckets.values() for d in arr]
        # de-dupe by (url, section) against already picked
        picked_keys = {(_source(d), d["section"]) for d in ordered}
        pool.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        for d in pool:
            key = (_source(d), d["section"])
            if key in picked_keys:
                continue
            ordered.append(d)
            picked_keys.add(key)
            if len(ordered) >= 4:
                break

    return ordered[:4]

def _build_context(docs: List[Dict]) -> str:
    """
    Number refs as [1], [2], ... so the generator can cite them easily.
    Each ref shows Title — Section — URL, then a compact snippet.
    """
    lines = []
    for i, d in enumerate(docs, 1):
        title = _title(d)
        url   = _source(d)
        sec   = d.get("section", "body")
        snippet = _normalize_ws(d.get("text") or "")
        header = f"[{i}] {title} — {sec} — {url}"
        lines.append(f"{header}\n{snippet}")
    return "\n\n".join(lines)

# ---------- main entry ----------
def chat_with_law_bot(query: str):
    global retriever
    msg = rb.answer(query)
    if msg is not None:
        return msg

    if retriever is None:
        from .retriever import LegalRetriever
        retriever = LegalRetriever()

    # Pull a bit more to allow diversification
    docs = retriever.retrieve(query, k=6)
    if not docs:
        return "No relevant jurisprudence found."

    wants_ruling, wants_facts, wants_issues = _intent(query)

    chosen = _dedupe_and_rank(docs, wants_ruling, wants_facts, wants_issues)
    context = _build_context(chosen)

    # Structured, instruction-rich prompt
    prompt = (
        "You are a legal assistant focused on Philippine Supreme Court jurisprudence.\n"
        "Use only the Refs below. When quoting the dispositive portion, quote it VERBATIM.\n"
        "If the user asks generally about a case, provide concise sections in this order when available:\n"
        "Facts — summarize key facts from refs; Issues — list the legal issues; Ruling — quote verbatim then explain briefly.\n"
        "Cite using bracketed numerals matching the Refs, e.g., [1], [2]. If a section is missing, say 'Not stated in sources.'\n\n"
        f"Refs:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    from .generator import generate_response
    return generate_response(prompt)
