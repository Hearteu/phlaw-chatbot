# backend/chatbot/rule_based.py
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# -----------------------------------------------------------------------------
# Patterns (PH jurisprudence-focused)
# -----------------------------------------------------------------------------
GR_RE = re.compile(r"\bG\.\s*R\.\s*(?:No\.|Nos\.)?\s*\d{1,6}(?:-\d{1,6})?\b", re.I)
CASE_NAME_RE = re.compile(r"\b([A-Z][A-Za-z'.()\- ]+)\s+v\.?s?\.?\s+([A-Z][A-Za-z'.()\- ]+)\b")
DIGEST_RE = re.compile(r"\b(case\s+digest|make\s+a\s+digest|full\s+digest|complete\s+digest|FIR)\b", re.I)
DEFINE_RE = re.compile(r"^(?:what\s+is|define|meaning\s+of|ano\s+ang)\s+(.+?)\??$", re.I)
ELEMENTS_RE = re.compile(r"\b(elements|tests?)\s+of\s+(.+)", re.I)

# Minimal jurisprudence FAQs (expand as needed)
_FAQ_RULES: Dict[str, str] = {
    "rule 45 vs 65": (
        "Rule 45 (Petition for Review on Certiorari): appeal on questions of law to the Supreme Court; "
        "assails errors of law in the judgment.\n"
        "Rule 65 (Certiorari/Prohibition/Mandamus): special civil actions to correct acts without or in excess "
        "of jurisdiction or with grave abuse of discretion; not a substitute for a lost appeal."
    ),
    "court hierarchy": (
        "Hierarchy: MTC/MeTC/MTCC → RTC → Court of Appeals/CTA → Supreme Court. "
        "Direct recourse to the SC requires special, compelling reasons."
    ),
    "standards of review": (
        "Standards (illustrative): errors of law (de novo), grave abuse of discretion (jurisdictional error), "
        "substantial evidence (administrative), probable cause (abuse of discretion)."
    ),
}

# Concise glossary/doctrine cards (keep short; retrieval will provide full text)
_GLOSSARY: Dict[str, str] = {
    "certiorari": "Extraordinary remedy under Rule 65 to correct acts done without or in excess of jurisdiction, or with grave abuse of discretion.",
    "prohibition": "Rule 65 remedy to restrain a tribunal from further acting without or in excess of jurisdiction.",
    "mandamus": "Rule 65 remedy to compel performance of a ministerial duty when there is no other plain, speedy, and adequate remedy.",
    "estoppel": "A party is precluded from asserting a claim or fact contrary to prior statements, conduct, or admissions.",
    "grave abuse of discretion": "Capricious and whimsical exercise of judgment equivalent to lack or excess of jurisdiction.",
    "substantial evidence": "Relevant evidence that a reasonable mind might accept as adequate to support a conclusion (administrative cases).",
    "burden of proof": "Duty of a party to establish by evidence the truth of his claims; in civil cases, by preponderance; in criminal, beyond reasonable doubt.",
}

_DOCTRINE_ELEMENTS: Dict[str, Dict[str, List[str]]] = {
    "warrantless search exceptions": {
        "title": "Exceptions to the warrant requirement",
        "elements": [
            "Search incidental to a lawful arrest",
            "Plain view doctrine",
            "Consent search",
            "Stop-and-frisk (Terry stop)",
            "Moving vehicle exception",
            "Customs/border searches",
        ],
    },
    "probable cause": {
        "title": "Probable cause (criminal complaints/informations)",
        "elements": [
            "Facts and circumstances that would lead a reasonably prudent person to believe a crime was committed",
            "That the person charged is probably guilty thereof",
        ],
    },
}

# -----------------------------------------------------------------------------
# Helper: if the query is an exact case reference, defer to retriever/LLM
# -----------------------------------------------------------------------------
def _looks_like_exact_case(q: str) -> bool:
    return bool(GR_RE.search(q) or CASE_NAME_RE.search(q))

# -----------------------------------------------------------------------------
# Main class (jurisprudence-only)
# -----------------------------------------------------------------------------
@dataclass
class RuleBasedResponder:
    bot_name: str = os.getenv("BOT_NAME", "PHLaw-Chatbot")

    def answer(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """
        Return a jurisprudence-related, deterministic response, or None to defer.
        This module intentionally omits non-legal chit-chat, time/date, math, etc.
        """
        q = (query or "").strip()
        if not q:
            return None

        # If user already gave a precise case reference, defer to retrieval/LLM
        if _looks_like_exact_case(q):
            return None

        lo = q.lower()

        # 1) Case digest scaffold (safe to return instantly)
        if DIGEST_RE.search(lo):
            return self._digest_scaffold()

        # 2) FAQs (static)
        for key, ans in _FAQ_RULES.items():
            if key in lo:
                return ans

        # 3) Elements / tests cards
        m = ELEMENTS_RE.search(q)
        if m:
            topic = m.group(2).strip().lower()
            card = _DOCTRINE_ELEMENTS.get(topic) or _DOCTRINE_ELEMENTS.get(topic.rstrip("s"))
            if card:
                bullets = "\n".join(f"- {x}" for x in card["elements"])
                return f"{card['title']}:\n{bullets}"
            # unknown topic → defer
            return None

        # 4) Short legal definitions (glossary)
        m = DEFINE_RE.search(q)
        if m:
            term = m.group(1).strip().lower().rstrip(".")
            # normalize common phrasing
            term = term.replace("rule 65 certiorari", "certiorari")
            if term in _GLOSSARY:
                return f"{term.title()}: {_GLOSSARY[term]}"
            return None  # defer to retrieval/LLM for richer, case-grounded definition

        # 5) Generic jurisprudence prompts without specifics → nudge
        if "ruling" in lo or "wherefore" in lo or "so ordered" in lo:
            return ("To quote the ruling/dispositive verbatim, please provide a G.R. number or case name "
                    "(e.g., 'G.R. No. 211089' or 'People v. Dizon').")

        if "issue" in lo and "whether" not in lo:
            return ("Frame the issue using the 'Whether or not …' convention. "
                    "Example: 'Whether or not the warrantless search was valid.'")

        # Nothing deterministic to do → let chat_engine handle (retrieval + LLM)
        return None

    # -------------------------------------------------------------------------
    # Scaffolds
    # -------------------------------------------------------------------------
    def _digest_scaffold(self) -> str:
        return (
            "Issue\n"
            "1) Whether or not ______.\n"
            "2) Whether or not ______. (if applicable)\n\n"
            "Facts\n"
            "- Substantive narrative first (who/what/when/where/why).\n"
            "- Then procedural history (RTC → CA → SC).\n\n"
            "Ruling\n"
            "- Doctrine/Rule: ______\n"
            "- Application to facts: ______\n"
            "- Lower courts:\n"
            "  • RTC: ______\n"
            "  • CA:  ______\n"
            "- Dispositive (quote if available): \"______ SO ORDERED.\"\n\n"
            "Discussion\n"
            "- Separate opinions (if any): ______\n\n"
            "Citations (when applicable)\n"
            "- [Case v. Case], G.R. No. _____, [Date] — [one-line takeaway].\n\n"
            "Note: If a detail isn't in the sources, write: 'Not stated in sources.'"
        )


# Manual smoke test (jurisprudence-only)
if __name__ == "__main__":
    rb = RuleBasedResponder()
    tests = [
        "Make a case digest of this case",
        "Rule 45 vs 65",
        "What is certiorari?",
        "Elements of probable cause",
        "What is the ruling in G.R. No. 211089?",
        "Facts and issues please",
        "Define grave abuse of discretion",
        "Court hierarchy",
        "People v. Dizon (2018)",
    ]
    for t in tests:
        print(t, "->", rb.answer(t) or "None (defer)")