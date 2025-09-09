# backend/chatbot/rule_based.py
from __future__ import annotations
import os, re, json, difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# -----------------------------------------------------------------------------
# Case-patterns: if these match, we DEFER to retriever/LLM (return None)
# -----------------------------------------------------------------------------
GR_RE = re.compile(r"\bG\.\s*R\.\s*(?:No\.|Nos\.)?\s*\d{1,6}(?:-\d{1,6})?\b", re.I)
CASE_NAME_RE = re.compile(r"\b([A-Z][A-Za-z'.()\- ]+)\s+v\.?s?\.?\s+([A-Z][A-Za-z'.()\- ]+)\b")

# -----------------------------------------------------------------------------
# Query-intent patterns we CAN answer deterministically (glossary fast path)
# -----------------------------------------------------------------------------
DEFINE_RE   = re.compile(r"^(what\s+is|define|meaning\s+of|ano\s+ang)\b", re.I)
ELEMENTS_RE = re.compile(r"\b(elements|tests?)\s+of\s+(.+)", re.I)
RULE_45_65_RE = re.compile(r"rule\s*45\s*(?:vs|versus|v\.?)\s*65", re.I)

# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------
NORMALIZE_MAP = {
    "versus": "v.", "vs.": "v.", "vs": "v.",
    "people of the philippines": "people",
    "sec.": "section", "§": "section",
    "ra ": "republic act ", "r.a.": "republic act",
}
def _norm(s: str) -> str:
    s = s or ""
    s = s.strip().lower()
    for a, b in NORMALIZE_MAP.items():
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s)
    return s

# -----------------------------------------------------------------------------
# Glossary index (supports multiple JSON sources; tolerant schema)
#  - Accepts: {"entries":[...]} OR a plain list of entries
#  - Each entry: {"term", "aliases"[], "definition", "elements"[], "anchors"[] ...}
#  - Also tolerates a file with {"terms":[...]} (no definitions) -> ignored
# -----------------------------------------------------------------------------
class GlossaryIndex:
    def __init__(self, entries: List[Dict[str, Any]]):
        # Keep only entries with definitions
        clean: List[Dict[str, Any]] = []
        for e in entries:
            term = (e.get("term") or "").strip()
            defi = (e.get("definition") or "").strip()
            if term and defi:
                # normalize aliases to list[str]
                aliases = e.get("aliases") or []
                if isinstance(aliases, str):
                    aliases = [aliases]
                e["aliases"] = [a for a in (x.strip() for x in aliases) if a]
                clean.append(e)

        self.entries = clean
        self.by_term: Dict[str, Dict[str, Any]] = {}
        self.alias_to_term: Dict[str, str] = {}

        for e in self.entries:
            canon = _norm(e["term"])
            self.by_term[canon] = e
            for a in e.get("aliases", []):
                self.alias_to_term[_norm(a)] = canon

        # Build a name set (terms + aliases) for fuzzy search
        self.name_set = set(self.by_term.keys()) | set(self.alias_to_term.keys())

    @staticmethod
    def _load_one(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(data, dict):
            if "entries" in data and isinstance(data["entries"], list):
                return data["entries"]
            # ignore {"terms": [...]} lists without definitions
            return []
        if isinstance(data, list):
            return data
        return []

    @classmethod
    def from_files(cls, paths: List[Path]) -> "GlossaryIndex":
        all_entries: List[Dict[str, Any]] = []
        for p in paths:
            all_entries.extend(cls._load_one(p))
        # Minimal seed if nothing loaded (so app still boots)
        if not all_entries:
            all_entries = SEED_ENTRIES
        return cls(all_entries)

    # quick intent test
    def is_definition_query(self, q: str) -> bool:
        qn = _norm(q)
        return bool(DEFINE_RE.search(qn) or ELEMENTS_RE.search(qn))

    # main search: exact → substring → fuzzy (difflib)
    def search(self, q: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        qn = _norm(q)
        results: List[Tuple[float, Dict[str, Any]]] = []

        # 1) exact alias
        if qn in self.alias_to_term:
            e = self.by_term[self.alias_to_term[qn]]
            return [(100.0, e)]

        # 2) exact canonical
        if qn in self.by_term:
            return [(99.0, self.by_term[qn])]

        # 3) substring match (aliases + terms)
        cand_idx: Dict[str, Dict[str, Any]] = {}
        for name in self.name_set:
            if name in qn or qn in name:
                canon = self.alias_to_term.get(name, name)
                e = self.by_term.get(canon)
                if e:
                    cand_idx[canon] = e
        for e in cand_idx.values():
            # small boost for more bar-relevant terms, if present
            score = 82.0 + (e.get("bar_importance", 3) - 3) * 2.0
            results.append((score, e))

        # 4) fuzzy (edit distance similarity)
        names = list(self.name_set)
        close = difflib.get_close_matches(qn, names, n=k*3, cutoff=0.72)
        for name in close:
            canon = self.alias_to_term.get(name, name)
            e = self.by_term.get(canon)
            if not e:
                continue
            ratio = difflib.SequenceMatcher(a=qn, b=name).ratio()
            score = 70.0 + ratio * 20.0 + (e.get("bar_importance", 3) - 3) * 2.0
            results.append((score, e))

        # de-dup + rank
        best: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        for sc, e in results:
            key = _norm(e["term"])
            if key not in best or sc > best[key][0]:
                best[key] = (sc, e)

        ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)[:k]
        return ranked

    def format_answer(self, entry: Dict[str, Any], *, want_elements: bool = False) -> str:
        parts: List[str] = [f"{entry['term']}: {entry['definition']}"]
        els = entry.get("elements") or []
        if want_elements and els:
            parts.append("Elements:\n" + "\n".join(f"- {x}" for x in els))
        anchors = entry.get("anchors") or []
        if anchors:
            a = "; ".join(f"{x.get('type','')}: {x.get('id','')}" for x in anchors if x)
            if a.strip():
                parts.append(f"Anchors: {a}")
        return "\n".join(parts)

# -----------------------------------------------------------------------------
# Optional seed (keeps bot usable if JSON files are missing)
# -----------------------------------------------------------------------------
SEED_ENTRIES: List[Dict[str, Any]] = [
    {
        "term": "Writ of Amparo",
        "aliases": ["amparo"],
        "definition": "Extraordinary remedy protecting the rights to life, liberty, and security against actual or threatened violations.",
        "anchors": [{"type": "AM", "id": "A.M. No. 07-9-12-SC"}],
        "bar_importance": 5,
        "elements": []
    },
    {
        "term": "Writ of Habeas Data",
        "aliases": ["habeas data"],
        "definition": "Remedy allowing access, correction, or destruction of personal data held by government or private entities to protect privacy in relation to life, liberty, or security.",
        "anchors": [{"type": "AM", "id": "A.M. No. 08-1-16-SC"}],
        "bar_importance": 5,
        "elements": []
    },
    {
        "term": "Writ of Kalikasan",
        "aliases": ["kalikasan"],
        "definition": "Special remedy addressing environmental damage of such magnitude as to prejudice life, health, or property of inhabitants in two or more cities or provinces.",
        "anchors": [{"type": "AM", "id": "A.M. No. 09-6-8-SC"}],
        "bar_importance": 5,
        "elements": []
    },
    {
        "term": "Rule 45",
        "aliases": ["petition for review on certiorari"],
        "definition": "Mode of appeal to the Supreme Court on questions of law.",
        "anchors": [{"type": "Rule", "id": "Rule 45"}],
        "bar_importance": 5,
        "elements": []
    },
    {
        "term": "Rule 65",
        "aliases": ["certiorari", "prohibition", "mandamus"],
        "definition": "Special civil actions to correct acts without or in excess of jurisdiction or with grave abuse of discretion; not a substitute for a lost appeal.",
        "anchors": [{"type": "Rule", "id": "Rule 65"}],
        "bar_importance": 5,
        "elements": []
    },
    {
        "term": "Chain of Custody (RA 9165)",
        "aliases": ["chain of custody", "section 21 ra 9165", "sec 21 ra 9165"],
        "definition": "Recorded movement and safekeeping of seized drugs from seizure to presentation in court (marking, inventory, photographing, witness presence, turnover).",
        "anchors": [{"type": "Statute", "id": "Sec. 21, RA 9165"}],
        "bar_importance": 5,
        "elements": []
    },
    {
        "term": "Forum shopping",
        "aliases": [],
        "definition": "Filing multiple actions involving the same parties and causes to secure a favorable judgment; prohibited and sanctionable.",
        "anchors": [],
        "bar_importance": 4,
        "elements": []
    },
    {
        "term": "Res judicata",
        "aliases": [],
        "definition": "A final judgment on the merits by a court of competent jurisdiction bars re-litigation of the same cause between the same parties.",
        "anchors": [],
        "bar_importance": 4,
        "elements": []
    },
]

# -----------------------------------------------------------------------------
# Resolve glossary files (you can list multiple files, comma-separated)
# Defaults try your merged + scribd JSON; falls back to seed if absent.
# -----------------------------------------------------------------------------
def _resolve_glossary_files() -> List[Path]:
    env = os.getenv("GLOSSARY_FILES", "")
    candidates: List[Path] = []
    if env.strip():
        for p in env.split(","):
            path = Path(p.strip())
            if path.suffix.lower() == ".json":
                candidates.append(path)
    else:
        # Common defaults in your repo
        candidates = [
            Path("backend/data/glossary.json"),
            Path("backend/data/ph_legal_dictionary_merged.json"),
            Path("backend/data/scribd_terms_with_meanings.json"),
            Path("/mnt/data/ph_legal_dictionary_merged.json"),
            Path("/mnt/data/scribd_terms_with_meanings.json"),
        ]
    # keep only existing, but let GlossaryIndex handle empty
    return [p for p in candidates if p.exists()]

_GLOSSARY = GlossaryIndex.from_files(_resolve_glossary_files())

# -----------------------------------------------------------------------------
# Public class used by chat_engine.py
# -----------------------------------------------------------------------------
@dataclass
class RuleBasedResponder:
    bot_name: str = os.getenv("BOT_NAME", "PHLaw-Chatbot")

    def answer(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """
        Jurisprudence-only, deterministic answers.
        Returns a string response or None (to defer to retrieval + LLM).
        """
        q = (query or "").strip()
        if not q:
            return None

        # If precise case reference, DEFER
        if GR_RE.search(q) or CASE_NAME_RE.search(q):
            return None

        # Rule 45 vs 65 quick distinction
        if RULE_45_65_RE.search(q):
            a = _GLOSSARY.search("Rule 45", k=1)
            b = _GLOSSARY.search("Rule 65", k=1)
            if a and b:
                defn45 = a[0][1].get("definition", "").strip()
                defn65 = b[0][1].get("definition", "").strip()
                if defn45 and defn65:
                    return (
                        f"Rule 45: {defn45}\n"
                        f"Rule 65: {defn65}\n"
                        "Use Rule 45 for errors of law in judgments; Rule 65 for jurisdictional errors (grave abuse of discretion)."
                    )
            return None  # let LLM explain if glossary lacks both

        # Definition / elements queries → glossary fast path
        qn = _norm(q)
        want_elements = False
        if DEFINE_RE.search(qn):
            # strip leading "what is/define/meaning of/ano ang"
            m = re.search(r"^(?:what\s+is|define|meaning\s+of|ano\s+ang)\s+(.+)", qn, re.I)
            key = (m.group(1) if m else q).strip(" ?.")
            matches = _GLOSSARY.search(key, k=3)
            if matches and matches[0][0] >= 78.0:
                return _GLOSSARY.format_answer(matches[0][1], want_elements=False)
            return None

        m_el = ELEMENTS_RE.search(q)
        if m_el:
            want_elements = True
            key = m_el.group(2).strip(" ?.")
            matches = _GLOSSARY.search(key, k=3)
            if matches and matches[0][0] >= 78.0:
                return _GLOSSARY.format_answer(matches[0][1], want_elements=True)
            return None

        # Not a deterministic glossary-style query → defer to retrieval/LLM
        return None

# -----------------------------------------------------------------------------
# Manual smoke test (run this file directly)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rb = RuleBasedResponder()
    tests = [
        "What is amparo?",
        "define rule 65",
        "meaning of chain of custody",
        "elements of forum shopping",
        "Rule 45 vs 65",
        "What is the ruling in G.R. No. 211089?",
        "People v. Dizon",
        "ano ang writ of kalikasan",
    ]
    for t in tests:
        ans = rb.answer(t)
        print(f"{t} -> {ans if ans is not None else 'None (defer)'}")
