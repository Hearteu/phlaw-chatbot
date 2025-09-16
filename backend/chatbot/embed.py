# embed.py — Minimal embedding pipeline for simplified GR-number vs keyword logic
# Keep only essential JSONL pipeline, simple overlapping chunking, and concise metadata
import gzip
import json
import os
import re
import time
import unicodedata
import uuid
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "jurisprudence")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Stern5497/sbert-legal-xlm-roberta-base")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 768))

# Source data (JSONL only)
DATA_FILE   = os.getenv("DATA_FILE", "backend/data/cases.jsonl.gz")
# Optional year range filter when processing JSONL
YEAR_START  = int(os.getenv("YEAR_START", 2005))
YEAR_END    = int(os.getenv("YEAR_END", 2005))


# Chunking/throughput
CHUNK_CHARS   = int(os.getenv("CHUNK_CHARS", 2000))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", 200))
BATCH_SIZE    = int(os.getenv("EMBED_BATCH_SIZE", 32))
UPSERT_BATCH  = int(os.getenv("UPSERT_BATCH", 1024))

# -----------------------------
# JSONL retrieval helper
# -----------------------------
def load_case_from_jsonl(case_id: str, jsonl_path: str = DATA_FILE) -> Optional[Dict[str, Any]]:
    """Load full case text from JSONL file by case ID"""
    try:
        with gzip.open(jsonl_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                case = json.loads(line)
                if case.get('id') == case_id or case.get('gr_number') == case_id:
                    return case
    except Exception as e:
        print(f"Error loading case {case_id}: {e}")
    return None

# -----------------------------
# Qdrant init
# -----------------------------
client = QdrantClient(
    host=QDRANT_HOST, port=QDRANT_PORT, grpc_port=6334, prefer_grpc=True, timeout=120.0
)
if not client.collection_exists(QDRANT_COLLECTION):
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
print(f"✅ Qdrant collection ready: {QDRANT_COLLECTION}")

# (TXT mode removed)

# -----------------------------
# Helpers
# -----------------------------
RULING_REGEX = re.compile(
    r"(WHEREFORE.*?SO ORDERED\.?|ACCORDINGLY.*?SO ORDERED\.?)",
    re.IGNORECASE | re.DOTALL,
)
WS_RE = re.compile(r"\s+")
PUNCT_FIX_RE = re.compile(r"\s+([,.;:!?])")
CASE_TITLE_LINE_RE = re.compile(
    r"^(?P<title>.+?)(?:\s*;\s*G\.R\.|\s*G\.R\.|\s*\(G\.R\.|\s*\[G\.R\.|\s*\d{4}|\s*Promulgated|\s*Decided)",
    re.IGNORECASE | re.MULTILINE,
)
CASE_VS_RE = re.compile(r"\b.+?\s+(v\.|vs\.?|versus)\s+.+?\b", re.IGNORECASE)
CASE_CAPTION_STRONG_RE = re.compile(
    r"^[A-Z0-9 ,.'\-()]+\s+(V\.|VS\.?|VERSUS)\s+[A-Z0-9 ,.'\-()]+$"
)
ADDRESS_NOISE_TERMS = (
    "padre faura",
    "ermita",
    "manila",
    "philippines",
)
GR_BLOCK_RE = re.compile(
    r"G\.R\.\s+No(?:s)?\.\?\s*([0-9][0-9\-*,\s]+)",
    re.IGNORECASE,
)

# -----------------------------
# Lightweight case-type detection
# -----------------------------
CASE_TYPE_PATTERNS: Dict[str, re.Pattern] = {
    "annulment": re.compile(r"\bannulment\b|\bnullity of marriage\b|\bvoid marriage\b", re.IGNORECASE),
    "habeas_corpus": re.compile(r"\bhabeas\s+corpus\b", re.IGNORECASE),
    "mandamus": re.compile(r"\bmandamus\b", re.IGNORECASE),
    "prohibition": re.compile(r"\bprohibition\b", re.IGNORECASE),
    "certiorari": re.compile(r"\bcertiorari\b", re.IGNORECASE),
    "election": re.compile(r"\belection\b|\bcomelec\b|\belectoral\b", re.IGNORECASE),
    "labor": re.compile(r"\blabor\b|\bill?egal\s+dismissal\b|\bnlrc\b", re.IGNORECASE),
    "criminal": re.compile(r"\bpeople of the philippines\b|\bcriminal\b|\binformation\b", re.IGNORECASE),
    "estafa": re.compile(r"\bestafa\b", re.IGNORECASE),
    "murder": re.compile(r"\bmurder\b", re.IGNORECASE),
    "homicide": re.compile(r"\bhomicide\b", re.IGNORECASE),
    "rape": re.compile(r"\brape\b", re.IGNORECASE),
    "robbery": re.compile(r"\brobbery\b", re.IGNORECASE),
    "theft": re.compile(r"\btheft\b", re.IGNORECASE),
    "tax": re.compile(r"\btax\b|\bbir\b|\btc\b(?![a-z])", re.IGNORECASE),
    "administrative": re.compile(r"\badministrative\b|\bombudsman\b|\bcsc\b", re.IGNORECASE),
    "agrarian": re.compile(r"\bagrarian\b|\bdar\b|\bcarp\b", re.IGNORECASE),
    "ip": re.compile(r"\bintellectual\s+property\b|\btrademark\b|\bpatent\b|\bcopyright\b", re.IGNORECASE),
    "family": re.compile(r"\bfamily\b|\bchild custody\b|\badoption\b|\bsupport\b", re.IGNORECASE),
    "contract": re.compile(r"\bcontract\b|\bspecific\s+performance\b|\brescission\b", re.IGNORECASE),
    "torts": re.compile(r"\btort\b|\bdamages\b|\bnegligence\b", re.IGNORECASE),
}

def derive_case_type(rec: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Infer a lightweight case_type and tags from header/body keywords.
    Returns (primary_type, tags).
    """
    sections = rec.get("sections") or {}
    header = sections.get("header") or ""
    body = sections.get("body") or rec.get("clean_text") or ""
    # Limit scanned text for performance
    text = f"{header}\n{body[:4000]}"

    tags: List[str] = []
    for label, pattern in CASE_TYPE_PATTERNS.items():
        try:
            if pattern.search(text):
                tags.append(label)
        except Exception:
            continue

    primary: Optional[str] = None
    if tags:
        # Prefer specific over generic when multiple match
        specificity_order = [
            "annulment", "habeas_corpus", "mandamus", "prohibition", "certiorari",
            "election", "labor", "estafa", "murder", "homicide", "rape", "robbery",
            "theft", "tax", "administrative", "agrarian", "ip", "family", "contract",
            "torts", "criminal",
        ]
        for label in specificity_order:
            if label in tags:
                primary = label
                break

    return primary, tags

def find_ruling(text: str) -> Tuple[int, int]:
    """Return (start, end) of the ruling section, or (-1, -1) if not found."""
    m = RULING_REGEX.search(text or "")
    return (m.start(), m.end()) if m else (-1, -1)

def normalize_text(s: str) -> str:
    """Collapse whitespace/newlines; lightly clean spacing around punctuation."""
    if not s:
        return ""
    # Normalize unicode, unify whitespace
    s = unicodedata.normalize("NFKC", s)
    s = WS_RE.sub(" ", s.replace("\r", "\n")).strip()
    # Remove stray spaces before punctuation: "word ." -> "word."
    s = PUNCT_FIX_RE.sub(r"\1", s)
    return s

def _clean_library_noise(s: str) -> str:
    if not s:
        return s
    return (
        s.replace("Supreme Court E-Library", "").replace("Information At Your Fingertips", "").strip()
    )

def derive_case_title(rec: Dict[str, Any]) -> str:
    """Derive a full case title (complete parties), preferring explicit captions.
    Preference order:
      1) explicit case_title/case_name
      2) strongest caption line in header containing v./vs./versus (choose longest reasonable)
      3) strongest caption line in early body
      4) regex-parsed title line from header
      5) provided title
      6) filename/id/source_url
    """
    # 1) Explicit field
    title = (rec.get("case_title") or rec.get("case_name") or "").strip()
    if title:
        return _clean_library_noise(title)

    def is_address_noise(line: str) -> bool:
        l = line.lower()
        return any(term in l for term in ADDRESS_NOISE_TERMS)

    def pick_strongest_caption(lines: List[str]) -> Optional[str]:
        candidates: List[str] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if is_address_noise(s):
                continue
            if CASE_CAPTION_STRONG_RE.match(s) or CASE_VS_RE.search(s):
                candidates.append(s)
        if not candidates:
            return None
        # Return the longest up to a soft cap (avoid absurd lines)
        candidates.sort(key=lambda x: len(x), reverse=True)
        best = candidates[0]
        return best if len(best) <= 240 else best[:240].rstrip()

    # 2) Parse from header section
    sections = rec.get("sections") or {}
    header = _clean_library_noise(sections.get("header") or "")
    if header:
        strongest = pick_strongest_caption(header.splitlines())
        if strongest:
            return strongest
        m = CASE_TITLE_LINE_RE.search(header)
        if m and m.group("title"):
            return m.group("title").strip()
        # fallback: first non-empty line without boilerplate
        for line in header.splitlines():
            line = line.strip()
            if not line:
                continue
            if len(line) > 3 and not is_address_noise(line):
                return line

    # 3) Scan the beginning of body for a case caption
    body = _clean_library_noise(sections.get("body") or rec.get("clean_text") or "")
    if body:
        strongest_body = pick_strongest_caption(body[:2000].splitlines())
        if strongest_body:
            return strongest_body

    # 4) Provided title, cleaned
    raw_title = _clean_library_noise(rec.get("title") or "").strip()
    if raw_title and not raw_title.lower().startswith("supreme court e-library"):
        return raw_title

    # 5) Filename fallback
    filename = (rec.get("filename") or rec.get("id") or rec.get("source_url") or "").strip()
    return filename or "Untitled case"

def derive_gr_numbers(rec: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Parse G.R. number(s) from record sections.
    Returns (primary_gr_number, all_gr_numbers_list).
    Keeps original rec['gr_number'] as primary if present; otherwise first parsed.
    """
    existing = (rec.get("gr_number") or "").strip()
    found: List[str] = []

    sections = rec.get("sections") or {}
    header = sections.get("header") or ""
    body = sections.get("body") or rec.get("clean_text") or ""

    def _collect(text: str):
        if not text:
            return
        for m in GR_BLOCK_RE.finditer(text):
            block = m.group(1)
            # Split by comma and whitespace, keep ranges like 225568-70 as-is
            parts = [p.strip().rstrip('*') for p in re.split(r"[,\s]+", block) if p.strip()]
            for p in parts:
                if p and p not in found:
                    found.append(p)

    _collect(header)
    # Only scan the first 2k chars of body for performance
    _collect(body[:2000])

    primary = existing or (found[0] if found else None)
    return primary, found

def derive_special_numbers(rec: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Parse special case numbers (A.M., OCA, etc.) from record.
    Returns (primary_special_number, all_special_numbers_list).
    """
    existing = (rec.get("special_number") or "").strip()
    existing_list = rec.get("special_numbers") or []
    
    if existing:
        return existing, existing_list if isinstance(existing_list, list) else [existing]
    
    # If no existing special numbers, try to parse from text
    text = rec.get("clean_text") or ""
    if not text:
        return None, []
    
    # Use the same patterns as crawler
    special_patterns = [
        (r"A\.M\.\s+No\.?\s*([0-9\-]+[A-Z]?)", "A.M. No. {}"),
        (r"OCA\s+No\.?\s*([0-9\-]+[A-Z]?)", "OCA No. {}"),
        (r"U\.C\.\s+No\.?\s*([0-9\-]+[A-Z]?)", "U.C. No. {}"),
        (r"ADM\s+No\.?\s*([0-9\-]+[A-Z]?)", "ADM No. {}"),
        (r"AC\s+No\.?\s*([0-9\-]+[A-Z]?)", "AC No. {}"),
        (r"B\.M\.\s+No\.?\s*([0-9\-]+[A-Z]?)", "B.M. No. {}"),
        (r"LRC\s+No\.?\s*([0-9\-]+[A-Z]?)", "LRC No. {}"),
        (r"SP\s+No\.?\s*([0-9\-]+[A-Z]?)", "SP No. {}"),
    ]
    
    found: List[str] = []
    for pattern, format_str in special_patterns:
        for m in re.finditer(pattern, text[:4000], re.IGNORECASE):
            number = m.group(1).strip()
            formatted = format_str.format(number)
            if formatted not in found:
                found.append(formatted)
    
    primary = found[0] if found else None
    return primary, found


# (TXT readers removed)

# -----------------------------
# JSONL readers (jsonl mode)
# -----------------------------
def iter_cases(path: str):
    """
    Yield JSONL records from .jsonl or .jsonl.gz (one object per line).
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip malformed lines but continue
                continue

def text_from_record(rec: Dict[str, Any]) -> str:
    """
    Build a single text stream from record relying on clean_text and optional ruling.
    Order: ruling (if present) → clean_text. Whitespace-normalized.
    """
    ruling = rec.get("ruling", "")
    body = rec.get("clean_text", "") or rec.get("body", "")
    parts: List[str] = []
    if isinstance(ruling, str) and ruling.strip():
        parts.append(normalize_text(ruling))
    if isinstance(body, str) and body.strip():
        parts.append(normalize_text(body))
    return " ".join(p for p in parts if p).strip()

def record_meta(rec):
    """Minimal metadata for retrieval and display.
    Ensures normalized gr_number and a clean title for both GR and keyword paths.
    """
    # Year extraction with fallbacks
    y = rec.get("promulgation_year")
    if not isinstance(y, int):
        d = rec.get("promulgation_date")
        if isinstance(d, str) and len(d) >= 4 and d[:4].isdigit():
            try:
                y = int(d[:4])
            except Exception:
                y = None

    primary_gr, all_grs = derive_gr_numbers(rec)
    primary_special, all_special = derive_special_numbers(rec)
    title = derive_case_title(rec)
    # Prefer crawler-provided case_type/subtypes if present; else derive lightweight type/tags
    crawler_case_type = rec.get("case_type")
    crawler_subtypes = rec.get("case_subtypes") or (
        [rec.get("case_subtype")] if rec.get("case_subtype") else None
    )
    if crawler_case_type:
        case_type = crawler_case_type
        case_tags = list(crawler_subtypes) if isinstance(crawler_subtypes, list) else None
    else:
        case_type, case_tags = derive_case_type(rec)
    
    return {
        "gr_number": primary_gr,
        "gr_numbers": all_grs or None,
        "special_number": primary_special,
        "special_numbers": all_special or None,
        "title": title,
        "case_type": case_type,
        "case_type_tags": case_tags or None,
        # Maintain explicit subtypes if provided by crawler
        "case_subtypes": list(case_tags) if case_tags else (list(crawler_subtypes) if isinstance(crawler_subtypes, list) else None),
        "case_subtype": (list(crawler_subtypes)[0] if isinstance(crawler_subtypes, list) and crawler_subtypes else rec.get("case_subtype")),
        "promulgation_year": y if isinstance(y, int) else None,
        "promulgation_date": rec.get("promulgation_date"),
        "source_url": rec.get("source_url"),
        # Optional commonly present fields (kept if available)
        "ponente": rec.get("ponente"),
        "division": rec.get("division"),
        "metadata_version": "minimal-3",
        "extraction_timestamp": int(time.time()),
    }


# -----------------------------
# Model (load once) - Use centralized cached model
# -----------------------------
print(f"📥 Loading model: {EMBED_MODEL}")
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from chatbot.model_cache import get_cached_embedding_model

model = get_cached_embedding_model()
if torch.cuda.is_available() and not str(model.device).startswith('cuda'):
    model = model.to("cuda")

# -----------------------------
# Point maker
# -----------------------------
def make_points(
    doc_id: str,
    text: str,
    meta: Dict[str, Any],
    extra_sections: Optional[Dict[str, str]] = None,
    ) -> List[PointStruct]:
    texts: List[str] = []
    payloads: List[Dict[str, Any]] = []
    ids: List[str] = []

    # CAPTION: lightweight metadata line to anchor title/GR/special/date/division for retrieval
    caption_bits = []
    if meta.get("title"):
        caption_bits.append(str(meta["title"]))
    if meta.get("gr_number"):
        caption_bits.append(f"G.R.: {meta['gr_number']}")
    if meta.get("special_number"):
        caption_bits.append(f"Special: {meta['special_number']}")
    if meta.get("promulgation_date"):
        caption_bits.append(str(meta["promulgation_date"]))
    if meta.get("division"):
        caption_bits.append(str(meta["division"]))
    if meta.get("case_type"):
        caption_bits.append(f"type:{meta['case_type']}")
    if meta.get("case_subtype"):
        caption_bits.append(f"subtype:{meta['case_subtype']}")
    if caption_bits:
        caption_text = normalize_text(" | ".join(caption_bits))
        texts.append(caption_text)
        payloads.append({**meta, "section": "caption"})
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#caption")))

    # RULING (from the full, normalized 'text')
    rs, re_ = find_ruling(text)
    if rs != -1:
        ruling_text = text[rs:re_].strip()
        if ruling_text:
            texts.append(ruling_text)
            payloads.append({**meta, "section": "ruling"})
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#ruling")))

    # HEADER (~first 2000 chars of normalized text) - Optimized size
    header_snippet = text[:2000].strip()
    if header_snippet:
        texts.append(header_snippet)
        payloads.append({**meta, "section": "header"})
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#header")))

    # BODY: straightforward overlapping chunks sized for the embed model
    body_text = (text[:max(0, rs - 1)] + " " + text[re_:]).strip() if rs != -1 else text
    if body_text:
        n = len(body_text)
        start = 0
        step = max(1, CHUNK_CHARS - OVERLAP_CHARS)
        chunk_index = 0
        while start < n:
            end = min(n, start + CHUNK_CHARS)
            chunk = body_text[start:end].strip()
            if chunk:
                chunk_index += 1
                texts.append(chunk)
                payloads.append({**meta, "section": "body", "chunk_index": chunk_index})
                ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#body-{chunk_index:03d}")))
            if end >= n:
                break
            start = end - OVERLAP_CHARS
    
    # Store full text reference for JSONL retrieval
    if rs != -1:
        body_text = (text[:max(0, rs - 1)] + " " + text[re_:]).strip()
        if body_text:
            texts.append(f"FULL_TEXT_REFERENCE: {doc_id}")  # Placeholder for full text
            payloads.append({**meta, "section": "full_text_ref", "has_full_text": True})
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#full_text_ref")))

    # EXTRA SECTIONS: facts, issues, etc. (normalized + chunked)
    # Only store key sections, not all content
    if extra_sections:
        key_sections = ["facts", "issues", "held", "disposition"]  # Only important sections
        for name, content in extra_sections.items():
            if not content or name.lower() not in key_sections:
                continue
            if name.lower() == "ruling":
                # avoid duplicating ruling
                continue
            norm = normalize_text(content)
            if not norm:
                continue
            # Store key sections as single chunks (not split)
            texts.append(norm)
            payloads.append({**meta, "section": name.lower()})
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#{name.lower()}")))

    # Encode in one batch
    vectors = model.encode(
        texts, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True
    )

    return [
        PointStruct(id=pid, vector=vec.tolist(), payload=pl)
        for pid, vec, pl in zip(ids, vectors, payloads)
    ]

def upsert_points(points: List[PointStruct]):
    # upsert in manageable batches
    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i : i + UPSERT_BATCH]
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)

# Multiple collection routing removed for simplicity

# Hierarchical/multi-collection upsert removed. Using single collection via upsert_points.

# Advanced legal chunking path removed

# (TXT process_dir removed)

def process_jsonl():
    """
    JSONL pipeline:
    - Streams records from DATA_FILE (.jsonl or .jsonl.gz)
    - Normalizes whitespace (newlines -> spaces)
    - Section-aware embedding with extra sections aligned to digest rules:
      facts, issues, arguments, discussion, citations, keywords
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"DATA_FILE not found: {DATA_FILE}")

    pending_points: List[PointStruct] = []
    added = 0
    year_stats = defaultdict(int)  # Track cases per year

    for rec in iter_cases(DATA_FILE):
        try:
            base_id = rec.get("id") or rec.get("source_url") or str(uuid.uuid4())
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, base_id))

            full_text = text_from_record(rec)
            if not full_text:
                continue

            # Year range filter (inclusive) based on promulgation_year/date if present
            y = rec.get("promulgation_year")
            if not isinstance(y, int):
                d = rec.get("promulgation_date")
                if isinstance(d, str) and len(d) >= 4 and d[:4].isdigit():
                    try:
                        y = int(d[:4])
                    except Exception:
                        y = None
            if isinstance(y, int) and (y < YEAR_START or y > YEAR_END):
                continue

            # Track year statistics
            if isinstance(y, int):
                year_stats[y] += 1

            meta = record_meta(rec)

            # Gather extra sections if present
            s = rec.get("sections") or {}
            extra = {}
            # Accept common keys and aliases → map to canonical section names used in retrieval
            alias_to_canonical = {
                # facts
                "facts": "facts",
                "statement_of_facts": "facts",
                "facts_and_issues": "facts",
                # issues
                "issues": "issues",
                "issue": "issues",
                # arguments / reasoning
                "arguments": "arguments",
                "reasoning": "arguments",
                "ratio": "arguments",
                # discussion / separate opinions
                "discussion": "discussion",
                "opinions": "discussion",
                "separate_opinions": "discussion",
                "concurring_dissenting": "discussion",
                # citations / authorities
                "citations": "citations",
                "authorities": "citations",
                # keywords / doctrines / legal terms
                "keywords": "keywords",
                "legal_terms": "keywords",
                "doctrines": "keywords",
                # administrative headings
                "resolution": "ruling",
            }

            for key, value in s.items():
                if not value or not isinstance(key, str):
                    continue
                canonical = alias_to_canonical.get(key.lower())
                if not canonical:
                    continue
                # Prefer first-found content per canonical name
                extra.setdefault(canonical, value)

            # Create points using straightforward chunking
            pts = make_points(doc_id, full_text, meta, extra_sections=extra)
            pending_points.extend(pts)
            added += 1

            if len(pending_points) >= UPSERT_BATCH:
                upsert_points(pending_points)
                pending_points.clear()

        except Exception as e:
            src = rec.get("source_url") if isinstance(rec, dict) else "unknown"
            print(f"⚠️  Skipping record ({src}) — {e}")

    if pending_points:
        upsert_points(pending_points)

    # Print year statistics
    print(f"\n📊 Year Statistics:")
    for year in sorted(year_stats.keys()):
        print(f"   {year}: {year_stats[year]} cases embedded")
    
    print(f"\n🚀 Total uploaded: {added} JSONL records")

if __name__ == "__main__":
        print(f"📦 Mode: JSONL — {DATA_FILE}")
        process_jsonl()
