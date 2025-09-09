# embed.py — Optimized batched, section-aware embedding + Qdrant upsert for 21k cases
# Optimizations:
# - Increased chunk size from 1200 to 2000 chars for better context
# - Optimized overlap from 150 to 200 chars (10% ratio)
# - Increased batch sizes for better throughput (32 embed, 1024 upsert)
# - Legal document boundary-aware chunking (respects WHEREFORE, ISSUES, FACTS, etc.)
# - Improved header chunk size from 1200 to 2000 chars
import gzip
import json
import os
import re
import unicodedata
import uuid
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

# Source data
DATA_FORMAT = os.getenv("DATA_FORMAT", "jsonl").lower()   # "txt" | "jsonl"
DATA_DIR    = os.getenv("DATA_DIR", "backend/jurisprudence")       # for txt mode
DATA_FILE   = os.getenv("DATA_FILE", "backend/data/cases.jsonl.gz")         # for jsonl mode
# Optional year range filter when processing JSONL
YEAR_START  = int(os.getenv("YEAR_START", 2005))
YEAR_END    = int(os.getenv("YEAR_END", 2006))

# Cache (for txt mode; list of processed filepaths)
CACHE_PATH = os.getenv("EMBED_CACHE_PATH", "backend/backend/chatbot/embedded_cache.json")

# Chunking/throughput - Optimized for 21k cases
CHUNK_CHARS   = int(os.getenv("CHUNK_CHARS", 2000))      # Increased from 1200 for better context
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", 200))     # Optimized overlap ratio (10%)
BATCH_SIZE    = int(os.getenv("EMBED_BATCH_SIZE", 32))   # Increased from 16 for better throughput
UPSERT_BATCH  = int(os.getenv("UPSERT_BATCH", 1024))     # Increased from 512 for efficiency

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

# -----------------------------
# Cache (txt mode: list of processed filepaths)
# -----------------------------
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        embedded_cache = set(json.load(f) or [])
else:
    embedded_cache = set()

def save_cache():
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(embedded_cache), f, indent=2)

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
    r"G\.R\.\s+No(?:s)?\.?\s*([0-9][0-9\-*,\s]+)",
    re.IGNORECASE,
)

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
    """Derive a clean case title from record fields.
    Preference order: explicit case_title -> parsed header -> provided title -> filename.
    Removes common library boilerplate and trims around G.R. markers.
    """
    # 1) Explicit field
    title = (rec.get("case_title") or rec.get("case_name") or "").strip()
    if title:
        return _clean_library_noise(title)

    def is_address_noise(line: str) -> bool:
        l = line.lower()
        return any(term in l for term in ADDRESS_NOISE_TERMS)

    # 2) Parse from header section
    sections = rec.get("sections") or {}
    header = _clean_library_noise(sections.get("header") or "")
    if header:
        # 2a) Prefer explicit case caption patterns with v./vs.
        for line in header.splitlines():
            line = line.strip()
            if not line:
                continue
            if is_address_noise(line):
                continue
            if CASE_CAPTION_STRONG_RE.match(line) or CASE_VS_RE.search(line):
                return line
        m = CASE_TITLE_LINE_RE.search(header)
        if m and m.group("title"):
            return m.group("title").strip()
        # fallback: first non-empty line without boilerplate
        for line in header.splitlines():
            line = line.strip()
            if not line:
                continue
            if "G.R." in line or line.lower().startswith(("promulgated", "decided")):
                break
            if len(line) > 3 and not is_address_noise(line):
                return line

    # 2b) As a last resort, scan the beginning of body for a case caption
    body = _clean_library_noise(sections.get("body") or rec.get("clean_text") or "")
    if body:
        start = body[:1200]
        for line in start.splitlines():
            line = line.strip()
            if is_address_noise(line):
                continue
            if CASE_CAPTION_STRONG_RE.match(line) or CASE_VS_RE.search(line):
                return line

    # 3) Provided title, cleaned
    raw_title = _clean_library_noise(rec.get("title") or "").strip()
    if raw_title and not raw_title.lower().startswith("supreme court e-library"):
        return raw_title

    # 4) Filename fallback
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

def chunkify(s: str, size: int, overlap: int) -> Iterable[str]:
    """Optimized chunker with legal document boundary awareness."""
    if not s:
        return
    n = len(s)
    start = 0
    step = max(1, size - overlap)
    
    while start < n:
        end = min(n, start + size)
        
        if end >= n:
            yield s[start:end]
            break
        
        # Try to break at legal document boundaries
        chunk = s[start:end]
        
        # Priority 1: Legal section boundaries (WHEREFORE, ISSUES, FACTS, etc.)
        legal_boundaries = ['WHEREFORE', 'ISSUES', 'FACTS', 'RULING', 'DECISION', 'ARGUMENTS']
        best_boundary = -1
        for boundary in legal_boundaries:
            boundary_pos = chunk.rfind(boundary)
            if boundary_pos > size * 0.6:  # If boundary is in last 40% of chunk
                best_boundary = max(best_boundary, boundary_pos)
        
        # Priority 2: Sentence boundary
        if best_boundary == -1:
            last_period = chunk.rfind('.')
            if last_period > size * 0.7:  # If period is in last 30% of chunk
                best_boundary = last_period + 1
        
        # Priority 3: Paragraph boundary
        if best_boundary == -1:
            last_newline = chunk.rfind('\n\n')
            if last_newline > size * 0.8:  # If double newline is in last 20% of chunk
                best_boundary = last_newline + 2
        
        # Priority 4: Single newline
        if best_boundary == -1:
            last_newline = chunk.rfind('\n')
            if last_newline > size * 0.8:  # If newline is in last 20% of chunk
                best_boundary = last_newline + 1
        
        # Apply the best boundary found
        if best_boundary > 0:
            end = start + best_boundary
        
        yield s[start:end]
        start = end - overlap

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

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
    Re-create a single text stream preserving your section-aware emphasis:
    ruling → header → body, with whitespace normalized.
    Enhanced to work with both current and legacy dataset formats.
    """
    s = rec.get("sections") or {}
    parts: List[str] = []
    
    # Priority 1: Ruling section (most important for legal decisions)
    ruling = s.get("ruling") or rec.get("ruling", "")
    if ruling:
        parts.append(normalize_text(ruling))
    
    # Priority 2: Header section (case title and basic info)
    header = s.get("header") or rec.get("header", "")
    if header:
        parts.append(normalize_text(header))
    
    # Priority 3: Body section (main content)
    body = s.get("body") or rec.get("body", "") or rec.get("clean_text", "")
    if body:
        parts.append(normalize_text(body))
    
    # Priority 4: Additional sections for comprehensive coverage
    additional_sections = ["facts", "issues", "arguments"]
    for section in additional_sections:
        content = s.get(section, "")
        if content and content.strip():
            parts.append(normalize_text(content))
    
    # Join with a single space (we already normalized internals)
    return " ".join(p for p in parts if p).strip()

def record_meta(rec):
    # Enhanced year extraction with multiple fallbacks
    y = None
    
    # 1) Try top-level year field first (for enhanced dataset)
    if 'year' in rec and rec['year']:
        y = int(rec['year']) if isinstance(rec['year'], (int, str)) and str(rec['year']).isdigit() else None
    
    # 2) Try promulgation_year
    if y is None:
        yh = rec.get("promulgation_year")
        if isinstance(yh, int) and yh > 0:
            y = yh
        elif isinstance(yh, str) and yh.isdigit():
            y = int(yh)
    
    # 3) Try promulgation_date
    if y is None:
        d = rec.get("promulgation_date")
        if isinstance(d, str) and len(d) >= 4 and d[:4].isdigit():
            y = int(d[:4])
    
    # 4) Try to extract from source_url or other fields
    if y is None:
        # Look for year patterns in various fields
        for field in ['source_url', 'title', 'case_title']:
            if field in rec and rec[field]:
                year_match = re.search(r'\b(19|20)\d{2}\b', str(rec[field]))
                if year_match:
                    y = int(year_match.group())
                    break
    
    # 5) Final fallback
    if y is None:
        y = 0

    # Enhanced metadata extraction
    ponente = rec.get("ponente") or rec.get("author") or rec.get("justice")
    
    # Division handling with better fallbacks
    division = rec.get("division") or rec.get("court_division")
    if not division and isinstance(rec.get("en_banc"), (bool, int)):
        division = "En Banc" if bool(rec.get("en_banc")) else None
    
    # G.R. number handling
    primary_gr, all_grs = derive_gr_numbers(rec)
    
    # Enhanced title extraction
    title = derive_case_title(rec)
    
    # Quality metrics
    quality_metrics = rec.get("quality_metrics", {})
    
    return {
        "gr_number": primary_gr,
        "gr_numbers": all_grs or None,
        "title": title,
        "year": y,  # Keep for backward compatibility
        "promulgation_year": y,  # Add the field that Qdrant expects
        "promulgation_date": rec.get("promulgation_date"),
        "source_url": rec.get("source_url"),
        "ponente": ponente,
        "division": division,
        "en_banc": bool(rec.get("en_banc")) if rec.get("en_banc") is not None else None,
        "sectioned": True,
        # Case classification
        "case_type": rec.get("case_type", "regular"),
        "case_subtype": rec.get("case_subtype"),
        "is_administrative": rec.get("is_administrative", False),
        "is_regular_case": rec.get("is_regular_case", True),
        # Quality metrics
        "quality_score": quality_metrics.get("sections_count", 0),
        "has_ruling": quality_metrics.get("has_ruling", False),
        "has_facts": quality_metrics.get("has_facts", False),
        "has_issues": quality_metrics.get("has_issues", False),
    }


# -----------------------------
# Model (load once) - Use centralized cached model
# -----------------------------
print(f"📥 Loading model: {EMBED_MODEL}")
# Import the centralized cached model function
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

    # CAPTION: lightweight metadata line to anchor title/GR/date/division for retrieval
    caption_bits = []
    if meta.get("title"):
        caption_bits.append(str(meta["title"]))
    if meta.get("gr_number"):
        caption_bits.append(f"G.R.: {meta['gr_number']}")
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

    # BODY (everything except ruling), normalized
    # OPTION: Skip body chunks to reduce storage and use JSONL for full text
    # Uncomment the next 4 lines to enable body chunking:
    # body_text = (text[:max(0, rs - 1)] + " " + text[re_:]).strip() if rs != -1 else text
    # for idx, chunk in enumerate(chunkify(body_text, CHUNK_CHARS, OVERLAP_CHARS), 1):
    #     texts.append(chunk)
    #     payloads.append({**meta, "section": "body", "chunk_index": idx})
    #     ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#body-{idx:03d}")))
    
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

# -----------------------------
# Main ingestion
# -----------------------------
def process_dir():
    """
    Legacy TXT folder layout:
      DATA_DIR/
        2005/
          jan2005_1.txt
          ...
        2006/
          ...
    Uses embedded_cache (filepaths) to avoid re-embedding the same files.
    """
    total_new = 0
    for year in sorted(os.listdir(DATA_DIR)):
        year_path = os.path.join(DATA_DIR, year)
        if not os.path.isdir(year_path) or not year.isdigit():
            continue

        print(f"\n📂 Year {year} — scanning {year_path}")
        files = [f for f in os.listdir(year_path) if f.endswith(".txt")]
        print(f"🔍 Found {len(files)} .txt files")

        pending_points: List[PointStruct] = []
        added_this_year = 0

        for filename in files:
            filepath = os.path.join(year_path, filename)
            if filepath in embedded_cache:
                continue

            try:
                text = read_text(filepath)
                meta = {"filename": filename, "year": int(year), "sectioned": False}
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, filepath))
                pts = make_points(doc_id, text, meta)
                pending_points.extend(pts)
                embedded_cache.add(filepath)
                added_this_year += 1
            except Exception as e:
                print(f"⚠️  Skipping (read/encode error): {filepath} — {e}")

            if len(pending_points) >= UPSERT_BATCH:
                upsert_points(pending_points)
                pending_points.clear()
                save_cache()

        if pending_points:
            upsert_points(pending_points)
            pending_points.clear()
            save_cache()

        print(f"🚀 Uploaded {added_this_year} new docs from {year}")
        total_new += added_this_year

    print(f"\n🎉 Done. New docs embedded: {total_new}")

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
                if not value:
                    continue
                canonical = alias_to_canonical.get(key.lower())
                if not canonical:
                    continue
                # Prefer first-found content per canonical name
                extra.setdefault(canonical, value)

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

    print(f"🚀 Uploaded {added} JSONL records")

if __name__ == "__main__":
    if DATA_FORMAT == "jsonl":
        print(f"📦 Mode: JSONL — {DATA_FILE}")
        process_jsonl()
    else:
        print(f"📦 Mode: TXT — {DATA_DIR}")
        process_dir()
