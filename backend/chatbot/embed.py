# embed.py — batched, section-aware embedding + Qdrant upsert (TXT or JSONL)
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
YEAR_START  = int(os.getenv("YEAR_START", 2012))
YEAR_END    = int(os.getenv("YEAR_END", 2012))

# Cache (for txt mode; list of processed filepaths)
CACHE_PATH = os.getenv("EMBED_CACHE_PATH", "backend/backend/chatbot/embedded_cache.json")

# Chunking/throughput
CHUNK_CHARS   = int(os.getenv("CHUNK_CHARS", 1200))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", 150))
BATCH_SIZE    = int(os.getenv("EMBED_BATCH_SIZE", 16))
UPSERT_BATCH  = int(os.getenv("UPSERT_BATCH", 512))

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
    """Simple char-based chunker with overlap."""
    if not s:
        return
    n = len(s)
    start = 0
    step = max(1, size - overlap)
    while start < n:
        end = min(n, start + size)
        yield s[start:end]
        if end >= n:
            break
        start += step

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
    """
    s = rec.get("sections") or {}
    parts: List[str] = []
    if s.get("ruling"):
        parts.append(normalize_text(s["ruling"]))
    if s.get("header"):
        parts.append(normalize_text(s["header"]))
    body = s.get("body") or rec.get("clean_text") or ""
    if body:
        parts.append(normalize_text(body))
    # Join with a single space (we already normalized internals)
    return " ".join(p for p in parts if p).strip()

def record_meta(rec):
    # 1) full date string present?
    y = None
    d = rec.get("promulgation_date")
    if isinstance(d, str) and len(d) >= 4 and d[:4].isdigit():
        y = int(d[:4])
    # 2) otherwise use year hint from crawler
    if y is None:
        yh = rec.get("promulgation_year")
        if isinstance(yh, int) and yh > 0:
            y = yh
    # 3) final fallback
    if y is None:
        y = 0

    # Optional richer header fields
    ponente = rec.get("ponente") or rec.get("author") or rec.get("justice")
    # Prefer explicit division; fall back to en_banc flag label
    division = rec.get("division") or rec.get("court_division")
    if not division and isinstance(rec.get("en_banc"), (bool, int)):
        division = "En Banc" if bool(rec.get("en_banc")) else None

    primary_gr, all_grs = derive_gr_numbers(rec)

    return {
        "gr_number": primary_gr,
        "gr_numbers": all_grs or None,
        "title": derive_case_title(rec),
        "year": y,
        "promulgation_date": rec.get("promulgation_date"),
        "source_url": rec.get("source_url"),
        "ponente": ponente,
        "division": division,
        "en_banc": bool(rec.get("en_banc")) if rec.get("en_banc") is not None else None,
        "sectioned": True,
        # Case classification
        "case_type": rec.get("case_type"),
        "case_subtype": rec.get("case_subtype"),
        "is_administrative": rec.get("is_administrative"),
        "is_regular_case": rec.get("is_regular_case"),
    }


# -----------------------------
# Model (load once)
# -----------------------------
print(f"📥 Loading model: {EMBED_MODEL}")
model = SentenceTransformer(EMBED_MODEL)
if torch.cuda.is_available():
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

    # HEADER (~first 1200 chars of normalized text)
    header_snippet = text[:1200].strip()
    if header_snippet:
        texts.append(header_snippet)
        payloads.append({**meta, "section": "header"})
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#header")))

    # BODY (everything except ruling), normalized
    body_text = (text[:max(0, rs - 1)] + " " + text[re_:]).strip() if rs != -1 else text
    for idx, chunk in enumerate(chunkify(body_text, CHUNK_CHARS, OVERLAP_CHARS), 1):
        texts.append(chunk)
        payloads.append({**meta, "section": "body", "chunk_index": idx})
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#body-{idx:03d}")))

    # EXTRA SECTIONS: facts, issues, etc. (normalized + chunked)
    if extra_sections:
        for name, content in extra_sections.items():
            if not content:
                continue
            if name.lower() == "ruling":
                # avoid duplicating ruling
                continue
            norm = normalize_text(content)
            if not norm:
                continue
            for idx, chunk in enumerate(chunkify(norm, CHUNK_CHARS, OVERLAP_CHARS), 1):
                texts.append(chunk)
                payloads.append({**meta, "section": name.lower(), "chunk_index": idx})
                ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#{name.lower()}-{idx:03d}")))

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
