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
DATA_DIR    = os.getenv("DATA_DIR", "backend/jurisprudence2")       # for txt mode
DATA_FILE   = os.getenv("DATA_FILE", "backend/data/cases.jsonl.gz")         # for jsonl mode

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

    return {
        "gr_number": rec.get("gr_number"),
        "title": rec.get("title"),
        "year": y,
        "source_url": rec.get("source_url"),
        "sectioned": True,
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
    - Section-aware embedding with extra sections: facts, issues
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

            meta = record_meta(rec)

            # Gather extra sections if present
            s = rec.get("sections") or {}
            extra = {}
            # accept common keys and simple aliases
            for key in ("facts", "issues", "issue", "statement_of_facts", "facts_and_issues"):
                if s.get(key):
                    # Map aliases to canonical names
                    canonical = "issues" if key in ("issues", "issue") else "facts"
                    # Prefer first-found content per canonical name
                    extra.setdefault(canonical, s[key])

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
