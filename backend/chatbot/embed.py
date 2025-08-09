# embed.py — batched, section-aware embedding + Qdrant upsert

import json
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Tuple

import torch
from cleaner import clean_and_split  # <-- add this import
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

DATA_DIR = os.getenv("DATA_DIR", "backend/jurisprudence2")
CACHE_PATH = os.getenv("EMBED_CACHE_PATH", "backend/backend/chatbot/embedded_cache2.json")

# Chunking
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", 1200))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", 150))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 16))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", 512))

# -----------------------------
# Qdrant init
# -----------------------------
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, grpc_port=6334, prefer_grpc=True, timeout=120.0  )
if not client.collection_exists(QDRANT_COLLECTION):
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
print(f"✅ Qdrant collection ready: {QDRANT_COLLECTION}")

# -----------------------------
# Cache (list of processed filepaths)
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
# -----------------------------
# Helpers
# -----------------------------
GR_NO_REGEX = re.compile(r"G\.R\.\s*No(?:s)?\.?\s*\d{5,}", re.IGNORECASE)

def extract_gr_nos(text: str) -> list[str]:
    raw = GR_NO_REGEX.findall(text)
    norm = []
    for m in raw:
        # unify spacing, collapse multiple spaces
        s = re.sub(r"\s+", " ", m.strip())
        # force singular "G.R. No."
        s = s.replace("G.R. Nos.", "G.R. No.").replace("G.R. No .", "G.R. No.")
        norm.append(s)
    return sorted(set(norm))

def find_ruling(text: str) -> Tuple[int, int]:
    """Return (start, end) of the ruling section, or (-1, -1) if not found."""
    m = RULING_REGEX.search(text)
    return (m.start(), m.end()) if m else (-1, -1)

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
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

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

PRIMARY_GR_RE = re.compile(
    r"\[\s*G\.R\.\s*No(?:s)?\.?\s*([0-9][0-9\- ,]*)\s*[,\]]",
    re.I
)

def extract_primary_gr_no(text: str) -> str | None:
        header = text[:2000]  # safer window around the centered caption
        m = PRIMARY_GR_RE.search(header)
        return f"G.R. No. {m.group(1)}" if m else None

def make_points(doc_id: str, raw_text: str, meta: Dict[str, Any]) -> List[PointStruct]:
    parts = clean_and_split(raw_text)
    texts, payloads, ids = [], [], []

    # RULING (highest recall for "Give me the ruling...")
    if parts["ruling"]:
        texts.append(parts["ruling"])
        payloads.append({**meta, "section": "ruling"})
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#ruling")))

    # HEADER (short)
    if parts["header"]:
        header_snippet = parts["header"][:1200].strip()
        if header_snippet:
            texts.append(header_snippet)
            payloads.append({**meta, "section": "header"})
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#header")))

    # BODY chunks (already cleaned; excludes ruling duplication)
    body_text = parts["body"]
    if parts["disposition"]:
        # Prefer to keep disposition independent; exclude it from body to avoid dup
        body_text = body_text.replace(parts["disposition"], "")

    # simple chunker (existing)
    for idx, chunk in enumerate(chunkify(body_text, CHUNK_CHARS, OVERLAP_CHARS), 1):
        c = chunk.strip()
        if not c:
            continue
        texts.append(c)
        payloads.append({**meta, "section": "body", "chunk_index": idx})
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#body-{idx:03d}")))

    # DISPOSITION as its own section (useful for retrieval)
    if parts["disposition"]:
        texts.append(parts["disposition"])
        payloads.append({**meta, "section": "disposition"})
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#disposition")))

    # METADATA can be short but searchable (optional)
    if parts["metadata"]:
        texts.append(parts["metadata"])
        payloads.append({**meta, "section": "metadata"})
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#metadata")))

    # Encode in one batch (unchanged)
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
                # Per-file metadata
                meta = {"filename": filename, "year": int(year)}
                primary = extract_primary_gr_no(text)
                if primary:
                    meta["primary_gr_no"] = primary
                # (optional) also keep all cited GRs you already collect:
                # meta["cited_gr_nos"] = list_of_all_mentions

                doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, filepath))
                pts = make_points(doc_id, text, meta)
                pending_points.extend(pts)
                embedded_cache.add(filepath)
                added_this_year += 1
            except Exception as e:
                print(f"⚠️  Skipping (read/encode error): {filepath} — {e}")

            # flush periodically to keep memory steady
            if len(pending_points) >= UPSERT_BATCH:
                upsert_points(pending_points)
                pending_points.clear()
                save_cache()

        # flush any remainder
        if pending_points:
            upsert_points(pending_points)
            pending_points.clear()
            save_cache()

        print(f"🚀 Uploaded {added_this_year} new docs from {year}")
        total_new += added_this_year

    print(f"\n🎉 Done. New docs embedded: {total_new}")
    from qdrant_client.http.models import PayloadSchemaType

    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="primary_gr_no",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="year",
        field_schema=PayloadSchemaType.INTEGER,
    )

if __name__ == "__main__":
    process_dir()
