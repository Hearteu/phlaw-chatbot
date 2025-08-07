import json
import os
import re
import uuid

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# --- Load configs & API key from .env (optional, for security) ---
load_dotenv()

QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "jurisprudence")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Stern5497/sbert-legal-xlm-roberta-base")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 768))

DATA_DIR = "backend/jurisprudence2"
CACHE_PATH = "backend/backend/chatbot/embedded_cache2.json"

# --- Initialize Qdrant Cloud Client ---
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
# print("🟢 Connected to Qdrant Cloud!")
# print(client.get_collections())

# Create collection if needed
if not client.collection_exists(QDRANT_COLLECTION):
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
print(f"✅ Qdrant collection: {QDRANT_COLLECTION}")

# Load embedding cache
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        embedded_cache = json.load(f)
else:
    embedded_cache = []

# --- Load model (with CUDA support if available) ---
print(f"📥 Loading model: {EMBED_MODEL}")
model = SentenceTransformer(EMBED_MODEL)
if torch.cuda.is_available():
    model = model.to('cuda')

# --- Main Embedding and Upload Loop ---
for year in sorted(os.listdir(DATA_DIR)):
    year_path = os.path.join(DATA_DIR, year)
    if not os.path.isdir(year_path) or not year.isdigit():
        continue

    print(f"\n📂 Processing year: {year}")
    print(f"🔍 Found {len(os.listdir(year_path))} files")
    print("Processing files...")
    points = []
    for filename in os.listdir(year_path):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(year_path, filename)
        if filepath in embedded_cache:
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract G.R. number(s)
        gr_nos = re.findall(r"G\.R\. No\. \d{5,}", content)
        gr_nos = [gr.strip() for gr in gr_nos]

        vector = model.encode(content).tolist()
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, filepath)),
            vector=vector,
            payload={
                "filename": filename,
                "year": year,
                "gr_no": gr_nos  # always a list
            }
        )
        points.append(point)
        embedded_cache.append(filepath)

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"🚀 Uploaded {len(points)} docs from {year} to Qdrant")
    else:
        print("✔️ No new docs found this year")

    # Save embedding cache
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(embedded_cache, f, indent=2)

print("\n🎉 Embedding process complete!")
