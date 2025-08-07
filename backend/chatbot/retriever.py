import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class LegalRetriever:
    def __init__(self):
        # Load env variables
        load_dotenv()
        QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
        QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        COLLECTION = os.getenv("QDRANT_COLLECTION", "jurisprudence2")
        EMBED_MODEL = os.getenv("EMBED_MODEL", "Stern5497/sbert-legal-xlm-roberta-base")

        self.model = SentenceTransformer(EMBED_MODEL)
        self.qdrant = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)
        self.collection = COLLECTION
        self.data_dir = os.getenv("DATA_DIR", "backend/jurisprudence2")  # fallback/default

    def retrieve(self, query, k=3):
        import re
        gr_no_match = re.search(r"G\.R\. No\. \d{5,}", query)
        query_vector = self.model.encode(query).tolist()
        try:
            if gr_no_match:
                gr_no = gr_no_match.group(0)
                hits = self.qdrant.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    limit=k,
                    query_filter={
                        "must": [
                            {"key": "gr_no", "match": {"value": gr_no}}
                        ]
                    }
                )
            else:
                hits = self.qdrant.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    limit=k
                )
        except Exception as e:
            print("Qdrant search error:", e)
            return [{"text": f"[Qdrant error: {e}]", "score": 0, "filename": "error"}]

        docs = []
        for hit in hits:
            year = hit.payload.get("year")
            filename = hit.payload.get("filename", "Unknown")
            path = os.path.join(self.data_dir, str(year), filename)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    docs.append({
                        "text": f.read(),
                        "score": hit.score,
                        "filename": filename
                    })
            else:
                docs.append({
                    "text": f"[Missing file: {path}]",
                    "score": hit.score,
                    "filename": filename
                })
        return docs
