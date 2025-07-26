import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class LegalRetriever:
    def __init__(self):
        self.model = SentenceTransformer("Stern5497/sbert-legal-xlm-roberta-base")
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.collection = "jurisprudence"

    def retrieve(self, query, k=3):
        query_vector = self.model.encode(query).tolist()
        hits = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=k
        )

        docs = []
        for hit in hits:
            path = hit.payload["path"]
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    docs.append({
                        "text": f.read(),
                        "score": hit.score,
                        "filename": hit.payload.get("filename", "Unknown")
                    })
            else:
                docs.append({
                    "text": f"[Missing file: {path}]",
                    "score": hit.score,
                    "filename": hit.payload.get("filename", "Unknown")
                })
        return docs
