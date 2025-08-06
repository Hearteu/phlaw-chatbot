import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class LegalRetriever:
    def __init__(self):
        self.model = SentenceTransformer("Stern5497/sbert-legal-xlm-roberta-base")
        self.qdrant = QdrantClient(host="qdrant", port=6333)
        self.collection = "jurisprudence2"
        self.data_dir = "/app/backend/jurisprudence2"  # base folder for .txt files

    def retrieve(self, query, k=3):
        import re
        gr_no_match = re.search(r"G\.R\. No\. \d{5,}", query)
        if gr_no_match:
            gr_no = gr_no_match.group(0)
            try:
                hits = self.qdrant.search(
                    collection_name=self.collection,
                    query_vector=self.model.encode(query).tolist(),
                    limit=k,
                    query_filter={
                        "must": [
                            {"key": "gr_no", "match": {"value": gr_no}}
                        ]
                    }
                )
            except Exception as e:
                print("Qdrant search error:", e)
                return [{"text": f"[Qdrant error: {e}]", "score": 0, "filename": "error"}]
        else:
            # regular vector search
            try:
                hits = self.qdrant.search(
                    collection_name=self.collection,
                    query_vector=self.model.encode(query).tolist(),
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
