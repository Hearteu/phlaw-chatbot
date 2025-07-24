from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class LegalRetriever:
    def __init__(self):
        self.model = SentenceTransformer("Stern5497/sbert-legal-xlm-roberta-base")
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.collection = "phlaw"

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
            with open(path, encoding="utf-8") as f:
                docs.append(f.read())
        return docs
