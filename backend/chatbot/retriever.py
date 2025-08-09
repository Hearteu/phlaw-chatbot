# retriever.py
import os
import re
from typing import Any, Dict, List, Tuple

from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

RULING_REGEX = re.compile(r"(WHEREFORE.*?SO ORDERED\.?|ACCORDINGLY.*?SO ORDERED\.?)",
                          re.IGNORECASE | re.DOTALL)
GR_IN_QUERY = re.compile(r"G\.R\.\s*No(?:s)?\.?\s*\d{5,}", re.I)

def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

def _extract_ruling(text: str) -> Tuple[int, int]:
    m = RULING_REGEX.search(text)
    return (m.start(), m.end()) if m else (-1, -1)


class LegalRetriever:
    def __init__(self):
        # Don't force CUDA; SentenceTransformers will use it if available.
        self.model = SentenceTransformer("Stern5497/sbert-legal-xlm-roberta-base")
        # Match your embed settings/collection
        self.qdrant = QdrantClient(host="localhost", port=6333, timeout=120.0)
        self.collection = "jurisprudence"
        default_data_dir = os.path.join(settings.BASE_DIR, "jurisprudence2")
        self.data_dir = os.path.abspath(os.getenv("DATA_DIR", default_data_dir))

    # ---------- helpers ----------

    def _load_docs_from_hits(self, hits) -> List[Dict[str, Any]]:
        """Turn Qdrant hits into the doc dicts the chat engine expects."""
        docs: List[Dict[str, Any]] = []
        for h in hits:
            payload = h.payload or {}
            filename = payload.get("filename")
            year = payload.get("year")
            if not filename or year is None:
                continue

            path = os.path.join(self.data_dir, str(int(year)), filename)
            if not os.path.exists(path):
                # still return something so the user sees the miss
                docs.append({
                    "text": f"[Missing file: {path}]",
                    "score": getattr(h, "score", 0.0),
                    "filename": filename
                })
                continue

            text = _read(path)
            rs, re_ = _extract_ruling(text)
            if rs != -1:
                excerpt = text[rs:re_].strip()
            else:
                # fallback to header-ish snippet
                excerpt = text[:1200].strip()

            docs.append({
                "text": excerpt,
                "score": getattr(h, "score", 0.0),
                "filename": filename
            })
        return docs

    def _scroll_by_primary_gr_no(self, gr_no_normalized: str) -> List[Dict[str, Any]]:
        """
        Pure filter retrieval against payload.primary_gr_no (exact match),
        then load full docs and prefer the ruling section.
        """
        results = []
        next_page = None
        seen = set()

        while True:
            points, next_page = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=[
                    FieldCondition(
                        key="primary_gr_no",
                        match=MatchValue(value=gr_no_normalized)
                    )
                ]),
                with_payload=True,
                limit=512,
                offset=next_page
            )
            for p in points:
                payload = p.payload or {}
                filename = payload.get("filename")
                year = payload.get("year")
                if filename and year is not None and (filename, int(year)) not in seen:
                    seen.add((filename, int(year)))
                    # Mimic a "hit" object minimally for _load_docs_from_hits
                    class _Hit:  # tiny shim
                        def __init__(self, payload):
                            self.payload = payload
                            self.score = 1.0
                    results.append(_Hit({"filename": filename, "year": int(year)}))

            if not next_page:
                break

        return self._load_docs_from_hits(results)

    # ---------- public API ----------

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts: {text, score, filename}.
        If query contains a G.R. No., try exact payload match first; otherwise semantic search.
        """
        # If the user asks for a specific G.R. number, try exact pull via payload.primary_gr_no
        m = GR_IN_QUERY.search(query)
        if m:
            gr = re.sub(r"\s+", " ", m.group(0).strip())
            # normalize "G.R. Nos." -> "G.R. No."
            gr = gr.replace("G.R. Nos.", "G.R. No.")
            docs = self._scroll_by_primary_gr_no(gr)
            if docs:
                return docs

        # Fallback: semantic search over all points
        qv = self.model.encode(query).tolist()
        hits = self.qdrant.search(
            collection_name=self.collection,
            query_vector=qv,
            limit=k,
            with_payload=True
        )
        return self._load_docs_from_hits(hits)
