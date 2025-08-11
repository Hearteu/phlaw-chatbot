# retriever.py — JSONL-aware retriever with exact GR match + sectioned snippets
import gzip
import json
import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

# Match ruling blocks when we must extract from raw text
RULING_REGEX = re.compile(r"(WHEREFORE.*?SO ORDERED\.?|ACCORDINGLY.*?SO ORDERED\.?)",
                          re.IGNORECASE | re.DOTALL)

# Detect a GR number in the user query (e.g., "G.R. No. 211089" or "G.R. Nos. 12345-67")
GR_IN_QUERY = re.compile(r"G\.?\s*R\.?\s*No(?:s)?\.?\s*[0-9\-]+", re.I)

# --- Env / config (align with embed.py) ---
DATA_FORMAT = os.getenv("DATA_FORMAT", "jsonl").lower()          # "txt" | "jsonl"
DATA_FILE   = os.getenv("DATA_FILE", "data/cases.jsonl.gz")     # for jsonl
DATA_DIR    = os.path.abspath(os.getenv(
    "DATA_DIR",
    os.path.join(settings.BASE_DIR, "jurisprudence2")           # legacy txt layout
))
CHUNK_CHARS   = int(os.getenv("CHUNK_CHARS", 1200))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", 150))

# ----------------- helpers -----------------
def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

def _extract_ruling(text: str) -> Tuple[int, int]:
    m = RULING_REGEX.search(text or "")
    return (m.start(), m.end()) if m else (-1, -1)

def _normalize_gr(raw: str) -> str:
    """Normalize various GR notations to 'G.R. No. XXXXX' (keep dashes if any)."""
    s = re.sub(r"\s+", " ", raw.strip())
    # Replace "Nos." -> "No."
    s = re.sub(r"\bNos?\.\b", "No.", s, flags=re.I)
    # Normalize 'GR' punctuation/spaces
    s = re.sub(r"^G\.?\s*R\.?\s*No\.\s*", "G.R. No. ", s, flags=re.I)
    return s

def _chunkify(s: str, size: int, overlap: int) -> List[str]:
    out: List[str] = []
    if not s:
        return out
    n = len(s)
    start = 0
    step = max(1, size - overlap)
    while start < n:
        end = min(n, start + size)
        out.append(s[start:end])
        if end >= n:
            break
        start += step
    return out

# -------- JSONL access with tiny LRU cache --------
class _JSONLCache:
    """
    Lazily loads records from .jsonl or .jsonl.gz and caches only what we touch.
    We key by 'source_url' (unique in your crawler output) and also expose by 'id'.
    Each cached entry stores only what's needed to assemble snippets.
    """
    def __init__(self, path: str, max_items: int = 512):
        self.path = path
        self.max_items = max_items
        self._by_url: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._by_id: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def _touch(self, d: "OrderedDict[str, Dict[str, Any]]", key: str, value: Dict[str, Any]):
        d[key] = value
        d.move_to_end(key)
        while len(d) > self.max_items:
            d.popitem(last=False)

    def _iter_lines(self):
        opener = gzip.open if self.path.endswith(".gz") else open
        with opener(self.path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    def _store_minimal(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only minimal fields needed to reconstruct snippets."""
        secs = rec.get("sections") or {}
        return {
            "id": rec.get("id"),
            "source_url": rec.get("source_url"),
            "gr_number": rec.get("gr_number"),
            "title": rec.get("title"),
            "promulgation_date": rec.get("promulgation_date"),
            "promulgation_year": rec.get("promulgation_year"),
            "sections": {
                # keep as-is; ruling/header are small, body can be long but needed to make the exact chunk
                "ruling": secs.get("ruling") or "",
                "header": secs.get("header") or "",
                "body":   secs.get("body") or "",
            },
            "clean_text": rec.get("clean_text") or "",
        }

    def get_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return None
        if url in self._by_url:
            rec = self._by_url[url]
            self._by_url.move_to_end(url)
            return rec
        # scan file once to find url
        for rec in self._iter_lines():
            if rec.get("source_url") == url:
                slim = self._store_minimal(rec)
                self._touch(self._by_url, url, slim)
                if slim.get("id"):
                    self._touch(self._by_id, slim["id"], slim)
                return slim
        return None

    def get_by_id(self, rid: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return None
        if rid in self._by_id:
            rec = self._by_id[rid]
            self._by_id.move_to_end(rid)
            return rec
        for rec in self._iter_lines():
            if rec.get("id") == rid:
                slim = self._store_minimal(rec)
                self._touch(self._by_id, rid, slim)
                if slim.get("source_url"):
                    self._touch(self._by_url, slim["source_url"], slim)
                return slim
        return None

# --------------- retriever ----------------
class LegalRetriever:
    def __init__(self):
        # SentenceTransformers will use CUDA if available
        self.model = SentenceTransformer("Stern5497/sbert-legal-xlm-roberta-base")

        # Qdrant client / collection must match embed.py
        self.qdrant = QdrantClient(host="localhost", port=6333, timeout=120.0)
        self.collection = "jurisprudence"

        # Legacy TXT dir (still supported)
        default_data_dir = os.path.join(settings.BASE_DIR, "jurisprudence2")
        self.data_dir = DATA_DIR or os.path.abspath(default_data_dir)

        # JSONL cache (only used when payloads are sectioned=True)
        self.jsonl_cache = _JSONLCache(DATA_FILE, max_items=512)

    # ---------- helpers ----------

    def _doc_from_legacy_payload(self, payload: Dict[str, Any], score: float) -> Dict[str, Any]:
        """Load from legacy TXT files using filename/year."""
        filename = payload.get("filename")
        year = payload.get("year")
        path = os.path.join(self.data_dir, str(int(year)), filename) if filename and year is not None else None

        if not path or not os.path.exists(path):
            return {
                "text": f"[Missing file: {path}]",
                "score": score,
                "filename": filename,
                "gr_number": payload.get("gr_number"),
                "title": payload.get("title"),
                "source_url": payload.get("source_url"),
            }

        text = _read(path)
        rs, re_ = _extract_ruling(text)
        excerpt = text[rs:re_].strip() if rs != -1 else text[:1200].strip()
        return {
            "text": excerpt,
            "score": score,
            "filename": filename,
            "gr_number": payload.get("gr_number"),
            "title": payload.get("title"),
            "source_url": payload.get("source_url"),
        }

    def _doc_from_jsonl_payload(self, payload: Dict[str, Any], score: float) -> Dict[str, Any]:
        src = payload.get("source_url") or ""
        rid = payload.get("id")
        rec = self.jsonl_cache.get_by_url(src) or (self.jsonl_cache.get_by_id(rid) if rid else None)

        if not rec:
            # Fallback to payload fields (never None)
            title = payload.get("title") or payload.get("gr_number") or "Untitled case"
            url   = payload.get("source_url") or "N/A"
            return {
                "text": "[Record not found in JSONL corpus]",
                "score": score,
                "gr_number": payload.get("gr_number"),
                "title": title,
                "source_url": url,
            }

        # Build snippet from sections…
        section = payload.get("section") or ""
        chunk_index = payload.get("chunk_index")
        secs = rec.get("sections") or {}
        text = ""
        if section == "ruling" and secs.get("ruling"):
            text = secs["ruling"].strip()
        elif section == "header" and secs.get("header"):
            text = secs["header"].strip()
        elif section == "body":
            body = secs.get("body") or rec.get("clean_text") or ""
            if body and isinstance(chunk_index, int) and chunk_index >= 1:
                chunks = _chunkify(body, CHUNK_CHARS, OVERLAP_CHARS)
                idx = min(chunk_index - 1, max(0, len(chunks) - 1))
                text = chunks[idx].strip() if chunks else (body[:1200] or "")
            else:
                text = (body[:1200] or "").strip()
        else:
            text = (secs.get("ruling") or secs.get("header") or rec.get("clean_text") or "")[:1200].strip()

        title = rec.get("title") or payload.get("title") or rec.get("gr_number") or "Untitled case"
        url   = rec.get("source_url") or payload.get("source_url") or "N/A"
        return {
            "text": text or "[Empty section]",
            "score": score,
            "gr_number": rec.get("gr_number") or payload.get("gr_number"),
            "title": title,
            "source_url": url,
        }


    def _load_docs_from_hits(self, hits) -> List[Dict[str, Any]]:
        """Turn Qdrant hits into the doc dicts the chat engine expects."""
        docs: List[Dict[str, Any]] = []
        for h in hits:
            payload = h.payload or {}
            score = getattr(h, "score", 0.0)

            # JSONL/sectioned path (new pipeline)
            is_jsonl = bool(payload.get("sectioned") or payload.get("source_url") or payload.get("id"))
            if is_jsonl:
                docs.append(self._doc_from_jsonl_payload(payload, score))
            else:
                docs.append(self._doc_from_legacy_payload(payload, score))

            # Legacy TXT path
            docs.append(self._doc_from_legacy_payload(payload, score))

        return docs

    def _scroll_by_gr_number(self, gr_no_normalized: str) -> List[Dict[str, Any]]:
        """
        Exact payload match on 'gr_number' (your embedder stores this for JSONL docs).
        Return the case's ruling as the primary snippet.
        """
        results = []
        next_page = None
        seen = set()

        while True:
            points, next_page = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=[
                    FieldCondition(
                        key="gr_number",
                        match=MatchValue(value=gr_no_normalized)
                    )
                ]),
                with_payload=True,
                limit=512,
                offset=next_page
            )
            for p in points:
                pl = p.payload or {}
                key = (pl.get("source_url"), pl.get("gr_number"))
                if key in seen:
                    continue
                seen.add(key)

                # Prefer JSONL path to return the ruling
                doc = self._doc_from_jsonl_payload({**pl, "section": "ruling"}, score=1.0)
                results.append(doc)

            if not next_page:
                break

        return results

    # ---------- public API ----------
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts: {text, score, gr_number, title, source_url, ...}.
        Strategy:
          1) If query contains a G.R. number → exact match on payload.gr_number, return ruling.
          2) Else vector search → return section-aware snippets via JSONL cache or legacy TXT.
        """
        # 1) Exact G.R. No. path
        m = GR_IN_QUERY.search(query or "")
        if m:
            gr = _normalize_gr(m.group(0))
            docs = self._scroll_by_gr_number(gr)
            if docs:
                return docs

        # 2) Semantic search path
        qv = self.model.encode(query).tolist()
        hits = self.qdrant.search(
            collection_name=self.collection,
            query_vector=qv,
            limit=k,
            with_payload=True
        )
        docs = self._load_docs_from_hits(hits)
        print("DEBUG first doc:", {k: docs[0].get(k) for k in ("title","source_url","gr_number")})

        return self._load_docs_from_hits(hits)
