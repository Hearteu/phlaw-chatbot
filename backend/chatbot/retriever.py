# retriever.py — JSONL-aware retriever with intent & section-weighted search
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

# ---------- Regex / signals ----------
RULING_REGEX = re.compile(r"(WHEREFORE.*?SO ORDERED\.?|ACCORDINGLY.*?SO ORDERED\.?)",
                          re.IGNORECASE | re.DOTALL)
GR_IN_QUERY = re.compile(r"G\.?\s*R\.?\s*No(?:s)?\.?\s*[0-9\-–]+", re.I)  # accepts dash/en-dash

WS_RE = re.compile(r"\s+")
PUNCT_FIX_RE = re.compile(r"\s+([,.;:!?])")

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = WS_RE.sub(" ", s.replace("\r", "\n")).strip()
    s = PUNCT_FIX_RE.sub(r"\1", s)
    return s

def _extract_ruling(text: str) -> Tuple[int, int]:
    m = RULING_REGEX.search(text or "")
    return (m.start(), m.end()) if m else (-1, -1)

def _normalize_gr(raw: str) -> str:
    s = re.sub(r"\s+", " ", raw.strip())
    s = re.sub(r"\bNos?\.\b", "No.", s, flags=re.I)
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

# ---------- Env / config (align with embed.py) ----------
DATA_FORMAT   = os.getenv("DATA_FORMAT", "jsonl").lower()            # "txt" | "jsonl"
DATA_FILE     = os.getenv("DATA_FILE", "data/cases.jsonl.gz")        # for jsonl
CHUNK_CHARS   = int(os.getenv("CHUNK_CHARS", 1200))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", 150))
QDRANT_HOST   = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT   = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_GRPC   = int(os.getenv("QDRANT_GRPC_PORT", 6334))
QDRANT_COLL   = os.getenv("QDRANT_COLLECTION", "jurisprudence")

# Preferred sections and weights (used in re-ranking)
SECTION_PRIORITY = ["ruling", "issues", "facts", "header", "body"]
SECTION_WEIGHTS  = {"ruling": 1.00, "issues": 0.96, "facts": 0.94, "header": 0.72, "body": 0.66}

# ---------- JSONL access with tiny LRU cache ----------
class _JSONLCache:
    def __init__(self, path: str, max_items: int = 512):
        self.path = path
        self.max_items = max_items
        self._by_url: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._by_id:  "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

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
        secs = rec.get("sections") or {}
        return {
            "id": rec.get("id"),
            "source_url": rec.get("source_url"),
            "gr_number": rec.get("gr_number"),
            "title": rec.get("title"),
            "promulgation_date": rec.get("promulgation_date"),
            "promulgation_year": rec.get("promulgation_year"),
            "sections": {
                "ruling": secs.get("ruling") or "",
                "header": secs.get("header") or "",
                "body":   secs.get("body")   or "",
                # If your JSONL already includes facts/issues, we’ll read from here too:
                "facts":  secs.get("facts")  or secs.get("statement_of_facts") or "",
                "issues": secs.get("issues") or secs.get("issue") or secs.get("facts_and_issues") or "",
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

# ---------- Retriever ----------
class LegalRetriever:
    def __init__(self):
        # Will use CUDA if available
        self.model = SentenceTransformer(os.getenv("EMBED_MODEL", "Stern5497/sbert-legal-xlm-roberta-base")).to("cuda")

        # Faster Qdrant via gRPC; collection must match embed.py
        self.qdrant = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            # grpc_port=QDRANT_GRPC,
            # prefer_grpc=True,
            timeout=120.0,
        )
        self.collection = QDRANT_COLL

        # JSONL cache
        self.jsonl_cache = _JSONLCache(DATA_FILE, max_items=512)

        # Legacy TXT (kept for backward compatibility)
        default_data_dir = os.path.join(settings.BASE_DIR, "jurisprudence2")
        self.data_dir = os.path.abspath(os.getenv("DATA_DIR", default_data_dir))

    # ---------- Intent detection ----------
    def _detect_intent_sections(self, query: str) -> List[str]:
        q = (query or "").lower()
        wants_ruling = any(t in q for t in ["ruling", "decision", "disposition", "wherefore", "so ordered"])
        wants_facts  = "fact" in q or "statement of facts" in q
        wants_issues = "issue" in q or "issues" in q or "question presented" in q

        if wants_facts and not (wants_ruling or wants_issues):
            return ["facts", "header", "body", "issues", "ruling"]
        if wants_issues and not (wants_ruling or wants_facts):
            return ["issues", "header", "body", "facts", "ruling"]
        if wants_ruling and not (wants_facts or wants_issues):
            return ["ruling", "issues", "facts", "header", "body"]
        # default priority
        return SECTION_PRIORITY

    # ---------- Builders for doc outputs ----------
    def _doc_from_legacy_payload(self, payload: Dict[str, Any], score: float) -> Dict[str, Any]:
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

        with open(path, encoding="utf-8") as f:
            text = f.read()
        rs, re_ = _extract_ruling(text)
        excerpt = text[rs:re_].strip() if rs != -1 else text[:1200].strip()
        return {
            "text": _normalize_text(excerpt),
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
            title = payload.get("title") or payload.get("gr_number") or "Untitled case"
            url   = payload.get("source_url") or "N/A"
            return {
                "text": "[Record not found in JSONL corpus]",
                "score": score,
                "gr_number": payload.get("gr_number"),
                "title": title,
                "source_url": url,
            }

        section = (payload.get("section") or "").lower()
        chunk_index = payload.get("chunk_index")
        secs = rec.get("sections") or {}
        text = ""

        if section in ("ruling", "header", "facts", "issues"):
            txt = secs.get(section) or ""
            text = txt.strip()
        elif section == "body":
            body = secs.get("body") or rec.get("clean_text") or ""
            if body and isinstance(chunk_index, int) and chunk_index >= 1:
                chunks = _chunkify(body, CHUNK_CHARS, OVERLAP_CHARS)
                idx = min(chunk_index - 1, max(0, len(chunks) - 1))
                text = (chunks[idx] if chunks else body[:1200]).strip()
            else:
                text = (body[:1200] or "").strip()
        else:
            # sensible fallback: prefer ruling > header > facts > issues > clean_text
            text = (secs.get("ruling") or secs.get("header") or secs.get("facts") or
                    secs.get("issues") or rec.get("clean_text") or "")[:1200].strip()

        title = rec.get("title") or payload.get("title") or rec.get("gr_number") or "Untitled case"
        url   = rec.get("source_url") or payload.get("source_url") or "N/A"
        return {
            "text": _normalize_text(text) or "[Empty section]",
            "score": score,
            "gr_number": rec.get("gr_number") or payload.get("gr_number"),
            "title": title,
            "source_url": url,
            "section": section or payload.get("section"),
        }

    def _hit_to_doc(self, h) -> Dict[str, Any]:
        payload = h.payload or {}
        score = getattr(h, "score", 0.0)
        is_jsonl = bool(payload.get("sectioned") or payload.get("source_url") or payload.get("id"))
        if is_jsonl:
            return self._doc_from_jsonl_payload(payload, score)
        return self._doc_from_legacy_payload(payload, score)

    # ---------- Exact G.R. No. path ----------
    def _scroll_by_gr_number(self, gr_no_normalized: str, target_section: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Exact payload match on 'gr_number'. If target_section is set ("facts"/"issues"/"ruling"),
        return that section when available; fallback to ruling, then header.
        """
        results = []
        next_page = None
        seen_urls = set()

        while True:
            points, next_page = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="gr_number", match=MatchValue(value=gr_no_normalized))
                ]),
                with_payload=True,
                limit=512,
                offset=next_page
            )
            for p in points:
                pl = p.payload or {}
                url = pl.get("source_url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                if target_section:
                    # Force-build doc for the requested section from JSONL cache
                    doc = self._doc_from_jsonl_payload({**pl, "section": target_section}, score=1.0)
                    if doc.get("text") and "[Record not found" not in doc["text"] and doc["text"] != "[Empty section]":
                        results.append(doc)
                        continue  # got the requested section

                # Fallback to ruling, then header
                doc = self._doc_from_jsonl_payload({**pl, "section": "ruling"}, score=1.0)
                if doc.get("text") and doc["text"] != "[Empty section]":
                    results.append(doc)
                else:
                    results.append(self._doc_from_jsonl_payload({**pl, "section": "header"}, score=1.0))

            if not next_page:
                break

        return results

    # ---------- Section-filtered Qdrant search ----------
    def _search_section(self, query_vector, section: str, limit: int) -> List[Any]:
        try:
            return self.qdrant.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                query_filter=Filter(must=[FieldCondition(key="section", match=MatchValue(value=section))]),
            )
        except Exception:
            return []

    def _merge_rerank(self, hits_by_section: Dict[str, List[Any]], query: str, k: int) -> List[Dict[str, Any]]:
        """
        Combine hits from multiple section-filtered searches, re-rank with simple weights
        and intent boosts, then return top-k docs (deduped by (url, section, chunk_index)).
        """
        desired = self._detect_intent_sections(query)
        intent_boost = {s: 1.0 for s in SECTION_WEIGHTS}
        # Small boost to user-desired lead section
        if desired and desired[0] in intent_boost:
            intent_boost[desired[0]] = 1.05

        pool = []
        for sec, hits in hits_by_section.items():
            w = SECTION_WEIGHTS.get(sec, 0.6) * intent_boost.get(sec, 1.0)
            for h in hits:
                pl = h.payload or {}
                chunk_idx = pl.get("chunk_index")
                url = pl.get("source_url")
                score = getattr(h, "score", 0.0)
                pool.append((sec, url, chunk_idx, score * w, h))

        # Deduplicate by (url, sec, chunk)
        seen = set()
        uniq = []
        for sec, url, chunk_idx, s, h in sorted(pool, key=lambda x: x[3], reverse=True):
            key = (url, sec, int(chunk_idx) if chunk_idx is not None else -1)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((s, h))
            if len(uniq) >= max(k * 3, 12):  # keep a bit more for doc building
                break

        docs = [self._hit_to_doc(h) for _, h in uniq]
        # Final trim to k, keeping order
        return docs[:k]

    # ---------- Public API ----------
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts: {text, score, gr_number, title, source_url, section?}.
        Strategy:
          1) If query has a G.R. No., do an exact payload match, return requested section
             if the user asked (facts/issues/ruling), else return ruling.
          2) Else, run section-filtered vector searches for the intent-priority sections,
             merge and re-rank, dedupe, and return top-k snippets.
        """
        # 1) Exact G.R. No. path, with section intent
        m = GR_IN_QUERY.search(query or "")
        desired_sections = self._detect_intent_sections(query)
        lead = desired_sections[0] if desired_sections else None

        if m:
            gr = _normalize_gr(m.group(0))
            target = None
            if lead in ("facts", "issues", "ruling"):
                target = lead
            docs = self._scroll_by_gr_number(gr, target_section=target)
            if docs:
                return docs[:k]

        # 2) Semantic search with section filters
        qv = self.model.encode(query, convert_to_numpy=True).tolist()

        # Search a few top-priority sections first; widen if needed
        plan = desired_sections or SECTION_PRIORITY
        hits_by_section: Dict[str, List[Any]] = {}
        # Pull more than k per section for better re-ranking
        per_sec = max(8, k * 4)

        for sec in plan:
            hits = self._search_section(qv, sec, limit=per_sec)
            if hits:
                hits_by_section[sec] = hits

        # If everything is empty (collection tiny / sections missing), do a global search as fallback
        if not hits_by_section:
            hits = self.qdrant.search(collection_name=self.collection, query_vector=qv, limit=max(20, k*5),
                                      with_payload=True)
            # Build docs and return
            docs = [self._hit_to_doc(h) for h in hits][:k]
            return docs

        return self._merge_rerank(hits_by_section, query, k)
