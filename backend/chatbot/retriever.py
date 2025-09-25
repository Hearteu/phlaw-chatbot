# retriever.py — Simplified legal document retriever with chunking support
import gzip
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

# Import centralized model cache
from .model_cache import get_cached_embedding_model

# =============================================================================
# GLOBAL CACHING FOR PERFORMANCE
# =============================================================================
_QDRANT_CLIENT = None
_COLLECTION_INFO = None

# JSONL data file path
DATA_FILE = os.getenv("DATA_FILE", os.path.join(os.path.dirname(__file__), "..", "data", "cases.jsonl.gz"))


def _get_cached_embedding_model():
    """Get cached SentenceTransformer model with lazy loading"""
    return get_cached_embedding_model()

def _get_cached_qdrant_client():
    """Get cached Qdrant client"""
    global _QDRANT_CLIENT
    
    if _QDRANT_CLIENT is None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        _QDRANT_CLIENT = QdrantClient(host=host, port=port, grpc_port=6334, prefer_grpc=True, timeout=30.0)
    
    return _QDRANT_CLIENT

def close_qdrant_client():
    """Explicitly close the cached Qdrant client to avoid __del__ warnings"""
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is not None:
        try:
            _QDRANT_CLIENT.close()
        except Exception:
            pass

def _get_collection_info(collection_name: str) -> Dict[str, Any]:
    """Get cached collection information"""
    global _COLLECTION_INFO
    
    if _COLLECTION_INFO is None:
        client = _get_cached_qdrant_client()
        try:
            info = client.get_collection(collection_name)
            _COLLECTION_INFO = {
                "name": collection_name,
                "vector_count": getattr(info, 'points_count', 0),
                "status": getattr(info, 'status', 'Unknown')
            }
        except Exception as e:
            print(f"❌ Failed to get collection info: {e}")
            _COLLECTION_INFO = {"name": collection_name, "vector_count": 0, "status": "Unknown"}
    
    return _COLLECTION_INFO

# =============================================================================
# JSONL RETRIEVAL FUNCTIONS
# =============================================================================

def load_case_from_jsonl(case_id: str, jsonl_path: str = DATA_FILE) -> Optional[Dict[str, Any]]:
    """Load full case text from JSONL file by case ID or GR number"""
    print(f"🔍 Looking for case {case_id} in {jsonl_path}")
    
    if not os.path.exists(jsonl_path):
        print(f"❌ JSONL file not found: {jsonl_path}")
        return None
    
    try:
        with gzip.open(jsonl_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                case = json.loads(line)
                
                case_gr = case.get('gr_number', '')
                case_special = case.get('special_number', '')
                case_id_field = case.get('id', '')
                
                if (case_gr == case_id or 
                    case_special == case_id or
                    case_id_field == case_id or 
                    case_gr == f"G.R. No. {case_id}" or
                    case_gr == f"GR No. {case_id}"):
                    print(f"✅ Found case {case_id} at line {line_num}")
                    return case
                    
    except Exception as e:
        print(f"❌ Error loading case {case_id}: {e}")
    return None


class LegalRetriever:
    """Simplified legal document retriever with structure-aware chunking support"""
    
    def __init__(self, collection: str = "jurisprudence"):
        self.collection = collection
        self.model = _get_cached_embedding_model()
        self.qdrant = _get_cached_qdrant_client()
        
        # Verify collection exists
        if not self.qdrant.collection_exists(collection):
            raise ValueError(f"Collection '{collection}' does not exist")
        
        # Get collection info
        self.collection_info = _get_collection_info(collection)
        vector_count = self.collection_info['vector_count']
        if vector_count is not None:
            print(f"📊 Collection: {collection} | Vectors: {vector_count:,}")
        else:
            print(f"📊 Collection: {collection} | Vectors: Unknown")
        
        # Section boost weights for retrieval
        self.section_boosts = {
            'summary': 0.3,
            'ruling': 0.25, 
            'dispositive': 0.25,
            'facts': 0.15,
            'issues': 0.15,
            'arguments': 0.1,
            'legal_analysis': 0.1,
            'body': 0.05
        }
    
    def retrieve(self, query: str, k: int = 8, is_case_digest: bool = False, 
                conversation_history: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Simplified two-path retrieval with chunking support: GR-number exact match or keyword search"""
        start_time = time.time()
        
        print(f"🎯 Retrieving: '{query}'")
        
        # Path 1: Check if query contains GR number
        gr_number = self._extract_gr_number(query)
        if gr_number:
            print(f"📋 GR-number path: {gr_number}")
            results = self._retrieve_by_gr_number(gr_number, k)
            return self._apply_section_boosts(results)
        
        # Path 2: Check if query contains special number (A.M., OCA, etc.)
        special_number = self._extract_special_number(query)
        if special_number:
            print(f"📋 Special-number path: {special_number}")
            results = self._retrieve_by_special_number(special_number, k)
            return self._apply_section_boosts(results)
        
        # Path 3: Keyword search with section-aware retrieval
        print(f"📋 Keyword path: {query}")
        results = self._retrieve_by_keywords(query, k, is_case_digest)
        return self._apply_section_boosts(results)
    
    def _extract_gr_number(self, query: str) -> Optional[str]:
        """Extract GR number from query, returns normalized number or None"""
        if not query:
            return None
        
        patterns = [
            r"G\.R\.?\s*No\.?\s*([0-9\-]+)",
            r"GR\s*No\.?\s*([0-9\-]+)",
            r"\b(\d{5,})\b"  # 5+ digit number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_special_number(self, query: str) -> Optional[str]:
        """Extract special number from query (A.M., OCA, etc.), returns formatted number or None"""
        if not query:
            return None
        
        special_patterns = [
            (r"A\.M\.\s+No\.?\s*([0-9\-]+[A-Z]?)", "A.M. No. {}"),
            (r"OCA\s+No\.?\s*([0-9\-]+[A-Z]?)", "OCA No. {}"),
            (r"U\.C\.\s+No\.?\s*([0-9\-]+[A-Z]?)", "U.C. No. {}"),
            (r"ADM\s+No\.?\s*([0-9\-]+[A-Z]?)", "ADM No. {}"),
            (r"A\.C\.\s+No\.?\s*([0-9\-]+[A-Z]?)", "A.C. No. {}"),
            (r"AC\s+No\.?\s*([0-9\-]+[A-Z]?)", "AC No. {}"),
            (r"B\.M\.\s+No\.?\s*([0-9\-]+[A-Z]?)", "B.M. No. {}"),
            (r"LRC\s+No\.?\s*([0-9\-]+[A-Z]?)", "LRC No. {}"),
            (r"SP\s+No\.?\s*([0-9\-]+[A-Z]?)", "SP No. {}"),
        ]
        
        for pattern, format_str in special_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                number = match.group(1).strip()
                return format_str.format(number)
        
        return None
    
    def _retrieve_by_gr_number(self, gr_number: str, k: int) -> List[Dict[str, Any]]:
        """Exact GR number search in metadata"""
        try:
            gr_formats = [
                gr_number,
                f"G.R. No. {gr_number}",
                f"GR No. {gr_number}",
            ]
            
            for gr_format in gr_formats:
                dummy_vector = [0.0] * 768
                
                search_results = self.qdrant.search(
                    collection_name=self.collection,
                    query_vector=dummy_vector,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="gr_number",
                                match=MatchValue(value=gr_format)
                            )
                        ]
                    ),
                    limit=k,
                    with_payload=True
                )
                
                if search_results:
                    results = []
                    for hit in search_results:
                        doc = self._convert_hit_to_doc(hit, 'gr_number_exact')
                        results.append(doc)
                    
                    print(f"✅ Found {len(results)} records for GR {gr_number}")
                    return results
            
            print(f"⚠️ No results found for GR {gr_number}")
            return []
            
        except Exception as e:
            print(f"❌ Error searching GR {gr_number}: {e}")
            return []

    def _retrieve_by_special_number(self, special_number: str, k: int) -> List[Dict[str, Any]]:
        """Exact special number search in metadata"""
        try:
            special_formats = [
                special_number,
                special_number.upper(),
                special_number.lower(),
            ]
            
            for special_format in special_formats:
                dummy_vector = [0.0] * 768
                
                search_results = self.qdrant.search(
                    collection_name=self.collection,
                    query_vector=dummy_vector,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="special_number",
                                match=MatchValue(value=special_format)
                            )
                        ]
                    ),
                    limit=k,
                    with_payload=True
                )
                
                if search_results:
                    results = []
                    for hit in search_results:
                        doc = self._convert_hit_to_doc(hit, 'special_number_exact')
                        results.append(doc)
                    
                    print(f"✅ Found {len(results)} records for Special {special_number}")
                    return results
            
            print(f"⚠️ No results found for Special {special_number}")
            return []
            
        except Exception as e:
            print(f"❌ Error searching Special {special_number}: {e}")
            return []
    
    def _retrieve_by_keywords(self, query: str, k: int, is_case_digest: bool = False) -> List[Dict[str, Any]]:
        """Vector search with section awareness and deduplication"""
        # Get more candidates to allow for section diversity
        vector_results = self._perform_vector_search(query, k=k*2, is_case_digest=is_case_digest)
        print(f"📋 Vector: {len(vector_results)} candidates")
        
        # Apply section diversity and deduplication
        diverse_results = self._ensure_section_diversity(vector_results, k)
        return self._dedupe_by_case_and_section(diverse_results)[:k]
    
    def _perform_vector_search(self, query: str, k: int, is_case_digest: bool) -> List[Dict[str, Any]]:
        """Perform vector search using Qdrant"""
        try:
            # Encode query to vector
            query_vector = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            
            # Search Qdrant
            search_results = self.qdrant.search(
                collection_name=self.collection,
                query_vector=query_vector.tolist(),
                limit=k,
                with_payload=True
            )
            
            # Convert results to document format
            results = []
            for hit in search_results:
                doc = self._convert_hit_to_doc(hit, 'vector_search')
                results.append(doc)
            
            return results
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []
    
    def _convert_hit_to_doc(self, hit: Any, match_type: str) -> Dict[str, Any]:
        """Convert Qdrant hit to document format"""
        payload = hit.payload or {}
        
        # Extract content from multiple possible fields
        content = (payload.get('content', '') or 
                  payload.get('text', '') or 
                  payload.get('clean_text', '') or 
                  payload.get('body', '') or 
                  payload.get('case_text', '') or '')
        
        doc = {
            'title': payload.get('title', ''),
            'content': content,
            'gr_number': payload.get('gr_number', ''),
            'year': payload.get('year', ''),
            'section': payload.get('section', ''),
            'section_type': payload.get('section_type', 'general'),
            'score': hit.score,
            'url': payload.get('source_url', ''),
            'match_type': match_type
        }
        
        # Add metadata with all available fields
        metadata = {}
        for key, value in payload.items():
            if key not in ['title', 'content', 'gr_number', 'year', 'section', 'source_url']:
                metadata[key] = value
        
        # Add common metadata fields
        metadata.update({
            'case_id': payload.get('gr_number') or payload.get('id') or payload.get('case_id'),
            'ponente': payload.get('ponente'),
            'division': payload.get('division'),
            'case_type': payload.get('case_type'),
            'date': payload.get('date'),
            'promulgation_year': payload.get('promulgation_year'),
            'is_en_banc': payload.get('is_en_banc', False),
            'is_administrative': payload.get('is_administrative', False)
        })
        
        doc['metadata'] = metadata
        return doc
    
    def _apply_section_boosts(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply section-based score boosts to results"""
        for result in results:
            section = result.get('section', 'body')
            section_type = result.get('section_type', 'general')
            
            # Apply boost based on section type
            boost = self.section_boosts.get(section, 0.05)
            if section_type == 'dispositive':
                boost = max(boost, 0.25)
            elif section_type == 'summary':
                boost = max(boost, 0.3)
            
            result['score'] = result.get('score', 0) * (1 + boost)
        
        return results
    
    def _ensure_section_diversity(self, results: List[Dict], target_k: int) -> List[Dict]:
        """Ensure diverse section representation in results"""
        if not results:
            return results
        
        # Group by section type
        by_section = {}
        for result in results:
            section_type = result.get('section_type', 'general')
            if section_type not in by_section:
                by_section[section_type] = []
            by_section[section_type].append(result)
        
        # Prioritize sections: summary > dispositive > factual > issues > others
        section_priority = ['summary', 'dispositive', 'factual', 'issues', 'legal_analysis', 'general']
        
        diverse_results = []
        section_quotas = {
            'summary': max(1, target_k // 6),      # ~17%
            'dispositive': max(1, target_k // 4),   # ~25%
            'factual': max(1, target_k // 5),       # ~20%
            'issues': max(1, target_k // 6),        # ~17%
            'legal_analysis': max(1, target_k // 8), # ~12%
            'general': target_k // 10               # ~10%
        }
        
        # Add results respecting quotas and priority
        for section_type in section_priority:
            if section_type in by_section:
                quota = section_quotas.get(section_type, 1)
                section_results = by_section[section_type][:quota]
                diverse_results.extend(section_results)
                
                if len(diverse_results) >= target_k:
                    break
        
        # Fill remaining slots with best remaining results
        used_ids = {r.get('id', '') for r in diverse_results}
        remaining = [r for r in results if r.get('id', '') not in used_ids]
        
        while len(diverse_results) < target_k and remaining:
            diverse_results.append(remaining.pop(0))
        
        return diverse_results
    
    def _dedupe_by_case_and_section(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicates based on case_id and section, keeping highest scoring"""
        seen = {}
        deduped = []
        
        for result in results:
            case_id = result.get('metadata', {}).get('case_id', '')
            section = result.get('section', '')
            key = f"{case_id}_{section}"
            
            if key not in seen:
                seen[key] = result
                deduped.append(result)
            else:
                # Keep the one with higher score
                existing_score = seen[key].get('score', 0)
                new_score = result.get('score', 0)
                if new_score > existing_score:
                    seen[key] = result
                    # Find and replace in deduped
                    for i, item in enumerate(deduped):
                        item_case_id = item.get('metadata', {}).get('case_id', '')
                        item_section = item.get('section', '')
                        if f"{item_case_id}_{item_section}" == key:
                            deduped[i] = result
                            break
        
        return deduped
    
    def _create_context_from_chunks(self, chunks: List[Dict[str, Any]], max_tokens: int = 2500) -> str:
        """Create compact context from retrieved chunks"""
        if not chunks:
            return ""
        
        context_parts = []
        current_tokens = 0
        
        # Sort chunks by section priority and score
        section_priority = {'summary': 0, 'ruling': 1, 'dispositive': 1, 'facts': 2, 'issues': 3, 'arguments': 4, 'body': 5}
        sorted_chunks = sorted(chunks, key=lambda x: (section_priority.get(x.get('section', 'body'), 5), -x.get('score', 0)))
        
        for chunk in sorted_chunks:
            content = chunk.get('content', '')
            chunk_tokens = chunk.get('token_count', len(content) // 4)
            
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            # Add section header for clarity
            section = chunk.get('section', 'content')
            case_info = ""
            metadata = chunk.get('metadata', {})
            if metadata.get('gr_number'):
                case_info = f" ({metadata['gr_number']})"
            elif metadata.get('special_number'):
                case_info = f" ({metadata['special_number']})"
            
            context_parts.append(f"[{section.title()}{case_info}] {content}")
            current_tokens += chunk_tokens
        
        return "\n\n".join(context_parts)
