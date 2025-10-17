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

# Import Contextual RAG system
from .contextual_rag import create_contextual_rag_system
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
    
    def __init__(self, collection: str = "jurisprudence", use_contextual_rag: bool = True):
        self.collection = collection
        self.model = _get_cached_embedding_model()
        self.qdrant = _get_cached_qdrant_client()
        self.use_contextual_rag = True  # Always use contextual RAG
        # Simple cache for retrieval results to ensure consistency
        self._retrieval_cache = {}
        
        # Check if the requested collection exists and has sufficient data
        collection_has_data = self._check_collection_availability(collection)
        
        # Collection validation
        if not collection_has_data:
            print(f"⚠️ {collection} is empty or doesn't exist")
            raise ValueError(f"Collection '{collection}' is not available")
        
        # Initialize Contextual RAG system - always required
        self.contextual_rag = None
        try:
            from .contextual_rag import create_contextual_rag_system
            self.contextual_rag = create_contextual_rag_system(collection=collection)
            print(f"✅ Contextual RAG system initialized for collection: {collection}")
        except Exception as e:
            print(f"❌ Failed to initialize Contextual RAG system: {e}")
            raise RuntimeError(f"Contextual RAG is required but failed to initialize: {e}")
        
        # Verify collection exists
        if not self.qdrant.collection_exists(collection):
            raise ValueError(f"Collection '{collection}' does not exist")
        
        # Get collection info
        self.collection_info = _get_collection_info(collection)
        vector_count = self.collection_info['vector_count']
        if vector_count is not None:
            print(f"Collection: {collection} | Vectors: {vector_count:,}")
        else:
            print(f"Collection: {collection} | Vectors: Unknown")
    
    def retrieve(self, query: str, k: int = 8, is_case_digest: bool = False, 
                conversation_history: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Simplified two-path retrieval with chunking support: GR-number exact match or keyword search"""
        start_time = time.time()
        
        print(f"🎯 Retrieving: '{query}'")
        
        # Create cache key
        cache_key = f"{query.strip()}_{k}_{is_case_digest}"
        
        # Check cache first
        if cache_key in self._retrieval_cache:
            print(f"📋 Cache hit for query: {query[:50]}...")
            return self._retrieval_cache[cache_key]
        
        # Path 1: Check if query contains GR number
        gr_number = self._extract_gr_number(query)
        if gr_number:
            print(f"📋 GR-number path: {gr_number}")
            results = self._retrieve_by_gr_number(gr_number, k)
            # Cache the results
            self._retrieval_cache[cache_key] = results
            return results
        
        # Path 2: Check if query contains special number (A.M., OCA, etc.)
        special_number = self._extract_special_number(query)
        if special_number:
            print(f"📋 Special-number path: {special_number}")
            results = self._retrieve_by_special_number(special_number, k)
            # Cache the results
            self._retrieval_cache[cache_key] = results
            return results
        
        # Path 3: Use Optimized Contextual RAG for keyword search (always required)
        print(f"🚀 Optimized Contextual RAG path: {query}")
        try:
            # Check if optimized method is available
            if hasattr(self.contextual_rag, 'retrieve_and_rank_fast'):
                results = self.contextual_rag.retrieve_and_rank_fast(
                    query, 
                    vector_k=50,   # Reduced for speed
                    bm25_k=50,     # Reduced for speed
                    final_k=k
                )
            else:
                # Fallback to original method
                results = self.contextual_rag.retrieve_and_rank(
                    query, 
                    vector_k=100,  # Reduced for speed
                    bm25_k=100,    # Reduced for speed
                    final_k=k
                )
            # Cache the results
            self._retrieval_cache[cache_key] = results
            return results
        except Exception as e:
            print(f"❌ Contextual RAG failed: {e}")
            raise RuntimeError(f"Contextual RAG retrieval failed: {e}")
    
    def _extract_gr_number(self, query: str) -> Optional[str]:
        """Extract GR number from query, returns normalized number or None"""
        if not query:
            return None
        
        # Normalize query for consistent matching
        query = query.strip()
        
        patterns = [
            r"G\.R\.?\s*NOS?\.?\s*([0-9\-]+)",  # G.R. No. or G.R. NOS.
            r"GR\s*NOS?\.?\s*([0-9\-]+)",       # GR No. or GR NOS.
            r"\b(\d{5,})\b"  # 5+ digit number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Normalize the extracted number
                number = match.group(1).strip()
                # Remove any leading zeros and ensure consistent format
                if '-' in number:
                    parts = number.split('-')
                    normalized = '-'.join(parts)
                else:
                    normalized = number.lstrip('0') or '0'
                return normalized
        
        return None
    
    def _check_collection_availability(self, collection: str) -> bool:
        """Check if collection exists and has sufficient data"""
        try:
            if not self.qdrant.collection_exists(collection):
                return False
            
            info = self.qdrant.get_collection(collection)
            # Consider collection available if it has at least 100 points
            return info.points_count >= 100
            
        except Exception as e:
            print(f"⚠️ Error checking collection {collection}: {e}")
            return False
    
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
    
    
    def clear_cache(self):
        """Clear the retrieval cache"""
        self._retrieval_cache.clear()
        print("📋 Retrieval cache cleared")
    
    
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
