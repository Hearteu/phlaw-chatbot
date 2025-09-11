# retriever.py — Simplified legal document retriever for GR-number vs keyword paths
import gzip
import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

# BM25 implementation for ensemble retrieval
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank-bm25 not available. Install with: pip install rank-bm25")

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

# Try to import legal cross-encoder
try:
    from .legal_cross_encoder import (LegalCrossEncoderReranker,
                                      LegalDomainCrossEncoder,
                                      create_legal_cross_encoder)
    LEGAL_CROSS_ENCODER_AVAILABLE = True
except ImportError:
    LEGAL_CROSS_ENCODER_AVAILABLE = False
    print("Warning: Legal cross-encoder not available. Using standard CrossEncoder.")
    print("Warning: Enhanced components not available. Using basic functionality.")

# Import centralized model cache
from .model_cache import (clear_bm25_cache, clear_embedding_cache,
                          get_cached_bm25, get_cached_embedding_model)

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
# JSONL RETRIEVAL FUNCTIONS FOR HYBRID APPROACH
# =============================================================================

def load_case_from_jsonl(case_id: str, jsonl_path: str = DATA_FILE) -> Optional[Dict[str, Any]]:
    """Load full case text from JSONL file by case ID or GR number"""
    try:
        with gzip.open(jsonl_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                case = json.loads(line)
                if case.get('id') == case_id or case.get('gr_number') == case_id:
                    return case
    except Exception as e:
        print(f"Error loading case {case_id}: {e}")
    return None

def load_cases_from_jsonl(case_ids: List[str], jsonl_path: str = DATA_FILE) -> Dict[str, Dict[str, Any]]:
    """Load multiple cases from JSONL file efficiently"""
    cases = {}
    try:
        with gzip.open(jsonl_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                case = json.loads(line)
                
                # Try multiple ID matching strategies
                case_id = case.get('id') or case.get('gr_number')
                gr_number = case.get('gr_number')
                
                # Check if any of the requested IDs match this case
                for requested_id in case_ids:
                    if (case_id == requested_id or 
                        gr_number == requested_id or
                        (gr_number and gr_number != 'None' and gr_number in requested_id) or
                        (requested_id.startswith('G.R. No.') and gr_number and gr_number != 'None' and requested_id.endswith(gr_number))):
                        cases[requested_id] = case
                        break
                
                if len(cases) >= len(case_ids):
                    break
    except Exception as e:
        print(f"Error loading cases: {e}")
    
    return cases

def extract_case_sections(case: Dict[str, Any]) -> Dict[str, str]:
    """Extract key sections from a case for detailed retrieval"""
    sections = {}
    
    # Extract main content from sections dict
    if 'sections' in case and isinstance(case['sections'], dict):
        for section_name, content in case['sections'].items():
            if content and isinstance(content, str) and content.strip():
                sections[section_name.lower()] = content
    
    # Extract top-level content fields for backward compatibility
    top_level_fields = ['header', 'body', 'ruling', 'facts', 'issues', 'arguments']
    for field in top_level_fields:
        if field in case and case[field] and isinstance(case[field], str) and case[field].strip():
            sections[field] = case[field]
    
    # Extract full text if available
    if 'clean_text' in case and case['clean_text']:
        sections['full_text'] = case['clean_text']
    elif 'text' in case and case['text']:
        sections['full_text'] = case['text']
    
    return sections


class LegalRetriever:
    """Simplified legal document retriever for GR-number vs keyword paths"""
    
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
        
        # Initialize BM25 for ensemble retrieval
        self.bm25_model = None
        self.bm25_corpus = []
        self.bm25_doc_metadata = []
        self._init_bm25()
        
        # Cross-encoder configuration
        self.cross_encoder_config = {
            'model_type': os.getenv('CROSS_ENCODER_MODEL_TYPE', 'legal'),
            'max_length': int(os.getenv('CROSS_ENCODER_MAX_LENGTH', '512')),
            'batch_size': int(os.getenv('CROSS_ENCODER_BATCH_SIZE', '16')),
            'use_legal_optimization': os.getenv('CROSS_ENCODER_LEGAL_OPT', 'true').lower() == 'true',
            'device': os.getenv('CROSS_ENCODER_DEVICE', 'auto')
        }
        
        # Initialize reranker
        self.reranker = None
        self._init_reranker()
        
        # Enhanced caching for performance
        self._rerank_cache = {}
        self._entity_cache = {}
        self._query_analysis_cache = {}
        self._similarity_cache = {}
        
        # Performance monitoring
        self._performance_stats = {
            'total_queries': 0,
            'avg_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def _init_bm25(self):
        """Initialize BM25 model for ensemble retrieval using cache"""
        if not BM25_AVAILABLE:
            print("BM25 not available - ensemble retrieval disabled")
            return
        
        # Try to get cached BM25 model first
        cached_result = get_cached_bm25()
        
        if cached_result[0] is not None:  # model is not None
            self.bm25_model, self.bm25_corpus, self.bm25_doc_metadata = cached_result
            print("✅ Using cached BM25 model")
            return
        
        try:
            print("🔄 Initializing BM25 corpus from Qdrant collection...")
            self._build_bm25_corpus()
            if self.bm25_corpus:
                self.bm25_model = BM25Okapi(self.bm25_corpus)
                print(f"✅ BM25 model initialized with {len(self.bm25_corpus)} documents")
            else:
                print("⚠️ No documents found for BM25 corpus")
        except Exception as e:
            print(f"❌ Failed to initialize BM25: {e}")
            self.bm25_model = None
    
    def _build_bm25_corpus(self):
        """Build BM25 corpus from Qdrant collection"""
        try:
            # Get all documents from Qdrant collection
            scroll_result = self.qdrant.scroll(
                collection_name=self.collection,
                limit=10000,  # Adjust based on your collection size
                with_payload=True
            )
            
            self.bm25_corpus = []
            self.bm25_doc_metadata = []
            
            for point in scroll_result[0]:  # scroll_result is (points, next_page_offset)
                payload = point.payload
                if not payload:
                    continue
                
                # Extract text content for BM25
                text_content = self._extract_text_for_bm25(payload)
                if text_content:
                    # Tokenize for BM25
                    tokens = self._tokenize_for_bm25(text_content)
                    if tokens:
                        self.bm25_corpus.append(tokens)
                        self.bm25_doc_metadata.append({
                            'point_id': point.id,
                            'payload': payload,
                            'text': text_content
                        })
            
            print(f"📚 Built BM25 corpus with {len(self.bm25_corpus)} documents")
            
        except Exception as e:
            print(f"❌ Error building BM25 corpus: {e}")
            self.bm25_corpus = []
            self.bm25_doc_metadata = []
    
    def _extract_text_for_bm25(self, payload: Dict[str, Any]) -> str:
        """Extract and combine text content from payload for BM25"""
        if not isinstance(payload, dict):
            return ""
            
        text_parts = []
        
        # Extract from different sections
        if 'section' in payload:
            section = payload['section']
            if section == 'caption' and 'title' in payload:
                text_parts.append(payload['title'])
            elif section == 'ruling' and 'content' in payload:
                text_parts.append(payload['content'])
            elif section == 'header' and 'content' in payload:
                text_parts.append(payload['content'])
            elif section in ['facts', 'issues', 'arguments'] and 'content' in payload:
                text_parts.append(payload['content'])
        
        # Add metadata fields
        for field in ['title', 'gr_number', 'ponente', 'division']:
            if field in payload and payload[field]:
                text_parts.append(str(payload[field]))
        
        return ' '.join(text_parts)
    
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize text for BM25 processing"""
        if not text:
            return []
        
        # Simple tokenization - can be enhanced with legal-specific tokenization
        import re

        # Remove special characters and split on whitespace
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _init_reranker(self):
        """Initialize the advanced legal cross-encoder reranker using cache"""
        # Try to get cached cross-encoder first
        from .model_cache import get_cached_cross_encoder
        cached_reranker = get_cached_cross_encoder()
        
        if cached_reranker is not None:
            self.reranker = cached_reranker
            print("✅ Using cached cross-encoder reranker")
            return
        
        # Fallback to direct loading if cache is empty
        if LEGAL_CROSS_ENCODER_AVAILABLE:
            try:
                print("Loading advanced legal cross-encoder reranker...")
                
                # Determine device
                device = self.cross_encoder_config['device']
                if device == 'auto':
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.reranker = create_legal_cross_encoder(
                    model_type=self.cross_encoder_config['model_type'],
                    device=device,
                    max_length=self.cross_encoder_config['max_length'],
                    batch_size=self.cross_encoder_config['batch_size'],
                    use_legal_optimization=self.cross_encoder_config['use_legal_optimization']
                )
                print(f"Advanced legal cross-encoder loaded successfully on {device}")
            except Exception as e:
                print(f"Failed to load legal cross-encoder: {e}")
                self.reranker = None
        else:
            print("No reranker available - using fallback ranking")
    
    def retrieve(self, query: str, k: int = 8, is_case_digest: bool = False, 
                conversation_history: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Simplified two-path retrieval: GR-number exact match or keyword search"""
        start_time = time.time()
        
        print(f"🎯 Retrieving: '{query}'")
        
        # Path 1: Check if query contains GR number
        gr_number = self._extract_gr_number(query)
        if gr_number:
            print(f"📋 GR-number path: {gr_number}")
            return self._retrieve_by_gr_number(gr_number, k)
        
        # Path 2: Keyword search with ensemble retrieval
        print(f"📋 Keyword path: {query}")
        return self._retrieve_by_keywords(query, k, is_case_digest)
    
    def _extract_gr_number(self, query: str) -> Optional[str]:
        """Extract GR number from query, returns normalized number or None"""
        if not query:
            return None
        
        # Common patterns: "G.R. No. 123456", "GR No. 123456", bare digits
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
    
    def _retrieve_by_gr_number(self, gr_number: str, k: int) -> List[Dict[str, Any]]:
        """Exact GR number search in metadata"""
        try:
            # Try multiple GR number formats
            gr_formats = [
                gr_number,  # Raw number
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
                    
                    print(f"✅ Found {len(results)} results for GR {gr_number}")
                    return results
            
            print(f"⚠️ No results found for GR {gr_number}")
            return []
            
        except Exception as e:
            print(f"❌ Error searching GR {gr_number}: {e}")
            return []
    
    def _retrieve_by_keywords(self, query: str, k: int, is_case_digest: bool = False) -> List[Dict[str, Any]]:
        """Keyword search using ensemble retrieval (BM25 + Legal RoBERTa)"""
        if not self.bm25_model:
            # Fallback to vector search only
            return self._perform_vector_search(query, k, is_case_digest)
        
        # Get BM25 results
        bm25_results = self._bm25_search(query, k=k*2)
        print(f"📋 BM25: {len(bm25_results)} candidates")
        
        # Get vector search results
        vector_results = self._perform_vector_search(query, k=k*2, is_case_digest=is_case_digest)
        print(f"📋 Vector: {len(vector_results)} candidates")
        
        # Combine and deduplicate by gr_number
        combined = self._combine_and_dedupe(bm25_results, vector_results, query)
        
        # Apply case_type boosting
        combined = self.boost_by_case_type(combined, query)
        
        # Apply reranking if available
        if self.reranker and len(combined) > 1:
            print("🔄 Applying reranking...")
            combined = self._rerank_results(query, combined)
        
        # Return top k unique cases
        return self._dedupe_by_gr_number(combined)[:k]
    
    def _combine_and_dedupe(self, bm25_results: List[Dict], vector_results: List[Dict], query: str) -> List[Dict]:
        """Combine BM25 and vector results, deduplicating by gr_number"""
        # Create a map of gr_number -> best result
        gr_map = {}
        
        # Add BM25 results first (they often have good keyword matches)
        for result in bm25_results:
            gr_num = result.get('gr_number', '')
            if gr_num and gr_num not in gr_map:
                result['match_type'] = 'bm25'
                gr_map[gr_num] = result
        
        # Add vector results, preferring higher scores
        for result in vector_results:
            gr_num = result.get('gr_number', '')
            if gr_num:
                if gr_num not in gr_map:
                    result['match_type'] = 'vector'
                    gr_map[gr_num] = result
                else:
                    # Keep the one with higher score
                    existing_score = gr_map[gr_num].get('score', 0)
                    new_score = result.get('score', 0)
                    if new_score > existing_score:
                        result['match_type'] = 'vector'
                        gr_map[gr_num] = result
        
        # Convert back to list and sort by score
        combined = list(gr_map.values())
        combined.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return combined
    
    def _dedupe_by_gr_number(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicates based on gr_number, keeping highest scoring"""
        seen = {}
        deduped = []
        
        for result in results:
            gr_num = result.get('gr_number', '')
            if gr_num and gr_num not in seen:
                seen[gr_num] = result
                deduped.append(result)
            elif gr_num and gr_num in seen:
                # Keep the one with higher score
                existing_score = seen[gr_num].get('score', 0)
                new_score = result.get('score', 0)
                if new_score > existing_score:
                    # Replace in both seen and deduped
                    seen[gr_num] = result
                    # Find and replace in deduped
                    for i, item in enumerate(deduped):
                        if item.get('gr_number') == gr_num:
                            deduped[i] = result
                            break
        
        return deduped
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Apply cross-encoder reranking if available"""
        if not self.reranker or len(results) <= 1:
            return results
        
        try:
            # Prepare query-document pairs
            pairs = []
            for result in results:
                content = result.get('content', '')[:512]  # Limit length
                pairs.append((query, content))
            
            # Get rerank scores
            scores = self.reranker.predict(pairs)
            
            # Update scores and sort
            for i, result in enumerate(results):
                result['rerank_score'] = float(scores[i])
                result['final_score'] = result.get('score', 0) + float(scores[i]) * 0.1
            
            # Sort by final score
            results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
        except Exception as e:
            print(f"⚠️ Reranking failed: {e}")
        
        return results
    
    def retrieve_by_case_type(self, case_type: str, k: int = 8) -> List[Dict[str, Any]]:
        """Retrieve cases by case_type (e.g., 'annulment', 'criminal')"""
        try:
            dummy_vector = [0.0] * 768
            
            search_results = self.qdrant.search(
                collection_name=self.collection,
                query_vector=dummy_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="case_type",
                            match=MatchValue(value=case_type)
                        )
                    ]
                ),
                limit=k,
                with_payload=True
            )
            
            results = []
            for hit in search_results:
                doc = self._convert_hit_to_doc(hit, 'case_type_filter')
                results.append(doc)
            
            print(f"✅ Found {len(results)} {case_type} cases")
            return results
            
        except Exception as e:
            print(f"❌ Error searching case_type {case_type}: {e}")
            return []
    
    def boost_by_case_type(self, results: List[Dict], query: str) -> List[Dict]:
        """Boost results that match case_type patterns in query"""
        if not results:
            return results
        
        # Extract potential case types from query
        query_lower = query.lower()
        case_type_boost = 0.1
        
        for result in results:
            case_type = result.get('case_type', '').lower()
            case_tags = result.get('case_type_tags', [])
            
            # Boost if case_type matches query terms
            if case_type and case_type in query_lower:
                result['score'] = result.get('score', 0) + case_type_boost
            
            # Boost if any case_type_tags match query terms
            for tag in case_tags:
                if tag.lower() in query_lower:
                    result['score'] = result.get('score', 0) + case_type_boost * 0.5
                    break
        
        # Re-sort by updated scores
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return results
    
    
    def ensemble_retrieve(self, query: str, k: int = 8, is_case_digest: bool = False, 
                         conversation_history: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Ensemble retrieval combining BM25 + Legal RoBERTa (Stern5497)"""
        start_time = time.time()
        self._performance_stats['total_queries'] += 1
        
        print(f"🎯 Ensemble retrieval (BM25 + Legal RoBERTa) for: '{query}'")
        
        # Step 1: Query analysis
        query_analysis = self._analyze_query(query, conversation_history)
        print(f"📊 Query type: {query_analysis.query_type.value if hasattr(query_analysis.query_type, 'value') else str(query_analysis.query_type)}, Complexity: {query_analysis.complexity.value if hasattr(query_analysis.complexity, 'value') else str(query_analysis.complexity)}")
        
        # Step 2: Get BM25 results
        bm25_results = self._bm25_search(query, k=k*2)
        print(f"📋 BM25 retrieved: {len(bm25_results)} candidates")
        
        # Step 3: Get Legal RoBERTa results
        roberta_results = self._perform_vector_search(query, k=k*2, is_case_digest=is_case_digest)
        print(f"📋 Legal RoBERTa retrieved: {len(roberta_results)} candidates")
        
        # Step 4: Combine and score results
        ensemble_results = self._combine_ensemble_results(
            bm25_results, roberta_results, query, query_analysis, k
        )
        print(f"🎯 Ensemble combined: {len(ensemble_results)} final results")
        
        # Step 5: Apply reranking if available
        if self.reranker and len(ensemble_results) > 1:
            print("🔄 Applying ensemble reranking...")
            ensemble_results = self._ensemble_rerank(query, ensemble_results, query_analysis)
        
        
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time)
        
        print(f"✅ Ensemble retrieval completed: {len(ensemble_results)} results in {processing_time:.2f}s")
        
        return ensemble_results
    
    
    def _bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform BM25 search"""
        if not self.bm25_model or not self.bm25_corpus:
            print("⚠️ BM25 model not available")
            return []
        
        try:
            # Tokenize query
            query_tokens = self._tokenize_for_bm25(query)
            if not query_tokens:
                return []
            
            # Get BM25 scores
            scores = self.bm25_model.get_scores(query_tokens)
            
            # Get top k results
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include results with positive scores
                    doc_metadata = self.bm25_doc_metadata[idx]
                    result = self._convert_bm25_to_doc(doc_metadata, scores[idx])
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in BM25 search: {e}")
            return []
    
    def _convert_bm25_to_doc(self, doc_metadata: Dict[str, Any], bm25_score: float) -> Dict[str, Any]:
        """Convert BM25 result to document format"""
        payload = doc_metadata.get('payload', {})
        
        # Debug: Check what's in doc_metadata
        if not payload:
            print(f"⚠️ Empty payload in doc_metadata: {doc_metadata.keys()}")
            # Try to use the text content if payload is empty
            if 'text' in doc_metadata:
                payload = {'content': doc_metadata['text']}
        
        # Ensure payload is a dictionary
        if not isinstance(payload, dict):
            print(f"⚠️ Payload is not a dict: {type(payload)} = {payload}")
            payload = {}
        
        # Extract all available fields from payload
        doc = {
            'title': payload.get('title', ''),
            'content': self._extract_text_for_bm25(payload),
            'gr_number': payload.get('gr_number', ''),
            'year': payload.get('year', ''),
            'section': payload.get('section', ''),
            'score': bm25_score,
            'url': payload.get('source_url', ''),
            'match_type': 'bm25',
            'source': 'bm25_corpus'
        }
        
        # Add metadata with all available fields
        metadata = {}
        for key, value in payload.items():
            if key not in ['title', 'content', 'gr_number', 'year', 'section', 'source_url']:
                metadata[key] = value
        
        # Add common metadata fields
        metadata.update({
            'case_id': payload.get('gr_number') or payload.get('id'),
            'ponente': payload.get('ponente'),
            'division': payload.get('division'),
            'case_type': payload.get('case_type'),
            'date': payload.get('date'),
            'promulgation_year': payload.get('promulgation_year'),
            'is_en_banc': payload.get('is_en_banc', False),
            'is_administrative': payload.get('is_administrative', False),
            'legal_areas': payload.get('legal_areas', []),
            'legal_concepts': payload.get('legal_concepts', []),
            'legal_doctrines': payload.get('legal_doctrines', []),
            'precedential_value': payload.get('precedential_value', ''),
            'citation_count': payload.get('citation_count', 0),
            'cited_cases': payload.get('cited_cases', []),
            'point_id': doc_metadata.get('point_id', '')
        })
        
        doc['metadata'] = metadata
        
        return doc
    
    def _combine_ensemble_results(self, bm25_results: List[Dict[str, Any]], 
                                 roberta_results: List[Dict[str, Any]], 
                                 query: str, query_analysis: Any, k: int) -> List[Dict[str, Any]]:
        """Combine BM25 and Legal RoBERTa results with ensemble scoring"""
        
        # Create a mapping of results by unique identifier
        combined_results = {}
        
        # Add BM25 results
        for result in bm25_results:
            key = self._get_result_key(result)
            if key not in combined_results:
                combined_results[key] = {
                    'bm25_score': result['score'],
                    'roberta_score': 0.0,
                    'result': result
                }
            else:
                combined_results[key]['bm25_score'] = max(
                    combined_results[key]['bm25_score'], result['score']
                )
        
        # Add Legal RoBERTa results
        for result in roberta_results:
            key = self._get_result_key(result)
            if key not in combined_results:
                combined_results[key] = {
                    'bm25_score': 0.0,
                    'roberta_score': result['score'],
                    'result': result
                }
            else:
                combined_results[key]['roberta_score'] = max(
                    combined_results[key]['roberta_score'], result['score']
                )
        
        # Calculate ensemble scores
        ensemble_results = []
        for key, scores in combined_results.items():
            # Normalize scores to [0, 1] range
            bm25_norm = self._normalize_score(scores['bm25_score'], 'bm25')
            roberta_norm = self._normalize_score(scores['roberta_score'], 'roberta')
            
            # Weighted ensemble score (can be tuned)
            ensemble_score = 0.4 * bm25_norm + 0.6 * roberta_norm
            
            # Update the result with ensemble information
            result = scores['result'].copy()
            result['score'] = ensemble_score
            result['bm25_score'] = scores['bm25_score']
            result['roberta_score'] = scores['roberta_score']
            result['ensemble_score'] = ensemble_score
            result['match_type'] = 'ensemble'
            
            ensemble_results.append(result)
        
        # Sort by ensemble score and return top k
        ensemble_results.sort(key=lambda x: x['ensemble_score'], reverse=True)
        return ensemble_results[:k]
    
    def _get_result_key(self, result: Dict[str, Any]) -> str:
        """Get unique key for result deduplication"""
        metadata = result.get('metadata', {})
        return f"{metadata.get('gr_number', '')}_{metadata.get('section', '')}_{metadata.get('point_id', '')}"
    
    def _normalize_score(self, score: float, score_type: str) -> float:
        """Normalize scores to [0, 1] range"""
        if score_type == 'bm25':
            # BM25 scores can be negative, normalize to [0, 1]
            return max(0.0, min(1.0, (score + 2) / 4))  # Rough normalization
        elif score_type == 'roberta':
            # RoBERTa cosine similarity is already in [0, 1]
            return max(0.0, min(1.0, score))
        else:
            return max(0.0, min(1.0, score))
    
    def _ensemble_rerank(self, query: str, results: List[Dict[str, Any]], 
                        query_analysis: Any) -> List[Dict[str, Any]]:
        """Apply advanced cross-encoder reranking to ensemble results"""
        if not self.reranker or len(results) <= 1:
            return results
        
        try:
            # Use advanced cross-encoder if available
            if hasattr(self.reranker, 'rerank'):
                print("🔄 Using advanced cross-encoder reranking...")
                reranked_results = self.reranker.rerank(
                    query=query,
                    documents=results,
                    top_k=len(results),
                    use_legal_model=True
                )
                
                # Update final scores combining ensemble and cross-encoder scores
                for result in reranked_results:
                    ensemble_score = result.get('ensemble_score', result.get('score', 0))
                    cross_encoder_score = result.get('cross_encoder_score', 0)
                    result['final_score'] = 0.6 * ensemble_score + 0.4 * cross_encoder_score
                
                return reranked_results
            else:
                # Fallback to standard CrossEncoder approach
                print("🔄 Using standard CrossEncoder reranking...")
                pairs = []
                for result in results:
                    content = result.get('content', '')
                    if content:
                        pairs.append([query, content])
                
                if not pairs:
                    return results
                
                # Get reranking scores
                rerank_scores = self.reranker.predict(pairs)
                
                # Update results with reranking scores
                for i, result in enumerate(results):
                    if i < len(rerank_scores):
                        result['rerank_score'] = float(rerank_scores[i])
                        # Combine ensemble score with rerank score
                        result['final_score'] = 0.7 * result['ensemble_score'] + 0.3 * result['rerank_score']
                    else:
                        result['final_score'] = result['ensemble_score']
                
                # Sort by final score
                results.sort(key=lambda x: x['final_score'], reverse=True)
                return results
            
        except Exception as e:
            print(f"❌ Error in ensemble reranking: {e}")
            return results
    
    def rebuild_bm25_corpus(self):
        """Rebuild BM25 corpus from Qdrant collection"""
        print("🔄 Rebuilding BM25 corpus...")
        self._build_bm25_corpus()
        if self.bm25_corpus:
            self.bm25_model = BM25Okapi(self.bm25_corpus)
            print(f"✅ BM25 corpus rebuilt with {len(self.bm25_corpus)} documents")
        else:
            print("⚠️ No documents found for BM25 corpus")
    
    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get status of ensemble retrieval components"""
        reranker_stats = {}
        if self.reranker and hasattr(self.reranker, 'get_stats'):
            reranker_stats = self.reranker.get_stats()
        
        return {
            'bm25_available': BM25_AVAILABLE,
            'bm25_model_loaded': self.bm25_model is not None,
            'bm25_corpus_size': len(self.bm25_corpus),
            'roberta_model_loaded': self.model is not None,
            'reranker_available': self.reranker is not None,
            'legal_cross_encoder_available': LEGAL_CROSS_ENCODER_AVAILABLE,
            'cross_encoder_loaded': self.reranker is not None and hasattr(self.reranker, 'rerank'),
            'ensemble_ready': self.bm25_model is not None and self.model is not None,
            'cross_encoder_config': self.cross_encoder_config,
            'reranker_stats': reranker_stats
        }
    
    def update_cross_encoder_config(self, **kwargs):
        """Update cross-encoder configuration and reload if needed"""
        config_updated = False
        
        for key, value in kwargs.items():
            if key in self.cross_encoder_config and self.cross_encoder_config[key] != value:
                self.cross_encoder_config[key] = value
                config_updated = True
                print(f"Updated cross-encoder config: {key} = {value}")
        
        if config_updated and LEGAL_CROSS_ENCODER_AVAILABLE:
            print("🔄 Reloading cross-encoder with new configuration...")
            self._init_reranker()
    
    def reload_cross_encoder(self):
        """Reload the cross-encoder with current configuration"""
        if LEGAL_CROSS_ENCODER_AVAILABLE:
            print("🔄 Reloading cross-encoder...")
            self._init_reranker()
        else:
            print("❌ Legal cross-encoder not available")
    
    def get_cross_encoder_performance(self) -> Dict[str, Any]:
        """Get cross-encoder performance statistics"""
        if self.reranker and hasattr(self.reranker, 'get_stats'):
            return self.reranker.get_stats()
        return {"error": "Cross-encoder not available or no stats available"}
    

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
    
    
    def _analyze_query(self, query: str, conversation_history: Optional[List[Dict]]) -> Any:
        """Analyze query using enhanced query processor"""
        # Check cache first
        cache_key = f"analysis_{hash(query)}_{hash(str(conversation_history))}"
        if cache_key in self._query_analysis_cache:
            self._performance_stats['cache_hits'] += 1
            return self._query_analysis_cache[cache_key]
        
        self._performance_stats['cache_misses'] += 1
        
        # Use basic analysis for simplified two-path logic
        analysis = self._basic_query_analysis(query, conversation_history)
        
        # Cache result
        self._query_analysis_cache[cache_key] = analysis
        
        return analysis
    
    def _basic_query_analysis(self, query: str, conversation_history: Optional[List[Dict]]) -> Any:
        """Basic query analysis fallback when enhanced components not available"""
        # Create a simple analysis object with basic structure
        class BasicQueryAnalysis:
            def __init__(self, query: str):
                self.original_query = query
                self.query_type = "general_legal"  # Default type
                self.complexity = "simple"  # Default complexity
                self.entities = {
                    'gr_numbers': [],
                    'case_names': [],
                    'legal_terms': [],
                    'people': [],
                    'organizations': [],
                    'dates': []
                }
                self.legal_terms = []
                self.intent_flags = {
                    'wants_facts': False,
                    'wants_ruling': False,
                    'wants_issues': False,
                    'wants_digest': False,
                    'wants_arguments': False
                }
                self.reformulated_queries = [query]
                self.suggested_filters = {}
                self.context_requirements = []
        
        return BasicQueryAnalysis(query)
    
    def _multi_strategy_retrieve(self, query: str, query_analysis: Any, k: int, is_case_digest: bool) -> List[Dict[str, Any]]:
        """Multi-strategy retrieval based on query analysis"""
        all_candidates = []
        
        # Strategy 1: Exact entity matching (highest priority)
        if hasattr(query_analysis, 'entities') and query_analysis.entities.get('gr_numbers'):
            gr_candidates = self._retrieve_by_gr_numbers(query_analysis.entities['gr_numbers'])
            all_candidates.extend(gr_candidates)
            print(f"🎯 G.R. number matches: {len(gr_candidates)}")
        
        if hasattr(query_analysis, 'entities') and query_analysis.entities.get('case_names'):
            case_candidates = self._retrieve_by_case_names(query_analysis.entities['case_names'])
            all_candidates.extend(case_candidates)
            print(f"📋 Case name matches: {len(case_candidates)}")
        
        # Strategy 2: Legal term matching
        if hasattr(query_analysis, 'entities') and query_analysis.entities.get('legal_terms'):
            term_candidates = self._retrieve_by_legal_terms(query_analysis.entities['legal_terms'])
            all_candidates.extend(term_candidates)
            print(f"⚖️ Legal term matches: {len(term_candidates)}")
        
        # Strategy 3: Semantic vector search
        semantic_candidates = self._semantic_search(query, k * 2, is_case_digest)
        all_candidates.extend(semantic_candidates)
        print(f"🔍 Semantic matches: {len(semantic_candidates)}")
        
        # Strategy 4: Query reformulation search
        if hasattr(query_analysis, 'reformulated_queries'):
            for reformulated_query in query_analysis.reformulated_queries[:3]:  # Limit to top 3
                if reformulated_query != query:
                    reformulated_candidates = self._semantic_search(reformulated_query, k, is_case_digest)
                    all_candidates.extend(reformulated_candidates)
                    print(f"🔄 Reformulated query matches: {len(reformulated_candidates)}")
        
        # Strategy 5: Contextual search (if conversation history available)
        if hasattr(query_analysis, 'context_requirements') and 'conversation_context' in query_analysis.context_requirements:
            contextual_candidates = self._contextual_search(query, query_analysis)
            all_candidates.extend(contextual_candidates)
            print(f"💬 Contextual matches: {len(contextual_candidates)}")
        
        # Deduplicate candidates
        unique_candidates = self._deduplicate_candidates(all_candidates)
        print(f"🔄 After deduplication: {len(unique_candidates)} unique candidates")
        
        return unique_candidates
    
    def _retrieve_by_gr_numbers(self, gr_numbers: List[str]) -> List[Dict[str, Any]]:
        """Retrieve cases by G.R. numbers with enhanced matching"""
        results = []
        
        for gr_number in gr_numbers:
            # Check cache first
            cache_key = f"gr_{gr_number}"
            if cache_key in self._entity_cache:
                results.extend(self._entity_cache[cache_key])
                continue
            
            try:
                # Try multiple G.R. number formats
                gr_formats = [
                    f"G.R. No. {gr_number}",
                    f"GR No. {gr_number}",
                    gr_number
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
                        limit=5,
                        with_payload=True
                    )
                    
                    if search_results:
                        break
                
                # Convert to our format
                for hit in search_results:
                    doc = self._convert_hit_to_doc(hit, 'gr_number_exact')
                    results.append(doc)
                
                # Cache results
                self._entity_cache[cache_key] = results[-len(search_results):]
                
            except Exception as e:
                print(f"Error searching G.R. number {gr_number}: {e}")
        
        return results
    
    def _retrieve_by_case_names(self, case_names: List[str]) -> List[Dict[str, Any]]:
        """Retrieve cases by case names with enhanced matching"""
        results = []
        
        for case_name in case_names:
            # Check cache first
            cache_key = f"case_{case_name.lower()}"
            if cache_key in self._entity_cache:
                results.extend(self._entity_cache[cache_key])
                continue
            
            try:
                dummy_vector = [0.0] * 768
                
                search_results = self.qdrant.search(
                    collection_name=self.collection,
                    query_vector=dummy_vector,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="title",
                                match=MatchValue(value=case_name)
                            )
                        ]
                    ),
                    limit=3,
                    with_payload=True
                )
                
                # Convert to our format
                for hit in search_results:
                    doc = self._convert_hit_to_doc(hit, 'case_name_match')
                    results.append(doc)
                
                # Cache results
                self._entity_cache[cache_key] = results[-len(search_results):]
                
            except Exception as e:
                print(f"Error searching case name {case_name}: {e}")
        
        return results
    
    def _retrieve_by_legal_terms(self, legal_terms: List[str]) -> List[Dict[str, Any]]:
        """Retrieve cases by legal terms with enhanced matching"""
        results = []
        
        for term in legal_terms:
            try:
                # Use semantic search for legal terms
                term_candidates = self._semantic_search(term, 3, False)
                for candidate in term_candidates:
                    candidate['match_type'] = 'legal_term_match'
                    candidate['matched_term'] = term
                results.extend(term_candidates)
                
            except Exception as e:
                print(f"Error searching legal term {term}: {e}")
        
        return results
    
    def _semantic_search(self, query: str, k: int, is_case_digest: bool) -> List[Dict[str, Any]]:
        """Perform semantic vector search"""
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
                doc = self._convert_hit_to_doc(hit, 'semantic_search')
                results.append(doc)
            
            return results
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def _contextual_search(self, query: str, query_analysis: Any) -> List[Dict[str, Any]]:
        """Perform contextual search based on conversation history"""
        # This would implement contextual search logic
        # For now, return empty list
        return []
    
    def _convert_hit_to_doc(self, hit: Any, match_type: str) -> Dict[str, Any]:
        """Convert Qdrant hit to document format"""
        payload = hit.payload or {}
        
        # Extract all available fields from payload
        doc = {
            'title': payload.get('title', ''),
            'content': payload.get('content', ''),
            'gr_number': payload.get('gr_number', ''),
            'year': payload.get('year', ''),
            'section': payload.get('section', ''),
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
            'case_id': payload.get('gr_number') or payload.get('id'),
            'ponente': payload.get('ponente'),
            'division': payload.get('division'),
            'case_type': payload.get('case_type'),
            'date': payload.get('date'),
            'promulgation_year': payload.get('promulgation_year'),
            'is_en_banc': payload.get('is_en_banc', False),
            'is_administrative': payload.get('is_administrative', False),
            'legal_areas': payload.get('legal_areas', []),
            'legal_concepts': payload.get('legal_concepts', []),
            'legal_doctrines': payload.get('legal_doctrines', []),
            'precedential_value': payload.get('precedential_value', ''),
            'citation_count': payload.get('citation_count', 0),
            'cited_cases': payload.get('cited_cases', [])
        })
        
        doc['metadata'] = metadata
        
        return doc
    
    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate candidates based on G.R. number and title"""
        seen = set()
        unique_candidates = []
        
        for candidate in candidates:
            # Create a unique key based on G.R. number and title
            gr_number = candidate.get('gr_number', '')
            title = candidate.get('title', '')
            key = f"{gr_number}_{title}"
            
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _advanced_rerank(self, query: str, candidates: List[Dict[str, Any]], query_analysis: Any) -> List[Dict[str, Any]]:
        """Advanced reranking using multiple factors"""
        if not self.reranker or len(candidates) <= 1:
            return candidates
        
        print(f"🔄 Advanced reranking {len(candidates)} candidates...")
        
        # Create cache key
        cache_key = f"rerank_{hash(query)}_{hash(str(candidates))}_{hash(str(query_analysis))}"
        if cache_key in self._rerank_cache:
            return self._rerank_cache[cache_key]
        
        try:
            # Use advanced cross-encoder if available
            if hasattr(self.reranker, 'rerank'):
                # Use the new advanced cross-encoder
                result = self.reranker.rerank(
                    query=query,
                    documents=candidates,
                    top_k=len(candidates)
                )
            else:
                # Fallback to standard CrossEncoder approach
                pairs = []
                for doc in candidates:
                    context = self._create_rerank_context(doc, query_analysis)
                    pairs.append([query, context])
                
                # Get relevance scores from CrossEncoder
                scores = self.reranker.predict(pairs)
                
                # Apply legal-specific reranking factors
                enhanced_scores = self._apply_legal_reranking_factors(scores, candidates, query, query_analysis)
                
                # Sort by enhanced scores
                ranked_candidates = sorted(zip(candidates, enhanced_scores), key=lambda x: x[1], reverse=True)
                result = [doc for doc, score in ranked_candidates]
            
            # Cache result
            self._rerank_cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"Error in advanced reranking: {e}")
            return self._fallback_ranking(candidates, query_analysis, query)
    
    def _create_rerank_context(self, doc: Dict[str, Any], query_analysis: Any) -> str:
        """Create rich context for reranking"""
        context_parts = []
        
        # Title (most important for case identification)
        if doc.get('title'):
            context_parts.append(f"Title: {doc['title']}")
        
        # G.R. number (critical for case matching)
        if doc.get('gr_number'):
            context_parts.append(f"G.R. Number: {doc['gr_number']}")
        
        # Year (important for legal relevance)
        if doc.get('year'):
            context_parts.append(f"Year: {doc['year']}")
        
        # Section (important for intent matching)
        if doc.get('section'):
            context_parts.append(f"Section: {doc['section']}")
        
        # Content snippet (for semantic relevance)
        content = doc.get('content', '')[:500]  # First 500 chars
        if content:
            context_parts.append(f"Content: {content}")
        
        return " | ".join(context_parts)
    
    def _apply_legal_reranking_factors(self, scores: np.ndarray, documents: List[Dict[str, Any]], 
                                     query: str, query_analysis: Any) -> np.ndarray:
        """Apply legal-specific reranking factors"""
        enhanced_scores = scores.copy()
        
        for i, (doc, score) in enumerate(zip(documents, enhanced_scores)):
            # G.R. number exact match bonus
            if query_analysis.entities['gr_numbers'] and doc.get('gr_number'):
                for gr_num in query_analysis.entities['gr_numbers']:
                    if gr_num in doc.get('gr_number', ''):
                        enhanced_scores[i] += 0.5  # Huge bonus for exact G.R. match
                        break
            
            # Case name match bonus
            if query_analysis.entities['case_names'] and doc.get('title'):
                title = doc.get('title', '').lower()
                for case_name in query_analysis.entities['case_names']:
                    if case_name.lower() in title:
                        enhanced_scores[i] += 0.3
                        break
            
            # Intent-specific section bonus
            if query_analysis.intent_flags.get('wants_facts', False) and doc.get('section') == 'facts':
                enhanced_scores[i] += 0.2
            elif query_analysis.intent_flags.get('wants_ruling', False) and doc.get('section') == 'ruling':
                enhanced_scores[i] += 0.2
            elif query_analysis.intent_flags.get('wants_digest', False) and doc.get('section') in ['facts', 'ruling', 'issues']:
                enhanced_scores[i] += 0.1
            
            # Recency bonus (newer cases are more relevant)
            if doc.get('year', 0) >= 2010:
                enhanced_scores[i] += 0.1
            
            # Content quality bonus
            content_length = len(doc.get('content', ''))
            if content_length > 500:
                enhanced_scores[i] += 0.05
            
            # Authority bonus (Supreme Court decisions)
            if 'supreme court' in doc.get('title', '').lower():
                enhanced_scores[i] += 0.1
            
            # Match type bonus
            if doc.get('match_type') == 'gr_number_exact':
                enhanced_scores[i] += 0.4
            elif doc.get('match_type') == 'case_name_match':
                enhanced_scores[i] += 0.2
        
        return enhanced_scores
    
    def _fallback_ranking(self, candidates: List[Dict[str, Any]], query_analysis: Any, query: str) -> List[Dict[str, Any]]:
        """Fallback ranking when CrossEncoder is not available"""
        def calculate_score(doc):
            score = doc.get('score', 0)
            
            # Apply same legal factors as reranker
            if query_analysis.entities['gr_numbers'] and doc.get('gr_number'):
                for gr_num in query_analysis.entities['gr_numbers']:
                    if gr_num in doc.get('gr_number', ''):
                        score += 0.5
                        break
            
            # Intent-specific bonuses
            if query_analysis.intent_flags.get('wants_facts', False) and doc.get('section') == 'facts':
                score += 0.2
            elif query_analysis.intent_flags.get('wants_ruling', False) and doc.get('section') == 'ruling':
                score += 0.2
            
            return score
        
        return sorted(candidates, key=calculate_score, reverse=True)
    
    
    def _final_ranking(self, candidates: List[Dict[str, Any]], query_analysis: Any, k: int) -> List[Dict[str, Any]]:
        """Final ranking and filtering of candidates"""
        # Sort by score
        candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Apply query type specific filtering
        if hasattr(query_analysis, 'query_type') and str(query_analysis.query_type) == 'CASE_DIGEST':
            # For case digests, prioritize comprehensive coverage
            return candidates[:k * 2]  # Return more candidates
        else:
            # For other queries, return top k
            return candidates[:k]
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        total_queries = self._performance_stats['total_queries']
        current_avg = self._performance_stats['avg_processing_time']
        
        # Calculate running average
        new_avg = ((current_avg * (total_queries - 1)) + processing_time) / total_queries
        self._performance_stats['avg_processing_time'] = new_avg
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_hit_rate = 0.0
        if self._performance_stats['total_queries'] > 0:
            cache_hit_rate = self._performance_stats['cache_hits'] / self._performance_stats['total_queries']
        
        return {
            **self._performance_stats,
            'cache_hit_rate': cache_hit_rate,
            'entity_cache_size': len(self._entity_cache),
            'query_analysis_cache_size': len(self._query_analysis_cache),
            'rerank_cache_size': len(self._rerank_cache)
        }
    
    def clear_caches(self):
        """Clear all caches"""
        self._entity_cache.clear()
        self._query_analysis_cache.clear()
        self._rerank_cache.clear()
        self._similarity_cache.clear()
        print("🧹 All caches cleared")
    
