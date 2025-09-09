# retriever.py — Enhanced legal document retriever with hybrid approach and reranking
import gzip
import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

# Try to import optimized Philippine legal reranker
try:
    from .reranker import PhilippineLegalReranker
    OPTIMIZED_RERANKER_AVAILABLE = True
except ImportError:
    OPTIMIZED_RERANKER_AVAILABLE = False
    print("Warning: Optimized reranker not available. Using standard CrossEncoder.")

# Import enhanced components
try:
    from .case_similarity_engine import CaseSimilarityEngine
    from .entity_extractor import PhilippineLegalEntityExtractor
    from .legal_aware_chunker import LegalAwareChunker
    from .query_processor import EnhancedQueryProcessor, QueryType
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False
    print("Warning: Enhanced components not available. Using basic functionality.")

# Import centralized model cache
from .model_cache import (clear_embedding_model_cache,
                          get_cached_embedding_model)

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
    """Enhanced legal document retriever with hybrid approach and reranking"""
    
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
        
        # Initialize enhanced components if available
        if ENHANCED_COMPONENTS_AVAILABLE:
            self.entity_extractor = PhilippineLegalEntityExtractor()
            self.query_processor = EnhancedQueryProcessor()
            self.legal_chunker = LegalAwareChunker()
            self.similarity_engine = CaseSimilarityEngine()
        else:
            self.entity_extractor = None
            self.query_processor = None
            self.legal_chunker = None
            self.similarity_engine = None
        
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
    
    def _init_reranker(self):
        """Initialize the optimized Philippine legal reranker"""
        if OPTIMIZED_RERANKER_AVAILABLE:
            try:
                print("Loading optimized Philippine legal reranker...")
                self.reranker = PhilippineLegalReranker()
                print("Optimized Philippine legal reranker loaded successfully")
            except Exception as e:
                print(f"Failed to load optimized reranker: {e}")
                self.reranker = None
        elif CROSSENCODER_AVAILABLE:
            try:
                print("Loading standard CrossEncoder reranker...")
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("CrossEncoder reranker loaded successfully")
            except Exception as e:
                print(f"Failed to load CrossEncoder: {e}")
                self.reranker = None
        else:
            print("No reranker available - using fallback ranking")
    
    def retrieve(self, query: str, k: int = 8, is_case_digest: bool = False, 
                conversation_history: Optional[List[Dict]] = None, use_hybrid: bool = True) -> List[Dict[str, Any]]:
        """Enhanced hybrid retrieval with all improvements"""
        start_time = time.time()
        self._performance_stats['total_queries'] += 1
        
        print(f"🔍 Enhanced hybrid retrieval for: '{query}'")
        
        if use_hybrid:
            # Use the new hybrid approach: Qdrant + JSONL
            return self._hybrid_enhanced_retrieve(query, k, is_case_digest, conversation_history)
        
        # Step 1: Advanced query analysis
        query_analysis = self._analyze_query(query, conversation_history)
        print(f"📊 Query type: {query_analysis.query_type.value}, Complexity: {query_analysis.complexity.value}")
        
        # Step 2: Enhanced entity extraction
        entities = query_analysis.entities
        print(f"🔍 Extracted entities: {sum(len(v) for v in entities.values())} total")
        
        # Step 3: Multi-strategy retrieval
        all_candidates = self._multi_strategy_retrieve(query, query_analysis, k, is_case_digest)
        print(f"📋 Retrieved {len(all_candidates)} candidates")
        
        # Step 4: Advanced reranking
        if self.reranker and len(all_candidates) > 1:
            ranked_candidates = self._advanced_rerank(query, all_candidates, query_analysis)
        else:
            ranked_candidates = all_candidates
        
        # Step 5: Add similarity recommendations
        if query_analysis.query_type == QueryType.CASE_DIGEST:
            ranked_candidates = self._add_similarity_recommendations(ranked_candidates, k)
        
        # Step 6: Final filtering and ranking
        final_results = self._final_ranking(ranked_candidates, query_analysis, k)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time)
        
        print(f"✅ Enhanced retrieval completed: {len(final_results)} results in {processing_time:.2f}s")
        
        return final_results
    
    def _hybrid_enhanced_retrieve(self, query: str, k: int, is_case_digest: bool, 
                                 conversation_history: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Hybrid enhanced retrieval using Qdrant + JSONL approach"""
        print("🔄 Hybrid enhanced retrieval: Qdrant filtering + JSONL content...")
        
        # Step 1: Advanced query analysis
        query_analysis = self._analyze_query(query, conversation_history)
        print(f"📊 Query type: {query_analysis.query_type.value}, Complexity: {query_analysis.complexity.value}")
        
        # Step 2: Enhanced entity extraction
        entities = query_analysis.entities
        print(f"🔍 Extracted entities: {sum(len(v) for v in entities.values())} total")
        
        # Step 3: Use base retriever's hybrid method
        all_candidates = super().retrieve(query, k=k*2, is_case_digest=is_case_digest, use_hybrid=True)
        print(f"📋 Retrieved {len(all_candidates)} candidates from base retriever")
        
        # Step 4: Advanced reranking if available
        if self.reranker and len(all_candidates) > 1:
            print("🔄 Advanced reranking...")
            ranked_candidates = self._advanced_rerank(query, all_candidates, query_analysis)
        else:
            ranked_candidates = all_candidates
        
        # Step 5: Apply case similarity recommendations
        if len(ranked_candidates) > 0:
            print("🔄 Applying case similarity recommendations...")
            ranked_candidates = self._apply_similarity_recommendations(query, ranked_candidates)
        
        # Step 6: Return top k results
        final_results = ranked_candidates[:k]
        print(f"✅ Hybrid enhanced retrieval: {len(final_results)} final results")
        
        return final_results
    
    def _apply_similarity_recommendations(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply case similarity recommendations to enhance results"""
        if not results or not self.similarity_engine:
            return results
        
        try:
            # Get case IDs from results
            case_ids = []
            for result in results:
                case_id = result.get('metadata', {}).get('case_id') or result.get('metadata', {}).get('gr_number')
                if case_id:
                    case_ids.append(case_id)
            
            if not case_ids:
                return results
            
            # Find similar cases for each result
            enhanced_results = []
            for result in results:
                case_id = result.get('metadata', {}).get('case_id') or result.get('metadata', {}).get('gr_number')
                if case_id:
                    try:
                        similar_cases = self.similarity_engine.find_similar_cases(case_id, top_k=3)
                        result['similar_cases'] = [
                            {
                                'title': sim_case.title,
                                'similarity_score': sim_case.similarity_score,
                                'shared_concepts': sim_case.shared_concepts
                            }
                            for sim_case in similar_cases
                        ]
                    except Exception as e:
                        print(f"⚠️ Similarity search failed for {case_id}: {e}")
                        result['similar_cases'] = []
                else:
                    result['similar_cases'] = []
                
                enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            print(f"⚠️ Similarity recommendations failed: {e}")
            return results
    
    def _analyze_query(self, query: str, conversation_history: Optional[List[Dict]]) -> Any:
        """Analyze query using enhanced query processor"""
        # Check cache first
        cache_key = f"analysis_{hash(query)}_{hash(str(conversation_history))}"
        if cache_key in self._query_analysis_cache:
            self._performance_stats['cache_hits'] += 1
            return self._query_analysis_cache[cache_key]
        
        self._performance_stats['cache_misses'] += 1
        
        # Perform analysis
        analysis = self.query_processor.process_query(query, conversation_history)
        
        # Cache result
        self._query_analysis_cache[cache_key] = analysis
        
        return analysis
    
    def _multi_strategy_retrieve(self, query: str, query_analysis: Any, k: int, is_case_digest: bool) -> List[Dict[str, Any]]:
        """Multi-strategy retrieval based on query analysis"""
        all_candidates = []
        
        # Strategy 1: Exact entity matching (highest priority)
        if query_analysis.entities['gr_numbers']:
            gr_candidates = self._retrieve_by_gr_numbers(query_analysis.entities['gr_numbers'])
            all_candidates.extend(gr_candidates)
            print(f"🎯 G.R. number matches: {len(gr_candidates)}")
        
        if query_analysis.entities['case_names']:
            case_candidates = self._retrieve_by_case_names(query_analysis.entities['case_names'])
            all_candidates.extend(case_candidates)
            print(f"📋 Case name matches: {len(case_candidates)}")
        
        # Strategy 2: Legal term matching
        if query_analysis.entities['legal_terms']:
            term_candidates = self._retrieve_by_legal_terms(query_analysis.entities['legal_terms'])
            all_candidates.extend(term_candidates)
            print(f"⚖️ Legal term matches: {len(term_candidates)}")
        
        # Strategy 3: Semantic vector search
        semantic_candidates = self._semantic_search(query, k * 2, is_case_digest)
        all_candidates.extend(semantic_candidates)
        print(f"🔍 Semantic matches: {len(semantic_candidates)}")
        
        # Strategy 4: Query reformulation search
        for reformulated_query in query_analysis.reformulated_queries[:3]:  # Limit to top 3
            if reformulated_query != query:
                reformulated_candidates = self._semantic_search(reformulated_query, k, is_case_digest)
                all_candidates.extend(reformulated_candidates)
                print(f"🔄 Reformulated query matches: {len(reformulated_candidates)}")
        
        # Strategy 5: Contextual search (if conversation history available)
        if query_analysis.context_requirements and 'conversation_context' in query_analysis.context_requirements:
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
            # Use the parent class semantic search
            results = super().retrieve(query, k, is_case_digest)
            # Add match type to results
            for result in results:
                result['match_type'] = 'semantic_search'
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
        return {
            'title': hit.payload.get('title', ''),
            'content': hit.payload.get('content', ''),
            'gr_number': hit.payload.get('gr_number', ''),
            'year': hit.payload.get('year', ''),
            'section': hit.payload.get('section', ''),
            'score': hit.score,
            'url': hit.payload.get('source_url', ''),
            'match_type': match_type
        }
    
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
            # Use optimized reranker if available
            if OPTIMIZED_RERANKER_AVAILABLE and hasattr(self.reranker, 'rerank'):
                base_scores = [candidate.get('score', 0.5) for candidate in candidates]
                ranked_results = self.reranker.rerank(query, candidates, base_scores)
                result = [doc for doc, score in ranked_results]
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
    
    def _add_similarity_recommendations(self, candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Add similarity recommendations for case digest queries"""
        if not candidates:
            return candidates
        
        # Get the top candidate for similarity search
        top_candidate = candidates[0]
        
        try:
            # Find similar cases
            similar_cases = self.similarity_engine.find_similar_cases_by_query(
                f"{top_candidate.get('title', '')} {top_candidate.get('content', '')[:200]}",
                top_k=3
            )
            
            # Add similar cases as recommendations
            for similar_case in similar_cases:
                recommendation = {
                    'title': similar_case.title,
                    'content': f"Similar case: {similar_case.title}",
                    'gr_number': similar_case.gr_number,
                    'year': similar_case.year,
                    'score': similar_case.similarity_score * 0.8,  # Lower score for recommendations
                    'match_type': 'similarity_recommendation',
                    'similarity_type': similar_case.similarity_type,
                    'shared_concepts': similar_case.shared_concepts
                }
                candidates.append(recommendation)
            
        except Exception as e:
            print(f"Error adding similarity recommendations: {e}")
        
        return candidates
    
    def _final_ranking(self, candidates: List[Dict[str, Any]], query_analysis: Any, k: int) -> List[Dict[str, Any]]:
        """Final ranking and filtering of candidates"""
        # Sort by score
        candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Apply query type specific filtering
        if query_analysis.query_type == QueryType.CASE_DIGEST:
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
    
    def add_case_to_similarity_engine(self, case_data: Dict[str, Any]) -> str:
        """Add a case to the similarity engine"""
        return self.similarity_engine.add_case(case_data)
    
    def get_similarity_engine_stats(self) -> Dict[str, Any]:
        """Get similarity engine statistics"""
        return self.similarity_engine.get_engine_statistics()
