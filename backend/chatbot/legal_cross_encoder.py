# legal_cross_encoder.py — Advanced BERT-based cross-encoder reranker for legal documents
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import \
    CrossEncoder as SentenceCrossEncoder


class LegalCrossEncoderReranker:
    """
    Advanced BERT-based cross-encoder reranker optimized for legal document retrieval.
    Supports multiple models and legal domain-specific optimizations.
    """
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 legal_model_name: Optional[str] = None,
                 device: Optional[str] = None,
                 max_length: int = 512,
                 batch_size: int = 32,
                 use_legal_optimization: bool = True):
        """
        Initialize the legal cross-encoder reranker.
        
        Args:
            model_name: Base cross-encoder model name
            legal_model_name: Legal domain-specific model (optional)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length for input
            batch_size: Batch size for inference
            use_legal_optimization: Whether to apply legal-specific optimizations
        """
        self.model_name = model_name
        self.legal_model_name = legal_model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_legal_optimization = use_legal_optimization
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize models (legal model only)
        self.base_model = None
        self.legal_model = None
        self._init_models()
        
        # Performance tracking
        self.stats = {
            'total_reranks': 0,
            'total_pairs': 0,
            'avg_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Caching for performance
        self._score_cache = {}
        self._cache_max_size = 10000
    
    def _init_models(self):
        """Initialize only the legal cross-encoder model"""
        if self.legal_model_name:
            try:
                print(f"🔄 Loading legal cross-encoder: {self.legal_model_name}")
                self.legal_model = CrossEncoder(self.legal_model_name, device=self.device)
                print(f"✅ Legal cross-encoder loaded on {self.device}")
            except Exception as e:
                print(f"❌ Failed to load legal cross-encoder {self.legal_model_name}: {e}")
                self.legal_model = None
        else:
            print("⚠️ No legal cross-encoder name provided; reranking disabled")
    
    def rerank(self, 
               query: str, 
               documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None,
               use_legal_model: bool = True) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance using cross-encoder.
        
        Args:
            query: Search query
            documents: List of document dictionaries to rerank
            top_k: Number of top results to return (None for all)
            use_legal_model: Whether to use legal domain model if available
            
        Returns:
            List of reranked documents with cross-encoder scores
        """
        if not self.legal_model or not documents:
            return documents
        
        start_time = time.time()
        self.stats['total_reranks'] += 1
        
        print(f"🔄 Cross-encoder reranking {len(documents)} documents...")
        
        # Prepare query-document pairs
        pairs = self._prepare_pairs(query, documents)
        if not pairs:
            return documents
        
        # Get cross-encoder scores
        scores = self._get_cross_encoder_scores(pairs, use_legal_model)
        
        # Apply legal domain optimizations
        if self.use_legal_optimization:
            scores = self._apply_legal_optimizations(scores, query, documents)
        
        # Update documents with scores
        reranked_docs = self._update_documents_with_scores(documents, scores)
        
        # Sort by cross-encoder score
        reranked_docs.sort(key=lambda x: x.get('cross_encoder_score', 0), reverse=True)
        
        # Return top_k results
        if top_k:
            reranked_docs = reranked_docs[:top_k]
        
        # Update stats
        processing_time = time.time() - start_time
        self.stats['total_pairs'] += len(pairs)
        self.stats['avg_processing_time'] = (
            (self.stats['avg_processing_time'] * (self.stats['total_reranks'] - 1) + processing_time) 
            / self.stats['total_reranks']
        )
        
        print(f"✅ Cross-encoder reranking completed in {processing_time:.2f}s")
        return reranked_docs
    
    def _prepare_pairs(self, query: str, documents: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Prepare query-document pairs for cross-encoder input"""
        pairs = []
        
        for doc in documents:
            # Extract relevant text content
            content = self._extract_document_content(doc)
            if content:
                pairs.append((query, content))
            else:
                # Fallback to empty content
                pairs.append((query, ""))
        
        return pairs
    
    def _extract_document_content(self, doc: Dict[str, Any]) -> str:
        """Extract and format document content for cross-encoder"""
        content_parts = []
        
        # Primary content
        if 'content' in doc and doc['content']:
            content_parts.append(doc['content'])
        
        # Metadata fields that might be relevant
        metadata = doc.get('metadata', {})
        for field in ['title', 'gr_number', 'section']:
            if field in metadata and metadata[field]:
                content_parts.append(f"{field}: {metadata[field]}")
        
        # Combine and truncate if necessary
        full_content = " | ".join(content_parts)
        
        # Truncate to max_length if needed
        if len(full_content) > self.max_length * 4:  # Rough character to token ratio
            full_content = full_content[:self.max_length * 4] + "..."
        
        return full_content
    
    def _get_cross_encoder_scores(self, 
                                 pairs: List[Tuple[str, str]], 
                                 use_legal_model: bool = True) -> np.ndarray:
        """Get cross-encoder scores for query-document pairs"""
        
        # Choose model - legal model only
        if use_legal_model and self.legal_model:
            model = self.legal_model
            print("🎯 Using legal cross-encoder for reranking")
        else:
            model = None
            print("⚠️ No cross-encoder model available")
        
        try:
            # Process in batches for memory efficiency
            all_scores = []
            
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i + self.batch_size]
                
                # Get scores for this batch
                batch_scores = model.predict(
                    batch_pairs,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                all_scores.extend(batch_scores)
            
            return np.array(all_scores)
            
        except Exception as e:
            print(f"❌ Error in cross-encoder scoring: {e}")
            # Return neutral scores as fallback
            return np.full(len(pairs), 0.5)
    
    def _apply_legal_optimizations(self, 
                                  scores: np.ndarray, 
                                  query: str, 
                                  documents: List[Dict[str, Any]]) -> np.ndarray:
        """Apply legal domain-specific optimizations to scores"""
        enhanced_scores = scores.copy()
        
        # Legal section importance weighting
        section_weights = {
            'ruling': 1.3,
            'issues': 1.2,
            'facts': 1.1,
            'arguments': 1.1,
            'header': 1.0,
            'body': 0.9,
            'caption': 0.8
        }
        
        for i, doc in enumerate(documents):
            if i < len(enhanced_scores):
                # Apply section weight
                section = (doc.get('metadata', {}).get('section') or 'body')
                weight = section_weights.get(section, 1.0)
                enhanced_scores[i] *= weight
                
                # Boost for exact G.R. number matches
                if 'gr_number' in query.lower():
                    gr_number = (doc.get('metadata', {}).get('gr_number') or '')
                    if gr_number and gr_number.lower() in query.lower():
                        enhanced_scores[i] *= 1.2
                
                # Boost for recent cases (if year available)
                year = doc.get('metadata', {}).get('year')
                if year and isinstance(year, int) and year >= 2010:
                    enhanced_scores[i] *= 1.05
                
                # Boost for En Banc cases
                division = (doc.get('metadata', {}).get('division') or '')
                if isinstance(division, str) and 'en banc' in division.lower():
                    enhanced_scores[i] *= 1.1
        
        return enhanced_scores
    
    def _update_documents_with_scores(self, 
                                    documents: List[Dict[str, Any]], 
                                    scores: np.ndarray) -> List[Dict[str, Any]]:
        """Update documents with cross-encoder scores"""
        updated_docs = []
        
        for i, doc in enumerate(documents):
            updated_doc = doc.copy()
            if i < len(scores):
                updated_doc['cross_encoder_score'] = float(scores[i])
                # Keep original score for comparison
                if 'score' not in updated_doc:
                    updated_doc['score'] = 0.0
                updated_doc['original_score'] = updated_doc['score']
            else:
                updated_doc['cross_encoder_score'] = 0.0
            
            updated_docs.append(updated_doc)
        
        return updated_docs
    
    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Predict relevance scores for query-document pairs (compatibility method)"""
        if not self.legal_model:
            return [0.5] * len(pairs)
        
        try:
            scores = self.legal_model.predict(pairs, convert_to_numpy=True)
            return scores.tolist()
        except Exception as e:
            print(f"❌ Error in predict method: {e}")
            return [0.5] * len(pairs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            'base_model_loaded': False,
            'legal_model_loaded': self.legal_model is not None,
            'device': self.device,
            'model_name': self.model_name,
            'legal_model_name': self.legal_model_name
        }
    
    def clear_cache(self):
        """Clear the score cache"""
        self._score_cache.clear()
        print("🧹 Cross-encoder cache cleared")
    
    def reload_models(self):
        """Reload the cross-encoder models"""
        print("🔄 Reloading cross-encoder models...")
        self._init_models()


class LegalDomainCrossEncoder(LegalCrossEncoderReranker):
    """
    Specialized cross-encoder for legal domain with additional optimizations.
    """
    
    def __init__(self, **kwargs):
        # Use legal-optimized model by default and avoid passing duplicate kwargs
        legal_model = kwargs.pop('legal_model_name', 'cross-encoder/ms-marco-MiniLM-L-12-v2')
        super().__init__(legal_model_name=legal_model, **kwargs)
    
    def _apply_legal_optimizations(self, 
                                  scores: np.ndarray, 
                                  query: str, 
                                  documents: List[Dict[str, Any]]) -> np.ndarray:
        """Enhanced legal optimizations for domain-specific reranking"""
        enhanced_scores = super()._apply_legal_optimizations(scores, query, documents)
        
        # Additional legal-specific optimizations
        legal_terms = [
            'constitution', 'statute', 'regulation', 'precedent', 'jurisdiction',
            'liability', 'negligence', 'contract', 'tort', 'criminal',
            'due process', 'equal protection', 'freedom', 'rights'
        ]
        
        query_lower = query.lower()
        
        for i, doc in enumerate(documents):
            if i < len(enhanced_scores):
                content = self._extract_document_content(doc).lower()
                
                # Boost for legal term matches
                legal_term_matches = sum(1 for term in legal_terms if term in content and term in query_lower)
                if legal_term_matches > 0:
                    enhanced_scores[i] *= (1 + legal_term_matches * 0.05)
                
                # Boost for case law citations
                if 'v.' in content or 'vs.' in content or 'versus' in content:
                    enhanced_scores[i] *= 1.05
                
                # Boost for legal procedural terms
                procedural_terms = ['motion', 'appeal', 'petition', 'writ', 'injunction']
                if any(term in content for term in procedural_terms):
                    enhanced_scores[i] *= 1.03
        
        return enhanced_scores


# Factory function for easy instantiation
def create_legal_cross_encoder(model_type: str = "base", **kwargs) -> LegalCrossEncoderReranker:
    """
    Factory function to create appropriate cross-encoder reranker.
    
    Args:
        model_type: Type of model ('base', 'legal', 'ms-marco', 'custom')
        **kwargs: Additional arguments passed to the constructor
    
    Returns:
        Configured cross-encoder reranker instance
    """
    model_configs = {
        'base': {
            'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'use_legal_optimization': True
        },
        'legal': {
            'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'legal_model_name': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
            'use_legal_optimization': True
        },
        'ms-marco': {
            'model_name': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
            'use_legal_optimization': True
        },
        'custom': kwargs
    }
    
    config = model_configs.get(model_type, model_configs['base'])
    config.update(kwargs)
    
    if model_type == 'legal':
        return LegalDomainCrossEncoder(**config)
    else:
        return LegalCrossEncoderReranker(**config)
