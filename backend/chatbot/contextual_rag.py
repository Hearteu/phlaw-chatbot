# optimized_contextual_rag.py — Performance-optimized Contextual RAG implementation
import asyncio
import concurrent.futures
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchValue,
                                  PointStruct, VectorParams)
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .chunker import LegalDocumentChunker
from .model_cache import get_cached_embedding_model, get_cached_llm


class ContextualRAG:
    """Performance-optimized Contextual RAG with caching and parallel processing"""
    
    def __init__(self, 
                 collection: str = "jurisprudence",
                 chunk_size: int = 640,
                 overlap_ratio: float = 0.15,
                 cache_dir: str = "backend/data/contextual_cache"):
        self.collection = collection
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize components
        self.chunker = LegalDocumentChunker(
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio
        )
        
        # Get cached models
        self.embedding_model = get_cached_embedding_model()
        self.qdrant = self._get_qdrant_client()
        
        # BM25 index for keyword search
        self.bm25_index = None
        
        # ID-based storage instead of list-based (critical fix for correctness)
        self.id_to_contextual_chunk = {}  # id -> contextual_chunk_text
        self.id_to_metadata = {}          # id -> metadata
        self.id_to_payload = {}           # id -> full_payload (for Qdrant compatibility)
        self.id_to_embedding = {}         # id -> embedding (optional, for reranking)
        
        # Legacy lists for backward compatibility (will be populated from maps)
        self.contextual_chunks = []
        self.chunk_metadata = []
        
        
        # Cache for contextual chunks
        self.contextual_cache = {}
        self.cache_file = os.path.join(cache_dir, "contextual_chunks.json")
        
        # Load existing cache
        self._load_contextual_cache()
        
        # Try to load existing indexes
        self._load_existing_indexes()
        
        # Optimized context generation prompt (shorter, more focused)
        self.CONTEXTUAL_RAG_PROMPT = """Briefly explain what this legal document section discusses (1 sentence):

Document: {DOCUMENT_EXCERPT}
Section: {CHUNK_CONTENT}

Explanation:"""
    
    def _get_qdrant_client(self) -> QdrantClient:
        """Get Qdrant client with optimized settings"""
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        return QdrantClient(
            host=host, 
            port=port, 
            grpc_port=6334, 
            prefer_grpc=True, 
            timeout=10.0  # Reduced timeout
        )
    
    def _load_contextual_cache(self) -> None:
        """Load contextual chunk cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.contextual_cache = json.load(f)
                print(f"✅ Loaded {len(self.contextual_cache)} cached contextual chunks")
            else:
                self.contextual_cache = {}
                print("📝 No existing contextual cache found")
        except Exception as e:
            print(f"⚠️ Failed to load contextual cache: {e}")
            self.contextual_cache = {}
    
    def _save_contextual_cache(self) -> None:
        """Save contextual chunk cache to disk"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.contextual_cache, f, ensure_ascii=False, indent=2)
            print(f"💾 Saved {len(self.contextual_cache)} contextual chunks to cache")
        except Exception as e:
            print(f"⚠️ Failed to save contextual cache: {e}")
    
    def _get_chunk_hash(self, chunk_content: str, document_context: str) -> str:
        """Generate hash for chunk to enable caching"""
        content = f"{chunk_content}|{document_context}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _load_existing_indexes(self) -> None:
        """Try to load existing contextual chunks from Qdrant collection"""
        try:
            # Use the collection name directly (no "_contextual" suffix)
            contextual_collection = self.collection
            if self.qdrant.collection_exists(contextual_collection):
                collection_info = self.qdrant.get_collection(contextual_collection)
                if collection_info.points_count > 0:
                    print(f"Found existing contextual collection with {collection_info.points_count} points")
                    self._load_contextual_chunks_from_qdrant(contextual_collection)
                else:
                    print(f"⚠️ Contextual collection exists but is empty")
            else:
                print(f"⚠️ No existing contextual collection found")
        except Exception as e:
            print(f"Could not check for existing indexes: {e}")
    
    def _load_contextual_chunks_from_qdrant(self, collection_name: str) -> None:
        """Load contextual chunks from Qdrant collection using ID-based storage"""
        try:
            print("🔄 Loading contextual chunks from Qdrant...")
            
            # Get all points from the collection
            points = self.qdrant.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True
            )[0]
            
            if not points:
                print("⚠️ No points found in contextual collection")
                return
            
            # Clear existing maps
            self.id_to_contextual_chunk.clear()
            self.id_to_metadata.clear()
            self.id_to_payload.clear()
            
            # Load chunks using stable IDs
            for point in points:
                payload = point.payload
                if not payload:
                    continue
                
                chunk_id = str(point.id)  # Use Qdrant's ID as the stable ID
                contextual_text = payload.get('content', '')
                
                # Reconstruct metadata from payload
                metadata = {
                    'content': payload.get('original_content', ''),
                    'section': payload.get('section', ''),
                    'section_type': payload.get('section_type', 'general'),
                    'chunk_type': payload.get('chunk_type', 'content'),
                    'token_count': payload.get('token_count', 0),
                    'metadata': {
                        'gr_number': payload.get('gr_number', ''),
                        'special_number': payload.get('special_number', ''),
                        'title': payload.get('title', ''),
                        'ponente': payload.get('ponente', ''),
                        'case_type': payload.get('case_type', ''),
                        'date': payload.get('date', '')
                    }
                }
                
                # Store in ID-based maps
                self.id_to_contextual_chunk[chunk_id] = contextual_text
                self.id_to_metadata[chunk_id] = metadata
                self.id_to_payload[chunk_id] = payload
            
            # Legacy lists will be populated as needed
            
            print(f"Loaded {len(self.id_to_contextual_chunk)} contextual chunks from Qdrant")
            
        except Exception as e:
            print(f"Failed to load contextual chunks from Qdrant: {e}")
            self.id_to_contextual_chunk.clear()
            self.id_to_metadata.clear()
            self.id_to_payload.clear()
            # Legacy lists will be populated as needed
    
    def generate_contextual_chunks_fast(self, document: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """Fast contextual chunk generation with caching and rule-based fallbacks"""
        print(f"🔄 Generating contextual chunks (fast) for {len(chunks)} chunks...")
        
        contextual_chunks = []
        
        # Use rule-based context generation for most chunks
        for i, chunk in enumerate(chunks):
            chunk_content = chunk.get('content', '')
            section_type = chunk.get('section_type', 'general')
            
            # Try cache first
            chunk_hash = self._get_chunk_hash(chunk_content, document[:500])
            if chunk_hash in self.contextual_cache:
                contextual_chunk = self.contextual_cache[chunk_hash]
                contextual_chunks.append(contextual_chunk)
                continue
            
            # Generate rule-based context (fast)
            if section_type == 'dispositive':
                context = "This section contains the court's final ruling and decision."
            elif section_type == 'factual':
                context = "This section describes the facts and circumstances of the case."
            elif section_type == 'issues':
                context = "This section identifies the legal issues to be resolved."
            elif section_type == 'legal_analysis':
                context = "This section provides the legal analysis and reasoning."
            else:
                context = "This section contains legal content from the case."
            
            contextual_chunk = f"{context} {chunk_content}"
            contextual_chunks.append(contextual_chunk)
            
            # Cache the result
            self.contextual_cache[chunk_hash] = contextual_chunk
        
        return contextual_chunks
    
    def generate_contextual_chunks_with_llm_parallel(self, document: str, chunks: List[Dict[str, Any]], 
                                                   max_llm_chunks: int = 5) -> List[str]:
        """Generate contextual chunks with limited LLM usage and parallel processing"""
        print(f"🔄 Generating contextual chunks (parallel LLM) for {len(chunks)} chunks...")
        
        contextual_chunks = []
        
        # Select only the most important chunks for LLM processing
        important_chunks = []
        for chunk in chunks:
            section_type = chunk.get('section_type', 'general')
            if section_type in ['dispositive', 'issues', 'legal_analysis']:
                important_chunks.append(chunk)
        
        # Limit to max_llm_chunks to control costs and speed
        important_chunks = important_chunks[:max_llm_chunks]
        
        # Generate rule-based context for non-important chunks
        for chunk in chunks:
            chunk_content = chunk.get('content', '')
            section_type = chunk.get('section_type', 'general')
            
            if chunk not in important_chunks:
                # Use rule-based context for non-important chunks
                if section_type == 'dispositive':
                    context = "This section contains the court's final ruling and decision."
                elif section_type == 'factual':
                    context = "This section describes the facts and circumstances of the case."
                elif section_type == 'issues':
                    context = "This section identifies the legal issues to be resolved."
                elif section_type == 'legal_analysis':
                    context = "This section provides the legal analysis and reasoning."
                else:
                    context = "This section contains legal content from the case."
                
                contextual_chunk = f"{context} {chunk_content}"
                contextual_chunks.append(contextual_chunk)
        
        # Generate LLM context for important chunks in parallel
        if important_chunks:
            print(f"🤖 Generating LLM context for {len(important_chunks)} important chunks...")
            
            # Prepare document excerpt (much smaller)
            doc_excerpt = document[:1000] + "..." if len(document) > 1000 else document
            
            # Process important chunks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_chunk = {}
                
                for chunk in important_chunks:
                    chunk_content = chunk.get('content', '')
                    
                    # Check cache first
                    chunk_hash = self._get_chunk_hash(chunk_content, doc_excerpt)
                    if chunk_hash in self.contextual_cache:
                        contextual_chunk = self.contextual_cache[chunk_hash]
                        contextual_chunks.append(contextual_chunk)
                        continue
                    
                    # Create prompt for LLM
                    prompt = self.CONTEXTUAL_RAG_PROMPT.format(
                        DOCUMENT_EXCERPT=doc_excerpt,
                        CHUNK_CONTENT=chunk_content
                    )
                    
                    # Submit to thread pool
                    future = executor.submit(self._generate_context_for_chunk, prompt, chunk_content, chunk_hash)
                    future_to_chunk[future] = (chunk, chunk_hash)
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk, chunk_hash = future_to_chunk[future]
                    try:
                        contextual_chunk = future.result()
                        contextual_chunks.append(contextual_chunk)
                        
                        # Cache the result
                        self.contextual_cache[chunk_hash] = contextual_chunk
                    except Exception as e:
                        print(f"⚠️ LLM generation failed for chunk: {e}")
                        # Fallback to rule-based
                        section_type = chunk.get('section_type', 'general')
                        if section_type == 'dispositive':
                            context = "This section contains the court's final ruling and decision."
                        elif section_type == 'factual':
                            context = "This section describes the facts and circumstances of the case."
                        elif section_type == 'issues':
                            context = "This section identifies the legal issues to be resolved."
                        elif section_type == 'legal_analysis':
                            context = "This section provides the legal analysis and reasoning."
                        else:
                            context = "This section contains legal content from the case."
                        
                        contextual_chunk = f"{context} {chunk.get('content', '')}"
                        contextual_chunks.append(contextual_chunk)
        
        return contextual_chunks
    
    def _generate_context_for_chunk(self, prompt: str, chunk_content: str, chunk_hash: str) -> str:
        """Generate context for a single chunk using local GGUF LLM"""
        try:
            llm = get_cached_llm()
            if not llm:
                print(f"⚠️ Local LLM not available, using rule-based context")
                return chunk_content
            
            # Generate context using local LLM
            result = llm(
                prompt,
                max_tokens=50,
                temperature=0.1,
                top_p=0.9,
                stop=["User:", "Human:", "\n\n"]
            )
            
            # Extract text from result
            if isinstance(result, dict):
                context = result.get('choices', [{}])[0].get('text', '')
            else:
                context = str(result)
            
            # Clean up context
            context = context.strip()
            if context.endswith('.'):
                context = context[:-1]
            
            # Create contextual chunk
            if context and len(context) > 5:
                return f"{context}. {chunk_content}"
            else:
                # Fallback to rule-based
                return chunk_content
                
        except Exception as e:
            print(f"⚠️ Local LLM generation failed: {e}")
            return chunk_content
    
    def build_hybrid_indexes_fast(self, cases: List[Dict[str, Any]]) -> None:
        """Build hybrid indexes with optimized contextual chunk generation"""
        print(f"🔄 Building optimized hybrid indexes for {len(cases)} cases...")
        
        all_chunks = []
        all_contextual_chunks = []
        
        for case in cases:
            # Chunk the case
            chunks = self.chunker.chunk_case(case)
            all_chunks.extend(chunks)
            
            # Generate contextual chunks using fast method
            clean_text = case.get('clean_text', '') or case.get('body', '')
            if clean_text:
                # Use fast method for most chunks, LLM for important ones
                contextual_chunks = self.generate_contextual_chunks_with_llm_parallel(
                    clean_text, chunks, max_llm_chunks=3
                )
                all_contextual_chunks.extend(contextual_chunks)
            else:
                # Fallback to original content
                contextual_chunks = [chunk.get('content', '') for chunk in chunks]
                all_contextual_chunks.extend(contextual_chunks)
        
        # Store chunks and metadata using ID-based approach
        # Clear existing ID-based maps
        self.id_to_contextual_chunk.clear()
        self.id_to_metadata.clear()
        self.id_to_payload.clear()
        
        # Store in ID-based maps (will be populated during embedding generation)
        # For now, just store the lists for backward compatibility
        self.contextual_chunks = all_contextual_chunks
        self.chunk_metadata = all_chunks
        
        # Build BM25 index
        print("🔄 Building BM25 index...")
        tokenized_chunks = [self._tokenize(chunk) for chunk in all_contextual_chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)
        
        # Build vector index (embeddings)
        print("🔄 Building vector embeddings...")
        self._build_vector_embeddings(all_contextual_chunks, all_chunks)
        
        # Save cache
        self._save_contextual_cache()
        
        print(f"✅ Built optimized hybrid indexes: {len(all_contextual_chunks)} contextual chunks")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [word for word in text.split() if len(word) > 1]
    
    def _build_vector_embeddings(self, contextual_chunks: List[str], chunk_metadata: List[Dict]) -> None:
        """Build vector embeddings with batch processing using stable IDs"""
        print(f"🔄 Generating embeddings for {len(contextual_chunks)} chunks...")
        
        # Generate embeddings in larger batches for better performance
        batch_size = 64  # Increased batch size
        embeddings = []
        chunk_ids = []
        
        for i in range(0, len(contextual_chunks), batch_size):
            batch = contextual_chunks[i:i + batch_size]
            batch_metadata = chunk_metadata[i:i + batch_size]
            
            batch_embeddings = self.embedding_model.encode(
                batch, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                show_progress_bar=False  # Disable progress bar for cleaner output
            )
            
            # Generate stable IDs for this batch
            for j, (chunk_text, metadata) in enumerate(zip(batch, batch_metadata)):
                stable_id = self._generate_stable_id(metadata, chunk_text)
                chunk_ids.append(stable_id)
                
                # Store in ID-based maps
                self.id_to_contextual_chunk[stable_id] = chunk_text
                self.id_to_metadata[stable_id] = metadata
                self.id_to_embedding[stable_id] = batch_embeddings[j]
            
            embeddings.extend(batch_embeddings)
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"  Processed {i + len(batch)}/{len(contextual_chunks)} chunks")
        
        # Store embeddings in Qdrant with stable IDs
        self._store_embeddings_in_qdrant_with_stable_ids(embeddings, chunk_ids)
    
    def _store_embeddings_in_qdrant_with_stable_ids(self, embeddings: List[np.ndarray], chunk_ids: List[str]) -> None:
        """Store embeddings in Qdrant using stable IDs"""
        print("🔄 Storing embeddings in Qdrant with stable IDs...")
        
        # Prepare points for Qdrant using stable IDs
        points = []
        for embedding, chunk_id in zip(embeddings, chunk_ids):
            if chunk_id not in self.id_to_metadata:
                continue
                
            metadata = self.id_to_metadata[chunk_id]
            contextual_text = self.id_to_contextual_chunk[chunk_id]
            
            point = PointStruct(
                id=chunk_id,  # Use stable ID instead of loop index
                vector=embedding.tolist(),
                payload={
                    "content": contextual_text,
                    "original_content": metadata.get('content', ''),
                    "section": metadata.get('section', ''),
                    "section_type": metadata.get('section_type', 'general'),
                    "gr_number": metadata.get('metadata', {}).get('gr_number', ''),
                    "special_number": metadata.get('metadata', {}).get('special_number', ''),
                    "title": metadata.get('metadata', {}).get('title', ''),
                    "ponente": metadata.get('metadata', {}).get('ponente', ''),
                    "case_type": metadata.get('metadata', {}).get('case_type', ''),
                    "date": metadata.get('metadata', {}).get('date', ''),
                    "chunk_type": metadata.get('chunk_type', 'content'),
                    "token_count": metadata.get('token_count', 0),
                    "contextual": True,
                    "stable_id": chunk_id  # Store the stable ID for verification
                }
            )
            points.append(point)
        
        # Create collection and upsert with optimized settings
        try:
            # Use the collection name directly (no "_contextual" suffix)
            collection_name = self.collection
            
            if not self.qdrant.collection_exists(collection_name):
                print(f"🔄 Creating collection: {collection_name}")
                vector_size = len(embeddings[0]) if embeddings else 768
                
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created collection: {collection_name}")
            
            # Upsert in smaller batches for better performance
            batch_size = 1000
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i + batch_size]
                self.qdrant.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )
            
            print(f"✅ Stored {len(points)} embeddings in Qdrant with stable IDs")
        except Exception as e:
            print(f"❌ Error storing embeddings: {e}")
    
    def _store_embeddings_in_qdrant(self, embeddings: List[np.ndarray], 
                                   contextual_chunks: List[str], 
                                   chunk_metadata: List[Dict]) -> None:
        """Store embeddings in Qdrant with optimized settings"""
        print("🔄 Storing embeddings in Qdrant...")
        
        # Prepare points for Qdrant
        points = []
        for i, (embedding, chunk_text, metadata) in enumerate(zip(embeddings, contextual_chunks, chunk_metadata)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "content": chunk_text,
                    "original_content": metadata.get('content', ''),
                    "section": metadata.get('section', ''),
                    "section_type": metadata.get('section_type', 'general'),
                    "gr_number": metadata.get('metadata', {}).get('gr_number', ''),
                    "special_number": metadata.get('metadata', {}).get('special_number', ''),
                    "title": metadata.get('metadata', {}).get('title', ''),
                    "ponente": metadata.get('metadata', {}).get('ponente', ''),
                    "case_type": metadata.get('metadata', {}).get('case_type', ''),
                    "date": metadata.get('metadata', {}).get('date', ''),
                    "chunk_type": metadata.get('chunk_type', 'content'),
                    "token_count": metadata.get('token_count', 0),
                    "contextual": True
                }
            )
            points.append(point)
        
        # Create collection and upsert with optimized settings
        try:
            # Use the collection name directly (no "_contextual" suffix)
            collection_name = self.collection
            
            if not self.qdrant.collection_exists(collection_name):
                print(f"🔄 Creating collection: {collection_name}")
                vector_size = len(embeddings[0]) if embeddings else 768
                
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created collection: {collection_name}")
            
            # Upsert in smaller batches for better performance
            batch_size = 1000
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i + batch_size]
                self.qdrant.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )
            
            print(f"✅ Stored {len(points)} embeddings in Qdrant")
        except Exception as e:
            print(f"❌ Error storing embeddings: {e}")
    
    def vector_retrieval(self, query: str, top_k: int = 50) -> List[str]:
        """Optimized vector search with reduced top_k"""
        try:
            # Encode query
            query_vector = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            
            # Search Qdrant with reduced limit
            # Use the collection name directly (no "_contextual" suffix)
            search_collection = self.collection
            
            results = self.qdrant.search(
                collection_name=search_collection,
                query_vector=query_vector.tolist(),
                limit=top_k,
                with_payload=True
            )
            
            # Extract chunk IDs (not indices!)
            chunk_ids = [str(hit.id) for hit in results]
            return chunk_ids
            
        except Exception as e:
            print(f"❌ Vector retrieval failed: {e}")
            return []
    
    def bm25_retrieval(self, query: str, k: int = 50) -> List[str]:
        """BM25 search returning chunk IDs"""
        if not self.bm25_index:
            print("⚠️ BM25 index not built")
            return []
        
        try:
            query_tokens = self._tokenize(query)
            scores = self.bm25_index.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:k]
            
            # Convert indices to chunk IDs
            chunk_ids = list(self.id_to_contextual_chunk.keys())
            return [chunk_ids[i] for i in top_indices if i < len(chunk_ids)]
            
        except Exception as e:
            print(f"❌ BM25 retrieval failed: {e}")
            return []
    
    def reciprocal_rank_fusion(self, *ranked_lists: List[str], k: int = 60) -> Tuple[List[Tuple[str, float]], List[str]]:
        """Optimized RRF with reduced k value"""
        rrf_map = defaultdict(float)
        
        for rank_list in ranked_lists:
            for rank, item in enumerate(rank_list, 1):
                rrf_map[item] += 1 / (rank + k)
        
        sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
        return sorted_items, [item for item, score in sorted_items]
    
    def retrieve_and_rank_fast(self, query: str, 
                              vector_k: int = 50, 
                              bm25_k: int = 50, 
                              final_k: int = 15) -> List[Dict[str, Any]]:
        """Optimized retrieval pipeline with reduced parameters"""
        print(f"🔍 Fast Contextual RAG retrieval for: '{query}'")
        
        # Step 1: Perform vector and BM25 retrieval with reduced limits
        vector_results = self.vector_retrieval(query, top_k=vector_k)
        bm25_results = self.bm25_retrieval(query, k=bm25_k)
        
        print(f"📊 Vector results: {len(vector_results)}, BM25 results: {len(bm25_results)}")
        
        # Step 2: Combine using RRF
        if vector_results and bm25_results:
            _, hybrid_results = self.reciprocal_rank_fusion(vector_results, bm25_results)
        elif vector_results:
            hybrid_results = vector_results
        elif bm25_results:
            hybrid_results = bm25_results
        else:
            return []
        
        print(f"🔄 Hybrid results: {len(hybrid_results)} chunks")
        
        # Step 3: Return top results directly (skip expensive reranking)
        final_results = hybrid_results[:final_k]
        print(f"✅ Final results: {len(final_results)} chunks")
        
        # Step 4: Return formatted results using ID-based lookup (CRITICAL FIX)
        results = []
        for chunk_id in final_results:
            # Use ID-based lookup instead of index arithmetic
            if chunk_id in self.id_to_contextual_chunk and chunk_id in self.id_to_metadata:
                contextual_text = self.id_to_contextual_chunk[chunk_id]
                metadata = self.id_to_metadata[chunk_id]
                
                result = {
                    'content': contextual_text,
                    'original_content': metadata.get('content', ''),
                    'metadata': metadata.get('metadata', {}),
                    'section': metadata.get('section', ''),
                    'section_type': metadata.get('section_type', 'general'),
                    'chunk_type': metadata.get('chunk_type', 'content'),
                    'token_count': metadata.get('token_count', 0),
                    'contextual': True,
                    'score': 1.0,
                    'title': metadata.get('metadata', {}).get('title', ''),
                    'case_title': metadata.get('metadata', {}).get('title', ''),
                    'gr_number': metadata.get('metadata', {}).get('gr_number', ''),
                    'special_number': metadata.get('metadata', {}).get('special_number', ''),
                    'chunk_id': chunk_id  # Include the stable ID for debugging
                }
                results.append(result)
            else:
                print(f"⚠️ Chunk ID {chunk_id} not found in ID-based maps")
        
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the built indexes"""
        # Use the collection name directly (no "_contextual" suffix)
        vector_collection = self.collection
            
        return {
            "total_chunks": len(self.id_to_contextual_chunk),
            "cached_contexts": len(self.contextual_cache),
            "bm25_available": self.bm25_index is not None,
            "vector_collection": vector_collection,
            "chunk_size": self.chunk_size,
            "overlap_ratio": self.overlap_ratio,
            "cache_file": self.cache_file,
            "id_based_storage": True,  # Indicate that ID-based storage is being used
            "unique_ids": len(self.id_to_contextual_chunk)
        }


    def _generate_stable_id(self, metadata: Dict[str, Any], chunk_text: str) -> str:
        """Generate a stable, content-derived ID for a chunk"""
        gr = metadata.get('metadata', {}).get('gr_number', '')
        special = metadata.get('metadata', {}).get('special_number', '')
        section = metadata.get('section', '')
        section_type = metadata.get('section_type', '')
        char_start = str(metadata.get('char_start', 0))
        
        # Create a stable identifier based on content and metadata
        content_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()[:16]
        stable_key = f"{gr}|{special}|{section}|{section_type}|{char_start}|{content_hash}"
        
        return hashlib.sha256(stable_key.encode('utf-8')).hexdigest()
    
    def _rebuild_legacy_lists(self):
        """Rebuild legacy lists from ID-based maps for backward compatibility"""
        self.contextual_chunks = list(self.id_to_contextual_chunk.values())
        self.chunk_metadata = list(self.id_to_metadata.values())
        
        # Rebuild BM25 index from contextual chunks
        if self.contextual_chunks:
            tokenized_chunks = [self._tokenize(chunk) for chunk in self.contextual_chunks]
            self.bm25_index = BM25Okapi(tokenized_chunks)


def create_contextual_rag_system(collection: str = "jurisprudence") -> ContextualRAG:
    """Create and initialize a Contextual RAG system"""
    return ContextualRAG(collection=collection)
