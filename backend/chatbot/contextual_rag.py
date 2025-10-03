# contextual_rag.py — Contextual RAG implementation with hybrid search and reranking
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
from .docker_model_client import generate_with_fallback
from .model_cache import get_cached_embedding_model


class ContextualRAG:
    """Contextual RAG implementation with hybrid search and reranking"""
    
    def __init__(self, 
                 collection: str = "jurisprudence",
                 chunk_size: int = 640,
                 overlap_ratio: float = 0.15,
                 context_model: str = "meta-llama/Meta-Llama-3.2-3B-Instruct-Turbo"):
        self.collection = collection
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.context_model = context_model
        
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
        self.contextual_chunks = []
        self.chunk_metadata = []
        
        # Reranker model
        self.reranker_model = None
        
        # Try to load existing indexes
        self._load_existing_indexes()
        
        # Context generation prompt template (optimized for shorter responses)
        self.CONTEXTUAL_RAG_PROMPT = """Given this legal document excerpt, briefly explain what the chunk discusses:

Document excerpt:
{WHOLE_DOCUMENT}

Chunk to explain:
{CHUNK_CONTENT}

Provide a brief explanation (1-2 sentences) of what this chunk covers in the document."""
    
    def _load_existing_indexes(self) -> None:
        """Try to load existing contextual chunks from Qdrant collection"""
        try:
            # Check if contextual collection exists
            contextual_collection = f"{self.collection}_contextual"
            if self.qdrant.collection_exists(contextual_collection):
                # Get collection info to see if it has data
                collection_info = self.qdrant.get_collection(contextual_collection)
                if collection_info.points_count > 0:
                    print(f"✅ Found existing contextual collection with {collection_info.points_count} points")
                    
                    # Load contextual chunks from Qdrant
                    self._load_contextual_chunks_from_qdrant(contextual_collection)
                    
                    # Note: BM25 index still needs to be rebuilt
                    print("⚠️ BM25 index needs to be rebuilt for full functionality")
                else:
                    print(f"⚠️ Contextual collection exists but is empty")
            else:
                print(f"⚠️ No existing contextual collection found")
        except Exception as e:
            print(f"⚠️ Could not check for existing indexes: {e}")
    
    def _load_contextual_chunks_from_qdrant(self, collection_name: str) -> None:
        """Load contextual chunks from Qdrant collection"""
        try:
            print("🔄 Loading contextual chunks from Qdrant...")
            
            # Get all points from the collection
            points = self.qdrant.scroll(
                collection_name=collection_name,
                limit=10000,  # Adjust based on your data size
                with_payload=True
            )[0]  # scroll returns (points, next_page_offset)
            
            if not points:
                print("⚠️ No points found in contextual collection")
                return
            
            # Extract chunks and metadata
            contextual_chunks = []
            chunk_metadata = []
            
            for point in points:
                payload = point.payload
                if payload:
                    # Add contextual chunk content
                    contextual_chunks.append(payload.get('content', ''))
                    
                    # Reconstruct metadata
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
                    chunk_metadata.append(metadata)
            
            # Store loaded data
            self.contextual_chunks = contextual_chunks
            self.chunk_metadata = chunk_metadata
            
            print(f"✅ Loaded {len(contextual_chunks)} contextual chunks from Qdrant")
            
        except Exception as e:
            print(f"❌ Failed to load contextual chunks from Qdrant: {e}")
            self.contextual_chunks = []
            self.chunk_metadata = []
    
    def _get_qdrant_client(self) -> QdrantClient:
        """Get Qdrant client"""
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        return QdrantClient(host=host, port=port, grpc_port=6334, prefer_grpc=True, timeout=30.0)
    
    def generate_contextual_chunks(self, document: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """Generate contextual chunks using LLM with context size optimization"""
        print(f"🔄 Generating contextual chunks for {len(chunks)} chunks...")
        
        contextual_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Truncate document to fit within context limits
                # Reserve space for prompt template, chunk content, and response
                max_doc_tokens = 3000  # Leave room for prompt + chunk + response
                truncated_doc = self._truncate_to_tokens(document, max_doc_tokens)
                
                # Create prompt for context generation
                prompt = self.CONTEXTUAL_RAG_PROMPT.format(
                    WHOLE_DOCUMENT=truncated_doc,
                    CHUNK_CONTENT=chunk.get('content', '')
                )
                
                # Check if prompt is still too long
                if self._estimate_tokens(prompt) > 3800:  # Leave buffer for response
                    print(f"⚠️ Chunk {i+1}: Prompt still too long, using fallback")
                    contextual_chunks.append(chunk.get('content', ''))
                    continue
                
                # Generate context using Docker model runner
                context = generate_with_fallback(
                    prompt,
                    max_tokens=150,  # Reduced to stay within limits
                    temperature=0.1,
                    top_p=0.9
                )
                
                # Clean up context
                context = context.strip()
                if context.endswith('.'):
                    context = context[:-1]
                
                # Create contextual chunk: context + original chunk
                contextual_chunk = f"{context}. {chunk.get('content', '')}"
                contextual_chunks.append(contextual_chunk)
                
                print(f"✅ Generated context for chunk {i+1}/{len(chunks)}")
                
            except Exception as e:
                print(f"⚠️ Failed to generate context for chunk {i+1}: {e}")
                # Fallback to original chunk
                contextual_chunks.append(chunk.get('content', ''))
        
        return contextual_chunks
    
    def generate_contextual_chunks_optimized(self, document: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """Generate contextual chunks using document summary approach for large documents"""
        print(f"🔄 Generating contextual chunks (optimized) for {len(chunks)} chunks...")
        
        contextual_chunks = []
        
        # For very long documents, create a summary first
        if self._estimate_tokens(document) > 4000:  # Lower threshold
            print("📝 Document is very long, creating summary for context generation...")
            try:
                # Create a summary of the document
                summary_prompt = f"""Create a brief summary (2-3 paragraphs) of this legal document:

{document[:3000]}...

Focus on the main legal issues, parties, and key points."""
                
                if self._estimate_tokens(summary_prompt) <= 3000:  # More aggressive
                    document_summary = generate_with_fallback(
                        summary_prompt,
                        max_tokens=200,  # Shorter summary
                        temperature=0.1
                    )
                    document_for_context = f"Document Summary: {document_summary}\n\nOriginal Document: {document[:1500]}..."
                else:
                    # Fallback to just using beginning of document
                    document_for_context = document[:1500] + "..."
            except Exception as e:
                print(f"⚠️ Failed to create summary, using document excerpt: {e}")
                document_for_context = document[:2000] + "..."
        else:
            document_for_context = document
        
        # Generate context for each chunk
        for i, chunk in enumerate(chunks):
            try:
                # Create prompt for context generation
                prompt = self.CONTEXTUAL_RAG_PROMPT.format(
                    WHOLE_DOCUMENT=document_for_context,
                    CHUNK_CONTENT=chunk.get('content', '')
                )
                
                # Check if prompt is still too long
                if self._estimate_tokens(prompt) > 3000:  # More aggressive limit
                    print(f"⚠️ Chunk {i+1}: Prompt still too long, using rule-based context")
                    # Generate rule-based context instead of skipping
                    section_type = chunk.get('section_type', 'general')
                    if section_type == 'dispositive':
                        basic_context = "This section contains the court's final ruling and decision."
                    elif section_type == 'factual':
                        basic_context = "This section describes the facts and circumstances of the case."
                    elif section_type == 'issues':
                        basic_context = "This section identifies the legal issues to be resolved."
                    elif section_type == 'legal_analysis':
                        basic_context = "This section provides the legal analysis and reasoning."
                    else:
                        basic_context = "This section contains legal content from the case."
                    
                    contextual_chunk = f"{basic_context} {chunk.get('content', '')}"
                    contextual_chunks.append(contextual_chunk)
                    continue
                
                # Generate context using Docker model runner
                context = generate_with_fallback(
                    prompt,
                    max_tokens=100,  # Keep it short
                    temperature=0.1,
                    top_p=0.9
                )
                
                # Clean up context
                context = context.strip()
                if context.endswith('.'):
                    context = context[:-1]
                
                # Always create contextual chunk - this is the core of Contextual RAG
                # Even if context is minimal, it's better than no context
                if context and len(context) > 5:  # Very low threshold - any context is valuable
                    contextual_chunk = f"{context}. {chunk.get('content', '')}"
                else:
                    # Generate a simple context if LLM failed
                    # Extract section info or create basic context
                    section = chunk.get('section', 'content')
                    section_type = chunk.get('section_type', 'general')
                    
                    if section_type == 'dispositive':
                        basic_context = "This section contains the court's final ruling and decision."
                    elif section_type == 'factual':
                        basic_context = "This section describes the facts and circumstances of the case."
                    elif section_type == 'issues':
                        basic_context = "This section identifies the legal issues to be resolved."
                    elif section_type == 'legal_analysis':
                        basic_context = "This section provides the legal analysis and reasoning."
                    else:
                        basic_context = "This section contains legal content from the case."
                    
                    contextual_chunk = f"{basic_context} {chunk.get('content', '')}"
                
                contextual_chunks.append(contextual_chunk)
                
                print(f"✅ Generated context for chunk {i+1}/{len(chunks)}")
                
            except Exception as e:
                print(f"⚠️ Failed to generate context for chunk {i+1}: {e}")
                # Fallback to original chunk
                contextual_chunks.append(chunk.get('content', ''))
        
        return contextual_chunks
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        if not text:
            return ""
        
        # Rough estimation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # Truncate and try to end at sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        if last_sentence_end > max_chars * 0.8:  # If we can find a sentence end in last 20%
            return truncated[:last_sentence_end + 1]
        else:
            return truncated
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
        if not text:
            return 0
        return max(1, len(text) // 4)
    
    def build_hybrid_indexes(self, cases: List[Dict[str, Any]]) -> None:
        """Build both vector and BM25 indexes with contextual chunks"""
        print(f"🔄 Building hybrid indexes for {len(cases)} cases...")
        
        all_chunks = []
        all_contextual_chunks = []
        
        for case in cases:
            # Chunk the case
            chunks = self.chunker.chunk_case(case)
            all_chunks.extend(chunks)
            
            # Generate contextual chunks using optimized approach
            clean_text = case.get('clean_text', '') or case.get('body', '')
            if clean_text:
                contextual_chunks = self.generate_contextual_chunks_optimized(clean_text, chunks)
                all_contextual_chunks.extend(contextual_chunks)
            else:
                # Fallback to original content (when no text available)
                contextual_chunks = [chunk.get('content', '') for chunk in chunks]
                all_contextual_chunks.extend(contextual_chunks)
        
        # Store chunks and metadata
        self.contextual_chunks = all_contextual_chunks
        self.chunk_metadata = all_chunks
        
        # Build BM25 index
        print("🔄 Building BM25 index...")
        tokenized_chunks = [self._tokenize(chunk) for chunk in all_contextual_chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)
        
        # Build vector index (embeddings)
        print("🔄 Building vector embeddings...")
        self._build_vector_embeddings(all_contextual_chunks, all_chunks)
        
        print(f"✅ Built hybrid indexes: {len(all_contextual_chunks)} contextual chunks")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split on whitespace/punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [word for word in text.split() if len(word) > 1]
    
    def _build_vector_embeddings(self, contextual_chunks: List[str], chunk_metadata: List[Dict]) -> None:
        """Build vector embeddings for contextual chunks"""
        print(f"🔄 Generating embeddings for {len(contextual_chunks)} chunks...")
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(contextual_chunks), batch_size):
            batch = contextual_chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch, 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )
            embeddings.extend(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(contextual_chunks)} chunks")
        
        # Store embeddings in Qdrant
        self._store_embeddings_in_qdrant(embeddings, contextual_chunks, chunk_metadata)
    
    def _get_original_case_title(self, gr_number: str, special_number: str) -> str:
        """Look up the original case title from JSONL data"""
        try:
            from .retriever import load_case_from_jsonl

            # Try to load the original case data
            case_id = gr_number or special_number
            if case_id:
                original_case = load_case_from_jsonl(case_id)
                if original_case and isinstance(original_case, dict):
                    # Get the original case title from JSONL
                    original_title = original_case.get('case_title') or original_case.get('title', '')
                    if original_title and len(original_title.strip()) > 10:
                        return original_title.strip()
        except Exception as e:
            print(f"⚠️ Could not load original case title for {case_id}: {e}")
        
        return ""  # Return empty string if lookup fails
    
    def _store_embeddings_in_qdrant(self, embeddings: List[np.ndarray], 
                                   contextual_chunks: List[str], 
                                   chunk_metadata: List[Dict]) -> None:
        """Store embeddings in Qdrant vector database"""
        print("🔄 Storing embeddings in Qdrant...")
        
        # Prepare points for Qdrant using proper PointStruct format
        points = []
        for i, (embedding, chunk_text, metadata) in enumerate(zip(embeddings, contextual_chunks, chunk_metadata)):
            # Try to get the original case title from JSONL
            gr_number = metadata.get('metadata', {}).get('gr_number', '')
            special_number = metadata.get('metadata', {}).get('special_number', '')
            original_title = self._get_original_case_title(gr_number, special_number)
            
            # Use original title if available, otherwise fall back to chunk title
            final_title = original_title if original_title else metadata.get('metadata', {}).get('title', '')
            
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "content": chunk_text,
                    "original_content": metadata.get('content', ''),
                    "section": metadata.get('section', ''),
                    "section_type": metadata.get('section_type', 'general'),
                    "gr_number": gr_number,
                    "special_number": special_number,
                    "title": final_title,
                    "ponente": metadata.get('metadata', {}).get('ponente', ''),
                    "case_type": metadata.get('metadata', {}).get('case_type', ''),
                    "date": metadata.get('metadata', {}).get('date', ''),
                    "chunk_type": metadata.get('chunk_type', 'content'),
                    "token_count": metadata.get('token_count', 0),
                    "contextual": True
                }
            )
            points.append(point)
        
        # Create collection if it doesn't exist and upsert to Qdrant
        try:
            collection_name = f"{self.collection}_contextual"
            
            # Check if collection exists, create if not
            if not self.qdrant.collection_exists(collection_name):
                print(f"🔄 Creating collection: {collection_name}")
                # Get vector size from first embedding
                vector_size = len(embeddings[0]) if embeddings else 768
                
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created collection: {collection_name}")
            
            # Upsert points to collection
            self.qdrant.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"✅ Stored {len(points)} embeddings in Qdrant")
        except Exception as e:
            print(f"❌ Error storing embeddings: {e}")
    
    def vector_retrieval(self, query: str, top_k: int = 150) -> List[int]:
        """Perform vector search and return chunk indices"""
        try:
            # Encode query
            query_vector = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            
            # Search Qdrant
            results = self.qdrant.search(
                collection_name=f"{self.collection}_contextual",
                query_vector=query_vector.tolist(),
                limit=top_k,
                with_payload=True
            )
            
            # Extract chunk indices
            chunk_indices = []
            for hit in results:
                chunk_indices.append(hit.id)
            
            return chunk_indices
            
        except Exception as e:
            print(f"❌ Vector retrieval failed: {e}")
            return []
    
    def bm25_retrieval(self, query: str, k: int = 150) -> List[int]:
        """Perform BM25 search and return chunk indices"""
        if not self.bm25_index:
            print("⚠️ BM25 index not built")
            return []
        
        try:
            # Tokenize query
            query_tokens = self._tokenize(query)
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:k]
            
            return top_indices.tolist()
            
        except Exception as e:
            print(f"❌ BM25 retrieval failed: {e}")
            return []
    
    def reciprocal_rank_fusion(self, *ranked_lists: List[int], k: int = 60) -> Tuple[List[Tuple[int, float]], List[int]]:
        """Fuse ranks from multiple IR systems using Reciprocal Rank Fusion"""
        rrf_map = defaultdict(float)
        
        # Calculate RRF score for each result in each list
        for rank_list in ranked_lists:
            for rank, item in enumerate(rank_list, 1):
                rrf_map[item] += 1 / (rank + k)
        
        # Sort items based on their RRF scores in descending order
        sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
        
        # Return tuple of list of sorted documents by score and sorted documents
        return sorted_items, [item for item, score in sorted_items]
    
    def rerank_results(self, query: str, chunk_indices: List[int], top_n: int = 20) -> List[int]:
        """Rerank results using cross-encoder model"""
        if not chunk_indices:
            return []
        
        try:
            # Get chunks for reranking (limit to avoid memory issues)
            rerank_limit = min(150, len(chunk_indices))
            indices_to_rerank = chunk_indices[:rerank_limit]
            
            # Check if we have contextual chunks loaded
            if not self.contextual_chunks:
                print("⚠️ No contextual chunks loaded, using vector search results as-is")
                return indices_to_rerank[:top_n]
            
            # Get chunks for reranking
            chunks_to_rerank = []
            valid_indices = []
            for idx in indices_to_rerank:
                if idx < len(self.contextual_chunks):
                    chunks_to_rerank.append(self.contextual_chunks[idx])
                    valid_indices.append(idx)
                else:
                    print(f"⚠️ Skipping invalid chunk index: {idx}")
            
            if not chunks_to_rerank:
                print("⚠️ No valid chunks found for reranking")
                return indices_to_rerank[:top_n]
            
            # Use simple similarity-based reranking for now
            # In production, you would use a proper cross-encoder model
            query_vector = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            
            rerank_scores = []
            for chunk_text in chunks_to_rerank:
                chunk_vector = self.embedding_model.encode([chunk_text], convert_to_numpy=True, normalize_embeddings=True)[0]
                similarity = np.dot(query_vector, chunk_vector)
                rerank_scores.append(similarity)
            
            # Get top-n indices from rerank scores
            top_indices = np.argsort(rerank_scores)[::-1][:top_n]
            
            # Map back to original chunk indices
            reranked_indices = [valid_indices[i] for i in top_indices if i < len(valid_indices)]
            
            return reranked_indices
            
        except Exception as e:
            print(f"❌ Reranking failed: {e}")
            return chunk_indices[:top_n]
    
    def retrieve_and_rank(self, query: str, 
                         vector_k: int = 150, 
                         bm25_k: int = 150, 
                         final_k: int = 20) -> List[Dict[str, Any]]:
        """Main retrieval pipeline: hybrid search + RRF + reranking"""
        print(f"🔍 Contextual RAG retrieval for: '{query}'")
        
        # Step 1: Perform vector and BM25 retrieval
        vector_results = self.vector_retrieval(query, top_k=vector_k)
        bm25_results = self.bm25_retrieval(query, k=bm25_k)
        
        print(f"📊 Vector results: {len(vector_results)}, BM25 results: {len(bm25_results)}")
        
        # Step 2: Combine using Reciprocal Rank Fusion
        if vector_results and bm25_results:
            _, hybrid_results = self.reciprocal_rank_fusion(vector_results, bm25_results)
        elif vector_results:
            hybrid_results = vector_results
        elif bm25_results:
            hybrid_results = bm25_results
        else:
            return []
        
        print(f"🔄 Hybrid results: {len(hybrid_results)} chunks")
        
        # Step 3: Rerank to get top results
        reranked_indices = self.rerank_results(query, hybrid_results, top_n=final_k)
        
        print(f"✅ Final results: {len(reranked_indices)} chunks")
        
        # Step 4: Return formatted results
        results = []
        for idx in reranked_indices:
            # Check if we have contextual chunks loaded
            if self.contextual_chunks and idx < len(self.contextual_chunks) and idx < len(self.chunk_metadata):
                # Use contextual chunks
                metadata = self.chunk_metadata[idx].get('metadata', {})
                result = {
                    'content': self.contextual_chunks[idx],
                    'original_content': self.chunk_metadata[idx].get('content', ''),
                    'metadata': metadata,
                    'section': self.chunk_metadata[idx].get('section', ''),
                    'section_type': self.chunk_metadata[idx].get('section_type', 'general'),
                    'chunk_type': self.chunk_metadata[idx].get('chunk_type', 'content'),
                    'token_count': self.chunk_metadata[idx].get('token_count', 0),
                    'contextual': True,
                    'score': 1.0,  # RRF doesn't provide individual scores
                    # Add title fields for proper display
                    'title': metadata.get('title', ''),
                    'case_title': metadata.get('title', ''),
                    'gr_number': metadata.get('gr_number', ''),
                    'special_number': metadata.get('special_number', '')
                }
                results.append(result)
            else:
                # Fallback: Get result from Qdrant directly
                try:
                    contextual_collection = f"{self.collection}_contextual"
                    point = self.qdrant.retrieve(
                        collection_name=contextual_collection,
                        ids=[idx],
                        with_payload=True
                    )
                    
                    if point and len(point) > 0:
                        payload = point[0].payload
                        result = {
                            'content': payload.get('content', ''),
                            'original_content': payload.get('original_content', ''),
                            'metadata': {
                                'gr_number': payload.get('gr_number', ''),
                                'special_number': payload.get('special_number', ''),
                                'title': payload.get('title', ''),
                                'ponente': payload.get('ponente', ''),
                                'case_type': payload.get('case_type', ''),
                                'date': payload.get('date', '')
                            },
                            'section': payload.get('section', ''),
                            'section_type': payload.get('section_type', 'general'),
                            'chunk_type': payload.get('chunk_type', 'content'),
                            'token_count': payload.get('token_count', 0),
                            'contextual': True,
                            'score': 1.0,
                            # Add title fields for proper display
                            'title': payload.get('title', ''),
                            'case_title': payload.get('title', ''),
                            'gr_number': payload.get('gr_number', ''),
                            'special_number': payload.get('special_number', '')
                        }
                        results.append(result)
                    else:
                        print(f"⚠️ No data found for chunk index {idx}")
                        
                except Exception as e:
                    print(f"⚠️ Failed to retrieve chunk {idx} from Qdrant: {e}")
        
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the built indexes"""
        return {
            "total_chunks": len(self.contextual_chunks),
            "bm25_available": self.bm25_index is not None,
            "vector_collection": f"{self.collection}_contextual",
            "chunk_size": self.chunk_size,
            "overlap_ratio": self.overlap_ratio,
            "context_model": self.context_model
        }


def create_contextual_rag_system(collection: str = "jurisprudence") -> ContextualRAG:
    """Create and initialize a Contextual RAG system"""
    return ContextualRAG(collection=collection)
