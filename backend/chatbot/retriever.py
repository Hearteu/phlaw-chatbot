# retriever.py — Optimized legal document retriever with enhanced caching
import gzip
import json
import os
import time
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# =============================================================================
# GLOBAL CACHING FOR PERFORMANCE
# =============================================================================
_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_LOADED = False
_QDRANT_CLIENT = None
_COLLECTION_INFO = None

def _get_cached_embedding_model():
    """Get cached SentenceTransformer model with lazy loading"""
    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_LOADED
    
    if _EMBEDDING_MODEL is None and not _EMBEDDING_MODEL_LOADED:
        print("🔄 Loading SentenceTransformer model...")
        start_time = time.time()
        
        # Use optimized model path
        model_name = os.getenv("EMBED_MODEL", "Stern5497/sbert-legal-xlm-roberta-base")
        _EMBEDDING_MODEL = SentenceTransformer(model_name, device="cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        _EMBEDDING_MODEL_LOADED = True
    
    return _EMBEDDING_MODEL

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
                "name": collection_name,  # Use the collection name we passed in
                "vector_count": getattr(info, 'points_count', 0),  # Use points_count instead of vectors_count
                "status": getattr(info, 'status', 'Unknown')
            }
        except Exception as e:
            print(f"❌ Failed to get collection info: {e}")
            _COLLECTION_INFO = {"name": collection_name, "vector_count": 0, "status": "Unknown"}
    
    return _COLLECTION_INFO

class LegalRetriever:
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

    def retrieve(self, query: str, k: int = 3, is_case_digest: bool = False) -> List[Dict[str, Any]]:
        """Fast vector search with optimized performance"""
        print(f"🔍 Query: '{query}' | Retrieving {k} documents...")
        
        # Handle G.R. No. queries specifically
        if "G.R. No." in query.upper():
            return self._retrieve_gr_case(query, k)
        
        print("🔄 Fast vector search...")
        start_time = time.time()
        
        try:
            # Encode query
            qv = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False).tolist()
            
            # For case digests, retrieve more documents to ensure comprehensive coverage
            limit = max(30, k * 6) if is_case_digest else max(20, k * 4)
            
            # Perform fast search with larger limit for better quality
            hits = self.qdrant.search(
                collection_name=self.collection, 
                query_vector=qv, 
                limit=limit,
                with_payload=True,
                score_threshold=0.4 if is_case_digest else 0.5  # Lower threshold for case digests
            )
            
            search_time = time.time() - start_time
            print(f"✅ Fast search completed: {len(hits)} results in {search_time:.2f}s")
            
            # Process and deduplicate results with case digest optimization
            if is_case_digest:
                docs = self._process_search_results_for_digest(hits, k)
            else:
                docs = self._process_search_results(hits, k)
            
            return docs[:k]
            
        except Exception as e:
            print(f"❌ Fast search failed: {e}")
            return []

    def _retrieve_gr_case(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Special handling for G.R. No. queries"""
        print("🎯 G.R. No. query detected - using specialized search...")
        
        # Extract G.R. number
        import re
        gr_match = re.search(r"G\.R\.\s*No\.\s*(\d+)", query, re.IGNORECASE)
        if not gr_match:
            return []
        
        gr_number = gr_match.group(1)
        print(f"🔍 Searching for G.R. No. {gr_number}")
        
        # Use payload filtering for exact matches
        try:
            hits = self.qdrant.search(
                collection_name=self.collection,
                query_vector=self.model.encode(query, convert_to_numpy=True, show_progress_bar=False).tolist(),
                limit=10,
                with_payload=True,
                query_filter={
                    "must": [
                        {
                            "key": "gr_number",
                            "match": {"text": gr_number}
                        }
                    ]
                }
            )
            
            if hits:
                return self._process_search_results(hits, k)
            else:
                print(f"⚠️ No exact G.R. No. {gr_number} found, falling back to semantic search")
                return self.retrieve(query, k)
                
        except Exception as e:
            print(f"❌ G.R. No. search failed: {e}")
            return self.retrieve(query, k)

    def _process_search_results(self, hits: List, k: int) -> List[Dict[str, Any]]:
        """Process and deduplicate search results"""
        seen_titles = set()
        docs = []
        
        for hit in hits:
            if not hit.payload:
                continue
                
            title = hit.payload.get("title", "Unknown Title")
            
            # Skip duplicates
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            # Extract relevant sections
            doc = {
                "title": title,
                "score": hit.score,
                "url": hit.payload.get("url", ""),
                "year": hit.payload.get("year", ""),
                "gr_number": hit.payload.get("gr_number", ""),
                "content": self._extract_best_section(hit.payload),
                "metadata": {
                    "section": hit.payload.get("section", "unknown"),
                    "chunk_id": hit.payload.get("chunk_id", ""),
                    "case_type": hit.payload.get("case_type", "")
                }
            }
            
            docs.append(doc)
            
            if len(docs) >= k:
                break
        
        return docs

    def _process_search_results_for_digest(self, hits: List, k: int) -> List[Dict[str, Any]]:
        """Process search results optimized for case digest generation"""
        # Group results by case to ensure we get comprehensive coverage
        case_groups = {}
        section_priority = ["issues", "facts", "ruling", "arguments", "header", "body"]
        
        for hit in hits:
            if not hit.payload:
                continue
                
            title = hit.payload.get("title", "Unknown Title")
            section = hit.payload.get("section", "body")
            
            # Group by case title
            if title not in case_groups:
                case_groups[title] = {}
            
            # Store the best result for each section
            if section not in case_groups[title] or hit.score > case_groups[title][section].get("score", 0):
                case_groups[title][section] = {
                    "title": title,
                    "score": hit.score,
                    "url": hit.payload.get("url", ""),
                    "year": hit.payload.get("year", ""),
                    "gr_number": hit.payload.get("gr_number", ""),
                    "content": self._extract_best_section(hit.payload),
                    "metadata": {
                        "section": section,
                        "chunk_id": hit.payload.get("chunk_id", ""),
                        "case_type": hit.payload.get("case_type", "")
                    }
                }
        
        # Select the best cases with comprehensive section coverage
        docs = []
        for title, sections in case_groups.items():
            # Prioritize sections for case digest
            case_docs = []
            for section in section_priority:
                if section in sections:
                    case_docs.append(sections[section])
            
            # Add remaining sections
            for section, doc in sections.items():
                if section not in section_priority:
                    case_docs.append(doc)
            
            # Add the best case to results
            if case_docs:
                # Use the highest scoring document as the primary document
                primary_doc = max(case_docs, key=lambda x: x["score"])
                docs.append(primary_doc)
                
                # Add additional sections if we have space
                for doc in case_docs:
                    if doc != primary_doc and len(docs) < k * 2:  # Allow more docs for digests
                        docs.append(doc)
        
        # Sort by score and return top results
        docs.sort(key=lambda x: x["score"], reverse=True)
        return docs[:k * 2]  # Return more documents for case digests

    def _extract_best_section(self, payload: Dict[str, Any]) -> str:
        """Extract the most relevant section content"""
        # Priority order: facts, issues, ruling, arguments, header
        sections = ["facts", "issues", "ruling", "arguments", "header"]
        
        for section in sections:
            if section in payload and payload[section]:
                content = payload[section]
                # Limit content length for context
                if len(content) > 800:
                    content = content[:800] + "..."
                return content
        
        # Fallback to any available content
        for key, value in payload.items():
            if isinstance(value, str) and len(value) > 100:
                return value[:800] + "..." if len(value) > 800 else value
        
        # If no content in payload, fetch from original source files
        return self._fetch_content_from_source(payload)

    def _fetch_content_from_source(self, payload: Dict[str, Any]) -> str:
        """Fetch content from original source files when Qdrant payload is missing content"""
        try:
            gr_number = payload.get("gr_number")
            if not gr_number:
                return "No content available - missing G.R. number"
            
            # Try to find the case in the enhanced data file
            content = self._find_case_content(gr_number)
            if content:
                return content
            
            return f"No content available for {gr_number}"
            
        except Exception as e:
            print(f"⚠️ Error fetching content from source: {e}")
            return "No content available - error fetching from source"

    def _find_case_content(self, gr_number: str) -> Optional[str]:
        """Find case content in the enhanced data file by G.R. number"""
        try:
            # Try the enhanced file first
            file_path = "data/cases_enhanced.jsonl.gz"
            if not os.path.exists(file_path):
                file_path = "data/cases_improved.jsonl.gz"
            if not os.path.exists(file_path):
                file_path = "data/cases.jsonl.gz"
            
            if not os.path.exists(file_path):
                print(f"⚠️ No source data file found")
                return None
            
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get("gr_number") == gr_number:
                            # Extract the best section content
                            sections = data.get("sections", {})
                            
                            # Priority order for sections
                            priority_sections = ["facts", "issues", "ruling", "arguments", "header", "body", "summary"]
                            
                            for section_name in priority_sections:
                                if section_name in sections and sections[section_name]:
                                    content = sections[section_name]
                                    # Clean and limit content
                                    if isinstance(content, str) and len(content.strip()) > 50:
                                        # Remove common headers and clean up
                                        content = content.replace("Supreme Court E-Library", "").strip()
                                        if len(content) > 800:
                                            content = content[:800] + "..."
                                        return content
                            
                            # Fallback to clean_text if no sections found
                            clean_text = data.get("clean_text", "")
                            if clean_text and len(clean_text.strip()) > 100:
                                # Remove common headers
                                clean_text = clean_text.replace("Supreme Court E-Library", "").strip()
                                if len(clean_text) > 800:
                                    clean_text = clean_text[:800] + "..."
                                return clean_text
                            
                            return f"Case found but no content available for {gr_number}"
                    
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"⚠️ Error processing line for {gr_number}: {e}")
                        continue
            
            return f"Case {gr_number} not found in source files"
            
        except Exception as e:
            print(f"❌ Error reading source file: {e}")
            return None

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.qdrant.get_collection(self.collection)
            vector_size = None
            if hasattr(info, 'config') and hasattr(info.config, 'params') and hasattr(info.config.params, 'vectors'):
                vector_size = getattr(info.config.params.vectors, 'size', None)
            return {
                "name": self.collection,  # Use the collection name we know
                "vector_count": getattr(info, 'points_count', 0),  # Use points_count instead of vectors_count
                "status": getattr(info, 'status', 'Unknown'),
                "vector_size": vector_size
            }
        except Exception as e:
            print(f"❌ Failed to get collection stats: {e}")
            return {"error": str(e)}
