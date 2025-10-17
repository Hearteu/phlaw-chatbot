# embed.py — Enhanced embedding pipeline with structure-aware chunking
import gzip
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

# Import the new chunker (handle both package and direct execution)
try:
    from .chunker import LegalDocumentChunker
except ImportError:
    # For direct execution
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    from chunker import LegalDocumentChunker

# -----------------------------
# Config
# -----------------------------
load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "jurisprudence")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Stern5497/sbert-legal-xlm-roberta-base")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 768))

# Source data (JSONL only)
DATA_FILE = os.getenv("DATA_FILE", "backend/data/cases.jsonl.gz")
YEAR_START = int(os.getenv("YEAR_START", 2005))
YEAR_END = int(os.getenv("YEAR_END", 2005))

# Chunking config
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", 640))
OVERLAP_RATIO = float(os.getenv("OVERLAP_RATIO", 0.15))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", 200))
MAX_DISPOSITIVE_SIZE = int(os.getenv("MAX_DISPOSITIVE_SIZE", 1200))

# Processing config
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 32))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", 1024))

# -----------------------------
# Initialize components
# -----------------------------
print(f"Connecting to Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
client = QdrantClient(
    host=QDRANT_HOST, port=QDRANT_PORT, grpc_port=6334, prefer_grpc=True, timeout=120.0
)

if not client.collection_exists(QDRANT_COLLECTION):
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
print(f"Qdrant collection ready: {QDRANT_COLLECTION}")

# Load embedding model using centralized cache
print(f"Loading model: {EMBED_MODEL}")
try:
    from .model_cache import get_cached_embedding_model
except ImportError:
    # For direct execution
    from model_cache import get_cached_embedding_model

model = get_cached_embedding_model()
if torch.cuda.is_available() and not str(model.device).startswith('cuda'):
    model = model.to("cuda")

# Initialize chunker
chunker = LegalDocumentChunker(
    chunk_size=CHUNK_SIZE_TOKENS,
    overlap_ratio=OVERLAP_RATIO,
    min_chunk_size=MIN_CHUNK_SIZE,
    max_dispositive_size=MAX_DISPOSITIVE_SIZE
)

# -----------------------------
# JSONL Helper Functions
# -----------------------------
def load_case_from_jsonl(case_id: str, jsonl_path: str = DATA_FILE):
    """Load full case text from JSONL file by case ID"""
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

def iter_cases(path: str):
    """Yield JSONL records from .jsonl or .jsonl.gz"""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

# -----------------------------
# Chunk Processing Functions
# -----------------------------
def create_points_from_chunks(chunks: List[Dict[str, Any]]) -> List[PointStruct]:
    """Convert chunked case data into Qdrant points"""
    if not chunks:
        return []
    
    # Extract all chunk content for batch embedding
    texts = [chunk['content'] for chunk in chunks]
    
    # Encode all chunks in one batch
    print(f"Encoding {len(texts)} chunks...")
    vectors = model.encode(
        texts, 
        batch_size=BATCH_SIZE, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )
    
    # Create points
    points = []
    for chunk, vector in zip(chunks, vectors):
        # Ensure chunk ID is a valid UUID string
        chunk_id = chunk['id']
        if not isinstance(chunk_id, str):
            chunk_id = str(chunk_id)
        
        # Create comprehensive payload for Qdrant
        payload = {
            # Content fields
            'content': chunk['content'],
            'section': chunk['section'],
            'section_type': chunk['section_type'],
            'chunk_type': chunk['chunk_type'],
            'paragraph_index': chunk['paragraph_index'],
            'token_count': chunk['token_count'],
            
            # Case metadata (from chunk metadata)
            'case_id': chunk['metadata']['case_id'],
            'gr_number': chunk['metadata']['gr_number'],
            'special_number': chunk['metadata']['special_number'],
            'title': chunk['metadata']['title'],
            'date': chunk['metadata']['date'],
            'ponente': chunk['metadata']['ponente'],
            'division': chunk['metadata']['division'],
            'en_banc': chunk['metadata']['en_banc'],
            'source_url': chunk['metadata']['source_url'],
            'promulgation_year': chunk['metadata']['promulgation_year'],
            'is_administrative': chunk['metadata']['is_administrative'],
            
            # Chunk identification
            'chunk_index': chunk.get('chunk_index', 0),
            'total_chunks': chunk.get('total_chunks', 1),
            
            # Processing metadata
            'chunk_version': 'v2.0',
            'processed_timestamp': int(time.time()),
            
            # Legal classification metadata (from Saibo classifier)
            'classification_method': chunk['metadata'].get('classification_method', 'unknown'),
            'classification_confidence': chunk['metadata'].get('classification_confidence', 0.0),
            'case_type_classification': chunk['metadata'].get('case_type', {}),
            'legal_area_classification': chunk['metadata'].get('legal_area', {}),
            'document_section_classification': chunk['metadata'].get('document_section', {}),
            'complexity_level_classification': chunk['metadata'].get('complexity_level', {}),
            'jurisdiction_level_classification': chunk['metadata'].get('jurisdiction_level', {}),
        }
        
        point = PointStruct(
            id=chunk_id,
            vector=vector.tolist(),
            payload=payload
        )
        points.append(point)
    
    return points

def upsert_points(points: List[PointStruct]):
    """Upsert points to Qdrant in batches"""
    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i : i + UPSERT_BATCH]
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
        print(f"Upserted batch {i//UPSERT_BATCH + 1}: {len(batch)} points")

# -----------------------------
# Main Processing Function
# -----------------------------
def process_jsonl():
    """
    Process JSONL file with structure-aware chunking:
    - Load cases from JSONL
    - Apply structure-aware chunking 
    - Create embeddings for each chunk
    - Store in Qdrant with rich metadata
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"DATA_FILE not found: {DATA_FILE}")

    pending_points: List[PointStruct] = []
    processed_cases = 0
    total_chunks = 0
    year_stats = defaultdict(int)
    section_stats = defaultdict(int)

    print(f"Starting structure-aware embedding pipeline")
    print(f"Input: {DATA_FILE}")
    print(f"Target: {QDRANT_COLLECTION}")
    print(f"Config: {CHUNK_SIZE_TOKENS} tokens, {OVERLAP_RATIO:.1%} overlap")
    
    for rec in iter_cases(DATA_FILE):
        try:
            # Year filtering
            y = rec.get("promulgation_year")
            if not isinstance(y, int):
                d = rec.get("promulgation_date")
                if isinstance(d, str) and len(d) >= 4 and d[:4].isdigit():
                    try:
                        y = int(d[:4])
                    except Exception:
                        y = None
            
            if isinstance(y, int) and (y < YEAR_START or y > YEAR_END):
                continue

            # Skip cases without sufficient content
            clean_text = rec.get('clean_text', '')
            if not clean_text or len(clean_text) < 500:  # Minimum content threshold
                continue

            # Classify case data using Saibo legal document classifier
            try:
                try:
                    from .legal_document_classifier import classify_legal_case
                except ImportError:
                    # For direct execution
                    from legal_document_classifier import classify_legal_case
                classification_result = classify_legal_case(rec.copy())
                
                # Add classification results to case metadata
                if classification_result.get('success', False):
                    rec['legal_classification'] = classification_result
                    rec['case_type'] = classification_result.get('predictions', {}).get('case_type', {})
                    rec['legal_area'] = classification_result.get('predictions', {}).get('legal_area', {})
                    rec['document_section'] = classification_result.get('predictions', {}).get('document_section', {})
                    rec['complexity_level'] = classification_result.get('predictions', {}).get('complexity_level', {})
                    rec['jurisdiction_level'] = classification_result.get('predictions', {}).get('jurisdiction_level', {})
                    rec['classification_method'] = classification_result.get('method', 'unknown')
                    rec['classification_confidence'] = classification_result.get('confidence', 0.0)
                    
                    print(f"Classified case {rec.get('gr_number', 'unknown')}: {classification_result.get('method', 'unknown')} method, confidence: {classification_result.get('confidence', 0.0):.3f}")
                else:
                    print(f"Classification failed for case {rec.get('gr_number', 'unknown')}")
                    
            except Exception as e:
                print(f"Saibo classification failed for case {rec.get('gr_number', 'unknown')}: {e}")
                # Continue with original case data
            
            # Apply structure-aware chunking
            chunks = chunker.chunk_case(rec)
            if not chunks:
                continue

            # Get chunking statistics
            stats = chunker.get_chunking_stats(chunks)
            for section, section_data in stats['sections'].items():
                section_stats[section] += section_data['count']

            # Create Qdrant points from chunks
            points = create_points_from_chunks(chunks)
            pending_points.extend(points)
            
            processed_cases += 1
            total_chunks += len(chunks)
            
            # Track year statistics
            if isinstance(y, int):
                year_stats[y] += 1

            # Upsert in batches
            if len(pending_points) >= UPSERT_BATCH:
                upsert_points(pending_points)
                pending_points.clear()

            # Progress logging
            if processed_cases % 100 == 0:
                print(f"Progress: {processed_cases} cases, {total_chunks} chunks")

        except Exception as e:
            src = rec.get("source_url") if isinstance(rec, dict) else "unknown"
            print(f"Skipping record ({src}) - {e}")

    # Upsert remaining points
    if pending_points:
        upsert_points(pending_points)

    # Print comprehensive statistics
    print(f"\nProcessing Complete!")
    print(f"Total Statistics:")
    print(f"   Cases processed: {processed_cases}")
    print(f"   Total chunks created: {total_chunks}")
    print(f"   Average chunks per case: {total_chunks / processed_cases:.1f}")
    
    print(f"\nYear Distribution:")
    for year in sorted(year_stats.keys()):
        print(f"   {year}: {year_stats[year]} cases")
    
    print(f"\nSection Distribution:")
    for section in sorted(section_stats.keys()):
        print(f"   {section}: {section_stats[section]} chunks")
    
    print(f"\nCollection: {QDRANT_COLLECTION}")
    collection_info = client.get_collection(QDRANT_COLLECTION)
    print(f"   Total points: {collection_info.points_count:,}")

def test_chunking_sample():
    """Test the chunking strategy with a sample case"""
    print("Testing chunking with sample case...")
    
    # Load first case for testing
    for rec in iter_cases(DATA_FILE):
        print(f"Testing with: {rec.get('case_title', 'Unknown')[:100]}...")
        
        chunks = chunker.chunk_case(rec)
        stats = chunker.get_chunking_stats(chunks)
        
        print(f"Generated {len(chunks)} chunks")
        print(f"Statistics: {stats}")
        print(f"Sample chunks:")
        
        for i, chunk in enumerate(chunks[:3]):
            print(f"   {i+1}. [{chunk['section']}] {chunk['token_count']} tokens")
            print(f"      {chunk['content'][:100]}...")
        
        break

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_chunking_sample()
    else:
        print(f"Mode: Structure-Aware JSONL Processing")
        process_jsonl()
