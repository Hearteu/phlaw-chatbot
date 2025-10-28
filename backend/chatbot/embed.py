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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
# Fine-tuned Model Classification
# -----------------------------

# Global variables for fine-tuned model
_finetuned_model = None
_finetuned_tokenizer = None
_finetuned_model_loaded = False

def load_finetuned_model(model_path: str = None):
    """Load the fine-tuned Legal RoBERTa model and tokenizer"""
    global _finetuned_model, _finetuned_tokenizer, _finetuned_model_loaded
    
    if _finetuned_model_loaded:
        return _finetuned_model, _finetuned_tokenizer
    
    # Determine the correct path based on execution context
    if model_path is None:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try different possible paths
        possible_paths = [
            "backend/legal_roberta_finetuned",  # When running from project root
            os.path.join(script_dir, "..", "legal_roberta_finetuned"),  # Relative to script
            os.path.join(os.path.dirname(script_dir), "legal_roberta_finetuned"),  # Backend directory
            "legal_roberta_finetuned",  # When running from project root (alternative)
            os.path.join(script_dir, "legal_roberta_finetuned"),  # Same directory as script
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("Fine-tuned model not found in any expected location, using original Saibo model")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Script directory: {script_dir}")
            print(f"Searched paths:")
            for i, path in enumerate(possible_paths):
                exists = os.path.exists(path)
                print(f"  {i+1}. {path} - {'EXISTS' if exists else 'NOT FOUND'}")
            return None, None
    
    try:
        if not os.path.exists(model_path):
            print(f"Fine-tuned model not found at {model_path}, using original Saibo model")
            return None, None
        
        print(f"Loading fine-tuned Legal RoBERTa model from {model_path}...")
        _finetuned_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _finetuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set to evaluation mode
        _finetuned_model.eval()
        _finetuned_model_loaded = True
        
        print("Fine-tuned model loaded successfully")
        print(f"Model info - Labels: {_finetuned_model.config.num_labels}, Problem type: {_finetuned_model.config.problem_type}")
        return _finetuned_model, _finetuned_tokenizer
        
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        return None, None

def _organize_flat_predictions(flat_predictions: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Organize flat predictions into categories"""
    # Define label-to-category mapping based on fine-tuned model labels
    LABEL_CATEGORIES = {
        # Case types
        'civil': 'case_type',
        'criminal': 'case_type',
        'administrative': 'case_type',
        'constitutional': 'case_type',
        'labor': 'case_type',
        'commercial': 'case_type',
        'family': 'case_type',
        'property': 'case_type',
        'tort': 'case_type',
        'tax': 'case_type',
        'environmental': 'case_type',
        'election': 'case_type',
        'agrarian': 'case_type',
        'intellectual_property': 'case_type',
        'special_civil_action': 'case_type',
        'special_proceedings': 'case_type',
        'appellate': 'case_type',
        'original_jurisdiction': 'case_type',
        'admiralty': 'case_type',
        'insurance': 'case_type',
        'banking': 'case_type',
        'corporate': 'case_type',
        'public_international': 'case_type',
        'private_international': 'case_type',
        
        # Legal areas
        'criminal_law': 'legal_area',
        'civil_law': 'legal_area',
        'administrative_law': 'legal_area',
        'constitutional_law': 'legal_area',
        'labor_law': 'legal_area',
        'commercial_law': 'legal_area',
        'family_law': 'legal_area',
        'property_law': 'legal_area',
        'tort_law': 'legal_area',
        'tax_law': 'legal_area',
        'environmental_law': 'legal_area',
        'election_law': 'legal_area',
        'agrarian_law': 'legal_area',
        'intellectual_property_law': 'legal_area',
        'remedial_law': 'legal_area',
        'political_law': 'legal_area',
        'public_corporation_law': 'legal_area',
        'private_corporation_law': 'legal_area',
        'banking_law': 'legal_area',
        'insurance_law': 'legal_area',
        'maritime_law': 'legal_area',
        'international_law': 'legal_area',
        'ethics': 'legal_area',
        'constitutional_remedies': 'legal_area',
        
        # Document sections
        'facts': 'document_section',
        'issues': 'document_section',
        'ruling': 'document_section',
        'ratio_decidendi': 'document_section',
        'disposition': 'document_section',
        'dissenting_opinion': 'document_section',
        'concurring_opinion': 'document_section',
        'separate_opinion': 'document_section',
        'legal_precedent': 'document_section',
        'procedural_history': 'document_section',
        'summary': 'document_section',
        'background': 'document_section',
        'analysis': 'document_section',
        'petition': 'document_section',
        'motion': 'document_section',
        'memorandum': 'document_section',
        'brief': 'document_section',
        'pleading': 'document_section',
        
        # Complexity level
        'simple': 'complexity_level',
        'moderate': 'complexity_level',
        'complex': 'complexity_level',
        'highly_complex': 'complexity_level',
        
        # Jurisdiction level
        'supreme_court': 'jurisdiction_level',
        'appellate_court': 'jurisdiction_level',
        'regional_trial_court': 'jurisdiction_level',
        'municipal_trial_court': 'jurisdiction_level',
        'quasi_judicial_agency': 'jurisdiction_level',
        'sandiganbayan': 'jurisdiction_level',
        'court_of_tax_appeals': 'jurisdiction_level',
        'court_of_appeals': 'jurisdiction_level',
        
        # Case status
        'pending': 'case_status',
        'decided': 'case_status',
        'dismissed': 'case_status',
        'settled': 'case_status',
        'appealed': 'case_status',
        'affirmed': 'case_status',
        'reversed': 'case_status',
        'modified': 'case_status',
        'remanded': 'case_status',
        
        # Issue categories
        'constitutional_question': 'issue_category',
        'administrative_matter': 'issue_category',
        'criminal_liability': 'issue_category',
        'civil_liability': 'issue_category',
        'contract_dispute': 'issue_category',
        'property_rights': 'issue_category',
        'family_dispute': 'issue_category',
        'labor_dispute': 'issue_category',
        'tax_dispute': 'issue_category',
        'environmental_violation': 'issue_category',
        'election_dispute': 'issue_category',
        'banking_dispute': 'issue_category',
        'insurance_claim': 'issue_category'
    }
    
    # Initialize organized structure
    organized = {}
    
    # Organize predictions by category
    for label, score in flat_predictions.items():
        category = LABEL_CATEGORIES.get(label)
        if category:
            if category not in organized:
                organized[category] = {}
            organized[category][label] = score
    
    # Sort each category by confidence
    for category in organized:
        organized[category] = dict(
            sorted(organized[category].items(), key=lambda x: x[1], reverse=True)
        )
    
    return organized


def classify_with_finetuned_model(case_data: Dict[str, Any], confidence_threshold: float = 0.1) -> Dict[str, Any]:
    """Classify case using the fine-tuned Legal RoBERTa model"""
    
    # Load model if not already loaded
    model, tokenizer = load_finetuned_model()
    
    if model is None or tokenizer is None:
        return {"success": False, "error": "Fine-tuned model not available"}
    
    try:
        # Extract text for classification
        text = case_data.get('clean_text', '') or case_data.get('content', '')
        title = case_data.get('case_title', '') or case_data.get('title', '')
        
        if not text or len(text) < 100:
            return {"success": False, "error": "Insufficient text for classification"}
        
        # Combine title and text for better context
        if title and len(title) < 200:
            full_text = f"{title}. {text[:1800]}"  # Limit total length
        else:
            full_text = text[:2000]
        
        # Tokenize input
        inputs = tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
        
        # Convert to CPU for processing
        probabilities = probabilities.cpu().numpy()[0]
        
        # Debug: Print comprehensive prediction info
        max_prob = max(probabilities) if len(probabilities) > 0 else 0.0
        print(f"Model predictions - Max probability: {max_prob:.4f}, Threshold: {confidence_threshold}")
        print(f"Total probabilities: {len(probabilities)}")
        
        # Get top 10 predictions for debugging
        if len(probabilities) > 0:
            top_indices = probabilities.argsort()[-10:][::-1]
            print("Top 10 predictions:")
            for i, idx in enumerate(top_indices):
                label_name = model.config.id2label.get(idx, f"label_{idx}")
                print(f"  {i+1}. {label_name}: {probabilities[idx]:.4f}")
        
        # Get predictions above threshold
        flat_predictions = {}
        for i, prob in enumerate(probabilities):
            if prob > confidence_threshold:
                # Get label name from model config
                label_name = model.config.id2label.get(i, f"label_{i}")
                flat_predictions[label_name] = float(prob)
        
        print(f"Predictions above threshold ({confidence_threshold}): {len(flat_predictions)}")
        
        # If no predictions above threshold, get top 5 predictions anyway
        if not flat_predictions and len(probabilities) > 0:
            print("No predictions above threshold, using top 5 predictions...")
            # Get top 5 predictions regardless of threshold
            top_indices = probabilities.argsort()[-5:][::-1]
            for idx in top_indices:
                if probabilities[idx] > 0.01:  # Very low threshold for fallback
                    label_name = model.config.id2label.get(idx, f"label_{idx}")
                    flat_predictions[label_name] = float(probabilities[idx])
                    print(f"  Fallback: {label_name}: {probabilities[idx]:.4f}")
        
        # Organize predictions by category
        organized_predictions = _organize_flat_predictions(flat_predictions)
        
        # Calculate overall confidence
        overall_confidence = max(probabilities) if len(probabilities) > 0 else 0.0
        
        return {
            "success": True,
            "predictions": organized_predictions,
            "raw_predictions": flat_predictions,
            "confidence": float(overall_confidence),
            "method": "finetuned_legal_roberta"
        }
        
    except Exception as e:
        print(f"Classification error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# -----------------------------
# Config
# -----------------------------
load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "jurisprudence2")
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
            
            # Legal classification metadata (from fine-tuned Legal RoBERTa)
            'classification_method': chunk['metadata'].get('classification_method', 'finetuned_legal_roberta'),
            'classification_confidence': chunk['metadata'].get('classification_confidence', 0.0),
            'case_type_classification': chunk['metadata'].get('case_type_classification', {}),
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

            # Classify case data using fine-tuned Legal RoBERTa model
            try:
                classification_result = classify_with_finetuned_model(rec.copy())
                
                # Add classification results to case metadata
                if classification_result.get('success', False):
                    rec['legal_classification'] = classification_result
                    rec['case_type_classification'] = classification_result.get('predictions', {})
                    rec['classification_method'] = classification_result.get('method', 'finetuned_legal_roberta')
                    rec['classification_confidence'] = classification_result.get('confidence', 0.0)
                    
                    print(f"Classified case {rec.get('gr_number', 'unknown')}: {classification_result.get('method', 'unknown')} method, confidence: {classification_result.get('confidence', 0.0):.3f}")
                else:
                    print(f"Classification failed for case {rec.get('gr_number', 'unknown')}: {classification_result.get('error', 'unknown error')}")
                    
            except Exception as e:
                print(f"Fine-tuned model classification failed for case {rec.get('gr_number', 'unknown')}: {e}")
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
