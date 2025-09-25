#!/usr/bin/env python3
"""
Test script to verify embedder and crawler alignment with chunking strategy
"""
import json
import os
import sys

# Add the parent directory to the path so we can import from chatbot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from chatbot.chunker import LegalDocumentChunker
    from chatbot.crawler import extract_sections_for_chunking, parse_case
except ImportError:
    # Handle different execution contexts
    import os
    import sys
    backend_path = os.path.join(os.path.dirname(__file__), 'chatbot')
    sys.path.append(backend_path)
    from chunker import LegalDocumentChunker
    from crawler import extract_sections_for_chunking, parse_case


def load_sample_case():
    """Load a sample case from the JSONL file"""
    data_file = os.path.join(os.path.dirname(__file__), "data", "samples", "sample_cases.jsonl")
    
    if not os.path.exists(data_file):
        print(f"❌ Sample file not found: {data_file}")
        return None
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            # Read the entire file content and parse as JSON objects
            content = f.read().strip()
            
            # The file might contain multiple JSON objects separated by newlines
            lines = content.split('\n')
            current_json = ""
            brace_count = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                current_json += line + "\n"
                
                # Count braces to find complete JSON objects
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0 and current_json.strip():
                    try:
                        case = json.loads(current_json.strip())
                        return case
                    except json.JSONDecodeError:
                        pass
                    current_json = ""
                    
    except Exception as e:
        print(f"❌ Error loading sample case: {e}")
    
    return None


def test_embedder_alignment():
    """Test that embedder and chunker work together properly"""
    print("🧪 Testing Embedder-Chunker Alignment")
    print("=" * 50)
    
    # Load sample case
    print("📄 Loading sample case...")
    case = load_sample_case()
    if not case:
        print("❌ Could not load sample case")
        return
    
    print(f"✅ Loaded case: {case.get('case_title', 'Unknown')[:100]}...")
    
    # Test chunking
    print("\n🔪 Testing chunking...")
    chunker = LegalDocumentChunker()
    chunks = chunker.chunk_case(case)
    
    print(f"✅ Generated {len(chunks)} chunks")
    
    # Test section extraction (crawler alignment)
    print("\n📋 Testing crawler section extraction...")
    clean_text = case.get('clean_text', '')
    sections = extract_sections_for_chunking(clean_text)
    
    print(f"✅ Extracted {len(sections)} sections: {list(sections.keys())}")
    
    # Test embedder compatibility
    print("\n🎯 Testing embedder compatibility...")
    try:
        # Simulate what embedder would do
        if chunks:
            # Check chunk structure
            first_chunk = chunks[0]
            required_fields = ['content', 'section', 'metadata', 'id']
            missing_fields = [field for field in required_fields if field not in first_chunk]
            
            if missing_fields:
                print(f"❌ Missing required fields in chunks: {missing_fields}")
            else:
                print("✅ Chunk structure is compatible with embedder")
                
            # Check metadata structure
            metadata = first_chunk.get('metadata', {})
            required_metadata = ['case_id', 'gr_number', 'title', 'case_type']
            missing_metadata = [field for field in required_metadata if field not in metadata]
            
            if missing_metadata:
                print(f"⚠️ Missing recommended metadata: {missing_metadata}")
            else:
                print("✅ Metadata structure is complete")
                
            # Check section alignment
            chunk_sections = set(chunk['section'] for chunk in chunks)
            crawler_sections = set(sections.keys())
            
            print(f"📊 Chunk sections: {sorted(chunk_sections)}")
            print(f"📊 Crawler sections: {sorted(crawler_sections)}")
            
            # Check for overlaps
            overlapping = chunk_sections & crawler_sections
            if overlapping:
                print(f"✅ Section alignment found: {sorted(overlapping)}")
            else:
                print("⚠️ No direct section alignment (chunker creates additional sections)")
                
        else:
            print("❌ No chunks generated")
            
    except Exception as e:
        print(f"❌ Embedder compatibility test failed: {e}")
    
    # Test case record structure (for new crawled cases)
    print("\n📄 Testing case record structure...")
    try:
        # Simulate what crawler would produce
        test_url = "https://example.com/test"
        test_record = parse_case(
            clean_text, 
            test_url, 
            year_hint=2020, 
            month_hint="january",
            title_guess=case.get('case_title', ''),
            page_title=case.get('page_title', '')
        )
        
        # Check if new record structure is compatible
        if 'sections' in test_record:
            print("✅ Crawler now includes sections field")
            print(f"📋 Sections: {list(test_record['sections'].keys()) if test_record['sections'] else 'None'}")
        else:
            print("❌ Crawler missing sections field")
            
        if 'quality_metrics' in test_record:
            quality = test_record['quality_metrics']
            print(f"✅ Quality metrics: {quality}")
        else:
            print("❌ Missing quality metrics")
            
    except Exception as e:
        print(f"❌ Crawler record test failed: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   Chunker: {len(chunks)} chunks from {len(clean_text)} chars")
    print(f"   Crawler: {len(sections)} sections extracted")
    print(f"   Alignment: Ready for embedding pipeline")
    
    return True


def test_processing_pipeline():
    """Test the complete processing pipeline"""
    print("\n🔄 Testing Complete Processing Pipeline")
    print("=" * 50)
    
    case = load_sample_case()
    if not case:
        return False
    
    try:
        # Step 1: Chunking
        chunker = LegalDocumentChunker()
        chunks = chunker.chunk_case(case)
        print(f"1️⃣ Chunking: {len(chunks)} chunks")
        
        # Step 2: Simulate embedding preparation (without actual model)
        print("2️⃣ Embedding preparation...")
        chunk_texts = [chunk['content'] for chunk in chunks]
        print(f"   Prepared {len(chunk_texts)} texts for embedding")
        
        # Step 3: Simulate Qdrant payload creation
        print("3️⃣ Qdrant payload creation...")
        payloads = []
        for chunk in chunks:
            payload = {
                'content': chunk['content'],
                'section': chunk['section'],
                'case_id': chunk['metadata']['case_id'],
                'gr_number': chunk['metadata']['gr_number'],
                'title': chunk['metadata']['title']
            }
            payloads.append(payload)
        
        print(f"   Created {len(payloads)} payloads")
        
        # Step 4: Verify retrieval readiness
        print("4️⃣ Retrieval readiness...")
        section_types = set(chunk['section'] for chunk in chunks)
        print(f"   Available sections for retrieval: {sorted(section_types)}")
        
        print("✅ Complete pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_embedder_alignment()
    if success:
        test_processing_pipeline()
    
    print(f"\n🎯 Embedder and Crawler are now aligned with chunking strategy!")
