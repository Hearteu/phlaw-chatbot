#!/usr/bin/env python3
"""
Test script for the new legal document chunking strategy
"""
import gzip
import json
import os
import sys

# Add the parent directory to the path so we can import from chatbot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.chunker import LegalDocumentChunker, chunk_legal_document


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
            # Let's try to find the first complete JSON object
            json_objects = []
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

def test_chunking_strategy():
    """Test the new chunking strategy"""
    print("🧪 Testing Legal Document Chunking Strategy")
    print("=" * 50)
    
    # Load sample case
    print("📄 Loading sample case...")
    case = load_sample_case()
    if not case:
        print("❌ Could not load sample case")
        return
    
    print(f"✅ Loaded case: {case.get('case_title', 'Unknown')[:100]}...")
    print(f"📏 Original text length: {len(case.get('clean_text', ''))} characters")
    
    # Test chunking
    print("\n🔪 Testing chunking strategy...")
    chunker = LegalDocumentChunker(
        chunk_size=640,  # tokens
        overlap_ratio=0.15,
        min_chunk_size=200,
        max_dispositive_size=1200
    )
    
    chunks = chunker.chunk_case(case)
    
    # Display results
    print(f"✅ Generated {len(chunks)} chunks")
    
    # Get chunking stats
    stats = chunker.get_chunking_stats(chunks)
    print(f"\n📊 Chunking Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
    print(f"  Chunk types: {stats['chunk_types']}")
    
    print(f"\n📋 Sections breakdown:")
    for section, section_stats in stats['sections'].items():
        print(f"  {section}: {section_stats['count']} chunks, {section_stats['tokens']} tokens")
    
    # Show sample chunks
    print(f"\n📄 Sample chunks:")
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        section = chunk.get('section', 'unknown')
        section_type = chunk.get('section_type', 'general')
        tokens = chunk.get('token_count', 0)
        preview = chunk.get('content_preview', '')
        
        print(f"\n  Chunk {i+1}: [{section}] ({section_type}) - {tokens} tokens")
        print(f"    Preview: {preview[:150]}...")
    
    if len(chunks) > 5:
        print(f"\n  ... and {len(chunks) - 5} more chunks")
    
    # Test context creation (if we have retriever available)
    try:
        from chatbot.retriever import LegalRetriever
        
        print(f"\n🔗 Testing context creation...")
        # Create a mock retriever just for context creation
        class MockRetriever:
            def _create_context_from_chunks(self, chunks, max_tokens=2500):
                context_parts = []
                current_tokens = 0
                
                section_priority = {'summary': 0, 'ruling': 1, 'dispositive': 1, 'facts': 2, 'issues': 3}
                sorted_chunks = sorted(chunks, key=lambda x: section_priority.get(x.get('section', 'body'), 5))
                
                for chunk in sorted_chunks:
                    content = chunk.get('content', '')
                    chunk_tokens = chunk.get('token_count', len(content) // 4)
                    
                    if current_tokens + chunk_tokens > max_tokens:
                        break
                    
                    section = chunk.get('section', 'content')
                    metadata = chunk.get('metadata', {})
                    case_info = ""
                    if metadata.get('gr_number'):
                        case_info = f" ({metadata['gr_number']})"
                    
                    context_parts.append(f"[{section.title()}{case_info}] {content}")
                    current_tokens += chunk_tokens
                
                return "\n\n".join(context_parts)
        
        mock_retriever = MockRetriever()
        context = mock_retriever._create_context_from_chunks(chunks, max_tokens=2500)
        
        print(f"✅ Generated context: {len(context)} characters")
        print(f"📄 Context preview:")
        print(context[:500] + "..." if len(context) > 500 else context)
        
    except ImportError:
        print("⚠️ Retriever not available for context testing")
    
    print(f"\n✅ Chunking test completed successfully!")

if __name__ == "__main__":
    test_chunking_strategy()
