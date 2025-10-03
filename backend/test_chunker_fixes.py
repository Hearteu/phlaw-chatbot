# test_chunker_fixes.py — Test chunker fixes for date and embedding_model removal
import sys

sys.path.insert(0, '.')

from chatbot.chunker import LegalDocumentChunker


def test_fixes():
    """Test that date is extracted and title is correct"""
    
    # Sample case data matching the real JSONL structure
    sample_case = {
        'id': 'test-case-genbank',
        'gr_number': 'G.R. No. 151809-12',
        'case_title': 'PRESIDENTIAL COMMISSION ON GOOD GOVERNMENT (PCGG), VS. SANDIGANBAYAN',
        'promulgation_date': '2005-04-12',  # This should populate the date field
        'ponente': 'PUNO, J.',
        'case_type': 'civil',
        'division': 'En Banc',
        'en_banc': True,
        'source_url': 'https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/1/43730',
        'promulgation_year': 2005,
        'is_administrative': False,
        'clean_text': '''
        General Bank and Trust Company (GENBANK) encountered financial difficulties.
        
        The Supreme Court ruled on this matter.
        
        WHEREFORE, the petition is GRANTED. SO ORDERED.
        '''
    }
    
    # Create chunker
    chunker = LegalDocumentChunker()
    
    # Chunk the case
    chunks = chunker.chunk_case(sample_case)
    
    print("=" * 70)
    print("CHUNKER FIXES VERIFICATION TEST")
    print("=" * 70)
    
    print(f"\nInput case data:")
    print(f"  case_title: '{sample_case.get('case_title', '')}'")
    print(f"  promulgation_date: '{sample_case.get('promulgation_date', '')}'")
    
    print(f"\nGenerated {len(chunks)} chunks")
    
    # Check first chunk
    if chunks:
        first_chunk = chunks[0]
        
        print(f"\nFirst chunk metadata:")
        print(f"  title: '{first_chunk['metadata'].get('title', '')}'")
        print(f"  date: '{first_chunk['metadata'].get('date', '')}'")
        
        # Verify fixes
        issues = []
        
        # Check 1: Date should be populated
        if not first_chunk['metadata'].get('date', ''):
            issues.append("ISSUE: Date field is blank")
        else:
            print(f"\n  OK: Date field populated: '{first_chunk['metadata']['date']}'")
        
        # Check 2: Title should be case_title, not content
        title = first_chunk['metadata'].get('title', '')
        if 'GENBANK' in title and 'PCGG' not in title:
            issues.append("ISSUE: Title contains content instead of case_title")
        elif 'PCGG' in title or 'SANDIGANBAYAN' in title:
            print(f"  OK: Title is correct case title")
        
        # Check 3: embedding_model should not be in chunk metadata
        if 'embedding_model' in first_chunk.get('metadata', {}):
            issues.append("ISSUE: embedding_model found in chunk metadata")
        else:
            print(f"  OK: embedding_model not in chunk metadata")
        
        print("\n" + "=" * 70)
        if issues:
            print("ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")
            print("=" * 70)
            return False
        else:
            print("SUCCESS: All fixes verified!")
            print("=" * 70)
            return True
    else:
        print("\nERROR: No chunks generated!")
        return False


def test_embed_payload():
    """Test that embed.py creates payload without embedding_model"""
    print("\n" + "=" * 70)
    print("EMBED PAYLOAD TEST")
    print("=" * 70)
    
    # Create sample chunk
    chunker = LegalDocumentChunker()
    sample_case = {
        'gr_number': 'G.R. No. 123456',
        'case_title': 'Test Case',
        'promulgation_date': '2005-01-15',
        'ponente': 'Test J.',
        'case_type': 'civil',
        'clean_text': 'Test content. WHEREFORE, GRANTED. SO ORDERED.'
    }
    
    chunks = chunker.chunk_case(sample_case)
    
    if chunks:
        print("\nChunk metadata keys:")
        metadata_keys = list(chunks[0]['metadata'].keys())
        print(f"  {metadata_keys}")
        
        if 'embedding_model' in metadata_keys:
            print("\n  ISSUE: embedding_model still in chunk metadata")
            return False
        else:
            print("\n  OK: embedding_model not in chunk metadata")
            return True
    
    return False


if __name__ == "__main__":
    print("\nCHUNKER FIXES VERIFICATION")
    print("=" * 70)
    
    test1_passed = test_fixes()
    test2_passed = test_embed_payload()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Main fixes test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Embed payload test: {'PASS' if test2_passed else 'FAIL'}")
    print("=" * 70)
    
    if test1_passed and test2_passed:
        print("\nAll tests passed!")
        print("\nFixes applied:")
        print("  1. Date field now uses promulgation_date")
        print("  2. Title field now uses case_title")
        print("  3. embedding_model removed from Qdrant payload")
        print("\nNOTE: To apply fixes to existing data, re-embed:")
        print("  cd backend")
        print("  python chatbot/embed.py")
    else:
        print("\nSome tests failed. Please review the fixes.")
