# test_chunker_date_fix.py — Test that chunker now correctly extracts date
import sys

sys.path.insert(0, '.')

from chatbot.chunker import LegalDocumentChunker


def test_date_extraction():
    """Test that date field is properly extracted from promulgation_date"""
    
    # Sample case data with promulgation_date
    sample_case = {
        'id': 'test-case-001',
        'gr_number': 'G.R. No. 137337',
        'special_number': 'SP No. 44846',
        'case_title': 'JUAN PADIN, et al. vs. HEIRS OF VIVENCIO OBIAS',
        'promulgation_date': '2005-07-28',  # This is the field in JSONL
        'ponente': 'SANDOVAL-GUTIERREZ, J.',
        'case_type': 'civil',
        'division': 'Third Division',
        'en_banc': False,
        'source_url': 'https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/1/41548',
        'promulgation_year': 2005,
        'is_administrative': False,
        'clean_text': '''
        The Supreme Court ruled on the issue of tenancy relationship.
        
        WHEREFORE, the petition is GRANTED. SO ORDERED.
        '''
    }
    
    # Create chunker
    chunker = LegalDocumentChunker()
    
    # Chunk the case
    chunks = chunker.chunk_case(sample_case)
    
    print("=" * 70)
    print("DATE FIELD EXTRACTION TEST")
    print("=" * 70)
    
    print(f"\nInput case data:")
    print(f"  promulgation_date: '{sample_case.get('promulgation_date', '')}'")
    print(f"  date: '{sample_case.get('date', '')}' (not in JSONL)")
    
    print(f"\nGenerated {len(chunks)} chunks")
    
    # Check each chunk for date field
    all_have_dates = True
    for i, chunk in enumerate(chunks):
        date_value = chunk['metadata'].get('date', '')
        has_date = bool(date_value)
        
        print(f"\nChunk {i + 1}:")
        print(f"  Section: {chunk['section']}")
        print(f"  Date field: '{date_value}'")
        print(f"  Has date: {'YES' if has_date else 'NO'}")
        
        if not has_date:
            all_have_dates = False
    
    print("\n" + "=" * 70)
    if all_have_dates:
        print("SUCCESS: All chunks have date field populated!")
        print("=" * 70)
        return True
    else:
        print("FAILURE: Some chunks are missing date field!")
        print("=" * 70)
        return False


def test_backward_compatibility():
    """Test that chunker still works with old 'date' field"""
    
    # Sample case with old 'date' field (for backward compatibility)
    sample_case_old = {
        'id': 'test-case-002',
        'gr_number': 'G.R. No. 123456',
        'case_title': 'Test Case',
        'date': '2004-05-15',  # Old field name
        'ponente': 'Test Justice',
        'case_type': 'criminal',
        'clean_text': 'Test content. WHEREFORE, petition is DENIED. SO ORDERED.'
    }
    
    chunker = LegalDocumentChunker()
    chunks = chunker.chunk_case(sample_case_old)
    
    print("\n" + "=" * 70)
    print("BACKWARD COMPATIBILITY TEST")
    print("=" * 70)
    
    print(f"\nInput case data (old format):")
    print(f"  date: '{sample_case_old.get('date', '')}'")
    print(f"  promulgation_date: '{sample_case_old.get('promulgation_date', '')}' (not present)")
    
    if chunks:
        date_value = chunks[0]['metadata'].get('date', '')
        print(f"\nFirst chunk date field: '{date_value}'")
        
        if date_value:
            print("\nSUCCESS: Backward compatibility maintained!")
            print("=" * 70)
            return True
    
    print("\nFAILURE: Backward compatibility broken!")
    print("=" * 70)
    return False


if __name__ == "__main__":
    print("\nCHUNKER DATE FIELD FIX - TEST SUITE")
    print("=" * 70)
    
    # Test 1: New format (promulgation_date)
    test1_passed = test_date_extraction()
    
    # Test 2: Old format (date) - backward compatibility
    test2_passed = test_backward_compatibility()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"New format (promulgation_date): {'PASS' if test1_passed else 'FAIL'}")
    print(f"Old format (date):              {'PASS' if test2_passed else 'FAIL'}")
    print("=" * 70)
    
    if test1_passed and test2_passed:
        print("\nAll tests passed! The date field fix is working correctly.")
        print("\nNOTE: To fix existing chunks in Qdrant, you need to re-embed the cases:")
        print("  python backend/chatbot/embed.py")
    else:
        print("\nSome tests failed. Please review the chunker implementation.")
