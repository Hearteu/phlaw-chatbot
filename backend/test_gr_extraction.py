# test_gr_extraction.py — Test GR number extraction improvements
import sys

sys.path.insert(0, '.')

from chatbot.chat_engine import _extract_gr_number


def test_gr_extraction():
    """Test that GR number extraction handles plural 'NOS.' correctly"""
    
    test_cases = [
        ("G.R. NOS. 151809-12", "151809-12"),
        ("G.R. No. 123456", "123456"),
        ("GR NOS. 151809-12", "151809-12"),
        ("G.R. NOS 151809-12", "151809-12"),
        ("G.R.NOS.151809-12", "151809-12"),
        ("gr nos 151809-12", "151809-12"),
        ("what is G.R. NOS. 151809-12 about?", "151809-12"),
    ]
    
    print("=" * 70)
    print("GR NUMBER EXTRACTION TEST")
    print("=" * 70)
    
    all_passed = True
    
    for query, expected in test_cases:
        result = _extract_gr_number(query)
        passed = result == expected
        
        status = "PASS" if passed else "FAIL"
        print(f"\n{status}: {query:50} -> {result:15} (expected: {expected})")
        
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("SUCCESS: All GR extraction tests passed!")
        print("The chatbot can now handle 'G.R. NOS. 151809-12' format")
    else:
        print("FAILURE: Some tests failed")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    test_gr_extraction()
