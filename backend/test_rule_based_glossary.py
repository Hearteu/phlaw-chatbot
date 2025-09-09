#!/usr/bin/env python3
"""
Test script to verify rule-based method works with glossary.json
"""
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from chatbot.rule_based import RuleBasedResponder, GlossaryIndex, _resolve_glossary_files

def test_glossary_loading():
    """Test that glossary.json is loaded correctly"""
    print("=== Testing Glossary Loading ===")
    
    # Test file resolution
    files = _resolve_glossary_files()
    print(f"Resolved glossary files: {files}")
    
    # Test glossary index creation
    glossary = GlossaryIndex.from_files(files)
    print(f"Loaded {len(glossary.entries)} entries")
    
    # Show some sample entries
    print("\nSample entries:")
    for i, entry in enumerate(glossary.entries[:5]):
        print(f"  {i+1}. {entry['term']}: {entry['definition'][:50]}...")
    
    return glossary

def test_definition_queries():
    """Test various definition query patterns"""
    print("\n=== Testing Definition Queries ===")
    
    rb = RuleBasedResponder()
    
    test_queries = [
        "What is abogado?",
        "define chain of custody",
        "meaning of corpus delicti",
        "What is constructive dismissal?",
        "define bill of attainder",
        "meaning of cause of action",
        "What is aberratio ictus?",
        "define acquisitive prescription",
        "meaning of absolute privileged communication",
        "What is accommodation party?",
    ]
    
    for query in test_queries:
        result = rb.answer(query)
        print(f"\nQuery: '{query}'")
        print(f"Result: {result if result else 'None (deferred to retriever)'}")

def test_elements_queries():
    """Test elements queries (though glossary.json doesn't have elements)"""
    print("\n=== Testing Elements Queries ===")
    
    rb = RuleBasedResponder()
    
    test_queries = [
        "elements of chain of custody",
        "elements of constructive dismissal",
        "elements of cause of action",
    ]
    
    for query in test_queries:
        result = rb.answer(query)
        print(f"\nQuery: '{query}'")
        print(f"Result: {result if result else 'None (deferred to retriever)'}")

def test_fuzzy_matching():
    """Test fuzzy matching and search functionality"""
    print("\n=== Testing Fuzzy Matching ===")
    
    rb = RuleBasedResponder()
    
    # Test with typos and variations
    test_queries = [
        "what is abogado",  # exact match
        "define abogado",   # exact match
        "what is abogad",   # typo
        "define chain of custudy",  # typo
        "meaning of corpus delict",  # partial match
        "what is constructive dismisal",  # typo
        "define bill of attaindr",  # typo
        "meaning of accomodation party",  # typo
    ]
    
    for query in test_queries:
        result = rb.answer(query)
        print(f"\nQuery: '{query}'")
        print(f"Result: {result if result else 'None (deferred to retriever)'}")

def test_case_deferral():
    """Test that case references are properly deferred"""
    print("\n=== Testing Case Reference Deferral ===")
    
    rb = RuleBasedResponder()
    
    test_queries = [
        "What is the ruling in G.R. No. 211089?",
        "People v. Dizon",
        "Smith vs. Jones",
        "G.R. No. 123456",
        "What happened in People of the Philippines v. Santos?",
    ]
    
    for query in test_queries:
        result = rb.answer(query)
        print(f"\nQuery: '{query}'")
        print(f"Result: {result if result else 'None (deferred to retriever)'}")

def test_rule_45_65_comparison():
    """Test Rule 45 vs 65 comparison"""
    print("\n=== Testing Rule 45 vs 65 Comparison ===")
    
    rb = RuleBasedResponder()
    
    test_queries = [
        "Rule 45 vs 65",
        "Rule 45 versus 65",
        "Rule 45 v. 65",
        "difference between rule 45 and 65",
    ]
    
    for query in test_queries:
        result = rb.answer(query)
        print(f"\nQuery: '{query}'")
        print(f"Result: {result if result else 'None (deferred to retriever)'}")

def test_glossary_search_functionality():
    """Test the glossary search functionality directly"""
    print("\n=== Testing Glossary Search Functionality ===")
    
    files = _resolve_glossary_files()
    glossary = GlossaryIndex.from_files(files)
    
    test_terms = [
        "abogado",
        "chain of custody", 
        "corpus delicti",
        "constructive dismissal",
        "bill of attainder",
        "cause of action",
        "aberratio ictus",
        "acquisitive prescription",
        "accommodation party",
        "absolute privileged communication",
    ]
    
    for term in test_terms:
        print(f"\nSearching for: '{term}'")
        results = glossary.search(term, k=3)
        if results:
            for i, (score, entry) in enumerate(results):
                print(f"  {i+1}. Score: {score:.1f} - {entry['term']}: {entry['definition'][:60]}...")
        else:
            print("  No results found")

def test_query_intent_detection():
    """Test query intent detection"""
    print("\n=== Testing Query Intent Detection ===")
    
    files = _resolve_glossary_files()
    glossary = GlossaryIndex.from_files(files)
    
    test_queries = [
        "What is abogado?",
        "define chain of custody",
        "meaning of corpus delicti",
        "elements of constructive dismissal",
        "How does constructive dismissal work?",
        "Tell me about abogado",
        "Explain chain of custody",
    ]
    
    for query in test_queries:
        is_definition = glossary.is_definition_query(query)
        print(f"'{query}' -> Definition query: {is_definition}")

def main():
    """Run all tests"""
    print("Testing Rule-Based Method with Glossary.json")
    print("=" * 50)
    
    try:
        # Test glossary loading
        glossary = test_glossary_loading()
        
        # Test various query types
        test_definition_queries()
        test_elements_queries()
        test_fuzzy_matching()
        test_case_deferral()
        test_rule_45_65_comparison()
        
        # Test internal functionality
        test_glossary_search_functionality()
        test_query_intent_detection()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

