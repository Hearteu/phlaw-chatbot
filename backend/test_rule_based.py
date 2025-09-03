#!/usr/bin/env python3
"""
Test script for rule_based.py without Docker dependencies
"""
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot.rule_based import RuleBasedResponder

def test_rule_based_system():
    """Test the rule-based system with various queries"""
    print("🧪 Testing PHLaw-Chatbot Rule-Based System")
    print("=" * 50)
    
    rb = RuleBasedResponder()
    
    # Test cases
    test_queries = [
        # Case digest requests
        "Make a case digest of this case",
        "I need a full digest",
        "Can you create a case digest?",
        
        # FAQ queries
        "Rule 45 vs 65",
        "What is the court hierarchy?",
        "Standards of review",
        
        # Definition queries
        "What is certiorari?",
        "Define grave abuse of discretion",
        "What is estoppel?",
        
        # Elements queries
        "Elements of probable cause",
        "Tests of warrantless search exceptions",
        
        # Specific case queries (should defer to LLM)
        "What is the ruling in G.R. No. 211089?",
        "People v. Dizon (2018)",
        "G.R. No. 123456",
        
        # Generic queries (should defer)
        "Facts and issues please",
        "What is the Civil Code?",
        "Supreme Court cases about contracts",
        
        # Edge cases
        "",
        "   ",
        None,
    ]
    
    print(f"Testing {len(test_queries)} queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i:2d}. Query: {repr(query)}")
        
        try:
            result = rb.answer(query)
            if result:
                print(f"    ✅ Rule-based response:")
                print(f"    {result[:100]}{'...' if len(result) > 100 else ''}")
            else:
                print(f"    ⏭️  Deferred to LLM/retrieval")
        except Exception as e:
            print(f"    ❌ Error: {e}")
        
        print()
    
    print("=" * 50)
    print("✅ Rule-based system test completed!")

def test_integration_with_chat_engine():
    """Test integration with chat_engine.py (without actual LLM calls)"""
    print("\n🔗 Testing Integration with chat_engine.py")
    print("=" * 50)
    
    try:
        # Import the chat engine
        from chatbot.chat_engine import chat_with_law_bot
        
        # Test queries that should be handled by rule-based system
        rule_based_queries = [
            "What is certiorari?",
            "Rule 45 vs 65",
            "Make a case digest",
        ]
        
        print("Testing rule-based queries (should get instant responses):")
        for query in rule_based_queries:
            print(f"\nQuery: {query}")
            try:
                # This will try to use the full system, but rule-based should handle it first
                response = chat_with_law_bot(query, [])
                print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
            except Exception as e:
                print(f"Error (expected if no LLM/DB): {e}")
        
    except ImportError as e:
        print(f"Could not import chat_engine: {e}")
        print("This is expected if dependencies are not fully set up")

if __name__ == "__main__":
    test_rule_based_system()
    test_integration_with_chat_engine()
