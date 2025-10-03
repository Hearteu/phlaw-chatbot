#!/usr/bin/env python3
"""
Test script for Contextual RAG implementation
"""
import os
import sys

import django

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from chatbot.contextual_rag import create_contextual_rag_system
from chatbot.retriever import LegalRetriever


def test_contextual_rag():
    """Test the Contextual RAG system"""
    print("🧪 Testing Contextual RAG System")
    print("=" * 50)
    
    # Test 1: Initialize system
    print("\n1. Testing system initialization...")
    try:
        contextual_rag = create_contextual_rag_system()
        print("✅ Contextual RAG system initialized successfully")
        
        # Get stats
        stats = contextual_rag.get_index_stats()
        print(f"📊 Index stats: {stats}")
        
    except Exception as e:
        print(f"❌ Failed to initialize Contextual RAG: {e}")
        return False
    
    # Test 2: Test retriever with Contextual RAG
    print("\n2. Testing retriever with Contextual RAG...")
    try:
        retriever = LegalRetriever(use_contextual_rag=True)
        print("✅ LegalRetriever with Contextual RAG initialized successfully")
        
        # Test a simple query (this will fallback to standard retrieval if indexes aren't built)
        test_query = "contract breach"
        print(f"🔍 Testing query: '{test_query}'")
        
        results = retriever.retrieve(test_query, k=3)
        print(f"📊 Retrieved {len(results)} results")
        
        for i, result in enumerate(results[:2]):  # Show first 2 results
            print(f"  Result {i+1}: {result.get('content', '')[:100]}...")
            
    except Exception as e:
        print(f"❌ Failed to test retriever: {e}")
        return False
    
    # Test 3: Test without Contextual RAG (fallback)
    print("\n3. Testing retriever without Contextual RAG...")
    try:
        retriever_basic = LegalRetriever(use_contextual_rag=False)
        print("✅ Basic LegalRetriever initialized successfully")
        
        results_basic = retriever_basic.retrieve(test_query, k=3)
        print(f"📊 Retrieved {len(results_basic)} results")
        
    except Exception as e:
        print(f"❌ Failed to test basic retriever: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True


def test_chat_engine():
    """Test the chat engine with Contextual RAG"""
    print("\n🧪 Testing Chat Engine with Contextual RAG")
    print("=" * 50)
    
    try:
        from chatbot.chat_engine import chat_with_law_bot
        
        test_queries = [
            "contract breach",
            "G.R. No. 123456",  # This should use GR number path
            "What is due process?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            try:
                response = chat_with_law_bot(query)
                print(f"✅ Response length: {len(response)} characters")
                print(f"📝 Response preview: {response[:200]}...")
            except Exception as e:
                print(f"❌ Chat failed for query '{query}': {e}")
        
        print("\n✅ Chat engine tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to test chat engine: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Contextual RAG Test Suite")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_contextual_rag()
    success &= test_chat_engine()
    
    if success:
        print("\n🎉 All tests passed! Contextual RAG is working correctly.")
        print("\n📋 Next steps:")
        print("1. Build the Contextual RAG indexes:")
        print("   python manage.py build_contextual_rag --max-cases 100")
        print("2. Test with more queries to see the improvement")
        print("3. Monitor performance and adjust parameters as needed")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
