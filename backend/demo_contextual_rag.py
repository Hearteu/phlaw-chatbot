#!/usr/bin/env python3
"""
Demo script for Contextual RAG implementation
This script demonstrates the Contextual RAG system with sample queries
"""
import os
import sys

import django

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from chatbot.chat_engine import chat_with_law_bot


def demo_queries():
    """Run demo queries to showcase Contextual RAG capabilities"""
    
    print("🎯 Contextual RAG Demo")
    print("=" * 60)
    print("This demo showcases the Contextual RAG system with various query types.")
    print("The system uses hybrid search (vector + BM25) with reranking for better results.\n")
    
    # Sample queries that demonstrate different capabilities
    demo_queries = [
        {
            "query": "What is due process in criminal cases?",
            "description": "Conceptual legal question - tests semantic understanding"
        },
        {
            "query": "contract breach damages",
            "description": "Keyword search - tests BM25 and hybrid search"
        },
        {
            "query": "G.R. No. 123456",
            "description": "GR number search - tests exact match path"
        },
        {
            "query": "estafa criminal law",
            "description": "Legal concept + area - tests contextual understanding"
        },
        {
            "query": "What are the elements of theft?",
            "description": "Structured legal question - tests comprehensive retrieval"
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n{'='*60}")
        print(f"DEMO {i}: {demo['description']}")
        print(f"Query: '{demo['query']}'")
        print(f"{'='*60}")
        
        try:
            print("🔍 Processing query...")
            response = chat_with_law_bot(demo['query'])
            
            print(f"✅ Response received ({len(response)} characters)")
            print("\n📝 Response:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print("This might be expected if Contextual RAG indexes aren't built yet.")
        
        # Pause between queries for readability
        if i < len(demo_queries):
            input("\nPress Enter to continue to the next demo...")
    
    print(f"\n{'='*60}")
    print("🎉 Demo completed!")
    print(f"{'='*60}")
    
    # Show next steps
    print("\n📋 Next Steps:")
    print("1. Build Contextual RAG indexes:")
    print("   python manage.py build_contextual_rag --max-cases 100")
    print("\n2. Run the test suite:")
    print("   python test_contextual_rag.py")
    print("\n3. Monitor performance and adjust parameters as needed")
    print("\n4. For production, build full indexes:")
    print("   python manage.py build_contextual_rag")


def show_system_info():
    """Show system information and status"""
    print("\n🔧 System Information")
    print("=" * 40)
    
    try:
        from chatbot.retriever import LegalRetriever

        # Test basic retriever
        print("Testing basic retriever...")
        retriever_basic = LegalRetriever(use_contextual_rag=False)
        print("✅ Basic retriever: Available")
        
        # Test contextual RAG retriever
        print("Testing Contextual RAG retriever...")
        try:
            retriever_contextual = LegalRetriever(use_contextual_rag=True)
            print("✅ Contextual RAG retriever: Available")
        except Exception as e:
            print(f"⚠️ Contextual RAG retriever: Not available ({e})")
            print("   This is expected if indexes aren't built yet.")
        
    except Exception as e:
        print(f"❌ System check failed: {e}")
    
    print("\n📊 Available Models:")
    try:
        from chatbot.model_cache import get_cached_embedding_model
        model = get_cached_embedding_model()
        print(f"✅ Embedding model: {model}")
    except Exception as e:
        print(f"❌ Embedding model: {e}")
    
    try:
        from chatbot.docker_model_client import get_docker_model_client
        client = get_docker_model_client()
        if client.is_available:
            print("✅ Docker model client: Available")
        else:
            print("⚠️ Docker model client: Not available")
    except Exception as e:
        print(f"❌ Docker model client: {e}")


if __name__ == "__main__":
    print("🚀 Contextual RAG Demo Suite")
    print("=" * 60)
    
    # Show system information first
    show_system_info()
    
    # Ask user if they want to run the demo
    print("\n" + "="*60)
    response = input("Would you like to run the demo queries? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        demo_queries()
    else:
        print("\nDemo skipped. You can run it later with:")
        print("python demo_contextual_rag.py")
        print("\nOr build the indexes first:")
        print("python manage.py build_contextual_rag --max-cases 100")
