#!/usr/bin/env python3
"""
Check and setup jurisprudence_contextual collection
"""
import os
import sys

import django
from dotenv import load_dotenv

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from chatbot.retriever import _get_cached_qdrant_client
from qdrant_client import QdrantClient


def check_collections():
    """Check available collections and their status"""
    print("🔍 Checking Qdrant Collections")
    print("="*50)
    
    try:
        qdrant = _get_cached_qdrant_client()
        
        # Get all collections
        collections = qdrant.get_collections()
        print(f"📊 Total collections found: {len(collections.collections)}")
        
        # Check each collection
        for collection_info in collections.collections:
            name = collection_info.name
            print(f"\n📁 Collection: {name}")
            
            try:
                # Get detailed info
                info = qdrant.get_collection(name)
                print(f"   Points: {info.points_count:,}")
                print(f"   Status: {info.status}")
                print(f"   Vector size: {info.config.params.vectors.size}")
                print(f"   Distance: {info.config.params.vectors.distance}")
                
                # Check if it's the contextual collection
                if name == "jurisprudence_contextual":
                    print("   ✅ This is the optimized contextual collection!")
                elif name == "jurisprudence":
                    print("   📝 This is the original collection")
                
            except Exception as e:
                print(f"   ❌ Error getting collection info: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return False


def check_contextual_collection():
    """Specifically check the jurisprudence_contextual collection"""
    print("\n🎯 Checking jurisprudence_contextual Collection")
    print("="*50)
    
    try:
        qdrant = _get_cached_qdrant_client()
        
        if qdrant.collection_exists("jurisprudence_contextual"):
            print("✅ jurisprudence_contextual collection exists")
            
            info = qdrant.get_collection("jurisprudence_contextual")
            print(f"📊 Points: {info.points_count:,}")
            print(f"📊 Status: {info.status}")
            
            if info.points_count > 0:
                print("✅ Collection has data - ready for use!")
                
                # Sample a few points to check structure
                try:
                    sample_points = qdrant.scroll(
                        collection_name="jurisprudence_contextual",
                        limit=3,
                        with_payload=True
                    )[0]
                    
                    print(f"\n📄 Sample point structure:")
                    if sample_points:
                        payload = sample_points[0].payload
                        print(f"   Keys: {list(payload.keys())}")
                        print(f"   Has contextual: {payload.get('contextual', False)}")
                        print(f"   Content length: {len(payload.get('content', ''))}")
                        print(f"   Section type: {payload.get('section_type', 'N/A')}")
                
                except Exception as e:
                    print(f"⚠️ Could not sample points: {e}")
                
            else:
                print("⚠️ Collection exists but is empty - needs indexing")
                
        else:
            print("❌ jurisprudence_contextual collection does not exist")
            print("💡 You may need to:")
            print("   1. Run the embedding process with contextual RAG")
            print("   2. Or migrate from the original jurisprudence collection")
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking contextual collection: {e}")
        return False


def test_retriever_with_contextual():
    """Test the retriever with the contextual collection"""
    print("\n🚀 Testing Retriever with Contextual Collection")
    print("="*50)
    
    try:
        from chatbot.retriever import LegalRetriever

        # Initialize retriever with contextual collection
        retriever = LegalRetriever(collection="jurisprudence_contextual", use_contextual_rag=True)
        print("✅ Retriever initialized successfully")
        
        # Test a simple query
        test_query = "What is annulment in Philippine law?"
        print(f"🔍 Testing query: '{test_query}'")
        
        results = retriever.retrieve(test_query, k=3)
        print(f"📊 Retrieved {len(results)} results")
        
        if results:
            print("✅ Retrieval successful!")
            for i, result in enumerate(results[:2], 1):
                title = result.get('title', '') or result.get('case_title', '')
                print(f"   {i}. {title[:60]}..." if title else f"   {i}. [No title]")
        else:
            print("⚠️ No results found - collection may be empty")
        
        return True
        
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        return False


def main():
    """Main function to check collection setup"""
    print("🧪 Collection Setup Checker")
    print("="*60)
    
    # Check 1: All collections
    collections_ok = check_collections()
    
    # Check 2: Contextual collection specifically
    contextual_ok = check_contextual_collection()
    
    # Check 3: Test retriever
    retriever_ok = test_retriever_with_contextual()
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    if collections_ok:
        print("✅ Qdrant connection: OK")
    else:
        print("❌ Qdrant connection: FAILED")
    
    if contextual_ok:
        print("✅ Collection check: OK")
    else:
        print("❌ Collection check: FAILED")
    
    if retriever_ok:
        print("✅ Retriever test: OK")
    else:
        print("❌ Retriever test: FAILED")
    
    if collections_ok and contextual_ok and retriever_ok:
        print("\n🎉 All checks passed! The jurisprudence_contextual collection is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Run your chatbot normally - it will use the optimized collection")
        print("   2. Run 'python test_performance.py' to test performance improvements")
        print("   3. Monitor performance with the built-in monitoring tools")
    else:
        print("\n⚠️ Some checks failed. You may need to:")
        print("   1. Ensure Qdrant is running")
        print("   2. Create the jurisprudence_contextual collection")
        print("   3. Index your data with contextual RAG")
    
    print("="*60)


if __name__ == "__main__":
    main()
