#!/usr/bin/env python3
"""
Migrate data from jurisprudence to jurisprudence_contextual collection
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


def check_both_collections():
    """Check both jurisprudence and jurisprudence_contextual collections"""
    print("🔍 Checking Both Collections")
    print("="*50)
    
    try:
        qdrant = _get_cached_qdrant_client()
        
        collections = ["jurisprudence", "jurisprudence_contextual"]
        
        for collection_name in collections:
            print(f"\n📁 Collection: {collection_name}")
            
            if qdrant.collection_exists(collection_name):
                info = qdrant.get_collection(collection_name)
                print(f"   ✅ Exists - Points: {info.points_count:,}")
                print(f"   Status: {info.status}")
                
                # Sample a few points to check structure
                try:
                    sample_points = qdrant.scroll(
                        collection_name=collection_name,
                        limit=3,
                        with_payload=True
                    )[0]
                    
                    if sample_points:
                        payload = sample_points[0].payload
                        print(f"   Sample keys: {list(payload.keys())}")
                        
                        # Check for GR numbers in the sample
                        gr_numbers = []
                        for point in sample_points[:3]:
                            gr_num = point.payload.get('gr_number', '')
                            if gr_num:
                                gr_numbers.append(gr_num)
                        
                        if gr_numbers:
                            print(f"   Sample GR numbers: {gr_numbers}")
                        
                except Exception as e:
                    print(f"   ⚠️ Could not sample points: {e}")
                    
            else:
                print(f"   ❌ Does not exist")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to check collections: {e}")
        return False


def search_for_gr_number(gr_number: str):
    """Search for a specific GR number in both collections"""
    print(f"\n🔍 Searching for GR No. {gr_number}")
    print("="*50)
    
    try:
        qdrant = _get_cached_qdrant_client()
        
        collections = ["jurisprudence", "jurisprudence_contextual"]
        
        for collection_name in collections:
            if not qdrant.collection_exists(collection_name):
                print(f"❌ Collection {collection_name} does not exist")
                continue
                
            print(f"\n📁 Searching in {collection_name}:")
            
            try:
                # Search by GR number filter
                results = qdrant.scroll(
                    collection_name=collection_name,
                    scroll_filter={
                        "must": [
                            {
                                "key": "gr_number",
                                "match": {"value": gr_number}
                            }
                        ]
                    },
                    limit=5,
                    with_payload=True
                )[0]
                
                if results:
                    print(f"   ✅ Found {len(results)} results")
                    for i, result in enumerate(results, 1):
                        payload = result.payload
                        title = payload.get('title', '') or payload.get('case_title', '')
                        print(f"   {i}. {title[:60]}..." if title else f"   {i}. [No title]")
                        print(f"      GR: {payload.get('gr_number', 'N/A')}")
                        print(f"      Special: {payload.get('special_number', 'N/A')}")
                else:
                    print(f"   ❌ No results found")
                    
            except Exception as e:
                print(f"   ⚠️ Search failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to search: {e}")
        return False


def suggest_solution():
    """Suggest solutions based on the findings"""
    print("\n💡 SOLUTION RECOMMENDATIONS")
    print("="*50)
    
    try:
        qdrant = _get_cached_qdrant_client()
        
        # Check if jurisprudence collection has more data
        jurisprudence_exists = qdrant.collection_exists("jurisprudence")
        contextual_exists = qdrant.collection_exists("jurisprudence_contextual")
        
        if jurisprudence_exists and contextual_exists:
            jur_info = qdrant.get_collection("jurisprudence")
            context_info = qdrant.get_collection("jurisprudence_contextual")
            
            print(f"📊 Data comparison:")
            print(f"   jurisprudence: {jur_info.points_count:,} points")
            print(f"   jurisprudence_contextual: {context_info.points_count:,} points")
            
            if jur_info.points_count > context_info.points_count:
                print(f"\n🔧 RECOMMENDED SOLUTION:")
                print(f"   The original 'jurisprudence' collection has more data.")
                print(f"   You have two options:")
                print(f"   1. Use the original collection temporarily:")
                print(f"      - Change collection back to 'jurisprudence' in the code")
                print(f"      - Or run the embedding process to populate 'jurisprudence_contextual'")
                print(f"   2. Rebuild the contextual collection:")
                print(f"      - Run the embedding process with contextual RAG")
                print(f"      - This will create the optimized collection with all data")
        
        elif jurisprudence_exists and not contextual_exists:
            print(f"🔧 RECOMMENDED SOLUTION:")
            print(f"   Only 'jurisprudence' exists. Create the contextual collection:")
            print(f"   1. Run the embedding process to create 'jurisprudence_contextual'")
            print(f"   2. Or temporarily use 'jurisprudence' collection")
            
        elif not jurisprudence_exists and contextual_exists:
            print(f"🔧 RECOMMENDED SOLUTION:")
            print(f"   Only 'jurisprudence_contextual' exists. It may be incomplete:")
            print(f"   1. Check if your data source is correct")
            print(f"   2. Rebuild the collection with all available data")
            
        else:
            print(f"🔧 RECOMMENDED SOLUTION:")
            print(f"   No collections found. You need to:")
            print(f"   1. Run the embedding process to create collections")
            print(f"   2. Ensure your data source is available")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to suggest solutions: {e}")
        return False


def main():
    """Main function"""
    print("🔄 Collection Migration Helper")
    print("="*60)
    
    # Check both collections
    check_both_collections()
    
    # Search for the specific GR number that failed
    search_for_gr_number("13744")
    
    # Suggest solutions
    suggest_solution()
    
    print("\n" + "="*60)
    print("📋 NEXT STEPS:")
    print("="*60)
    print("1. If 'jurisprudence' has more data, temporarily use it:")
    print("   - Change collection back to 'jurisprudence' in retriever.py")
    print("   - Or run the embedding process to populate 'jurisprudence_contextual'")
    print("")
    print("2. To rebuild the contextual collection:")
    print("   - Run: python backend/embed.py")
    print("   - Or use the contextual RAG building process")
    print("")
    print("3. Test with the collection checker:")
    print("   - Run: python backend/check_collection_setup.py")
    print("="*60)


if __name__ == "__main__":
    main()
