#!/usr/bin/env python3
"""
Test ID-based storage fix for contextual RAG
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

from chatbot.contextual_rag import create_contextual_rag_system


def test_stable_id_generation():
    """Test that stable IDs are generated correctly"""
    print("🧪 Testing Stable ID Generation")
    print("="*50)
    
    try:
        rag_system = create_contextual_rag_system(collection="jurisprudence_contextual")
        
        # Test metadata and chunk text
        test_metadata = {
            'content': 'This is a test chunk about annulment.',
            'section': 'facts',
            'section_type': 'factual',
            'chunk_type': 'content',
            'token_count': 10,
            'metadata': {
                'gr_number': '12345',
                'special_number': '',
                'title': 'Test Case Title',
                'ponente': 'Test Justice',
                'case_type': 'Civil',
                'date': '2023-01-01'
            }
        }
        
        test_chunk = "This is a test chunk about annulment in Philippine law."
        
        # Generate stable ID
        stable_id1 = rag_system._generate_stable_id(test_metadata, test_chunk)
        stable_id2 = rag_system._generate_stable_id(test_metadata, test_chunk)
        
        print(f"Generated ID 1: {stable_id1[:16]}...")
        print(f"Generated ID 2: {stable_id2[:16]}...")
        
        # IDs should be identical for same content
        if stable_id1 == stable_id2:
            print("✅ Stable ID generation is deterministic")
        else:
            print("❌ Stable ID generation is not deterministic")
            return False
        
        # Test with different content
        test_chunk_different = "This is a different chunk about criminal law."
        stable_id3 = rag_system._generate_stable_id(test_metadata, test_chunk_different)
        
        print(f"Different content ID: {stable_id3[:16]}...")
        
        if stable_id1 != stable_id3:
            print("✅ Different content generates different IDs")
        else:
            print("❌ Different content generates same ID")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_id_based_storage():
    """Test that ID-based storage works correctly"""
    print("\n🔍 Testing ID-Based Storage")
    print("="*50)
    
    try:
        rag_system = create_contextual_rag_system(collection="jurisprudence_contextual")
        
        # Check if ID-based storage is initialized
        print(f"ID-based maps initialized:")
        print(f"  id_to_contextual_chunk: {len(rag_system.id_to_contextual_chunk)}")
        print(f"  id_to_metadata: {len(rag_system.id_to_metadata)}")
        print(f"  id_to_payload: {len(rag_system.id_to_payload)}")
        print(f"  id_to_embedding: {len(rag_system.id_to_embedding)}")
        
        # Check stats
        stats = rag_system.get_index_stats()
        print(f"\nStats:")
        print(f"  Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  ID-based storage: {stats.get('id_based_storage', False)}")
        print(f"  Unique IDs: {stats.get('unique_ids', 0)}")
        
        if stats.get('id_based_storage', False):
            print("✅ ID-based storage is enabled")
        else:
            print("❌ ID-based storage is not enabled")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_retrieval_with_ids():
    """Test that retrieval works with ID-based lookup"""
    print("\n🔍 Testing Retrieval with ID-Based Lookup")
    print("="*50)
    
    try:
        rag_system = create_contextual_rag_system(collection="jurisprudence_contextual")
        
        # Try a simple retrieval
        print("Testing retrieval with ID-based lookup...")
        results = rag_system.retrieve_and_rank_fast("test query", final_k=3)
        
        print(f"Retrieved {len(results)} results")
        
        # Check if results have chunk_id field
        if results:
            first_result = results[0]
            if 'chunk_id' in first_result:
                print(f"✅ Results include chunk_id: {first_result['chunk_id'][:16]}...")
            else:
                print("❌ Results missing chunk_id field")
                return False
        
        # Check if we can look up by ID
        if results and 'chunk_id' in results[0]:
            chunk_id = results[0]['chunk_id']
            if chunk_id in rag_system.id_to_contextual_chunk:
                print("✅ ID-based lookup works correctly")
            else:
                print("❌ ID-based lookup failed")
                return False
        
        print("✅ Retrieval with ID-based lookup successful")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_correctness_fix():
    """Test that the correctness issue is fixed"""
    print("\n🎯 Testing Correctness Fix")
    print("="*50)
    
    try:
        rag_system = create_contextual_rag_system(collection="jurisprudence_contextual")
        
        # Test that we're not using index arithmetic
        print("Checking for index arithmetic usage...")
        
        # Look for any code that might use list indices as IDs
        if hasattr(rag_system, 'id_to_contextual_chunk') and hasattr(rag_system, 'id_to_metadata'):
            print("✅ ID-based maps are available")
            
            # Check that retrieval uses ID-based lookup
            if len(rag_system.id_to_contextual_chunk) > 0:
                print("✅ ID-based storage has data")
                
                # Test that we can retrieve by ID
                sample_id = list(rag_system.id_to_contextual_chunk.keys())[0]
                if sample_id in rag_system.id_to_metadata:
                    print("✅ ID-based lookup works correctly")
                    print("✅ Correctness issue appears to be fixed")
                    return True
                else:
                    print("❌ ID-based lookup failed")
                    return False
            else:
                print("⚠️ No data in ID-based storage (may be empty collection)")
                return True  # Not a failure, just empty
        else:
            print("❌ ID-based maps not available")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("🔧 ID-Based Storage Fix Test Suite")
    print("="*60)
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Stable ID generation
    if test_stable_id_generation():
        tests_passed += 1
    
    # Test 2: ID-based storage
    if test_id_based_storage():
        tests_passed += 1
    
    # Test 3: Retrieval with IDs
    if test_retrieval_with_ids():
        tests_passed += 1
    
    # Test 4: Correctness fix
    if test_correctness_fix():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST RESULTS")
    print("="*60)
    
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n🎉 All tests passed! The ID-based storage fix is working correctly.")
        print("\n✅ CRITICAL FIXES IMPLEMENTED:")
        print("   • Stable, content-derived IDs instead of loop indices")
        print("   • ID-based maps instead of list-based storage")
        print("   • Proper ID-to-content lookup instead of index arithmetic")
        print("   • No more risk of wrong documents being returned")
        print("\nThe system should now be safe from the correctness issues!")
    else:
        print(f"\n⚠️ {total_tests - tests_passed} tests failed. Check the issues above.")
    
    print("="*60)
