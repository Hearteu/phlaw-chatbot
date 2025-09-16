#!/usr/bin/env python3
"""
Test script to run the crawler with a small sample and generate TXT files
"""
import os
import sys

# Set environment variables for testing
os.environ['YEAR_START'] = '2012'
os.environ['YEAR_END'] = '2012'
os.environ['EXPORT_TXT'] = '1'
os.environ['TXT_EXPORT_DIR'] = 'backend/jurisprudence'
os.environ['CONCURRENCY'] = '2'
os.environ['WRITE_CHUNK'] = '3'

# Add the backend directory to Python path
sys.path.insert(0, 'backend')

from chatbot import crawler as c

def test_crawler():
    print("🧪 Testing Supreme Court E-Library Crawler")
    print("=" * 50)
    
    # Discover case URLs
    print("🔍 Discovering case URLs...")
    items = c.discover_case_urls()
    
    if not items:
        print("❌ No case URLs found!")
        return
    
    print(f"📋 Found {len(items)} case URLs")
    
    # Take only first 2 items for testing
    test_items = items[:2]
    print(f"🧪 Testing with first {len(test_items)} cases:")
    for i, item in enumerate(test_items):
        print(f"  {i+1}. {item['url']}")
    
    print("\n🚀 Starting crawl...")
    
    try:
        import asyncio
        asyncio.run(c.crawl_all(test_items, 'backend/data/cases.jsonl.gz'))
        print("✅ Test crawl completed!")
        
        # List created TXT files
        print("\n📄 TXT Files Created:")
        jurisprudence_dir = 'backend/jurisprudence'
        if os.path.exists(jurisprudence_dir):
            for root, dirs, files in os.walk(jurisprudence_dir):
                for file in files:
                    if file.endswith('.txt'):
                        rel_path = os.path.relpath(os.path.join(root, file), jurisprudence_dir)
                        print(f"  • {rel_path}")
        else:
            print("  No jurisprudence directory found")
            
    except Exception as e:
        print(f"❌ Test crawl failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_crawler()

