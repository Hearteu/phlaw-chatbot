#!/usr/bin/env python3
"""
Simple test to run crawler and create TXT files
"""
import os
import sys
import asyncio
import gzip
import json

# Set environment variables
os.environ['YEAR_START'] = '2010'
os.environ['YEAR_END'] = '2010'
os.environ['EXPORT_TXT'] = '1'
os.environ['TXT_EXPORT_DIR'] = 'backend/jurisprudence'
os.environ['CONCURRENCY'] = '1'
os.environ['WRITE_CHUNK'] = '1'

# Add backend to path
sys.path.insert(0, 'backend')

def test_crawler():
    print("🧪 Testing crawler with real data...")
    
    try:
        from chatbot import crawler as c
        
        # Discover URLs
        print("🔍 Discovering URLs...")
        items = c.discover_case_urls()
        print(f"Found {len(items)} URLs")
        
        if not items:
            print("❌ No URLs found!")
            return
        
        # Take first 2 items
        test_items = items[:2]
        print(f"Testing with {len(test_items)} items:")
        for i, item in enumerate(test_items):
            print(f"  {i+1}. {item['url']}")
        
        # Run crawler
        print("\n🚀 Running crawler...")
        asyncio.run(c.crawl_all(test_items, 'backend/data/cases.jsonl.gz'))
        print("✅ Crawler completed!")
        
        # Check results
        print("\n📊 Checking results...")
        
        # Check JSONL file
        if os.path.exists('backend/data/cases.jsonl.gz'):
            with gzip.open('backend/data/cases.jsonl.gz', 'rt', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"📄 cases.jsonl.gz has {len(lines)} records")
                
                if lines:
                    first_record = json.loads(lines[0])
                    print(f"First record title: {first_record.get('title', 'N/A')}")
                    print(f"First record G.R.: {first_record.get('gr_number', 'N/A')}")
        
        # Check TXT files
        print("\n📁 TXT files in jurisprudence folder:")
        jurisprudence_dir = 'backend/jurisprudence'
        if os.path.exists(jurisprudence_dir):
            txt_count = 0
            for root, dirs, files in os.walk(jurisprudence_dir):
                for file in files:
                    if file.endswith('.txt'):
                        rel_path = os.path.relpath(os.path.join(root, file), jurisprudence_dir)
                        print(f"  • {rel_path}")
                        txt_count += 1
            print(f"\n📊 Total TXT files: {txt_count}")
        else:
            print("  No jurisprudence folder found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_crawler()


