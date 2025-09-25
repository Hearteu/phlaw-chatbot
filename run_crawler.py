#!/usr/bin/env python3
"""
Run the crawler to scrape real data and create TXT files
"""
import os
import sys
import asyncio
from datetime import datetime

# Set environment variables
os.environ['YEAR_START'] = '2010'
os.environ['YEAR_END'] = '2010'
os.environ['EXPORT_TXT'] = '1'
os.environ['TXT_EXPORT_DIR'] = 'backend/jurisprudence'
os.environ['CONCURRENCY'] = '2'
os.environ['WRITE_CHUNK'] = '10'

# Add backend to path
sys.path.insert(0, 'backend')

from chatbot import crawler as c

async def run_crawler():
    print("🚀 Starting REAL Supreme Court E-Library Crawler")
    print("=" * 60)
    print(f"📅 Year range: {os.environ['YEAR_START']}-{os.environ['YEAR_END']}")
    print(f"📁 TXT export: {'Enabled' if os.environ['EXPORT_TXT'] == '1' else 'Disabled'}")
    print(f"📂 TXT directory: {os.environ['TXT_EXPORT_DIR']}")
    print("=" * 60)
    
    # Discover case URLs
    print("🔍 Discovering case URLs...")
    items = c.discover_case_urls()
    
    if not items:
        print("❌ No case URLs found!")
        return
    
    print(f"📋 Found {len(items)} case URLs to crawl")
    
    # For testing, let's start with first 5 cases
    test_items = items[:5]
    print(f"🧪 Starting with first {len(test_items)} cases for testing...")
    
    for i, item in enumerate(test_items):
        print(f"  {i+1}. {item['url']}")
    
    print("\n🔄 Starting crawl process...")
    start_time = datetime.now()
    
    try:
        await c.crawl_all(test_items, 'backend/data/cases.jsonl.gz')
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Crawl completed in {duration:.1f} seconds!")
        
        # Show summary of TXT files created
        print("\n📄 TXT Files Created:")
        jurisprudence_dir = os.environ['TXT_EXPORT_DIR']
        if os.path.exists(jurisprudence_dir):
            txt_count = 0
            for root, dirs, files in os.walk(jurisprudence_dir):
                for file in files:
                    if file.endswith('.txt'):
                        rel_path = os.path.relpath(os.path.join(root, file), jurisprudence_dir)
                        print(f"  • {rel_path}")
                        txt_count += 1
            print(f"\n📊 Total TXT files created: {txt_count}")
        else:
            print("  No jurisprudence directory found")
            
    except Exception as e:
        print(f"❌ Crawl failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_crawler())


