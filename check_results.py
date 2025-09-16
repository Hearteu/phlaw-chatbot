#!/usr/bin/env python3
"""
Check the results of the crawler
"""
import gzip
import json
import os

def check_results():
    print("📊 Checking crawler results...")
    
    # Check JSONL file
    print("\n📄 Checking cases.jsonl.gz...")
    if os.path.exists('backend/data/cases.jsonl.gz'):
        with gzip.open('backend/data/cases.jsonl.gz', 'rt', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Total records in JSONL: {len(lines)}")
            
            if lines:
                print("\nFirst record:")
                first_record = json.loads(lines[0])
                print(f"  Title: {first_record.get('title', 'N/A')}")
                print(f"  G.R. Number: {first_record.get('gr_number', 'N/A')}")
                print(f"  Year: {first_record.get('promulgation_year', 'N/A')}")
    else:
        print("❌ cases.jsonl.gz does not exist")
    
    # Check TXT files
    print("\n📁 Checking TXT files in jurisprudence folder...")
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
        print("❌ No jurisprudence folder found")

if __name__ == "__main__":
    check_results()

