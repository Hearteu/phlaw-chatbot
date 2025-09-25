#!/usr/bin/env python3
"""
Manual demo of the crawler TXT export functionality
"""
import os
import sys
import asyncio
import gzip
import json
from datetime import datetime

# Set environment variables
os.environ['YEAR_START'] = '2010'
os.environ['YEAR_END'] = '2010'
os.environ['EXPORT_TXT'] = '1'
os.environ['TXT_EXPORT_DIR'] = 'backend/jurisprudence'
os.environ['CONCURRENCY'] = '1'
os.environ['WRITE_CHUNK'] = '1'

# Add backend to path
sys.path.insert(0, 'backend')

def create_sample_real_data():
    """Create sample data that mimics what the crawler would produce"""
    print("🔧 Creating sample real data to demonstrate TXT export...")
    
    # Sample real case data (mimicking what crawler.py would produce)
    sample_cases = [
        {
            "id": "sha256:abc123",
            "gr_number": "G.R. No. 190582",
            "gr_numbers": ["G.R. No. 190582"],
            "title": "PEOPLE OF THE PHILIPPINES vs. MARIO SANTOS",
            "case_title": "PEOPLE OF THE PHILIPPINES vs. MARIO SANTOS",
            "promulgation_date": "2010-03-15",
            "promulgation_year": 2010,
            "promulgation_month": "march",
            "court": "Supreme Court",
            "source_url": "https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/1/53585",
            "clean_version": "v2.0",
            "checksum": "sha256:abc123",
            "crawl_ts": datetime.now().isoformat() + "Z",
            "ponente": "Justice Antonio Carpio",
            "division": "First Division",
            "en_banc": False,
            "sections": {
                "header": "[G.R. No. 190582, March 15, 2010]\nPEOPLE OF THE PHILIPPINES, Plaintiff-Appellee,\nvs.\nMARIO SANTOS, Accused-Appellant.",
                "body": "This is a criminal case for murder. The accused-appellant Mario Santos was charged with killing Juan Dela Cruz on the night of January 15, 2008. The trial court found him guilty and sentenced him to reclusion perpetua. The Court of Appeals affirmed the decision. The accused-appellant now appeals to this Court.",
                "ruling": "WHEREFORE, the appeal is hereby DENIED. The decision of the Court of Appeals is AFFIRMED. The accused-appellant Mario Santos is found GUILTY beyond reasonable doubt of the crime of murder and is sentenced to suffer the penalty of reclusion perpetua. SO ORDERED.",
                "facts": "This case involves the killing of Juan Dela Cruz on the night of January 15, 2008. The prosecution presented witnesses who testified that they saw the accused-appellant Mario Santos stab the victim with a knife. The defense claimed alibi, stating that the accused was in another city at the time of the incident.",
                "issues": "Whether the accused-appellant is guilty beyond reasonable doubt of the crime of murder.",
                "arguments": "The Court finds that the prosecution has successfully established the guilt of the accused beyond reasonable doubt. The testimonies of the prosecution witnesses are credible and consistent. The defense of alibi cannot prevail over positive identification by credible witnesses."
            },
            "clean_text": "This is the complete case text with all the details about the murder case involving Mario Santos and Juan Dela Cruz. The case was decided by the Supreme Court on March 15, 2010.",
            "has_gr_number": True,
            "case_type": "regular",
            "case_subtype": "criminal",
            "is_administrative": False,
            "is_regular_case": True,
            "quality_metrics": {
                "has_facts": True,
                "has_issues": True,
                "has_ruling": True,
                "has_arguments": True,
                "text_length": 500,
                "sections_count": 5
            }
        },
        {
            "id": "sha256:def456",
            "gr_number": "G.R. No. 190583",
            "gr_numbers": ["G.R. No. 190583"],
            "title": "SMITH CORPORATION vs. JONES ENTERPRISES",
            "case_title": "SMITH CORPORATION vs. JONES ENTERPRISES",
            "promulgation_date": "2010-06-20",
            "promulgation_year": 2010,
            "promulgation_month": "june",
            "court": "Supreme Court",
            "source_url": "https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/1/53531",
            "clean_version": "v2.0",
            "checksum": "sha256:def456",
            "crawl_ts": datetime.now().isoformat() + "Z",
            "ponente": "Justice Renato Corona",
            "division": "Second Division",
            "en_banc": False,
            "sections": {
                "header": "[G.R. No. 190583, June 20, 2010]\nSMITH CORPORATION, Petitioner,\nvs.\nJONES ENTERPRISES, Respondent.",
                "body": "This is a civil case involving a contract dispute. The petitioner Smith Corporation seeks to enforce the terms of the service agreement against the respondent Jones Enterprises.",
                "ruling": "WHEREFORE, the petition is hereby GRANTED. The decision of the Court of Appeals is REVERSED and SET ASIDE. The contract between the parties is declared valid and binding. SO ORDERED.",
                "facts": "This case involves a contract dispute between Smith Corporation and Jones Enterprises. The parties entered into a service agreement in 2008. Jones Enterprises failed to deliver the services as agreed, leading to this legal action.",
                "issues": "Whether the contract between the parties is valid and enforceable.",
                "arguments": "The Court finds that the contract is valid and binding. The terms and conditions were clearly stated and agreed upon by both parties. The failure of Jones Enterprises to perform its obligations constitutes a breach of contract."
            },
            "clean_text": "This is the complete case text with all the details about the contract dispute between Smith Corporation and Jones Enterprises. The case was decided by the Supreme Court on June 20, 2010.",
            "has_gr_number": True,
            "case_type": "regular",
            "case_subtype": "civil",
            "is_administrative": False,
            "is_regular_case": True,
            "quality_metrics": {
                "has_facts": True,
                "has_issues": True,
                "has_ruling": True,
                "has_arguments": True,
                "text_length": 400,
                "sections_count": 4
            }
        }
    ]
    
    # Import the TXT export function
    from chatbot.crawler import export_record_as_txt
    
    # Create TXT files for each case
    print("📄 Creating TXT files...")
    for i, case in enumerate(sample_cases, 1):
        print(f"  Processing case {i}: {case['title']}")
        export_record_as_txt(case)
    
    # Also save to JSONL file
    print("💾 Saving to cases.jsonl.gz...")
    with gzip.open('backend/data/cases.jsonl.gz', 'at', encoding='utf-8') as f:
        for case in sample_cases:
            f.write(json.dumps(case) + '\n')
    
    print("✅ Sample data created successfully!")
    
    # Show results
    print("\n📊 Results:")
    print(f"📄 JSONL records: {len(sample_cases)}")
    
    # Count TXT files
    jurisprudence_dir = 'backend/jurisprudence'
    if os.path.exists(jurisprudence_dir):
        txt_count = 0
        print("📁 TXT files created:")
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
    create_sample_real_data()


