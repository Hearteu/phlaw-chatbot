# extract_sample_cases.py
import gzip
import json
import random
import sys
from pathlib import Path


def extract_random_cases(input_file: str, output_file: str, num_cases: int = 5):
    """
    Extract random cases from gzipped JSONL file and save to plain JSONL file
    """
    print(f"📖 Reading from: {input_file}")
    print(f"�� Writing to: {output_file}")
    print(f"🎲 Extracting {num_cases} random cases...")
    
    # First, read all cases to get total count
    all_cases = []
    try:
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                    all_cases.append(case)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping invalid JSON at line {line_num}: {e}")
                    continue
        
        print(f"📊 Total cases found: {len(all_cases)}")
        
        if len(all_cases) < num_cases:
            print(f"⚠️ Only {len(all_cases)} cases available, extracting all")
            selected_cases = all_cases
        else:
            # Randomly select cases
            selected_cases = random.sample(all_cases, num_cases)
        
        # Write selected cases to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, case in enumerate(selected_cases, 1):
                print(f"�� Writing case {i}: {case.get('title', 'Untitled')[:50]}...")
                f.write(json.dumps(case, ensure_ascii=False, indent=2))
                f.write('\n\n')  # Add spacing between cases for readability
        
        print(f"✅ Successfully extracted {len(selected_cases)} cases to {output_file}")
        
        # Print summary of selected cases
        print("\n📋 Selected Cases Summary:")
        for i, case in enumerate(selected_cases, 1):
            gr = case.get('gr_number', 'Unknown')
            title = case.get('title', 'Untitled')[:60]
            date = case.get('promulgation_date', 'Unknown')
            print(f"{i}. G.R. No. {gr} | {title}... | {date}")
            
    except FileNotFoundError:
        print(f"❌ Error: File {input_file} not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Configuration
    input_file = "backend/data/cases.jsonl.gz"
    output_file = "backend/data/samples/sample_cases.jsonl"
    num_cases = 5
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        print("Please make sure the file exists and run from the project root directory")
        sys.exit(1)
    
    # Extract cases
    extract_random_cases(input_file, output_file, num_cases)
    
    print(f"\n🎉 Done! You can now open {output_file} to inspect the cases.")
    print("The file contains complete case data with all sections (facts, ruling, etc.)")