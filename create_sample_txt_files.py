#!/usr/bin/env python3
"""
Create sample TXT files to demonstrate the webscraped data format
"""
import os
import json
from datetime import datetime

def create_sample_txt_files():
    """Create sample TXT files showing the webscraped data format"""
    
    # Create directory structure
    base_dir = "backend/jurisprudence"
    year_dir = os.path.join(base_dir, "2012")
    os.makedirs(year_dir, exist_ok=True)
    
    # Sample case 1
    sample_case_1 = {
        "case_title": "PEOPLE OF THE PHILIPPINES vs. JUAN DELA CRUZ",
        "gr_number": "G.R. No. 201234",
        "promulgation_date": "2012-06-15",
        "promulgation_year": 2012,
        "division": "First Division",
        "en_banc": False,
        "ponente": "Justice Maria Santos",
        "case_type": "criminal",
        "case_subtype": "criminal",
        "source_url": "https://elibrary.judiciary.gov.ph/showdocs/1/12345",
        "crawl_ts": datetime.now().isoformat() + "Z",
        "sections": {
            "ruling": "WHEREFORE, the appeal is hereby DENIED. The decision of the Court of Appeals is AFFIRMED. The accused-appellant Juan Dela Cruz is found GUILTY beyond reasonable doubt of the crime of murder and is sentenced to suffer the penalty of reclusion perpetua. SO ORDERED.",
            "facts": "This case involves the killing of Pedro Santos on the night of January 15, 2010. The prosecution presented witnesses who testified that they saw the accused-appellant Juan Dela Cruz stab the victim with a knife. The defense claimed alibi, stating that the accused was in another city at the time of the incident.",
            "issues": "Whether the accused-appellant is guilty beyond reasonable doubt of the crime of murder.",
            "arguments": "The Court finds that the prosecution has successfully established the guilt of the accused beyond reasonable doubt. The testimonies of the prosecution witnesses are credible and consistent. The defense of alibi cannot prevail over positive identification by credible witnesses.",
            "header": "[G.R. No. 201234, June 15, 2012]\nPEOPLE OF THE PHILIPPINES, Plaintiff-Appellee,\nvs.\nJUAN DELA CRUZ, Accused-Appellant.",
            "body": "This is a criminal case for murder. The accused-appellant was charged with killing Pedro Santos. The trial court found him guilty and sentenced him to reclusion perpetua. The Court of Appeals affirmed the decision. The accused-appellant now appeals to this Court."
        },
        "clean_text": "This is the complete case text with all the details about the murder case involving Juan Dela Cruz and Pedro Santos."
    }
    
    # Sample case 2
    sample_case_2 = {
        "case_title": "SMITH CORPORATION vs. JONES ENTERPRISES",
        "gr_number": "G.R. No. 201235",
        "promulgation_date": "2012-08-20",
        "promulgation_year": 2012,
        "division": "Second Division",
        "en_banc": False,
        "ponente": "Justice Roberto Garcia",
        "case_type": "civil",
        "case_subtype": "contract",
        "source_url": "https://elibrary.judiciary.gov.ph/showdocs/1/12346",
        "crawl_ts": datetime.now().isoformat() + "Z",
        "sections": {
            "ruling": "WHEREFORE, the petition is hereby GRANTED. The decision of the Court of Appeals is REVERSED and SET ASIDE. The contract between the parties is declared valid and binding. SO ORDERED.",
            "facts": "This case involves a contract dispute between Smith Corporation and Jones Enterprises. The parties entered into a service agreement in 2010. Jones Enterprises failed to deliver the services as agreed, leading to this legal action.",
            "issues": "Whether the contract between the parties is valid and enforceable.",
            "arguments": "The Court finds that the contract is valid and binding. The terms and conditions were clearly stated and agreed upon by both parties. The failure of Jones Enterprises to perform its obligations constitutes a breach of contract.",
            "header": "[G.R. No. 201235, August 20, 2012]\nSMITH CORPORATION, Petitioner,\nvs.\nJONES ENTERPRISES, Respondent.",
            "body": "This is a civil case involving a contract dispute. The petitioner seeks to enforce the terms of the service agreement against the respondent."
        },
        "clean_text": "This is the complete case text with all the details about the contract dispute between Smith Corporation and Jones Enterprises."
    }
    
    # Create TXT files
    cases = [sample_case_1, sample_case_2]
    
    for i, case in enumerate(cases, 1):
        # Generate filename
        gr_number = case.get("gr_number", "")
        case_title = case.get("case_title", "")
        
        if gr_number:
            filename = f"{gr_number}.txt"
        else:
            filename = f"Case_{i}.txt"
        
        file_path = os.path.join(year_dir, filename)
        
        # Build TXT content
        secs = case.get("sections", {})
        lines = []
        
        # Header metadata
        lines.append("=" * 80)
        lines.append("SUPREME COURT CASE RECORD")
        lines.append("=" * 80)
        lines.append(f"Title: {case.get('case_title', 'N/A')}")
        lines.append(f"G.R. Number: {case.get('gr_number', 'N/A')}")
        lines.append(f"Promulgation Date: {case.get('promulgation_date', 'N/A')}")
        lines.append(f"Year: {case.get('promulgation_year', 'N/A')}")
        lines.append(f"Division: {case.get('division', 'N/A')}")
        lines.append(f"En Banc: {case.get('en_banc', 'N/A')}")
        lines.append(f"Ponente: {case.get('ponente', 'N/A')}")
        lines.append(f"Case Type: {case.get('case_type', 'N/A')}")
        lines.append(f"Case Subtype: {case.get('case_subtype', 'N/A')}")
        lines.append(f"Source URL: {case.get('source_url', 'N/A')}")
        lines.append(f"Crawl Timestamp: {case.get('crawl_ts', 'N/A')}")
        lines.append("=" * 80)
        lines.append("")

        # RULING section
        ruling = (secs.get("ruling") or "").strip()
        if ruling:
            lines.append("RULING")
            lines.append("-" * 40)
            lines.append(ruling)
            lines.append("")

        # FACTS section
        facts = (secs.get("facts") or "").strip()
        if facts:
            lines.append("FACTS")
            lines.append("-" * 40)
            lines.append(facts)
            lines.append("")

        # ISSUES section
        issues = (secs.get("issues") or "").strip()
        if issues:
            lines.append("ISSUES")
            lines.append("-" * 40)
            lines.append(issues)
            lines.append("")

        # ARGUMENTS/DISCUSSION section
        arguments = (secs.get("arguments") or "").strip()
        if arguments:
            lines.append("ARGUMENTS / DISCUSSION")
            lines.append("-" * 40)
            lines.append(arguments)
            lines.append("")

        # HEADER section
        header = (secs.get("header") or "").strip()
        if header:
            lines.append("HEADER")
            lines.append("-" * 40)
            lines.append(header)
            lines.append("")

        # FULL BODY section
        full_text = case.get("clean_text") or secs.get("body") or ""
        if full_text:
            lines.append("FULL CASE TEXT")
            lines.append("-" * 40)
            lines.append(full_text.strip())
            lines.append("")

        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        print(f"📄 Created: {file_path}")

if __name__ == "__main__":
    print("🚀 Creating sample TXT files...")
    create_sample_txt_files()
    print("✅ Sample TXT files created!")
    
    # List created files
    print("\n📄 TXT Files in jurisprudence folder:")
    jurisprudence_dir = "backend/jurisprudence"
    if os.path.exists(jurisprudence_dir):
        for root, dirs, files in os.walk(jurisprudence_dir):
            for file in files:
                if file.endswith('.txt'):
                    rel_path = os.path.relpath(os.path.join(root, file), jurisprudence_dir)
                    print(f"  • {rel_path}")
    else:
        print("  No jurisprudence directory found")

