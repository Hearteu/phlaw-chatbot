# -*- coding: utf-8 -*-
"""
Async e-Library crawler:
- Discover month pages (YEAR_START..YEAR_END)
- Extract case URLs (/showdocs) with year/month hints (Fix A)
- Fetch concurrently with Crawl4AI AsyncWebCrawler (Playwright) or HTTP fallback
- Extract robust title (HTML preferred, then Markdown)
- Save JSONL.gz ready for embed.py (sections: header/body/ruling)
- Dedupe/resume: skips records already present in JSONL.gz
"""

import asyncio
import gzip
import hashlib
# --- Windows event-loop policy MUST be set early (Playwright subprocess on Windows) ---
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict
from urllib.parse import urljoin

import aiohttp
import orjson
import requests
from bs4 import BeautifulSoup
# Crawl4AI (current docs API)
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from dateutil import parser as dtp
from selectolax.parser import HTMLParser

# if sys.platform == 'win32':
#     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
# -------------------------------------------------------------------------------------



# -----------------------------
# Config (env-overridable)
# -----------------------------
BASE_URL     = os.getenv("ELIBRARY_BASE", "https://elibrary.judiciary.gov.ph/")
OUT_PATH     = os.getenv("CASES_JSONL", "backend/data/cases2.jsonl.gz")
UA           = os.getenv("CRAWLER_UA", "Mozilla/5.0 (compatible; PHLawBot/1.0)")
YEAR_START   = int(os.getenv("YEAR_START", 2005))
YEAR_END     = int(os.getenv("YEAR_END", 2025))
CONCURRENCY  = int(os.getenv("CONCURRENCY", 12))  # Higher for HTTP-first approach
SLOWDOWN_MS  = int(os.getenv("SLOWDOWN_MS", 250))
TIMEOUT_S    = int(os.getenv("TIMEOUT_S", 45))
WRITE_CHUNK  = int(os.getenv("WRITE_CHUNK", 1000))  # tasks per gather batch

HEADERS = {"User-Agent": UA}

RULING_REGEX = re.compile(
    r"(WHEREFORE.*?SO ORDERED\.?|ACCORDINGLY.*?SO ORDERED\.?|IN VIEW WHEREOF.*?SO ORDERED\.?)",
    re.IGNORECASE | re.DOTALL,
)

# Date patterns - prioritize header dates over body dates
HEADER_DATE_PATTERNS = [
    # Case header format: [ G.R. No. 176692, June 27, 2012 ]
    r"\[.*?G\.R\.\s+No\.?\s*\d+.*?,\s*(\w+\s+\d{1,2},\s+\d{4})\s*\]",
    # Alternative header formats without brackets
    r"G\.R\.\s+No\.?\s*\d+.*?,\s*(\w+\s+\d{1,2},\s+\d{4})",
    # EN BANC [ A.M. No. P-11-2907, January 31, 2012 ]
    r"\[.*?A\.M\.\s+No\.?\s*[^,]+,\s*(\w+\s+\d{1,2},\s+\d{4})\s*\]",
    # Division headers: FIRST DIVISION [ G.R. No. 176692, June 27, 2012 ]
    r"(?:FIRST|SECOND|THIRD|EN\s+BANC)\s+DIVISION\s*\[.*?G\.R\.\s+No\.?\s*\d+.*?,\s*(\w+\s+\d{1,2},\s+\d{4})\s*\]",
    # More flexible header patterns
    r"\[.*?(\w+\s+\d{1,2},\s+\d{4})\s*\]",
]

BODY_DATE_PATTERNS = [
    r"Promulgated\s*(?:on)?\s*:\s*(\w+\s+\d{1,2},\s+\d{4})",
    r"Promulgated\s*(?:on)?\s*(\w+\s+\d{1,2},\s+\d{4})",
    r"Decided\s*(?:on)?\s*:\s*(\w+\s+\d{1,2},\s+\d{4})",
    r"Decided\s*(?:on)?\s*(\w+\s+\d{1,2},\s+\d{4})",
    # Simple date pattern without lookbehind
    r"(\w+\s+\d{1,2},\s+\d{4})",
]

# -----------------------------
# Utils
# -----------------------------
def sha256(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()

def clean_website_headers(text: str) -> str:
    """Remove Supreme Court E-Library website headers and navigation"""
    if not text:
        return ""
    
    # Find the start of actual case content (look for G.R. pattern)
    gr_match = re.search(r'\[.*?G\.R\.\s+No\.?\s*\d+.*?\]', text)
    if gr_match:
        # Start from the G.R. line
        text = text[gr_match.start():]
    
    # Remove all website-related content more aggressively
    website_patterns = [
        r"Supreme Court E-Library.*?Information At Your Fingertips",
        r"HOME\s*PHILIPPINE REPORTS E-BOOKS\s*REPUBLIC ACTS\s*CHIEF JUSTICES\s*NEWS & ADVISORIES\s*SITE MAP\s*ABOUT US",
        r"The Supreme Court E-Library.*?Toggle posts",
        r"CLICK THE IMAGE TO SEARCH.*?libraryservices\.sc@judiciary\.gov\.ph",
        r"Foreign Supreme Courts.*?United States of America",
        r"Volume in drive.*?bytes free",
        r"Directory of.*?Volume Serial Number",
        r"File\(s\)\s+\d+\s+bytes",
        r"Dir\(s\)\s+\d+\s+bytes free",
        r"Toggle posts\s*A\s*A\+\s*A\+\+",
        r"CONTACT:\s*Supreme Court of the Philippines.*?libraryservices\.sc@judiciary\.gov\.ph",
        r"\(632\)\s*\d{3}-\d{4}",
        r"^A\s*A\+\s*A\+\+.*?View printer friendly version",
        r"View printer friendly version.*?Phil\.",
        r"^\d+\s+Phil\.\s+\d+",
        r"Supreme Court E-Library\s*Information At Your Fingertips",
        r"The E-Library Development Team",
        r"Toggle posts",
        r"CLICK THE IMAGE TO SEARCH",
        r"CONTACT:",
        r"Supreme Court of the Philippines",
        r"Library Services,",
        r"Padre Faura, Ermita, Manila, Philippines 1000",
        # Additional aggressive patterns
        r"Supreme Court E-Library",
        r"Information At Your Fingertips",
        r"The E-Library Development Team",
        r"Toggle posts",
        r"CLICK THE IMAGE TO SEARCH",
        r"CONTACT:",
        r"Supreme Court of the Philippines",
        r"Library Services,",
        r"Padre Faura, Ermita, Manila, Philippines 1000",
        r"Foreign Supreme Courts",
        r"Korea, South",
        r"Malaysia",
        r"Singapore",
        r"United States of America",
        # Footer copyright lines
        r"©\s*\d{4}.*E-?Library.*",
    ]
    
    for pattern in website_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
    # Remove lines that are just navigation elements
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            # Skip if line contains website elements
            if any(phrase in line.upper() for phrase in [
                'SUPREME COURT E-LIBRARY',
                'INFORMATION AT YOUR FINGERTIPS',
                'THE E-LIBRARY DEVELOPMENT TEAM',
                'TOGGLE POSTS',
                'CLICK THE IMAGE TO SEARCH',
                'CONTACT:',
                'SUPREME COURT OF THE PHILIPPINES',
                'LIBRARY SERVICES,',
                'PADRE FAURA, ERMITA, MANILA, PHILIPPINES 1000',
                'FOREIGN SUPREME COURTS',
                'KOREA, SOUTH',
                'MALAYSIA',
                'SINGAPORE',
                'UNITED STATES OF AMERICA',
                'HOME',
                'ABOUT US',
                'SITE MAP',
                'NEWS & ADVISORIES',
                'A+',
                'A++',
            ]):
                continue
            
            # Skip known navigation tokens and reporter citation lines only
            if (
                re.match(r'^\d+\s+Phil\.\s+\d+$', line) or  # e.g., 123 Phil. 456
                line in ['A', 'A+', 'A++', 'HOME', 'ABOUT US', 'SITE MAP', 'NEWS & ADVISORIES']
            ):
                continue
            
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

def normalize_text(s: str) -> str:
    if not s:
        return ""
    
    # Clean website headers first
    s = clean_website_headers(s)
    
    # Convert literal "\n" sequences to real newlines
    s = s.replace("\\n", "\n")
    # Normalize whitespace
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"\r", "\n", s)
    # Preserve paragraph breaks first
    s = re.sub(r"\n{3,}", "\n\n", s)
    # Join inline broken lines inside sentences
    s = re.sub(r",[ \t]*\n", ", ", s)       # comma + newline -> comma space
    s = re.sub(r";[ \t]*\n", "; ", s)       # semicolon + newline -> semicolon space
    s = re.sub(r"(?<=\w)[ \t]*\n[ \t]*(?=\w)", " ", s)  # word\nword -> word word
    # Fix Id. variants
    s = re.sub(r"\bId\s*\n\s*\.", "Id.", s)
    s = re.sub(r"\bId\s*\n\s*at\b", "Id. at", s, flags=re.IGNORECASE)
    # Collapse excessive spaces
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"^\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s+$", "", s, flags=re.MULTILINE)
    return s.strip()

# -----------------------------
# Field extractors (caption, GR numbers, division, ponente)
# -----------------------------
TITLE_NOISE = (
    "SUPREME COURT", "EN BANC", "FIRST DIVISION",
    "SECOND DIVISION", "THIRD DIVISION", "PER CURIAM",
)

CAPTION_RE = re.compile(r"(.+?)\s+(v\.|vs\.?|versus)\s+(.+?)(?:\.|$)", re.IGNORECASE)
CAPTION_WITH_NOTATION_RE = re.compile(r"^(.+?,\[?\d*\]?)\s+(PETITIONER|RESPONDENT|COMPLAINANT|PLAINTIFF|DEFENDANT)S?,\s*(v\.|vs\.?|versus)\s+(.+?),\s*(PETITIONER|RESPONDENT|COMPLAINANT|PLAINTIFF|DEFENDANT)S?\.?$", re.IGNORECASE)
CAPTION_WITH_GR_NOTATION_RE = re.compile(r"^(XXX\d+,\*?)\s+(PETITIONER|RESPONDENT|COMPLAINANT|PLAINTIFF|DEFENDANT)S?,\s*(v\.|vs\.?|versus)\s+(.+?),\s*(PETITIONER|RESPONDENT|COMPLAINANT|PLAINTIFF|DEFENDANT)S?\.?$", re.IGNORECASE)
CAPTION_STRONG_RE = re.compile(r"^[A-Z0-9 ,.'\-()*\[\]]+\s+(V\.|VS\.?|VERSUS)\s+[A-Z0-9 ,.'\-()*\[\]]+$")

# Additional patterns for edge cases
ADMIN_COMPLAINT_RE = re.compile(r"^(.+?),\s*COMPLAINANT,\s*VS\.?\s*(.+?),\s*RESPONDENT", re.IGNORECASE)

# Enhanced RE: prefix patterns for various administrative case formats
RE_PREFIX_PATTERNS = [
    # RE: [COMPLAINANT] VS. [RESPONDENT] (most common format)
    re.compile(r"^RE:\s*(.+?)\s+VS\.?\s+(.+?)(?:\s+RESOLUTION|\s+DECISION|\s+ORDER)?$", re.IGNORECASE),
    # RE: [COMPLAINANT], COMPLAINANT, VS. [RESPONDENT], RESPONDENT
    re.compile(r"^RE:\s*(.+?),\s*COMPLAINANT,\s*VS\.?\s*(.+?),\s*RESPONDENT", re.IGNORECASE),
    # RE: VERIFIED COMPLAINT OF [COMPLAINANT] AGAINST [RESPONDENTS]
    re.compile(r"^RE:\s*VERIFIED\s+COMPLAINT\s+OF\s+(.+?)\s+AGAINST\s+(.+?)$", re.IGNORECASE),
    # RE: COMPLAINT FILED BY [COMPLAINANT] AGAINST [RESPONDENT]
    re.compile(r"^RE:\s*COMPLAINT\s+FILED\s+BY\s+(.+?)\s+AGAINST\s+(.+?)$", re.IGNORECASE),
    # RE: LETTER-COMPLAINT AGAINST [RESPONDENTS] FILED BY [COMPLAINANT]
    re.compile(r"^RE:\s*LETTER-COMPLAINT\s+AGAINST\s+(.+?)\s+FILED\s+BY\s+(.+?)$", re.IGNORECASE),
    # RE: ANONYMOUS LETTER... COMPLAINING AGAINST [RESPONDENT]
    re.compile(r"^RE:\s*ANONYMOUS\s+LETTER.*?COMPLAINING.*?AGAINST\s+(.+?)$", re.IGNORECASE),
    # RE: SUBPOENA DUCES TECUM... (special case)
    re.compile(r"^RE:\s*SUBPOENA\s+DUCES\s+TECUM.*?OF\s+(.+?)$", re.IGNORECASE),
    # Generic RE: pattern as fallback
    re.compile(r"^RE:\s*(.+?)(?:,\s*COMPLAINANT|\s*VS\.?|\s*AGAINST)", re.IGNORECASE),
]

# Additional patterns for special administrative case formats
SPECIAL_ADMIN_PATTERNS = [
    # IN RE: ANONYMOUS LETTER... COMPLAINING AGAINST [RESPONDENT]
    re.compile(r"^IN\s+RE:\s*ANONYMOUS\s+LETTER.*?COMPLAINING\s+AGAINST\s+(.+?)$", re.IGNORECASE),
    # IN RE: [CASE DESCRIPTION]
    re.compile(r"^IN\s+RE:\s*(.+?)$", re.IGNORECASE),
]
BODY_TEXT_INDICATORS = [
    "this is an", "this case is", "the case at bar", "the instant case", 
    "the present case", "this petition", "this appeal", "the foregoing",
    "of the court of appeals", "dated", "resolution", "decision",
    "against respondent", "for grave misconduct", "dishonesty and breach",
    "records officer", "office of administrative services", "court administrator",
    "this is a", "the instant", "the foregoing", "complainant claims",
    "respondent served", "while respondent", "complainant states",
    # Historical/narrative indicators that signal body text
    "in february", "in march", "in april", "in may", "in june", 
    "in july", "in august", "in september", "in october", "in november", "in december",
    "in january", "in 19", "in 20", "on january", "on february", "on march",
    "the facts", "the case involve", "the case stem", "the case arise",
    "the edsa", "revolution toppled", "one of the first acts",
    "pursuant to this", "pursuant to the mandate", "pursuant to its mandate",
    "his family and his cronies", "alleged ill-gotten wealth",
    "the commission", "to recover", "following the", "after the",
    "president corazon", "president aquino", "president marcos",
    "was to establish", "was established", "encountered financial",
    # Additional patterns for factual content from case bodies
    "the marriage was", "respondent complained that", "their sexual relationship",
    "the couple talked about", "is a product of a broken family",
    "her father always subjected", "her mother to physical abuse",
    "both parents are living", "her auntie took care",
    "her mother would give her away", "in exchange for money",
    "x x x", "is a product of", "broken family", "physical abuse",
    "sexual relationship", "marriage was blissful", "turned sour",
    "too direct", "outspoken", "domineering", "unsatisfying"
]
GR_BLOCK_RE = re.compile(r"G\.R\.\s+No(?:s)?\.?\s*([0-9][0-9\-*,\s]+)", re.IGNORECASE)

def _clean_noise(line: str) -> str:
    s = line.strip()
    for t in TITLE_NOISE:
        s = s.replace(t, "")
    # Remove "PROMULGATED:" and similar prefixes that sometimes appear in titles
    s = re.sub(r"^PROMULGATED:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^DECIDED:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^RESOLVED:\s*", "", s, flags=re.IGNORECASE)
    # Remove decision headers that sometimes get appended to titles
    s = re.sub(r"\s+D\s+E\s+C\s+I\s+S\s+I\s+O\s+N.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+DECISION.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+R\s+E\s+S\s+O\s+L\s+U\s+T\s+I\s+O\s+N.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+RESOLUTION.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+ORDER.*$", "", s, flags=re.IGNORECASE)
    # Remove ponente names that get appended to titles
    s = re.sub(r"\s+[A-Z][A-Z\-']+,\s*J\.?.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+[A-Z][A-Z\-']+\s+J\.?.*$", "", s, flags=re.IGNORECASE)
    # Clean up special characters but preserve them in names (like asterisks in names)
    # Only remove trailing special characters, not those within names
    s = re.sub(r"[,\s]*$", "", s)  # Remove trailing commas and spaces
    # Normalize whitespace but preserve special characters within names
    s = re.sub(r"\s+", " ", s)  # Normalize whitespace
    return s.strip()

def _is_body_text(line: str) -> bool:
    """Check if a line looks like body text rather than a title"""
    line_lower = line.lower().strip()
    return any(indicator in line_lower for indicator in BODY_TEXT_INDICATORS)

def _extract_admin_title(line: str) -> str | None:
    """Extract title from administrative case formats"""
    # Try COMPLAINANT VS. RESPONDENT format
    match = ADMIN_COMPLAINT_RE.match(line)
    if match:
        complainant = match.group(1).strip()
        respondent = match.group(2).strip()
        return f"{complainant}, COMPLAINANT, VS. {respondent}, RESPONDENT"
    
    # Try enhanced RE: prefix patterns
    for pattern in RE_PREFIX_PATTERNS:
        match = pattern.match(line)
        if match:
            if len(match.groups()) == 2:
                # Pattern with complainant and respondent
                complainant = match.group(1).strip()
                respondent = match.group(2).strip()
                # Clean up the extracted parts
                complainant = _clean_noise(complainant)
                respondent = _clean_noise(respondent)
                return f"{complainant}, COMPLAINANT, VS. {respondent}, RESPONDENT"
            elif len(match.groups()) == 1:
                # Pattern with single group (like SUBPOENA or ANONYMOUS LETTER)
                return _clean_noise(match.group(1).strip())
    
    # Try special administrative patterns (IN RE: format)
    for pattern in SPECIAL_ADMIN_PATTERNS:
        match = pattern.match(line)
        if match:
            if len(match.groups()) == 1:
                # Extract the main subject/person from the case
                case_description = match.group(1).strip()
                # For ANONYMOUS LETTER cases, extract the person being complained against
                if 'COMPLAINING AGAINST' in case_description.upper():
                    # Extract person after "COMPLAINING AGAINST"
                    parts = case_description.upper().split('COMPLAINING AGAINST')
                    if len(parts) > 1:
                        person_part = parts[1].strip()
                        # Take the first part (person's name and title)
                        person_parts = person_part.split(',')
                        if person_parts:
                            return person_parts[0].strip()
                return _clean_noise(case_description)
    
    return None

def _extract_am_case_title(text: str) -> str | None:
    """Extract title specifically for A.M. cases with RE: format"""
    # Look for A.M. NO. pattern followed by RE: pattern
    am_pattern = re.compile(r"A\.M\.\s+NO\.\s+([0-9\-]+[A-Z]?)\s*-\s*RE:\s*(.+?)(?:\s+RESOLUTION|\s+DECISION|\s+ORDER)?$", re.IGNORECASE | re.MULTILINE)
    match = am_pattern.search(text)
    if match:
        am_number = match.group(1).strip()
        case_title = match.group(2).strip()
        return f"A.M. NO. {am_number} - {case_title}"
    
    # Look for A.M. cases without RE: pattern
    am_pattern2 = re.compile(r"A\.M\.\s+NO\.\s+([0-9\-]+[A-Z]?)\s*\([^)]+\)", re.IGNORECASE | re.MULTILINE)
    match2 = am_pattern2.search(text)
    if match2:
        am_number = match2.group(1).strip()
        # Look for the case title after the A.M. number
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if f"A.M. NO. {am_number}" in line.upper():
                # Look for the title in the next few lines
                for j in range(i + 1, min(i + 5, len(lines))):
                    title_line = lines[j].strip()
                    if (title_line and 
                        len(title_line) > 10 and 
                        not any(indicator in title_line.lower() for indicator in BODY_TEXT_INDICATORS) and
                        ('V.' in title_line.upper() or 'VS.' in title_line.upper())):
                        return f"A.M. NO. {am_number} - {title_line}"
    
    # Enhanced RE: pattern detection for administrative cases
    lines = text.split('\n')[:15]  # Check first 15 lines for better coverage
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for RE: pattern with various administrative case formats
        if line.startswith('RE:'):
            # Extract the full RE: title, but be more specific about stopping conditions
            re_match = re.match(r'^RE:\s*(.+?)(?:\s+(?:RESOLUTION|DECISION|ORDER|OF\s+THE\s+COURT|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|\d{4}))?$', line, re.IGNORECASE)
            if re_match:
                title_part = re_match.group(1).strip()
                # Validate it's not body text
                if not any(indicator in title_part.lower() for indicator in BODY_TEXT_INDICATORS):
                    return f"RE: {title_part}"
        
        # Look for administrative case patterns without RE: prefix
        if 'COMPLAINANT' in line.upper() and 'VS.' in line.upper():
            # This looks like an administrative case title
            if not any(indicator in line.lower() for indicator in BODY_TEXT_INDICATORS):
                return line
    
    return None

def derive_case_title_from_text(text: str) -> str | None:
    if not text:
        return None
    head = text[:5000]  # Increased window for very long multi-party titles
    
    # Priority 1: A.M. cases with RE: format (most specific)
    am_title = _extract_am_case_title(text)
    if am_title and 10 <= len(am_title) <= 500:
        return am_title
    
    # Priority 2: Enhanced multi-line VS. format (handle long titles that span multiple lines)
    # Look for the pattern across multiple lines with better logic
    lines = head.splitlines()
    
    # First, try to find complete single-line titles
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Check if this line contains a case title with GR notation (like XXX264870,*)
        if CAPTION_WITH_GR_NOTATION_RE.match(line):
            # This is a complete case title with GR notation
            if len(line) <= 3000:  # Allow very long titles for multi-party cases
                cleaned = _clean_noise(line)
                if 20 <= len(cleaned) <= 3000:
                    return cleaned
        
        # Check if this line contains a case title with notation (like XXX,[1])
        elif CAPTION_WITH_NOTATION_RE.match(line):
            # This is a complete case title with notation
            if len(line) <= 3000:  # Allow very long titles for multi-party cases
                cleaned = _clean_noise(line)
                if 20 <= len(cleaned) <= 3000:
                    return cleaned
        
        # Check if this line contains a complete case title pattern
        elif CAPTION_RE.match(line) and re.search(r"\b(PETITIONER|RESPONDENT|COMPLAINANT|PLAINTIFF|DEFENDANT)\b", line, re.IGNORECASE):
            # Check if this line starts with comma - might be continuation of previous line
            if line.startswith(','):
                # This is likely a continuation line, skip it and let multi-line logic handle it
                continue
            # This line looks like a complete case title
            if len(line) <= 3000:  # Allow very long titles for multi-party cases
                cleaned = _clean_noise(line)
                if 20 <= len(cleaned) <= 3000:
                    return cleaned
        
        # Special handling for titles with special characters (asterisks, brackets)
        elif re.search(r"\b(VS\.|V\.|VERSUS)\b", line, re.IGNORECASE) and re.search(r"\b(PETITIONER|RESPONDENT|COMPLAINANT|PLAINTIFF|DEFENDANT)\b", line, re.IGNORECASE):
            # Check if this line starts with comma - might be continuation of previous line
            if line.startswith(','):
                # This is likely a continuation line, skip it and let multi-line logic handle it
                continue
            # This might be a complete title with special characters
            if len(line) <= 3000:
                cleaned = _clean_noise(line)
                if 20 <= len(cleaned) <= 3000:
                    return cleaned
    
    # Priority 3: Multi-line title reconstruction
    # Look for partial titles and try to complete them across lines
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        
        # Check if this line starts a case title (contains PETITIONER/PLAINTIFF/COMPLAINANT)
        if re.search(r"\b(PETITIONER|PLAINTIFF|COMPLAINANT|DEFENDANT)S?\b", line, re.IGNORECASE):
            # This might be the start of a case title
            title_parts = [line]
            
            # Look ahead to find the complete title
            for j in range(i + 1, min(i + 8, len(lines))):  # Look ahead up to 8 lines
                next_line = lines[j].strip()
                if not next_line:
                    continue
                
                # Stop if we hit decision content or other non-title content
                if re.search(r"\b(DECISION|RESOLUTION|ORDER|PER CURIAM|J\.|CHICO-NAZARIO|CARPIO|MORALES|THE\s+CASES?|ANTECEDENTS?)\b", next_line, re.IGNORECASE):
                    break
                if re.search(r"D\s+E\s+C\s+I\s+S\s+I\s+O\s+N", next_line, re.IGNORECASE):
                    break
                if re.search(r"\[G\.R\.\s+No\.\s+\d+\]", next_line):  # Stop at GR number lines
                    break
                
                # Add the line if it looks like part of a title
                if (re.search(r"\b(PETITIONER|RESPONDENT|COMPLAINANT|PLAINTIFF|DEFENDANT|VS\.|V\.|VERSUS)\b", next_line, re.IGNORECASE) or
                    re.search(r"\b(AND|&)\b", next_line, re.IGNORECASE) or  # Handle "AND" in multi-party cases
                    re.search(r"^[A-Z][A-Z0-9\s,.'\-()]+$", next_line) or  # All caps legal names
                    next_line.startswith(',') or  # Handle continuation lines that start with comma
                    (next_line.startswith(' ') and re.search(r"\b(VS\.|V\.|VERSUS|RESPONDENT|PETITIONER|COMPLAINANT|PLAINTIFF|DEFENDANT)\b", next_line, re.IGNORECASE))):  # Indented continuation lines
                    title_parts.append(next_line)
                else:
                    break
            
            # Join the parts and check if it forms a valid title
            if len(title_parts) > 1:  # Only if we found multiple parts
                full_title = " ".join(title_parts)
                # Clean up the title
                full_title = re.sub(r"\s+", " ", full_title)  # Normalize whitespace
                
                # Check if this looks like a complete case title
                if (CAPTION_RE.search(full_title) and 
                    len(full_title) >= 20 and len(full_title) <= 3000 and
                    re.search(r"\b(PETITIONER|RESPONDENT|COMPLAINANT|PLAINTIFF|DEFENDANT)\b", full_title, re.IGNORECASE)):
                    cleaned = _clean_noise(full_title)
                    if 20 <= len(cleaned) <= 3000:
                        return cleaned
    
    # Priority 3.5: Special handling for Supreme Court E-Library format
    # Look for the specific pattern: NAME, PETITIONER, VS. RESPONDENT
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        
        # Look for lines that start with a name followed by PETITIONER
        if re.search(r"^[A-Z][A-Z0-9\s,.'\-()]+\s*,\s*PETITIONER", line, re.IGNORECASE):
            # This looks like the start of a case title
            title_parts = [line]
            
            # Look for the VS. line
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    continue
                
                # Stop if we hit non-title content
                if re.search(r"\b(DECISION|RESOLUTION|ORDER|PER CURIAM|J\.|CHICO-NAZARIO|CARPIO|MORALES|THE\s+CASES?|ANTECEDENTS?)\b", next_line, re.IGNORECASE):
                    break
                if re.search(r"D\s+E\s+C\s+I\s+S\s+I\s+O\s+N", next_line, re.IGNORECASE):
                    break
                if re.search(r"\[G\.R\.\s+No\.\s+\d+\]", next_line):
                    break
                
                # Look for VS. line
                if re.search(r"\b(VS\.|V\.|VERSUS)\b", next_line, re.IGNORECASE):
                    title_parts.append(next_line)
                    # Look for the respondent line
                    for k in range(j + 1, min(j + 3, len(lines))):
                        resp_line = lines[k].strip()
                        if not resp_line:
                            continue
                        if re.search(r"\b(RESPONDENT|RESPONDENTS)\b", resp_line, re.IGNORECASE):
                            title_parts.append(resp_line)
                            break
                    break
                else:
                    break
            
            # Join the parts and check if it forms a valid title
            if len(title_parts) >= 2:  # At least petitioner and vs line
                full_title = " ".join(title_parts)
                full_title = re.sub(r"\s+", " ", full_title)  # Normalize whitespace
                
                # Check if this looks like a complete case title
                if (CAPTION_RE.search(full_title) and 
                    len(full_title) >= 20 and len(full_title) <= 3000 and
                    re.search(r"\b(PETITIONER|RESPONDENT|COMPLAINANT|PLAINTIFF|DEFENDANT)\b", full_title, re.IGNORECASE)):
                    cleaned = _clean_noise(full_title)
                    if 20 <= len(cleaned) <= 3000:
                        return cleaned
    
    # Priority 3.6: Handle incomplete titles by looking for complete versions
    # This addresses cases where titles are truncated or incomplete
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        
        # Look for incomplete titles (ending with VS. or V. without respondent)
        if re.search(r"\b(VS\.|V\.|VERSUS)\s*$", line, re.IGNORECASE):
            # This might be an incomplete title, look for the respondent
            title_parts = [line]
            
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    continue
                
                # Stop if we hit non-title content
                if re.search(r"\b(DECISION|RESOLUTION|ORDER|PER CURIAM|J\.|CHICO-NAZARIO|CARPIO|MORALES|THE\s+CASES?|ANTECEDENTS?)\b", next_line, re.IGNORECASE):
                    break
                if re.search(r"D\s+E\s+C\s+I\s+S\s+I\s+O\s+N", next_line, re.IGNORECASE):
                    break
                if re.search(r"\[G\.R\.\s+No\.\s+\d+\]", next_line):
                    break
                
                # Add if it looks like a respondent name (including special characters)
                if (re.search(r"\b(RESPONDENT|RESPONDENTS|DEFENDANT|DEFENDANTS|ACCUSED|ACCUSED-APPELLANT|ACCUSED-APPELLEE)\b", next_line, re.IGNORECASE) or
                    (len(next_line) > 5 and not re.search(r"\b(THE|OF|IN|FOR|WITH|AND|OR)\b", next_line, re.IGNORECASE))):
                    title_parts.append(next_line)
                    break
                else:
                    break
            
            # Join the parts and check if it forms a valid title
            if len(title_parts) >= 2:  # At least vs and respondent
                full_title = " ".join(title_parts)
                full_title = re.sub(r"\s+", " ", full_title)  # Normalize whitespace
                
                # Check if this looks like a complete case title
                if (CAPTION_RE.search(full_title) and 
                    len(full_title) >= 20 and len(full_title) <= 3000):
                    cleaned = _clean_noise(full_title)
                    if 20 <= len(cleaned) <= 3000:
                        return cleaned
        
        # Special handling for titles that might be complete but have special characters
        # Look for patterns like "NAME, PETITIONER, VS. NAME* RESPONDENT" or "NAME[*] RESPONDENT"
        elif re.search(r"\b(VS\.|V\.|VERSUS)\b", line, re.IGNORECASE) and re.search(r"\b(PETITIONER|RESPONDENT|COMPLAINANT|PLAINTIFF|DEFENDANT)\b", line, re.IGNORECASE):
            # This might be a complete title with special characters
            if len(line) <= 3000:
                cleaned = _clean_noise(line)
                if 20 <= len(cleaned) <= 3000:
                    return cleaned
    
    # Priority 4: Single-line VS. format (fallback)
    for raw in head.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Try the GR notation pattern first
        if CAPTION_WITH_GR_NOTATION_RE.match(line):
            cleaned = _clean_noise(line)
            if 10 <= len(cleaned) <= 3000:  # Allow long titles
                return cleaned
        # Try the notation pattern
        elif CAPTION_WITH_NOTATION_RE.match(line):
            cleaned = _clean_noise(line)
            if 10 <= len(cleaned) <= 3000:  # Allow long titles
                return cleaned
        # Then try other patterns
        elif CAPTION_STRONG_RE.match(line) or CAPTION_RE.search(line):
            cleaned = _clean_noise(line)
            if 10 <= len(cleaned) <= 3000:  # Allow long titles
                return cleaned
    
    # Priority 5: Administrative case formats
    for raw in head.splitlines():
        line = raw.strip()
        if not line:
            continue
        admin_title = _extract_admin_title(line)
        if admin_title and 10 <= len(admin_title) <= 500:
            return admin_title
    
    # Priority 6: Fallback candidates (avoid body text)
    candidates = []
    for raw in head.splitlines():
        line = raw.strip()
        if not line or _is_body_text(line):
            continue
        cleaned = _clean_noise(line)
        if (10 <= len(cleaned) <= 3000 and  # Allow long titles
            not any(n in line.upper() for n in TITLE_NOISE) and
            not _is_body_text(cleaned)):
            candidates.append(cleaned)
    
    # Return the best candidate that looks like a title
    if candidates:
        # Prefer candidates with VS. or similar patterns
        vs_candidates = [c for c in candidates if 'VS.' in c.upper() or 'V.' in c.upper()]
        if vs_candidates:
            # Among VS candidates, prefer those that look like proper case titles
            proper_case_candidates = [c for c in vs_candidates if re.search(r"\b(PETITIONER|RESPONDENT|COMPLAINANT|PLAINTIFF|DEFENDANT)\b", c, re.IGNORECASE)]
            if proper_case_candidates:
                return max(proper_case_candidates, key=len)
            return max(vs_candidates, key=len)
        
        # If no VS candidates, prefer candidates that look like case titles
        case_title_candidates = [c for c in candidates if re.search(r"\b(PETITIONER|RESPONDENT|COMPLAINANT|PLAINTIFF|DEFENDANT)\b", c, re.IGNORECASE)]
        if case_title_candidates:
            return max(case_title_candidates, key=len)
        
        return max(candidates, key=len)
    
    return None

def parse_gr_numbers_from_text(text: str) -> tuple[str | None, list[str]]:
    if not text:
        return None, []
    
    # First, try to find GR numbers in the header (first 10 lines)
    header_lines = text.split('\n')[:10]
    header_text = '\n'.join(header_lines)
    
    header_found = []
    for m in GR_BLOCK_RE.finditer(header_text):
        block = m.group(1)
        parts = [p.strip().rstrip('*') for p in re.split(r"[,\s]+", block) if p.strip()]
        for p in parts:
            if p and p not in header_found:
                header_found.append(p)
    
    # If we found GR numbers in the header, use those
    if header_found:
        found = header_found
    else:
        # Fallback to searching the first 4000 characters
        found = []
        for m in GR_BLOCK_RE.finditer(text[:4000]):
            block = m.group(1)
            parts = [p.strip().rstrip('*') for p in re.split(r"[,\s]+", block) if p.strip()]
            for p in parts:
                if p and p not in found:
                    found.append(p)
    
    # Format GR numbers as "G.R. No. XXXX"
    formatted_found = []
    for gr_num in found:
        if gr_num and gr_num.isdigit():
            formatted_found.append(f"G.R. No. {gr_num}")
        elif gr_num and '-' in gr_num and all(part.isdigit() for part in gr_num.split('-')):
            # Handle ranges like "177857-58"
            formatted_found.append(f"G.R. No. {gr_num}")
        else:
            # Keep as is if already formatted or has special characters
            formatted_found.append(gr_num)
    
    primary = formatted_found[0] if formatted_found else None
    return primary, formatted_found

def parse_special_numbers_from_text(text: str) -> tuple[str | None, list[str]]:
    """Extract special case numbers (A.M., OCA, etc.) for non-GR cases"""
    if not text:
        return None, []
    
    # Patterns for different special case types
    special_patterns = [
        (r"A\.M\.\s+No\.?\s*([A-Z0-9\-]+)", "A.M. No. {}"),
        (r"OCA\s+No\.?\s*([A-Z0-9\-]+)", "OCA No. {}"),
        (r"U\.C\.\s+No\.?\s*([A-Z0-9\-]+)", "U.C. No. {}"),
        (r"ADM\s+No\.?\s*([A-Z0-9\-]+)", "ADM No. {}"),
        (r"A\.C\.\s+No\.?\s*([A-Z0-9\-]+)", "A.C. No. {}"),
        (r"AC\s+No\.?\s*([A-Z0-9\-]+)", "AC No. {}"),
        (r"B\.M\.\s+No\.?\s*([A-Z0-9\-]+)", "B.M. No. {}"),
        (r"LRC\s+No\.?\s*([A-Z0-9\-]+)", "LRC No. {}"),
        (r"SP\s+No\.?\s*([A-Z0-9\-]+)", "SP No. {}"),
    ]
    
    found: list[str] = []
    for pattern, format_str in special_patterns:
        for m in re.finditer(pattern, text[:4000], re.IGNORECASE):
            number = m.group(1).strip()
            formatted = format_str.format(number)
            if formatted not in found:
                found.append(formatted)
    
    primary = found[0] if found else None
    return primary, found

def extract_division_enbanc(text: str) -> tuple[str | None, bool | None]:
    if not text:
        return None, None
    U = text[:800].upper()
    if "EN BANC" in U:
        return "En Banc", True
    for div in ("FIRST DIVISION", "SECOND DIVISION", "THIRD DIVISION"):
        if div in U:
            return div.title(), False
    return None, None

def extract_promulgation_date(text: str) -> str | None:
    """Extract promulgation date, prioritizing header dates over body dates"""
    if not text:
        return None
    
    # First, try to find date in header (first 2000 characters)
    header_text = text[:2000]
    for pattern in HEADER_DATE_PATTERNS:
        match = re.search(pattern, header_text, flags=re.IGNORECASE)
        if match:
            try:
                date_str = match.group(1)
                parsed_date = dtp.parse(date_str).date()
                return parsed_date.isoformat()
            except Exception:
                continue
    
    # If no header date found, try body patterns (but avoid WHEREFORE references)
    for pattern in BODY_DATE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                date_str = match.group(1)
                parsed_date = dtp.parse(date_str).date()
                return parsed_date.isoformat()
            except Exception:
                continue
    
    return None

def extract_ponente(text: str) -> str | None:
    if not text:
        return None
    window = text[:5000]
    
    def _clean_justice_name(raw: str) -> str:
        name = raw.strip()
        # Remove leading DECISION/RESOLUTION tokens, including spaced variants (D E C I S I O N)
        name = re.sub(r"^(?:DECISION\s+|RESOLUTION\s+)", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^(?:D\s*E\s*C\s*I\s*S\s*I\s*O\s*N\s+)", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^(?:R\s*E\s*S\s*O\s*L\s*U\s*T\s*I\s*O\s*N\s+)", "", name, flags=re.IGNORECASE)
        return name.strip()
    # Prefer explicit justice suffixes: J., JJ., CJ, SAJ (optionally followed by ':' or '.')
    # Handle titles like SR., JR., III, etc. before the justice suffix
    # Also handle compound names like "CARPIO MORALES"
    patterns = [
        # Simple pattern for common cases like "GESMUNDO, C.J.:" - no word boundary
        r"([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)*)\s*,\s*(C\.J\.|J\.|JJ\.|CJ\.|CJ|SAJ)\s*[:\.]?",
        # Handle spaced DECISION pattern: "D E C I S I O N GESMUNDO, C.J.:"
        r"(?:D\s*E\s*C\s*I\s*S\s*I\s*O\s*N\s+)([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)*)\s*,\s*(C\.J\.|J\.|JJ\.|CJ\.|CJ|SAJ)\s*[:\.]?",
        # More complex patterns for edge cases
        r"([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)*(?:\s+(?:SR\.|JR\.|III|IV|V|VI|VII|VIII|IX|X))*)\s*,\s*(J\.|JJ\.|C\.J\.|CJ\.|CJ|SAJ)\s*[:\.]?",
        # Some pages omit dot after J
        r"([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)*(?:\s+(?:SR\.|JR\.|III|IV|V|VI|VII|VIII|IX|X))*)\s*,\s*(J|JJ|C\.J\.|CJ\.|CJ|SAJ)\s*[:\.]?",
    ]
    for pat in patterns:
        m = re.search(pat, window)
        if m:
            name = _clean_justice_name(m.group(1))
            suffix = m.group(2).strip()
            if suffix and not suffix.endswith('.') and suffix in ("J", "JJ"):
                suffix = suffix + "."
            return f"{name}, {suffix}"
    # As a fallback, keep prior behavior (name only before J.)
    m = re.search(r"\b([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)*)\s*,\s*J\.?\b", window)
    if m:
        return f"{_clean_justice_name(m.group(1))}, J."
    return None

# -----------------------------
# Section Extraction for Chunking Alignment
# -----------------------------
def extract_sections_for_chunking(text: str) -> Dict[str, str]:
    """Extract structured sections that align with the chunking strategy"""
    if not text:
        return {}
    
    sections = {}
    
    # Extract dispositive/ruling first (highest priority)
    ruling_match = RULING_REGEX.search(text)
    if ruling_match:
        sections['ruling'] = ruling_match.group(0).strip()
    
    # Simple heuristic patterns for other sections
    section_patterns = [
        (r'THE FACTS?[\s\S]*?(?=THE ISSUE|THE RULING|WHEREFORE|$)', 'facts'),
        (r'ANTECEDENT FACTS?[\s\S]*?(?=THE ISSUE|THE RULING|WHEREFORE|$)', 'facts'),
        (r'THE ISSUE[S]?[\s\S]*?(?=THE RULING|WHEREFORE|$)', 'issues'),
        (r'ISSUE[S]?[\s\S]*?(?=THE RULING|WHEREFORE|$)', 'issues'),
        (r'THE RULING[\s\S]*?(?=WHEREFORE|$)', 'arguments'),
        (r'DISCUSSION[\s\S]*?(?=WHEREFORE|$)', 'arguments'),
    ]
    
    for pattern, section_name in section_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and section_name not in sections:
            content = match.group(0).strip()
            if len(content) > 100:  # Only add substantial content
                sections[section_name] = content
    
    # Add header section (first part of document)
    lines = text.split('\n')
    header_lines = []
    for i, line in enumerate(lines):
        if i > 20:  # Limit header to first 20 lines
            break
        if any(indicator in line.upper() for indicator in ['DECISION', 'RESOLUTION', 'ORDER']):
            break
        header_lines.append(line)
    
    if header_lines:
        sections['header'] = '\n'.join(header_lines).strip()
    
    return sections

def load_existing_ids(path: str) -> set[str]:
    ids: set[str] = set()
    if not os.path.exists(path):
        return ids
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = orjson.loads(line)
                    rid = rec.get("id")
                    if isinstance(rid, str):
                        ids.add(rid)
                except Exception:
                    continue
    except Exception:
        pass
    return ids

# -----------------------------
# Discovery (requests + BS4): months → case URLs (+ year/month hints)
# -----------------------------
def fetch_page(url: str, timeout=15, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 503:
                print(f"⚠️ Server unavailable (503), retrying in {2**attempt} seconds...")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2**attempt)
                    continue
            r.raise_for_status()
            return BeautifulSoup(r.content, "html.parser")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Request failed (attempt {attempt + 1}), retrying...")
                import time
                time.sleep(2**attempt)
            else:
                print(f"❌ All retry attempts failed: {e}")
    return None

def extract_month_links():
    soup = fetch_page(BASE_URL)
    if not soup:
        return []
    container = soup.find("div", id="container_date")
    if not container:
        return []

    out, current_year = [], None
    for tag in container.find_all(["h2", "a"]):
        if tag.name == "h2":
            try:
                current_year = int(tag.text.strip())
            except ValueError:
                current_year = None
        elif tag.name == "a" and current_year and YEAR_START <= current_year <= YEAR_END:
            href = tag.get("href")
            if not href:
                continue
            month_url = href if href.startswith("http") else urljoin(BASE_URL, href)
            out.append({"year": current_year, "month_text": tag.text.strip().lower(), "url": month_url})
    return out

def extract_decision_links(month_url: str):
    soup = fetch_page(month_url)
    if not soup:
        return []
    container = soup.find("div", id="container_title")
    if not container:
        return []
    links = []
    for a in container.find_all("a", href=True):
        href = a["href"]
        if "showdocs" in href:
            links.append(href if href.startswith("http") else urljoin(BASE_URL, href))
    return links

def discover_case_urls():
    months = extract_month_links()
    print(f"🗓️ Found {len(months)} month pages for {YEAR_START}-{YEAR_END}")
    items, seen = [], set()
    for m in months:
        links = extract_decision_links(m["url"])
        print(f"  • {m['month_text'].title()} {m['year']}: {len(links)} cases")
        for u in links:
            if u in seen:
                continue
            seen.add(u)
            items.append({"url": u, "year_hint": m["year"], "month_hint": m["month_text"]})
    print(f"🔗 Total unique case URLs: {len(items)}")
    return items

# -----------------------------
# Title extraction (HTML → MD fallback)
# -----------------------------
TITLE_BAD = {
    "EN BANC", "FIRST DIVISION", "SECOND DIVISION", "THIRD DIVISION",
    "PER CURIAM", "REPUBLIC OF THE PHILIPPINES", "SUPREME COURT"
}

def _is_title_like(line: str) -> bool:
    s = line.strip()
    if not (10 <= len(s) <= 3000):  # Allow very long titles for multi-party cases
        return False
    U = s.upper()
    if any(bad in U for bad in TITLE_BAD):
        return False
    return bool(re.search(r"\b(v\.|vs\.?| versus )\b", s, flags=re.I)) or "People of the Philippines" in s

def extract_title_from_html(html: str) -> tuple[str | None, str | None]:
    """
    Returns (title, page_title) — page_title is raw <title> (for debugging).
    """
    if not html:
        return None, None
    tree = HTMLParser(html)
    root = tree.css_first("div.single_content") or tree.body

    # 1) <h1>
    if root:
        h1 = root.css_first("h1")
        if h1:
            t = h1.text(strip=True)
            if _is_title_like(t) or len(t) >= 10:
                pt = (tree.css_first("title").text(strip=True) if tree.css_first("title") else None)
                return t, pt

    # 2) scan common blocks in order: all <center>, all <h2>, first 25 lines of single_content
    def scan_lines(lines):
        for l in lines:
            if _is_title_like(l):
                pt = (tree.css_first("title").text(strip=True) if tree.css_first("title") else None)
                return l, pt
        candidates = [l for l in lines if not any(b in l.upper() for b in TITLE_BAD)]
        if candidates:
            t = max(candidates, key=len)
            pt = (tree.css_first("title").text(strip=True) if tree.css_first("title") else None)
            return t, pt
        return None, None

    if root:
        centers = root.css("center") or []
        for cent in centers:
            lines = [l.strip() for l in cent.text(separator="\n").splitlines() if l.strip()]
            t, pt = scan_lines(lines)
            if t:
                return t, pt

        h2s = root.css("h2") or []
        for h2 in h2s:
            t = h2.text(strip=True)
            if t and _is_title_like(t):
                pt = (tree.css_first("title").text(strip=True) if tree.css_first("title") else None)
                return t, pt

        # first 25 lines under single_content
        block = root.text(separator="\n").splitlines()[:25]
        lines = [l.strip() for l in block if l.strip()]
        t, pt = scan_lines(lines)
        if t:
            return t, pt

    # 3) <title>
    tnode = tree.css_first("title")
    page_title = tnode.text(strip=True) if tnode else None
    if page_title:
        t = re.split(r"\s+[\|\-–]\s+", page_title)[0]
        if len(t) >= 10:
            return t, page_title

    return None, page_title

def extract_title_from_markdown(md: str) -> str | None:
    if not md:
        return None
    m = re.search(r"^#{1,2}\s+(.+)$", md, flags=re.M)
    if m:
        return m.group(1).strip()
    for line in md.splitlines():
        if _is_title_like(line):
            return line.strip()
    return None

def extract_title(md: str | None, html: str | None) -> tuple[str | None, str | None]:
    # Prefer HTML (more structure), then MD
    t, pt = extract_title_from_html(html or "")
    if t:
        return t, pt
    return extract_title_from_markdown(md or ""), pt

# -----------------------------
# Parse case (from text) + build record
# -----------------------------
def classify_case_type(title: str, gr_number: str | None = None, full_text: str | None = None) -> dict[str, Any]:
    """Classify primary case_type as 'civil' or 'criminal' and list subtypes.

    Heuristics:
    - Primary type = 'criminal' if title or early body mentions 'People of the Philippines' or common criminal terms; else 'civil'.
    - Subtypes include: certiorari, mandamus, prohibition, habeas_corpus, quo_warranto, appeal, petition,
      administrative_matter, disciplinary_complaint, motion, resolution, order, consolidated, election, labor, tax, etc.
    """
    title_upper = (title or "").upper()
    text_scan = (full_text or "")[:4000].upper()

    subtypes: list[str] = []

    # Helper to add subtype if any keyword appears
    def add_if_any(keywords: list[str], label: str):
        if any(k in title_upper or k in text_scan for k in keywords):
            if label not in subtypes:
                subtypes.append(label)

    # Special Civil Actions / Writs
    add_if_any(["CERTIORARI"], "certiorari")
    add_if_any(["MANDAMUS"], "mandamus")
    add_if_any(["PROHIBITION"], "prohibition")
    add_if_any(["HABEAS CORPUS"], "habeas_corpus")
    add_if_any(["QUO WARRANTO"], "quo_warranto")
    add_if_any(["DECLARATORY RELIEF"], "declaratory_relief")
    add_if_any(["INTERPLEADER"], "interpleader")
    add_if_any(["INJUNCTION", "PRELIMINARY INJUNCTION"], "injunction")

    # Civil: Family and Persons
    add_if_any(["ANNULMENT OF MARRIAGE"], "annulment_of_marriage")
    add_if_any(["DECLARATION OF NULLITY"], "declaration_of_nullity")
    add_if_any(["LEGAL SEPARATION"], "legal_separation")
    add_if_any(["ADOPTION"], "adoption")
    add_if_any(["GUARDIANSHIP"], "guardianship")
    add_if_any(["SUPPORT"], "support")
    add_if_any(["PATERNITY", "FILIATION"], "paternity_filiation")
    add_if_any(["CHANGE OF NAME", "CORRECTION OF NAME", "CORRECTION OF ENTRY", "CIVIL STATUS"], "change_or_correction_of_name_or_status")

    # Civil: Property and Ownership
    add_if_any(["EJECTMENT", "FORCIBLE ENTRY", "UNLAWFUL DETAINER"], "ejectment")
    add_if_any(["QUIETING OF TITLE"], "quieting_of_title")
    add_if_any(["PARTITION"], "partition")
    add_if_any(["RECONVEYANCE"], "reconveyance")
    add_if_any(["ACCION REIVINDICATORIA", "ACCION PUBLICIANA", "RECOVERY OF POSSESSION"], "recovery_of_possession")
    add_if_any(["FORECLOSURE OF MORTGAGE", "FORECLOSURE OF REAL ESTATE MORTGAGE"], "foreclosure_of_mortgage")
    add_if_any(["REFORMATION OF INSTRUMENTS"], "reformation_of_instruments")

    # Civil: Obligations and Contracts
    add_if_any(["COLLECTION OF SUM", "SUM OF MONEY"], "collection_of_sum_of_money")
    add_if_any(["BREACH OF CONTRACT"], "breach_of_contract")
    add_if_any(["SPECIFIC PERFORMANCE"], "specific_performance")
    add_if_any(["RESCISSION"], "rescission_of_contract")
    add_if_any(["DAMAGES", "NEGLIGENCE", "QUASI-DELICT", "TORT"], "damages")

    # Civil: Succession and Estates
    add_if_any(["SETTLEMENT OF ESTATE", "SPECIAL PROCEEDINGS"], "settlement_of_estate")
    add_if_any(["PROBATE OF WILL", "PROBATE"], "probate")
    add_if_any(["INTESTATE"], "intestate")

    # Civil: Other topics
    add_if_any(["EXPROPRIATION", "EMINENT DOMAIN"], "expropriation")

    # Administrative flavor (still civil primary) - but not for criminal cases
    # Only add administrative_matter if it's not a criminal case
    if not any(tok in title_upper for tok in ["PEOPLE OF THE PHILIPPINES", "INFORMATION", "ESTAFA", "MURDER", "HOMICIDE"]):
        add_if_any(["ADMINISTRATIVE"], "administrative_matter")
        if "COMPLAINANT" in title_upper and "RESPONDENT" in title_upper:
            add_if_any(["COMPLAINANT"], "disciplinary_complaint")

    # Motions / Resolutions / Orders
    add_if_any(["MOTION"], "motion")
    add_if_any(["RESOLUTION"], "resolution")
    add_if_any(["ORDER"], "order")

    # Labor / Tax / Election
    add_if_any(["LABOR", "NLRC"], "labor")
    add_if_any(["TAX", "BIR"], "tax")
    add_if_any(["ELECTION", "COMELEC", "ELECTORAL"], "election")

    # Consolidated
    if gr_number and '-' in str(gr_number):
        if "consolidated" not in subtypes:
            subtypes.append("consolidated")

    # Criminal indicators
    criminal_indicators = [
        "PEOPLE OF THE PHILIPPINES", "INFORMATION", "ESTAFA", "MURDER", "HOMICIDE",
        "RAPE", "ROBBERY", "THEFT", "ILLEGAL DRUGS", "DANGEROUS DRUGS", "QUALIFIED",
        "KIDNAPPING", "DETENTION", "CARNAPPING", "ARSON", "LIBEL", "SLANDER",
        "REBEL", "SEDITION", "COUP", "ASSAULT", "PERJURY", "FORGERY", "COUNTERFEIT",
        "FALSIFICATION", "BRIBERY", "MALVERSATION", "GAMBLING", "OBSCENITY", "PORNOGRAPHY",
        "TRAFFICKING", "MONEY LAUNDERING", "TERRORISM"
    ]
    is_criminal = any(tok in title_upper for tok in criminal_indicators) or any(
        tok in text_scan for tok in criminal_indicators
    )

    # Criminal subtypes
    add_if_any(["MURDER"], "murder")
    add_if_any(["HOMICIDE"], "homicide")
    add_if_any(["PARRICIDE"], "parricide")
    add_if_any(["INFANTICIDE"], "infanticide")
    add_if_any(["PHYSICAL INJURIES"], "physical_injuries")
    add_if_any(["RAPE"], "rape")
    add_if_any(["ACTS OF LASCIVIOUSNESS"], "acts_of_lasciviousness")
    add_if_any(["FRUSTRATED HOMICIDE", "ATTEMPTED HOMICIDE"], "frustrated_or_attempted_homicide")

    add_if_any(["THEFT"], "theft")
    add_if_any(["ROBBERY"], "robbery")
    add_if_any(["CARNAPPING"], "carnapping")
    add_if_any(["MALICIOUS MISCHIEF"], "malicious_mischief")
    add_if_any(["ESTAFA"], "estafa")
    add_if_any(["ARSON"], "arson")

    add_if_any(["KIDNAPPING", "ILLEGAL DETENTION"], "kidnapping_or_illegal_detention")
    add_if_any(["GRAVE THREATS"], "grave_threats")
    add_if_any(["GRAVE COERCION"], "grave_coercion")
    add_if_any(["SLAVERY", "CHILD LABOR"], "slavery_or_child_labor")

    add_if_any(["ADULTERY"], "adultery")
    add_if_any(["CONCUBINAGE"], "concubinage")
    add_if_any(["SEDuction".upper()], "seduction")

    add_if_any(["LIBEL"], "libel")
    add_if_any(["SLANDER", "ORAL DEFAMATION"], "slander")
    add_if_any(["INCRIMINATING INNOCENT PERSONS"], "incriminating_innocent_persons")
    add_if_any(["INTRIGUING AGAINST HONOR"], "intriguing_against_honor")

    add_if_any(["FORGERY"], "forgery")
    add_if_any(["COUNTERFEIT", "COUNTERFEITING"], "counterfeiting")
    add_if_any(["USE OF FALSIFIED", "FALSIFIED DOCUMENTS"], "use_of_falsified_documents")
    add_if_any(["PERJURY"], "perjury")

    add_if_any(["REBELLION"], "rebellion")
    add_if_any(["SEDITION"], "sedition")
    add_if_any(["COUP D", "COUP D’ETAT", "COUP D'ETAT"], "coup_detat")
    add_if_any(["DIRECT ASSAULT"], "direct_assault")
    add_if_any(["RESISTANCE AND DISOBEDIENCE", "DISOBEDIENCE"], "resistance_or_disobedience")

    add_if_any(["GAMBLING", "JUETENG"], "gambling")
    add_if_any(["OBSCENITY", "PORNOGRAPHY"], "obscenity_or_pornography")

    add_if_any(["BRIBERY"], "bribery")
    add_if_any(["CORRUPTION OF PUBLIC OFFICIALS"], "corruption_of_public_officials")
    add_if_any(["MALVERSATION"], "malversation")
    add_if_any(["FALSIFICATION BY PUBLIC OFFICERS"], "falsification_by_public_officers")

    add_if_any(["DANGEROUS DRUGS", "RA 9165"], "dangerous_drugs")
    add_if_any(["CYBERCRIME", "RA 10175"], "cybercrime")
    add_if_any(["ANTI-VAWC", "VAWC", "RA 9262"], "anti_vawc")
    add_if_any(["ANTI-HAZING", "HAZING", "RA 11053"], "anti_hazing")
    add_if_any(["TERRORISM", "ANTI-TERRORISM", "RA 11479"], "anti_terrorism")
    add_if_any(["TRAFFICKING IN PERSONS", "RA 9208"], "anti_trafficking")
    add_if_any(["MONEY LAUNDERING", "RA 9160"], "anti_money_laundering")

    # Special handling for A.M. cases - they should be administrative, not criminal
    is_am_case = "A.M." in title_upper or "A.M." in text_scan
    if is_am_case:
        case_type = "administrative"
    else:
        case_type = "criminal" if is_criminal else "civil"

    # Basic fallbacks for petitions without explicit label
    if ("PETITION" in title_upper or "PETITION" in text_scan) and not any(
        s in subtypes for s in ("certiorari", "mandamus", "prohibition", "habeas_corpus", "appeal")
    ):
        subtypes.append("petition")
    
    return {
        "case_type": case_type,
        "case_subtypes": subtypes or None,
    }

def parse_case(text_base: str, url: str, year_hint: int | None = None, month_hint: str | None = None,
               title_guess: str | None = None, page_title: str | None = None):
    # Clean site boilerplate then normalize whitespace
    text = normalize_text(clean_website_headers(text_base or ""))

    # Case title (prefer derived caption over generic page guess)
    derived_title = derive_case_title_from_text(text)
    case_title = derived_title or (title_guess if title_guess and CAPTION_RE.search(title_guess) else title_guess)

    # GR numbers (primary + list)
    gr_primary, gr_all = parse_gr_numbers_from_text(text)
    
    # Special numbers for non-GR cases (A.M., OCA, etc.)
    special_primary, special_all = parse_special_numbers_from_text(text)

    # Date (prioritize header dates over body dates)
    date_iso = extract_promulgation_date(text)

    # Simple ruling presence flag from full text
    has_ruling_flag = bool(RULING_REGEX.search(text or ""))

    # Division / En Banc and Ponente
    division, en_banc = extract_division_enbanc(text)
    ponente = extract_ponente(text)
    # If en banc or PER CURIAM indicated and no individual justice found, set ponente accordingly
    if (en_banc is True or "PER CURIAM" in text[:3000].upper()) and not ponente:
        ponente = "PER CURIAM"
    
    # Classify case type
    classification = classify_case_type(case_title or title_guess or "", gr_primary, text)

    # Robust title fallback if still missing (avoid body text)
    def is_valid_title_candidate(title: str) -> bool:
        if not title or len(title) < 10:
            return False
        title_lower = title.lower()
        
        # Check if it looks like body text
        if any(indicator in title_lower for indicator in BODY_TEXT_INDICATORS):
            return False
        
        # Check for case patterns that indicate legitimate long titles
        # Many Philippine SC cases have numerous co-petitioners/respondents
        is_multi_party_case = ('vs.' in title_lower or 'versus' in title_lower) and title_lower.count(',') > 3
        is_admin_case = any(pattern in title_lower for pattern in ['re:', 'a.m. no.', 'administrative matter'])
        
        # Dynamic max length based on case type:
        # - Multi-party cases (many commas): up to 3000 chars (legitimate full party lists)
        # - Administrative cases: up to 500 chars
        # - Regular cases: up to 400 chars
        if is_multi_party_case:
            max_length = 3000
        elif is_admin_case:
            max_length = 500
        else:
            max_length = 400
            
        if len(title) > max_length:
            return False
            
        # Additional validation: reject titles that look like procedural text
        procedural_indicators = [
            'held in abeyance', 'payment of', 'retirement benefits',
            'subject to whatever', 'penalties the court', 'may impose',
            'relative to another', 'docketed as', 'resolution of',
            'decision of', 'order of', 'the court hereby'
        ]
        if any(indicator in title_lower for indicator in procedural_indicators):
            return False
            
        return True
    
    # Try fallback options in order of preference
    fallback_title = case_title
    if not fallback_title or not is_valid_title_candidate(fallback_title):
        fallback_title = title_guess if is_valid_title_candidate(title_guess or "") else None
    if not fallback_title:
        fallback_title = page_title if is_valid_title_candidate(page_title or "") else None
    
    # If still no valid title, try to construct one from available metadata
    if not fallback_title:
        title_parts = []
        
        # Try to build title from special number or GR number
        if special_primary:
            title_parts.append(special_primary)
        elif gr_primary:
            title_parts.append(f"G.R. No. {gr_primary}")
            
        # Add division info if available
        if division:
            title_parts.append(f"({division})")
            
        # Add case type if it's administrative
        if classification.get('case_type') == 'administrative':
            title_parts.append("Administrative Matter")
        
        if title_parts:
            fallback_title = " ".join(title_parts)
        else:
            # Last resort: use URL fragment or generic title
            url_fragment = url.rsplit('/', 1)[-1] if url else None
            fallback_title = url_fragment if is_valid_title_candidate(url_fragment or "") else "Untitled Case"

    # Extract structured sections for chunking alignment
    sections = extract_sections_for_chunking(text)
    
    # Add quality metrics (drop per-section booleans)
    quality_metrics = {
        "has_ruling": has_ruling_flag,
        "text_length": len(text),
        "sections_extracted": len(sections),
        "has_structured_sections": len(sections) > 1,
    }

    record = {
        "id": sha256(url),
        "gr_number": gr_primary,
        "gr_numbers": gr_all or None,
        "special_number": special_primary,
        "special_numbers": special_all or None,
        "case_title": case_title,
        "page_title": page_title,              # debug/optional
        "promulgation_date": date_iso,         # may be None
        "promulgation_year": year_hint,        # Fix A: carry year hint from listing
        "promulgation_month": month_hint,      # optional
        "court": "Supreme Court",
        "source_url": url,
        "clean_version": "v2.2",  # Updated version for sections + chunking alignment
        "checksum": sha256(text),
        "crawl_ts": datetime.utcnow().isoformat() + "Z",
        "ponente": ponente,
        "division": division,
        "en_banc": en_banc,
        # Always include cleaned full text for hybrid retrieval
        "clean_text": text,
        # Structured sections for chunking alignment
        "sections": sections if sections else None,
        # Helpful flags
        "has_gr_number": bool(gr_primary),
        "has_special_number": bool(special_primary),
        # Case classification
        "case_type": classification.get("case_type"),
        "case_subtypes": classification.get("case_subtypes"),
        # Back-compat single subtype for older consumers (first of list)
        "case_subtype": (classification.get("case_subtypes") or [None])[0],
        # Quality metrics
        "quality_metrics": quality_metrics,
    }
    return record

# -----------------------------
# Async fetching
# -----------------------------
async def fetch_playwright(crawler: AsyncWebCrawler, url: str) -> tuple[str, str]:
    run_cfg = CrawlerRunConfig(
        exclude_external_links=True,
        remove_overlay_elements=True,
        page_timeout=TIMEOUT_S * 1000,  # Convert to milliseconds
        # render_js=True,  # uncomment only if truly needed
    )
    res = await crawler.arun(
        url=url,
        config=run_cfg,
        browser_config=BrowserConfig(headless=True, browser_type="chromium"),
    )
    md = getattr(res, "markdown", "") or ""
    html = getattr(res, "html", "") or ""
    return md, html

async def crawl_all_with_playwright(items: list[dict], out_path: str, existing_ids: set[str]) -> int:
    # Note: Playwright is not thread-safe, but we use asyncio (not threading) with semaphore for concurrency control
    sem = asyncio.Semaphore(CONCURRENCY)
    written = 0

    async with AsyncWebCrawler() as crawler:
        with gzip.open(out_path, "at", encoding="utf-8") as f:

            async def worker(item: dict, idx: int, total: int):
                nonlocal written
                async with sem:
                    u = item["url"]
                    rid = sha256(u)
                    if rid in existing_ids:
                        return
                    try:
                        md, html = await fetch_playwright(crawler, u)
                        title_guess, page_title = extract_title(md, html)
                        text_base = md or (HTMLParser(html).body.text(separator="\n") if html else "")
                        rec = parse_case(text_base, u, year_hint=item.get("year_hint"),
                                         month_hint=item.get("month_hint"),
                                         title_guess=title_guess, page_title=page_title)
                        f.write(orjson.dumps(rec).decode("utf-8") + "\n")
                        existing_ids.add(rec["id"])
                        written += 1
                        await asyncio.sleep(SLOWDOWN_MS / 1000.0)
                    except Exception as e:
                        print(f"⚠️  PW fail {idx}/{total}: {u} — {e}")

            tasks = [worker(item, i + 1, len(items)) for i, item in enumerate(items)]
            for start in range(0, len(tasks), WRITE_CHUNK):
                await asyncio.gather(*tasks[start:start + WRITE_CHUNK])
                print(f"📝 PW progress: {written}/{start + min(WRITE_CHUNK, len(tasks) - start)} written")

    return written

async def crawl_all_fallback_requests(items: list[dict], out_path: str, existing_ids: set[str]) -> int:
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    sem = asyncio.Semaphore(CONCURRENCY)
    written = 0

    async with aiohttp.ClientSession(connector=connector, headers={"User-Agent": UA}) as session:
        with gzip.open(out_path, "at", encoding="utf-8") as f:

            async def fetch_http(u: str, retries: int = 2) -> str:
                delay = 0.5
                for attempt in range(retries + 1):
                    try:
                        async with session.get(u, timeout=TIMEOUT_S) as resp:
                            resp.raise_for_status()
                            return await resp.text()
                    except Exception:
                        if attempt >= retries:
                            raise
                        await asyncio.sleep(delay)
                        delay *= 2

            async def worker(item: dict, idx: int, total: int):
                nonlocal written
                async with sem:
                    u = item["url"]
                    rid = sha256(u)
                    if rid in existing_ids:
                        return
                    try:
                        html = await fetch_http(u)
                        if not html or len(html) < 1000:  # Skip very short responses
                            return
                        
                        # Extract title and text more carefully
                        title_guess, page_title = extract_title(None, html)
                        text = HTMLParser(html).body.text(separator="\n") if html else ""
                        
                        # Skip if we only got boilerplate
                        if "Supreme Court E-Library" in text and len(text) < 2000:
                            return
                            
                        rec = parse_case(text, u, year_hint=item.get("year_hint"),
                                         month_hint=item.get("month_hint"),
                                         title_guess=title_guess, page_title=page_title)
                        f.write(orjson.dumps(rec).decode("utf-8") + "\n")
                        existing_ids.add(rec["id"])
                        written += 1
                        await asyncio.sleep(SLOWDOWN_MS / 1000.0)
                    except Exception as e:
                        print(f"⚠️  HTTP fail {idx}/{total}: {u} — {e}")

            tasks = [worker(item, i + 1, len(items)) for i, item in enumerate(items)]
            for start in range(0, len(tasks), WRITE_CHUNK * 2):
                await asyncio.gather(*tasks[start:start + WRITE_CHUNK * 2])
                print(f"📝 HTTP progress: {written}/{start + min(WRITE_CHUNK * 2, len(tasks) - start)} written")

    return written

async def crawl_all(items: list[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    existing_ids = load_existing_ids(out_path)
    existing_grs = set()
    # Build lightweight GR dedupe set
    opener = gzip.open if out_path.endswith(".gz") else open
    try:
        with opener(out_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = orjson.loads(line)
                    g = rec.get("gr_number")
                    if isinstance(g, str):
                        existing_grs.add(g)
                except Exception:
                    continue
    except Exception:
        pass
    print(f"🧹 Skipping {len(existing_ids)} by URL and {len(existing_grs)} by GR; crawling {len(items)} candidates")

    # Try HTTP first (faster), then Playwright fallback if needed
    try:
        print("▶ Trying HTTP mode (faster)…")
        w = await crawl_all_fallback_requests(items, out_path, existing_ids)
        print(f"✅ HTTP mode wrote {w} records")
        
        # If HTTP got very few records, try Playwright for the remaining
        if w < len(items) * 0.5:  # If less than 50% success rate
            print(f"⚠️  HTTP success rate low ({w}/{len(items)}), trying Playwright for remaining...")
            remaining_items = [item for item in items if sha256(item["url"]) not in existing_ids]
            if remaining_items:
                pw_w = await crawl_all_with_playwright(remaining_items, out_path, existing_ids)
                print(f"✅ Playwright fallback wrote {pw_w} additional records")
    except Exception as e:
        print(f"⚠️  HTTP failed: {e}, trying Playwright...")
        try:
            w = await crawl_all_with_playwright(items, out_path, existing_ids)
            print(f"✅ Playwright mode wrote {w} records")
        except Exception as pw_e:
            print(f"❌ Both HTTP and Playwright failed: {pw_e}")
            raise

def main():
    items = discover_case_urls()
    asyncio.run(crawl_all(items, OUT_PATH))

if __name__ == "__main__":
    main()
