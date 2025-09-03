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
from typing import Any
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
OUT_PATH     = os.getenv("CASES_JSONL", "backend/data/cases.jsonl.gz")
UA           = os.getenv("CRAWLER_UA", "Mozilla/5.0 (compatible; PHLawBot/1.0)")
YEAR_START   = int(os.getenv("YEAR_START", 2012))
YEAR_END     = int(os.getenv("YEAR_END", 2012))
CONCURRENCY  = int(os.getenv("CONCURRENCY", 12))  # Higher for HTTP-first approach
SLOWDOWN_MS  = int(os.getenv("SLOWDOWN_MS", 250))
TIMEOUT_S    = int(os.getenv("TIMEOUT_S", 45))
WRITE_CHUNK  = int(os.getenv("WRITE_CHUNK", 1000))  # tasks per gather batch

HEADERS = {"User-Agent": UA}

RULING_REGEX = re.compile(
    r"(WHEREFORE.*?SO ORDERED\.?|ACCORDINGLY.*?SO ORDERED\.?)",
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
    # Avoid dates in WHEREFORE clauses that reference lower court decisions
    r"(?<!WHEREFORE.*?promulgated\s+on\s+)(?<!affirms.*?promulgated\s+on\s+)(?<!decision\s+promulgated\s+on\s+)(\w+\s+\d{1,2},\s+\d{4})",
]

# -----------------------------
# Utils
# -----------------------------
def sha256(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()

# -----------------------------
# Field extractors (caption, GR numbers, division, ponente)
# -----------------------------
TITLE_NOISE = (
    "SUPREME COURT", "REPUBLIC OF THE PHILIPPINES", "EN BANC", "FIRST DIVISION",
    "SECOND DIVISION", "THIRD DIVISION", "PER CURIAM",
)

CAPTION_RE = re.compile(r"\b(.+?)\s+(v\.|vs\.?|versus)\s+(.+?)\b", re.IGNORECASE)
CAPTION_STRONG_RE = re.compile(r"^[A-Z0-9 ,.'\-()]+\s+(V\.|VS\.?|VERSUS)\s+[A-Z0-9 ,.'\-()]+$")

# Additional patterns for edge cases
ADMIN_COMPLAINT_RE = re.compile(r"^(.+?),\s*COMPLAINANT,\s*VS\.?\s*(.+?),\s*RESPONDENT", re.IGNORECASE)

# Enhanced RE: prefix patterns for various administrative case formats
RE_PREFIX_PATTERNS = [
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
    "of the court of appeals", "dated", "resolution", "decision"
]
GR_BLOCK_RE = re.compile(r"G\.R\.\s+No(?:s)?\.?\s*([0-9][0-9\-*,\s]+)", re.IGNORECASE)

def _clean_noise(line: str) -> str:
    s = line.strip()
    for t in TITLE_NOISE:
        s = s.replace(t, "")
    return re.sub(r"\s{2,}", " ", s).strip(",;:- ")

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
                return f"{complainant}, COMPLAINANT, VS. {respondent}, RESPONDENT"
            elif len(match.groups()) == 1:
                # Pattern with single group (like SUBPOENA or ANONYMOUS LETTER)
                return match.group(1).strip()
    
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
                return case_description
    
    return None

def derive_case_title_from_text(text: str) -> str | None:
    if not text:
        return None
    head = text[:1500]
    
    # Priority 1: Standard VS. format
    for raw in head.splitlines():
        line = raw.strip()
        if not line:
            continue
        if CAPTION_STRONG_RE.match(line) or CAPTION_RE.search(line):
            cleaned = _clean_noise(line)
            if 10 <= len(cleaned) <= 400:
                return cleaned
    
    # Priority 2: Administrative case formats
    for raw in head.splitlines():
        line = raw.strip()
        if not line:
            continue
        admin_title = _extract_admin_title(line)
        if admin_title and 10 <= len(admin_title) <= 400:
            return admin_title
    
    # Priority 3: Fallback candidates (avoid body text)
    candidates = []
    for raw in head.splitlines():
        line = raw.strip()
        if not line or _is_body_text(line):
            continue
        cleaned = _clean_noise(line)
        if (10 <= len(cleaned) <= 400 and 
            not any(n in line.upper() for n in TITLE_NOISE) and
            not _is_body_text(cleaned)):
            candidates.append(cleaned)
    
    # Return the longest candidate that looks like a title
    if candidates:
        # Prefer candidates with VS. or similar patterns
        vs_candidates = [c for c in candidates if 'VS.' in c.upper() or 'V.' in c.upper()]
        if vs_candidates:
            return max(vs_candidates, key=len)
        return max(candidates, key=len)
    
    return None

def parse_gr_numbers_from_text(text: str) -> tuple[str | None, list[str]]:
    if not text:
        return None, []
    found: list[str] = []
    for m in GR_BLOCK_RE.finditer(text[:4000]):
        block = m.group(1)
        parts = [p.strip().rstrip('*') for p in re.split(r"[,\s]+", block) if p.strip()]
        for p in parts:
            if p and p not in found:
                found.append(p)
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
    for pat in [
        r"\b([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)*)\s*,\s*J\.?\b",
        r"\b([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)*)\s*,\s*SAJ\b",
        r"\b([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)*)\s*,\s*Chairperson\b",
    ]:
        m = re.search(pat, text[:4000])
        if m:
            return m.group(1)
    return None

def split_sections(text: str):
    m = RULING_REGEX.search(text or "")
    if not m:
        return {
            "header": (text[:1200] or "").strip(),
            "body": (text or "").strip(),
            "ruling": "",
        }
    rs, re_ = m.span()
    header = (text[:1200] or "").strip()
    ruling = text[rs:re_].strip()
    body = (text[:max(0, rs)] + "\n\n" + text[re_:]).strip()
    return {"header": header, "body": body, "ruling": ruling}

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
def fetch_page(url: str, timeout=15):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return BeautifulSoup(r.content, "html.parser")
    except Exception:
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
    if not (10 <= len(s) <= 200):
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
def classify_case_type(title: str, gr_number: str | None = None) -> dict[str, Any]:
    """Classify case type and extract relevant metadata"""
    title_upper = title.upper()
    
    # Disciplinary complaints (can have A.M. numbers) - check this first
    if 'COMPLAINANT' in title_upper and 'RESPONDENT' in title_upper:
        return {
            "case_type": "administrative", 
            "case_subtype": "disciplinary_complaint",
            "is_administrative": True,
            "is_regular_case": False
        }
    
    # Other administrative cases (A.M. numbers without complaint structure)
    if 'A.M.' in title_upper or 'ADMINISTRATIVE' in title_upper:
        return {
            "case_type": "administrative",
            "case_subtype": "administrative_matter",
            "is_administrative": True,
            "is_regular_case": False
        }
    
    # Regular cases with G.R. numbers
    if gr_number and (str(gr_number).isdigit() or 'G.R.' in str(gr_number)):
        # Determine subtype based on title patterns
        if 'PETITIONER' in title_upper and 'RESPONDENT' in title_upper:
            if 'CERTIORARI' in title_upper:
                subtype = "certiorari"
            elif 'MANDAMUS' in title_upper:
                subtype = "mandamus"
            elif 'PROHIBITION' in title_upper:
                subtype = "prohibition"
            elif 'HABEAS CORPUS' in title_upper:
                subtype = "habeas_corpus"
            elif 'QUO WARRANTO' in title_upper:
                subtype = "quo_warranto"
            elif 'APPEAL' in title_upper:
                subtype = "appeal"
            else:
                subtype = "petition"
        elif 'PEOPLE OF THE PHILIPPINES' in title_upper:
            subtype = "criminal"
        else:
            subtype = "civil"
            
        return {
            "case_type": "regular",
            "case_subtype": subtype,
            "is_administrative": False,
            "is_regular_case": True
        }
    
    # Consolidated cases (GR ranges)
    if gr_number and '-' in str(gr_number) and any(char.isdigit() for char in str(gr_number)):
        return {
            "case_type": "regular",
            "case_subtype": "consolidated",
            "is_administrative": False,
            "is_regular_case": True
        }
    
    # Other cases
    if 'MOTION' in title_upper:
        subtype = "motion"
    elif 'RESOLUTION' in title_upper:
        subtype = "resolution"
    elif 'ORDER' in title_upper:
        subtype = "order"
    elif 'RE:' in title_upper:
        subtype = "administrative_matter"
    else:
        subtype = "other"
    
    return {
        "case_type": "other",
        "case_subtype": subtype,
        "is_administrative": subtype == "administrative_matter",
        "is_regular_case": False
    }

def parse_case(text_base: str, url: str, year_hint: int | None = None, month_hint: str | None = None,
               title_guess: str | None = None, page_title: str | None = None):
    text = normalize_text(text_base)

    # Case title (prefer derived caption over generic page guess)
    derived_title = derive_case_title_from_text(text)
    case_title = derived_title or (title_guess if title_guess and CAPTION_RE.search(title_guess) else title_guess)

    # GR numbers (primary + list)
    gr_primary, gr_all = parse_gr_numbers_from_text(text)

    # Date (prioritize header dates over body dates)
    date_iso = extract_promulgation_date(text)

    secs = split_sections(text)

    # Division / En Banc and Ponente
    division, en_banc = extract_division_enbanc(text)
    ponente = extract_ponente(text)
    
    # Classify case type
    classification = classify_case_type(case_title or title_guess or "", gr_primary)

    record = {
        "id": sha256(url),
        "gr_number": gr_primary,
        "gr_numbers": gr_all or None,
        "title": case_title or title_guess,
        "case_title": case_title,
        "page_title": page_title,              # debug/optional
        "promulgation_date": date_iso,         # may be None
        "promulgation_year": year_hint,        # Fix A: carry year hint from listing
        "promulgation_month": month_hint,      # optional
        "court": "Supreme Court",
        "source_url": url,
        "clean_version": "v1.0",
        "checksum": sha256(text),
        "crawl_ts": datetime.utcnow().isoformat() + "Z",
        "ponente": ponente,
        "division": division,
        "en_banc": en_banc,
        "sections": secs,
        "clean_text": text,
        # Case classification
        "case_type": classification["case_type"],
        "case_subtype": classification["case_subtype"],
        "is_administrative": classification["is_administrative"],
        "is_regular_case": classification["is_regular_case"],
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
