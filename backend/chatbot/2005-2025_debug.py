#!/usr/bin/env python3
# clean_split_scraper.py — eLibrary scraper with section-splitting + aggressive cleaning

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString
from clean_cases import clean_case_text, save_case_sections

BASE_URL = "https://elibrary.judiciary.gov.ph/"
DEFAULT_OUTPUT_DIR = "2005-2025_cleaned_structured"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PHLawBot/1.0)"}

# =========================
# Cleaning patterns
# =========================
RE_DUP_HEADERS = re.compile(
    r'^(View printer friendly version|[0-9]{3,4}\s+Phil\.\s+\d+|FIRST DIVISION|SECOND DIVISION|THIRD DIVISION|EN BANC)$',
    re.IGNORECASE | re.MULTILINE
)
RE_CASE_CAPTION_LINE = re.compile(r'^[A-Z0-9 ,\.\-/&\(\)]+VS\.[A-Z0-9 ,\.\-/&\(\)]+$', re.MULTILINE)
RE_URL = re.compile(r'https?://\S+')
RE_ROLLO = re.compile(r'\bRollo\b.*?(pp?\.|pages?).*?(?=\n|$)', re.IGNORECASE)
RE_ID_IBID = re.compile(r'^\s*(Id\.?|Ibid\.?|supra).*$', re.IGNORECASE | re.MULTILINE)
RE_FOOTNOTE_BRACKETS = re.compile(r'\[\d+\]|\(fn\.?\s*\d+\)', re.IGNORECASE)
RE_PARENTHESES_CITES = re.compile(r'\((G\.R\.\s*No\.?.*?|Phil\.\s*\d+.*?)\)', re.IGNORECASE)
RE_MULTI_BLANKS = re.compile(r'\n{3,}')
RE_MULTISPACE = re.compile(r'[ \t]{2,}')

# Disposition detection (case-insensitive), covers common openers and closers
RE_DISPO = re.compile(
    r'\b(WHEREFORE|ACCORDINGLY|IN VIEW OF THE FOREGOING|FOR THESE REASONS)\b.*?\bSO ORDERED\.?',
    re.IGNORECASE | re.DOTALL
)

# Metadata helpers
RE_GRNO = re.compile(r'G\.\s*R\.\s*No\.?\s*[\.:]?\s*([0-9A-Z\-]+)', re.IGNORECASE)
RE_DATE_IN_BRACKETS = re.compile(r'\[\s*G\.\s*R\.\s*No\..*?,\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})\s*\]')
RE_DATE_FALLBACK = re.compile(r'([A-Za-z]+\s+\d{1,2},\s+\d{4})')
RE_DIVISION = re.compile(r'^\s*(FIRST|SECOND|THIRD)\s+DIVISION|EN\s+BANC\s*$', re.IGNORECASE | re.MULTILINE)

def clean_decision_text(text: str) -> str:
    """Remove boilerplate, duplicate headers, citations, footnotes, urls, etc."""
    if not text:
        return ""
    text = RE_DUP_HEADERS.sub('', text)
    text = RE_CASE_CAPTION_LINE.sub('', text)
    text = RE_URL.sub('', text)
    text = RE_ROLLO.sub('', text)
    text = RE_ID_IBID.sub('', text)
    text = RE_FOOTNOTE_BRACKETS.sub('', text)
    text = RE_PARENTHESES_CITES.sub('', text)
    text = RE_MULTISPACE.sub(' ', text)
    text = RE_MULTI_BLANKS.sub('\n\n', text)
    return text.strip()

# =========================
# IO helpers
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def file_exists(path: str) -> bool:
    return os.path.exists(path)

def save_text(text: str, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# =========================
# HTTP
# =========================
def make_session(max_retries: int = 3, backoff: float = 0.6) -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    orig = s.request

    def request_with_retry(method, url, **kwargs):
        for attempt in range(max_retries):
            try:
                resp = orig(method, url, timeout=kwargs.pop("timeout", 20), **kwargs)
                if resp.status_code == 200 and resp.content:
                    return resp
                resp.raise_for_status()
                return resp
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff * (2 ** attempt))
    s.request = request_with_retry
    return s

def fetch_page(session: requests.Session, url: str) -> BeautifulSoup | None:
    try:
        resp = session.get(url)
        parser = "lxml" if "lxml" in BeautifulSoup().builder.features else "html.parser"
        return BeautifulSoup(resp.content, parser)
    except Exception:
        return None

# =========================
# Scraping
# =========================
def extract_month_links(session: requests.Session, start_year: int, end_year: int) -> list[dict]:
    soup = fetch_page(session, BASE_URL)
    if not soup:
        return []
    container = soup.find("div", id="container_date") or soup
    all_links = []
    current_year = None

    for tag in container.find_all(["h2", "a"]):
        if tag.name == "h2":
            m = re.search(r"(19|20)\d{2}", tag.get_text(strip=True))
            current_year = int(m.group(0)) if m else None
        elif tag.name == "a" and current_year and (start_year <= current_year <= end_year):
            href = tag.get("href", "")
            if not href:
                continue
            abs_url = urljoin(BASE_URL, href)
            all_links.append({
                "year": current_year,
                "month_text": tag.get_text(strip=True).lower(),
                "url": abs_url
            })
    return all_links

def extract_decision_links(session: requests.Session, month_url: str) -> list[str]:
    soup = fetch_page(session, month_url)
    if not soup:
        return []
    container = soup.find("div", id="container_title") or soup
    links = []
    for a in container.find_all("a", href=True):
        href = a["href"]
        if "showdocs" in href:
            links.append(urljoin(BASE_URL, href))
    return links

def remove_formatting_tags(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(["em", "i", "b", "u"]):
        tag.unwrap()

def extract_full_text_block(div: BeautifulSoup) -> str:
    """Get a reasonably linearized text from the main content div."""
    # Many decisions wrap content between a <center> and a final <hr>
    all_hrs = div.find_all("hr", attrs={"align": "LEFT", "width": "60%", "size": "1"})
    end_tag = all_hrs[-1] if all_hrs else None
    start_tag = div.find("center")

    parts = []
    added_center = (start_tag is None)  # if no center, just read whole div
    for elem in div.contents:
        if start_tag and elem == start_tag:
            added_center = True
            parts.append(start_tag.get_text(separator="\n", strip=True))
            continue
        if end_tag and elem == end_tag:
            break
        if not added_center:
            continue
        if isinstance(elem, NavigableString):
            txt = str(elem).strip()
            if txt:
                parts.append(txt)
        elif getattr(elem, "name", None):
            if elem.name in {"script", "style"}:
                continue
            txt = elem.get_text(separator="\n", strip=True)
            if txt:
                parts.append(txt)
    text = "\n".join([p for p in parts if p]).strip()
    return text if text else div.get_text(separator="\n", strip=True)

def extract_metadata(full_text: str, source_url: str) -> dict:
    md = {"gr_number": "", "date": "", "division": "", "source_url": source_url}

    # GR No.
    m = RE_GRNO.search(full_text)
    if m:
        md["gr_number"] = m.group(1).strip()

    # Date (prefer bracketed header line)
    m = RE_DATE_IN_BRACKETS.search(full_text)
    if m:
        md["date"] = m.group(1).strip()
    else:
        # Fallback: take the first plausible Month DD, YYYY near the top
        top_slice = full_text[:2000]
        m2 = RE_DATE_FALLBACK.search(top_slice)
        if m2:
            md["date"] = m2.group(1).strip()

    # Division
    m = RE_DIVISION.search(full_text)
    if m:
        div_text = m.group(0).strip().title()
        # Normalize "En Banc"
        if "En" in div_text and "Banc" in div_text:
            md["division"] = "En Banc"
        else:
            md["division"] = div_text
    return md

def split_sections(full_text: str) -> dict:
    """
    Heuristically split into header, body, disposition.
    - HEADER: caption + ponente block from the top down to the first major break
    - DISPOSITION: WHEREFORE/ACCORDINGLY... SO ORDERED.
    - BODY: remainder after removing header + disposition
    """
    text = full_text.strip()

    # Disposition
    dispo_match = RE_DISPO.search(text)
    disposition = ""
    dispo_span = (None, None)
    if dispo_match:
        disposition = dispo_match.group(0).strip()
        dispo_span = dispo_match.span()

    # Header heuristic:
    # Try to capture from start through "D E C I S I O N" (or RESOLUTION) and the ponente line "J.:"
    header = ""
    header_end = 0
    header_regexes = [
        re.compile(r'^(.*?\b(D\s*E\s*C\s*I\s*S\s*I\s*O\s*N|RESOLUTION)\b.*?(?:J\.:|,\s*J\.|,?\s*J\.)[^\n]*\n)', re.IGNORECASE | re.DOTALL),
        re.compile(r'^(.*?\[?\s*G\.\s*R\.\s*No\..*?\]\s*\n)', re.IGNORECASE | re.DOTALL),
    ]
    for rx in header_regexes:
        hm = rx.search(text)
        if hm:
            header = hm.group(0).strip()
            header_end = hm.end()
            break
    if not header:
        # fallback: take first ~1200 chars up to a double newline boundary
        cut = min(len(text), 1200)
        chunk = text[:cut]
        # extend to the next double newline if nearby
        nn = text.find("\n\n", cut)
        if nn != -1 and nn - cut < 400:
            cut = nn + 2
        header = text[:cut].strip()
        header_end = cut

    # Body: remove header + disposition spans
    body_text = text
    # Remove disposition first (so header_end index remains valid relative to original text)
    if dispo_span[0] is not None:
        body_text = body_text[:dispo_span[0]] + body_text[dispo_span[1]:]
    # Remove header region
    body_text = body_text[header_end:].strip()

    return {
        "header": header,
        "body": body_text,
        "disposition": disposition
    }

def extract_case_sections(session: requests.Session, gr_url: str) -> dict | None:
    """Fetch a case page, clean HTML, extract full text, then split + clean sections."""
    soup = fetch_page(session, gr_url)
    if not soup:
        return None

    div = soup.find("div", class_="single_content")
    if not div:
        return None

    # Strip in-document footnote superscripts and styling noise
    for sup in div.find_all("sup"):
        sup.decompose()
    remove_formatting_tags(div)

    full_text_raw = extract_full_text_block(div)

    # Clean once lightly to make metadata/regexes more reliable (but keep content)
    # Only remove obvious junk here; deep cleaning happens per-section below
    prelim = RE_URL.sub('', full_text_raw)
    prelim = RE_DUP_HEADERS.sub('', prelim)

    metadata = extract_metadata(prelim, gr_url)
    sections = split_sections(prelim)

    # Final cleaning per section
    header_clean = clean_decision_text(sections["header"])
    body_clean = clean_decision_text(sections["body"])
    dispo_clean = clean_decision_text(sections["disposition"])

    return {
        "metadata": metadata,
        "header": header_clean,
        "body": body_clean,
        "disposition": dispo_clean
    }

# =========================
# Orchestration
# =========================
def process_month(session: requests.Session, month: dict, output_dir: str, rate_delay: float, max_workers: int = 3):
    year = month["year"]
    month_label = month["month_text"][:3].title() + str(year)
    month_url = month["url"]

    print(f"\n📂 Processing {month_label} -> {month_url}")
    decision_links = extract_decision_links(session, month_url)
    print(f"🔗 Found {len(decision_links)} case links")

    def handle(idx_url):
        idx, gr_url = idx_url
        target_dir = os.path.join(output_dir, str(year))
        out_path = os.path.join(target_dir, f"{month_label}_{idx:03d}.txt")
        if file_exists(out_path):
            return f"⏩ Skip existing: {out_path}"

        data = extract_case_sections(session, gr_url)
        time.sleep(rate_delay)

        if not data:
            return f"⚠️ Failed to extract: {gr_url}"

        # Assemble structured text
        lines = []
        lines.append("=== METADATA ===")
        md = data["metadata"]
        # Always output metadata keys even if blank (keeps format predictable)
        lines.append(f"gr_number: {md.get('gr_number','')}")
        lines.append(f"date: {md.get('date','')}")
        lines.append(f"division: {md.get('division','')}")
        lines.append(f"source_url: {md.get('source_url','')}")

        lines.append("\n=== HEADER ===")
        lines.append(data["header"])

        lines.append("\n=== BODY ===")
        lines.append(data["body"])

        lines.append("\n=== DISPOSITION ===")
        lines.append(data["disposition"])

        save_text("\n".join(lines).strip() + "\n", out_path)
        return f"✅ Saved: {out_path}"

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(handle, (i, u)) for i, u in enumerate(decision_links, start=1)]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(res)
    return results

def main():
    parser = argparse.ArgumentParser(description="Scrape PH e-Library decisions into cleaned, sectioned .txt files.")
    parser.add_argument("--start-year", type=int, default=2025)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-delay", type=float, default=0.6)
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()

    session = make_session()
    print("🔍 Fetching month links...")
    months = extract_month_links(session, args.start_year, args.end_year)
    print(f"🗓️ Found {len(months)} months to process.")

    for month in months:
        process_month(session, month, args.output_dir, args.rate_delay, args.max_workers)

    print("\n🎉 DONE: All decisions scraped and saved (clean & sectioned).")

if __name__ == "__main__":
    main()
