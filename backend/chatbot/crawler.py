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
# --- Windows event-loop policy MUST be set early (Playwright subprocess on Windows) ---
import os
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# -------------------------------------------------------------------------------------

import gzip
import hashlib
import re
import time
from datetime import datetime
from urllib.parse import urljoin

import aiohttp
import orjson
import requests
from bs4 import BeautifulSoup
# Crawl4AI (current docs API)
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from dateutil import parser as dtp
from selectolax.parser import HTMLParser

# -----------------------------
# Config (env-overridable)
# -----------------------------
BASE_URL     = os.getenv("ELIBRARY_BASE", "https://elibrary.judiciary.gov.ph/")
OUT_PATH     = os.getenv("CASES_JSONL", "backend/data/cases.jsonl.gz")
UA           = os.getenv("CRAWLER_UA", "Mozilla/5.0 (compatible; PHLawBot/1.0)")
YEAR_START   = int(os.getenv("YEAR_START", 2005))
YEAR_END     = int(os.getenv("YEAR_END", 2025))
CONCURRENCY  = int(os.getenv("CONCURRENCY", 8))
SLOWDOWN_MS  = int(os.getenv("SLOWDOWN_MS", 250))
TIMEOUT_S    = int(os.getenv("TIMEOUT_S", 45))
WRITE_CHUNK  = int(os.getenv("WRITE_CHUNK", 1000))  # tasks per gather batch

HEADERS = {"User-Agent": UA}

RULING_REGEX = re.compile(
    r"(WHEREFORE.*?SO ORDERED\.?|ACCORDINGLY.*?SO ORDERED\.?)",
    re.IGNORECASE | re.DOTALL,
)

# Broader date patterns (helps when markdown extractor misses labels)
DATE_PATTERNS = [
    r"Promulgated\s*(?:on)?\s*:\s*(\w+\s+\d{1,2},\s+\d{4})",
    r"Promulgated\s*(?:on)?\s*(\w+\s+\d{1,2},\s+\d{4})",
    r"\b(\w+\s+\d{1,2},\s+\d{4})\b",
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

    # 2) first <center>
    if root:
        cent = root.css_first("center")
        if cent:
            lines = [l.strip() for l in cent.text(separator="\n").splitlines() if l.strip()]
            for l in lines:
                if _is_title_like(l):
                    pt = (tree.css_first("title").text(strip=True) if tree.css_first("title") else None)
                    return l, pt
            candidates = [l for l in lines if not any(b in l.upper() for b in TITLE_BAD)]
            if candidates:
                t = max(candidates, key=len)
                pt = (tree.css_first("title").text(strip=True) if tree.css_first("title") else None)
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
def parse_case(text_base: str, url: str, year_hint: int | None = None, month_hint: str | None = None,
               title_guess: str | None = None, page_title: str | None = None):
    text = normalize_text(text_base)

    # GR number
    gr = None
    m = re.search(r"G\.\s*R\.\s*No\.\s*[0-9\-]+", text, flags=re.I)
    if m:
        gr = m.group(0).strip()

    # Date (try patterns)
    date_iso = None
    for pat in DATE_PATTERNS:
        md = re.search(pat, text, flags=re.I)
        if md:
            try:
                date_iso = dtp.parse(md.group(1)).date().isoformat()
                break
            except Exception:
                pass

    secs = split_sections(text)

    record = {
        "id": sha256(url),
        "gr_number": gr,
        "title": title_guess,                  # set from extractor (may be None)
        "page_title": page_title,              # debug/optional
        "promulgation_date": date_iso,         # may be None
        "promulgation_year": year_hint,        # Fix A: carry year hint from listing
        "promulgation_month": month_hint,      # optional
        "court": "Supreme Court",
        "source_url": url,
        "clean_version": "v1.0",
        "checksum": sha256(text),
        "crawl_ts": datetime.utcnow().isoformat() + "Z",
        "sections": secs,
        "clean_text": text,
    }
    return record

# -----------------------------
# Async fetching
# -----------------------------
async def fetch_playwright(crawler: AsyncWebCrawler, url: str) -> tuple[str, str]:
    run_cfg = CrawlerRunConfig(
        exclude_external_links=True,
        remove_overlay_elements=True,
        timeout=TIMEOUT_S,
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

            async def fetch_http(u: str) -> str:
                async with session.get(u, timeout=TIMEOUT_S) as resp:
                    resp.raise_for_status()
                    return await resp.text()

            async def worker(item: dict, idx: int, total: int):
                nonlocal written
                async with sem:
                    u = item["url"]
                    rid = sha256(u)
                    if rid in existing_ids:
                        return
                    try:
                        html = await fetch_http(u)
                        title_guess, page_title = extract_title(None, html)
                        text = HTMLParser(html).body.text(separator="\n") if html else ""
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
    print(f"🧹 Skipping {len(existing_ids)} already saved; crawling {len(items)} candidates")

    # Try Playwright first; if it can’t spawn (Windows subprocess issue), fall back gracefully.
    try:
        print("▶ Trying Playwright mode…")
        w = await crawl_all_with_playwright(items, out_path, existing_ids)
        print(f"✅ Playwright mode wrote {w} records")
    except NotImplementedError as e:
        print("⛔ Playwright cannot spawn subprocess. Falling back to HTTP-only.", e)
        w = await crawl_all_fallback_requests(items, out_path, existing_ids)
        print(f"✅ HTTP fallback wrote {w} records")
    except Exception as e:
        print("⛔ Playwright error. Falling back to HTTP-only.", e)
        w = await crawl_all_fallback_requests(items, out_path, existing_ids)
        print(f"✅ HTTP fallback wrote {w} records")

def main():
    items = discover_case_urls()
    asyncio.run(crawl_all(items, OUT_PATH))

if __name__ == "__main__":
    main()
