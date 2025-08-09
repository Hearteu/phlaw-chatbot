#!/usr/bin/env python3
# clean_cases.py — Supreme Court e-Library text cleaner & section splitter
# Usage:
#   python clean_cases.py --in-dir raw_txt --out-dir cleaned_cases
#
# You can also import clean_case_text(text) in your scraper and skip the CLI.

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

# -----------------------------
# Patterns (compiled once)
# -----------------------------
SPACED_WORD = re.compile(r"\b([A-Z])(?:\s+([A-Z]))+\b")  # e.g., D E C I S I O N
MULTISPACE = re.compile(r"[ \t]+")
MULTIBLANKS = re.compile(r"\n{3,}")

# Common garbage lines (exact-ish matches)
GARBAGE_LINE_PATTERNS = [
    r"^View printer friendly version$",
    r"^\d+\s*Phil\.\s*\d+\s*$",               # 922 Phil. 797
    r"^(FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+DIVISION$",
    r"^EN\s+BANC$",
    r"^\[?\s*G\.R\.\s*No\.\s*\d[\d\-]*,?\s*[A-Za-z]+\s+\d{1,2},\s*\d{4}\s*\]?$",  # [ G.R. No. 227655, April 27, 2022 ]
    r"^D\s*E\s*C\s*I\s*S\s*I\s*O\s*N$",      # D E C I S I O N (spaced)
    r"^R\s*E\s*S\s*O\s*L\s*U\s*T\s*I\s*O\s*N$",
    r"^\(*Per\s+Cur[ií]am\)*\.?$",
    r"^x\s*x\s*x\s*x\s*$",                    # x x x
]

GARBAGE_LINE_RX = re.compile("|".join(f"(?:{p})" for p in GARBAGE_LINE_PATTERNS), re.IGNORECASE)

# Judge/author line (e.g., "LOPEZ, M., J.:" / "GAERLAN, J.:")
JUDGE_LINE_RX = re.compile(r"^[A-ZÀ-Ÿ][A-ZÀ-Ÿ\.\-\' ]+,\s*(J\.|J\.'|C\.J\.|S\.A\.J\.)\s*:?\s*$")

# Case caption all-caps parties (rough heuristic)
CAPTION_LINE_RX = re.compile(r"^[A-Z0-9\.,&/()\"' \-\[\]]{15,}$")

# Inline junk
INLINE_SUPERSCRIPT_RX = re.compile(r"\[\s*\d+\s*\]")  # [12]
INLINE_URL_RX = re.compile(r"https?://\S+")
INLINE_CITE_CHUNKS_RX = re.compile(r"\s+\(\s*Citation(?:s)?\s+omitted\s*\)\.?$", re.IGNORECASE)

# Footnote/citation block starters (any of these → likely the “junk tail”)
FOOT_BLOCK_STARTERS = [
    r"^Rollo\b",
    r"^Id\.?\b",
    r"^Ibid\.?\b",
    r"^\(?see\b",
    r"^\(?See\b",
    r"^\d{3,}\s*Phil\.\s*\d+",  # 716 Phil. 267 (2013).
    r"^G\.R\.\s*No\.\s*\d",
    r"^\(?[A-Za-z].*v\.\s*[A-Za-z].*,?\s*G\.R\.\s*No\.",  # Case v. Case, G.R. No...
    r"^\<?https?://",  # raw URLs
]
FOOT_BLOCK_START_RX = re.compile("|".join(FOOT_BLOCK_STARTERS))

# Disposition delimiters
DISPO_START_RX = re.compile(r"\b(WHEREFORE|ACCORDINGLY)\b", re.IGNORECASE)
DISPO_END_RX = re.compile(r"\bSO\s+ORDERED\.?\b", re.IGNORECASE)

# Lightweight metadata keys in your sample files
META_LINE_RX = re.compile(r"^(gr_number|date|division|source_url)\s*:\s*(.+)$", re.IGNORECASE)

# -----------------------------
# Data model
# -----------------------------
@dataclass
class CleanedCase:
    gr_number: Optional[str] = None
    date: Optional[str] = None
    division: Optional[str] = None
    source_url: Optional[str] = None
    header: str = ""
    body: str = ""
    disposition: str = ""

# -----------------------------
# Utilities
# -----------------------------
def _normalize_spacing(text: str) -> str:
    # Fix weird spaced words like D E C I S I O N → DECISION
    def _unspace(m: re.Match) -> str:
        chunk = m.group(0)
        return chunk.replace(" ", "")
    text = SPACED_WORD.sub(_unspace, text)
    # Normalize multi-spaces and trim trailing spaces
    text = "\n".join(MULTISPACE.sub(" ", ln).rstrip() for ln in text.splitlines())
    # Collapse excessive blank lines
    text = MULTIBLANKS.sub("\n\n", text).strip()
    return text

def _strip_inline_junk(line: str) -> str:
    line = INLINE_SUPERSCRIPT_RX.sub("", line)
    line = INLINE_URL_RX.sub("", line)
    line = INLINE_CITE_CHUNKS_RX.sub("", line)
    return line.strip()

def _is_garbage_line(line: str) -> bool:
    if not line.strip():
        return False
    if GARBAGE_LINE_RX.search(line):
        return True
    # Kill short “Phil.” cite fragments too
    if re.search(r"\bPhil\.\b", line) and len(line) <= 30:
        return True
    return False

def _strip_footnote_block(lines: list) -> list:
    """
    Remove the trailing big block of citations/footnotes that usually starts with
    'Rollo', 'Id.', 'G.R. No.', URLs, etc.
    We scan from bottom up; once we hit a “starter”, we cut everything from there to the end,
    but only if that block is "citation heavy" (to avoid nuking actual reasoning).
    """
    n = len(lines)
    cut_at = None
    # Find the top of the last citation cluster of at least ~5 lines
    for i in range(n - 1, -1, -1):
        if FOOT_BLOCK_START_RX.search(lines[i].strip()):
            # ensure the tail has at least a few similar lines
            tail = lines[i:]
            starters = sum(1 for t in tail if FOOT_BLOCK_START_RX.search(t.strip()))
            if len(tail) >= 5 and starters >= 2:
                cut_at = i
            break
    if cut_at is not None:
        return lines[:cut_at]
    return lines

def _parse_meta_if_present(text: str) -> Tuple[Dict[str, str], str]:
    """
    If the file already has a === METADATA === block, parse it and return meta + remainder.
    Otherwise, return empty meta and original text.
    """
    meta = {}
    if "=== METADATA ===" in text:
        _, rest = text.split("=== METADATA ===", 1)
        # Stop at next === HEADER === or end
        parts = rest.split("=== HEADER ===", 1)
        meta_block = parts[0]
        for ln in meta_block.splitlines():
            m = META_LINE_RX.match(ln.strip())
            if m:
                key, val = m.group(1).lower(), m.group(2).strip()
                meta[key] = val
        remainder = ("=== HEADER ===" + parts[1]) if len(parts) == 2 else ""
        return meta, remainder or ""
    return meta, text

def _drop_repeated_header_from_body(header: str, body: str) -> str:
    """
    If the header lines repeat at the top of the body (very common), remove the overlapping prefix.
    """
    h_lines = [ln for ln in header.splitlines() if ln.strip()]
    b_lines = body.splitlines()
    if not h_lines or not b_lines:
        return body
    max_k = min(len(h_lines), 30)  # compare only the first 30 lines of header
    k = 0
    for i in range(max_k):
        if i < len(b_lines) and h_lines[i].strip() == b_lines[i].strip():
            k += 1
        else:
            break
    if k >= 3:
        return "\n".join(b_lines[k:]).lstrip("\n")
    return body

def _extract_disposition(text: str) -> Tuple[str, str]:
    """
    Returns (disposition_text, remainder_without_dispo).
    We search for WHEREFORE/ACCORDINGLY … SO ORDERED.
    """
    m_start = DISPO_START_RX.search(text)
    if not m_start:
        return "", text
    m_end = DISPO_END_RX.search(text, m_start.start())
    if not m_end:
        # Sometimes the dispositive ends without 'SO ORDERED' (rare). Take till end.
        dispo = text[m_start.start():].strip()
        body = text[:m_start.start()].rstrip()
        return dispo, body
    end_idx = m_end.end()
    dispo = text[m_start.start():end_idx].strip()
    body = (text[:m_start.start()] + text[end_idx:]).strip()
    return dispo, body

# -----------------------------
# Core cleaner
# -----------------------------
def clean_case_text(raw_text: str) -> CleanedCase:
    """
    Ingest raw extracted text (from your scraper or a saved .txt),
    remove junk, split into sections, and return a CleanedCase.
    """
    # Normalize spacing early so matchers work
    text = _normalize_spacing(raw_text)

    # If the file already has === METADATA ===, parse it
    meta, text = _parse_meta_if_present(text)

    # If the file also has explicit === HEADER === / === BODY === markers, use them
    header, body = "", ""
    if "=== HEADER ===" in text:
        # Expect: === HEADER === ... === BODY === ... (optional === DISPOSITION ===)
        after_header = text.split("=== HEADER ===", 1)[1]
        if "=== BODY ===" in after_header:
            header, after_body = after_header.split("=== BODY ===", 1)
            header = header.strip()
            if "=== DISPOSITION ===" in after_body:
                body, dispo = after_body.split("=== DISPOSITION ===", 1)
                body = body.strip()
                disposition_raw = dispo.strip()
            else:
                body = after_body.strip()
                disposition_raw = ""
        else:
            # Only header present
            header = after_header.strip()
            body = ""
            disposition_raw = ""
    else:
        # No explicit markers → try to heuristically separate header from the rest.
        # Strategy: collect up top all obvious caption/judge/division lines, then stop at first normal paragraph.
        lines = text.splitlines()
        head_lines = []
        rest_lines = []
        phase_header = True
        for ln in lines:
            s = ln.strip()
            if not s:
                # keep a single blank inside header; stop when we see normal prose after header
                if phase_header and head_lines and head_lines[-1] != "":
                    head_lines.append("")
                else:
                    rest_lines.append(ln)
                continue
            if phase_header and (GARBAGE_LINE_RX.search(s) or JUDGE_LINE_RX.search(s) or CAPTION_LINE_RX.match(s)):
                head_lines.append(s)
                continue
            if phase_header and len(s) <= 120 and s.isupper():
                head_lines.append(s)
                continue
            # First “normal” line flips us into body mode
            if phase_header:
                phase_header = False
                rest_lines.append(ln)
            else:
                rest_lines.append(ln)
        header = "\n".join([ln for ln in head_lines if ln is not None]).strip()
        body = "\n".join(rest_lines).strip()
        disposition_raw = ""

    # Clean header lines (remove obvious junk)
    h_clean = []
    for ln in header.splitlines():
        if _is_garbage_line(ln):
            continue
        ln = _strip_inline_junk(ln)
        if ln:
            h_clean.append(ln)
    header = "\n".join(h_clean).strip()

    # Get full working text for body cleanup
    working = body

    # If no BODY marker present and the header leaked into body, drop it
    working = _drop_repeated_header_from_body(header, working)

    # Line-by-line cleaning
    lines = []
    for ln in working.splitlines():
        s = ln.strip()
        if _is_garbage_line(s):
            continue
        s = _strip_inline_junk(s)
        if not s:
            lines.append("")  # keep structure, collapse later
            continue
        lines.append(s)

    # Remove trailing footnote/citation block (Rollo/Id./G.R./URLs)
    lines = _strip_footnote_block(lines)

    # Rebuild body text
    body_text = "\n".join(lines)
    body_text = MULTIBLANKS.sub("\n\n", body_text).strip()

    # Extract disposition from the cleaned body (or from disposition_raw if the markers had one)
    if disposition_raw:
        # Even explicit marker content might still have cites — quick clean
        d_lines = []
        for ln in disposition_raw.splitlines():
            s = _strip_inline_junk(ln.strip())
            if _is_garbage_line(s):
                continue
            d_lines.append(s)
        disposition = "\n".join(d_lines).strip()
        # Also remove dispo text from body if user accidentally put it under body too
        dispo_clean, _ = _extract_disposition(body_text)
        if dispo_clean:
            body_text = body_text.replace(dispo_clean, "").strip()
    else:
        disposition, body_text = _extract_disposition(body_text)

    # Final tidy
    header = _normalize_spacing(header)
    body_text = _normalize_spacing(body_text)
    disposition = _normalize_spacing(disposition)

    # Return sections + any meta we picked up
    return CleanedCase(
        gr_number=meta.get("gr_number"),
        date=meta.get("date"),
        division=meta.get("division"),
        source_url=meta.get("source_url"),
        header=header,
        body=body_text,
        disposition=disposition,
    )

# -----------------------------
# I/O helpers
# -----------------------------
def save_case_sections(cleaned: CleanedCase, out_dir: str, base: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # JSON (sections + light meta)
    with open(os.path.join(out_dir, f"{base}.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cleaned), f, ensure_ascii=False, indent=2)
    # Separate section files (handy for embedding/QA)
    def _write(name: str, txt: str):
        path = os.path.join(out_dir, f"{base}_{name}.txt")
        with open(path, "w", encoding="utf-8") as wf:
            wf.write(txt.strip() + ("\n" if txt and not txt.endswith("\n") else ""))
    _write("header", cleaned.header)
    _write("body", cleaned.body)
    _write("disposition", cleaned.disposition)

# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Clean e-Library case texts and split into sections.")
    p.add_argument("--in-dir", required=True, help="Directory of raw .txt case files")
    p.add_argument("--out-dir", required=True, help="Directory to write cleaned JSON + sections")
    p.add_argument("--suffix", default="", help="Optional suffix to append to basenames")
    args = p.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    suffix = args.suffix

    files = [f for f in os.listdir(in_dir) if f.lower().endswith(".txt")]
    files.sort()
    print(f"Found {len(files)} files in {in_dir}")
    for i, fn in enumerate(files, 1):
        src = os.path.join(in_dir, fn)
        try:
            with open(src, "r", encoding="utf-8") as rf:
                raw = rf.read()
            cleaned = clean_case_text(raw)
            base = os.path.splitext(fn)[0] + (f"_{suffix}" if suffix else "")
            save_case_sections(cleaned, out_dir, base)
            print(f"[{i:04d}/{len(files)}] ✅ {fn} → cleaned")
        except Exception as e:
            print(f"[{i:04d}/{len(files)}] ⚠️  {fn} → {e}")

if __name__ == "__main__":
    main()
