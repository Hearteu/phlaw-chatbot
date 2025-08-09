# cleaner.py — normalize, de-duplicate, and sectionize PH SC decisions

import re
from typing import Dict, Tuple

# --- compiled regexes (fast) ---

# junk banners / repeated headings / boilerplate
RX_JUNK_LINES = re.compile(
    r"""
    ^\s*(?:View\ printer\ friendly\ version)\s*$|
    ^\s*\d{3,4}\s*Phil\.\s*\d+\s*$|                  # "922 Phil. 797"
    ^\s*(?:FIRST|SECOND|THIRD|EN\ BANC|SPECIAL)\ DIVISION\s*$|
    ^\s*\[\s*G\.R\.\ No\.\s*[\w\.\-]+\s*,?\s*[A-Za-z]+\s+\d{1,2},\s*\d{4}\s*\]\s*$|  # [ G.R. No. 227655, April 27, 2022 ]
    ^\s*D\s*E\s*C\s*I\s*S\s*I\s*O\s*N\s*$|           # D E C I S I O N (spaced)
    ^\s*R\s*E\s*S\s*O\s*L\s*U\s*T\s*I\s*O\s*N\s*$|   # R E S O L U T I O N
    ^\s*[A-Z][\w\.\- ]+,\ J\.\s*:$|                  # "LOPEZ, M., J.:"
    ^\s*J\.\s*:$|                                    # "J.:"
    ^\s*Facts\s*$|^\\s*The\ Facts\s*$|
    ^\s*Antecedents\s*$|
    ^\s*Ruling\s*$|
    ^\s*Issues?\s*$|
    ^\s*SO ORDERED\.?\s*$                            # trailing lone "SO ORDERED."
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE
)

# footnote / citation noise (inline + blocks)
RX_INLINE_CITES = re.compile(
    r"""
    \s*\(\s*Citation\s+omitted\s*\)\s*|
    \s*\(\s*Emphases?\s+(?:in\ the\ original|supplied)\s*\)\s*|
    \s*\[\s*Per\ .*?\]\s*|
    \s*<https?://[^>\s]+>\s*|
    \s*\bid\s*\.\s*|
    \s*\(G\.R\.\ No\.[^)]+\)\s*|
    \bRollo\b.*?(?:\)|\.)\s*|
    \[\s*\d+\s*\]                                    # bracketed footnote numbers
    """,
    re.IGNORECASE | re.VERBOSE
)

# footnote trails / bibliographic lists at the end (start lines like "Rollo (vol...)", case lists)
RX_TRAIL_BLOCKS = re.compile(
    r"""
    (?:^\s*Rollo\s*\(?.*?$)|
    (?:^\s*\d{3,4}\s*Phil\..*$)|
    (?:^\s*G\.R\.\ No\..*$)|
    (?:^\s*[A-Z][\w\ \.\-]+\ v\.\ [A-Z].*$)
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE
)

# whitespace cleanup
RX_MULTISPACE = re.compile(r"[ \t]+")
RX_MULTI_NL = re.compile(r"\n{3,}")

# section sentinels
RX_METADATA_HDR = re.compile(r"^===\s*METADATA\s*===", re.IGNORECASE | re.MULTILINE)
RX_HEADER_HDR   = re.compile(r"^===\s*HEADER\s*===",   re.IGNORECASE | re.MULTILINE)
RX_BODY_HDR     = re.compile(r"^===\s*BODY\s*===",     re.IGNORECASE | re.MULTILINE)
RX_DISP_HDR     = re.compile(r"^===\s*DISPOSITION\s*===", re.IGNORECASE | re.MULTILINE)

# ruling / disposition capture (robust)
RX_RULING = re.compile(
    r"""
    (?P<prefix>\bWHEREFORE\b|\bACCORDINGLY\b|\bIN\s+LIGHT\s+OF\s+THE\s+FOREGOING\b)
    .*?
    \bSO\s+ORDERED\.?
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE
)

def _strip_junk_lines(text: str) -> str:
    # remove obvious banner/junk lines
    text = RX_JUNK_LINES.sub("", text)
    # drop trailing citation blocks that are each on their own lines
    text = RX_TRAIL_BLOCKS.sub("", text)
    # normalize weird spacing inside ALL-CAPS spaced words already removed by RX_JUNK_LINES, but keep generic clean-up
    return text

def _normalize_ws(text: str) -> str:
    # fix hyphenation artifacts like "in\ nto" not handled here; mostly normalize spacing/newlines
    text = RX_INLINE_CITES.sub("", text)
    text = RX_MULTISPACE.sub(" ", text)
    text = text.replace(" \n", "\n").replace("\n ", "\n")
    text = RX_MULTI_NL.sub("\n\n", text)
    return text.strip()

def _extract_sections(raw: str) -> Dict[str, str]:
    """Return dict: metadata, header, body, disposition, ruling, full_clean."""
    # locate main tagged blocks if present
    def _slice_between(start_rx, next_rxs) -> Tuple[int, int]:
        start = start_rx.search(raw)
        if not start:
            return (-1, -1)
        s = start.end()
        # next header among provided
        ends = [rx.search(raw, s) for rx in next_rxs]
        ends = [m.start() for m in ends if m]
        e = min(ends) if ends else len(raw)
        return (s, e)

    m_s, m_e = _slice_between(RX_METADATA_HDR, [RX_HEADER_HDR, RX_BODY_HDR, RX_DISP_HDR])
    h_s, h_e = _slice_between(RX_HEADER_HDR,   [RX_BODY_HDR, RX_DISP_HDR])
    b_s, b_e = _slice_between(RX_BODY_HDR,     [RX_DISP_HDR])
    d_s, d_e = _slice_between(RX_DISP_HDR,     [])

    meta = raw[m_s:m_e] if m_s != -1 else ""
    head = raw[h_s:h_e] if h_s != -1 else ""
    body = raw[b_s:b_e] if b_s != -1 else ""
    disp = raw[d_s:d_e] if d_s != -1 else ""

    # If no tagged sections, treat entire text as body candidate
    if not any([meta, head, body, disp]):
        body = raw

    # Clean each part
    def clean(part: str) -> str:
        return _normalize_ws(_strip_junk_lines(part))

    meta_c = clean(meta)
    head_c = clean(head)
    body_c = clean(body)
    disp_c = clean(disp)

    # Pull ruling (prefer from disposition; else from body)
    search_space = disp_c if disp_c else body_c
    ruling_match = RX_RULING.search(search_space)
    ruling_c = _normalize_ws(ruling_match.group(0)) if ruling_match else ""

    # de-dup: if header repeats top of body, drop it
    if head_c and body_c.startswith(head_c[: min(200, len(head_c))]):
        head_c = ""

    # Make an overall cleaned text (for debugging/export if you ever need it)
    pieces = []
    if meta_c:
        pieces.append("=== METADATA ===\n" + meta_c.strip())
    if head_c:
        pieces.append("=== HEADER ===\n" + head_c.strip())
    if body_c:
        pieces.append("=== BODY ===\n" + body_c.strip())
    if disp_c:
        pieces.append("=== DISPOSITION ===\n" + disp_c.strip())
    full_clean = "\n\n".join(pieces).strip()

    return {
        "metadata": meta_c,
        "header": head_c,
        "body": body_c,
        "disposition": disp_c,
        "ruling": ruling_c,
        "full_clean": full_clean,
    }

def clean_and_split(raw: str) -> Dict[str, str]:
    """
    Public API: take raw file text with/without section sentinels,
    drop duplicate headings/citations/footnotes, and return clean parts.
    """
    return _extract_sections(raw)
