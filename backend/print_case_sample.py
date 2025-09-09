#!/usr/bin/env python3
"""
Print a single case from 2005 or 2006 from backend/data/cases.jsonl.gz
"""
import gzip
import json
import os
import sys
from typing import Any, Dict, Optional


def parse_year(rec: Dict[str, Any]) -> Optional[int]:
    y = rec.get("promulgation_year")
    if isinstance(y, str) and y.isdigit():
        try:
            return int(y)
        except Exception:
            pass
    if isinstance(y, int):
        return y
    y = rec.get("year")
    if isinstance(y, str) and y.isdigit():
        try:
            return int(y)
        except Exception:
            return None
    if isinstance(y, int):
        return y
    return None


def main() -> int:
    # Resolve dataset path relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "data", "cases.jsonl.gz")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    continue
                y = parse_year(rec)
                if isinstance(y, int) and y in (2005, 2006):
                    out = {
                        "id": rec.get("id"),
                        "gr_number": rec.get("gr_number"),
                        "title": rec.get("title") or rec.get("case_title") or rec.get("short_title"),
                        "promulgation_date": rec.get("promulgation_date"),
                        "promulgation_year": y,
                        "source_url": rec.get("source_url"),
                    }
                    print(json.dumps(out, ensure_ascii=False))
                    return 0
        print("NO_MATCH")
        return 0
    except FileNotFoundError:
        print("MISSING")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


