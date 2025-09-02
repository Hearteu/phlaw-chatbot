import gzip
import json
import os

DATA_PATH = os.path.join("backend", "data", "cases_enhanced.jsonl.gz")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"MISSING {DATA_PATH}")
        return

    total = 0
    by_year = {2010: 0, 2011: 0, 2012: 0}
    samples = {2010: [], 2011: [], 2012: []}

    with gzip.open(DATA_PATH, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            y = obj.get("promulgation_year")
            if not isinstance(y, int):
                # try from date string
                d = obj.get("promulgation_date")
                if isinstance(d, str) and len(d) >= 4 and d[:4].isdigit():
                    y = int(d[:4])
            if y not in (2010, 2011, 2012):
                continue

            total += 1
            by_year[y] += 1

            if len(samples[y]) < 5:
                secs = obj.get('sections') or {}
                samples[y].append({
                    'title': obj.get('title'),
                    'case_title': obj.get('case_title'),
                    'gr_number': obj.get('gr_number'),
                    'source_url': obj.get('source_url'),
                    'ponente': obj.get('ponente'),
                    'division': obj.get('division'),
                    'header_len': len((secs.get('header') or '')),
                    'facts_len': len((secs.get('facts') or '')),
                    'issues_len': len((secs.get('issues') or '')),
                    'ruling_len': len((secs.get('ruling') or '')),
                    'body_len': len((secs.get('body') or obj.get('clean_text') or '')),
                })

    print("SUMMARY 2010-2012")
    print(by_year)
    for y in (2010, 2011, 2012):
        print(f"\nYEAR {y} - {by_year[y]} records; showing up to 5 samples:")
        for i, s in enumerate(samples[y], 1):
            print(f" {i}. title={s['title']!r} case_title={s['case_title']!r} gr={s['gr_number']!r}")
            print(f"    sections: header={s['header_len']} facts={s['facts_len']} issues={s['issues_len']} ruling={s['ruling_len']} body={s['body_len']}")

if __name__ == "__main__":
    main()


