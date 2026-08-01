"""Resumable download of AgenticDataBench datasets for the full run.

Skips loan_risk (16 GB, 25 tasks that carry no data_sources) and marketing
(3.3 GB, 9 tasks likewise). Everything else is fetched, including financial,
whose tasks name their files inline rather than in a data_sources field.
"""
import json, sys, time, urllib.parse, urllib.request
from pathlib import Path

TESTBED = Path(sys.argv[1])
BASE = "https://huggingface.co/datasets/shawnzzzh/AgenticDataBench/resolve/main/"
API = "https://huggingface.co/api/datasets/shawnzzzh/AgenticDataBench/tree/main/"

DOMAINS = ["agriculture", "ecommerce", "energy", "entertainment", "financial",
           "healthcare", "loan_model", "real_estate", "social_network",
           "sports", "strategy", "tourism", "transportation"]

def listing(path):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(API + urllib.parse.quote(path)
                                        + "?recursive=true", timeout=120) as r:
                return json.load(r)
        except Exception as exc:
            print(f"  listing retry {attempt+1} for {path}: {exc}", flush=True)
            time.sleep(3 * (attempt + 1))
    return []

total_bytes = 0
for dom in DOMAINS:
    items = listing(f"datasets/{dom}")
    files = [i for i in items if i["type"] == "file"]
    print(f"[{dom}] {len(files)} files", flush=True)
    for it in files:
        rel = it["path"]                      # datasets/<dom>/...
        dest = TESTBED / rel
        if dest.exists() and dest.stat().st_size == it.get("size", -1):
            total_bytes += dest.stat().st_size
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = BASE + urllib.parse.quote(rel)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=1800) as r, \
                     open(dest, "wb") as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                total_bytes += dest.stat().st_size
                print(f"  ok {rel} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
                break
            except Exception as exc:
                print(f"  retry {attempt+1} {rel}: {exc}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            print(f"  FAILED {rel}", flush=True)
print(f"DONE total {total_bytes/1e9:.2f} GB", flush=True)
