"""Audit the probe regexes in analyze_rediscovery.py by printing what they
actually matched. Written after a reviewer flagged that `\\blen\\s*\\(\\s*\\w+\\s*\\)`
matches ordinary Python (`len(cols)`) and not just row-count profiling.
"""
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
raw = (HERE / "run_log_glm_learn.json").read_text(encoding="utf-8")
rows = [r for r in json.loads(raw[raw.find("["):]) if r.get("trajectory")]


def unesc(s):
    try:
        return s.encode().decode("unicode_escape")
    except Exception:
        return s


def code_of(t):
    text = t.get("text", "")
    m = re.search(r"code=(['\"])(.*?)\1,\s*stdout=", text, re.S)
    return unesc(m.group(2)) if m else ""


SUSPECT = {
    "row count OLD": r"\.shape\b|\blen\s*\(\s*\w+\s*\)|COUNT\s*\(\s*\*",
    "  |- len(x) only": r"\blen\s*\(\s*\w+\s*\)",
    "  |- .shape only": r"\.shape\b",
    "  |_ COUNT(*) only": r"COUNT\s*\(\s*\*",
    "grain groupby.size": r"groupby\([^)]*\)\.(size|count)\s*\(",
    "join merge/JOIN": r"merge\(|\bJOIN\b",
    "minmax OLD (dot)": r"\.describe\s*\(|\.min\s*\(|\.max\s*\(",
    "minmax SQL forms": r"\b(MIN|MAX)\s*\(",
    "distinct SQL": r"COUNT\s*\(\s*DISTINCT",
}

print("=== sample of what each pattern matches (code only) ===\n")
for label, pat in SUSPECT.items():
    hits = []
    for r in rows:
        for t in r["trajectory"]:
            c = code_of(t)
            for m in re.finditer(pat, c):
                s = max(0, m.start() - 30)
                hits.append(re.sub(r"\s+", " ", c[s:m.end() + 30]))
    print(f"--- {label}: {len(hits)} matches")
    for h in hits[:6]:
        print("      …", h)
    print()

# The decisive question: how many len() calls are on a genuine table/dataframe?
print("=== len() argument frequency ===")
args = Counter()
for r in rows:
    for t in r["trajectory"]:
        for m in re.finditer(r"\blen\s*\(\s*(\w+)\s*\)", code_of(t)):
            args[m.group(1)] += 1
for k, v in args.most_common(25):
    print(f"   len({k}) : {v}")

