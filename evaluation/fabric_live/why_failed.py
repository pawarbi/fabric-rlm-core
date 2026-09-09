"""Why did the Fabric runs produce so many unusable answers?

Phase 1 (local) had 1.4% unusable; both Fabric arms have ~40%. The workbook
duty is ruled out by the control, so the cause is environmental. This prints
the model's own words for every failure so the mechanism is visible rather
than inferred.
"""
import json
import sys
from collections import Counter
from pathlib import Path

log = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

blocked = Counter()
print(f"=== {len(log)} records ===\n")
for r in log:
    a = r.get("answer") or {}
    v = a.get("value")
    if v is not None and not isinstance(v, str):
        continue
    print(f"--- {r['id']}  turns={r.get('turns')}  "
          f"failure_reason={r.get('failure_reason')}")
    print(f"    value    : {str(v)[:160]}")
    print(f"    reasoning: {str(a.get('reasoning'))[:260]}")
    blob = f"{v} {a.get('reasoning')} {a.get('sql')}".lower()
    for kw in ("restrict", "permission", "denied", "not allowed", "read-only",
               "unable", "no access", "timeout", "error", "empty", "failed"):
        if kw in blob:
            blocked[kw] += 1
    print()

print("=== keyword frequency across unusable answers ===")
for k, c in blocked.most_common():
    print(f"  {k:14} {c}")
