"""Reference-defect audit.

q23 announced itself with a specific signature: every trial in BOTH arms
converged on the same value, and that value was not my reference. When the
library independently and repeatedly disagrees with me in exactly the same
way, the cheap explanation is that my reference is wrong -- not that the
library failed six times identically.

Dual-engine agreement (pandas + DuckDB) does NOT protect against this,
because I authored both encodings of the same misreading.

This flags every question with that signature so the reference can be
re-derived by hand before any of them are scored as library failures.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

data = json.load(open("dbo_eval_results.json"))
gt = {r["id"]: r for r in json.load(open("ground_truth.json"))}


def num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"[-+]?\d[\d,]*\.?\d*", v)
        if m:
            try:
                return float(m.group(0).replace(",", ""))
            except ValueError:
                return None
    return None


per_q = defaultdict(list)
for t in data["trials"]:
    if t.get("ok"):
        per_q[t["question_id"]].append(t)

print("=" * 100)
print("REFERENCE-DEFECT AUDIT -- questions where the library converges on a value I did not expect")
print("=" * 100)

suspects = []
for qid in sorted(per_q):
    trials = per_q[qid]
    n_correct = sum(1 for t in trials if t.get("grade", {}).get("analytic_correct"))
    if n_correct:
        continue  # at least one arm reproduced the reference -> reference is reachable

    ref = gt[qid]["reference"]
    vals = [num(t.get("grade", {}).get("value")) for t in trials]
    vals = [v for v in vals if v is not None]
    if not vals:
        print(f"\n{qid}: 0/{len(trials)} correct, but no parseable values "
              f"-> formatting failure, not a reference defect")
        continue

    # cluster values within 0.5%
    clusters = Counter()
    for v in vals:
        placed = False
        for k in list(clusters):
            if k != 0 and abs(v - k) <= abs(k) * 0.005:
                clusters[k] += 1
                placed = True
                break
        if not placed:
            clusters[v] += 1
    top_val, top_n = clusters.most_common(1)[0]
    frac = top_n / len(vals)

    verdict = ("STRONG reference-defect suspect" if frac >= 0.8 and len(vals) >= 4
               else "weak -- library is inconsistent, likely a genuine miss")
    print(f"\n{qid}: 0/{len(trials)} correct across BOTH arms")
    print(f"   question   : {gt[qid]['question'][:110]}")
    print(f"   my ref     : {ref}")
    print(f"   library    : {top_val}  ({top_n}/{len(vals)} trials agree, {frac:.0%})")
    if ref not in (0, None) and isinstance(ref, (int, float)):
        print(f"   ratio      : library/ref = {top_val/ref:.6f}")
    print(f"   verdict    : {verdict}")
    if frac >= 0.8 and len(vals) >= 4:
        suspects.append(qid)

print()
print("=" * 100)
print(f"STRONG suspects requiring manual re-derivation: {suspects or 'none'}")
print("Questions with >=1 correct trial are NOT suspects -- the reference is demonstrably reachable.")
print("=" * 100)
