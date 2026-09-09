"""Quantify two hypotheses about why the learned arm loses on this dataset.

H1  Scale substitution: the wrong answer is the reference off by exactly 100x,
    i.e. a proportion returned where a percentage was asked for.

H2  Package short-circuit: the run cites a precomputed/host-provided aggregate
    from the knowledge package in its reasoning.

Both are read off the recorded trajectories; nothing is re-run.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

CITE = re.compile(
    r"precomputed|knowledge_result|host-provided|host provided|"
    r"provided in knowledge|from the knowledge package|already computed",
    re.I,
)

data = json.load(open("dbo_eval_results.json"))
gt = {r["id"]: r for r in json.load(open("ground_truth.json"))}


def num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.fullmatch(r"\s*([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)\s*", v)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return None
    return None


stats = defaultdict(lambda: defaultdict(int))
scale_rows, cite_rows = [], []

for t in data["trials"]:
    arm = t["arm"]
    if not t.get("ok"):
        continue
    g = t.get("grade", {})
    ref = gt[t["question_id"]]["reference"]
    stats[arm]["n"] += 1
    if g.get("analytic_correct"):
        stats[arm]["correct"] += 1
        continue
    stats[arm]["wrong"] += 1

    v = num(g.get("value"))
    if v is not None and isinstance(ref, (int, float)):
        for factor, label in ((100.0, "ref/100"), (0.01, "ref*100")):
            if ref != 0 and abs(v - ref / factor) <= abs(ref / factor) * 0.005:
                stats[arm]["scale_error"] += 1
                scale_rows.append((t["question_id"], arm, t["rep"], v, ref, label))
                break

    ans = t.get("answer") or {}
    blob = json.dumps(ans, default=str)
    if CITE.search(blob):
        stats[arm]["cites_package"] += 1
        cite_rows.append((t["question_id"], arm, t["rep"],
                          str(ans.get("reasoning", ""))[:150]))

print("=" * 92)
print("WRONG-ANSWER ANATOMY")
print("=" * 92)
print(f'{"":<26}{"arm A":>18}{"arm B":>18}')
for k in ("n", "correct", "wrong", "scale_error", "cites_package"):
    print(f'{k:<26}{stats["A"][k]:>18}{stats["B"][k]:>18}')

print()
print(f'Scale errors (proportion returned where percentage asked): '
      f'A={stats["A"]["scale_error"]}  B={stats["B"]["scale_error"]}')
for q, arm, rep, v, ref, lbl in sorted(scale_rows):
    print(f'   {q} arm {arm} rep{rep}: got {v} vs ref {ref}  ({lbl})')

print()
print(f'Runs citing a package-supplied aggregate: '
      f'A={stats["A"]["cites_package"]}  B={stats["B"]["cites_package"]}')
for q, arm, rep, why in sorted(cite_rows):
    print(f'   {q} arm {arm} rep{rep}: {why}')

# How often does citing the package coincide with being wrong?
cited_wrong = defaultdict(int)
cited_total = defaultdict(int)
for t in data["trials"]:
    if not t.get("ok"):
        continue
    blob = json.dumps(t.get("answer") or {}, default=str)
    if CITE.search(blob):
        cited_total[t["arm"]] += 1
        if not t.get("grade", {}).get("analytic_correct"):
            cited_wrong[t["arm"]] += 1
print()
for arm in ("A", "B"):
    tot = cited_total[arm]
    if tot:
        print(f'  arm {arm}: {cited_wrong[arm]}/{tot} runs that cited the package '
              f'were wrong ({100*cited_wrong[arm]/tot:.0f}%)')
