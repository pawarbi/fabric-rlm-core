"""Analyse the 25-question dbo run. Offline: no model calls.

Reports the two axes separately, per question and overall, and lists
regressions explicitly so an overall average cannot hide one.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963985
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d) * 100, min(1.0, (c + m) / d) * 100)


def main(path: str) -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    trials = data["trials"]

    arms = sorted({t["arm"] for t in trials})
    agg = {a: defaultdict(int) for a in arms}
    num = {a: defaultdict(list) for a in arms}
    per_q = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    per_q_n = defaultdict(lambda: defaultdict(int))

    for t in trials:
        a = t["arm"]
        q = t["question_id"]
        agg[a]["n"] += 1
        per_q_n[q][a] += 1
        if not t.get("ok"):
            agg[a]["error"] += 1
            continue
        g = t.get("grade", {})
        for k in ("contract_correct", "analytic_correct", "hit_hazard",
                  "abstained", "machine_readable"):
            if g.get(k):
                agg[a][k] += 1
                per_q[q][a][k] += 1
        for k in ("turns", "prompt_tokens", "completion_tokens", "wall_seconds"):
            v = t.get(k)
            if isinstance(v, (int, float)):
                num[a][k].append(float(v))

    print("=" * 78)
    print("OVERALL".center(78))
    print("=" * 78)
    hdr = f'{"metric":<26}' + "".join(f"{('arm ' + a):>24}" for a in arms)
    print(hdr)
    print("-" * 78)

    def row(label, fn):
        print(f"{label:<26}" + "".join(f"{fn(a):>24}" for a in arms))

    n = {a: agg[a]["n"] for a in arms}
    ok = {a: n[a] - agg[a]["error"] for a in arms}

    row("trials", lambda a: n[a])
    row("completed", lambda a: ok[a])
    row("errors / crashes", lambda a: agg[a]["error"])

    def pct(a, key):
        d = ok[a] or 1
        lo, hi = wilson(agg[a][key], ok[a])
        return f'{agg[a][key]}/{ok[a]} ({100*agg[a][key]/d:.1f}%) [{lo:.0f}-{hi:.0f}]'

    row("analytic correct", lambda a: pct(a, "analytic_correct"))
    row("contract correct", lambda a: pct(a, "contract_correct"))
    row("hit named hazard", lambda a: f'{agg[a]["hit_hazard"]}/{ok[a]}')
    row("abstained", lambda a: f'{agg[a]["abstained"]}/{ok[a]}')
    row("not machine-readable",
        lambda a: f'{ok[a] - agg[a]["machine_readable"]}/{ok[a]}')

    for k, label in (("turns", "mean turns"),
                     ("prompt_tokens", "mean prompt tokens"),
                     ("completion_tokens", "mean completion tokens"),
                     ("wall_seconds", "mean wall seconds")):
        row(label, lambda a, k=k: (f"{sum(num[a][k])/len(num[a][k]):.2f}"
                                   if num[a][k] else "-"))

    if len(arms) == 2:
        a, b = arms
        print("-" * 78)
        for k, label in (("turns", "turns"), ("prompt_tokens", "prompt tokens")):
            if num[a][k] and num[b][k]:
                ma = sum(num[a][k]) / len(num[a][k])
                mb = sum(num[b][k]) / len(num[b][k])
                delta = (mb - ma) / ma * 100 if ma else 0
                print(f"  {label:<22} arm {b} vs arm {a}: {delta:+.1f}%")

    # ---- per question -------------------------------------------------
    print()
    print("=" * 78)
    print("PER QUESTION  (analytic correct / completed)".center(78))
    print("=" * 78)
    regress, improve = [], []
    for q in sorted(per_q_n):
        cells = []
        for a in arms:
            cells.append(f'{per_q[q][a]["analytic_correct"]}/{per_q_n[q][a]}')
        line = f'{q:<6}' + "".join(f"{c:>12}" for c in cells)
        if len(arms) == 2:
            a, b = arms
            ra = per_q[q][a]["analytic_correct"] / max(per_q_n[q][a], 1)
            rb = per_q[q][b]["analytic_correct"] / max(per_q_n[q][b], 1)
            if rb < ra:
                line += "   <-- LEARNING HURT"
                regress.append(q)
            elif rb > ra:
                line += "   <-- learning helped"
                improve.append(q)
        print(line)

    print()
    if regress:
        print(f"REGRESSIONS ({len(regress)}): {', '.join(regress)}")
    else:
        print("REGRESSIONS: none")
    if improve:
        print(f"IMPROVEMENTS ({len(improve)}): {', '.join(improve)}")

    # ---- gate ---------------------------------------------------------
    if len(arms) == 2:
        a, b = arms
        acc_ok = agg[b]["analytic_correct"] / max(ok[b], 1) >= \
            agg[a]["analytic_correct"] / max(ok[a], 1)
        ta = sum(num[a]["turns"]) / len(num[a]["turns"]) if num[a]["turns"] else 0
        tb = sum(num[b]["turns"]) / len(num[b]["turns"]) if num[b]["turns"] else 0
        turn_ok = tb < ta
        print()
        print("=" * 78)
        print(f"GATE: learned accuracy >= cold  -> {'PASS' if acc_ok else 'FAIL'}")
        print(f"      in fewer turns            -> "
              f"{'PASS' if turn_ok else 'FAIL'} ({ta:.2f} -> {tb:.2f})")
        print(f"      OVERALL                   -> "
              f"{'PASS' if (acc_ok and turn_ok) else 'FAIL'}")
        print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "dbo_eval_results.json"))
