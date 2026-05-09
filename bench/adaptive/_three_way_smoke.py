"""Three-way honest comparison: naive vs forced-best-of-N vs Speculative Fanout.

Reads per-question result JSONs from bench/adaptive/results/srlm_eval_p4e_smoke/
(populated by srlm_eval.py for configs `default`, `adaptive_current_minrung3`,
and `adaptive_e_minrung3` on the same 27Q smoke set with model openai/gpt-4.1).

Reports honest per-domain breakdowns and the three pairwise comparisons that
matter:

    naive          vs adaptive_current  -> does best-of-N actually buy accuracy?
    adaptive_current vs adaptive_e      -> what does Speculative Fanout cost vs
                                            full fanout when you've already
                                            committed to best-of-N?
    naive          vs adaptive_e        -> the full picture for someone choosing
                                            between "ship one rollout" and
                                            "ship Speculative Fanout".
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("bench/adaptive/results/srlm_eval_p4e_smoke")
CONFIGS = ("default", "adaptive_current_minrung3", "adaptive_e_minrung3")


def load() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {c: {} for c in CONFIGS}
    for p in RESULTS_DIR.glob("*.json"):
        name = p.name
        if "__" not in name:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = data.get("config_name")
        qid = data.get("question_id")
        if cfg in out and qid:
            out[cfg][qid] = data
    return out


def agg(rows: dict[str, dict]) -> dict:
    if not rows:
        return {"n": 0, "passed": 0, "tokens": 0, "elapsed": 0.0}
    n = len(rows)
    passed = sum(1 for r in rows.values() if r.get("passed"))
    tokens = sum(int(r.get("total_cost_tokens", 0) or 0) for r in rows.values())
    elapsed = sum(float(r.get("elapsed_s", 0.0) or 0.0) for r in rows.values())
    return {"n": n, "passed": passed, "tokens": tokens, "elapsed": elapsed}


def by_domain(rows: dict[str, dict]) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows.values():
        buckets[r.get("domain") or "?"].append(r)
    out: dict[str, dict] = {}
    for d, rs in buckets.items():
        n = len(rs)
        passed = sum(1 for r in rs if r.get("passed"))
        tokens = sum(int(r.get("total_cost_tokens", 0) or 0) for r in rs)
        out[d] = {"n": n, "passed": passed, "tokens": tokens}
    return out


def main() -> int:
    data = load()
    for c in CONFIGS:
        if not data[c]:
            print(f"ERROR: no results for {c} in {RESULTS_DIR}")
            return 1

    # Aggregate overall
    print("=" * 78)
    print("3-way comparison — bench smoke (27Q, model openai/gpt-4.1, single seed)")
    print("=" * 78)
    print(f"{'config':32s} {'pass':>10s} {'tokens':>14s} {'elapsed (sum)':>16s}")
    print("-" * 78)
    aggs = {}
    for c in CONFIGS:
        a = agg(data[c])
        aggs[c] = a
        acc = a["passed"] / a["n"] if a["n"] else 0.0
        print(f"{c:32s} {a['passed']:>3d}/{a['n']:<3d} ({acc*100:>4.1f}%) "
              f"{a['tokens']:>14d} {a['elapsed']:>14.1f}s")

    # Per-domain
    print("\nPer-domain pass-rate (passed / n):")
    domains = sorted({r.get("domain") for c in CONFIGS for r in data[c].values()})
    print(f"{'domain':24s}", *[f"{c:>26s}" for c in CONFIGS], sep="")
    for d in domains:
        line = f"{d:24s}"
        for c in CONFIGS:
            bd = by_domain(data[c]).get(d, {"n": 0, "passed": 0})
            line += f" {bd['passed']:>4d}/{bd['n']:<4d} {('('+str(round(bd['passed']/max(bd['n'],1)*100))+'%)'):>15s}"
        print(line)

    # Per-domain token costs
    print("\nPer-domain mean tokens/question:")
    print(f"{'domain':24s}", *[f"{c:>20s}" for c in CONFIGS], sep="")
    for d in domains:
        line = f"{d:24s}"
        for c in CONFIGS:
            bd = by_domain(data[c]).get(d, {"n": 0, "tokens": 0})
            mean = bd["tokens"] / bd["n"] if bd["n"] else 0.0
            line += f" {mean:>20.0f}"
        print(line)

    # Per-question matrix on the failure-prone subset
    print("\nPer-question outcomes (only questions where at least one config differs):")
    qids = sorted(set().union(*[set(data[c].keys()) for c in CONFIGS]))
    diffs = []
    for qid in qids:
        out = [data[c].get(qid, {}).get("passed") for c in CONFIGS]
        if len(set(out)) > 1:
            diffs.append((qid, out))
    print(f"  {len(diffs)}/{len(qids)} questions where configs disagree")
    if diffs:
        print(f"  {'qid':28s}", *[f"{c[:14]:>16s}" for c in CONFIGS], sep="")
        for qid, out in diffs:
            row = f"  {qid:28s}"
            for v in out:
                row += f" {('✓' if v else ('✗' if v is False else '-')):>16s}"
            print(row)

    # Pairwise summaries
    print("\nPairwise (deltas):")

    def pair(a_name: str, b_name: str) -> None:
        a = aggs[a_name]
        b = aggs[b_name]
        d_pass = b["passed"] - a["passed"]
        d_tok = b["tokens"] - a["tokens"]
        d_pct = (d_tok / a["tokens"] * 100) if a["tokens"] else 0.0
        d_elapsed = b["elapsed"] - a["elapsed"]
        print(f"  {a_name:32s} -> {b_name:32s}  "
              f"acc {a['passed']:>3d} -> {b['passed']:>3d} ({d_pass:+d}), "
              f"tokens {d_pct:+5.1f}%, elapsed {d_elapsed:+6.1f}s")

    pair("default", "adaptive_current_minrung3")
    pair("default", "adaptive_e_minrung3")
    pair("adaptive_current_minrung3", "adaptive_e_minrung3")

    return 0


if __name__ == "__main__":
    sys.exit(main())
