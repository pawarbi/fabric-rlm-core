"""Aggregate turns, tokens and latency per arm from the Stage 3 raw results.

Offline only: reads the recorded trials, makes no model calls.
"""

from __future__ import annotations

import json
import statistics as st

DISCRIMINATING = [
    "q_ecom_avg_payment",
    "q_ecom_order_count",
    "q_retail_top_product",
]


def load(path):
    data = json.load(open(path))
    return data["trials"] if isinstance(data, dict) and "trials" in data else data


def agg(rows):
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return len(rows), None, None, None, None
    return (
        len(rows),
        round(st.mean([r["turns"] for r in ok]), 2),
        round(st.mean([r["prompt_tokens"] for r in ok])),
        round(st.mean([r["completion_tokens"] for r in ok])),
        round(st.mean([r["wall_seconds"] for r in ok]), 1),
    )


def main():
    trials = load("stage3_results.json")
    header = ("question", "arm", "n", "turns", "ptok", "ctok", "sec")
    print("{:24} {:3} {:>2} {:>6} {:>7} {:>6} {:>6}".format(*header))
    for qid in DISCRIMINATING:
        for arm in ("A", "B"):
            rows = [r for r in trials if r["question_id"] == qid and r["arm"] == arm]
            n, turns, ptok, ctok, sec = agg(rows)
            print(
                "{:24} {:3} {:>2} {!s:>6} {!s:>7} {!s:>6} {!s:>6}".format(
                    qid, arm, n, turns, ptok, ctok, sec
                )
            )

    print()
    totals = {}
    for arm in ("A", "B"):
        rows = [
            r
            for r in trials
            if r["question_id"] in DISCRIMINATING and r["arm"] == arm
        ]
        totals[arm] = agg(rows)
        n, turns, ptok, ctok, sec = totals[arm]
        print(
            f"OVERALL arm {arm}: n={n} turns={turns} prompt_tok={ptok} "
            f"completion_tok={ctok} sec={sec}"
        )

    a, b = totals["A"], totals["B"]
    print()
    for label, idx in (("turns", 1), ("prompt_tokens", 2), ("completion_tokens", 3), ("wall_seconds", 4)):
        if a[idx] and b[idx]:
            change = (b[idx] - a[idx]) / a[idx] * 100
            print(f"{label:18} A={a[idx]!s:>8} B={b[idx]!s:>8}  change={change:+.1f}%")

    crashed = [
        r
        for r in trials
        if r["question_id"] in DISCRIMINATING and not r.get("ok")
    ]
    print(f"\ncrashed trials (excluded from means): {len(crashed)}")
    for r in crashed:
        print(f"  {r['question_id']} arm={r['arm']} rep={r['rep']} reason={r.get('failure_reason')}")


if __name__ == "__main__":
    main()
