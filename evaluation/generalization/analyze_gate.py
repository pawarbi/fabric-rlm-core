"""Decide the F8 gate from a recorded A/B run.

The gate the user set: learned accuracy >= cold, in fewer turns, or on
novel/complex questions.

Reports per question, per domain and overall, and never lets an average
hide a regression: any question where learning loses is listed
explicitly. Abstentions and errors count as incomplete, and confidently
wrong answers are counted separately from abstentions.

Offline: reads recorded trials, makes no model calls.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict

import stage3_fabric_live as harness


def wilson(correct: int, total: int) -> tuple[float, float]:
    """95% Wilson interval, so small-n results are not over-read."""
    if total == 0:
        return (0.0, 1.0)
    z = 1.96
    phat = correct / total
    denom = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denom
    margin = z * ((phat * (1 - phat) + z**2 / (4 * total)) / total) ** 0.5 / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def main(path: str) -> None:
    refs = harness.references()
    data = json.load(open(path))
    trials = data["trials"] if isinstance(data, dict) and "trials" in data else data

    per_q = defaultdict(lambda: defaultdict(list))
    for trial in trials:
        qid = trial["question_id"]
        if refs[qid].get("degenerate"):
            continue
        per_q[qid][trial["arm"]].append(trial)

    def summarise(rows):
        done = [r for r in rows if r.get("ok")]
        regraded = [
            (r, harness.grade(r["question_id"], r.get("answer") or {}, refs))
            for r in done
        ]
        correct = sum(1 for _, g in regraded if g["correct"])
        analytic = sum(1 for _, g in regraded if g["analytic_correct"])
        hazard = sum(1 for _, g in regraded if g["hit_hazard"] or g["analytic_hazard"])
        wrong = sum(1 for _, g in regraded if g["confidently_wrong"])
        abst = sum(1 for _, g in regraded if g["abstained"])
        unread = sum(1 for _, g in regraded if not g["machine_readable"])
        turns = [r["turns"] for r in done] or [0]
        ptok = [r["prompt_tokens"] for r in done] or [0]
        return {
            "n": len(rows),
            "errors": len(rows) - len(done),
            "correct": correct,
            "analytic": analytic,
            "hazard": hazard,
            "confidently_wrong": wrong,
            "abstained": abst,
            "unreadable": unread,
            "turns": round(st.mean(turns), 2),
            "prompt_tokens": round(st.mean(ptok)),
        }

    print("PER QUESTION   (ok = contract-compliant, an = analytically correct)")
    header = ("question", "arm", "ok", "an", "haz", "abs", "err", "turns", "ptok")
    print("{:34} {:3} {:>5} {:>5} {:>4} {:>4} {:>4} {:>6} {:>7}".format(*header))
    regressions, improvements = [], []
    for qid in sorted(per_q):
        stats = {}
        for arm in ("A", "B"):
            s = summarise(per_q[qid][arm])
            stats[arm] = s
            print(
                "{:34} {:3} {:>5} {:>5} {:>4} {:>4} {:>4} {:>6} {:>7}".format(
                    qid,
                    arm,
                    f"{s['correct']}/{s['n']}",
                    f"{s['analytic']}/{s['n']}",
                    s["hazard"],
                    s["abstained"],
                    s["errors"],
                    s["turns"],
                    s["prompt_tokens"],
                )
            )
        if stats["B"]["analytic"] < stats["A"]["analytic"]:
            regressions.append((qid, stats))
        elif stats["B"]["analytic"] > stats["A"]["analytic"]:
            improvements.append((qid, stats))

    print("\nPER DOMAIN")
    dom = defaultdict(lambda: defaultdict(list))
    for trial in trials:
        if refs[trial["question_id"]].get("degenerate"):
            continue
        dom[trial["domain"]][trial["arm"]].append(trial)
    for domain in sorted(dom):
        for arm in ("A", "B"):
            s = summarise(dom[domain][arm])
            print(
                f"  {domain:14} {arm}  contract {s['correct']}/{s['n']}  "
                f"analytic {s['analytic']}/{s['n']}  hazard={s['hazard']}  "
                f"turns={s['turns']}  ptok={s['prompt_tokens']}"
            )

    print("\nOVERALL")
    overall = {}
    for arm in ("A", "B"):
        rows = [
            t
            for t in trials
            if t["arm"] == arm and not refs[t["question_id"]].get("degenerate")
        ]
        s = summarise(rows)
        overall[arm] = s
        lo, hi = wilson(s["analytic"], s["n"])
        print(
            f"  Arm {arm}: contract {s['correct']}/{s['n']} "
            f"({s['correct']/s['n']:.1%})   "
            f"analytic {s['analytic']}/{s['n']} "
            f"({s['analytic']/s['n']:.1%}, 95% CI {lo:.1%}-{hi:.1%})"
        )
        print(
            f"          hazard={s['hazard']} abstained={s['abstained']} "
            f"errors={s['errors']} unreadable={s['unreadable']}  "
            f"turns={s['turns']} ptok={s['prompt_tokens']}"
        )

    a, b = overall["A"], overall["B"]
    print("\nGATE: learned accuracy >= cold, in fewer turns")
    acc_ok = b["analytic"] >= a["analytic"]
    contract_ok = b["correct"] >= a["correct"]
    turns_ok = b["turns"] < a["turns"]
    print(
        f"  analytic accuracy  B={b['analytic']} A={a['analytic']}  "
        f"-> {'PASS' if acc_ok else 'FAIL'}"
    )
    print(
        f"  contract accuracy  B={b['correct']} A={a['correct']}  "
        f"-> {'PASS' if contract_ok else 'FAIL'}"
    )
    print(f"  turns              B={b['turns']} A={a['turns']}  -> {'PASS' if turns_ok else 'FAIL'}")
    a_lo, a_hi = wilson(a["analytic"], a["n"])
    b_lo, b_hi = wilson(b["analytic"], b["n"])
    overlap = b_hi >= a_lo and a_hi >= b_lo
    print(f"  CIs overlap: {overlap} (if True, the accuracy gap itself is not established)")
    print(f"\n  VERDICT: {'PASS' if acc_ok and contract_ok and turns_ok else 'FAIL'}")

    if regressions:
        print("\nQUESTIONS WHERE LEARNING LOSES (not hidden by the average):")
        for qid, stats in regressions:
            print(
                f"  {qid}: A={stats['A']['analytic']}/{stats['A']['n']} "
                f"B={stats['B']['analytic']}/{stats['B']['n']} "
                f"hazard_B={stats['B']['hazard']}"
            )
    else:
        print("\nNo question regressed under learning.")
    if improvements:
        print("\nQUESTIONS WHERE LEARNING WINS:")
        for qid, stats in improvements:
            print(
                f"  {qid}: A={stats['A']['analytic']}/{stats['A']['n']} "
                f"B={stats['B']['analytic']}/{stats['B']['n']}"
            )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stage4_gate.json")
