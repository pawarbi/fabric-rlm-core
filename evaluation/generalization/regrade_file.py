"""Re-grade a recorded result file with the current grader.

Offline: no model calls. Used to confirm that a grader change recovers
answers that were correct but unreadable, without re-running trials.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import stage3_fabric_live as harness


def main(path: str) -> None:
    refs = harness.references()
    data = json.load(open(path))
    trials = data["trials"] if isinstance(data, dict) and "trials" in data else data

    totals = defaultdict(lambda: [0, 0])
    unreadable = defaultdict(int)
    fixed = 0

    print("regraded trials:")
    for trial in trials:
        if not trial.get("ok"):
            continue
        qid = trial["question_id"]
        if refs[qid].get("degenerate"):
            continue
        grade = harness.grade(qid, trial.get("answer") or {}, refs)
        was = trial.get("grade", {}).get("correct")
        note = ""
        if grade["correct"] and not was:
            note = "  <-- recovered"
            fixed += 1
        arm = trial["arm"]
        totals[arm][1] += 1
        totals[arm][0] += int(grade["correct"])
        if not grade["machine_readable"]:
            unreadable[arm] += 1
        print(
            f"  {qid:34} {arm} was={str(was):5} now={str(grade['correct']):5} "
            f"value={grade['value']} readable={grade['machine_readable']}{note}"
        )

    print()
    for arm in sorted(totals):
        correct, total = totals[arm]
        print(
            f"Arm {arm}: {correct}/{total} correct, "
            f"{unreadable[arm]} not machine-readable"
        )
    print(f"\nrecovered by the grader fix: {fixed}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stage4_smoke.json")
