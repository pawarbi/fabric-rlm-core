"""Offline re-grading of stage 3 results. No model calls.

The first-pass grader read only ``answer["value"]`` as a scalar. Some correct
answers arrive as a nested mapping, or as a numpy scalar that failed to
serialise, so a numerically correct result was scored wrong. That is a grading
artifact and would understate the cold arm.

This regrader separates two distinct things that the first pass conflated:

    numerically_correct  the reference value is present in the answer
    machine_readable     answer["value"] is directly usable as a number

An answer can be the first without the second, and that gap is itself a
finding rather than a grading detail.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")


def deep_numbers(node, out: list[float]) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        out.append(float(node))
        return
    if isinstance(node, str):
        for token in NUMBER.findall(node):
            try:
                out.append(float(token.replace(",", "")))
            except ValueError:
                pass
        return
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"__type__", "__serializable__"}:
                continue
            deep_numbers(value, out)
        return
    if isinstance(node, list):
        for value in node:
            deep_numbers(value, out)


def scalar(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").replace("$", "").strip())
        except ValueError:
            return None
    return None


def close(a, b) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= max(0.005, abs(b) * 0.001)


def main() -> None:
    report = json.loads(Path("stage3_results.json").read_text(encoding="utf-8"))
    refs = report["references"]

    degenerate = {
        key
        for key, value in refs.items()
        if close(float(value["value"]), float(value["hazard"]))
    }

    rows = []
    for trial in report["trials"]:
        row = {k: trial[k] for k in ("question_id", "arm", "rep")}
        row["ok"] = bool(trial.get("ok"))
        if not row["ok"]:
            row["error"] = trial["error"]
            rows.append(row)
            continue
        answer = trial.get("answer") or {}
        ref = refs[trial["question_id"]]
        found: list[float] = []
        deep_numbers(answer.get("value"), found)
        direct = scalar(answer.get("value"))
        row["numerically_correct"] = any(close(v, float(ref["value"])) for v in found)
        row["machine_readable"] = direct is not None
        row["direct_value"] = direct
        row["hit_hazard"] = (
            any(close(v, float(ref["hazard"])) for v in found)
            and not row["numerically_correct"]
        )
        if "entity" in ref:
            text = json.dumps(answer).lower()
            row["entity_ok"] = ref["entity"].lower() in text
            row["numerically_correct"] = row["numerically_correct"] and row["entity_ok"]
        row["abstained"] = str(answer.get("status", "")).lower() in {
            "abstain", "abstained", "needs_definition", "unknown"
        }
        row["turns"] = trial.get("turns")
        row["wall_seconds"] = trial.get("wall_seconds")
        row["prompt_tokens"] = trial.get("prompt_tokens")
        rows.append(row)

    agg = defaultdict(lambda: defaultdict(int))
    perf = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cell = agg[(row["question_id"], row["arm"])]
        cell["n"] += 1
        if not row["ok"]:
            cell["error"] += 1
            continue
        cell["correct"] += int(row["numerically_correct"])
        cell["hazard"] += int(row["hit_hazard"])
        cell["unreadable"] += int(not row["machine_readable"])
        cell["abstained"] += int(row["abstained"])
        for metric in ("turns", "wall_seconds", "prompt_tokens"):
            if row.get(metric) is not None:
                perf[(row["question_id"], row["arm"])][metric].append(row[metric])

    def mean(values):
        return round(sum(values) / len(values), 1) if values else None

    print("Degenerate questions (reference within tolerance of hazard):", sorted(degenerate) or "none")
    print()
    header = f"{'question':22}{'arm':4}{'correct':9}{'hazard':7}{'unread':7}{'err':5}{'turns':7}{'secs':7}{'ptok':7}"
    print(header)
    print("-" * len(header))
    for key in sorted(agg):
        cell, p = agg[key], perf[key]
        flag = "  <- non-discriminating" if key[0] in degenerate else ""
        print(
            f"{key[0]:22}{key[1]:4}{cell['correct']}/{cell['n']:<7}{cell['hazard']:<7}"
            f"{cell['unreadable']:<7}{cell['error']:<5}"
            f"{str(mean(p['turns'])):<7}{str(mean(p['wall_seconds'])):<7}"
            f"{str(mean(p['prompt_tokens'])):<7}{flag}"
        )

    print("\nPer-arm totals over discriminating questions only:")
    for arm in ("A", "B"):
        cells = [agg[k] for k in agg if k[1] == arm and k[0] not in degenerate]
        n = sum(c["n"] for c in cells)
        print(
            f"  arm {arm}: correct {sum(c['correct'] for c in cells)}/{n}  "
            f"hazard {sum(c['hazard'] for c in cells)}  "
            f"errors {sum(c['error'] for c in cells)}  "
            f"unreadable {sum(c['unreadable'] for c in cells)}"
        )

    Path("stage3_regraded.json").write_text(
        json.dumps({"degenerate": sorted(degenerate), "rows": rows}, indent=1),
        encoding="utf-8",
    )
    print("\nwrote stage3_regraded.json")


if __name__ == "__main__":
    main()
