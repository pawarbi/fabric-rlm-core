"""Summarize recorded trials without rerunning models or changing the grader."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .runner import _write_json

SUCCESS = {"answered", "success", "ok", "complete", "final"}


def _marker(value):
    if isinstance(value, dict):
        return value.get("__serializable__") is False or any(_marker(v) for v in value.values())
    return isinstance(value, list) and any(_marker(v) for v in value)


def summarize(rows: list[dict]) -> dict:
    numeric = [r for r in rows if r["expected"].get("expected_status") != "abstain"]
    expected_abstain = [r for r in rows if r["expected"].get("expected_status") == "abstain"]

    def status(row):
        return str(row.get("answer", {}).get("status", "")).strip().lower()

    def value_identity(row):
        fields = [key for key in ("value", "entity_id", "entity_ids") if key in row["expected"]]
        checks = row["grade"].get("checks", {})
        return bool(fields) and all(checks.get(field) is True for field in fields)

    costs = [r.get("metrics", {}).get("provider_cost_usd") for r in rows]
    known_costs = [value for value in costs if isinstance(value, (int, float))]
    walls = [r.get("metrics", {}).get("wall_seconds") for r in rows]
    walls = [value for value in walls if isinstance(value, (int, float))]
    result = {
        "trials": len(rows), "answerable_tasks": len(numeric),
        "answer_correct": sum(r["grade"].get("correct") is True for r in numeric),
        "strict_correct": sum(r["grade"].get("strict", {}).get("correct") is True for r in numeric),
        "value_identity_correct": sum(value_identity(r) for r in numeric),
        "expected_abstention_cases": len(expected_abstain),
        "expected_abstention_detected": sum(r["grade"].get("correct") is True for r in expected_abstain),
        "incomplete_tasks": sum(
            status(r) not in SUCCESS or bool(r.get("error")) or _marker(r.get("answer"))
            or r.get("answer", {}).get("value") is None for r in rows
        ),
        "wrong_value_or_identity": sum(
            status(r) in SUCCESS and not value_identity(r)
            and r.get("answer", {}).get("value") is not None
            and not _marker(r.get("answer")) for r in numeric
        ),
        "framing_only_mismatch": sum(value_identity(r) and not r["grade"].get("correct") for r in numeric),
        "serialization_marker_trials": sum(_marker(r.get("answer")) for r in rows),
        "grader_outcomes": dict(Counter(r["grade"]["outcome"] for r in rows)),
        "provider_cost_usd": sum(known_costs) if known_costs else None,
        "cost_known_trials": len(known_costs),
        "wall_seconds": sum(walls) if walls else None,
        "wall_median_seconds": statistics.median(walls) if walls else None,
        "wall_known_trials": len(walls),
        "verification_outcomes": dict(Counter(r.get("metrics", {}).get("verification_outcome", "unknown") for r in rows)),
        "knowledge_modes": dict(Counter(str(r.get("metrics", {}).get("knowledge_mode")) for r in rows)),
        "lessons_injected_total": sum(r.get("metrics", {}).get("lessons_injected", 0) for r in rows),
    }
    for field in ("value", "units", "period", "entity_id", "entity_ids"):
        applicable = [r for r in numeric if field in r["expected"]]
        result[field] = {
            "correct": sum(r["grade"].get("checks", {}).get(field) is True for r in applicable),
            "applicable": len(applicable),
        }
    for field in ("prompt_tokens", "completion_tokens", "cached_tokens", "source_calls",
                  "failed_source_calls", "operation_selection_lm_calls"):
        known = [r.get("metrics", {}).get(field) for r in rows]
        known = [v for v in known if isinstance(v, (int, float))]
        result[field] = {"sum": sum(known) if known else None, "known_trials": len(known)}
    return result


def analyze(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("status") != "complete":
        raise ValueError(f"refusing incomplete run: {path}")
    if document.get("core_freeze_mismatches") or document.get("fixture_changes"):
        raise ValueError(f"freeze/fixture integrity failure: {path}")
    rows = document["trials"]
    groups = defaultdict(list)
    for row in rows:
        for kind, fields in (
            ("arm", ("arm",)), ("domain", ("domain", "variant", "arm")),
            ("question", ("question_id", "variant", "arm")),
            ("repetition", ("variant", "repetition", "arm")),
        ):
            groups[(kind, *(str(row[f]) for f in fields))].append(row)
    groups_summary = {"|".join(key): summarize(items) for key, items in sorted(groups.items())}
    regressions = []
    for row in rows:
        if row["arm"] != "A":
            continue
        key = f"question|{row['question_id']}|{row['variant']}|"
        if any(r["key"] == key for r in regressions):
            continue
        baseline = groups_summary[key + "A"]
        for arm in ("B", "C"):
            candidate = groups_summary.get(key + arm)
            if candidate is not None and (
                candidate["answer_correct"] < baseline["answer_correct"]
                or candidate["value_identity_correct"] < baseline["value_identity_correct"]
            ):
                regressions.append({"key": key, "arm": arm, "A": baseline, "candidate": candidate})
    return {
        "input": str(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "repetitions": document.get("repetitions"),
        "provenance": document.get("provenance", {"warning": "legacy run; actual SHA not captured"}),
        "packages": document.get("packages"), "overall": summarize(rows),
        "groups": groups_summary, "observed_regressions": regressions,
        "interpretation": {
            "correctness": "Original value/units/period/identity criterion retained; strict reported separately.",
            "abstentions": "Incomplete analytical tasks, including correctly identified unanswerable tasks.",
            "verification": "Runtime checker outcome only; does not establish semantic correctness.",
            "evidence_coverage": "Independent claim-to-evidence coverage not measured; supported flags are self-reported.",
            "source_calls": "Instrumented adapter calls only; direct filesystem/pandas calls excluded.",
            "sampling": "Repeated fixed synthetic questions; not an IID population or proof of general reliability.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input)
    _write_json(args.output, result)
    print(json.dumps({key: value for key, value in result["groups"].items() if key.startswith("arm|")}, indent=2))


if __name__ == "__main__":
    main()
