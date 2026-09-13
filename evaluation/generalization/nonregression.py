"""Observed per-question parity, not a claim of population-wide reliability."""

from collections import Counter

from .evidence_summary import summarize


def _key(row):
    return row["question_id"], row["variant"], row["repetition"], row["arm"]


def evaluate(document: dict) -> dict:
    schedule = document.get("planned_schedule", [])
    rows = document.get("trials", [])
    expected = Counter(_key(row) for row in schedule)
    actual = Counter(_key(row) for row in rows)
    coverage_ok = bool(expected) and expected == actual and all(n == 1 for n in expected.values())
    integrity_ok = (
        document.get("status") == "complete"
        and document.get("core_freeze_mismatches") == []
        and document.get("fixture_changes") == []
    )
    groups = {}
    regressions = []
    for question_id, variant in sorted({(row["question_id"], row["variant"]) for row in schedule}):
        arms = {
            arm: summarize([
                row for row in rows if row["question_id"] == question_id
                and row["variant"] == variant and row["arm"] == arm
            ])
            for arm in ("A", "B", "C")
        }
        groups[f"{question_id}|{variant}"] = arms
        cold = arms["A"]
        for arm in ("B", "C"):
            candidate = arms[arm]
            reasons = []
            for field in ("answer_correct", "value_identity_correct", "expected_abstention_detected"):
                if candidate[field] < cold[field]:
                    reasons.append(field)
            if cold["answerable_tasks"] and candidate["incomplete_tasks"] > cold["incomplete_tasks"]:
                reasons.append("completion")
            if reasons:
                regressions.append({
                    "question_id": question_id, "variant": variant, "arm": arm,
                    "reasons": reasons,
                })
    return {
        "passed": coverage_ok and integrity_ok and not regressions,
        "coverage_ok": coverage_ok, "integrity_ok": integrity_ok,
        "regressions": regressions, "questions": groups,
        "scope": (
            "Fixed tested questions only. All abstentions are reported as incomplete; "
            "completion parity applies to answerable tasks. Expected abstention is "
            "graded separately. A parity pass is not an absolute accuracy floor "
            "or statistical proof of generalization."
        ),
    }
