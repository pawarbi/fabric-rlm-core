from __future__ import annotations

import math
from typing import Any, Mapping


def _equal(actual: object, expected: object) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


def grade_answer(answer: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    status = answer.get("status")
    if status in {"timeout", "failed", "incomplete"} or not status:
        return {"outcome": "incomplete", "correct": False}
    expected_status = expected.get("expected_status", "answered")
    if expected_status == "abstain":
        correct = status in {"abstain", "needs_definition", "uncertain"}
        return {
            "outcome": "correct" if correct else "confident_wrong",
            "correct": correct,
        }
    if status != "answered":
        return {"outcome": "incomplete", "correct": False}
    unsupported = [
        claim
        for claim in answer.get("claims", ())
        if isinstance(claim, Mapping) and claim.get("supported") is False
    ]
    checks = {
        field: _equal(answer.get(field), expected.get(field))
        for field in ("value", "units", "grain", "period")
        if field in expected
    }
    for identity_field in ("entity_id", "entity_ids"):
        if identity_field in expected:
            checks[identity_field] = _equal(
                answer.get(identity_field), expected.get(identity_field)
            )
    if unsupported:
        return {
            "outcome": "unsupported_claim",
            "correct": all(checks.values()),
            "checks": checks,
            "unsupported_claims": unsupported,
        }
    correct = all(checks.values())
    return {
        "outcome": "correct" if correct else "confident_wrong",
        "correct": correct,
        "checks": checks,
    }


__all__ = ["grade_answer"]
