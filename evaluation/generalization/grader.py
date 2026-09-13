"""Grading for the cross-domain generalization evaluation.

The first version of this grader compared every field with ``==``, including
the free-text ``grain`` and ``period``. On the first live smoke run that scored
6 of 6 answered trials as ``confident_wrong`` while every one of them had the
correct value: ``'latest inventory snapshot product grain'`` is not string-equal
to ``'latest warehouse-product snapshot rows'``, so a correct answer landed in
the most alarming bucket the report has.

This version separates two things the old one conflated.

**Correctness** is value, units and period, compared after normalization.
Those three are checkable and low-variance: a number is a number, units come
from a small vocabulary, and a reporting period reduces to a set of dates.

**Grain** is scored on its own and never decides correctness. It is free text
with no controlled vocabulary, so a binary equal/not-equal on it measures
wording rather than analysis -- but it is still worth recording, because losing
a qualifier ('production rows' for 'complete production periods') is a real
loss of meaning.

Every result also carries ``strict``: the original verbatim verdict. Loosening
a grader is exactly the kind of change that can quietly inflate a number, so
both scores are reported side by side and the difference stays auditable.

The normalizations are deliberately one-sided. They forgive *representation*
(``"1,800"`` for ``1800``, ``ratio`` for ``proportion``, extra date precision)
and never forgive *magnitude or meaning* (66.67 percent is not the ratio
0.6667, minutes are not hours, 187 is not 186). Tests pin both directions,
with most of them pinning what must keep failing.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping

# Units denoting the same quantity. Each frozenset is one equivalence class;
# anything outside a shared class counts as a different unit, so percent and
# ratio stay distinct and a magnitude error cannot hide as a wording
# difference.
_UNIT_CLASSES: tuple[frozenset[str], ...] = (
    frozenset({"unit", "units", "count", "rows", "row", "records", "record"}),
    frozenset({"ratio", "proportion", "fraction", "share", "rate"}),
    frozenset({"percent", "percentage", "pct", "%"}),
    frozenset({"minute", "minutes", "min", "mins"}),
    frozenset({"hour", "hours", "hr", "hrs"}),
    frozenset({"day", "days"}),
    frozenset({"usd", "dollars", "dollar", "$", "currency"}),
    frozenset({"ticket", "tickets"}),
    frozenset({"customer", "customers"}),
    frozenset({"product", "products"}),
    frozenset({"line", "lines"}),
)

# Words carrying no analytical meaning in a grain description.
_GRAIN_STOPWORDS = frozenset({
    "a", "an", "the", "of", "per", "by", "at", "in", "on", "for", "and",
    "level", "grain", "granularity", "id", "ids", "key", "each", "every",
})

_GRAIN_EQUIVALENT_THRESHOLD = 0.5

_DATE_TOKEN = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")
_NUMERIC = re.compile(r"^[-+]?[\d,\s]*\.?\d+(?:[eE][-+]?\d+)?$")


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------
def _text(value: object) -> str:
    return re.sub(r"[\s_\-]+", " ", str(value).strip().lower()).strip()


def is_unserializable(value: object) -> bool:
    """True for the opaque marker ``freeze()`` emits for a non-JSON value.

    An answer carrying one of these is incomplete, never correct, however good
    the underlying computation was -- the caller never receives the number.
    """
    return isinstance(value, Mapping) and value.get("__serializable__") is False


def _as_number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and _NUMERIC.match(value.strip()):
        try:
            return float(value.replace(",", "").replace(" ", ""))
        except ValueError:
            return None
    return None


def _unit_class(value: object) -> frozenset[str] | str:
    token = _text(value).rstrip(".")
    for cls in _UNIT_CLASSES:
        if token in cls:
            return cls
    return token


def _date_tokens(value: object) -> list[str]:
    return _DATE_TOKEN.findall(str(value)) if value is not None else []


# ---------------------------------------------------------------------------
# field comparisons
# ---------------------------------------------------------------------------
def values_match(actual: object, expected: object) -> bool:
    """Compare answer values, forgiving representation but not magnitude."""
    if is_unserializable(actual) or actual is None:
        return False
    a, e = _as_number(actual), _as_number(expected)
    if a is not None and e is not None:
        return math.isclose(a, e, rel_tol=1e-6, abs_tol=1e-9)
    if a is not None or e is not None:
        return False
    if isinstance(expected, (list, tuple, set)) or isinstance(actual, (list, tuple, set)):
        if not (isinstance(actual, (list, tuple, set))
                and isinstance(expected, (list, tuple, set))):
            return False
        return {_text(x) for x in actual} == {_text(x) for x in expected}
    if isinstance(expected, str) or isinstance(actual, str):
        return _text(actual) == _text(expected)
    return actual == expected


def units_match(actual: object, expected: object) -> bool:
    if actual is None:
        return expected is None
    return _unit_class(actual) == _unit_class(expected)


def periods_match(actual: object, expected: object) -> bool:
    """A period matches when it covers every date the reference names.

    Reduced to date tokens, so ``2026-01`` is satisfied by
    ``2026-01-01 through 2026-01-31`` (extra precision) and an open-ended
    ``through 2026-03-18`` is satisfied by a bounded range ending there. An
    answer naming no date at all -- ``None``, or a filter condition such as
    ``'complete reporting periods'`` -- never matches a dated reference, which
    is the behaviour that exposed a real weakness the verbatim grader hid.
    """
    want, got = _date_tokens(expected), _date_tokens(actual)
    if not want:
        if actual is None:
            return expected is None
        return _text(actual) == _text(expected)
    if not got:
        return False
    return all(any(g.startswith(w) or w.startswith(g) for g in got) for w in want)


def score_grain(actual: object, expected: object) -> dict[str, Any]:
    """Score grain on its own scale: exact, equivalent, mismatch or missing."""
    if expected is None:
        return {"match": "not_specified", "actual": actual, "expected": expected}
    if actual is None or not str(actual).strip():
        return {"match": "missing", "actual": actual, "expected": expected}
    if _text(actual) == _text(expected):
        return {"match": "exact", "actual": actual, "expected": expected,
                "overlap": 1.0}

    def words(value: object) -> set[str]:
        return {w.rstrip("s") for w in _text(value).split()
                if w not in _GRAIN_STOPWORDS}

    a, e = words(actual), words(expected)
    overlap = len(a & e) / len(e) if e else 0.0
    match = "equivalent" if overlap >= _GRAIN_EQUIVALENT_THRESHOLD else "mismatch"
    return {"match": match, "actual": actual, "expected": expected,
            "overlap": round(overlap, 3)}


# ---------------------------------------------------------------------------
# the original verbatim grader, preserved for comparison
# ---------------------------------------------------------------------------
def _strict_equal(actual: object, expected: object) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


def grade_answer_strict(answer: Mapping[str, Any],
                        expected: Mapping[str, Any]) -> dict[str, Any]:
    """The original grader, unchanged, reported alongside the current one."""
    status = answer.get("status")
    if isinstance(status, str):
        status = status.strip().lower()
    if status in {"timeout", "failed", "incomplete"} or not status:
        return {"outcome": "incomplete", "correct": False}
    if expected.get("expected_status", "answered") == "abstain":
        correct = status in {"abstain", "needs_definition", "uncertain"}
        return {"outcome": "correct" if correct else "confident_wrong",
                "correct": correct}
    if status not in {"answered", "success", "ok"}:
        return {"outcome": "incomplete", "correct": False}
    checks = {
        field: _strict_equal(answer.get(field), expected.get(field))
        for field in ("value", "units", "grain", "period")
        if field in expected
    }
    for field in ("entity_id", "entity_ids"):
        if field in expected:
            checks[field] = _strict_equal(answer.get(field), expected.get(field))
    correct = all(checks.values())
    return {"outcome": "correct" if correct else "confident_wrong",
            "correct": correct, "checks": checks}


# ---------------------------------------------------------------------------
# the grader
# ---------------------------------------------------------------------------
def grade_answer(answer: Mapping[str, Any],
                 expected: Mapping[str, Any]) -> dict[str, Any]:
    strict = grade_answer_strict(answer, expected)

    status = answer.get("status")
    if isinstance(status, str):
        status = status.strip().lower()

    if status in {"timeout", "failed", "incomplete"} or not status:
        return {"outcome": "incomplete", "correct": False, "strict": strict,
                "reason": "no usable status"}

    if expected.get("expected_status", "answered") == "abstain":
        correct = status in {"abstain", "needs_definition", "uncertain",
                             "insufficient_metadata"}
        return {"outcome": "correct" if correct else "confident_wrong",
                "correct": correct, "strict": strict,
                "expected_behavior": "abstain"}

    if status not in {"answered", "success", "ok", "complete", "final"}:
        return {"outcome": "incomplete", "correct": False, "strict": strict,
                "reason": f"unrecognized status {status!r}"}

    value = answer.get("value")
    unserializable = is_unserializable(value)
    if value is None or unserializable:
        return {
            "outcome": "incomplete",
            "correct": False,
            "strict": strict,
            "unserializable": unserializable,
            "reason": "opaque serialization marker in value" if unserializable
                      else "no value submitted",
        }

    checks: dict[str, bool] = {}
    if "value" in expected:
        checks["value"] = values_match(value, expected["value"])
    if "units" in expected:
        checks["units"] = units_match(answer.get("units"), expected["units"])
    if "period" in expected:
        checks["period"] = periods_match(answer.get("period"), expected["period"])
    for field in ("entity_id", "entity_ids"):
        if field in expected:
            checks[field] = values_match(answer.get(field), expected[field])

    grain = score_grain(answer.get("grain"), expected.get("grain"))
    correct = all(checks.values())

    unsupported = [
        claim for claim in answer.get("claims", ())
        if isinstance(claim, Mapping) and claim.get("supported") is False
    ]

    result: dict[str, Any] = {
        "outcome": "correct" if correct else "confident_wrong",
        "correct": correct,
        "checks": checks,
        "grain": grain,
        "unserializable": False,
        "strict": strict,
    }
    if unsupported:
        result["outcome"] = "unsupported_claim"
        result["unsupported_claims"] = unsupported
    return result


__all__ = [
    "grade_answer",
    "grade_answer_strict",
    "is_unserializable",
    "periods_match",
    "score_grain",
    "units_match",
    "values_match",
]
