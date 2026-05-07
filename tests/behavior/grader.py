"""Comparators for behavior-CI question results.

The grader compares the LM-emitted answer to the locally-computed expected value
for each behavior question.  Comparators are intentionally narrow and explicit
so failures produce useful messages, not silent wrong-type passes.

Supported comparators:
    "exact"  - Python ``==`` after light normalization (string strip; int/float
               equivalence).  Used for integer/string answers.
    "near"   - numeric within ``abs_tol=1e-6, rel_tol=1e-3``.  Used for floats
               where a tiny rounding difference is expected.
    "string" - ``repr(actual) == repr(expected)`` after stripping whitespace
               around any string actual.  Used for ordered sequences and
               structures where element identity matters but native ``==``
               might compare loosely (e.g. ``[1, 2] == (1, 2)`` is False but
               ``["1", "2"]`` vs ``[1, 2]`` should also fail — this comparator
               surfaces both).
    "set"    - both sides converted to ``set``; order-insensitive equality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GradeResult:
    """Outcome of grading a single (answer, expected) pair."""

    passed: bool
    cmp: str
    expected: Any
    actual: Any
    reason: str  # human-readable explanation; empty string when passed


def _coerce_numeric(value: Any) -> Any:
    """Convert numeric-looking strings to int/float; pass everything else through."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return value
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return value
    return value


def _grade_exact(actual: Any, expected: Any) -> tuple[bool, str]:
    a = _coerce_numeric(actual)
    e = _coerce_numeric(expected)
    if isinstance(a, (int, float)) and isinstance(e, (int, float)) and not isinstance(a, bool):
        return (a == e, "" if a == e else f"numeric mismatch: {a!r} != {e!r}")
    if isinstance(a, str) and isinstance(e, str):
        return (
            a.strip() == e.strip(),
            "" if a.strip() == e.strip() else f"string mismatch (after strip): {a.strip()!r} != {e.strip()!r}",
        )
    if a == e:
        return (True, "")
    return (False, f"value mismatch: {a!r} != {e!r}")


def _grade_near(actual: Any, expected: Any, abs_tol: float, rel_tol: float) -> tuple[bool, str]:
    a = _coerce_numeric(actual)
    e = _coerce_numeric(expected)
    if not isinstance(a, (int, float)) or isinstance(a, bool):
        return (False, f"non-numeric actual for near-cmp: {actual!r}")
    if not isinstance(e, (int, float)) or isinstance(e, bool):
        return (False, f"non-numeric expected for near-cmp: {expected!r}")
    if math.isclose(a, e, abs_tol=abs_tol, rel_tol=rel_tol):
        return (True, "")
    return (False, f"numeric not within tol: {a!r} vs {e!r} (abs_tol={abs_tol}, rel_tol={rel_tol})")


def _grade_string(actual: Any, expected: Any) -> tuple[bool, str]:
    a = actual.strip() if isinstance(actual, str) else actual
    e = expected.strip() if isinstance(expected, str) else expected
    if repr(a) == repr(e):
        return (True, "")
    return (False, f"repr mismatch: {a!r} != {e!r}")


def _grade_set(actual: Any, expected: Any) -> tuple[bool, str]:
    try:
        a_set = set(actual)
    except TypeError as exc:
        return (False, f"actual not set-convertible ({type(actual).__name__}): {exc}")
    try:
        e_set = set(expected)
    except TypeError as exc:
        return (False, f"expected not set-convertible ({type(expected).__name__}): {exc}")
    if a_set == e_set:
        return (True, "")
    return (False, f"set mismatch: missing={e_set - a_set!r} extra={a_set - e_set!r}")


def grade(
    actual: Any,
    expected: Any,
    cmp: str = "exact",
    *,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-3,
) -> GradeResult:
    """Grade a single (actual, expected) pair using the named comparator."""
    if actual is None:
        return GradeResult(False, cmp, expected, actual, "no answer (None)")

    if cmp == "exact":
        passed, reason = _grade_exact(actual, expected)
    elif cmp == "near":
        passed, reason = _grade_near(actual, expected, abs_tol, rel_tol)
    elif cmp == "string":
        passed, reason = _grade_string(actual, expected)
    elif cmp == "set":
        passed, reason = _grade_set(actual, expected)
    else:
        return GradeResult(False, cmp, expected, actual, f"unknown comparator: {cmp!r}")

    return GradeResult(passed, cmp, expected, actual, reason)
