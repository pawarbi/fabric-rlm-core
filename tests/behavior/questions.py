"""Pinned question set for behavior CI.

Five stratified questions (2 compute + 2 messy + 1 selfcorrect) selected from the
n=30 dspy-vs-fabric parity set (see session ec8cf4f1.../files/dspy_vs_fabric/
ab_n30.py).  Picked because gpt-4.1-mini passed them reliably in the parity run
AND they exercise different reasoning patterns:

    C1_sum_squares_mod        - deterministic numeric reduction
    C5_palindrome_count       - simple iteration + character indexing
    M1_optional_field_count   - dict with optional keys (defensive access)
    M3_mixed_string_sum       - parse-or-skip mixed input
    S5_dedup_preserve_order   - naive trap (set() loses order)

All ground-truth values are computed locally below (no memorization risk).
The inputs are seeded deterministically so the suite is reproducible.

To add a question: append to QUESTIONS, run calibration to confirm gpt-4.1-mini
passes >=4/5 runs, then commit the new baseline.
"""

from __future__ import annotations

import hashlib
import inspect
import random
import string
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Question:
    qid: str
    category: str  # "compute" | "messy" | "selfcorrect"
    task: str
    inputs: dict[str, Any] = field(default_factory=dict)
    expected: Any = None
    cmp: str = "exact"  # see grader.py


def _seeded_words(seed: int, n: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append("".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8))))
    return out


def _seeded_ints(seed: int, n: int, lo: int = 1, hi: int = 100) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(lo, hi) for _ in range(n)]


def _build_questions() -> list[Question]:
    qs: list[Question] = []

    # C1 - sum of squares mod 9973 over 200 deterministic ints
    nums1 = _seeded_ints(seed=11, n=200, lo=1, hi=999)
    qs.append(
        Question(
            qid="C1_sum_squares_mod",
            category="compute",
            task=(
                "Given a list of integers, compute the sum of x*x for each x, "
                "then return the result modulo 9973."
            ),
            inputs={"numbers": nums1},
            expected=sum(x * x for x in nums1) % 9973,
        )
    )

    # C5 - count strings whose first char equals last char
    words5 = _seeded_words(seed=55, n=500)
    qs.append(
        Question(
            qid="C5_palindrome_count",
            category="compute",
            task=(
                "Given a list of lowercase strings, return the count of strings "
                "whose first character equals their last character."
            ),
            inputs={"words": words5},
            expected=sum(1 for w in words5 if w and w[0] == w[-1]),
        )
    )

    # M1 - optional 'score' field, count where present and > 50
    rng_m1 = random.Random(2025)
    items_m1: list[dict[str, Any]] = []
    for i in range(80):
        d: dict[str, Any] = {"id": i}
        if rng_m1.random() < 0.7:
            d["score"] = rng_m1.randint(0, 100)
        if rng_m1.random() < 0.5:
            d["name"] = "".join(rng_m1.choices(string.ascii_lowercase, k=5))
        items_m1.append(d)
    qs.append(
        Question(
            qid="M1_optional_field_count",
            category="messy",
            task=(
                "Given a list of dicts where each may or may not have a 'score' "
                "key, return the integer count of dicts where 'score' is present "
                "AND > 50."
            ),
            inputs={"items": items_m1},
            expected=sum(1 for d in items_m1 if "score" in d and d["score"] > 50),
        )
    )

    # M3 - sum the numeric items in a mixed list, return integer floor
    rng_m3 = random.Random(2025 + 3)
    items_m3: list[str] = []
    expected_m3 = 0.0
    for _ in range(60):
        choice = rng_m3.random()
        if choice < 0.3:
            v = rng_m3.randint(1, 100)
            items_m3.append(str(v))
            expected_m3 += v
        elif choice < 0.5:
            v = round(rng_m3.random() * 100, 2)
            items_m3.append(str(v))
            expected_m3 += v
        elif choice < 0.7:
            items_m3.append("".join(rng_m3.choices(string.ascii_letters, k=5)))
        else:
            items_m3.append(f"item-{rng_m3.randint(1, 99)}")
    qs.append(
        Question(
            qid="M3_mixed_string_sum",
            category="messy",
            task=(
                "Given a list of strings, some are numeric (int or float), others "
                "are non-numeric. Sum the numeric ones (parsing as float when "
                "needed) and return the integer floor of the total."
            ),
            inputs={"items": items_m3},
            expected=int(expected_m3),
        )
    )

    # S5 - dedup preserving order; naive set() loses order
    src_s5 = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    seen: set[int] = set()
    expected_s5: list[int] = []
    for x in src_s5:
        if x not in seen:
            seen.add(x)
            expected_s5.append(x)
    qs.append(
        Question(
            qid="S5_dedup_preserve_order",
            category="selfcorrect",
            task=(
                "Given a list of integers with duplicates, return a list of "
                "unique integers in the SAME ORDER as their first appearance. "
                "(Note: using set() loses order; you must preserve insertion "
                "order.)"
            ),
            inputs={"numbers": src_s5},
            expected=expected_s5,
            cmp="string",  # repr-based: order matters
        )
    )

    return qs


QUESTIONS: list[Question] = _build_questions()


def questions_sha256() -> str:
    """SHA-256 of this module's source (without the cached hash itself).

    Used in baselines.json to detect when the suite changes without a recalibration.
    Source-based (not data-based) so adding a new question or changing ``task``
    text invalidates the baseline; pure-formatting changes also invalidate but
    that's an acceptable false positive.
    """
    src = inspect.getsource(_build_questions).encode("utf-8")
    src += inspect.getsource(_seeded_words).encode("utf-8")
    src += inspect.getsource(_seeded_ints).encode("utf-8")
    return hashlib.sha256(src).hexdigest()


def get_question(qid: str) -> Question:
    for q in QUESTIONS:
        if q.qid == qid:
            return q
    raise KeyError(f"unknown qid: {qid}")
