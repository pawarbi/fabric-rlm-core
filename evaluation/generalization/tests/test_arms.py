"""Tests for arm configuration in the generalization runner.

The original three arms confounded the thing being measured. Arms B and C
received a knowledge package *and no data sources at all*, while arm A
received sources and no package. Any difference between them therefore mixes
two independent changes, and the natural reading -- "learning hurts" -- is not
something that design can support. It can only support the much narrower
"a knowledge package is not a substitute for data access".

These tests pin the arm definitions so the distinction cannot quietly collapse
again, and cover the added arms BS and CS, which hold source access fixed and
vary only the knowledge package.
"""
from __future__ import annotations

import pytest

from evaluation.generalization.runner import (
    ALL_ARMS,
    ARMS,
    arm_knowledge_kind,
    arm_uses_sources,
    build_schedule,
)


def test_default_arms_unchanged_for_reproducibility() -> None:
    assert ARMS == ("A", "B", "C")


def test_added_arms_are_available() -> None:
    assert set(ALL_ARMS) == {"A", "B", "C", "BS", "CS"}


@pytest.mark.parametrize("arm,expected", [
    ("A", True),    # sources, no package
    ("B", False),   # package only -- the original, confounded arm
    ("C", False),   # package only
    ("BS", True),   # package AND sources
    ("CS", True),   # enriched package AND sources
])
def test_which_arms_receive_sources(arm: str, expected: bool) -> None:
    assert arm_uses_sources(arm) is expected


@pytest.mark.parametrize("arm,expected", [
    ("A", None),
    ("B", "learned"),
    ("BS", "learned"),
    ("C", "enriched"),
    ("CS", "enriched"),
])
def test_which_package_each_arm_receives(arm: str, expected: str | None) -> None:
    assert arm_knowledge_kind(arm) == expected


def test_bs_differs_from_a_only_by_the_package() -> None:
    """The comparison the evaluation actually wants: same data, one variable."""
    assert arm_uses_sources("A") == arm_uses_sources("BS")
    assert arm_knowledge_kind("A") is None
    assert arm_knowledge_kind("BS") == "learned"


def test_cs_differs_from_bs_only_by_enrichment() -> None:
    assert arm_uses_sources("BS") == arm_uses_sources("CS")
    assert arm_knowledge_kind("BS") == "learned"
    assert arm_knowledge_kind("CS") == "enriched"


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(ValueError):
        arm_uses_sources("Z")
    with pytest.raises(ValueError):
        arm_knowledge_kind("Z")


# ---------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------
def _questions() -> list[dict[str, object]]:
    return [
        {"question_id": "q1", "domain": "inventory", "variant": "descriptive"},
        {"question_id": "q2", "domain": "service", "variant": "descriptive"},
    ]


def test_schedule_defaults_to_the_original_arms() -> None:
    schedule = build_schedule(_questions(), repetitions=1, seed=1)
    assert {row["arm"] for row in schedule} == {"A", "B", "C"}


def test_schedule_accepts_explicit_arms() -> None:
    schedule = build_schedule(_questions(), repetitions=2, seed=1,
                              arms=("A", "BS", "CS"))
    assert {row["arm"] for row in schedule} == {"A", "BS", "CS"}
    assert len(schedule) == 2 * 2 * 3


def test_schedule_is_seeded_and_reproducible() -> None:
    a = build_schedule(_questions(), repetitions=2, seed=7, arms=("A", "BS"))
    b = build_schedule(_questions(), repetitions=2, seed=7, arms=("A", "BS"))
    assert a == b


def test_every_question_gets_every_arm_in_every_repetition() -> None:
    schedule = build_schedule(_questions(), repetitions=3, seed=3,
                              arms=("A", "BS", "CS"))
    seen = {(row["question_id"], row["repetition"], row["arm"]) for row in schedule}
    assert len(seen) == 2 * 3 * 3
