"""Answers that the run did not derive.

Built from a real leaderboard audit. Four trials reported an exact ground-truth
value having either never queried the data or never derived the figure; three
submitted on their first turn in two milliseconds. The cases below are those
traces reduced to their essentials, so a regression here means the check would
stop catching something it has already caught in the wild.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fabric_rlm.grounding import (
    evidence_report,
    numbers_in,
    observed_numbers,
    submitted_without_evidence,
    ungrounded_figures,
)


@dataclass
class Turn:
    stdout: str = ""
    stderr: str = ""


# -- the case this exists for ----------------------------------------------


def test_a_first_turn_submit_with_no_output_is_caught():
    """agnews q3 r2/r3/r4: one turn, no output, answer 336.6363... which is
    exactly the ground truth 3703/11. Not a heuristic - a run that printed
    nothing cannot have derived anything."""
    turns = [Turn()]
    assert submitted_without_evidence(turns) is True
    assert ungrounded_figures(turns, {"answer": "The average is 336.6363636"}) == [
        336.6364
    ]


def test_a_figure_never_printed_is_flagged_even_after_real_work():
    """agnews q3 r5: eleven turns of genuine querying, and the reported value
    still appears from nowhere. Work happening is not evidence that this
    number came from it."""
    turns = [Turn(stdout="rows: 120000\ncategories: 4"), Turn(stdout="europe: 88214")]
    assert submitted_without_evidence(turns) is False
    assert 3703.0 in ungrounded_figures(turns, {"answer": "3703.0 over 11.0 years"})


def test_a_derived_average_is_not_flagged():
    """The run printed the numerator and the denominator. Writing the quotient
    is arithmetic, not invention."""
    turns = [Turn(stdout="business articles: 3703.0\nyears: 11.0")]
    assert ungrounded_figures(turns, {"answer": "336.6363636 per year"}) == []


def test_a_printed_figure_is_never_flagged():
    turns = [Turn(stdout="total ARR 18118056834.46")]
    assert ungrounded_figures(turns, {"answer": "Total is 18118056834.46"}) == []


# -- not flagging things that are fine --------------------------------------


def test_years_are_not_figures():
    turns = [Turn(stdout="loaded")]
    assert ungrounded_figures(turns, {"answer": "from 2010 to 2020 inclusive"}) == []


def test_small_counts_are_not_figures():
    """Thresholds, list lengths and loop bounds are not findings."""
    turns = [Turn(stdout="ok")]
    assert ungrounded_figures(turns, {"answer": "across 8 lines, 3 shifts"}) == [3.0, 8.0]
    assert ungrounded_figures(turns, {"answer": "1 of them"}, ignore_below=10) == []


def test_stderr_counts_as_having_been_seen():
    turns = [Turn(stderr="ValueError: expected 4096 rows")]
    assert ungrounded_figures(turns, {"answer": "4096 rows"}) == []
    assert submitted_without_evidence(turns) is False


def test_arithmetic_can_be_switched_off_for_a_stricter_read():
    turns = [Turn(stdout="3703.0 and 11.0")]
    assert ungrounded_figures(turns, {"answer": "336.6363636"}) == []
    assert ungrounded_figures(turns, {"answer": "336.6363636"},
                              allow_arithmetic=False) == [336.6364]


# -- shape ------------------------------------------------------------------


def test_numbers_are_rounded_so_precision_does_not_matter():
    assert numbers_in("3.14159") == numbers_in("3.1416")


def test_thousands_separators_are_understood():
    assert 1234567.0 in numbers_in("we saw 1,234,567 rows")


def test_observed_numbers_spans_every_turn():
    turns = [Turn(stdout="1000"), Turn(stdout="2000")]
    assert {1000.0, 2000.0} <= observed_numbers(turns)


def test_evidence_report_carries_both_checks():
    turns = [Turn()]
    rep = evidence_report(turns, {"answer": "336.6363636"})
    assert rep["submitted_without_evidence"] is True
    assert rep["ungrounded_figures"] == [336.6364]
    assert rep["observed_figures"] == 0


@pytest.mark.parametrize("payload", [
    {"answer": "336.6363636"},
    "336.6363636",
    {"answer": "336.6363636", "note": "see above"},
])
def test_payload_may_be_a_string_or_a_dict(payload):
    assert ungrounded_figures([Turn()], payload) == [336.6364]
