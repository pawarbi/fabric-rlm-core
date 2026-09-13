"""Tests for the noise-floor calibration arithmetic.

These are pure-function tests: no model, no network, no fixtures larger than a
literal. The point of the module is that an evaluation conclusion can be
re-derived and challenged, so its arithmetic has to be pinned.
"""

from __future__ import annotations

import pytest

from evaluation.calibration.noise import (
    Metric,
    Verdict,
    compare,
    flip_rate,
    spread,
    spread_kind,
)

ACCURACY = Metric(
    key="accuracy",
    label="Accuracy",
    unit="pp",
    absolute_floor=2.0,
    relative_floor=0.02,
    lower_is_better=False,
)
LATENCY = Metric(
    key="latency",
    label="Latency",
    unit="s",
    absolute_floor=1.0,
    relative_floor=0.10,
    lower_is_better=True,
)


# --- spread -------------------------------------------------------------


def test_spread_of_a_single_sample_is_zero_not_an_error() -> None:
    assert spread([42.0]) == 0.0
    assert spread([]) == 0.0


def test_spread_below_four_samples_uses_the_full_range() -> None:
    """A quartile of three points is false precision; the range is honest."""
    assert spread([10.0, 14.0]) == 4.0
    assert spread([10.0, 12.0, 14.0]) == 4.0
    assert spread_kind(3) == "range"


def test_spread_from_four_samples_uses_the_interquartile_range() -> None:
    assert spread_kind(4) == "iqr"
    assert spread([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.5)


def test_spread_ignores_a_single_outlier_once_there_are_enough_samples() -> None:
    tight = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7]
    assert spread(tight) < 1.0
    assert spread([*tight, 900.0]) < 1.0


# --- the three-part gate ------------------------------------------------


def test_a_change_inside_the_noise_is_not_reported() -> None:
    """The calibration thesis: same-config variation must not read as a result."""
    noisy = [80.0, 92.0, 86.0]
    result = compare(ACCURACY, noisy, [88.0, 84.0, 90.0])

    assert result.verdict is Verdict.NO_CLEAR_CHANGE
    assert not result.is_significant


def test_a_change_clearing_all_three_floors_is_reported() -> None:
    result = compare(ACCURACY, [70.0, 70.5, 70.2], [90.0, 90.4, 90.1])

    assert result.verdict is Verdict.IMPROVEMENT
    assert result.is_significant
    assert result.delta == pytest.approx(19.9)


def test_a_change_below_the_absolute_floor_is_not_reported() -> None:
    result = compare(ACCURACY, [70.0, 70.0, 70.0], [71.0, 71.0, 71.0])

    assert result.verdict is Verdict.NO_CLEAR_CHANGE


def test_a_change_below_the_relative_floor_is_not_reported() -> None:
    """A 2.5 point move on a 1000 point baseline is not a finding."""
    result = compare(ACCURACY, [1000.0] * 3, [1002.5] * 3)

    assert result.verdict is Verdict.NO_CLEAR_CHANGE


def test_direction_respects_lower_is_better() -> None:
    slower = compare(LATENCY, [10.0] * 3, [30.0] * 3)
    assert slower.verdict is Verdict.REGRESSION

    more_accurate = compare(ACCURACY, [10.0] * 3, [30.0] * 3)
    assert more_accurate.verdict is Verdict.IMPROVEMENT


def test_an_empty_side_is_incomplete_never_a_result() -> None:
    assert compare(ACCURACY, [], [90.0]).verdict is Verdict.INCOMPLETE
    assert compare(ACCURACY, [90.0], []).verdict is Verdict.INCOMPLETE


def test_a_missing_repetition_is_incomplete_not_a_result() -> None:
    """A dropped trial must not silently become a smaller, cleaner sample."""
    result = compare(
        ACCURACY, [70.0, 70.0, 70.0], [95.0, 95.0], expected_samples=3
    )

    assert result.verdict is Verdict.INCOMPLETE
    assert not result.is_significant


# --- calibrated floor ---------------------------------------------------


def test_a_calibrated_floor_overrides_the_within_comparison_spread() -> None:
    """Two tight-but-different arms look significant until noise is measured.

    Each arm is internally consistent, so their own spread is ~0 and the change
    clears the gate. A same-versus-same run that showed 25 points of drift says
    otherwise, and that is the measurement the gate must honour.
    """
    baseline, candidate = [70.0, 70.0, 70.0], [90.0, 90.0, 90.0]

    assert compare(ACCURACY, baseline, candidate).verdict is Verdict.IMPROVEMENT

    calibrated = compare(
        ACCURACY, baseline, candidate, calibrated_floor=25.0
    )
    assert calibrated.verdict is Verdict.NO_CLEAR_CHANGE
    assert calibrated.observed_spread == 25.0


def test_a_calibrated_floor_of_zero_is_honoured_not_treated_as_absent() -> None:
    """A genuinely noiseless baseline must not fall back to the spread."""
    result = compare(
        ACCURACY, [70.0, 90.0], [95.0, 95.0], calibrated_floor=0.0
    )

    assert result.observed_spread == 0.0
    assert result.verdict is Verdict.IMPROVEMENT


# --- flip rate ----------------------------------------------------------


def test_flip_rate_counts_items_that_disagree_with_themselves() -> None:
    outcomes = {
        "q1": [True, True, True],
        "q2": [True, False, True],
        "q3": [False, False, False],
        "q4": [False, True, False],
    }

    assert flip_rate(outcomes) == pytest.approx(0.5)


def test_flip_rate_ignores_items_with_a_single_run() -> None:
    """One run cannot disagree with itself, and must not dilute the rate."""
    assert flip_rate({"q1": [True], "q2": [True, False]}) == pytest.approx(1.0)
    assert flip_rate({"q1": [True]}) == 0.0
    assert flip_rate({}) == 0.0


# --- metric validation --------------------------------------------------


def test_a_negative_floor_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Metric("k", "K", "u", -1.0, 0.1)
    with pytest.raises(ValueError, match="non-negative"):
        Metric("k", "K", "u", 1.0, -0.1)
