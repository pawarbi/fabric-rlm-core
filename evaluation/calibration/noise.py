"""Noise-floor calibration for evaluation metrics.

A measured difference between two configurations only means something once we
know how much the *same* configuration varies against itself. This module keeps
that question separate from any model call: everything here is pure arithmetic
over already-collected trial records, so a conclusion can be recomputed and
argued with long after the runs are gone.

The gate follows the one Prime Intellect use for their PR benchmarks: a change
is reported only when it clears an absolute floor, a relative floor, *and* the
observed spread of the samples. Any one of those alone is easy to fool.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Metric",
    "Verdict",
    "Comparison",
    "spread",
    "spread_kind",
    "compare",
    "flip_rate",
]


class Verdict(str, Enum):
    """What a comparison is entitled to claim."""

    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    NO_CLEAR_CHANGE = "no_clear_change"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class Metric:
    """A measurable quantity and the smallest change worth reporting.

    ``absolute_floor`` is in the metric's own units and encodes the smallest
    difference that matters in practice. ``relative_floor`` is a fraction of the
    baseline and guards against large absolute swings on small baselines.
    ``lower_is_better`` decides which direction counts as a regression.
    """

    key: str
    label: str
    unit: str
    absolute_floor: float
    relative_floor: float
    lower_is_better: bool = True

    def __post_init__(self) -> None:
        if self.absolute_floor < 0 or self.relative_floor < 0:
            raise ValueError(f"{self.key}: floors must be non-negative")


@dataclass(frozen=True)
class Comparison:
    """The result of holding a candidate against a baseline."""

    metric: Metric
    baseline: float | None
    candidate: float | None
    delta: float | None
    relative_change: float | None
    observed_spread: float
    verdict: Verdict
    baseline_n: int
    candidate_n: int

    @property
    def is_significant(self) -> bool:
        return self.verdict in (Verdict.REGRESSION, Verdict.IMPROVEMENT)

    def describe(self) -> str:
        if self.baseline is None or self.candidate is None:
            return f"{self.metric.label}: unavailable"
        arrow = {
            Verdict.REGRESSION: "^",
            Verdict.IMPROVEMENT: "v",
            Verdict.NO_CLEAR_CHANGE: "~",
            Verdict.INCOMPLETE: "?",
        }[self.verdict]
        percent = (
            "n/a"
            if self.relative_change is None
            else f"{self.relative_change * 100:+.1f}%"
        )
        return (
            f"{self.metric.label}: {self.baseline:.4g} -> {self.candidate:.4g} "
            f"{self.metric.unit} {arrow} {self.delta:+.4g} ({percent}), "
            f"noise {self.observed_spread:.4g}"
        )


def spread_kind(sample_count: int) -> str:
    """Which dispersion measure ``spread`` will use for this many samples."""
    if sample_count < 2:
        return "none"
    return "iqr" if sample_count >= 4 else "range"


def spread(values: list[float]) -> float:
    """Dispersion of ``values``, robust at the sample sizes evaluations reach.

    Below four samples a quartile is not meaningful, so the full range is used
    instead. That is deliberately pessimistic: with n=2 or n=3 the honest
    statement is that the noise could be as wide as everything we have seen.
    """
    if len(values) < 2:
        return 0.0
    if len(values) < 4:
        return max(values) - min(values)
    first, _, third = statistics.quantiles(values, n=4, method="inclusive")
    return third - first


def compare(
    metric: Metric,
    baseline: list[float],
    candidate: list[float],
    *,
    expected_samples: int | None = None,
    calibrated_floor: float | None = None,
) -> Comparison:
    """Hold ``candidate`` against ``baseline`` under the three-part gate.

    ``calibrated_floor`` is the noise floor measured by running the baseline
    configuration against itself. When supplied it replaces the within-comparison
    spread, which is the entire point of calibration: the spread of two runs of
    two *different* configurations conflates their difference with their noise,
    while a same-versus-same run measures the noise alone.
    """
    baseline_n, candidate_n = len(baseline), len(candidate)
    if not baseline or not candidate:
        return Comparison(
            metric, None, None, None, None, 0.0,
            Verdict.INCOMPLETE, baseline_n, candidate_n,
        )

    baseline_value = statistics.median(baseline)
    candidate_value = statistics.median(candidate)
    delta = candidate_value - baseline_value
    relative_change = delta / baseline_value if baseline_value else None

    observed = (
        calibrated_floor
        if calibrated_floor is not None
        else max(spread(baseline), spread(candidate))
    )

    incomplete = expected_samples is not None and (
        baseline_n != expected_samples or candidate_n != expected_samples
    )

    clears_absolute = abs(delta) > metric.absolute_floor
    clears_relative = (
        relative_change is not None
        and abs(relative_change) > metric.relative_floor
    )
    clears_noise = abs(delta) > observed

    if incomplete:
        verdict = Verdict.INCOMPLETE
    elif clears_absolute and clears_relative and clears_noise:
        worse = delta > 0 if metric.lower_is_better else delta < 0
        verdict = Verdict.REGRESSION if worse else Verdict.IMPROVEMENT
    else:
        verdict = Verdict.NO_CLEAR_CHANGE

    return Comparison(
        metric,
        baseline_value,
        candidate_value,
        delta,
        relative_change,
        observed,
        verdict,
        baseline_n,
        candidate_n,
    )


def flip_rate(outcomes_by_item: dict[str, list[bool]]) -> float:
    """Fraction of items whose repeated identical runs disagree with each other.

    This is the most direct statement of instability available: it needs no
    baseline and no threshold. An item that is sometimes right and sometimes
    wrong under an unchanged configuration has no single accuracy to report.
    """
    comparable = [
        outcomes for outcomes in outcomes_by_item.values() if len(outcomes) >= 2
    ]
    if not comparable:
        return 0.0
    flipped = sum(1 for outcomes in comparable if len(set(outcomes)) > 1)
    return flipped / len(comparable)
