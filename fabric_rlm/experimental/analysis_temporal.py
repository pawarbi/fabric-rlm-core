"""Deterministic temporal relevance classification for analytical evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from fabric_rlm.experimental.analysis_contracts import (
    OperatorResult,
    _freeze_json,
    _thaw_json,
)


TemporalStatus = Literal[
    "current_change",
    "current_level",
    "persistent",
    "recurring_seasonal",
    "historical",
    "stale",
    "not_applicable",
]

_TEMPORAL_INTENTS = {
    "current_state",
    "recent_change",
    "historical_context",
    "structural_pattern",
}


@dataclass(frozen=True)
class TemporalAssessment:
    """Temporal claim status and the evidence context supporting it."""

    status: TemporalStatus
    supports_current_action: bool
    reason: str
    context: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "context",
            _freeze_json(self.context, "context"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "supports_current_action": self.supports_current_action,
            "reason": self.reason,
            "context": _thaw_json(self.context),
        }


def _nonnegative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def classify_temporal_relevance(
    *,
    temporal_intent: str,
    coverage: OperatorResult,
    time_basis: str | None,
    window: OperatorResult | None = None,
    has_change_evidence: bool = False,
    persistence_periods: int = 0,
    seasonal_cycles: int = 0,
) -> TemporalAssessment:
    """Classify whether evidence is current, historical, persistent, or stale."""

    if temporal_intent not in _TEMPORAL_INTENTS:
        raise ValueError(
            "temporal_intent must be current_state, recent_change, "
            "historical_context, or structural_pattern"
        )
    if (
        not isinstance(coverage, OperatorResult)
        or coverage.operator not in {
            "profile_time_coverage.v1",
            "profile_joined_time_coverage.v1",
        }
        or coverage.status != "completed"
    ):
        raise ValueError(
            "coverage must be a completed temporal coverage result"
        )
    if window is not None and (
        not isinstance(window, OperatorResult)
        or window.operator != "select_latest_complete_window.v1"
    ):
        raise ValueError(
            "window must be a select_latest_complete_window.v1 result"
        )
    if type(has_change_evidence) is not bool:
        raise ValueError("has_change_evidence must be boolean")
    persistence_periods = _nonnegative_int(
        persistence_periods,
        "persistence_periods",
    )
    seasonal_cycles = _nonnegative_int(seasonal_cycles, "seasonal_cycles")
    if time_basis is not None and (
        not isinstance(time_basis, str) or not time_basis.strip()
    ):
        raise ValueError("time_basis must be non-empty text or None")

    watermarks = coverage.diagnostics.get("watermarks", {})
    current_window = (
        window.values.get("current_window")
        if window is not None and window.status == "completed"
        else None
    )
    comparator = (
        window.values.get("comparator")
        if window is not None and window.status == "completed"
        else None
    )
    context = {
        "time_basis": time_basis.strip() if isinstance(time_basis, str) else None,
        "timezone": watermarks.get("timezone"),
        "requested_as_of": watermarks.get("requested_as_of"),
        "data_as_of": coverage.values.get("source_watermark"),
        "trustworthy_through": coverage.values.get("trustworthy_through"),
        "latest_complete_period": coverage.values.get(
            "latest_complete_period"
        ),
        "current_window": current_window,
        "comparators": (comparator,) if isinstance(comparator, Mapping) else (),
        "partial_period_policy": "exclude",
        "completeness_basis": (
            "calendar_complete_and_source_marked_trustworthy"
        ),
        "recency_status": coverage.values.get("freshness_status"),
    }

    if time_basis is None:
        return TemporalAssessment(
            status="not_applicable",
            supports_current_action=False,
            reason="No event-time basis was declared for this finding.",
            context=context,
        )
    if coverage.values.get("freshness_status") in {"stale", "mismatched"}:
        return TemporalAssessment(
            status="stale",
            supports_current_action=False,
            reason=(
                "Source freshness is stale or inconsistent across required "
                "inputs, so it cannot support a current claim."
            ),
            context=context,
        )
    if temporal_intent == "historical_context":
        return TemporalAssessment(
            status="historical",
            supports_current_action=False,
            reason=(
                "The analysis intent permits historical context but does not "
                "establish current operating conditions."
            ),
            context=context,
        )
    if temporal_intent == "current_state":
        if coverage.values.get("latest_complete_period") is None:
            return TemporalAssessment(
                status="historical",
                supports_current_action=False,
                reason="No complete trustworthy period supports a current level.",
                context=context,
            )
        return TemporalAssessment(
            status="current_level",
            supports_current_action=True,
            reason="A recent trustworthy complete period supports the level claim.",
            context=context,
        )
    if temporal_intent == "recent_change":
        if (
            window is None
            or window.status != "completed"
            or not isinstance(comparator, Mapping)
            or not has_change_evidence
        ):
            return TemporalAssessment(
                status="historical",
                supports_current_action=False,
                reason=(
                    "Recent change requires a complete current window, "
                    "comparable history, and measured change evidence."
                ),
                context=context,
            )
        return TemporalAssessment(
            status="current_change",
            supports_current_action=True,
            reason=(
                "Measured change reconciles a complete current window with "
                "comparable history."
            ),
            context=context,
        )
    if seasonal_cycles >= 2:
        return TemporalAssessment(
            status="recurring_seasonal",
            supports_current_action=False,
            reason=(
                "The pattern recurs across at least two comparable seasonal "
                "cycles but does not by itself establish a current change."
            ),
            context=context,
        )
    if persistence_periods >= 3:
        return TemporalAssessment(
            status="persistent",
            supports_current_action=True,
            reason=(
                "The pattern persists across at least three trustworthy "
                "complete periods."
            ),
            context=context,
        )
    return TemporalAssessment(
        status="historical",
        supports_current_action=False,
        reason=(
            "The available history does not meet the deterministic persistence "
            "or recurrence threshold."
        ),
        context=context,
    )
