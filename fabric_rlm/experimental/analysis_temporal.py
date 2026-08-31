"""Deterministic temporal relevance classification for analytical evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from fabric_rlm.experimental.analysis_contracts import (
    AnalysisBrief,
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
_TEMPORAL_STATUSES = {
    "current_change",
    "current_level",
    "persistent",
    "recurring_seasonal",
    "historical",
    "stale",
    "not_applicable",
}
_CURRENT_ACTION_STATUSES = {"current_change", "current_level", "persistent"}


@dataclass(frozen=True)
class TemporalAssessment:
    """Temporal claim status and the evidence context supporting it."""

    status: TemporalStatus
    supports_current_action: bool
    reason: str
    context: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.status not in _TEMPORAL_STATUSES:
            raise ValueError("status is invalid")
        if type(self.supports_current_action) is not bool:
            raise ValueError("supports_current_action must be boolean")
        if self.supports_current_action != (
            self.status in _CURRENT_ACTION_STATUSES
        ):
            raise ValueError(
                "supports_current_action contradicts temporal status"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be non-empty text")
        if not isinstance(self.context, Mapping):
            raise ValueError("context must be a mapping")
        object.__setattr__(self, "reason", self.reason.strip())
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


def _period_evidence(
    value: object,
    field_name: str,
    observed_periods: set[str],
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{field_name} must be a sequence of observed period labels"
        )
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"{field_name}[{index}] must be a non-empty period label"
            )
        label = item.strip()
        if label not in observed_periods:
            raise ValueError(
                f"{field_name}[{index}] is not present in coverage evidence"
            )
        normalized.append(label)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return tuple(normalized)


def _seasonal_evidence(
    value: object,
    observed_periods: set[str],
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "seasonal_cycles must be a sequence of observed-period sequences"
        )
    cycles: list[tuple[str, ...]] = []
    for index, cycle in enumerate(value):
        labels = _period_evidence(
            cycle,
            f"seasonal_cycles[{index}]",
            observed_periods,
        )
        if not labels:
            raise ValueError(f"seasonal_cycles[{index}] must not be empty")
        cycles.append(labels)
    return tuple(cycles)


def classify_temporal_relevance(
    *,
    temporal_intent: str | None = None,
    brief: AnalysisBrief | None = None,
    coverage: OperatorResult,
    time_basis: str | None,
    window: OperatorResult | None = None,
    has_change_evidence: bool = False,
    persistence_periods: object = (),
    seasonal_cycles: object = (),
    recency_policy: str = "strict",
    latest_complete_period_only: bool = True,
) -> TemporalAssessment:
    """Classify whether evidence is current, historical, persistent, or stale."""

    if brief is not None:
        if not isinstance(brief, AnalysisBrief):
            raise ValueError("brief must be an AnalysisBrief")
        if temporal_intent is not None and temporal_intent != brief.temporal_intent:
            raise ValueError("temporal_intent conflicts with brief")
        temporal_intent = brief.temporal_intent
        recency_policy = brief.recency_policy
        latest_complete_period_only = brief.latest_complete_period_only
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
    if recency_policy not in {"strict", "allow_historical"}:
        raise ValueError("recency_policy must be strict or allow_historical")
    if type(latest_complete_period_only) is not bool:
        raise ValueError("latest_complete_period_only must be boolean")
    observed_periods = set(coverage.values.get("observed_periods", ()))
    persistence_periods = _period_evidence(
        persistence_periods,
        "persistence_periods",
        observed_periods,
    )
    seasonal_cycles = _seasonal_evidence(
        seasonal_cycles,
        observed_periods,
    )
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
        "evidence_fingerprints": {
            "coverage": coverage.diagnostics.get("input_fingerprint"),
            "window": (
                window.diagnostics.get("input_fingerprint")
                if window is not None
                else None
            ),
        },
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
    if (
        temporal_intent in {"current_state", "recent_change"}
        and (
            coverage.values.get("freshness_status") != "current"
            or not latest_complete_period_only
        )
    ):
        return TemporalAssessment(
            status="historical",
            supports_current_action=False,
            reason=(
                "The configured recency and complete-period policy does not "
                "permit this evidence to be classified as current."
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
    if len(seasonal_cycles) >= 2:
        return TemporalAssessment(
            status="recurring_seasonal",
            supports_current_action=False,
            reason=(
                "The pattern recurs across at least two comparable seasonal "
                "cycles but does not by itself establish a current change."
            ),
            context=context,
        )
    if len(persistence_periods) >= 3:
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
