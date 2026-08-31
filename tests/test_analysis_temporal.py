from __future__ import annotations

import pytest

from fabric_rlm.experimental import (
    profile_time_coverage,
    select_latest_complete_window,
)
from fabric_rlm.experimental.analysis_contracts import AnalysisBrief


def test_analysis_brief_defaults_to_non_current_temporal_claims() -> None:
    brief = AnalysisBrief(objective="Find durable revenue patterns")

    assert brief.temporal_intent == "historical_context"
    assert brief.requested_as_of is None
    assert brief.recency_policy == "strict"
    assert brief.latest_complete_period_only is True
    assert brief.to_dict()["temporal_intent"] == "historical_context"


def test_analysis_brief_records_explicit_current_change_intent() -> None:
    brief = AnalysisBrief(
        objective="Explain the latest complete quarter's revenue change",
        temporal_intent="recent_change",
        requested_as_of="2026-08-31",
        recency_policy="strict",
        latest_complete_period_only=True,
    )

    assert brief.requested_as_of == "2026-08-31"
    assert brief.temporal_intent == "recent_change"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temporal_intent": "now"}, "temporal_intent"),
        ({"requested_as_of": "08/31/2026"}, "requested_as_of"),
        ({"recency_policy": "loose"}, "recency_policy"),
        ({"latest_complete_period_only": 1}, "latest_complete_period_only"),
    ],
)
def test_analysis_brief_rejects_ambiguous_temporal_policy(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        AnalysisBrief(objective="Find changes", **kwargs)


def test_profile_time_coverage_separates_event_and_trustworthy_watermarks() -> None:
    result = profile_time_coverage(
        node_id="orders-time-coverage",
        timestamps=(
            "2025-04-02T10:00:00Z",
            "2025-04-29T09:00:00Z",
            "2025-05-14T12:00:00Z",
            "2025-07-03T08:00:00Z",
            "2025-07-30T18:00:00Z",
            "2025-08-14T11:00:00Z",
        ),
        seed=17,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-07-31T23:59:59Z",
        timezone="UTC",
    )

    assert result.values["event_time_min"] == "2025-04-02T10:00:00Z"
    assert result.values["event_time_max"] == "2025-08-14T11:00:00Z"
    assert result.values["source_watermark"] == "2025-08-14T23:59:59Z"
    assert result.values["trustworthy_through"] == "2025-07-31T23:59:59Z"
    assert result.values["observed_periods"] == (
        "2025-04",
        "2025-05",
        "2025-07",
        "2025-08",
    )
    assert result.values["missing_periods"] == ("2025-06",)
    assert result.values["latest_complete_period"] == {
        "grain": "month",
        "start": "2025-07-01",
        "end": "2025-07-31",
    }
    assert result.values["partial_final_period"] == {
        "grain": "month",
        "start": "2025-08-01",
        "end": "2025-08-31",
    }
    assert result.diagnostics["watermarks"]["requested_as_of"] == "2025-08-15"


def test_profile_time_coverage_marks_stale_source_without_inventing_currentness() -> None:
    result = profile_time_coverage(
        node_id="stale-orders",
        timestamps=("2024-01-03", "2024-11-20", "2024-12-20"),
        seed=9,
        grain="month",
        requested_as_of="2026-08-31",
        source_watermark="2024-12-31T23:59:59Z",
        trustworthy_through="2024-11-30T23:59:59Z",
    )

    assert result.values["latest_complete_period"]["start"] == "2024-11-01"
    assert result.values["freshness_status"] == "stale"
    assert result.values["freshness_lag_days"] == 608


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"timestamps": ()}, "timestamps"),
        ({"timestamps": ("not-a-date",)}, "timestamps"),
        ({"timestamps": ("2025-01-01",), "grain": "hour"}, "grain"),
        (
            {
                "timestamps": ("2025-01-01",),
                "requested_as_of": "2025-01-31",
                "trustworthy_through": "2025-02-01",
            },
            "trustworthy_through",
        ),
    ],
)
def test_profile_time_coverage_rejects_invalid_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        profile_time_coverage(node_id="invalid", seed=1, **kwargs)


def test_latest_complete_window_excludes_partial_period_and_builds_yoy_comparator() -> None:
    coverage = profile_time_coverage(
        node_id="coverage",
        timestamps=(
            "2024-05-10",
            "2024-06-10",
            "2024-07-10",
            "2025-05-10",
            "2025-06-10",
            "2025-07-10",
            "2025-08-10",
        ),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-07-31T23:59:59Z",
    )

    result = select_latest_complete_window(
        node_id="window",
        coverage=coverage,
        seed=2,
        window_periods=3,
        comparator_kind="same_period_prior_year",
    )

    assert result.status == "completed"
    assert result.values["current_window"] == {
        "grain": "month",
        "start": "2025-05-01",
        "end": "2025-07-31",
        "periods": ("2025-05", "2025-06", "2025-07"),
    }
    assert result.values["comparator"] == {
        "kind": "same_period_prior_year",
        "start": "2024-05-01",
        "end": "2024-07-31",
        "periods": ("2024-05", "2024-06", "2024-07"),
    }
    assert result.values["excluded_partial_period"] == {
        "grain": "month",
        "start": "2025-08-01",
        "end": "2025-08-31",
    }


def test_latest_complete_window_abstains_when_current_periods_are_missing() -> None:
    coverage = profile_time_coverage(
        node_id="coverage",
        timestamps=("2025-05-10", "2025-07-10", "2025-08-10"),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-07-31T23:59:59Z",
    )

    result = select_latest_complete_window(
        node_id="window",
        coverage=coverage,
        seed=2,
        window_periods=3,
        comparator_kind="none",
    )

    assert result.status == "failed"
    assert result.failure_code == "incomplete_current_window"
    assert "2025-06" in result.failure_message


def test_latest_complete_window_abstains_without_comparable_history() -> None:
    coverage = profile_time_coverage(
        node_id="coverage",
        timestamps=("2025-05-10", "2025-06-10", "2025-07-10"),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-07-31T23:59:59Z",
    )

    result = select_latest_complete_window(
        node_id="window",
        coverage=coverage,
        seed=2,
        window_periods=3,
        comparator_kind="same_period_prior_year",
    )

    assert result.status == "failed"
    assert result.failure_code == "insufficient_comparable_history"
    assert result.limitations == (
        "Current evidence cannot support a recent-change claim without the "
        "requested comparable periods.",
    )


def test_daily_yoy_window_abstains_for_leap_day_without_exact_comparator() -> None:
    coverage = profile_time_coverage(
        node_id="coverage",
        timestamps=("2024-02-29",),
        seed=1,
        grain="day",
        requested_as_of="2024-03-01",
        source_watermark="2024-03-01T00:00:00Z",
        trustworthy_through="2024-02-29T23:59:59Z",
    )

    result = select_latest_complete_window(
        node_id="window",
        coverage=coverage,
        seed=2,
        window_periods=1,
        comparator_kind="same_period_prior_year",
    )

    assert result.status == "failed"
    assert result.failure_code == "non_comparable_calendar_period"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"window_periods": 0}, "window_periods"),
        ({"comparator_kind": "trend"}, "comparator_kind"),
        ({"coverage": {"latest_complete_period": None}}, "coverage"),
    ],
)
def test_latest_complete_window_rejects_invalid_contracts(
    kwargs: dict[str, object],
    match: str,
) -> None:
    defaults = {
        "coverage": profile_time_coverage(
            node_id="coverage",
            timestamps=("2025-07-10",),
            seed=1,
            grain="month",
            requested_as_of="2025-08-15",
            source_watermark="2025-08-14T23:59:59Z",
            trustworthy_through="2025-07-31T23:59:59Z",
        ),
        "window_periods": 1,
        "comparator_kind": "none",
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=match):
        select_latest_complete_window(
            node_id="invalid",
            seed=2,
            **defaults,
        )
