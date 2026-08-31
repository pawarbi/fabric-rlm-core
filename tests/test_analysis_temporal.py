from __future__ import annotations

import pytest

from fabric_rlm.experimental import profile_time_coverage
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
