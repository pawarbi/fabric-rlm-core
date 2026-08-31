from __future__ import annotations

import pytest

from fabric_rlm.experimental import (
    AnalysisBrief,
    TemporalAssessment,
    assess_cohort_exposure,
    classify_temporal_relevance,
    combine_time_coverage,
    profile_time_coverage,
    select_latest_complete_window,
)


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
        trustworthy_through="2025-08-01T00:00:00Z",
        timezone="UTC",
    )

    assert result.values["event_time_min"] == "2025-04-02T10:00:00Z"
    assert result.values["event_time_max"] == "2025-08-14T11:00:00Z"
    assert result.values["source_watermark"] == "2025-08-14T23:59:59Z"
    assert result.values["trustworthy_through"] == "2025-08-01T00:00:00Z"
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
        trustworthy_through="2024-12-01T00:00:00Z",
    )

    assert result.values["latest_complete_period"]["start"] == "2024-11-01"
    assert result.values["freshness_status"] == "stale"
    assert result.values["freshness_lag_days"] == 608


def test_profile_time_coverage_buckets_periods_in_source_timezone() -> None:
    result = profile_time_coverage(
        node_id="local-calendar",
        timestamps=("2025-03-01T00:30:00Z",),
        seed=1,
        grain="month",
        requested_as_of="2025-03-02",
        source_watermark="2025-03-01T08:00:00Z",
        trustworthy_through="2025-03-01T08:00:00Z",
        timezone="America/Los_Angeles",
    )

    assert result.values["observed_periods"] == ("2025-02",)


@pytest.mark.parametrize(
    ("timestamp", "timezone", "expected_day"),
    [
        ("2025-02-28T20:00:00Z", "Asia/Kolkata", "2025-03-01"),
        ("2025-03-09T07:30:00Z", "America/Los_Angeles", "2025-03-08"),
        ("2025-11-02T07:30:00Z", "America/Los_Angeles", "2025-11-02"),
    ],
)
def test_daily_period_bucketing_handles_offsets_and_dst(
    timestamp: str,
    timezone: str,
    expected_day: str,
) -> None:
    result = profile_time_coverage(
        node_id="local-day",
        timestamps=(timestamp,),
        seed=1,
        grain="day",
        requested_as_of="2025-12-31",
        source_watermark=timestamp,
        trustworthy_through=timestamp,
        timezone=timezone,
    )

    assert result.values["observed_periods"] == (expected_day,)


def test_period_is_complete_only_at_exclusive_next_period_boundary() -> None:
    incomplete = profile_time_coverage(
        node_id="incomplete",
        timestamps=("2025-06-10", "2025-07-10"),
        seed=1,
        grain="month",
        requested_as_of="2025-08-01",
        source_watermark="2025-07-31T00:00:00Z",
        trustworthy_through="2025-07-31T00:00:00Z",
    )
    complete = profile_time_coverage(
        node_id="complete",
        timestamps=("2025-06-10", "2025-07-10"),
        seed=1,
        grain="month",
        requested_as_of="2025-08-01",
        source_watermark="2025-08-01T00:00:00Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )

    assert incomplete.values["latest_complete_period"]["start"] == "2025-06-01"
    assert complete.values["latest_complete_period"]["start"] == "2025-07-01"


@pytest.mark.parametrize(
    ("grain", "timestamp", "before_boundary", "boundary"),
    [
        ("day", "2025-07-10T12:00:00Z", "2025-07-10T23:59:59Z", "2025-07-11T00:00:00Z"),
        ("week", "2025-07-09T12:00:00Z", "2025-07-13T23:59:59Z", "2025-07-14T00:00:00Z"),
        ("month", "2025-07-10T12:00:00Z", "2025-07-31T23:59:59Z", "2025-08-01T00:00:00Z"),
        ("quarter", "2025-05-10T12:00:00Z", "2025-06-30T23:59:59Z", "2025-07-01T00:00:00Z"),
    ],
)
def test_all_grains_use_exclusive_completion_boundary(
    grain: str,
    timestamp: str,
    before_boundary: str,
    boundary: str,
) -> None:
    incomplete = profile_time_coverage(
        node_id="incomplete",
        timestamps=(timestamp,),
        seed=1,
        grain=grain,
        requested_as_of="2025-12-31",
        source_watermark=before_boundary,
        trustworthy_through=before_boundary,
    )
    complete = profile_time_coverage(
        node_id="complete",
        timestamps=(timestamp,),
        seed=1,
        grain=grain,
        requested_as_of="2025-12-31",
        source_watermark=boundary,
        trustworthy_through=boundary,
    )

    assert incomplete.values["latest_complete_period"] is None
    assert complete.values["latest_complete_period"] is not None


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
        (
            {
                "timestamps": ("1900-01-01", "2100-01-01"),
                "grain": "day",
            },
            "period span",
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
        trustworthy_through="2025-08-01T00:00:00Z",
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
        "grain": "month",
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
        trustworthy_through="2025-08-01T00:00:00Z",
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
        trustworthy_through="2025-08-01T00:00:00Z",
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
        trustworthy_through="2024-03-01T00:00:00Z",
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


def test_weekly_yoy_comparator_preserves_iso_week_boundaries() -> None:
    coverage = profile_time_coverage(
        node_id="coverage",
        timestamps=("2024-01-02", "2024-12-31"),
        seed=1,
        grain="week",
        requested_as_of="2025-01-06",
        source_watermark="2025-01-06T00:00:00Z",
        trustworthy_through="2025-01-06T00:00:00Z",
    )

    result = select_latest_complete_window(
        node_id="window",
        coverage=coverage,
        seed=2,
        window_periods=1,
        comparator_kind="same_period_prior_year",
    )

    assert result.status == "completed"
    assert result.values["current_window"]["periods"] == ("2025-W01",)
    assert result.values["comparator"] == {
        "kind": "same_period_prior_year",
        "grain": "week",
        "start": "2024-01-01",
        "end": "2024-01-07",
        "periods": ("2024-W01",),
    }


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
            trustworthy_through="2025-08-01T00:00:00Z",
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


def test_recent_change_requires_current_comparable_change_evidence() -> None:
    coverage = profile_time_coverage(
        node_id="coverage",
        timestamps=(
            "2024-06-10",
            "2024-07-10",
            "2025-06-10",
            "2025-07-10",
            "2025-08-10",
        ),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )
    window = select_latest_complete_window(
        node_id="window",
        coverage=coverage,
        seed=2,
        window_periods=2,
        comparator_kind="same_period_prior_year",
    )

    assessment = classify_temporal_relevance(
        temporal_intent="recent_change",
        coverage=coverage,
        window=window,
        time_basis="booking_created_at",
        has_change_evidence=True,
    )

    assert assessment.status == "current_change"
    assert assessment.supports_current_action is True
    assert assessment.context["current_window"]["start"] == "2025-06-01"
    assert assessment.context["comparators"][0]["kind"] == (
        "same_period_prior_year"
    )
    assert assessment.context["partial_period_policy"] == "exclude"


def test_recent_change_without_comparator_is_historical_not_current() -> None:
    coverage = profile_time_coverage(
        node_id="coverage",
        timestamps=("2025-06-10", "2025-07-10"),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )
    window = select_latest_complete_window(
        node_id="window",
        coverage=coverage,
        seed=2,
        window_periods=2,
        comparator_kind="none",
    )

    assessment = classify_temporal_relevance(
        temporal_intent="recent_change",
        coverage=coverage,
        window=window,
        time_basis="booking_created_at",
        has_change_evidence=True,
    )

    assert assessment.status == "historical"
    assert assessment.supports_current_action is False
    assert "comparable" in assessment.reason


def test_stale_extract_cannot_support_current_action() -> None:
    coverage = profile_time_coverage(
        node_id="coverage",
        timestamps=("2024-11-10",),
        seed=1,
        grain="month",
        requested_as_of="2026-08-31",
        source_watermark="2024-12-31T23:59:59Z",
        trustworthy_through="2024-12-01T00:00:00Z",
    )

    assessment = classify_temporal_relevance(
        temporal_intent="current_state",
        coverage=coverage,
        time_basis="order_created_at",
    )

    assert assessment.status == "stale"
    assert assessment.supports_current_action is False
    assert assessment.context["data_as_of"] == "2024-12-31T23:59:59Z"


def test_structural_pattern_distinguishes_persistence_from_seasonality() -> None:
    seasonal_coverage = profile_time_coverage(
        node_id="coverage",
        timestamps=("2024-07-10", "2025-07-10"),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )
    persistent_coverage = profile_time_coverage(
        node_id="persistent-coverage",
        timestamps=(
            "2025-04-10",
            "2025-05-10",
            "2025-06-10",
            "2025-07-10",
        ),
        seed=2,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )

    seasonal = classify_temporal_relevance(
        temporal_intent="structural_pattern",
        coverage=seasonal_coverage,
        time_basis="booking_created_at",
        seasonal_cycles=(("2024-07",), ("2025-07",)),
    )
    persistent = classify_temporal_relevance(
        temporal_intent="structural_pattern",
        coverage=persistent_coverage,
        time_basis="booking_created_at",
        persistence_periods=(
            "2025-04",
            "2025-05",
            "2025-06",
            "2025-07",
        ),
    )

    assert seasonal.status == "recurring_seasonal"
    assert seasonal.supports_current_action is False
    assert persistent.status == "persistent"
    assert persistent.supports_current_action is True


def test_missing_time_basis_is_explicitly_not_applicable() -> None:
    coverage = profile_time_coverage(
        node_id="coverage",
        timestamps=("2025-07-10",),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )

    assessment = classify_temporal_relevance(
        temporal_intent="historical_context",
        coverage=coverage,
        time_basis=None,
    )

    assert assessment.status == "not_applicable"
    assert assessment.supports_current_action is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temporal_intent": "latest"}, "temporal_intent"),
        ({"has_change_evidence": 1}, "has_change_evidence"),
        ({"persistence_periods": 3}, "persistence_periods"),
        ({"seasonal_cycles": 2}, "seasonal_cycles"),
    ],
)
def test_temporal_relevance_rejects_invalid_classification_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    coverage = profile_time_coverage(
        node_id="coverage",
        timestamps=("2025-07-10",),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )
    defaults = {
        "temporal_intent": "current_state",
        "coverage": coverage,
        "time_basis": "order_created_at",
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=match):
        classify_temporal_relevance(**defaults)


def test_strict_recency_policy_does_not_treat_eight_day_lag_as_current() -> None:
    coverage = profile_time_coverage(
        node_id="coverage",
        timestamps=("2025-07-10",),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-07T23:59:59Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )
    brief = AnalysisBrief(
        objective="Describe the current level",
        temporal_intent="current_state",
        requested_as_of="2025-08-15",
        recency_policy="strict",
    )

    assessment = classify_temporal_relevance(
        brief=brief,
        coverage=coverage,
        time_basis="order_created_at",
    )

    assert assessment.status == "historical"
    assert assessment.supports_current_action is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "status": "bogus",
            "supports_current_action": False,
            "reason": "reason",
            "context": {},
        },
        {
            "status": "historical",
            "supports_current_action": True,
            "reason": "reason",
            "context": {},
        },
        {
            "status": "historical",
            "supports_current_action": False,
            "reason": "",
            "context": {},
        },
    ],
)
def test_temporal_assessment_rejects_invalid_direct_construction(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        TemporalAssessment(**kwargs)


def test_joined_coverage_uses_least_fresh_required_source() -> None:
    orders = profile_time_coverage(
        node_id="orders",
        timestamps=("2025-05-10", "2025-06-10", "2025-07-10"),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )
    reviews = profile_time_coverage(
        node_id="reviews",
        timestamps=("2025-05-20", "2025-06-20", "2025-07-20"),
        seed=2,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-13T23:59:59Z",
        trustworthy_through="2025-06-01T00:00:00Z",
    )

    joined = combine_time_coverage(
        node_id="orders-reviews",
        coverages={"orders": orders, "reviews": reviews},
        seed=3,
    )

    assert joined.values["trustworthy_through"] == "2025-06-01T00:00:00Z"
    assert joined.values["latest_complete_period"] == {
        "grain": "month",
        "start": "2025-05-01",
        "end": "2025-05-31",
    }
    assert joined.values["observed_periods"] == (
        "2025-05",
        "2025-06",
        "2025-07",
    )
    assert joined.values["freshness_status"] == "mismatched"
    assert joined.diagnostics["source_trustworthy_through"] == {
        "orders": "2025-08-01T00:00:00Z",
        "reviews": "2025-06-01T00:00:00Z",
    }


def test_joined_freshness_mismatch_blocks_current_action() -> None:
    current = profile_time_coverage(
        node_id="current",
        timestamps=("2025-07-10",),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )
    lagged = profile_time_coverage(
        node_id="lagged",
        timestamps=("2025-05-10", "2025-07-10"),
        seed=2,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-06-01T00:00:00Z",
    )
    joined = combine_time_coverage(
        node_id="joined",
        coverages={"current": current, "lagged": lagged},
        seed=3,
    )

    assessment = classify_temporal_relevance(
        temporal_intent="current_state",
        coverage=joined,
        time_basis="order_id",
    )

    assert assessment.status == "stale"
    assert assessment.supports_current_action is False
    assert "freshness" in assessment.reason


def test_joined_coverage_reports_common_period_gaps() -> None:
    first = profile_time_coverage(
        node_id="first",
        timestamps=("2025-05-10", "2025-07-10"),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )
    second = profile_time_coverage(
        node_id="second",
        timestamps=("2025-05-20", "2025-07-20"),
        seed=2,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )

    joined = combine_time_coverage(
        node_id="joined",
        coverages={"first": first, "second": second},
        seed=3,
    )

    assert joined.values["missing_periods"] == ("2025-06",)


def test_joined_coverage_abstains_when_sources_do_not_overlap() -> None:
    old = profile_time_coverage(
        node_id="old",
        timestamps=("2024-01-10",),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2024-02-01T00:00:00Z",
        trustworthy_through="2024-02-01T00:00:00Z",
    )
    new = profile_time_coverage(
        node_id="new",
        timestamps=("2025-07-10",),
        seed=2,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-01T00:00:00Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )

    joined = combine_time_coverage(
        node_id="joined",
        coverages={"old": old, "new": new},
        seed=3,
    )

    assert joined.status == "failed"
    assert joined.failure_code == "no_common_time_coverage"


@pytest.mark.parametrize(
    ("coverages", "match"),
    [
        ({}, "coverages"),
        ({"orders": "not coverage"}, "coverages.orders"),
    ],
)
def test_joined_coverage_rejects_invalid_sources(
    coverages: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        combine_time_coverage(
            node_id="invalid",
            coverages=coverages,
            seed=1,
        )


def test_temporal_source_and_cohort_names_reject_normalized_collisions() -> None:
    coverage = profile_time_coverage(
        node_id="coverage",
        timestamps=("2025-07-10",),
        seed=1,
        grain="month",
        requested_as_of="2025-08-15",
        source_watermark="2025-08-14T23:59:59Z",
        trustworthy_through="2025-08-01T00:00:00Z",
    )
    with pytest.raises(ValueError, match="duplicate normalized source"):
        combine_time_coverage(
            node_id="joined",
            coverages={"orders": coverage, " orders ": coverage},
            seed=1,
        )
    with pytest.raises(ValueError, match="duplicate normalized cohort"):
        assess_cohort_exposure(
            node_id="cohorts",
            cohorts={
                "2024": {
                    "customers": 10,
                    "repeat_customers": 1,
                    "exposure_days": 400,
                },
                " 2024 ": {
                    "customers": 20,
                    "repeat_customers": 2,
                    "exposure_days": 400,
                },
            },
            seed=1,
            minimum_exposure_days=365,
            identity_key="customer_unique_id",
            repeat_definition="More than one delivered order",
            cohort_basis="First observed delivered order",
        )


def test_cohort_exposure_quantifies_right_censoring_sensitivity() -> None:
    result = assess_cohort_exposure(
        node_id="repeat-purchase-exposure",
        cohorts={
            "2024-H1": {
                "customers": 100,
                "repeat_customers": 5,
                "exposure_days": 500,
            },
            "2024-H2": {
                "customers": 100,
                "repeat_customers": 5,
                "exposure_days": 400,
            },
            "2025-H1": {
                "customers": 800,
                "repeat_customers": 16,
                "exposure_days": 100,
            },
        },
        seed=4,
        minimum_exposure_days=365,
        identity_key="customer_unique_id",
        repeat_definition="More than one delivered order",
        cohort_basis="First observed delivered order",
        material_difference=0.01,
    )

    assert result.values["all_cohort_rate"] == pytest.approx(0.026)
    assert result.values["mature_cohort_rate"] == pytest.approx(0.05)
    assert result.values["censoring_sensitivity"] == pytest.approx(0.024)
    assert result.values["sensitivity_status"] == "material"
    assert result.values["mature_cohorts"] == ("2024-H1", "2024-H2")
    assert result.values["immature_cohorts"] == ("2025-H1",)
    assert result.values["denominators"] == {
        "all_customers": 1000,
        "mature_customers": 200,
    }
    assert result.diagnostics["identity_key"] == "customer_unique_id"


def test_cohort_exposure_abstains_without_mature_cohorts() -> None:
    result = assess_cohort_exposure(
        node_id="repeat-purchase-exposure",
        cohorts={
            "2025-H1": {
                "customers": 100,
                "repeat_customers": 3,
                "exposure_days": 90,
            }
        },
        seed=4,
        minimum_exposure_days=365,
        identity_key="customer_unique_id",
        repeat_definition="More than one delivered order",
        cohort_basis="First observed delivered order",
    )

    assert result.status == "failed"
    assert result.failure_code == "no_mature_cohorts"
    assert "right-censoring" in result.failure_message


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cohorts": {}}, "cohorts"),
        (
            {
                "cohorts": {
                    "2025": {
                        "customers": 10,
                        "repeat_customers": 11,
                        "exposure_days": 400,
                    }
                }
            },
            "repeat_customers",
        ),
        ({"minimum_exposure_days": 0}, "minimum_exposure_days"),
        ({"identity_key": ""}, "identity_key"),
    ],
)
def test_cohort_exposure_rejects_invalid_denominator_contract(
    kwargs: dict[str, object],
    match: str,
) -> None:
    defaults = {
        "cohorts": {
            "2024": {
                "customers": 10,
                "repeat_customers": 1,
                "exposure_days": 400,
            }
        },
        "minimum_exposure_days": 365,
        "identity_key": "customer_unique_id",
        "repeat_definition": "More than one delivered order",
        "cohort_basis": "First observed delivered order",
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=match):
        assess_cohort_exposure(
            node_id="invalid",
            seed=1,
            **defaults,
        )
