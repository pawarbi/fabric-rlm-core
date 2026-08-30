from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from fabric_rlm.experimental import (
    BenchmarkCaseScore,
    BenchmarkReport,
    score_decomposition_case,
)


def test_exact_decomposition_score_reports_zero_error() -> None:
    score = score_decomposition_case(
        dataset_id="decomposition-ground-truth-v1",
        case_id="additive-001",
        task="additive",
        expected_observed_change=15.0,
        actual_observed_change=15.0,
        expected_components={"north": 30.0, "south": -10.0, "closed": -20.0, "new": 15.0},
        actual_components={"new": 15.0, "closed": -20.0, "south": -10.0, "north": 30.0},
        tolerance=1e-12,
        sample_size=4,
        runtime_seconds=0.02,
    )

    assert score.passed is True
    assert score.metrics == {
        "attribution_mae": 0.0,
        "attribution_max_abs_error": 0.0,
        "observed_change_error": 0.0,
        "reconciliation_error": 0.0,
    }
    assert all(score.invariants.values())
    with pytest.raises(FrozenInstanceError):
        score.sample_size = 5


def test_decomposition_score_fails_inaccurate_attribution_even_when_total_matches() -> None:
    score = score_decomposition_case(
        dataset_id="decomposition-ground-truth-v1",
        case_id="additive-001",
        task="additive",
        expected_observed_change=15.0,
        actual_observed_change=15.0,
        expected_components={"north": 30.0, "south": -15.0},
        actual_components={"north": 20.0, "south": -5.0},
        tolerance=1e-9,
        sample_size=2,
    )

    assert score.metrics["reconciliation_error"] == 0.0
    assert score.metrics["attribution_mae"] == 10.0
    assert score.invariants["reconciliation_within_tolerance"] is True
    assert score.invariants["attribution_within_tolerance"] is False
    assert score.passed is False


def test_decomposition_score_preserves_component_key_mismatch() -> None:
    score = score_decomposition_case(
        dataset_id="decomposition-ground-truth-v1",
        case_id="rate-001",
        task="rate",
        expected_observed_change=0.05,
        actual_observed_change=0.05,
        expected_components={"numerator": 0.08, "denominator": -0.03},
        actual_components={"numerator": 0.05},
        tolerance=1e-12,
        sample_size=1,
    )

    assert score.invariants["component_keys_match"] is False
    assert score.metrics["attribution_max_abs_error"] == pytest.approx(0.03)
    assert score.passed is False


def test_report_cannot_hide_failed_slice_behind_good_average() -> None:
    passing = score_decomposition_case(
        dataset_id="synthetic",
        case_id="case-1",
        task="additive",
        slice_id="overall",
        expected_observed_change=10.0,
        actual_observed_change=10.0,
        expected_components={"a": 10.0},
        actual_components={"a": 10.0},
        tolerance=1e-9,
        sample_size=1000,
    )
    failing = score_decomposition_case(
        dataset_id="synthetic",
        case_id="case-1",
        task="additive",
        slice_id="rare",
        expected_observed_change=1.0,
        actual_observed_change=1.0,
        expected_components={"a": 1.0},
        actual_components={"a": 0.0, "proxy": 1.0},
        tolerance=1e-9,
        sample_size=12,
    )

    report = BenchmarkReport(cases=(passing, failing))

    assert report.passed is False
    assert report.summary["passed_case_count"] == 1
    assert report.summary["failed_case_count"] == 1
    assert report.summary["failed_invariants"] == (
        {
            "case_id": "case-1",
            "dataset_id": "synthetic",
            "invariant": "attribution_within_tolerance",
            "slice_id": "rare",
        },
        {
            "case_id": "case-1",
            "dataset_id": "synthetic",
            "invariant": "component_keys_match",
            "slice_id": "rare",
        },
    )


def test_score_and_report_fingerprints_ignore_runtime_only() -> None:
    first = BenchmarkCaseScore(
        dataset_id="dataset",
        case_id="case",
        task="task",
        metrics={"error": 0.0},
        invariants={"safe": True},
        sample_size=10,
        runtime_seconds=0.1,
    )
    second = BenchmarkCaseScore(
        dataset_id="dataset",
        case_id="case",
        task="task",
        metrics={"error": 0.0},
        invariants={"safe": True},
        sample_size=10,
        runtime_seconds=9.5,
    )

    assert first.fingerprint == second.fingerprint
    assert BenchmarkReport(cases=(first,)).fingerprint == BenchmarkReport(
        cases=(second,)
    ).fingerprint
    assert first.to_dict()["runtime_seconds"] != second.to_dict()["runtime_seconds"]


def test_failed_case_requires_structured_failure_and_fails_report() -> None:
    failed = BenchmarkCaseScore(
        dataset_id="dataset",
        case_id="case",
        task="task",
        status="failed",
        failure_code="operator_failed",
        failure_message="The operator did not produce a result",
    )

    assert failed.passed is False
    report = BenchmarkReport(cases=(failed,))
    assert report.passed is False
    assert report.summary["failed_cases"] == (
        {
            "case_id": "case",
            "dataset_id": "dataset",
            "failure_code": "operator_failed",
            "slice_id": "all",
        },
    )

    with pytest.raises(ValueError, match="failure_code"):
        BenchmarkCaseScore(
            dataset_id="dataset",
            case_id="bad",
            task="task",
            status="failed",
        )


def test_report_rejects_duplicate_slice_identities() -> None:
    case = BenchmarkCaseScore(
        dataset_id="dataset",
        case_id="case",
        task="task",
    )

    with pytest.raises(ValueError, match="duplicate"):
        BenchmarkReport(cases=(case, case))


def test_completed_score_rejects_failure_details() -> None:
    with pytest.raises(ValueError, match="completed scores"):
        BenchmarkCaseScore(
            dataset_id="dataset",
            case_id="case",
            task="task",
            failure_code="unexpected",
            failure_message="must not be present",
        )


def test_benchmark_scores_reject_non_finite_metrics() -> None:
    with pytest.raises(ValueError, match="finite"):
        BenchmarkCaseScore(
            dataset_id="dataset",
            case_id="case",
            task="task",
            metrics={"error": math.nan},
        )
