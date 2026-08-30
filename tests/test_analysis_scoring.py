from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from fabric_rlm.experimental import (
    BenchmarkCaseScore,
    BenchmarkReport,
    score_binary_classification_case,
    score_clustering_case,
    score_count_rate_case,
    score_detection_case,
    score_decomposition_case,
    score_regression_case,
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


def test_perfect_binary_classification_score_has_ideal_metrics() -> None:
    score = score_binary_classification_case(
        dataset_id="distribution-shift-ground-truth-v1",
        case_id="final-holdout",
        task="classification",
        labels=(0, 0, 1, 1),
        probabilities=(0.0, 0.1, 0.9, 1.0),
        sample_size=4,
        minimum_sample_size=4,
        calibration_bins=2,
    )

    assert score.passed is True
    assert score.metrics["roc_auc"] == 1.0
    assert score.metrics["pr_auc"] == 1.0
    assert score.metrics["brier_score"] == pytest.approx(0.005)
    assert score.metrics["log_loss"] < 0.053
    assert score.metrics["expected_calibration_error"] == pytest.approx(0.05)
    assert score.invariants == {
        "both_classes_present": True,
        "minimum_sample_size_met": True,
        "probabilities_in_range": True,
    }


def test_binary_classification_auc_handles_tied_probabilities() -> None:
    score = score_binary_classification_case(
        dataset_id="dataset",
        case_id="ties",
        task="classification",
        labels=(0, 1, 0, 1),
        probabilities=(0.5, 0.5, 0.5, 0.5),
        sample_size=4,
        calibration_bins=2,
    )

    assert score.metrics["roc_auc"] == pytest.approx(0.5)
    assert score.metrics["pr_auc"] == pytest.approx(0.5)
    assert score.metrics["brier_score"] == pytest.approx(0.25)


def test_binary_classification_score_flags_small_subgroup() -> None:
    score = score_binary_classification_case(
        dataset_id="dataset",
        case_id="holdout",
        task="classification",
        slice_id="rare",
        labels=(0, 1, 0, 1),
        probabilities=(0.2, 0.8, 0.3, 0.7),
        sample_size=4,
        minimum_sample_size=30,
    )

    assert score.invariants["minimum_sample_size_met"] is False
    assert score.passed is False
    report = BenchmarkReport(cases=(score,))
    assert report.summary["failed_invariants"][0]["slice_id"] == "rare"


@pytest.mark.parametrize(
    ("labels", "probabilities", "match"),
    [
        ((0, 1), (0.2,), "same length"),
        ((0, 2), (0.2, 0.8), "labels"),
        ((0, 1), (-0.1, 0.8), "probabilities"),
        ((0, 0), (0.2, 0.3), "both classes"),
    ],
)
def test_binary_classification_score_rejects_invalid_inputs(
    labels: tuple[object, ...],
    probabilities: tuple[object, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        score_binary_classification_case(
            dataset_id="dataset",
            case_id="case",
            task="classification",
            labels=labels,
            probabilities=probabilities,
            sample_size=len(labels),
        )


def test_binary_classification_score_requires_matching_sample_size() -> None:
    with pytest.raises(ValueError, match="sample_size"):
        score_binary_classification_case(
            dataset_id="dataset",
            case_id="case",
            task="classification",
            labels=(0, 1),
            probabilities=(0.2, 0.8),
            sample_size=3,
        )


def test_regression_score_reports_error_bias_and_interval_quality() -> None:
    score = score_regression_case(
        dataset_id="correlated-tabular-ground-truth-v1",
        case_id="final-holdout",
        task="regression",
        actual=(10.0, 20.0, 30.0, 40.0),
        predicted=(11.0, 19.0, 28.0, 42.0),
        interval_lower=(8.0, 18.0, 25.0, 39.0),
        interval_upper=(12.0, 22.0, 31.0, 41.0),
        sample_size=4,
        minimum_sample_size=4,
        minimum_interval_coverage=0.75,
    )

    assert score.metrics["bias"] == 0.0
    assert score.metrics["mae"] == 1.5
    assert score.metrics["rmse"] == pytest.approx(math.sqrt(2.5))
    assert score.metrics["interval_coverage"] == 1.0
    assert score.metrics["mean_interval_width"] == 4.0
    assert score.passed is True


def test_regression_score_flags_undercoverage_and_small_slice() -> None:
    score = score_regression_case(
        dataset_id="dataset",
        case_id="holdout",
        task="regression",
        slice_id="rare",
        actual=(0.0, 1.0, 2.0),
        predicted=(0.0, 1.0, 2.0),
        interval_lower=(0.1, 0.9, 2.1),
        interval_upper=(0.2, 1.1, 2.2),
        sample_size=3,
        minimum_sample_size=30,
        minimum_interval_coverage=0.8,
    )

    assert score.metrics["interval_coverage"] == pytest.approx(1 / 3)
    assert score.invariants["minimum_sample_size_met"] is False
    assert score.invariants["minimum_interval_coverage_met"] is False
    assert score.passed is False


def test_regression_score_can_require_uncertainty_intervals() -> None:
    score = score_regression_case(
        dataset_id="dataset",
        case_id="case",
        task="regression",
        actual=(1.0, 2.0),
        predicted=(1.0, 2.0),
        sample_size=2,
        require_intervals=True,
    )

    assert score.invariants["required_intervals_present"] is False
    assert score.passed is False
    assert "interval_coverage" not in score.metrics

    coverage_required = score_regression_case(
        dataset_id="dataset",
        case_id="case",
        task="regression",
        actual=(1.0, 2.0),
        predicted=(1.0, 2.0),
        sample_size=2,
        minimum_interval_coverage=0.8,
    )
    assert (
        coverage_required.invariants["minimum_interval_coverage_met"] is False
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"actual": (1.0, 2.0), "predicted": (1.0,)},
            "same length",
        ),
        (
            {
                "actual": (1.0, 2.0),
                "predicted": (1.0, 2.0),
                "interval_lower": (0.0, 1.0),
            },
            "provided together",
        ),
        (
            {
                "actual": (1.0, 2.0),
                "predicted": (1.0, 2.0),
                "interval_lower": (0.0, 3.0),
                "interval_upper": (2.0, 2.5),
            },
            "lower",
        ),
    ],
)
def test_regression_score_rejects_invalid_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        score_regression_case(
            dataset_id="dataset",
            case_id="case",
            task="regression",
            sample_size=2,
            **kwargs,
        )


def test_regression_score_rejects_non_finite_derived_errors() -> None:
    with pytest.raises(ValueError, match="errors"):
        score_regression_case(
            dataset_id="dataset",
            case_id="overflow",
            task="regression",
            actual=(1e308,),
            predicted=(-1e308,),
            sample_size=1,
        )


def test_exact_detection_score_has_no_false_alarms_or_delay() -> None:
    score = score_detection_case(
        dataset_id="time-series-ground-truth-v1",
        case_id="anomaly-missing",
        task="anomaly_detection",
        expected_indices=(10, 20, 30),
        detected_indices=(10, 20, 30),
        total_opportunities=100,
        tolerance=0,
        minimum_precision=1.0,
        minimum_recall=1.0,
        maximum_false_alarm_rate=0.0,
        maximum_mean_absolute_delay=0.0,
    )

    assert score.metrics == {
        "f1": 1.0,
        "false_alarm_rate": 0.0,
        "mean_absolute_detection_delay": 0.0,
        "mean_detection_delay": 0.0,
        "precision": 1.0,
        "recall": 1.0,
    }
    assert score.passed is True


def test_detection_score_matches_events_once_within_tolerance() -> None:
    score = score_detection_case(
        dataset_id="dataset",
        case_id="events",
        task="change_detection",
        expected_indices=(10, 20),
        detected_indices=(11, 18, 50),
        total_opportunities=60,
        tolerance=2,
    )

    assert score.metrics["precision"] == pytest.approx(2 / 3)
    assert score.metrics["recall"] == 1.0
    assert score.metrics["f1"] == pytest.approx(0.8)
    assert score.metrics["false_alarm_rate"] == pytest.approx(1 / 60)
    assert score.metrics["mean_detection_delay"] == pytest.approx(-0.5)
    assert score.metrics["mean_absolute_detection_delay"] == pytest.approx(1.5)


def test_detection_score_maximizes_matches_before_minimizing_delay() -> None:
    score = score_detection_case(
        dataset_id="dataset",
        case_id="overlapping-windows",
        task="change_detection",
        expected_indices=(15, 20, 24, 31, 36),
        detected_indices=(7, 17, 23, 25),
        total_opportunities=50,
        tolerance=10,
    )

    assert score.metrics["precision"] == 1.0
    assert score.metrics["recall"] == pytest.approx(4 / 5)
    assert score.metrics["mean_absolute_detection_delay"] == pytest.approx(4.5)


def test_detection_score_supports_anomaly_free_ranges() -> None:
    clean = score_detection_case(
        dataset_id="dataset",
        case_id="clean",
        task="anomaly_detection",
        expected_indices=(),
        detected_indices=(),
        total_opportunities=24,
    )
    noisy = score_detection_case(
        dataset_id="dataset",
        case_id="clean",
        task="anomaly_detection",
        expected_indices=(),
        detected_indices=(5, 12),
        total_opportunities=24,
        maximum_false_alarm_rate=0.05,
    )

    assert clean.metrics["precision"] == 1.0
    assert clean.metrics["recall"] == 1.0
    assert clean.passed is True
    assert noisy.metrics["precision"] == 0.0
    assert noisy.metrics["recall"] == 1.0
    assert noisy.metrics["false_alarm_rate"] == pytest.approx(2 / 24)
    assert noisy.invariants["maximum_false_alarm_rate_met"] is False


def test_detection_score_preserves_failed_thresholds() -> None:
    score = score_detection_case(
        dataset_id="dataset",
        case_id="weak",
        task="anomaly_detection",
        expected_indices=(10, 20, 30),
        detected_indices=(10, 70),
        total_opportunities=100,
        minimum_precision=0.75,
        minimum_recall=0.75,
    )

    assert score.invariants["minimum_precision_met"] is False
    assert score.invariants["minimum_recall_met"] is False
    assert score.passed is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"expected_indices": (1, 1), "detected_indices": ()},
            "duplicates",
        ),
        (
            {"expected_indices": (1,), "detected_indices": (-1,)},
            "non-negative",
        ),
        (
            {
                "expected_indices": (0,),
                "detected_indices": (0,),
                "total_opportunities": 1,
                "tolerance": -1,
            },
            "tolerance",
        ),
    ],
)
def test_detection_score_rejects_invalid_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    defaults = {
        "expected_indices": (),
        "detected_indices": (),
        "total_opportunities": 10,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        score_detection_case(
            dataset_id="dataset",
            case_id="case",
            task="detection",
            **defaults,
        )


def test_clustering_score_is_invariant_to_cluster_label_permutation() -> None:
    score = score_clustering_case(
        dataset_id="clustered-contaminated-ground-truth-v1",
        case_id="clean-clusters",
        task="clustering",
        expected_labels=("a", "a", "b", "b", "c", "c"),
        predicted_labels=("z", "z", "x", "x", "y", "y"),
        sample_size=6,
        silhouette=0.82,
        stability_ari=0.91,
        minimum_ari=0.9,
        minimum_silhouette=0.5,
        minimum_stability_ari=0.8,
    )

    assert score.metrics == {
        "adjusted_rand_index": 1.0,
        "cluster_count_absolute_error": 0.0,
        "silhouette": 0.82,
        "stability_adjusted_rand_index": 0.91,
    }
    assert score.invariants == {
        "minimum_ari_met": True,
        "minimum_silhouette_met": True,
        "minimum_stability_ari_met": True,
    }
    assert score.passed is True


def test_clustering_score_detects_collapsed_clusters() -> None:
    score = score_clustering_case(
        dataset_id="dataset",
        case_id="collapsed",
        task="clustering",
        expected_labels=("a", "a", "b", "b"),
        predicted_labels=("one", "one", "one", "one"),
        sample_size=4,
        minimum_ari=0.5,
        require_expected_cluster_count=True,
    )

    assert score.metrics["adjusted_rand_index"] == 0.0
    assert score.metrics["cluster_count_absolute_error"] == 1.0
    assert score.invariants["minimum_ari_met"] is False
    assert score.invariants["expected_cluster_count_met"] is False
    assert score.passed is False


def test_clustering_score_handles_independent_pairings() -> None:
    score = score_clustering_case(
        dataset_id="dataset",
        case_id="independent",
        task="clustering",
        expected_labels=("a", "a", "b", "b"),
        predicted_labels=("x", "y", "x", "y"),
        sample_size=4,
    )

    assert score.metrics["adjusted_rand_index"] == pytest.approx(-0.5)


@pytest.mark.parametrize(
    ("expected", "predicted"),
    [
        (("a",), ("x",)),
        (("a", "b", "c"), ("x", "y", "z")),
        (("a", "a", "a"), ("x", "x", "x")),
    ],
)
def test_clustering_score_matches_standard_degenerate_ari_convention(
    expected: tuple[str, ...],
    predicted: tuple[str, ...],
) -> None:
    score = score_clustering_case(
        dataset_id="dataset",
        case_id="degenerate",
        task="clustering",
        expected_labels=expected,
        predicted_labels=predicted,
        sample_size=len(expected),
    )

    assert score.metrics["adjusted_rand_index"] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "expected_labels": ("a", "b"),
                "predicted_labels": ("x",),
                "sample_size": 2,
            },
            "same length",
        ),
        (
            {
                "expected_labels": ("a", "b"),
                "predicted_labels": ("x", "y"),
                "sample_size": 2,
                "silhouette": 1.5,
            },
            "silhouette",
        ),
        (
            {
                "expected_labels": ("a", "b"),
                "predicted_labels": ("x", "y"),
                "sample_size": 2,
                "stability_ari": -1.5,
            },
            "stability_ari",
        ),
    ],
)
def test_clustering_score_rejects_invalid_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        score_clustering_case(
            dataset_id="dataset",
            case_id="case",
            task="clustering",
            **kwargs,
        )


def test_count_rate_score_reports_exact_cohort_result() -> None:
    score = score_count_rate_case(
        dataset_id="panel-ground-truth-v1",
        case_id="2026-01",
        task="day_90_retention",
        expected_count=24,
        actual_count=24,
        expected_denominator=30,
        actual_denominator=30,
        rate_tolerance=1e-12,
    )

    assert score.metrics == {
        "count_absolute_error": 0.0,
        "denominator_absolute_error": 0.0,
        "rate_absolute_error": 0.0,
    }
    assert score.invariants == {
        "count_exact": True,
        "denominator_exact": True,
        "rate_within_tolerance": True,
    }
    assert score.sample_size == 30
    assert score.passed is True


def test_count_rate_score_catches_wrong_retention_denominator() -> None:
    score = score_count_rate_case(
        dataset_id="panel-ground-truth-v1",
        case_id="2026-01",
        task="day_90_retention",
        expected_count=24,
        actual_count=24,
        expected_denominator=30,
        actual_denominator=60,
        rate_tolerance=1e-12,
    )

    assert score.metrics["count_absolute_error"] == 0.0
    assert score.metrics["denominator_absolute_error"] == 30.0
    assert score.metrics["rate_absolute_error"] == pytest.approx(0.4)
    assert score.invariants["denominator_exact"] is False
    assert score.invariants["rate_within_tolerance"] is False
    assert score.passed is False


def test_count_rate_score_supports_count_only_funnel_steps() -> None:
    score = score_count_rate_case(
        dataset_id="panel-ground-truth-v1",
        case_id="converted",
        task="funnel_count",
        expected_count=98,
        actual_count=96,
    )

    assert score.metrics == {"count_absolute_error": 2.0}
    assert score.invariants == {"count_exact": False}
    assert score.sample_size == 98
    assert score.passed is False


def test_count_rate_score_requires_censored_outcome_to_remain_unreported() -> None:
    respected = score_count_rate_case(
        dataset_id="panel-ground-truth-v1",
        case_id="2026-04",
        task="day_90_retention",
        censored=True,
    )
    leaked = score_count_rate_case(
        dataset_id="panel-ground-truth-v1",
        case_id="2026-04",
        task="day_90_retention",
        censored=True,
        actual_count=12,
        actual_denominator=20,
    )

    assert respected.metrics == {}
    assert respected.invariants == {"censoring_respected": True}
    assert respected.passed is True
    assert leaked.invariants["censoring_respected"] is False
    assert leaked.passed is False


def test_count_rate_score_rejects_visible_truth_for_censored_case() -> None:
    with pytest.raises(ValueError, match="visible expected outcomes"):
        score_count_rate_case(
            dataset_id="panel-ground-truth-v1",
            case_id="2026-04",
            task="day_90_retention",
            censored=True,
            expected_count=12,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "expected_count": 1,
                "actual_count": 1,
                "expected_denominator": 0,
                "actual_denominator": 0,
            },
            "positive",
        ),
        (
            {
                "expected_count": 2,
                "actual_count": 2,
                "expected_denominator": 1,
                "actual_denominator": 1,
            },
            "must not exceed",
        ),
        (
            {
                "expected_count": 1,
                "actual_count": None,
            },
            "actual_count",
        ),
    ],
)
def test_count_rate_score_rejects_invalid_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        score_count_rate_case(
            dataset_id="dataset",
            case_id="case",
            task="rate",
            **kwargs,
        )
