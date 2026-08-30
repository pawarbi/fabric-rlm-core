"""Deterministic scoring records for analysis benchmark results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
import math
from types import MappingProxyType
from typing import Literal

from fabric_rlm.experimental.analysis_reproducibility import fingerprint


BenchmarkStatus = Literal["completed", "failed"]
_BENCHMARK_STATUSES = {"completed", "failed"}


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    return number + 0.0


def _numeric_mapping(
    values: Mapping[str, object],
    field_name: str,
) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    normalized: dict[str, float] = {}
    for key in sorted(values):
        identity = _required_text(key, f"{field_name} key")
        normalized[identity] = _finite_number(
            values[key],
            f"{field_name}.{identity}",
        )
    return MappingProxyType(normalized)


def _boolean_mapping(
    values: Mapping[str, object],
    field_name: str,
) -> Mapping[str, bool]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    normalized: dict[str, bool] = {}
    for key in sorted(values):
        identity = _required_text(key, f"{field_name} key")
        value = values[key]
        if type(value) is not bool:
            raise ValueError(f"{field_name}.{identity} must be a boolean")
        normalized[identity] = value
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class BenchmarkCaseScore:
    """Immutable metrics and hard invariants for one dataset/task slice."""

    dataset_id: str
    case_id: str
    task: str
    slice_id: str = "all"
    metrics: Mapping[str, float] = field(default_factory=dict)
    invariants: Mapping[str, bool] = field(default_factory=dict)
    sample_size: int = 0
    runtime_seconds: float = 0.0
    status: BenchmarkStatus = "completed"
    failure_code: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("dataset_id", "case_id", "task", "slice_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        if self.status not in _BENCHMARK_STATUSES:
            raise ValueError("status must be completed or failed")
        if type(self.sample_size) is not int or self.sample_size < 0:
            raise ValueError("sample_size must be a non-negative integer")
        runtime = _finite_number(self.runtime_seconds, "runtime_seconds")
        if runtime < 0:
            raise ValueError("runtime_seconds must be non-negative")
        object.__setattr__(self, "runtime_seconds", runtime)
        object.__setattr__(
            self,
            "metrics",
            _numeric_mapping(self.metrics, "metrics"),
        )
        object.__setattr__(
            self,
            "invariants",
            _boolean_mapping(self.invariants, "invariants"),
        )
        if self.status == "failed":
            if self.failure_code is None:
                raise ValueError("failure_code is required for failed scores")
            if self.failure_message is None:
                raise ValueError("failure_message is required for failed scores")
        else:
            if self.failure_code is not None or self.failure_message is not None:
                raise ValueError(
                    "completed scores must not contain failure details"
                )
        if self.failure_code is not None:
            object.__setattr__(
                self,
                "failure_code",
                _required_text(self.failure_code, "failure_code"),
            )
        if self.failure_message is not None:
            object.__setattr__(
                self,
                "failure_message",
                _required_text(self.failure_message, "failure_message"),
            )

    @property
    def passed(self) -> bool:
        return self.status == "completed" and all(self.invariants.values())

    def _result_payload(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "case_id": self.case_id,
            "task": self.task,
            "slice_id": self.slice_id,
            "metrics": dict(self.metrics),
            "invariants": dict(self.invariants),
            "sample_size": self.sample_size,
            "status": self.status,
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self._result_payload())

    def to_dict(self) -> dict[str, object]:
        return {
            **self._result_payload(),
            "runtime_seconds": self.runtime_seconds,
            "fingerprint": self.fingerprint,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregate benchmark results without masking failed slices."""

    cases: tuple[BenchmarkCaseScore, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cases, (list, tuple)) or not self.cases:
            raise ValueError("cases must be a non-empty sequence")
        normalized = tuple(self.cases)
        if any(not isinstance(case, BenchmarkCaseScore) for case in normalized):
            raise ValueError("cases must contain BenchmarkCaseScore values")
        identities = [
            (case.dataset_id, case.case_id, case.task, case.slice_id)
            for case in normalized
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("cases must not contain duplicate slice identities")
        object.__setattr__(self, "cases", normalized)

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)

    @cached_property
    def summary(self) -> Mapping[str, object]:
        metric_names = sorted(
            {name for case in self.cases for name in case.metrics}
        )
        metric_means = {
            name: math.fsum(
                case.metrics[name] for case in self.cases if name in case.metrics
            )
            / sum(name in case.metrics for case in self.cases)
            for name in metric_names
        }
        failed_invariants = tuple(
            MappingProxyType(
                {
                    "case_id": case.case_id,
                    "dataset_id": case.dataset_id,
                    "invariant": name,
                    "slice_id": case.slice_id,
                }
            )
            for case in self.cases
            for name, passed in case.invariants.items()
            if not passed
        )
        failed_cases = tuple(
            MappingProxyType(
                {
                    "case_id": case.case_id,
                    "dataset_id": case.dataset_id,
                    "failure_code": case.failure_code,
                    "slice_id": case.slice_id,
                }
            )
            for case in self.cases
            if case.status == "failed"
        )
        return MappingProxyType(
            {
                "case_count": len(self.cases),
                "passed_case_count": sum(case.passed for case in self.cases),
                "failed_case_count": sum(not case.passed for case in self.cases),
                "metric_means": MappingProxyType(metric_means),
                "failed_cases": failed_cases,
                "failed_invariants": failed_invariants,
            }
        )

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            {
                "cases": [case._result_payload() for case in self.cases],
                "summary": {
                    "case_count": self.summary["case_count"],
                    "passed_case_count": self.summary["passed_case_count"],
                    "failed_case_count": self.summary["failed_case_count"],
                    "metric_means": dict(self.summary["metric_means"]),
                    "failed_cases": [
                        dict(item) for item in self.summary["failed_cases"]
                    ],
                    "failed_invariants": [
                        dict(item) for item in self.summary["failed_invariants"]
                    ],
                },
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cases": [case.to_dict() for case in self.cases],
            "summary": {
                "case_count": self.summary["case_count"],
                "passed_case_count": self.summary["passed_case_count"],
                "failed_case_count": self.summary["failed_case_count"],
                "metric_means": dict(self.summary["metric_means"]),
                "failed_cases": [
                    dict(item) for item in self.summary["failed_cases"]
                ],
                "failed_invariants": [
                    dict(item) for item in self.summary["failed_invariants"]
                ],
            },
            "fingerprint": self.fingerprint,
            "passed": self.passed,
        }


def score_decomposition_case(
    *,
    dataset_id: str,
    case_id: str,
    task: str,
    expected_observed_change: object,
    actual_observed_change: object,
    expected_components: Mapping[str, object],
    actual_components: Mapping[str, object],
    tolerance: object,
    sample_size: int,
    slice_id: str = "all",
    runtime_seconds: float = 0.0,
) -> BenchmarkCaseScore:
    """Score exact decomposition totals, attribution, and reconciliation."""

    expected_change = _finite_number(
        expected_observed_change,
        "expected_observed_change",
    )
    actual_change = _finite_number(
        actual_observed_change,
        "actual_observed_change",
    )
    expected = _numeric_mapping(expected_components, "expected_components")
    actual = _numeric_mapping(actual_components, "actual_components")
    if not expected:
        raise ValueError("expected_components must not be empty")
    numeric_tolerance = _finite_number(tolerance, "tolerance")
    if numeric_tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    component_names = sorted(set(expected) | set(actual))
    component_errors = [
        abs(actual.get(name, 0.0) - expected.get(name, 0.0))
        for name in component_names
    ]
    observed_change_error = abs(actual_change - expected_change)
    reconciliation_error = abs(actual_change - math.fsum(actual.values()))
    attribution_mae = math.fsum(component_errors) / len(component_errors)
    attribution_max_abs_error = max(component_errors)
    return BenchmarkCaseScore(
        dataset_id=dataset_id,
        case_id=case_id,
        task=task,
        slice_id=slice_id,
        metrics={
            "attribution_mae": attribution_mae,
            "attribution_max_abs_error": attribution_max_abs_error,
            "observed_change_error": observed_change_error,
            "reconciliation_error": reconciliation_error,
        },
        invariants={
            "component_keys_match": set(expected) == set(actual),
            "observed_change_within_tolerance": (
                observed_change_error <= numeric_tolerance
            ),
            "reconciliation_within_tolerance": (
                reconciliation_error <= numeric_tolerance
            ),
            "attribution_within_tolerance": (
                attribution_max_abs_error <= numeric_tolerance
            ),
        },
        sample_size=sample_size,
        runtime_seconds=runtime_seconds,
    )


def _binary_labels(values: object) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("labels must be a non-empty sequence")
    normalized = []
    for index, value in enumerate(values):
        if type(value) is not int or value not in {0, 1}:
            raise ValueError(f"labels[{index}] must be 0 or 1")
        normalized.append(value)
    if set(normalized) != {0, 1}:
        raise ValueError("labels must contain both classes")
    return tuple(normalized)


def _probabilities(values: object) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("probabilities must be a non-empty sequence")
    normalized = tuple(
        _finite_number(value, f"probabilities[{index}]")
        for index, value in enumerate(values)
    )
    if any(value < 0 or value > 1 for value in normalized):
        raise ValueError("probabilities must be between 0 and 1")
    return normalized


def _numeric_sequence(values: object, field_name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{field_name} must be a non-empty sequence")
    return tuple(
        _finite_number(value, f"{field_name}[{index}]")
        for index, value in enumerate(values)
    )


def _roc_auc(
    labels: tuple[int, ...],
    probabilities: tuple[float, ...],
) -> float:
    ordered = sorted(
        zip(probabilities, labels),
        key=lambda item: item[0],
    )
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = 0.5 * ((index + 1) + end)
        positive_rank_sum += average_rank * sum(
            label for _, label in ordered[index:end]
        )
        index = end
    positives = sum(labels)
    negatives = len(labels) - positives
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def _average_precision(
    labels: tuple[int, ...],
    probabilities: tuple[float, ...],
) -> float:
    """Return grouped-threshold step-function average precision."""

    ordered = sorted(
        zip(probabilities, labels),
        key=lambda item: item[0],
        reverse=True,
    )
    positives = sum(labels)
    true_positives = 0
    seen = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        true_positives += sum(label for _, label in ordered[index:end])
        seen += end - index
        recall = true_positives / positives
        precision = true_positives / seen
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


def _expected_calibration_error(
    labels: tuple[int, ...],
    probabilities: tuple[float, ...],
    calibration_bins: int,
) -> float:
    bins: list[list[tuple[int, float]]] = [
        [] for _ in range(calibration_bins)
    ]
    for label, probability in zip(labels, probabilities):
        bin_index = min(int(probability * calibration_bins), calibration_bins - 1)
        bins[bin_index].append((label, probability))
    return math.fsum(
        len(items)
        / len(labels)
        * abs(
            math.fsum(probability for _, probability in items) / len(items)
            - math.fsum(label for label, _ in items) / len(items)
        )
        for items in bins
        if items
    )


def score_binary_classification_case(
    *,
    dataset_id: str,
    case_id: str,
    task: str,
    labels: tuple[object, ...] | list[object],
    probabilities: tuple[object, ...] | list[object],
    sample_size: int,
    slice_id: str = "all",
    minimum_sample_size: int = 1,
    calibration_bins: int = 10,
    runtime_seconds: float = 0.0,
) -> BenchmarkCaseScore:
    """Score classification, grouped-threshold PR AUC, and calibration."""

    normalized_labels = _binary_labels(labels)
    normalized_probabilities = _probabilities(probabilities)
    if len(normalized_labels) != len(normalized_probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if type(sample_size) is not int or sample_size != len(normalized_labels):
        raise ValueError("sample_size must match labels length")
    if type(minimum_sample_size) is not int or minimum_sample_size <= 0:
        raise ValueError("minimum_sample_size must be a positive integer")
    if type(calibration_bins) is not int or calibration_bins < 2:
        raise ValueError("calibration_bins must be an integer of at least 2")

    epsilon = 1e-15
    clipped = tuple(
        min(max(probability, epsilon), 1.0 - epsilon)
        for probability in normalized_probabilities
    )
    log_loss = -math.fsum(
        label * math.log(probability)
        + (1 - label) * math.log(1.0 - probability)
        for label, probability in zip(normalized_labels, clipped)
    ) / sample_size
    brier_score = math.fsum(
        (probability - label) ** 2
        for label, probability in zip(
            normalized_labels,
            normalized_probabilities,
        )
    ) / sample_size
    return BenchmarkCaseScore(
        dataset_id=dataset_id,
        case_id=case_id,
        task=task,
        slice_id=slice_id,
        metrics={
            "brier_score": brier_score,
            "expected_calibration_error": _expected_calibration_error(
                normalized_labels,
                normalized_probabilities,
                calibration_bins,
            ),
            "log_loss": log_loss,
            "pr_auc": _average_precision(
                normalized_labels,
                normalized_probabilities,
            ),
            "roc_auc": _roc_auc(
                normalized_labels,
                normalized_probabilities,
            ),
        },
        invariants={
            "both_classes_present": True,
            "minimum_sample_size_met": sample_size >= minimum_sample_size,
            "probabilities_in_range": True,
        },
        sample_size=sample_size,
        runtime_seconds=runtime_seconds,
    )


def score_regression_case(
    *,
    dataset_id: str,
    case_id: str,
    task: str,
    actual: tuple[object, ...] | list[object],
    predicted: tuple[object, ...] | list[object],
    sample_size: int,
    slice_id: str = "all",
    interval_lower: tuple[object, ...] | list[object] | None = None,
    interval_upper: tuple[object, ...] | list[object] | None = None,
    minimum_sample_size: int = 1,
    minimum_interval_coverage: float | None = None,
    require_intervals: bool = False,
    runtime_seconds: float = 0.0,
) -> BenchmarkCaseScore:
    """Score regression error, bias, and optional uncertainty intervals."""

    actual_values = _numeric_sequence(actual, "actual")
    predicted_values = _numeric_sequence(predicted, "predicted")
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual and predicted must have the same length")
    if type(sample_size) is not int or sample_size != len(actual_values):
        raise ValueError("sample_size must match actual length")
    if type(minimum_sample_size) is not int or minimum_sample_size <= 0:
        raise ValueError("minimum_sample_size must be a positive integer")
    if type(require_intervals) is not bool:
        raise ValueError("require_intervals must be a boolean")
    if (interval_lower is None) != (interval_upper is None):
        raise ValueError("interval_lower and interval_upper must be provided together")
    if minimum_interval_coverage is not None:
        numeric_minimum_coverage = _finite_number(
            minimum_interval_coverage,
            "minimum_interval_coverage",
        )
        if numeric_minimum_coverage < 0 or numeric_minimum_coverage > 1:
            raise ValueError(
                "minimum_interval_coverage must be between 0 and 1"
            )
    else:
        numeric_minimum_coverage = None

    errors = tuple(
        _finite_number(
            predicted_value - actual_value,
            f"errors[{index}]",
        )
        for index, (actual_value, predicted_value) in enumerate(
            zip(actual_values, predicted_values)
        )
    )
    metrics = {
        "bias": math.fsum(errors) / sample_size,
        "mae": math.fsum(abs(error) for error in errors) / sample_size,
        "rmse": math.sqrt(
            math.fsum(error**2 for error in errors) / sample_size
        ),
    }
    intervals_present = interval_lower is not None
    invariants = {
        "minimum_sample_size_met": sample_size >= minimum_sample_size,
        "required_intervals_present": not require_intervals or intervals_present,
    }
    if intervals_present:
        lower_values = _numeric_sequence(interval_lower, "interval_lower")
        upper_values = _numeric_sequence(interval_upper, "interval_upper")
        if len(lower_values) != sample_size or len(upper_values) != sample_size:
            raise ValueError("interval bounds must match actual length")
        if any(lower > upper for lower, upper in zip(lower_values, upper_values)):
            raise ValueError("interval lower bounds must not exceed upper bounds")
        coverage = sum(
            lower <= value <= upper
            for value, lower, upper in zip(
                actual_values,
                lower_values,
                upper_values,
            )
        ) / sample_size
        metrics["interval_coverage"] = coverage
        metrics["mean_interval_width"] = math.fsum(
            upper - lower for lower, upper in zip(lower_values, upper_values)
        ) / sample_size
        invariants["interval_bounds_valid"] = True
        if numeric_minimum_coverage is not None:
            invariants["minimum_interval_coverage_met"] = (
                coverage >= numeric_minimum_coverage
            )
    elif numeric_minimum_coverage is not None:
        invariants["minimum_interval_coverage_met"] = False

    return BenchmarkCaseScore(
        dataset_id=dataset_id,
        case_id=case_id,
        task=task,
        slice_id=slice_id,
        metrics=metrics,
        invariants=invariants,
        sample_size=sample_size,
        runtime_seconds=runtime_seconds,
    )
