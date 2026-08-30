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
