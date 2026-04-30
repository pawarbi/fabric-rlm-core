"""Validation reports, quality gates, and trajectory metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .trajectory import Trajectory

FAILURE_TAXONOMY = {
    "syntax",
    "runtime",
    "api_knowledge",
    "serialization",
    "validation",
    "timeout",
    "formatting",
    "cost",
    "prompt_over_engineering",
    "unknown",
}


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    failure_type: str | None = None

    def __post_init__(self) -> None:
        if self.failure_type is not None and self.failure_type not in FAILURE_TAXONOMY:
            raise ValueError(f"Unknown failure type: {self.failure_type}")


@dataclass
class ValidationReport:
    run_id: str
    checks: list[ValidationCheck] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def add_check(
        self,
        name: str,
        passed: bool,
        *,
        details: dict[str, Any] | None = None,
        failure_type: str | None = None,
    ) -> None:
        self.checks.append(
            ValidationCheck(
                name=name,
                passed=passed,
                details=details or {},
                failure_type=None if passed else failure_type or "unknown",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def summarize_trajectory(trajectory: Trajectory) -> dict[str, Any]:
    turns = list(trajectory)
    errors = [turn for turn in turns if turn.error]
    validation_failures = [turn for turn in turns if turn.validation_errors]
    submitted = any(turn.submitted and not turn.validation_errors for turn in turns)
    wall_time_s = sum(turn.duration_s or 0 for turn in turns)
    return {
        "turns": len(turns),
        "submitted": submitted,
        "error_turns": len(errors),
        "validation_failed_turns": len(validation_failures),
        "worker_error_rate": len(errors) / len(turns) if turns else 0.0,
        "wall_time_s": wall_time_s,
        "state_keys": turns[-1].state_keys if turns else [],
    }


def classify_failure(message: str) -> str:
    text = message.lower()
    if "syntaxerror" in text:
        return "syntax"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "not json serializable" in text or "serialization" in text:
        return "serialization"
    if "validation" in text or "validator" in text or "assert" in text:
        return "validation"
    if "format" in text or "solution =" in text:
        return "formatting"
    if "token" in text or "quota" in text or "cost" in text:
        return "cost"
    if "attributeerror" in text or "nameerror" in text or "importerror" in text:
        return "api_knowledge"
    if "traceback" in text or "error" in text or "exception" in text:
        return "runtime"
    return "unknown"


def assert_report_passed(report: ValidationReport) -> None:
    failed = [check for check in report.checks if not check.passed]
    if failed:
        names = ", ".join(check.name for check in failed)
        raise AssertionError(f"Validation report failed checks: {names}")

