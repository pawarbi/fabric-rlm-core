"""Tests for adaptive_eval.would_signal_escalate / compare_escalation_timing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from fabric_rlm.experimental.adaptive_eval import (
    compare_escalation_timing,
    would_signal_escalate,
)
from fabric_rlm.experimental.adaptive_policy import (
    AttemptConfig,
    AttemptRecord,
    DifficultyVerdict,
    ValidationVerdict,
    ValidatorOnly,
)

pytestmark = pytest.mark.experimental


@dataclass
class _StubResult:
    submitted: bool = True
    payload: dict = field(default_factory=dict)
    failure_reason: str | None = None
    trajectory: Any = None
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None


def _record(rung: int, *, passed: bool, rollout_index: int = 0) -> AttemptRecord:
    cfg = AttemptConfig(rung=rung, max_turns=10, rollout_index=rollout_index)
    return AttemptRecord(
        rung=rung,
        rollout_index=rollout_index,
        config=cfg,
        result=_StubResult(),
        verdict=ValidationVerdict(passed=passed),
        elapsed_seconds=0.1,
        turns_used=1,
    )


def test_validator_only_agrees_with_actual_failures() -> None:
    # Two failing attempts then one passing — a fail-fail-pass log.
    records = [
        _record(0, passed=False),
        _record(1, passed=False),
        _record(2, passed=True),
    ]
    replay = would_signal_escalate(records, ValidatorOnly())
    # ValidatorOnly says "escalate" at every failed attempt (indices 0 and 1).
    # Actual run also escalated at indices 0 and 1 (rung went 0->1->2).
    assert replay.would_have_escalated_at == [0, 1]
    assert replay.matches_actual is True
    assert replay.actual_passed is True


def test_compare_escalation_timing_aggregates() -> None:
    log_a = [_record(0, passed=False), _record(1, passed=True)]
    log_b = [_record(0, passed=True)]
    out = compare_escalation_timing([log_a, log_b], [ValidatorOnly()])
    row = out["ValidatorOnly"]
    assert row["cases"] == 2
    assert row["agreement"] == 2
    assert row["disagreement"] == 0
