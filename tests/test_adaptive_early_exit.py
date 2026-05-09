"""Feature E (early-exit probe) tests for AdaptiveRunner.

Feature E is opt-in default-OFF. When enabled, the rung-3 best-of-N step
runs as a probe-then-fanout pattern: launch one candidate first, await
its result, and skip launching the remaining N-1 if the probe passes
the validator. Otherwise launch the suffix in parallel as before.

Empirical justification (`bench/adaptive/p4_prefix_replay_findings.md`):
- 35% of rung-3 rollouts fire the safe `all_pass` predicate at K=1.
- 0/196 pass-flips on captured data — provably no accuracy loss when
  validator is the grader.
- Per-domain savings: math 90-100%, easy_calibration 100%, dabench
  12-21% fire rate.

Contract (per duck review of the design):
- Pass/fail preservation only. Selected-candidate identity may differ
  from full-fanout selection (especially with Feature A on).
- Default-OFF; existing behaviour byte-identical when flag is unset.
- After probe completes, wall-budget is re-checked before launching
  suffix (so a slow probe near the deadline can't push over budget
  by launching N-1 more rollouts).
- Distinct stop_reason ``"early-exit: probe passed"`` for downstream
  attribution.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from fabric_rlm.experimental.adaptive_policy import (
    AttemptConfig,
    Budget,
    LadderPolicy,
    ValidationVerdict,
)
from fabric_rlm.experimental.adaptive_runner import AdaptiveRunner

pytestmark = pytest.mark.experimental


# Reuse the StubResult / verdict_validator pattern from test_adaptive_runner.
@dataclass
class StubTurn:
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class StubTrajectory:
    turns: list[StubTurn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class StubResult:
    submitted: bool = True
    payload: dict | None = None
    failure_reason: str | None = None
    trajectory: StubTrajectory = field(default_factory=StubTrajectory)
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None


class StubRLM:
    def __init__(self, responder, *, calls: list[AttemptConfig] | None = None):
        self.responder = responder
        self.calls = calls

    def run(self, inputs, **kwargs):
        cfg = self._cfg
        if self.calls is not None:
            self.calls.append(cfg)
        result, _ = self.responder(cfg, inputs)
        return result


def make_factory(responder, *, calls=None):
    def factory(cfg: AttemptConfig):
        rlm = StubRLM(responder, calls=calls)
        rlm._cfg = cfg
        return rlm
    return factory


def verdict_validator(result):
    payload = getattr(result, "payload", None) or {}
    v = payload.get("_verdict")
    if isinstance(v, ValidationVerdict):
        return v
    return bool(v)


# ----- 1. Default-OFF byte-identical behaviour ------------------------------


def test_early_exit_default_off_runs_all_n_rollouts() -> None:
    """When the flag is not set, every parallel rollout still launches."""
    calls: list[AttemptConfig] = []
    lock = threading.Lock()

    def responder(cfg, inputs):
        with lock:
            calls.append(cfg)
        # All rung-3 rollouts pass — full fanout still happens by default.
        v = ValidationVerdict(passed=(cfg.rung == 3))
        return (
            StubResult(
                submitted=True,
                payload={"answer": f"r{cfg.rollout_index}", "_verdict": v},
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(
            base_max_turns=5, parallel_rollouts=3,
            skip_more_turns_when_submitted=False,
        ),
        budget=Budget(max_attempts=10),
        validator=verdict_validator,
        # early_exit_probe NOT passed -> default OFF
    )
    result = runner.run({"q": "..."})
    assert result.passed
    rung3_calls = [c for c in calls if c.rung == 3]
    assert len(rung3_calls) == 3, "Default OFF must launch all N rollouts"


# ----- 2. Probe passes -> only 1 rung-3 rollout launches --------------------


def test_early_exit_probe_passes_skips_suffix() -> None:
    """Probe candidate (rollout_index 0) passes; remaining N-1 skipped."""
    calls: list[AttemptConfig] = []
    lock = threading.Lock()

    def responder(cfg, inputs):
        with lock:
            calls.append(cfg)
        # All rung-3 candidates would pass, but we should never see suffix.
        v = ValidationVerdict(passed=(cfg.rung == 3))
        return (
            StubResult(
                submitted=True,
                payload={"answer": f"r{cfg.rollout_index}", "_verdict": v},
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(
            base_max_turns=5, parallel_rollouts=3,
            skip_more_turns_when_submitted=False,
        ),
        budget=Budget(max_attempts=10),
        validator=verdict_validator,
        early_exit_probe=True,
    )
    result = runner.run({"q": "..."})
    assert result.passed
    rung3_calls = [c for c in calls if c.rung == 3]
    assert len(rung3_calls) == 1, "Probe passed -> suffix should not launch"
    assert rung3_calls[0].rollout_index == 0
    # Distinct stop_reason for accounting.
    assert "early" in result.stop_reason.lower()
    # Winner is the probe candidate.
    assert result.winner.rollout_index == 0
    assert result.winner.rung == 3


# ----- 3. Probe fails -> full fanout (probe + N-1) --------------------------


def test_early_exit_probe_fails_runs_full_fanout() -> None:
    """Probe candidate fails; remaining N-1 still launch as parallel suffix."""
    calls: list[AttemptConfig] = []
    lock = threading.Lock()

    def responder(cfg, inputs):
        with lock:
            calls.append(cfg)
        # Rung 3: probe fails, rollout 1 passes, rollout 2 fails.
        if cfg.rung == 3 and cfg.rollout_index == 1:
            v = ValidationVerdict(passed=True)
        else:
            v = ValidationVerdict(passed=False)
        return (
            StubResult(
                submitted=True,
                payload={"answer": f"r{cfg.rollout_index}", "_verdict": v},
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(
            base_max_turns=5, parallel_rollouts=3,
            skip_more_turns_when_submitted=False,
        ),
        budget=Budget(max_attempts=10),
        validator=verdict_validator,
        early_exit_probe=True,
    )
    result = runner.run({"q": "..."})
    rung3_calls = [c for c in calls if c.rung == 3]
    # Probe (idx 0) launched first, then suffix (idx 1, 2).
    assert len(rung3_calls) == 3
    assert {c.rollout_index for c in rung3_calls} == {0, 1, 2}
    assert result.passed
    assert result.winner.rollout_index == 1


# ----- 4. n=1 ignores the flag (no probe semantics for single-rollout) ------


def test_early_exit_with_n1_is_noop() -> None:
    """parallel_rollouts == 1 should never trigger probe semantics."""
    calls: list[AttemptConfig] = []

    def responder(cfg, inputs):
        calls.append(cfg)
        return (
            StubResult(
                submitted=True,
                payload={
                    "answer": "x",
                    "_verdict": ValidationVerdict(passed=(cfg.rung == 0)),
                },
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(base_max_turns=5),
        validator=verdict_validator,
        early_exit_probe=True,
    )
    result = runner.run({"q": "..."})
    assert result.passed
    assert len(calls) == 1


# ----- 5. on_attempt fires only for actually-launched rollouts --------------


def test_early_exit_on_attempt_only_for_launched() -> None:
    """on_attempt callback must fire exactly once per real launch."""
    seen: list[int] = []

    def responder(cfg, inputs):
        v = ValidationVerdict(passed=(cfg.rung == 3))
        return (
            StubResult(
                submitted=True,
                payload={"answer": f"r{cfg.rollout_index}", "_verdict": v},
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    def on_attempt(rec):
        if rec.rung == 3:
            seen.append(rec.rollout_index)

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(
            base_max_turns=5, parallel_rollouts=3,
            skip_more_turns_when_submitted=False,
        ),
        budget=Budget(max_attempts=10),
        validator=verdict_validator,
        early_exit_probe=True,
        on_attempt=on_attempt,
    )
    runner.run({"q": "..."})
    assert seen == [0], (
        "on_attempt must fire only for the probe when probe passes; "
        f"saw {seen}"
    )


# ----- 6. Wall-budget re-checked after probe --------------------------------


def test_early_exit_respects_wall_budget_after_probe() -> None:
    """If wall budget is exhausted by the probe, suffix must NOT launch."""
    calls: list[AttemptConfig] = []
    lock = threading.Lock()

    def slow_responder(cfg, inputs):
        with lock:
            calls.append(cfg)
        # Probe at rung 3 burns wall time; suffix would otherwise launch.
        if cfg.rung == 3 and cfg.rollout_index == 0:
            time.sleep(0.30)
        v = ValidationVerdict(passed=False)
        return (
            StubResult(
                submitted=True,
                payload={"answer": f"r{cfg.rollout_index}", "_verdict": v},
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(slow_responder),
        policy=LadderPolicy(
            base_max_turns=5, parallel_rollouts=3,
            skip_more_turns_when_submitted=False,
        ),
        # Wall budget tight: probe will exceed it, suffix must be skipped.
        budget=Budget(max_attempts=10, max_wall_seconds=0.20),
        validator=verdict_validator,
        early_exit_probe=True,
    )
    result = runner.run({"q": "..."})
    rung3_calls = [c for c in calls if c.rung == 3]
    # Only probe should have launched at rung 3.
    assert len(rung3_calls) == 1, (
        f"wall budget exhausted by probe; suffix must not launch. "
        f"saw rung3 calls: {[c.rollout_index for c in rung3_calls]}"
    )
    assert not result.passed


# ----- 7. Stop-reason distinguishes early-exit from regular BoN -------------


def test_early_exit_stop_reason_is_distinct() -> None:
    """Distinct stop_reason for offline accounting / replay."""
    def responder(cfg, inputs):
        v = ValidationVerdict(passed=(cfg.rung == 3))
        return (
            StubResult(
                submitted=True,
                payload={"answer": "x", "_verdict": v},
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    # With probe ON: probe passes -> distinct stop_reason
    r1 = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(
            base_max_turns=5, parallel_rollouts=3,
            skip_more_turns_when_submitted=False,
        ),
        budget=Budget(max_attempts=10),
        validator=verdict_validator,
        early_exit_probe=True,
    ).run({"q": "..."})
    assert "early" in r1.stop_reason.lower()
    assert "probe" in r1.stop_reason.lower()

    # With probe OFF: standard BoN stop_reason
    r2 = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(
            base_max_turns=5, parallel_rollouts=3,
            skip_more_turns_when_submitted=False,
        ),
        budget=Budget(max_attempts=10),
        validator=verdict_validator,
    ).run({"q": "..."})
    assert "early" not in r2.stop_reason.lower()


# ----- 8. Probe launching twice when n>1 stays at probe-then-suffix --------


def test_early_exit_n5_probe_passes_skips_4() -> None:
    """Confirm scaling: at parallel_rollouts=5, probe passing skips 4."""
    calls: list[AttemptConfig] = []
    lock = threading.Lock()

    def responder(cfg, inputs):
        with lock:
            calls.append(cfg)
        v = ValidationVerdict(passed=(cfg.rung == 3))
        return (
            StubResult(
                submitted=True,
                payload={"answer": f"r{cfg.rollout_index}", "_verdict": v},
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(
            base_max_turns=5, parallel_rollouts=5,
            skip_more_turns_when_submitted=False,
        ),
        budget=Budget(max_attempts=10),
        validator=verdict_validator,
        early_exit_probe=True,
    )
    runner.run({"q": "..."})
    rung3_calls = [c for c in calls if c.rung == 3]
    assert len(rung3_calls) == 1
