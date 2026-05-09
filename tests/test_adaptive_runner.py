"""Mechanics tests for AdaptiveRunner.

All tests use a stub ``RLM`` so they run deterministically with no LM keys.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from fabric_rlm.experimental.adaptive_policy import (
    AttemptConfig,
    Budget,
    LadderPolicy,
    ValidationVerdict,
    ValidatorOnly,
)
from fabric_rlm.experimental.adaptive_runner import AdaptiveRunner

pytestmark = pytest.mark.experimental


# --- minimal RLM/Trajectory stubs ---------------------------------------------


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
    """An RLM lookalike whose response depends on the AttemptConfig.

    ``responder(cfg, inputs) -> (StubResult, ValidationVerdict | bool)``.
    """

    def __init__(self, responder, *, calls: list[AttemptConfig] | None = None):
        self.responder = responder
        self.calls = calls

    def run(self, inputs, **kwargs):
        cfg = self._cfg
        if self.calls is not None:
            self.calls.append(cfg)
        result, _verdict = self.responder(cfg, inputs)
        return result


def make_factory(responder, *, calls=None):
    """Closure factory that captures the cfg and uses the responder."""

    def factory(cfg: AttemptConfig):
        rlm = StubRLM(responder, calls=calls)
        rlm._cfg = cfg
        return rlm

    return factory


# Validator that consults the responder's intent: we encode the verdict in
# the StubResult.payload['_verdict'] for these tests.
def verdict_validator(result):
    payload = getattr(result, "payload", None) or {}
    v = payload.get("_verdict")
    if isinstance(v, ValidationVerdict):
        return v
    return bool(v)


# --- 1. Primary attempt passes -> no escalation -------------------------------


def test_primary_passing_attempt_returns_immediately() -> None:
    calls: list[AttemptConfig] = []

    def responder(cfg, inputs):
        return (
            StubResult(
                submitted=True,
                payload={"answer": "42", "_verdict": ValidationVerdict(passed=True)},
                trajectory=StubTrajectory(turns=[StubTurn(), StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder, calls=calls),
        policy=LadderPolicy(base_max_turns=10),
        validator=verdict_validator,
    )
    result = runner.run({"q": "..."})
    assert result.passed
    assert len(result.attempts) == 1
    assert calls[0].rung == 0


# --- 2. Validator keeps failing -> ladder climbs in cost order ---------------


def test_ladder_climbs_rung_by_rung_until_budget() -> None:
    calls: list[AttemptConfig] = []

    def responder(cfg, inputs):
        return (
            StubResult(
                submitted=True,
                payload={"answer": "wrong", "_verdict": ValidationVerdict(passed=False)},
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder, calls=calls),
        policy=LadderPolicy(
            base_max_turns=5,
            base_reasoning_effort="low",
            strong_lm_spec="strong-model",
            skip_more_turns_when_submitted=False,
        ),
        budget=Budget(max_attempts=7),
        validator=verdict_validator,
    )
    result = runner.run({"q": "..."})
    assert not result.passed
    rungs = [c.rung for c in calls]
    # rung 3 fans out to 3 rollouts -> 3 calls at rung 3
    # so we expect: 0, 1, 2, 3, 3, 3, 4
    assert rungs[:4] == [0, 1, 2, 3]
    assert rungs.count(3) == 3
    assert 4 in rungs


# --- 3. Best-of-N selects passing rollout when present ------------------------


def test_best_of_n_selects_passing_rollout() -> None:
    state = {"calls": 0}
    lock = threading.Lock()

    def responder(cfg, inputs):
        # at rung 3 (parallel), only rollout_index == 1 passes
        with lock:
            state["calls"] += 1
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
            base_max_turns=5,
            parallel_rollouts=3,
            skip_more_turns_when_submitted=False,
        ),
        budget=Budget(max_attempts=5),
        validator=verdict_validator,
    )
    result = runner.run({"q": "..."})
    assert result.passed
    assert result.winner.rollout_index == 1
    assert result.winner.rung == 3


def test_best_of_n_falls_back_to_most_populated_when_none_pass() -> None:
    def responder(cfg, inputs):
        # rollout 0: empty; rollout 1: one field; rollout 2: two fields
        if cfg.rollout_index == 0:
            payload = {"_verdict": ValidationVerdict(passed=False)}
        elif cfg.rollout_index == 1:
            payload = {"answer": "x", "_verdict": ValidationVerdict(passed=False)}
        else:
            payload = {
                "answer": "y",
                "evidence": "z",
                "_verdict": ValidationVerdict(passed=False),
            }
        return (
            StubResult(
                submitted=True,
                payload=payload,
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(
            base_max_turns=5,
            parallel_rollouts=3,
            skip_more_turns_when_submitted=False,
        ),
        budget=Budget(max_attempts=4),  # 0,1,2,3 -> stops at end of rung 3
        validator=verdict_validator,
    )
    result = runner.run({"q": "..."})
    assert not result.passed
    # winner should be the most-populated rollout from rung 3
    assert result.winner.rung == 3
    assert result.winner.rollout_index == 2


# --- 4. Budget exhaustion returns best-partial with passed=False --------------


def test_budget_max_attempts_returns_best_partial() -> None:
    def responder(cfg, inputs):
        return (
            StubResult(
                submitted=True,
                payload={"answer": "wrong", "_verdict": ValidationVerdict(passed=False)},
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(base_max_turns=5),
        budget=Budget(max_attempts=2),
        validator=verdict_validator,
    )
    result = runner.run({"q": "..."})
    assert not result.passed
    assert len(result.attempts) == 2
    assert "max_attempts" in result.stop_reason


# --- 5. Adaptive metadata is attached to winner trajectory --------------------


def test_metadata_attached_to_winner_trajectory() -> None:
    seq = iter([False, True])

    def responder(cfg, inputs):
        passed = next(seq)
        return (
            StubResult(
                submitted=True,
                payload={
                    "answer": "x",
                    "_verdict": ValidationVerdict(
                        passed=passed,
                        feedback=None if passed else "wrong, try again",
                    ),
                },
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(base_max_turns=5),
        validator=verdict_validator,
    )
    result = runner.run({"q": "..."})
    assert result.passed
    meta = result.result.trajectory.metadata.get("adaptive")
    assert meta is not None
    assert meta["winner_rung"] == 1
    assert len(meta["attempts"]) == 2
    assert meta["attempts"][0]["passed"] is False
    assert meta["attempts"][1]["passed"] is True


# --- 6. Feedback injection prepends marker to the first text input ----------


def test_feedback_injection_appends_to_inputs() -> None:
    seen_inputs: list[dict] = []
    seq = iter([False, True])

    def responder(cfg, inputs):
        seen_inputs.append(dict(inputs))
        passed = next(seq)
        return (
            StubResult(
                submitted=True,
                payload={
                    "answer": "x",
                    "_verdict": ValidationVerdict(
                        passed=passed,
                        feedback=None if passed else "answer X is wrong",
                    ),
                },
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(base_max_turns=5),
        validator=verdict_validator,
    )
    runner.run({"question": "what?"})
    assert "## PRIOR_ATTEMPT_FEEDBACK" not in seen_inputs[0]["question"]
    assert "## PRIOR_ATTEMPT_FEEDBACK" in seen_inputs[1]["question"]
    assert "what?" in seen_inputs[1]["question"]


def test_feedback_injection_fires_on_no_submit_failure() -> None:
    """REFLECT: synthesize feedback even when validator never ran."""
    seen_inputs: list[dict] = []
    seq = iter([False, True])

    def responder(cfg, inputs):
        seen_inputs.append(dict(inputs))
        if next(seq):
            return (
                StubResult(
                    submitted=True,
                    payload={
                        "answer": "x",
                        "_verdict": ValidationVerdict(passed=True),
                    },
                    trajectory=StubTrajectory(turns=[StubTurn()]),
                ),
                None,
            )
        # First attempt: no submit, no validator feedback at all
        return (
            StubResult(
                submitted=False,
                payload=None,
                trajectory=StubTrajectory(turns=[StubTurn()]),
                failure_reason="worker_timeout",
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(base_max_turns=5),
        validator=verdict_validator,
    )
    runner.run({"question": "what?"})
    # second attempt should have the REFLECT block synthesized from
    # failure_reason rather than from the (absent) validator feedback
    assert "## PRIOR_ATTEMPT_FEEDBACK" in seen_inputs[1]["question"]
    assert "worker_timeout" in seen_inputs[1]["question"]
    assert "submitted=False" in seen_inputs[1]["question"]


# --- 7. Factory exception is captured as failed verdict, not propagated ----


def test_factory_exception_recorded_as_failed_attempt() -> None:
    def responder(cfg, inputs):
        if cfg.rung == 0:
            raise RuntimeError("factory boom")
        return (
            StubResult(
                submitted=True,
                payload={"answer": "ok", "_verdict": ValidationVerdict(passed=True)},
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(
            base_max_turns=5, skip_more_turns_when_submitted=False
        ),
        validator=verdict_validator,
    )
    result = runner.run({"q": "..."})
    assert result.passed
    assert result.attempts[0].verdict.passed is False
    assert "boom" in (result.attempts[0].verdict.feedback or "")


# --- 8. on_attempt callback invoked once per attempt ------------------------


def test_on_attempt_callback_invoked() -> None:
    seen: list[int] = []
    seq = iter([False, True])

    def responder(cfg, inputs):
        passed = next(seq)
        return (
            StubResult(
                submitted=True,
                payload={"answer": "x", "_verdict": ValidationVerdict(passed=passed)},
                trajectory=StubTrajectory(turns=[StubTurn()]),
            ),
            None,
        )

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(base_max_turns=5),
        validator=verdict_validator,
        on_attempt=lambda rec: seen.append(rec.rung),
    )
    runner.run({"q": "..."})
    # Default LadderPolicy.skip_more_turns_when_submitted=True bumps rung 1
    # straight to rung 2 since the prior attempt was submitted with no feedback.
    assert seen == [0, 2]


# --- 9. Feature A scope: prefer_shorter_traces is rung-3-only ---------------
#
# Regression test for SRLM Phase 2 blocker B3: the ``prefer_shorter_traces``
# flag must NOT influence ``_best_partial`` (the exhausted-run fallback that
# selects across attempts of all rungs). It is scoped to the rung-3 best-of-N
# tiebreaker only. Without this scoping, the flag silently changes which
# failed attempt is reported on exhausted runs, contaminating winner_rung
# and answer/cost metrics.


def test_prefer_shorter_traces_does_not_affect_best_partial() -> None:
    """When all attempts fail, ``_best_partial`` ignores prefer_shorter_traces.

    Two failing attempts with identical (passed, score, confidence,
    completeness) tuples differ only in completion_tokens. Default selection
    breaks ties by lowest (rung, rollout_index) — i.e. the first attempt.
    With ``prefer_shorter_traces=True``, the SHORTER trace would win in
    rung-3 BoN. ``_best_partial`` must NOT honor that flag and must return
    the deterministic-tie winner (the first attempt).
    """
    # Two attempts, both fail with identical verdicts and identical payloads.
    # The first has a longer trace, the second is shorter. Default tie-break
    # keeps the first; if Feature A leaked into _best_partial it would pick
    # the second.
    seq = iter([False, False])

    def responder(cfg, inputs):
        is_first = next(seq)  # noqa: F841 — flag flipped via iter only
        n_turns = 2 if is_first else 5
        turns = [StubTurn(completion_tokens=100) for _ in range(n_turns)]
        return (
            StubResult(
                submitted=True,
                payload={
                    "answer": "x",
                    "_verdict": ValidationVerdict(passed=False),
                },
                trajectory=StubTrajectory(turns=turns),
                total_completion_tokens=n_turns * 100,
            ),
            None,
        )

    # Re-do iter so the LONG (higher completion_tokens) attempt comes first.
    seq = iter([True, False])  # True -> long, False -> short

    runner = AdaptiveRunner(
        rlm_factory=make_factory(responder),
        policy=LadderPolicy(base_max_turns=5),
        budget=Budget(max_attempts=2),
        validator=verdict_validator,
        prefer_shorter_traces=True,  # would prefer SHORT in rung-3 BoN
    )
    result = runner.run({"q": "..."})

    assert not result.passed
    assert len(result.attempts) == 2
    # _best_partial must use deterministic tie-break (lowest rung, rollout_index)
    # NOT the prefer_shorter_traces preference. Both attempts are at rung 0/1
    # with rollout_index 0, so the FIRST recorded attempt wins.
    assert result.winner is result.attempts[0], (
        "Feature A leaked into _best_partial: the shorter (later) attempt was "
        "selected instead of the deterministic first-attempt tie-break winner. "
        "prefer_shorter_traces must be scoped to rung-3 BoN selection only."
    )
