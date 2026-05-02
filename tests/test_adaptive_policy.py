"""Sanity tests for the adaptive policy primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from fabric_rlm.experimental.adaptive_policy import (
    AnswerConsensus,
    AttemptConfig,
    AttemptRecord,
    Budget,
    Confidence,
    DifficultyVerdict,
    Ensemble,
    LadderPolicy,
    NoSubmit,
    ToolErrorRate,
    ValidationVerdict,
    ValidatorOnly,
    as_verdict,
    inject_feedback,
    render_feedback_block,
    select_best_of_n,
)

pytestmark = pytest.mark.experimental


# --- minimal stubs that walk like an RLMResult ---------------------------------


@dataclass
class StubTurn:
    error: str | None = None
    turn: int = 0
    turn_type: str = "normal"
    submitted: bool = False
    response_text: str = ""
    code: str = ""
    stdout: str = ""


@dataclass
class StubTrajectory:
    turns: list[StubTurn] = field(default_factory=list)


@dataclass
class StubResult:
    submitted: bool = True
    payload: dict[str, Any] | None = None
    failure_reason: str | None = None
    trajectory: StubTrajectory = field(default_factory=StubTrajectory)


def _record(
    rung: int,
    *,
    passed: bool,
    payload: dict[str, Any] | None = None,
    submitted: bool = True,
    failure_reason: str | None = None,
    feedback: str | None = None,
    confidence: float | None = None,
    score: float | None = None,
    rollout_index: int = 0,
    config: AttemptConfig | None = None,
    turn_errors: int = 0,
    turn_count: int = 0,
) -> AttemptRecord:
    cfg = config or AttemptConfig(rung=rung)
    turns = [StubTurn(error=("boom" if i < turn_errors else None)) for i in range(turn_count)]
    return AttemptRecord(
        rung=rung,
        rollout_index=rollout_index,
        config=cfg,
        result=StubResult(
            submitted=submitted,
            payload=payload,
            failure_reason=failure_reason,
            trajectory=StubTrajectory(turns=turns),
        ),
        verdict=ValidationVerdict(
            passed=passed,
            confidence=confidence,
            score=score,
            feedback=feedback,
        ),
        elapsed_seconds=0.1,
        turns_used=turn_count,
    )


# --- ValidationVerdict adapter -------------------------------------------------


def test_as_verdict_passes_through_existing_verdict() -> None:
    v = ValidationVerdict(passed=True, score=0.9)
    assert as_verdict(v) is v


def test_as_verdict_wraps_bool() -> None:
    assert as_verdict(True) == ValidationVerdict(passed=True)
    assert as_verdict(False) == ValidationVerdict(passed=False)


# --- ValidatorOnly -------------------------------------------------------------


def test_validator_only_no_attempts_runs_baseline() -> None:
    v = ValidatorOnly().assess([], max_rung=4)
    assert v.action == "escalate"
    assert v.target_rung == 0


def test_validator_only_passing_attempt_stops() -> None:
    rec = _record(0, passed=True)
    v = ValidatorOnly().assess([rec], max_rung=4)
    assert v.action == "stop_pass"


def test_validator_only_failing_attempt_climbs_one_rung() -> None:
    rec = _record(0, passed=False)
    v = ValidatorOnly().assess([rec], max_rung=4)
    assert v.action == "escalate"
    assert v.target_rung == 1


def test_validator_only_exhausts_ladder() -> None:
    rec = _record(4, passed=False)
    v = ValidatorOnly().assess([rec], max_rung=4)
    assert v.action == "stop_fail"


# --- Confidence ---------------------------------------------------------------


def test_confidence_low_jumps_to_effort_rung() -> None:
    rec = _record(0, passed=False, payload={"answer": "x", "confidence": 0.3})
    v = Confidence(threshold=0.7).assess([rec], max_rung=4)
    assert v.action == "escalate"
    assert v.target_rung == 2


def test_confidence_high_falls_back_to_validator_only() -> None:
    rec = _record(0, passed=False, payload={"answer": "x", "confidence": 0.9})
    v = Confidence(threshold=0.7).assess([rec], max_rung=4)
    assert v.action == "escalate"
    assert v.target_rung == 1  # ValidatorOnly behavior


# --- NoSubmit ----------------------------------------------------------------


def test_nosubmit_max_turns_failure_climbs_to_more_turns() -> None:
    rec = _record(0, passed=False, submitted=False, failure_reason="max_turns")
    v = NoSubmit().assess([rec], max_rung=4)
    assert v.action == "escalate"
    assert v.target_rung == 1


def test_nosubmit_tool_error_skips_to_effort() -> None:
    rec = _record(
        0,
        passed=False,
        submitted=False,
        failure_reason="tool_execution_error",
    )
    v = NoSubmit().assess([rec], max_rung=4)
    assert v.action == "escalate"
    assert v.target_rung == 2


# --- AnswerConsensus ---------------------------------------------------------


def test_answer_consensus_repeated_answer_fans_out() -> None:
    a = _record(0, passed=False, payload={"answer": "wrong"})
    b = _record(1, passed=False, payload={"answer": "wrong"})
    v = AnswerConsensus(fanout_n=4).assess([a, b], max_rung=4)
    assert v.action == "fanout"
    assert v.rollouts == 4


def test_answer_consensus_diverging_answers_switches_model() -> None:
    a = _record(0, passed=False, payload={"answer": "x"})
    b = _record(1, passed=False, payload={"answer": "y"})
    v = AnswerConsensus().assess([a, b], max_rung=4)
    assert v.action == "escalate"
    assert v.target_rung == 4


# --- ToolErrorRate -----------------------------------------------------------


def test_tool_error_rate_high_raises_effort() -> None:
    rec = _record(
        0,
        passed=False,
        turn_count=10,
        turn_errors=5,
    )
    v = ToolErrorRate(error_rate_threshold=0.3).assess([rec], max_rung=4)
    assert v.action == "escalate"
    assert v.target_rung == 2


def test_tool_error_rate_low_falls_through_to_validator() -> None:
    rec = _record(0, passed=False, turn_count=10, turn_errors=1)
    v = ToolErrorRate(error_rate_threshold=0.3).assess([rec], max_rung=4)
    assert v.target_rung == 1


# --- Ensemble ----------------------------------------------------------------


def test_ensemble_picks_highest_target_rung() -> None:
    rec = _record(0, passed=False, payload={"answer": "x", "confidence": 0.2})
    v = Ensemble().assess([rec], max_rung=4)
    # Confidence votes rung 2; ValidatorOnly votes rung 1; NoSubmit votes rung 1
    assert v.action == "escalate"
    assert v.target_rung == 2


def test_ensemble_stop_pass_short_circuits() -> None:
    rec = _record(0, passed=True)
    v = Ensemble().assess([rec], max_rung=4)
    assert v.action == "stop_pass"


# --- LadderPolicy ------------------------------------------------------------


def test_ladder_policy_baseline_is_rung_zero() -> None:
    p = LadderPolicy(base_max_turns=8, base_reasoning_effort="low")
    cfg = p.baseline_config()
    assert cfg.rung == 0
    assert cfg.max_turns == 8
    assert cfg.reasoning_effort == "low"


def test_ladder_policy_climbs_in_cost_order() -> None:
    p = LadderPolicy(
        base_max_turns=10,
        base_reasoning_effort="low",
        strong_lm_spec="gpt-5",
    )
    # rung 1: more turns
    cfg = p._build_config(1)
    assert cfg.max_turns == 20 and cfg.reasoning_effort == "low"
    # rung 2: raise effort
    cfg = p._build_config(2)
    assert cfg.reasoning_effort == "medium"
    # rung 3: parallel rollouts
    cfg = p._build_config(3)
    assert cfg.parallel_rollouts == p.parallel_rollouts
    # rung 4: switch lm
    cfg = p._build_config(4)
    assert cfg.lm_spec == "gpt-5"


def test_ladder_policy_max_rung_depends_on_strong_lm() -> None:
    assert LadderPolicy().max_rung == 3
    assert LadderPolicy(strong_lm_spec="gpt-5").max_rung == 4


def test_ladder_skip_more_turns_when_submitted_no_feedback() -> None:
    p = LadderPolicy(skip_more_turns_when_submitted=True)
    rec = _record(
        0,
        passed=False,
        submitted=True,
        feedback=None,
        payload={"answer": "wrong"},
    )
    verdict, cfg = p.next_decision([rec])
    # ValidatorOnly says "escalate to rung 1", but the policy bumps to rung 2
    # because the prior was submitted with no feedback.
    assert verdict.action == "escalate"
    assert cfg is not None
    assert cfg.rung == 2


def test_ladder_does_not_skip_when_feedback_present() -> None:
    p = LadderPolicy(skip_more_turns_when_submitted=True)
    rec = _record(
        0,
        passed=False,
        submitted=True,
        feedback="answer was wrong because X",
        payload={"answer": "wrong"},
    )
    verdict, cfg = p.next_decision([rec])
    assert cfg is not None
    assert cfg.rung == 1
    assert cfg.failure_feedback == "answer was wrong because X"


def test_ladder_propagates_stop_pass() -> None:
    p = LadderPolicy()
    rec = _record(0, passed=True)
    verdict, cfg = p.next_decision([rec])
    assert verdict.action == "stop_pass"
    assert cfg is None


# --- Feedback injection ------------------------------------------------------


def test_render_feedback_block_marks_with_sentinel() -> None:
    block = render_feedback_block("X is wrong", {"answer": "X"})
    assert block.startswith("## PRIOR_ATTEMPT_FEEDBACK")
    assert block.rstrip().endswith("## END_PRIOR_ATTEMPT_FEEDBACK")
    assert "X is wrong" in block


def test_render_feedback_block_includes_attempt_metadata() -> None:
    block = render_feedback_block(
        "X is wrong",
        {"answer": "X"},
        rung=2,
        reasoning_effort="medium",
        submitted=True,
    )
    assert "rung=2" in block
    assert "effort=medium" in block
    assert "submitted=True" in block


def test_render_feedback_block_handles_no_payload() -> None:
    block = render_feedback_block(
        "worker_timeout",
        None,
        rung=0,
        reasoning_effort="minimal",
        submitted=False,
    )
    assert "<no payload" in block
    assert "submitted=False" in block


def test_inject_feedback_prepends_to_first_text_field() -> None:
    inputs = {"question": "what is X?", "max_rows": 100}
    out = inject_feedback(inputs, "wrong", {"answer": "X"})
    assert out["max_rows"] == 100
    assert out["question"].startswith("## PRIOR_ATTEMPT_FEEDBACK")
    assert "what is X?" in out["question"]


def test_inject_feedback_falls_back_to_synthetic_key() -> None:
    inputs = {"rows": 100, "limit": 10}
    out = inject_feedback(inputs, "wrong", {})
    assert "_adaptive_feedback" in out
    assert out["rows"] == 100


# --- Best-of-N selection -----------------------------------------------------


def test_select_best_of_n_prefers_passing_rollout() -> None:
    a = _record(3, passed=False, rollout_index=0)
    b = _record(3, passed=True, rollout_index=1)
    c = _record(3, passed=False, rollout_index=2)
    winner = select_best_of_n([a, b, c])
    assert winner is b


def test_select_best_of_n_breaks_ties_by_score_then_confidence_then_rollout() -> None:
    a = _record(3, passed=True, score=0.5, confidence=0.8, rollout_index=2)
    b = _record(3, passed=True, score=0.9, confidence=0.5, rollout_index=1)
    c = _record(3, passed=True, score=0.9, confidence=0.9, rollout_index=0)
    # b and c tie on score; c has higher confidence
    assert select_best_of_n([a, b, c]) is c


def test_select_best_of_n_deterministic_on_full_tie() -> None:
    rs = [_record(3, passed=False, rollout_index=i) for i in range(4)]
    # all identical except rollout_index — lowest index wins
    assert select_best_of_n(rs).rollout_index == 0
    # same call repeated returns same answer
    for _ in range(3):
        assert select_best_of_n(rs).rollout_index == 0


# --- Budget ------------------------------------------------------------------


def test_budget_clamp_turns_to_remaining() -> None:
    b = Budget(max_total_turns=15)
    assert b.clamp_turns(planned_max_turns=10, turns_used_so_far=8) == 7
    assert b.clamp_turns(planned_max_turns=10, turns_used_so_far=20) == 0


def test_budget_clamp_no_total_returns_planned() -> None:
    b = Budget()
    assert b.clamp_turns(planned_max_turns=10, turns_used_so_far=999) == 10


def test_budget_cap_parallel() -> None:
    b = Budget(max_parallel=3)
    assert b.cap_parallel(10) == 3
    assert b.cap_parallel(0) == 1
    assert b.cap_parallel(2) == 2


# --- AttemptRecord.to_summary -----------------------------------------------


def test_attempt_record_summary_truncates_long_strings() -> None:
    long_str = "x" * 1000
    rec = _record(2, passed=False, payload={"answer": long_str, "evidence": "short"})
    s = rec.to_summary()
    assert s["passed"] is False
    assert s["payload_preview"]["evidence"] == "short"
    assert s["payload_preview"]["answer"].endswith("…")
    assert len(s["payload_preview"]["answer"]) <= 220


def test_attempt_record_summary_omits_turns_by_default(monkeypatch) -> None:
    monkeypatch.delenv("FABRIC_RLM_CAPTURE_TURNS", raising=False)
    rec = _record(2, passed=True, payload={"answer": "ok"})
    s = rec.to_summary()
    assert "turns" not in s


def test_attempt_record_summary_captures_turns_when_env_set(monkeypatch) -> None:
    monkeypatch.setenv("FABRIC_RLM_CAPTURE_TURNS", "1")
    cfg = AttemptConfig(rung=2)
    long_text = "PLAN " * 600  # 3000 chars
    turn = StubTurn(
        turn=1,
        turn_type="normal",
        submitted=True,
        response_text=long_text,
        code="print('hi')",
        stdout="hi",
    )
    rec = AttemptRecord(
        rung=2,
        rollout_index=0,
        config=cfg,
        result=StubResult(
            submitted=True,
            payload={"answer": "ok"},
            trajectory=StubTrajectory(turns=[turn]),
        ),
        verdict=ValidationVerdict(passed=True),
        elapsed_seconds=0.1,
        turns_used=1,
    )
    s = rec.to_summary()
    assert isinstance(s["turns"], list) and len(s["turns"]) == 1
    t = s["turns"][0]
    assert t["turn"] == 1
    assert t["submitted"] is True
    assert t["response_text"].endswith("…")  # truncated
    assert len(t["response_text"]) <= 1501
    assert t["code"] == "print('hi')"
    assert t["stdout"] == "hi"
