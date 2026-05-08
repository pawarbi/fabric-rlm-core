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
    _trace_length,
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
    completion_tokens: int | None = None,
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
        completion_tokens=completion_tokens,
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


def test_nosubmit_stuck_loop_skips_to_effort() -> None:
    """NEW-H: stuck_loop should NOT climb to 'more turns' (rung 1) — more of
    the same identical-failure turns is exactly what was just proven useless.
    Skip to rung 2 to raise effort/diversity instead.
    """
    rec = _record(0, passed=False, submitted=False, failure_reason="stuck_loop")
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




# --- SRLM Feature A: trace-length tiebreaker (prefer_shorter_traces) ---------


def _equiv(rollout_index: int, completion_tokens: int, **overrides: Any) -> AttemptRecord:
    """Helper: build a passing rollout with identical correctness signals."""
    base = dict(
        passed=True,
        score=1.0,
        confidence=0.9,
        payload={"answer": "x"},
        rollout_index=rollout_index,
        completion_tokens=completion_tokens,
    )
    base.update(overrides)
    return _record(3, **base)


def test_feat_a_tiebreaker_fires_when_enabled() -> None:
    """With prefer_shorter_traces=True, the shorter trace wins on full tie."""
    long_rec = _equiv(rollout_index=0, completion_tokens=500)
    short_rec = _equiv(rollout_index=1, completion_tokens=100)
    winner = select_best_of_n([long_rec, short_rec], prefer_shorter_traces=True)
    assert winner is short_rec
    # And with the flag off (default), the existing -rollout_index tie-break
    # picks the lower-indexed rollout.
    winner_off = select_best_of_n([long_rec, short_rec])
    assert winner_off is long_rec


def test_feat_a_default_byte_identical_to_legacy_call() -> None:
    """Regression: default (flag=False) must match the no-kwarg invocation."""
    import random
    rng = random.Random(0xA)
    rollouts = []
    for i in range(5):
        rollouts.append(
            _record(
                3,
                passed=rng.choice([True, False]),
                score=rng.random(),
                confidence=rng.random(),
                payload={"answer": str(i), "ev": "y" if i % 2 else ""},
                rollout_index=i,
                completion_tokens=rng.randint(50, 1000),
            )
        )
    # Reset metadata between calls so observability writes don't taint the
    # equality check (record identity is what matters).
    for r in rollouts:
        r.metadata.clear()
    legacy = select_best_of_n(rollouts)
    for r in rollouts:
        r.metadata.clear()
    explicit_off = select_best_of_n(rollouts, prefer_shorter_traces=False)
    assert legacy is explicit_off


def test_feat_a_does_not_override_score() -> None:
    """Higher score wins even if its trace is longer."""
    short_lower = _record(
        3, passed=True, score=0.9, confidence=0.9,
        payload={"answer": "x"}, rollout_index=0, completion_tokens=100,
    )
    long_higher = _record(
        3, passed=True, score=0.95, confidence=0.9,
        payload={"answer": "y"}, rollout_index=1, completion_tokens=500,
    )
    assert select_best_of_n([short_lower, long_higher], prefer_shorter_traces=True) is long_higher


def test_feat_a_does_not_override_confidence() -> None:
    """Higher confidence wins even if its trace is longer."""
    short_low_conf = _record(
        3, passed=True, score=1.0, confidence=0.7,
        payload={"answer": "x"}, rollout_index=0, completion_tokens=100,
    )
    long_high_conf = _record(
        3, passed=True, score=1.0, confidence=0.9,
        payload={"answer": "y"}, rollout_index=1, completion_tokens=500,
    )
    assert select_best_of_n([short_low_conf, long_high_conf], prefer_shorter_traces=True) is long_high_conf


def test_feat_a_does_not_override_completeness() -> None:
    """More-complete payload wins even if its trace is longer."""
    short_sparse = _record(
        3, passed=True, score=1.0, confidence=0.9,
        payload={"answer": "x"},  # 1 non-blank
        rollout_index=0, completion_tokens=100,
    )
    long_full = _record(
        3, passed=True, score=1.0, confidence=0.9,
        payload={"answer": "y", "evidence": "z", "notes": "n"},  # 3 non-blank
        rollout_index=1, completion_tokens=500,
    )
    assert select_best_of_n([short_sparse, long_full], prefer_shorter_traces=True) is long_full


def test_feat_a_inverse_failure_universality() -> None:
    """Documented universality trade-off: when validator deems a verbose-correct
    answer and a concise (but happens-to-pass) wrong answer EQUALLY correct,
    Feature A picks the shorter one. We accept this because:

    1. Trace-length is gated to LATE TIE-BREAK only (after passed/score/
       confidence/completeness all equal).
    2. If the validator can't distinguish them, no signal we have can.
    3. This test is a REGRESSION DETECTOR: any future change that broadens
       Feature A's scope (e.g., letting trace-length override score) would
       break the other Feat-A tests above. This one asserts the documented
       behavior so we notice if someone "fixes" the trade-off and silently
       changes the late-tie-break contract.
    """
    verbose_correct = _record(
        3, passed=True, score=1.0, confidence=0.9,
        payload={"answer": "42"}, rollout_index=0, completion_tokens=2000,
    )
    concise_wrong_but_passes = _record(
        3, passed=True, score=1.0, confidence=0.9,
        payload={"answer": "42"}, rollout_index=1, completion_tokens=100,
    )
    winner = select_best_of_n(
        [verbose_correct, concise_wrong_but_passes],
        prefer_shorter_traces=True,
    )
    assert winner is concise_wrong_but_passes


def test_feat_a_observability_metadata_populated() -> None:
    """Every rollout gets trace_length_completion; winner gets selector_key."""
    a = _equiv(rollout_index=0, completion_tokens=300)
    b = _equiv(rollout_index=1, completion_tokens=100)
    c = _equiv(rollout_index=2, completion_tokens=500)
    winner = select_best_of_n([a, b, c], prefer_shorter_traces=True)
    for rec, expected in [(a, 300), (b, 100), (c, 500)]:
        assert "srlm" in rec.metadata
        assert rec.metadata["srlm"]["trace_length_completion"] == expected
    # Winner has selector_key with the trace-length component embedded.
    sk = winner.metadata["srlm"]["selector_key"]
    assert isinstance(sk, tuple)
    # When prefer_shorter_traces=True the key has 7 elements:
    # (passed, score, confidence, required_filled, total_non_blank,
    #  -trace_length, -rollout_index)
    assert len(sk) == 7
    # The -trace_length slot should equal -winner.completion_tokens.
    assert sk[-2] == -100  # b had the shortest trace
    # And losers should not have selector_key set on this call.
    for loser in (a, c):
        assert "selector_key" not in loser.metadata.get("srlm", {})


def test_feat_a_missing_completion_tokens_failsafe() -> None:
    """completion_tokens=None or 0 must not crash; treated as 0."""
    none_tokens = _equiv(rollout_index=0, completion_tokens=None)
    zero_tokens = _equiv(rollout_index=1, completion_tokens=0)
    fifty_tokens = _equiv(rollout_index=2, completion_tokens=50)
    # None and 0 tie at trace_length=0; -rollout_index breaks tie -> idx 0.
    winner = select_best_of_n(
        [none_tokens, zero_tokens, fifty_tokens],
        prefer_shorter_traces=True,
    )
    # Both 0-token rollouts beat the 50-token one; among 0-token, lower index wins.
    assert winner is none_tokens
    assert _trace_length(none_tokens, "completion") == 0
    assert _trace_length(zero_tokens, "completion") == 0
    assert _trace_length(fifty_tokens, "completion") == 50


# --- SRLM Feature A: end-to-end plumbing test --------------------------------


def test_feat_a_plumbing_from_rlm_from_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: RLM.from_task(adaptive={"prefer_shorter_traces": True}) wires
    the flag through runtime._run_adaptive -> AdaptiveRunner -> select_best_of_n.

    Uses the same monkeypatch-AdaptiveRunner pattern as test_adaptive_runtime.
    Instead of invoking real LMs, we capture the AdaptiveRunner kwargs and
    independently exercise select_best_of_n with two synthetic rung-3 rollouts
    of different completion-token counts.
    """
    import warnings as _w
    from fabric_rlm import RLM
    from fabric_rlm.experimental import adaptive_runner as ar_mod

    captured: dict = {}

    class _FakeAdaptiveResult:
        def __init__(self) -> None:
            class _R:
                class trajectory:
                    metadata: dict = {}
                payload = {"answer": "stub"}
                submitted = True
                failure_reason = None
            self.result = _R()
            self.passed = True
            self.attempts = []
            self.winner = None
            self.stop_reason = "ok"
            self.elapsed_seconds = 0.0

    class _CapturingRunner:
        def __init__(self, **kw) -> None:
            captured["kwargs"] = kw

        def run(self, inputs, **_kw):
            return _FakeAdaptiveResult()

    monkeypatch.setattr(ar_mod, "AdaptiveRunner", _CapturingRunner)

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        rlm = RLM.from_task(
            "Compute the answer.",
            inputs={"question": "2+2?"},
            outputs=["answer"],
            lm="gpt-4.1-mini",
            engine="adaptive",
            adaptive={
                "validator": lambda _r: True,
                "prefer_shorter_traces": True,
            },
        )
        rlm.run()

    assert captured["kwargs"]["prefer_shorter_traces"] is True

    # And independently verify the selector behaves correctly with the same
    # flag (this is the "shorter wins" assertion the plan asked for).
    long_rec = _equiv(rollout_index=0, completion_tokens=500)
    short_rec = _equiv(rollout_index=1, completion_tokens=100)
    assert (
        select_best_of_n([long_rec, short_rec], prefer_shorter_traces=True)
        is short_rec
    )
