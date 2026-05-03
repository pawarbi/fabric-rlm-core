"""Tests for the single-LM effort-climbing ladder + bandit pairing.

The cross-model :class:`LadderPolicy` swaps between cheap and strong LMs.
:class:`EffortLadderPolicy` keeps a single LM and walks up the
``reasoning_effort`` axis instead. These tests pin the rung-to-effort
mapping, the parallel-rollouts threshold, and the bandit's cost-aware
tie-break behavior on the new cost map.
"""

from __future__ import annotations

import random

import pytest

from fabric_rlm.experimental.adaptive_policy import AttemptRecord, ValidationVerdict
from fabric_rlm.experimental.bandit_policy import BanditState, _RUNG_COST
from fabric_rlm.experimental.effort_ladder_policy import (
    EFFORT_RUNG_COST,
    EffortBanditPolicy,
    EffortLadderPolicy,
)


@pytest.fixture
def base_policy_kwargs():
    return dict(
        base_lm_spec="azure/gpt-5",
        base_reasoning_effort="minimal",
        parallel_rollouts=3,
    )


# ---- EffortLadderPolicy mapping -------------------------------------------


def test_rung_0_uses_minimal_effort(base_policy_kwargs):
    policy = EffortLadderPolicy(**base_policy_kwargs)
    cfg = policy._build_config(0)
    assert cfg.reasoning_effort == "minimal"
    assert cfg.parallel_rollouts == 1
    assert cfg.lm_spec == "azure/gpt-5"


def test_rung_1_keeps_low_effort_doubles_turns(base_policy_kwargs):
    policy = EffortLadderPolicy(**base_policy_kwargs)
    base = policy._build_config(0)
    rung1 = policy._build_config(1)
    assert rung1.reasoning_effort == "low"
    assert rung1.max_turns == base.max_turns * 2
    assert rung1.parallel_rollouts == 1


def test_rung_2_uses_medium_effort(base_policy_kwargs):
    policy = EffortLadderPolicy(**base_policy_kwargs)
    cfg = policy._build_config(2)
    assert cfg.reasoning_effort == "medium"
    assert cfg.parallel_rollouts == 1


def test_rung_3_uses_high_effort_no_parallel_by_default(base_policy_kwargs):
    policy = EffortLadderPolicy(**base_policy_kwargs)
    cfg = policy._build_config(3)
    assert cfg.reasoning_effort == "high"
    # Default parallel_at_rung = max_rung (4), so rung 3 is still N=1
    assert cfg.parallel_rollouts == 1


def test_rung_4_high_with_parallel_rollouts(base_policy_kwargs):
    policy = EffortLadderPolicy(**base_policy_kwargs)
    cfg = policy._build_config(4)
    assert cfg.reasoning_effort == "high"
    assert cfg.parallel_rollouts == 3


def test_parallel_at_rung_starts_earlier(base_policy_kwargs):
    policy = EffortLadderPolicy(parallel_at_rung=2, **base_policy_kwargs)
    assert policy._build_config(1).parallel_rollouts == 1
    assert policy._build_config(2).parallel_rollouts == 3


def test_lm_spec_never_swaps_even_with_strong_lm_set(base_policy_kwargs):
    """``strong_lm_spec`` is intentionally ignored by EffortLadderPolicy."""
    policy = EffortLadderPolicy(strong_lm_spec="azure/gpt-9", **base_policy_kwargs)
    for rung in range(5):
        assert policy._build_config(rung).lm_spec == "azure/gpt-5"


def test_custom_effort_ladder_respected(base_policy_kwargs):
    policy = EffortLadderPolicy(
        effort_ladder=("low", "low", "medium", "high", "high"),
        **base_policy_kwargs,
    )
    assert policy._build_config(0).reasoning_effort == "low"
    assert policy._build_config(2).reasoning_effort == "medium"


def test_short_ladder_falls_back_to_top_effort(base_policy_kwargs):
    # If ladder shorter than max_rung+1, extra rungs reuse the last entry
    policy = EffortLadderPolicy(
        effort_ladder=("low", "medium"),
        **base_policy_kwargs,
    )
    assert policy._build_config(0).reasoning_effort == "low"
    assert policy._build_config(4).reasoning_effort == "medium"


def test_full_ladder_progression_via_next_decision(base_policy_kwargs):
    """Climbing through the ladder via next_decision() should walk
    minimal → low → medium → high → high+parallel."""
    policy = EffortLadderPolicy(**base_policy_kwargs)
    attempts: list[AttemptRecord] = []
    seen_efforts = []
    seen_parallels = []
    for _ in range(5):
        verdict, cfg = policy.next_decision(attempts)
        if cfg is None:
            break
        seen_efforts.append(cfg.reasoning_effort)
        seen_parallels.append(cfg.parallel_rollouts)
        fake_result = type("R", (), {"submitted": False, "payload": None, "failure_reason": None})()
        attempts.append(
            AttemptRecord(
                rung=cfg.rung,
                rollout_index=0,
                config=cfg,
                result=fake_result,
                verdict=ValidationVerdict(passed=False, feedback="nope"),
                elapsed_seconds=0.1,
                turns_used=1,
            )
        )
    assert seen_efforts == ["minimal", "low", "medium", "high", "high"]
    assert seen_parallels == [1, 1, 1, 1, 3]


# ---- EffortBanditPolicy: bandit + effort ladder integration ---------------


def test_effort_bandit_uses_effort_cost_map_by_default(base_policy_kwargs):
    state = BanditState()
    policy = EffortBanditPolicy(state=state, task_key="X", **base_policy_kwargs)
    # Should default to the effort cost map, NOT the cross-model one
    assert policy.rung_cost is EFFORT_RUNG_COST
    assert policy._cost_for(0) == 1.0
    assert policy._cost_for(4) == 75.0


def test_effort_bandit_respects_explicit_rung_cost_override(base_policy_kwargs):
    custom = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    state = BanditState()
    policy = EffortBanditPolicy(
        state=state, task_key="X", rung_cost=custom, **base_policy_kwargs
    )
    assert policy.rung_cost is custom
    assert policy._cost_for(4) == 1.0


def test_effort_bandit_starting_rung_lands_on_evidenced_high(base_policy_kwargs):
    """If templated evidence shows rung 3 (high effort) consistently passes,
    the bandit should pick it as the starting rung — and use the effort
    cost map, not the cross-model one."""
    state = BanditState()
    # Strong evidence: rung 3 succeeds 20/20; other rungs all fail
    for _ in range(20):
        state.record("X", 3, True)
    for r in (0, 1, 2, 4):
        for _ in range(10):
            state.record("X", r, False)

    rng = random.Random(7)
    policy = EffortBanditPolicy(
        state=state, task_key="X", rng=rng, **base_policy_kwargs
    )
    picks = [policy._bandit_starting_rung()[0] for _ in range(200)]
    # Rung 3 should dominate
    assert picks.count(3) > 100, picks.count(3)
    # Rung 0 should be very rare given negative evidence
    assert picks.count(0) < 20, picks.count(0)


def test_effort_bandit_tiebreak_prefers_cheaper_rung(base_policy_kwargs):
    """When two rungs have similar success rates, the cheaper one wins.
    Critical for the effort axis where minimal vs high is 25× cost."""
    state = BanditState()
    # Both rung 0 (minimal) and rung 3 (high) succeed 10/10
    for _ in range(10):
        state.record("X", 0, True)
        state.record("X", 3, True)
    rng = random.Random(13)
    policy = EffortBanditPolicy(
        state=state, task_key="X", rng=rng, **base_policy_kwargs
    )
    picks = [policy._bandit_starting_rung()[0] for _ in range(200)]
    # Rung 0 should win the tie-break overwhelmingly (cost 1 vs 25)
    assert picks.count(0) > picks.count(3), (picks.count(0), picks.count(3))


def test_effort_bandit_falls_back_to_ladder_when_warmup_unmet(base_policy_kwargs):
    """Below the warmup threshold, bandit defers to plain ladder."""
    state = BanditState()
    # Only 1 observation — warmup default is 2
    state.record("X", 0, False)
    rng = random.Random(0)
    policy = EffortBanditPolicy(
        state=state, task_key="X", rng=rng, warmup=5, **base_policy_kwargs
    )
    verdict, cfg = policy.next_decision([])
    # Should land on rung 0 (the standard ladder cold-start)
    assert cfg is not None and cfg.rung == 0
    assert cfg.reasoning_effort == "minimal"


# ---- Regression: lm_instance must propagate to all rungs ------------------
# (rung-skip bug: when bandit warm-state hops directly to rung 3 with a
#  single-LM EffortLadderPolicy that was built with a base_lm_instance, the
#  config was emitted with both lm_spec=None and lm_instance=None, so the
#  runtime had no LM to construct → KeyError(max_tokens) downstream.
#  EffortLadderPolicy never swaps LMs, so lm_instance must always carry
#  through, regardless of rung.)


class _FakeLM:
    """Minimal stand-in for dspy.LM/FabricLM — has a .copy() that returns
    self with merged kwargs, and stores .kwargs as a dict."""
    def __init__(self, **kw):
        self.kwargs = dict(kw)
    def copy(self, **overrides):
        merged = {**self.kwargs, **overrides}
        return _FakeLM(**merged)


@pytest.mark.parametrize("rung", [0, 1, 2, 3, 4])
def test_lm_instance_propagates_to_all_rungs(rung):
    fake = _FakeLM(model="azure/gpt-5", max_tokens=16000)
    policy = EffortLadderPolicy(
        base_lm_instance=fake,
        base_reasoning_effort="minimal",
        parallel_rollouts=3,
    )
    cfg = policy._build_config(rung)
    # At every rung — including the rung-skip case the bandit triggers when
    # warm state says "go straight to rung 3" — the runtime must receive an
    # LM. EffortLadderPolicy does not swap LMs, so lm_instance must remain.
    assert cfg.lm_instance is fake or cfg.lm_spec is not None, (
        f"rung {rung} produced cfg with both lm_spec=None and lm_instance=None"
    )


@pytest.mark.parametrize("rung", [0, 1, 2, 3, 4])
def test_lm_spec_propagates_to_all_rungs(rung):
    policy = EffortLadderPolicy(
        base_lm_spec="azure/gpt-5",
        base_reasoning_effort="minimal",
        parallel_rollouts=3,
    )
    cfg = policy._build_config(rung)
    assert cfg.lm_spec == "azure/gpt-5", f"rung {rung} dropped lm_spec"
