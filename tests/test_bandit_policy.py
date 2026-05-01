"""Unit tests for :class:`BanditPolicy` and :class:`BanditState`.

Tests are deterministic: every Thompson-sampling test seeds its own
``random.Random``, and posterior assertions use the closed-form Beta
mean / variance rather than empirical draws.
"""

from __future__ import annotations

import json
import random
import threading
from pathlib import Path

import pytest

from fabric_rlm.experimental.adaptive_policy import (
    AttemptConfig,
    AttemptRecord,
    DifficultyVerdict,
    LadderPolicy,
    ValidationVerdict,
)
from fabric_rlm.experimental.bandit_policy import (
    BanditPolicy,
    BanditState,
    _beta_sample,
    _rung_cost,
)


# ----------------------------------------------------------------------------
# BanditState — persistence & posterior bookkeeping
# ----------------------------------------------------------------------------


def test_beta_for_unseen_returns_uniform():
    state = BanditState()
    assert state.beta_for("MFMC", 0) == (1.0, 1.0)
    assert state.beta_for("anything", 99) == (1.0, 1.0)


def test_record_pass_increments_alpha():
    state = BanditState()
    state.record("MFMC", 4, passed=True)
    state.record("MFMC", 4, passed=True)
    state.record("MFMC", 4, passed=False)
    alpha, beta = state.beta_for("MFMC", 4)
    assert alpha == 3.0  # uniform 1 + two passes
    assert beta == 2.0  # uniform 1 + one fail


def test_record_keeps_keys_isolated():
    state = BanditState()
    state.record("MFMC", 0, passed=False)
    state.record("MCM", 0, passed=True)
    assert state.beta_for("MFMC", 0) == (1.0, 2.0)
    assert state.beta_for("MCM", 0) == (2.0, 1.0)
    assert state.beta_for("MFMC", 1) == (1.0, 1.0)


def test_total_observations_excludes_unseen_rungs():
    state = BanditState()
    state.record("MFMC", 0, passed=False)
    state.record("MFMC", 0, passed=False)
    state.record("MFMC", 4, passed=True)
    # rung 0: alpha=1, beta=3 -> 2 obs; rung 4: alpha=2, beta=1 -> 1 obs
    assert state.total_observations("MFMC") == 3
    assert state.total_observations("never_seen") == 0


def test_save_and_load_roundtrip(tmp_path: Path):
    p = tmp_path / "bandit.json"
    state = BanditState.from_path(p)
    state.record("MFMC", 4, passed=True)
    state.record("MFMC", 0, passed=False)
    state.record("MCM", 2, passed=True)
    state.save()
    assert p.exists()

    loaded = BanditState.from_path(p)
    assert loaded.beta_for("MFMC", 4) == (2.0, 1.0)
    assert loaded.beta_for("MFMC", 0) == (1.0, 2.0)
    assert loaded.beta_for("MCM", 2) == (2.0, 1.0)


def test_load_missing_path_yields_empty_state(tmp_path: Path):
    state = BanditState.from_path(tmp_path / "does-not-exist.json")
    assert state.priors == {}
    assert state.beta_for("anything", 0) == (1.0, 1.0)


def test_load_corrupt_json_yields_empty_state(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text("{ not valid json", encoding="utf-8")
    state = BanditState.from_path(p)
    assert state.priors == {}


def test_save_is_atomic_via_tmp_rename(tmp_path: Path, monkeypatch):
    p = tmp_path / "bandit.json"
    state = BanditState.from_path(p)
    state.record("MFMC", 4, passed=True)

    written: list[Path] = []
    real_replace = Path.replace

    def tracking_replace(self: Path, target: Path) -> Path:
        # Confirm we wrote to a .tmp first
        assert self.suffix.endswith(".tmp"), self
        written.append(target)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", tracking_replace)
    state.save()
    assert written == [p]
    assert p.exists()


def test_record_is_thread_safe():
    state = BanditState()
    n_per_thread = 200
    threads = [
        threading.Thread(
            target=lambda: [state.record("MFMC", 0, passed=True) for _ in range(n_per_thread)]
        )
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    alpha, beta = state.beta_for("MFMC", 0)
    assert alpha == 1.0 + 8 * n_per_thread
    assert beta == 1.0


# ----------------------------------------------------------------------------
# BanditPolicy — Thompson sampling on the starting rung
# ----------------------------------------------------------------------------


def _ladder() -> dict:
    """Common kwargs producing a ladder with rungs 0..4."""

    return dict(
        base_max_turns=4,
        strong_lm_spec=object(),  # presence enables rung 4
        parallel_rollouts=2,
    )


def test_no_state_falls_back_to_ladder():
    policy = BanditPolicy(state=None, task_key="MFMC", **_ladder())
    verdict, cfg = policy.next_decision(attempts=[])
    # Plain LadderPolicy starts at rung 0 with no attempts
    assert cfg is not None
    assert cfg.rung == 0
    assert "no attempts" in (verdict.reason or "").lower()


def test_no_task_key_falls_back_to_ladder():
    state = BanditState()
    state.record("MFMC", 4, passed=True)
    state.record("MFMC", 4, passed=True)
    policy = BanditPolicy(state=state, task_key="", **_ladder())
    verdict, cfg = policy.next_decision(attempts=[])
    assert cfg is not None
    assert cfg.rung == 0


def test_below_warmup_falls_back_to_ladder():
    state = BanditState()
    state.record("MFMC", 4, passed=True)  # only 1 obs, warmup default 2
    policy = BanditPolicy(state=state, task_key="MFMC", **_ladder())
    verdict, cfg = policy.next_decision(attempts=[])
    assert cfg is not None
    assert cfg.rung == 0  # ladder default


def test_picks_strong_rung_when_strongly_evidenced():
    """After 10 strong-LM passes and 10 cheap-LM fails, bandit should jump high.

    With raw Thompson sampling (no cost discount) plus a 5% epsilon
    tie-breaker, rung 4's posterior (Beta(11, 1)) dominates rungs 0/1/2/3
    (Beta(1, 11) for rung 0, Beta(1, 1) for the unobserved rungs).
    """

    state = BanditState()
    for _ in range(10):
        state.record("MFMC", 0, passed=False)
        state.record("MFMC", 4, passed=True)
    rng = random.Random(42)
    policy = BanditPolicy(state=state, task_key="MFMC", warmup=1, rng=rng, **_ladder())

    chosen = []
    for _ in range(200):
        verdict, cfg = policy.next_decision(attempts=[])
        chosen.append(cfg.rung if cfg is not None else None)

    counts = {rung: chosen.count(rung) for rung in range(5)}
    # Rung 4 should be picked the majority of the time.
    assert counts[4] > 100, counts
    # Rung 0 (overwhelmingly evidenced as failing) should be picked rarely.
    assert counts[0] < 20, counts


def test_cost_tiebreak_prefers_cheaper_rung():
    """Two rungs with similar success rates: pick the cheaper."""

    state = BanditState()
    # Both rungs look similarly successful (~70%)
    for _ in range(20):
        state.record("MCM", 2, passed=True)
        state.record("MCM", 4, passed=True)
    for _ in range(8):
        state.record("MCM", 2, passed=False)
        state.record("MCM", 4, passed=False)
    rng = random.Random(0)
    policy = BanditPolicy(state=state, task_key="MCM", warmup=1, rng=rng, **_ladder())
    chosen = [policy.next_decision(attempts=[])[1].rung for _ in range(200)]
    counts = {rung: chosen.count(rung) for rung in range(5)}
    # When both 2 and 4 sampled within ε, prefer 2 (cheaper).
    assert counts[2] > counts[4], counts


def test_after_first_attempt_falls_back_to_super():
    """Bandit only drives the *starting* rung; subsequent escalations are
    standard ladder behavior."""

    state = BanditState()
    for _ in range(10):
        state.record("MFMC", 0, passed=False)
        state.record("MFMC", 4, passed=True)
    rng = random.Random(0)
    policy = BanditPolicy(state=state, task_key="MFMC", warmup=1, rng=rng, **_ladder())

    # Fabricate a single failed attempt at rung 2
    cfg = policy._build_config(2)
    fake_result = type("R", (), {"submitted": False, "payload": None, "failure_reason": None})()
    rec = AttemptRecord(
        rung=2,
        rollout_index=0,
        config=cfg,
        result=fake_result,
        verdict=ValidationVerdict(passed=False),
        elapsed_seconds=1.0,
        turns_used=1,
    )
    verdict, cfg2 = policy.next_decision(attempts=[rec])
    assert cfg2 is not None
    # Standard ladder escalation: next_rung = last.rung + 1 = 3
    assert cfg2.rung == 3


def test_bandit_pick_records_outcome_through_state():
    """End-to-end: caller drives state updates after each task."""

    state = BanditState()
    policy = BanditPolicy(state=state, task_key="MFMC", warmup=0, **_ladder())
    # Cold start: even with warmup=0 and no priors, Thompson sampling should
    # eventually fire (priors are uniform, no evidence). Just check it doesn't
    # crash and produces a valid rung.
    for _ in range(20):
        verdict, cfg = policy.next_decision(attempts=[])
        assert cfg is not None
        assert 0 <= cfg.rung <= policy.max_rung
        # Simulate task outcome
        state.record("MFMC", cfg.rung, passed=(cfg.rung >= 4))

    # After 20 outcomes, strong rung should be more frequent
    assert state.beta_for("MFMC", 4)[0] > state.beta_for("MFMC", 4)[1]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def test_beta_sample_returns_value_in_range():
    rng = random.Random(0)
    for _ in range(100):
        x = _beta_sample(2.0, 5.0, rng)
        assert 0.0 <= x <= 1.0


def test_beta_sample_handles_degenerate_zero():
    """If both shape params are tiny enough that gammavariate returns 0."""

    rng = random.Random(0)
    # Very small alpha + beta to maximize chance of 0/0
    for _ in range(100):
        x = _beta_sample(0.001, 0.001, rng)
        assert 0.0 <= x <= 1.0


def test_rung_cost_monotonic():
    """Cost ordering must match the ladder cost ordering."""

    assert _rung_cost(0) < _rung_cost(1) < _rung_cost(2) < _rung_cost(3) < _rung_cost(4)


def test_subclass_relationship():
    """BanditPolicy must be a true LadderPolicy so the runner accepts it."""

    policy = BanditPolicy()
    assert isinstance(policy, LadderPolicy)
