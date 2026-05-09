"""Unit tests for the prefix-replay simulator.

Prefix-replay simulates Feature E (early-exit). For each captured rung-3
rollout, we walk candidates in execution order and ask: if we had stopped
after K rollouts, would the selected winner have changed (winner-flip)?
Would the rollout's pass/fail outcome have changed (pass-flip)? How many
completion tokens would we have saved?

Tested predicates:
- ``all_pass(prefix)``: every candidate in prefix passed the validator.
  Provably no accuracy loss when validator IS the grader. Safe default.
- ``all_fail_same_canonical(prefix)``: every candidate in prefix failed
  AND they all share the same ``consensus_cluster_id``. Empirically risky
  (a suffix candidate could rescue) — strict opt-in only.
"""

from __future__ import annotations

import math

from bench.adaptive._prefix_replay import (  # type: ignore[import-not-found]
    PrefixReplayResult,
    all_fail_same_canonical,
    all_pass,
    replay_rollout,
)


def _obs(*, idx: int, passed: bool, cluster: str, completion: int,
         score: float = 0.0, conf: float = 0.0) -> dict:
    """Build a synthetic observability row matching live shape."""
    selector_key = [1 if passed else 0, score, conf, 0, 1, -completion, -idx]
    return {
        "selector_key": selector_key,
        "trace_length_completion": completion,
        "trace_length_turns": 1,
        "consensus_cluster_id": cluster,
        "consensus_cluster_size": 1,
        "candidate_answer_preview": cluster[-4:],
    }


# ----- predicate tests -----


def test_all_pass_predicate() -> None:
    p1 = _obs(idx=0, passed=True, cluster="c:a", completion=100)
    p2 = _obs(idx=1, passed=True, cluster="c:a", completion=120)
    f1 = _obs(idx=0, passed=False, cluster="c:b", completion=80)
    assert all_pass([p1, p2]) is True
    assert all_pass([p1, f1]) is False
    assert all_pass([f1]) is False
    # Empty prefix: vacuously true is ambiguous; replay code never calls it
    # with an empty prefix, but we want a sane default for safety.
    assert all_pass([]) is False


def test_all_fail_same_canonical_predicate() -> None:
    f1 = _obs(idx=0, passed=False, cluster="c:x", completion=200)
    f2 = _obs(idx=1, passed=False, cluster="c:x", completion=210)
    f3 = _obs(idx=2, passed=False, cluster="c:y", completion=190)
    p1 = _obs(idx=0, passed=True, cluster="c:x", completion=100)
    assert all_fail_same_canonical([f1, f2]) is True
    assert all_fail_same_canonical([f1, f3]) is False  # different canonical
    assert all_fail_same_canonical([f1, p1]) is False  # one passed
    assert all_fail_same_canonical([f1]) is False  # need ≥2 to be meaningful
    assert all_fail_same_canonical([]) is False


def test_all_fail_same_canonical_unclusterable_is_safe() -> None:
    # Per Phase 3: cluster_id can be None when canonicalization fails.
    # Two unclusterable failures are NOT "the same canonical answer".
    f1 = _obs(idx=0, passed=False, cluster="c:x", completion=200)
    f1["consensus_cluster_id"] = None
    f2 = _obs(idx=1, passed=False, cluster="c:x", completion=210)
    f2["consensus_cluster_id"] = None
    assert all_fail_same_canonical([f1, f2]) is False


# ----- replay_rollout tests -----


def test_all_pass_at_k2_with_3_passing_candidates_fires_no_flip() -> None:
    obs = [
        _obs(idx=0, passed=True, cluster="c:a", completion=100, score=0.9),
        _obs(idx=1, passed=True, cluster="c:a", completion=120, score=0.8),
        _obs(idx=2, passed=True, cluster="c:a", completion=110, score=0.7),
    ]
    res = replay_rollout(obs, predicate=all_pass)
    # Predicate fires at K=2 (first valid K with prefix len ≥ 1 is K=1)
    assert res.first_fire_k == 1  # K=1 also passes "all_pass" with single passing cand
    assert res.fired is True
    # Winner from full = highest score = obs[0]; prefix at K=1 also obs[0]
    assert res.winner_flip is False
    assert res.pass_flip is False
    # Saved tokens = sum of suffix completion = 120 + 110 = 230
    assert res.completion_tokens_saved == 230


def test_all_pass_does_not_fire_when_first_candidate_fails() -> None:
    obs = [
        _obs(idx=0, passed=False, cluster="c:b", completion=100),
        _obs(idx=1, passed=True, cluster="c:a", completion=120),
        _obs(idx=2, passed=True, cluster="c:a", completion=110),
    ]
    res = replay_rollout(obs, predicate=all_pass)
    assert res.fired is False
    assert res.first_fire_k is None
    assert res.completion_tokens_saved == 0


def test_pass_flip_is_recorded_when_safety_violated() -> None:
    # all_pass is provably safe (no pass-flip), so to test the flip-detection
    # mechanics we use all_fail_same_canonical with a suffix that DOES rescue.
    obs = [
        _obs(idx=0, passed=False, cluster="c:wrong", completion=100),
        _obs(idx=1, passed=False, cluster="c:wrong", completion=120),
        _obs(idx=2, passed=True, cluster="c:right", completion=200),
    ]
    res = replay_rollout(obs, predicate=all_fail_same_canonical)
    assert res.fired is True
    assert res.first_fire_k == 2  # need ≥2 fails to fire
    # Full-set winner is obs[2] (passed); prefix winner is one of the failures
    assert res.pass_flip is True
    assert res.winner_flip is True
    # Saved would have been 200, but we'd have lost a passing candidate.
    assert res.completion_tokens_saved == 200


def test_winner_flip_without_pass_flip() -> None:
    # All three pass, but score order would change selector pick if we
    # truncated. Full set winner = obs[2] (highest score).
    # Prefix at K=1 fires immediately, picks obs[0].
    obs = [
        _obs(idx=0, passed=True, cluster="c:a", completion=100, score=0.5),
        _obs(idx=1, passed=True, cluster="c:b", completion=120, score=0.6),
        _obs(idx=2, passed=True, cluster="c:c", completion=110, score=0.9),
    ]
    res = replay_rollout(obs, predicate=all_pass)
    assert res.fired is True
    assert res.first_fire_k == 1
    # Prefix winner is obs[0] (only candidate, score=0.5)
    # Full winner is obs[2] (score=0.9)
    assert res.winner_flip is True
    assert res.pass_flip is False  # both pass, no overall outcome change


def test_singleton_rollout_does_not_fire() -> None:
    obs = [_obs(idx=0, passed=True, cluster="c:a", completion=100)]
    res = replay_rollout(obs, predicate=all_pass)
    # With N=1 there is no suffix to save; we report did-not-fire.
    assert res.fired is False
    assert res.completion_tokens_saved == 0


def test_replay_handles_unsorted_input_by_execution_order() -> None:
    # Input rows in scrambled order (e.g., as JSON file order may not match
    # execution). selector_key[-1] = -idx is the canonical ordering field.
    obs_unsorted = [
        _obs(idx=2, passed=True, cluster="c:a", completion=110),
        _obs(idx=0, passed=False, cluster="c:b", completion=100),
        _obs(idx=1, passed=True, cluster="c:a", completion=120),
    ]
    res = replay_rollout(obs_unsorted, predicate=all_pass)
    # In execution order: idx0=fail, idx1=pass, idx2=pass.
    # all_pass fires at K=1? No: idx0 failed.
    assert res.fired is False


def test_replay_result_dataclass_fields() -> None:
    """Stable schema for downstream aggregation."""
    obs = [
        _obs(idx=0, passed=True, cluster="c:a", completion=100),
        _obs(idx=1, passed=True, cluster="c:a", completion=200),
    ]
    res = replay_rollout(obs, predicate=all_pass)
    assert isinstance(res, PrefixReplayResult)
    # Schema check — these names are used by the aggregator.
    for attr in ("fired", "first_fire_k", "winner_flip", "pass_flip",
                 "completion_tokens_saved", "n_candidates"):
        assert hasattr(res, attr), attr
    assert res.n_candidates == 2
