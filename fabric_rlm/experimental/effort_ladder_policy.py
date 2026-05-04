"""Effort-climbing escalation policy for single-LM reasoning workloads.

Where :class:`LadderPolicy` climbs by *swapping LMs* (cheap → strong) plus
some intermediate steps (more turns, parallel rollouts, bumped effort),
:class:`EffortLadderPolicy` keeps a single LM the entire way and uses the
``reasoning_effort`` axis as the primary ladder. This is the right shape
when the cheap LM is unsuitable (multi-step reasoning that minimum effort
won't crack) but you still want to spend as little reasoning budget as
possible per task.

Default rung mapping (all using ``base_lm_spec``):

    rung 0: minimal effort
    rung 1: low effort + retry feedback (2× max_turns)
    rung 2: medium effort
    rung 3: high effort
    rung 4: high effort + parallel-N rollouts (best-of-N)

This pairs well with :class:`BanditPolicy` because each rung produces
qualitatively different reasoning depth (not just "more attempts at the
same depth"), so per-template posteriors separate quickly. Pass
``rung_cost=_EFFORT_RUNG_COST`` to the BanditPolicy so its tie-break
math reflects the new economics (each step ~3× cost).

Example
-------

    from fabric_rlm.experimental import (
        BanditPolicy, BanditState, EffortLadderPolicy, EFFORT_RUNG_COST,
    )

    state = BanditState.from_path("/lakehouse/.../bandit/state.json")
    policy = EffortBanditPolicy(  # convenience subclass below
        base_lm_spec="azure/gpt-5",
        base_reasoning_effort="minimal",
        parallel_rollouts=3,
        state=state,
        task_key="MCM",
        rung_cost=EFFORT_RUNG_COST,
    )
    rlm = RLM(engine='adaptive', adaptive=dict(policy=policy, ...))
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from .adaptive_policy import (
    AttemptConfig,
    LadderPolicy,
    ReasoningEffort,
)
from .bandit_policy import BanditPolicy, _EFFORT_RUNG_COST


# Public re-export so callers don't have to reach into bandit_policy internals.
EFFORT_RUNG_COST = _EFFORT_RUNG_COST


# The default effort progression for an effort-only ladder. Rung 0 starts
# at ``base_reasoning_effort`` and each rung climbs one step. ``None`` at
# rung 0 means "use whatever the base config says"; subsequent rungs
# resolve relative to that starting point.
_EFFORT_LADDER: tuple[ReasoningEffort, ...] = (
    "minimal",
    "low",
    "medium",
    "high",
    "high",  # rung 4 stays at "high" but adds parallel rollouts
)


@dataclass
class EffortLadderPolicy(LadderPolicy):
    """Single-LM ladder that escalates by reasoning effort.

    Parameters
    ----------
    effort_ladder
        Tuple of (rung 0..max_rung) effort levels. Defaults to
        ``("minimal", "low", "medium", "high", "high")``.
    parallel_at_rung
        Rung at which to start adding parallel rollouts (best-of-N).
        Defaults to ``max_rung`` (i.e. only the top rung uses parallel).
        Set to a lower rung to add parallel earlier.

    Notes
    -----
    The base class field ``strong_lm_spec`` is *intentionally ignored* —
    this policy never swaps LMs. Pass it anyway and it's simply unused.
    """

    effort_ladder: tuple[ReasoningEffort, ...] = _EFFORT_LADDER
    parallel_at_rung: int | None = None  # None -> max_rung
    # When True, an extra top rung is appended that runs decompose-then-
    # synthesize (Phase A: 1 LM call breaks problem into 2..max_subs sub-
    # problems; Phase B: parallel sub-solves; Phase C: 1 LM call merges).
    # Universal — works for any prompt, no task-specific config.
    #
    # ADVISORY DEPRECATION (dev11, 2026-05): showed 0/16 wins on hard-only
    # single-shot CS-template suite. Not a fair test of decomposition (those
    # tasks need sequential reasoning, not parallel split), but until a
    # workload shows ≥2 positive wins, do NOT enable for production benches.
    # See ``decompose_rung.py`` module docstring. Slated for removal 2026-Q4
    # if no positive evidence accumulates.
    enable_decompose_top_rung: bool = False
    decompose_max_subs: int = 6

    def __post_init__(self) -> None:
        if self.enable_decompose_top_rung:
            warnings.warn(
                "enable_decompose_top_rung=True: this rung showed 0/16 wins on the "
                "dev11 hard-CS bench (single-shot deep reasoning) and is currently on "
                "advisory deprecation. It may still help on multi-step planning / "
                "multi-doc workloads — re-enable only with a measurement plan. "
                "See fabric_rlm/experimental/decompose_rung.py module docstring.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def max_rung(self) -> int:
        # Independent of strong_lm_spec — driven by the effort_ladder length.
        # Always at least 4 by default (matching the canonical 5-rung ladder).
        base = max(0, len(self.effort_ladder) - 1)
        if self.enable_decompose_top_rung:
            base += 1
        return base

    def _build_config(
        self,
        rung: int,
        *,
        rollouts: int = 1,
        feedback: str | None = None,
    ) -> AttemptConfig:
        cfg = self.baseline_config()
        max_turns = cfg.max_turns

        # Top rung is decompose-phase when enabled — it bypasses the turn
        # loop entirely (handled by the runtime factory). Effort is set to
        # the highest available so synthesize/decompose calls reason hard.
        if self.enable_decompose_top_rung and rung == self.max_rung:
            return AttemptConfig(
                rung=rung,
                max_turns=max_turns,  # ignored by adapter, kept for logging
                reasoning_effort=self.effort_ladder[-1],
                lm_spec=cfg.lm_spec,
                lm_instance=cfg.lm_instance,
                inner_engine=cfg.inner_engine,
                parallel_rollouts=1,
                failure_feedback=feedback,
                decompose_phase=True,
                decompose_max_subs=self.decompose_max_subs,
            )

        # Pick the effort for this rung from the ladder, clamped to
        # the available steps. Falls back to "high" if the ladder is
        # shorter than max_rung+1.
        if rung < len(self.effort_ladder):
            reasoning_effort = self.effort_ladder[rung]
        else:
            reasoning_effort = self.effort_ladder[-1]

        # Rung 1 keeps the same effort as rung 0 by default, but doubles
        # max_turns so the retry has actual room to land. This mirrors
        # the cross-model LadderPolicy's "2× turns at rung 1" behavior.
        if rung >= 1:
            max_turns = cfg.max_turns * 2

        # Parallel rollouts kick in at parallel_at_rung (default: top rung)
        parallel = 1
        # When decompose-top is on, the "parallel-eligible" top is the
        # rung *below* decompose, since decompose itself is single-rollout.
        effective_top = self.max_rung - (1 if self.enable_decompose_top_rung else 0)
        threshold = self.parallel_at_rung if self.parallel_at_rung is not None else effective_top
        if rung >= threshold and rung != self.max_rung or (rung == self.max_rung and not self.enable_decompose_top_rung):
            parallel = max(rollouts, self.parallel_rollouts)

        return AttemptConfig(
            rung=rung,
            max_turns=max_turns,
            reasoning_effort=reasoning_effort,
            lm_spec=cfg.lm_spec,
            # EffortLadderPolicy never swaps LMs — it only varies
            # reasoning_effort. The runtime needs an LM at every rung
            # (including when the bandit's warm-state hops directly to a
            # high rung without seeing rung 0 first), so the lm_instance
            # must propagate. The cross-model LadderPolicy zeroes
            # lm_instance at rung >= 1 because it switches to
            # strong_lm_spec at rung 4 and re-resolves from a dict spec;
            # that doesn't apply here.
            lm_instance=cfg.lm_instance,
            inner_engine=cfg.inner_engine,
            parallel_rollouts=parallel,
            failure_feedback=feedback,
        )


@dataclass
class EffortBanditPolicy(BanditPolicy, EffortLadderPolicy):
    """Convenience: BanditPolicy on top of an EffortLadderPolicy.

    Equivalent to writing your own subclass of both — but supplies
    :data:`EFFORT_RUNG_COST` as the default ``rung_cost`` so the bandit's
    tie-break math is calibrated for the effort axis (each step ~3×).

    When ``enable_decompose_top_rung`` is True, the appended decompose
    rung is gated behind ``decompose_min_lower_obs`` observations on the
    next-lower rung. Until that floor is met the bandit only samples
    from rungs 0..N-1, preventing uniform-prior exploration from picking
    the most expensive (and potentially broken) new rung first. This is
    a universal pattern: any newly-introduced top rung benefits from
    waiting for evidence to accumulate on the cheaper rungs first.
    """

    decompose_min_lower_obs: int = 3

    def __post_init__(self) -> None:
        # Default rung_cost to the effort cost map if the caller didn't
        # override (BanditPolicy alone would use the cheap-vs-strong map).
        if self.rung_cost is None:
            self.rung_cost = EFFORT_RUNG_COST
        # Inherit the EffortLadderPolicy advisory-deprecation warning.
        super().__post_init__()

    def _bandit_eligible_rungs(self) -> list[int]:
        all_rungs = list(range(self.max_rung + 1))
        if not self.enable_decompose_top_rung or self.state is None:
            return all_rungs
        decompose_rung = self.max_rung
        lower_rung = decompose_rung - 1
        if lower_rung < 0:
            return all_rungs
        alpha, beta = self.state.beta_for(self.task_key, lower_rung)
        # Beta(1,1) is the uniform prior — observations are (alpha-1)+(beta-1).
        lower_obs = max(0, int(alpha) - 1) + max(0, int(beta) - 1)
        if lower_obs < self.decompose_min_lower_obs:
            return [r for r in all_rungs if r != decompose_rung]
        return all_rungs


__all__ = [
    "EFFORT_RUNG_COST",
    "EffortLadderPolicy",
    "EffortBanditPolicy",
]
