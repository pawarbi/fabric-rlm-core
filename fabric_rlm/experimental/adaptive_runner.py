"""Adaptive meta-controller.

Wraps :class:`fabric_rlm.RLM` with an escalation ladder: when an attempt fails
its validator, the runner consults a :class:`DifficultySignal` and builds a
new :class:`AttemptConfig` (more turns / higher reasoning effort / parallel
rollouts / stronger LM) for the next attempt — bounded by a :class:`Budget`.

Two surfaces:

* ``AdaptiveRunner.run()`` returns an :class:`AdaptiveResult` wrapper for
  power users (full per-attempt logs as live ``RLMResult`` objects).
* The ``RLM(engine='adaptive')`` facade (in :mod:`fabric_rlm.runtime`) returns
  the winning attempt's plain ``RLMResult`` with a compact summary attached at
  ``result.trajectory.metadata['adaptive']``.
"""

from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .adaptive_policy import (
    AttemptConfig,
    AttemptRecord,
    Budget,
    DifficultyVerdict,
    LadderPolicy,
    ValidationVerdict,
    as_verdict,
    inject_feedback,
    select_best_of_n,
)


@dataclass
class AdaptiveResult:
    """Power-user return type from :meth:`AdaptiveRunner.run`.

    ``winner`` is the chosen attempt (the one whose ``result`` is what the
    facade returns to ``RLM(engine='adaptive').run()`` callers).
    """

    winner: AttemptRecord
    attempts: list[AttemptRecord]
    passed: bool
    stop_reason: str
    elapsed_seconds: float

    @property
    def result(self) -> Any:
        """The winning ``RLMResult``."""
        return self.winner.result


class AdaptiveRunner:
    """Meta-controller that escalates an underlying ``RLM`` until it passes.

    Parameters
    ----------
    rlm_factory
        Builds an ``RLM`` from an :class:`AttemptConfig`. The factory is
        invoked once per attempt; reusing one ``RLM`` across attempts is
        unsafe because ``RLM.run()`` mutates internal state.
    policy
        Escalation policy. The default :class:`LadderPolicy` climbs cheap →
        expensive in cost order.
    budget
        Hard limits on attempts/turns/parallelism, plus a best-effort wall
        clock. See :class:`Budget`.
    validator
        Returns ``True``/``False`` or a :class:`ValidationVerdict` for each
        attempt's ``RLMResult``. Bare ``bool`` is auto-adapted.
    on_attempt
        Optional callback invoked after every attempt with the recorded
        :class:`AttemptRecord`.
    feedback_injection
        When True (default), prepends a documented feedback block to the
        first textual input field of the next attempt when the prior
        validator returned a ``feedback`` string.
    prefer_shorter_traces
        SRLM Feature A — when True, :func:`select_best_of_n` adds a late-tier
        trace-length tie-breaker after passed/score/confidence/completeness.
        Off by default; default behavior is byte-identical to before. See
        :func:`select_best_of_n` docstring for the full sort key.
    prefer_consensus
        SRLM Feature C — when True, :func:`select_best_of_n` inserts a
        cluster-size tie-breaker after completeness (and before any trace-
        length slot). Off by default; default behavior is byte-identical
        to before. Scoped to rung-3 best-of-N selection ONLY (same scoping
        rule as Feature A — see ``_best_partial`` comment).
    consensus_answer_keys
        Payload field names to scan for the canonical answer when
        ``prefer_consensus`` is enabled. Defaults match
        :class:`AnswerConsensus` (``"answer", "output", "result", "report"``).
    """

    def __init__(
        self,
        rlm_factory: Callable[[AttemptConfig], Any],
        *,
        policy: LadderPolicy | None = None,
        budget: Budget | None = None,
        validator: Callable[[Any], Any] | None = None,
        on_attempt: Callable[[AttemptRecord], None] | None = None,
        feedback_injection: bool = True,
        pre_run: Callable[[Mapping[str, Any], "AdaptiveRunner"], None] | None = None,
        prefer_shorter_traces: bool = False,
        prefer_consensus: bool = False,
        consensus_answer_keys: tuple[str, ...] = ("answer",),
        early_exit_probe: bool = False,
    ):
        self.rlm_factory = rlm_factory
        self.policy = policy or LadderPolicy()
        self.budget = budget or Budget()
        self.validator = validator or _default_validator()
        self.on_attempt = on_attempt
        self.feedback_injection = feedback_injection
        # One-shot hook fired before the first decision. Used by the task
        # classifier integration to seed bandit priors from a single LM call,
        # but generic — any caller can plug in custom warm-up logic.
        self.pre_run = pre_run
        self.prefer_shorter_traces = prefer_shorter_traces
        self.prefer_consensus = prefer_consensus
        self.consensus_answer_keys = tuple(consensus_answer_keys)
        # SRLM Feature E (early-exit probe). When True, the rung-3
        # best-of-N step uses a probe-then-fanout pattern: launch one
        # candidate (rollout_index 0), await its result, and skip
        # launching the remaining N-1 if the probe passes the validator.
        # Otherwise launch the suffix in parallel as before. Off by
        # default; default behavior is byte-identical to before. See
        # ``_run_rollouts`` for the contract.
        self.early_exit_probe = early_exit_probe

    def run(self, inputs: Mapping[str, Any] | None = None, **kwargs: Any) -> AdaptiveResult:
        run_inputs = dict(inputs or {})
        attempts: list[AttemptRecord] = []
        turns_used_so_far = 0
        wall_start = time.perf_counter()

        if self.pre_run is not None:
            try:
                self.pre_run(run_inputs, self)
            except Exception:
                # pre_run is best-effort — failures must not derail the run
                pass

        verdict, next_config = self.policy.next_decision(attempts)
        stop_reason = ""

        while next_config is not None:
            if len(attempts) >= self.budget.max_attempts:
                stop_reason = "budget: max_attempts reached"
                break
            if self.budget.max_wall_seconds is not None:
                if (time.perf_counter() - wall_start) >= self.budget.max_wall_seconds:
                    stop_reason = "budget: max_wall_seconds reached"
                    break

            cfg = self._apply_budget(next_config, turns_used_so_far)
            if cfg.max_turns <= 0:
                stop_reason = "budget: no turn budget left"
                break

            this_inputs = self._with_feedback(run_inputs, attempts)

            if cfg.parallel_rollouts > 1:
                rollout_records = self._run_rollouts(
                    cfg, this_inputs, kwargs, wall_start=wall_start,
                )
                # record every rollout
                for rec in rollout_records:
                    attempts.append(rec)
                    turns_used_so_far += rec.turns_used
                    if self.on_attempt is not None:
                        try:
                            self.on_attempt(rec)
                        except Exception:
                            pass
                # selection — best becomes the "current" tail
                winner = select_best_of_n(
                    rollout_records,
                    prefer_shorter_traces=self.prefer_shorter_traces,
                    prefer_consensus=self.prefer_consensus,
                    consensus_answer_keys=self.consensus_answer_keys,
                )
                if winner.verdict.passed:
                    # SRLM Feature E (early-exit probe): if the rollout
                    # batch came back with fewer than the configured N
                    # rollouts AND the flag is on, the suffix was skipped
                    # because the probe passed. Emit a distinct stop_reason
                    # so offline replay can attribute savings correctly.
                    if (
                        self.early_exit_probe
                        and len(rollout_records) < cfg.parallel_rollouts
                    ):
                        sr = "early-exit: probe passed"
                    else:
                        sr = "best-of-N rollout passed"
                    return self._make_result(
                        winner=winner,
                        attempts=attempts,
                        stop_reason=sr,
                        wall_start=wall_start,
                    )
            else:
                rec = self._run_one(cfg, 0, this_inputs, kwargs)
                attempts.append(rec)
                turns_used_so_far += rec.turns_used
                if self.on_attempt is not None:
                    try:
                        self.on_attempt(rec)
                    except Exception:
                        pass
                if rec.verdict.passed:
                    return self._make_result(
                        winner=rec,
                        attempts=attempts,
                        stop_reason="validator passed",
                        wall_start=wall_start,
                    )

            verdict, next_config = self.policy.next_decision(attempts)
            if next_config is None and verdict.action == "stop_pass":
                # policy says we're done and last attempt passed
                last_passing = next(
                    (a for a in reversed(attempts) if a.verdict.passed), None
                )
                if last_passing is not None:
                    return self._make_result(
                        winner=last_passing,
                        attempts=attempts,
                        stop_reason="policy: stop_pass",
                        wall_start=wall_start,
                    )
            if next_config is None:
                stop_reason = f"policy: {verdict.action} ({verdict.reason})"
                break

        # Exhausted — return best partial result
        if not attempts:
            raise RuntimeError(
                "AdaptiveRunner.run produced no attempts; check budget and policy"
            )
        winner = self._best_partial(attempts)
        return self._make_result(
            winner=winner,
            attempts=attempts,
            stop_reason=stop_reason or "exhausted without passing",
            wall_start=wall_start,
        )

    # ---- internals --------------------------------------------------------

    def _apply_budget(self, cfg: AttemptConfig, turns_used: int) -> AttemptConfig:
        clamped_turns = self.budget.clamp_turns(cfg.max_turns, turns_used)
        capped_parallel = self.budget.cap_parallel(cfg.parallel_rollouts)
        if clamped_turns == cfg.max_turns and capped_parallel == cfg.parallel_rollouts:
            return cfg
        # AttemptConfig is frozen — replace via dataclasses.replace pattern
        from dataclasses import replace

        return replace(
            cfg,
            max_turns=clamped_turns,
            parallel_rollouts=capped_parallel,
        )

    def _with_feedback(
        self,
        inputs: Mapping[str, Any],
        attempts: list[AttemptRecord],
    ) -> dict[str, Any]:
        if not self.feedback_injection or not attempts:
            return dict(inputs)
        # Ablation switch: when PVR is disabled, fall back to the legacy
        # "validator-feedback only" behavior so the A/B comparison is clean.
        # In `reflect_only` mode we keep synthesis ON (that *is* the point).
        mode = os.environ.get("FABRIC_RLM_PVR_MODE", "").strip().lower()
        if mode in ("full", "off", "reflect_only"):
            pvr_enabled = mode != "off"
        else:
            pvr_enabled = os.environ.get("FABRIC_RLM_PVR", "1") == "1"
        last = attempts[-1]
        if last.verdict.passed:
            return dict(inputs)

        # REFLECT: synthesize feedback even for non-validator failures (worker
        # timeout, exception, no-submit) so the next attempt sees what went
        # wrong rather than starting blind. Disabled when FABRIC_RLM_PVR=0.
        feedback = last.verdict.feedback
        if not feedback:
            if not pvr_enabled:
                return dict(inputs)
            payload = getattr(last.result, "payload", None)
            failure_reason = getattr(last.result, "failure_reason", None)
            submitted = bool(payload)
            if not submitted and failure_reason:
                feedback = f"Previous attempt did not submit (reason: {failure_reason})."
            elif not submitted:
                feedback = "Previous attempt did not submit a payload."
            elif failure_reason:
                feedback = f"Previous attempt was rejected (reason: {failure_reason})."
            else:
                feedback = "Previous attempt was rejected by the validator."

        prior_payload = getattr(last.result, "payload", None)
        return inject_feedback(
            inputs,
            feedback=feedback,
            prior_payload=prior_payload,
            rung=last.rung,
            reasoning_effort=getattr(last.config, "reasoning_effort", None),
            submitted=bool(prior_payload),
        )

    def _run_one(
        self,
        cfg: AttemptConfig,
        rollout_index: int,
        inputs: Mapping[str, Any],
        run_kwargs: dict[str, Any],
    ) -> AttemptRecord:
        from dataclasses import replace

        rollout_cfg = replace(cfg, rollout_index=rollout_index)
        t0 = time.perf_counter()
        try:
            rlm = self.rlm_factory(rollout_cfg)
            result = rlm.run(inputs, **run_kwargs)
            verdict = as_verdict(self.validator(result))
        except Exception as exc:  # noqa: BLE001 — surface every error as failed verdict
            elapsed = time.perf_counter() - t0
            return AttemptRecord(
                rung=rollout_cfg.rung,
                rollout_index=rollout_index,
                config=rollout_cfg,
                result=_FailedResult(reason=repr(exc)),
                verdict=ValidationVerdict(
                    passed=False, feedback=f"factory or run raised: {exc!r}"
                ),
                elapsed_seconds=elapsed,
                turns_used=0,
            )
        elapsed = time.perf_counter() - t0
        traj = getattr(result, "trajectory", None)
        turns_used = len(getattr(traj, "turns", []) or []) if traj is not None else 0
        return AttemptRecord(
            rung=rollout_cfg.rung,
            rollout_index=rollout_index,
            config=rollout_cfg,
            result=result,
            verdict=verdict,
            elapsed_seconds=elapsed,
            turns_used=turns_used,
            prompt_tokens=getattr(result, "total_prompt_tokens", None),
            completion_tokens=getattr(result, "total_completion_tokens", None),
            cached_tokens=getattr(result, "total_cached_tokens", None),
            reasoning_tokens=getattr(result, "total_reasoning_tokens", None),
        )

    def _run_rollouts(
        self,
        cfg: AttemptConfig,
        inputs: Mapping[str, Any],
        run_kwargs: dict[str, Any],
        *,
        wall_start: float | None = None,
    ) -> list[AttemptRecord]:
        """Run ``cfg.parallel_rollouts`` rollouts and return their records.

        Default behavior: launch all N futures at once and collect them.

        When ``self.early_exit_probe`` is True and ``n > 1``, use a
        probe-then-fanout pattern:

        1. Launch ONE rollout (``rollout_index=0``) and await it.
        2. If its validator verdict passes, return ``[probe_record]``
           and skip launching the remaining N-1. This is the all_pass
           predicate at K=1, empirically validated by
           ``bench/adaptive/_prefix_replay.py`` to fire on 35% of
           captured rung-3 rollouts with 0/196 pass-flips.
        3. Otherwise, re-check the wall budget (the probe may have
           burned the remaining time); if exceeded, return just the
           probe so the caller can record it and stop. Else launch
           the remaining N-1 rollouts in parallel as before and
           return all N records.

        The contract is **pass/fail preservation only**. The selected
        winner identity (and therefore reported trace length, answer
        text, and downstream cost analyses) may differ from the
        full-fanout selection, especially when Feature A
        (``prefer_shorter_traces``) is enabled. Document accordingly.
        """
        n = cfg.parallel_rollouts
        # safety: if user provided pre-resolved lm_instance, downgrade to sequential
        if cfg.lm_instance is not None and cfg.lm_spec is None and n > 1:
            warnings.warn(
                "AdaptiveRunner: lm_instance is not safely shareable across "
                "rollouts; downgrading parallel_rollouts to sequential. "
                "Pass lm_spec for parallel best-of-N.",
                UserWarning,
                stacklevel=2,
            )
            return [self._run_one(cfg, i, inputs, run_kwargs) for i in range(n)]

        if self.early_exit_probe and n > 1:
            return self._run_rollouts_with_probe(
                cfg, inputs, run_kwargs, wall_start=wall_start,
            )

        records: list[AttemptRecord] = [None] * n  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {
                ex.submit(self._run_one, cfg, i, inputs, run_kwargs): i for i in range(n)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                records[idx] = fut.result()
        return records

    def _run_rollouts_with_probe(
        self,
        cfg: AttemptConfig,
        inputs: Mapping[str, Any],
        run_kwargs: dict[str, Any],
        *,
        wall_start: float | None,
    ) -> list[AttemptRecord]:
        """Probe-then-fanout implementation of Feature E.

        See ``_run_rollouts`` docstring for the contract.
        """
        n = cfg.parallel_rollouts
        # 1. Probe (rollout_index 0). Run inline — no executor needed for one
        # task, and this avoids a thread-pool round-trip on the fast path.
        probe = self._run_one(cfg, 0, inputs, run_kwargs)
        if probe.verdict.passed:
            # All-pass predicate fired at K=1: skip suffix entirely.
            return [probe]

        # 2. Probe failed. Re-check wall budget before launching the suffix
        # so a slow probe near the deadline can't push us over budget by
        # launching N-1 more rollouts. (Other budget gates — max_attempts
        # and turn budget — are enforced by the outer ``run`` loop using
        # ``len(attempts)`` and ``turns_used_so_far``; the wall clock is
        # the only one that can change DURING a rollout batch.)
        if wall_start is not None and self.budget.max_wall_seconds is not None:
            if (time.perf_counter() - wall_start) >= self.budget.max_wall_seconds:
                return [probe]

        # 3. Launch suffix in parallel.
        records: list[AttemptRecord] = [probe]
        suffix_n = n - 1
        records.extend([None] * suffix_n)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=suffix_n) as ex:
            futures = {
                ex.submit(self._run_one, cfg, i, inputs, run_kwargs): i
                for i in range(1, n)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                records[idx] = fut.result()
        return records

    def _best_partial(self, attempts: list[AttemptRecord]) -> AttemptRecord:
        # No passing attempt — pick the most-populated payload, breaking ties
        # by lowest (rung, rollout_index) for determinism.
        #
        # Note: ``prefer_shorter_traces`` and ``prefer_consensus`` are
        # intentionally NOT forwarded here. SRLM Features A and C are scoped
        # to rung-3 best-of-N selection only (the late tie-breakers after a
        # parallel rollout batch). Forwarding them to exhausted-run selection
        # would let the flags silently change which failed attempt is
        # reported across all rungs — contradicting the "rung-3 BoN tie-break
        # only" design intent and contaminating comparisons of winner_rung /
        # answer / cost on failed runs.
        return select_best_of_n(attempts)

    def _make_result(
        self,
        *,
        winner: AttemptRecord,
        attempts: list[AttemptRecord],
        stop_reason: str,
        wall_start: float,
    ) -> AdaptiveResult:
        elapsed = time.perf_counter() - wall_start
        # attach compact summary to winner.result.trajectory.metadata['adaptive']
        traj = getattr(winner.result, "trajectory", None)
        if traj is not None and hasattr(traj, "metadata"):
            try:
                traj.metadata["adaptive"] = {
                    "stop_reason": stop_reason,
                    "elapsed_seconds": elapsed,
                    "winner_rung": winner.rung,
                    "winner_rollout_index": winner.rollout_index,
                    "attempts": [a.to_summary() for a in attempts],
                }
            except Exception:
                pass
        return AdaptiveResult(
            winner=winner,
            attempts=attempts,
            passed=winner.verdict.passed,
            stop_reason=stop_reason,
            elapsed_seconds=elapsed,
        )


# --- helpers / fallbacks ---------------------------------------------------


def _default_validator() -> Callable[[Any], bool]:
    """Importable shim around :func:`adaptive.universal_validator`."""

    from .adaptive import universal_validator

    return universal_validator()


@dataclass
class _FailedResult:
    """Sentinel used when the factory or `.run()` raises.

    Mimics the public surface of :class:`fabric_rlm.RLMResult` so it can flow
    through any code that pattern-matches on attribute access (``.payload``,
    ``.trajectory``, ``.failure_reason``). Carries a real ``Trajectory`` so
    that ``_make_result`` can still attach the adaptive metadata block when
    the winning attempt is itself a failure.
    """

    reason: str
    submitted: bool = False
    payload: Any = None
    failure_reason: str = field(init=False)
    trajectory: Any = field(default=None)
    final_state: dict[str, Any] = field(default_factory=dict)
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_reasoning_tokens: int | None = None
    total_lm_seconds: float | None = None
    total_worker_seconds: float | None = None

    def __post_init__(self) -> None:
        self.failure_reason = self.reason
        if self.trajectory is None:
            # Lazy import to avoid a circular import at module load.
            from fabric_rlm.trajectory import Trajectory

            self.trajectory = Trajectory(metadata={"failed": True, "reason": self.reason})


def _failed_to_rlm_result(failed: "_FailedResult") -> Any:
    """Convert a ``_FailedResult`` to a real :class:`RLMResult`.

    Used at the ``RLM(engine='adaptive')`` boundary so callers always get a
    uniform ``RLMResult`` instance, never the internal sentinel.
    """
    from fabric_rlm.runtime import RLMResult

    return RLMResult(
        submitted=False,
        payload=None,
        trajectory=failed.trajectory,
        final_state=dict(failed.final_state),
        failure_reason=failed.failure_reason,
        total_prompt_tokens=failed.total_prompt_tokens,
        total_completion_tokens=failed.total_completion_tokens,
        total_cached_tokens=failed.total_cached_tokens,
        total_reasoning_tokens=failed.total_reasoning_tokens,
        total_lm_seconds=failed.total_lm_seconds,
        total_worker_seconds=failed.total_worker_seconds,
    )


__all__ = ["AdaptiveResult", "AdaptiveRunner"]
