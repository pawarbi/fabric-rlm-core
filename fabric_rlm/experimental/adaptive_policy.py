"""Adaptive engine policy primitives.

Defines the small data types and pluggable interfaces the meta-controller
(:mod:`fabric_rlm.experimental.adaptive_runner`) uses to decide whether and
how to escalate when an attempt fails its validator.

Everything in this module is opt-in and stable enough to import from outside
the package — but the module itself remains under ``experimental/`` because
the broader adaptive runtime is still being validated against real benchmarks
(see ``bench/adaptive/``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, runtime_checkable

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationVerdict:
    """Richer validator return type.

    A bare ``Callable[[RLMResult], bool]`` validator is still accepted by the
    runner; it is adapted into ``ValidationVerdict(passed=bool)`` via
    :func:`as_verdict`.
    """

    passed: bool
    score: float | None = None
    confidence: float | None = None
    feedback: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


def as_verdict(value: Any) -> ValidationVerdict:
    """Coerce a validator return value into a :class:`ValidationVerdict`.

    Accepts:
        * an existing :class:`ValidationVerdict` (returned as-is),
        * a ``bool`` (wrapped as ``ValidationVerdict(passed=bool)``),
        * any other truthy/falsy value (interpreted as bool).
    """

    if isinstance(value, ValidationVerdict):
        return value
    return ValidationVerdict(passed=bool(value))


# ----------------------------------------------------------------------------
# Per-attempt configuration
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptConfig:
    """What the policy mutates between attempts.

    The runner builds an ``RLM`` from this config for each attempt. ``lm_spec``
    is preferred over ``lm_instance`` because spec-based LMs can be safely
    re-resolved in parallel rollouts and have their reasoning effort adjusted;
    pre-resolved instances cannot.
    """

    rung: int
    rollout_index: int = 0
    max_turns: int = 10
    reasoning_effort: ReasoningEffort | None = None
    lm_spec: Any = None
    lm_instance: Any = None
    parallel_rollouts: int = 1
    inner_engine: str = "v6-custom"
    failure_feedback: str | None = None
    # When True, runtime should bypass the standard inner RLM loop and
    # invoke decompose_then_synthesize against the raw question instead.
    # Universal building block — task-agnostic — see SPEC-decompose-rung.md.
    decompose_phase: bool = False
    decompose_max_subs: int = 6


# ----------------------------------------------------------------------------
# Captured attempt records (the data substrate for offline analysis)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptRecord:
    """One attempt's complete (in-memory) record.

    A compact serializable summary of this is what gets stored in
    ``RLMResult.trajectory.metadata['adaptive']`` for the
    ``engine='adaptive'`` facade. Full ``result`` objects live only on
    :class:`AdaptiveResult.attempts`.
    """

    rung: int
    rollout_index: int
    config: AttemptConfig
    result: Any  # RLMResult; not annotated to avoid an import cycle
    verdict: ValidationVerdict
    elapsed_seconds: float
    turns_used: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    # Free-form per-rollout observability bag. Frozen-dataclass safe: the dict
    # object reference is immutable but its contents are mutable, which is
    # exactly what selectors / signals (SRLM features A-D) need to record
    # diagnostics like selector_key / trace_length_completion / vc_aggregate.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> dict[str, Any]:
        """JSON-serializable summary, safe for trajectory metadata."""

        payload = getattr(self.result, "payload", None)
        # truncate the preview so we never dump 10KB strings into trajectory metadata
        preview: Any
        if isinstance(payload, Mapping):
            preview = {
                k: (v[:200] + "…") if isinstance(v, str) and len(v) > 200 else v
                for k, v in payload.items()
            }
        else:
            preview = None
        return {
            "rung": self.rung,
            "rollout_index": self.rollout_index,
            "passed": self.verdict.passed,
            "score": self.verdict.score,
            "confidence": self.verdict.confidence,
            "feedback": self.verdict.feedback,
            "submitted": bool(getattr(self.result, "submitted", False)),
            "failure_reason": getattr(self.result, "failure_reason", None),
            "turns_used": self.turns_used,
            "elapsed_seconds": self.elapsed_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "max_turns": self.config.max_turns,
            "reasoning_effort": self.config.reasoning_effort,
            "parallel_rollouts": self.config.parallel_rollouts,
            "payload_preview": preview,
            **(
                {"turns": _capture_turns(self.result)}
                if os.environ.get("FABRIC_RLM_CAPTURE_TURNS", "").lower()
                in {"1", "true", "yes", "on"}
                else {}
            ),
        }


def _capture_turns(result: Any) -> list[dict[str, Any]]:
    """Compact per-turn capture for offline analysis (PVR PLAN/VERIFY/REFLECT)."""

    traj = getattr(result, "trajectory", None)
    turns = getattr(traj, "turns", None) if traj is not None else None
    if not turns:
        return []
    out: list[dict[str, Any]] = []
    for t in turns:
        rt = getattr(t, "response_text", "") or ""
        code = getattr(t, "code", "") or ""
        stdout = getattr(t, "stdout", "") or ""
        out.append(
            {
                "turn": getattr(t, "turn", None),
                "turn_type": getattr(t, "turn_type", "normal"),
                "submitted": bool(getattr(t, "submitted", False)),
                "has_error": bool(getattr(t, "error", None)),
                "response_text": (rt[:1500] + "…") if len(rt) > 1500 else rt,
                "code": (code[:800] + "…") if len(code) > 800 else code,
                "stdout": (stdout[:400] + "…") if len(stdout) > 400 else stdout,
            }
        )
    return out


# ----------------------------------------------------------------------------
# Difficulty signal (the trigger)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class DifficultyVerdict:
    """What a :class:`DifficultySignal` decides to do next."""

    action: Literal["stop_pass", "stop_fail", "repeat", "escalate", "fanout"]
    target_rung: int | None = None  # for "escalate"
    rollouts: int | None = None  # for "fanout"
    reason: str = ""

    @classmethod
    def stop_pass(cls, reason: str = "validator passed") -> "DifficultyVerdict":
        return cls(action="stop_pass", reason=reason)

    @classmethod
    def stop_fail(cls, reason: str) -> "DifficultyVerdict":
        return cls(action="stop_fail", reason=reason)

    @classmethod
    def repeat(cls, reason: str = "retry same rung") -> "DifficultyVerdict":
        return cls(action="repeat", reason=reason)

    @classmethod
    def escalate(cls, target_rung: int, reason: str = "") -> "DifficultyVerdict":
        return cls(action="escalate", target_rung=target_rung, reason=reason)

    @classmethod
    def fanout(cls, rollouts: int, reason: str = "") -> "DifficultyVerdict":
        return cls(action="fanout", rollouts=rollouts, reason=reason)


@runtime_checkable
class DifficultySignal(Protocol):
    """Pluggable trigger that decides when (and how) to escalate.

    Implementations inspect the prior attempts and return a
    :class:`DifficultyVerdict`. The :class:`LadderPolicy` consults this signal
    after each attempt; ``ValidatorOnly`` is the default and the only one wired
    into the headline ``engine='adaptive'`` facade.
    """

    def assess(
        self,
        attempts: list[AttemptRecord],
        max_rung: int,
    ) -> DifficultyVerdict: ...


@dataclass(frozen=True)
class ValidatorOnly:
    """Default trigger.

    ``DifficultyVerdict.escalate(next_rung)`` iff the last attempt's validator
    failed; otherwise ``stop_pass``. No heuristics. Provably correct about
    *whether* the previous attempt was good (that's by definition what the
    validator decides) — it can only be wrong about *which* rung to climb,
    and the ladder order itself encodes a cost-aware default.
    """

    def assess(
        self,
        attempts: list[AttemptRecord],
        max_rung: int,
    ) -> DifficultyVerdict:
        if not attempts:
            return DifficultyVerdict.escalate(0, reason="no attempts yet")
        last = attempts[-1]
        if last.verdict.passed:
            return DifficultyVerdict.stop_pass()
        next_rung = last.rung + 1
        if next_rung > max_rung:
            return DifficultyVerdict.stop_fail(reason="ladder exhausted")
        return DifficultyVerdict.escalate(
            next_rung, reason=f"validator failed at rung {last.rung}"
        )


@dataclass(frozen=True)
class Confidence:
    """Trigger on self-reported confidence.

    Requires the caller to wrap their signature with ``with_self_report`` so
    the model emits a ``confidence`` field. Low confidence escalates *effort*
    (rung 2) instead of just adding turns.
    """

    threshold: float = 0.7

    def assess(
        self,
        attempts: list[AttemptRecord],
        max_rung: int,
    ) -> DifficultyVerdict:
        if not attempts:
            return DifficultyVerdict.escalate(0, reason="no attempts yet")
        last = attempts[-1]
        if last.verdict.passed:
            return DifficultyVerdict.stop_pass()
        payload = getattr(last.result, "payload", None) or {}
        try:
            conf = float(payload.get("confidence")) if isinstance(payload, Mapping) else None
        except (TypeError, ValueError):
            conf = None
        if conf is not None and conf < self.threshold:
            target = max(2, last.rung + 1)
            if target > max_rung:
                return DifficultyVerdict.stop_fail(reason="ladder exhausted")
            return DifficultyVerdict.escalate(
                target, reason=f"low confidence ({conf:.2f}) -> jump to effort rung"
            )
        # fall back to ValidatorOnly behavior
        return ValidatorOnly().assess(attempts, max_rung)


@dataclass(frozen=True)
class NoSubmit:
    """Distinguish 'ran out of turns' from 'tool-error spiral'.

    * ``submitted=False`` + ``failure_reason in {"max_turns", None}`` ⇒ rung 1
      (more turns is the right escalation).
    * ``submitted=False`` + tool-error-flavored failure ⇒ jump to rung 2 (raise
      effort) or rung 4 (stronger model) on repeat offence.
    """

    def assess(
        self,
        attempts: list[AttemptRecord],
        max_rung: int,
    ) -> DifficultyVerdict:
        if not attempts:
            return DifficultyVerdict.escalate(0, reason="no attempts yet")
        last = attempts[-1]
        if last.verdict.passed:
            return DifficultyVerdict.stop_pass()
        submitted = bool(getattr(last.result, "submitted", False))
        reason = (getattr(last.result, "failure_reason", "") or "").lower()
        if not submitted:
            if reason in ("", "max_turns"):
                target = max(1, last.rung + 1)
            elif reason == "stuck_loop":
                # NEW-H: an LM that re-emitted identical failing code N times
                # is unlikely to break out by getting MORE of the same turns.
                # Skip past rung-1 (more_turns) and raise effort/diversity.
                target = max(2, last.rung + 1)
            else:
                # tool-error-y reason — skip past more_turns
                target = max(2, last.rung + 1)
            if target > max_rung:
                return DifficultyVerdict.stop_fail(reason="ladder exhausted")
            return DifficultyVerdict.escalate(
                target, reason=f"non-submit ({reason or 'unknown'})"
            )
        return ValidatorOnly().assess(attempts, max_rung)


@dataclass(frozen=True)
class AnswerConsensus:
    """Look at agreement across prior attempts.

    Same wrong answer twice ⇒ fan out for diversity. Different wrong answers
    each time ⇒ switch to a stronger model.
    """

    answer_keys: tuple[str, ...] = ("answer", "output", "result", "report")
    fanout_n: int = 3

    def _answer_of(self, record: AttemptRecord) -> Any:
        payload = getattr(record.result, "payload", None) or {}
        if not isinstance(payload, Mapping):
            return None
        for k in self.answer_keys:
            if k in payload:
                return payload[k]
        return None

    def assess(
        self,
        attempts: list[AttemptRecord],
        max_rung: int,
    ) -> DifficultyVerdict:
        if not attempts:
            return DifficultyVerdict.escalate(0, reason="no attempts yet")
        last = attempts[-1]
        if last.verdict.passed:
            return DifficultyVerdict.stop_pass()
        if len(attempts) >= 2:
            answers = [self._answer_of(a) for a in attempts]
            unique = {repr(a) for a in answers}
            if len(unique) == 1:
                return DifficultyVerdict.fanout(
                    self.fanout_n,
                    reason="repeated identical wrong answer -> diversify",
                )
            if len(unique) == len(answers):
                # every attempt produced a different answer — switch model
                target = min(4, max_rung)
                return DifficultyVerdict.escalate(
                    target, reason="answers diverging -> stronger model"
                )
        return ValidatorOnly().assess(attempts, max_rung)


@dataclass(frozen=True)
class ToolErrorRate:
    """Repeated tool misuse (errors in trajectory turns) ⇒ raise effort."""

    error_rate_threshold: float = 0.3

    def assess(
        self,
        attempts: list[AttemptRecord],
        max_rung: int,
    ) -> DifficultyVerdict:
        if not attempts:
            return DifficultyVerdict.escalate(0, reason="no attempts yet")
        last = attempts[-1]
        if last.verdict.passed:
            return DifficultyVerdict.stop_pass()
        traj = getattr(last.result, "trajectory", None)
        turns = list(getattr(traj, "turns", []) or []) if traj is not None else []
        if turns:
            errors = sum(1 for t in turns if getattr(t, "error", None))
            rate = errors / len(turns)
            if rate >= self.error_rate_threshold:
                target = max(2, last.rung + 1)
                if target > max_rung:
                    return DifficultyVerdict.stop_fail(reason="ladder exhausted")
                return DifficultyVerdict.escalate(
                    target, reason=f"tool-error rate {rate:.2f} -> raise effort"
                )
        return ValidatorOnly().assess(attempts, max_rung)


@dataclass(frozen=True)
class Ensemble:
    """Combine multiple signals with a documented voting rule.

    Each signal votes; the **highest target_rung** among ``escalate`` votes
    wins (most aggressive policy). If any signal says ``stop_pass``, we stop.
    If none say escalate or fanout, we ``stop_fail``.
    """

    signals: tuple[DifficultySignal, ...] = (
        ValidatorOnly(),
        Confidence(),
        NoSubmit(),
    )

    def assess(
        self,
        attempts: list[AttemptRecord],
        max_rung: int,
    ) -> DifficultyVerdict:
        verdicts = [s.assess(attempts, max_rung) for s in self.signals]
        if any(v.action == "stop_pass" for v in verdicts):
            return DifficultyVerdict.stop_pass()
        fanouts = [v for v in verdicts if v.action == "fanout"]
        if fanouts:
            best = max(fanouts, key=lambda v: v.rollouts or 0)
            return best
        escalations = [v for v in verdicts if v.action == "escalate"]
        if escalations:
            best = max(escalations, key=lambda v: v.target_rung or -1)
            return best
        return DifficultyVerdict.stop_fail(reason="no signal asked for more compute")


# ----------------------------------------------------------------------------
# Escalation policy (the action)
# ----------------------------------------------------------------------------


@dataclass
class LadderPolicy:
    """Default escalation policy.

    Climbs rungs in the user-confirmed cost order. Each rung produces an
    :class:`AttemptConfig` with one knob raised relative to the baseline.

    Rungs:
        0. baseline (as configured)
        1. ``max_turns *= 2``
        2. ``reasoning_effort`` step up (low -> medium -> high)
        3. ``parallel_rollouts = N`` (best-of-N selected by validator)
        4. ``lm_spec = strong_lm`` (caller-supplied)
    """

    base_max_turns: int = 10
    base_reasoning_effort: ReasoningEffort | None = None
    base_lm_spec: Any = None
    base_lm_instance: Any = None
    base_inner_engine: str = "v6-custom"
    strong_lm_spec: Any = None
    parallel_rollouts: int = 3
    signal: DifficultySignal = field(default_factory=ValidatorOnly)
    skip_more_turns_when_submitted: bool = True

    @property
    def max_rung(self) -> int:
        return 4 if self.strong_lm_spec is not None else 3

    def baseline_config(self) -> AttemptConfig:
        return AttemptConfig(
            rung=0,
            max_turns=self.base_max_turns,
            reasoning_effort=self.base_reasoning_effort,
            lm_spec=self.base_lm_spec,
            lm_instance=self.base_lm_instance,
            inner_engine=self.base_inner_engine,
        )

    def next_decision(
        self,
        attempts: list[AttemptRecord],
    ) -> tuple[DifficultyVerdict, AttemptConfig | None]:
        """Consult the signal and translate its verdict into an attempt config.

        Returns ``(verdict, config_or_none)``. ``config_or_none`` is ``None``
        when the verdict is ``stop_pass``/``stop_fail``.
        """

        verdict = self.signal.assess(attempts, self.max_rung)
        if verdict.action in ("stop_pass", "stop_fail"):
            return verdict, None

        last = attempts[-1] if attempts else None

        if verdict.action == "repeat":
            cfg = last.config if last is not None else self.baseline_config()
            return verdict, cfg

        if verdict.action == "fanout":
            rung = max(3, (last.rung if last else 0))
            return verdict, self._build_config(
                rung,
                rollouts=verdict.rollouts or self.parallel_rollouts,
                feedback=self._feedback_for(last),
            )

        # escalate
        target = verdict.target_rung if verdict.target_rung is not None else 0
        # Skip rung 1 when the prior attempt was submitted=True with no feedback —
        # just rerunning more_turns will have the model resubmit the same wrong
        # answer. Bump straight to rung 2 (raise effort) instead.
        if (
            target == 1
            and self.skip_more_turns_when_submitted
            and last is not None
            and bool(getattr(last.result, "submitted", False))
            and not last.verdict.feedback
        ):
            target = 2
        target = min(target, self.max_rung)
        return verdict, self._build_config(
            target, feedback=self._feedback_for(last)
        )

    def _feedback_for(self, last: AttemptRecord | None) -> str | None:
        if last is None:
            return None
        return last.verdict.feedback

    def _build_config(
        self,
        rung: int,
        *,
        rollouts: int = 1,
        feedback: str | None = None,
    ) -> AttemptConfig:
        cfg = self.baseline_config()
        max_turns = cfg.max_turns
        reasoning_effort = cfg.reasoning_effort
        lm_spec = cfg.lm_spec
        lm_instance = cfg.lm_instance
        parallel = 1

        if rung >= 1:
            max_turns = cfg.max_turns * 2
        if rung >= 2:
            reasoning_effort = _next_effort(reasoning_effort)
        if rung >= 3:
            parallel = max(rollouts, self.parallel_rollouts)
        if rung >= 4 and self.strong_lm_spec is not None:
            lm_spec = self.strong_lm_spec
            lm_instance = None  # strong-LM rung always re-resolves from spec

        return AttemptConfig(
            rung=rung,
            max_turns=max_turns,
            reasoning_effort=reasoning_effort,
            lm_spec=lm_spec,
            lm_instance=lm_instance,
            inner_engine=cfg.inner_engine,
            parallel_rollouts=parallel,
            failure_feedback=feedback,
        )


_EFFORT_ORDER: tuple[ReasoningEffort, ...] = ("minimal", "low", "medium", "high")


def _next_effort(current: ReasoningEffort | None) -> ReasoningEffort:
    if current is None:
        return "medium"
    try:
        idx = _EFFORT_ORDER.index(current)
    except ValueError:
        return "medium"
    return _EFFORT_ORDER[min(idx + 1, len(_EFFORT_ORDER) - 1)]


# ----------------------------------------------------------------------------
# Budget guardrails
# ----------------------------------------------------------------------------


@dataclass
class Budget:
    """Bounds on adaptive execution.

    Hard guarantees:
        * ``max_attempts`` enforced before scheduling each attempt.
        * Per-attempt ``max_turns`` clamped to remaining ``max_total_turns``.
        * Parallel rollouts are reserved upfront against remaining budget.

    Soft (best-effort):
        * ``max_wall_seconds`` is checked between attempts only — Python
          threads cannot be force-killed. Each inner ``RLM`` should set its own
          timeout if hard wall-time matters.

    Default ``max_attempts=5`` means rungs 0..4 are reachable when
    ``strong_lm`` is configured; otherwise rung 4 is skipped automatically.
    """

    max_attempts: int = 5
    max_total_turns: int | None = None
    max_parallel: int = 4
    max_wall_seconds: float | None = None
    compute_optimal: bool = False

    def clamp_turns(self, planned_max_turns: int, turns_used_so_far: int) -> int:
        if self.max_total_turns is None:
            return planned_max_turns
        remaining = max(0, self.max_total_turns - turns_used_so_far)
        return min(planned_max_turns, remaining)

    def cap_parallel(self, planned: int) -> int:
        return max(1, min(planned, self.max_parallel))


# ----------------------------------------------------------------------------
# Feedback-injection — used by the runner to wire prior failure into next attempt
# ----------------------------------------------------------------------------


FEEDBACK_BLOCK_HEADER = "## PRIOR_ATTEMPT_FEEDBACK"
FEEDBACK_BLOCK_FOOTER = "## END_PRIOR_ATTEMPT_FEEDBACK"


def render_feedback_block(
    feedback: str,
    prior_payload: Any,
    *,
    rung: int | None = None,
    reasoning_effort: str | None = None,
    submitted: bool | None = None,
) -> str:
    """Build the REFLECT block prepended to the next attempt's input.

    Recognised by the always-on ``core.md`` skill. Includes prior-attempt
    metadata (rung, effort, submission status) so the model can reason
    about what changed between attempts. Payload preview is truncated.
    """

    preview: str
    try:
        if prior_payload is None:
            preview = "<no payload — attempt did not submit>"
        elif isinstance(prior_payload, Mapping):
            items = list(prior_payload.items())[:5]
            preview = "; ".join(
                f"{k}={_short(v)}" for k, v in items
            )
        else:
            preview = _short(prior_payload)
    except Exception:
        preview = "<unserializable>"

    meta_bits: list[str] = []
    if rung is not None:
        meta_bits.append(f"rung={rung}")
    if reasoning_effort:
        meta_bits.append(f"effort={reasoning_effort}")
    if submitted is not None:
        meta_bits.append(f"submitted={submitted}")
    meta_line = (" (" + ", ".join(meta_bits) + ")") if meta_bits else ""

    return (
        f"{FEEDBACK_BLOCK_HEADER}\n"
        f"A previous attempt{meta_line} failed. Use this to choose a different approach.\n"
        f"- Reason: {feedback}\n"
        f"- Submitted payload: {preview}\n"
        f"- Guidance: Do not repeat the same payload or the same line of reasoning. "
        f"Re-decompose in your PLAN block; address the failure reason directly.\n"
        f"{FEEDBACK_BLOCK_FOOTER}\n\n"
    )


def _short(value: Any, limit: int = 200) -> str:
    s = repr(value)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def inject_feedback(
    inputs: Mapping[str, Any],
    feedback: str,
    prior_payload: Any,
    *,
    rung: int | None = None,
    reasoning_effort: str | None = None,
    submitted: bool | None = None,
) -> dict[str, Any]:
    """Return a new inputs dict with a feedback block prepended to the first
    text-typed input field. If no text field exists, the block is added under a
    new ``_adaptive_feedback`` key.
    """

    block = render_feedback_block(
        feedback,
        prior_payload,
        rung=rung,
        reasoning_effort=reasoning_effort,
        submitted=submitted,
    )
    out: dict[str, Any] = dict(inputs)
    text_key: str | None = None
    for k, v in out.items():
        if isinstance(v, str):
            text_key = k
            break
    if text_key is None:
        out["_adaptive_feedback"] = block
    else:
        out[text_key] = block + str(out[text_key])
    return out


# ----------------------------------------------------------------------------
# Best-of-N selection (used by the runner)
# ----------------------------------------------------------------------------


def payload_completeness(
    result: Any,
    required_keys: tuple[str, ...] = (),
) -> tuple[int, int]:
    """Score (required_filled, total_non_blank) for deterministic tie-break."""

    payload = getattr(result, "payload", None)
    if not isinstance(payload, Mapping):
        return (0, 0)
    required_filled = sum(
        1 for k in required_keys if k in payload and not _is_blank(payload[k])
    )
    total_non_blank = sum(1 for v in payload.values() if not _is_blank(v))
    return (required_filled, total_non_blank)


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, tuple, set, frozenset, dict)):
        return len(v) == 0
    return False


def _trace_length(
    record: AttemptRecord,
    mode: Literal["completion", "turns", "step_sum"] = "completion",
) -> int:
    """Provider-independent trace-length signal (shared helper).

    Defined in the SRLM integration plan ("Trace-length definition" section)
    so Features A, B, and D all read the same source of truth and can't drift.

    Modes
    -----
    ``"completion"`` (Feature A — tie-breaker)
        ``record.completion_tokens`` if available, else 0. Excludes prompt
        tokens and provider-specific reasoning tokens. Treats missing as 0
        (no preference / no penalty), so the selector fails safely on
        providers that don't report token counts.
    ``"turns"`` (Feature B — DifficultySignal)
        ``record.turns_used`` (always present).
    ``"step_sum"`` (Feature D — VC × Len)
        Per-step completion-token sum. Today this falls back to
        ``record.completion_tokens`` because per-step counts are not yet
        captured; Feature D may specialize this when the per-step capture
        lands.
    """
    if mode == "turns":
        return int(record.turns_used or 0)
    # "completion" and "step_sum" both use completion_tokens today; "step_sum"
    # is reserved for Feature D specialization (per-step accumulation).
    return int(record.completion_tokens or 0)


def select_best_of_n(
    rollouts: list[AttemptRecord],
    *,
    required_keys: tuple[str, ...] = (),
    prefer_shorter_traces: bool = False,
) -> AttemptRecord:
    """Deterministic best-of-N selection.

    Default sort key (all descending where higher = better, except
    ``rollout_index`` which is ascending so the lowest-indexed rollout wins
    on ultimate tie):

        (passed, score, confidence, required_filled, total_non_blank, -rollout_index)

    SRLM Feature A — opt-in via ``prefer_shorter_traces=True`` — appends a
    LATE-TIER trace-length tie-breaker AFTER all correctness/completeness
    signals and BEFORE ``-rollout_index`` (which remains the ultimate
    determinism tie-break):

        (passed, score, confidence, required_filled, total_non_blank,
         -trace_length_completion, -rollout_index)

    Trace length uses :func:`_trace_length(rec, "completion")` — provider-
    independent and missing-safe (None/0 → no penalty/no preference).

    Late-tier-only is non-negotiable: trace length must never override
    validator-passed, score, confidence, or payload completeness. The
    inverse-failure case (concise wrong answer beats verbose correct answer)
    is gated by the validator already calling them equally correct, which we
    accept as the documented trade-off.

    Plumbing choice: this function takes ``prefer_shorter_traces`` as a
    direct keyword instead of stuffing it into :class:`AttemptConfig`. The
    flag is a property of *how rollouts are compared*, not of any individual
    attempt's execution config, so it lives on the call site that does the
    comparison. :class:`AdaptiveRunner` carries it as a constructor field.

    Observability: every rollout (winner and losers) gets
    ``record.metadata['srlm']['trace_length_completion']`` populated; the
    winner additionally gets ``record.metadata['srlm']['selector_key']`` set
    to the actual sort tuple used.
    """

    if not rollouts:
        raise ValueError("select_best_of_n: empty rollouts")

    def key(rec: AttemptRecord) -> tuple:
        rf, tn = payload_completeness(rec.result, required_keys=required_keys)
        base = (
            int(rec.verdict.passed),
            (rec.verdict.score if rec.verdict.score is not None else float("-inf")),
            (rec.verdict.confidence if rec.verdict.confidence is not None else float("-inf")),
            rf,
            tn,
        )
        if prefer_shorter_traces:
            return (*base, -_trace_length(rec, "completion"), -rec.rollout_index)
        return (*base, -rec.rollout_index)

    keys: list[tuple[tuple, AttemptRecord]] = [(key(r), r) for r in rollouts]

    # Per-rollout observability hook (always recorded, even when flag is off,
    # so callers can inspect what *would* have happened). One-line addition.
    for k, r in keys:
        srlm = r.metadata.setdefault("srlm", {})
        srlm["trace_length_completion"] = _trace_length(r, "completion")

    winner_key, winner = max(keys, key=lambda kr: kr[0])
    winner.metadata.setdefault("srlm", {})["selector_key"] = winner_key
    return winner


__all__ = [
    "AnswerConsensus",
    "AttemptConfig",
    "AttemptRecord",
    "Budget",
    "Confidence",
    "DifficultySignal",
    "DifficultyVerdict",
    "Ensemble",
    "FEEDBACK_BLOCK_FOOTER",
    "FEEDBACK_BLOCK_HEADER",
    "LadderPolicy",
    "NoSubmit",
    "ReasoningEffort",
    "ToolErrorRate",
    "ValidationVerdict",
    "ValidatorOnly",
    "_trace_length",
    "as_verdict",
    "inject_feedback",
    "payload_completeness",
    "render_feedback_block",
    "select_best_of_n",
]


def _stub_callable(fn: Callable[..., Any]) -> Callable[..., Any]:
    # placeholder to keep imports happy; not used externally
    return fn
