"""Core Recursive Language Model driver."""

from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import re
import time
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .interpreter import ExecResult, Interpreter, WorkerProtocolError, WorkerTimeout
from .lakehouse import resolve_lakehouse_inputs
from .security import SecurityPolicy
from .serializers import DEFAULT_MAX_SUBMIT_BYTES, validate_max_submit_bytes

if TYPE_CHECKING:
    from .inspector import RunInspector
    from .knowledge_api import Knowledge, KnowledgeStore
    from .knowledge_sources import ProfileLimits, SourceAdapterRegistry
    from .onelake_knowledge_store import OneLakeKnowledgeTransport


# ---------------------------------------------------------------------------
# Stuck-loop detection helpers (NEW-H)
# ---------------------------------------------------------------------------
#
# Trace mining of v4 SSB runs surfaced trajectories where the LM emits the
# SAME failing code with the SAME error type 3+ times in a row, then exhausts
# the turn budget. Even on gpt-5 the count is small (~2/399), but when it
# triggers it burns the full ``max_turns`` budget for zero diagnostic value
# beyond the second attempt.
#
# Detection uses a normalized-whitespace code hash plus the bare exception
# *type* (e.g., ``NameError``) — not the full message — so a flake-induced
# message diff doesn't mask a real loop.


def _normalize_code_for_loop_detection(code: str | None) -> str:
    """Hashable representation of code that ignores trivial whitespace.

    Strips trailing whitespace per line and drops blank lines, so the LM
    re-emitting identical code with cosmetic differences is still detected
    as a loop. Indentation is preserved (semantic).
    """
    if not code:
        return ""
    lines = [line.rstrip() for line in code.splitlines()]
    lines = [line for line in lines if line.strip()]
    return "\n".join(lines)


def _extract_error_signature(error: str | None) -> tuple[str | None, str | None]:
    """Return ``(error_type, normalized_message)`` from an error string.

    Real Python tracebacks are multi-line with the type+msg on the LAST line
    (``ValueError: bad input``). Single-line errors split on the first colon.
    Returns ``(None, None)`` if the input is empty.

    The full message is returned (not just the type) so that stateful runs
    where the LM re-emits identical code but the error text changes — e.g.
    interpreter state has advanced between attempts — are NOT classified as
    stuck. This keeps the breaker conservative against false positives.
    """
    if not error:
        return (None, None)
    lines = error.strip().splitlines()
    if not lines:
        return (None, None)
    last_line = lines[-1].strip()
    head, _, msg = last_line.partition(":")
    error_type = head.strip() or None
    message = msg.strip() or None
    return (error_type, message)


def _loop_signature(turn: "TurnRecord") -> tuple[str, str | None, str | None]:
    """Signature for stuck-loop detection: (normalized_code, error_type, msg).

    Including the full message text means an identical code cell whose error
    *message* changes between attempts (e.g. state advancing in a stateful
    interpreter) is treated as making progress, not stuck. Only when ALL
    three components match across N consecutive turns do we abort.
    """
    error_type, message = _extract_error_signature(turn.error)
    return (
        _normalize_code_for_loop_detection(turn.code),
        error_type,
        message,
    )


# Repair-turn diversity nudge (reviewer item: "Nudge diversity on repair turns").
# A short line appended to repair feedback that operationalizes REFLECT *within*
# an attempt — pushing the model off a stuck trajectory before the hard
# stuck-loop abort fires. Controlled by FABRIC_RLM_REPAIR_NUDGE:
#   "escalating" -> append only on the 2nd+ failure of the same repair key
#                   (repeat-aware; plain on first repair). DEFAULT — chosen via
#                   A/B (matches "static" accuracy gain with ~1/3 the nudge
#                   volume and less destabilization of correct first repairs).
#   "static"     -> always append on every repair turn (reviewer's version)
#   "off"        -> never append (pre-0.2.x baseline behavior)
_REPAIR_DIVERSITY_LINE = (
    "If this is a repeat failure on the same field, recompute it via a "
    "different method and cross-check the two results before you SUBMIT."
)
_REPAIR_NUDGE_MODES = {"off", "static", "escalating"}
_REPAIR_NUDGE_DEFAULT = "escalating"
_PYTHON_LITERAL_RECOVERY_HINT = (
    "This REPL executes Python: use `None`, `True`, and `False`, "
    "not JSON `null`, `true`, or `false`."
)


def _repair_nudge_mode() -> str:
    """Return the active repair-nudge mode from the environment (default 'escalating')."""
    mode = os.environ.get("FABRIC_RLM_REPAIR_NUDGE", _REPAIR_NUDGE_DEFAULT).strip().lower()
    return mode if mode in _REPAIR_NUDGE_MODES else _REPAIR_NUDGE_DEFAULT


from .lm import resolve_lm
from .prompts import (
    _task_and_outputs,
    build_initial_user_message,
    build_system_prompt,
)
from .skill_loader import Skill, SkillLoader, compose_skills
from .skill_router import RouteDecision, SkillRouter
from .trajectory import Trajectory, TurnRecord, detect_ranking_drift
from .analytical_integrity import (
    check_answer_hygiene,
    check_directional_claims,
    check_ranking_disclosure,
    infer_requested_ranking,
)


logger = logging.getLogger(__name__)


CORE_FINAL_OUTPUT_FIELDS = frozenset({"output", "answer", "result", "report"})

_ACTIVATE_MARKER = "[FABRIC_RLM_ACTIVATE]"


def _is_supported_knowledge_operation(operation: Any) -> bool:
    supported = {
        ("semantic_model.measure", "semantic_model.measure.v1"),
        ("tabular.aggregate", "tabular.aggregate.v1"),
        ("lakehouse.aggregate", "lakehouse.aggregate.v1"),
        (
            "lakehouse.preaggregate_join",
            "lakehouse.preaggregate_join.v1",
        ),
    }
    return (
        operation.status == "active"
        and (operation.operation, operation.host_implementation_id) in supported
    )


def _estimate_tokens(text: str) -> int:
    """Cheap token-count proxy used for budget heuristics."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _scan_activation_markers(stdout: str) -> list[str]:
    if not stdout or _ACTIVATE_MARKER not in stdout:
        return []
    out: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(_ACTIVATE_MARKER + ":"):
            name = line[len(_ACTIVATE_MARKER) + 1 :].strip()
            if name:
                out.append(name)
    return out


def _digest_for_skill(skill: Skill) -> str:
    """Build a compact digest of a skill body: title + summary + verifier signature."""
    lines: list[str] = [f"## Skill (digest): {skill.name}"]
    if skill.title and skill.title != skill.name:
        lines.append(f"- title: {skill.title}")
    if skill.summary:
        lines.append(f"- summary: {skill.summary}")
    if skill.verifier_present:
        lines.append("- verifier: active (call your computed solution through it before SUBMIT).")
    if skill.applies_when_output_fields:
        lines.append(f"- output fields: {', '.join(skill.applies_when_output_fields)}")
    if skill.dependencies:
        lines.append(f"- depends on: {', '.join(skill.dependencies)}")
    return "\n".join(lines)


def _int_env(name: str, default: int) -> int:
    """Read an int env var, falling back to ``default`` on missing/malformed
    input. A bad value must never crash ``import fabric_rlm``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        warnings.warn(
            f"{name}={raw!r} is not a valid integer; using default {default}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return default


STDOUT_FEEDBACK_LIMIT = _int_env("FABRIC_RLM_STDOUT_LIMIT", 5000)
STDERR_FEEDBACK_LIMIT = _int_env("FABRIC_RLM_STDERR_LIMIT", 5000)
ERROR_FEEDBACK_LIMIT = _int_env("FABRIC_RLM_ERROR_LIMIT", 2000)

# Default fraction of each budget reserved for the tail (freshest signal).
# Errors lean tail-heavy: a traceback's diagnostic line is last. Read at call
# time (not frozen here) so they can be flipped per-run — set any to 0 for the
# historic head-only behavior (also the A/B control arm and a user escape hatch).
_STDOUT_TAIL_RATIO_DEFAULT = 0.4
_STDERR_TAIL_RATIO_DEFAULT = 0.5
_ERROR_TAIL_RATIO_DEFAULT = 0.7


def _tail_ratio(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return default


# Failure-time truncation hint. When a turn prints more than the stdout budget,
# the model only sees the head/tail of its own output; trace analysis showed
# models that over-print then *guess* the items they could not see. Firing a
# concrete recovery instruction at exactly that moment follows the project's
# "signal at failure time, don't teach always-on" rule. Default OFF: A/B on real
# tasks showed the deployed-class models (gpt-3.5-turbo, gpt-4.1-mini, gpt-4.1)
# self-limit printing (they sample rather than dump), so the trigger rarely
# fires on classification/counting workloads. Kept as an opt-in safety net for
# genuine large-output dumps (big logs, full-document prints).
_TRUNCATION_HINT_DEFAULT = "off"
_TRUNCATION_HINT_MARKER = "you are NOT seeing all of the output"


def _truncation_hint_enabled() -> bool:
    raw = os.environ.get("FABRIC_RLM_TRUNCATION_HINT", _TRUNCATION_HINT_DEFAULT)
    return raw.strip().lower() in ("1", "on", "true", "yes")


def _build_truncation_hint(full_len: int, limit: int) -> str:
    return (
        f"\n\nNOTE: this turn printed {full_len} characters but only about {limit} "
        f"are shown above, so {_TRUNCATION_HINT_MARKER}. Do not count, classify, "
        "label, or otherwise judge items you cannot see — guessing the hidden ones "
        "is the most common cause of wrong answers. Instead either (a) compute the "
        "result in Python and print only a short summary, or (b) if each item needs "
        "a language-model judgment, process the data in bounded chunks with "
        "`await predict(...)` and combine the results."
    )


def _truncate_for_feedback(text: str, limit: int, *, tail_ratio: float = 0.0) -> str:
    """Trim ``text`` to ``limit`` characters for LM feedback.

    With ``tail_ratio == 0`` (default) this keeps the head and drops the rest —
    the historic behavior. When ``tail_ratio`` is in ``(0, 1]`` it reserves that
    fraction of the budget for the *tail* of the text and spends the remainder
    on the head, joining them with an omitted-count marker. The tail carries the
    freshest signal: the last prints of a turn and — critically — the final
    ``SomeError: ...`` line of a Python traceback, which pure head-truncation
    would discard exactly when the model needs it to recover. Same token budget,
    strictly better signal.
    """
    if not text:
        return ""
    if len(text) <= limit:
        return text
    if tail_ratio <= 0.0:
        extra = len(text) - limit
        return text[:limit] + f"\n... (truncated {extra} more chars)"
    tail = int(limit * min(tail_ratio, 1.0))
    head = limit - tail
    omitted = len(text) - head - tail
    head_part = text[:head]
    tail_part = text[len(text) - tail:] if tail else ""
    return f"{head_part}\n... ({omitted} chars omitted) ...\n{tail_part}"


def _final_turn_suffix() -> str:
    """BUG-LIB-3: addendum appended to the user message immediately preceding
    the LM's final permitted turn. Tells the model that this is its last shot
    and that submitting an imperfect answer beats submitting nothing.

    Library-general; applies to any iterative RLM workload, not the SSB
    benchmark specifically. ASCII-only for consistency with prompts.py.
    """
    return (
        "\n\n*** FINAL TURN ***\n"
        "This is your LAST turn - no further code will be executed after this "
        "one. Respond with one complete ```python code block that calls "
        "SUBMIT(...) with your best current answer derived from the state "
        "above. Submitting an imperfect answer is strictly better than "
        "submitting nothing."
    )


def _build_routing_text(
    task_text: str | None,
    bound_inputs: Mapping[str, Any] | None,
    *,
    per_input_chars: int = 4000,
) -> str:
    """Compose router input from the bound input values.

    Routing scores skills by keyword matches against the actual question/inputs.
    The signature ``task_text`` describes the *menu* (often listing every
    template/skill name) and would inflate scores for unrelated skills, so it
    is intentionally excluded; only bound input values are used. Universal
    across signatures: any string-coercible input value is appended, capped
    per-field to keep routing cheap. ``task_text`` is accepted for API
    stability but ignored.
    """

    del task_text  # routing is driven by inputs only; see docstring
    parts: list[str] = []
    if bound_inputs:
        for value in bound_inputs.values():
            if value is None:
                continue
            try:
                text = value if isinstance(value, str) else str(value)
            except Exception:
                continue
            if not text:
                continue
            if len(text) > per_input_chars:
                text = text[:per_input_chars]
            parts.append(text)
    return "\n".join(parts)


def _route_with_task_fallback(
    router: SkillRouter,
    task_text: str | None,
    bound_inputs: Mapping[str, Any] | None,
    *,
    explicit_skills: list[str] | None,
) -> tuple[RouteDecision, bool]:
    """Route on bound inputs first; fall back to the task text when inputs
    produce zero keyword signal.

    This fixes the ``from_task`` blind spot: when the user's actual question
    lives in ``task=`` and the inputs are just file paths
    (``RLM.from_task(task="Summarize the termination clauses...",
    inputs={"pdf_path": ...})``), inputs-only routing scores nothing and every
    run gets the same always-on bundle. The fallback is gated on *zero* input
    scores so the original menu-inflation protection is preserved: whenever
    bound inputs match any skill keyword (the benchmark-menu shape, where the
    task text enumerates every template), the task text is never consulted.

    Returns ``(decision, used_task_text_fallback)``.
    """

    routing_text = _build_routing_text(task_text, bound_inputs)
    decision = router.route(routing_text, explicit_skills=explicit_skills)
    if decision.scores or not task_text:
        return decision, False
    fallback_text = (routing_text + "\n" + task_text) if routing_text else task_text
    return router.route(fallback_text, explicit_skills=explicit_skills), True


@dataclass(frozen=True)
class OutputValidationResult:
    """Result for validating a worker SUBMIT payload against declared outputs."""

    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class RLMResult:
    submitted: bool
    payload: dict[str, Any] | None
    trajectory: Trajectory
    final_state: dict[str, Any]
    failure_reason: str | None = None
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    # Sub-fields of the totals above. Cached prompt tokens are billed at a
    # ~10x discount on gpt-5; reasoning tokens dominate completion cost on
    # reasoning models. Both ``None`` when no turn reported them.
    total_cached_tokens: int | None = None
    total_reasoning_tokens: int | None = None
    total_lm_seconds: float | None = None
    total_worker_seconds: float | None = None
    # The turn cap this run was given. Compare against ``len(turns)`` to see
    # whether the loop finished because it was done or because it ran out of
    # room -- models differ widely in how many turns they take for the same
    # work, so a cap tuned on one can quietly truncate another.
    max_turns: int | None = None

    @property
    def integrity_problems(self) -> list[str]:
        """Analytical-integrity findings the run ended with, if any.

        In ``"repair"`` mode (the default) these are the problems still
        present when the repair budget ran out and the answer was accepted
        anyway; in ``"strict"`` mode they are the problems of the last
        rejected submission. Empty when the screen was off or satisfied.
        """
        metadata = getattr(self.trajectory, "metadata", None) or {}
        return list(metadata.get("analytical_integrity_unresolved") or [])

    @property
    def integrity_ok(self) -> bool:
        """False when the answer was accepted with unresolved integrity findings."""
        return not self.integrity_problems

    @property
    def turns(self) -> list[TurnRecord]:
        """Convenience alias for ``self.trajectory.turns``.

        Lets callers write ``result.turns`` instead of
        ``result.trajectory.turns``. Returns an empty list when the trajectory
        has no recorded turns.
        """
        return self.trajectory.turns

    @property
    def n_turns(self) -> int:
        """Number of turns recorded in the trajectory."""
        return len(self.trajectory.turns)

    @property
    def outputs(self) -> dict[str, Any]:
        """Submitted payload as a dict; empty dict when no submission occurred.

        Always returns a dict so callers can safely chain ``.outputs.get(...)``
        even on runs that ended without a SUBMIT.
        """
        return self.payload if self.payload is not None else {}

    @property
    def ran_any_code(self) -> bool:
        """True when at least one turn executed code.

        False means nothing ran, which has two quite different causes and the
        report distinguishes them: ``max_turns`` was 0 or exhausted before any
        code block appeared, or the model replied in prose and never emitted a
        code block at all. Either way the run says nothing about whether the
        model can do the task.

        Note this is deliberately *not* called "reached the model". An LM error
        (bad key, wrong provider prefix, rejected parameter) raises out of
        ``run()`` rather than returning a result, so a result in hand means the
        LM answered. Harnesses that wrap ``run()`` in a broad ``except`` and
        record a row are where that distinction gets lost - and it has been lost
        twice on this project, once when an expired key produced 37 rows scored
        as wrong answers and moved a measured delta by four points.
        """
        return self.n_turns > 0

    def report(self, as_dict: bool = False) -> Any:
        """Human-readable summary of what this run did, or a dict of the facts.

        Deterministic and offline: counts, splits and timings straight off the
        trajectory. No model is consulted, nothing is inferred, and it is safe
        to call in CI or on a 100-turn trajectory.

        The fields exist because each one cost real time to reconstruct by hand
        from raw turns: whether the run reached the model at all, whether it
        stopped because it was done or because it ran out of turns, where the
        time went (LM vs worker), and what share of prompt tokens were cached -
        the last because cache-adjusted cost has reversed a conclusion here
        before, when a naive token count overstated a feature's cost by 3x.
        """
        s = self.trajectory.summary()
        turns = self.turns
        ceiling = self.max_turns
        hit_ceiling = bool(ceiling and self.n_turns >= ceiling and not self.submitted)
        prompt = self.total_prompt_tokens
        cached = self.total_cached_tokens
        cache_share = (cached / prompt) if (prompt and cached is not None) else None
        lm_s = self.total_lm_seconds
        wk_s = self.total_worker_seconds
        wall = (lm_s or 0.0) + (wk_s or 0.0)
        timed = [t for t in turns if t.duration_s is not None]
        slowest = max(timed, key=lambda t: t.duration_s or 0.0) if timed else None
        errored = [t for t in turns if t.error]
        last_error = errored[-1] if errored else None
        # A single error class recurring on 3+ turns means the loop is not
        # recovering from it, which is different from a turn that errored once
        # and moved on.
        kinds_count: dict[str, int] = {}
        for t in errored:
            error_lines = (t.error or "").strip().splitlines()
            k = (
                error_lines[-1].split(":", 1)[0].strip()
                if error_lines
                else "Unknown"
            ) or "Unknown"
            kinds_count[k] = kinds_count.get(k, 0) + 1
        repeated_error = next(
            ((k, v) for k, v in sorted(kinds_count.items(), key=lambda kv: -kv[1]) if v >= 3),
            None,
        )
        repairs: dict[str, int] = {}
        for t in turns:
            tt = getattr(t, "turn_type", "normal") or "normal"
            if tt != "normal":
                repairs[tt] = repairs.get(tt, 0) + 1

        facts: dict[str, Any] = {
            "submitted": self.submitted,
            "ran_any_code": self.ran_any_code,
            "failure_reason": self.failure_reason,
            "turns": self.n_turns,
            "max_turns": ceiling,
            "hit_ceiling": hit_ceiling,
            "errors": s.get("errors"),
            "error_kinds": s.get("error_kinds") or {},
            "prompt_tokens": prompt,
            "completion_tokens": self.total_completion_tokens,
            "cached_tokens": cached,
            "cache_share": round(cache_share, 3) if cache_share is not None else None,
            "reasoning_tokens": self.total_reasoning_tokens,
            "lm_seconds": round(lm_s, 2) if lm_s is not None else None,
            "worker_seconds": round(wk_s, 2) if wk_s is not None else None,
            # Repair turns are the closest thing here to a signal about answer
            # *quality* rather than run mechanics. A run can submit successfully
            # and still have been rejected three times on the way, which is the
            # difference between "worked" and "barely worked" - and a user whose
            # output_validator keeps firing has a concrete thing to look at.
            "submit_turn": next((t.turn for t in reversed(turns) if t.submitted), None),
            "repair_turns": repairs,
            "validation_errors": sum(
                len(t.validation_errors or []) for t in turns
            ),
            "issues": [
                {"turn": i.turn, "kind": i.kind, "message": i.message}
                for i in self.trajectory.diagnose()
            ],
            "outputs": sorted(self.outputs.keys()),
        }
        if as_dict:
            return facts

        def n(v: Any, suffix: str = "") -> str:
            if v is None:
                return "n/a"
            if isinstance(v, bool):
                return f"{v}{suffix}"
            if isinstance(v, int):
                return f"{v:,}{suffix}"
            if isinstance(v, float):
                return f"{v:,.2f}{suffix}"
            return f"{v}{suffix}"

        lines: list[str] = []
        if not self.ran_any_code:
            # Lead with this: a run where nothing executed says nothing about
            # whether the model can do the task, and reading it as a wrong
            # answer is the mistake this guards against.
            lines.append("NO CODE RAN - not a wrong answer, and not a model verdict.")
            lines.append(f"  max_turns was {n(ceiling)}; the model never produced an "
                         "executable code block.")
            if self.failure_reason:
                lines.append(f"  failure_reason: {self.failure_reason}")
        elif self.submitted:
            # The LAST submitting turn, not the first. A payload rejected by an
            # output_validator still sets submitted on its turn, so naming the
            # first one reports the attempt that failed rather than the one that
            # stuck - and reads as a contradiction next to the turn count.
            accepted = next((t.turn for t in reversed(turns) if t.submitted), None)
            lines.append(f"SUBMITTED on turn {accepted} of {n(ceiling)}"
                         f"  outputs: {', '.join(facts['outputs']) or '(none)'}")
        else:
            why = self.failure_reason or "no SUBMIT call"
            lines.append(f"NOT SUBMITTED after {self.n_turns} turns - {why}")
            if hit_ceiling:
                lines.append("  Ran out of turns. Raising max_turns may be all this needs.")

        lines.append(
            f"turns {self.n_turns}"
            + (f"/{ceiling}" if ceiling else "")
            + f"   errors {n(s.get('errors'))}"
            + (f" {facts['error_kinds']}" if facts["error_kinds"] else "")
        )
        if prompt or self.total_completion_tokens:
            tok = f"tokens  prompt {n(prompt)}  completion {n(self.total_completion_tokens)}"
            if cache_share is not None:
                tok += f"  cached {cache_share:.0%} of prompt"
            if self.total_reasoning_tokens:
                tok += f"  reasoning {n(self.total_reasoning_tokens)}"
            lines.append(tok)
        if lm_s is not None or wk_s is not None:
            share = f" ({lm_s / wall:.0%} in the model)" if wall and lm_s is not None else ""
            lines.append(f"time    lm {n(lm_s, 's')}  worker {n(wk_s, 's')}{share}")
        if repairs or facts["validation_errors"]:
            bits = ", ".join(f"{k} x{v}" for k, v in sorted(repairs.items()))
            line = f"repairs {bits or 'none'}"
            if facts["validation_errors"]:
                line += f"   validation errors {facts['validation_errors']}"
            line += "   (submitted, but not on the first attempt)" if self.submitted else ""
            lines.append(line)
        if facts["issues"]:
            lines.append(f"issues  {len(facts['issues'])} detected:")
            for i in facts["issues"][:5]:
                lines.append(f"  turn {i['turn']}: {i['kind']} - {i['message'][:80]}")

        # Where to look. A line per turn is unreadable past ~15 turns, so name
        # only the two that answer "which turn do I open first": the slowest,
        # and the last one that errored.
        if slowest is not None and (slowest.duration_s or 0) > 0:
            lines.append(f"slowest turn {slowest.turn} ({slowest.duration_s:.1f}s)")
        if last_error is not None:
            error_lines = (last_error.error or "").strip().splitlines()
            msg = error_lines[-1][:100] if error_lines else "Unknown error"
            lines.append(f"last error  turn {last_error.turn}: {msg}")

        # What to try. Every hint is tied to a fact printed above, so none of
        # this is a guess about intent - which matters, because a confident
        # wrong suggestion costs more than no suggestion.
        hints: list[str] = []
        if hit_ceiling:
            hints.append(f"raise max_turns - all {ceiling} were used and nothing was submitted")
        if repeated_error:
            hints.append(
                f"{repeated_error[0]} recurs on {repeated_error[1]} turns; the loop is "
                f"stuck rather than progressing - start at turn {errored[0].turn}"
            )
        if repairs.get("verifier_repair"):
            hints.append("an output_validator rejected a payload; make sure its message "
                         "says what to change, not just that it was wrong")
        if wall > 5 and wk_s is not None and (wk_s / wall) > 0.8:
            hints.append(f"{wk_s / wall:.0%} of the time was worker execution rather than "
                         "the model - the bottleneck is the data work, not the LM")
        if cache_share is not None and self.n_turns >= 4 and cache_share < 0.3:
            hints.append(f"only {cache_share:.0%} of prompt tokens were cached across "
                         f"{self.n_turns} turns; something in the prompt changes every turn, "
                         "which inflates cost")
        if hints:
            lines.append("what to try:")
            for h in hints:
                lines.append(f"  - {h}")
        return "\n".join(lines)

    def inspect(
        self,
        *,
        max_chars: int = 20_000,
        slow_turn_seconds: float = 10.0,
        expanded: bool = True,
        visible_turns: int = 15,
    ) -> "RunInspector":
        """Return an interactive notebook view of this run's observable turns.

        The returned object renders automatically as HTML in Jupyter and Fabric
        notebooks. The view starts open with individually collapsed turns and a
        scrollable turn viewport. It can also be exported with
        ``.save_html(path)``.
        """

        from .inspector import RunInspector

        return RunInspector(
            self,
            max_chars=max_chars,
            slow_turn_seconds=slow_turn_seconds,
            expanded=expanded,
            visible_turns=visible_turns,
        )

    def __getattribute__(self, name: str) -> Any:
        if name == "inspect":
            payload = object.__getattribute__(self, "__dict__").get("payload")
            if isinstance(payload, dict) and name in payload:
                return payload[name]
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal attribute lookup fails. Never satisfy dunder
        # probes from the payload, and read ``payload`` straight from ``__dict__``
        # so pickling/copying (which reconstruct the instance with an empty
        # ``__dict__`` before ``payload`` is set) cannot recurse infinitely.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        payload = self.__dict__.get("payload")
        if payload and name in payload:
            return payload[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "payload": self.payload,
            "final_state": self.final_state,
            "failure_reason": self.failure_reason,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "total_lm_seconds": self.total_lm_seconds,
            "total_worker_seconds": self.total_worker_seconds,
            "trajectory": self.trajectory.to_dict(),
        }


def _aggregate_trajectory_metrics(
    trajectory: Trajectory,
    unbilled: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Sum per-turn token + timing fields across the trajectory.

    Token aggregates are ``None`` when no turn reported usage (so we don't
    confuse "unknown" with "zero"). Timing aggregates are always summed when
    any turn was recorded.

    ``unbilled`` carries usage from LM calls that never became a trajectory
    turn: a response that was truncated mid-fence, or that contained no
    runnable code at all. Those calls are retried rather than executed, so
    they have no turn to hang off, but the provider still charged for them.
    Leaving them out understated the reported spend of any run that hit either
    guard, and made a run where *every* call hit one report zero tokens.
    """
    extra = unbilled or []

    def _sum_optional(values: list[Any]) -> Any:
        present = [v for v in values if v is not None]
        if not present:
            return None
        return sum(present)

    prompts = [t.prompt_tokens for t in trajectory.turns] + [e.get("prompt_tokens") for e in extra]
    completions = ([t.completion_tokens for t in trajectory.turns]
                   + [e.get("completion_tokens") for e in extra])
    cached = [t.cached_tokens for t in trajectory.turns] + [e.get("cached_tokens") for e in extra]
    reasoning = ([t.reasoning_tokens for t in trajectory.turns]
                 + [e.get("reasoning_tokens") for e in extra])
    lm_secs = ([t.lm_call_seconds for t in trajectory.turns]
               + [e.get("lm_call_seconds") for e in extra])
    worker_secs = [t.worker_execute_seconds for t in trajectory.turns]
    return {
        "total_prompt_tokens": _sum_optional(prompts),
        "total_completion_tokens": _sum_optional(completions),
        "total_cached_tokens": _sum_optional(cached),
        "total_reasoning_tokens": _sum_optional(reasoning),
        "total_lm_seconds": _sum_optional(lm_secs),
        "total_worker_seconds": _sum_optional(worker_secs),
    }


_ENGINE_ALIASES: dict[str, str] = {
    "default": "v6-custom",
    "dspy": "v7-dspy",
}

# Canonical engine names that map 1:1 to a runtime path.
_CANONICAL_ENGINES: tuple[str, ...] = ("v6-custom", "v7-dspy", "adaptive")

# Capability-router pseudo-engines: valid as user input, but resolved to a
# canonical name at __init__ based on what kwargs were passed (e.g. ``tools=``).
# ``self.engine`` always exposes the resolved canonical name, never the
# pseudo-engine literal — see ``_unresolved_engine`` for the original input.
_ROUTER_ENGINES: tuple[str, ...] = ("auto",)

# Phase 5: legacy canonical names still resolve correctly (back-compat) but
# emit ``DeprecationWarning`` to steer new code toward the public aliases.
# The warning is emitted by the public entry points (``RLM.__init__`` and
# ``RLM.from_task`` / ``RLM.task``) at the correct ``stacklevel`` for each —
# NOT inside
# ``_normalize_engine_name`` itself, because that helper is also re-entered
# by internal call sites (notably the adaptive inner-RLM factory) that pass
# canonical names by design and must not self-trigger the warning.
_DEPRECATED_LEGACY_ENGINES: dict[str, str] = {
    "v6-custom": "default",
    "v7-dspy": "dspy",
}


def _emit_engine_deprecation_warning_if_legacy(name: Any, stacklevel: int) -> None:
    """Emit a ``DeprecationWarning`` if ``name`` is a legacy canonical engine
    spelling (``"v6-custom"`` / ``"v7-dspy"``).

    ``stacklevel`` is passed straight through to ``warnings.warn`` and must be
    set by the caller so the warning points at the user's call site rather
    than at library internals. The convention used by ``RLM`` is:

    * Called directly from ``RLM.__init__`` body: ``stacklevel=3``
      (warn -> helper -> __init__ -> user).
    * Called from the shared task-constructor implementation:
      ``stacklevel=4`` (warn -> helper -> _from_task_impl -> public
      constructor alias -> user).

    Non-string inputs are silently ignored — ``_normalize_engine_name`` is
    responsible for raising on those.
    """
    if isinstance(name, str) and name in _DEPRECATED_LEGACY_ENGINES:
        replacement = _DEPRECATED_LEGACY_ENGINES[name]
        warnings.warn(
            f"engine={name!r} is deprecated as a public spelling; "
            f"use engine={replacement!r} instead. Behavior is unchanged: "
            f"{replacement!r} resolves to the same engine. Use engine='auto' "
            f"to let fabric_rlm pick the engine based on whether tools= is "
            f"passed.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )


def _normalize_engine_name(name: str) -> str:
    """Resolve user-friendly engine aliases to canonical internal names.

    Aliases (Phase 1):
        ``"default"`` -> ``"v6-custom"``
        ``"dspy"``    -> ``"v7-dspy"``

    Canonical names (``"v6-custom"``, ``"v7-dspy"``, ``"adaptive"``) and the
    capability-router pseudo-engine ``"auto"`` (Phase 2) pass through unchanged
    — ``"auto"`` is resolved to a canonical name later in ``__init__`` once
    routing-relevant kwargs (``tools=``) are materialized.

    Phase 5 deprecation warnings for legacy canonical names are emitted by
    public entry points (``RLM.__init__`` / task constructors) using
    ``_emit_engine_deprecation_warning_if_legacy``, NOT here — this function
    is re-entered by internal call sites that legitimately pass canonical
    names and must not self-warn.

    Unknown values raise ``ValueError`` listing the valid set.
    """
    valid = sorted(
        set(_CANONICAL_ENGINES) | set(_ENGINE_ALIASES) | set(_ROUTER_ENGINES)
    )
    # Preserve old behavior: invalid engine values raised ValueError, not
    # TypeError. Keep the same exception class even for non-string inputs
    # so existing callers' exception handlers continue to work. (Phase 1
    # is alias-only; type-strictness is out of scope.)
    if not isinstance(name, str):
        raise ValueError(
            f"engine must be one of {valid}, got {name!r}"
        )
    if name in _ENGINE_ALIASES:
        return _ENGINE_ALIASES[name]
    if name in _CANONICAL_ENGINES or name in _ROUTER_ENGINES:
        return name
    raise ValueError(
        f"engine must be one of {valid}, got {name!r}"
    )


class RLM:
    """LM-driven Python REPL loop with persistent worker state.

    Parameters
    ----------
    tools : Iterable[Callable], optional
        Host callables exposed to the worker via the JSON-RPC tool protocol.
        Supported by the DSPy/tool-call engine. Prefer ``engine='auto'``
        (auto-routes to the tool-call engine when ``tools=`` is passed) or
        ``engine='dspy'``; the canonical ``engine='v7-dspy'`` is also
        accepted. Passing tools with an engine that resolves to
        ``v6-custom`` or ``adaptive`` raises ``NotImplementedError``. The
        iterable is snapshotted to a list at construction time, so
        generators are safe but consumed once. Tool names are inferred from
        each callable's
        ``__name__``; wrap with ``functools.wraps`` if you need to preserve
        the underlying name through a decorator. Pass ``dspy.Tool(fn,
        name='...')`` for an explicit name override.
    max_submit_bytes : int, optional
        Maximum UTF-8 JSON byte size for a final ``SUBMIT`` payload. Final
        values are serialized losslessly up to this limit; larger payloads
        fail explicitly rather than being truncated. Defaults to 64 MiB.
    digest_after_turn : int, optional
        Swap full skill bodies in the system message for short digests once a
        skill has been present for this many turns. Defaults to ``None``, which
        leaves the playbook intact for the whole run.

        Leave it off unless you are running out of context. Rewriting the system
        message invalidates the provider's cached prefix, and cached input is
        billed at a tenth of fresh input on Fabric's gpt-5 series, so the tokens
        the digest keeps get re-priced at 10x. On a 400-question benchmark run,
        87% of prompt tokens came from cache and caching cut total cost by 51%;
        digesting gives that back to buy context headroom you may not need.
    """

    def __init__(
        self,
        signature: Any = None,
        *,
        lm: Any,
        sub_lm: Any | None = None,
        max_turns: int = 20,
        timeout: float = 300.0,
        verbose: bool = False,
        skills: list[str] | None = None,
        enable_skill_autoloading: bool = False,
        skill_loader: SkillLoader | None = None,
        enable_verifier: bool = True,
        enable_router: bool = False,
        max_active_skills: int = 2,
        router_baseline_skills: list[str] | None = None,
        router_candidate_specificities: list[str] | None = None,
        router_include_dependencies: bool = True,
        reserve_finalize_turns: int = 0,
        recover_worker_timeouts: int = 1,
        skills_as_cards: bool = False,
        block_network: bool = False,
        max_prompt_tokens: int | None = None,
        digest_after_turn: int | None = None,
        output_validator: Callable[[Mapping[str, Any]], None] | None = None,
        output_validator_context: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
        analytical_integrity: bool | str = True,
        halve_max_iter_on_retry: bool = True,
        engine: str = "auto",
        adaptive: dict | None = None,
        inner_engine: str = "v6-custom",
        stuck_loop_threshold: int | None = 3,
        tools: Iterable[Callable[..., Any]] | None = None,
        security: SecurityPolicy | None = None,
        max_submit_bytes: int = DEFAULT_MAX_SUBMIT_BYTES,
        knowledge: Knowledge | None = None,
    ):
        # ---- engine validation (early, before any heavy resolution) ---------
        # Resolve aliases ('default' -> 'v6-custom', 'dspy' -> 'v7-dspy')
        # before any further engine-based logic so internal switches keep
        # using canonical literals. Phase 1 of engine consolidation.
        # Capture the user's literal input BEFORE normalization so debug
        # tooling and Phase 5 deprecation logic can distinguish e.g.
        # `engine="default"` from `engine="v6-custom"`.
        # Phase 5: emit DeprecationWarning here (not inside the normalizer)
        # so the warning points at the user's call site at stacklevel=3
        # (warn -> helper -> __init__ -> user). ``RLM.from_task`` performs
        # its own emit-then-translate to keep its own callers' stacklevel
        # correct without double-warning.
        _emit_engine_deprecation_warning_if_legacy(engine, stacklevel=3)
        self._unresolved_engine: str = engine if isinstance(engine, str) else ""
        engine = _normalize_engine_name(engine)
        # `inner_engine` is only consulted when engine == "adaptive". For
        # non-adaptive engines it is silently ignored (and overwritten with
        # the resolved engine name a few lines down). Validate it ONLY in
        # the adaptive branch to preserve the prior behavior of accepting
        # arbitrary `inner_engine` values when they are ignored — config-
        # driven callers that always set both must keep working.
        # ---- tools validation (engine-gated, normalize once) ---------------
        # The JSON-RPC tool-call protocol lives only on SubprocessPythonInterpreter
        # (used by engine='v7-dspy'). Legacy Interpreter (v6-custom) and the
        # adaptive wrapper don't carry tools through. Refuse loudly rather than
        # silently dropping the caller's tools.
        # Snapshot the iterable now so generators/etc. are exhausted exactly
        # once at construction time and downstream code sees a stable list.
        tool_list: list[Callable[..., Any]] = list(tools) if tools is not None else []
        if knowledge is not None:
            from .knowledge_api import Knowledge as KnowledgeType

            if not isinstance(knowledge, KnowledgeType):
                raise TypeError("knowledge must be a Knowledge instance")
        self._knowledge = knowledge
        # ---- engine="auto" capability router (Phase 2) ---------------------
        # Resolve the pseudo-engine to a canonical name based on what the
        # caller actually needs. Decision is made AFTER tool_list is
        # materialized so generators are exhausted exactly once and the
        # router sees the real (possibly empty) length.
        # Routing rule: tools= present -> v7-dspy (the only engine that
        # currently supports the JSON-RPC tool-call protocol); otherwise
        # v6-custom (the historic default).
        if engine == "auto":
            engine = "v7-dspy" if tool_list else "v6-custom"
        if tool_list and engine != "v7-dspy":
            raise NotImplementedError(
                "tools= requires the DSPy/tool-call engine "
                "(canonical name: 'v7-dspy'). "
                "Use engine='dspy' or engine='auto' (which auto-selects "
                "the tool-call engine when tools= is passed). "
                f"Got resolved engine={engine!r}."
            )
        for i, t in enumerate(tool_list):
            if not callable(t):
                raise TypeError(
                    f"tools[{i}] is not callable: got {type(t).__name__}. "
                    "Pass a list/iterable of plain callables (function names "
                    "are inferred from __name__) or dspy.Tool objects."
                )
        # ---- stuck-loop threshold validation -------------------------------
        if stuck_loop_threshold is not None:
            # Reject bool explicitly: bool is a subclass of int in Python.
            if isinstance(stuck_loop_threshold, bool) or not isinstance(stuck_loop_threshold, int):
                raise TypeError(
                    "stuck_loop_threshold must be an int or None; "
                    f"got {type(stuck_loop_threshold).__name__}"
                )
            if stuck_loop_threshold < 2:
                raise ValueError(
                    "stuck_loop_threshold must be >= 2 (a single retry is normal); "
                    f"got {stuck_loop_threshold!r}. Pass None to disable."
                )
        self.stuck_loop_threshold = stuck_loop_threshold
        # Default-on security baseline. The contract is: by default, every
        # LM-facing interpreter created by this RLM scrubs secret-bearing
        # env vars and rejects destructive/network/dynamic-dispatch APIs
        # statically before the code reaches the worker. Pass
        # ``security=SecurityPolicy.disabled()`` to opt out entirely, or
        # supply a customised ``SecurityPolicy`` instance to tune the
        # denylist / env-strip globs. See ``fabric_rlm.security`` for the
        # honest framing of what this does and does not protect against.
        self._security: SecurityPolicy = security if security is not None else SecurityPolicy.default()
        self.max_submit_bytes = validate_max_submit_bytes(max_submit_bytes)
        if engine == "adaptive":
            # Resolve aliases for inner_engine ONLY when it is actually used.
            # Phase 1: keeps non-adaptive callers free of new strictness on
            # an arg they pass-but-don't-use.
            if not isinstance(inner_engine, str):
                raise ValueError(
                    "inner_engine must be 'v6-custom' or 'v7-dspy' "
                    f"(got {inner_engine!r}); 'adaptive' cannot be its own inner engine"
                )
            inner_engine = _ENGINE_ALIASES.get(inner_engine, inner_engine)
            if inner_engine not in ("v6-custom", "v7-dspy"):
                raise ValueError(
                    "inner_engine must be 'v6-custom' or 'v7-dspy' "
                    f"(got {inner_engine!r}); 'adaptive' cannot be its own inner engine"
                )
            warnings.warn(
                "engine='adaptive' is experimental; API may change before 0.2.x",
                UserWarning,
                stacklevel=2,
            )
            # Keep the outer wrapper minimal — do not resolve LM, do not load
            # skills. Each inner attempt builds its own RLM via the factory.
            self.engine = "adaptive"
            self.inner_engine = inner_engine
            self.adaptive_config: dict[str, Any] = dict(adaptive or {})
            self.signature = signature
            self.outer_lm = None
            self.sub_lm_spec = sub_lm if sub_lm is not None else (
                lm if isinstance(lm, (str, dict)) else None
            )
            self.max_turns = max_turns
            self.timeout = timeout
            self.verbose = verbose
            self.skills = list(skills or [])
            self.enable_skill_autoloading = enable_skill_autoloading
            self.skill_loader = skill_loader or SkillLoader()
            self.enable_verifier = enable_verifier
            self.enable_router = enable_router
            self.max_active_skills = max(0, int(max_active_skills))
            self.router_baseline_skills = (
                list(router_baseline_skills) if router_baseline_skills is not None else None
            )
            self.router_candidate_specificities = (
                list(router_candidate_specificities)
                if router_candidate_specificities is not None
                else None
            )
            self.router_include_dependencies = bool(router_include_dependencies)
            self.reserve_finalize_turns = max(0, int(reserve_finalize_turns))
            self.recover_worker_timeouts = max(0, int(recover_worker_timeouts))
            self.skills_as_cards = bool(skills_as_cards)
            self.block_network = bool(block_network)
            _reject_block_network_with_sub_lm(block_network, sub_lm)
            self.max_prompt_tokens = max_prompt_tokens
            self.digest_after_turn = digest_after_turn
            self.output_validator = output_validator
            self.output_validator_context = output_validator_context
            self.analytical_integrity = _normalize_integrity_mode(analytical_integrity)
            self._integrity_rejections = 0
            self.halve_max_iter_on_retry = bool(halve_max_iter_on_retry)
            self._loaded_skills: list[Skill] = []
            self._activated_skills: set[str] = set()
            self._inline_task: str | None = None
            self._inline_outputs: list[str] | None = None
            self._inline_output_types: dict[str, type] = {}
            self._inline_inputs: dict[str, Any] = {}
            # Capture the construction args needed to spin up inner RLMs.
            # ``lm`` is preserved as the user passed it (spec or instance) so
            # the inner factory can re-resolve specs in parallel rollouts.
            self._adaptive_inner_kwargs: dict[str, Any] = dict(
                signature=signature,
                lm=lm,
                sub_lm=sub_lm,
                timeout=timeout,
                verbose=verbose,
                skills=list(skills or []),
                enable_skill_autoloading=enable_skill_autoloading,
                skill_loader=skill_loader,
                enable_verifier=enable_verifier,
                enable_router=enable_router,
                max_active_skills=max_active_skills,
                router_baseline_skills=router_baseline_skills,
                router_candidate_specificities=router_candidate_specificities,
                router_include_dependencies=router_include_dependencies,
                reserve_finalize_turns=reserve_finalize_turns,
                recover_worker_timeouts=recover_worker_timeouts,
                skills_as_cards=skills_as_cards,
                block_network=block_network,
                max_prompt_tokens=max_prompt_tokens,
                digest_after_turn=digest_after_turn,
                output_validator=output_validator,
                output_validator_context=output_validator_context,
                analytical_integrity=analytical_integrity,
                halve_max_iter_on_retry=halve_max_iter_on_retry,
                stuck_loop_threshold=stuck_loop_threshold,
                security=self._security,
                max_submit_bytes=self.max_submit_bytes,
                knowledge=None,
            )
            return

        # ---- legacy v6/v7 path (unchanged below this line) -----------------
        self.tools: list[Callable[..., Any]] = tool_list
        self.signature = signature
        self.outer_lm = resolve_lm(lm)
        self.sub_lm_spec = sub_lm if sub_lm is not None else (lm if isinstance(lm, (str, dict)) else None)
        self.max_turns = max_turns
        self.timeout = timeout
        self.verbose = verbose
        self.skills = list(skills or [])
        self.enable_skill_autoloading = enable_skill_autoloading
        self.skill_loader = skill_loader or SkillLoader()
        self.enable_verifier = enable_verifier
        self.enable_router = enable_router
        self.max_active_skills = max(0, int(max_active_skills))
        self.router_baseline_skills = (
            list(router_baseline_skills) if router_baseline_skills is not None else None
        )
        self.router_candidate_specificities = (
            list(router_candidate_specificities)
            if router_candidate_specificities is not None
            else None
        )
        self.router_include_dependencies = bool(router_include_dependencies)
        self.reserve_finalize_turns = max(0, int(reserve_finalize_turns))
        self.recover_worker_timeouts = max(0, int(recover_worker_timeouts))
        self.skills_as_cards = bool(skills_as_cards)
        self.block_network = bool(block_network)
        _reject_block_network_with_sub_lm(block_network, sub_lm)
        self.max_prompt_tokens = max_prompt_tokens
        self.digest_after_turn = digest_after_turn
        self.output_validator = output_validator
        self.output_validator_context = output_validator_context
        self.analytical_integrity = _normalize_integrity_mode(analytical_integrity)
        self._integrity_rejections = 0
        self.halve_max_iter_on_retry = bool(halve_max_iter_on_retry)
        # Per-run counter for repeat-aware repair nudges (keyed by repair target);
        # reset at the start of each run().
        self._repair_counts: dict[str, int] = {}
        self.engine = engine
        self.inner_engine = engine  # for symmetry — non-adaptive RLMs are their own inner
        self.adaptive_config = None
        self._loaded_skills: list[Skill] = (
            [self.skill_loader.load(name) for name in self.skills]
            if self.enable_verifier
            else []
        )
        self._activated_skills: set[str] = set()
        self._inline_task: str | None = None
        self._inline_outputs: list[str] | None = None
        self._inline_output_types: dict[str, type] = {}
        self._inline_inputs: dict[str, Any] = {}

    @classmethod
    def learn(
        cls,
        *,
        sources: Mapping[str, object],
        store: KnowledgeStore | None = None,
        roles: Mapping[str, object] | None = None,
        package_id: str | None = None,
        limits: ProfileLimits | None = None,
        registry: SourceAdapterRegistry | None = None,
        transport: OneLakeKnowledgeTransport | None = None,
        overwrite: bool = False,
    ) -> Knowledge:
        """Learn a deterministic package from explicitly approved sources."""

        from .knowledge_api import learn as learn_knowledge

        return learn_knowledge(
            sources=sources,
            store=store,
            roles=roles,
            package_id=package_id,
            limits=limits,
            registry=registry,
            transport=transport,
            overwrite=overwrite,
        )

    def _bind_knowledge_inputs(
        self,
        explicit_inputs: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._knowledge is None:
            return dict(explicit_inputs), {}
        from .knowledge_preflight import preflight_knowledge

        conflicts = sorted(set(explicit_inputs) & set(self._knowledge.bindings))
        if conflicts:
            raise ValueError(
                "task inputs conflict with knowledge source aliases: "
                + ", ".join(conflicts)
            )
        if (
            "knowledge_result" in explicit_inputs
            and any(
                _is_supported_knowledge_operation(operation)
                for operation in self._knowledge.package.operations
            )
        ):
            raise ValueError(
                "task input alias knowledge_result is reserved for registered operations"
            )
        blocked = [
            source.source_id
            for source in self._knowledge.package.sources
            if source.status not in {"candidate", "active"}
        ]
        if blocked:
            raise ValueError(
                "knowledge source is not reusable: " + ", ".join(sorted(blocked))
            )
        preflight = preflight_knowledge(
            self._knowledge.package,
            self._knowledge.bindings,
            limits=self._knowledge._limits,
            registry=self._knowledge._registry,
        )
        if preflight.drift:
            raise ValueError(
                "stale knowledge sources detected: "
                + ", ".join(sorted(preflight.drift))
            )
        bound = dict(explicit_inputs)
        bound.update(self._knowledge.bindings)
        supported = any(
            _is_supported_knowledge_operation(operation)
            for operation in self._knowledge.package.operations
        )
        if supported:
            mode = "registered_operations_available"
        elif self._knowledge.package.operations:
            mode = "registered_operations_unavailable"
        else:
            mode = "fallback_no_registered_operations"
        return bound, {
            "knowledge_fingerprint": self._knowledge.package.fingerprint,
            "knowledge_mode": mode,
        }

    def _prepare_registered_operation(
        self,
        bound_inputs: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if (
            self._knowledge is None
            or metadata.get("knowledge_mode") != "registered_operations_available"
        ):
            return bound_inputs, metadata

        from .knowledge import canonical_json
        from .knowledge_execution import (
            OperationPlanError,
            OperationPlanFallback,
            execute_registered_operation,
            parse_operation_plan,
        )

        operations = [
            {
                "operation_id": operation.operation_id,
                "operation": operation.operation,
                "parameter_schema": operation.to_dict()["parameter_schema"],
                "parameter_defaults": operation.to_dict()["parameter_defaults"],
                "grain": operation.grain,
                "max_output_rows": operation.max_output_rows,
            }
            for operation in self._knowledge.package.operations
            if _is_supported_knowledge_operation(operation)
        ]
        task_description, _ = _task_and_outputs(
            self.signature,
            self._inline_task,
            self._inline_outputs,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Select one registered host operation for the task. "
                    "Return JSON only, with exactly "
                    '{"operation_id":"...","parameters":{...}}. '
                    "Use only operation IDs and parameter values allowed by the "
                    "provided contracts. Never write DAX, SQL, Python, or another "
                    "executable language. If no operation is compatible, return "
                    '{"fallback":true,"reason":"short reason"}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{task_description}\n\n"
                    f"Registered operations:\n{canonical_json(operations)}"
                ),
            },
        ]
        response_text, raw_response, selection_seconds = _call_lm_with_meta(
            self.outer_lm,
            messages,
        )
        usage = _extract_usage(raw_response)
        metadata = {
            **metadata,
            "operation_selection_lm_calls": 1,
            "operation_selection_lm_seconds": selection_seconds,
        }
        for metadata_name, usage_name in (
            ("operation_selection_prompt_tokens", "prompt_tokens"),
            ("operation_selection_completion_tokens", "completion_tokens"),
        ):
            value = _usage_field(usage, usage_name)
            if value is not None:
                metadata[metadata_name] = value
        for metadata_name, block_name, usage_name in (
            (
                "operation_selection_cached_tokens",
                "prompt_tokens_details",
                "cached_tokens",
            ),
            (
                "operation_selection_reasoning_tokens",
                "completion_tokens_details",
                "reasoning_tokens",
            ),
        ):
            value = _usage_nested_field(usage, block_name, usage_name)
            if value is not None:
                metadata[metadata_name] = value
        try:
            plan = parse_operation_plan(response_text)
        except ValueError:
            logger.warning(
                "Registered operation plan was rejected: invalid planner response"
            )
            metadata["operation_fallback_reason"] = "operation_plan_parse_invalid"
            metadata["knowledge_mode"] = "fallback_operation_plan_rejected"
            return bound_inputs, metadata
        if isinstance(plan, OperationPlanFallback):
            logger.info(
                "Registered operation planner declined the available operations"
            )
            metadata["operation_fallback_reason"] = "no_compatible_operation"
            metadata["knowledge_mode"] = "fallback_no_compatible_operation"
            return bound_inputs, metadata
        try:
            execution = execute_registered_operation(
                self._knowledge,
                operation_id=plan.operation_id,
                parameters=plan.parameters,
            )
        except OperationPlanError:
            logger.warning(
                "Registered operation plan was rejected by the host contract"
            )
            metadata["operation_fallback_reason"] = (
                "operation_plan_contract_rejected"
            )
            metadata["knowledge_mode"] = "fallback_operation_plan_rejected"
            return bound_inputs, metadata

        packet = execution.to_packet()
        metadata.update(
            {
                "knowledge_mode": "registered_operation",
                "operation_id": execution.operation_id,
                "operation_version": execution.operation_version,
                "operation_fingerprint": execution.operation_fingerprint,
                "operation_result_fingerprint": execution.result_fingerprint,
                "operation_source_fingerprints": dict(
                    execution.source_fingerprints
                ),
                "operation_audit_status": execution.audit_status,
                "operation_host_seconds": execution.elapsed_seconds,
            }
        )
        synthesis_inputs = {
            name: value
            for name, value in bound_inputs.items()
            if name not in self._knowledge.bindings
        }
        synthesis_inputs["knowledge_result"] = packet
        return synthesis_inputs, metadata

    @staticmethod
    def _attach_knowledge_metadata(
        result: RLMResult,
        metadata: Mapping[str, Any],
    ) -> RLMResult:
        if metadata:
            result.trajectory.metadata.update(metadata)
        for result_field, metadata_field in (
            ("total_prompt_tokens", "operation_selection_prompt_tokens"),
            ("total_completion_tokens", "operation_selection_completion_tokens"),
            ("total_cached_tokens", "operation_selection_cached_tokens"),
            ("total_reasoning_tokens", "operation_selection_reasoning_tokens"),
        ):
            extra = metadata.get(metadata_field)
            if isinstance(extra, int):
                current = getattr(result, result_field)
                setattr(result, result_field, extra if current is None else current + extra)
        selection_seconds = metadata.get("operation_selection_lm_seconds")
        if isinstance(selection_seconds, (int, float)):
            current_seconds = result.total_lm_seconds
            result.total_lm_seconds = (
                float(selection_seconds)
                if current_seconds is None
                else current_seconds + float(selection_seconds)
            )
        return result

    @classmethod
    def _from_task_impl(
        cls,
        task: str,
        inputs: dict[str, Any] | None = None,
        outputs: list[str] | Mapping[str, type] | None = None,
        *,
        knowledge: Knowledge | None = None,
        deprecation_stacklevel: int,
        **kwargs: Any,
    ) -> "RLM":
        # Don't pass `None` positionally — callers may supply `signature=...`
        # via kwargs (v6.5+) and we'd otherwise hit "multiple values for 'signature'".
        kwargs.setdefault("signature", None)
        if knowledge is not None:
            kwargs["knowledge"] = knowledge
        # Phase 5 deprecation: emit at the from_task call site stacklevel
        # (warn -> helper -> from_task -> user = 3). Suppress only the
        # inner re-emission of OUR engine-deprecation warning during the
        # delegated ``cls(**kwargs)`` call via a narrow message-scoped
        # filter -- we must NOT mutate ``kwargs["engine"]``, because
        # subclasses are entitled to observe the user's literal value in
        # their ``__init__`` (caught by Phase 7 v1 duck as a subclass
        # regression: rewriting kwargs before delegating broke the
        # ``from_task`` extension surface for ``cls != RLM``).
        #
        # The filter is installed ONLY when the user passed a legacy
        # literal -- otherwise an unconditional filter would silently
        # swallow any matching ``DeprecationWarning`` emitted by subclass
        # ``__init__`` even on non-legacy paths, defeating ``-W error``
        # policy on the subclass extension surface (Phase 7 v2 duck).
        _user_engine = kwargs.get("engine")
        _emit_engine_deprecation_warning_if_legacy(
            _user_engine, stacklevel=deprecation_stacklevel
        )
        if isinstance(_user_engine, str) and _user_engine in _DEPRECATED_LEGACY_ENGINES:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=rf"engine={_user_engine!r} is deprecated as a public spelling",
                    category=DeprecationWarning,
                )
                instance = cls(**kwargs)
        else:
            instance = cls(**kwargs)
        output_names, output_types = _normalize_inline_outputs(outputs)
        if output_types and instance.signature is not None:
            _, signature_outputs = _task_and_outputs(instance.signature, None, None)
            if len(output_names) != len(signature_outputs) or set(output_names) != set(
                signature_outputs
            ):
                raise ValueError(
                    f"Typed output fields {output_names!r} do not match explicit "
                    f"signature outputs {signature_outputs!r}."
                )
        instance._inline_task = task
        instance._inline_outputs = output_names
        instance._inline_output_types = output_types
        instance._inline_inputs = dict(inputs or {})
        return instance

    @classmethod
    def from_task(
        cls,
        task: str,
        inputs: dict[str, Any] | None = None,
        outputs: list[str] | Mapping[str, type] | None = None,
        *,
        knowledge: Knowledge | None = None,
        **kwargs: Any,
    ) -> "RLM":
        """Construct an RLM from task text and named inputs and outputs.

        ``outputs`` may be a list of field names for backward-compatible,
        name-only validation, or a mapping from field names to concrete Python
        types for runtime type enforcement and repair feedback.
        """
        return cls._from_task_impl(
            task,
            inputs=inputs,
            outputs=outputs,
            knowledge=knowledge,
            deprecation_stacklevel=4,
            **kwargs,
        )

    @classmethod
    def task(
        cls,
        task: str,
        inputs: dict[str, Any] | None = None,
        outputs: list[str] | Mapping[str, type] | None = None,
        *,
        knowledge: Knowledge | None = None,
        **kwargs: Any,
    ) -> "RLM":
        """Construct an RLM using the same contract as :meth:`from_task`."""
        return cls._from_task_impl(
            task,
            inputs=inputs,
            outputs=outputs,
            knowledge=knowledge,
            deprecation_stacklevel=4,
            **kwargs,
        )

    def __call__(self, **inputs: Any) -> RLMResult:
        return self.run(inputs or None)

    def _run_adaptive(self, bound_inputs: dict[str, Any]) -> RLMResult:
        """Route ``run()`` through ``AdaptiveRunner`` for ``engine='adaptive'``.

        Returns the winning attempt's :class:`RLMResult` with the full attempt
        log attached at ``result.trajectory.metadata['adaptive']``.
        """
        # Local import to avoid pulling experimental code into the cold path.
        from .experimental.adaptive_policy import (
            Budget,
            LadderPolicy,
            ValidationVerdict,
        )
        from .experimental.adaptive_runner import AdaptiveRunner

        cfg = self.adaptive_config or {}
        inner_kwargs = self._adaptive_inner_kwargs
        inner_engine = self.inner_engine
        # Skill loader is shared across attempts to avoid re-reading from disk.
        shared_loader = inner_kwargs.get("skill_loader") or self.skill_loader

        def factory(attempt_cfg):
            # Decompose-phase rungs bypass the standard inner RLM loop entirely
            # — they run a depth-1 decompose -> parallel sub-solve -> synthesize
            # pipeline. Universal building block; works on any prompt key.
            if getattr(attempt_cfg, "decompose_phase", False):
                from .experimental.decompose_engine import DecomposeRLMAdapter

                # Resolve the LM the same way the standard branch would.
                lm_for_decompose = (
                    attempt_cfg.lm_instance
                    if attempt_cfg.lm_instance is not None
                    else attempt_cfg.lm_spec
                )
                # Honor reasoning_effort if the LM supports .copy(...)
                if attempt_cfg.reasoning_effort and hasattr(lm_for_decompose, "copy"):
                    try:
                        lm_for_decompose = lm_for_decompose.copy(
                            reasoning_effort=attempt_cfg.reasoning_effort
                        )
                    except Exception:
                        pass
                return DecomposeRLMAdapter(
                    lm=lm_for_decompose,
                    sub_lm=lm_for_decompose,
                    max_subs=getattr(attempt_cfg, "decompose_max_subs", 6),
                )

            kwargs = dict(inner_kwargs)
            kwargs["skill_loader"] = shared_loader
            # Phase 5: pass the public alias spelling instead of the canonical
            # legacy literal so the inner ``RLM(**kwargs)`` call does not
            # self-emit a DeprecationWarning. ``inner_engine`` is already a
            # validated canonical name ("v6-custom" or "v7-dspy") — translate
            # to its public alias ("default" / "dspy") via the inverse map.
            # Behavior is identical because ``_normalize_engine_name`` resolves
            # both back to the same canonical name.
            kwargs["engine"] = _DEPRECATED_LEGACY_ENGINES.get(inner_engine, inner_engine)
            kwargs["max_turns"] = attempt_cfg.max_turns
            if attempt_cfg.lm_spec is not None:
                kwargs["lm"] = attempt_cfg.lm_spec
            elif attempt_cfg.lm_instance is not None:
                kwargs["lm"] = attempt_cfg.lm_instance
            # reasoning_effort is forwarded into the inner RLM. When the lm
            # spec is a dict, merge in-place. When it's a dspy.LM-like
            # instance with .copy(), clone it so the original isn't mutated
            # (callers commonly pass a single FabricLM instance shared
            # across attempts and across tasks).
            if attempt_cfg.reasoning_effort:
                lm_obj = kwargs.get("lm")
                if isinstance(lm_obj, dict):
                    kwargs["lm"] = {**lm_obj, "reasoning_effort": attempt_cfg.reasoning_effort}
                elif lm_obj is not None and hasattr(lm_obj, "copy"):
                    try:
                        kwargs["lm"] = lm_obj.copy(reasoning_effort=attempt_cfg.reasoning_effort)
                    except Exception:
                        # If the LM doesn't accept reasoning_effort via copy(),
                        # leave it alone — non-reasoning chat models will reject it.
                        pass
            inner = RLM(**kwargs)
            # Propagate from_task inline state. When the outer adaptive RLM is
            # constructed via ``RLM.from_task(...)``, that classmethod sets
            # ``_inline_task`` / ``_inline_outputs`` / ``_inline_inputs`` on the
            # outer instance AFTER ``__init__`` returns (see ``from_task`` above).
            # ``inner_kwargs`` is a snapshot taken inside outer ``__init__``, so
            # those attributes are NOT in it — without this copy, every inner
            # RLM would run "blind" (no task description, no declared outputs)
            # and consistently emit ``answer=None``. Mirror the post-init
            # assignment that ``from_task`` does so inner RLMs see the same task.
            if getattr(self, "_inline_task", None) is not None:
                inner._inline_task = self._inline_task
                inner._inline_outputs = (
                    list(self._inline_outputs) if self._inline_outputs else []
                )
                inner._inline_output_types = dict(self._inline_output_types)
                inner._inline_inputs = dict(self._inline_inputs or {})
            return inner

        # ---- policy + budget construction from adaptive_config ---------------
        policy_kwargs = {
            "base_max_turns": self.max_turns,
        }
        if "strong_lm" in cfg:
            policy_kwargs["strong_lm_spec"] = cfg["strong_lm"]
        if "parallel_rollouts" in cfg:
            policy_kwargs["parallel_rollouts"] = int(cfg["parallel_rollouts"])
        if "feedback_injection" in cfg:
            policy_kwargs["feedback_injection"] = bool(cfg["feedback_injection"])
        if "skip_more_turns_when_submitted" in cfg:
            policy_kwargs["skip_more_turns_when_submitted"] = bool(
                cfg["skip_more_turns_when_submitted"]
            )
        # If the user passed an LM with reasoning_effort already set, seed
        # the policy so its rung-0 baseline matches the user's intent. Works
        # for both dict specs and dspy.LM instances (which carry .kwargs).
        lm_obj = self._adaptive_inner_kwargs.get("lm")
        if isinstance(lm_obj, dict):
            eff = lm_obj.get("reasoning_effort")
        elif lm_obj is not None and hasattr(lm_obj, "kwargs"):
            eff = lm_obj.kwargs.get("reasoning_effort") if isinstance(lm_obj.kwargs, dict) else None
        else:
            eff = None
        if eff:
            policy_kwargs["base_reasoning_effort"] = eff
        policy = cfg.get("policy") or LadderPolicy(**policy_kwargs)

        budget_kwargs: dict[str, Any] = {}
        for key in ("max_attempts", "max_total_turns", "max_parallel", "max_wall_seconds"):
            if key in cfg:
                budget_kwargs[key] = cfg[key]
        budget = cfg.get("budget") or Budget(**budget_kwargs)

        # Warn if the user's max_attempts can't reach the policy's top rung.
        # Common footgun: bumping max_rung without bumping max_attempts means
        # the top rungs are silently unreachable. We don't auto-bump (would
        # surprise users who explicitly capped attempts) — just be loud.
        try:
            policy_max_rung = int(getattr(policy, "max_rung"))
        except (AttributeError, TypeError, ValueError):
            policy_max_rung = -1
        if policy_max_rung >= 0 and budget.max_attempts < policy_max_rung + 1:
            warnings.warn(
                f"adaptive: budget.max_attempts={budget.max_attempts} cannot "
                f"reach policy.max_rung={policy_max_rung} (need at least "
                f"{policy_max_rung + 1}). Top rungs will not be tried. "
                f"Pass adaptive={{'max_attempts': {policy_max_rung + 1}}} to fix.",
                UserWarning,
                stacklevel=3,
            )

        validator = cfg.get("validator")
        if validator is None:
            warnings.warn(
                "engine='adaptive' was constructed without an explicit validator; "
                "the default checks structural validity only, not semantic correctness. "
                "Pass adaptive={'validator': your_validator} for meaningful escalation.",
                UserWarning,
                stacklevel=3,
            )

        # Optional task classifier — one-shot pre-run hook seeds bandit priors
        # from a single low-effort LM call so the bandit doesn't waste its
        # cold-start observations on rung 0 for obviously-hard questions.
        # Universal: works on any prompt; classifier output classes are
        # reasoning-shape, not domain. Best-effort — failure degrades to
        # ordinary cold-start behaviour.
        pre_run_hook = None
        if cfg.get("enable_task_classifier"):
            try:
                from .experimental.task_classifier import make_classifier_pre_run

                classifier_lm = cfg.get("task_classifier_lm")
                if classifier_lm is None:
                    # Fall back to the same LM the inner engine uses
                    classifier_lm = self._adaptive_inner_kwargs.get("lm")
                if classifier_lm is not None:
                    pre_run_hook = make_classifier_pre_run(
                        classifier_lm=classifier_lm,
                        question_input_key=cfg.get("task_classifier_input_key"),
                        prior_table=cfg.get("task_classifier_prior_table"),
                        overwrite_existing=bool(
                            cfg.get("task_classifier_overwrite_existing", False)
                        ),
                        on_classify=cfg.get("on_classify"),
                    )
            except Exception:
                pre_run_hook = None

        runner = AdaptiveRunner(
            rlm_factory=factory,
            policy=policy,
            budget=budget,
            validator=validator,  # AdaptiveRunner uses its default when None
            on_attempt=cfg.get("on_attempt"),
            feedback_injection=policy_kwargs.get("feedback_injection", True),
            pre_run=pre_run_hook,
        )
        adaptive_result = runner.run(bound_inputs)
        # AdaptiveRunner already attaches metadata to the winning trajectory
        # and returns the underlying RLMResult on .result. If every attempt
        # raised, the winner is a sentinel ``_FailedResult`` — convert it to
        # a real RLMResult so callers see a uniform shape.
        winner = adaptive_result.result
        from .experimental.adaptive_runner import _FailedResult, _failed_to_rlm_result

        if isinstance(winner, _FailedResult):
            winner = _failed_to_rlm_result(winner)
            # Re-attach the adaptive summary on the synthesized trajectory so
            # downstream tooling can still read ``trajectory.metadata['adaptive']``.
            if winner.trajectory is not None and hasattr(winner.trajectory, "metadata"):
                winner.trajectory.metadata.setdefault(
                    "adaptive",
                    {
                        "stop_reason": adaptive_result.stop_reason,
                        "elapsed_seconds": adaptive_result.elapsed_seconds,
                        "winner_rung": adaptive_result.winner.rung,
                        "winner_rollout_index": adaptive_result.winner.rollout_index,
                        "attempts": [a.to_summary() for a in adaptive_result.attempts],
                        "all_attempts_failed": True,
                    },
                )
        return winner

    def run(self, inputs: dict[str, Any] | None = None) -> RLMResult:
        bound_inputs = dict(self._inline_inputs)
        if inputs:
            bound_inputs.update(inputs)
        bound_inputs, knowledge_metadata = self._bind_knowledge_inputs(bound_inputs)
        bound_inputs, knowledge_metadata = self._prepare_registered_operation(
            bound_inputs,
            knowledge_metadata,
        )
        bound_inputs = resolve_lakehouse_inputs(bound_inputs)

        if self.engine == "adaptive":
            return self._attach_knowledge_metadata(
                self._run_adaptive(bound_inputs),
                knowledge_metadata,
            )

        required_output_fields = _required_output_fields(self.signature, self._inline_task, self._inline_outputs)

        if self.engine == "v7-dspy":
            return self._attach_knowledge_metadata(
                self._run_via_dspy(bound_inputs, required_output_fields),
                knowledge_metadata,
            )

        trajectory = Trajectory(
            metadata={
                "max_turns": self.max_turns,
                "skills": list(self.skills),
                "skill_autoloading": self.enable_skill_autoloading,
                "router_enabled": self.enable_router,
                **knowledge_metadata,
            }
        )

        # Router-driven setup: pick active skills + build cards based on the
        # task text (signature description or inline task). Falls back to the
        # legacy preloaded-skills flow when the router is disabled.
        route_decision: RouteDecision | None = None
        cards_text: str | None = None
        active_skill_objects: list[Skill] = []
        if self.enable_router:
            task_text_for_routing, _ = _task_and_outputs(
                self.signature, self._inline_task, self._inline_outputs
            )
            router = SkillRouter.from_loader(
                self.skill_loader,
                max_active_skills=self.max_active_skills,
                baseline_skill_names=self.router_baseline_skills,
                candidate_specificities=self.router_candidate_specificities,
                include_dependencies=self.router_include_dependencies,
            )
            # Route on bound input values first (so keywords in the actual
            # question dominate), falling back to the task text only when the
            # inputs carry zero keyword signal — the from_task case where the
            # question lives in task= and inputs are just file paths.
            route_decision, used_task_fallback = _route_with_task_fallback(
                router,
                task_text_for_routing,
                bound_inputs,
                explicit_skills=self.skills or None,
            )
            trajectory.metadata["router_used_task_text_fallback"] = used_task_fallback
            active_skill_objects = [
                router._by_name[name]
                for name in route_decision.active
                if name in router._by_name
            ]
            self._loaded_skills = list(active_skill_objects) if self.enable_verifier else []
            self._activated_skills = {sk.name for sk in active_skill_objects}
            if route_decision.cards:
                cards_text = "\n".join(router.card_text(n) for n in route_decision.cards)
            preloaded_skills = (
                compose_skills(
                    [sk.name for sk in active_skill_objects],
                    loader=self.skill_loader,
                    include_dependencies=False,
                )
                if active_skill_objects
                else None
            )
            skill_index = self.skill_loader.format_index()
            trajectory.metadata["router_active"] = list(route_decision.active)
            trajectory.metadata["router_cards"] = list(route_decision.cards)
        else:
            self._activated_skills = set()
            skill_index = self.skill_loader.format_index() if self.enable_skill_autoloading or self.skills else None
            if self.skills and self.skills_as_cards:
                # Advertise the chosen skills by name and summary and let the
                # model pull a body with load_skill(name) when it decides it
                # needs one. A preloaded body is resent on every turn, so its
                # cost is roughly size x turns; a card is a line, and the body
                # is paid for once if at all. Measured on AgenticDataBench,
                # preloading the router's picks cost +124% in cache-adjusted
                # spend and did not improve the score.
                cards_text = "\n".join(
                    _skill_card(self.skill_loader, name) for name in self.skills
                )
                preloaded_skills = None
            else:
                preloaded_skills = (
                    compose_skills(self.skills, loader=self.skill_loader, include_dependencies=True)
                    if self.skills
                    else None
                )

        messages = [
            {
                "role": "system",
                "content": build_system_prompt(
                    signature=self.signature,
                    inline_task=self._inline_task,
                    inline_outputs=self._inline_outputs,
                    inline_output_types=self._inline_output_types,
                    inputs=bound_inputs,
                    skill_index=skill_index,
                    preloaded_skills=preloaded_skills,
                    skill_cards=cards_text,
                    router_active=self.enable_router,
                ),
            },
            {"role": "user", "content": build_initial_user_message(bound_inputs)},
        ]
        # Track digest state per active skill to swap full bodies for digests
        # after `digest_after_turn` turns of being present.
        skill_first_seen_turn: dict[str, int] = {
            sk.name: 1 for sk in active_skill_objects
        }
        digested_skills: set[str] = set()
        final_state: dict[str, Any] = {}
        task_description, _ = _task_and_outputs(self.signature, self._inline_task, self._inline_outputs)

        with Interpreter(
            timeout=self.timeout,
            security=self._security,
            max_submit_bytes=self.max_submit_bytes,
            block_network=self.block_network,
        ) as interpreter:
            if self.sub_lm_spec is not None:
                interpreter.configure_lm(self.sub_lm_spec)
            if bound_inputs:
                interpreter.set_inputs(bound_inputs)

            turn_counter = 0
            # LM calls that never became a turn (truncated / no code block).
            # Retried rather than executed, but still billed.
            unbilled_calls: list[dict[str, Any]] = []
            if knowledge_metadata.get("operation_selection_lm_calls") == 1:
                unbilled_calls.append(
                    {
                        "prompt_tokens": knowledge_metadata.get(
                            "operation_selection_prompt_tokens"
                        ),
                        "completion_tokens": knowledge_metadata.get(
                            "operation_selection_completion_tokens"
                        ),
                        "cached_tokens": knowledge_metadata.get(
                            "operation_selection_cached_tokens"
                        ),
                        "reasoning_tokens": knowledge_metadata.get(
                            "operation_selection_reasoning_tokens"
                        ),
                        "lm_call_seconds": knowledge_metadata.get(
                            "operation_selection_lm_seconds"
                        ),
                    }
                )
            next_turn_type = "normal"
            reached_max = False
            verifier_repair_history: list[dict[str, Any]] = []
            self._repair_counts = {}
            self._integrity_rejections = 0
            timeout_recoveries = 0

            while turn_counter < self.max_turns:
                # Pre-LM-call: budget urgency hint + digest swap for the system message.
                budget_remaining = self.max_turns - turn_counter
                if (
                    self.reserve_finalize_turns > 0
                    and budget_remaining <= self.reserve_finalize_turns
                    and messages
                    and messages[-1].get("role") == "user"
                    and "[BUDGET]" not in messages[-1].get("content", "")
                ):
                    messages[-1] = {
                        "role": messages[-1]["role"],
                        "content": (
                            "[BUDGET] Only "
                            f"{budget_remaining} turn(s) remain. Stop exploring; "
                            "produce your best answer THIS turn and call SUBMIT(...) before the end of the block.\n\n"
                            + messages[-1]["content"]
                        ),
                    }
                if (
                    self.enable_router
                    and self.digest_after_turn is not None
                    and turn_counter >= self.digest_after_turn
                ):
                    self._maybe_digest_active_skills(
                        messages,
                        active_skill_objects,
                        digested_skills,
                        force=False,
                    )
                if (
                    self.enable_router
                    and self.max_prompt_tokens is not None
                    and _estimate_tokens(messages[0].get("content", "")) > self.max_prompt_tokens
                ):
                    self._maybe_digest_active_skills(
                        messages,
                        active_skill_objects,
                        digested_skills,
                        force=True,
                    )
                response_text, raw_response, lm_call_seconds = _call_lm_with_meta(
                    self.outer_lm, messages
                )
                usage = _extract_usage(raw_response)
                prompt_tokens = _usage_field(usage, "prompt_tokens")
                completion_tokens = _usage_field(usage, "completion_tokens")
                total_tokens = _usage_field(usage, "total_tokens")
                cached_tokens = _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens")
                reasoning_tokens = _usage_nested_field(usage, "completion_tokens_details", "reasoning_tokens")
                if _looks_truncated(response_text):
                    messages.append({"role": "assistant", "content": response_text})
                    # truncated turns still count toward the budget to avoid loops.
                    turn_counter += 1
                    unbilled_calls.append({
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cached_tokens": cached_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "lm_call_seconds": lm_call_seconds,
                    })
                    truncation_msg = (
                        "Your previous response was truncated before the closing code fence. "
                        "Rewrite that turn in one complete ```python block under 30 lines."
                    )
                    # BUG-LIB-3: if the truncation retry IS the final turn,
                    # tell the LM that explicitly (predicate matches the
                    # normal-path one: after-increment, next iteration is the
                    # last iff turn_counter + 1 == max_turns).
                    if (turn_counter + 1) == self.max_turns:
                        truncation_msg += _final_turn_suffix()
                    messages.append({"role": "user", "content": truncation_msg})
                    if turn_counter >= self.max_turns:
                        reached_max = True
                    continue

                # No runnable code at all (pure prose, or a fence-less response
                # that isn't valid Python). Sibling of the truncation guard
                # above: don't ship prose to the worker just to get a SyntaxError
                # back — short-circuit with a clean "resend a block" signal.
                selected_code = _select_code_block(response_text)
                if selected_code is None:
                    messages.append({"role": "assistant", "content": response_text})
                    turn_counter += 1
                    unbilled_calls.append({
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cached_tokens": cached_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "lm_call_seconds": lm_call_seconds,
                    })
                    no_code_msg = (
                        "Your previous response contained no complete ```python code block. "
                        "Reply with exactly one complete ```python block (or call SUBMIT(...) "
                        "if you are done)."
                    )
                    if (turn_counter + 1) == self.max_turns:
                        no_code_msg += _final_turn_suffix()
                    messages.append({"role": "user", "content": no_code_msg})
                    if turn_counter >= self.max_turns:
                        reached_max = True
                    continue

                turn_counter += 1
                current_turn_type = next_turn_type
                next_turn_type = "normal"

                code = selected_code
                self._log(f"\n=== Turn {turn_counter}/{self.max_turns} ({current_turn_type}) ===\n{code}")
                started = time.perf_counter()
                worker_started = time.monotonic()
                try:
                    result = interpreter.execute(code)
                except (WorkerTimeout, WorkerProtocolError) as exc:
                    worker_died = not isinstance(exc, WorkerTimeout)
                    worker_execute_seconds = time.monotonic() - worker_started
                    duration = time.perf_counter() - started
                    trajectory.append(
                        TurnRecord(
                            turn=turn_counter,
                            code=code,
                            stdout="",
                            stderr="",
                            error=f"{type(exc).__name__}: {exc}",
                            submitted=False,
                            state=dict(final_state),
                            response_text=response_text,
                            duration_s=duration,
                            token_usage=usage,
                            validation_errors=[],
                            turn_type=current_turn_type,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            cached_tokens=cached_tokens,
                            reasoning_tokens=reasoning_tokens,
                            lm_call_seconds=lm_call_seconds,
                            worker_execute_seconds=worker_execute_seconds,
                        )
                    )
                    # A timeout kills the worker, so recovery means restarting it
                    # and telling the model its namespace is gone. Ending the run
                    # here instead throws away every prior turn: on a 246-task
                    # benchmark this lost 20 tasks outright, several of which had
                    # already done the analysis and were formatting output.
                    #
                    # A worker that *dies* leaves the run in the same state, so it
                    # recovers the same way. This is not hypothetical: an OOM kill
                    # is the usual way a Fabric worker goes, and on Python 3.10 a
                    # deep enough recursion overflows the C stack and segfaults
                    # before any timeout can fire.
                    if timeout_recoveries < self.recover_worker_timeouts:
                        timeout_recoveries += 1
                        try:
                            # A timed-out worker is already killed, but one that
                            # died on a closed pipe may still be running, and
                            # start() refuses to run twice.
                            interpreter.kill()
                            interpreter.start()
                            if self.sub_lm_spec is not None:
                                interpreter.configure_lm(self.sub_lm_spec)
                            if bound_inputs:
                                interpreter.set_inputs(bound_inputs)
                            interpreter.warmup()
                        except Exception as restart_error:
                            trajectory.turns[-1].error = (
                                f"{trajectory.turns[-1].error}; worker restart "
                                f"failed: {type(restart_error).__name__}: "
                                f"{restart_error}"
                            )
                        else:
                            if worker_died:
                                what_happened = (
                                    "Your code killed the Python worker process (it ran "
                                    "out of memory, crashed, or exited) and the worker "
                                    "was restarted."
                                )
                                what_to_do = (
                                    "That approach cannot finish as written. Do not "
                                    "repeat it. Use far less memory: read only the "
                                    "columns you need, process in chunks rather than "
                                    "loading everything, avoid building large "
                                    "intermediate copies, and check your logic on a "
                                    "sample first. Do not call sys.exit() or os._exit()."
                                )
                            else:
                                what_happened = (
                                    f"Your code was still running after {self.timeout:.0f}s "
                                    "and the worker was restarted."
                                )
                                what_to_do = (
                                    "That approach is too slow for this data. Do not "
                                    "repeat it. Work in a way that finishes: read only "
                                    "the columns you need, aggregate or filter in the "
                                    "query rather than in Python, process in chunks, or "
                                    "sample to check your logic before running on "
                                    "everything."
                                )
                            messages.append({"role": "assistant", "content": response_text})
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"{what_happened} EVERY variable is gone; "
                                    "inputs are re-bound and importable again.\n"
                                    f"{what_to_do}\n"
                                    f"Recovery {timeout_recoveries} of "
                                    f"{self.recover_worker_timeouts}."
                                ),
                            })
                            turn_counter += 1
                            continue

                    return RLMResult(
                        submitted=False,
                        payload=None,
                        trajectory=trajectory,
                        final_state=final_state,
                        max_turns=self.max_turns,
                        failure_reason="worker_died" if worker_died else "worker_timeout",
                        **_aggregate_trajectory_metrics(trajectory, unbilled_calls),
                    )
                except Exception as exc:
                    # Capture any unexpected worker/interpreter failure as a turn and stop.
                    # Re-raising would lose the trajectory; silent retry can mask systemic bugs.
                    worker_execute_seconds = time.monotonic() - worker_started
                    duration = time.perf_counter() - started
                    trajectory.append(
                        TurnRecord(
                            turn=turn_counter,
                            code=code,
                            stdout="",
                            stderr="",
                            error=f"{type(exc).__name__}: {exc}",
                            submitted=False,
                            state=dict(final_state),
                            response_text=response_text,
                            duration_s=duration,
                            token_usage=usage,
                            validation_errors=[],
                            turn_type=current_turn_type,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            cached_tokens=cached_tokens,
                            reasoning_tokens=reasoning_tokens,
                            lm_call_seconds=lm_call_seconds,
                            worker_execute_seconds=worker_execute_seconds,
                        )
                    )
                    return RLMResult(
                        submitted=False,
                        payload=None,
                        trajectory=trajectory,
                        final_state=final_state,
                        max_turns=self.max_turns,
                        failure_reason="worker_error",
                        **_aggregate_trajectory_metrics(trajectory, unbilled_calls),
                    )
                worker_execute_seconds = time.monotonic() - worker_started
                duration = time.perf_counter() - started
                # Turns that never reached the worker (parent-side security
                # rejection) carry a placeholder empty state; keep the last
                # real snapshot instead of wiping it. getattr() because
                # duck-typed interpreter fakes/shims may not carry the flag —
                # absent means "reached the worker" (the historic behavior).
                if getattr(result, "reached_worker", True):
                    final_state = result.state
                else:
                    result = ExecResult(
                        ok=result.ok,
                        submitted=result.submitted,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        state=dict(final_state),
                        error=result.error,
                        submit_payload=result.submit_payload,
                        reached_worker=False,
                    )
                # Router: pick up any `activate_skill(name)` calls from this turn.
                if self.enable_router and result.stdout:
                    for activated_name in _scan_activation_markers(result.stdout):
                        if activated_name in self._activated_skills:
                            continue
                        try:
                            new_skill = self.skill_loader.load(activated_name)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "activate_skill(%r) failed to load: %s", activated_name, exc
                            )
                            continue
                        self._activated_skills.add(activated_name)
                        if self.enable_verifier and not any(
                            sk.name == activated_name for sk in self._loaded_skills
                        ):
                            self._loaded_skills.append(new_skill)
                        skill_first_seen_turn.setdefault(activated_name, turn_counter)
                validation = (
                    validate_submit_payload(
                        result.submit_payload,
                        required_output_fields,
                        self._inline_output_types,
                    )
                    if result.submitted
                    else OutputValidationResult()
                )

                trajectory.append(
                    TurnRecord(
                        turn=turn_counter,
                        code=code,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        error=result.error,
                        submitted=result.submitted,
                        state=result.state,
                        response_text=response_text,
                        duration_s=duration,
                        token_usage=usage,
                        validation_errors=list(validation.errors),
                        turn_type=current_turn_type,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cached_tokens=cached_tokens,
                        reasoning_tokens=reasoning_tokens,
                        lm_call_seconds=lm_call_seconds,
                        worker_execute_seconds=worker_execute_seconds,
                        submit_payload=result.submit_payload if result.submitted else None,
                    )
                )

                # NEW-H stuck-loop circuit breaker: bail early if the last
                # ``stuck_loop_threshold`` CONSECUTIVE turns are all failures
                # with the same normalized code and the same error type.
                # An intervening success or non-erroring turn resets the chain.
                if (
                    not result.submitted
                    and result.error is not None
                    and self.stuck_loop_threshold is not None
                ):
                    threshold = self.stuck_loop_threshold
                    if len(trajectory.turns) >= threshold:
                        tail = trajectory.turns[-threshold:]
                        all_failed = all(
                            (not t.submitted) and t.error is not None
                            for t in tail
                        )
                        if all_failed:
                            recent_sigs = {_loop_signature(t) for t in tail}
                            if len(recent_sigs) == 1:
                                return RLMResult(
                                    submitted=False,
                                    payload=None,
                                    trajectory=trajectory,
                                    final_state=final_state,
                                    max_turns=self.max_turns,
                        failure_reason="stuck_loop",
                                    **_aggregate_trajectory_metrics(trajectory, unbilled_calls),
                                )

                if result.submitted:
                    if not validation.ok:
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append(
                            {
                                "role": "user",
                                "content": self._format_validation_feedback(result, turn_counter, validation),
                            }
                        )
                        next_turn_type = "validation_repair"
                        if turn_counter >= self.max_turns:
                            reached_max = True
                        continue

                    verifier_feedback = self._run_skill_verifiers(interpreter, result.submit_payload)
                    if verifier_feedback is not None:
                        feedback_text, history_entry = verifier_feedback
                        if history_entry is not None:
                            history_entry["turn"] = turn_counter
                            verifier_repair_history.append(history_entry)
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": feedback_text})
                        next_turn_type = "verifier_repair"
                        if turn_counter >= self.max_turns:
                            reached_max = True
                        continue

                    output_feedback = self._run_output_validator(result.submit_payload)
                    if output_feedback is None:
                        output_feedback = self._run_output_validator_context(
                            result.submit_payload,
                            {
                                "inputs": bound_inputs,
                                "state": result.state,
                                "turn": turn_counter,
                                "trajectory": trajectory,
                            },
                        )
                    if output_feedback is None:
                        output_feedback = self._run_analytical_integrity(
                            result.submit_payload,
                            {
                                "inputs": bound_inputs,
                                "state": result.state,
                                "turn": turn_counter,
                                "trajectory": trajectory,
                            },
                        )
                    if output_feedback is not None:
                        feedback_text, history_entry = output_feedback
                        if history_entry is not None:
                            history_entry["turn"] = turn_counter
                            verifier_repair_history.append(history_entry)
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": feedback_text})
                        next_turn_type = "verifier_repair"
                        if turn_counter >= self.max_turns:
                            reached_max = True
                        continue

                    if verifier_repair_history:
                        trajectory.metadata["verifier_repair_history"] = verifier_repair_history
                    return RLMResult(
                        submitted=True,
                        payload=result.submit_payload,
                        trajectory=trajectory,
                        final_state=result.state,
                        max_turns=self.max_turns,
                        **_aggregate_trajectory_metrics(trajectory, unbilled_calls),
                    )

                messages.append({"role": "assistant", "content": response_text})
                # BUG-LIB-3: detect whether the next while-iteration will be
                # the FINAL LM turn so the user message can include an explicit
                # submit-now nudge. After this body, turn_counter has already
                # been incremented for this turn; the loop will run once more
                # iff turn_counter < max_turns, and that next run is the LAST
                # iff turn_counter + 1 == max_turns.
                is_final_turn = (turn_counter + 1) == self.max_turns
                messages.append({
                    "role": "user",
                    "content": self._format_feedback(
                        result, turn_counter, is_final_turn=is_final_turn
                    ),
                })

            if not reached_max:
                reached_max = True

        if verifier_repair_history:
            trajectory.metadata["verifier_repair_history"] = verifier_repair_history
        last_turn = trajectory.turns[-1] if trajectory.turns else None
        last_verifier_turn = (
            verifier_repair_history[-1].get("turn")
            if verifier_repair_history
            else None
        )
        validation_failed = bool(
            last_turn
            and last_turn.submitted
            and (
                last_turn.validation_errors
                or last_verifier_turn == last_turn.turn
            )
        )
        if not validation_failed:
            # Say so out loud. A truncated run returns a wrong-looking answer
            # that is indistinguishable from a wrong one, and the fix is a
            # config change the caller can only make if they know to. How many
            # turns a model needs for the same task varies a lot between
            # models, so a cap that suits one can silently starve another.
            logger.warning(
                "RLM ran out of turns after %d of %d without calling SUBMIT. "
                "The result is truncated, not necessarily wrong. Raise max_turns "
                "if this recurs -- models differ in how many turns they take.",
                turn_counter,
                self.max_turns,
            )
        else:
            latest_assertion = (
                verifier_repair_history[-1].get("assertion")
                if last_verifier_turn == last_turn.turn
                else "; ".join(last_turn.validation_errors)
            )
            logger.warning(
                "RLM exhausted its %d-turn budget after the latest submitted "
                "payload failed validation: %s",
                self.max_turns,
                latest_assertion,
            )
        return RLMResult(
            submitted=False,
            payload=None,
            trajectory=trajectory,
            final_state=final_state,
            max_turns=self.max_turns,
            failure_reason=(
                "output_validation_failed" if validation_failed else "max_turns"
            ),
            **_aggregate_trajectory_metrics(trajectory, unbilled_calls),
        )

    def _repair_nudge_suffix(self, key: str) -> str:
        """Return the diversity-nudge text to append to a repair message.

        Behavior depends on ``FABRIC_RLM_REPAIR_NUDGE`` (see ``_repair_nudge_mode``):
        ``off`` -> always empty; ``static`` -> always the line; ``escalating`` ->
        the line only once ``key`` has failed at least twice in this run. ``key``
        identifies the repair target (validation, a skill verifier, or the output
        validator) so the escalating counter tracks "same field" repeat failures.
        """
        mode = _repair_nudge_mode()
        if mode == "off":
            return ""
        count = self._repair_counts.get(key, 0) + 1
        self._repair_counts[key] = count
        if mode == "static" or count >= 2:
            return "\n" + _REPAIR_DIVERSITY_LINE
        return ""

    def _format_validation_feedback(
        self, result: ExecResult, turn: int, validation: OutputValidationResult
    ) -> str:
        failures = "\n".join(f"- {error}" for error in validation.errors)
        return (
            f"SUBMIT from turn {turn} failed output validation.\n"
            f"Validation failures:\n{failures}\n"
            f"Submitted payload preview: {_preview_payload(result.submit_payload)}\n"
            f"State keys: {', '.join(result.state.keys()) or '(none)'}\n"
            "Repair the final answer and call SUBMIT() again with all required fields present and non-blank."
            + self._repair_nudge_suffix("output_validation")
        )

    def _maybe_digest_active_skills(
        self,
        messages: list[dict[str, Any]],
        active_skill_objects: list[Skill],
        digested_skills: set[str],
        *,
        force: bool,
    ) -> None:
        """Replace active skill bodies in the system message with short digests.

        Mutates ``messages[0]['content']`` in-place. Idempotent: skills already
        in ``digested_skills`` are skipped. When ``force=True`` digests every
        active skill regardless.

        Cost warning -- this defeats prompt caching. Providers match cached
        prefixes byte-for-byte from the start of the prompt, so rewriting the
        system message invalidates the cache for that request and every request
        after it. Cached input is billed at a tenth of fresh input on Fabric's
        gpt-5 series (4.20 vs 42.02 CU-seconds per 1k tokens), so a digest that
        removes N tokens from the prompt re-prices the *surviving* prefix at 10x.
        Digesting is a net loss unless the skill bodies being dropped are large
        relative to what remains.

        Measured on a 400-question benchmark run: 87% of prompt tokens were
        served from cache, and prompt caching accounted for a 51% reduction in
        total cost. Turn 1 was already 86% cached because the skill prefix is
        identical across tasks -- an effect this method destroys.

        Only reach for this when the prompt would otherwise exceed the context
        window. ``digest_after_turn`` defaults to ``None`` (off) for that reason.
        """

        if not messages or messages[0].get("role") != "system":
            return
        sys_content = messages[0].get("content") or ""
        changed = False
        for skill in active_skill_objects:
            if not force and skill.name in digested_skills:
                continue
            full_block = f"## Skill: {skill.name}"
            if full_block not in sys_content:
                continue
            digest = _digest_for_skill(skill)
            # Replace from "## Skill: <name>" up to (but not including) the next
            # "## Skill:" header or end of string.
            start = sys_content.find(full_block)
            if start == -1:
                continue
            tail_search_start = start + len(full_block)
            next_header = sys_content.find("\n## Skill: ", tail_search_start)
            end = len(sys_content) if next_header == -1 else next_header
            sys_content = sys_content[:start] + digest + "\n" + sys_content[end:]
            digested_skills.add(skill.name)
            changed = True
        if changed:
            messages[0] = {"role": "system", "content": sys_content}

    def _run_skill_verifiers(
        self, interpreter: Interpreter, payload: Mapping[str, Any] | None
    ) -> tuple[str, dict[str, Any] | None] | None:
        """Execute each loaded skill's ``verify(payload)`` against the SUBMIT payload.

        Returns ``None`` when every verifier passes (or there are none / the
        payload cannot be JSON-serialized / the verifier is buggy — see graceful
        degrade below). Returns ``(feedback_str, history_entry)`` for the LM
        when any verifier raises ``AssertionError``: that fails the SUBMIT and
        triggers a ``verifier_repair`` turn. ``history_entry`` is a dict with
        ``skill``, ``rejected_payload``, and ``assertion`` keys (or ``None`` if
        the rejection couldn't be summarized) so the caller can thread it into
        verifier-repair feedback metadata. Non-AssertionError failures are logged
        and skipped to avoid blocking valid answers behind a buggy verifier.
        """

        if not self.enable_verifier:
            return None
        if not self._loaded_skills:
            return None

        try:
            payload_json = json.dumps(payload)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Skipping skill verifiers: SUBMIT payload is not JSON-serializable (%s).",
                exc,
            )
            return None

        for skill in self._loaded_skills:
            if not skill.verifier_source:
                continue
            if self.enable_router and skill.name not in self._activated_skills:
                continue
            verifier_code = (
                f"{skill.verifier_source}\n\n"
                "import json as _fabric_rlm_json\n"
                f"_fabric_rlm_payload = _fabric_rlm_json.loads({json.dumps(payload_json)})\n"
                "verify(_fabric_rlm_payload)\n"
            )
            try:
                exec_result = interpreter.execute(verifier_code)
            except WorkerTimeout:
                logger.warning(
                    "Skill %r verifier timed out; accepting payload (graceful degrade).",
                    skill.name,
                )
                continue
            except Exception as exc:  # noqa: BLE001 - any host-side failure means a buggy verifier
                logger.warning(
                    "Skill %r verifier raised host error %s: %s; accepting payload (graceful degrade).",
                    skill.name,
                    type(exc).__name__,
                    exc,
                )
                continue

            if exec_result.ok:
                continue

            error_text = exec_result.error or ""
            if "AssertionError" in error_text:
                message = _extract_assertion_message(error_text)
                feedback = (
                    f"Your SUBMIT was rejected by the `{skill.name}` skill verifier:\n\n"
                    f"AssertionError: {message}\n\n"
                    f"Submitted payload preview: {_preview_payload(payload)}\n"
                    "Recompute the offending field(s) and call SUBMIT(...) again."
                    + self._repair_nudge_suffix(f"verifier:{skill.name}")
                )
                history_entry: dict[str, Any] = {
                    "skill": skill.name,
                    "rejected_payload": dict(payload) if isinstance(payload, Mapping) else payload,
                    "assertion": message,
                }
                return feedback, history_entry
            logger.warning(
                "Skill %r verifier raised non-AssertionError; accepting payload anyway. "
                "Traceback tail: %s",
                skill.name,
                error_text[-500:],
            )
        return None

    def _run_output_validator(
        self, payload: Mapping[str, Any] | None
    ) -> tuple[str, dict[str, Any] | None] | None:
        """Run the configured global output validator on a SUBMIT payload.

        The validator is a callable (typically a host contract verifier)
        that raises :class:`AssertionError` on contract violation. Returns
        ``None`` when no validator is configured, the validator passes, or
        the validator itself misbehaves (graceful degrade — never block a
        valid SUBMIT behind a buggy host-side validator). Returns a
        ``(feedback, history_entry)`` tuple when the validator raises an
        ``AssertionError`` so the caller can drive a verifier-repair turn.
        """

        if self.output_validator is None:
            return None
        try:
            self.output_validator(payload or {})
        except AssertionError as exc:
            message = str(exc) or "output validator rejected the SUBMIT payload."
            feedback = (
                "Your SUBMIT was rejected by the output-format validator:\n\n"
                f"AssertionError: {message}\n\n"
                f"Submitted payload preview: {_preview_payload(payload)}\n"
                "Repair the `output` field and call SUBMIT(...) again."
                + self._repair_nudge_suffix("output_validator")
            )
            history_entry: dict[str, Any] = {
                "skill": "output_validator",
                "rejected_payload": dict(payload) if isinstance(payload, Mapping) else payload,
                "assertion": message,
            }
            return feedback, history_entry
        except Exception as exc:  # noqa: BLE001 - graceful degrade for buggy validators
            logger.warning(
                "Output validator raised non-AssertionError %s: %s; "
                "accepting payload (graceful degrade).",
                type(exc).__name__,
                exc,
            )
        return None

    # Two rejections per run is the repair budget for the heuristic screens
    # below; "strict" mode has no budget and keeps rejecting until the run
    # runs out of turns, so a known-bad answer is never returned as success.
    # They catch prose that contradicts its own numbers; they do not decide
    # the analysis, so past that point the answer is accepted and the
    # unresolved findings are recorded on the trajectory instead.
    _ANALYTICAL_INTEGRITY_MAX_REJECTIONS = 2

    def _analytical_integrity_problems(
        self, payload: Mapping[str, Any] | None, context: Mapping[str, Any]
    ) -> list[str]:
        """Source-agnostic checks on a SUBMIT payload and the code behind it.

        Directional prose must agree with its printed numbers; a task that
        asks to rank by a concept must get an answer that shows that metric
        and code that sorted by it; multidimensional candidates must not
        have been collapsed into independent per-dimension filters. Each
        check reads only the payload and the trajectory, never the sources.
        """
        texts: list[str] = []
        stack: list[Any] = list((payload or {}).values())
        while stack:
            item = stack.pop()
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, Mapping):
                stack.extend(item.values())
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
        combined = "\n".join(texts)
        problems: list[str] = []
        problems.extend(check_answer_hygiene(combined))
        problems.extend(check_directional_claims(combined))

        task_text, _ = _task_and_outputs(self.signature, self._inline_task, self._inline_outputs)
        request = infer_requested_ranking(task_text or "")
        trajectory = context.get("trajectory")
        turns = list(getattr(trajectory, "turns", None) or [])
        if request is not None:
            problems.extend(check_ranking_disclosure(combined, request))
            drift = detect_ranking_drift(turns, request, answer_text=combined)
            if drift is not None:
                problems.append(drift.message)
        if trajectory is not None and hasattr(trajectory, "diagnose"):
            # The detector only reports a tuple loss that no later statement
            # repaired on the same dimensions, so every issue here is live.
            for issue in trajectory.diagnose():
                if issue.kind == "cartesian_candidate_filter":
                    problems.append(f"Turn {issue.turn}: {issue.message}")
        return problems

    def _run_analytical_integrity(
        self, payload: Mapping[str, Any] | None, context: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any] | None] | None:
        """Drive a verifier-repair turn when the analytical integrity screen fails.

        Off when ``analytical_integrity=False`` or the environment sets
        ``FABRIC_RLM_ANALYTICAL_INTEGRITY=0``. Graceful degrade like the
        other validators: a bug in the screen never blocks a valid answer.
        """
        mode = getattr(self, "analytical_integrity", "repair")
        if mode == "off":
            return None
        env_mode = os.environ.get("FABRIC_RLM_ANALYTICAL_INTEGRITY", "").strip().lower()
        if env_mode in {"0", "false", "off", "no"}:
            return None
        if env_mode == "strict":
            mode = "strict"
        try:
            problems = self._analytical_integrity_problems(payload, context)
        except Exception as exc:  # noqa: BLE001 - graceful degrade
            logger.warning(
                "Analytical integrity check raised %s: %s; accepting payload (graceful degrade).",
                type(exc).__name__,
                exc,
            )
            return None
        if not problems:
            cleared = context.get("trajectory")
            if cleared is not None and hasattr(cleared, "metadata"):
                cleared.metadata.pop("analytical_integrity_unresolved", None)
            return None
        trajectory = context.get("trajectory")
        if trajectory is not None and hasattr(trajectory, "metadata"):
            # Always visible on the result: in repair mode this is what the
            # answer was accepted with, in strict mode what it was last
            # rejected for. Cleared again when a later submission passes.
            trajectory.metadata["analytical_integrity_unresolved"] = list(problems)
        if mode != "strict" and self._integrity_rejections >= self._ANALYTICAL_INTEGRITY_MAX_REJECTIONS:
            return None
        self._integrity_rejections += 1
        message = "\n".join(f"- {p}" for p in problems)
        feedback = (
            "Your SUBMIT was rejected by the analytical integrity check:\n\n"
            f"{message}\n\n"
            f"Submitted payload preview: {_preview_payload(payload)}\n"
            "Fix the analysis or the wording so the answer is faithful to the "
            "request and the numbers, then call SUBMIT(...) again."
            + self._repair_nudge_suffix("analytical_integrity")
        )
        history_entry: dict[str, Any] = {
            "skill": "analytical_integrity",
            "rejected_payload": dict(payload) if isinstance(payload, Mapping) else payload,
            "assertion": message,
        }
        return feedback, history_entry

    def _run_output_validator_context(
        self, payload: Mapping[str, Any] | None, context: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any] | None] | None:
        """Run an optional validator that can inspect runtime context.

        ``output_validator`` is intentionally payload-only for simple answer
        contracts. Artifact tasks often need to verify side effects (saved
        files, database rows, interpreter state), so this opt-in hook receives
        the same payload plus non-serialized runtime context.
        """

        if self.output_validator_context is None:
            return None
        try:
            self.output_validator_context(payload or {}, context)
        except AssertionError as exc:
            message = str(exc) or "context validator rejected the submitted artifact/state."
            feedback = (
                "Your SUBMIT was rejected by the context-aware output validator:\n\n"
                f"AssertionError: {message}\n\n"
                f"Submitted payload preview: {_preview_payload(payload)}\n"
                f"Context keys: {', '.join(context.keys()) or '(none)'}\n"
                "Inspect the artifact/state, fix the issue, and call SUBMIT(...) again."
                + self._repair_nudge_suffix("output_validator_context")
            )
            history_entry: dict[str, Any] = {
                "skill": "output_validator_context",
                "rejected_payload": dict(payload) if isinstance(payload, Mapping) else payload,
                "assertion": message,
                "context_keys": list(context.keys()),
            }
            return feedback, history_entry
        except Exception as exc:  # noqa: BLE001 - graceful degrade for buggy validators
            logger.warning(
                "Context-aware output validator raised non-AssertionError %s: %s; "
                "accepting payload (graceful degrade).",
                type(exc).__name__,
                exc,
            )
        return None

    def _format_feedback(
        self, result: ExecResult, turn: int, *, is_final_turn: bool = False
    ) -> str:
        stdout_text = _truncate_for_feedback(
            result.stdout,
            STDOUT_FEEDBACK_LIMIT,
            tail_ratio=_tail_ratio("FABRIC_RLM_STDOUT_TAIL_RATIO", _STDOUT_TAIL_RATIO_DEFAULT),
        )
        parts = [f"REPL output from turn {turn}:\n```\n{stdout_text}\n```"]
        if len(result.stdout) > STDOUT_FEEDBACK_LIMIT and _truncation_hint_enabled():
            parts.append(_build_truncation_hint(len(result.stdout), STDOUT_FEEDBACK_LIMIT))
        if not result.ok:
            error_text = _truncate_for_feedback(
                result.error or "",
                ERROR_FEEDBACK_LIMIT,
                tail_ratio=_tail_ratio("FABRIC_RLM_ERROR_TAIL_RATIO", _ERROR_TAIL_RATIO_DEFAULT),
            )
            parts.append(f"\nERROR:\n```\n{error_text}\n```\nWrite a recovery turn.")
            if re.search(
                r"NameError: name ['\"](?:null|true|false)['\"] is not defined",
                error_text,
            ):
                parts.append(f"\n{_PYTHON_LITERAL_RECOVERY_HINT}")
        elif result.stderr:
            stderr_text = _truncate_for_feedback(
                result.stderr,
                STDERR_FEEDBACK_LIMIT,
                tail_ratio=_tail_ratio("FABRIC_RLM_STDERR_TAIL_RATIO", _STDERR_TAIL_RATIO_DEFAULT),
            )
            parts.append(f"\nstderr:\n```\n{stderr_text}\n```")
        parts.append(f"\nState keys: {', '.join(result.state.keys()) or '(none)'}")
        if is_final_turn:
            # BUG-LIB-3: explicit final-turn nudge. Trace mining identified
            # turns burning the budget without ever submitting; making
            # "this is your last shot" salient recovers many of those.
            parts.append(_final_turn_suffix())
        else:
            parts.append("\nContinue with one complete Python code block, or call SUBMIT() if done.")
        return "".join(parts)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    # =====================================================================
    # v7 engine: delegate the loop to dspy.predict.RLM, keeping our
    # subprocess worker as the CodeInterpreter backend.
    # =====================================================================
    def _run_via_dspy(
        self,
        bound_inputs: dict[str, Any],
        required_output_fields: tuple[str, ...],
    ) -> RLMResult:
        """Run via dspy.predict.RLM using SubprocessPythonInterpreter.

        Slice 2 contract: same per-iter prompt, same code-fence rules, same REPL
        contract (llm_query/llm_query_batched/SUBMIT), same `extract` fallback at
        max_iterations as upstream dspy.

        Slice 3 additions:
          - Router-elected skill markdown is concatenated into the dspy action
            signature's instructions (lean: ≤ ``self.max_active_skills``).
          - Verifier wrapper: registered skill verifiers + ``output_validator``
            are run against the dspy ``Prediction``. On rejection we re-call
            ``dspy.RLM`` with feedback prepended to the first string input and
            ``max_iterations`` halved. Bounded retry (≤ 2).

        Source citation: dspy/predict/rlm.py:102-164 (RLM ctor, action+extract sigs),
        :576-625 (forward + Prediction shape).
        """
        import dspy
        from dspy.predict import RLM as DspyRLM

        from .interpreter import SubprocessPythonInterpreter

        trajectory = Trajectory(
            metadata={
                "max_turns": self.max_turns,
                "skills": list(self.skills),
                "skill_autoloading": self.enable_skill_autoloading,
                "router_enabled": self.enable_router,
                "engine": "v7-dspy",
            }
        )

        # ---- Slice 3: router-driven skill election + composed instructions
        skill_instructions, active_skill_objects = self._gather_skill_text_for_v7(
            bound_inputs, trajectory
        )
        self._loaded_skills = list(active_skill_objects) if self.enable_verifier else []
        self._activated_skills = {sk.name for sk in active_skill_objects}

        signature = self._build_dspy_signature(
            required_output_fields, extra_instructions=skill_instructions
        )

        outer_lm = self.outer_lm
        sub_lm = resolve_lm(self.sub_lm_spec) if self.sub_lm_spec is not None else outer_lm

        # Token accounting: dspy appends one history entry (with a usage dict)
        # per LM call. Snapshot history lengths now and harvest everything new
        # after the run so v7 results report the same total_* token fields as
        # the v6 engine. Track outer and sub LM separately unless identical.
        usage_tracked_lms: list[Any] = [outer_lm]
        if sub_lm is not outer_lm:
            usage_tracked_lms.append(sub_lm)
        pre_history_lens: list[int] = [
            len(h) if isinstance((h := getattr(lm, "history", None)), list) else 0
            for lm in usage_tracked_lms
        ]

        # ---- Slice 3: bounded verifier-wrapper retry loop
        MAX_VERIFIER_RETRIES = 2
        verifier_repair_history: list[dict[str, Any]] = []
        current_inputs = dict(bound_inputs)
        max_iter = self.max_turns
        prediction = None
        elapsed_total = 0.0
        last_failure_reason: str | None = None
        retry_instruction: str | None = None

        for attempt in range(MAX_VERIFIER_RETRIES + 1):
            attempt_signature = signature
            if retry_instruction:
                attempt_signature = signature.with_instructions(
                    (
                        f"{signature.instructions or ''}\n\n"
                        "REPAIR REQUIRED: The previous submission was rejected. "
                        f"{retry_instruction}"
                    ).strip()
                )
            interpreter = SubprocessPythonInterpreter(
                timeout=self.timeout,
                security=self._security,
                max_submit_bytes=self.max_submit_bytes,
            )
            t0 = time.time()
            try:
                with dspy.context(lm=outer_lm):
                    dspy_rlm_kwargs: dict[str, Any] = dict(
                        signature=attempt_signature,
                        sub_lm=sub_lm,
                        interpreter=interpreter,
                        max_iterations=max_iter,
                    )
                    if self.tools:
                        dspy_rlm_kwargs["tools"] = self.tools
                    rlm = DspyRLM(**dspy_rlm_kwargs)
                    prediction = rlm(**current_inputs)
            except Exception as exc:
                last_failure_reason = (
                    f"dspy.RLM raised {type(exc).__name__}: {exc}"
                )
                prediction = None
                break
            finally:
                elapsed_total += time.time() - t0
                try:
                    interpreter.shutdown()
                except Exception:
                    pass

            payload = self._extract_payload_from_prediction(prediction, attempt_signature)
            if not payload:
                if attempt < MAX_VERIFIER_RETRIES:
                    last_failure_reason = "dspy.RLM produced no output payload"
                    feedback_text = (
                        "Your previous run did not produce a valid final output payload. "
                        f"Call SUBMIT(...) with all required fields: "
                        f"{tuple(required_output_fields) or ('output',)}."
                    )
                    current_inputs = self._inputs_with_verifier_feedback(
                        current_inputs, feedback_text
                    )
                    retry_instruction = (
                        None
                        if any(isinstance(value, str) for value in current_inputs.values())
                        else feedback_text
                    )
                    if self.halve_max_iter_on_retry:
                        max_iter = max(1, (max_iter + 1) // 2)
                    continue
                last_failure_reason = "dspy.RLM produced no output payload"
                break

            validation = validate_submit_payload(
                payload,
                required_output_fields,
                self._inline_output_types,
            )
            if not validation.ok:
                last_failure_reason = "; ".join(validation.errors)
                if attempt < MAX_VERIFIER_RETRIES:
                    current_inputs = self._inputs_with_verifier_feedback(
                        current_inputs,
                        last_failure_reason,
                    )
                    retry_instruction = (
                        None
                        if any(isinstance(value, str) for value in current_inputs.values())
                        else last_failure_reason
                    )
                    if self.halve_max_iter_on_retry:
                        max_iter = max(1, (max_iter + 1) // 2)
                    continue
                break

            # Run verifiers + output_validator using a fresh legacy interpreter.
            # Must use context manager — Interpreter requires explicit .start().
            # Verifier code is loaded from trusted skill .md files; it does
            # not need to be subjected to the SecurityPolicy. Leaving this
            # interpreter policy-free also means skill verifiers that
            # happen to use APIs the policy would block (none in-tree do
            # today, but future contributors get a stable contract) still
            # work as authored.
            try:
                with Interpreter(
                    timeout=self.timeout,
                    max_submit_bytes=self.max_submit_bytes,
                    # Verifier code is trusted enough to skip the policy, but a
                    # skill can come from anywhere, and "the sandbox has no
                    # network" should not have a hole in it.
                    block_network=self.block_network,
                ) as verifier_interp:
                    verifier_feedback = self._run_skill_verifiers(verifier_interp, payload)
            except Exception as exc:
                logger.warning(
                    "v7 verifier interpreter failed to start (%s); "
                    "skipping skill verifiers (graceful degrade).",
                    exc,
                )
                verifier_feedback = None

            if verifier_feedback is None:
                verifier_feedback = self._run_output_validator(payload)
            if verifier_feedback is None:
                verifier_feedback = self._run_output_validator_context(
                    payload,
                    {
                        "inputs": current_inputs,
                        "attempt": attempt,
                        "prediction": prediction,
                        "trajectory": trajectory,
                    },
                )
            if verifier_feedback is None:
                verifier_feedback = self._run_analytical_integrity(
                    payload,
                    {
                        "inputs": current_inputs,
                        "attempt": attempt,
                        "prediction": prediction,
                        "trajectory": trajectory,
                    },
                )

            if verifier_feedback is None:
                # All checks pass – ship the prediction.
                last_failure_reason = None
                break

            # Verifier rejected. Record + maybe retry.
            feedback_text, history_entry = verifier_feedback
            if history_entry is not None:
                history_entry["turn"] = attempt
                verifier_repair_history.append(history_entry)
            last_failure_reason = (
                f"verifier rejected payload after attempt {attempt + 1}: "
                f"{feedback_text[:200]}"
            )
            if attempt >= MAX_VERIFIER_RETRIES:
                # Out of retries. The run is reported as not-submitted with
                # payload=None (matching the v6 engine's contract); the
                # rejected payload itself is preserved for inspection in
                # trajectory.metadata['verifier_repair_history'].
                break

            # Prepend feedback to first string input we can find.
            current_inputs = self._inputs_with_verifier_feedback(
                current_inputs, feedback_text
            )
            retry_instruction = (
                None
                if any(isinstance(value, str) for value in current_inputs.values())
                else feedback_text
            )
            if self.halve_max_iter_on_retry:
                max_iter = max(1, (max_iter + 1) // 2)  # ceil(remaining/2), floor 1

        # ---- Adapt prediction → RLMResult
        payload = (
            self._extract_payload_from_prediction(prediction, signature)
            if prediction is not None
            else {}
        )
        dspy_traj = (
            getattr(prediction, "trajectory", None) or [] if prediction is not None else []
        )
        for idx, event in enumerate(dspy_traj):
            if not isinstance(event, dict):
                continue
            output_text = str(event.get("output") or event.get("stdout") or "")
            # dspy marks final-output frames with 'FINAL:' prefix in the output.
            submitted_flag = bool(
                event.get("submitted")
                or event.get("final")
                or output_text.startswith("FINAL:")
            )
            error_text = event.get("error")
            # Heuristic: dspy puts errors as '[Error] ...' lines in the output.
            if not error_text and output_text.startswith("[Error]"):
                error_text = output_text
            trajectory.turns.append(
                TurnRecord(
                    turn=idx,
                    code=str(event.get("code") or event.get("action") or ""),
                    stdout=output_text,
                    stderr="",
                    error=error_text,
                    submitted=submitted_flag,
                    state={},
                    response_text=str(event.get("reasoning") or ""),
                    turn_type=str(event.get("type") or "normal"),
                )
            )
        if verifier_repair_history:
            trajectory.metadata["verifier_repair_history"] = verifier_repair_history

        token_totals = _sum_usage_from_histories(usage_tracked_lms, pre_history_lens)
        submitted = bool(payload) and last_failure_reason is None
        return RLMResult(
            submitted=submitted,
            payload=payload if submitted else None,
            trajectory=trajectory,
            final_state={},
            failure_reason=last_failure_reason,
            total_prompt_tokens=token_totals["prompt_tokens"],
            total_completion_tokens=token_totals["completion_tokens"],
            total_cached_tokens=token_totals["cached_tokens"],
            total_reasoning_tokens=token_totals["reasoning_tokens"],
            total_lm_seconds=elapsed_total,
        )

    def _gather_skill_text_for_v7(
        self, bound_inputs: dict[str, Any], trajectory: Trajectory
    ) -> tuple[str, list[Skill]]:
        """Run the v6 router in a v7-friendly way and return composed skill text.

        Returns ``(skill_instructions, active_skill_objects)``.
        ``skill_instructions`` is the string block we will append to the dspy
        Signature instructions. Empty when no skills elected.
        """
        active_skill_objects: list[Skill] = []
        composed_text = ""

        if self.enable_router:
            task_text_for_routing, _ = _task_and_outputs(
                self.signature, self._inline_task, self._inline_outputs
            )
            router = SkillRouter.from_loader(
                self.skill_loader,
                max_active_skills=self.max_active_skills,
                baseline_skill_names=self.router_baseline_skills,
                candidate_specificities=self.router_candidate_specificities,
                include_dependencies=self.router_include_dependencies,
            )
            # Same two-stage routing as the v6 path: inputs first, task-text
            # fallback only on zero input signal. See _route_with_task_fallback.
            route_decision, used_task_fallback = _route_with_task_fallback(
                router,
                task_text_for_routing,
                bound_inputs,
                explicit_skills=self.skills or None,
            )
            trajectory.metadata["router_used_task_text_fallback"] = used_task_fallback
            active_skill_objects = [
                router._by_name[name]
                for name in route_decision.active
                if name in router._by_name
            ]
            trajectory.metadata["router_active"] = list(route_decision.active)
            trajectory.metadata["router_cards"] = list(route_decision.cards)
            if active_skill_objects:
                composed_text = compose_skills(
                    [sk.name for sk in active_skill_objects],
                    loader=self.skill_loader,
                    include_dependencies=False,
                )
        elif self.skills:
            composed_text = compose_skills(
                self.skills, loader=self.skill_loader, include_dependencies=True
            )
            active_skill_objects = [
                sk for sk in (self.skill_loader.load(n) for n in self.skills) if sk is not None
            ]

        if composed_text:
            instructions = (
                "You also have access to the following SKILLS. Read them carefully "
                "and follow their procedures when applicable.\n\n"
                + composed_text
            )
            return instructions, active_skill_objects
        return "", active_skill_objects

    def _extract_payload_from_prediction(self, prediction: Any, signature: Any) -> dict[str, Any]:
        if prediction is None:
            return {}
        payload: dict[str, Any] = {}
        try:
            for field_name in signature.output_fields:
                if field_name == "trajectory":
                    continue
                if hasattr(prediction, field_name):
                    payload[field_name] = getattr(prediction, field_name)
        except Exception:
            return {}
        return payload

    @staticmethod
    def _inputs_with_verifier_feedback(
        inputs: dict[str, Any], feedback: str
    ) -> dict[str, Any]:
        """Return a copy of ``inputs`` with verifier feedback prepended to the
        first string-valued field. Retry instructions carry feedback when no
        string input exists."""
        new_inputs = dict(inputs)
        for key, value in new_inputs.items():
            if isinstance(value, str):
                new_inputs[key] = (
                    "VERIFIER FEEDBACK (your previous SUBMIT was rejected — fix it):\n"
                    f"{feedback}\n\n"
                    "ORIGINAL INPUT:\n"
                    f"{value}"
                )
                return new_inputs
        return new_inputs

    def _build_dspy_signature(
        self,
        required_output_fields: tuple[str, ...],
        *,
        extra_instructions: str = "",
    ) -> Any:
        """Build (or pass through) a dspy.Signature for the v7 dspy engine.

        Order of preference (matches v6 semantics):
        1. self.signature is already a dspy.Signature class → use as-is.
        2. self.signature is a "in -> out" string → pass to dspy.Signature(...).
        3. self._inline_task + self._inline_outputs → synthesise "inputs -> outputs".

        When ``extra_instructions`` is non-empty (Slice 3 skills wiring) the
        text is appended to the Signature's instructions block.
        """
        import dspy

        sig = self.signature
        if sig is not None and inspect.isclass(sig):
            built = sig
        elif isinstance(sig, str):
            built = dspy.Signature(sig)
        elif self._inline_task is not None:
            input_names = list(self._inline_inputs.keys()) or ["question"]
            output_names = list(required_output_fields) or ["output"]
            sig_str = f"{', '.join(input_names)} -> {', '.join(output_names)}"
            built = dspy.Signature(sig_str, instructions=self._inline_task)
        else:
            raise ValueError(
                "RLM(engine='v7-dspy') requires either a signature (dspy.Signature or 'in -> out' string) "
                "or from_task(task=..., inputs=..., outputs=...)."
            )

        if self._inline_output_types:
            type_contract = ", ".join(
                f"{name} must be {expected_type.__name__}"
                for name, expected_type in self._inline_output_types.items()
            )
            base = built.instructions or ""
            built = built.with_instructions(
                f"{base}\n\nRequired output type contract: {type_contract}.".strip()
            )
            for name in self._inline_output_types:
                # Keep raw SUBMIT values intact for strict post-execution validation.
                # DSPy's parser preserves strings only for nullable unions that include
                # str; Any alone JSON-decodes them, and concrete types may coerce them.
                built = built.with_updated_fields(name, type_=str | None | Any)

        if extra_instructions:
            base = built.instructions or ""
            new_instructions = (base + "\n\n" + extra_instructions).strip()
            built = built.with_instructions(new_instructions)
        return built


_FENCE_RE = re.compile(r"```([A-Za-z0-9_+\-]*)[^\n]*\n(.*?)```", re.DOTALL)
_PYTHON_FENCE_LANGS = {"python", "py", "python3"}


def _is_parseable_python(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """All COMPLETE fenced blocks as ``(lang, body)`` in document order.

    Truncated blocks (an opening fence with no closing ```) don't match and are
    handled upstream by ``_looks_truncated``.
    """
    return [(m.group(1).lower(), m.group(2).strip()) for m in _FENCE_RE.finditer(text)]


def _select_code_block(text: str) -> str | None:
    """Pick the runnable Python to execute, or ``None`` when the response has no
    runnable code (caller should ask the model to resend a code block).

    Selection order:

    * Prefer language-tagged (```python) blocks over untagged ``` blocks — an
      untagged trailing block is usually an *expected output* example, not code.
    * Within the chosen group, take the model's LAST parseable block — revisions
      come last — falling back to the last non-empty block so genuinely broken
      code still surfaces a real SyntaxError on the model's latest attempt.
    * With no fenced block at all, run the bare text only if it is itself valid
      Python (lenient fence-less fallback); otherwise it's prose -> ``None``.
    """
    blocks = _fenced_blocks(text)
    tagged = [body for lang, body in blocks if lang in _PYTHON_FENCE_LANGS]
    untagged = [body for lang, body in blocks if lang not in _PYTHON_FENCE_LANGS]
    for group in (tagged, untagged):
        non_empty = [b for b in group if b]
        if not non_empty:
            continue
        for body in reversed(non_empty):
            if _is_parseable_python(body):
                return body
        return non_empty[-1]
    stripped = text.strip()
    if stripped and _is_parseable_python(stripped):
        return stripped
    return None


def _extract_code(text: str) -> str:
    """Back-compat wrapper: the selected code block, or "" when none is runnable."""
    return _select_code_block(text) or ""


def validate_submit_payload(
    payload: Mapping[str, Any] | None,
    required_fields: Iterable[str],
    required_types: Mapping[str, type] | None = None,
) -> OutputValidationResult:
    """Validate declared SUBMIT outputs before accepting success.

    Declared output fields are required. ``None`` is invalid for every declared
    field, blank strings/bytes are invalid, and empty containers are invalid for
    core final-output names (``output``, ``answer``, ``result``, ``report``).
    Empty lists/dicts on more specific fields are allowed so tasks can validly
    return "no items found" without inventing placeholder content.
    """

    fields = _normalize_required_fields(required_fields)
    if not fields:
        return OutputValidationResult()
    if payload is None:
        return OutputValidationResult(tuple(f"Missing required output field {name!r}." for name in fields))
    if not isinstance(payload, Mapping):
        return OutputValidationResult((f"SUBMIT payload must be a mapping, got {type(payload).__name__}.",))

    errors: list[str] = []
    types = dict(required_types or {})
    for name in fields:
        if name not in payload:
            errors.append(f"Missing required output field {name!r}.")
            continue
        value = payload[name]
        error = _validate_required_value(name, value)
        if error:
            errors.append(error)
            continue
        expected_type = types.get(name)
        if expected_type is not None and not _matches_output_type(value, expected_type):
            errors.append(
                f"Required output field {name!r} must be "
                f"{expected_type.__name__}, got {type(value).__name__}."
            )
    return OutputValidationResult(tuple(errors))


def _validate_required_value(name: str, value: Any) -> str | None:
    if value is None:
        return f"Required output field {name!r} is None."
    if isinstance(value, str) and not value.strip():
        return f"Required output field {name!r} is a blank string."
    if isinstance(value, bytes) and not value.strip():
        return f"Required output field {name!r} is blank bytes."
    if _is_core_final_output_field(name) and isinstance(value, (Mapping, list, tuple, set, frozenset)) and not value:
        return f"Required core output field {name!r} is an empty {type(value).__name__}."
    return None


def _is_core_final_output_field(name: str) -> bool:
    return name.strip().lower() in CORE_FINAL_OUTPUT_FIELDS


def _matches_output_type(value: Any, expected_type: type) -> bool:
    strict_builtins = {bool, bytes, dict, float, int, list, set, str, tuple}
    if expected_type in strict_builtins:
        return type(value) is expected_type
    return isinstance(value, expected_type)


def _normalize_required_fields(required_fields: Iterable[str]) -> tuple[str, ...]:
    fields: list[str] = []
    for name in required_fields:
        if not isinstance(name, str):
            raise TypeError(f"Output field names must be strings, got {type(name).__name__}.")
        if not name:
            raise ValueError("Output field names must not be empty.")
        fields.append(name)
    return tuple(fields)


def _normalize_inline_outputs(
    outputs: list[str] | Mapping[str, type] | None,
) -> tuple[list[str], dict[str, type]]:
    if outputs is None:
        return [], {}
    if isinstance(outputs, Mapping):
        fields = list(_normalize_required_fields(outputs))
        types: dict[str, type] = {}
        for name, expected_type in outputs.items():
            if not isinstance(expected_type, type):
                raise TypeError(
                    f"Output type for {name!r} must be a Python type "
                    "(a concrete class; parameterized generics and unions are not supported), "
                    f"got {type(expected_type).__name__}."
                )
            types[name] = expected_type
        return fields, types
    return list(_normalize_required_fields(outputs)), {}


def _required_output_fields(
    signature: Any,
    inline_task: str | None,
    inline_outputs: list[str] | None,
) -> tuple[str, ...]:
    _, outputs = _task_and_outputs(signature, inline_task, inline_outputs)
    return _normalize_required_fields(outputs)


def _normalize_integrity_mode(value: Any) -> str:
    """``analytical_integrity`` accepts True/"repair", "strict", or False/"off"."""
    if value is True or value is None:
        return "repair"
    if value is False:
        return "off"
    text = str(value).strip().lower()
    if text in {"repair", "strict", "off"}:
        return text
    if text in {"1", "true", "yes", "on"}:
        return "repair"
    if text in {"0", "false", "no"}:
        return "off"
    raise ValueError("analytical_integrity must be True, False, 'repair', 'strict' or 'off'")


def _preview_payload(payload: Mapping[str, Any] | None) -> str:
    text = repr(payload)
    return text if len(text) <= 1000 else text[:997] + "..."


def _extract_assertion_message(traceback_text: str) -> str:
    """Pull the AssertionError message out of a traceback string."""

    for line in reversed(traceback_text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("AssertionError"):
            _, _, msg = stripped.partition(":")
            return msg.strip() or stripped
    return traceback_text.strip().splitlines()[-1] if traceback_text.strip() else ""


def _looks_truncated(text: str) -> bool:
    return "```" in text and text.count("```") % 2 == 1


def _call_lm_text(lm: Any, messages: list[dict[str, str]]) -> str:
    text, _response, _seconds = _call_lm_with_meta(lm, messages)
    return text


def _call_lm_with_meta(
    lm: Any, messages: list[dict[str, str]]
) -> tuple[str, Any, float]:
    """Call the LM and return (text, raw_response, elapsed_seconds).

    The raw response is returned so callers can extract structured metadata
    such as token usage. ``elapsed_seconds`` is measured around the call.

    For ``dspy.LM`` instances the public ``__call__`` returns only the
    completion text (a list of strings), so we fall back to peeking at
    ``lm.history[-1]`` which carries the canonical ``usage`` dict and the
    underlying litellm response object. This is essential for the adaptive
    engine's cost accounting.
    """

    started = time.monotonic()
    pre_history_len = 0
    history = getattr(lm, "history", None)
    if isinstance(history, list):
        pre_history_len = len(history)
    response = _call_lm(lm, messages)
    elapsed = time.monotonic() - started

    history = getattr(lm, "history", None)
    if isinstance(history, list) and len(history) > pre_history_len:
        # dspy appends one entry per LM call. Use the new entry as the
        # raw_response so _extract_usage finds the usage dict.
        new_entry = history[-1]
        if isinstance(new_entry, dict):
            return _response_to_text(response), new_entry, elapsed

    return _response_to_text(response), response, elapsed



def _reject_block_network_with_sub_lm(block_network: Any, sub_lm: Any) -> None:
    """A worker with no egress cannot call an LM from inside the sandbox.

    Only an *explicit* ``sub_lm=`` is rejected. ``sub_lm_spec`` is also set
    implicitly whenever ``lm`` is a spec, purely so ``call_lm()`` is available
    if the model reaches for it; refusing on that would make ``block_network``
    unusable for nearly everyone. Failing here beats failing on turn three with
    a connection error the model then tries to debug.
    """
    if block_network and sub_lm is not None:
        raise ValueError(
            "block_network=True cannot be combined with an explicit sub_lm: "
            "sub-LM calls are made from inside the worker, which has no "
            "network egress. Drop one of the two, or route the sub-LM through "
            "a proxy on 127.0.0.1 (loopback is permitted)."
        )


def _skill_card(loader: Any, name: str) -> str:
    """One-line advertisement for a skill: name, verifier flag, summary.

    Mirrors SkillRouter.card_text so both paths describe a skill the same way,
    without requiring the router to be enabled.
    """
    try:
        sk = loader.load(name)
    except Exception:
        return f"- {name}: (unavailable)"
    summary = getattr(sk, "summary", None) or getattr(sk, "title", None) or name
    tag = " [verifier]" if getattr(sk, "verifier_present", False) else ""
    return f"- {name}{tag}: {summary}"


def _call_lm(lm: Any, messages: list[dict[str, str]]) -> Any:
    try:
        signature = inspect.signature(lm)
        accepts_messages = "messages" in signature.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_messages = True

    if accepts_messages:
        return lm(messages=messages)
    return lm(messages[-1]["content"])


def _response_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        return _response_to_text(response[0]) if response else ""
    if isinstance(response, tuple):
        return _response_to_text(response[0]) if response else ""
    if isinstance(response, dict):
        for key in ("content", "text", "message"):
            if key in response:
                return _response_to_text(response[key])
        if "choices" in response:
            return _response_to_text(response["choices"])
    for attr in ("content", "text", "message"):
        if hasattr(response, attr):
            return _response_to_text(getattr(response, attr))
    return str(response)


def _normalize_usage_details(value: Any) -> dict[str, Any] | None:
    """Coerce a ``*_tokens_details`` payload to a plain ``dict[str, Any]``.

    OpenAI / LiteLLM / Anthropic SDKs expose the nested
    ``prompt_tokens_details`` and ``completion_tokens_details`` blocks as
    Pydantic models or duck-typed objects with attributes such as
    ``cached_tokens`` / ``reasoning_tokens``. DSPy stores
    ``lm.history[-1]['usage']`` as a top-level dict but does NOT deep-convert
    these nested values, so they reach ``_extract_usage`` as objects, not
    dicts. This helper normalizes both shapes so downstream
    ``_usage_nested_field`` can read the integers it needs.

    Conversion strategy (in order):

    1. ``None`` → ``None``.
    2. ``dict`` → shallow copy with ``None`` values stripped (preserves all
       provider-specific fields the caller may want to record in
       ``token_usage`` for debugging).
    3. Pydantic-style (``model_dump`` / ``dict``) → use the SDK's own
       serialization so any provider-specific fields are preserved.
    4. Duck-typed object → probe a known set of child attributes
       (``cached_tokens``, ``audio_tokens``, ``reasoning_tokens``,
       ``accepted_prediction_tokens``, ``rejected_prediction_tokens``).

    Returns ``None`` if no usable child fields are found.
    """

    if value is None:
        return None
    if isinstance(value, dict):
        out = {k: v for k, v in value.items() if v is not None}
        return out or None
    # Pydantic v2 / v1 SDK objects (OpenAI / LiteLLM responses).
    for serializer in ("model_dump", "dict"):
        method = getattr(value, serializer, None)
        if callable(method):
            try:
                serialized = method()
            except Exception:
                continue
            if isinstance(serialized, dict):
                out = {k: v for k, v in serialized.items() if v is not None}
                return out or None
    # Plain duck-typed object: probe well-known child names.
    candidate_keys = (
        "cached_tokens",
        "audio_tokens",
        "reasoning_tokens",
        "accepted_prediction_tokens",
        "rejected_prediction_tokens",
    )
    out = {k: getattr(value, k) for k in candidate_keys if getattr(value, k, None) is not None}
    return out or None


# Alternate top-level field names produced by the OpenAI Responses API
# (``o1`` / ``o3`` / ``o4`` direct calls) and Anthropic's native usage shape.
# We map them onto the Chat-Completions-style names that the rest of the
# library already understands so callers do not need to special-case providers.
_USAGE_FIELD_ALIASES: dict[str, str] = {
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
    "input_tokens_details": "prompt_tokens_details",
    "output_tokens_details": "completion_tokens_details",
}


def _apply_usage_aliases(usage: dict[str, Any]) -> dict[str, Any]:
    """Copy values from alias keys into their canonical names if missing.

    Never overwrites an existing canonical value — providers that emit BOTH
    shapes (rare; some LiteLLM debug paths) keep the canonical one.

    Anthropic's ``cache_read_input_tokens`` is mapped into
    ``prompt_tokens_details.cached_tokens`` because it has the same
    semantics (a count of cached prompt tokens billed at the cache-read
    rate). ``cache_creation_input_tokens`` is intentionally left as a
    top-level field — billing semantics differ from cache reads, and
    silently folding it into ``cached_tokens`` would distort cost reports.
    """

    out = dict(usage)
    for alias, canonical in _USAGE_FIELD_ALIASES.items():
        if alias in out and canonical not in out:
            out[canonical] = out[alias]
    cache_read = out.get("cache_read_input_tokens")
    if cache_read is not None:
        details = out.get("prompt_tokens_details")
        if details is None:
            out["prompt_tokens_details"] = {"cached_tokens": cache_read}
        elif isinstance(details, dict) and details.get("cached_tokens") is None:
            details["cached_tokens"] = cache_read
    return out


def _extract_usage(response: Any) -> dict[str, Any]:
    """Pull a usage dict out of an LM response, if any.

    Looks at common shapes: an object with a ``usage`` attribute, a dict with a
    ``"usage"`` key, or a list/tuple whose first element carries usage. Returns
    an empty dict when nothing usable is found.

    Nested ``prompt_tokens_details`` / ``completion_tokens_details`` blocks are
    preserved (and normalized to plain dicts) so callers can read
    ``cached_tokens`` and ``reasoning_tokens`` for cost analysis. Normalization
    happens regardless of whether the top-level usage is a dict or an SDK
    object, because DSPy's ``lm.history[-1]['usage']`` is a dict whose nested
    detail values are usually still SDK objects.

    Provider aliases (Responses API ``input_tokens`` / ``output_tokens`` and
    Anthropic ``cache_read_input_tokens``) are mapped onto the canonical
    Chat-Completions-style names so downstream code does not branch by
    provider.
    """

    if response is None:
        return {}
    if isinstance(response, (list, tuple)):
        return _extract_usage(response[0]) if response else {}
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        out = _apply_usage_aliases(usage)
        for nested_key in ("prompt_tokens_details", "completion_tokens_details"):
            if nested_key in out:
                normalized = _normalize_usage_details(out[nested_key])
                if normalized is None:
                    out.pop(nested_key, None)
                else:
                    out[nested_key] = normalized
        return out
    if usage is not None and not isinstance(usage, (str, int, float, bool)):
        # Object-shaped usage. Probe canonical names AND aliases.
        out: dict[str, Any] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                    "input_tokens", "output_tokens",
                    "cache_read_input_tokens", "cache_creation_input_tokens"):
            value = getattr(usage, key, None)
            if value is not None:
                out[key] = value
        for nested_key in ("prompt_tokens_details", "completion_tokens_details",
                           "input_tokens_details", "output_tokens_details"):
            nested = getattr(usage, nested_key, None)
            normalized = _normalize_usage_details(nested)
            if normalized is not None:
                out[nested_key] = normalized
        return _apply_usage_aliases(out) if out else {}
    return {}


def _sum_usage_from_histories(
    lms: list[Any], pre_lens: list[int]
) -> dict[str, int | None]:
    """Sum token usage across new ``lm.history`` entries since ``pre_lens``.

    Used by the v7/dspy engine, where the loop runs inside ``dspy.predict.RLM``
    and per-turn usage is not surfaced through our own ``_call_lm_with_meta``
    path. Each field is ``None`` when no entry reported it (matching the
    "unknown, not zero" convention of ``_aggregate_trajectory_metrics``).
    LMs without a list-shaped ``history`` (mocks, custom callables) contribute
    nothing.
    """

    totals: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "reasoning_tokens": None,
    }
    for lm, pre_len in zip(lms, pre_lens):
        history = getattr(lm, "history", None)
        if not isinstance(history, list):
            continue
        for entry in history[pre_len:]:
            usage = _extract_usage(entry)
            if not usage:
                continue
            for key, value in (
                ("prompt_tokens", _usage_field(usage, "prompt_tokens")),
                ("completion_tokens", _usage_field(usage, "completion_tokens")),
                ("cached_tokens", _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens")),
                ("reasoning_tokens", _usage_nested_field(usage, "completion_tokens_details", "reasoning_tokens")),
            ):
                if value is None:
                    continue
                totals[key] = value if totals[key] is None else totals[key] + value
    return totals


def _usage_field(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _usage_nested_field(usage: dict[str, Any], parent: str, child: str) -> int | None:
    """Read a nested integer from ``usage[parent][child]`` safely.

    Used for OpenAI's ``prompt_tokens_details.cached_tokens`` and
    ``completion_tokens_details.reasoning_tokens``. Returns ``None`` when the
    parent is missing, not a mapping, or the child is absent / non-numeric.
    """

    nested = usage.get(parent)
    if not isinstance(nested, dict):
        return None
    return _usage_field(nested, child)
