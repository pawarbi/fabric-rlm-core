"""Core Recursive Language Model driver."""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .interpreter import ExecResult, Interpreter, WorkerTimeout
from .lm import resolve_lm
from .prompts import (
    _task_and_outputs,
    build_initial_user_message,
    build_reflection_prompt,
    build_system_prompt,
)
from .skill_loader import Skill, SkillLoader, compose_skills
from .skill_router import RouteDecision, SkillRouter
from .trajectory import Trajectory, TurnRecord


logger = logging.getLogger(__name__)


CORE_FINAL_OUTPUT_FIELDS = frozenset({"output", "answer", "result", "report"})

_ACTIVATE_MARKER = "[FABRIC_RLM_ACTIVATE]"


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


STDOUT_FEEDBACK_LIMIT = int(os.environ.get("FABRIC_RLM_STDOUT_LIMIT", "5000"))
STDERR_FEEDBACK_LIMIT = int(os.environ.get("FABRIC_RLM_STDERR_LIMIT", "5000"))


def _truncate_for_feedback(text: str, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    extra = len(text) - limit
    return text[:limit] + f"\n... (truncated {extra} more chars)"


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
    reflection_used: bool = False
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_lm_seconds: float | None = None
    total_worker_seconds: float | None = None

    def __getattr__(self, name: str) -> Any:
        if self.payload and name in self.payload:
            return self.payload[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "payload": self.payload,
            "final_state": self.final_state,
            "failure_reason": self.failure_reason,
            "reflection_used": self.reflection_used,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_lm_seconds": self.total_lm_seconds,
            "total_worker_seconds": self.total_worker_seconds,
            "trajectory": self.trajectory.to_dict(),
        }


def _aggregate_trajectory_metrics(trajectory: Trajectory) -> dict[str, Any]:
    """Sum per-turn token + timing fields across the trajectory.

    Token aggregates are ``None`` when no turn reported usage (so we don't
    confuse "unknown" with "zero"). Timing aggregates are always summed when
    any turn was recorded.
    """

    def _sum_optional(values: list[Any]) -> Any:
        present = [v for v in values if v is not None]
        if not present:
            return None
        return sum(present)

    prompts = [t.prompt_tokens for t in trajectory.turns]
    completions = [t.completion_tokens for t in trajectory.turns]
    lm_secs = [t.lm_call_seconds for t in trajectory.turns]
    worker_secs = [t.worker_execute_seconds for t in trajectory.turns]
    return {
        "total_prompt_tokens": _sum_optional(prompts),
        "total_completion_tokens": _sum_optional(completions),
        "total_lm_seconds": _sum_optional(lm_secs),
        "total_worker_seconds": _sum_optional(worker_secs),
    }


class RLM:
    """LM-driven Python REPL loop with persistent worker state.

    Parameters
    ----------
    enable_reflection:
        When True (default), the runtime injects one reflection turn after each
        validated SUBMIT, asking the model to attack its own answer. The
        reflection turn may either (a) print ``REFLECTION_OK`` to confirm the
        original payload, (b) emit a corrected ``SUBMIT(...)`` call, or (c)
        raise — in which case the runtime falls through to the existing
        validation/repair feedback loop. At most one reflection turn ever runs
        per ``run()``; subsequent SUBMITs (e.g. from validation/repair after a
        reflection) are accepted without re-reflecting.
    """

    def __init__(
        self,
        signature: Any = None,
        *,
        lm: Any,
        sub_lm: Any | None = None,
        max_turns: int = 10,
        timeout: float = 300.0,
        verbose: bool = False,
        skills: list[str] | None = None,
        enable_skill_autoloading: bool = False,
        skill_loader: SkillLoader | None = None,
        enable_reflection: bool = True,
        enable_verifier: bool = True,
        enable_router: bool = False,
        max_active_skills: int = 2,
        router_baseline_skills: list[str] | None = None,
        router_candidate_specificities: list[str] | None = None,
        router_include_dependencies: bool = True,
        reserve_finalize_turns: int = 0,
        max_prompt_tokens: int | None = None,
        digest_after_turn: int | None = None,
        output_validator: Callable[[Mapping[str, Any]], None] | None = None,
        halve_max_iter_on_retry: bool = True,
        engine: str = "v6-custom",
    ):
        self.signature = signature
        self.outer_lm = resolve_lm(lm)
        self.sub_lm_spec = sub_lm if sub_lm is not None else (lm if isinstance(lm, (str, dict)) else None)
        self.max_turns = max_turns
        self.timeout = timeout
        self.verbose = verbose
        self.skills = list(skills or [])
        self.enable_skill_autoloading = enable_skill_autoloading
        self.skill_loader = skill_loader or SkillLoader()
        self.enable_reflection = enable_reflection
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
        self.max_prompt_tokens = max_prompt_tokens
        self.digest_after_turn = digest_after_turn
        self.output_validator = output_validator
        self.halve_max_iter_on_retry = bool(halve_max_iter_on_retry)
        if engine not in ("v6-custom", "v7-dspy"):
            raise ValueError(
                f"engine must be 'v6-custom' or 'v7-dspy', got {engine!r}"
            )
        self.engine = engine
        self._loaded_skills: list[Skill] = (
            [self.skill_loader.load(name) for name in self.skills]
            if self.enable_verifier
            else []
        )
        self._activated_skills: set[str] = set()
        self._inline_task: str | None = None
        self._inline_outputs: list[str] | None = None
        self._inline_inputs: dict[str, Any] = {}

    @classmethod
    def from_task(
        cls,
        task: str,
        inputs: dict[str, Any] | None = None,
        outputs: list[str] | None = None,
        **kwargs: Any,
    ) -> "RLM":
        # Don't pass `None` positionally — callers may supply `signature=...`
        # via kwargs (v6.5+) and we'd otherwise hit "multiple values for 'signature'".
        kwargs.setdefault("signature", None)
        instance = cls(**kwargs)
        instance._inline_task = task
        instance._inline_outputs = list(outputs or [])
        instance._inline_inputs = dict(inputs or {})
        return instance

    def __call__(self, **inputs: Any) -> RLMResult:
        return self.run(inputs or None)

    def run(self, inputs: dict[str, Any] | None = None) -> RLMResult:
        bound_inputs = dict(self._inline_inputs)
        if inputs:
            bound_inputs.update(inputs)
        required_output_fields = _required_output_fields(self.signature, self._inline_task, self._inline_outputs)

        if self.engine == "v7-dspy":
            return self._run_via_dspy(bound_inputs, required_output_fields)

        trajectory = Trajectory(
            metadata={
                "max_turns": self.max_turns,
                "skills": list(self.skills),
                "skill_autoloading": self.enable_skill_autoloading,
                "router_enabled": self.enable_router,
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
            # Include bound input values so routing matches keywords that appear
            # in the actual question, not only the generic task description.
            # Universal: applies to any signature/input shape.
            routing_text = _build_routing_text(task_text_for_routing, bound_inputs)
            router = SkillRouter.from_loader(
                self.skill_loader,
                max_active_skills=self.max_active_skills,
                baseline_skill_names=self.router_baseline_skills,
                candidate_specificities=self.router_candidate_specificities,
                include_dependencies=self.router_include_dependencies,
            )
            route_decision = router.route(
                routing_text, explicit_skills=self.skills or None
            )
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

        with Interpreter(timeout=self.timeout) as interpreter:
            if self.sub_lm_spec is not None:
                interpreter.configure_lm(self.sub_lm_spec)
            if bound_inputs:
                interpreter.set_inputs(bound_inputs)

            turn_counter = 0
            reflection_done = False
            next_turn_type = "normal"
            reached_max = False
            verifier_repair_history: list[dict[str, Any]] = []

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
                if _looks_truncated(response_text):
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was truncated before the closing code fence. "
                                "Rewrite that turn in one complete ```python block under 30 lines."
                            ),
                        }
                    )
                    # truncated turns still count toward the budget to avoid loops.
                    turn_counter += 1
                    if turn_counter >= self.max_turns:
                        reached_max = True
                    continue

                turn_counter += 1
                current_turn_type = next_turn_type
                next_turn_type = "normal"

                code = _extract_code(response_text)
                self._log(f"\n=== Turn {turn_counter}/{self.max_turns} ({current_turn_type}) ===\n{code}")
                started = time.perf_counter()
                worker_started = time.monotonic()
                try:
                    result = interpreter.execute(code)
                except WorkerTimeout as exc:
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
                            lm_call_seconds=lm_call_seconds,
                            worker_execute_seconds=worker_execute_seconds,
                        )
                    )
                    return RLMResult(
                        submitted=False,
                        payload=None,
                        trajectory=trajectory,
                        final_state=final_state,
                        failure_reason="worker_timeout",
                        reflection_used=reflection_done,
                        **_aggregate_trajectory_metrics(trajectory),
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
                            lm_call_seconds=lm_call_seconds,
                            worker_execute_seconds=worker_execute_seconds,
                        )
                    )
                    return RLMResult(
                        submitted=False,
                        payload=None,
                        trajectory=trajectory,
                        final_state=final_state,
                        failure_reason="worker_error",
                        reflection_used=reflection_done,
                        **_aggregate_trajectory_metrics(trajectory),
                    )
                worker_execute_seconds = time.monotonic() - worker_started
                duration = time.perf_counter() - started
                final_state = result.state
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
                    validate_submit_payload(result.submit_payload, required_output_fields)
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
                        lm_call_seconds=lm_call_seconds,
                        worker_execute_seconds=worker_execute_seconds,
                    )
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

                    if not (self.enable_reflection and not reflection_done):
                        return RLMResult(
                            submitted=True,
                            payload=result.submit_payload,
                            trajectory=trajectory,
                            final_state=result.state,
                            reflection_used=reflection_done,
                            **_aggregate_trajectory_metrics(trajectory),
                        )

                    # Run a single inline reflection turn.
                    reflection_done = True
                    original_payload = result.submit_payload
                    original_state = result.state
                    messages.append({"role": "assistant", "content": response_text})
                    reflection_prompt = build_reflection_prompt(
                        original_payload,
                        task_description,
                        verifier_repair_history=verifier_repair_history,
                    )
                    messages.append({"role": "user", "content": reflection_prompt})

                    (
                        reflect_response_text,
                        reflect_raw_response,
                        reflect_lm_seconds,
                    ) = _call_lm_with_meta(self.outer_lm, messages)
                    reflect_usage = _extract_usage(reflect_raw_response)
                    reflect_prompt_tokens = _usage_field(reflect_usage, "prompt_tokens")
                    reflect_completion_tokens = _usage_field(reflect_usage, "completion_tokens")
                    reflect_total_tokens = _usage_field(reflect_usage, "total_tokens")
                    turn_counter += 1
                    reflect_code = _extract_code(reflect_response_text)
                    self._log(
                        f"\n=== Turn {turn_counter}/{self.max_turns} (reflection) ===\n{reflect_code}"
                    )
                    reflect_started = time.perf_counter()
                    reflect_worker_started = time.monotonic()
                    try:
                        reflect_result = interpreter.execute(reflect_code)
                    except (WorkerTimeout, Exception) as exc:  # noqa: BLE001 - capture all worker failures uniformly
                        reflect_worker_seconds = time.monotonic() - reflect_worker_started
                        reflect_duration = time.perf_counter() - reflect_started
                        trajectory.append(
                            TurnRecord(
                                turn=turn_counter,
                                code=reflect_code,
                                stdout="",
                                stderr="",
                                error=f"{type(exc).__name__}: {exc}",
                                submitted=False,
                                state=dict(final_state),
                                response_text=reflect_response_text,
                                duration_s=reflect_duration,
                                token_usage=reflect_usage,
                                validation_errors=[],
                                turn_type="reflection",
                                prompt_tokens=reflect_prompt_tokens,
                                completion_tokens=reflect_completion_tokens,
                                total_tokens=reflect_total_tokens,
                                lm_call_seconds=reflect_lm_seconds,
                                worker_execute_seconds=reflect_worker_seconds,
                            )
                        )
                        # Fall through to repair: feed the reflection error back to the LM.
                        messages.append({"role": "assistant", "content": reflect_response_text})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Reflection turn raised {type(exc).__name__}: {exc}\n"
                                    f"The earlier SUBMIT payload preview: {_preview_payload(original_payload)}\n"
                                    "Write a recovery turn that diagnoses the issue and re-calls SUBMIT(...) "
                                    "with a corrected payload."
                                ),
                            }
                        )
                        next_turn_type = "validation_repair"
                        if turn_counter >= self.max_turns:
                            reached_max = True
                        continue

                    reflect_worker_seconds = time.monotonic() - reflect_worker_started
                    reflect_duration = time.perf_counter() - reflect_started
                    final_state = reflect_result.state
                    reflect_validation = (
                        validate_submit_payload(reflect_result.submit_payload, required_output_fields)
                        if reflect_result.submitted
                        else OutputValidationResult()
                    )
                    trajectory.append(
                        TurnRecord(
                            turn=turn_counter,
                            code=reflect_code,
                            stdout=reflect_result.stdout,
                            stderr=reflect_result.stderr,
                            error=reflect_result.error,
                            submitted=reflect_result.submitted,
                            state=reflect_result.state,
                            response_text=reflect_response_text,
                            duration_s=reflect_duration,
                            token_usage=reflect_usage,
                            validation_errors=list(reflect_validation.errors),
                            turn_type="reflection",
                            prompt_tokens=reflect_prompt_tokens,
                            completion_tokens=reflect_completion_tokens,
                            total_tokens=reflect_total_tokens,
                            lm_call_seconds=reflect_lm_seconds,
                            worker_execute_seconds=reflect_worker_seconds,
                        )
                    )

                    if reflect_result.submitted:
                        if not reflect_validation.ok:
                            # Reflection produced a structurally bad SUBMIT; defer to validation/repair
                            # without recursing into another reflection.
                            messages.append({"role": "assistant", "content": reflect_response_text})
                            messages.append(
                                {
                                    "role": "user",
                                    "content": self._format_validation_feedback(
                                        reflect_result, turn_counter, reflect_validation
                                    ),
                                }
                            )
                            next_turn_type = "validation_repair"
                            if turn_counter >= self.max_turns:
                                reached_max = True
                            continue
                        return RLMResult(
                            submitted=True,
                            payload=reflect_result.submit_payload,
                            trajectory=trajectory,
                            final_state=reflect_result.state,
                            reflection_used=True,
                            **_aggregate_trajectory_metrics(trajectory),
                        )

                    if reflect_result.error:
                        # Worker reported an error (e.g. assertion) but didn't raise to the host —
                        # treat the same as an exception: feed back and let repair run.
                        messages.append({"role": "assistant", "content": reflect_response_text})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Reflection turn errored:\n{(reflect_result.error or '')[:2000]}\n"
                                    f"Earlier SUBMIT payload preview: {_preview_payload(original_payload)}\n"
                                    "Write a recovery turn and re-call SUBMIT(...) with a corrected payload."
                                ),
                            }
                        )
                        next_turn_type = "validation_repair"
                        if turn_counter >= self.max_turns:
                            reached_max = True
                        continue

                    # Reflection survived without a new SUBMIT and without an error -> accept original.
                    return RLMResult(
                        submitted=True,
                        payload=original_payload,
                        trajectory=trajectory,
                        final_state=original_state,
                        reflection_used=True,
                        **_aggregate_trajectory_metrics(trajectory),
                    )

                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": self._format_feedback(result, turn_counter)})

            if not reached_max:
                reached_max = True

        return RLMResult(
            submitted=False,
            payload=None,
            trajectory=trajectory,
            final_state=final_state,
            failure_reason=(
                "output_validation_failed"
                if trajectory.turns and trajectory.turns[-1].validation_errors
                else "max_turns"
            ),
            reflection_used=reflection_done,
            **_aggregate_trajectory_metrics(trajectory),
        )

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
        the eventual reflection prompt. Non-AssertionError failures are logged
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

        The validator is a callable (typically ``verify_longcot_output``)
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

    def _format_feedback(self, result: ExecResult, turn: int) -> str:
        stdout_text = _truncate_for_feedback(result.stdout, STDOUT_FEEDBACK_LIMIT)
        parts = [f"REPL output from turn {turn}:\n```\n{stdout_text}\n```"]
        if not result.ok:
            parts.append(f"\nERROR:\n```\n{(result.error or '')[:2000]}\n```\nWrite a recovery turn.")
        elif result.stderr:
            stderr_text = _truncate_for_feedback(result.stderr, STDERR_FEEDBACK_LIMIT)
            parts.append(f"\nstderr:\n```\n{stderr_text}\n```")
        parts.append(f"\nState keys: {', '.join(result.state.keys()) or '(none)'}")
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

        # ---- Slice 3: bounded verifier-wrapper retry loop
        MAX_VERIFIER_RETRIES = 2
        verifier_repair_history: list[dict[str, Any]] = []
        current_inputs = dict(bound_inputs)
        max_iter = self.max_turns
        prediction = None
        elapsed_total = 0.0
        last_failure_reason: str | None = None

        for attempt in range(MAX_VERIFIER_RETRIES + 1):
            interpreter = SubprocessPythonInterpreter(timeout=self.timeout)
            t0 = time.time()
            try:
                with dspy.context(lm=outer_lm):
                    rlm = DspyRLM(
                        signature=signature,
                        sub_lm=sub_lm,
                        interpreter=interpreter,
                        max_iterations=max_iter,
                    )
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

            payload = self._extract_payload_from_prediction(prediction, signature)
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
                    if self.halve_max_iter_on_retry:
                        max_iter = max(1, (max_iter + 1) // 2)
                    continue
                last_failure_reason = "dspy.RLM produced no output payload"
                break

            # Run verifiers + output_validator using a fresh legacy interpreter.
            # Must use context manager — Interpreter requires explicit .start().
            try:
                with Interpreter(timeout=self.timeout) as verifier_interp:
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
                # Out of retries; still return the (rejected) payload but flag.
                break

            # Prepend feedback to first string input we can find.
            current_inputs = self._inputs_with_verifier_feedback(
                current_inputs, feedback_text
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

        submitted = bool(payload) and last_failure_reason is None
        return RLMResult(
            submitted=submitted,
            payload=payload if submitted else None,
            trajectory=trajectory,
            final_state={},
            failure_reason=last_failure_reason,
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
            routing_text = _build_routing_text(task_text_for_routing, bound_inputs)
            router = SkillRouter.from_loader(
                self.skill_loader,
                max_active_skills=self.max_active_skills,
                baseline_skill_names=self.router_baseline_skills,
                candidate_specificities=self.router_candidate_specificities,
                include_dependencies=self.router_include_dependencies,
            )
            route_decision = router.route(routing_text, explicit_skills=self.skills or None)
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
        first string-valued field. Falls back to a new ``_verifier_feedback``
        key (which dspy will ignore) if no string input exists."""
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
        new_inputs["_verifier_feedback"] = feedback
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

        if extra_instructions:
            base = built.instructions or ""
            new_instructions = (base + "\n\n" + extra_instructions).strip()
            built = built.with_instructions(new_instructions)
        return built


def _extract_code(text: str) -> str:
    for fence in ("```python", "```py", "```"):
        if fence in text:
            start = text.index(fence) + len(fence)
            rest = text[start:]
            if "```" in rest:
                return rest[: rest.index("```")].strip()
    return text.strip()


def validate_submit_payload(
    payload: Mapping[str, Any] | None,
    required_fields: Iterable[str],
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
    for name in fields:
        if name not in payload:
            errors.append(f"Missing required output field {name!r}.")
            continue
        value = payload[name]
        error = _validate_required_value(name, value)
        if error:
            errors.append(error)
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


def _normalize_required_fields(required_fields: Iterable[str]) -> tuple[str, ...]:
    fields: list[str] = []
    for name in required_fields:
        if not isinstance(name, str):
            raise TypeError(f"Output field names must be strings, got {type(name).__name__}.")
        if not name:
            raise ValueError("Output field names must not be empty.")
        fields.append(name)
    return tuple(fields)


def _required_output_fields(
    signature: Any,
    inline_task: str | None,
    inline_outputs: list[str] | None,
) -> tuple[str, ...]:
    _, outputs = _task_and_outputs(signature, inline_task, inline_outputs)
    return _normalize_required_fields(outputs)


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
    """

    started = time.monotonic()
    response = _call_lm(lm, messages)
    elapsed = time.monotonic() - started
    return _response_to_text(response), response, elapsed


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


def _extract_usage(response: Any) -> dict[str, Any]:
    """Pull a usage dict out of an LM response, if any.

    Looks at common shapes: an object with a ``usage`` attribute, a dict with a
    ``"usage"`` key, or a list/tuple whose first element carries usage. Returns
    an empty dict when nothing usable is found.
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
        return dict(usage)
    if usage is not None and not isinstance(usage, (str, int, float, bool)):
        return {
            key: getattr(usage, key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if getattr(usage, key, None) is not None
        }
    return {}


def _usage_field(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None

