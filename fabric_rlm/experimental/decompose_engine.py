"""Adapter that lets ``decompose_then_synthesize`` plug into the adaptive runner.

Universal — works on any prompt key. Task-agnostic by design.

The adaptive runner's contract is ``factory(attempt_cfg).run(inputs) -> RLMResult``.
This module provides a tiny adapter so a rung whose ``decompose_phase=True``
gets the decompose-then-synthesize building block instead of the standard
inner RLM loop.

Why a wrapper rather than a full RLM subclass: decompose-then-synthesize is
a *fundamentally different control flow* (1 LM + N parallel LMs + 1 LM, no
turn loop, no submit/reflect cycle). Forcing it into the turn loop would
muddy both. A wrapper keeps responsibilities separate and lets the rest of
the adaptive machinery (validator, bandit, attempt logging) work unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from .decompose_rung import DecomposeResult, decompose_then_synthesize


# Common keys that downstream callers tend to bind the question text to.
# We probe these in order. Falls back to the first string-valued input.
_DEFAULT_QUESTION_KEYS: tuple[str, ...] = (
    "question",
    "input",
    "prompt",
    "task",
    "query",
    "problem",
)


def _extract_question(inputs: Mapping[str, Any] | None, override: str | None) -> str:
    """Pull the natural-language prompt out of ``inputs``.

    Order of precedence:
        1. ``override`` (caller-supplied key name)
        2. Each common key in :data:`_DEFAULT_QUESTION_KEYS`
        3. First string-valued field in ``inputs``
        4. Empty string
    """
    if not inputs:
        return ""
    if override and override in inputs and isinstance(inputs[override], str):
        return inputs[override]
    for k in _DEFAULT_QUESTION_KEYS:
        v = inputs.get(k)
        if isinstance(v, str) and v.strip():
            return v
    for v in inputs.values():
        if isinstance(v, str) and v.strip():
            return v
    return ""


@dataclass
class DecomposeRLMAdapter:
    """Mimics ``RLM.run(inputs) -> RLMResult`` using ``decompose_then_synthesize``.

    Parameters
    ----------
    lm
        The high-effort LM used for the decompose (Phase A) and synthesize
        (Phase C) calls.
    sub_lm
        Optional LM for sub-problem solves (Phase B). Defaults to ``lm``.
    max_subs
        Upper bound on sub-problem count.
    output_field
        Payload key under which the synthesized answer is returned. Default
        ``"answer"`` matches the convention used by all bench validators.
    question_input_key
        If set, look up the question under this exact input key first
        before falling back to defaults.
    parallel
        Whether Phase B sub-solves run in parallel (default ``True``).
    """

    lm: Any
    sub_lm: Any | None = None
    max_subs: int = 6
    output_field: str = "answer"
    question_input_key: str | None = None
    parallel: bool = True

    def run(
        self,
        inputs: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> Any:
        # Local imports avoid circular import at module load.
        from fabric_rlm.runtime import RLMResult
        from fabric_rlm.trajectory import Trajectory

        events: list[dict[str, Any]] = []

        def _on_event(name: str, payload: dict[str, Any]) -> None:
            events.append({"event": name, **payload})

        question = _extract_question(inputs, self.question_input_key)
        t_start = time.monotonic()
        result: DecomposeResult = decompose_then_synthesize(
            question=question,
            lm=self.lm,
            sub_lm=self.sub_lm,
            max_subs=self.max_subs,
            min_subs=2,
            parallel=self.parallel,
            on_event=_on_event,
        )
        elapsed = time.monotonic() - t_start

        traj = Trajectory(
            metadata={
                "decompose": {
                    "sub_problems": list(result.sub_problems),
                    "sub_answers": list(result.sub_answers),
                    "rung_failure": result.rung_failure,
                    "llm_calls": result.llm_calls,
                    "error": result.error,
                    "events": events,
                    "question_chars": len(question),
                }
            }
        )

        if result.rung_failure or not result.final_answer:
            return RLMResult(
                submitted=False,
                payload=None,
                trajectory=traj,
                final_state={},
                failure_reason=result.error or "decompose_rung_failure",
                total_lm_seconds=elapsed,
                total_worker_seconds=elapsed,
            )

        payload = {self.output_field: result.final_answer}
        return RLMResult(
            submitted=True,
            payload=payload,
            trajectory=traj,
            final_state=dict(payload),
            failure_reason=None,
            total_lm_seconds=elapsed,
            total_worker_seconds=elapsed,
        )


__all__ = ["DecomposeRLMAdapter", "_extract_question"]
