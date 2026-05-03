"""Decompose-then-synthesize as a single callable building block.

Pure library function, no policy / no I/O. The ``decompose_then_synthesize``
function is meant to back **rung 5** of the effort ladder (see
``SPEC-decompose-rung.md``) — but it is intentionally usable on its own so
callers can experiment without touching the bandit.

Two-phase, depth-1 only:

    Phase A — decompose: one LM call → 2..max_subs sub-problems.
    Phase B — solve:    each sub-problem → one ``sub_lm`` call (parallel).
    Phase C — synthesize: one LM call merges the partial answers into the
                          final answer.

If the model degenerates (1 sub-problem, or zero), we treat that as a
*decomposition failure* and surface a structured ``DecomposeResult``
indicating the rung should be reported as a no-op to the caller's policy.
The caller can then fall back to whatever rung produced the previous
attempt's result.

The function is **task-agnostic** — the prompt mentions no template names,
no JSON schema, no integer-list shape. Sub-problems are free-form strings;
synthesis is free-form text. This is what lets the same rung work on Spark
log RCA, multi-doc QA, planning, code review, etc.

Speculative. See ``bench/adaptive/SPEC-decompose-rung.md`` for the gating
plan: a 5-trial micro-bench on a synthetic compositional task family must
show 0 → ≥2 wins on at least one task before the full-bench is wired in.
"""

from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class DecomposeResult:
    """Outcome of a single decompose-then-synthesize call.

    Attributes
    ----------
    final_answer
        The synthesized answer string. Empty string when ``rung_failure`` is
        ``True`` and no upstream fallback was available.
    sub_problems
        Plain-text sub-problem strings as parsed from Phase A.
    sub_answers
        Parallel-position-matched answers from Phase B.
    rung_failure
        ``True`` when decomposition produced fewer than 2 sub-problems
        (degenerate). Caller should treat the rung as no-op and report
        the *previous* rung's result to the bandit, not this one.
    llm_calls
        Total number of LM invocations. Always ``1 + N + 1`` on success
        where ``N == len(sub_problems)``. ``0`` on a hard error.
    error
        First exception caught (if any) — for diagnostic logging only;
        the function never re-raises.
    """

    final_answer: str = ""
    sub_problems: list[str] = field(default_factory=list)
    sub_answers: list[str] = field(default_factory=list)
    rung_failure: bool = False
    llm_calls: int = 0
    error: str | None = None


_DECOMPOSE_PROMPT = (
    "Break the following problem into between 2 and {max_subs} INDEPENDENT "
    "sub-problems whose individual answers can be combined to answer the "
    "original. Each sub-problem must be self-contained (no references to "
    "other sub-problems). Do NOT solve them — just enumerate.\n\n"
    "Format your response as a numbered list, one sub-problem per line:\n"
    "1. <sub-problem 1>\n2. <sub-problem 2>\n...\n\n"
    "Original problem:\n{question}\n\nSub-problems:"
)

_SYNTH_PROMPT = (
    "Below is the original problem followed by the partial answers to a set "
    "of sub-problems that someone broke it into. Combine the partial answers "
    "into the final answer to the ORIGINAL problem.\n\n"
    "Original problem:\n{question}\n\n"
    "Sub-answers:\n{sub_answers_block}\n\n"
    "Final answer:"
)


def _invoke_lm(lm: Any, prompt: str) -> str:
    """Call ``lm`` accommodating several common signatures; return a string.

    Mirrors the duck-typing in :mod:`task_classifier` so any LM the rest of
    the project accepts works here too.
    """

    result: Any
    try:
        result = lm(prompt)
    except TypeError:
        try:
            result = lm(messages=[{"role": "user", "content": prompt}])
        except TypeError:
            for attr in ("complete", "generate", "predict"):
                fn = getattr(lm, attr, None)
                if callable(fn):
                    result = fn(prompt)
                    break
            else:
                raise TypeError(f"Unsupported LM type: {type(lm).__name__}")

    if isinstance(result, list):
        result = result[0] if result else ""
    if result is None:
        return ""
    return str(result)


_NUMBERED_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*\S)\s*$")


def _parse_sub_problems(raw: str, max_subs: int) -> list[str]:
    """Extract sub-problems from a numbered/bulleted-list response.

    Tolerant of leading prose, blank lines, and bullets (``-``, ``*``,
    ``1.``, ``1)``). Returns at most ``max_subs`` items.
    """

    out: list[str] = []
    for line in raw.splitlines():
        m = _NUMBERED_RE.match(line)
        if m:
            text = m.group(1).strip()
            if text:
                out.append(text)
        if len(out) >= max_subs:
            break

    if not out:
        # Plain newline-separated fallback (model ignored numbering). Only
        # engage when there are multiple non-trivial lines — otherwise a
        # single sentence like "I refuse" would parse as one sub-problem.
        candidates = [
            line.strip().lstrip("-*•").strip()
            for line in raw.splitlines()
            if line.strip() and len(line.strip()) > 4
        ]
        if len(candidates) >= 2:
            out = candidates[:max_subs]

    return out


def decompose_then_synthesize(
    question: str,
    lm: Any,
    sub_lm: Any | None = None,
    *,
    max_subs: int = 4,
    min_subs: int = 2,
    parallel: bool = True,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> DecomposeResult:
    """Run depth-1 decompose-then-synthesize on ``question``.

    Parameters
    ----------
    question
        The original task prompt.
    lm
        LM used for the decompose (Phase A) and synthesize (Phase C) calls
        — typically the high-effort/expensive model.
    sub_lm
        LM used for each sub-problem solve (Phase B). Defaults to ``lm``.
    max_subs, min_subs
        Bounds on sub-problem count. Below ``min_subs`` we report
        ``rung_failure=True``.
    parallel
        When ``True`` (default), Phase B uses a thread pool. Set to
        ``False`` for deterministic ordering in tests.
    on_event
        Optional callback ``(event_name, payload)`` invoked at phase
        boundaries — useful for trace capture without coupling this
        module to a logger.

    Returns
    -------
    :class:`DecomposeResult`
        Total function — never raises. Errors surface in ``.error`` and
        ``.rung_failure``.
    """

    if sub_lm is None:
        sub_lm = lm

    result = DecomposeResult()
    if not question or not str(question).strip() or lm is None:
        result.rung_failure = True
        result.error = "empty question or no LM"
        return result

    if max_subs < min_subs:
        result.rung_failure = True
        result.error = f"max_subs ({max_subs}) < min_subs ({min_subs})"
        return result

    if on_event:
        on_event("decompose_begin", {"max_subs": max_subs})

    try:
        raw = _invoke_lm(lm, _DECOMPOSE_PROMPT.format(max_subs=max_subs, question=question))
        result.llm_calls += 1
    except Exception as exc:
        result.rung_failure = True
        result.error = f"decompose call failed: {exc}"
        return result

    subs = _parse_sub_problems(raw, max_subs=max_subs)
    result.sub_problems = subs

    if on_event:
        on_event("decompose_end", {"sub_problems": list(subs)})

    if len(subs) < min_subs:
        result.rung_failure = True
        result.error = f"degenerate decomposition: {len(subs)} sub-problems"
        return result

    def _solve_one(sub: str) -> str:
        try:
            return _invoke_lm(sub_lm, sub)
        except Exception as exc:
            return f"<sub-solve error: {exc}>"

    if parallel and len(subs) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(subs)) as pool:
            answers = list(pool.map(_solve_one, subs))
    else:
        answers = [_solve_one(s) for s in subs]

    result.sub_answers = answers
    result.llm_calls += len(subs)

    if on_event:
        on_event("solve_end", {"sub_answers": list(answers)})

    sub_block = "\n".join(f"{i+1}. Q: {q}\n   A: {a}" for i, (q, a) in enumerate(zip(subs, answers)))
    try:
        synth = _invoke_lm(lm, _SYNTH_PROMPT.format(question=question, sub_answers_block=sub_block))
        result.llm_calls += 1
    except Exception as exc:
        result.rung_failure = True
        result.error = f"synthesize call failed: {exc}"
        return result

    result.final_answer = synth.strip()

    if on_event:
        on_event("synthesize_end", {"final_answer": result.final_answer})

    return result


# ----------------------------------------------------------------------------
# Cost vector — extends _EFFORT_RUNG_COST by one rung
# ----------------------------------------------------------------------------


def extended_effort_rung_cost(base: dict[int, float] | None = None) -> dict[int, float]:
    """Return an effort-cost dict extended with rung 5 ≈ 2× rung 4.

    Default ``base`` is :data:`fabric_rlm.experimental.bandit_policy._EFFORT_RUNG_COST`.
    Caller passes the result to ``BanditPolicy(rung_cost=...)`` to keep the
    bandit's tie-break math calibrated.
    """

    if base is None:
        from .bandit_policy import _EFFORT_RUNG_COST  # local import avoids cycle
        base = _EFFORT_RUNG_COST
    out = dict(base)
    top = max(out)
    out[top + 1] = out[top] * 2.0
    return out


# Default extended ladder for callers that want rung 5 enabled
EFFORT_LADDER_WITH_DECOMPOSE: tuple[str, ...] = (
    "minimal",
    "low",
    "medium",
    "high",
    "high",     # rung 4: high + parallel-N
    "high",     # rung 5: high + decompose (parallel handled by rung implementation)
)


__all__ = [
    "DecomposeResult",
    "decompose_then_synthesize",
    "extended_effort_rung_cost",
    "EFFORT_LADDER_WITH_DECOMPOSE",
]
