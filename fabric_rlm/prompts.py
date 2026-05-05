"""Prompt builders for the RLM driver."""

from __future__ import annotations

import inspect
from typing import Any


SYSTEM_PROMPT_TEMPLATE = """You are an RLM (Recursive Language Model) running in a Python REPL.

You solve the task by writing Python code. Each block you write is executed in
a persistent namespace, then stdout is returned to you. Variables persist across
turns. Build your answer incrementally.

## Sandbox API

`File(path)` wraps a file path.
`await predict(signature, instructions=None, pydantic_schemas=None, **kwargs)` calls the configured sub-LM.
Use instructions for task-specific guidance, pydantic_schemas for typed outputs, and dspy.Image input fields for images.
`SUBMIT(**fields)` finishes the task. You MUST call SUBMIT once ready.
{skill_section}

## Code style - critical

- Keep each turn under 40 lines of code.
- Do not define large helper libraries or broad validators.
- Inline simple operations. One turn = one focused action.
- Always close your ```python fence.
- Avoid triple-backtick characters in string literals.
- Never call `exit()`, `quit()`, `sys.exit()`, or `os._exit()`.
- Use print() to inspect intermediate state.
- Do not catch broad exceptions unless the task requires explicit recovery.

## State management

- Variables persist across turns. Reuse them.
- The driver shows namespace keys after each turn.
- Heavy objects should stay on disk. Keep paths and metadata in state.

## Recovery

- If a turn fails or returns wrong data, write a recovery turn.
- Diagnose briefly with code/prints, then change approach.

## Task

{task_description}

## Inputs available in namespace

{input_listing}

## Required output fields for SUBMIT()

{output_listing}

Submit every listed field. Required fields may not be None or blank strings. Fields named output, answer, result, or report may not be empty containers.

## Answering rules

- Always attempt a concrete answer, even when the prompt is ambiguous or appears to be missing context.
- Do NOT submit clarification requests, acknowledgements, or "please confirm" messages as your answer. Phrases like "Acknowledged", "Please confirm/clarify/specify/provide", "I need more information", "Could you please...", or "Before I can answer..." are NEVER valid SUBMIT payloads.
- If information appears missing, make the most reasonable assumption, state it inline, and answer based on that assumption.
- If the prompt enumerates sub-questions (Q1..Qn, numbered list, or "Part N"), produce ONE answer per sub-question in the same order. Partial answers (e.g. 3 elements when 50 sub-questions are listed) will be rejected.

Begin. Write your first code block.
"""


def build_system_prompt(
    *,
    signature: Any = None,
    inline_task: str | None = None,
    inline_outputs: list[str] | None = None,
    inputs: dict[str, Any] | None = None,
    skill_index: str | None = None,
    preloaded_skills: str | None = None,
    skill_cards: str | None = None,
    router_active: bool = False,
) -> str:
    inputs = inputs or {}
    task_description, outputs = _task_and_outputs(signature, inline_task, inline_outputs)
    input_listing = "\n".join(f"  {name}: {_describe_value(value)}" for name, value in inputs.items())
    output_listing = "\n".join(f"  - {name}" for name in outputs)
    return SYSTEM_PROMPT_TEMPLATE.format(
        task_description=task_description or "(no task description)",
        input_listing=input_listing or "  (none)",
        output_listing=output_listing or "  (none)",
        skill_section=_format_skill_section(
            skill_index, preloaded_skills, skill_cards=skill_cards, router_active=router_active
        ),
    )


def build_initial_user_message(inputs: dict[str, Any]) -> str:
    input_names = ", ".join(inputs) if inputs else "no named inputs"
    return (
        f"The inputs are already bound in your namespace ({input_names}). "
        "Write one concise Python code block for the first step."
    )


def _parse_string_signature_outputs(sig: str) -> list[str]:
    """Extract declared output field names from a ``"a, b -> c, d"`` style string.

    Mirrors dspy's loose convention. Strips type hints (``"answer: int"`` ->
    ``"answer"``). Returns ``[]`` if the string lacks a ``"->"`` arrow.
    """
    if "->" not in sig:
        return []
    _, _, rhs = sig.partition("->")
    out: list[str] = []
    for chunk in rhs.split(","):
        name = chunk.strip().split(":", 1)[0].strip()
        if name and name.isidentifier():
            out.append(name)
    return out


def _task_and_outputs(
    signature: Any,
    inline_task: str | None,
    inline_outputs: list[str] | None,
) -> tuple[str | None, list[str]]:
    if signature is None:
        return inline_task, list(inline_outputs or [])

    if isinstance(signature, str):
        # String signatures of the form ``"inputs -> outputs"`` (dspy convention).
        # The string itself becomes the task description; we parse out declared
        # output fields so SUBMIT-payload validation actually fires.
        extra = (inline_task or "").strip()
        task_description = f"{signature}\n\n{extra}" if extra else signature
        return task_description, _parse_string_signature_outputs(signature)

    base = inspect.getdoc(signature) or getattr(signature, "__name__", str(signature))
    extra = (inline_task or "").strip()
    task_description = f"{base}\n\n{extra}" if extra else base
    output_fields = getattr(signature, "output_fields", None)
    if isinstance(output_fields, dict):
        return task_description, list(output_fields.keys())
    if output_fields is not None:
        return task_description, list(output_fields)
    return task_description, []


def _describe_value(value: Any) -> str:
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, tuple):
        return f"tuple[{len(value)}]"
    if isinstance(value, dict):
        return f"dict keys={list(value.keys())[:20]}"
    if hasattr(value, "path"):
        return f"{type(value).__name__} path={getattr(value, 'path')}"
    return type(value).__name__


def _format_skill_section(
    skill_index: str | None,
    preloaded_skills: str | None,
    *,
    skill_cards: str | None = None,
    router_active: bool = False,
) -> str:
    if not skill_index and not preloaded_skills and not skill_cards:
        return ""

    parts = [
        "",
        "## SKILL playbooks",
        "",
        "`list_skills()` returns task-generic playbook names. `load_skill(name)` returns Markdown for one playbook.",
        "Use SKILLs for gotchas, verifier patterns, and pre-flight checks; do not include SKILL text in final answers.",
    ]
    if router_active:
        parts.append(
            "`activate_skill(name)` loads a SKILL **and** turns on its verifier for the rest of this run. "
            "Activate a skill only when its card matches your task — extra active skills add prompt cost and verifier checks."
        )
    if skill_index:
        parts.extend(["", "Available SKILLs:", skill_index])
    if skill_cards:
        parts.extend(
            [
                "",
                "Skill cards (not active; use `activate_skill(name)` to enable):",
                skill_cards,
            ]
        )
    if preloaded_skills:
        parts.extend(["", "Preloaded SKILLs:", preloaded_skills])
    return "\n".join(parts)

