"""Prompt builders for the RLM driver."""

from __future__ import annotations

import inspect
import re
from typing import Any, Mapping, Sequence


REFLECTION_HISTORY_HEADER = "PREVIOUS VERIFIER REJECTIONS"
_MAX_REFLECTION_HISTORY_ENTRIES = 5

SYSTEM_PROMPT_TEMPLATE = """You are an RLM (Recursive Language Model) running in a Python REPL.

You solve the task by writing Python code. Each block you write is executed in
a persistent namespace, then stdout is returned to you. Variables persist across
turns. Build your answer incrementally.

## Sandbox API

`File(path)` wraps a file path.
`await predict(signature, instructions=None, pydantic_schemas=None, **kwargs)` calls the configured sub-LM.
Use instructions for task-specific guidance, pydantic_schemas for typed outputs, and dspy.Image input fields for images.
`SUBMIT(**fields)` finishes the task. You MUST call SUBMIT once ready.

After you call SUBMIT, you may receive a reflection turn asking you to attack your own answer. Treat it as required: verify invariants, then either re-SUBMIT the corrected payload or print REFLECTION_OK.
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


def build_reflection_prompt(
    submitted_payload: Any,
    original_question: str | None = None,
    verifier_repair_history: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Build the reflection-turn user message that asks the LM to attack its own SUBMIT.

    When ``verifier_repair_history`` is non-empty, prepends a section listing
    each prior runtime-verifier rejection so the model's self-check is aware
    of failures the runtime previously caught. Capped at the most recent
    ``_MAX_REFLECTION_HISTORY_ENTRIES`` entries to avoid prompt bloat.
    """
    payload_text = repr(submitted_payload)
    if len(payload_text) > 4000:
        payload_text = payload_text[:3997] + "..."
    question_block = (
        f"Original task:\n{original_question}\n\n" if original_question else ""
    )
    history_block = _format_verifier_history(verifier_repair_history)
    return (
        f"{history_block}"
        "You are about to submit the following answer:\n"
        f"<payload>\n{payload_text}\n</payload>\n\n"
        f"{question_block}"
        "Before this is finalized, ATTACK your own answer:\n"
        "1. List the invariants the answer must satisfy (ranges, signs, cross-field consistency, format).\n"
        "2. Write a short Python snippet that asserts each invariant against the submitted values. "
        "If any assertion fails, raise.\n"
        "3. If you find ANY issue, write corrected code that ends with a new SUBMIT(...) call with the fixed payload.\n"
        "4. If the answer survives all checks, print \"REFLECTION_OK: <one-line justification>\" "
        "and do NOT call SUBMIT again.\n"
        "\nThis is your one reflection opportunity for this submission - no further reflection will run."
    )


def _format_verifier_history(
    history: Sequence[Mapping[str, Any]] | None,
) -> str:
    if not history:
        return ""
    recent = list(history)[-_MAX_REFLECTION_HISTORY_ENTRIES:]
    lines = [
        f"{REFLECTION_HISTORY_HEADER} (your current SUBMIT must not reintroduce these):"
    ]
    for idx, entry in enumerate(recent, start=1):
        skill = entry.get("skill", "?")
        assertion = entry.get("assertion", "") or ""
        rejected = entry.get("rejected_payload")
        summary = _summarize_rejected_payload(rejected, assertion)
        lines.append(f"{idx}. Skill `{skill}` rejected `{summary}` because: {assertion}")
    lines.append("")
    lines.append(
        "Confirm none of the above invariants is violated in your current SUBMIT. "
        "If any past rejection looks structurally similar to your current answer, raise an error to retry."
    )
    return "\n".join(lines) + "\n\n"


_FIELD_TOKEN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


def _summarize_rejected_payload(
    rejected_payload: Any, assertion_message: str
) -> str:
    """Show only the offending field from the rejected payload when we can parse it.

    Looks for the first identifier in the assertion message that is also a
    key in the payload (e.g. ``"Q5 must equal..."`` -> show ``Q5=<value>``).
    Falls back to a 200-char truncation of ``repr(payload)`` if parsing fails.
    """
    if isinstance(rejected_payload, Mapping) and assertion_message:
        for token in _FIELD_TOKEN_RE.findall(assertion_message):
            if token in rejected_payload:
                value_repr = repr(rejected_payload[token])
                if len(value_repr) > 80:
                    value_repr = value_repr[:77] + "..."
                return f"{token}={value_repr}"
    text = repr(rejected_payload)
    if len(text) > 200:
        text = text[:197] + "..."
    return text


def build_initial_user_message(inputs: dict[str, Any]) -> str:
    input_names = ", ".join(inputs) if inputs else "no named inputs"
    return (
        f"The inputs are already bound in your namespace ({input_names}). "
        "Write one concise Python code block for the first step."
    )


def _task_and_outputs(
    signature: Any,
    inline_task: str | None,
    inline_outputs: list[str] | None,
) -> tuple[str | None, list[str]]:
    if signature is None:
        return inline_task, list(inline_outputs or [])

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

