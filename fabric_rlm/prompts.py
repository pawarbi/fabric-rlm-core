"""Prompt builders for the RLM driver."""

from __future__ import annotations

import inspect
from typing import Mapping, Any


SYSTEM_PROMPT_TEMPLATE = """You are an RLM (Recursive Language Model) running in a Python REPL.

You solve the task by writing Python code. Each block you write is executed in
a persistent namespace, then stdout is returned to you. Variables persist across
turns. Build your answer incrementally.

## Sandbox API

`File(path)` wraps a file path.
`await predict(signature, instructions=None, pydantic_schemas=None, **kwargs)` calls the configured sub-LM.
`predict_sync(signature, instructions=None, pydantic_schemas=None, **kwargs)` is the synchronous form.
Both return a Prediction object; read outputs by field name, for example
`predict_sync("text -> label", text=text).label`.
Use instructions for task-specific guidance, pydantic_schemas for typed outputs, and dspy.Image input fields for images.
`SUBMIT(**fields)` finishes the task. You MUST call SUBMIT once ready. SUBMIT is already defined in your namespace; never import it.
`is_material_change(current, baseline, absolute_tolerance=0, relative_tolerance=0, direction=None)` and `restrict_to_candidate_tuples(frame, candidates, keys=[...])` are predefined; `validate_analysis_integrity(...)` runs pre-SUBMIT analytical checks.
{skill_section}{cross_source_section}

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
- Analytical integrity, whatever the data source: never call a value increasing, decreasing, improving, or deteriorating just because one float is larger than another; use is_material_change with a materiality rule you state. When asked to rank by a concept (impact, risk, deterioration), define the metric for it, sort by that metric, and show it in the answer; name and justify any proxy. Keep multidimensional candidates as tuples (restrict_to_candidate_tuples), never independent per-dimension lists. Keep the requested grain or say why it changed. Attribute each material figure to its input; do not combine inputs with different periods, units, or metric definitions silently, and surface contradictions instead of resolving them. Submissions whose prose contradicts their numbers, or that hide the requested ranking metric, are rejected.

Begin. Write your first code block.
"""


def build_system_prompt(
    *,
    signature: Any = None,
    inline_task: str | None = None,
    inline_outputs: list[str] | None = None,
    inline_output_types: dict[str, type] | None = None,
    inputs: dict[str, Any] | None = None,
    skill_index: str | None = None,
    preloaded_skills: str | None = None,
    skill_cards: str | None = None,
    router_active: bool = False,
) -> str:
    inputs = inputs or {}
    task_description, outputs = _task_and_outputs(signature, inline_task, inline_outputs)
    input_listing = "\n".join(f"  {name}: {_describe_value(value)}" for name, value in inputs.items())
    output_types = inline_output_types or {}
    output_listing = "\n".join(
        f"  - {name}: {output_types[name].__name__}" if name in output_types else f"  - {name}"
        for name in outputs
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        task_description=task_description or "(no task description)",
        input_listing=input_listing or "  (none)",
        output_listing=output_listing or "  (none)",
        skill_section=_format_skill_section(
            skill_index, preloaded_skills, skill_cards=skill_cards, router_active=router_active
        ),
        cross_source_section=_cross_source_section(inputs),
    )


def is_evidence_source(value: Any) -> bool:
    """True for an input that carries evidence the analysis will reason over.

    Decided by the ``__rlm_evidence_source__`` class marker rather than by a
    list of classes, so a new source type opts in with one attribute and
    the harness never has to learn its name. ``File``, ``LakehouseSource``
    and ``SemanticModel`` carry the marker; an output sink such as
    ``FileDestination`` does not.
    """
    return bool(getattr(type(value), "__rlm_evidence_source__", False))


def evidence_leaves(inputs: Any, prefix: str = "") -> list[str]:
    """Dotted paths of every evidence source inside ``inputs``, at any depth.

    ``{"customer": {"arr": SemanticModel(...), "usage": LakehouseSource(...)}}``
    yields ``customer.arr`` and ``customer.usage``; a list yields
    ``sources[0]``, ``sources[1]``. Counting leaves rather than top-level
    keys is what makes a nested bundle of sources count as several.
    """
    leaves: list[str] = []
    if is_evidence_source(inputs):
        return [prefix or "input"]
    if isinstance(inputs, Mapping):
        for key, value in inputs.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            leaves.extend(evidence_leaves(value, path))
    elif isinstance(inputs, (list, tuple)):
        for index, value in enumerate(inputs):
            leaves.extend(evidence_leaves(value, f"{prefix}[{index}]"))
    return leaves


def _evidence_input_names(inputs: dict[str, Any]) -> list[str]:
    return evidence_leaves(inputs)


def _cross_source_section(inputs: dict[str, Any]) -> str:
    """A compact checklist, only when a finding may draw on several inputs.

    Activation is by analytical context (two or more evidence-bearing
    inputs), not by input class: a CSV plus a PDF triggers it exactly as a
    semantic model plus a Lakehouse does. Costs nothing on single-source
    tasks.
    """
    names = _evidence_input_names(inputs)
    if len(names) < 2:
        return ""
    listed = ", ".join(names)
    return (
        "\n\n## Several evidence inputs are bound (" + listed + ")\n"
        "- Keep every material figure attributed to the input it came from.\n"
        "- Join entities across inputs on an explicit shared key; a name-based match is inferred evidence and must say so.\n"
        "- A similar metric name in two inputs is not the same metric: check population, aggregation, and time basis before comparing.\n"
        "- Compare as-of dates, data availability, and reporting periods; align to a common period or state the mismatch.\n"
        "- Reconcile unit, currency, and scale explicitly before comparing numbers.\n"
        "- If the inputs disagree, report the disagreement; do not force one story."
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
    # Inputs that know how to introduce themselves get to. Used by
    # SemanticModel, where naming the handle's methods is the difference
    # between the model querying it and not knowing it can.
    describe = getattr(value, "__rlm_describe__", None)
    if callable(describe):
        try:
            return str(describe())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        kind = type(value).__name__
        described_items = []
        for index, item in enumerate(value[:10]):
            item_describe = getattr(item, "__rlm_describe__", None)
            if not callable(item_describe):
                continue
            try:
                described_items.append(f"    [{index}] {item_describe()}")
            except Exception:
                continue
        suffix = "\n" + "\n".join(described_items) if described_items else ""
        return f"{kind}[{len(value)}]{suffix}"
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
