"""Prompt builders for the RLM driver."""

from __future__ import annotations

import inspect
from typing import Mapping, Any


# The sub-LM helpers, as the model is told about them when one is configured.
_PREDICT_AVAILABLE = """`await predict(signature, instructions=None, pydantic_schemas=None, **kwargs)` calls the configured sub-LM.
`predict_sync(signature, instructions=None, pydantic_schemas=None, **kwargs)` is the synchronous form.
Both return a Prediction object; read outputs by field name, for example
`predict_sync("text -> label", text=text).label`.
Use instructions for task-specific guidance, pydantic_schemas for typed outputs, and dspy.Image input fields for images.
"""
# A live LM object (FabricLM(...), a dspy.LM, any callable) cannot cross into
# the worker, so no sub-LM is configured unless ``sub_lm=`` or a string or dict
# ``lm`` names one. Advertising the helpers anyway cost a turn in every
# PDF-skill run traced in Fabric: the model called them first and got an error.
_PREDICT_UNAVAILABLE = (
    "`predict` and `predict_sync` are not available in this run: no sub-LM is configured. "
    "Where a skill suggests them, do that step yourself: print the text you need and read it, or use Python.\n"
)

# ask_each is opt-in (RLM(ask_each=...)); the model is told about it only then.
_ASK_EACH_AVAILABLE = """`ask_each(items, question, output, columns=None, concurrency=8, retries=2, batch_size=1, max_seconds=None)` asks one question about EVERY item with an LM, in parallel on the host, and validates each answer.
Use it for a judgement per item over many items; do not write your own loop or batches around an LM. `items` is a list, a pandas Series or a DataFrame (`columns=` picks the columns each item shows).
`output` maps field names to `str`, `int`, `float`, `bool`, or a list of allowed strings; add an "unsure" choice where a forced answer would be a guess,
e.g. `ask_each(df, "Classify the complaint.", {"theme": ["Brakes", "Steering", "Other"], "safety_critical": bool}, columns=["summary"])`.
It returns a list aligned with `items` (None where an item failed or the time limit ran out) with `.errors`, `.stats` and `.to_frame(df)`. Check `.stats["failed"]` and `.stats["unfinished"]` before aggregating, and say in the answer how many items were left out.
Good uses: classify or extract from free text once the analysis shows where it matters; match records across sources that share no key (build candidate pairs in Python first, ask about each pair, then keep at most one match per record); check each claim in a draft against the evidence it came from (claim and evidence in one item; recompute numbers in code rather than asking); map the columns of messy files to a known schema; find the measure a question needs in a semantic model with many measures: call `model.find_measures("<the question>")`, which screens EVERY measure (do not narrow by keyword first) and returns the best candidates with their DAX, then read those and choose.
When a number in the answer counts items by what their free text means (a theme, cause or complaint type), keyword or regex matches miss paraphrases and redacted words and give only a floor: label every relevant item with ask_each and count those labels, using keywords only to find examples.
Keep items short: filter first, then ask. For many short items (under ~2,000 characters) pass batch_size=10 to 20; keep batch_size=1 for long items or subtle judgements.
"""
# Only when a document is among the inputs: how to find the pages a task depends on.
_ASK_EACH_DOCUMENTS = """Finding the pages a task depends on in a long document (roughly 30+ pages): screen its pages with ask_each before relying on keyword search. Search only finds rules worded the way you guess, and a missed exception, override or amendment silently breaks the answer.
Get the pages with `pages = <document input>.pages()`: one item per PDF page, or for text and markdown one per page marker, else a chunk of about 2,000 characters at a heading. Each item is a `str` with `.label` ("page 57", or "chunk 12 · Article 14") to cite and `.number` (the page number, or None for a chunk); pass `pages` straight to ask_each.
Do not ask one broad `relevant` field (nearly every page of a long document looks relevant to a broad question). Ask one narrow `bool` field per rule, one idea per field (never "X or Y"), named for what the page would do. Include:
a field for each input column or attribute that could change the result ("<result>_depends_on_<column>", e.g. shipping_fee_depends_on_region, premium_depends_on_age), one for each rule the task names, one for amendments or updates to earlier terms, and a catch-all "other_exception_or_adjustment_to_<result>",
e.g. `{"sets_refund_window": bool, "refund_depends_on_plan_type": bool, "refund_depends_on_usage": bool, "charges_cancellation_fee": bool, "amends_earlier_terms": bool, "other_exception_or_adjustment_to_refund": bool}`.
Describe each rule by its effect, since the document may use different words from the task.
Before relying on a clause, check whether other text narrows or overrides it: "notwithstanding", "provided, however", "except" or "excluding" (an exception can itself have an exception), a later amendment, or the definition of a term it uses; search `pages` for the clause's number and its defined terms.
"""
# Only with AskEach(sub_runs=True) and more than one document input.
_RUN_EACH_DOCUMENTS = """Several documents: when each document needs its own full review (many fields to work out per document), give each one its own child run with `run_each(items, task, outputs, context=None)`, e.g. `rows = run_each([docs[k] for k in keys], "<what to work out for this document, in full>", {"field": str, "source": str})`. Each child gets the document as `item` (plus any `context` dict you pass), the same tools and guidance, and its own turn budget, and returns its submitted fields (None where it did not finish; see `.errors`). Write one complete per-document task, then check and combine the rows.
When instead one data file is checked against the rules in several documents (e.g. many records, each governed by one of the documents), keep it in this run: screen all the documents' pages together and apply the rules in one calculation; splitting it gives each child a separate chance to miss a rule.
"""
# How to read the flagged pages: whole pages (default) or, with a text model, verified quotes.
_ASK_EACH_READ_PAGES = """Then read the full text of EVERY page flagged for any field (not a subset you pick), and confirm each rule in the text before relying on it.
Each turn shows at most {output_limit} characters of output and cuts anything longer, so print only as many pages as fit in that (add up `len(page)`), and keep a list of the pages still to read until it is empty.
"""
_ASK_EACH_READ_QUOTES = """Then, instead of printing whole pages, have the text model copy the rule text out of EVERY page flagged for any field (not a subset you pick) in one call:
`quotes = ask_each([pages[i] for i in to_read], "Copy word for word every sentence or table row on this page that sets, changes, limits or makes an exception to a rule about: <the rules you screened for>, plus any definition or cross-reference on this page that those sentences depend on. Separate passages with ' || '. Empty if none.", {"quote": str}, model="text")`.
Check each passage is really on its page (compare with whitespace collapsed, e.g. `" ".join(p.split()) in " ".join(page.split())`), then print the quotes with each page's `.label` and read them. An empty quote means the page has no rule: skip it (pages added only for their rank usually have none). Print a full page only when a passage fails that check, or a quote depends on text it does not include (a definition, another clause, a table); for another clause, search `pages` for its number or defined term and print just that passage. Confirm each rule before relying on it.
Each turn shows at most {output_limit} characters of output and cuts anything longer, so print only as much as fits.
"""
_ASK_EACH_DECISION_MODEL = """In this run ask_each is backed by a decision model: it answers ONLY a list of choices or `bool` (no str/int/float fields; bucket numbers into choice ranges instead).
Each choice field also returns `<field>_confidence` and each bool returns `<field>_p` (probability of true). Items are answered one per fast call, so `batch_size` has no effect; use `concurrency=32` or more.
Ask one narrow question per field, and use the confidence to set aside uncertain items (e.g. confidence < 0.6) for a closer look instead of trusting every answer.
When screening pages, the page a rule is on usually scores highest for that field even when its probability is below 0.5, so read every flagged page plus each field's top 3 by `<field>_p`:
`to_read = sorted({i for f in fields for i in range(len(pages)) if screen[i] and screen[i][f]} | {i for f in fields for i in sorted(range(len(pages)), key=lambda i: -(screen[i] or {}).get(f + "_p", 0))[:3]})`.
"""
_ASK_EACH_TEXT_MODEL = """This run also has a text model for ask_each: pass `model="text"` to use it for `str`, `int` or `float` fields (the default decision model rejects them), e.g. `ask_each(items, question, {"quote": str}, model="text")`. Keep choice and bool fields on the default model; use `model="text"` for them only to re-check items the default model was unsure about.
"""


def ask_each_section(*, decision_model: bool = False, documents: bool = False, output_limit: int = 5000,
                     text_model: bool = False, several_documents: bool = False) -> str:
    """The prompt text for ask_each: the tool, the document guidance when a document is an input, the decision-model notes.

    ``output_limit`` is the per-turn stdout budget the run actually has, so the
    advice on how many pages to print at once follows the configured limit.
    ``text_model`` means ``AskEach(text_lm=...)`` is set: flagged pages are read
    as verified quotes pulled by that model instead of whole pages.
    ``several_documents`` means ``AskEach(sub_runs=True)`` and more than one
    document input: the run is told to give each document its own child run.
    """
    reading = _ASK_EACH_READ_QUOTES if text_model else _ASK_EACH_READ_PAGES
    documents_text = (_ASK_EACH_DOCUMENTS + reading).replace("{output_limit}", f"{output_limit:,}")
    if several_documents:
        documents_text = _RUN_EACH_DOCUMENTS + documents_text
    if not text_model:
        return (_ASK_EACH_AVAILABLE
                + (documents_text if documents else "")
                + (_ASK_EACH_DECISION_MODEL if decision_model else ""))
    # With a text model the reading step is quotes, so the decision-model notes
    # (which say which pages to read) come before it and must not say "read".
    decision = _ASK_EACH_DECISION_MODEL.replace("so read every flagged page plus", "so the pages to read are every flagged page plus")
    return (_ASK_EACH_AVAILABLE
            + (decision if decision_model else "")
            + _ASK_EACH_TEXT_MODEL
            + (documents_text if documents else ""))

SYSTEM_PROMPT_TEMPLATE = """You are an RLM (Recursive Language Model) running in a Python REPL.

You solve the task by writing Python code. Each block you write is executed in
a persistent namespace, then stdout is returned to you. Variables persist across
turns. Build your answer incrementally.

## Sandbox API

`File(path)` wraps a file path.
{predict_section}`SUBMIT(**fields)` finishes the task. You MUST call SUBMIT once ready. SUBMIT is already defined in your namespace; never import it.
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
- One code block per turn. The runtime executes it and returns the real output; output you write yourself is not evidence and is never executed.
- Every number you submit must come from code that ran: pass computed values to SUBMIT and print them first. A literal you typed that no output showed is rejected.
- When repeated failures leave no computed result, call ABSTAIN('what is missing') rather than inventing values. An abstention is recorded as a failed run with your reason; an invented answer is a wrong one.

## Task

{task_description}
{validator_rules_section}
## Inputs available in namespace

{input_listing}
{learned_guidance_section}
## Required output fields for SUBMIT()

{output_listing}

Submit every listed field. Required fields may not be None or blank strings. Never invent values to fill a field: if the data has nothing for a list or dict field, submit it empty and confirm it when asked.

## Answering rules

- Always attempt a concrete answer, even when the prompt is ambiguous or appears to be missing context.
- Do NOT submit clarification requests, acknowledgements, or "please confirm" messages as your answer. Phrases like "Acknowledged", "Please confirm/clarify/specify/provide", "I need more information", "Could you please...", or "Before I can answer..." are NEVER valid SUBMIT payloads.
- If information appears missing, make the most reasonable assumption, state it inline, and answer based on that assumption.
- If the prompt enumerates sub-questions (Q1..Qn, numbered list, or "Part N"), produce ONE answer per sub-question in the same order. Partial answers (e.g. 3 elements when 50 sub-questions are listed) will be rejected.
- Analytical integrity, whatever the data source: never call a value increasing, decreasing, improving, or deteriorating just because one float is larger than another; use is_material_change with a materiality rule you state. When asked to rank by a concept (impact, risk, deterioration), define the metric for it, sort by that metric, and show it in the answer; name and justify any proxy. Keep multidimensional candidates as tuples (restrict_to_candidate_tuples), never independent per-dimension lists. Keep the requested grain or say why it changed. Attribute each material figure to its input; do not combine inputs with different periods, units, or metric definitions silently, and surface contradictions instead of resolving them. Submissions whose prose contradicts their numbers, or that hide the requested ranking metric, are rejected.

Begin. Write your first code block.
"""




def _type_label(expected_type: Any) -> str:
    """``float``, or ``dict[str, float]`` for a parameterized type."""
    import typing

    origin = typing.get_origin(expected_type)
    if origin is not None:
        return f"{origin.__name__}[{', '.join(_type_label(a) for a in typing.get_args(expected_type))}]"
    return getattr(expected_type, "__name__", str(expected_type))

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
    learned_guidance: str | None = None,
    sub_lm_available: bool = True,
    validator_rules: str | None = None,
    ask_each: str | None = None,
) -> str:
    inputs = inputs or {}
    task_description, outputs = _task_and_outputs(signature, inline_task, inline_outputs)
    input_listing = "\n".join(f"  {name}: {_describe_value(value)}" for name, value in inputs.items())
    output_types = inline_output_types or {}
    output_listing = "\n".join(
        f"  - {name}: {_type_label(output_types[name])}" if name in output_types else f"  - {name}"
        for name in outputs
    )
    # Learned guidance (retrieved lessons from a knowledge package) sits
    # right after the inputs it describes. Absent, the prompt is byte for
    # byte what it was before learning existed.
    guidance = (learned_guidance or "").strip()
    return SYSTEM_PROMPT_TEMPLATE.format(
        task_description=task_description or "(no task description)",
        input_listing=input_listing or "  (none)",
        output_listing=output_listing or "  (none)",
        skill_section=_format_skill_section(
            skill_index, preloaded_skills, skill_cards=skill_cards, router_active=router_active
        ),
        cross_source_section=_cross_source_section(inputs),
        predict_section=(_PREDICT_AVAILABLE if sub_lm_available else _PREDICT_UNAVAILABLE) + (ask_each or ""),
        learned_guidance_section=f"\n{guidance}\n" if guidance else "",
        # The rules the configured validators check, stated up front so the
        # first answer can follow them. Absent, the prompt is unchanged.
        validator_rules_section=f"\n{validator_rules.strip()}\n" if (validator_rules or "").strip() else "",
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
            "`activate_skill(name)` loads a SKILL and registers its available verifier when verification is enabled. "
            "Activate a skill only when its card matches your task — extra active skills add prompt cost and verifier checks."
        )
    if skill_index:
        parts.extend(["", "Available SKILLs:", skill_index])
    if skill_cards:
        parts.extend(
            [
                "",
                "Skill cards (bodies not preloaded; use `load_skill(name)` to read):",
                skill_cards,
            ]
        )
    if preloaded_skills:
        parts.extend(["", "Preloaded SKILLs:", preloaded_skills])
    return "\n".join(parts)
