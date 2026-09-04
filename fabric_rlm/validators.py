"""Composable output_validator helpers for fabric_rlm.RLM.

A validator is any ``Callable[[Mapping[str, Any]], None]`` that raises
``AssertionError`` to signal rejection. The verifier loop catches the
AssertionError, surfaces the message back to the model, and re-prompts.

This module provides:

  * :func:`signature_validator` — auto-derive a shape validator from a
    ``dspy.Signature`` by reusing its output-field type annotations as a
    Pydantic model. Catches type/missing-field/coercion errors.

  * Composable primitives — small functions that each enforce a single
    invariant. Combine via :func:`chain`.

Example::

    from fabric_rlm.validators import (
        signature_validator, chain, assert_keys, assert_list_len,
    )

    validator = chain(
        signature_validator(MySignature),
        assert_keys("solution"),
        assert_list_len("solution", n=490, key_in_payload="solution"),
    )
    rlm = RLM(signature=MySignature, output_validator=validator, ...)
"""
from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping

try:  # pydantic is a dspy dep
    import pydantic
    from pydantic import ValidationError, create_model
except ImportError:  # pragma: no cover
    pydantic = None
    ValidationError = Exception  # type: ignore[misc,assignment]

Validator = Callable[[Mapping[str, Any]], None]

from .analytical_integrity import (  # noqa: E402 - after the Validator alias on purpose
    check_directional_claims,
    check_ranking_disclosure,
    infer_requested_ranking,
    validate_grain,
    AnalyticalIntegrityError,
)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def chain(*validators: Validator) -> Validator:
    """Run validators in order; raise on first failure.

    None entries are skipped (so ``chain(maybe_a, maybe_b)`` works when one
    is conditionally None).
    """
    real = [v for v in validators if v is not None]

    def composed(payload: Mapping[str, Any]) -> None:
        for v in real:
            v(payload)
    return composed


# ---------------------------------------------------------------------------
# Auto-derived from dspy.Signature
# ---------------------------------------------------------------------------

def _is_dspy_output_field(field_info: Any) -> bool:
    extra = getattr(field_info, "json_schema_extra", None)
    if isinstance(extra, dict):
        return extra.get("__dspy_field_type") == "output"
    return False


def signature_validator(signature: type) -> Validator | None:
    """Build a shape validator from a ``dspy.Signature`` class.

    The returned validator constructs a Pydantic model from the signature's
    OUTPUT fields (preserving each field's annotation) and runs
    ``model_validate`` on the SUBMIT payload. Pydantic ``ValidationError``
    is converted to ``AssertionError`` so the verifier loop can surface
    the message back to the model.

    Returns ``None`` if pydantic isn't importable or the signature has no
    typed output fields (graceful no-op so callers can do
    ``chain(signature_validator(sig), ...)`` unconditionally).
    """
    if pydantic is None:
        return None

    out_fields: dict[str, tuple[Any, Any]] = {}
    model_fields = getattr(signature, "model_fields", None)
    if not isinstance(model_fields, Mapping):
        return None

    for name, fi in model_fields.items():
        if not _is_dspy_output_field(fi):
            continue
        # Skip untyped fields ("answer: <unspecified>"). dspy uses str for those
        # which is fine — Pydantic validation on str is a no-op anyway, but we
        # still include them so missing-field rejection works.
        annotation = getattr(fi, "annotation", str) or str
        out_fields[name] = (annotation, ...)  # required

    if not out_fields:
        return None

    OutputModel = create_model(
        f"{getattr(signature, '__name__', 'Output')}_OutputShape",
        **out_fields,  # type: ignore[arg-type]
    )

    def validate(payload: Mapping[str, Any]) -> None:
        try:
            OutputModel.model_validate(dict(payload))
        except ValidationError as exc:
            # Pydantic produces multiline error reports; preserve them — the
            # model needs the field names to know what to fix.
            raise AssertionError(
                f"Output schema validation failed against {OutputModel.__name__}:\n{exc}"
            ) from exc

    return validate


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def assert_keys(*required: str) -> Validator:
    """Assert that all listed keys are present (and not None) in the payload."""
    required_set = tuple(required)

    def v(payload: Mapping[str, Any]) -> None:
        missing = [k for k in required_set
                   if k not in payload or payload.get(k) is None]
        assert not missing, (
            f"SUBMIT payload is missing required keys: {missing}. "
            f"Required: {list(required_set)}."
        )
    return v


def _resolve(payload: Mapping[str, Any], key: str) -> Any:
    """Pull `key` from payload OR from a nested dict at `payload['solution']`.

    Some outputs use ``payload = {'solution': '<json str>'}`` with the actual
    fields inside the JSON. We accept both forms.
    """
    if key in payload:
        return payload[key]
    # Try parsing payload['solution'] as JSON
    sol = payload.get("solution") or payload.get("output") or payload.get("answer")
    if isinstance(sol, str):
        # Try parsing payload['solution'] as a JSON object containing the key
        try:
            import json as _json
            parsed = _json.loads(sol)
            if isinstance(parsed, Mapping) and key in parsed:
                return parsed[key]
        except (ValueError, TypeError):
            pass
    return _MISSING


_MISSING = object()


def assert_list_len(key: str, n: int, *, exact: bool = True) -> Validator:
    """Assert that ``payload[key]`` is a list of length ``n`` (exact by default).

    Resolves ``key`` from the top-level payload or from a JSON string in
    ``payload['solution']``.
    """
    def v(payload: Mapping[str, Any]) -> None:
        val = _resolve(payload, key)
        assert val is not _MISSING, (
            f"SUBMIT payload missing key '{key}' (looked at top-level and inside "
            f"payload['solution'] JSON)."
        )
        assert isinstance(val, list), (
            f"Expected '{key}' to be a list, got {type(val).__name__}."
        )
        actual = len(val)
        if exact:
            assert actual == n, (
                f"'{key}' has length {actual} but {n} was expected. "
                f"If you submitted a worked-example output instead of computing "
                f"the actual instance, re-read the puzzle and recompute."
            )
        else:
            assert actual >= n, (
                f"'{key}' has length {actual}; expected at least {n}."
            )
    return v


def assert_list_of(key: str, item_type: type) -> Validator:
    """Assert that ``payload[key]`` is a list and every item is ``item_type``."""
    def v(payload: Mapping[str, Any]) -> None:
        val = _resolve(payload, key)
        assert val is not _MISSING, f"Missing key '{key}'."
        assert isinstance(val, list), (
            f"Expected '{key}' to be a list, got {type(val).__name__}."
        )
        bad = [i for i, x in enumerate(val) if not isinstance(x, item_type)]
        assert not bad, (
            f"'{key}' must be list[{item_type.__name__}]; "
            f"non-{item_type.__name__} items at indices {bad[:5]}"
            f"{' ...' if len(bad) > 5 else ''}."
        )
    return v


def assert_in_range(key: str, lo: float, hi: float, *, inclusive: bool = True) -> Validator:
    """Assert that ``payload[key]`` is a number in ``[lo, hi]`` (or ``(lo, hi)``)."""
    def v(payload: Mapping[str, Any]) -> None:
        val = _resolve(payload, key)
        assert val is not _MISSING, f"Missing key '{key}'."
        assert isinstance(val, (int, float)) and not isinstance(val, bool), (
            f"Expected '{key}' to be a number, got {type(val).__name__}."
        )
        if inclusive:
            assert lo <= val <= hi, f"'{key}'={val} not in [{lo}, {hi}]."
        else:
            assert lo < val < hi, f"'{key}'={val} not in ({lo}, {hi})."
    return v


def assert_matches_regex(key: str, pattern: str, *, flags: int = 0) -> Validator:
    """Assert that ``payload[key]`` is a string matching ``pattern`` (re.fullmatch)."""
    rx = re.compile(pattern, flags)

    def v(payload: Mapping[str, Any]) -> None:
        val = _resolve(payload, key)
        assert val is not _MISSING, f"Missing key '{key}'."
        assert isinstance(val, str), (
            f"Expected '{key}' to be a string, got {type(val).__name__}."
        )
        assert rx.fullmatch(val), (
            f"'{key}'={val!r} does not match required pattern {pattern!r}."
        )
    return v


def assert_predicate(predicate: Callable[[Mapping[str, Any]], bool],
                     message: str = "custom predicate failed") -> Validator:
    """Wrap any boolean predicate as a validator.

    Useful for one-off semantic checks (e.g. consistency between fields)
    that don't fit the standard primitives.
    """
    def v(payload: Mapping[str, Any]) -> None:
        assert predicate(payload), message
    return v


# ---------------------------------------------------------------------------
# Universal multi-part question guards
# ---------------------------------------------------------------------------
# These primitives target a class of failure that recurs across any task
# family with enumerated sub-questions (CS-hard benchmarks, RFP responses,
# multi-section reports, code-review checklists, etc.): the model "answers"
# a 50-part question with a 3-element list and the validator passes it.

_ENUM_PATTERNS = (
    # Q1, Q2, ..., Qn (case-insensitive)
    re.compile(r"\bQ\s*(\d+)\s*[:.\)]", re.IGNORECASE),
    # 1., 2., ... at line start (markdown / numbered lists)
    re.compile(r"(?m)^\s*(\d+)[.\)]\s+\S"),
    # Question 1, Question 2, ...
    re.compile(r"\bQuestion\s+(\d+)\b", re.IGNORECASE),
    # Part 1, Part 2, ...
    re.compile(r"\bPart\s+(\d+)\b", re.IGNORECASE),
    # Step 1, Step 2, ...
    re.compile(r"\bStep\s+(\d+)\b", re.IGNORECASE),
)


def infer_subquestion_count(question_text: str) -> int:
    """Infer how many sub-questions a prompt enumerates.

    Universal heuristic — works for any task that uses Q1..Qn / 1. 2. 3. /
    Question N / Part N / Step N enumeration. Returns 0 when no enumeration
    is detected (caller should treat 0 as "unknown shape, no guard").

    Picks the *largest* contiguous run starting at 1 across all detected
    schemes, so a complete enumerated field sequence wins over a stray step
    reference elsewhere in the prompt.
    """
    if not question_text:
        return 0
    best = 0
    for rx in _ENUM_PATTERNS:
        nums = sorted({int(m) for m in rx.findall(question_text)})
        if not nums or nums[0] != 1:
            continue
        run = 1
        for prev, cur in zip(nums, nums[1:]):
            if cur == prev + 1:
                run += 1
            else:
                break
        if run > best:
            best = run
    return best


def assert_answers_all_subquestions(
    key: str,
    question_text: str,
    *,
    min_count: int | None = None,
) -> Validator | None:
    """Reject submissions that under-answer an enumerated multi-part prompt.

    Inspects ``question_text`` for enumeration (Q1..Qn, 1./2./3., Question N,
    Part N, Step N) and, when at least ``min_count`` (default 3) sub-parts
    are detected, builds a validator that rejects ``payload[key]`` when it
    is a list/tuple/dict shorter than the inferred count.

    Returns ``None`` (silently no-op) when no enumeration is detected — so
    you can chain it unconditionally:

        validator = chain(
            signature_validator(Sig),
            assert_answers_all_subquestions("answer", question_text=q),
        )

    Universal: no template names, no hard-coded counts. The actual count
    comes from the prompt itself.
    """
    threshold = 3 if min_count is None else min_count
    expected = infer_subquestion_count(question_text or "")
    if expected < threshold:
        return None

    def v(payload: Mapping[str, Any]) -> None:
        val = _resolve(payload, key)
        if val is _MISSING or val is None:
            return  # let other validators report missing-key
        # We can size lists, tuples, dicts, and parsable JSON strings.
        if isinstance(val, str):
            try:
                import json as _json
                parsed = _json.loads(val)
                if isinstance(parsed, (list, tuple, dict)):
                    val = parsed
            except (ValueError, TypeError):
                pass
        if isinstance(val, (list, tuple)):
            actual = len(val)
        elif isinstance(val, dict):
            actual = len(val)
        else:
            return  # scalar — no shape to check
        assert actual >= expected, (
            f"'{key}' contains {actual} items but the prompt enumerates "
            f"{expected} sub-questions. Submit one answer per sub-question "
            f"(in order), not a partial or summary list."
        )

    return v


_CLARIFICATION_OPENERS = (
    re.compile(r"^\s*acknowledg", re.IGNORECASE),
    re.compile(r"^\s*please\s+(confirm|clarify|provide|share|specify)", re.IGNORECASE),
    re.compile(r"^\s*(could|can|would)\s+you\s+(please\s+)?(confirm|clarify|provide|share|specify)", re.IGNORECASE),
    re.compile(r"^\s*i\s+(need|require|would\s+need|would\s+require)\s+(more|additional|further|the)\s+", re.IGNORECASE),
    re.compile(r"^\s*to\s+(answer|proceed|continue)[, ].{0,40}\bplease\b", re.IGNORECASE),
    re.compile(r"^\s*before\s+(i\s+)?(can\s+)?(answer|proceed|begin)", re.IGNORECASE),
)


def assert_not_clarification_request(key: str) -> Validator:
    """Reject submissions whose answer is actually a clarification request.

    Universal: looks at the *opening* of the answer string for canonical
    "I need more information" / "Please confirm" / "Acknowledged. ..."
    patterns. Anything matching is rejected so the verifier loop can
    re-prompt with "produce a concrete answer".

    Catches the rung-0 failure mode where the model defers the question
    instead of attempting it. Works for any task family — chatbot, code
    gen, data extraction — because clarification openers are
    domain-agnostic English.
    """
    def v(payload: Mapping[str, Any]) -> None:
        val = _resolve(payload, key)
        if val is _MISSING or val is None:
            return
        if not isinstance(val, str):
            return
        text = val.strip()
        if not text:
            return
        for rx in _CLARIFICATION_OPENERS:
            if rx.search(text[:200]):
                assert False, (
                    f"'{key}' looks like a clarification request, not an "
                    f"answer ({text[:80]!r}...). Provide a concrete answer; "
                    f"if information is missing, make a reasonable assumption "
                    f"and state it inline."
                )
    return v


# ---------------------------------------------------------------------------
# Analytical integrity (source-agnostic)
# ---------------------------------------------------------------------------


def _payload_texts(payload: Mapping[str, Any], key: str | None) -> list[str]:
    """Collect the string content a reader would see, at ``key`` or everywhere."""
    values: list[Any]
    if key is not None:
        val = _resolve(payload, key)
        values = [] if val is _MISSING else [val]
    else:
        values = list(payload.values())
    texts: list[str] = []
    stack = list(values)
    depth = 0
    while stack and depth < 10_000:
        depth += 1
        item = stack.pop()
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return texts


def assert_directional_claims_consistent(
    key: str | None = None,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> Validator:
    """Reject prose whose direction disagrees with its own numbers.

    Scans the submitted text (``key``, or every string in the payload) for
    "<moved> from A to B" claims and rejects "declined from 926,400.00 to
    926,400.00" (effectively equal) or "fell from 3.9M to 4.2M" (opposite
    direction). Tolerances are the task's materiality rule; the defaults
    only reject float noise. Works the same whatever produced the numbers.
    """

    def v(payload: Mapping[str, Any]) -> None:
        problems: list[str] = []
        for text in _payload_texts(payload, key):
            problems.extend(
                check_directional_claims(
                    text,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
            )
        assert not problems, "Directional claims disagree with their numbers:\n" + "\n".join(
            f"- {p}" for p in problems
        )

    return v


def assert_ranking_disclosed(
    key: str | None,
    question_text: str,
) -> Validator | None:
    """Reject an answer to a "rank by <concept>" task that hides the metric.

    Returns ``None`` when the task does not ask for a ranking by a concept,
    so it chains unconditionally. When it does, the answer must mention the
    concept (the ranking metric must be visible or explained) and must not
    say "ranked by <something else>" without calling it a proxy.
    """
    request = infer_requested_ranking(question_text or "")
    if request is None:
        return None

    def v(payload: Mapping[str, Any]) -> None:
        text = "\n".join(_payload_texts(payload, key))
        problems = check_ranking_disclosure(text, request)
        assert not problems, "\n".join(problems)

    return v


def assert_grain_preserved(key: str, dimensions: Iterable[str]) -> Validator:
    """Reject a result that silently drops or adds analytical dimensions.

    ``payload[key]`` may be a list of record dicts (each must carry every
    dimension key) or text (each dimension label must appear). A grain
    change is tolerated only when the payload also explains it: a string
    field named ``grain_note`` or ``grain_explanation``, or the word
    "grain" in the text.
    """
    wanted = [str(d) for d in dimensions]

    def v(payload: Mapping[str, Any]) -> None:
        val = _resolve(payload, key)
        if val is _MISSING or val is None:
            return
        explanation = payload.get("grain_note") or payload.get("grain_explanation")
        if isinstance(val, (list, tuple)) and val and all(isinstance(r, Mapping) for r in val):
            # Hand the record keys to validate_grain as they are: it matches
            # leaf names when they are unique and requires the table
            # qualification when they are not, so Sold To[Region] cannot
            # stand in for Products[Region]. Extra record columns (measures)
            # are not grain, so only the requested dimensions are compared.
            actual = [str(k) for k in val[0].keys()]
            requested_leaves = {_norm(d) for d in wanted}
            actual_dims = [k for k in actual if _norm(k) in requested_leaves]
            try:
                validate_grain(requested=wanted, actual=actual_dims, explanation=explanation)
            except AnalyticalIntegrityError as exc:
                assert False, str(exc)
            return
        text = val if isinstance(val, str) else str(val)
        lowered = _norm(text)
        present = [d for d in wanted if _norm(d) in lowered]
        if len(present) != len(wanted) and not explanation and "grain" not in text.lower():
            missing = [d for d in wanted if d not in present]
            assert False, (
                f"The result does not show the requested grain {wanted}; "
                f"{missing} never appear. Report at the requested grain or state why it changed."
            )

    return v


def _norm(value: Any) -> str:
    text = str(value)
    m = re.match(r"^\s*'?([^'\[\]]+?)'?\s*\[([^\[\]]+)\]\s*$", text)
    if m:
        text = m.group(2)
    return re.sub(r"[^a-z0-9]", "", text.lower())


__all__ = [
    "Validator",
    "assert_directional_claims_consistent",
    "assert_ranking_disclosed",
    "assert_grain_preserved",
    "chain",
    "signature_validator",
    "assert_keys",
    "assert_list_len",
    "assert_list_of",
    "assert_in_range",
    "assert_matches_regex",
    "assert_predicate",
    "assert_answers_all_subquestions",
    "assert_not_clarification_request",
    "infer_subquestion_count",
]
