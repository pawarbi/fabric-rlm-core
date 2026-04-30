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

    LongCoT-style outputs typically come as ``payload = {'solution': '<json str>'}``
    where the actual fields are inside the JSON. We try both forms.
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

    Resolves ``key`` from the top-level payload OR from a JSON string in
    ``payload['solution']`` (longcot pattern).
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


__all__ = [
    "Validator",
    "chain",
    "signature_validator",
    "assert_keys",
    "assert_list_len",
    "assert_list_of",
    "assert_in_range",
    "assert_matches_regex",
    "assert_predicate",
]
