"""Tests for fabric_rlm.validators (auto-generator + composable primitives)."""
from __future__ import annotations

import pytest
import dspy

from fabric_rlm.validators import (
    chain,
    signature_validator,
    assert_keys,
    assert_list_len,
    assert_list_of,
    assert_in_range,
    assert_matches_regex,
    assert_predicate,
)


# ---------------------------------------------------------------------------
# signature_validator (auto-generator)
# ---------------------------------------------------------------------------

class _MathSig(dspy.Signature):
    """add."""
    a: int = dspy.InputField()
    b: int = dspy.InputField()
    result: int = dspy.OutputField()


def test_signature_validator_accepts_correct_type() -> None:
    v = signature_validator(_MathSig)
    assert v is not None
    v({"result": 7})  # no exception


def test_signature_validator_rejects_wrong_type() -> None:
    v = signature_validator(_MathSig)
    with pytest.raises(AssertionError) as ei:
        v({"result": "not an int"})
    assert "result" in str(ei.value)


def test_signature_validator_rejects_missing_field() -> None:
    v = signature_validator(_MathSig)
    with pytest.raises(AssertionError) as ei:
        v({})
    msg = str(ei.value)
    assert "result" in msg


def test_signature_validator_only_validates_output_fields() -> None:
    """Input fields must NOT be required for SUBMIT-side validation."""
    v = signature_validator(_MathSig)
    # No 'a' or 'b' present, but they're inputs — only 'result' should be required.
    v({"result": 99})


class _ListSig(dspy.Signature):
    """list out."""
    q: str = dspy.InputField()
    items: list[int] = dspy.OutputField()


def test_signature_validator_list_of_int_accepts_ints() -> None:
    v = signature_validator(_ListSig)
    v({"items": [1, 2, 3]})


def test_signature_validator_list_of_int_rejects_strings() -> None:
    v = signature_validator(_ListSig)
    with pytest.raises(AssertionError) as ei:
        v({"items": ["x", "y"]})
    assert "int" in str(ei.value).lower()


class _NoOutputs(dspy.Signature):
    """only input."""
    q: str = dspy.InputField()


def test_signature_validator_returns_none_when_no_output_fields() -> None:
    """Edge case: signature with no OUTPUT fields => no validator."""
    assert signature_validator(_NoOutputs) is None


def test_signature_validator_handles_non_signature_gracefully() -> None:
    """Passing something that isn't a dspy.Signature returns None (no crash)."""
    assert signature_validator(int) is None
    assert signature_validator(object) is None


# ---------------------------------------------------------------------------
# Composable primitives
# ---------------------------------------------------------------------------

def test_assert_keys_passes_when_present() -> None:
    assert_keys("a", "b")({"a": 1, "b": 2})


def test_assert_keys_rejects_missing() -> None:
    with pytest.raises(AssertionError) as ei:
        assert_keys("a", "b")({"a": 1})
    assert "b" in str(ei.value)


def test_assert_keys_rejects_none_value() -> None:
    with pytest.raises(AssertionError):
        assert_keys("a")({"a": None})


def test_assert_list_len_exact_passes() -> None:
    assert_list_len("xs", 3)({"xs": [1, 2, 3]})


def test_assert_list_len_exact_rejects_short() -> None:
    with pytest.raises(AssertionError) as ei:
        assert_list_len("xs", 3)({"xs": [1]})
    msg = str(ei.value)
    assert "1" in msg and "3" in msg


def test_assert_list_len_min_mode() -> None:
    assert_list_len("xs", 2, exact=False)({"xs": [1, 2, 3]})
    with pytest.raises(AssertionError):
        assert_list_len("xs", 5, exact=False)({"xs": [1, 2]})


def test_assert_list_len_resolves_from_solution_json() -> None:
    """A JSON solution string may contain the requested field."""
    payload = {"solution": '{"final_capacities": [1, 2, 3, 4, 5]}'}
    assert_list_len("final_capacities", 5)(payload)
    with pytest.raises(AssertionError):
        assert_list_len("final_capacities", 8)(payload)


def test_assert_list_of_passes() -> None:
    assert_list_of("xs", int)({"xs": [1, 2, 3]})


def test_assert_list_of_rejects_mixed() -> None:
    with pytest.raises(AssertionError) as ei:
        assert_list_of("xs", int)({"xs": [1, "two", 3]})
    assert "1" in str(ei.value)  # the bad index


def test_assert_in_range_inclusive() -> None:
    assert_in_range("p", 0, 1)({"p": 0.5})
    assert_in_range("p", 0, 1)({"p": 0})
    assert_in_range("p", 0, 1)({"p": 1})
    with pytest.raises(AssertionError):
        assert_in_range("p", 0, 1)({"p": 1.5})


def test_assert_in_range_rejects_bool() -> None:
    """Booleans are subclass of int but shouldn't pass numeric checks."""
    with pytest.raises(AssertionError):
        assert_in_range("p", 0, 1)({"p": True})


def test_assert_matches_regex() -> None:
    assert_matches_regex("v", r"v\d+")({"v": "v42"})
    with pytest.raises(AssertionError):
        assert_matches_regex("v", r"v\d+")({"v": "abc"})


def test_assert_predicate() -> None:
    assert_predicate(lambda p: p["a"] < p["b"], "a must be < b")(
        {"a": 1, "b": 2}
    )
    with pytest.raises(AssertionError) as ei:
        assert_predicate(lambda p: p["a"] < p["b"], "a must be < b")(
            {"a": 5, "b": 2}
        )
    assert "a must be < b" in str(ei.value)


# ---------------------------------------------------------------------------
# chain composition
# ---------------------------------------------------------------------------

def test_chain_runs_in_order_and_short_circuits() -> None:
    calls: list[str] = []

    def v1(_):
        calls.append("v1")

    def v2(_):
        calls.append("v2")
        raise AssertionError("v2 failed")

    def v3(_):
        calls.append("v3")

    composed = chain(v1, v2, v3)
    with pytest.raises(AssertionError) as ei:
        composed({})
    assert "v2 failed" in str(ei.value)
    assert calls == ["v1", "v2"]  # v3 never ran


def test_chain_skips_none() -> None:
    """None entries are skipped so chain(maybe_validator, ...) works."""
    composed = chain(None, assert_keys("x"), None)
    composed({"x": 1})
    with pytest.raises(AssertionError):
        composed({})


def test_chain_signature_validator_plus_primitive() -> None:
    """End-to-end: combine auto-derived shape check with a semantic primitive."""
    composed = chain(
        signature_validator(_ListSig),
        assert_list_len("items", 3),
    )
    composed({"items": [1, 2, 3]})

    # Wrong type — caught by signature_validator
    with pytest.raises(AssertionError):
        composed({"items": ["a", "b", "c"]})

    # Right type, wrong length — caught by primitive
    with pytest.raises(AssertionError) as ei:
        composed({"items": [1, 2]})
    assert "length" in str(ei.value).lower() or "2" in str(ei.value)
