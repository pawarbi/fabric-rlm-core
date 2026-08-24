"""Tests for universal multi-part-question shape and clarification guards.

These guards target failure modes observed in the 2026-05-03 Fabric run:
- Backprop_hard_10 submitted [6362, 8, -125039336] for a 50-element prompt.
- Rung-0 submissions were "Acknowledged. Please confirm..." — a question, not an answer.

The helpers must remain task-agnostic — they look at the prompt itself for
enumeration shape, and at the answer surface for clarification openers.
"""

from __future__ import annotations

import pytest

from fabric_rlm.validators import (
    assert_answers_all_subquestions,
    assert_not_clarification_request,
    chain,
    infer_subquestion_count,
)
from fabric_rlm.runtime import validate_submit_payload


def test_validate_submit_payload_enforces_declared_types():
    result = validate_submit_payload(
        {"result": "South"},
        ["result"],
        {"result": dict},
    )

    assert result.errors == ("Required output field 'result' must be dict, got str.",)


@pytest.mark.parametrize(
    ("expected_type", "value"),
    [
        (dict, {"top_region": "South"}),
        (list, ["South"]),
        (str, "South"),
        (int, 42),
        (float, 42.5),
        (bool, True),
    ],
)
def test_validate_submit_payload_accepts_declared_types(expected_type, value):
    result = validate_submit_payload({"result": value}, ["result"], {"result": expected_type})

    assert result.ok


@pytest.mark.parametrize(
    ("expected_type", "value"),
    [
        (dict, ["South"]),
        (list, {"region": "South"}),
        (str, 42),
        (int, True),
        (float, 42),
        (bool, 1),
    ],
)
def test_validate_submit_payload_rejects_wrong_declared_types(expected_type, value):
    result = validate_submit_payload({"result": value}, ["result"], {"result": expected_type})

    assert not result.ok
    assert expected_type.__name__ in result.errors[0]
    assert type(value).__name__ in result.errors[0]


def test_validate_submit_payload_accepts_subclasses_for_custom_types():
    class OutputBase:
        pass

    class OutputChild(OutputBase):
        pass

    result = validate_submit_payload(
        {"result": OutputChild()},
        ["result"],
        {"result": OutputBase},
    )

    assert result.ok


# ---------- infer_subquestion_count ----------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", 0),
        (None, 0),
        ("hello world", 0),
        ("Q1: foo  Q2: bar  Q3: baz", 3),
        ("q1) what  q2) why  q3) when  q4) where", 4),
        ("1. step\n2. step\n3. step\n4. step\n5. step", 5),
        ("Question 1 ... Question 2 ... Question 3", 3),
        ("Part 1: a\nPart 2: b\nPart 3: c\nPart 4: d", 4),
        ("Step 1: do thing\nStep 2: do other\nStep 3: done", 3),
        # Non-contiguous starting numbers should NOT count
        ("Q5: foo  Q6: bar", 0),
        # Stops at first gap
        ("Q1: a Q2: b Q3: c Q5: e Q6: f", 3),
        # Picks largest scheme
        ("Q1 Q2 Step 1 Step 2 Step 3 Step 4", 4),
    ],
)
def test_infer_subquestion_count(text, expected):
    assert infer_subquestion_count(text or "") == expected


def test_infer_handles_50_part_pattern():
    text = " ".join(f"Q{i}:" for i in range(1, 51))
    assert infer_subquestion_count(text) == 50


# ---------- assert_answers_all_subquestions ----------


def test_no_validator_when_question_has_no_enumeration():
    v = assert_answers_all_subquestions("answer", "Just a single normal question?")
    assert v is None


def test_no_validator_below_threshold():
    # Only Q1, Q2 detected -> below default threshold of 3
    v = assert_answers_all_subquestions("answer", "Q1: a  Q2: b")
    assert v is None


def test_validator_built_when_enumeration_present():
    v = assert_answers_all_subquestions("answer", "Q1: a  Q2: b  Q3: c  Q4: d")
    assert v is not None
    # 4-element list passes
    v({"answer": [1, 2, 3, 4]})
    v({"answer": [1, 2, 3, 4, 5]})  # over-answering OK


def test_validator_rejects_short_list():
    v = assert_answers_all_subquestions("answer", "Q1: a  Q2: b  Q3: c  Q4: d")
    with pytest.raises(AssertionError, match="3 items but the prompt enumerates 4"):
        v({"answer": [1, 2, 3]})


def test_validator_handles_50_part_backprop_failure_case():
    """Reproduce the canonical Backprop_hard_10 failure: 3-element answer for 50-Q."""
    text = "\n".join(f"Q{i}: compute backprop step {i}" for i in range(1, 51))
    v = assert_answers_all_subquestions("answer", text)
    assert v is not None
    with pytest.raises(AssertionError, match="3 items but the prompt enumerates 50"):
        v({"answer": [6362, 8, -125039336]})


def test_validator_accepts_dict_with_enough_keys():
    v = assert_answers_all_subquestions("answer", "Q1: x  Q2: y  Q3: z")
    v({"answer": {"q1": "a", "q2": "b", "q3": "c"}})


def test_validator_parses_json_string_answer():
    v = assert_answers_all_subquestions("answer", "Q1: a  Q2: b  Q3: c")
    v({"answer": "[1, 2, 3]"})
    with pytest.raises(AssertionError):
        v({"answer": "[1]"})


def test_validator_silent_on_scalar_answers():
    """Some prompts enumerate sub-questions but want a single integer answer
    (e.g. "considering Q1..Q5, what is the final value?"). Don't reject scalars."""
    v = assert_answers_all_subquestions("answer", "Q1: a  Q2: b  Q3: c")
    v({"answer": 42})
    v({"answer": "the answer is 42"})  # not JSON, leave alone


def test_validator_silent_on_missing_key():
    v = assert_answers_all_subquestions("answer", "Q1: a  Q2: b  Q3: c")
    v({})  # let assert_keys handle missing-key reporting
    v({"answer": None})


def test_validator_custom_threshold():
    v = assert_answers_all_subquestions("answer", "Q1: a  Q2: b", min_count=2)
    assert v is not None
    with pytest.raises(AssertionError):
        v({"answer": [1]})


# ---------- assert_not_clarification_request ----------


@pytest.mark.parametrize(
    "answer",
    [
        "Acknowledged. Please confirm the exact target value before I proceed.",
        "acknowledged — i will wait for clarification.",
        "Please confirm the input format.",
        "Please clarify the desired output.",
        "Could you please specify which algorithm to use?",
        "Can you provide the missing data?",
        "Would you confirm the constraint?",
        "I need more information about the dataset.",
        "I require additional context to answer this.",
        "I would need the full input to proceed.",
        "Before I can answer, please share the schema.",
        "Before answering, please provide the input.",
    ],
)
def test_clarification_openers_rejected(answer):
    v = assert_not_clarification_request("answer")
    with pytest.raises(AssertionError, match="clarification request"):
        v({"answer": answer})


@pytest.mark.parametrize(
    "answer",
    [
        "The answer is 42.",
        "[1, 2, 3, 4, 5]",
        "After computing, I get x=7.",
        "Sorting the array yields [1,2,3].",
        # Even if "please" appears later it shouldn't trigger
        "x=5. Please verify if you wish.",
        # Edge: includes the word "confirm" but not as opener
        "I confirm that the result is correct.",
    ],
)
def test_concrete_answers_pass(answer):
    v = assert_not_clarification_request("answer")
    v({"answer": answer})


def test_clarification_silent_on_non_string():
    v = assert_not_clarification_request("answer")
    v({"answer": [1, 2, 3]})
    v({"answer": 42})
    v({"answer": None})
    v({})


# ---------- chain integration ----------


def test_chain_with_optional_subquestion_validator():
    """assert_answers_all_subquestions returns None for non-enumerated prompts;
    chain() must skip None entries (already supported, just verify here)."""
    sub = assert_answers_all_subquestions("answer", "single question?")
    not_clar = assert_not_clarification_request("answer")
    assert sub is None
    composed = chain(sub, not_clar)
    composed({"answer": "42"})
    with pytest.raises(AssertionError):
        composed({"answer": "Please confirm the question."})
