"""TDD for verify_shape_tolerant — template-agnostic generic grader."""
from __future__ import annotations

import pytest

from bench.adaptive.longcot_adapter import verify_shape_tolerant


# ---------------------------------------------------------------- list[int]

LIST_GOLD = [2, 1, 0]


@pytest.mark.parametrize(
    "response",
    [
        "[2, 1, 0]",
        "solution = [2, 1, 0]",
        "Answer: [2,1,0]",
        "```json\n[2, 1, 0]\n```",
        "```\n[2, 1, 0]\n```",
        '{"answer": [2, 1, 0]}',
        '{"solution":[2,1,0]}',
        '{"result": [2, 1, 0]}',
        "Q1: 2\nQ2: 1\nQ3: 0",
        "**Answer:** [2, 1, 0]",
        "<think>let me work it out for a while...</think>\n[2, 1, 0]",
        "2, 1, 0",
        "After much thought, the answer is [2, 1, 0].",
    ],
)
def test_list_gold_positives(response: str) -> None:
    assert verify_shape_tolerant(LIST_GOLD, response) is True, response


@pytest.mark.parametrize(
    "response",
    [
        "[2, 1, 1]",
        "[2, 1]",
        "[0, 1, 2]",
        "the answer is unknown",
        "",
    ],
)
def test_list_gold_negatives(response: str) -> None:
    assert verify_shape_tolerant(LIST_GOLD, response) is False, response


# ---------------------------------------------------------------- int

INT_GOLD = 5


@pytest.mark.parametrize(
    "response",
    [
        "5",
        "Answer: 5",
        "**Answer:** 5",
        r"\boxed{5}",
        '{"answer": 5}',
        "After analysis, the answer is 5.",
        "solution = 5",
    ],
)
def test_int_gold_positives(response: str) -> None:
    assert verify_shape_tolerant(INT_GOLD, response) is True, response


@pytest.mark.parametrize(
    "response",
    [
        "6",
        "the answer is fifty",
        "",
    ],
)
def test_int_gold_negatives(response: str) -> None:
    assert verify_shape_tolerant(INT_GOLD, response) is False, response


# ---------------------------------------------------------------- dict

DICT_GOLD = {"a": 1, "b": 2}


@pytest.mark.parametrize(
    "response",
    [
        '{"a": 1, "b": 2}',
        '{"b": 2, "a": 1}',
        '{"a":"1","b":"2"}',
        '```json\n{"a":1,"b":2}\n```',
        'Final answer: {"a": 1, "b": 2}',
    ],
)
def test_dict_gold_positives(response: str) -> None:
    assert verify_shape_tolerant(DICT_GOLD, response) is True, response


@pytest.mark.parametrize(
    "response",
    [
        '{"a": 1, "b": 2, "c": 3}',
        '{"a": 1}',
        '{"a": 2, "b": 1}',
        "",
    ],
)
def test_dict_gold_negatives(response: str) -> None:
    assert verify_shape_tolerant(DICT_GOLD, response) is False, response


# ---------------------------------------------------------------- str gold

def test_str_gold_whitespace_tolerant() -> None:
    assert verify_shape_tolerant("hello world", "  hello world  \n") is True
    assert verify_shape_tolerant("hello world", "**hello world**") is True
    assert verify_shape_tolerant("hello world", "Answer: hello world.") is True
    assert verify_shape_tolerant("hello world", "goodbye world") is False


# ---------------------------------------------------------------- json gold

def test_json_string_gold_is_parsed() -> None:
    assert verify_shape_tolerant("[2, 1, 0]", "[2, 1, 0]") is True
    assert verify_shape_tolerant('{"a": 1}', '{"a": 1}') is True


# ---------------------------------------------------------------- defensive

def test_none_response() -> None:
    assert verify_shape_tolerant([1, 2, 3], None) is False


def test_none_expected_only_matches_none_text() -> None:
    assert verify_shape_tolerant(None, "") is True
    assert verify_shape_tolerant(None, "anything") is False
