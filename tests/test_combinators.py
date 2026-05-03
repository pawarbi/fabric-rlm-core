"""Tests for the 7 combinator primitives.

Per SPEC-combinators-skill.md: pure-Python primitives the model can compose
in the REPL. No template-specific behaviour, no I/O inside primitives.
"""
from __future__ import annotations

import pytest

from fabric_rlm.skills._combinators import (
    concat,
    cross,
    filter_,
    map_,
    peek,
    reduce_,
    split,
)


# ---- split ---------------------------------------------------------------

def test_split_word_aware_two_chunks() -> None:
    assert split("a b c d", 2) == ["a b", "c d"]


def test_split_with_more_chunks_than_words_pads_empty() -> None:
    out = split("a b", 4)
    assert len(out) == 4
    assert "".join(out).replace(" ", "") == "ab"


def test_split_k_one_returns_whole() -> None:
    assert split("hello world", 1) == ["hello world"]


def test_split_empty_input() -> None:
    assert split("", 3) == ["", "", ""]


def test_split_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        split("a b", 0)
    with pytest.raises(ValueError):
        split("a b", -1)


def test_split_non_ascii_preserved() -> None:
    out = split("café résumé naïve", 3)
    assert "".join(out).replace(" ", "") == "caférésuménaïve"


# ---- peek ----------------------------------------------------------------

def test_peek_basic_window() -> None:
    assert peek("abcdefghij", 2, 4) == "cdef"


def test_peek_clamps_to_end() -> None:
    assert peek("abc", 1, 100) == "bc"


def test_peek_offset_past_end_returns_empty() -> None:
    assert peek("abc", 10, 5) == ""


def test_peek_negative_offset_or_n_raises() -> None:
    with pytest.raises(ValueError):
        peek("abc", -1, 1)
    with pytest.raises(ValueError):
        peek("abc", 0, -1)


def test_peek_records_cost() -> None:
    from fabric_rlm.skills import _combinators as c

    c.reset_peek_counter()
    peek("abcdefgh", 0, 3)
    peek("abcdefgh", 3, 2)
    assert c.get_peek_counter() == {"calls": 2, "chars_read": 5}


# ---- map_ / filter_ / reduce_ -------------------------------------------

def test_map_applies_callable() -> None:
    assert map_([1, 2, 3], lambda x: x * 2) == [2, 4, 6]


def test_map_empty_input() -> None:
    assert map_([], lambda x: x) == []


def test_map_non_callable_raises() -> None:
    with pytest.raises(TypeError):
        map_([1, 2], 5)  # type: ignore[arg-type]


def test_filter_keeps_truthy() -> None:
    assert filter_([1, 2, 3, 4], lambda x: x % 2 == 0) == [2, 4]


def test_filter_empty_input() -> None:
    assert filter_([], lambda x: True) == []


def test_filter_non_callable_raises() -> None:
    with pytest.raises(TypeError):
        filter_([1, 2], 5)  # type: ignore[arg-type]


def test_reduce_sum_with_initial() -> None:
    assert reduce_([1, 2, 3, 4], lambda a, b: a + b, 0) == 10


def test_reduce_no_initial_uses_first_element() -> None:
    assert reduce_([1, 2, 3, 4], lambda a, b: a + b) == 10


def test_reduce_empty_no_initial_raises() -> None:
    with pytest.raises(TypeError):
        reduce_([], lambda a, b: a + b)


def test_reduce_empty_with_initial_returns_initial() -> None:
    assert reduce_([], lambda a, b: a + b, 42) == 42


# ---- concat --------------------------------------------------------------

def test_concat_strings() -> None:
    assert concat(["ab", "cd", "ef"]) == "abcdef"


def test_concat_lists() -> None:
    assert concat([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]


def test_concat_with_separator_string() -> None:
    assert concat(["a", "b", "c"], sep=", ") == "a, b, c"


def test_concat_separator_only_for_strings() -> None:
    with pytest.raises(TypeError):
        concat([[1], [2]], sep=", ")


def test_concat_empty_input() -> None:
    assert concat([]) == ""
    assert concat([], sep=",") == ""


def test_concat_mixed_types_raises() -> None:
    with pytest.raises(TypeError):
        concat(["a", [1, 2]])


# ---- cross ---------------------------------------------------------------

def test_cross_two_lists() -> None:
    assert cross([1, 2], ["a", "b"]) == [(1, "a"), (1, "b"), (2, "a"), (2, "b")]


def test_cross_three_lists() -> None:
    out = cross([0, 1], [0, 1], [0, 1])
    assert len(out) == 8
    assert (0, 0, 0) in out
    assert (1, 1, 1) in out


def test_cross_with_empty_factor_returns_empty() -> None:
    assert cross([1, 2], []) == []


def test_cross_zero_args_raises() -> None:
    with pytest.raises(ValueError):
        cross()


def test_cross_size_cap_raises_to_protect_oom() -> None:
    # 7 lists of 100 = 100^7 = 10^14 — must reject.
    with pytest.raises(ValueError, match="too large"):
        cross(*([list(range(100))] * 7))
