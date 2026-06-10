"""Unit tests for the v7 token-accounting helper.

``_sum_usage_from_histories`` harvests dspy ``lm.history`` entries appended
during a v7/dspy run so ``RLMResult`` reports the same ``total_*`` token
fields as the v6 engine (pre-fix they were always ``None``).
"""

from __future__ import annotations

from fabric_rlm.runtime import _sum_usage_from_histories


class _LMWithHistory:
    def __init__(self, history):
        self.history = history


def _entry(prompt, completion, cached=None, reasoning=None):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    if reasoning is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return {"usage": usage}


def test_sums_only_entries_after_pre_len() -> None:
    lm = _LMWithHistory(
        [
            _entry(999, 999),  # pre-existing: must be excluded
            _entry(100, 10, cached=40, reasoning=5),
            _entry(200, 20, cached=60, reasoning=15),
        ]
    )
    totals = _sum_usage_from_histories([lm], [1])
    assert totals == {
        "prompt_tokens": 300,
        "completion_tokens": 30,
        "cached_tokens": 100,
        "reasoning_tokens": 20,
    }


def test_sums_across_outer_and_sub_lm() -> None:
    outer = _LMWithHistory([_entry(100, 10)])
    sub = _LMWithHistory([_entry(50, 5)])
    totals = _sum_usage_from_histories([outer, sub], [0, 0])
    assert totals["prompt_tokens"] == 150
    assert totals["completion_tokens"] == 15


def test_unknown_stays_none_not_zero() -> None:
    """No usage reported -> None (unknown), matching the v6 convention."""
    lm = _LMWithHistory([{"no_usage_here": True}])
    totals = _sum_usage_from_histories([lm], [0])
    assert totals == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "reasoning_tokens": None,
    }


def test_lm_without_history_contributes_nothing() -> None:
    class _Mock:
        pass

    totals = _sum_usage_from_histories([_Mock()], [0])
    assert totals["prompt_tokens"] is None


def test_partial_fields_accumulate_independently() -> None:
    lm = _LMWithHistory(
        [
            _entry(100, 10),                 # no cached/reasoning details
            _entry(200, 20, cached=50),      # cached only
        ]
    )
    totals = _sum_usage_from_histories([lm], [0])
    assert totals["prompt_tokens"] == 300
    assert totals["cached_tokens"] == 50
    assert totals["reasoning_tokens"] is None
