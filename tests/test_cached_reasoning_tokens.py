"""Tests for cached + reasoning token capture.

OpenAI/litellm responses for reasoning models include nested usage detail blocks:

    usage = {
      "prompt_tokens": 1234,
      "completion_tokens": 5678,
      "prompt_tokens_details": {"cached_tokens": 800},
      "completion_tokens_details": {"reasoning_tokens": 5000},
    }

These are critical for cost analysis on gpt-5: cached input is billed at ~10x
discount, and reasoning tokens are the dominant completion cost. Older code
silently dropped them; this regression suite pins the new plumbing so they
flow all the way from ``_extract_usage`` -> ``TurnRecord`` ->
``_aggregate_trajectory_metrics`` -> ``RLMResult``.
"""

from __future__ import annotations

from fabric_rlm.runtime import (
    RLMResult,
    _aggregate_trajectory_metrics,
    _extract_usage,
    _usage_field,
    _usage_nested_field,
)
from fabric_rlm.trajectory import Trajectory, TurnRecord


def _make_turn(turn=1, **tokens):
    return TurnRecord(
        turn=turn,
        code="",
        stdout="",
        stderr="",
        error=None,
        submitted=False,
        state={},
        **tokens,
    )


# ---------------------------------------------------------------------------
# _extract_usage / _usage_nested_field
# ---------------------------------------------------------------------------


def test_extract_usage_preserves_nested_details_for_dict_response():
    """Dict responses already preserved nested details — pin that behavior."""

    response = {
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 5000,
            "prompt_tokens_details": {"cached_tokens": 800},
            "completion_tokens_details": {"reasoning_tokens": 4500},
        }
    }
    usage = _extract_usage(response)
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") == 800
    assert _usage_nested_field(usage, "completion_tokens_details", "reasoning_tokens") == 4500


def test_extract_usage_preserves_nested_details_for_object_response():
    """REGRESSION: previously the object branch dropped nested *_details blocks
    entirely, leaving cached_tokens unobservable for litellm/openai SDK
    responses. The fix walks attribute access on the nested objects too.
    """

    class _NestedDetails:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _Usage:
        prompt_tokens = 2000
        completion_tokens = 6000
        total_tokens = 8000
        prompt_tokens_details = _NestedDetails(cached_tokens=1500)
        completion_tokens_details = _NestedDetails(reasoning_tokens=5500)

    class _Response:
        usage = _Usage()

    usage = _extract_usage(_Response())
    assert _usage_field(usage, "prompt_tokens") == 2000
    assert _usage_field(usage, "completion_tokens") == 6000
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") == 1500
    assert _usage_nested_field(usage, "completion_tokens_details", "reasoning_tokens") == 5500


def test_usage_nested_field_returns_none_for_missing_or_malformed():
    assert _usage_nested_field({}, "prompt_tokens_details", "cached_tokens") is None
    # Parent present but not a mapping (defensive against odd SDK shapes)
    assert _usage_nested_field({"prompt_tokens_details": 42}, "prompt_tokens_details", "cached_tokens") is None
    # Parent is a mapping but child missing
    assert _usage_nested_field({"prompt_tokens_details": {}}, "prompt_tokens_details", "cached_tokens") is None
    # Child present but non-numeric
    assert (
        _usage_nested_field({"prompt_tokens_details": {"cached_tokens": "??"}}, "prompt_tokens_details", "cached_tokens")
        is None
    )


# ---------------------------------------------------------------------------
# Aggregation onto RLMResult
# ---------------------------------------------------------------------------


def test_aggregate_sums_cached_and_reasoning_tokens():
    traj = Trajectory()
    traj.append(_make_turn(turn=1, prompt_tokens=1000, completion_tokens=5000, cached_tokens=800, reasoning_tokens=4500))
    traj.append(_make_turn(turn=2, prompt_tokens=1200, completion_tokens=6000, cached_tokens=1100, reasoning_tokens=5400))

    out = _aggregate_trajectory_metrics(traj)
    assert out["total_prompt_tokens"] == 2200
    assert out["total_completion_tokens"] == 11000
    assert out["total_cached_tokens"] == 1900
    assert out["total_reasoning_tokens"] == 9900


def test_aggregate_returns_none_when_nested_fields_absent():
    """Backward compatibility: if turns don't report cached/reasoning (old
    code path or non-reasoning model), totals must be None — never 0 — so
    callers can distinguish "not measured" from "measured zero".
    """

    traj = Trajectory()
    traj.append(_make_turn(turn=1, prompt_tokens=100, completion_tokens=50))
    traj.append(_make_turn(turn=2, prompt_tokens=200, completion_tokens=80))

    out = _aggregate_trajectory_metrics(traj)
    assert out["total_prompt_tokens"] == 300
    assert out["total_completion_tokens"] == 130
    assert out["total_cached_tokens"] is None
    assert out["total_reasoning_tokens"] is None


def test_rlmresult_to_dict_includes_new_aggregates():
    """Bench notebooks read RLMResult.to_dict — pin that the new fields
    are surfaced so downstream JSON artifacts get them."""

    traj = Trajectory()
    traj.append(_make_turn(turn=1, prompt_tokens=10, completion_tokens=20, cached_tokens=3, reasoning_tokens=15))
    result = RLMResult(
        submitted=True,
        payload={"answer": "x"},
        trajectory=traj,
        final_state={},
        **_aggregate_trajectory_metrics(traj),
    )
    payload = result.to_dict()
    assert payload["total_cached_tokens"] == 3
    assert payload["total_reasoning_tokens"] == 15
    # legacy fields still present and correct
    assert payload["total_prompt_tokens"] == 10
    assert payload["total_completion_tokens"] == 20
