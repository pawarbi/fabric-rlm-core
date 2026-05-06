"""Tests for ``cached_tokens`` and ``reasoning_tokens`` extraction (NEW-A).

Background
----------
v4 SSB trace mining showed ``cached_seen=0`` and ``reasoning_seen=0`` across
all 1354 gpt-5 turns (and 1129 gpt-4.1 turns), even though gpt-5 ALWAYS
returns ``completion_tokens_details.reasoning_tokens`` and OpenAI/Anthropic
return ``prompt_tokens_details.cached_tokens`` whenever prompt caching kicks
in.

Root cause: when DSPy / LiteLLM stores the usage object on
``lm.history[-1]['usage']``, the nested ``*_details`` values are SDK objects
(Pydantic models or duck-typed objects with attributes), NOT plain dicts.
``_usage_nested_field`` does ``isinstance(nested, dict)`` and returns ``None``
for non-dicts, silently dropping the data. ``_extract_usage`` only normalizes
nested details on the OBJECT-shaped usage path (the ``getattr(usage, ...)``
branch); the dict-shaped usage path (the ``isinstance(usage, dict)`` branch)
returns ``dict(usage)`` shallowly, leaving the nested objects intact.

Fix: normalize nested details to plain dicts on BOTH paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fabric_rlm import RLM
import fabric_rlm.runtime as runtime_module
from fabric_rlm.runtime import _extract_usage, _usage_nested_field


# ---------------------------------------------------------------------------
# Unit tests for _extract_usage
# ---------------------------------------------------------------------------


class _ObjUsageDetails:
    """Mimics OpenAI/litellm ``CompletionTokensDetails`` / ``PromptTokensDetails``
    objects which expose attributes but are NOT dicts and do not implement
    ``__iter__`` over keys."""

    def __init__(self, **fields):
        for key, value in fields.items():
            setattr(self, key, value)


def test_extract_usage_normalizes_object_shaped_prompt_details_in_dict_usage() -> None:
    """response is a dict (dspy history entry shape); usage is a dict;
    nested ``prompt_tokens_details`` is an OBJECT — must normalize to dict."""
    response = {
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "prompt_tokens_details": _ObjUsageDetails(cached_tokens=512, audio_tokens=0),
        }
    }
    usage = _extract_usage(response)
    assert isinstance(usage["prompt_tokens_details"], dict), (
        f"nested details must be normalized to dict, got {type(usage['prompt_tokens_details'])}"
    )
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") == 512


def test_extract_usage_normalizes_object_shaped_completion_details_in_dict_usage() -> None:
    response = {
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 5000,
            "total_tokens": 6000,
            "completion_tokens_details": _ObjUsageDetails(
                reasoning_tokens=4500,
                accepted_prediction_tokens=0,
                rejected_prediction_tokens=0,
            ),
        }
    }
    usage = _extract_usage(response)
    assert isinstance(usage["completion_tokens_details"], dict)
    assert _usage_nested_field(usage, "completion_tokens_details", "reasoning_tokens") == 4500


def test_extract_usage_handles_both_details_object_shaped() -> None:
    response = {
        "usage": {
            "prompt_tokens": 8000,
            "completion_tokens": 2000,
            "total_tokens": 10000,
            "prompt_tokens_details": _ObjUsageDetails(cached_tokens=4096),
            "completion_tokens_details": _ObjUsageDetails(reasoning_tokens=1500),
        }
    }
    usage = _extract_usage(response)
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") == 4096
    assert _usage_nested_field(usage, "completion_tokens_details", "reasoning_tokens") == 1500


def test_extract_usage_dict_path_with_plain_dict_details_still_works() -> None:
    """Regression: plain-dict details (already normalized) must continue to work."""
    response = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 50},
            "completion_tokens_details": {"reasoning_tokens": 10},
        }
    }
    usage = _extract_usage(response)
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") == 50
    assert _usage_nested_field(usage, "completion_tokens_details", "reasoning_tokens") == 10


def test_extract_usage_object_path_still_works() -> None:
    """Regression: object-shaped usage (existing branch) still normalizes."""
    usage_obj = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=_ObjUsageDetails(cached_tokens=25),
        completion_tokens_details=_ObjUsageDetails(reasoning_tokens=10),
    )
    response = SimpleNamespace(usage=usage_obj)
    usage = _extract_usage(response)
    assert usage["prompt_tokens"] == 100
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") == 25
    assert _usage_nested_field(usage, "completion_tokens_details", "reasoning_tokens") == 10


def test_extract_usage_handles_missing_details_gracefully() -> None:
    response = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    usage = _extract_usage(response)
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") is None
    assert _usage_nested_field(usage, "completion_tokens_details", "reasoning_tokens") is None


def test_extract_usage_handles_none_nested_value() -> None:
    """Some providers send ``prompt_tokens_details=None`` explicitly."""
    response = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        }
    }
    usage = _extract_usage(response)
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") is None
    assert _usage_nested_field(usage, "completion_tokens_details", "reasoning_tokens") is None


def test_extract_usage_drops_none_child_attributes() -> None:
    """If the nested details object has cached_tokens=None, the normalized dict
    should either omit the key or store None — either way ``_usage_nested_field``
    returns None.
    """
    response = {
        "usage": {
            "prompt_tokens": 100,
            "prompt_tokens_details": _ObjUsageDetails(cached_tokens=None, audio_tokens=0),
        }
    }
    usage = _extract_usage(response)
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") is None


# ---------------------------------------------------------------------------
# End-to-end: trajectory must surface cached/reasoning when LM reports them
# ---------------------------------------------------------------------------


class _LMResponseObj:
    """Stand-in matching the real dspy/litellm history entry shape."""

    def __init__(self, content: str, usage: dict | None = None):
        self.content = content
        if usage is not None:
            self.usage = usage


class _UsageScriptedLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def __call__(self, *, messages):
        text, usage = self.responses.pop(0)
        return _LMResponseObj(text, usage)


class _FakeExec:
    def __init__(self):
        self.stdout = "ok"
        self.stderr = ""
        self.error = None
        self.submitted = True
        self.submit_payload = {"answer": 1}
        self.state: dict = {}
        self.ok = True


class _FakeInterp:
    def __init__(self, **_):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def configure_lm(self, _):
        pass

    def set_inputs(self, _):
        pass

    def execute(self, _):
        return _FakeExec()


def test_trajectory_records_cached_tokens_from_object_shaped_details(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "Interpreter", lambda **kw: _FakeInterp())
    lm = _UsageScriptedLM(
        [
            (
                "```python\nSUBMIT(answer=1)\n```",
                {
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "total_tokens": 1200,
                    "prompt_tokens_details": _ObjUsageDetails(cached_tokens=512),
                },
            )
        ]
    )
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=2, timeout=5)
    result = rlm.run()
    assert result.submitted
    assert result.trajectory[0].cached_tokens == 512, (
        f"cached_tokens not surfaced: trajectory[0].cached_tokens="
        f"{result.trajectory[0].cached_tokens}"
    )
    assert result.total_cached_tokens == 512


def test_trajectory_records_reasoning_tokens_from_object_shaped_details(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "Interpreter", lambda **kw: _FakeInterp())
    lm = _UsageScriptedLM(
        [
            (
                "```python\nSUBMIT(answer=1)\n```",
                {
                    "prompt_tokens": 500,
                    "completion_tokens": 4500,
                    "total_tokens": 5000,
                    "completion_tokens_details": _ObjUsageDetails(reasoning_tokens=4000),
                },
            )
        ]
    )
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=2, timeout=5)
    result = rlm.run()
    assert result.submitted
    assert result.trajectory[0].reasoning_tokens == 4000
    assert result.total_reasoning_tokens == 4000


def test_trajectory_aggregates_cached_and_reasoning_across_turns(monkeypatch) -> None:
    """Multi-turn run: per-turn caches/reasoning sum into trajectory totals."""
    monkeypatch.setattr(runtime_module, "Interpreter", lambda **kw: _FakeInterp())

    class _MultiTurnInterp:
        def __init__(self, **_):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def configure_lm(self, _):
            pass

        def set_inputs(self, _):
            pass

        def execute(self, _):
            self.calls += 1
            r = _FakeExec()
            if self.calls < 2:
                r.submitted = False
                r.submit_payload = {}
            return r

    monkeypatch.setattr(runtime_module, "Interpreter", lambda **kw: _MultiTurnInterp())

    lm = _UsageScriptedLM(
        [
            (
                "```python\nprint('thinking')\n```",
                {
                    "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
                    "prompt_tokens_details": _ObjUsageDetails(cached_tokens=200),
                    "completion_tokens_details": _ObjUsageDetails(reasoning_tokens=50),
                },
            ),
            (
                "```python\nSUBMIT(answer=1)\n```",
                {
                    "prompt_tokens": 1500, "completion_tokens": 80, "total_tokens": 1580,
                    "prompt_tokens_details": _ObjUsageDetails(cached_tokens=900),
                    "completion_tokens_details": _ObjUsageDetails(reasoning_tokens=30),
                },
            ),
        ]
    )
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=3, timeout=5)
    result = rlm.run()
    assert result.submitted
    assert len(result.trajectory) == 2
    assert result.total_cached_tokens == 1100  # 200 + 900
    assert result.total_reasoning_tokens == 80  # 50 + 30


# ---------------------------------------------------------------------------
# Rubber-duck review follow-ups (PR #5):
#   * Responses API alias normalization (input_tokens, output_tokens_details, ...)
#   * Anthropic native cache fields (cache_read_input_tokens, cache_creation_*)
#   * Pydantic-style model_dump fallback (preserves provider-specific fields)
#   * dspy lm.history[-1] path through _call_lm_with_meta with object-shaped details
# ---------------------------------------------------------------------------


class _PydanticishDetails:
    """Mimics a Pydantic model: has model_dump() returning a plain dict.

    Used to verify the helper prefers SDK serialization over hardcoded probes
    so that provider-specific fields are preserved for debugging.
    """

    def __init__(self, **fields):
        self._fields = dict(fields)

    def model_dump(self) -> dict:
        return dict(self._fields)


def test_extract_usage_uses_model_dump_when_available() -> None:
    """Pydantic model_dump path preserves provider-specific fields."""
    response = {
        "usage": {
            "prompt_tokens": 100,
            "prompt_tokens_details": _PydanticishDetails(
                cached_tokens=42,
                some_provider_specific_field="xyz",
            ),
        }
    }
    usage = _extract_usage(response)
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") == 42
    assert usage["prompt_tokens_details"]["some_provider_specific_field"] == "xyz"


def test_extract_usage_handles_responses_api_input_output_token_aliases() -> None:
    """OpenAI Responses API uses input_tokens/output_tokens.

    Ensure they're mapped onto the canonical prompt_tokens/completion_tokens
    so existing callers (cost reporting, _usage_field) keep working.
    """
    response = {
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "input_tokens_details": {"cached_tokens": 256},
            "output_tokens_details": {"reasoning_tokens": 400},
        }
    }
    usage = _extract_usage(response)
    assert usage.get("prompt_tokens") == 1000
    assert usage.get("completion_tokens") == 500
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") == 256
    assert _usage_nested_field(usage, "completion_tokens_details", "reasoning_tokens") == 400


def test_extract_usage_handles_responses_api_with_object_shaped_details() -> None:
    """Combined: Responses API aliases + object-shaped nested details."""
    response = {
        "usage": {
            "input_tokens": 8000,
            "output_tokens": 2000,
            "input_tokens_details": _ObjUsageDetails(cached_tokens=4096),
            "output_tokens_details": _ObjUsageDetails(reasoning_tokens=1500),
        }
    }
    usage = _extract_usage(response)
    assert usage["prompt_tokens"] == 8000
    assert usage["completion_tokens"] == 2000
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") == 4096
    assert _usage_nested_field(usage, "completion_tokens_details", "reasoning_tokens") == 1500


def test_extract_usage_handles_anthropic_cache_read_input_tokens() -> None:
    """Anthropic native shape: cache_read_input_tokens at top level.

    Map it into prompt_tokens_details.cached_tokens (same semantics: count of
    prompt tokens billed at the cache-read rate).
    """
    response = {
        "usage": {
            "input_tokens": 5000,
            "output_tokens": 200,
            "cache_read_input_tokens": 4500,
            "cache_creation_input_tokens": 0,
        }
    }
    usage = _extract_usage(response)
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") == 4500
    # cache_creation_input_tokens stays as a top-level field; do NOT silently
    # fold it into cached_tokens (different billing semantics).
    assert usage.get("cache_creation_input_tokens") == 0


def test_extract_usage_anthropic_cache_does_not_overwrite_existing_cached_tokens() -> None:
    """If both Anthropic cache_read AND prompt_tokens_details.cached_tokens are
    present, prefer the explicit nested value (don't double-count)."""
    response = {
        "usage": {
            "input_tokens": 5000,
            "cache_read_input_tokens": 4500,
            "prompt_tokens_details": {"cached_tokens": 9999},  # explicit, take this
        }
    }
    usage = _extract_usage(response)
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") == 9999


def test_extract_usage_does_not_overwrite_existing_canonical_fields() -> None:
    """If both `input_tokens` and `prompt_tokens` are present, keep the
    canonical one (defensive: never silently shift a value)."""
    response = {
        "usage": {
            "prompt_tokens": 100,
            "input_tokens": 999,  # alias must NOT win
        }
    }
    usage = _extract_usage(response)
    assert usage["prompt_tokens"] == 100


# ---------------------------------------------------------------------------
# Production path: dspy LM with history-based usage extraction
# ---------------------------------------------------------------------------


class _DspyHistoryLM:
    """Mimics dspy.LM behavior: __call__ returns a list of strings, and the
    LM appends a dict to ``lm.history`` whose ``usage`` is a dict containing
    OBJECT-shaped nested details. This is the production failure mode.
    """

    def __init__(self, *, content: str, usage: dict):
        self.content = content
        self.usage = usage
        self.history: list[dict] = []

    def __call__(self, *, messages):
        # dspy returns just the text(s).
        self.history.append(
            {
                "messages": messages,
                "response": object(),  # opaque; not used by us
                "outputs": [self.content],
                "usage": self.usage,
                "cost": 0.0,
            }
        )
        return [self.content]


def test_call_lm_with_meta_extracts_object_details_via_history_path() -> None:
    """End-to-end: when the LM exposes usage only via lm.history[-1]['usage'],
    AND the nested details are SDK objects, _call_lm_with_meta + _extract_usage
    must still surface cached_tokens / reasoning_tokens. This is the exact path
    that silently dropped 1354/1354 turns of data in the v4 gpt-5 SSB run.
    """
    from fabric_rlm.runtime import _call_lm_with_meta

    lm = _DspyHistoryLM(
        content="```python\nSUBMIT(answer=1)\n```",
        usage={
            "prompt_tokens": 2000,
            "completion_tokens": 1500,
            "total_tokens": 3500,
            "prompt_tokens_details": _ObjUsageDetails(cached_tokens=1024),
            "completion_tokens_details": _ObjUsageDetails(reasoning_tokens=1200),
        },
    )

    text, raw, _seconds = _call_lm_with_meta(lm, [{"role": "user", "content": "hi"}])
    usage = _extract_usage(raw)

    assert "SUBMIT" in text
    assert _usage_nested_field(usage, "prompt_tokens_details", "cached_tokens") == 1024, (
        f"production-shape extraction failed: usage={usage}"
    )
    assert _usage_nested_field(usage, "completion_tokens_details", "reasoning_tokens") == 1200
