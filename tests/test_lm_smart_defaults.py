"""Tests for fabric_rlm.lm smart defaults and reasoning-model detection."""

from __future__ import annotations

import pytest

from fabric_rlm.lm import (
    OpenAILM,
    _smart_defaults,
    is_reasoning_model,
    resolve_lm,
)


# ---------------------------------------------------------------------------
# is_reasoning_model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model",
    [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-2025-04-14",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
        "openai/gpt-5",
        "azure/gpt-5",
        "openrouter/openai/gpt-5",
        "fabric/gpt-5-mini",
    ],
)
def test_reasoning_models_detected(model: str) -> None:
    assert is_reasoning_model(model), model


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "gpt-5-chat",                 # Azure non-reasoning variant
        "gpt-5-chat-latest",
        "claude-3-5-sonnet",
        "openai/gpt-4.1",
        "fabric/gpt-4.1-mini",
    ],
)
def test_chat_models_not_detected(model: str) -> None:
    assert not is_reasoning_model(model), model


# ---------------------------------------------------------------------------
# _smart_defaults
# ---------------------------------------------------------------------------

def test_reasoning_defaults_omit_temperature() -> None:
    d = _smart_defaults("gpt-5")
    assert "temperature" not in d
    assert d["max_tokens"] >= 16_000


def test_chat_defaults_include_temperature() -> None:
    d = _smart_defaults("gpt-4.1-mini")
    assert d["temperature"] == 1.0
    assert d["max_tokens"] == 16_000


# ---------------------------------------------------------------------------
# OpenAILM (no live API call — just argument-shape checks)
# ---------------------------------------------------------------------------

def test_openailm_strips_temperature_for_reasoning_model(monkeypatch) -> None:
    """User-supplied temperature on a reasoning model gets dropped silently
    so dspy doesn't raise."""
    captured: dict = {}

    class FakeLM:
        def __init__(self, model: str, **kw):
            captured["model"] = model
            captured["kw"] = kw

    import fabric_rlm.lm as lm_mod
    monkeypatch.setattr(lm_mod, "dspy", type("S", (), {"LM": FakeLM})(), raising=False)
    # The factory imports dspy locally; patch sys.modules instead.
    import sys, types
    fake_dspy = types.SimpleNamespace(LM=FakeLM)
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)

    OpenAILM("gpt-5", api_key="sk-test", temperature=0.3, max_tokens=20_000)

    assert captured["model"] == "openai/gpt-5"
    assert "temperature" not in captured["kw"]
    assert captured["kw"]["max_tokens"] == 20_000


def test_openailm_keeps_temperature_for_chat_model(monkeypatch) -> None:
    captured: dict = {}

    class FakeLM:
        def __init__(self, model: str, **kw):
            captured["kw"] = kw

    import sys, types
    fake_dspy = types.SimpleNamespace(LM=FakeLM)
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)

    OpenAILM("gpt-4.1-mini", api_key="sk-test", temperature=0.0)

    assert captured["kw"]["temperature"] == 0.0


def test_openailm_passes_through_reasoning_effort(monkeypatch) -> None:
    captured: dict = {}

    class FakeLM:
        def __init__(self, model: str, **kw):
            captured["kw"] = kw

    import sys, types
    fake_dspy = types.SimpleNamespace(LM=FakeLM)
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)

    OpenAILM("gpt-5", api_key="sk-test", reasoning_effort="low")

    assert captured["kw"]["reasoning_effort"] == "low"
    assert "temperature" not in captured["kw"]


# ---------------------------------------------------------------------------
# resolve_lm string-spec path
# ---------------------------------------------------------------------------

def test_resolve_lm_string_spec_uses_smart_defaults(monkeypatch) -> None:
    captured: dict = {}

    class FakeLM:
        def __init__(self, model: str, **kw):
            captured["model"] = model
            captured["kw"] = kw

    import sys, types
    fake_dspy = types.SimpleNamespace(LM=FakeLM)
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)

    # Reasoning model — temperature must NOT be in kwargs.
    resolve_lm("openai/gpt-5")
    assert "temperature" not in captured["kw"]

    # Chat model — temperature should default to 1.0.
    resolve_lm("openai/gpt-4.1-mini")
    assert captured["kw"]["temperature"] == 1.0
