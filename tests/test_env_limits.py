"""A malformed ``FABRIC_RLM_*_LIMIT`` env var must not crash ``import fabric_rlm``.

The feedback-limit module constants are read from the environment at import
time. ``_int_env`` guarantees a bad value falls back to the default (with a
warning) instead of raising a bare ``ValueError`` during import.
"""

from __future__ import annotations

import warnings

from fabric_rlm.runtime import _int_env


def test_int_env_uses_default_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("FABRIC_RLM_STDOUT_LIMIT", raising=False)
    assert _int_env("FABRIC_RLM_STDOUT_LIMIT", 5000) == 5000


def test_int_env_uses_default_when_blank(monkeypatch) -> None:
    monkeypatch.setenv("FABRIC_RLM_STDOUT_LIMIT", "   ")
    assert _int_env("FABRIC_RLM_STDOUT_LIMIT", 5000) == 5000


def test_int_env_parses_valid_value(monkeypatch) -> None:
    monkeypatch.setenv("FABRIC_RLM_STDOUT_LIMIT", "1234")
    assert _int_env("FABRIC_RLM_STDOUT_LIMIT", 5000) == 1234


def test_int_env_falls_back_on_malformed_value(monkeypatch) -> None:
    monkeypatch.setenv("FABRIC_RLM_STDOUT_LIMIT", "not-a-number")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _int_env("FABRIC_RLM_STDOUT_LIMIT", 5000) == 5000
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
