"""Tests for `_RefreshingLM` — the wrapper that retries once on auth-expiry.

The wrapper must be provider-agnostic: it works for any short-lived bearer
token (Azure AAD, GCP IAM, AWS IAM, custom OIDC) by accepting a
`token_provider: Callable[[], str]` and re-applying it on 401.

The wrapper subclasses `dspy.LM` so it passes the `isinstance(lm, dspy.BaseLM)`
check that `dspy.Predict` performs on the configured LM. This is essential
for the worker-side `predict()` helper that wraps a sub-LM in `dspy.Predict`.
"""

from __future__ import annotations

from typing import Any

import pytest

from fabric_rlm.lm import _get_refreshing_lm_cls, _is_auth_expired


class _FakeAuthError(Exception):
    """Mimics litellm.AuthenticationError shape (class name + message)."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)


# Force the class name to match what litellm raises
_FakeAuthError.__name__ = "AuthenticationError"


_RefreshingLM = _get_refreshing_lm_cls()


class _ScriptedLM(_RefreshingLM):  # type: ignore[misc, valid-type]
    """Test double: replays a scripted sequence of return values / exceptions.

    Overrides BOTH `_do_call` (used by `__call__`) and `forward` (used by
    `dspy.Predict`) so a single script drives whichever entry point a test
    exercises. The forward override re-implements the same refresh-then-retry
    logic so we can verify retry behaviour without actually relying on the
    base class's forward implementation.
    """

    def __init__(self, script: list[Any], **kwargs: Any) -> None:
        super().__init__("openai/gpt-4o", api_key="dummy", cache=False, **kwargs)
        self._script = list(script)
        self.call_count = 0

    def _next(self) -> Any:
        self.call_count += 1
        if not self._script:
            raise RuntimeError("script exhausted")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _do_call(self, *args: Any, **kwargs: Any) -> Any:
        return self._next()

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        try:
            return self._next()
        except Exception as exc:
            if not _is_auth_expired(exc):
                raise
            if not self._refresh():
                raise
            return self._next()


def test_refresh_on_auth_error_succeeds() -> None:
    """First call raises 401, refresh runs, second call succeeds."""
    refresh_calls: list[int] = []

    def provider() -> str:
        refresh_calls.append(1)
        return f"Bearer fresh-{len(refresh_calls)}"

    err = _FakeAuthError("AuthException - Error code: 401 - User Aad Token is expired")
    lm = _ScriptedLM(
        script=[err, ["ok"]],
        token_provider=provider,
        extra_headers={"Authorization": "Bearer stale"},
    )
    out = lm("hello")
    assert out == ["ok"]
    assert lm.call_count == 2
    assert len(refresh_calls) == 1
    assert lm.kwargs["extra_headers"]["Authorization"] == "Bearer fresh-1"


def test_non_auth_error_propagates_no_refresh() -> None:
    """A non-auth error (rate limit, 5xx, etc.) is NOT retried."""
    refresh_calls: list[int] = []

    def provider() -> str:
        refresh_calls.append(1)
        return "Bearer x"

    err = RuntimeError("rate limited 429")
    lm = _ScriptedLM(script=[err], token_provider=provider)
    with pytest.raises(RuntimeError, match="rate limited"):
        lm("hi")
    assert lm.call_count == 1
    assert refresh_calls == []  # provider was NOT called


def test_double_auth_error_propagates_after_one_retry() -> None:
    """If the refresh-then-retry also returns 401, surface the error — no loop."""
    err1 = _FakeAuthError("401 expired")
    err2 = _FakeAuthError("401 still expired")
    refresh_calls: list[int] = []

    def provider() -> str:
        refresh_calls.append(1)
        return "Bearer fresh"

    lm = _ScriptedLM(script=[err1, err2], token_provider=provider)
    with pytest.raises(Exception) as exc_info:
        lm("hi")
    assert "AuthenticationError" in type(exc_info.value).__name__
    assert lm.call_count == 2
    assert len(refresh_calls) == 1


def test_no_token_provider_propagates_immediately() -> None:
    """When no provider is set, behave like plain dspy.LM (no retry)."""
    err = _FakeAuthError("401 expired")
    lm = _ScriptedLM(script=[err], token_provider=None)
    with pytest.raises(Exception) as exc_info:
        lm("hi")
    assert "AuthenticationError" in type(exc_info.value).__name__
    assert lm.call_count == 1


def test_token_provider_failure_does_not_mask_original_error() -> None:
    """If the provider itself raises, the ORIGINAL auth error must surface."""
    err = _FakeAuthError("401 expired")

    def provider() -> str:
        raise RuntimeError("token endpoint unreachable")

    lm = _ScriptedLM(script=[err], token_provider=provider)
    with pytest.raises(Exception) as exc_info:
        lm("hi")
    msg = str(exc_info.value) + " | " + str(getattr(exc_info.value, "__cause__", ""))
    assert "401" in msg or "AuthenticationError" in type(exc_info.value).__name__ or "expired" in msg.lower()
    assert lm.call_count == 1  # we did not retry because refresh failed


def test_custom_token_header() -> None:
    """Header name is configurable (e.g. X-Api-Key for non-Azure providers)."""

    def provider() -> str:
        return "new-key"

    err = _FakeAuthError("401 expired")
    lm = _ScriptedLM(
        script=[err, ["ok"]],
        token_provider=provider,
        token_header="X-Api-Key",
        extra_headers={"X-Api-Key": "old-key"},
    )
    out = lm("hi")
    assert out == ["ok"]
    assert lm.kwargs["extra_headers"]["X-Api-Key"] == "new-key"


def test_auth_detection_recognizes_aad_message() -> None:
    """The exact litellm/Azure error string from production traces is detected."""
    msg = (
        "litellm.AuthenticationError: AzureException AuthenticationError - "
        "Error code: 401 - {'Message': 'User Aad Token is expired.', "
        "'Source': 'GENERAL', 'error_code': 'CUSTOMER_UNAUTHORIZED'}"
    )
    err = _FakeAuthError(msg)
    refresh_calls: list[int] = []

    def provider() -> str:
        refresh_calls.append(1)
        return "Bearer fresh"

    lm = _ScriptedLM(script=[err, ["ok"]], token_provider=provider)
    assert lm("hi") == ["ok"]
    assert refresh_calls == [1]


def test_auth_detection_ignores_unrelated_401_in_message() -> None:
    """A '401' substring in a non-AuthenticationError type is NOT treated as auth."""
    err = RuntimeError("Connection failed after 401 milliseconds")
    refresh_calls: list[int] = []

    def provider() -> str:
        refresh_calls.append(1)
        return "x"

    lm = _ScriptedLM(script=[err], token_provider=provider)
    with pytest.raises(RuntimeError):
        lm("hi")
    assert refresh_calls == []  # not detected as auth — class name didn't match


def test_copy_preserves_wrapper_and_refresh_capability() -> None:
    """REGRESSION: lm.copy(reasoning_effort=...) must NOT strip the auth-refresh wrapper.

    The runtime calls ``lm.copy(reasoning_effort=...)`` per attempt to clone
    the LM without mutating the shared instance. The wrapper-preserving
    behaviour is now provided by `dspy.LM.copy` itself (deepcopy), but we
    keep this regression test to guard against accidental loss.
    """
    refresh_calls: list[int] = []

    def provider() -> str:
        refresh_calls.append(1)
        return f"Bearer fresh-{len(refresh_calls)}"

    err = _FakeAuthError("AuthException - Error code: 401 - User Aad Token is expired")
    lm = _ScriptedLM(
        script=[err, ["recovered"]],
        token_provider=provider,
        extra_headers={"Authorization": "Bearer stale"},
    )
    cloned = lm.copy(temperature=1.0)
    assert isinstance(cloned, _RefreshingLM), \
        f"copy() returned {type(cloned).__name__}, expected _RefreshingLM"
    # Cloned wrapper carries the same provider + header config
    assert cloned._token_provider is provider
    assert cloned._token_header == "Authorization"
    # And the cloned instance is a distinct object
    assert cloned is not lm


def test_max_tokens_mirrored_for_reasoning_models() -> None:
    """REGRESSION: dspy 3.2 stores cap as `max_completion_tokens` for
    reasoning models (lm.py L77) but its truncation log path indexes
    `self.kwargs['max_tokens']` (L301), raising `KeyError('max_tokens')`
    on every truncated response. `_RefreshingLM` mirrors the value into
    both keys so dspy's broken format string finds it.
    """
    lm = _RefreshingLM(
        "openai/gpt-5",
        api_key="dummy",
        cache=False,
        temperature=1.0,
        max_tokens=16000,
    )
    assert lm.kwargs["max_tokens"] == 16000
    assert lm.kwargs["max_completion_tokens"] == 16000

    cloned = lm.copy(temperature=1.0)
    assert cloned.kwargs["max_tokens"] == 16000
    assert cloned.kwargs["max_completion_tokens"] == 16000


def test_max_tokens_not_mirrored_for_non_reasoning_models() -> None:
    """For non-reasoning models dspy already stores `max_tokens` directly
    (lm.py L81) so we MUST NOT clobber or duplicate it.
    """
    lm = _RefreshingLM(
        "openai/gpt-4o",
        api_key="dummy",
        cache=False,
        temperature=0.0,
        max_tokens=4000,
    )
    assert lm.kwargs["max_tokens"] == 4000
    assert "max_completion_tokens" not in lm.kwargs


def test_is_dspy_baselm_subclass() -> None:
    """REGRESSION: `_RefreshingLM` must subclass `dspy.BaseLM` so that
    `dspy.Predict._forward_preprocess` accepts it as the configured LM.

    Pre-fix bug: composition-only wrapper failed dspy 3.x's hard
    `isinstance(lm, dspy.BaseLM)` check, breaking the worker-side
    `predict()` helper for any sub_lm built via `FabricLM`.
    """
    import dspy

    lm = _RefreshingLM("openai/gpt-4o", api_key="dummy", cache=False)
    assert isinstance(lm, dspy.BaseLM)
    assert isinstance(lm, dspy.LM)


def test_dspy_predict_accepts_refreshing_lm() -> None:
    """REGRESSION: `dspy.Predict(...)` must accept `_RefreshingLM` as the LM
    inside `dspy.context(...)`. This is the path the worker `predict()`
    helper takes when the agent calls `await predict(signature, ...)`.

    Verifies that the isinstance check inside `Predict._forward_preprocess`
    does NOT raise `ValueError("LM must be an instance of dspy.BaseLM, ...")`.
    """
    import dspy

    lm = _RefreshingLM("openai/gpt-4o", api_key="dummy", cache=False)
    pr = dspy.Predict("english -> french")

    with dspy.context(lm=lm):
        try:
            pr(english="Hello")
        except ValueError as exc:
            if "BaseLM" in str(exc):
                raise AssertionError(
                    f"_RefreshingLM rejected by dspy.Predict isinstance check: {exc}"
                )
        except Exception:
            # Other failures (no real backend, parse errors) are out of scope.
            pass
