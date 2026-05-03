"""Language-model backend resolution."""

from __future__ import annotations

import os
import re
from typing import Any, Callable

BackendFactory = Callable[[str], Any]

_BACKENDS: dict[str, BackendFactory] = {}

# Defaults for non-reasoning chat models (gpt-4.1, gpt-4o, etc.).
# Reasoning models (gpt-5, o1/o3/o4 family) override these in `_smart_defaults`.
_DEFAULT_LM_KWARGS = {"temperature": 1.0, "max_tokens": 16_000}

# Mirrors dspy 3.2's reasoning-model detector. Keep in sync with
# dspy/clients/lm.py:LM.__init__ "model_pattern" regex.
_REASONING_MODEL_RE = re.compile(
    r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?"
    r"|gpt-5(?!-chat)(?:-.*)?)$"
)


def is_reasoning_model(model: str) -> bool:
    """Return True if `model` is an OpenAI reasoning model (gpt-5 / o1-o4 family).

    Recognizes the same family dspy enforces 16k+ max_tokens on. Excludes
    `gpt-5-chat*` (Azure non-reasoning variant).
    """
    short = model.split("/")[-1].lower()
    return _REASONING_MODEL_RE.match(short) is not None


def _smart_defaults(model: str) -> dict[str, Any]:
    """Pick sensible defaults based on model family.

    - Reasoning models (gpt-5, o1/o3/o4): omit `temperature` (it's a no-op /
      rejected on these models — the only API knob is `reasoning_effort`),
      keep `max_tokens=16000` (dspy's hard floor; reasoning tokens count
      against this budget).
    - Everything else: standard chat defaults (temperature=1.0,
      max_tokens=16000).
    """
    if is_reasoning_model(model):
        return {"max_tokens": 16_000}   # NO temperature key
    return dict(_DEFAULT_LM_KWARGS)


def register_backend(prefix: str, factory: BackendFactory) -> None:
    """Register a model-prefix backend factory."""

    _BACKENDS[prefix] = factory


def resolve_lm(spec: Any) -> Any:
    """Resolve strings, dictionaries, dspy.LM instances, or callables to an LM."""

    if spec is None:
        raise TypeError("LM spec cannot be None")
    if callable(spec) and not isinstance(spec, (str, dict)):
        return spec
    if isinstance(spec, str):
        for prefix, factory in _BACKENDS.items():
            if spec.startswith(prefix):
                return factory(spec)
        import dspy

        return dspy.LM(spec, **_smart_defaults(spec))
    if isinstance(spec, dict):
        import dspy

        spec = dict(spec)
        model = spec.pop("model", None)
        if model is None:
            return dspy.LM(**{**_DEFAULT_LM_KWARGS, **spec})
        kwargs = {**_smart_defaults(model), **spec}
        return dspy.LM(model, **kwargs)
    raise TypeError(f"Unsupported LM spec: {type(spec).__name__}")


def _fabric_factory(model_name: str, **overrides: Any) -> Any:
    from synapse.ml.fabric.service_discovery import get_fabric_env_config
    from synapse.ml.fabric.token_utils import TokenUtils

    env = get_fabric_env_config().fabric_env_config
    token_provider = lambda: TokenUtils().get_openai_auth_header()  # noqa: E731
    auth_header = token_provider()
    base = f"{env.ml_workload_endpoint}cognitive/openai"
    model = model_name.split("/", 1)[-1]
    kwargs = {
        "api_key": "fabric-token",
        "api_base": base,
        "api_version": "2025-04-01-preview",
        "extra_headers": {"Authorization": auth_header},
        **_smart_defaults(model),
        **overrides,
    }
    # If the caller passed temperature on a reasoning model, drop it silently
    # (OpenAI rejects non-1.0 values; dspy raises on non-1.0 values; user
    # intent is "I want low randomness" which is a no-op for these models).
    if is_reasoning_model(model) and kwargs.get("temperature") not in (None, 1.0):
        kwargs.pop("temperature")
    return _RefreshingLM(f"azure/{model}", token_provider=token_provider, **kwargs)


def FabricLM(model: str, **kwargs: Any) -> Any:
    """Create a DSPy LM using Fabric's built-in OpenAI endpoint.

    Smart defaults by model family:
      - Reasoning (gpt-5, o1/o3/o4 family): max_tokens=16000, no temperature.
        Pass `reasoning_effort="minimal"|"low"|"medium"|"high"` to control depth.
      - Chat (gpt-4.1, gpt-4o, etc.): temperature=1.0, max_tokens=16000.

    The returned LM transparently refreshes its Azure AAD bearer token on
    401 (`AuthenticationError` from litellm) by calling Fabric's
    `TokenUtils().get_openai_auth_header()` and retrying once. Long-running
    Fabric jobs (>= ~1 hour) no longer fail mid-run when the AAD token
    expires.

    Examples
    --------
    >>> lm = FabricLM("gpt-5")                                        # defaults OK
    >>> lm = FabricLM("gpt-5", reasoning_effort="high", max_tokens=32000)
    >>> lm = FabricLM("gpt-4.1-mini", temperature=0.0)                # determinstic chat
    """

    return _fabric_factory(model, **kwargs)


def OpenAILM(model: str, **kwargs: Any) -> Any:
    import dspy

    api_key = kwargs.pop("api_key", os.environ.get("OPENAI_API_KEY"))
    merged = {"api_key": api_key, **_smart_defaults(model), **kwargs}
    if is_reasoning_model(model) and merged.get("temperature") not in (None, 1.0):
        merged.pop("temperature")
    return dspy.LM(f"openai/{model}", **merged)


def AnthropicLM(model: str, **kwargs: Any) -> Any:
    import dspy

    merged = {
        "api_key": kwargs.pop("api_key", os.environ.get("ANTHROPIC_API_KEY")),
        **_DEFAULT_LM_KWARGS,
        **kwargs,
    }
    return dspy.LM(f"anthropic/{model}", **merged)


def _is_auth_expired(exc: BaseException) -> bool:
    """Detect whether an exception means 'bearer token rejected, refresh needed'.

    Recognised by:
      1. Class name contains 'AuthenticationError' (litellm.AuthenticationError,
         azure.core.exceptions.ClientAuthenticationError, etc.), AND
      2. Message contains a recognised expiry/401 marker.

    The class-name guard avoids false positives like a generic RuntimeError
    that happens to mention 401 (e.g. timing values, unrelated codes).
    """
    cls = type(exc).__name__
    if "AuthenticationError" not in cls:
        return False
    msg = str(exc).lower()
    return (
        "401" in msg
        or "expired" in msg
        or "customer_unauthorized" in msg
        or "unauthorized" in msg
    )


class _RefreshingLM:
    """Wraps `dspy.LM` and retries once on bearer-token expiry.

    Provider-agnostic: works for any short-lived bearer auth (Azure AAD,
    GCP IAM, AWS IAM, custom OIDC) by accepting a `token_provider`
    callable. On `AuthenticationError`, calls `token_provider()`, replaces
    the configured header (default `Authorization`), and retries once.

    The class is intentionally not a `dspy.LM` subclass — composing rather
    than inheriting keeps the wrapper safe across dspy versions where
    `__init__` / `__call__` signatures may shift.

    Parameters
    ----------
    model
        Same as `dspy.LM(model, ...)`.
    token_provider
        Zero-arg callable returning a fresh bearer header value. When
        `None`, the wrapper degrades to plain `dspy.LM` behaviour.
    token_header
        Name of the header to update (default `"Authorization"`). Set to
        `"X-Api-Key"` etc. for non-bearer providers.
    **kwargs
        Forwarded to `dspy.LM(...)`.
    """

    def __init__(
        self,
        model: str,
        *,
        token_provider: Callable[[], str] | None = None,
        token_header: str = "Authorization",
        **kwargs: Any,
    ) -> None:
        import dspy

        self._inner = dspy.LM(model, **kwargs)
        self._token_provider = token_provider
        self._token_header = token_header

    # Forward attribute access for everything we don't own (model, kwargs,
    # history, callbacks, etc.) — keeps duck-typing with dspy.LM intact.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def kwargs(self) -> dict[str, Any]:
        return self._inner.kwargs

    def copy(self, **overrides: Any) -> "_RefreshingLM":
        """Return a new wrapper around `self._inner.copy(**overrides)`.

        The runtime calls ``lm.copy(reasoning_effort=...)`` to clone an LM
        per attempt without mutating the shared instance. If we let this
        fall through to ``__getattr__`` it would return a plain ``dspy.LM``
        and silently strip our auth-refresh capability — long Fabric runs
        would then start failing with 401 around the AAD token expiry.
        """
        new = _RefreshingLM.__new__(_RefreshingLM)
        new._inner = self._inner.copy(**overrides)
        new._token_provider = self._token_provider
        new._token_header = self._token_header
        return new

    def _refresh(self) -> bool:
        if self._token_provider is None:
            return False
        try:
            new_value = self._token_provider()
        except Exception:  # provider failure — let the original error surface
            return False
        hdrs = dict(self._inner.kwargs.get("extra_headers") or {})
        hdrs[self._token_header] = new_value
        self._inner.kwargs["extra_headers"] = hdrs
        return True

    def _do_call(self, *args: Any, **kwargs: Any) -> Any:
        """Hook for tests and subclasses. Production path delegates to dspy.LM."""
        return self._inner(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._do_call(*args, **kwargs)
        except Exception as exc:
            if not _is_auth_expired(exc):
                raise
            if not self._refresh():
                raise
            # one retry, then surface whatever happens
            return self._do_call(*args, **kwargs)


register_backend("fabric/", _fabric_factory)

