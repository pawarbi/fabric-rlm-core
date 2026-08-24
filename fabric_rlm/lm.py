"""Language-model backend resolution."""

from __future__ import annotations

import os
import re
from typing import Any, Callable

BackendFactory = Callable[[str], Any]

_BACKENDS: dict[str, BackendFactory] = {}

# Defaults for non-reasoning chat models (gpt-4.1, gpt-4o, etc.).
# Reasoning models (gpt-5, o1/o3/o4 family) override these in `_smart_defaults`.
#
# `timeout` bounds each REQUEST at the HTTP layer. Without it, a connection the
# provider silently drops blocks forever: the task-level timeout is checked
# between turns and a blocked read never returns to let it run. Observed in a
# five-worker batch that froze 35+ minutes with zero CPU and zero spend until
# killed. 600s is generous for a single call (reasoning models at high effort
# stream for minutes) while turning a dead connection into an ordinary error
# that the retry/turn machinery already handles. Callers override by passing
# `timeout` in the spec.
_DEFAULT_LM_KWARGS = {"temperature": 1.0, "max_tokens": 16_000, "timeout": 600}

# Mirrors dspy 3.2's reasoning-model detector, extended to cover dotted
# generations (gpt-5.1, gpt-5.2, ...). Keep in sync with
# dspy/clients/lm.py:LM.__init__ "model_pattern" regex.
_REASONING_MODEL_RE = re.compile(
    r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?"
    r"|gpt-5(?:\.\d+)?(?!-chat)(?:-.*)?)$"
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

    - Reasoning models (gpt-5, o1/o3/o4): omit `temperature` (it is a no-op or
      rejected on these models; the only API knob is `reasoning_effort`), keep
      `max_tokens=16000` (dspy's hard floor; reasoning tokens count against this
      budget), and set `reasoning_effort="medium"`.

      The effort default is deliberate. Providers disagree about what an unset
      `reasoning_effort` means: the same gpt-5 family model reached through one
      route thinks by default and through another does not, and a model that is
      not thinking fails multi-step document work badly while looking confident
      (it invents plausible values rather than reporting that it could not find
      them). Pinning the value here makes a run reproducible across routes
      instead of inheriting whatever the endpoint happens to do. Pass
      `reasoning_effort` explicitly to override; allowed values vary by model
      generation, so check the OpenAI/Azure model docs.
    - Everything else: standard chat defaults (temperature=1.0,
      max_tokens=16000).
    """
    if is_reasoning_model(model):
        return {"max_tokens": 16_000, "reasoning_effort": "medium",
                "timeout": _DEFAULT_LM_KWARGS["timeout"]}   # NO temperature key
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

    The set of hosted models changes over time (gpt-5 and gpt-4.1 were retired
    in June 2026; gpt-5.1 and gpt-5-mini are the current language models).
    Check the list before pinning a model name:
    https://learn.microsoft.com/en-us/fabric/data-science/ai-services/ai-services-overview#consumption-rate

    Smart defaults by model family:
      - Reasoning (gpt-5.x, gpt-5-mini, o1/o3/o4 family): max_tokens=16000,
        no temperature, `reasoning_effort="medium"`. The effort is pinned rather
        than left to the endpoint because provider defaults differ, and a
        reasoning model running without reasoning degrades sharply on
        multi-step work while still sounding confident. Pass `reasoning_effort`
        to override (allowed values vary by model generation; see the
        OpenAI/Azure model docs).
      - Chat (gpt-4o, gpt-5.1-chat, etc.): temperature=1.0, max_tokens=16000.

    The returned LM transparently refreshes its Azure AAD bearer token on
    401 (`AuthenticationError` from litellm) by calling Fabric's
    `TokenUtils().get_openai_auth_header()` and retrying once. Long-running
    Fabric jobs (>= ~1 hour) no longer fail mid-run when the AAD token
    expires.

    Examples
    --------
    >>> lm = FabricLM("gpt-5.1")                                      # defaults OK
    >>> lm = FabricLM("gpt-5.1", reasoning_effort="high", max_tokens=32000)
    >>> lm = FabricLM("gpt-5-mini")                                   # cheaper tier
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


def _make_refreshing_lm_class() -> type:
    """Build the `_RefreshingLM` class lazily so we can subclass dspy.LM.

    Importing dspy at module top-level is fine in production but slows
    down `import fabric_rlm.lm` when callers only need `is_reasoning_model`
    or `register_backend`. We defer the import to first use.
    """
    import dspy

    class _RefreshingLM(dspy.LM):
        """A `dspy.LM` subclass that refreshes its bearer token on auth-expiry.

        Subclasses `dspy.LM` (and therefore `dspy.BaseLM`) so it passes the
        `isinstance(lm, dspy.BaseLM)` check that `dspy.Predict` and other
        dspy primitives perform on the configured LM. This is critical for
        the worker-side `predict()` helper that wraps a sub-LM in
        `dspy.Predict` — composition-only wrappers fail that check.

        The wrapper is provider-agnostic: it works for any short-lived bearer
        token (Azure AAD, GCP IAM, AWS IAM, custom OIDC) by accepting a
        `token_provider: Callable[[], str]` and re-applying it on 401.

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
            super().__init__(model, **kwargs)
            self._token_provider = token_provider
            self._token_header = token_header
            self._mirror_max_tokens()

        def _mirror_max_tokens(self) -> None:
            # dspy 3.2 reasoning-model branch stores the cap as
            # ``max_completion_tokens`` (lm.py L77) but its truncation log
            # path indexes ``self.kwargs['max_tokens']`` (L301), raising
            # ``KeyError('max_tokens')`` whenever a long-prompt response is
            # truncated (e.g. the synthesize call in decompose_rung). Mirror
            # the value into both keys so the format string finds it.
            if "max_tokens" not in self.kwargs and "max_completion_tokens" in self.kwargs:
                self.kwargs["max_tokens"] = self.kwargs["max_completion_tokens"]

        def copy(self, **overrides: Any) -> "_RefreshingLM":
            """Return a copy with refresh capability preserved.

            ``dspy.LM.copy`` uses ``copy.deepcopy(self)`` so the resulting
            instance is the correct subclass. We re-mirror max_tokens
            because dspy may have re-set kwargs based on overrides.
            """
            new = super().copy(**overrides)
            new._mirror_max_tokens()
            return new

        def _refresh(self) -> bool:
            if self._token_provider is None:
                return False
            try:
                new_value = self._token_provider()
            except Exception:  # provider failure — let the original error surface
                return False
            hdrs = dict(self.kwargs.get("extra_headers") or {})
            hdrs[self._token_header] = new_value
            self.kwargs["extra_headers"] = hdrs
            return True

        def _do_call(self, *args: Any, **kwargs: Any) -> Any:
            """Hook for tests and subclasses. Production path delegates to dspy.LM.__call__."""
            return dspy.LM.__call__(self, *args, **kwargs)

        # Wrap __call__, forward, and aforward so refresh fires regardless
        # of which entry point is used:
        #   - `lm("...")` → __call__ (used by tests, runtime, worker `predict()`
        #     fallbacks)
        #   - `dspy.Predict(...)(...)` → forward (sync) or aforward (async)
        # Wrapping all three keeps the refresh-on-401 behaviour reachable
        # from every call site.
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            try:
                return self._do_call(*args, **kwargs)
            except Exception as exc:
                if not _is_auth_expired(exc):
                    raise
                if not self._refresh():
                    raise
                return self._do_call(*args, **kwargs)

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            try:
                return super().forward(*args, **kwargs)
            except Exception as exc:
                if not _is_auth_expired(exc):
                    raise
                if not self._refresh():
                    raise
                return super().forward(*args, **kwargs)

        async def aforward(self, *args: Any, **kwargs: Any) -> Any:
            try:
                return await super().aforward(*args, **kwargs)
            except Exception as exc:
                if not _is_auth_expired(exc):
                    raise
                if not self._refresh():
                    raise
                return await super().aforward(*args, **kwargs)

    return _RefreshingLM


# Cache the class after first construction so isinstance checks remain stable.
_REFRESHING_LM_CLS: type | None = None


def _get_refreshing_lm_cls() -> type:
    global _REFRESHING_LM_CLS
    if _REFRESHING_LM_CLS is None:
        _REFRESHING_LM_CLS = _make_refreshing_lm_class()
    return _REFRESHING_LM_CLS


def _RefreshingLM(*args: Any, **kwargs: Any) -> Any:
    """Factory that returns a `_RefreshingLM` instance.

    Kept as a callable (not the class itself) so the dspy import stays
    lazy. ``isinstance(x, _RefreshingLM)`` is not supported; use
    ``isinstance(x, _get_refreshing_lm_cls())`` if needed (rare).
    """
    cls = _get_refreshing_lm_cls()
    return cls(*args, **kwargs)


register_backend("fabric/", _fabric_factory)


def _try_register_openai_models_for_fabric() -> None:
    """Alias bare OpenAI model names to the Fabric backend when in Fabric.

    Inside Fabric, ``synapse.ml.fabric`` is importable. When that's true, we
    register the common OpenAI model-family prefixes (``gpt-``, ``o1``,
    ``o3``, ``o4``) to route through ``_fabric_factory`` so that string
    specs like ``sub_lm="gpt-4.1"`` resolve to a Fabric-authenticated LM
    in the worker subprocess (which cannot inherit a live ``FabricLM``
    instance from the parent).

    Outside Fabric (local dev, CI, etc.) ``synapse.ml.fabric`` is not
    importable and these aliases are never registered, so bare ``"gpt-4o"``
    spec strings continue to use ``dspy.LM(...)`` with whatever
    ``OPENAI_API_KEY`` the developer has configured.
    """
    try:
        import synapse.ml.fabric  # noqa: F401
    except ImportError:
        return
    # Idempotent: register_backend overwrites a duplicate prefix.
    register_backend("gpt-", _fabric_factory)
    register_backend("o1", _fabric_factory)
    register_backend("o3", _fabric_factory)
    register_backend("o4", _fabric_factory)


_try_register_openai_models_for_fabric()

