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
    import dspy
    from synapse.ml.fabric.service_discovery import get_fabric_env_config
    from synapse.ml.fabric.token_utils import TokenUtils

    env = get_fabric_env_config().fabric_env_config
    auth_header = TokenUtils().get_openai_auth_header()
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
    return dspy.LM(f"azure/{model}", **kwargs)


def FabricLM(model: str, **kwargs: Any) -> Any:
    """Create a DSPy LM using Fabric's built-in OpenAI endpoint.

    Smart defaults by model family:
      - Reasoning (gpt-5, o1/o3/o4 family): max_tokens=16000, no temperature.
        Pass `reasoning_effort="minimal"|"low"|"medium"|"high"` to control depth.
      - Chat (gpt-4.1, gpt-4o, etc.): temperature=1.0, max_tokens=16000.

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


register_backend("fabric/", _fabric_factory)

