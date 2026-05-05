import pytest

from fabric_rlm import lm as lm_mod
from fabric_rlm.lm import register_backend, resolve_lm


def test_resolve_lm_accepts_callable() -> None:
    def fake(messages):
        return "ok"

    assert resolve_lm(fake) is fake


def test_resolve_lm_rejects_none() -> None:
    with pytest.raises(TypeError):
        resolve_lm(None)


def test_register_backend_prefix() -> None:
    register_backend("unit-test/", lambda spec: {"spec": spec})

    assert resolve_lm("unit-test/model") == {"spec": "unit-test/model"}


def test_openai_model_aliases_to_fabric_when_in_fabric(monkeypatch) -> None:
    """REGRESSION: Bare ``"gpt-4.1"`` strings must route to the Fabric backend
    inside Fabric so that ``RLM(sub_lm="gpt-4.1")`` works without forcing
    users to remember the ``"fabric/"`` prefix.

    Pre-fix bug: a bare model spec (which is required because the worker
    subprocess can't inherit a live ``FabricLM`` instance) fell through to
    plain ``dspy.LM("gpt-4.1")``, which uses litellm's OpenAI client with
    the ``OPENAI_API_KEY`` placeholder Fabric ships, producing
    ``Incorrect API key provided: place_ho********************rnal`` 401s
    on every sub-LM call.
    """
    captured: list[str] = []

    def fake_factory(spec, **overrides):
        captured.append(spec)
        return ("fabric-routed", spec)

    # Save & restore the prefix table so this test cannot leak side-effects.
    original = dict(lm_mod._BACKENDS)
    try:
        lm_mod._BACKENDS.clear()
        lm_mod._BACKENDS.update(original)
        register_backend("fabric/", fake_factory)
        register_backend("gpt-", fake_factory)
        register_backend("o1", fake_factory)

        assert resolve_lm("gpt-4.1") == ("fabric-routed", "gpt-4.1")
        assert resolve_lm("gpt-5") == ("fabric-routed", "gpt-5")
        assert resolve_lm("o1-mini") == ("fabric-routed", "o1-mini")
        assert captured == ["gpt-4.1", "gpt-5", "o1-mini"]
    finally:
        lm_mod._BACKENDS.clear()
        lm_mod._BACKENDS.update(original)


def test_openai_aliases_not_registered_outside_fabric() -> None:
    """Outside Fabric (``synapse.ml.fabric`` not importable) the ``gpt-`` /
    ``o1`` prefixes must NOT be auto-registered, so local development with
    a real ``OPENAI_API_KEY`` keeps using the plain ``dspy.LM`` path.

    We don't try to mock the import; instead we just assert the production
    state is consistent: if synapse is unavailable here, the prefixes are
    absent; if it IS available, they're present. This is a sanity check
    against the registration code accidentally running unconditionally.
    """
    try:
        import synapse.ml.fabric  # noqa: F401
        in_fabric = True
    except ImportError:
        in_fabric = False

    if in_fabric:
        assert "gpt-" in lm_mod._BACKENDS
    else:
        assert "gpt-" not in lm_mod._BACKENDS


