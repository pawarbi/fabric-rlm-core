import pytest

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

