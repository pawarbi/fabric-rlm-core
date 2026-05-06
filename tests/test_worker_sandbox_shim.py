"""Tests for the ``sandbox`` module shim exposed inside the worker namespace.

Background: traces from gpt-5 SSB runs show the model frequently emits code like::

    from sandbox import SUBMIT
    SUBMIT(answer="42")

This is a reasonable convention (other agentic-Python harnesses expose a module
named ``sandbox``), but our worker only injects names into the eval namespace.
The import therefore raises ``ModuleNotFoundError: No module named 'sandbox'``,
costing a turn (and on gpt-5 specifically, ~7.7% of turns).

The fix exposes a synthetic ``sandbox`` module via ``sys.modules`` that carries
the same names that ``_install_runtime_api`` puts into ``_namespace``.
"""

from __future__ import annotations

import sys

import pytest

from fabric_rlm import _worker


def setup_function(_func) -> None:
    """Ensure each test starts from a fresh runtime install."""
    _worker._install_runtime_api()


def test_sandbox_module_registered_in_sys_modules() -> None:
    assert "sandbox" in sys.modules, (
        "sandbox shim must be registered in sys.modules so `import sandbox` works"
    )


def test_sandbox_exposes_SUBMIT() -> None:
    sandbox = sys.modules["sandbox"]
    assert getattr(sandbox, "SUBMIT", None) is _worker.SUBMIT


def test_sandbox_exposes_predict_and_helpers() -> None:
    sandbox = sys.modules["sandbox"]
    for name in ("predict", "predict_sync", "load_skill", "activate_skill", "list_skills", "File"):
        assert hasattr(sandbox, name), f"sandbox shim missing public name: {name}"


def test_sandbox_module_has_dunder_name() -> None:
    sandbox = sys.modules["sandbox"]
    assert sandbox.__name__ == "sandbox"


def test_user_code_can_import_SUBMIT_from_sandbox() -> None:
    code = "from sandbox import SUBMIT\nSUBMIT(answer='ok')"
    result = _worker._execute(code)
    assert result["ok"] is True, f"expected ok, got: {result}"
    assert result["submitted"] is True
    assert result["submit_payload"] == {"answer": "ok"}


def test_user_code_can_import_sandbox_as_module() -> None:
    code = "import sandbox\nsandbox.SUBMIT(answer='alpha')"
    result = _worker._execute(code)
    assert result["ok"] is True, f"expected ok, got: {result}"
    assert result["submit_payload"] == {"answer": "alpha"}


def test_user_code_can_import_multiple_names_from_sandbox() -> None:
    code = (
        "from sandbox import SUBMIT, predict, File\n"
        "assert callable(SUBMIT)\n"
        "assert callable(predict)\n"
        "assert File is not None\n"
        "SUBMIT(answer='multi')"
    )
    result = _worker._execute(code)
    assert result["ok"] is True, f"expected ok, got: {result}"
    assert result["submit_payload"] == {"answer": "multi"}


def test_sandbox_shim_survives_re_install() -> None:
    """Calling _install_runtime_api again must not orphan the shim."""
    _worker._install_runtime_api()
    _worker._install_runtime_api()
    assert "sandbox" in sys.modules
    assert sys.modules["sandbox"].SUBMIT is _worker.SUBMIT


def test_sandbox_shim_picks_up_dynamic_outputs() -> None:
    """When _registered_output_fields is set, SUBMIT positional ordering changes;
    the sandbox-imported SUBMIT must use the same (live) function, not a stale snapshot.
    """
    _worker._registered_output_fields = ["first", "second"]
    try:
        code = (
            "from sandbox import SUBMIT\n"
            "SUBMIT('a', 'b')"
        )
        result = _worker._execute(code)
        assert result["ok"] is True, f"expected ok, got: {result}"
        assert result["submit_payload"] == {"first": "a", "second": "b"}
    finally:
        _worker._registered_output_fields = []


def test_sandbox_does_not_pollute_with_private_names() -> None:
    sandbox = sys.modules["sandbox"]
    public = [n for n in dir(sandbox) if not n.startswith("_")]
    # Only the curated runtime-API names should be exposed.
    expected = {"File", "SUBMIT", "predict", "predict_sync",
                "load_skill", "activate_skill", "list_skills"}
    assert set(public) == expected, (
        f"sandbox public surface drift: extra={set(public)-expected}, "
        f"missing={expected-set(public)}"
    )
