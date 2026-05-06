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

import json
import subprocess
import sys

import pytest

from fabric_rlm import _worker


_MISSING = object()


@pytest.fixture(autouse=True)
def _sandbox_shim_isolation():
    """Restore sys.modules['sandbox'] and worker globals after each test.

    Without this fixture, the shim leaks across tests and could interfere
    with any future test that asserts on `'sandbox' not in sys.modules`
    or imports a real ``sandbox`` package.
    """
    original_module = sys.modules.get("sandbox", _MISSING)
    original_output_fields = list(_worker._registered_output_fields)
    _worker._install_runtime_api()
    try:
        yield
    finally:
        if original_module is _MISSING:
            sys.modules.pop("sandbox", None)
        else:
            sys.modules["sandbox"] = original_module
        _worker._registered_output_fields = original_output_fields


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
    code = (
        "from sandbox import SUBMIT\n"
        "SUBMIT('a', 'b')"
    )
    result = _worker._execute(code)
    assert result["ok"] is True, f"expected ok, got: {result}"
    assert result["submit_payload"] == {"first": "a", "second": "b"}


def test_sandbox_does_not_pollute_with_private_names() -> None:
    sandbox = sys.modules["sandbox"]
    public = [n for n in dir(sandbox) if not n.startswith("_")]
    expected = {"File", "SUBMIT", "predict", "predict_sync",
                "load_skill", "activate_skill", "list_skills"}
    assert set(public) == expected, (
        f"sandbox public surface drift: extra={set(public)-expected}, "
        f"missing={expected-set(public)}"
    )


# ---------------------------------------------------------------------------
# Rubber-duck review follow-ups (PR #4): isolation, sync, ownership, sentinel,
# subprocess integration.
# ---------------------------------------------------------------------------


def test_sandbox_public_names_match_eval_namespace() -> None:
    """Guard against drift: every name exposed via the eval namespace
    (``_namespace`` after ``_install_runtime_api``) MUST also be on the
    sandbox module, and vice versa. Adding a runtime API name without
    updating ``_SANDBOX_PUBLIC_NAMES`` would silently recreate the original
    bug for the new helper.
    """
    _worker._install_runtime_api()
    sandbox = sys.modules["sandbox"]

    namespace_names = set(_worker._namespace)
    shim_names = set(_worker._SANDBOX_PUBLIC_NAMES)
    assert namespace_names == shim_names, (
        f"runtime API drift: in namespace but not shim={namespace_names - shim_names}; "
        f"in shim but not namespace={shim_names - namespace_names}"
    )

    public_attrs = {n for n in dir(sandbox) if not n.startswith("_")}
    assert public_attrs == namespace_names


def test_sandbox_sentinel_is_object_identity_not_bool() -> None:
    """The sentinel must be a private object so a real module that happens to
    set ``__fabric_rlm_shim__ = True`` is not misidentified as our shim and
    quietly clobbered.
    """
    sentinel = _worker._SANDBOX_SHIM_SENTINEL
    assert sentinel is not True
    assert sentinel is not False
    assert sentinel is not None
    assert type(sentinel) is object


def test_existing_real_sandbox_module_is_replaced_by_shim() -> None:
    """OWNERSHIP POLICY: the worker subprocess intentionally owns the
    ``sandbox`` module name. If a real package with the same name was
    imported before _install_runtime_api ran, _install_sandbox_shim
    replaces it. This is documented behavior; this test pins it.
    """
    import types as _types

    fake_real = _types.ModuleType("sandbox")
    fake_real.something_unrelated = 42  # type: ignore[attr-defined]
    # Note: NO __fabric_rlm_shim__ attribute => not our shim.
    sys.modules["sandbox"] = fake_real

    _worker._install_runtime_api()

    sandbox = sys.modules["sandbox"]
    assert sandbox is not fake_real, (
        "worker must own `sandbox` -- a pre-existing real module should be replaced"
    )
    assert getattr(sandbox, "__fabric_rlm_shim__", None) is _worker._SANDBOX_SHIM_SENTINEL
    assert hasattr(sandbox, "SUBMIT")
    # The unrelated attr from the fake real module is gone.
    assert not hasattr(sandbox, "something_unrelated")


def test_existing_shim_is_reused_not_recreated() -> None:
    """Idempotency: repeated _install_runtime_api calls must keep the SAME
    module object so ``import sandbox`` in user code still resolves to the
    cached object after a JSON-RPC reset.
    """
    _worker._install_runtime_api()
    first = sys.modules["sandbox"]
    _worker._install_runtime_api()
    _worker._install_runtime_api()
    second = sys.modules["sandbox"]
    assert first is second, (
        "shim module identity changed across re-installs; user code holding "
        "`import sandbox` would now reference a stale object"
    )


def test_sandbox_shim_works_in_real_worker_subprocess() -> None:
    """End-to-end: spawn the worker as ``python -m fabric_rlm._worker`` and drive
    it via legacy stdin protocol. The subprocess MUST recognize the import.
    This is the path used in production (Fabric runtime spawns the worker as
    a subprocess), so direct _execute() unit coverage is not enough.
    """
    msg = json.dumps({
        "op": "exec",
        "code": "from sandbox import SUBMIT\nSUBMIT(answer='subproc-ok')",
    })
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "fabric_rlm._worker"],
        input=msg + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"worker crashed: stderr={proc.stderr}"
    # Worker emits a single JSON envelope per request on stdout.
    line = proc.stdout.strip().splitlines()[0]
    response = json.loads(line)
    assert response.get("ok") is True, f"unexpected response: {response}"
    assert response.get("submitted") is True
    assert response.get("submit_payload") == {"answer": "subproc-ok"}


def test_sandbox_shim_idempotent_with_repeated_imports_in_user_code() -> None:
    """Stress: user code may repeatedly import sandbox across cells/turns.
    Each import must yield the same module object (Python caches in sys.modules,
    but we want to verify our shim doesn't disturb that cache)."""
    code = (
        "import sandbox as s1\n"
        "import sandbox as s2\n"
        "from sandbox import SUBMIT as sub1\n"
        "from sandbox import SUBMIT as sub2\n"
        "assert s1 is s2\n"
        "assert sub1 is sub2\n"
        "SUBMIT(answer='cached')"
    )
    result = _worker._execute(code)
    assert result["ok"] is True, f"expected ok, got: {result}"
    assert result["submit_payload"] == {"answer": "cached"}

