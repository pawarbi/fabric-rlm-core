"""Failure integrity for cold, directly bound SemanticModel inputs."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

from fabric_rlm import Interpreter, SemanticModel
from fabric_rlm.interpreter import SubprocessPythonInterpreter


def _install_fake_sempy(monkeypatch, *, dax_result=None, dax_error=None):
    fabric = ModuleType("sempy.fabric")

    def evaluate_dax(dataset, query, **kwargs):
        del dataset, query, kwargs
        if dax_error is not None:
            raise dax_error
        return dax_result

    fabric.evaluate_dax = evaluate_dax
    sempy = ModuleType("sempy")
    sempy.fabric = fabric
    monkeypatch.setitem(sys.modules, "sempy", sempy)
    monkeypatch.setitem(sys.modules, "sempy.fabric", fabric)


def test_failed_semantic_model_query_cannot_be_replaced_by_numeric_placeholder(
    monkeypatch,
):
    _install_fake_sempy(
        monkeypatch,
        dax_error=PermissionError("source access denied"),
    )
    with Interpreter(timeout=5) as interpreter:
        interpreter.set_inputs(
            {"sales": SemanticModel("Sales", validate=False)}
        )
        result = interpreter.execute(
            """
try:
    sales.dax("EVALUATE ROW(\\"value\\", [Revenue])")
except Exception:
    pass
SUBMIT(answer=0)
"""
        )

    assert result.submitted is False
    assert "SemanticModel" in (result.error or "")


def test_schema_that_reports_unavailable_metadata_cannot_be_followed_by_placeholder(
    monkeypatch,
):
    _install_fake_sempy(monkeypatch)
    with Interpreter(timeout=5) as interpreter:
        interpreter.set_inputs(
            {"sales": SemanticModel("Sales", validate=False)}
        )
        result = interpreter.execute(
            """
schema = sales.schema()
assert "unavailable" in schema
SUBMIT(answer=0)
"""
        )

    assert result.submitted is False
    assert "SemanticModel" in (result.error or "")


def test_successful_cold_semantic_model_query_can_submit_its_numeric_result(
    monkeypatch,
):
    _install_fake_sempy(monkeypatch, dax_result=[[42]])
    with Interpreter(timeout=5) as interpreter:
        interpreter.set_inputs(
            {"sales": SemanticModel("Sales", validate=False)}
        )
        result = interpreter.execute(
            """
rows = sales.dax("EVALUATE ROW(\\"value\\", [Revenue])")
SUBMIT(answer=rows[0][0])
"""
        )

    assert result.submitted is True
    assert result.submit_payload == {"answer": 42}


def test_object_setattr_cannot_reset_semantic_model_failure_latch(monkeypatch):
    _install_fake_sempy(
        monkeypatch,
        dax_error=PermissionError("source access denied"),
    )

    with Interpreter(timeout=5) as interpreter:
        interpreter.set_inputs(
            {"sales": SemanticModel("Sales", validate=False)}
        )
        result = interpreter.execute(
            """
try:
    sales.dax("EVALUATE ROW(\\"value\\", [Revenue])")
except Exception:
    pass
object.__setattr__(sales, "_source_access_failed", False)
SUBMIT(answer=0)
"""
        )

    assert result.submitted is False
    assert "SemanticModel" in (result.error or "")


def test_submit_globals_cannot_clear_semantic_model_failure_latch(monkeypatch):
    _install_fake_sempy(
        monkeypatch,
        dax_error=PermissionError("source access denied"),
    )

    with Interpreter(timeout=5) as interpreter:
        interpreter.set_inputs(
            {"sales": SemanticModel("Sales", validate=False)}
        )
        result = interpreter.execute(
            """
try:
    sales.dax("EVALUATE ROW(\\"value\\", [Revenue])")
except Exception:
    pass
SUBMIT.__globals__["_bound_semantic_models"].clear()
SUBMIT(answer=0)
"""
        )

    assert result.submitted is False
    assert "SemanticModel" in (result.error or "")


def test_later_success_cannot_recover_failed_task_run(monkeypatch):
    calls = 0
    fabric = ModuleType("sempy.fabric")

    def evaluate_dax(dataset, query, **kwargs):
        nonlocal calls
        del dataset, query, kwargs
        calls += 1
        if calls == 1:
            raise PermissionError("source access denied")
        return [[42]]

    fabric.evaluate_dax = evaluate_dax
    sempy = ModuleType("sempy")
    sempy.fabric = fabric
    monkeypatch.setitem(sys.modules, "sempy", sempy)
    monkeypatch.setitem(sys.modules, "sempy.fabric", fabric)

    with Interpreter(timeout=5) as interpreter:
        interpreter.set_inputs(
            {"sales": SemanticModel("Sales", validate=False)}
        )
        first = interpreter.execute(
            """
try:
    sales.dax("EVALUATE ROW(\\"value\\", [Revenue])")
except Exception:
    pass
"""
        )
        second = interpreter.execute(
            """
rows = sales.dax("EVALUATE ROW(\\"value\\", [Revenue])")
SUBMIT(answer=rows[0][0])
"""
        )

    assert first.ok
    assert second.submitted is False
    assert "SemanticModel" in (second.error or "")


def test_fresh_input_binding_resets_semantic_model_failure_latch(monkeypatch):
    calls = 0
    fabric = ModuleType("sempy.fabric")

    def evaluate_dax(dataset, query, **kwargs):
        nonlocal calls
        del dataset, query, kwargs
        calls += 1
        if calls == 1:
            raise PermissionError("source access denied")
        return [[42]]

    fabric.evaluate_dax = evaluate_dax
    sempy = ModuleType("sempy")
    sempy.fabric = fabric
    monkeypatch.setitem(sys.modules, "sempy", sempy)
    monkeypatch.setitem(sys.modules, "sempy.fabric", fabric)

    with Interpreter(timeout=5) as interpreter:
        inputs = {"sales": SemanticModel("Sales", validate=False)}
        interpreter.set_inputs(inputs)
        interpreter.execute(
            """
try:
    sales.dax("EVALUATE ROW(\\"value\\", [Revenue])")
except Exception:
    pass
"""
        )
        interpreter.set_inputs(inputs)
        result = interpreter.execute(
            """
rows = sales.dax("EVALUATE ROW(\\"value\\", [Revenue])")
SUBMIT(answer=rows[0][0])
"""
        )

    assert result.submitted is True
    assert result.submit_payload == {"answer": 42}


def test_subprocess_submit_globals_exploit_cannot_bypass_parent_latch(
    monkeypatch,
):
    _install_fake_sempy(
        monkeypatch,
        dax_error=PermissionError("source access denied"),
    )

    with SubprocessPythonInterpreter(timeout=5) as interpreter:
        with pytest.raises(CodeInterpreterError, match="SemanticModel"):
            interpreter.execute(
                """
try:
    sales.dax("EVALUATE ROW(\\"value\\", [Revenue])")
except Exception:
    pass
SUBMIT.__globals__["_bound_semantic_models"].clear()
SUBMIT(answer=0)
""",
                {"sales": SemanticModel("Sales", validate=False)},
            )


def test_cold_semantic_model_uses_parent_notebookutils_credential(monkeypatch):
    observed_credentials = []
    fabric = ModuleType("sempy.fabric")

    def evaluate_dax(dataset, query, **kwargs):
        del dataset, query
        observed_credentials.append(kwargs.get("credential"))
        return [[42]]

    fabric.evaluate_dax = evaluate_dax
    sempy = ModuleType("sempy")
    sempy.fabric = fabric
    monkeypatch.setitem(sys.modules, "sempy", sempy)
    monkeypatch.setitem(sys.modules, "sempy.fabric", fabric)

    with Interpreter(timeout=5) as interpreter:
        interpreter.set_inputs(
            {
                "sales": SemanticModel(
                    "Sales",
                    credential_provider="notebookutils",
                    validate=False,
                )
            }
        )
        result = interpreter.execute(
            """
rows = sales.dax("EVALUATE ROW(\\"value\\", [Revenue])")
SUBMIT(answer=rows[0][0])
"""
        )

    assert result.submitted is True
    assert len(observed_credentials) == 1
    assert callable(getattr(observed_credentials[0], "get_token", None))
