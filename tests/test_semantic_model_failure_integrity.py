"""Failure integrity for cold, directly bound SemanticModel inputs."""

from __future__ import annotations

import sys
from types import ModuleType

from fabric_rlm import _worker


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
    _worker._install_runtime_api()
    _worker._set_inputs(
        {
            "sales": {
                "__fabric_rlm_semantic_model__": {
                    "dataset": "Sales",
                    "workspace": None,
                }
            }
        }
    )

    result = _worker._execute(
        """
try:
    sales.dax("EVALUATE ROW(\\"value\\", [Revenue])")
except PermissionError:
    pass
SUBMIT(answer=0)
"""
    )

    assert result["submitted"] is False
    assert "SemanticModel" in result["error"]


def test_schema_that_reports_unavailable_metadata_cannot_be_followed_by_placeholder(
    monkeypatch,
):
    _install_fake_sempy(monkeypatch)
    _worker._install_runtime_api()
    _worker._set_inputs(
        {
            "sales": {
                "__fabric_rlm_semantic_model__": {
                    "dataset": "Sales",
                    "workspace": None,
                }
            }
        }
    )

    result = _worker._execute(
        """
schema = sales.schema()
assert "unavailable" in schema
SUBMIT(answer=0)
"""
    )

    assert result["submitted"] is False
    assert "SemanticModel" in result["error"]


def test_successful_cold_semantic_model_query_can_submit_its_numeric_result(
    monkeypatch,
):
    _install_fake_sempy(monkeypatch, dax_result=[[42]])
    _worker._install_runtime_api()
    _worker._set_inputs(
        {
            "sales": {
                "__fabric_rlm_semantic_model__": {
                    "dataset": "Sales",
                    "workspace": None,
                }
            }
        }
    )

    result = _worker._execute(
        """
rows = sales.dax("EVALUATE ROW(\\"value\\", [Revenue])")
SUBMIT(answer=rows[0][0])
"""
    )

    assert result["submitted"] is True
    assert result["submit_payload"] == {"answer": 42}
