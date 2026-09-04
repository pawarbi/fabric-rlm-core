"""Fixes taken from the first live trajectory that used SemanticModel.aggregate().

Four things cost that run turns or bloated its record: the handle's cached
catalog and telemetry landing in every state snapshot, a preflight budget with
no headroom over the engine's ~8s round trip, `columns("ARR Data")` raising
TypeError, and `from fabric import SUBMIT` on the final turn.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from fabric_rlm import SemanticModel
from fabric_rlm.prompts import build_system_prompt
from fabric_rlm.semantic_model import DEFAULT_PREFLIGHT_TIMEOUT_SECONDS
from fabric_rlm.serializers import freeze, snapshot

COLUMNS = [
    {"Table Name": "Products", "Column Name": "Line Of Business",
     "Data Type": "String", "Description": ""},
    {"Table Name": "Sold To", "Column Name": "Sold_To Region",
     "Data Type": "String", "Description": ""},
]
MEASURES = [
    {"Table Name": "Measures", "Measure Name": "ARR $",
     "Measure Expression": "SUM(Fact[Amount])", "Measure Description": "",
     "Measure Display Folder": ""},
]


@pytest.fixture
def fake_sempy(monkeypatch):
    calls: list[tuple[str, tuple, dict]] = []

    def recorder(name, frame):
        def fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return frame() if callable(frame) else frame
        return fn

    def evaluate_dax(dataset, query, **kwargs):
        calls.append(("evaluate_dax", (dataset, query), kwargs))
        if "group_count" in query:
            return pd.DataFrame({"[group_count]": [5]})
        return pd.DataFrame({"Products[Line Of Business]": ["Cloud"], "[__m0]": [1.0]})

    fabric = types.ModuleType("sempy.fabric")
    fabric.list_tables = recorder("list_tables", pd.DataFrame([{"Name": "Products"}]))
    fabric.list_columns = recorder("list_columns", lambda: pd.DataFrame(COLUMNS))
    fabric.list_measures = recorder("list_measures", lambda: pd.DataFrame(MEASURES))
    fabric.list_relationships = recorder("list_relationships", pd.DataFrame())
    fabric.evaluate_dax = evaluate_dax
    sempy = types.ModuleType("sempy")
    sempy.fabric = fabric
    monkeypatch.setitem(sys.modules, "sempy", sempy)
    monkeypatch.setitem(sys.modules, "sempy.fabric", fabric)
    return calls


def test_snapshot_records_only_the_handle_identity_after_aggregate(fake_sempy):
    model = SemanticModel("ARR Model", workspace="WS", validate=False, max_groups=500)
    model.aggregate(["ARR $"], groupby=["Products[Line Of Business]"])
    assert model.query_telemetry, "precondition: telemetry was recorded"

    frozen = freeze(model)

    assert frozen == {
        "dataset": "ARR Model",
        "workspace": "WS",
        "credential_provider": None,
        "validate": False,
        "max_groups": 500,
    }
    assert "_catalog" not in frozen and "_query_telemetry" not in frozen
    state = snapshot({"business_model": model})
    assert state["business_model"] == frozen


def test_preflight_budget_leaves_headroom_over_the_engine_round_trip():
    """Live preflights took 8.7s and 8.3s for 44 and 2,418 groups."""
    assert DEFAULT_PREFLIGHT_TIMEOUT_SECONDS >= 20


def test_columns_can_be_narrowed_to_one_table(fake_sempy):
    model = SemanticModel("D", workspace="WS", validate=False)

    model.columns("ARR Data")
    model.columns()

    narrowed, everything = fake_sempy
    assert narrowed[0] == "list_columns" and narrowed[2]["table"] == "ARR Data"
    assert narrowed[2]["workspace"] == "WS"
    assert everything[0] == "list_columns" and "table" not in everything[2]


def test_prompt_says_submit_is_predefined():
    prompt = build_system_prompt(
        inline_task="answer",
        inputs={},
        inline_outputs=["answer"],
    )

    assert "SUBMIT is already defined" in prompt
    assert "never import it" in prompt
