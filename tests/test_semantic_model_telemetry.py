"""Semantic-model telemetry can say where a number came from (issue #121).

Live on the ecommerce model, a run's read_table call left no record and its DAX
query was recorded as a fingerprint only, so a provenance check could not tell
[Total Price] from SUM(Sales[Price]). dax records now carry the measures,
columns and tables a query references, and read_table records itself. Names
only: comments and string literals (filter values, aliases) are never read.
sempy is faked.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from fabric_rlm import SemanticModel
from fabric_rlm.semantic_model import dax_references

CATALOG = {
    "measures": {"total price": "Total Price", "total revenue": "Total Revenue", "net sales": "Net Sales"},
    "columns": {
        "sales[price]": ("Sales", "Price"),
        "sales[order status]": ("Sales", "Order Status"),
        "date[year]": ("Date", "Year"),
        "customer's table[name]": ("Customer's Table", "Name"),
    },
}


def test_measures_columns_and_tables_are_told_apart_and_values_never_read():
    refs = dax_references(
        """EVALUATE ROW("v", CALCULATE([Total Price], 'Date'[Year] = 2017, Sales[Order Status] <> "canceled"))"""
    )
    assert refs["measures"] == ["Total Price"]
    assert refs["columns"] == ["Date[Year]", "Sales[Order Status]"]
    assert refs["tables"] == ["Date", "Sales"]
    assert "canceled" not in repr(refs) and refs["names_resolved"] is False


def test_a_raw_column_sum_is_not_a_measure():
    refs = dax_references("EVALUATE ROW(\"v\", SUM(Sales[Price]))")
    assert refs["measures"] == [] and refs["columns"] == ["Sales[Price]"]


def test_comments_strings_and_aliases_are_not_names():
    refs = dax_references(
        """// uses [Fake Measure] in a comment
        /* 'Ghost'[Column] */
        EVALUATE SUMMARIZECOLUMNS('dim_brands'[name], "Avg OSA", AVERAGE('fct'[osa]), "note", "[not a ref]")
        ORDER BY [Avg OSA] DESC -- and [Another Fake]"""
    )
    assert refs["measures"] == []
    assert refs["columns"] == ["dim_brands[name]", "fct[osa]"]
    assert "Ghost" not in refs["tables"]


def test_escaped_quotes_and_brackets():
    refs = dax_references("EVALUATE ROW(\"v\", COUNTROWS(FILTER('Customer''s Table', 'Customer''s Table'[Name] <> BLANK())) + [Amount]]USD])")
    assert refs["columns"] == ["Customer's Table[Name]"]
    assert "Customer's Table" in refs["tables"]
    assert refs["measures"] == ["Amount]USD"]


def test_a_measure_the_query_defines_is_not_a_model_measure_or_a_column():
    refs = dax_references(
        """DEFINE MEASURE Sales[Margin %] = DIVIDE([Total Revenue] - [Total Price], [Total Revenue])
        EVALUATE ROW("m", [Margin %])"""
    )
    assert refs["measures"] == ["Total Price", "Total Revenue"]
    assert refs["columns"] == [] and refs["other_refs"] == ["Margin %"]


def test_with_the_catalog_names_are_canonical_and_row_context_columns_are_set_apart():
    refs = dax_references(
        "EVALUATE ROW(\"v\", [total price], \"n\", COUNTROWS(FILTER(Sales, [Price] > 5)), \"y\", MAX('date'[year]))",
        CATALOG,
    )
    assert refs["names_resolved"] is True
    assert refs["measures"] == ["Total Price"]
    assert refs["other_refs"] == ["Price"]
    assert refs["columns"] == ["Date[Year]"]
    assert refs["tables"] == ["Date", "Sales"]


# -- on the handle -----------------------------------------------------------------------------------


@pytest.fixture
def fabric(monkeypatch):
    calls = {"list_measures": 0}
    module = types.ModuleType("sempy.fabric")

    def list_measures(*args, **kwargs):
        calls["list_measures"] += 1
        return pd.DataFrame([{"Table Name": "Sales", "Measure Name": "Total Price", "Measure Expression": "SUM(Sales[Price])"}])

    module.list_measures = list_measures
    module.list_columns = lambda *a, **k: pd.DataFrame(
        [{"Table Name": "Sales", "Column Name": "Price", "Data Type": "Double"},
         {"Table Name": "Date", "Column Name": "Year", "Data Type": "Int64"}])
    module.list_tables = lambda *a, **k: pd.DataFrame([{"Name": "Sales"}])
    module.evaluate_dax = lambda dataset, query, **k: pd.DataFrame({"[v]": [1.0]})
    module.read_table = lambda dataset, table, **k: pd.DataFrame({"Category": ["a", "b", "c"], "Code": [1, 2, 3]})
    sempy = types.ModuleType("sempy")
    sempy.fabric = module
    monkeypatch.setitem(sys.modules, "sempy", sempy)
    monkeypatch.setitem(sys.modules, "sempy.fabric", module)
    return module, calls


def test_dax_records_names_without_any_extra_engine_call(fabric):
    _module, calls = fabric
    model = SemanticModel("Sales Model", validate=False)
    model.dax("EVALUATE ROW(\"v\", CALCULATE([Total Price], 'Date'[Year] = 2017))")
    model.dax("EVALUATE ROW(\"v\", SUM(Sales[Price]))")
    first, second = model.query_telemetry
    assert first["query_type"] == "dax" and first["names_resolved"] is False
    assert first["measures"] == ["Total Price"] and first["columns"] == ["Date[Year]"]
    assert second["measures"] == [] and second["columns"] == ["Sales[Price]"]
    assert "query_fingerprint" in first and first["returned_rows"] == 1
    assert calls["list_measures"] == 0   # telemetry never fetches metadata


def test_once_a_name_lookup_loaded_the_catalog_dax_names_are_canonical(fabric):
    model = SemanticModel("Sales Model", validate=False)
    model._catalog_names()   # what aggregate() does before it resolves names
    model.dax("EVALUATE ROW(\"v\", [total price], \"n\", COUNTROWS(FILTER(Sales, [Price] > 5)))")
    (record,) = model.query_telemetry
    assert record["names_resolved"] is True
    assert record["measures"] == ["Total Price"] and record["other_refs"] == ["Price"]
    assert record["tables"] == ["Sales"]


def test_read_table_is_recorded(fabric):
    model = SemanticModel("Sales Model", validate=False)
    frame = model.read_table("Products", num_rows=3)
    assert len(frame) == 3
    (record,) = model.query_telemetry
    assert record == {**record, "query_type": "read_table", "table": "Products", "num_rows": 3,
                      "executed": True, "returned_rows": 3, "column_count": 2}
    assert "execution_seconds" in record


def test_a_failed_read_table_is_recorded_and_raised(fabric):
    module, _calls = fabric

    def fails(*args, **kwargs):
        raise RuntimeError("table not found")

    module.read_table = fails
    model = SemanticModel("Sales Model", validate=False)
    with pytest.raises(RuntimeError, match="table not found"):
        model.read_table("Nope")
    (record,) = model.query_telemetry
    assert record["reason"] == "execution_error" and "table not found" in record["error"]
