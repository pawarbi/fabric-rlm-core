"""What SemanticModel accepts and what it hands back.

Three things found by running real models in Fabric:

* A measure the engine types as variant comes back from SemPy as text. The
  frame prints like numbers, ``sort_values`` orders it as text, and a run
  returned a lexicographic top five with nothing raised.
* Generated code writes ``order_by`` the way pandas and SQL do: a list, a
  ``(name, direction)`` pair, a list of pairs. Only a bare string was accepted.
* SemPy's measure endpoint refuses some valid groupings (a column on a fact
  table) with a bare HTTP 400, while the DAX path answers the same request.

sempy is faked so this runs in CI, which has no Fabric.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from fabric_rlm import SemanticModel, SemanticModelQueryError
from fabric_rlm.semantic_model import (
    _normalize_order_by,
    _restore_numeric_columns,
)

COLUMNS = [
    {"Table Name": "Products", "Column Name": "Category", "Data Type": "String", "Description": ""},
    {"Table Name": "Sales", "Column Name": "Region", "Data Type": "String", "Description": ""},
    {"Table Name": "Date", "Column Name": "Year", "Data Type": "Int64", "Description": ""},
]
MEASURES = [
    {"Table Name": "Measures", "Measure Name": name, "Measure Expression": "1",
     "Measure Description": "", "Measure Display Folder": ""}
    for name in ("Total Revenue", "Late Delivery Pct", "Status Label")
]
# What SemPy returns for variant-typed measures: every value is a string.
VARIANT_RESULT = pd.DataFrame(
    {
        "Products[Category]": ["small_appliances", "health_beauty", "market_place", None],
        "[__m0]": ["97210.18000000001", "885191.1199999951", "9364.010000000004", None],
        "[__m1]": ["0.06442577030812324", "0.1030078978322971", "0", None],
    }
)


class Engine:
    def __init__(self):
        self.queries: list[str] = []
        self.result = VARIANT_RESULT
        self.measure_calls: list[dict] = []
        self.measure_error: Exception | None = None
        self.measure_result = pd.DataFrame({"Total Revenue": ["15843553.24"]})

    def evaluate_dax(self, dataset, query, **kwargs):
        self.queries.append(query)
        if "group_count" in query:
            return pd.DataFrame({"[group_count]": [4]})
        return self.result.copy()

    def evaluate_measure(self, dataset, measure, **kwargs):
        self.measure_calls.append(dict(kwargs, measure=measure))
        if self.measure_error is not None and kwargs.get("groupby_columns"):
            raise self.measure_error
        return self.measure_result.copy()


@pytest.fixture
def engine(monkeypatch):
    eng = Engine()
    fabric = types.ModuleType("sempy.fabric")
    fabric.list_tables = lambda *a, **k: pd.DataFrame([{"Name": "Products", "Description": ""}])
    fabric.list_columns = lambda *a, **k: pd.DataFrame(COLUMNS)
    fabric.list_measures = lambda *a, **k: pd.DataFrame(MEASURES)
    fabric.list_relationships = lambda *a, **k: pd.DataFrame(
        columns=["From Table", "From Column", "To Table", "To Column"]
    )
    fabric.evaluate_dax = eng.evaluate_dax
    fabric.evaluate_measure = eng.evaluate_measure
    sempy = types.ModuleType("sempy")
    sempy.fabric = fabric
    monkeypatch.setitem(sys.modules, "sempy", sempy)
    monkeypatch.setitem(sys.modules, "sempy.fabric", fabric)
    return eng


def model():
    return SemanticModel("ecommerce", validate=False)


# -- variant-typed measures ----------------------------------------------------


@pytest.mark.parametrize("normalize", [True, False])
def test_aggregate_returns_numeric_measure_columns(engine, normalize):
    frame = model().aggregate(
        ["Total Revenue", "Late Delivery Pct"],
        groupby=["Products[Category]"],
        normalize_columns=normalize,
    )
    revenue = "total_revenue" if normalize else "[Total Revenue]"
    category = "products_category" if normalize else "Products[Category]"

    assert pd.api.types.is_numeric_dtype(frame[revenue])
    top = frame.sort_values(revenue, ascending=False).iloc[0]
    assert top[category] == "health_beauty"          # text order would say small_appliances
    assert frame[revenue].sum() == pytest.approx(991765.31)
    assert pd.isna(frame[revenue].iloc[3])            # blanks stay blank


def test_dax_restores_expression_columns_and_leaves_model_columns_alone(engine):
    engine.result = pd.DataFrame(
        {
            "Customers[Zip Code]": ["02134", "10001"],   # a model column: text stays text
            "[revenue]": ["10.5", "9"],
            "[label]": ["A leads by 303", "B leads by 2"],
            "[code]": ["007", "010"],                    # leading zeros: an identifier
        }
    )
    frame = model().dax("EVALUATE ...")

    assert list(frame["Customers[Zip Code]"]) == ["02134", "10001"]
    assert frame["[revenue]"].tolist() == [10.5, 9.0]
    assert frame["[label]"].tolist() == ["A leads by 303", "B leads by 2"]
    assert frame["[code]"].tolist() == ["007", "010"]


def test_measure_returns_numeric_values(engine):
    frame = model().measure("Total Revenue")
    assert frame["Total Revenue"].iloc[0] == pytest.approx(15843553.24)


@pytest.mark.parametrize(
    "values",
    [["1,234.5", "2"], ["12abc", "3"], ["0012", "7"], [1.5, 2.5], ["", "3"], ["NaN", "1"]],
)
def test_restore_numeric_leaves_anything_doubtful_alone(values):
    frame = pd.DataFrame({"[x]": values})
    assert _restore_numeric_columns(frame, ["[x]"])["[x]"].tolist() == values


def test_restore_numeric_does_not_mutate_the_input():
    frame = pd.DataFrame({"[x]": ["1", "2"]})
    result = _restore_numeric_columns(frame, ["[x]", "[missing]"])
    assert frame["[x]"].tolist() == ["1", "2"]
    assert result["[x]"].tolist() == [1, 2]


# -- order_by shapes ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("order_by", "descending", "expected"),
    [
        ("Total Revenue", True, "ORDER BY [__m0] DESC"),
        (["Total Revenue"], True, "ORDER BY [__m0] DESC"),
        (("Total Revenue", "desc"), False, "ORDER BY [__m0] DESC"),
        ([("Total Revenue", "DESC")], False, "ORDER BY [__m0] DESC"),
        ([("Total Revenue", "asc")], True, "ORDER BY [__m0] ASC"),
        ([["Total Revenue", "ascending"]], True, "ORDER BY [__m0] ASC"),
        ({"Total Revenue": "desc"}, False, "ORDER BY [__m0] DESC"),
        ([("Products[Category]", "asc")], True, "ORDER BY 'Products'[Category] ASC"),
        (("Total Revenue", False), True, "ORDER BY [__m0] ASC"),
    ],
)
def test_order_by_accepts_the_shapes_generated_code_writes(engine, order_by, descending, expected):
    model().aggregate(
        ["Total Revenue"], groupby=["Products[Category]"], order_by=order_by, descending=descending
    )
    assert engine.queries[-1].rstrip().endswith(expected)


def test_several_sort_keys_are_refused_with_the_accepted_form(engine):
    with pytest.raises(SemanticModelQueryError) as raised:
        model().aggregate(
            ["Total Revenue", "Late Delivery Pct"],
            groupby=["Products[Category]"],
            order_by=[("Total Revenue", "desc"), ("Late Delivery Pct", "asc")],
        )
    message = str(raised.value)
    assert "one key" in message and 'order_by="Total Sales", descending=True' in message
    assert engine.queries == []


@pytest.mark.parametrize("bad", [3, [("Total Revenue", "sideways")], [3], [()]])
def test_unusable_order_by_names_the_accepted_form(bad):
    with pytest.raises(SemanticModelQueryError, match="descending=True"):
        _normalize_order_by(bad, True)


def test_unknown_order_by_name_still_lists_the_options(engine):
    with pytest.raises(SemanticModelQueryError, match="Options:"):
        model().aggregate(["Total Revenue"], groupby=["Products[Category]"], order_by=["Status Label"])


# -- measure() when the endpoint refuses ---------------------------------------


def test_measure_falls_back_to_dax_when_the_endpoint_refuses_a_grouping(engine):
    engine.measure_error = RuntimeError("400 Bad Request for url: .../internalMetrics/query")
    engine.result = pd.DataFrame({"Sales[Region]": ["South", "North"], "[__m0]": ["125", "50"]})
    sm = model()

    frame = sm.measure("Total Revenue", groupby=["Sales[Region]"], filters={"Date[Year]": [2018]})

    assert list(frame.columns) == ["Region", "Total Revenue"]
    assert frame["Total Revenue"].tolist() == [125, 50]
    assert "TREATAS({2018}, 'Date'[Year])" in engine.queries[-1]
    records = [r for r in sm.query_telemetry if r["query_type"] == "measure"]
    assert len(records) == 1 and records[0]["fallback"] == "dax" and "reason" not in records[0]


def test_measure_without_grouping_still_raises_the_endpoint_error(engine):
    engine.measure_error = RuntimeError("boom")

    def always(dataset, measure, **kwargs):
        raise RuntimeError("boom")

    sys.modules["sempy.fabric"].evaluate_measure = always
    sm = model()
    with pytest.raises(RuntimeError, match="boom"):
        sm.measure("Total Revenue")
    assert sm.query_telemetry[-1]["reason"] == "execution_error"


def test_measure_reports_both_failures_when_the_fallback_fails_too(engine):
    engine.measure_error = RuntimeError("400 Bad Request")
    with pytest.raises(SemanticModelQueryError) as raised:
        model().measure("Total Revenue", groupby=["Sales[Nope]"])
    message = str(raised.value)
    assert "400 Bad Request" in message and "aggregate(" in message and "Sales[Nope]" in message
