"""SemanticModel.aggregate(): grouped analysis behind a query-size guardrail.

One run asked for Sub Product Line x Region x Customer Group x Quarter with
five measures, waited out the 300s worker timeout, retried per quarter and
waited it out again. The query was reasonable; the runtime just never said the
grain was too wide. These tests pin the fast-fail: names are checked first, a
short cardinality preflight runs next, and the expensive query only runs when
the estimate is under the limit. A rejection is a normal error, not a poisoned
handle.

sempy is faked so this runs in CI, which has no Fabric.
"""

from __future__ import annotations

import sys
import time
import types

import pandas as pd
import pytest

from fabric_rlm import (
    SemanticModel,
    SemanticModelQueryError,
    SemanticModelQueryRiskUnknown,
    SemanticModelQueryTooBroad,
)
from fabric_rlm.artifacts import decode_from_worker_wire, encode_for_worker
from fabric_rlm.prompts import _describe_value
from fabric_rlm.semantic_model import (
    DEFAULT_MAX_GROUPS,
    MAX_GROUPS_ENV,
    PREFLIGHT_TIMEOUT_ENV,
)


def _column(table, name):
    return {
        "Table Name": table,
        "Column Name": name,
        "Data Type": "String",
        "Description": "",
    }


COLUMNS = [
    _column("Products", "Line Of Business"),
    _column("Products", "Sub Product Line"),
    _column("Sold To", "Sold_To Region"),
    _column("Sold To", "Sold_To Customer Group"),
    _column("Period", "YearQuarter"),
]
MEASURE_NAMES = ("ARR $", "New $", "Upgrade $", "Churn $", "Downgrade $")
MEASURES = [
    {
        "Table Name": "Measures",
        "Measure Name": name,
        "Measure Expression": "SUM(Fact[Amount])",
        "Measure Description": "",
        "Measure Display Folder": "",
    }
    for name in MEASURE_NAMES
]

WIDE_GROUPBY = [
    "Products[Sub Product Line]",
    "Sold To[Sold_To Region]",
    "Sold To[Sold_To Customer Group]",
    "Period[YearQuarter]",
]


class FakeEngine:
    """Fake sempy.fabric: answers the preflight with a configured count and
    records every DAX query so tests can see what ran, and in what order."""

    def __init__(self):
        self.group_count = 500
        self.preflight_delay = 0.0
        self.queries: list[str] = []
        self.result = pd.DataFrame(
            {
                "Products[Line Of Business]": ["Cloud", "Devices"],
                "[__m0]": [10.0, 5.0],
                "[__m1]": [1.0, 2.0],
            }
        )

    def evaluate_dax(self, dataset, query, **kwargs):
        self.queries.append(query)
        if "group_count" in query:
            if self.preflight_delay:
                time.sleep(self.preflight_delay)
            count = self.group_count
            if callable(count):
                count = count(query)
            return pd.DataFrame({"[group_count]": [count]})
        return self.result.copy()

    @property
    def preflights(self):
        return [q for q in self.queries if "group_count" in q]

    @property
    def full_queries(self):
        return [q for q in self.queries if "group_count" not in q]


@pytest.fixture
def engine(monkeypatch):
    eng = FakeEngine()
    fabric = types.ModuleType("sempy.fabric")
    fabric.list_tables = lambda *a, **k: pd.DataFrame(
        [{"Name": "Products", "Description": ""}]
    )
    fabric.list_columns = lambda *a, **k: pd.DataFrame(COLUMNS)
    fabric.list_measures = lambda *a, **k: pd.DataFrame(MEASURES)
    fabric.list_relationships = lambda *a, **k: pd.DataFrame(
        columns=["From Table", "From Column", "To Table", "To Column"]
    )
    fabric.evaluate_dax = eng.evaluate_dax
    sempy = types.ModuleType("sempy")
    sempy.fabric = fabric
    monkeypatch.setitem(sys.modules, "sempy", sempy)
    monkeypatch.setitem(sys.modules, "sempy.fabric", fabric)
    monkeypatch.delenv(MAX_GROUPS_ENV, raising=False)
    monkeypatch.delenv(PREFLIGHT_TIMEOUT_ENV, raising=False)
    return eng


def model(**kwargs):
    return SemanticModel("ARR Model", validate=False, **kwargs)


# -- the happy path ---------------------------------------------------------


def test_small_query_runs_after_the_preflight(engine):
    result = model().aggregate(
        ["ARR $", "New $"], groupby=["Products[Line Of Business]"]
    )

    assert len(engine.preflights) == 1
    assert len(engine.full_queries) == 1
    assert engine.queries[0] in engine.preflights, "preflight runs first"
    assert result.columns.tolist() == ["products_line_of_business", "arr", "new"]
    assert result["arr"].tolist() == [10.0, 5.0]


def test_preflight_counts_groups_without_the_business_measures(engine):
    model().aggregate(
        ["ARR $", "New $"],
        groupby=["Products[Line Of Business]", "Sold To[Sold_To Region]"],
        filters={"Period[YearQuarter]": "2026/Q2"},
    )

    preflight = engine.preflights[0]
    assert "COUNTROWS(" in preflight and "SUMMARIZECOLUMNS(" in preflight
    assert "'Products'[Line Of Business]" in preflight
    assert "'Sold To'[Sold_To Region]" in preflight
    assert "TREATAS({\"2026/Q2\"}, 'Period'[YearQuarter])" in preflight
    assert "[ARR $]" not in preflight and "[New $]" not in preflight


def test_full_query_is_summarizecolumns_with_aliased_measures(engine):
    model().aggregate(
        ["ARR $", "New $"],
        groupby=["Products[Line Of Business]"],
        filters={"Period[YearQuarter]": "2026/Q2"},
    )

    query = engine.full_queries[0]
    assert query.startswith("EVALUATE")
    assert "'Products'[Line Of Business]" in query
    assert "TREATAS({\"2026/Q2\"}, 'Period'[YearQuarter])" in query
    assert '"__m0", [ARR $]' in query
    assert '"__m1", [New $]' in query
    assert "TOPN(" not in query and "ORDER BY" not in query


def test_normalize_columns_false_keeps_sempy_names_with_measure_names(engine):
    result = model().aggregate(
        ["ARR $", "New $"],
        groupby=["Products[Line Of Business]"],
        normalize_columns=False,
    )

    assert result.columns.tolist() == [
        "Products[Line Of Business]", "[ARR $]", "[New $]",
    ]


def test_success_telemetry_records_estimate_rows_and_timing(engine):
    m = model()
    m.aggregate(["ARR $", "New $"], groupby=["Products[Line Of Business]"])

    record = m.query_telemetry[-1]
    assert record["query_type"] == "aggregate"
    assert record["executed"] is True
    assert record["estimated_groups"] == 500
    assert record["returned_rows"] == 2
    assert record["groupby_count"] == 1 and record["measure_count"] == 2
    assert isinstance(record["preflight_seconds"], float)
    assert isinstance(record["execution_seconds"], float)
    assert "reason" not in record


# -- rejection ----------------------------------------------------------------


def test_large_query_is_rejected_before_the_measures_run(engine):
    engine.group_count = 50_000

    with pytest.raises(SemanticModelQueryTooBroad) as err:
        model().aggregate(list(MEASURE_NAMES), groupby=WIDE_GROUPBY[:3])

    assert engine.full_queries == [], "the expensive query must never run"
    assert len(engine.preflights) == 1
    assert err.value.estimated_groups == 50_000
    assert err.value.max_groups == DEFAULT_MAX_GROUPS


def test_rejection_message_tells_the_model_how_to_narrow(engine):
    engine.group_count = 83_000

    with pytest.raises(SemanticModelQueryTooBroad) as err:
        model().aggregate(list(MEASURE_NAMES), groupby=WIDE_GROUPBY[:3])

    text = str(err.value)
    assert "83,000" in text and "10,000" in text
    for column in WIDE_GROUPBY[:3]:
        assert column in text
    for measure in MEASURE_NAMES:
        assert measure in text
    assert "Try one of:" in text
    assert "coarser" in text.lower()
    assert "filter" in text.lower()
    assert "TOP N" in text
    assert "max_groups" not in text, "do not advertise raising the limit"


def test_the_observed_wide_pattern_fails_fast(engine):
    """Acceptance criterion: Sub Product Line x Region x Customer Group x
    Quarter with five measures fails in well under the worker timeout."""
    engine.group_count = 83_000

    started = time.monotonic()
    with pytest.raises(SemanticModelQueryTooBroad):
        model().aggregate(list(MEASURE_NAMES), groupby=WIDE_GROUPBY)
    elapsed = time.monotonic() - started

    assert elapsed < 5.0
    assert engine.full_queries == []


def test_preflight_timeout_is_treated_as_unknown_risk(engine, monkeypatch):
    monkeypatch.setenv(PREFLIGHT_TIMEOUT_ENV, "0.2")
    engine.preflight_delay = 1.5
    engine.group_count = 10

    started = time.monotonic()
    with pytest.raises(SemanticModelQueryRiskUnknown) as err:
        model().aggregate(["ARR $"], groupby=WIDE_GROUPBY)
    elapsed = time.monotonic() - started

    assert elapsed < 1.2, "must give up at the budget, not wait for the engine"
    assert engine.full_queries == []
    assert err.value.timeout_seconds == 0.2
    text = str(err.value)
    assert "could not be estimated within 0.2 seconds" in text
    assert "narrower filter" in text


def test_filters_bring_a_wide_grain_under_the_limit(engine):
    engine.group_count = lambda query: 4_000 if "TREATAS" in query else 40_000

    with pytest.raises(SemanticModelQueryTooBroad):
        model().aggregate(["ARR $"], groupby=WIDE_GROUPBY[:3])
    assert engine.full_queries == []

    result = model().aggregate(
        ["ARR $"],
        groupby=WIDE_GROUPBY[:3],
        filters={"Period[YearQuarter]": "2026/Q2"},
    )

    assert len(engine.full_queries) == 1
    assert "TREATAS({\"2026/Q2\"}, 'Period'[YearQuarter])" in engine.full_queries[0]
    assert not result.empty


def test_top_n_does_not_bypass_the_preflight(engine):
    engine.group_count = 50_000

    with pytest.raises(SemanticModelQueryTooBroad):
        model().aggregate(
            ["ARR $"], groupby=WIDE_GROUPBY[:3], order_by="ARR $", top=100
        )

    assert len(engine.preflights) == 1
    assert engine.full_queries == []


def test_rejected_query_does_not_poison_the_handle(engine):
    """try / except SemanticModelQueryTooBroad / narrower query must work."""
    m = model()
    engine.group_count = lambda query: 50_000 if "Sub Product Line" in query else 20

    with pytest.raises(SemanticModelQueryTooBroad):
        m.aggregate(list(MEASURE_NAMES), groupby=WIDE_GROUPBY[:3])

    result = m.aggregate(["ARR $", "New $"], groupby=["Products[Line Of Business]"])
    assert result["arr"].tolist() == [10.0, 5.0]

    raw = m.dax("EVALUATE 1")
    assert isinstance(raw, pd.DataFrame)
    assert not getattr(m, "_source_access_failed", False)
    reasons = [r.get("reason") for r in m.query_telemetry]
    assert reasons == ["cardinality_limit", None]


def test_rejection_telemetry_records_why_nothing_ran(engine):
    engine.group_count = 83_000
    m = model()

    with pytest.raises(SemanticModelQueryTooBroad):
        m.aggregate(list(MEASURE_NAMES), groupby=WIDE_GROUPBY[:3])

    record = m.query_telemetry[-1]
    assert record["executed"] is False
    assert record["reason"] == "cardinality_limit"
    assert record["estimated_groups"] == 83_000
    assert record["groupby_count"] == 3 and record["measure_count"] == 5
    assert record["max_groups"] == DEFAULT_MAX_GROUPS
    assert isinstance(record["preflight_seconds"], float)


def test_preflight_timeout_telemetry(engine, monkeypatch):
    monkeypatch.setenv(PREFLIGHT_TIMEOUT_ENV, "0.1")
    engine.preflight_delay = 0.8
    m = model()

    with pytest.raises(SemanticModelQueryRiskUnknown):
        m.aggregate(["ARR $"], groupby=WIDE_GROUPBY)

    record = m.query_telemetry[-1]
    assert record["executed"] is False
    assert record["reason"] == "preflight_timeout"
    assert "estimated_groups" not in record


def test_preflight_engine_errors_propagate_without_running_the_query(engine):
    def boom(dataset, query, **kwargs):
        engine.queries.append(query)
        raise ValueError("DAX error")

    engine.evaluate_dax = boom
    sys.modules["sempy.fabric"].evaluate_dax = boom

    with pytest.raises(ValueError, match="DAX error"):
        model().aggregate(["ARR $"], groupby=["Products[Line Of Business]"])

    assert len(engine.queries) == 1


def test_query_errors_are_recoverable_runtime_errors():
    assert issubclass(SemanticModelQueryError, RuntimeError)
    assert issubclass(SemanticModelQueryTooBroad, SemanticModelQueryError)
    assert issubclass(SemanticModelQueryRiskUnknown, SemanticModelQueryError)


# -- bounded DAX ----------------------------------------------------------------


def test_top_and_order_by_generate_topn_over_the_aliased_measure(engine):
    model().aggregate(
        ["ARR $", "New $"],
        groupby=["Products[Line Of Business]"],
        order_by="ARR $",
        top=100,
    )

    query = engine.full_queries[0]
    assert "TOPN(" in query
    assert "    100," in query
    assert "[__m0], DESC" in query
    assert query.rstrip().endswith("ORDER BY [__m0] DESC")


def test_order_by_can_name_a_later_measure_ascending(engine):
    model().aggregate(
        ["ARR $", "New $"],
        groupby=["Products[Line Of Business]"],
        order_by="[New $]",
        descending=False,
        top=5,
    )

    query = engine.full_queries[0]
    assert "[__m1], ASC" in query
    assert query.rstrip().endswith("ORDER BY [__m1] ASC")


def test_top_without_order_by_orders_by_the_first_measure(engine):
    model().aggregate(["New $", "ARR $"], groupby=["Products[Line Of Business]"], top=3)

    query = engine.full_queries[0]
    assert '"__m0", [New $]' in query
    assert "[__m0], DESC" in query


def test_order_by_can_name_a_groupby_column(engine):
    model().aggregate(
        ["ARR $"],
        groupby=["Products[Line Of Business]"],
        order_by="Products[Line Of Business]",
        descending=False,
    )

    query = engine.full_queries[0]
    assert "TOPN(" not in query
    assert query.rstrip().endswith("ORDER BY 'Products'[Line Of Business] ASC")


def test_order_by_must_be_a_requested_measure_or_groupby_column(engine):
    with pytest.raises(SemanticModelQueryError, match="order_by"):
        model().aggregate(
            ["ARR $"], groupby=["Products[Line Of Business]"], order_by="Churn $"
        )
    assert engine.queries == []


def test_scalar_and_list_filter_values_become_treatas_lists(engine):
    model().aggregate(
        ["ARR $"],
        groupby=["Products[Line Of Business]"],
        filters={
            "Period[YearQuarter]": "2026/Q2",
            "Sold To[Sold_To Region]": ["EMEA", "APAC"],
        },
    )

    query = engine.full_queries[0]
    assert "TREATAS({\"2026/Q2\"}, 'Period'[YearQuarter])" in query
    assert "TREATAS({\"EMEA\", \"APAC\"}, 'Sold To'[Sold_To Region])" in query


def test_string_filter_values_are_escaped(engine):
    model().aggregate(
        ["ARR $"],
        groupby=["Products[Line Of Business]"],
        filters={"Sold To[Sold_To Region]": 'Ame"ricas'},
    )

    assert 'TREATAS({"Ame""ricas"}' in engine.full_queries[0]


def test_no_groupby_skips_the_preflight(engine):
    m = model()
    result = m.aggregate("ARR $", filters={"Period[YearQuarter]": "2026/Q2"})

    assert engine.preflights == []
    assert len(engine.full_queries) == 1
    assert '"__m0", [ARR $]' in engine.full_queries[0]
    assert m.query_telemetry[-1]["estimated_groups"] == 1
    assert result is not None


def test_preflight_can_be_switched_off_explicitly(engine):
    engine.group_count = 50_000

    model().aggregate(
        ["ARR $"], groupby=["Products[Line Of Business]"], preflight=False
    )

    assert engine.preflights == []
    assert len(engine.full_queries) == 1


# -- validation before any call ---------------------------------------------------


def test_unknown_measure_names_close_matches_without_querying(engine):
    with pytest.raises(SemanticModelQueryError) as err:
        model().aggregate(["ARR"], groupby=["Products[Line Of Business]"])

    text = str(err.value)
    assert "Unknown semantic-model measure: ARR" in text
    assert "ARR $" in text
    assert engine.queries == []


def test_unknown_column_names_close_matches_without_querying(engine):
    with pytest.raises(SemanticModelQueryError) as err:
        model().aggregate(["ARR $"], groupby=["Products[ProductSegment]"])

    text = str(err.value)
    assert "Unknown semantic-model column" in text
    assert "Products[ProductSegment]" in text
    assert "Products[Sub Product Line]" in text or "Products[Line Of Business]" in text
    assert engine.queries == []


def test_unknown_filter_column_is_reported_as_a_filter(engine):
    with pytest.raises(SemanticModelQueryError, match="filter"):
        model().aggregate(["ARR $"], filters={"Period[Quarter]": "Q2"})
    assert engine.queries == []


def test_names_resolve_case_insensitively_to_the_models_spelling(engine):
    model().aggregate(
        ["arr $"],
        groupby=["products[line of business]"],
        filters={"PERIOD[yearquarter]": "2026/Q2"},
    )

    query = engine.full_queries[0]
    assert "'Products'[Line Of Business]" in query
    assert '"__m0", [ARR $]' in query
    assert "'Period'[YearQuarter]" in query


def test_malformed_column_reference_is_rejected(engine):
    with pytest.raises(SemanticModelQueryError, match=r"Table\[Column\]"):
        model().aggregate(["ARR $"], groupby=["Line Of Business"])
    assert engine.queries == []


def test_bad_arguments_are_rejected_before_any_call(engine):
    m = model()
    with pytest.raises(SemanticModelQueryError):
        m.aggregate([])
    with pytest.raises(SemanticModelQueryError, match="top"):
        m.aggregate(["ARR $"], groupby=["Products[Line Of Business]"], top=0)
    with pytest.raises(SemanticModelQueryError, match="groupby"):
        m.aggregate(["ARR $"], groupby="Products[Line Of Business]")
    with pytest.raises(SemanticModelQueryError, match="filters"):
        m.aggregate(["ARR $"], filters=["Period[YearQuarter]"])
    assert engine.queries == []
    assert all(r["reason"] == "validation" for r in m.query_telemetry)


def test_metadata_for_validation_is_fetched_once_per_handle(engine, monkeypatch):
    calls = []
    fabric = sys.modules["sempy.fabric"]
    real = fabric.list_columns
    fabric.list_columns = lambda *a, **k: (calls.append("columns"), real())[1]

    m = model()
    m.aggregate(["ARR $"], groupby=["Products[Line Of Business]"])
    m.aggregate(["New $"], groupby=["Sold To[Sold_To Region]"])

    assert calls == ["columns"]


# -- configuration ------------------------------------------------------------------


def test_max_groups_can_be_raised_per_call(engine):
    engine.group_count = 50_000

    model().aggregate(["ARR $"], groupby=WIDE_GROUPBY[:3], max_groups=60_000)

    assert len(engine.full_queries) == 1


def test_max_groups_can_be_set_on_the_handle(engine):
    engine.group_count = 50_000

    model(max_groups=60_000).aggregate(["ARR $"], groupby=WIDE_GROUPBY[:3])

    assert len(engine.full_queries) == 1


def test_max_groups_env_default_applies_when_nothing_else_is_set(engine, monkeypatch):
    engine.group_count = 50_000
    monkeypatch.setenv(MAX_GROUPS_ENV, "60000")

    model().aggregate(["ARR $"], groupby=WIDE_GROUPBY[:3])

    assert len(engine.full_queries) == 1


def test_invalid_env_limit_falls_back_to_the_default(engine, monkeypatch):
    engine.group_count = 50_000
    monkeypatch.setenv(MAX_GROUPS_ENV, "lots")

    with pytest.raises(SemanticModelQueryTooBroad) as err:
        model().aggregate(["ARR $"], groupby=WIDE_GROUPBY[:3])

    assert err.value.max_groups == DEFAULT_MAX_GROUPS


def test_handle_max_groups_must_be_positive(engine):
    with pytest.raises(ValueError, match="max_groups"):
        SemanticModel("D", validate=False, max_groups=0)
    with pytest.raises(SemanticModelQueryError, match="max_groups"):
        model().aggregate(["ARR $"], max_groups=-1)
    assert engine.queries == []


def test_handle_max_groups_survives_the_worker_wire():
    wire = encode_for_worker({"m": SemanticModel("D", validate=False, max_groups=25)})
    assert wire["m"]["__fabric_rlm_semantic_model__"]["max_groups"] == 25
    assert decode_from_worker_wire(wire)["m"].max_groups == 25


def test_wire_format_is_unchanged_when_max_groups_is_unset():
    wire = encode_for_worker({"m": SemanticModel("D", workspace="WS", validate=False)})
    assert wire["m"] == {
        "__fabric_rlm_semantic_model__": {"dataset": "D", "workspace": "WS"}
    }
    assert decode_from_worker_wire(wire)["m"].max_groups is None


# -- raw dax and the prompt -------------------------------------------------------


def test_dax_is_unchanged_and_unguarded(engine):
    m = model()
    m.dax("EVALUATE 1")

    assert engine.queries == ["EVALUATE 1"]
    assert m.query_telemetry == ()


def test_prompt_listing_prefers_aggregate_over_dax():
    line = _describe_value(SemanticModel("Sales", validate=False))

    assert ".aggregate(" in line
    assert line.index(".aggregate(") < line.index(".dax(")
    assert "size" in line
    assert len(line) < 400
