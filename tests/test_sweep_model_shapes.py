"""what_moved on the semantic-model shapes that broke it on real models.

Run over 18 models it had never seen, the sweep found nothing on most of them, and
the reasons were plumbing, not judgement: fact tables hidden behind measures, date
keys spelled "Order Date Key" or "InvoiceDateID", a calendar whose "Month" column
holds dates, a calendar stored entirely as text ("2014", "February", "Unknown"),
monthly snapshots dated on the 1st read as incomplete months, and a named fact that
was dropped without a word. These pin each one.
"""

from __future__ import annotations

from types import SimpleNamespace

from fabric_rlm.semantic_checks import _AS_OF_HINT
from fabric_rlm.source_model import SourceSchema, _date_candidates
from fabric_rlm.sweep import SemanticModelProbe, _Dax, _dax_axis, _explicit_axis, sweep


def schema(tables, types, relationships=(), measures=()):
    return SourceSchema("m", "semantic_model", {t: tuple(c) for t, c in tables.items()}, tuple(measures), tuple(relationships), types)


def joins_of(s):
    return {(a, b): (c, d) for a, b, c, d in s.relationships}


class FakeModel:
    def __init__(self, tables, columns, relationships, measures=()):
        self._meta = SimpleNamespace(tables=tables, columns=columns, relationships=relationships,
                                     measures=[{"measure_name": m, "table_name": "Fact"} for m in measures])
        self.queries = []

    def metadata(self):
        return self._meta

    def dax(self, query):
        self.queries.append(query)
        return []


def test_hidden_fact_tables_are_part_of_the_model():
    model = FakeModel(
        tables=[{"table_name": "Metrics", "hidden": True}, {"table_name": "Date", "hidden": False}, {"table_name": "LocalDateTable_9", "hidden": True}],
        columns=[{"table_name": "Metrics", "column_name": "Date", "data_type": "DateTime"}, {"table_name": "Metrics", "column_name": "Qty", "data_type": "Integer"},
                 {"table_name": "Date", "column_name": "Date", "data_type": "DateTime"}, {"table_name": "LocalDateTable_9", "column_name": "Date", "data_type": "DateTime"}],
        relationships=[{"from_table": "Metrics", "from_column": "Date", "to_table": "Date", "to_column": "Date", "active": True}],
    )
    probe = SemanticModelProbe(model)
    assert "Metrics" in probe.schema.tables and ("Metrics", "Date", "Date", "Date") in probe.schema.relationships
    assert "LocalDateTable_9" not in probe.schema.tables


def test_date_keys_with_a_space_underscore_or_id_are_date_columns():
    found = _date_candidates(["Order Date Key", "order_date_key", "InvoiceDateID", "Amount", "Customer ID"])
    assert set(found) == {"Order Date Key", "order_date_key", "InvoiceDateID"}


def test_a_date_key_joins_the_fact_to_its_calendar():
    s = schema({"Sales": ["Order Date Key", "Amount"], "Date": ["Date Key", "Date", "Year"]},
               {"Sales": {"Order Date Key": "Integer", "Amount": "Decimal"}, "Date": {"Date Key": "Integer", "Date": "Date", "Year": "Integer"}},
               [("Sales", "Order Date Key", "Date", "Date Key")])
    assert _dax_axis(s, "Sales", joins_of(s)) == {"kind": "date", "table": "Date", "column": "Date", "via": "Order Date Key"}


def test_a_month_column_holding_dates_is_a_monthly_date_axis_not_a_month_number():
    s = schema({"Spend": ["Date", "Dollars"], "Months": ["Month", "Year", "Season"]},
               {"Spend": {"Date": "Date", "Dollars": "Decimal"}, "Months": {"Month": "Date", "Year": "Integer", "Season": "Text"}},
               [("Spend", "Date", "Months", "Month")])
    assert _dax_axis(s, "Spend", joins_of(s)) == {"kind": "date", "table": "Months", "column": "Month", "via": "Date"}


def test_a_text_calendar_uses_the_month_number_and_compares_as_text():
    s = schema({"Invoice": ["InvoiceDateID", "Total"], "Date": ["DateID", "MonthNo", "Month", "Year"]},
               {"Invoice": {"InvoiceDateID": "Text", "Total": "Decimal"}, "Date": {"DateID": "Text", "MonthNo": "Text", "Month": "Text", "Year": "Text"}},
               [("Invoice", "InvoiceDateID", "Date", "DateID")])
    axis = _dax_axis(s, "Invoice", joins_of(s))
    assert axis["kind"] == "parts" and axis["month"] == "MonthNo" and axis["year"] == "Year" and axis["text"]
    fact = {"table": "Invoice", "date": axis}
    dax = _Dax(s, joins_of(s))
    assert dax._period(fact, {"year": 2014, "month": 2}) == "FILTER(ALL('Date'[Year], 'Date'[MonthNo]), 'Date'[Year] = \"2014\" && 'Date'[MonthNo] IN {\"2\", \"02\"})"
    assert dax._period(fact, {"year": 2013}) == "FILTER(ALL('Date'[Year]), 'Date'[Year] = \"2013\")"


def test_a_numeric_calendar_still_uses_treatas():
    s = schema({"Sales": ["DateKey", "Amount"], "Date": ["DateKey", "Year", "MonthNumber"]},
               {"Sales": {"DateKey": "Integer", "Amount": "Decimal"}, "Date": {"DateKey": "Integer", "Year": "Integer", "MonthNumber": "Integer"}},
               [("Sales", "DateKey", "Date", "DateKey")])
    axis = _dax_axis(s, "Sales", joins_of(s))
    assert "text" not in axis
    assert _Dax(s, joins_of(s))._period({"table": "Sales", "date": axis}, {"year": 2014, "month": 2}).startswith("TREATAS({(2014, 2)}")


def test_the_caller_can_name_the_time_axis():
    s = schema({"Fact": ["YearPeriod", "Revenue", "Booked"], "Date": ["YearPeriod", "Date"], "Other": ["Date"]},
               {"Fact": {"YearPeriod": "Text", "Revenue": "Decimal", "Booked": "Date"}, "Date": {"YearPeriod": "Text", "Date": "Date"}, "Other": {"Date": "Date"}},
               [("Fact", "YearPeriod", "Date", "YearPeriod")])
    j = joins_of(s)
    assert _explicit_axis(s, "Fact", j, "'Date'[Date]") == {"kind": "date", "table": "Date", "column": "Date", "via": "YearPeriod"}
    assert _explicit_axis(s, "Fact", j, "Fact[Booked]") == {"kind": "date", "table": "Fact", "column": "Booked"}
    assert _explicit_axis(s, "Fact", j, "'Other'[Date]") is None      # not related to the fact
    assert _explicit_axis(s, "Fact", j, "'Date'[Nope]") is None


def test_a_monthly_snapshot_can_be_told_from_an_incomplete_month():
    s = schema({"Fact": ["Date", "Amount"]}, {"Fact": {"Date": "Date", "Amount": "Decimal"}})
    query = _Dax(s, {}).days_after_first({"table": "Fact", "date": {"kind": "date", "table": "Fact", "column": "Date"}})
    assert query == "EVALUATE ROW(\"value\", COUNTROWS(FILTER(VALUES('Fact'[Date]), DAY('Fact'[Date]) > 1)))"


def test_a_named_fact_or_measure_that_does_not_exist_is_said_out_loud():
    model = FakeModel(
        tables=[{"table_name": "Sales", "hidden": False}],
        columns=[{"table_name": "Sales", "column_name": "Amount", "data_type": "Decimal"}],
        relationships=[], measures=["Total Sales"],
    )
    result = sweep(SemanticModelProbe(model), facts=["Salez"], measures=["[Total Sales]", "[Net Sales]"])
    notes = " ".join(result.notes)
    assert "'Salez' is not a table in the source" in notes and "Sales" in notes
    assert "measure [Net Sales] is not in the model" in notes and "[Total Sales] is not" not in notes
    assert "no fact table recognised" not in notes


def test_as_of_measures_written_with_underscores_are_found():
    assert _AS_OF_HINT.search("Msr_ARR_As_Of_Date") and _AS_OF_HINT.search("Data_Through") and not _AS_OF_HINT.search("Total Sales")


def test_months_from_the_current_one_on_are_set_aside(monkeypatch):
    import datetime as dt
    import fabric_rlm.sweep as sw
    monkeypatch.setattr(sw, "_today", lambda: dt.date(2026, 9, 25))
    months = [{"year": 2026, "month": 7, "n": 30}, {"year": 2026, "month": 8, "n": 31}, {"year": 2026, "month": 9, "n": 22},
              {"year": 2030, "month": 12, "n": 5}, {"year": 2027, "month": None, "n": 3}]
    kept, dropped = sw._before_today(months)
    assert [(m["year"], m["month"]) for m in kept] == [(2026, 7), (2026, 8)] and dropped == 3


def test_as_of_sets_aside_its_month_only_when_it_falls_mid_month():
    import datetime as dt
    from fabric_rlm.sweep import _as_of_boundary
    assert _as_of_boundary(dt.date(2014, 12, 7)) == dt.date(2014, 12, 7)      # data ends Dec 7: December is not compared
    assert _as_of_boundary(dt.date(2018, 5, 1)) == dt.date(2018, 6, 1)       # a monthly snapshot's label: May is complete
    assert _as_of_boundary(dt.date(2014, 12, 31)) == dt.date(2015, 1, 1)     # a month end: December is complete


def test_a_named_grouping_that_is_the_join_key_is_grouped_by_the_facts_own_key():
    from fabric_rlm.sweep import _choose_paths
    s = schema({"Sales": ["Prod", "Date", "Amount"], "Products": ["Product Name", "Category"]},
               {"Sales": {"Prod": "Text", "Date": "Date", "Amount": "Decimal"}, "Products": {"Product Name": "Text", "Category": "Text"}},
               [("Sales", "Prod", "Products", "Product Name")])
    chosen = _choose_paths(s, "Sales", joins_of(s), set(), [], ["'Products'[Product Name]", "'Products'[Category]"])
    assert [p["column"] for p in chosen] == ["Prod", "Category"]


def test_a_model_measure_over_the_facts_own_date_is_evaluated_per_day():
    s = schema({"Sales": ["Date", "Amount"]}, {"Sales": {"Date": "Date", "Amount": "Decimal"}})
    fact = {"table": "Sales", "date": {"kind": "date", "table": "Sales", "column": "Date"}, "measure": "[Total Sales]"}
    query = _Dax(s, {}).series(fact, ["[Total Sales]"])
    assert "SUMMARIZECOLUMNS('Sales'[Date]" in query and "[[" not in query and '"day"' in query


def test_a_time_of_day_key_is_not_a_date_key_and_sort_helpers_are_not_groupings():
    from fabric_rlm.sweep import _choose_paths
    assert _date_candidates(["DIM_TimeId", "InvoiceDateID"]) == ["InvoiceDateID"]
    s = schema({"Usage": ["DIM_TimeId", "DIM_CalendarKey", "Units"], "Time": ["DIM_TimeId", "TwelveHourClockTime"], "Calendar": ["DIM_CalendarKey", "Calendar Date"],
                "Segment": ["Seg", "CE Segment", "CE Segment Sort"]},
               {"Usage": {"DIM_TimeId": "Integer", "DIM_CalendarKey": "Date", "Units": "Integer"}, "Time": {"DIM_TimeId": "Integer", "TwelveHourClockTime": "DateTime"},
                "Calendar": {"DIM_CalendarKey": "Date", "Calendar Date": "Date"}, "Segment": {"Seg": "Text", "CE Segment": "Text", "CE Segment Sort": "Integer"}},
               [("Usage", "DIM_TimeId", "Time", "DIM_TimeId"), ("Usage", "DIM_CalendarKey", "Calendar", "DIM_CalendarKey")])
    axis = _dax_axis(s, "Usage", joins_of(s))
    assert axis["table"] == "Calendar"
    s2 = schema({"Usage": ["Seg", "Units"], "Segment": ["Seg", "CE Segment", "CE Segment Sort"]},
                {"Usage": {"Seg": "Text", "Units": "Integer"}, "Segment": {"Seg": "Text", "CE Segment": "Text", "CE Segment Sort": "Integer"}},
                [("Usage", "Seg", "Segment", "Seg")])
    names = [p["column"] for p in _choose_paths(s2, "Usage", joins_of(s2), set(), [], 8)]
    assert "CE Segment" in names and "CE Segment Sort" not in names


def test_a_quoted_fact_and_an_unusable_time_column_are_handled():
    model = FakeModel(
        tables=[{"table_name": "Sales", "hidden": False}],
        columns=[{"table_name": "Sales", "column_name": "Amount", "data_type": "Decimal"}, {"table_name": "Sales", "column_name": "Code", "data_type": "Text"}],
        relationships=[], measures=["Total Sales"],
    )
    result = sweep(SemanticModelProbe(model), facts=["'Sales'"], measures=["[Total Sales]"], time_column="'Sales'[Code]")
    notes = " ".join(result.notes)
    assert "is not a table" not in notes
    assert "the time column 'Sales'[Code] is not a date column" in notes and "no other time axis was found" in notes


def test_months_are_judged_by_the_governed_measures_not_by_rows():
    from fabric_rlm.sweep import _measured_months
    months = [{"year": 2023, "month": 10, "v0": 35.3, "v1": 10.6}, {"year": 2023, "month": 11, "v0": 28.8, "v1": None},
              {"year": 2024, "month": 1, "v0": None, "v1": None}, {"year": 2024, "month": 11, "v0": 0, "v1": 0.0}]
    kept, blank = _measured_months(months, ["[Sales]", "[Margin]"])
    assert [(m["year"], m["month"]) for m in kept] == [(2023, 10), (2023, 11)] and blank == 2
    assert _measured_months(months, ["Amount"]) == (months, 0)                       # a raw column: rows are the data
    assert _measured_months(months[2:], ["[Sales]", "[Margin]"]) == (months[2:], 0)  # nothing measured: leave it to the caller


def test_a_named_date_table_resolves_when_the_fact_column_also_joins_another_calendar():
    s = schema({"Activation": ["Date", "Units"], "Date_Table": ["Date", "Year"], "commondate": ["Date", "Year"]},
               {"Activation": {"Date": "DateTime", "Units": "Double"}, "Date_Table": {"Date": "DateTime", "Year": "Int64"},
                "commondate": {"Date": "DateTime", "Year": "Int64"}},
               [("Activation", "Date", "Date_Table", "Date"), ("Activation", "Date", "commondate", "Date")])
    axis = _explicit_axis(s, "Activation", joins_of(s), "'Date_Table'[Date]")
    assert axis == {"kind": "date", "table": "Date_Table", "column": "Date", "via": "Date"}


def test_a_measure_named_with_its_table_is_the_model_measure():
    from fabric_rlm.sweep import _bare_measure
    known = {"total revenue"}
    assert _bare_measure("'Sales'[Total Revenue]", known) == "[Total Revenue]"
    assert _bare_measure("Sales[Total Revenue]", known) == "[Total Revenue]"
    assert _bare_measure("Sales[Amount]", known) == "Sales[Amount]"
    assert _bare_measure("[Total Revenue]", known) == "[Total Revenue]"
