"""Period coverage and claim checks for semantic-model runs.

The shapes here are the ones that misled real runs on unseen models: a 9-day
last month, forward-dated rows years ahead, a January-to-May "year", and a
share whose year filter was silently dropped. Two shapes must NOT be flagged:
a weekday-only business and a model with one row per month. sempy is faked.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from fabric_rlm import SemanticModel, semantic_model_checks
from fabric_rlm.semantic_checks import (
    claim_expression,
    discover_date_column,
    period_bounds,
    period_coverage,
)

TODAY = dt.date(2026, 9, 24)


def days(start, end, weekdays_only=False, monthly=False):
    out, d = [], start
    while d <= end:
        if monthly:
            if d.day == 1:
                out.append(d)
        elif not weekdays_only or d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


class FakeModel:
    def __init__(self, dates, *, columns=None, relationships=None, measures=None, values=None):
        self.dates = dates
        self._columns = columns or [
            {"Table Name": "Date", "Column Name": "Date", "Data Type": "DateTime"},
            {"Table Name": "Sales", "Column Name": "Order Date", "Data Type": "DateTime"},
        ]
        self._relationships = relationships if relationships is not None else [
            {"From Table": "Sales", "From Column": "Order Date", "To Table": "Date", "To Column": "Date"}
        ]
        self._measures = measures or [{"Measure Name": "Total Sales", "Measure Expression": "SUM(Sales[Amount])"}]
        self.values = values or {}
        self.queries: list[str] = []

    def columns(self):
        return pd.DataFrame(self._columns)

    def relationships(self):
        return pd.DataFrame(self._relationships)

    def measures(self):
        return pd.DataFrame(self._measures)

    def dax(self, query):
        self.queries.append(query)
        if "ADDCOLUMNS(VALUES(" in query:
            return pd.DataFrame({"[Date]": [pd.Timestamp(d) for d in self.dates], "[v]": [1.0] * len(self.dates)})
        if query.startswith('EVALUATE ROW("v", '):
            expr = query[len('EVALUATE ROW("v", '):-1]
            if expr not in self.values:
                raise KeyError(expr)
            return pd.DataFrame({"[v]": [self.values[expr]]})
        raise AssertionError(f"unexpected query {query}")


def status_map(cov):
    return {p["period"]: p["status"] for p in cov.periods}


def test_a_nine_day_last_month_is_partial_and_the_month_before_is_the_headline():
    model = FakeModel(days(dt.date(2025, 9, 1), dt.date(2026, 8, 9)))
    cov = period_coverage(model, "Total Sales", today=TODAY)
    s = status_map(cov)
    assert s["2026-08"] == "partial" and s["2026-07"] == "complete" and s["2026-09"] == "in progress"
    assert cov.latest_complete == "2026-07" and cov.latest_with_data == "2026-08"
    assert "partial" in repr(cov) and "'Date'[Date]" in repr(cov)


def test_a_weekday_only_business_is_not_partial():
    model = FakeModel(days(dt.date(2025, 9, 1), dt.date(2026, 8, 31), weekdays_only=True))
    cov = period_coverage(model, "Total Sales", today=TODAY)
    assert cov.latest_complete == "2026-08"
    assert all(p["status"] in {"complete", "unknown", "in progress"} for p in cov.periods)


def test_one_row_per_month_is_complete_by_month_and_a_five_month_year_is_partial():
    model = FakeModel(days(dt.date(2016, 1, 1), dt.date(2018, 5, 1), monthly=True))
    by_month = period_coverage(model, "Total Sales", today=TODAY)
    assert by_month.latest_complete == "2018-05"
    by_year = period_coverage(model, "Total Sales", grain="year", today=TODAY)
    s = status_map(by_year)
    assert s["2018"] == "partial" and s["2017"] == "complete" and by_year.latest_complete == "2017"


def test_forward_dated_rows_are_future_not_current():
    model = FakeModel(days(dt.date(2025, 1, 1), dt.date(2030, 12, 31)))
    cov = period_coverage(model, "Total Sales", today=TODAY)
    s = status_map(cov)
    assert s["2030-12"] == "future" and s["2026-09"] == "in progress"
    assert cov.latest_complete == "2026-08" and cov.future_periods_with_data > 40
    assert "forward-dated" in repr(cov)


def test_a_given_as_of_date_moves_the_boundary():
    model = FakeModel(days(dt.date(2025, 1, 1), dt.date(2030, 12, 31)))
    cov = period_coverage(model, "Total Sales", as_of="2026-05-12", today=TODAY)
    assert cov.latest_complete == "2026-04" and status_map(cov)["2026-05"] == "in progress"


def test_the_date_column_is_the_one_facts_filter_through():
    col, why = discover_date_column(FakeModel([]))
    assert col == "'Date'[Date]" and "relationship" in why
    only_fact = FakeModel([], columns=[{"Table Name": "Sales", "Column Name": "Order Date", "Data Type": "DateTime"}],
                          relationships=[])
    assert discover_date_column(only_fact)[0] == "'Sales'[Order Date]"


def test_period_labels():
    assert period_bounds("2026-07")[:2] == (dt.date(2026, 7, 1), dt.date(2026, 8, 1))
    assert period_bounds("2026-Q4")[:2] == (dt.date(2026, 10, 1), dt.date(2027, 1, 1))
    assert period_bounds("2017")[2] == "year"
    with pytest.raises(ValueError):
        period_bounds("last month")


def test_the_validator_sends_back_a_partial_headline_and_accepts_it_labelled_period_to_date():
    model = FakeModel(days(dt.date(2025, 9, 1), dt.date(2026, 8, 9)))
    checks = semantic_model_checks(model, require_claims=False, today=TODAY)
    bad = {"headline_measure": "Total Sales", "headline_period": "2026-08", "period_to_date": False}
    with pytest.raises(AssertionError, match="2026-08 is partial.*latest complete month is 2026-07"):
        checks(bad)
    checks({**bad, "period_to_date": True})
    checks({**bad, "headline_period": "2026-07"})


def test_the_validator_sends_back_a_future_headline_even_when_labelled_period_to_date():
    model = FakeModel(days(dt.date(2025, 1, 1), dt.date(2030, 12, 31)))
    checks = semantic_model_checks(model, require_claims=False, today=TODAY)
    with pytest.raises(AssertionError, match="2030-12 is future"):
        checks({"headline_measure": "Total Sales", "headline_period": "2030-12", "period_to_date": True})


def test_claims_are_recomputed_with_the_harness_dax():
    share = {"value": 33.87, "numerator": {"aggregate": "sum", "column": "'Trip Purpose Statistics'[Visits]",
                                           "filters": {"'Trip Purpose Statistics'[New or Repeat Visitor]": "First Time Visitor"}},
             "denominator": {"aggregate": "sum", "column": "'Trip Purpose Statistics'[Visits]"},
             "filters": {"Islands[Island Name]": "O'ahu"}, "period": "2017"}
    expr = claim_expression(share, "'Months'[Month]")
    assert "'Islands'[Island Name] = \"O'ahu\"" in expr
    assert "FILTER(ALL('Months'[Month]), 'Months'[Month] >= DATE(2017,1,1) && 'Months'[Month] < DATE(2018,1,1))" in expr
    assert expr.count("CALCULATE(") == 2 and expr.startswith("DIVIDE(")
    model = FakeModel(days(dt.date(2016, 1, 1), dt.date(2018, 5, 1), monthly=True),
                      columns=[{"Table Name": "Months", "Column Name": "Month", "Data Type": "DateTime"}], relationships=[],
                      values={expr: 0.3351})
    checks = semantic_model_checks(model, today=TODAY)
    base = {"headline_measure": "Total Sales", "headline_period": "2017", "period_to_date": False}
    with pytest.raises(AssertionError, match="claim 1 says 33.87 but .* returns 0.3351"):
        checks({**base, "claims": [share]})
    checks({**base, "claims": [{**share, "value": 33.51}]})      # percent of the same share passes
    checks({**base, "claims": [{**share, "value": 0.3351}]})


def test_a_claim_that_cannot_be_recomputed_is_reported_and_the_budget_stops_the_loop():
    model = FakeModel(days(dt.date(2025, 9, 1), dt.date(2026, 7, 31)))
    checks = semantic_model_checks(model, today=TODAY, max_rejections=1)
    payload = {"headline_measure": "Total Sales", "headline_period": "2026-07", "period_to_date": False,
               "claims": [{"value": 5, "aggregate": "sum", "column": "not a column"}]}
    with pytest.raises(AssertionError, match="could not be recomputed"):
        checks(payload)
    checks(payload)                                                 # budget spent: accepted and logged
    assert checks.log[-1]["check"] == "budget"


def test_semantic_model_exposes_period_coverage(monkeypatch):
    sm = SemanticModel("m", validate=False)
    fake = FakeModel(days(dt.date(2025, 9, 1), dt.date(2026, 8, 9)))
    for name in ("columns", "relationships", "measures", "dax"):
        monkeypatch.setattr(SemanticModel, name, lambda self, *a, _f=getattr(fake, name), **k: _f(*a, **k))
    cov = sm.period_coverage("Total Sales")
    assert cov.date_column == "'Date'[Date]" and cov.periods
