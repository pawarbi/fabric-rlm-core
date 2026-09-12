"""The Monday Morning Brief: last week in context, with the drivers, the level shifts, the patterns and what moves together."""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

from fabric_rlm.brief import brief, brief_request
from fabric_rlm.data_agent_review import LakehouseExecutor, schema_from_tables
from fabric_rlm.reports import parse_request, report
from fabric_rlm.sweep import LakehouseProbe

_sweep_tests = importlib.util.spec_from_file_location("test_sweep", Path(__file__).with_name("test_sweep.py"))
_module = importlib.util.module_from_spec(_sweep_tests)
_sweep_tests.loader.exec_module(_module)
FakeModel = _module.FakeModel

FACTORS = (1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.3)  # Monday to Sunday
SHARES = {"North": 0.6, "South": 0.4}
FIRST_MONDAY = dt.date(2023, 1, 2)
WEEKS = 104
SHIFT_WEEK = 81  # the level goes from 1000 to 1300 a day


def _weekly_lakehouse():
    """Two years of daily sales in two regions: a weekly rhythm, a level shift at week 81, and a Saturday spike in the last week."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE sales (sale_date DATE, region VARCHAR, amount DOUBLE, orders INTEGER)")
    for week in range(1, WEEKS + 1):
        level = 1000.0 if week < SHIFT_WEEK else 1300.0
        for weekday in range(7):
            day = FIRST_MONDAY + dt.timedelta(days=(week - 1) * 7 + weekday)
            for region, share in SHARES.items():
                amount = level * FACTORS[weekday] * share
                if week == WEEKS and weekday == 5 and region == "North":
                    amount *= 3
                con.execute("INSERT INTO sales VALUES (?, ?, ?, ?)", [day.isoformat(), region, amount, round(amount / 50)])
    tables = {"sales": ("sale_date", "region", "amount", "orders")}
    types = {"sales": {"sale_date": "DATE", "region": "VARCHAR", "amount": "DOUBLE", "orders": "INTEGER"}}

    def query(sql, *, sources, timeout=None):
        relation = con.execute(sql)
        return {"columns": [d[0] for d in relation.description], "rows": relation.fetchall(), "truncated": False}

    return LakehouseProbe.from_executor(LakehouseExecutor(query, tables), schema_from_tables("lh", tables, types=types), name="Shop")


INSTRUCTIONS = "Revenue = SUM(amount)."


def test_the_brief_puts_last_week_in_context_and_explains_it():
    probe = _weekly_lakehouse()
    result = brief(probe, ["revenue by region", "orders"], instructions=INSTRUCTIONS, budget=60)
    assert result.week is not None and result.week.start == "2024-12-23" and "Monday 23 December to Sunday 29 December 2024" == result.week_label
    revenue = result.metrics[0]
    assert revenue.name == "sales revenue" and revenue.groupings == ("region",) and len(revenue.weeks) == WEEKS
    c = revenue.context
    assert c["value"] == 12740.0 and c["previous"] == 10400.0 and round(c["wow_pct"], 4) == 0.225
    assert c["prior_year"] == 8000.0 and round(c["yoy_pct"], 4) == 0.5925 and c["avg4"] == 10400.0 and c["avg13"] == 10400.0
    assert c["expected"] == pytest.approx(10400.0) and "prior year" in c["expected_source"] and c["z"] > 2 and "unusual" in c["verdict"]
    assert c["rank_note"].startswith("The highest week on record")
    assert [p.start for p in revenue.change_points] == ["2024-07-15"] and revenue.change_points[0].before == 8000.0 and revenue.change_points[0].after == pytest.approx(10497.5)
    assert revenue.pattern == "Saturday carried 34% of the week against 19% usually."
    finding = revenue.finding
    assert finding is not None and finding.movement.comparison.label == "week of 16 Dec 2024 to week of 23 Dec 2024"
    assert (finding.movement.before_value, finding.movement.after_value) == (10400.0, 12740.0) and finding.best.path["column"] == "region"
    assert finding.best.concentration == "single" and finding.best.groups[0].group == "North"
    assert revenue.mix["rate"] == pytest.approx(1.0) and revenue.mix["volume"] == pytest.approx(0.0)
    assert any(text.startswith("Had North held at the week before, sales revenue would have moved +0.0% instead of +22.5%.") for text in revenue.explanations)
    assert any("Volume explains 0% of the move" in text and "the value per row 100%" in text for text in revenue.explanations)
    assert revenue.prior_year_finding is not None and revenue.prior_year_finding.movement.before_value == 8000.0
    assert revenue.headline.startswith("Sales revenue came in at 12,740 for the week of 23 Dec 2024: +22.5% on the week before, +59.2% on the same week last year, +22.5% against the 13-week average.")
    orders = result.metrics[1]
    assert orders.name == "sales orders" and orders.context["wow_pct"] > 0.1
    assert any("moved together" in line for line in result.comovement) and any("associations" in line for line in result.comovement)
    assert result.watch and any(line.startswith("Sales revenue:") for line in result.watch)
    assert result.verified and result.recomputed > 0 and result.mismatches == ()
    assert result.queries <= 60 and result.queries == result.metrics[0].sweep.queries + result.metrics[1].sweep.queries + 2
    markdown = result.to_markdown()
    assert "# Monday Morning Brief: Shop" in markdown and "Level shift the week of 15 Jul 2024: the weekly average went from 8,000 to 10,498 (+31%)." in markdown
    html = result.to_html()
    assert "Monday Morning Brief: Shop" in html and "In one look" in html and "Why it moved" in html and "Pattern within the week" in html
    assert html.count("<svg") >= 6 and "Level shift the week of 15 Jul 2024" in html and "Week over week, by region" in html and "figures recomputed, 0 mismatches" in html


def test_a_named_week_and_a_spec_mapping_are_honoured():
    probe = _weekly_lakehouse()
    result = brief(probe, [{"measure": "amount", "fact": "sales", "by": ["region"], "name": "revenue"}], week="2024-07-17", instructions=INSTRUCTIONS, budget=30)
    assert result.week is not None and result.week.start == "2024-07-15"
    metric = result.metrics[0]
    assert metric.name == "revenue" and metric.context["value"] == 10400.0 and metric.context["previous"] == 8000.0 and round(metric.context["wow_pct"], 2) == 0.30
    assert metric.change_points == () or all(p.start <= "2024-07-15" for p in metric.change_points)  # nothing after the briefed week is looked at
    assert metric.context["verdict"].startswith("very unusual") and result.watch


def test_a_brief_over_a_semantic_model_writes_daily_dax_and_copes_with_little_history():
    model = FakeModel()
    result = brief(model, ["sales amount by color"], budget=40)
    assert result.kind == "semantic_model" and result.week is not None and result.week.start == "2013-12-09"
    metric = result.metrics[0]
    assert metric.context["value"] == 210.0 and metric.context["previous"] == 1020.0 and round(metric.context["wow_pct"], 3) == -0.794
    assert metric.context["verdict"] == "not enough history to say whether this is unusual"
    assert any(q.startswith("EVALUATE SELECTCOLUMNS(SUMMARIZECOLUMNS('Date'[Date], \"n\", COUNTROWS('Sales')") for q in model.queries)
    assert metric.finding is not None and "DATE(2013,12,9)" in metric.finding.movement.query and metric.finding.best.path["column"] == "Color"
    assert result.mismatches == () and result.recomputed > 0
    assert "Monday Morning Brief: Sales model" in result.to_html()


def test_a_brief_request_names_its_metrics():
    assert brief_request("monday morning brief: revenue by region, orders") == ["revenue by region", "orders"]
    assert brief_request("weekly brief for revenue by product category and territory; order quantity") == ["revenue by product category and territory", "order quantity"]
    probe = _weekly_lakehouse()
    spec = parse_request("monday morning brief: revenue by region, orders", probe, instructions=INSTRUCTIONS)
    assert spec.kind == "brief" and spec.metrics == ("revenue by region", "orders") and spec.title == "Monday Morning Brief"
    result = report(probe, "monday morning brief: revenue by region", instructions=INSTRUCTIONS, budget=40)
    assert result.title == "Monday Morning Brief: Shop" and result.metrics[0].name == "sales revenue"
