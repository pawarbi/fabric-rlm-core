"""KPIs from structure: entities and their lifecycle, ratios, crossings, concentration; every definition printed."""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

from fabric_rlm.brief import brief
from fabric_rlm.data_agent_review import LakehouseExecutor, schema_from_tables
from fabric_rlm.kpis import crossings, measure_phrase_filters, parse_kpi
from fabric_rlm.sweep import LakehouseProbe

_sweep_tests = importlib.util.spec_from_file_location("test_sweep", Path(__file__).with_name("test_sweep.py"))
_module = importlib.util.module_from_spec(_sweep_tests)
_sweep_tests.loader.exec_module(_module)
FakeModel = _module.FakeModel

FIRST_MONDAY = dt.date(2023, 1, 2)
WEEKS = 104


def _shop():
    """Two years of daily sales: a permanent customer, one new customer a week who stays four weeks, five products always present; a level shift at week 81."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE sales (sale_date DATE, customer_id INTEGER, product_id INTEGER, region VARCHAR, amount DOUBLE)")
    for week in range(1, WEEKS + 1):
        level = 1000.0 if week < 81 else 1300.0
        active = [0] + [c for c in range(1, 101) if c <= week <= c + 3]
        for weekday in range(7):
            day = FIRST_MONDAY + dt.timedelta(days=(week - 1) * 7 + weekday)
            for customer in active:
                region = "North" if customer % 5 != 4 else "South"
                con.execute("INSERT INTO sales VALUES (?, ?, ?, ?, ?)", [day.isoformat(), customer, customer % 5 + 1, region, level / len(active)])
    tables = {"sales": ("sale_date", "customer_id", "product_id", "region", "amount")}
    types = {"sales": {"sale_date": "DATE", "customer_id": "INTEGER", "product_id": "INTEGER", "region": "VARCHAR", "amount": "DOUBLE"}}

    def query(sql, *, sources, timeout=None):
        relation = con.execute(sql)
        return {"columns": [d[0] for d in relation.description], "rows": relation.fetchall(), "truncated": False}

    return LakehouseProbe.from_executor(LakehouseExecutor(query, tables), schema_from_tables("lh", tables, types=types), name="Shop")


def test_a_thin_tail_moves_the_briefed_week_back_and_says_so():
    # the shop's last weeks hold 1 to 3 customers against a usual 5, so their rows fall below half the typical week: read as data still arriving, unless a week is named
    result = brief(_shop(), ["revenue"], instructions="Revenue = SUM(amount).", budget=30)
    assert result.week is not None and result.week.start == "2024-12-09"
    assert any("7 against a typical 35" in note and "say week=2024-12-23" in note for note in result.notes), result.notes
    assert "The last 2 weeks hold far fewer rows" in result.to_markdown() or "the last 2 weeks hold far fewer rows" in result.to_markdown()


def test_kpi_phrases_are_read_by_kind():
    assert parse_kpi("new customers").kind == "new" and parse_kpi("new customers").entity_words == "customers"
    churn = parse_kpi("churned resellers over 4 weeks")
    assert churn.kind == "churned" and churn.entity_words == "resellers" and churn.window == 4
    assert parse_kpi("attrition of accounts").kind == "churned" and parse_kpi("active users").kind == "active"
    ratio = parse_kpi("average order value = sales amount / order quantity")
    assert ratio.kind == "ratio" and ratio.name == "average order value" and ratio.numerator == "sales amount" and ratio.denominator == "order quantity"
    assert parse_kpi("revenue per rows").kind == "ratio"
    crossing = parse_kpi("order quantity where channel = Internet vs order quantity where channel = Reseller")
    assert crossing.kind == "crossing" and crossing.left == "order quantity where channel = Internet"
    assert parse_kpi("amount crossing 8000").kind == "crossing"
    share = parse_kpi("top 3 share of revenue by reseller")
    assert share.kind == "concentration" and share.top == 3 and share.measure_words == "revenue" and share.grouping_words == "reseller"
    assert parse_kpi("revenue by region") is None  # an ordinary metric
    assert measure_phrase_filters("order quantity where channel = Internet and color is Red") == ("order quantity", [("channel", "Internet"), ("color", "Red")])
    assert crossings({1: 5.0, 2: 6.0, 3: 9.0, 4: 12.0}, {1: 8.0, 2: 8.0, 3: 8.0, 4: 8.0}) == [(3, "left overtook right")]


def test_entities_are_ranked_by_structure_and_the_lifecycle_counts_are_exact():
    probe = _shop()
    result = brief(probe, ["revenue"], kpis=["new customers", "churned customers", "active customers"], week="2024-12-23", instructions="Revenue = SUM(amount).", budget=90)
    assert [e.column for e in result.entities] == ["customer_id", "product_id"]
    customers, products = result.entities
    assert customers.entities == 101 and customers.name == "customers" and "named like an entity" in customers.reasons and customers.score > products.score
    assert products.entities == 5 and any("recur every month" in reason for reason in products.reasons)
    assert result.entity_choice.startswith("entity customers (customer_id on sales: 101 distinct")
    new = next(m for m in result.metrics if m.kind == "new")
    churned = next(m for m in result.metrics if m.kind == "churned")
    active = next(m for m in result.metrics if m.kind == "active")
    assert new.name == "new customers" and new.definition == "customers whose first activity falls in the week (customer_id on sales)"
    assert new.weeks[-1].start == "2024-12-23" and new.target.value == 0.0 and new.weeks[49].value == 1.0  # one new customer a week until week 100
    assert churned.target.value == 1.0 and churned.weeks[49].value == 1.0 and churned.definition.startswith("customers active in the 1 week before and not in the week")
    assert active.target.value == 1.0 and active.weeks[49].value == 5.0
    growth = new.extra["growth"][-1]
    assert (growth.active, growth.new, growth.retained, growth.resurrected, growth.churned) == (1, 0, 1, 0, 1)
    mid = next(p for p in new.extra["growth"] if p.start == "2024-09-16")  # week 90: the growth chart keeps the last 26 weeks
    assert (mid.active, mid.new, mid.retained, mid.resurrected, mid.churned) == (5, 1, 4, 0, 1)
    assert new.headline.startswith("New customers came in at 0 for the week of 23 Dec 2024: -100.0% on the same week last year")  # the week before was 0 too, so no week-over-week figure
    assert result.mismatches == () and result.queries <= 90
    html = result.to_html()
    assert "<h2>Definitions</h2>" in html and "customers whose first activity falls in the week" in html and "who stayed, who joined, who came back, who left" in html
    assert "Entities: entity customers (customer_id on sales: 101 distinct" in html
    markdown = result.to_markdown()
    assert "Definition: customers whose first activity falls in the week (customer_id on sales)." in markdown


def test_an_entity_override_a_ratio_a_crossing_and_a_concentration():
    probe = _shop()
    result = brief(probe, [], kpis=["new products", "amount / rows", "amount crossing 8000", "top 1 share of amount by region", "amount where region = North vs amount where region = South"], entity="product_id", week="2024-12-23", instructions="Revenue = SUM(amount).", budget=90)
    new_products = next(m for m in result.metrics if m.kind == "new")
    assert new_products.name == "new products" and result.entity_choice.endswith("as specified") and new_products.target.value == 0.0
    ratio = next(m for m in result.metrics if m.kind == "ratio")
    assert ratio.definition == "amount divided by rows, both summed over the week" and ratio.target.value == pytest.approx(1300.0 * 7 / 7)  # one row a day for the permanent customer
    crossing = next(m for m in result.metrics if m.kind == "crossing" and m.extra["right_name"] == "8,000")
    assert crossing.extra["crossings"] == [("2024-07-15", "left")] and crossing.explanations[0].startswith("Amount overtook 8,000 in the week of 15 Jul 2024; the gap this week is +1,100.")
    regions = next(m for m in result.metrics if m.kind == "crossing" and m.extra["right_name"] != "8,000")
    assert regions.extra["crossings"] == [] and regions.explanations[0].startswith("No crossing in the last 104 weeks; amount where region = North is above amount where region = South by 9,100 this week.")
    share = next(m for m in result.metrics if m.kind == "concentration")
    assert share.name == "share of the top 1 region in sales revenue" and share.target.value == pytest.approx(1.0) and share.extra["leaders_now"] == ("North",)
    assert share.headline.startswith("Share of the top 1 region in sales revenue came in at 100% for the week of 23 Dec 2024")
    html = result.to_html()
    assert "dashed: a crossing" in html and "Share of the top 1 region in sales revenue by week" in html and "Definition: amount divided by rows" in html


def test_lifecycle_kpis_over_a_semantic_model_write_dax_per_period():
    model = FakeModel()
    result = brief(model, ["sales amount"], kpis=["new products"], budget=60)
    new = next((m for m in result.metrics if m.kind == "new"), None)
    assert new is not None and result.entities and result.entities[0].column == "ProductKey"
    assert any(q.startswith('EVALUATE ROW("entities", DISTINCTCOUNT(') for q in model.queries)
    assert any(q.startswith('EVALUATE ROW("active", CALCULATE(DISTINCTCOUNT(') and "INTERSECT(CALCULATETABLE(VALUES(" in q for q in model.queries)
    assert new.target.start == "2013-12-09" and new.target.value == 0.0
    assert result.mismatches == ()
