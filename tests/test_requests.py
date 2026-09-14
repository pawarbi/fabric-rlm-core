"""A question in plain words becomes the report it asks for: filters on named values, a period read against the trend, and what was not understood said on the page."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from fabric_rlm.reports import parse_request, report
from fabric_rlm.series import Month, SeriesStory, analyse, expected_value, period_check
from fabric_rlm.source_model import schema_from_tables
from fabric_rlm.sweep import LakehouseProbe, Point, SemanticModelProbe

_sweep_tests = importlib.util.spec_from_file_location("test_sweep", Path(__file__).with_name("test_sweep.py"))
_module = importlib.util.module_from_spec(_sweep_tests)
_sweep_tests.loader.exec_module(_module)
_reseller_lakehouse, _saas_payments, FakeModel, INSTRUCTIONS = _module._reseller_lakehouse, _module._saas_payments, _module.FakeModel, _module.INSTRUCTIONS


def test_a_text_column_named_in_a_question_is_a_grouping_not_a_measure():
    probe = _saas_payments()
    spec = parse_request("which region to focus on. whats moving them", probe)
    assert spec.kind == "root_cause" and spec.groupings == ("region",) and spec.measures == () and spec.filters == ()
    assert "by: region" in spec.reading and not any(line.startswith("ignored") for line in spec.reading)
    result = report(probe, "which region to focus on. whats moving them", years=[2023, 2024], budget=100)
    assert result.sweep.findings and all(d.path["column"] == "region" for f in result.sweep.findings for d in f.decompositions)


def test_a_named_value_narrows_every_query_and_the_page_says_so():
    probe = _saas_payments()
    result = report(probe, "why did payments amount rise in 2024 for sector = Technology", years=[2023, 2024], budget=100)
    spec = result.spec
    assert spec.kind == "root_cause" and spec.measures == ("amount",) and [(p["column"], v) for p, v in spec.filters] == [("sector", "Technology")]
    assert "only: sector = Technology (4 payments)" in spec.reading and spec.lookups >= 1 and result.queries == result.sweep.queries + spec.lookups
    finding = result.sweep.findings[0]
    assert (finding.movement.before_value, finding.movement.after_value) == (100.0, 400.0) and finding.movement.comparison.label == "2023 to 2024"
    assert {d.path["column"] for d in finding.decompositions} == {"region"}  # the grouping the request fixed has one group left, so it is not split
    assert finding.best.groups[0].group == "Europe" and finding.best.groups[0].after_value == 350.0
    assert result.sweep.mismatches == () and result.sweep.recomputed > 0
    assert "sector = 'Technology'" in finding.movement.query and all("sector = 'Technology'" in q for q in finding.movement.verification.values())
    assert result.sweep.filter_phrase() == "sector is Technology"
    html = result.to_html()
    assert "Only rows where sector is Technology." in html and "only: sector = Technology (4 payments)" in html
    assert "Only rows where sector is Technology." in result.to_markdown()


def test_a_bare_value_is_looked_up_in_the_groupings_and_a_partial_or_unknown_one_is_explained():
    probe = _saas_payments()
    europe = parse_request("what moved for Europe", probe)
    assert [(p["column"], v) for p, v in europe.filters] == [("region", "Europe")] and "only: region = Europe (4 payments)" in europe.reading
    partial = parse_request("what moved for tech", probe)
    assert [(p["column"], v) for p, v in partial.filters] == [("sector", "Technology")] and "only: sector = Technology (matched 'tech'; 4 payments)" in partial.reading
    unknown = parse_request("what moved for Mars", probe)
    assert unknown.filters == () and "nothing reachable from payments is called 'Mars'; not filtered" in unknown.reading
    assert not any(line.startswith("ignored") for line in unknown.reading)  # said once, as a filter that failed, not again as an ignored word
    soft = parse_request("what moved in Mars", probe)
    assert soft.filters == () and any(line == "ignored: mars (nothing in the source matched)" for line in soft.reading)
    named = parse_request("what moved for region Americas", probe)
    assert [(p["column"], v) for p, v in named.filters] == [("region", "Americas")]
    reversed_order = parse_request("what moved for the Americas region", probe)
    assert [(p["column"], v) for p, v in reversed_order.filters] == [("region", "Americas")]
    after_a_period = parse_request("what changed in June 2024 for sector Technology", probe)  # the blanked period must not hand the clause to "in"
    assert [(p["column"], v) for p, v in after_a_period.filters] == [("sector", "Technology")] and after_a_period.period == {"year": 2024, "month": 6}
    kept_measure = parse_request("was amount in June 2024 normal for Europe", probe)  # the check strips only its own words
    assert kept_measure.check and kept_measure.measures == ("amount",) and [(p["column"], v) for p, v in kept_measure.filters] == [("region", "Europe")]


def _payments_and_tickets():
    """Two facts: payments carry a region, tickets do not, and only tickets carry a channel."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE payments (payment_id INTEGER, payment_date DATE, amount DOUBLE, region VARCHAR)")
    con.execute("CREATE TABLE tickets (ticket_id INTEGER, opened_at DATE, channel VARCHAR, hours DOUBLE)")
    for i, (day, amount, region) in enumerate([("2023-06-15", 100.0, "Europe"), ("2023-06-15", 50.0, "Americas"), ("2024-06-15", 300.0, "Europe"), ("2024-06-15", 60.0, "Americas")]):
        con.execute("INSERT INTO payments VALUES (?, ?, ?, ?)", [i, day, amount, region])
    for i, (day, channel, hours) in enumerate([("2023-06-15", "email", 2.0), ("2023-06-15", "phone", 1.0), ("2024-06-15", "email", 5.0), ("2024-06-15", "phone", 1.5)]):
        con.execute("INSERT INTO tickets VALUES (?, ?, ?, ?)", [i, day, channel, hours])
    tables = {"payments": ("payment_id", "payment_date", "amount", "region"), "tickets": ("ticket_id", "opened_at", "channel", "hours")}
    types = {"payments": {"payment_id": "INTEGER", "payment_date": "DATE", "amount": "DOUBLE", "region": "VARCHAR"}, "tickets": {"ticket_id": "INTEGER", "opened_at": "DATE", "channel": "VARCHAR", "hours": "DOUBLE"}}
    return LakehouseProbe.from_executor(_module._executor(con, tables), schema_from_tables("lh-2", tables, types=types), name="Two facts")


def test_a_filter_is_carried_by_the_facts_that_can_carry_it_and_looked_for_in_every_fact():
    probe = _payments_and_tickets()
    europe = report(probe, "what moved for Europe", years=[2023, 2024], budget=100)
    assert [(p["column"], v) for p, v in europe.spec.filters] == [("region", "Europe")]
    assert {m.fact for m in europe.sweep.ledger} == {"payments"} and europe.sweep.findings[0].movement.after_value == 300.0
    assert any(note.startswith("tickets: no column reachable from it carries region is Europe") for note in europe.sweep.notes)
    phone = report(probe, "what changed in June 2024 for phone", years=[2023, 2024], budget=100)  # the value lives in the second fact
    assert [(p["column"], v) for p, v in phone.spec.filters] == [("channel", "phone")] and phone.spec.facts == ("tickets",)
    assert any("the fact that holds the value asked for" in line for line in phone.spec.reading)
    assert {m.fact for m in phone.sweep.ledger} == {"tickets"} and phone.sweep.findings[0].movement.after_value == 1.5


def test_a_period_asked_about_the_trend_is_root_cause_with_a_check():
    probe = _reseller_lakehouse()
    result = report(probe, "what changed in December 2013 and was it in line with the trend", years=[2012, 2013], instructions=INSTRUCTIONS, budget=100)
    spec = result.spec
    assert spec.kind == "root_cause" and spec.check and spec.period == {"year": 2013, "month": 12}
    assert "check: December 2013 against the trend and season of each measure" in spec.reading
    assert result.checks and all("December 2013" in line for line in result.checks) and any("implied" in line for line in result.checks)
    assert "Against the trend and season" in result.to_html() and "Against the trend and season:" in result.to_markdown()
    latest = parse_request("was the latest month normal for revenue", probe, instructions=INSTRUCTIONS)
    assert latest.check and latest.kind == "recap" and latest.period is None and "check: the latest complete month against the trend and season of each measure" in latest.reading
    plain_trend = parse_request("trend of revenue by product category", probe, instructions=INSTRUCTIONS)
    assert plain_trend.kind == "trend" and not plain_trend.check


def test_which_focus_and_per_read_as_groupings_and_ignored_words_are_reported():
    probe = _reseller_lakehouse()
    which = parse_request("which territory should we focus on for revenue", probe, instructions=INSTRUCTIONS)
    assert which.kind == "root_cause" and which.groupings == ("SalesTerritoryRegion",) and which.measures == ("SalesAmount",) and which.filters == ()
    per = parse_request("revenue per product category", probe, instructions=INSTRUCTIONS)
    assert per.groupings == ("EnglishProductCategoryName",)
    noise = parse_request("show me blorp what moved", probe, instructions=INSTRUCTIONS)
    assert noise.kind == "recap" and "ignored: blorp (nothing in the source matched)" in noise.reading
    clean = parse_request("what moved", probe, instructions=INSTRUCTIONS)
    assert not any(line.startswith("ignored") for line in clean.reading) and clean.filters == ()


def test_the_dax_dialect_narrows_every_query_shape_and_can_look_a_value_up():
    model = FakeModel()
    probe = SemanticModelProbe(model)
    joins = probe.joins("")
    dialect = probe.dialect(joins)
    axis = dialect.axis("Sales")
    fact = {"table": "Sales", "date": axis, "measure": "Sales Amount", "aggregate": "sum"}
    color = {"column": "Color", "table": "Product", "hops": ({"from_column": "ProductKey", "table": "Product", "key": "ProductKey"},)}
    narrow = [(color, "Yellow")]
    assert "TREATAS({\"Yellow\"}, 'Product'[Color])" in dialect.series(fact, ["Sales Amount"], narrow)
    assert dialect.series(fact, ["Sales Amount"]) == dialect.series(fact, ["Sales Amount"], [])
    assert "TREATAS" in dialect.top_groups(fact, color, [2012, 2013], 6, narrow) and "TREATAS" not in dialect.top_groups(fact, color, [2012, 2013], 6)
    assert "TREATAS" in dialect.grouped_series(fact, color, [2012, 2013], ["Yellow", "Black"], narrow)
    assert "CALCULATE(" in dialect.max_date(fact, narrow) and "CALCULATE(" not in dialect.max_date(fact)
    assert 'CONTAINSSTRING(FORMAT(\'Product\'[Color], "General"), "yel")' in dialect.lookup(fact, color, "yel")
    members = dialect.movement(fact, __import__("fabric_rlm.sweep", fromlist=["Comparison"]).Comparison("year", {"year": 2012}, {"year": 2013}), [(color, ("Yellow", "Black"))])
    assert 'TREATAS({"Yellow", "Black"}, \'Product\'[Color])' in members


def test_a_month_is_read_against_the_trend_and_the_season():
    # three years of a rising series with a strong December: the check finds December in line and a planted collapse outside the spread
    points = []
    for year in (2021, 2022, 2023):
        for month in range(1, 13):
            level = 1000 * (1.01 ** ((year - 2021) * 12 + month - 1))
            season = 1.6 if month == 12 else 0.8 if month == 1 else 1.0
            points.append(Point(year, month, level * season, 100))
    story = analyse(points, name="orders")
    assert story.decomposition is not None
    within = period_check(story, 2022, 12, name="orders")
    assert within is not None and "December 2022" in within and "within the usual spread" in within and "trend and season" in within
    extended = expected_value(story, 2023, 12)
    assert extended is not None and "extended from" in extended[1]
    collapsed = analyse(points[:-1] + [Point(2023, 12, 100.0, 100)], name="orders")
    outside = period_check(collapsed, 2023, 12, name="orders")
    assert outside is not None and "below what the trend" in outside and "outside the usual spread" in outside
    assert period_check(story, 2019, 5, name="orders") is None
    short = SeriesStory(months=(Month(2023, 1, 10.0, 1), Month(2023, 2, 11.0, 1)), window=0, average=(None, None), shifts=(), yoy=(), decomposition=None, trend_per_year=None)
    assert expected_value(short, 2023, 2) is None
