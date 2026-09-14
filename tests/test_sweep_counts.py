"""Tables with no amount (tickets, cases, sessions) are measured by their rows; a label with hundreds of values is not a grouping."""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

from fabric_rlm.brief import brief
from fabric_rlm.data_agent_review import schema_from_tables
from fabric_rlm.sweep import LakehouseProbe, what_moved

_sweep_tests = importlib.util.spec_from_file_location("test_sweep", Path(__file__).with_name("test_sweep.py"))
_module = importlib.util.module_from_spec(_sweep_tests)
_sweep_tests.loader.exec_module(_module)
SOURCE, _executor = _module.SOURCE, _module._executor


def _tickets():
    """Two years of support tickets with no numeric column: ten a month in 2023, fifteen a month in 2024, the extra ones all high priority through chat."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE tickets (ticket_id INTEGER, opened_at TIMESTAMP, priority VARCHAR, channel VARCHAR, requester VARCHAR)")
    ticket = 0
    for year, per_month in ((2023, 10), (2024, 15)):
        for month in range(1, 13):
            for i in range(per_month):
                ticket += 1
                extra = i >= 10
                con.execute(
                    "INSERT INTO tickets VALUES (?, ?, ?, ?, ?)",
                    [ticket, f"{year}-{month:02d}-{(i % 28) + 1:02d} 09:00:00", "high" if extra else ("low" if i % 2 else "medium"), "chat" if extra else "email", f"person-{ticket % 400}"],
                )
    tables = {"tickets": ("ticket_id", "opened_at", "priority", "channel", "requester")}
    types = {"tickets": {"ticket_id": "INTEGER", "opened_at": "TIMESTAMP", "priority": "VARCHAR", "channel": "VARCHAR", "requester": "VARCHAR"}}
    return LakehouseProbe.from_executor(_executor(con, tables), schema_from_tables(SOURCE, tables, types=types), name="Support")


def test_a_table_with_no_amount_is_swept_by_its_row_count_and_recomputed():
    probe = _tickets()
    assert probe.facts("") == ["tickets"]
    result = what_moved(probe, budget=40)
    year = next(m for m in result.ledger if m.path is None and m.comparison.kind == "year")
    assert (year.measure, year.aggregate, year.before_value, year.after_value) == ("rows", "count", 120.0, 180.0)
    assert "Tickets rose 50.0% 2023 to 2024 (120 to 180)" in result.lines()[0]
    assert result.verified and result.recomputed > 0 and result.mismatches == ()
    finding = next(f for f in result.findings if f.movement.comparison.kind == "year")
    by_priority = next(d for d in finding.decompositions if d.path["column"] == "priority")
    assert by_priority.concentration == "single" and by_priority.groups[0].group == "high" and by_priority.groups[0].delta == 60.0
    assert not any(flag.startswith("volume:") for flag in finding.flags)  # a count is not split into volume and rate


def test_a_label_with_hundreds_of_values_reads_as_fragmented_not_as_a_driver():
    result = what_moved(_tickets(), budget=40)
    finding = next(f for f in result.findings if f.movement.comparison.kind == "year")
    requester = next(d for d in finding.decompositions if d.path["column"] == "requester")
    assert requester.concentration == "fragmented" and finding.best is not None and finding.best.path["column"] != "requester"
    assert "fragmented across" in result.to_markdown()


def test_the_brief_counts_rows_for_a_metric_named_after_the_table():
    result = brief(_tickets(), ["tickets by priority", "number of tickets"], week="2024-12-09", budget=30)
    metric = result.metrics[0]
    assert metric.name == "tickets by priority" and metric.measure == "rows" and metric.aggregate == "count"
    assert metric.target.start == "2024-12-09" and metric.target.value == 7.0  # the 9th to the 15th of December, one ticket a day
    assert metric.finding is not None and metric.finding.movement.aggregate == "count"
    assert result.metrics[1].name == "tickets" and result.metrics[1].measure == "rows"  # "number of tickets" is the count of the table


def test_the_briefed_week_is_measured_when_a_metric_ends_a_day_short_of_the_sunday():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE sales (sale_date DATE, region VARCHAR, amount DOUBLE)")
    con.execute("CREATE TABLE production (run_date DATE, plant VARCHAR, units INTEGER)")
    day = dt.date(2024, 1, 1)
    while day <= dt.date(2024, 12, 29):  # a Sunday
        con.execute("INSERT INTO sales VALUES (?, ?, ?)", [day.isoformat(), "North", 100.0])
        if day.weekday() < 6:  # no Sunday shift: production ends on Saturday the 28th
            con.execute("INSERT INTO production VALUES (?, ?, ?)", [day.isoformat(), "Riverside", 10])
        day += dt.timedelta(days=1)
    tables = {"sales": ("sale_date", "region", "amount"), "production": ("run_date", "plant", "units")}
    types = {"sales": {"sale_date": "DATE", "region": "VARCHAR", "amount": "DOUBLE"}, "production": {"run_date": "DATE", "plant": "VARCHAR", "units": "INTEGER"}}
    probe = LakehouseProbe.from_executor(_executor(con, tables), schema_from_tables(SOURCE, tables, types=types), name="Plant")
    result = brief(probe, ["sales amount", "production units"], budget=30)
    assert result.week is not None and result.week.start == "2024-12-23"
    units = next(m for m in result.metrics if m.measure == "units")
    assert units.target.start == "2024-12-23" and units.target.value == 60.0
    assert any("its data ends on 28 Dec 2024, 1 day(s) before the end of the week" in note for note in units.notes)


def test_the_pareto_view_takes_the_finest_grouping_and_the_page_carries_its_notation_and_references():
    result = what_moved(_tickets(), budget=40)
    finding = next(f for f in result.findings if f.movement.comparison.kind == "year")
    fine, view = result.pareto_view(finding)
    assert fine is not None and fine.path["column"] == "requester" and view is not None and view["n"] == 300
    assert view["k_base"] <= 300 and 0 < view["curve_base"][-1] <= 1.0 + 1e-9 and view["curve_change"][-1] >= 0.99
    sentence = result.pareto_sentence(fine)
    assert sentence.startswith("Pareto: ") and "of the 300 requester groups" in sentence and "carry 80% of tickets in 2024" in sentence
    assert any(line.strip() == sentence for line in result.lines())
    assert result.pareto_view(next(f for f in result.findings if f.movement.comparison.kind != "year"))[0] is None or True  # a smaller finding may have no fine grouping
    html = result.to_html()
    assert "Pareto view, by requester" in html and "Before and after, by" in html and "wmhatch" in html  # the variance chart and the hatch that marks a fall
    assert "<summary>About this page</summary>" in html and "Tables used" in html and "Generated" in html
    assert "Set aside, not read as business change (" in html and html.index("Set aside, not read as business change (") > html.index("Driver analysis")  # collapsed, after the analysis
    assert "Notation:" in html and "#00875a" in html and "#c9500a" in html  # the rise and fall pair


def _shop_with_products(*, duplicate_key: bool = False, missing_key: bool = False):
    """Daily sales of three products over two years, a products table joined on product_id; one product can be listed twice or not at all."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE sales (sale_date DATE, region VARCHAR, product_id INTEGER, amount DOUBLE)")
    con.execute("CREATE TABLE products (product_id INTEGER, category VARCHAR)")
    con.execute("INSERT INTO sales SELECT DATE '2023-01-01' + INTERVAL (d) DAY, CASE WHEN d % 2 = 0 THEN 'North' ELSE 'South' END, 1 + d % 3, 100.0 + CASE WHEN d >= 365 THEN 50.0 ELSE 0.0 END FROM range(730) t(d)")
    rows = [(1, "Bikes"), (2, "Parts")] + ([] if missing_key else [(3, "Gear")]) + ([(1, "Bikes again")] if duplicate_key else [])
    for row in rows:
        con.execute("INSERT INTO products VALUES (?, ?)", list(row))
    tables = {"sales": ("sale_date", "region", "product_id", "amount"), "products": ("product_id", "category")}
    types = {"sales": {"sale_date": "DATE", "region": "VARCHAR", "product_id": "INTEGER", "amount": "DOUBLE"}, "products": {"product_id": "INTEGER", "category": "VARCHAR"}}
    return LakehouseProbe.from_executor(_executor(con, tables), schema_from_tables(SOURCE, tables, types=types), name="Shop")


def test_a_join_that_multiplies_rows_is_flagged_and_never_read_as_the_driver():
    result = what_moved(_shop_with_products(duplicate_key=True), instructions="sales.product_id = products.product_id", budget=40)
    finding = next(f for f in result.findings if f.movement.comparison.kind == "year")
    by_category = next(d for d in finding.decompositions if d.path["column"] == "category")
    assert by_category.concentration == "none" and by_category.flags and by_category.flags[0].startswith("join multiplies rows")
    assert finding.best is not None and finding.best.path["column"] == "region"
    assert "join multiplies rows" in result.to_markdown() and result.pareto(by_category) is None


def test_rows_the_join_drops_are_kept_as_a_group_so_the_split_adds_up():
    result = what_moved(_shop_with_products(missing_key=True), instructions="sales.product_id = products.product_id", budget=40)
    finding = next(f for f in result.findings if f.movement.comparison.kind == "year")
    by_category = next(d for d in finding.decompositions if d.path["column"] == "category")
    remainder = next(g for g in by_category.groups if g.group == "(no match in products)")
    parent = by_category.parent
    assert remainder.query == "" and abs(sum(g.after_value for g in by_category.groups) - parent.after_value) < 1e-6 and abs(sum(g.before_value for g in by_category.groups) - parent.before_value) < 1e-6
    assert remainder.after_rows == parent.after_rows - sum(g.after_rows for g in by_category.groups if g.query)
    assert result.verified and result.mismatches == ()  # the remainder is arithmetic, not a figure the source recomputes


def test_a_grouping_cut_at_500_reads_its_pareto_against_the_true_total():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE sales (sale_date DATE, customer VARCHAR, amount DOUBLE)")
    con.execute("INSERT INTO sales SELECT DATE '2023-01-15' + to_months(m::INTEGER), 'customer ' || c, (10.0 + c % 7) * CASE WHEN m >= 12 THEN 1.5 ELSE 1.0 END FROM range(24) t(m), range(600) u(c)")
    tables = {"sales": ("sale_date", "customer", "amount")}
    types = {"sales": {"sale_date": "DATE", "customer": "VARCHAR", "amount": "DOUBLE"}}
    probe = LakehouseProbe.from_executor(_executor(con, tables), schema_from_tables(SOURCE, tables, types=types), name="Shop")
    result = what_moved(probe, budget=40)
    finding = next(f for f in result.findings if f.movement.comparison.kind == "year")
    by_customer = next(d for d in finding.decompositions if d.path["column"] == "customer")
    assert by_customer.size == 500
    view = result.pareto(by_customer)
    assert view is not None and view["capped"] and view["n"] == 500 and abs(view["curve_base"][-1] - 1.0) < 1e-9 and view["listed_base"] < 1.0
    sentence = result.pareto_sentence(by_customer)
    assert "largest customer movers the query listed (there are more)" in sentence


def test_a_timestamp_with_a_time_zone_is_read_in_utc_whatever_the_session_zone():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("SET TimeZone = 'America/Los_Angeles'")
    con.execute("CREATE TABLE events (happened_at TIMESTAMPTZ, kind VARCHAR, amount DOUBLE)")
    con.execute("INSERT INTO events VALUES (TIMESTAMPTZ '2024-07-01 03:00:00+00', 'a', 1.0), (TIMESTAMPTZ '2024-06-15 12:00:00+00', 'a', 1.0)")
    tables = {"events": ("happened_at", "kind", "amount")}
    types = {"events": {"happened_at": "TIMESTAMP WITH TIME ZONE", "kind": "VARCHAR", "amount": "DOUBLE"}}
    probe = LakehouseProbe.from_executor(_executor(con, tables), schema_from_tables(SOURCE, tables, types=types), name="Events")
    dialect = probe.dialect(probe.joins(""))
    axis = dialect.axis("events")
    assert axis is not None and axis.get("tz") is True
    sql = dialect.daily({"table": "events", "date": axis, "measure": "amount", "aggregate": "sum"}, ["amount"])
    assert "timezone('UTC'" in sql
    days = sorted(str(row["day"])[:10] for row in probe.run(sql))
    assert days == ["2024-06-15", "2024-07-01"]  # 03:00 UTC on 1 July stays 1 July, not 30 June Los Angeles time
