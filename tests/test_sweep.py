"""The sweep: what moved in a source, measured by the source, recomputable, bounded."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fabric_rlm.data_agent_review import AgentDataSource, AgentSnapshot, LakehouseExecutor, schema_from_tables
from fabric_rlm.sweep import Sweep, sweep, verify_sweep

SOURCE = "lh-1"


def _executor(con, tables):
    def query(sql, *, sources, timeout=None):
        relation = con.execute(sql)
        return {"columns": [d[0] for d in relation.description], "rows": relation.fetchall(), "truncated": False}

    return LakehouseExecutor(query, tables)


def _reseller_lakehouse():
    """A reseller star with a date dimension, four resellers, a snowflaked category and a visible December-to-December collapse."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE dimdate (DateKey INTEGER, CalendarYear INTEGER, MonthNumberOfYear INTEGER, CalendarQuarter INTEGER)")
    for year, month in [(2012, 10), (2012, 11), (2012, 12)] + [(2013, m) for m in range(1, 13)]:
        con.execute("INSERT INTO dimdate VALUES (?, ?, ?, ?)", [year * 10000 + month * 100 + 1, year, month, (month - 1) // 3 + 1])
    for key, year, month in ((20131129, 2013, 11), (20131202, 2013, 12), (20140115, 2014, 1)):
        con.execute("INSERT INTO dimdate VALUES (?, ?, ?, ?)", [key, year, month, (month - 1) // 3 + 1])
    con.execute("CREATE TABLE dimproductcategory AS SELECT * FROM (VALUES (1, 'Bikes'), (2, 'Accessories')) t(ProductCategoryKey, EnglishProductCategoryName)")
    con.execute("CREATE TABLE dimproductsubcategory AS SELECT * FROM (VALUES (1, 'Mountain Bikes', 1), (2, 'Helmets', 2)) t(ProductSubcategoryKey, EnglishProductSubcategoryName, ProductCategoryKey)")
    con.execute("CREATE TABLE dimproduct AS SELECT * FROM (VALUES (1, 'Mountain-200', 1), (2, 'Road-350', 1), (3, 'Sport Helmet', 2)) t(ProductKey, EnglishProductName, ProductSubcategoryKey)")
    con.execute("CREATE TABLE dimsalesterritory AS SELECT * FROM (VALUES (1, 'Northwest', 'United States', 'North America'), (2, 'Germany', 'Germany', 'Europe')) t(SalesTerritoryKey, SalesTerritoryRegion, SalesTerritoryCountry, SalesTerritoryGroup)")
    con.execute("CREATE TABLE dimreseller AS SELECT * FROM (VALUES (1, 'Bike World'), (2, 'Trail Co'), (3, 'Pedal Shop'), (4, 'Old Shop')) t(ResellerKey, ResellerName)")
    rows = [
        (1, 20121001, 1, 1, "RO0a", 2000.0), (2, 20121101, 1, 4, "RO0b", 400.0),
        (1, 20121201, 1, 1, "RO1", 3000.0), (2, 20121201, 1, 2, "RO2", 1000.0), (3, 20121201, 1, 3, "RO3", 200.0),
        (1, 20130101, 1, 1, "RO4", 500.0), (2, 20130101, 1, 2, "RO5", 900.0), (3, 20130101, 1, 3, "RO6", 250.0),
        (1, 20130201, 1, 1, "RO7", 1200.0), (2, 20130201, 1, 2, "RO8", 800.0),
        (1, 20130301, 2, 1, "RO9", 1500.0), (3, 20130601, 1, 3, "RO10", 300.0), (2, 20131001, 2, 2, "RO11", 700.0),
        (1, 20131129, 1, 1, "RO12", 300.0), (2, 20131202, 1, 2, "RO13", 200.0), (1, 20140115, 1, 1, "RO14", 150.0),
    ]
    con.execute("CREATE TABLE factresellersales (ProductKey INTEGER, OrderDateKey INTEGER, SalesTerritoryKey INTEGER, ResellerKey INTEGER, SalesOrderNumber VARCHAR, SalesAmount DOUBLE, OrderQuantity INTEGER, TotalProductCost DOUBLE)")
    for product, date, territory, reseller, order, amount in rows:
        con.execute("INSERT INTO factresellersales VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [product, date, territory, reseller, order, amount, 1, amount * 0.6])
    tables = {
        "dimdate": ("DateKey", "CalendarYear", "MonthNumberOfYear", "CalendarQuarter"),
        "dimproductcategory": ("ProductCategoryKey", "EnglishProductCategoryName"),
        "dimproductsubcategory": ("ProductSubcategoryKey", "EnglishProductSubcategoryName", "ProductCategoryKey"),
        "dimproduct": ("ProductKey", "EnglishProductName", "ProductSubcategoryKey"),
        "dimsalesterritory": ("SalesTerritoryKey", "SalesTerritoryRegion", "SalesTerritoryCountry", "SalesTerritoryGroup"),
        "dimreseller": ("ResellerKey", "ResellerName"),
        "factresellersales": ("ProductKey", "OrderDateKey", "SalesTerritoryKey", "ResellerKey", "SalesOrderNumber", "SalesAmount", "OrderQuantity", "TotalProductCost"),
    }
    source = AgentDataSource(id=SOURCE, kind="lakehouse", name="AWLakehouse", instructions="Use dbo.factresellersales for reseller sales. Revenue = SUM(SalesAmount).", description="Reseller sales.")
    snapshot = AgentSnapshot(agent_id="a", name="Sales Agent", instructions="Answer with figures.", datasources=(source,))
    return _executor(con, tables), schema_from_tables(SOURCE, tables), snapshot


def _saas_payments():
    """A flat payments table where one sector, and one region inside it, carries the whole rise; a satisfaction score that must be averaged."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE payments (payment_id INTEGER, payment_date DATE, amount DOUBLE, sector VARCHAR, region VARCHAR, satisfaction_score DOUBLE)")
    rows = [
        (1, "2023-06-15", 50.0, "Technology", "Europe", 4.0), (2, "2023-06-15", 50.0, "Technology", "Americas", 4.0),
        (3, "2023-06-15", 100.0, "Services", "Europe", 4.0), (4, "2023-06-15", 100.0, "Commerce", "Americas", 4.0),
        (5, "2024-06-15", 350.0, "Technology", "Europe", 3.0), (6, "2024-06-15", 50.0, "Technology", "Americas", 3.0),
        (7, "2024-06-15", 110.0, "Services", "Europe", 3.0), (8, "2024-06-15", 90.0, "Commerce", "Americas", 3.0),
    ]
    for row in rows:
        con.execute("INSERT INTO payments VALUES (?, ?, ?, ?, ?, ?)", list(row))
    tables = {"payments": ("payment_id", "payment_date", "amount", "sector", "region", "satisfaction_score")}
    types = {"payments": {"payment_id": "INTEGER", "payment_date": "DATE", "amount": "DOUBLE", "sector": "VARCHAR", "region": "VARCHAR", "satisfaction_score": "DOUBLE"}}
    source = AgentDataSource(id=SOURCE, kind="lakehouse", name="Billing", instructions="", description="")
    snapshot = AgentSnapshot(agent_id="a", name="Billing agent", instructions="", datasources=(source,))
    return _executor(con, tables), schema_from_tables(SOURCE, tables, types=types), snapshot


def test_the_sweep_measures_every_movement_the_axis_supports_and_recomputes_the_ones_it_reports():
    executor, schema, snapshot = _reseller_lakehouse()
    result = sweep(executor, schema, snapshot, [2012, 2013], budget=200)
    assert result.queries <= 200 and result.queries == len({m.sql for m in result.ledger}) + 2  # one query per grouping, plus the month series and the max date
    assert verify_sweep(result, executor) == []
    kinds = {(f.movement.measure, f.movement.comparison.kind): f for f in result.findings}
    assert ("SalesAmount", "year") not in kinds  # 6,600 to 6,650 is not material
    year_total = next(m for m in result.ledger if m.measure == "SalesAmount" and m.comparison.kind == "year" and m.path is None)
    assert (year_total.before_value, year_total.after_value) == (6600.0, 6650.0)
    december = kinds[("SalesAmount", "same_month_prior_year")]
    assert december.movement.comparison.label == "December 2012 to December 2013" and (december.movement.before_value, december.movement.after_value) == (4200.0, 200.0)
    assert round(december.movement.pct, 3) == -0.952 and december.movement.before_rows == 3 and december.movement.after_rows == 1
    assert any("row counts moved -67%" in flag for flag in december.flags) and any("small base" in flag for flag in december.flags)
    by_reseller = next(d for d in december.decompositions if d.path["column"] == "ResellerName")
    assert by_reseller.concentration == "proportional" and [g.group for g in by_reseller.groups[:2]] == ["Bike World", "Trail Co"]
    assert round(by_reseller.share_of_change(by_reseller.groups[0]), 2) == 0.75 and round(by_reseller.share_of_base(by_reseller.groups[0]), 2) == 0.71
    november = kinds[("SalesAmount", "month")]
    assert november.movement.comparison.label == "November 2013 to December 2013" and round(november.movement.pct, 3) == -0.333
    assert november.best.path["column"] == "ResellerName" and november.best.concentration == "offsetting"
    markdown = result.to_markdown()
    assert "Reseller sales revenue fell 95.2% December 2012 to December 2013 (4,200 to 200)." in markdown
    assert "in proportion to size, nothing stands out: Bike World 75% of the change on 71% of the base" in markdown
    assert "offsetting moves: Bike World" in markdown and "while Trail Co moved the other way" in markdown
    assert "SalesOrderNumber" not in {m.measure for m in result.ledger}  # an order number is not a measure
    html = result.to_html()
    assert "<table>" in html and "Share of change" in html and "<details><summary>Queries</summary>" in html


def test_the_sweep_finds_a_concentrated_driver_and_drills_into_it():
    executor, schema, snapshot = _saas_payments()
    result = sweep(executor, schema, snapshot, [2023, 2024], budget=100)
    assert verify_sweep(result, executor) == []
    rise = next(f for f in result.findings if f.movement.measure == "amount" and f.movement.comparison.kind == "year")
    assert (rise.movement.before_value, rise.movement.after_value) == (300.0, 600.0) and rise.movement.pct == 1.0
    assert rise.best.path["column"] == "sector" and rise.best.concentration == "single"
    technology = rise.best.groups[0]
    assert technology.group == "Technology" and rise.best.share_of_change(technology) == 1.0 and round(rise.best.share_of_base(technology), 2) == 0.33
    assert rise.drill and rise.drill[0].path["column"] == "region" and rise.drill[0].concentration == "single"
    europe = rise.drill[0].groups[0]
    assert europe.group == "Europe" and europe.parent == (("sector", "Technology"),) and (europe.before_value, europe.after_value) == (50.0, 350.0)
    assert "one group carries it, Technology 100% of the change on 33% of the base" in result.to_markdown()
    assert "Within Technology, by region: one group carries it, Europe 100% of the change on 50% of the base" in result.to_markdown()
    assert not any("coverage" in flag for flag in rise.flags)  # rows 4 to 4: the value moved, the volume did not

    score = next(f for f in result.findings if f.movement.measure == "satisfaction_score" and f.movement.comparison.kind == "year")
    assert score.movement.aggregate == "avg" and (score.movement.before_value, score.movement.after_value) == (4.0, 3.0)
    assert score.best.concentration == "none" and "fell 25.0% on average 2023 to 2024 (4 to 3)" in result.to_markdown()
    assert "largest moves in the average" in result.to_markdown()
    assert result.findings[0] is rise  # a concentrated finding outranks one with nothing to decompose


def test_the_sweep_stops_at_its_budget_and_says_so():
    executor, schema, snapshot = _reseller_lakehouse()
    result = sweep(executor, schema, snapshot, [2012, 2013], budget=3)
    assert result.queries == 3 and any("budget of 3 queries was spent" in note for note in result.notes)
    assert isinstance(result, Sweep) and result.to_markdown().startswith("0 material movement(s)") or result.findings


def test_verification_reports_a_figure_that_does_not_recompute():
    executor, schema, snapshot = _saas_payments()
    result = sweep(executor, schema, snapshot, [2023, 2024], budget=100)
    finding = result.findings[0]
    tampered = replace(result, findings=(replace(finding, movement=replace(finding.movement, after_value=finding.movement.after_value + 1)),))
    mismatches = verify_sweep(tampered, executor)
    assert len(mismatches) == 1 and mismatches[0].startswith("payments.amount 2023 to 2024 after: expected 601.0, recomputed 600.0")


def test_the_review_runs_the_sweep_when_given_a_budget_and_reports_what_moved():
    from fabric_rlm.data_agent_review import review_agent

    executor, schema, snapshot = _reseller_lakehouse()
    report = review_agent(snapshot, [schema], {SOURCE: executor}, lambda q: "Reseller revenue was $1,000.00 in 2013.", years={SOURCE: [2012, 2013]}, top=3, limit_per_source=6, sweep_budget=200)
    assert len(report.sweeps) == 1 and report.sweeps[0].findings
    assert any("sweep found" in note and "0 figure(s) failed to recompute" in note for note in report.notes)
    markdown = report.to_markdown()
    assert "## What moved (AWLakehouse)" in markdown and "Reseller sales revenue fell 95.2% December 2012 to December 2013" in markdown
    html = report.to_html()
    assert "<h2>What moved (AWLakehouse)</h2>" in html and "Share of change" in html
    quiet = review_agent(snapshot, [schema], {SOURCE: executor}, lambda q: "x", years={SOURCE: [2012, 2013]}, top=3, limit_per_source=4)
    assert quiet.sweeps == () and "What moved" not in quiet.to_markdown()
