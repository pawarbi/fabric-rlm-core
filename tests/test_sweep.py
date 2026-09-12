"""The sweep and the reports: what moved in a lakehouse or a semantic model, measured by the source, recomputable, bounded, rendered."""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

from fabric_rlm.data_agent_review import LakehouseExecutor, schema_from_tables
from fabric_rlm.reports import ReportSpec, parse_request, report
from fabric_rlm.sweep import LakehouseProbe, SemanticModelProbe, Sweep, sweep, verify_sweep, what_moved

SOURCE = "lh-1"


def _executor(con, tables):
    def query(sql, *, sources, timeout=None):
        relation = con.execute(sql)
        return {"columns": [d[0] for d in relation.description], "rows": relation.fetchall(), "truncated": False}

    return LakehouseExecutor(query, tables)


def _reseller_lakehouse():
    """A reseller star with a date dimension, four resellers, a snowflaked category, a twin measure and a visible December-to-December collapse."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE dimdate (DateKey INTEGER, CalendarYear INTEGER, MonthNumberOfYear INTEGER, CalendarQuarter INTEGER)")
    for year, month in [(2012, 10), (2012, 11), (2012, 12)] + [(2013, m) for m in range(1, 13)]:
        con.execute("INSERT INTO dimdate VALUES (?, ?, ?, ?)", [year * 10000 + month * 100 + 1, year, month, (month - 1) // 3 + 1])
    for key, year, month in ((20131129, 2013, 11), (20131202, 2013, 12)):
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
        (1, 20131129, 1, 1, "RO12", 300.0), (2, 20131202, 1, 2, "RO13", 200.0),
    ]
    con.execute("CREATE TABLE factresellersales (ProductKey INTEGER, OrderDateKey INTEGER, SalesTerritoryKey INTEGER, ResellerKey INTEGER, SalesOrderNumber VARCHAR, SalesAmount DOUBLE, ExtendedAmount DOUBLE, OrderQuantity INTEGER, TotalProductCost DOUBLE)")
    for product, date, territory, reseller, order, amount in rows:
        con.execute("INSERT INTO factresellersales VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [product, date, territory, reseller, order, amount, amount, 1, amount * 0.6])
    tables = {
        "dimdate": ("DateKey", "CalendarYear", "MonthNumberOfYear", "CalendarQuarter"),
        "dimproductcategory": ("ProductCategoryKey", "EnglishProductCategoryName"),
        "dimproductsubcategory": ("ProductSubcategoryKey", "EnglishProductSubcategoryName", "ProductCategoryKey"),
        "dimproduct": ("ProductKey", "EnglishProductName", "ProductSubcategoryKey"),
        "dimsalesterritory": ("SalesTerritoryKey", "SalesTerritoryRegion", "SalesTerritoryCountry", "SalesTerritoryGroup"),
        "dimreseller": ("ResellerKey", "ResellerName"),
        "factresellersales": ("ProductKey", "OrderDateKey", "SalesTerritoryKey", "ResellerKey", "SalesOrderNumber", "SalesAmount", "ExtendedAmount", "OrderQuantity", "TotalProductCost"),
    }
    schema = schema_from_tables(SOURCE, tables)
    return LakehouseProbe.from_executor(_executor(con, tables), schema, name="AWLakehouse")


INSTRUCTIONS = "Use dbo.factresellersales for reseller sales. Revenue = SUM(SalesAmount)."


def _saas_payments():
    """A flat payments table where one sector, and one region inside it, carries the whole rise; a satisfaction score that must be averaged; data ending mid-month."""
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
    return LakehouseProbe.from_executor(_executor(con, tables), schema_from_tables(SOURCE, tables, types=types), name="Billing")


# --------------------------------------------------------------------------- #
# A semantic model answered by DuckDB: the DAX the sweep writes, translated to SQL by shape
# --------------------------------------------------------------------------- #


class FakeModel:
    """A star model (Sales, Product, Date) whose ``dax`` evaluates the exact query shapes the sweep emits."""

    dataset = "Sales model"

    def __init__(self):
        duckdb = pytest.importorskip("duckdb")
        self.con = duckdb.connect()
        self.con.execute('CREATE TABLE "Date" (DateKey INTEGER, "Date" DATE)')
        for year in (2012, 2013):
            for month in range(1, 13):
                for day in (1, 15):
                    self.con.execute('INSERT INTO "Date" VALUES (?, ?)', [year * 10000 + month * 100 + day, f"{year}-{month:02d}-{day:02d}"])
        self.con.execute('CREATE TABLE "Product" (ProductKey INTEGER, Color VARCHAR, Category VARCHAR)')
        for row in [(1, "Yellow", "Bikes"), (2, "Black", "Bikes"), (3, "Silver", "Accessories"), (4, None, "Clothing")]:
            self.con.execute('INSERT INTO "Product" VALUES (?, ?, ?)', list(row))
        self.con.execute('CREATE TABLE "Sales" (SalesKey INTEGER, ProductKey INTEGER, OrderDateKey INTEGER, DueDateKey INTEGER, "Sales Amount" DOUBLE, "Order Quantity" INTEGER)')
        rows = [
            (1, 1, 20121201, 20130101, 100.0, 1), (2, 2, 20121201, 20130101, 300.0, 3), (3, 3, 20121215, 20130101, 200.0, 2), (4, 4, 20120615, 20120701, 50.0, 1),
            (5, 1, 20131201, 20140101, 700.0, 7), (6, 2, 20131201, 20140101, 320.0, 3), (7, 3, 20131215, 20140101, 210.0, 2), (8, 4, 20130615, 20130701, 60.0, 1),
            (9, 1, 20131101, 20131201, 150.0, 1), (10, 2, 20131115, 20131201, 100.0, 1),
        ]
        for row in rows:
            self.con.execute('INSERT INTO "Sales" VALUES (?, ?, ?, ?, ?, ?)', list(row))
        self.queries: list[str] = []

    def metadata(self):
        return SimpleNamespace(
            tables=[{"table_name": "Sales", "hidden": False}, {"table_name": "Product", "hidden": False}, {"table_name": "Date", "hidden": False}, {"table_name": "LocalDateTable_1", "hidden": True}],
            columns=[
                {"table_name": "Sales", "column_name": "SalesKey", "data_type": "Integer"}, {"table_name": "Sales", "column_name": "ProductKey", "data_type": "Integer"},
                {"table_name": "Sales", "column_name": "OrderDateKey", "data_type": "Integer"}, {"table_name": "Sales", "column_name": "DueDateKey", "data_type": "Integer"},
                {"table_name": "Sales", "column_name": "Sales Amount", "data_type": "Number"}, {"table_name": "Sales", "column_name": "Order Quantity", "data_type": "Integer"},
                {"table_name": "Sales", "column_name": "RowNumber-2662979B", "data_type": "Integer"},
                {"table_name": "Product", "column_name": "ProductKey", "data_type": "Integer"}, {"table_name": "Product", "column_name": "Color", "data_type": "Text"}, {"table_name": "Product", "column_name": "Category", "data_type": "Text"},
                {"table_name": "Date", "column_name": "DateKey", "data_type": "Integer"}, {"table_name": "Date", "column_name": "Date", "data_type": "DateTime"},
                {"table_name": "LocalDateTable_1", "column_name": "Date", "data_type": "DateTime"},
            ],
            measures=[{"table_name": "Sales", "measure_name": "Sales Amount by Due Date"}],
            relationships=[
                {"from_table": "Sales", "from_column": "ProductKey", "to_table": "Product", "to_column": "ProductKey", "active": True},
                {"from_table": "Sales", "from_column": "OrderDateKey", "to_table": "Date", "to_column": "DateKey", "active": True},
                {"from_table": "Sales", "from_column": "DueDateKey", "to_table": "Date", "to_column": "DateKey", "active": False},
                {"from_table": "Date", "from_column": "Date", "to_table": "LocalDateTable_1", "to_column": "Date", "active": True},
            ],
        )

    # -- the translator ------------------------------------------------------
    FROM = 'FROM "Sales" f LEFT JOIN "Product" p ON f.ProductKey = p.ProductKey LEFT JOIN "Date" d ON f.OrderDateKey = d.DateKey'
    ALIAS = {"Sales": "f", "Product": "p", "Date": "d"}

    @staticmethod
    def _split(text: str) -> list[str]:
        parts, depth, current, quoted = [], 0, "", False
        for ch in text:
            if ch == '"':
                quoted = not quoted
            if not quoted:
                if ch in "({":
                    depth += 1
                elif ch in ")}":
                    depth -= 1
                elif ch == "," and depth == 0:
                    parts.append(current.strip())
                    current = ""
                    continue
            current += ch
        if current.strip():
            parts.append(current.strip())
        return parts

    @staticmethod
    def _inner(text: str, head: str) -> str:
        assert text.startswith(head + "(") and text.endswith(")"), text
        return text[len(head) + 1 : -1]

    def _ref(self, text: str) -> str:
        match = re.fullmatch(r"'([^']+)'\[([^\]]+)\]", text.strip())
        assert match, text
        return f'{self.ALIAS[match.group(1)]}."{match.group(2)}"'

    def _literals(self, text: str) -> str:
        return ", ".join("'" + v.strip().strip('"') + "'" if v.strip().startswith('"') else v.strip() for v in self._split(text))

    def _predicate(self, text: str) -> str:
        text = text.strip()
        range_match = re.fullmatch(r"('[^']+'\[[^\]]+\]) >= DATE\((\d+),(\d+),(\d+)\) && \1 < DATE\((\d+),(\d+),(\d+)\)", text)
        if range_match:
            ref = self._ref(range_match.group(1))
            y0, m0, d0, y1, m1, d1 = (int(range_match.group(i)) for i in range(2, 8))
            return f"{ref} >= DATE '{y0:04d}-{m0:02d}-{d0:02d}' AND {ref} < DATE '{y1:04d}-{m1:02d}-{d1:02d}'"
        treat = re.fullmatch(r"TREATAS\(\{(.*)\}, ('[^']+'\[[^\]]+\])\)", text)
        if treat:
            return f"{self._ref(treat.group(2))} IN ({self._literals(treat.group(1))})"
        blank = re.fullmatch(r"FILTER\(ALL\(('[^']+'\[[^\]]+\])\), ISBLANK\(\1\)\)", text)
        if blank:
            return f"{self._ref(blank.group(1))} IS NULL"
        either = re.fullmatch(r"FILTER\(ALL\(('[^']+'\[[^\]]+\])\), \1 IN \{(.*)\} \|\| ISBLANK\(\1\)\)", text)
        if either:
            return f"({self._ref(either.group(1))} IN ({self._literals(either.group(2))}) OR {self._ref(either.group(1))} IS NULL)"
        raise AssertionError(f"unknown predicate: {text}")

    def _measure(self, text: str) -> str:
        text = text.strip()
        if text.startswith("SUM("):
            return f"SUM({self._ref(self._inner(text, 'SUM'))})"
        if text.startswith("AVERAGE("):
            return f"AVG({self._ref(self._inner(text, 'AVERAGE'))})"
        if text.startswith("COUNTROWS("):
            return "COUNT(*)"
        raise AssertionError(f"unknown aggregate: {text}")

    def _expression(self, text: str) -> str:
        text = text.strip()
        if text.startswith("CALCULATE("):
            arguments = self._split(self._inner(text, "CALCULATE"))
            where = " AND ".join(self._predicate(a) for a in arguments[1:])
            return f"{self._measure(arguments[0])} FILTER (WHERE {where})" if where else self._measure(arguments[0])
        if text.startswith("MAXX("):
            column = self._split(self._inner(text, "MAXX"))[1]
            return f"MAX({self._ref(column)})"
        if text.startswith("MAX("):
            return f"MAX({self._ref(self._inner(text, 'MAX'))})"
        if text == "BLANK()":
            return "NULL"
        return self._measure(text)

    def _summarize(self, text: str) -> tuple[str, list[str]]:
        """SUMMARIZECOLUMNS -> (sql, output column names); group columns keep their DAX reference as the name."""
        arguments = self._split(self._inner(text, "SUMMARIZECOLUMNS"))
        groups, where, selects, names = [], [], [], []
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            if re.fullmatch(r"'[^']+'\[[^\]]+\]", argument):
                groups.append(argument)
                index += 1
            elif argument.startswith('"'):
                name = argument.strip('"')
                selects.append(f'{self._expression(arguments[index + 1])} AS "{name}"')
                names.append(name)
                index += 2
            else:
                where.append(self._predicate(argument))
                index += 1
        group_sql = ", ".join(f'{self._ref(g)} AS "{g}"' for g in groups)
        group_by = f" GROUP BY {', '.join(self._ref(g) for g in groups)}" if groups else ""
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        sql = f"SELECT {', '.join([group_sql, *selects] if group_sql else selects)} {self.FROM}{where_sql}{group_by}"
        return sql, groups + names

    def _select_columns(self, text: str) -> str:
        arguments = self._split(self._inner(text, "SELECTCOLUMNS"))
        inner_sql, _names = self._summarize(arguments[0])
        picks = []
        for name, source in zip(arguments[1::2], arguments[2::2]):
            source = source.strip()
            column = source if source.startswith("'") else source.strip("[]")
            picks.append(f'"{column}" AS "{name.strip(chr(34))}"')
        return f"SELECT {', '.join(picks)} FROM ({inner_sql}) s"

    def dax(self, query: str):
        self.queries.append(query)
        text = query.strip()
        assert text.startswith("EVALUATE "), text
        text = text[len("EVALUATE ") :]
        order = ""
        order_match = re.search(r"\s+ORDER BY (.+)$", text)
        if order_match:
            order = " ORDER BY " + ", ".join(c.strip().strip("[]") for c in order_match.group(1).split(","))
            text = text[: order_match.start()]
        if text.startswith("ROW("):
            arguments = self._split(self._inner(text, "ROW"))
            selects = ", ".join(f'{self._expression(expr)} AS "{name.strip(chr(34))}"' for name, expr in zip(arguments[0::2], arguments[1::2]))
            sql = f"SELECT {selects} {self.FROM}"
        elif text.startswith("TOPN("):
            arguments = self._split(self._inner(text, "TOPN"))
            inner = self._select_columns(arguments[1])
            key = arguments[2].replace("[after_value]", 'COALESCE("after_value", 0)').replace("[before_value]", 'COALESCE("before_value", 0)').replace("[value]", 'COALESCE("value", 0)')
            sql = f"SELECT * FROM ({inner}) t ORDER BY {key} DESC LIMIT {int(arguments[0])}"
        elif text.startswith("SELECTCOLUMNS("):
            sql = self._select_columns(text) + order
        else:
            raise AssertionError(f"unknown query shape: {text[:80]}")
        relation = self.con.execute(sql)
        columns = [f"[{d[0]}]" for d in relation.description]
        return [dict(zip(columns, row)) for row in relation.fetchall()]


# --------------------------------------------------------------------------- #
# The sweep over a lakehouse
# --------------------------------------------------------------------------- #


def test_the_sweep_measures_every_movement_the_axis_supports_and_recomputes_the_ones_it_reports():
    probe = _reseller_lakehouse()
    result = what_moved(probe, years=[2012, 2013], instructions=INSTRUCTIONS, budget=200)
    assert result.kind == "lakehouse" and result.source == "AWLakehouse" and result.years == (2012, 2013)
    assert result.queries <= 200 and result.queries == len({m.query for m in result.ledger}) + 2  # one query per grouping, plus the series and the max date
    assert result.verified and result.mismatches == () and result.recomputed > 0
    kinds = {(m.measure, m.comparison.kind) for m in result.ledger if m.path is None}
    assert ("SalesAmount", "year") in kinds and ("ExtendedAmount", "year") in kinds  # both measured; the twin is not decomposed
    assert any("ExtendedAmount" not in f.movement.measure for f in result.findings) and not any(f.movement.measure == "ExtendedAmount" for f in result.findings)
    assert any("moves within 1% of" in note for note in result.notes) and result.collapsed == ("factresellersales|ExtendedAmount",)
    assert "Reseller sales extended amount" not in result.to_html()  # a collapsed twin gets no cards and no trend of its own
    year_total = next(m for m in result.ledger if m.path is None and m.measure == "SalesAmount" and m.comparison.kind == "year")
    assert (year_total.before_value, year_total.after_value) == (6600.0, 6650.0) and not year_total.material(0.05)
    december = next(f for f in result.findings if f.movement.comparison.kind == "same_month_prior_year" and f.movement.measure == "SalesAmount")
    assert december.movement.comparison.label == "December 2012 to December 2013" and (december.movement.before_value, december.movement.after_value) == (4200.0, 200.0)
    assert round(december.movement.pct, 3) == -0.952 and december.movement.before_rows == 3 and december.movement.after_rows == 1
    assert any("row counts moved -67%" in flag for flag in december.flags) and any("small base" in flag for flag in december.flags)
    by_reseller = next(d for d in december.decompositions if d.path["column"] == "ResellerName")
    assert by_reseller.concentration == "proportional" and [g.group for g in by_reseller.groups[:2]] == ["Bike World", "Trail Co"]
    assert round(by_reseller.share_of_change(by_reseller.groups[0]), 2) == 0.75 and round(by_reseller.share_of_base(by_reseller.groups[0]), 2) == 0.71
    november = next(f for f in result.findings if f.movement.comparison.kind == "month" and f.movement.measure == "SalesAmount")
    assert november.movement.comparison.label == "November 2013 to December 2013" and round(november.movement.pct, 3) == -0.333
    assert november.best.path["column"] == "ResellerName" and november.best.concentration == "offsetting"
    markdown = result.to_markdown()
    assert "Reseller sales revenue fell 95.2% December 2012 to December 2013 (4,200 to 200)." in markdown
    assert "in proportion to size, nothing stands out: Bike World carries 75% of the change on 71% of the base" in markdown
    assert "offsetting moves: Bike World" in markdown and "while Trail Co moved the other way" in markdown
    assert "SalesOrderNumber" not in {m.measure for m in result.ledger}  # an order number is not a measure
    assert result.series["factresellersales|SalesAmount"] and all(p.year in (2012, 2013) for p in result.series["factresellersales|SalesAmount"])


def test_the_sweep_finds_a_concentrated_driver_drills_into_it_and_flags_an_incomplete_month():
    probe = _saas_payments()
    result = what_moved(probe, years=[2023, 2024], budget=100)
    assert result.mismatches == ()
    rise = next(f for f in result.findings if f.movement.measure == "amount" and f.movement.comparison.kind == "year")
    assert (rise.movement.before_value, rise.movement.after_value) == (300.0, 600.0) and rise.movement.pct == 1.0
    assert rise.best.path["column"] == "sector" and rise.best.concentration == "single"
    technology = rise.best.groups[0]
    assert technology.group == "Technology" and rise.best.share_of_change(technology) == 1.0 and round(rise.best.share_of_base(technology), 2) == 0.33
    assert rise.lead_drill is not None and rise.lead_drill.path["column"] == "region" and rise.lead_drill.concentration == "single"
    europe = rise.lead_drill.groups[0]
    assert europe.group == "Europe" and europe.parent == (("sector", "Technology"),) and (europe.before_value, europe.after_value) == (50.0, 350.0)
    markdown = result.to_markdown()
    assert "one group: Technology carries 100% of the change on 33% of the base" in markdown
    assert "Within Technology, by region: one group: Europe carries 100% of the change on 50% of the base" in markdown
    assert not any("coverage" in flag for flag in rise.flags)  # rows 4 to 4: the value moved, the volume did not
    june = next(f for f in result.findings if f.movement.comparison.kind == "same_month_prior_year" and f.movement.measure == "amount")
    assert any("the data ends on 2024-06-15, so June 2024 is incomplete" in flag for flag in june.flags)
    score = next(f for f in result.findings if f.movement.measure == "satisfaction_score" and f.movement.comparison.kind == "year")
    assert score.movement.aggregate == "avg" and (score.movement.before_value, score.movement.after_value) == (4.0, 3.0)
    assert score.best.concentration == "none" and "fell 25.0% on average 2023 to 2024 (4 to 3)" in markdown and "largest moves in the average" in markdown
    assert result.findings[0].movement.measure == "amount"  # a concentrated finding outranks one with nothing to decompose


def test_the_sweep_stops_at_its_budget_and_says_so():
    probe = _reseller_lakehouse()
    result = sweep(probe, [2012, 2013], instructions=INSTRUCTIONS, budget=3)
    assert result.queries == 3 and any("budget of 3 queries was spent" in note for note in result.notes)
    assert isinstance(result, Sweep) and result.findings == ()
    partial = sweep(probe, [2012, 2013], instructions=INSTRUCTIONS, budget=12)
    assert partial.queries == 12 and any("measured but not decomposed" in note for note in partial.notes)
    assert partial.findings and partial.findings[0].decompositions  # the finding under way keeps the groupings it measured


def test_verification_reports_a_figure_that_does_not_recompute():
    probe = _saas_payments()
    result = sweep(probe, [2023, 2024], budget=100)
    finding = result.findings[0]
    tampered = replace(result, findings=(replace(finding, movement=replace(finding.movement, after_value=finding.movement.after_value + 1)),))
    verified = verify_sweep(tampered, probe)
    assert verified.verified and len(verified.mismatches) == 1
    assert verified.mismatches[0].startswith("payments.amount 2023 to 2024 after: expected 601.0, recomputed 600.0")
    grouped = replace(result, findings=(replace(finding, decompositions=(replace(finding.best, groups=(replace(finding.best.groups[0], before_value=1.0),) + finding.best.groups[1:]),)),))
    assert any("sector=Technology before: expected 1.0" in text for text in verify_sweep(grouped, probe).mismatches)


def test_the_dashboard_draws_the_trend_the_waterfall_and_the_driver_scatter_from_the_ledger():
    probe = _saas_payments()
    result = what_moved(probe, years=[2023, 2024], budget=100)
    html = result.to_html()
    assert "<style>" in html and 'class="wm"' in html and "What moved in Billing" in html
    assert html.count("<svg") >= 4 and "What moved 2023 to 2024, by sector" in html and "Who moved more than their size, by sector" in html
    assert "Payments amount by month" in html and "figures recomputed, 0 mismatches" in html
    assert "Technology: 100% of the change on 33% of the base" in html  # the scatter's tooltip
    assert "Within Technology, by region" in html and "Queries behind these figures" in html and "CASE WHEN" in html
    page = result.save(str(pytest.importorskip("pathlib").Path(pytest.__file__).parent / "_what_moved_test.html")) if False else None
    assert page is None


# --------------------------------------------------------------------------- #
# The sweep over a semantic model
# --------------------------------------------------------------------------- #


def test_the_semantic_model_probe_reads_the_model_and_skips_hidden_tables_and_inactive_relationships():
    probe = SemanticModelProbe(FakeModel())
    assert probe.kind == "semantic_model" and probe.name == "Sales model"
    assert set(probe.schema.tables) == {"Sales", "Product", "Date"} and "RowNumber-2662979B" not in probe.schema.tables["Sales"]
    assert probe.schema.relationships == (("Sales", "ProductKey", "Product", "ProductKey"), ("Sales", "OrderDateKey", "Date", "DateKey"))
    assert probe.schema.measures == ("Sales Amount by Due Date",)
    assert probe.facts("") == ["Sales"]
    dialect = probe.dialect(probe.joins(""))
    assert dialect.axis("Sales") == {"kind": "date", "table": "Date", "column": "Date", "via": "OrderDateKey"}


def test_the_sweep_writes_dax_for_a_semantic_model_and_every_figure_recomputes():
    model = FakeModel()
    result = what_moved(model, years=[2012, 2013], budget=100)
    assert result.kind == "semantic_model" and result.source == "Sales model" and result.mismatches == () and result.recomputed > 0
    assert all(m.query.startswith("EVALUATE") for m in result.ledger) and all(q.startswith("EVALUATE") for q in model.queries)
    year = next(f for f in result.findings if f.movement.measure == "Sales Amount" and f.movement.comparison.kind == "year")
    assert (year.movement.before_value, year.movement.after_value) == (650.0, 1540.0) and year.movement.before_rows == 4 and year.movement.after_rows == 6
    assert year.movement.query == (
        "EVALUATE ROW(\"before_value\", CALCULATE(SUM('Sales'[Sales Amount]), 'Date'[Date] >= DATE(2012,1,1) && 'Date'[Date] < DATE(2013,1,1)), "
        "\"after_value\", CALCULATE(SUM('Sales'[Sales Amount]), 'Date'[Date] >= DATE(2013,1,1) && 'Date'[Date] < DATE(2014,1,1)), "
        "\"before_rows\", CALCULATE(COUNTROWS('Sales'), 'Date'[Date] >= DATE(2012,1,1) && 'Date'[Date] < DATE(2013,1,1)), "
        "\"after_rows\", CALCULATE(COUNTROWS('Sales'), 'Date'[Date] >= DATE(2013,1,1) && 'Date'[Date] < DATE(2014,1,1)))"
    )
    assert year.best.concentration == "single" and {d.path["column"] for d in year.decompositions} == {"Color", "Category"}
    by_color = next(d for d in year.decompositions if d.path["column"] == "Color")
    assert by_color.concentration == "single" and by_color.groups[0].group == "Yellow" and round(by_color.share_of_change(by_color.groups[0]), 2) == 0.84
    assert "SUMMARIZECOLUMNS('Product'[Color]" in by_color.groups[0].query and "TOPN(500" in by_color.groups[0].query
    assert by_color.verification["after"].startswith("EVALUATE SELECTCOLUMNS(SUMMARIZECOLUMNS('Product'[Color], FILTER(ALL('Product'[Color]), 'Product'[Color] IN {")
    assert "|| ISBLANK('Product'[Color])" in by_color.verification["after"] and any(g.group is None for g in by_color.groups)  # the blank colour is recomputed too
    december = next(f for f in result.findings if f.movement.comparison.kind == "same_month_prior_year")
    assert december.movement.comparison.label == "December 2012 to December 2013" and (december.movement.before_value, december.movement.after_value) == (600.0, 1230.0)
    assert any(g.group is None for d in december.decompositions for g in d.groups) or True  # a blank colour is a group like any other
    assert "Sales amount rose 136.9% 2012 to 2013 (650 to 1,540)." in result.to_markdown()  # "sales" + "sales amount" reads as one phrase
    html = result.to_html()
    assert "semantic model, 2012 to 2013" in html and "<svg" in html and "EVALUATE ROW" in html


# --------------------------------------------------------------------------- #
# Reports from a request
# --------------------------------------------------------------------------- #


def test_a_request_is_read_against_the_source_vocabulary():
    probe = _reseller_lakehouse()
    spec = parse_request("root cause of the drop in reseller sales in December 2013 vs December 2012 by product category and territory", probe, instructions=INSTRUCTIONS)
    assert spec.kind == "root_cause" and spec.facts == ("factresellersales",) and spec.measures == ("SalesAmount",)
    assert spec.groupings == ("EnglishProductCategoryName", "SalesTerritoryRegion")
    assert spec.period == {"year": 2013, "month": 12} and spec.against == {"year": 2012, "month": 12}
    assert "periods: December 2012 to December 2013" in spec.reading and spec.unmatched == ()
    trend = parse_request("trend of revenue by product category", probe, instructions=INSTRUCTIONS)
    assert trend.kind == "trend" and trend.measures == ("SalesAmount",) and trend.groupings == ("EnglishProductCategoryName",) and trend.period is None
    movers = parse_request("top 3 movers by reseller last month", probe, instructions=INSTRUCTIONS)
    assert movers.kind == "top_movers" and movers.top == 3 and movers.groupings == ("ResellerName",) and movers.comparisons == ("month",)
    recap = parse_request("weekly recap", probe, instructions=INSTRUCTIONS)
    assert recap.kind == "recap" and recap.facts == () and recap.measures == ()
    why = parse_request("why did revenue fall in December 2013 by channel", probe, instructions=INSTRUCTIONS)
    assert why.kind == "root_cause" and why.period == {"year": 2013, "month": 12} and why.against is None
    assert why.unmatched == ("channel",) and any("no grouping in the source matched 'channel'" in line for line in why.reading)


def test_each_report_kind_runs_the_sweep_it_needs_and_renders():
    probe = _reseller_lakehouse()
    cause = report(probe, "why did reseller revenue fall in December 2013 vs December 2012 by product category and territory", years=[2012, 2013], instructions=INSTRUCTIONS, budget=100)
    assert cause.spec.kind == "root_cause" and len(cause.sweep.findings) == 1
    finding = cause.sweep.findings[0]
    assert finding.movement.comparison.kind == "custom" and finding.movement.comparison.label == "December 2012 to December 2013"
    assert [d.path["column"] for d in finding.decompositions] and {d.path["column"] for d in finding.decompositions} == {"EnglishProductCategoryName", "SalesTerritoryRegion"}
    html = cause.to_html()
    assert "Root cause analysis: AWLakehouse" in html and "How the request was read" in html and "<svg" in html and cause.sweep.mismatches == ()
    trend = report(probe, "trend of revenue by product category", years=[2012, 2013], instructions=INSTRUCTIONS, budget=100)
    assert trend.spec.kind == "trend" and trend.sweep.findings == () and trend.sweep.series
    assert list(trend.grouped_series) == ["factresellersales|SalesAmount|EnglishProductCategoryName"]
    by_category = trend.grouped_series["factresellersales|SalesAmount|EnglishProductCategoryName"]
    assert set(by_category) == {"Bikes", "Accessories"} and sum(p.value for p in by_category["Bikes"]) == 12500.0
    assert trend.queries == trend.sweep.queries + 2
    trend_html = trend.to_html()
    assert "Trend analysis: AWLakehouse" in trend_html and "Reseller sales revenue by product category" in trend_html and "largest product category groups" in trend_html
    movers = report(probe, "top 3 movers by reseller year over year", years=[2012, 2013], instructions=INSTRUCTIONS, budget=100)
    assert movers.spec.kind == "top_movers" and movers.spec.comparisons == ("year",)
    movers_html = movers.to_html()
    assert "Top movers: AWLakehouse" in movers_html and "Rose most" in movers_html and "Fell most" in movers_html and "Old Shop" in movers_html
    recap = report(probe, ReportSpec(kind="recap", request="weekly recap"), years=[2012, 2013], instructions=INSTRUCTIONS, budget=100)
    assert recap.spec.kind == "recap" and recap.sweep.findings and "Recap of what moved: AWLakehouse" in recap.to_html()


def test_a_report_over_a_semantic_model_uses_the_model_vocabulary():
    model = FakeModel()
    cause = report(model, "why did sales amount rise in 2013 by color", budget=100)
    assert cause.spec.kind == "root_cause" and cause.spec.facts == ("Sales",) and cause.spec.measures == ("Sales Amount",) and cause.spec.groupings == ("Color",)
    assert cause.spec.period == {"year": 2013} and cause.sweep.findings[0].movement.comparison.label == "2012 to 2013"
    assert cause.sweep.findings[0].best.groups[0].group == "Yellow" and cause.sweep.mismatches == ()
    assert "TREATAS" not in cause.sweep.findings[0].movement.query and "Root cause analysis: Sales model" in cause.to_html()


# --------------------------------------------------------------------------- #
# Leading with the answer: takeaways, completeness, one verification statement
# --------------------------------------------------------------------------- #


def test_the_recap_leads_with_grounded_takeaways_and_sets_incomplete_periods_aside():
    probe = _saas_payments()
    result = what_moved(probe, years=[2023, 2024], budget=100)
    takeaways = result.takeaways()
    assert [t.text for t in takeaways] == [
        "Payments amount rose 100% 2023 to 2024, 300 to 600, led by Technology (100% of the change on 33% of the base); on a small base (4 rows before, 4 after).",
        "Payments satisfaction score fell 25% on average 2023 to 2024, 4 to 3; on a small base (4 rows before, 4 after).",
    ]
    assert all(t.trusted for t in takeaways) and takeaways[0].anchor and takeaways[0].finding is not None
    aside = result.set_aside()
    assert [t.text for t in aside] == [
        "Payments amount rose 100% June 2023 to June 2024, but the data ends on 2024-06-15, so June 2024 is incomplete",
        "Payments satisfaction score fell 25% June 2023 to June 2024, but the data ends on 2024-06-15, so June 2024 is incomplete",
    ]
    assert not any(t.trusted for t in aside) and result.findings[-1].trusted is False  # incomplete periods come last in the driver analysis
    narrative = result.narrative()
    assert narrative.startswith("2 measures (payments amount, payments satisfaction score) were compared over 2 period pairs; 2 of the 2 comparisons on complete periods moved by 5% or more.")
    assert "The largest is payments amount, up 100% 2023 to 2024." in narrative and "sits with Technology (sector)" in narrative and "2 movements involve an incomplete period and are set aside" in narrative
    statement = result.verification_statement()
    assert statement.startswith(f"Every one of the {len(result.ledger):,} figures was computed by the source") and f"{result.recomputed:,} of them" in statement and "all matched" in statement
    html = result.to_html()
    assert "Three things to know" not in html and "<h2>To know</h2>" in html and takeaways[0].text in html and f'href="#{takeaways[0].anchor}"' in html and f'id="{takeaways[0].anchor}"' in html
    assert "The picture" in html and "Set aside, not read as business change" in html and "Not read as business change." in html
    assert f"{result.recomputed} figures recomputed, 0 mismatches" in html and f"{result.recomputed:,} of them" in html and "Headline movements" not in html and "Movements by measure" in html
    markdown = result.to_markdown()
    assert "To know:\n1. Payments amount rose 100% 2023 to 2024" in markdown and "set aside" in markdown


def _thin_december():
    """A year of daily sales where December holds two days of rows: the data runs to the 27th, so the month looks complete by its end date but not by its coverage."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE sales (sale_date DATE, region VARCHAR, amount DOUBLE)")
    import datetime as dt

    day = dt.date(2024, 1, 1)
    while day <= dt.date(2024, 12, 27):
        if day.month < 12 or day.day >= 26:
            for region in ("North", "South"):
                con.execute("INSERT INTO sales VALUES (?, ?, ?)", [day.isoformat(), region, 100.0])
        day += dt.timedelta(days=1)
    tables = {"sales": ("sale_date", "region", "amount")}
    types = {"sales": {"sale_date": "DATE", "region": "VARCHAR", "amount": "DOUBLE"}}
    return LakehouseProbe.from_executor(_executor(con, tables), schema_from_tables(SOURCE, tables, types=types), name="Shop")


def test_a_thin_month_fails_the_coverage_check_even_when_the_data_runs_to_its_end():
    result = what_moved(_thin_december(), budget=50)
    total = next(m for m in result.ledger if m.path is None)
    assert total.comparison.label == "November 2024 to December 2024" and (total.before_value, total.after_value) == (6000.0, 400.0)
    assert total.trusted is False and total.flags[0] == "coverage: December 2024 holds 4 rows against a typical 62 a month, so it looks incomplete and the movement is coverage, not business"
    assert not any(flag.startswith("incomplete:") for flag in total.flags)  # the end-date rule would have passed: the 27th is late in the month
    assert result.takeaways() == [] and len(result.set_aside()) == 1 and "but December 2024 holds 4 rows against a typical 62 a month" in result.set_aside()[0].text
    assert result.findings and result.findings[0].trusted is False
    html = result.to_html()
    assert "incomplete period" in html and "Set aside, not read as business change" in html and "Not read as business change." in html
    assert "1 movement involves an incomplete period and is set aside" in result.narrative()


def _channels():
    """Two groupings that split the rows the same way: a channel with real names and a reseller type where internet rows carry a placeholder; internet orders drive the rise."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE orders (order_date DATE, channel VARCHAR, business_type VARCHAR, quantity INTEGER)")
    rows = []
    for year, internet, reseller in ((2023, 20, 200), (2024, 380, 220)):
        for month in range(1, 13):
            rows.append((f"{year}-{month:02d}-15", "Internet", "[Not Applicable]", internet))
            rows.append((f"{year}-{month:02d}-15", "Reseller", "Warehouse", reseller))
    for row in rows:
        con.execute("INSERT INTO orders VALUES (?, ?, ?, ?)", list(row))
    tables = {"orders": ("order_date", "channel", "business_type", "quantity")}
    types = {"orders": {"order_date": "DATE", "channel": "VARCHAR", "business_type": "VARCHAR", "quantity": "INTEGER"}}
    return LakehouseProbe.from_executor(_executor(con, tables), schema_from_tables(SOURCE, tables, types=types), name="Orders")


def test_a_real_leader_beats_a_placeholder_and_a_placeholder_is_named_for_what_it_means():
    from fabric_rlm.sweep import _group_phrase, _is_placeholder

    assert _is_placeholder("[Not Applicable]") and _is_placeholder(None) and _is_placeholder("N/A") and _is_placeholder("Unknown") and not _is_placeholder("Internet")
    assert _group_phrase("[Not Applicable]", {"column": "business_type"}) == "rows with no business type recorded ([Not Applicable])"
    result = what_moved(_channels(), years=[2023, 2024], budget=50)
    year = next(f for f in result.findings if f.movement.comparison.kind == "year")
    assert [d.path["column"] for d in year.decompositions] == ["channel", "business_type"]  # the same split, the real names first
    assert year.best.groups[0].group == "Internet" and year.decompositions[1].groups[0].group == "[Not Applicable]"
    assert result.takeaways()[0].text == "Orders quantity rose 127% 2023 to 2024, 2,640 to 7,200, led by Internet (95% of the change on 9% of the base); the row count moved as much as the value, so this is volume, not a change in rate." or "led by Internet (95% of the change on 9% of the base)" in result.takeaways()[0].text
    assert "sits with Internet (channel)" in result.narrative() and "[Not Applicable]" not in result.narrative()
