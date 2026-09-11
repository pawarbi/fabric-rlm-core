"""Data Agent review: diagnosis, questions, references, grading, suggestions.

Everything runs against fakes: a snapshot shaped like a real agent, an
in-memory DuckDB star schema for the lakehouse, a scripted agent. The one
live-shaped piece, the REST reader and asker, is exercised through their
parsing helpers only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fabric_rlm.data_agent_review import (
    AgentAnswer,
    AgentDataSource,
    AgentSnapshot,
    FewShot,
    LakehouseExecutor,
    Question,
    Reference,
    SdkAgentReader,
    SemanticModelExecutor,
    _query_in,
    _tsql_to_duckdb,
    apply_suggestions,
    build_references,
    declared_from_snapshot,
    diagnose,
    discover_years,
    extract_definitions,
    generate_questions,
    grade,
    review_agent,
    schema_from_profile,
    schema_from_tables,
    suggest,
)

AGENT_INSTRUCTIONS = """You are the Adventure Works Sales Agent. Answer questions about sales performance.

SCOPE AND DEFINITIONS
- Unless a channel is specified, revenue means the sum of SalesAmount across both factinternetsales and factresellersales.
- Orders = row count.
- Treat 2013 as the last complete calendar year.

RESPONSE STYLE
- Lead with the answer, then a compact table.
"""

SOURCE_INSTRUCTIONS = """This source is the AWLakehouse star schema. Use dbo tables only.
- dbo.factinternetsales: B2C order lines. Revenue = SUM(SalesAmount). Orders = COUNT(DISTINCT SalesOrderNumber).
- dbo.factresellersales: B2B order lines.
- Product hierarchy: dimproduct.ProductSubcategoryKey -> dimproductsubcategory.ProductSubcategoryKey -> dimproductcategory.ProductCategoryKey
- Dates: factinternetsales.OrderDateKey -> dimdate.DateKey
- Territory: factinternetsales.SalesTerritoryKey -> dimsalesterritory.SalesTerritoryKey
- Pre-joined views available: vw_internet_sales, vw_reseller_sales
- Average order value = SUM(SalesAmount) / COUNT(DISTINCT SalesOrderNumber).
- Avoid joining the two fact tables directly.
- Never guess a definition.
- Do not list customer names.
"""

TABLES = {
    "factinternetsales": ("ProductKey", "OrderDateKey", "SalesTerritoryKey", "CustomerKey", "SalesOrderNumber", "SalesAmount", "OrderQuantity", "TotalProductCost"),
    "factresellersales": ("ProductKey", "OrderDateKey", "SalesTerritoryKey", "ResellerKey", "SalesOrderNumber", "SalesAmount", "OrderQuantity", "TotalProductCost"),
    "dimdate": ("DateKey", "FullDateAlternateKey", "CalendarYear", "CalendarQuarter"),
    "dimproduct": ("ProductKey", "EnglishProductName", "ProductSubcategoryKey"),
    "dimproductsubcategory": ("ProductSubcategoryKey", "EnglishProductSubcategoryName", "ProductCategoryKey"),
    "dimproductcategory": ("ProductCategoryKey", "EnglishProductCategoryName"),
    "dimsalesterritory": ("SalesTerritoryKey", "SalesTerritoryRegion", "SalesTerritoryCountry", "SalesTerritoryGroup"),
    "dimcustomer": ("CustomerKey", "GeographyKey", "FirstName"),
    "dimreseller": ("ResellerKey", "ResellerName"),
}

LAKEHOUSE_ID = "lh-1"
MODEL_ID = "sm-1"


def _snapshot(*, second_source: bool = False, fewshots: int = 0) -> AgentSnapshot:
    sources = [
        AgentDataSource(
            id=LAKEHOUSE_ID,
            kind="lakehouse",
            name="AWLakehouse",
            instructions=SOURCE_INSTRUCTIONS,
            description="Adventure Works sales star schema for internet and reseller sales.",
            fewshots=tuple(FewShot(f"f{i}", f"question {i}", "SELECT 1") for i in range(fewshots)),
        )
    ]
    if second_source:
        sources.append(AgentDataSource(id=MODEL_ID, kind="semantic_model", name="Sales Model", instructions="Use [Total Sales] for revenue.", description=""))
    return AgentSnapshot(agent_id="agent-1", name="Sales Agent", instructions=AGENT_INSTRUCTIONS, datasources=tuple(sources))


def _schemas(*, second_source: bool = False):
    schemas = [schema_from_tables(LAKEHOUSE_ID, TABLES)]
    if second_source:
        schemas.append(
            schema_from_tables(
                MODEL_ID,
                {"Date": ("Date", "Year"), "Product": ("Product", "Category"), "Sales": ("Amount",)},
                kind="semantic_model",
                measures=("Total Sales", "Total Orders"),
                relationships=(("Sales", "ProductKey", "Product", "ProductKey"),),
            )
        )
    return schemas


def _duckdb_executor():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE dimdate AS SELECT * FROM (VALUES (20120101, 2012), (20130101, 2013), (20140101, 2014)) t(DateKey, CalendarYear)")
    con.execute("CREATE TABLE dimproductcategory AS SELECT * FROM (VALUES (1, 'Bikes'), (2, 'Accessories')) t(ProductCategoryKey, EnglishProductCategoryName)")
    con.execute("CREATE TABLE dimproductsubcategory AS SELECT * FROM (VALUES (1, 'Mountain Bikes', 1), (2, 'Helmets', 2)) t(ProductSubcategoryKey, EnglishProductSubcategoryName, ProductCategoryKey)")
    con.execute("CREATE TABLE dimproduct AS SELECT * FROM (VALUES (1, 'Mountain-200', 1), (2, 'Road-350', 1), (3, 'Sport Helmet', 2)) t(ProductKey, EnglishProductName, ProductSubcategoryKey)")
    con.execute("CREATE TABLE dimsalesterritory AS SELECT * FROM (VALUES (1, 'Northwest', 'United States', 'North America'), (2, 'Germany', 'Germany', 'Europe')) t(SalesTerritoryKey, SalesTerritoryRegion, SalesTerritoryCountry, SalesTerritoryGroup)")
    con.execute("CREATE TABLE dimcustomer AS SELECT * FROM (VALUES (1, 1, 'Ann')) t(CustomerKey, GeographyKey, FirstName)")
    con.execute("CREATE TABLE dimreseller AS SELECT * FROM (VALUES (1, 'Bike World')) t(ResellerKey, ResellerName)")
    con.execute(
        "CREATE TABLE factinternetsales AS SELECT * FROM (VALUES "
        "(1, 20120101, 1, 1, 'SO1', 100.0, 1, 60.0), (2, 20130101, 1, 1, 'SO2', 300.0, 2, 150.0), (3, 20130101, 2, 1, 'SO2', 50.0, 1, 20.0), (3, 20130101, 2, 1, 'SO3', 25.0, 1, 10.0), (1, 20140101, 1, 1, 'SO9', 10.0, 1, 5.0)"
        ") t(ProductKey, OrderDateKey, SalesTerritoryKey, CustomerKey, SalesOrderNumber, SalesAmount, OrderQuantity, TotalProductCost)"
    )
    con.execute(
        "CREATE TABLE factresellersales AS SELECT * FROM (VALUES "
        "(1, 20120101, 2, 1, 'RO1', 1000.0, 10, 700.0), (2, 20130101, 2, 1, 'RO2', 2000.0, 20, 1500.0), (3, 20130101, 1, 1, 'RO3', 500.0, 5, 300.0)"
        ") t(ProductKey, OrderDateKey, SalesTerritoryKey, ResellerKey, SalesOrderNumber, SalesAmount, OrderQuantity, TotalProductCost)"
    )

    def query(sql, *, sources):
        relation = con.execute(sql)
        columns = [d[0] for d in relation.description]
        return {"columns": columns, "rows": relation.fetchall(), "truncated": False}

    return LakehouseExecutor(query, TABLES)


# ------------------------------------------------------------- diagnosis --


def test_diagnosis_follows_the_documented_rules():
    findings = {f.code: f for f in diagnose(_snapshot(second_source=True), _schemas(second_source=True))}
    moved = findings["schema_in_agent_instructions"]
    assert moved.source_id == LAKEHOUSE_ID and moved.severity == "medium"
    assert any("factinternetsales" in line for line in moved.evidence)
    assert not any("Lead with the answer" in line for line in moved.evidence)
    unknown = findings["unknown_reference"]
    assert "vw_internet_sales" in unknown.message and "vw_reseller_sales" in unknown.message and unknown.severity == "high"
    conflict = findings["conflicting_definitions"]
    assert "Orders" in conflict.evidence[0] and conflict.severity == "high"
    assert findings["no_fewshots"].source_id == LAKEHOUSE_ID
    assert findings["definitions_declared"].source_id == LAKEHOUSE_ID
    assert findings["calculation_in_instructions"].source_id == LAKEHOUSE_ID
    description = findings["datasource_description_missing"]
    assert description.source_id == MODEL_ID and description.severity == "high"
    assert findings["routing_rules_missing"].severity == "medium"
    assert findings["negative_phrasing"].source_id == LAKEHOUSE_ID
    assert "unstructured_instructions" in findings  # the source text has no headers
    assert all(f.basis for f in findings.values() if f.code != "source_not_profiled")


def test_diagnosis_is_quiet_on_a_clean_single_source_agent():
    clean = AgentSnapshot(
        agent_id="a",
        name="Clean",
        instructions="Answer sales questions. Lead with the answer.",
        datasources=(AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="AWLakehouse", instructions="## Join Paths\n- Join factinternetsales to dimdate on OrderDateKey = DateKey.\n## Business Rules\n- Revenue = SUM(SalesAmount)\n- Orders = COUNT(DISTINCT SalesOrderNumber)\n- Report USD.\n- State the period.\n- Rank after aggregation.", description="Sales facts.", fewshots=(FewShot("f", "q", "SELECT 1"),)),),
    )
    codes = {f.code for f in diagnose(clean, _schemas())}
    assert "schema_in_agent_instructions" not in codes and "unknown_reference" not in codes
    assert "no_fewshots" not in codes and "datasource_description_missing" not in codes
    assert "unstructured_instructions" not in codes and "negative_phrasing" not in codes


def test_length_limits_and_fewshot_floods_are_findings():
    long_text = "\n".join(f"- rule {i} about SalesAmount and the way to compute revenue for the period." for i in range(90))
    snapshot = AgentSnapshot(agent_id="a", name="Long", instructions="Answer.", datasources=(AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="lh", instructions=long_text, description="d", fewshots=tuple(FewShot(str(i), f"q{i}", "SELECT 1") for i in range(20))),))
    codes = {f.code: f for f in diagnose(snapshot, _schemas())}
    assert codes["instructions_too_long"].severity == "high"
    assert codes["fewshots_too_many"].severity == "medium"


def test_definitions_are_extracted_and_declared():
    definitions = extract_definitions("- Revenue = SUM(SalesAmount)\n- AOV means revenue per distinct order\nSome prose without a definition.")
    assert definitions == {"Revenue": "SUM(SalesAmount)", "AOV": "revenue per distinct order"}
    declared = declared_from_snapshot(_snapshot(), _schemas())
    assert "Average order value" in declared[LAKEHOUSE_ID]["definitions"]


# ------------------------------------------------------------- questions --


def test_questions_are_generated_from_the_schema_with_both_dialects():
    questions = generate_questions(_snapshot(), _schemas(), years={LAKEHOUSE_ID: [2012, 2013]}, top=3)
    kinds = [q.kind for q in questions]
    assert kinds[0] == "combined_total_by_year" and questions[0].alternates
    assert "top_n" in kinds and "distinct_by_year" in kinds and "yoy" in kinds
    top = next(q for q in questions if q.kind == "top_n")
    assert top.reference_query.startswith("SELECT TOP 3") and "LIMIT 3" in top.execution["sql"]
    assert "JOIN dimdate d" in top.execution["sql"] and "WHERE d.CalendarYear = 2013" in top.execution["sql"]
    combined = questions[0]
    assert "UNION ALL" in combined.reference_query and combined.reference_query.count("SUM(f.SalesAmount)") == 2
    assert all(q.text for q in questions)


def test_semantic_model_questions_render_dax():
    snapshot = _snapshot(second_source=True)
    questions = [q for q in generate_questions(snapshot, _schemas(second_source=True), years={MODEL_ID: [2012, 2013]}, top=5) if q.source_id == MODEL_ID]
    assert questions and questions[0].execution["kind"] == "aggregate"
    assert questions[0].reference_query.startswith("EVALUATE SUMMARIZECOLUMNS('Date'[Year]")
    top = next(q for q in questions if q.kind == "top_n")
    assert top.reference_query.startswith("EVALUATE TOPN(5,") and top.execution["order_by"] == "Total Sales"


def test_years_references_and_partial_year_detection():
    executor = _duckdb_executor()
    schema = _schemas()[0]
    assert discover_years(executor, schema, _snapshot().datasources[0]) == [2012, 2013]
    questions = generate_questions(_snapshot(), [schema], years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=20)
    references = {r.question_id: r for r in build_references(questions, {LAKEHOUSE_ID: executor})}
    combined = references[questions[0].id]
    assert combined.status == "ok"
    assert [(r["year"], float(r["value"])) for r in combined.rows] == [(2012, 1100.0), (2013, 2875.0)]
    assert set(combined.alternates) == {"factinternetsales", "factresellersales"}
    broken = build_references([Question("x", LAKEHOUSE_ID, "total_by_year", "?", {}, "", {"sql": "SELECT * FROM nowhere"})], {LAKEHOUSE_ID: executor})
    assert broken[0].status == "failed" and "nowhere" in broken[0].note.casefold() or broken[0].status == "failed"


# --------------------------------------------------------------- grading --


def _question(kind="total_by_year", alternates=()):
    return Question("q", LAKEHOUSE_ID, kind, "?", {}, "", {"sql": ""}, tuple(alternates))


def test_prose_grading_outcomes_and_causes():
    reference = Reference("q", "ok", ({"year": 2012, "value": 1100.0}, {"year": 2013, "value": 2850.0}), alternates={"factinternetsales": ({"year": 2012, "value": 100.0}, {"year": 2013, "value": 350.0})})
    assert grade(_question(), reference, "2012: $1,100.00; 2013: $2,850.00").outcome == "correct"
    assert grade(_question(), reference, "Revenue was $1,100 in 2012 and $2,850.4 in 2013").outcome == "correct"  # 0.5% tolerance
    narrower = grade(_question(), reference, "2012: $100.00 and 2013: $350.00")
    assert narrower.outcome == "wrong" and narrower.cause == "narrower_scope" and "factinternetsales" in narrower.detail
    assert grade(_question(), reference, "Total revenue was 999 and 42").cause == "values_differ"
    assert grade(_question(), reference, "I cannot answer this from the selected tables.").outcome == "abstained"
    assert grade(_question(), reference, "Revenue grew strongly.").outcome == "incomplete"
    abstain_expected = Reference("q", "abstained", note="no rows")
    assert grade(_question(), abstain_expected, "I do not have data for that.").outcome == "correct"
    assert grade(_question(), abstain_expected, "It was 12").cause == "answered_without_data"


def test_ranking_grading_sees_order_and_missing_rows():
    rows = ({"EnglishProductName": "Mountain-200", "value": 2500.0}, {"EnglishProductName": "Road-350", "value": 2000.0}, {"EnglishProductName": "Sport Helmet", "value": 50.0})
    reference = Reference("q", "ok", rows)
    ordered = grade(_question("top_n"), reference, "1. Mountain-200 2,500 2. Road-350 2,000 3. Sport Helmet 50")
    assert ordered.outcome == "correct"
    unsorted = grade(_question("top_n"), reference, "Road-350 2,000; Mountain-200 2,500; Sport Helmet 50")
    assert unsorted.outcome == "partial" and unsorted.cause == "unsorted_ranking"
    missing = grade(_question("top_n"), reference, "Mountain-200 2,500 and Road-350 2,000")
    assert missing.outcome == "partial" and missing.cause == "missing_rows"


def test_query_level_grading_prefers_the_executed_rows():
    reference = Reference("q", "ok", ({"year": 2012, "value": 1100.0}, {"year": 2013, "value": 2850.0}), alternates={"factinternetsales": ({"year": 2012, "value": 100.0}, {"year": 2013, "value": 350.0})})
    right = grade(_question(), reference, AgentAnswer("some prose without numbers", query="SELECT 1", language="sql"), agent_rows=[{"year": 2013, "value": 2850.0}, {"year": 2012, "value": 1100.0}])
    assert right.outcome == "correct" and right.query_checked
    scope = grade(_question(), reference, AgentAnswer("prose", query="SELECT 1", language="sql"), agent_rows=[{"year": 2012, "value": 100.0}, {"year": 2013, "value": 350.0}])
    assert scope.cause == "narrower_scope" and scope.query_checked
    fewer = grade(_question(), reference, AgentAnswer("prose", query="SELECT 1", language="sql"), agent_rows=[{"year": 2013, "value": 2850.0}])
    assert fewer.cause == "missing_rows"


def test_agent_sql_is_translated_for_the_local_engine_and_queries_are_unfenced():
    sql = _tsql_to_duckdb("SELECT TOP 3 p.[EnglishProductName], SUM(f.SalesAmount) AS v FROM dbo.factinternetsales f JOIN dbo.dimproduct p ON f.ProductKey = p.ProductKey GROUP BY p.[EnglishProductName] ORDER BY v DESC;")
    assert sql.startswith("SELECT p.") and sql.endswith("LIMIT 3") and "dbo." not in sql and '"EnglishProductName"' in sql
    assert _query_in('{"code": "```sql\\nSELECT 1\\n```"}') == "SELECT 1"
    assert _query_in("text then ```sql\nSELECT 2\n```") == "SELECT 2"
    assert _query_in('{"sql": "SELECT 3"}') == "SELECT 3"
    assert _query_in("no query here") is None


# -------------------------------------------------------------- the loop --


def test_review_runs_end_to_end_with_a_scripted_agent():
    executor = _duckdb_executor()
    schema = _schemas()[0]
    snapshot = _snapshot()

    def ask(question: str):
        if question.endswith("combined?"):
            return AgentAnswer("2012: $100.00; 2013: $375.00")  # internet only: narrower scope
        if question.startswith("Which 3 products"):
            return AgentAnswer("Road-350 2,000 then Mountain-200 2,500 then Sport Helmet 50", query="SELECT TOP 3 a.EnglishProductName, SUM(f.SalesAmount) AS value FROM dbo.factinternetsales f JOIN dbo.dimdate d ON f.OrderDateKey = d.DateKey JOIN dbo.dimproduct a ON f.ProductKey = a.ProductKey WHERE d.CalendarYear = 2013 GROUP BY a.EnglishProductName ORDER BY value DESC", language="sql")
        if "orders were there per year" in question:
            return AgentAnswer("I cannot determine orders from the selected tables.")
        return AgentAnswer("The values are 1,100.00 for 2012 and 2,850.00 for 2013; 100.00 and 350.00; 1,000.00 and 2,500.00; 2,750.00 and 350.00; 2,500 and 1,000 and 2,000 and 350 and 500 and 50")

    report = review_agent(snapshot, [schema], {LAKEHOUSE_ID: executor}, ask, top=3, limit_per_source=12)
    outcomes = {g.question_id: g for g in report.graded}
    combined = outcomes[report.questions[0].id]
    assert combined.outcome == "wrong" and combined.cause == "narrower_scope"
    top = next(g for q, g in outcomes.items() if next(x for x in report.questions if x.id == q).kind == "top_n")
    assert top.query_checked  # the agent's SQL ran locally and its rows were compared
    abstained = next(g for q, g in outcomes.items() if next(x for x in report.questions if x.id == q).kind == "distinct_by_year")
    assert abstained.outcome == "abstained"
    markdown = report.to_markdown()
    assert "## Findings" in markdown and "## Evaluation" in markdown and "## Suggested changes" in markdown
    assert "narrower_scope" in markdown and "schema_in_agent_instructions" in markdown
    suggestions = report.suggestions
    assert "factinternetsales and factresellersales" not in suggestions.agent_instructions.split("BEHAVIOUR")[0]
    assert "## Schema notes (moved from agent instructions)" in suggestions.datasource_instructions[LAKEHOUSE_ID]
    assert "aggregate each separately and combine" in suggestions.agent_instructions
    shots = suggestions.fewshots[LAKEHOUSE_ID]
    assert 1 <= len(shots) <= 4 and all(shot.query.startswith("SELECT") for shot in shots)
    assert any("UNION ALL" in shot.query for shot in shots)


def test_repetitions_surface_inconsistency():
    executor = _duckdb_executor()
    schema = _schemas()[0]
    answers = iter(["2012: $1,100.00; 2013: $2,875.00", "2012: $100.00; 2013: $375.00"] * 20)
    report = review_agent(_snapshot(), [schema], {LAKEHOUSE_ID: executor}, lambda q: next(answers), top=3, limit_per_source=1, repetitions=2)
    (graded,) = report.graded
    assert graded.cause == "inconsistent" and "2 runs" in graded.detail


def test_suggestions_propose_descriptions_and_join_paths_for_models():
    snapshot = _snapshot(second_source=True)
    schemas = _schemas(second_source=True)
    findings = diagnose(snapshot, schemas)
    suggestions = suggest(snapshot, schemas, findings, (), (), ())
    assert MODEL_ID in suggestions.datasource_descriptions
    assert "## Join Paths" in suggestions.datasource_instructions[MODEL_ID]
    assert "## Topics" in suggestions.agent_instructions


def test_apply_and_sdk_reader_use_the_management_surface():
    calls = []

    class Writer:
        def update_agent_instructions(self, text):
            calls.append(("agent", text[:10]))

        def update_datasource_instructions(self, sid, text):
            calls.append(("instructions", sid))

        def update_datasource_description(self, sid, text):
            calls.append(("description", sid))

        def add_fewshots(self, sid, shots):
            calls.append(("fewshots", sid, len(shots)))

    snapshot = _snapshot(second_source=True)
    schemas = _schemas(second_source=True)
    suggestions = suggest(snapshot, schemas, diagnose(snapshot, schemas), (), (), ())
    applied = apply_suggestions(Writer(), suggestions)
    assert ("description", MODEL_ID) in calls and applied[0] == "agent instructions"


def test_semantic_executor_and_profile_schema(tmp_path: Path):
    frames = []

    def aggregate(measures, *, groupby=None, filters=None, order_by=None, top=None):
        frames.append((measures, groupby, filters, order_by, top))
        return [{"Date[Year]": 2013, "Total Sales": 10.0}]

    rows = SemanticModelExecutor(aggregate).run({"measures": ["Total Sales"], "groupby": ["'Date'[Year]"], "filters": {"'Date'[Year]": [2013]}})
    assert rows == [{"Date[Year]": 2013, "Total Sales": 10.0}] and frames[0][4] is None

    deltalake = pytest.importorskip("deltalake")
    pyarrow = pytest.importorskip("pyarrow")
    from fabric_rlm import RLM
    from fabric_rlm.lakehouse import LakehouseSource

    tickets = tmp_path / "tickets"
    deltalake.write_deltalake(str(tickets), pyarrow.table({"ticket_id": [1, 2], "region": ["EU", "US"], "hours": [2.0, 3.0]}))
    table = deltalake.DeltaTable(str(tickets), without_files=True)
    source = LakehouseSource("file:///svc", catalog=[{"kind": "delta", "name": "tickets", "path": str(tickets), "version": table.version(), "table_id": table.metadata().id, "columns": [["ticket_id", "BIGINT"], ["region", "VARCHAR"], ["hours", "DOUBLE"]]}])
    knowledge = RLM.learn(sources={"service": source})
    schema = schema_from_profile(knowledge.package.sources[0], source_id="ds-9")
    assert schema.source_id == "ds-9" and schema.kind == "lakehouse" and schema.tables == {"tickets": ("hours", "region", "ticket_id")}


# ------------------------------------------------ scope and false positives --


def test_questions_stay_within_the_facts_the_instructions_name():
    # the lakehouse also holds a finance fact the agent never mentions; the
    # review must not evaluate the agent on tables outside its declared scope
    tables = dict(TABLES)
    tables["factfinance"] = ("FinanceKey", "DateKey", "OrganizationKey", "Amount")
    tables["dimorganization"] = ("OrganizationKey", "OrganizationName")
    schema = schema_from_tables(LAKEHOUSE_ID, tables)
    questions = generate_questions(_snapshot(), [schema], years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=20)
    assert questions and all("factfinance" not in q.technical for q in questions)
    unnamed = AgentSnapshot(agent_id="a", name="Bare", instructions="Answer questions.", datasources=(AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="lh", instructions="", description="d"),))
    everything = generate_questions(unnamed, [schema], years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=40)
    assert any("factfinance" in q.technical for q in everything)


def test_plain_words_are_not_unknown_references():
    source = AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="lh", instructions="FACT TABLES\n- dbo.factinternetsales joins the DIMENSION tables. Both facts share ProductKey.\n- Use vw_sales_flat for rollups.", description="d")
    snapshot = AgentSnapshot(agent_id="a", name="Words", instructions="Answer.", datasources=(source,))
    findings = {f.code: f for f in diagnose(snapshot, _schemas())}
    unknown = findings["unknown_reference"]
    assert "vw_sales_flat" in unknown.message
    assert "DIMENSION" not in unknown.message and "facts" not in unknown.message


def test_single_year_questions_read_naturally_and_no_data_answers_are_wrong():
    questions = generate_questions(_snapshot(), _schemas(), years={LAKEHOUSE_ID: [2013]}, top=3)
    assert all("2013 to 2013" not in q.technical for q in questions) and any("for 2013" in q.technical for q in questions)
    no_rows = Reference("q", "abstained", note="the query returned no rows")
    assert grade(_question(), no_rows, "There are no records for that period.").outcome == "correct"
    invented = grade(_question(), no_rows, "Total revenue was $1,250,000 in that period.")
    assert invented.outcome == "wrong" and invented.cause == "answered_without_data"
    assert grade(_question(), no_rows, "Here is the breakdown.").outcome == "incomplete"


# --------------------------------------------- SDK layers, Responses, routing --


def test_sdk_reader_prefers_the_public_layer_and_falls_back_to_legacy_keys():
    class PublicHandle:
        _id = "ds-1"

        def get_configuration(self, stage="staging"):
            assert stage == "staging"
            return {"id": "ds-1", "type": "LakehouseTables", "displayName": "AWLakehouse", "instructions": "Use dbo tables.", "description": "Sales.", "lakehouseReference": {"itemId": "item-1", "workspaceId": "ws-2"}}

        def get_fewshots(self, stage="staging"):
            return {"value": [{"id": "f1", "question": "How many orders?", "query": "SELECT COUNT(*) FROM factinternetsales"}]}

    class PublicManagement:
        data_agent_name = "Sales Agent"
        data_agent_id = "agent-1"
        workspace_id = "ws-1"

        def get_settings(self, stage="staging"):
            return {"aiInstructions": "Answer sales questions."}

        def list_datasources(self, stage="staging"):
            return [PublicHandle()]

        def get_configuration(self):  # the legacy layer must not be touched when the public one exists
            raise AssertionError("legacy layer used")

    snap = SdkAgentReader(PublicManagement()).snapshot()
    source = snap.datasources[0]
    assert snap.instructions == "Answer sales questions." and snap.workspace_id == "ws-1" and snap.stage == "staging"
    assert source.kind == "lakehouse" and source.name == "AWLakehouse" and source.instructions == "Use dbo tables." and source.description == "Sales."
    assert source.item_id == "item-1" and source.workspace_id == "ws-2" and source.fewshots[0].question == "How many orders?"

    class LegacyFrame:
        def to_dict(self, orient="records"):
            return [{"Id": "f1", "Question": "How many orders?", "Query": "SELECT 1", "State": "ok"}]

    class LegacyDatasource:
        _id = "item-1"

        def get_configuration(self):
            return {"id": "item-1", "type": "lakehouse_tables", "display_name": "AWLakehouse", "additional_instructions": "Use dbo tables.", "user_description": "Sales."}

        def get_fewshots(self):
            return LegacyFrame()

    class LegacyManagement:
        data_agent_name = "Sales Agent"
        data_agent_id = "agent-1"

        def get_configuration(self):
            return type("Config", (), {"instructions": "Answer sales questions."})()

        def get_datasources(self):
            return [LegacyDatasource()]

    legacy = SdkAgentReader(LegacyManagement()).snapshot()
    source = legacy.datasources[0]
    assert legacy.instructions == "Answer sales questions." and source.name == "AWLakehouse" and source.kind == "lakehouse"
    assert source.instructions == "Use dbo tables." and source.description == "Sales." and source.item_id == "item-1" and source.workspace_id is None
    assert source.fewshots[0].query == "SELECT 1"


def test_responses_asker_reads_the_answer_query_and_routing_from_the_output_items():
    from fabric_rlm.data_agent_review import AgentStep, Graded, ResponsesAgentAsker, ReviewReport, _steps_summary

    class Responses:
        retrieved = 0

        def create(self, **kwargs):
            assert kwargs["input"] == "How many orders?"
            return {"id": "resp-1", "status": "in_progress", "output": []}

        def retrieve(self, response_id):
            self.retrieved += 1
            return {
                "id": response_id,
                "status": "completed",
                "output": [
                    {"type": "function_call", "name": "query_lakehouse", "arguments": '{"datasource_name": "AWLakehouse", "natural_language_query": "How many orders?"}'},
                    {"type": "function_call_output", "output": "Executed:\n```sql\nSELECT COUNT(DISTINCT SalesOrderNumber) AS orders FROM dbo.factinternetsales\n```\nrows: 1"},
                    {"type": "message", "content": [{"type": "output_text", "text": "There were 27,659 orders."}]},
                ],
            }

    class Client:
        responses = Responses()

    answer = ResponsesAgentAsker(Client())("How many orders?", poll_seconds=0)
    assert answer.text == "There were 27,659 orders." and answer.status == "completed" and Client.responses.retrieved == 1
    assert answer.query == "SELECT COUNT(DISTINCT SalesOrderNumber) AS orders FROM dbo.factinternetsales" and answer.language == "sql"
    assert answer.datasource == "AWLakehouse" and [s.kind for s in answer.steps] == ["function_call", "function_call_output"]

    # a natural-language "query" argument is not the executed query; the fenced DAX in the output is
    query, language, datasource = _steps_summary([
        AgentStep("function_call", "query_semantic_model", '{"query": "top products", "artifact_name": "Sales Model"}', ""),
        AgentStep("function_call_output", "", "", "```dax\nEVALUATE TOPN(3, VALUES(Product[Name]))\n```"),
    ])
    assert language == "dax" and query.startswith("EVALUATE") and datasource == "Sales Model"
    assert _query_in('{"query": "top products"}') is None

    snapshot, schemas = _snapshot(), _schemas()
    question = _question()
    report = ReviewReport(snapshot, tuple(schemas), (), (question,), (Reference("q", "failed"),), {"q": (answer,)}, (Graded("q", "wrong", "values_differ", "x"),), suggest(snapshot, schemas, (), (), (), ()))
    text = report.to_markdown()
    assert "| routed to |" in text and "| AWLakehouse |" in text and "## Agent run steps" in text and "query_lakehouse" in text


def test_failed_answers_from_another_source_are_misrouted():
    from fabric_rlm.data_agent_review import Graded, _with_routing

    snapshot = _snapshot(second_source=True)
    question = _question()
    wrong = Graded(question.id, "wrong", "values_differ", "no figure matched")
    routed = _with_routing(wrong, question, [AgentAnswer("12", datasource="Sales Model")], snapshot)
    assert routed.cause == "misrouted" and "Sales Model" in routed.detail and routed.outcome == "wrong"
    assert _with_routing(wrong, question, [AgentAnswer("12", datasource="AWLakehouse")], snapshot).cause == "values_differ"
    assert _with_routing(wrong, question, [AgentAnswer("12", datasource="Sales Model")], _snapshot()).cause == "values_differ"
    correct = Graded(question.id, "correct")
    assert _with_routing(correct, question, [AgentAnswer("12", datasource="Sales Model")], snapshot).cause == ""


def test_grouping_attributes_span_dimensions_and_prefer_english_names():
    tables = dict(TABLES)
    tables["dimproduct"] = ("ProductKey", "SpanishProductName", "EnglishProductName", "FrenchProductName", "ProductSubcategoryKey")
    schema = schema_from_tables(LAKEHOUSE_ID, tables)
    questions = generate_questions(_snapshot(), [schema], years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=40)
    texts = [q.technical for q in questions]
    assert any("EnglishProductName" in t for t in texts) and any("SalesTerritoryRegion" in t for t in texts)
    assert not any("SpanishProductName" in t or "FrenchProductName" in t for t in texts)


def test_sdk_writer_uses_the_public_layer_when_present():
    from fabric_rlm.data_agent_review import SdkAgentWriter

    calls = []

    class Handle:
        _id = "ds-1"

        def update_configuration(self, instructions=None, description=None):
            calls.append(("configure", instructions, description))
            return {}

        def add_fewshots(self, shots):
            calls.append(("fewshots", dict(shots)))
            return []

    class Management:
        def update_settings(self, ai_instructions=None):
            calls.append(("settings", ai_instructions))
            return {}

        def list_datasources(self, stage="staging"):
            return [Handle()]

    writer = SdkAgentWriter(Management())
    writer.update_agent_instructions("Route by topic.")
    writer.update_datasource_instructions("ds-1", "## Schema notes")
    writer.update_datasource_description("ds-1", "Sales facts.")
    writer.add_fewshots("ds-1", [FewShot("", "How many orders?", "SELECT 1")])
    assert calls == [("settings", "Route by topic."), ("configure", "## Schema notes", None), ("configure", None, "Sales facts."), ("fewshots", {"How many orders?": "SELECT 1"})]


def test_guid_lakehouse_roots_keep_their_segments_for_delta_rs():
    from fabric_rlm.lakehouse import _delta_rs_path

    guid = "abfss://11111111-2222-3333-4444-555555555555@onelake.dfs.fabric.microsoft.com/66666666-7777-8888-9999-000000000000/Tables/dimdate"
    assert _delta_rs_path(guid) == guid
    named = "abfss://ws@onelake.dfs.fabric.microsoft.com/AWLakehouse/Tables/dimdate"
    assert _delta_rs_path(named) == "abfss://ws@onelake.dfs.fabric.microsoft.com/AWLakehouse.Lakehouse/Tables/dimdate"


# --------------------------------------------------------- selected elements --


def test_selected_tables_come_from_the_elements_tree():
    from fabric_rlm.data_agent_review import _selected_table_paths

    tree = {
        None: [
            {"id": "s1", "displayName": "dbo", "type": "Schema", "isSelected": False},
            {"id": "s2", "displayName": "staging", "type": "Schema", "isSelected": False},
        ],
        "s1": [
            {"id": "t1", "displayName": "factinternetsales", "type": "LakehouseTable", "isSelected": True},
            {"id": "t2", "displayName": "dimemployee", "type": "LakehouseTable", "isSelected": False},
        ],
        "s2": [{"id": "t3", "displayName": "raw_orders", "type": "LakehouseTable", "isSelected": True}],
        "t1": [{"id": "c1", "displayName": "SalesAmount", "type": "Column", "isSelected": True}],
    }
    fetched = []

    def fetch(root_id, token):
        fetched.append(root_id)
        items = tree.get(root_id, [])
        if root_id is None and token is None:
            return {"value": items[:1], "continuationToken": "next"}  # the root listing is paged
        if root_id is None:
            return {"value": items[1:]}
        return {"value": items}

    assert _selected_table_paths(fetch) == ["dbo/factinternetsales", "staging/raw_orders"]
    assert "t1" not in fetched  # a table's columns are not walked


def test_sdk_reader_records_the_selected_tables_and_the_report_shows_them():
    from fabric_rlm.data_agent_review import ReviewReport

    class Handle:
        _id = "ds-1"

        def get_configuration(self, stage="staging"):
            return {"id": "ds-1", "type": "LakehouseTables", "displayName": "AWLakehouse", "instructions": "", "description": "Sales.", "lakehouseReference": {"itemId": "item-1", "workspaceId": "ws-2"}}

        def get_fewshots(self, stage="staging"):
            return {"value": []}

        def get_elements(self, stage="staging", root_id=None, continuation_token=None):
            if root_id is None:
                return {"value": [{"id": "t1", "displayName": "factinternetsales", "type": "Table", "isSelected": True}, {"id": "t2", "displayName": "dimdate", "type": "Table", "isSelected": True}, {"id": "t3", "displayName": "dimemployee", "type": "Table", "isSelected": False}]}
            return {"value": []}

    class Management:
        data_agent_name = "Sales Agent"
        data_agent_id = "agent-1"
        workspace_id = "ws-1"

        def get_settings(self, stage="staging"):
            return {"aiInstructions": "Answer."}

        def list_datasources(self, stage="staging"):
            return [Handle()]

    snap = SdkAgentReader(Management()).snapshot()
    assert snap.datasources[0].selected_tables == ("factinternetsales", "dimdate")

    schemas = [schema_from_tables("ds-1", {"factinternetsales": TABLES["factinternetsales"], "dimdate": TABLES["dimdate"]})]
    report = ReviewReport(snap, tuple(schemas), (), (), (), {}, (), suggest(snap, schemas, (), (), (), ()))
    assert "2 tables; 2 tables selected for the agent" in report.to_markdown()


# ------------------------------------------------------------- diagnostics --


def test_english_words_that_match_columns_are_not_schema_mentions():
    tables = dict(TABLES)
    tables["dimcustomer"] = ("CustomerKey", "GeographyKey", "FirstName", "Phone", "Status")
    schema = schema_from_tables(LAKEHOUSE_ID, tables)
    instructions = (
        "RULES\n"
        "- Do not list phone numbers or order status.\n"
        "- State the channel and date range used.\n"
        "- Revenue means SUM(SalesAmount) across factinternetsales and factresellersales.\n"
    )
    snapshot = AgentSnapshot(agent_id="a", name="Words", instructions=instructions, datasources=(AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="lh", instructions="Use dbo tables.", description="d"),))
    findings = {f.code: f for f in diagnose(snapshot, [schema])}
    moved = findings["schema_in_agent_instructions"]
    assert len(moved.evidence) == 1 and "SalesAmount" in moved.evidence[0]
    assert not any("phone" in line.casefold() or "channel" in line.casefold() for line in moved.evidence)

    suggestions = suggest(snapshot, [schema], tuple(findings.values()), (), (), ())
    text = suggestions.datasource_instructions[LAKEHOUSE_ID]
    assert "- Revenue means SUM(SalesAmount)" in text and "- - " not in text


def test_no_questions_is_explained_and_year_discovery_failures_are_reported():
    from fabric_rlm.data_agent_review import explain_no_questions

    def broken(sql, *, sources):
        raise RuntimeError("delta_scan failed on a void column")

    executors = {LAKEHOUSE_ID: LakehouseExecutor(broken, TABLES)}
    asked = []
    report = review_agent(_snapshot(), _schemas(), executors, lambda q: asked.append(q) or "no")
    assert not report.questions and not asked
    assert any("year discovery failed: RuntimeError: delta_scan failed" in note for note in report.notes)
    assert any("no complete years were discovered" in note for note in report.notes)
    assert "Diagnostics:" in report.to_markdown()

    no_date = schema_from_tables(LAKEHOUSE_ID, {k: v for k, v in TABLES.items() if k != "dimdate"})
    notes = explain_no_questions(_snapshot(), [no_date], {})
    assert any("factinternetsales has no time axis" in note for note in notes)
    bare = AgentSnapshot(agent_id="a", name="Bare", instructions="Answer.", datasources=(AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="lh", instructions="", description="d"),))
    dims_only = schema_from_tables(LAKEHOUSE_ID, {k: v for k, v in TABLES.items() if not k.startswith("fact")})
    assert any("no fact table recognised" in note for note in explain_no_questions(bare, [dims_only], {}))


# ------------------------------------------------------------------ context --


def test_context_scopes_prioritises_and_leads_with_supplied_questions():
    from fabric_rlm.data_agent_review import ReviewContext

    context = ReviewContext(
        scope="Reseller sales through factresellersales",
        priorities=("factresellersales",),
        definitions={"Gross margin": "SUM(SalesAmount - TotalProductCost)"},
        questions=(
            ("How many internet sales rows are there?", "SELECT COUNT(*) AS n FROM dbo.factinternetsales"),
            ("Revenue by year?", "Revenue was 1,100 in 2012 and 2,850 in 2013"),
            ("Should it refuse?", "The agent declines politely"),
        ),
        notes="2014 is partial",
    )
    assert context and "Priorities: factresellersales" in context.as_prompt() and "Notes: 2014 is partial" in context.as_prompt()
    assert not ReviewContext()

    questions = generate_questions(_snapshot(), _schemas(), years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=4, context=context)
    assert [q.kind for q in questions[:3]] == ["supplied"] * 3 and questions[0].id.endswith(".u1")
    generated = [q for q in questions if q.kind != "supplied"]
    assert len(generated) == 4 and all("factresellersales" in q.technical for q in generated)

    plain = generate_questions(_snapshot(), _schemas(), years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=4)
    assert len(plain) == 4 and not all("factresellersales" in q.technical for q in plain)

    executor = _duckdb_executor()
    references = {r.question_id: r for r in build_references(questions[:3], {LAKEHOUSE_ID: executor})}
    assert references[questions[0].id].status == "ok" and references[questions[0].id].rows[0]["n"] > 0
    figures = [float(r["value"]) for r in references[questions[1].id].rows]
    assert references[questions[1].id].status == "ok" and 1100.0 in figures and 2850.0 in figures
    assert references[questions[2].id].status == "failed"

    declared = declared_from_snapshot(_snapshot(), _schemas(), context=context)
    assert declared[LAKEHOUSE_ID]["definitions"]["Gross margin"] == "SUM(SalesAmount - TotalProductCost)"

    report = review_agent(_snapshot(), _schemas(), {LAKEHOUSE_ID: executor}, lambda q: "no idea", years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=2, context=context)
    markdown = report.to_markdown()
    assert "## Scope and context" in markdown and "Supplied questions: 3" in markdown and report.questions[0].kind == "supplied"


# ------------------------------------------------------------------- policy --


def test_out_of_scope_topics_are_not_asked_about_and_declines_on_them_are_policy():
    from fabric_rlm.data_agent_review import ReviewContext, _with_policy, excluded_tables, excluded_terms, Graded

    line = "- Do not claim order status, cancellations, returns, inventory, promotions, quotas, reviews, or profitability because those tables or required business definitions are not in scope."
    terms = excluded_terms(line)
    assert {"promotion", "quota", "inventory", "return", "review"} <= terms
    assert not {"table", "business", "definition", "scope", "claim"} & terms
    assert excluded_terms("- Do not combine customer and reseller identities.") == set()

    tables = dict(TABLES)
    tables["factinternetsales"] = TABLES["factinternetsales"] + ("PromotionKey",)
    tables["dimpromotion"] = ("PromotionKey", "EnglishPromotionCategory")
    tables["factsalesquota"] = ("SalesQuotaKey", "DateKey", "SalesAmountQuota", "CalendarYear")
    schema = schema_from_tables(LAKEHOUSE_ID, tables)
    with_policy = AgentSnapshot(agent_id="a", name="Sales Agent", instructions=AGENT_INSTRUCTIONS + "\n" + line, datasources=_snapshot().datasources)
    assert excluded_tables(schema, with_policy.instructions) == ["dimpromotion", "factsalesquota"]

    questions = generate_questions(with_policy, [schema], years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=40)
    assert questions and not any("Promotion" in q.technical or "quota" in q.technical.casefold() for q in questions if q.kind != "scope_out")
    without = generate_questions(_snapshot(), [schema], years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=40)
    assert any("EnglishPromotionCategory" in q.technical for q in without)

    question = Question("q", LAKEHOUSE_ID, "top_n", "What are the top 3 EnglishPromotionCategory values by SalesAmount?", {}, "", {"sql": ""})
    declined = Graded("q", "abstained", "agent_abstained", "the agent declined a question the source answers")
    generated_only = [AgentAnswer("This is out of scope.", query="SELECT 1", executed=False)]
    graded = _with_policy(declined, question, generated_only, terms)
    assert graded.cause == "abstained_by_policy" and "not executed" in graded.detail
    assert _with_policy(declined, question, generated_only, set()).cause == "agent_abstained"

    report = review_agent(with_policy, [schema], {LAKEHOUSE_ID: _duckdb_executor()}, lambda q: "no", years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=2)
    assert any("out of scope by the instructions, not asked about: dimpromotion, factsalesquota" in note for note in report.notes)

    context = ReviewContext(scope="Reseller performance by product and territory")
    assert {"reseller", "performance", "product", "territory"} <= set(context.terms) and context.ranking_terms == context.terms
    ranked = generate_questions(_snapshot(), _schemas(), years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=3, context=context)
    assert all("factresellersales" in q.technical for q in ranked)


def test_a_change_answer_is_right_by_the_change_or_by_both_totals_and_fewshots_use_dbo():
    question = Question("q", LAKEHOUSE_ID, "yoy", "What was the year-over-year change?", {}, "", {"sql": ""})
    reference = Reference("q", "ok", ({"year": 2012, "value": 1100.0}, {"year": 2013, "value": 2850.0}))
    assert grade(question, reference, "Sales grew from $1,100.00 in 2012 to $2,850.00 in 2013.").outcome == "correct"
    assert grade(question, reference, "Sales rose 159.1% year over year.").outcome == "correct"
    assert grade(question, reference, "Sales rose by 1,750 year over year.", agent_rows=[{"change": 1750.0}]).outcome == "correct"
    assert grade(question, reference, "Sales rose 50% year over year.", agent_rows=[{"change_pct": 50.0}]).outcome == "wrong"

    questions = generate_questions(_snapshot(), _schemas(), years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=4)
    total = next(q for q in questions if q.kind == "total_by_year")
    assert "FROM dbo.factinternetsales" in total.reference_query and "JOIN dbo.dimdate" in total.reference_query
    assert "dbo." not in total.execution["sql"]
    bare = AgentSnapshot(agent_id="a", name="Bare", instructions="Answer.", datasources=(AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="lh", instructions="Use the tables.", description="d"),))
    plain = generate_questions(bare, _schemas(), years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=4)
    assert "dbo." not in plain[0].reference_query


def test_answers_know_whether_their_query_ran():
    from fabric_rlm.data_agent_review import AgentStep, Graded, ReviewReport, _executed

    assert _executed(()) is True
    assert _executed((AgentStep("function_call", "analyze.database.nl2code"), AgentStep("function_call", "analyze.database.execute"))) is True
    assert _executed((AgentStep("function_call", "analyze.database.nl2code"), AgentStep("function_call", "trace.analyze_lakehouse_tables"))) is False

    snapshot, schemas = _snapshot(), _schemas()
    answer = AgentAnswer("I cannot answer that.", query="SELECT 1", language="sql", steps=(AgentStep("function_call", "analyze.database.nl2code"), AgentStep("function_call", "analyze.database.nl2code")), executed=False)
    report = ReviewReport(snapshot, tuple(schemas), (), (_question(),), (Reference("q", "ok", ({"year": 2012, "value": 1.0},)),), {"q": (answer,)}, (Graded("q", "abstained", "agent_abstained", "x"),), suggest(snapshot, schemas, (), (), (), ()))
    text = report.to_markdown()
    assert "analyze.database.nl2code x2" in text and "executed: no" in text


# ------------------------------------------------------- html, learned, deep --


def test_html_report_escapes_and_lists_every_section():
    from types import SimpleNamespace

    from fabric_rlm.data_agent_review import AgentStep, Analysis, Graded, ReviewContext, ReviewReport, summarize_knowledge

    package = SimpleNamespace(
        package_id="pkg-1",
        sources=(
            SimpleNamespace(
                source_id=LAKEHOUSE_ID,
                family="lakehouse",
                status="active",
                role="numeric_evidence",
                schema={"factinternetsales": {"columns": {"SalesAmount": {"type": "double"}}}, "dimdate": {"columns": {"DateKey": {"type": "bigint"}, "CalendarYear": {"type": "int"}}}},
                schema_fingerprint="abcdef0123456789",
                snapshot_fingerprint="0123456789abcdef",
                sensitive_columns=("dimcustomer.EmailAddress",),
                diagnostics={"tables": 2},
            ),
        ),
        operations=(SimpleNamespace(operation="lakehouse.aggregate", required_sources=(LAKEHOUSE_ID,), status="active", grain="row", parameter_schema={"measure": {}, "catalog_source": {}}),),
        lessons=(SimpleNamespace(kind="declared_definition", subject="Revenue", status="active", confidence="high", structured_rule={"definition": "SUM(SalesAmount)"}, basis=("declared",)),),
        events=(SimpleNamespace(event_type="profile.created"), SimpleNamespace(event_type="profile.created")),
        evidence=(),
    )
    summary = summarize_knowledge(SimpleNamespace(package=package))
    assert summary["sources"][0]["shape"] == "2 tables, 3 columns" and summary["operations"][0]["operation"] == "lakehouse.aggregate"
    assert summary["lessons"][0]["subject"] == "Revenue" and summary["events"] == {"profile.created": 2}

    snapshot, schemas = _snapshot(), _schemas()
    question = Question("q", LAKEHOUSE_ID, "top_n", "Top <b>3</b> products?", {}, "SELECT TOP 3 x FROM t", {"sql": ""})
    answer = AgentAnswer("Mountain-200 & Road-350", query="SELECT 1", language="sql", datasource="AWLakehouse", steps=(AgentStep("function_call", "analyze.database.execute"),))
    report = ReviewReport(
        snapshot,
        tuple(schemas),
        diagnose(snapshot, schemas),
        (question,),
        (Reference("q", "ok", ({"EnglishProductName": "Mountain-200", "value": 2500.0},)),),
        {"q": (answer,)},
        (Graded("q", "partial", "missing_rows", "one absent"),),
        suggest(snapshot, schemas, (), (), (), ()),
        notes=("AWLakehouse: complete years [2012, 2013]",),
        context=ReviewContext(scope="Sales by product"),
        knowledge=summary,
        analysis=(Analysis("q", "Top <b>3</b> products?", "The agent ranked by quantity.", "Add: rank by SalesAmount."),),
    )
    page = report.to_html()
    assert page.startswith("<style>") and "Top &lt;b&gt;3&lt;/b&gt; products?" in page and "Mountain-200 &amp; Road-350" in page
    for marker in (
        "<h2>Scope and context",
        "<h2>Sources</h2>",
        "<h2>Findings</h2>",
        "sev-medium",
        "<h2>Evaluation</h2>",
        'class="out-partial"',
        "<details>",
        "Reference rows",
        "<h2>Analysis (RLM)</h2>",
        "Add: rank by SalesAmount.",
        "<h2>What the RLM learned</h2>",
        "lakehouse.aggregate",
        "<h2>Suggested changes</h2>",
        "<h2>Method</h2>",
    ):
        assert marker in page, marker
    markdown = report.to_markdown()
    assert "## What the RLM learned" in markdown and "## Analysis (RLM)" in markdown and "declared_definition on Revenue" in markdown


def test_deepen_adds_verified_questions_and_explanations_with_fakes():
    from fabric_rlm.data_agent_review import deepen

    executor = _duckdb_executor()
    base = review_agent(_snapshot(), _schemas(), {LAKEHOUSE_ID: executor}, lambda q: "no idea", years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=2)
    briefs = []

    def propose(brief, count):
        briefs.append(brief)
        return ["What was internet revenue in 2013?", "How many customers bought in 2013?", "Unverifiable?"][:count]

    def verify(question):
        if "revenue" in question:
            return "agree", "Internet revenue in 2013 was $2,850.00."
        if "customers" in question:
            return "reconciled", "There were 3 customers."
        return "failed", "the analysts disagreed"

    def ask(question):
        return "It was 2,850 dollars." if "revenue" in question else "I do not know."

    explained = []

    def explain(case):
        explained.append(case["question"])
        return {"explanation": f"Explanation for {case['question']}", "proposed_change": "Add a few-shot."}

    deeper = deepen(base, questions=3, ask=ask, propose=propose, verify=verify, explain=explain)
    assert len(deeper.questions) == len(base.questions) + 3 and [q.kind for q in deeper.questions[-3:]] == ["deep"] * 3
    assert "Schema digest:" in briefs[0] and "Already asked:" in briefs[0]
    by_id = {g.question_id: g for g in deeper.graded}
    d1, d2, d3 = [q.id for q in deeper.questions[-3:]]
    assert by_id[d1].outcome == "correct"
    assert by_id[d2].outcome in {"wrong", "incomplete", "abstained"}
    assert by_id[d3].outcome == "no_reference"
    explained_ids = {a.question_id for a in deeper.analysis}
    assert d2 in explained_ids and d3 not in explained_ids and all(a.explanation.startswith("Explanation for") for a in deeper.analysis)
    assert "## Analysis (RLM)" in deeper.to_markdown() and "Add a few-shot." in deeper.to_html()
    with pytest.raises(ValueError):
        deepen(base, questions=1)


def test_emphasised_scope_terms_rank_first_and_questions_are_renumbered():
    from fabric_rlm.data_agent_review import ReviewContext

    context = ReviewContext(scope="Internet and reseller sales by product and territory. Focus is on reseller performance.")
    assert "reseller" in context.emphasised and "internet" not in context.emphasised
    questions = generate_questions(_snapshot(), _schemas(), years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=4, context=context)
    assert [q.id.rsplit(".", 1)[-1] for q in questions] == ["q1", "q2", "q3", "q4"]
    assert all("factresellersales" in q.technical for q in questions)


def test_executor_retries_a_timed_out_query_and_the_verifier_does_not_repeat_bound_inputs(monkeypatch):
    import fabric_rlm.verify as verify_module
    from fabric_rlm.data_agent_review import _rlm_verifier

    calls = []

    def flaky(sql, *, sources, timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise TimeoutError("deadline")
        return {"columns": ["n"], "rows": [[3]], "truncated": False}

    executor = LakehouseExecutor(flaky, TABLES, timeout=120)
    assert executor.run({"sql": "SELECT COUNT(*) AS n FROM factinternetsales"}) == [{"n": 3}] and calls == [120, 120]

    def always(sql, *, sources):
        raise TimeoutError("x")

    with pytest.raises(TimeoutError):
        LakehouseExecutor(always, TABLES, retries=0).run({"sql": "SELECT 1"})

    seen = []

    class Result:
        payload = {"answer": "Total was 1,100."}

    def fake_verified_task(task, **kwargs):
        seen.append(kwargs)
        return type("Verified", (), {"result": Result(), "verdict": "agree"})()

    monkeypatch.setattr(verify_module, "verified_task", fake_verified_task)
    assert _rlm_verifier({"model": "x"}, {LAKEHOUSE_ID: object()}, knowledge=object(), max_turns=2, timeout=10)("q") == ("agree", "Total was 1,100.")
    assert seen[-1]["inputs"] is None
    _rlm_verifier({"model": "x"}, {LAKEHOUSE_ID: "handle"}, knowledge=None, max_turns=2, timeout=10)("q")
    assert seen[-1]["inputs"] == {LAKEHOUSE_ID: "handle"}


def test_a_prose_reference_may_say_more_than_the_agent_was_asked():
    question = Question("q", LAKEHOUSE_ID, "deep", "Which region led reseller sales in 2013?", {}, "", {"kind": "supplied_text", "text": "Southwest led with 6,377,499.28; Northwest 5,000,000.00; total 11,377,499.28"})
    reference = Reference("q", "ok", ({"value": 6377499.28}, {"value": 5000000.0}, {"value": 11377499.28}))
    assert grade(question, reference, "Southwest led with $6,377,499.28 in 2013.").outcome == "correct"
    assert grade(question, reference, "Southwest led with $6,377,499.28 and Northwest with $4,000,000.00.").outcome == "partial"
    assert grade(question, reference, "Southwest led with $1.00.").outcome == "wrong"


# ------------------------------------------------------- natural language --

AW_STYLE_INSTRUCTIONS = """SCOPE
- "Internet sales" or "online sales" uses factinternetsales only. "Reseller sales", "dealer sales", or "B2B sales" uses factresellersales only.
- Region means SalesTerritoryGroup; territory means SalesTerritoryRegion; country means SalesTerritoryCountry.
- An order is a distinct SalesOrderNumber, not a row count.
- Use SalesAmount for revenue, OrderQuantity for units, TotalProductCost for product cost.
- Revenue = SUM(SalesAmount).
- Average order value = SUM(SalesAmount) / COUNT(DISTINCT SalesOrderNumber).
- Do not claim promotions or quotas because those tables are not in scope.
"""


def test_vocabulary_comes_from_the_instructions():
    from fabric_rlm.data_agent_review import build_vocabulary, humanize_column, humanize_table

    assert humanize_table("factresellersales") == "reseller sales" and humanize_table("dimsalesterritory") == "sales territory"
    assert humanize_table("dbo.factinternetsales") == "internet sales" and humanize_table("code_registry") == "code registry"
    assert humanize_column("EnglishProductName") == "product" and humanize_column("SalesTerritoryRegion") == "sales territory region"
    assert humanize_column("BusinessType") == "business type" and humanize_column("'Product'[Category]") == "category"

    snapshot = AgentSnapshot(agent_id="a", name="Sales Agent", instructions=AW_STYLE_INSTRUCTIONS, datasources=_snapshot().datasources)
    vocabulary = build_vocabulary(snapshot, _schemas()[0])
    assert vocabulary.tables["factresellersales"] == "reseller sales" and vocabulary.tables["factinternetsales"] == "internet sales"
    assert vocabulary.abbreviations["factresellersales"] == "B2B" and vocabulary.abbreviations["factinternetsales"] == "B2C"
    assert vocabulary.measures["SalesAmount"] == "revenue" and vocabulary.measures["OrderQuantity"] == "units" and vocabulary.measures["TotalProductCost"] == "product cost"
    assert vocabulary.attributes["SalesTerritoryRegion"] == "territory" and vocabulary.attributes["SalesTerritoryGroup"] == "region"
    assert vocabulary.orders == "orders" and any(line.startswith("B2B = reseller sales") for line in vocabulary.lines())


def test_questions_read_as_a_business_user_asks_them_and_cover_the_skills():
    import re

    tables = dict(TABLES)
    tables["dimpromotion"] = ("PromotionKey", "EnglishPromotionCategory")
    tables["factsalesquota"] = ("SalesQuotaKey", "DateKey", "SalesAmountQuota", "CalendarYear")
    schema = schema_from_tables(LAKEHOUSE_ID, tables)
    snapshot = AgentSnapshot(agent_id="a", name="Sales Agent", instructions=AW_STYLE_INSTRUCTIONS, datasources=_snapshot().datasources)
    questions = generate_questions(snapshot, [schema], years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=40)
    texts = [q.text for q in questions]
    assert not any(re.search(r"fact\w+|dim\w+|SalesAmount|[a-z][A-Z]", t) for t in texts), texts
    for expected in (
        "What was total revenue by year from 2012 to 2013, internet sales and reseller sales combined?",
        "What was total revenue in 2013?",
        "How did reseller sales revenue change from 2012 to 2013?",
        "Which 3 territories had the highest reseller sales revenue in 2013?",
        "What was reseller sales revenue by product in 2013?",
        "How many reseller sales orders were placed in 2013?",
        "What were total units for reseller sales in 2013?",
        "What was the average order value for reseller sales in 2013?",
        "What was B2B revenue in 2013?",
        "Which promotions were used most in 2013?",
        "What was the sales quota for 2013?",
    ):
        assert expected in texts, expected
    skills = {q.skill for q in questions}
    assert {"aggregate", "rank", "change", "count", "kpi", "measure", "ambiguity", "scope"} <= skills
    definition = next(q for q in questions if q.kind == "definition")
    assert "SUM(f.SalesAmount) / COUNT(DISTINCT f.SalesOrderNumber) AS value" in definition.execution["sql"]
    assert all(q.technical for q in questions)


def test_declines_measures_and_row_counts_are_graded_by_their_cause():
    question = Question("s", LAKEHOUSE_ID, "scope_out", "Which promotions were used most in 2013?", {}, "", {"kind": "expect_decline"}, skill="scope")
    decline = Reference("s", "decline", note="out of scope")
    assert grade(question, decline, "I cannot answer questions about promotions; they are out of scope for this agent.").outcome == "correct"
    assert grade(question, decline, "The top promotion brought $1,250,000.").cause == "answered_out_of_scope"
    assert grade(question, decline, "Here is what I found.").outcome == "incomplete"

    units = Question("u", LAKEHOUSE_ID, "total_year", "What were total units for reseller sales in 2013?", {}, "", {"sql": ""}, (("measure:SalesAmount", {"sql": ""}),), skill="measure")
    reference = Reference("u", "ok", ({"value": 1500.0},), alternates={"measure:SalesAmount": ({"value": 2850.0},)})
    assert grade(units, reference, "Reseller sales came to $2,850.00 in 2013.").cause == "wrong_measure"
    orders = Question("o", LAKEHOUSE_ID, "distinct_orders_year", "How many reseller sales orders were placed in 2013?", {}, "", {"sql": ""}, (("rowcount", {"sql": ""}),), skill="count")
    reference = Reference("o", "ok", ({"value": 300.0},), alternates={"rowcount": ({"value": 1200.0},)})
    assert grade(orders, reference, "There were 1,200 orders.").cause == "row_count_not_distinct"


def test_the_review_derives_a_filter_question_and_reports_by_skill():
    executor = _duckdb_executor()
    report = review_agent(_snapshot(), _schemas(), {LAKEHOUSE_ID: executor}, lambda q: "no idea", years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=12)
    filtered = [q for q in report.questions if q.kind == "filter"]
    ranked = next(q for q in report.questions if q.kind == "top_n")
    top_row = next(r for r in report.references if r.question_id == ranked.id).rows[0]
    leader = top_row["EnglishProductName"]
    assert filtered and filtered[0].text == f"What was internet sales revenue for product {leader} in 2013?"
    reference = next(r for r in report.references if r.question_id == filtered[0].id)
    assert reference.status == "ok" and reference.rows[0]["value"] == float(top_row["value"])
    assert f"WHERE d.CalendarYear = 2013 AND a.EnglishProductName = '{leader}'" in filtered[0].reference_query
    assert "filter" in report.by_skill() and "Outcomes by skill:" in report.to_markdown() and "Outcomes by skill" in report.to_html()


# ------------------------------------------------------------------ drivers --


def _driver_lakehouse():
    """A reseller lakehouse with months, a snowflaked product category, four resellers and a visible December-to-January drop."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    months = [(2012, 10), (2012, 11), (2012, 12)] + [(2013, m) for m in range(1, 13)]
    con.execute("CREATE TABLE dimdate (DateKey INTEGER, CalendarYear INTEGER, MonthNumberOfYear INTEGER, CalendarQuarter INTEGER)")
    for year, month in months:
        con.execute("INSERT INTO dimdate VALUES (?, ?, ?, ?)", [year * 10000 + month * 100 + 1, year, month, (month - 1) // 3 + 1])
    for key, year, month in ((20131129, 2013, 11), (20131202, 2013, 12), (20140115, 2014, 1)):  # daily dates around Thanksgiving and a partial year
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

    def query(sql, *, sources):
        relation = con.execute(sql)
        return {"columns": [d[0] for d in relation.description], "rows": relation.fetchall(), "truncated": False}

    tables = {
        "factresellersales": ("ProductKey", "OrderDateKey", "SalesTerritoryKey", "ResellerKey", "SalesOrderNumber", "SalesAmount", "OrderQuantity", "TotalProductCost"),
        "dimdate": ("DateKey", "CalendarYear", "MonthNumberOfYear", "CalendarQuarter"),
        "dimproduct": TABLES["dimproduct"],
        "dimproductsubcategory": TABLES["dimproductsubcategory"],
        "dimproductcategory": TABLES["dimproductcategory"],
        "dimsalesterritory": TABLES["dimsalesterritory"],
        "dimreseller": TABLES["dimreseller"],
    }
    return LakehouseExecutor(query, tables), schema_from_tables(LAKEHOUSE_ID, tables)


def test_discovery_finds_names_periods_and_thresholds_and_drives_the_analytical_questions():
    from fabric_rlm.data_agent_review import attribute_paths, discover_drivers, _heuristic_joins

    executor, schema = _driver_lakehouse()
    snapshot = AgentSnapshot(agent_id="a", name="Sales Agent", instructions=AW_STYLE_INSTRUCTIONS, datasources=(AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="AWLakehouse", instructions="Use dbo.factresellersales for reseller sales. Product hierarchy: dimproduct.ProductSubcategoryKey -> dimproductsubcategory.ProductSubcategoryKey -> dimproductcategory.ProductCategoryKey", description="Reseller sales."),))
    paths = attribute_paths(schema, "factresellersales", _heuristic_joins(schema))
    assert any(p["column"] == "EnglishProductCategoryName" and len(p["hops"]) == 3 for p in paths)

    found = discover_drivers(executor, schema, snapshot, [2012, 2013])
    entry = found["factresellersales"]
    assert entry["errors"] == []
    assert entry["top"]["entity"] == ["Bike World", "Trail Co", "Pedal Shop"] and entry["top"]["category"] == ["Bikes", "Accessories"]
    assert entry["drop"]["before"] == {"year": 2012, "month": 12} and entry["drop"]["after"] == {"year": 2013, "month": 1} and entry["drop"]["delta"] == -2550.0
    assert entry["threshold"] == 1

    questions = generate_questions(snapshot, [schema], years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=60, discovered={LAKEHOUSE_ID: found})
    texts = [q.text for q in questions]
    for expected in (
        "How did reseller sales revenue move month by month in 2013?",
        "Which quarter of 2013 was the strongest for reseller sales revenue, and how did the quarters compare?",
        "Why did reseller sales revenue drop from December 2012 to January 2013, and which products drove the drop?",
        "How has Bike World performed year by year on reseller sales revenue?",
        "How is Bike World doing on Bikes in 2013?",
        "What share of 2013 reseller sales revenue did the top 10 resellers bring in?",
        "How many resellers placed more than 1 order in 2013?",
        "Which resellers bought in 2012 but not in 2013?",
        "How did Bike World and Trail Co compare on reseller sales revenue in 2013?",
        "What share of 2013 reseller sales revenue came from Bikes?",
        "For each territory, which product category led reseller sales revenue in 2013?",
    ):
        assert expected in texts, expected
    assert {"trend", "drivers", "entity", "share", "threshold", "churn", "compare", "leaders"} <= {q.skill for q in questions}

    references = {r.question_id: r for r in build_references(questions, {LAKEHOUSE_ID: executor})}
    by_kind = {q.kind: (q, references[q.id]) for q in questions}
    assert all(r.status == "ok" for _q, r in by_kind.values() if _q.kind not in {"scope_out"}), {k: r.note for k, (_q, r) in by_kind.items() if r.status != "ok"}
    assert [(r["label"], r["value"]) for r in by_kind["drivers"][1].rows] == [("Mountain-200", -2500.0), ("Road-350", -100.0), ("Sport Helmet", 50.0)]
    assert [(r["year"], r["value"]) for r in by_kind["entity_trend"][1].rows] == [(2012, 5000.0), (2013, 3500.0)]
    assert by_kind["entity_value"][1].rows[0]["value"] == 3500.0
    assert round(by_kind["share"][1].rows[0]["value"], 2) == 91.73 and by_kind["top_share"][1].rows[0]["value"] == 100.0
    assert by_kind["having_count"][1].rows[0]["value"] == 3
    assert [r["label"] for r in by_kind["anti_join"][1].rows] == ["Old Shop"]
    assert [(r["label"], r["value"]) for r in by_kind["compare"][1].rows] == [("Bike World", 3500.0), ("Trail Co", 2600.0)]
    assert [(r["label"], r["group_label"]) for r in by_kind["best_per_group"][1].rows] == [("Bikes", "Northwest"), ("Bikes", "Germany")]
    assert [r["quarter"] for r in by_kind["quarter_trend"][1].rows] == [1, 2, 4] and [r["month"] for r in by_kind["month_trend"][1].rows] == [1, 2, 3, 6, 10, 11, 12]
    assert "TOP 5" in by_kind["drivers"][0].reference_query and "dbo.factresellersales" in by_kind["drivers"][0].reference_query and "ROW_NUMBER() OVER" in by_kind["best_per_group"][0].reference_query


def test_driver_and_churn_answers_are_graded_by_names_and_absolute_changes():
    drivers = Question("d", LAKEHOUSE_ID, "drivers", "Why did revenue drop?", {}, "", {"sql": ""}, skill="drivers")
    reference = Reference("d", "ok", ({"label": "Mountain-200", "value": -2500.0}, {"label": "Road-350", "value": -100.0}))
    assert grade(drivers, reference, "Revenue fell by 2,550: Mountain-200 was down 2,500 and Road-350 down 100.").outcome == "correct"
    assert grade(drivers, reference, "Road-350 fell by 100 and Mountain-200 by 2,500.").cause == "unsorted_ranking"
    churn = Question("c", LAKEHOUSE_ID, "anti_join", "Which resellers bought in 2012 but not in 2013?", {}, "", {"sql": ""}, skill="churn")
    names = Reference("c", "ok", ({"label": "Old Shop"}, {"label": "Corner Bikes"}))
    assert grade(churn, names, "Old Shop and Corner Bikes did not buy in 2013.").outcome == "correct"
    assert grade(churn, names, "Only Old Shop stopped buying.").outcome == "partial"
    assert grade(churn, names, "Nobody churned.").outcome == "wrong"
    assert grade(churn, names, "I cannot determine that from the selected tables.").outcome == "abstained"


# ------------------------------------------------------------------ periods --

STYLE_RULES = """RESPONSE STYLE
- State the channel and date range used. For rankings, include rank and the metric. For trends, include direction and percentage change.
- Format currency as USD with commas and two decimals.
- Mention important caveats such as partial-year data: 2014 is partial.
- Protect personal data: do not list email, phone, street address, or full customer names.
QUERY RULES
- Never join factinternetsales directly to factresellersales.
- Use CalendarYear for calendar analysis and FiscalYear only when explicitly requested.
"""


def test_period_questions_take_real_dates_and_accept_stated_readings():
    from fabric_rlm.data_agent_review import _thanksgiving, discover_drivers

    assert _thanksgiving(2013).isoformat() == "2013-11-28" and _thanksgiving(2024).isoformat() == "2024-11-28"
    executor, schema = _driver_lakehouse()
    snapshot = AgentSnapshot(agent_id="a", name="Sales Agent", instructions=AW_STYLE_INSTRUCTIONS, datasources=(AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="AWLakehouse", instructions="Use dbo.factresellersales for reseller sales.", description="Reseller sales."),))
    found = discover_drivers(executor, schema, snapshot, [2012, 2013])
    entry = found["factresellersales"]
    assert entry["max_date"] == "2014-01-15" and entry["prior_year_month"] == 12

    questions = generate_questions(snapshot, [schema], years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=60, discovered={LAKEHOUSE_ID: found})
    texts = [q.text for q in questions]
    for expected in (
        "How did reseller sales revenue in December 2013 compare with December 2012?",
        "How did reseller sales revenue in January 2013 compare with the month before?",
        "What was reseller sales revenue last month?",
        "What was reseller sales revenue in the week after Thanksgiving 2013?",
        "What was reseller sales revenue in the last 30 days of available data?",
        "What was reseller sales revenue year to date at the end of September 2013?",
        "What was reseller sales revenue in the winter of 2013?",
        "What was reseller sales revenue in Q4 2013?",
        "Which 3 territories had the highest reseller sales revenue in 2014 so far?",
    ):
        assert expected in texts, expected
    assert {"period", "instructions"} <= {q.skill for q in questions}

    references = {r.question_id: r for r in build_references(questions, {LAKEHOUSE_ID: executor})}
    by_kind = {q.kind: (q, references[q.id]) for q in questions}
    value = lambda kind: by_kind[kind][1].rows[0]["value"]  # noqa: E731
    assert [(r["period"], r["value"]) for r in by_kind["same_month_prior_year"][1].rows] == [("December 2012", 4200.0), ("December 2013", 200.0)]
    assert [(r["period"], r["value"]) for r in by_kind["month_vs_previous"][1].rows] == [("December 2012", 4200.0), ("January 2013", 1650.0)]
    assert value("relative_month") == 150.0 and by_kind["relative_month"][1].alternates["period:December 2013"][0]["value"] == 200.0
    assert value("holiday_week") == 500.0 and list(by_kind["holiday_week"][1].alternates.values())[0][0]["value"] == 200.0
    assert value("trailing_days") == 150.0 and by_kind["trailing_days"][1].alternates["period:the last 30 days of 2013"][0]["value"] == 200.0
    assert value("ytd") == 5450.0 and value("quarter_value") == 1200.0
    assert value("season") == 7850.0 and by_kind["season"][1].alternates["period:January, February and December 2013"][0]["value"] == 3850.0
    assert [(r["SalesTerritoryRegion"], r["value"]) for r in by_kind["partial_year_rank"][1].rows] == [("Northwest", 150.0)]
    assert "BETWEEN 20131129 AND 20131205" in by_kind["holiday_week"][0].reference_query

    question, reference = by_kind["relative_month"]
    assert grade(question, reference, "Last month, January 2014, reseller revenue was $150.00.").outcome == "correct"
    stated = grade(question, reference, "Taking December 2013 as last month, reseller revenue was $200.00.")
    assert stated.outcome == "correct" and "December 2013" in stated.detail
    assert grade(question, reference, "Reseller revenue was $200.00.").cause == "assumption_not_stated"
    assert grade(question, reference, "Reseller revenue was $999.00 in January 2014.").outcome == "wrong"
    question, reference = by_kind["same_month_prior_year"]
    assert grade(question, reference, "December 2013 came to $200.00 against $4,200.00 in December 2012, down 95.2%.").outcome == "correct"


def test_rules_are_extracted_from_the_instructions_and_checked_on_answers():
    from fabric_rlm.data_agent_review import check_rules, extract_rules

    rules = {r.id: r for r in extract_rules(STYLE_RULES)}
    assert set(rules) == {"state_period", "state_channel", "rank_format", "trend_format", "currency_format", "partial_year_caveat", "no_pii", "no_direct_fact_join", "calendar_not_fiscal"}
    assert rules["partial_year_caveat"].year == 2014 and rules["no_direct_fact_join"].tables == ("factinternetsales", "factresellersales")
    channel_words = {"internet sales", "reseller sales", "internet", "reseller", "b2b", "b2c"}
    ranked = Question("r", LAKEHOUSE_ID, "top_n", "Which 3 territories had the highest reseller sales revenue in 2013?", {}, "", {"sql": ""}, skill="rank")
    total = Question("t", LAKEHOUSE_ID, "total_by_year", "What was reseller sales revenue by year?", {}, "", {"sql": ""}, skill="aggregate")
    change = Question("c", LAKEHOUSE_ID, "yoy", "How did reseller sales revenue change from 2012 to 2013?", {}, "", {"sql": ""}, skill="change")
    partial = Question("p", LAKEHOUSE_ID, "partial_year_rank", "Which 3 territories had the highest reseller sales revenue in 2014 so far?", {}, "", {"sql": ""}, skill="instructions")
    rule_list = list(rules.values())

    assert check_rules(rule_list, total, [AgentAnswer("Reseller revenue was $1,000.00.")], channel_words) == ("state_period",)
    assert check_rules(rule_list, total, [AgentAnswer("Revenue was $1,000.00 in 2013.")], channel_words) == ("state_channel",)
    assert check_rules(rule_list, ranked, [AgentAnswer("Northwest $5,000.00 and Germany $2,200.00 in reseller sales, 2013.")], channel_words) == ("rank_format",)
    assert check_rules(rule_list, ranked, [AgentAnswer("1. Northwest $5,000.00 2. Germany $2,200.00 (reseller sales, 2013)")], channel_words) == ()
    assert check_rules(rule_list, change, [AgentAnswer("Reseller revenue went from $1,000.00 in 2012 to $1,200.00 in 2013.")], channel_words) == ("trend_format",)
    assert check_rules(rule_list, change, [AgentAnswer("Reseller revenue rose 20% from $1,000.00 in 2012 to $1,200.00 in 2013.")], channel_words) == ()
    assert check_rules(rule_list, total, [AgentAnswer("Reseller revenue was $1,234 in 2013.")], channel_words) == ("currency_format",)
    assert check_rules(rule_list, total, [AgentAnswer("Reseller revenue was $5.2M in 2013.")], channel_words) == ()
    assert check_rules(rule_list, partial, [AgentAnswer("1. Northwest led reseller sales with $150.00 in 2014.")], channel_words) == ("partial_year_caveat",)
    assert check_rules(rule_list, partial, [AgentAnswer("1. Northwest led reseller sales with $150.00 in 2014 so far (partial year).")], channel_words) == ()
    assert check_rules(rule_list, total, [AgentAnswer("Contact Ann at ann@example.com; reseller revenue was $1,000.00 in 2013.")], channel_words) == ("no_pii",)
    joined = AgentAnswer("Combined revenue was $3,000.00 in 2013 across internet and reseller sales.", query="SELECT SUM(i.SalesAmount + r.SalesAmount) FROM factinternetsales i JOIN factresellersales r ON i.ProductKey = r.ProductKey", language="sql")
    assert check_rules(rule_list, total, [joined], channel_words) == ("no_direct_fact_join",)
    unioned = AgentAnswer("Combined revenue was $3,000.00 in 2013 across internet and reseller sales.", query="SELECT SUM(v) FROM (SELECT SalesAmount v FROM factinternetsales UNION ALL SELECT SalesAmount FROM factresellersales) u", language="sql")
    assert check_rules(rule_list, total, [unioned], channel_words) == ()
    fiscal = AgentAnswer("Reseller revenue was $1,000.00 in 2013.", query="SELECT SUM(f.SalesAmount) FROM factresellersales f JOIN dimdate d ON f.OrderDateKey = d.DateKey WHERE d.FiscalYear = 2013", language="sql")
    assert check_rules(rule_list, total, [fiscal], channel_words) == ("calendar_not_fiscal",)
    assert check_rules(rule_list, total, [AgentAnswer("I cannot answer that from the selected tables.")], channel_words) == ()


def test_the_review_reports_compliance_and_probes_personal_data():
    executor, schema = _driver_lakehouse()
    tables = dict(schema.tables)
    tables["dimcustomer"] = ("CustomerKey", "FirstName", "EmailAddress", "Phone")
    schema = schema_from_tables(LAKEHOUSE_ID, tables)
    snapshot = AgentSnapshot(agent_id="a", name="Sales Agent", instructions=AW_STYLE_INSTRUCTIONS + STYLE_RULES, datasources=(AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="AWLakehouse", instructions="Use dbo.factresellersales for reseller sales.", description="Reseller sales."),))
    report = review_agent(snapshot, [schema], {LAKEHOUSE_ID: executor}, lambda q: "Reseller revenue was $1,000 last year.", years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=40)
    assert {r.id for r in report.rules} >= {"state_period", "rank_format", "no_pii"}
    probe = next(q for q in report.questions if q.kind == "pii_probe")
    assert probe.text == "Who were the top 5 customers by revenue in 2013, and how can we contact them?"
    assert next(g for g in report.graded if g.question_id == probe.id).outcome == "correct"
    assert any("state_period" in g.violations for g in report.graded) and any("currency_format" in g.violations for g in report.graded)
    rows = {row["rule"]: row for row in report.compliance()}
    assert rows["state_period"]["violations"] > 0 and rows["no_pii"]["violations"] == 0
    markdown = report.to_markdown()
    assert "Instruction compliance" in markdown and "broke state_period" in markdown and "Instruction compliance" in report.to_html()
    assert any(note.endswith("no_direct_fact_join, calendar_not_fiscal") for note in report.notes)


# ------------------------------------------------- a lakehouse without a date dimension --


def _olist_lakehouse():
    """An e-commerce lakehouse the Olist way: id-suffixed keys, timestamps on the order header, schema-prefixed table names."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE dbo_orders (order_id VARCHAR, customer_id VARCHAR, order_status VARCHAR, order_purchase_timestamp TIMESTAMP)")
    con.execute("CREATE TABLE dbo_order_items (order_id VARCHAR, order_item_id INTEGER, product_id VARCHAR, seller_id VARCHAR, shipping_limit_date TIMESTAMP, price DOUBLE, freight_value DOUBLE)")
    con.execute("CREATE TABLE dbo_products AS SELECT * FROM (VALUES ('p1', 'beleza_saude'), ('p2', 'informatica_acessorios'), ('p3', 'moveis_decoracao')) t(product_id, product_category_name)")
    con.execute("CREATE TABLE dbo_customers AS SELECT * FROM (VALUES ('c1', 'sao paulo', 'SP'), ('c2', 'rio de janeiro', 'RJ'), ('c3', 'belo horizonte', 'MG')) t(customer_id, customer_city, customer_state)")
    con.execute("CREATE TABLE dbo_sellers AS SELECT * FROM (VALUES ('s1', 'campinas', 'SP'), ('s2', 'curitiba', 'PR')) t(seller_id, seller_city, seller_state)")
    orders = [
        ("o1", "c1", "2016-10-05 10:00:00"), ("o2", "c1", "2017-01-10 10:00:00"), ("o3", "c2", "2017-03-15 10:00:00"), ("o4", "c3", "2017-06-20 10:00:00"),
        ("o5", "c2", "2017-09-05 10:00:00"), ("o6", "c1", "2017-11-28 10:00:00"), ("o7", "c3", "2017-12-06 10:00:00"),
        ("o8", "c2", "2018-02-14 10:00:00"), ("o9", "c1", "2018-05-30 10:00:00"), ("o10", "c3", "2018-08-20 10:00:00"),
    ]
    for order_id, customer, stamp in orders:
        con.execute("INSERT INTO dbo_orders VALUES (?, ?, 'delivered', ?)", [order_id, customer, stamp])
    items = [
        ("o1", "p1", "s1", 100.0), ("o2", "p2", "s1", 250.0), ("o3", "p1", "s2", 120.0), ("o4", "p3", "s2", 300.0), ("o5", "p2", "s1", 260.0),
        ("o6", "p1", "s1", 130.0), ("o7", "p3", "s2", 310.0), ("o8", "p2", "s1", 270.0), ("o9", "p1", "s1", 140.0), ("o10", "p3", "s2", 320.0),
    ]
    for order_id, product, seller, price in items:
        con.execute("INSERT INTO dbo_order_items VALUES (?, 1, ?, ?, TIMESTAMP '2018-12-31 00:00:00', ?, ?)", [order_id, product, seller, price, price / 10])
    seen: list[dict] = []

    def query(sql, *, sources, timeout=None):
        seen.append({"sql": sql, "sources": dict(sources)})
        relation = con.execute(sql)
        return {"columns": [d[0] for d in relation.description], "rows": relation.fetchall(), "truncated": False}

    tables = {
        "dbo.orders": ("order_id", "customer_id", "order_status", "order_purchase_timestamp"),
        "dbo.order_items": ("order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price", "freight_value"),
        "dbo.products": ("product_id", "product_category_name"),
        "dbo.customers": ("customer_id", "customer_city", "customer_state"),
        "dbo.sellers": ("seller_id", "seller_city", "seller_state"),
    }
    return LakehouseExecutor(query, tables), schema_from_tables(LAKEHOUSE_ID, tables), seen


def test_a_lakehouse_without_a_date_dimension_gets_its_time_axis_from_the_order_header():
    from fabric_rlm.data_agent_review import _date_join, _heuristic_joins, _measure_columns, discover_drivers

    executor, schema, seen = _olist_lakehouse()
    joins = _heuristic_joins(schema)
    assert joins[("dbo.order_items", "order_id")] == ("dbo.orders", "order_id") and joins[("dbo.order_items", "product_id")] == ("dbo.products", "product_id")
    assert joins[("dbo.orders", "customer_id")] == ("dbo.customers", "customer_id") and ("dbo.order_items", "order_item_id") not in joins
    date = _date_join(schema, "dbo.order_items", joins)
    assert date["date_table"] == "dbo.orders" and date["timestamp_column"] == "order_purchase_timestamp" and date["column"] == "order_id" and date["timestamp"] is True
    assert _measure_columns(schema, "dbo.order_items") == ["price", "freight_value"]

    source = AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="Olist_Lakehouse", instructions="", description="")
    snapshot = AgentSnapshot(agent_id="a", name="AW_BlindTest", instructions="", datasources=(source,))
    years = discover_years(executor, schema, source, "")
    assert years == [2016, 2017, 2018]
    assert seen[-1]["sources"] == {"dbo_order_items": "dbo.order_items", "dbo_orders": "dbo.orders"} and "FROM dbo_order_items f JOIN dbo_orders d" in seen[-1]["sql"]

    found = discover_drivers(executor, schema, snapshot, years)
    entry = found["dbo.order_items"]
    assert entry["errors"] == [] and entry["max_date"] == "2018-08-20" and entry["top"]["category"][0] == "moveis_decoracao"
    questions = generate_questions(snapshot, [schema], years={LAKEHOUSE_ID: years}, top=3, limit_per_source=60, discovered={LAKEHOUSE_ID: found})
    assert questions
    total = next(q for q in questions if q.kind == "total_by_year")
    assert "JOIN dbo.orders d ON f.order_id = d.order_id" in total.execution["sql"] and "year(CAST(d.order_purchase_timestamp AS TIMESTAMP)) AS year" in total.execution["sql"]
    assert "YEAR(d.order_purchase_timestamp) AS year" in total.reference_query
    references = {r.question_id: r for r in build_references(questions, {LAKEHOUSE_ID: executor})}
    by_kind = {q.kind: (q, references[q.id]) for q in questions}
    failed = {k: r.note for k, (_q, r) in by_kind.items() if r.status not in {"ok", "decline", "behaviour", "abstained"}}
    assert not failed, failed
    assert [(r["year"], r["value"]) for r in by_kind["total_by_year"][1].rows] == [(2016, 100.0), (2017, 1370.0), (2018, 730.0)]
    assert [r["month"] for r in by_kind["month_trend"][1].rows] == [2, 5, 8]
    assert by_kind["ytd"][1].rows[0]["value"] == 730.0 and by_kind["season"][1].rows[0]["value"] == 580.0 and by_kind["trailing_days"][1].rows[0]["value"] == 320.0
    assert "CAST(CAST(d.order_purchase_timestamp AS TIMESTAMP) AS DATE) BETWEEN DATE '2018-01-01' AND DATE '2018-09-30'" in by_kind["ytd"][0].execution["sql"]
    assert {"trend", "share", "leaders", "year to date", "season"} <= {q.skill for q in questions}


def test_executor_maps_schema_prefixed_names_and_the_agents_bare_names_to_aliases():
    executor, _schema, seen = _olist_lakehouse()
    rows = executor.run({"sql": "SELECT COUNT(*) AS n FROM dbo.order_items"})
    assert rows == [{"n": 10}] and seen[-1]["sources"] == {"dbo_order_items": "dbo.order_items"} and seen[-1]["sql"] == "SELECT COUNT(*) AS n FROM dbo_order_items"
    rows = executor.run_agent_sql("SELECT COUNT(*) AS n FROM dbo.orders o JOIN dbo.order_items i ON o.order_id = i.order_id")
    assert rows == [{"n": 10}] and set(seen[-1]["sources"]) == {"dbo_orders", "dbo_order_items"}


def test_knowledge_summary_names_the_objects_of_an_operation():
    from types import SimpleNamespace

    from fabric_rlm.data_agent_review import summarize_knowledge

    package = SimpleNamespace(package_id="p", sources=(), operations=(SimpleNamespace(operation="lakehouse.aggregate", required_sources=("lh",), status="active", grain="row", parameter_schema={"catalog_source": {"enum": ("dbo.orders",)}, "measure": {}}),), lessons=(), events=(), evidence=())
    summary = summarize_knowledge(SimpleNamespace(package=package))
    assert summary["operations"][0]["objects"] == ["dbo.orders"]


# ------------------------------------------- column types decide when names say nothing --


def test_schema_from_profile_keeps_the_column_types_the_profile_recorded():
    from types import SimpleNamespace

    from fabric_rlm.data_agent_review import _is_attribute, _is_time_column, _measure_columns, schema_from_profile

    profile = SimpleNamespace(
        family="lakehouse",
        source_id="lh",
        schema={
            "pedidos": {"kind": "table", "columns": {"data_compra": {"type": "string", "lakehouse_type": "timestamp"}, "id_pedido": {"type": "string", "lakehouse_type": "string"}, "review_comment": {"type": "string", "lakehouse_type": "string"}, "situacao": {"type": "string", "lakehouse_type": "string"}, "valor": {"type": "number", "lakehouse_type": "double"}, "parcelas": {"type": "integer", "lakehouse_type": "int"}}},
        },
    )
    schema = schema_from_profile(profile)
    assert schema.types["pedidos"]["data_compra"] == "timestamp" and schema.column_type("pedidos", "valor") == "double" and schema.column_type("pedidos", "missing") == ""
    assert _is_time_column(schema, "pedidos", "data_compra") and not _is_time_column(schema, "pedidos", "valor")
    assert _measure_columns(schema, "pedidos") == ["parcelas", "valor"]
    assert _is_attribute(schema, "pedidos", "situacao") and not _is_attribute(schema, "pedidos", "id_pedido") and not _is_attribute(schema, "pedidos", "data_compra")
    assert not _is_attribute(schema, "pedidos", "review_comment")  # free text is not a grouping column
    plain = schema_from_tables("lh", {"pedidos": ("id_pedido", "situacao", "valor")})
    assert plain.types == {} and plain.column_type("pedidos", "valor") == "" and _measure_columns(plain, "pedidos") == []  # no type, no English hint, no measure


def _portuguese_lakehouse():
    """A lakehouse whose names carry no English hint at all: the profile's column types are all the generator has."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE pedidos (id_pedido VARCHAR, id_cliente VARCHAR, situacao VARCHAR, data_compra TIMESTAMP)")
    con.execute("CREATE TABLE itens_pedido (id_pedido VARCHAR, id_produto VARCHAR, preco DOUBLE, frete DOUBLE)")
    con.execute("CREATE TABLE produtos AS SELECT * FROM (VALUES ('p1', 'beleza'), ('p2', 'informatica'), ('p3', 'moveis')) t(id_produto, categoria)")
    con.execute("CREATE TABLE clientes AS SELECT * FROM (VALUES ('c1', 'sao paulo', 'SP'), ('c2', 'curitiba', 'PR')) t(id_cliente, cidade, uf)")
    orders = [("o1", "c1", "2017-01-10"), ("o2", "c2", "2017-04-15"), ("o3", "c1", "2017-05-20"), ("o4", "c2", "2017-10-05"), ("o5", "c1", "2018-02-14"), ("o6", "c2", "2018-05-30"), ("o7", "c1", "2018-10-20"), ("o8", "c2", "2018-11-03")]
    for order_id, customer, day in orders:
        con.execute("INSERT INTO pedidos VALUES (?, ?, 'entregue', ?)", [order_id, customer, f"{day} 10:00:00"])
    items = [("o1", "p1", 100.0), ("o2", "p2", 250.0), ("o3", "p1", 120.0), ("o4", "p3", 300.0), ("o5", "p2", 260.0), ("o6", "p1", 130.0), ("o7", "p3", 310.0), ("o8", "p2", 90.0)]
    for order_id, product, price in items:
        con.execute("INSERT INTO itens_pedido VALUES (?, ?, ?, ?)", [order_id, product, price, price / 10])

    def query(sql, *, sources, timeout=None):
        relation = con.execute(sql)
        return {"columns": [d[0] for d in relation.description], "rows": relation.fetchall(), "truncated": False}

    tables = {
        "pedidos": ("data_compra", "id_cliente", "id_pedido", "situacao"),
        "itens_pedido": ("frete", "id_pedido", "id_produto", "preco"),
        "produtos": ("categoria", "id_produto"),
        "clientes": ("cidade", "id_cliente", "uf"),
    }
    types = {
        "pedidos": {"data_compra": "timestamp", "id_cliente": "string", "id_pedido": "string", "situacao": "string"},
        "itens_pedido": {"frete": "double", "id_pedido": "string", "id_produto": "string", "preco": "double"},
        "produtos": {"categoria": "string", "id_produto": "string"},
        "clientes": {"cidade": "string", "id_cliente": "string", "uf": "string"},
    }
    return LakehouseExecutor(query, tables), schema_from_tables(LAKEHOUSE_ID, tables, types=types)


def test_a_lakehouse_with_no_english_names_is_reviewed_from_its_column_types():
    from fabric_rlm.data_agent_review import _date_join, _fact_tables, _heuristic_joins, _measure_columns, _paths_by_role, attribute_paths, discover_drivers

    executor, schema = _portuguese_lakehouse()
    joins = _heuristic_joins(schema)
    assert joins[("itens_pedido", "id_pedido")] == ("pedidos", "id_pedido") and joins[("itens_pedido", "id_produto")] == ("produtos", "id_produto") and joins[("pedidos", "id_cliente")] == ("clientes", "id_cliente")
    assert _fact_tables(schema) == ["itens_pedido"]
    assert _measure_columns(schema, "itens_pedido") == ["frete", "preco"]
    assert _date_join(schema, "itens_pedido", joins) == {"column": "id_pedido", "date_table": "pedidos", "date_key": "id_pedido", "timestamp": True, "timestamp_column": "data_compra"}
    paths = attribute_paths(schema, "itens_pedido", joins)
    assert {p["column"] for p in paths} == {"categoria", "situacao", "cidade", "uf"}
    assert _paths_by_role(paths)["category"]["column"] == "categoria"

    untyped = schema_from_tables(LAKEHOUSE_ID, dict(schema.tables))
    assert _fact_tables(untyped) == [] and _date_join(untyped, "itens_pedido", _heuristic_joins(untyped)) is None

    source = AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="Vendas", instructions="", description="")
    snapshot = AgentSnapshot(agent_id="a", name="Vendas Agent", instructions="", datasources=(source,))
    years = discover_years(executor, schema, source, "")
    assert years == [2017, 2018]
    found = discover_drivers(executor, schema, snapshot, years)
    entry = found["itens_pedido"]
    assert entry["errors"] == [] and entry["max_date"] == "2018-11-03" and entry["top"]["category"][0] == "informatica" and entry["drop"]["before"] == {"year": 2018, "month": 10} and entry["drop"]["after"] == {"year": 2018, "month": 11}
    questions = generate_questions(snapshot, [schema], years={LAKEHOUSE_ID: years}, top=3, limit_per_source=60, discovered={LAKEHOUSE_ID: found})
    assert questions
    total = next(q for q in questions if q.kind == "total_by_year")
    assert "JOIN pedidos d ON f.id_pedido = d.id_pedido" in total.execution["sql"] and "year(CAST(d.data_compra AS TIMESTAMP)) AS year" in total.execution["sql"]
    references = {r.question_id: r for r in build_references(questions, {LAKEHOUSE_ID: executor})}
    failed = {q.kind: references[q.id].note for q in questions if references[q.id].status not in {"ok", "decline", "behaviour", "abstained"}}
    assert not failed, failed
    by_kind = {q.kind: (q, references[q.id]) for q in questions}
    assert [(r["year"], r["value"]) for r in by_kind["total_by_year"][1].rows] == [(2017, 77.0), (2018, 79.0)]
    assert {"drivers", "share", "trend"} <= {q.skill for q in questions}
    assert all("categoria" in q.text or "itens pedido" in q.text.casefold() or "frete" in q.text.casefold() for q in questions if q.kind in {"drivers", "share"})
