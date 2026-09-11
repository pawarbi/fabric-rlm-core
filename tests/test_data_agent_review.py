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
        if question.startswith("What was total SalesAmount by year") and "combined" in question:
            return AgentAnswer("2012: $100.00; 2013: $375.00")  # internet only: narrower scope
        if question.startswith("What are the top"):
            return AgentAnswer("Road-350 2,000 then Mountain-200 2,500 then Sport Helmet 50", query="SELECT TOP 3 a.EnglishProductName, SUM(f.SalesAmount) AS value FROM dbo.factinternetsales f JOIN dbo.dimdate d ON f.OrderDateKey = d.DateKey JOIN dbo.dimproduct a ON f.ProductKey = a.ProductKey WHERE d.CalendarYear = 2013 GROUP BY a.EnglishProductName ORDER BY value DESC", language="sql")
        if "distinct" in question:
            return AgentAnswer("I cannot determine orders from the selected tables.")
        return AgentAnswer("The values are 1,100.00 for 2012 and 2,850.00 for 2013; 100.00 and 350.00; 1,000.00 and 2,500.00; 2,750.00 and 350.00; 2,500 and 1,000 and 2,000 and 350 and 500 and 50")

    report = review_agent(snapshot, [schema], {LAKEHOUSE_ID: executor}, ask, top=3, limit_per_source=6)
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
    assert questions and all("factfinance" not in q.text for q in questions)
    unnamed = AgentSnapshot(agent_id="a", name="Bare", instructions="Answer questions.", datasources=(AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="lh", instructions="", description="d"),))
    everything = generate_questions(unnamed, [schema], years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=40)
    assert any("factfinance" in q.text for q in everything)


def test_plain_words_are_not_unknown_references():
    source = AgentDataSource(id=LAKEHOUSE_ID, kind="lakehouse", name="lh", instructions="FACT TABLES\n- dbo.factinternetsales joins the DIMENSION tables. Both facts share ProductKey.\n- Use vw_sales_flat for rollups.", description="d")
    snapshot = AgentSnapshot(agent_id="a", name="Words", instructions="Answer.", datasources=(source,))
    findings = {f.code: f for f in diagnose(snapshot, _schemas())}
    unknown = findings["unknown_reference"]
    assert "vw_sales_flat" in unknown.message
    assert "DIMENSION" not in unknown.message and "facts" not in unknown.message


def test_single_year_questions_read_naturally_and_no_data_answers_are_wrong():
    questions = generate_questions(_snapshot(), _schemas(), years={LAKEHOUSE_ID: [2013]}, top=3)
    assert all("2013 to 2013" not in q.text for q in questions) and any("for 2013" in q.text for q in questions)
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
    texts = [q.text for q in questions]
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
    assert any("factinternetsales has no join to a date table" in note for note in notes)
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
    assert len(generated) == 4 and all("factresellersales" in q.text for q in generated)

    plain = generate_questions(_snapshot(), _schemas(), years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=4)
    assert len(plain) == 4 and not all("factresellersales" in q.text for q in plain)

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
    assert questions and not any("Promotion" in q.text or "quota" in q.text.casefold() for q in questions)
    without = generate_questions(_snapshot(), [schema], years={LAKEHOUSE_ID: [2012, 2013]}, top=3, limit_per_source=40)
    assert any("EnglishPromotionCategory" in q.text for q in without)

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
    assert all("factresellersales" in q.text for q in ranked)


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
    assert all("factresellersales" in q.text for q in questions)


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
