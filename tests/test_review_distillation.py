"""The review on a schema whose names say nothing: questions reach the order lines through a join only the data reveals, a stale copy is never asked about, and the query the agent ran becomes instruction lines."""

from __future__ import annotations

import datetime as dt

import pytest

from fabric_rlm.experimental.data_agent_review import (
    AgentAnswer,
    Graded,
    Question,
    Reference,
    generate_questions,
    suggest,
)
from fabric_rlm.source_model import (
    AgentDataSource,
    AgentSnapshot,
    LakehouseExecutor,
    joins_from_data,
    schema_from_tables,
    shadow_tables,
)

SOURCE = "lh-murky"
TABLES = {
    "sls_hdr": ("ord_no", "cust_cd", "dt1", "src", "stat", "cur"),
    "sls_dtl": ("ord_no", "ln", "cd", "qty", "amt1", "amt2"),
    "sls_dtl_v2": ("ord_no", "ln", "cd", "qty", "amt1", "amt2"),
    "cust_mstr": ("cust_cd", "nm", "rgn"),
}
TYPES = {
    "sls_hdr": {"ord_no": "varchar", "cust_cd": "varchar", "dt1": "date", "src": "varchar", "stat": "varchar", "cur": "varchar"},
    "sls_dtl": {"ord_no": "varchar", "ln": "bigint", "cd": "varchar", "qty": "bigint", "amt1": "double", "amt2": "double"},
    "sls_dtl_v2": {"ord_no": "varchar", "ln": "bigint", "cd": "varchar", "qty": "bigint", "amt1": "double", "amt2": "double"},
    "cust_mstr": {"cust_cd": "varchar", "nm": "varchar", "rgn": "varchar"},
}


def _source():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    for table, columns in TABLES.items():
        con.execute(f"CREATE TABLE {table} ({', '.join(f'{c} {TYPES[table][c]}' for c in columns)})")
    con.execute("INSERT INTO cust_mstr VALUES ('C1', 'Acme', 'R1'), ('C2', 'Globex', 'R2')")
    order = 0
    for year in (2024, 2025):
        for month in range(1, 13):
            for customer, status in (("C1", "C"), ("C2", "X")):
                order += 1
                number = f"O{order:04d}"
                con.execute("INSERT INTO sls_hdr VALUES (?, ?, ?, ?, ?, ?)", [number, customer, dt.date(year, month, 10), "ERP", status, "USD"])
                con.execute("INSERT INTO sls_dtl VALUES (?, 1, 'P1', 2, 120.0, 100.0)", [number])
                if year == 2024:  # the stale copy holds the first year only
                    con.execute("INSERT INTO sls_dtl_v2 VALUES (?, 1, 'P1', 2, 120.0, 108.0)", [number])

    def query(sql, *, sources=None, timeout=None):
        relation = con.execute(sql)
        return {"columns": [d[0] for d in relation.description], "rows": relation.fetchall(), "truncated": False}

    snapshot = AgentSnapshot(
        agent_id="a", name="murky", instructions="", description="",
        datasources=(AgentDataSource(id=SOURCE, kind="lakehouse", name="murky_sales", instructions="", description="",
                                     item_id="i", workspace_id="w", fewshots=(), selected_tables=tuple(TABLES)),),
    )
    return LakehouseExecutor(query, TABLES), schema_from_tables(SOURCE, TABLES, types=TYPES), snapshot


def _facts(questions) -> set[str]:
    return {str(f.get("table")) for q in questions for f in ((q.spec or {}).get("facts") or []) if isinstance(f, dict)}


def test_the_order_lines_are_asked_about_only_once_the_data_reveals_their_join() -> None:
    executor, schema, snapshot = _source()
    years = {SOURCE: [2024, 2025]}
    assert generate_questions(snapshot, [schema], years=years) == ()  # no name joins sls_dtl to its header
    joins = joins_from_data(schema, executor)
    assert joins[("sls_dtl", "ord_no")] == ("sls_hdr", "ord_no")
    asked = generate_questions(snapshot, [schema], years=years, data_joins={SOURCE: joins})
    assert asked and "sls_dtl" in _facts(asked)


def test_a_stale_copy_is_never_asked_about() -> None:
    executor, schema, snapshot = _source()
    years = {SOURCE: [2024, 2025]}
    joins = joins_from_data(schema, executor)
    stale = shadow_tables(schema, executor)
    assert stale == {"sls_dtl_v2": "sls_dtl"}
    with_copy = generate_questions(snapshot, [schema], years=years, data_joins={SOURCE: joins})
    without = generate_questions(snapshot, [schema], years=years, data_joins={SOURCE: joins}, shadow={SOURCE: stale})
    assert "sls_dtl_v2" in _facts(with_copy)  # left alone it looks like a fact and gets questions of its own
    assert without and "sls_dtl_v2" not in _facts(without)


def test_the_query_the_agent_ran_becomes_lines_in_the_data_source_instructions() -> None:
    _executor, schema, snapshot = _source()
    reference_sql = (
        "SELECT SUM(d.amt2 * CASE h.cur WHEN 'EUR' THEN 1.10 ELSE 1.0 END) AS value "
        "FROM sls_dtl d JOIN sls_hdr h ON h.ord_no = d.ord_no WHERE h.src = 'ERP' AND h.stat <> 'X'"
    )
    question = Question(id="q1", source_id=SOURCE, kind="supplied", text="What was total revenue?",
                        spec={"kind": "supplied", "measure": "revenue"}, reference_query=reference_sql,
                        execution={"kind": "supplied_sql", "sql": reference_sql})
    answer = AgentAnswer(text="Revenue was 5,280.", query="SELECT SUM(d.amt1 + d.amt2) AS revenue FROM dbo.sls_dtl d",
                         language="sql", datasource="murky_sales", seconds=1.0, steps=(), status="completed", executed=True)
    graded = Graded("q1", "wrong", "values_differ", "the figure differs")
    suggestions = suggest(snapshot, [schema], (), [question], [Reference("q1", "ok", rows=({"value": 2400.0},))], [graded],
                          answers={"q1": (answer,)})
    text = suggestions.datasource_instructions[SOURCE]
    assert "## From the queries the agent ran" in text
    assert "Do not add amt1" in text
    assert "sls_hdr.cur" in text
    assert "sls_hdr.src = ERP" in text and "sls_hdr.stat <> X" in text


def test_a_correct_answer_adds_no_lines() -> None:
    _executor, schema, snapshot = _source()
    question = Question(id="q1", source_id=SOURCE, kind="supplied", text="What was total revenue?",
                        spec={"kind": "supplied", "measure": "revenue"}, reference_query="SELECT SUM(amt2) AS value FROM sls_dtl",
                        execution={"kind": "supplied_sql", "sql": "SELECT SUM(amt2) AS value FROM sls_dtl"})
    answer = AgentAnswer(text="2,400", query="SELECT SUM(amt1) AS value FROM sls_dtl", language="sql", datasource="murky_sales",
                         seconds=1.0, steps=(), status="completed", executed=True)
    suggestions = suggest(snapshot, [schema], (), [question], [Reference("q1", "ok", rows=({"value": 2400.0},))],
                          [Graded("q1", "correct")], answers={"q1": answer})
    assert "## From the queries the agent ran" not in suggestions.datasource_instructions[SOURCE]
