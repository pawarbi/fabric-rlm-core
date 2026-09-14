"""A schema whose names say nothing: keys spelled the way an export spells them, and joins the data has to settle."""

from __future__ import annotations

import pytest

from fabric_rlm.source_model import (
    LakehouseExecutor,
    _fact_tables,
    _is_key,
    _measure_columns,
    joins_from_data,
    schema_from_tables,
)

TABLES = {
    "sls_hdr": ("ord_no", "cust_cd", "dt1", "stat"),
    "sls_dtl": ("ord_no", "ln", "cd", "qty", "amt1", "amt2"),
    "cust_mstr": ("cust_cd", "nm", "rgn"),
    "prd": ("cd", "nm"),
}
TYPES = {
    "sls_hdr": {"ord_no": "varchar", "cust_cd": "varchar", "dt1": "date", "stat": "varchar"},
    "sls_dtl": {"ord_no": "varchar", "ln": "bigint", "cd": "varchar", "qty": "bigint", "amt1": "double", "amt2": "double"},
    "cust_mstr": {"cust_cd": "varchar", "nm": "varchar", "rgn": "varchar"},
    "prd": {"cd": "varchar", "nm": "varchar"},
}


def _source():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE sls_hdr (ord_no VARCHAR, cust_cd VARCHAR, dt1 DATE, stat VARCHAR)")
    con.execute("CREATE TABLE sls_dtl (ord_no VARCHAR, ln BIGINT, cd VARCHAR, qty BIGINT, amt1 DOUBLE, amt2 DOUBLE)")
    con.execute("CREATE TABLE cust_mstr (cust_cd VARCHAR, nm VARCHAR, rgn VARCHAR)")
    con.execute("CREATE TABLE prd (cd VARCHAR, nm VARCHAR)")
    con.execute("INSERT INTO sls_hdr VALUES ('O1','C1',DATE '2026-07-01','C'), ('O2','C1',DATE '2026-07-02','C'), ('O3','C2',DATE '2026-08-01','X')")
    con.execute("INSERT INTO sls_dtl VALUES ('O1',1,'P1',2,100.0,90.0), ('O1',2,'P2',1,50.0,50.0), ('O2',1,'P1',3,150.0,140.0), ('O3',1,'P2',1,50.0,45.0)")
    con.execute("INSERT INTO cust_mstr VALUES ('C1','Acme','R1'), ('C2','Globex','R2')")
    con.execute("INSERT INTO prd VALUES ('P1','Road Elite'), ('P2','Helmet Air')")

    def query(sql, *, sources=None, timeout=None):
        relation = con.execute(sql)
        return {"columns": [d[0] for d in relation.description], "rows": relation.fetchall(), "truncated": False}

    return LakehouseExecutor(query, {t: c for t, c in TABLES.items()}), schema_from_tables("lh", TABLES, types=TYPES)


def test_a_key_can_be_spelled_the_way_an_export_spells_it() -> None:
    assert all(_is_key(c) for c in ("ord_no", "cust_cd", "item_nbr", "plant_code", "line_num", "doc_ref"))
    assert all(_is_key(c) for c in ("CustomerKey", "customer_id", "id_cliente"))  # the old spellings still hold
    assert not any(_is_key(c) for c in ("amount_paid", "is_valid", "amt2", "stat", "revenue"))


def test_an_amount_spelled_amt_is_a_measure() -> None:
    _executor, schema = _source()
    measures = _measure_columns(schema, "sls_dtl")
    assert "amt1" in measures and "amt2" in measures
    # the key and the line number are not measures however they are typed
    assert "ln" not in measures and "cd" not in measures


def test_the_data_settles_a_join_no_name_reveals() -> None:
    executor, schema = _source()
    joins = joins_from_data(schema, executor)
    # ord_no identifies a row in the header and repeats in the lines, so the header is the parent
    assert joins[("sls_dtl", "ord_no")] == ("sls_hdr", "ord_no")
    assert joins[("sls_hdr", "cust_cd")] == ("cust_mstr", "cust_cd")
    assert joins[("sls_dtl", "cd")] == ("prd", "cd")
    assert ("sls_hdr", "ord_no") not in joins  # a table is never joined to itself


def test_a_key_unique_in_every_table_or_in_none_is_left_alone() -> None:
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE a (k VARCHAR, v BIGINT)")
    con.execute("CREATE TABLE b (k VARCHAR, v BIGINT)")
    con.execute("CREATE TABLE c (k VARCHAR, v BIGINT)")
    con.execute("INSERT INTO a VALUES ('x',1), ('y',2)")  # unique
    con.execute("INSERT INTO b VALUES ('x',1), ('y',2)")  # unique too: nothing says which is the parent
    con.execute("INSERT INTO c VALUES ('x',1), ('x',2)")  # unique nowhere else
    tables = {"a": ("k", "v"), "b": ("k", "v"), "c": ("k", "v")}

    def query(sql, *, sources=None, timeout=None):
        relation = con.execute(sql)
        return {"columns": [d[0] for d in relation.description], "rows": relation.fetchall(), "truncated": False}

    schema = schema_from_tables("lh2", tables, types={t: {"k": "varchar", "v": "bigint"} for t in tables})
    assert joins_from_data(schema, LakehouseExecutor(query, tables)) == {}


def test_a_source_that_will_not_answer_yields_no_joins_rather_than_an_error() -> None:
    def refuse(sql, *, sources=None, timeout=None):
        raise RuntimeError("read-only gate refused the query")

    schema = schema_from_tables("lh3", TABLES, types=TYPES)
    assert joins_from_data(schema, LakehouseExecutor(refuse, TABLES)) == {}


def test_order_lines_become_a_fact_only_once_the_discovered_join_is_known() -> None:
    executor, schema = _source()
    # the date lives on the header; without the path to it the lines are not a fact
    assert "sls_dtl" not in _fact_tables(schema)
    assert "sls_dtl" in _fact_tables(schema, joins=joins_from_data(schema, executor))


def test_the_query_budget_is_bounded() -> None:
    executor, schema = _source()
    calls = {"n": 0}
    inner = executor.run

    def counting(spec):
        calls["n"] += 1
        return inner(spec)

    executor.run = counting  # type: ignore[assignment]
    joins_from_data(schema, executor, limit=2)
    assert calls["n"] <= 2
