"""A migration leaves a copy of a table behind; the data says which one is the copy."""

from __future__ import annotations

import pytest

from fabric_rlm.source_model import LakehouseExecutor, schema_from_tables, shadow_tables

COLUMNS = ("ord_no", "ln", "amt2")
TYPES = {"ord_no": "varchar", "ln": "bigint", "amt2": "double"}


def _source(rows: dict[str, list[tuple]], columns: dict[str, tuple[str, ...]] | None = None):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    tables = columns or {name: COLUMNS for name in rows}
    for name, cols in tables.items():
        spec = ", ".join(f"{c} {TYPES.get(c, 'varchar')}" for c in cols)
        con.execute(f"CREATE TABLE {name} ({spec})")
        for row in rows.get(name, []):
            con.execute(f"INSERT INTO {name} VALUES ({', '.join('?' for _ in row)})", list(row))

    def query(sql, *, sources=None, timeout=None):
        relation = con.execute(sql)
        return {"columns": [d[0] for d in relation.description], "rows": relation.fetchall(), "truncated": False}

    schema = schema_from_tables("lh", tables, types={t: {c: TYPES.get(c, "varchar") for c in cols} for t, cols in tables.items()})
    return LakehouseExecutor(query, tables), schema


def test_a_smaller_copy_whose_keys_all_appear_in_the_other_is_a_stale_copy() -> None:
    executor, schema = _source({
        "sls_dtl": [("O1", 1, 10.0), ("O2", 1, 20.0), ("O3", 1, 30.0)],
        "sls_dtl_v2": [("O1", 1, 11.0), ("O2", 1, 22.0)],
    })
    assert shadow_tables(schema, executor) == {"sls_dtl_v2": "sls_dtl"}


def test_two_tables_that_each_hold_rows_the_other_lacks_are_both_kept() -> None:
    executor, schema = _source({
        "sales_eu": [("O1", 1, 10.0), ("O2", 1, 20.0)],
        "sales_us": [("O3", 1, 30.0)],
    })
    # a partition is not a stale copy: O3 is in neither the other table nor its keys
    assert shadow_tables(schema, executor) == {}


def test_tables_with_different_columns_are_never_compared() -> None:
    executor, schema = _source(
        {"orders": [("O1", 1, 10.0)], "customers": [("C1",)]},
        columns={"orders": COLUMNS, "customers": ("cust_cd",)},
    )
    assert shadow_tables(schema, executor) == {}


def test_a_copy_of_equal_size_is_not_called_stale() -> None:
    executor, schema = _source({
        "a_tbl": [("O1", 1, 10.0), ("O2", 1, 20.0)],
        "b_tbl": [("O1", 1, 10.0), ("O2", 1, 20.0)],
    })
    # same rows on both sides: nothing says which one anybody should read
    assert shadow_tables(schema, executor) == {}


def test_a_source_that_refuses_to_answer_reports_nothing_rather_than_guessing() -> None:
    def refuse(sql, *, sources=None, timeout=None):
        raise RuntimeError("the read-only gate refused this query")

    tables = {"sls_dtl": COLUMNS, "sls_dtl_v2": COLUMNS}
    schema = schema_from_tables("lh", tables, types={t: TYPES for t in tables})
    assert shadow_tables(schema, LakehouseExecutor(refuse, tables)) == {}


def test_the_query_budget_is_bounded() -> None:
    executor, schema = _source({
        "sls_dtl": [("O1", 1, 10.0), ("O2", 1, 20.0)],
        "sls_dtl_v2": [("O1", 1, 10.0)],
    })
    calls = {"n": 0}
    inner = executor.run

    def counting(spec):
        calls["n"] += 1
        return inner(spec)

    executor.run = counting  # type: ignore[assignment]
    shadow_tables(schema, executor, limit=1)
    assert calls["n"] <= 1
