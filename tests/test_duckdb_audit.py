"""Tests for the internal DuckDB deep-insight audit executor."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from fabric_rlm._deep_insight_audit import AuditCheck
from fabric_rlm._duckdb_audit import DuckDBAuditError, DuckDBAuditExecutor


def _write_csv(path: Path, rows: str) -> Path:
    path.write_text(rows, encoding="utf-8")
    return path


def _check(
    expression: object,
    *,
    method: object = "sql",
    sources: object = None,
) -> AuditCheck:
    return AuditCheck(
        path="insights[0]",
        expected=0,
        verification={
            "method": method,
            "expression": expression,
            "sources": {"orders": "orders"} if sources is None else sources,
        },
    )


def test_executes_aggregate_join_and_cte_queries(tmp_path: Path) -> None:
    orders = _write_csv(
        tmp_path / "orders'quoted.csv",
        "customer_id,amount\n1,10\n1,20\n2,5\n",
    )
    customers = _write_csv(
        tmp_path / "customers.csv",
        "customer_id,active\n1,true\n2,false\n",
    )
    executor = DuckDBAuditExecutor({"orders": orders, "customers": customers})

    aggregate = executor(
        _check("SELECT SUM(amount) FROM orders")
    )
    joined_cte = executor(
        _check(
            """
            WITH active_orders AS (
                SELECT o.amount
                FROM orders AS o
                JOIN customers AS c USING (customer_id)
                WHERE c.active
            )
            SELECT SUM(amount) FROM active_orders
            """,
            sources={"orders": "orders", "customers": "customers"},
        )
    )

    assert aggregate == [[35]]
    assert joined_cte == [[30]]


def test_reuses_one_locked_source_snapshot_across_checks(tmp_path: Path) -> None:
    orders = _write_csv(tmp_path / "orders.csv", "amount\n10\n20\n")
    executor = DuckDBAuditExecutor({"orders": orders})

    assert executor(_check("SELECT SUM(amount) FROM orders")) == [[30]]
    orders.write_text("amount\n100\n200\n", encoding="utf-8")
    assert executor(_check("SELECT SUM(amount) FROM orders")) == [[30]]


def test_binds_query_aliases_to_declared_source_identities(tmp_path: Path) -> None:
    orders = _write_csv(tmp_path / "orders.csv", "amount\n10\n20\n")
    executor = DuckDBAuditExecutor({"olist_orders": orders})

    result = executor(
        _check(
            "SELECT SUM(amount) FROM recent_orders",
            sources={"recent_orders": "olist_orders"},
        )
    )

    assert result == [[30]]


@pytest.mark.parametrize(
    ("verification", "message"),
    [
        ({"method": "python", "expression": "SELECT 1", "sources": {}}, "method"),
        ({"method": "sql", "expression": "", "sources": {}}, "expression"),
        ({"method": "sql", "expression": 1, "sources": {}}, "expression"),
        ({"method": "sql", "expression": "SELECT 1", "sources": ["orders"]}, "sources"),
    ],
)
def test_rejects_malformed_verification(
    tmp_path: Path, verification: dict, message: str
) -> None:
    orders = _write_csv(tmp_path / "orders.csv", "amount\n1\n")
    executor = DuckDBAuditExecutor({"orders": orders})
    check = AuditCheck(path="insights[2]", expected=1, verification=verification)

    with pytest.raises(DuckDBAuditError, match=rf"insights\[2\].*{message}"):
        executor(check)


def test_rejects_unknown_and_mismatched_source_declarations(tmp_path: Path) -> None:
    orders = _write_csv(tmp_path / "orders.csv", "amount\n1\n")
    executor = DuckDBAuditExecutor({"orders": orders})

    with pytest.raises(DuckDBAuditError, match="unknown source.*missing"):
        executor(_check("SELECT COUNT(*) FROM missing", sources={"missing": "missing"}))
    with pytest.raises(DuckDBAuditError, match="unknown source.*other"):
        executor(_check("SELECT COUNT(*) FROM orders", sources={"orders": "other"}))
    with pytest.raises(DuckDBAuditError, match="Catalog|orders"):
        executor(_check("SELECT COUNT(*) FROM orders", sources={}))


@pytest.mark.parametrize(
    "expression",
    [
        "SELECT 1; SELECT 2",
        "SELECT 1; -- hide the mutation\nDROP TABLE orders",
        "/* harmless-looking */ CoPy orders TO 'stolen.csv'",
        "ATTACH 'other.db' AS other",
        "PRAGMA version",
        "SET enable_external_access = true",
        "INSTALL httpfs",
        "LOAD httpfs",
        "CALL checkpoint()",
        "SELECT * FROM read_csv('secret.csv')",
        "SELECT * FROM READ_PARQUET('secret.parquet')",
        "WITH x AS (SELECT * FROM parquet_scan('secret.parquet')) SELECT * FROM x",
    ],
)
def test_rejects_unsafe_sql(tmp_path: Path, expression: str) -> None:
    orders = _write_csv(tmp_path / "orders.csv", "amount\n1\n")
    executor = DuckDBAuditExecutor({"orders": orders})

    with pytest.raises(DuckDBAuditError, match="insights\\[0\\].*(read-only|unsafe|single)"):
        executor(_check(expression))


@pytest.mark.parametrize(
    "expression",
    [
        "SELECT amount FROM orders",
        "SELECT amount, amount + 1 FROM orders LIMIT 1",
    ],
)
def test_requires_exactly_one_row_and_column(
    tmp_path: Path, expression: str
) -> None:
    orders = _write_csv(tmp_path / "orders.csv", "amount\n1\n2\n")
    executor = DuckDBAuditExecutor({"orders": orders})

    with pytest.raises(
        DuckDBAuditError,
        match=r"insights\[0\].*exactly one row and one column",
    ):
        executor(_check(expression))


@pytest.mark.parametrize("alias", ["", "two words", "x;DROP", "1orders", "a.b"])
def test_rejects_invalid_aliases(tmp_path: Path, alias: str) -> None:
    source = _write_csv(tmp_path / "source.csv", "amount\n1\n")

    with pytest.raises(DuckDBAuditError, match="invalid source alias"):
        DuckDBAuditExecutor({alias: source})


def test_rejects_case_insensitive_alias_collisions(tmp_path: Path) -> None:
    first = _write_csv(tmp_path / "first.csv", "amount\n1\n")
    second = _write_csv(tmp_path / "second.csv", "amount\n2\n")

    with pytest.raises(DuckDBAuditError, match="duplicate source alias"):
        DuckDBAuditExecutor({"orders": first, "ORDERS": second})


def test_rejects_missing_source_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(DuckDBAuditError, match="source file.*does not exist"):
        DuckDBAuditExecutor({"orders": missing})


def test_missing_duckdb_has_actionable_optional_dependency_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_csv(tmp_path / "source.csv", "amount\n1\n")
    real_import_module = importlib.import_module

    def missing_duckdb(name: str, package: str | None = None):
        if name == "duckdb":
            raise ModuleNotFoundError("No module named 'duckdb'", name="duckdb")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_duckdb)

    with pytest.raises(
        DuckDBAuditError,
        match=r"optional.*duckdb.*fabric-rlm\[analytics\]",
    ):
        DuckDBAuditExecutor({"orders": source})
