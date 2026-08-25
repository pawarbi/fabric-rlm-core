"""Tests for catalog-bounded Lakehouse queries executed in the parent process."""

from __future__ import annotations

import json
import sys
import threading
from types import SimpleNamespace

import pytest

from fabric_rlm import LakehouseSource


def _serialized_select(table_name: str) -> str:
    return json.dumps(
        {
            "error": False,
            "statements": [
                {
                    "node": {
                        "type": "SELECT_NODE",
                        "cte_map": {"map": []},
                        "from_table": {
                            "type": "BASE_TABLE",
                            "table_name": table_name,
                            "catalog_name": "",
                            "schema_name": "",
                            "at_clause": None,
                        },
                    }
                }
            ],
        }
    )


class _SelectStatement:
    type = "StatementType.SELECT"


class _MetadataCursor:
    def __init__(self, table_name: str) -> None:
        self._table_name = table_name

    def fetchone(self):
        return (_serialized_select(self._table_name),)

    def fetchall(self):
        return []


def test_lakehouse_query_reads_only_named_catalog_sources(tmp_path) -> None:
    csv_path = tmp_path / "companies.csv"
    csv_path.write_text(
        "region,mrr\nNorth America,10.5\nEurope,7.0\nNorth America,4.5\n",
        encoding="utf-8",
    )
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[
            {
                "kind": "csv",
                "name": "files.companies",
                "path": str(csv_path),
                "columns": [["region", "VARCHAR"], ["mrr", "DOUBLE"]],
            }
        ],
    )

    result = source.query(
        """
        SELECT region, SUM(mrr) AS active_mrr
        FROM companies
        GROUP BY region
        ORDER BY active_mrr DESC
        """,
        sources={"companies": "files.companies"},
    )

    assert result == {
        "columns": ["region", "active_mrr"],
        "rows": [["North America", 15.0], ["Europe", 7.0]],
        "truncated": False,
    }


def test_lakehouse_query_rejects_sources_outside_the_catalog() -> None:
    source = LakehouseSource(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse",
        catalog=[
            {"kind": "delta", "name": "dbo.companies", "path": "abfss://companies"}
        ],
    )

    with pytest.raises(ValueError, match="not in this LakehouseSource catalog"):
        source.query(
            "SELECT * FROM subscriptions",
            sources={"subscriptions": "dbo.subscriptions"},
        )


@pytest.mark.parametrize(
    "sql",
    [
        "COPY (SELECT * FROM companies) TO 'out.csv'",
        "SELECT * FROM read_csv_auto('C:/secrets.txt')",
        "SELECT * FROM read_csv_auto/**/('C:/secrets.txt')",
        "SELECT * FROM delta_scan('abfss://other/Tables/private')",
        "SELECT * FROM duckdb_secrets()",
        "PRAGMA version",
    ],
)
def test_lakehouse_query_rejects_external_or_non_query_sql(sql: str) -> None:
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[
            {"kind": "csv", "name": "files.companies", "path": "companies.csv"}
        ],
    )

    with pytest.raises(ValueError, match="read-only catalog query"):
        source.query(sql, sources={"companies": "files.companies"})


def test_lakehouse_query_rejects_dynamic_sql_that_reads_local_files(
    tmp_path,
) -> None:
    payload_path = tmp_path / "payload.txt"
    payload_path.write_text("must-not-be-readable", encoding="utf-8")
    csv_path = tmp_path / "companies.csv"
    csv_path.write_text("id\n1\n", encoding="utf-8")
    escaped_path = str(payload_path).replace("\\", "/").replace("'", "''")
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[
            {"kind": "csv", "name": "files.companies", "path": str(csv_path)}
        ],
    )

    with pytest.raises(ValueError, match="read-only catalog query"):
        source.query(
            "SELECT * FROM query("
            "'SELECT content FROM rea' || "
            f"'d_text(''{escaped_path}'')'"
            ")",
            sources={"companies": "files.companies"},
        )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM parquet_metadata('private.parquet')",
        "SELECT (SELECT count(*) FROM query('SELECT * FROM companies'))",
        "SELECT * FROM (SELECT * FROM query('SELECT * FROM companies')) nested",
    ],
)
def test_lakehouse_query_rejects_all_user_table_functions(sql: str, tmp_path) -> None:
    csv_path = tmp_path / "companies.csv"
    csv_path.write_text("id\n1\n", encoding="utf-8")
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[
            {"kind": "csv", "name": "files.companies", "path": str(csv_path)}
        ],
    )

    with pytest.raises(ValueError, match="read-only catalog query"):
        source.query(sql, sources={"companies": "files.companies"})


def test_lakehouse_query_allows_ctes_derived_from_authorized_sources(
    tmp_path,
) -> None:
    csv_path = tmp_path / "companies.csv"
    csv_path.write_text("region,mrr\nNorth America,10\nEurope,7\n", encoding="utf-8")
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[
            {"kind": "csv", "name": "files.companies", "path": str(csv_path)}
        ],
    )

    result = source.query(
        """
        WITH regional AS (
            SELECT region, SUM(mrr) AS total_mrr
            FROM companies
            GROUP BY region
        )
        SELECT region, total_mrr
        FROM regional
        WHERE total_mrr >= (SELECT MIN(mrr) FROM companies)
        ORDER BY total_mrr DESC
        """,
        sources={"companies": "files.companies"},
    )

    assert result["rows"] == [["North America", 10], ["Europe", 7]]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT unregistered_transform(id) FROM companies",
        "SELECT current_query() FROM companies",
    ],
)
def test_lakehouse_query_rejects_unrecognized_or_side_effecting_functions(
    sql: str,
    tmp_path,
) -> None:
    csv_path = tmp_path / "companies.csv"
    csv_path.write_text("id\n1\n", encoding="utf-8")
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[
            {"kind": "csv", "name": "files.companies", "path": str(csv_path)}
        ],
    )

    with pytest.raises(ValueError, match="read-only catalog query"):
        source.query(sql, sources={"companies": "files.companies"})


def test_lakehouse_query_rejects_unbounded_result_limits() -> None:
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[
            {"kind": "csv", "name": "files.companies", "path": "companies.csv"}
        ],
    )

    with pytest.raises(ValueError, match="at most 10000"):
        source.query(
            "SELECT * FROM companies",
            sources={"companies": "files.companies"},
            max_rows=10_001,
        )


def test_lakehouse_query_bounds_returned_rows(tmp_path) -> None:
    csv_path = tmp_path / "values.csv"
    csv_path.write_text("value\n1\n2\n3\n", encoding="utf-8")
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[
            {
                "kind": "csv",
                "name": "files.values",
                "path": str(csv_path),
            }
        ],
    )

    result = source.query(
        "SELECT value FROM values ORDER BY value",
        sources={"values": "files.values"},
        max_rows=2,
    )

    assert result == {
        "columns": ["value"],
        "rows": [[1], [2]],
        "truncated": True,
    }


def test_lakehouse_query_rejects_results_above_transfer_limit(
    monkeypatch,
    tmp_path,
) -> None:
    csv_path = tmp_path / "values.csv"
    csv_path.write_text(f"value\n{'x' * 100}\n", encoding="utf-8")
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[
            {"kind": "csv", "name": "files.values", "path": str(csv_path)}
        ],
    )
    monkeypatch.setattr("fabric_rlm.lakehouse._MAX_QUERY_RESULT_BYTES", 50)

    with pytest.raises(ValueError, match="transfer limit"):
        source.query(
            "SELECT value FROM values",
            sources={"values": "files.values"},
        )


def test_lakehouse_query_fetches_and_sizes_results_incrementally(
    monkeypatch,
) -> None:
    class _ResultCursor:
        description = [("value",)]

        def __init__(self) -> None:
            self._rows = iter([(1,), (2,), (3,)])

        def fetchmany(self, size):
            rows = []
            for _ in range(size):
                try:
                    rows.append(next(self._rows))
                except StopIteration:
                    break
            return rows

        def fetchall(self):
            raise AssertionError("Lakehouse queries must not materialize all rows.")

    class _Connection:
        def __init__(self) -> None:
            self.settings = []

        def extract_statements(self, _sql):
            return [_SelectStatement()]

        def sql(self, _sql):
            return None

        def execute(self, sql, parameters=None):
            if sql == "SELECT json_serialize_sql(?)":
                return _MetadataCursor("values")
            if sql.startswith("SELECT DISTINCT function_name "):
                return _MetadataCursor("values")
            if sql.startswith("SET "):
                self.settings.append((sql, parameters))
                return self
            if sql.startswith("CREATE TEMP VIEW"):
                return self
            if sql.startswith("SELECT * FROM ("):
                return _ResultCursor()
            return self

        def close(self):
            return None

    connection = _Connection()
    monkeypatch.setitem(
        sys.modules,
        "duckdb",
        SimpleNamespace(connect=lambda: connection),
    )
    monkeypatch.setattr("fabric_rlm.lakehouse._QUERY_MEMORY_LIMIT", "64MB")
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[{"kind": "csv", "name": "files.values", "path": "values.csv"}],
    )

    result = source.query(
        "SELECT value FROM values",
        sources={"values": "files.values"},
        max_rows=2,
    )

    assert result == {
        "columns": ["value"],
        "rows": [[1], [2]],
        "truncated": True,
    }
    assert ("SET memory_limit = ?", ["64MB"]) in connection.settings
    assert ("SET temp_directory = ''", None) in connection.settings


def test_lakehouse_query_stops_after_first_oversized_row(monkeypatch) -> None:
    class _ResultCursor:
        description = [("value",)]

        def __init__(self) -> None:
            self._calls = 0

        def fetchmany(self, _size):
            self._calls += 1
            if self._calls == 1:
                return [("x" * 100,)]
            raise AssertionError("Fetching must stop once the transfer limit is exceeded.")

    class _Connection:
        def extract_statements(self, _sql):
            return [_SelectStatement()]

        def sql(self, _sql):
            return None

        def execute(self, sql, _parameters=None):
            if sql == "SELECT json_serialize_sql(?)":
                return _MetadataCursor("values")
            if sql.startswith("SELECT DISTINCT function_name "):
                return _MetadataCursor("values")
            if sql.startswith("SELECT * FROM ("):
                return _ResultCursor()
            return self

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "duckdb",
        SimpleNamespace(connect=lambda: _Connection()),
    )
    monkeypatch.setattr("fabric_rlm.lakehouse._MAX_QUERY_RESULT_BYTES", 50)
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[{"kind": "csv", "name": "files.values", "path": "values.csv"}],
    )

    with pytest.raises(ValueError, match="transfer limit"):
        source.query(
            "SELECT value FROM values",
            sources={"values": "files.values"},
        )


def test_lakehouse_query_interrupts_execution_after_deadline(monkeypatch) -> None:
    class _Connection:
        def __init__(self) -> None:
            self.interrupted = threading.Event()

        def extract_statements(self, _sql):
            return [_SelectStatement()]

        def sql(self, _sql):
            return None

        def execute(self, sql, _parameters=None):
            if sql == "SELECT json_serialize_sql(?)":
                return _MetadataCursor("values")
            if sql.startswith("SELECT DISTINCT function_name "):
                return _MetadataCursor("values")
            if sql.startswith("SELECT * FROM ("):
                if not self.interrupted.wait(0.2):
                    raise RuntimeError("query completed without interruption")
                raise RuntimeError("INTERRUPT Error: interrupted")
            return self

        def interrupt(self):
            self.interrupted.set()

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "duckdb",
        SimpleNamespace(connect=lambda: _Connection()),
    )
    monkeypatch.setattr("fabric_rlm.lakehouse._QUERY_TIMEOUT_SECONDS", 0.01)
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[{"kind": "csv", "name": "files.values", "path": "values.csv"}],
    )

    with pytest.raises(TimeoutError, match="deadline"):
        source.query(
            "SELECT value FROM values",
            sources={"values": "files.values"},
        )


def test_lakehouse_query_interrupts_real_duckdb_work(monkeypatch, tmp_path) -> None:
    pytest.importorskip("duckdb")
    csv_path = tmp_path / "values.csv"
    csv_path.write_text(
        "value\n" + "\n".join(str(value) for value in range(500)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("fabric_rlm.lakehouse._QUERY_TIMEOUT_SECONDS", 0.01)
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[{"kind": "csv", "name": "files.values", "path": str(csv_path)}],
    )

    with pytest.raises(TimeoutError, match="deadline"):
        source.query(
            """
            SELECT SUM(a.value * b.value + c.value * d.value)
            FROM values a
            CROSS JOIN values b
            CROSS JOIN values c
            CROSS JOIN values d
            """,
            sources={"values": "files.values"},
        )


def test_lakehouse_query_redacts_storage_token_from_errors(monkeypatch) -> None:
    token = "sensitive-storage-token"

    class _Statement:
        type = "StatementType.SELECT"

    class _Cursor:
        def fetchone(self):
            return (
                json.dumps(
                    {
                        "error": False,
                        "statements": [
                            {
                                "node": {
                                    "type": "SELECT_NODE",
                                    "cte_map": {"map": []},
                                    "from_table": {
                                        "type": "BASE_TABLE",
                                        "table_name": "values",
                                        "catalog_name": "",
                                        "schema_name": "",
                                        "at_clause": None,
                                    },
                                }
                            }
                        ],
                    }
                ),
            )

        def fetchall(self):
            return []

    class _Connection:
        def extract_statements(self, _sql):
            return [_Statement()]

        def sql(self, _sql):
            return None

        def execute(self, sql, _parameters=None):
            if sql == "SELECT json_serialize_sql(?)" or sql.startswith(
                "SELECT DISTINCT function_name "
            ):
                return _Cursor()
            if sql.startswith("CREATE TEMP VIEW"):
                raise RuntimeError(f"storage failure for {token}")
            return self

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "duckdb",
        SimpleNamespace(connect=lambda: _Connection()),
    )
    monkeypatch.setattr("fabric_rlm.lakehouse._storage_token", lambda: token)
    source = LakehouseSource(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse",
        catalog=[
            {
                "kind": "csv",
                "name": "files.values",
                "path": "abfss://workspace/lakehouse/Files/values.csv",
            }
        ],
    )

    with pytest.raises(RuntimeError) as exc_info:
        source.query(
            "SELECT value FROM values",
            sources={"values": "files.values"},
        )

    assert token not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)
