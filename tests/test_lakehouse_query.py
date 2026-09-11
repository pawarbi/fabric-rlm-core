"""Tests for catalog-bounded Lakehouse queries executed in the parent process."""

from __future__ import annotations

import json
import sys
import threading
from types import SimpleNamespace

import pytest

from fabric_rlm import LakehouseSource
from fabric_rlm import lakehouse as lakehouse_module


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


def test_lakehouse_query_rejects_catalog_name_collisions_after_construction() -> None:
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[
            {"kind": "csv", "name": "files.orders", "path": "orders.csv"},
            {"kind": "parquet", "name": "files.customers", "path": "customers.parquet"},
        ],
    )
    assert source.catalog is not None
    source.catalog[1]["name"] = "Files.Orders"

    with pytest.raises(ValueError, match="unique names"):
        source.query(
            "SELECT * FROM orders",
            sources={"orders": "files.orders"},
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
        "SELECT * FROM unnest([1, 2, 3])",
        "SELECT * FROM generate_series(1, 10)",
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
    ("operator", "expected"),
    [
        ("UNION", [[1], [2], [3]]),
        ("EXCEPT", [[1]]),
        ("INTERSECT", [[2]]),
    ],
)
def test_lakehouse_query_allows_set_operations_over_authorized_sources(
    operator: str,
    expected: list[list[int]],
    tmp_path,
) -> None:
    csv_path = tmp_path / "values.csv"
    csv_path.write_text("value\n1\n2\n3\n", encoding="utf-8")
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[{"kind": "csv", "name": "files.values", "path": str(csv_path)}],
    )

    result = source.query(
        f"""
        SELECT value FROM values WHERE value <= 2
        {operator}
        SELECT value FROM values WHERE value >= 2
        ORDER BY value
        """,
        sources={"values": "files.values"},
    )

    assert result["rows"] == expected


def test_lakehouse_query_allows_bounded_recursive_ctes_from_authorized_sources(
    tmp_path,
) -> None:
    csv_path = tmp_path / "values.csv"
    csv_path.write_text("value\n1\n", encoding="utf-8")
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[{"kind": "csv", "name": "files.values", "path": str(csv_path)}],
    )

    result = source.query(
        """
        WITH RECURSIVE running(value, depth) AS (
            SELECT value, 1 FROM values
            UNION ALL
            SELECT value + 1, depth + 1
            FROM running
            WHERE depth < 3
        )
        SELECT value FROM running ORDER BY value
        """,
        sources={"values": "files.values"},
    )

    assert result["rows"] == [[1], [2], [3]]


def test_lakehouse_query_allows_internal_side_effect_free_window_functions(
    tmp_path,
) -> None:
    csv_path = tmp_path / "values.csv"
    csv_path.write_text("value\n3\n1\n2\n", encoding="utf-8")
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[{"kind": "csv", "name": "files.values", "path": str(csv_path)}],
    )

    result = source.query(
        """
        SELECT value, row_number() OVER (ORDER BY value) AS row_number
        FROM values
        ORDER BY value
        """,
        sources={"values": "files.values"},
    )

    assert result["rows"] == [[1, 1], [2, 2], [3, 3]]


@pytest.mark.parametrize(
    ("window_type", "function_name"),
    [
        ("WINDOW_CUME_DIST", "cume_dist"),
        ("WINDOW_FIRST_VALUE", "first_value"),
        ("WINDOW_LAG", "lag"),
        ("WINDOW_LAST_VALUE", "last_value"),
        ("WINDOW_LEAD", "lead"),
        ("WINDOW_NTH_VALUE", "nth_value"),
        ("WINDOW_NTILE", "ntile"),
        ("WINDOW_PERCENT_RANK", "percent_rank"),
        ("WINDOW_RANK", "rank"),
        ("WINDOW_RANK_DENSE", "dense_rank"),
        ("WINDOW_ROW_NUMBER", "row_number"),
    ],
)
def test_catalog_validation_allows_builtin_window_missing_from_function_catalog(
    window_type: str,
    function_name: str,
) -> None:
    document = json.loads(_serialized_select("values"))
    document["statements"][0]["node"]["select_list"] = [
        {
            "class": "WINDOW",
            "type": window_type,
            "function_name": function_name,
            "schema": "",
            "catalog": "",
        }
    ]

    class _Cursor:
        def __init__(self, *, serialized=None, rows=None) -> None:
            self._serialized = serialized
            self._rows = rows or []

        def fetchone(self):
            return (self._serialized,)

        def fetchall(self):
            return self._rows

    class _Connection:
        def execute(self, sql, _parameters=None):
            if sql == "SELECT json_serialize_sql(?)":
                return _Cursor(serialized=json.dumps(document))
            if sql.startswith("SELECT DISTINCT function_name "):
                return _Cursor()
            raise AssertionError(f"Unexpected SQL: {sql}")

    assert (
        lakehouse_module._validate_catalog_query(
            _Connection(),
            f"SELECT {function_name}() OVER () FROM values",
            aliases=["values"],
        )
        == f"SELECT {function_name}() OVER () FROM values"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT unregistered_transform(id) FROM companies",
        "SELECT current_query() FROM companies",
        "SELECT unregistered_transform(id) OVER () FROM companies",
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


# ---------------------------------------------------- Delta tables via Parquet --


def _delta_source(name: str, path, columns) -> LakehouseSource:
    return LakehouseSource(
        "file:///lakehouse",
        catalog=[{"kind": "delta", "name": name, "path": str(path), "columns": columns}],
    )


def _void_table(root, name: str, with_deletion_vectors: bool = False):
    """A Delta table whose schema declares a Spark void column, which the Delta readers reject."""
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    table = root / name
    (table / "_delta_log").mkdir(parents=True)
    parquet.write_table(pyarrow.table({"id": [1, 2], "amount": [1.5, 2.5]}), str(table / "part-0.parquet"))
    schema = {"type": "struct", "fields": [
        {"name": "id", "type": "long", "nullable": True, "metadata": {}},
        {"name": "amount", "type": "double", "nullable": True, "metadata": {}},
        {"name": "tracking", "type": "void", "nullable": True, "metadata": {}},
    ]}
    protocol = {"minReaderVersion": 3, "minWriterVersion": 7, "readerFeatures": ["deletionVectors"], "writerFeatures": ["deletionVectors"]} if with_deletion_vectors else {"minReaderVersion": 1, "minWriterVersion": 2}
    lines = [
        {"protocol": protocol},
        {"metaData": {"id": name, "format": {"provider": "parquet", "options": {}}, "schemaString": json.dumps(schema), "partitionColumns": [], "configuration": {}, "createdTime": 1}},
        {"add": {"path": "part-0.parquet", "size": 1, "modificationTime": 1, "dataChange": True, "partitionValues": {}}},
    ]
    (table / "_delta_log" / "00000000000000000000.json").write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return table


def test_a_table_the_delta_reader_rejects_is_read_from_its_own_files(tmp_path) -> None:
    pytest.importorskip("duckdb")
    table = _void_table(tmp_path, "shipments")
    source = _delta_source("shipments", table, [["id", "BIGINT"], ["amount", "DOUBLE"], ["tracking", "void"]])
    result = source.query("SELECT COUNT(*) AS n, SUM(amount) AS total, COUNT(tracking) AS tracked FROM shipments", sources={"shipments": "shipments"})
    assert result["rows"] == [[2, 4.0, 0]]
    assert str(table) in lakehouse_module._DELTA_READER_REJECTED  # the failing reader attempt is not repeated

    vectors = _void_table(tmp_path, "vectors", with_deletion_vectors=True)
    source = _delta_source("vectors", vectors, [["id", "BIGINT"], ["amount", "DOUBLE"], ["tracking", "void"]])
    with pytest.raises(RuntimeError, match="cannot stand in"):
        source.query("SELECT COUNT(*) AS n FROM vectors", sources={"vectors": "vectors"})


def test_delta_tables_go_through_the_delta_reader_with_missing_catalog_columns_as_null(tmp_path) -> None:
    deltalake = pytest.importorskip("deltalake")
    pyarrow = pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    table = tmp_path / "orders"
    deltalake.write_deltalake(str(table), pyarrow.table({"order_id": [1, 2], "amount": [10.0, 20.0]}))
    deltalake.write_deltalake(str(table), pyarrow.table({"order_id": [3], "amount": [5.0]}), mode="append")
    # the catalog declares a column the table does not have yet; it comes back as NULL
    source = _delta_source("orders", table, [["order_id", "BIGINT"], ["amount", "DOUBLE"], ["tracking", "void"]])

    result = source.query(
        "SELECT COUNT(*) AS n, SUM(amount) AS total, COUNT(tracking) AS tracked FROM orders",
        sources={"orders": "orders"},
    )
    assert result["rows"] == [[3, 35.0, 0]]

    deltalake.write_deltalake(str(table), pyarrow.table({"order_id": [9], "amount": [1.0]}), mode="overwrite")
    result = source.query("SELECT order_id FROM orders", sources={"orders": "orders"})
    assert result["rows"] == [[9]]  # the overwritten files left the log, and the cache noticed


def test_partition_columns_come_from_the_file_paths(tmp_path) -> None:
    deltalake = pytest.importorskip("deltalake")
    pyarrow = pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    table = tmp_path / "sales"
    deltalake.write_deltalake(
        str(table),
        pyarrow.table({"year": [2012, 2013, 2013], "amount": [1.0, 2.0, 3.0]}),
        partition_by=["year"],
    )
    source = _delta_source("sales", table, [["year", "BIGINT"], ["amount", "DOUBLE"]])

    result = source.query(
        "SELECT year, SUM(amount) AS total FROM sales GROUP BY year ORDER BY year",
        sources={"sales": "sales"},
    )
    assert result["rows"] == [[2012, 1.0], [2013, 5.0]]


def test_log_replay_applies_removes_decodes_paths_and_defers_unsupported_features(tmp_path) -> None:
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()

    plain = tmp_path / "plain" / "_delta_log"
    plain.mkdir(parents=True)
    (plain / "00000000000000000000.json").write_text(
        '{"protocol":{"minReaderVersion":1,"minWriterVersion":2}}\n'
        '{"metaData":{"id":"t","schemaString":"{\\"type\\":\\"struct\\",\\"fields\\":[]}","partitionColumns":["a"],"configuration":{}}}\n'
        '{"add":{"path":"a=1/part-0.parquet","size":1,"modificationTime":1,"dataChange":true}}\n'
        '{"add":{"path":"part-1%20x.parquet","size":1,"modificationTime":1,"dataChange":true}}\n',
        encoding="utf-8",
    )
    (plain / "00000000000000000001.json").write_text(
        '{"remove":{"path":"part-1%20x.parquet","deletionTimestamp":1,"dataChange":true}}\n',
        encoding="utf-8",
    )
    files = lakehouse_module._delta_parquet_files(con, str(tmp_path / "plain"))
    assert files == [str(tmp_path / "plain") + "/a=1/part-0.parquet"]

    vectors = tmp_path / "dv" / "_delta_log"
    vectors.mkdir(parents=True)
    (vectors / "00000000000000000000.json").write_text(
        '{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["deletionVectors"],"writerFeatures":["deletionVectors"]}}\n'
        '{"add":{"path":"part-0.parquet","size":1,"modificationTime":1,"dataChange":true}}\n',
        encoding="utf-8",
    )
    assert lakehouse_module._delta_parquet_files(con, str(tmp_path / "dv")) is None

    mapped = tmp_path / "mapped" / "_delta_log"
    mapped.mkdir(parents=True)
    (mapped / "00000000000000000000.json").write_text(
        '{"metaData":{"id":"m","schemaString":"{\\"type\\":\\"struct\\",\\"fields\\":[]}","partitionColumns":[],"configuration":{"delta.columnMapping.mode":"name"}}}\n'
        '{"add":{"path":"part-0.parquet","size":1,"modificationTime":1,"dataChange":true}}\n',
        encoding="utf-8",
    )
    assert lakehouse_module._delta_parquet_files(con, str(tmp_path / "mapped")) is None

    gap = tmp_path / "gap" / "_delta_log"
    gap.mkdir(parents=True)
    (gap / "00000000000000000000.json").write_text('{"add":{"path":"part-0.parquet","size":1,"modificationTime":1,"dataChange":true}}\n', encoding="utf-8")
    (gap / "00000000000000000002.json").write_text('{"add":{"path":"part-2.parquet","size":1,"modificationTime":1,"dataChange":true}}\n', encoding="utf-8")
    assert lakehouse_module._delta_parquet_files(con, str(tmp_path / "gap")) is None


def test_checkpoints_seed_the_replay(tmp_path) -> None:
    duckdb = pytest.importorskip("duckdb")
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    log = tmp_path / "cp" / "_delta_log"
    log.mkdir(parents=True)
    file_type = pyarrow.struct([("path", pyarrow.string())])
    protocol_type = pyarrow.struct([("minReaderVersion", pyarrow.int32()), ("readerFeatures", pyarrow.list_(pyarrow.string()))])
    checkpoint = pyarrow.table(
        {
            "add": pyarrow.array([{"path": "old.parquet"}, {"path": "kept.parquet"}, None], type=file_type),
            "remove": pyarrow.array([None, None, {"path": "kept.parquet"}], type=file_type),  # a tombstone is history, not state
            "protocol": pyarrow.array([None, None, {"minReaderVersion": 1, "readerFeatures": None}], type=protocol_type),
        }
    )
    parquet.write_table(checkpoint, str(log / "00000000000000000010.checkpoint.parquet"))
    (log / "_last_checkpoint").write_text('{"version":10,"size":3}', encoding="utf-8")
    (log / "00000000000000000011.json").write_text(
        '{"remove":{"path":"old.parquet","deletionTimestamp":1,"dataChange":true}}\n'
        '{"add":{"path":"new.parquet","size":1,"modificationTime":1,"dataChange":true}}\n',
        encoding="utf-8",
    )

    files = lakehouse_module._delta_parquet_files(duckdb.connect(), str(tmp_path / "cp"))
    assert sorted(item.rsplit("/", 1)[-1] for item in files) == ["kept.parquet", "new.parquet"]

    # a commit missing between the checkpoint and the newest log entry: leave it to the Delta reader
    (log / "00000000000000000013.json").write_text('{"add":{"path":"later.parquet","size":1,"modificationTime":1,"dataChange":true}}\n', encoding="utf-8")
    assert lakehouse_module._delta_parquet_files(duckdb.connect(), str(tmp_path / "cp")) is None


def test_query_timeout_is_validated_and_passed_to_the_deadline(tmp_path) -> None:
    csv_path = tmp_path / "companies.csv"
    csv_path.write_text("region,mrr\nNorth America,10.5\n", encoding="utf-8")
    source = LakehouseSource(
        "file:///lakehouse",
        catalog=[{"kind": "csv", "name": "files.companies", "path": str(csv_path), "columns": [["region", "VARCHAR"], ["mrr", "DOUBLE"]]}],
    )
    assert source.query("SELECT SUM(mrr) AS total FROM companies", sources={"companies": "files.companies"}, timeout=120)["rows"] == [[10.5]]
    for bad in (0, -1, True, 10_000):
        with pytest.raises(ValueError, match="timeout"):
            source.query("SELECT 1 FROM companies", sources={"companies": "files.companies"}, timeout=bad)
