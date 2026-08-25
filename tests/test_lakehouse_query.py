"""Tests for catalog-bounded Lakehouse queries executed in the parent process."""

from __future__ import annotations

import pytest

from fabric_rlm import LakehouseSource


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
