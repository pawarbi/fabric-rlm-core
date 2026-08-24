from __future__ import annotations

import pytest

from fabric_rlm import LakehouseSource
from fabric_rlm.artifacts import decode_from_worker_wire, encode_for_worker


def test_lakehouse_source_is_public_and_normalizes_scopes() -> None:
    source = LakehouseSource(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse.Lakehouse/",
        tables="Tables/dbo/",
        files=["Files/data/", "Files/reference"],
        catalog=[],
    )

    assert source.root == (
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse.Lakehouse"
    )
    assert source.tables == ("Tables/dbo",)
    assert source.files == ("Files/data", "Files/reference")
    assert source.catalog == ()


@pytest.mark.parametrize(
    ("path", "expected_root", "expected_tables", "expected_files"),
    [
        (
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse",
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse",
            ("Tables",),
            (),
        ),
        (
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables",
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse",
            ("Tables",),
            (),
        ),
        (
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/dbo",
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse",
            ("Tables/dbo",),
            (),
        ),
        (
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/dbo/orders",
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse",
            ("Tables/dbo/orders",),
            (),
        ),
        (
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Files/data",
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse",
            (),
            ("Files/data",),
        ),
    ],
)
def test_lakehouse_source_infers_scope_from_abfss_path(
    path: str,
    expected_root: str,
    expected_tables: tuple[str, ...],
    expected_files: tuple[str, ...],
) -> None:
    source = LakehouseSource(path, catalog=[])

    assert source.root == expected_root
    assert source.tables == expected_tables
    assert source.files == expected_files


def test_one_lakehouse_source_can_include_tables_and_files() -> None:
    source = LakehouseSource(
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse",
        tables=["Tables/dbo", "Tables/reference"],
        files=["Files/data", "Files/lookups"],
        catalog=[],
    )

    assert source.tables == ("Tables/dbo", "Tables/reference")
    assert source.files == ("Files/data", "Files/lookups")


def test_multiple_lakehouse_sources_round_trip_in_one_input() -> None:
    sources = [
        LakehouseSource(
            "abfss://ws1@onelake.dfs.fabric.microsoft.com/lh1/Tables",
            catalog=[{"kind": "delta", "name": "orders", "path": "abfss://one"}],
        ),
        LakehouseSource(
            "abfss://ws2@onelake.dfs.fabric.microsoft.com/lh2/Files/data",
            catalog=[{"kind": "csv", "name": "targets", "path": "abfss://two"}],
        ),
    ]

    decoded = decode_from_worker_wire(encode_for_worker(sources))

    assert decoded == sources


def test_lakehouse_source_requires_a_root_or_explicit_catalog() -> None:
    with pytest.raises(ValueError, match="root"):
        LakehouseSource("", catalog=[])


def test_lakehouse_source_validates_catalog_entries() -> None:
    with pytest.raises(ValueError, match="kind, name, and path"):
        LakehouseSource(
            "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse",
            catalog=[{"kind": "delta", "name": "dbo.orders"}],
        )


def test_explicit_catalog_round_trips_through_worker_wire() -> None:
    source = LakehouseSource(
        "abfss://workspace-id@onelake.dfs.fabric.microsoft.com/lakehouse-id",
        catalog=[
            {
                "kind": "delta",
                "name": "dbo.orders",
                "path": (
                    "abfss://workspace-id@onelake.dfs.fabric.microsoft.com/"
                    "lakehouse-id/Tables/dbo/orders"
                ),
                "columns": [["order_id", "BIGINT"]],
            }
        ],
    )

    decoded = decode_from_worker_wire(encode_for_worker(source))

    assert isinstance(decoded, LakehouseSource)
    assert decoded.root == source.root
    assert decoded.catalog == source.catalog
    assert decoded.is_resolved


def test_lakehouse_source_freezes_as_compact_catalog_input() -> None:
    source = LakehouseSource(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse.Lakehouse",
        catalog=[
            {
                "kind": "csv",
                "name": "files.orders",
                "path": "abfss://workspace/path/Files/data/orders.csv",
                "columns": [["order_id", "BIGINT"]],
            }
        ],
    )

    assert source.__frozen__() == {
        "root": source.root,
        "tables": ["Tables"],
        "files": [],
        "catalog": [
            {
                "kind": "csv",
                "name": "files.orders",
                "path": "abfss://workspace/path/Files/data/orders.csv",
                "columns": [["order_id", "BIGINT"]],
            }
        ],
    }
