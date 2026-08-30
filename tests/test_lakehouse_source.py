from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from fabric_rlm import LakehouseSource, RLM
from fabric_rlm import lakehouse as lakehouse_module
from fabric_rlm.artifacts import decode_from_worker_wire, encode_for_worker
from fabric_rlm.lakehouse import LakehouseDiscoveryError, resolve_lakehouse_inputs


def test_delta_schema_discovery_uses_current_pyarrow_conversion(
    monkeypatch,
) -> None:
    fields = [
        SimpleNamespace(name="company_id", type="Int64"),
        SimpleNamespace(name="region", type="Utf8"),
    ]

    class Schema:
        def to_pyarrow(self):
            return fields

    class DeltaTable:
        def __init__(self, path, storage_options=None):
            assert path == "local-delta-table"
            assert storage_options is None

        def schema(self):
            return Schema()

    monkeypatch.setitem(
        sys.modules,
        "deltalake",
        SimpleNamespace(DeltaTable=DeltaTable),
    )

    assert lakehouse_module._read_delta_columns("local-delta-table") == [
        ["company_id", "Int64"],
        ["region", "Utf8"],
    ]


def test_delta_schema_discovery_supports_arrow_conversion(
    monkeypatch,
) -> None:
    fields = [SimpleNamespace(name="invoice_id", type="Int64")]

    class Schema:
        def to_arrow(self):
            return fields

    class DeltaTable:
        def __init__(self, _path, storage_options=None):
            assert storage_options is None

        def schema(self):
            return Schema()

    monkeypatch.setitem(
        sys.modules,
        "deltalake",
        SimpleNamespace(DeltaTable=DeltaTable),
    )

    assert lakehouse_module._read_delta_columns("local-delta-table") == [
        ["invoice_id", "Int64"],
    ]


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


def test_lakehouse_source_rejects_duplicate_explicit_catalog_names() -> None:
    with pytest.raises(ValueError, match="unique names"):
        LakehouseSource(
            "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse",
            catalog=[
                {
                    "kind": "csv",
                    "name": "files.orders",
                    "path": "abfss://root/Files/orders.csv",
                },
                {
                    "kind": "parquet",
                    "name": "Files.Orders",
                    "path": "abfss://root/Files/orders.parquet",
                },
            ],
        )


@pytest.mark.parametrize(
    "scope",
    [
        "../Tables",
        "Files/../secrets",
        r"Files\data",
        "Tables/%2e%2e/private",
        "Tables//private",
        "Tables/private?version=1",
        "Tables/private#fragment",
        "Tables/dbo/Files/private",
        "Files/data/Tables/private",
    ],
)
def test_lakehouse_source_rejects_unsafe_relative_scopes(scope: str) -> None:
    with pytest.raises(ValueError, match="relative paths"):
        LakehouseSource(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh",
            tables=scope,
        )


@pytest.mark.parametrize(
    "path",
    [
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh/Tables/../Files/private",
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh/Tables/%2e%2e/Files/private",
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh/Tables/dbo/Files/private",
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh//Tables",
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh/Tables?version=1",
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh/Tables#fragment",
    ],
)
def test_lakehouse_source_rejects_unsafe_inferred_scopes(path: str) -> None:
    with pytest.raises(ValueError, match="canonical|relative paths"):
        LakehouseSource(path, catalog=[])


@pytest.mark.parametrize(
    "root",
    [
        "abfss://onelake.dfs.fabric.microsoft.com/lh",
        "abfss://@onelake.dfs.fabric.microsoft.com/lh",
        "abfss://ws@other.example/lh",
        "abfss://user:password@onelake.dfs.fabric.microsoft.com/lh",
        "abfss://ws@onelake.dfs.fabric.microsoft.com:443/lh",
        "abfss://ws@onelake.dfs.fabric.microsoft.com/",
    ],
)
def test_lakehouse_source_rejects_malformed_abfss_roots(root: str) -> None:
    with pytest.raises(ValueError, match="canonical"):
        LakehouseSource(root, catalog=[])


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


def test_lakehouse_source_lists_and_filters_resolved_catalog() -> None:
    source = LakehouseSource(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse",
        catalog=[
            {
                "kind": "delta",
                "name": "dbo.orders",
                "path": "abfss://root/Tables/dbo/orders",
                "columns": [["order_id", "BIGINT"], ["amount", "DOUBLE"]],
            },
            {
                "kind": "csv",
                "name": "files.data.targets",
                "path": "abfss://root/Files/data/targets.csv",
                "columns": [["region", "VARCHAR"]],
            },
        ],
    )

    assert [item["name"] for item in source.list_sources()] == [
        "dbo.orders",
        "files.data.targets",
    ]
    assert [item["name"] for item in source.list_sources(kind="DELTA")] == [
        "dbo.orders"
    ]


def test_lakehouse_source_finds_sources_by_name_path_or_column() -> None:
    source = LakehouseSource(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse",
        catalog=[
            {
                "kind": "delta",
                "name": "dbo.usage_logs",
                "path": "abfss://root/Tables/dbo/usage_logs",
                "columns": [["user_id", "BIGINT"], ["api_calls", "BIGINT"]],
            },
            {
                "kind": "csv",
                "name": "files.data.support",
                "path": "abfss://root/Files/data/support.csv",
                "columns": [["ticket_id", "VARCHAR"]],
            },
        ],
    )

    assert source.find_sources("usage")[0]["name"] == "dbo.usage_logs"
    assert source.find_sources("api_calls")[0]["name"] == "dbo.usage_logs"
    assert source.find_sources("Files/data")[0]["name"] == "files.data.support"
    assert source.find_sources("ticket", kind="delta") == ()


def test_lakehouse_source_catalog_helpers_return_defensive_copies() -> None:
    source = LakehouseSource(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse",
        catalog=[
            {
                "kind": "delta",
                "name": "dbo.orders",
                "path": "abfss://root/Tables/dbo/orders",
                "columns": [["order_id", "BIGINT"]],
            }
        ],
    )

    listed = source.list_sources()
    listed[0]["columns"][0][0] = "changed"

    assert source.catalog[0]["columns"][0][0] == "order_id"


def test_lakehouse_source_description_explains_resolved_worker_contract() -> None:
    source = LakehouseSource(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse",
        catalog=[
            {"kind": "delta", "name": "dbo.orders", "path": "abfss://orders"}
        ],
    )

    description = source.__rlm_describe__()

    assert "resolved" in description
    assert "1 source" in description
    assert "list_sources" in description
    assert "find_sources" in description
    assert ".query(" in description
    assert "credentials remain in the parent" in description
    assert "Do not call notebookutils" in description


def test_lakehouse_source_find_requires_a_non_empty_query() -> None:
    source = LakehouseSource(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse",
        catalog=[],
    )

    with pytest.raises(ValueError, match="query"):
        source.find_sources(" ")


class _Item:
    def __init__(self, path: str, *, is_dir: bool):
        self.path = path
        self.name = path.rstrip("/").rsplit("/", 1)[-1]
        self.isDir = is_dir


class _FakeFS:
    def __init__(self, listings: dict[str, list[_Item]], heads: dict[str, str] | None = None):
        self.listings = listings
        self.heads = heads or {}

    def ls(self, path: str):
        if path not in self.listings:
            raise FileNotFoundError(path)
        return self.listings[path]

    def head(self, path: str, _max_bytes: int):
        return self.heads[path]


def test_auto_discovery_rejects_duplicate_normalized_names(monkeypatch) -> None:
    root = "abfss://ws@onelake.dfs.fabric.microsoft.com/lh"
    files = f"{root}/Files"
    csv_path = f"{files}/orders.csv"
    parquet_path = f"{files}/orders.parquet"
    fs = _FakeFS(
        {
            files: [
                _Item(csv_path, is_dir=False),
                _Item(parquet_path, is_dir=False),
            ]
        },
        {csv_path: "order_id\n1\n"},
    )
    monkeypatch.setattr("fabric_rlm.lakehouse._get_fs", lambda: fs)

    with pytest.raises(LakehouseDiscoveryError, match="duplicate catalog name"):
        LakehouseSource(root, tables=[], files="Files").resolve()


def test_explicit_catalog_bypasses_auto_discovery(monkeypatch) -> None:
    source = LakehouseSource(
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh",
        catalog=[{"kind": "delta", "name": "orders", "path": "abfss://orders"}],
    )
    monkeypatch.setattr(
        "fabric_rlm.lakehouse.build_lakehouse_catalog",
        lambda _source: pytest.fail("explicit catalog must bypass discovery"),
    )

    assert source.resolve() is source


def test_auto_discovery_builds_delta_and_files_catalog(monkeypatch) -> None:
    root = "abfss://ws@onelake.dfs.fabric.microsoft.com/lh"
    tables = f"{root}/Tables"
    dbo = f"{tables}/dbo"
    orders = f"{dbo}/orders"
    products = f"{tables}/products"
    files = f"{root}/Files/data"
    nested = f"{files}/archive"
    orders_csv = f"{files}/orders.csv"
    readme = f"{nested}/README.txt"
    fs = _FakeFS(
        {
            tables: [_Item(dbo, is_dir=True), _Item(products, is_dir=True)],
            dbo: [_Item(orders, is_dir=True)],
            orders: [_Item(f"{orders}/_delta_log", is_dir=True)],
            products: [_Item(f"{products}/_delta_log", is_dir=True)],
            files: [_Item(orders_csv, is_dir=False), _Item(nested, is_dir=True)],
            nested: [_Item(readme, is_dir=False)],
        },
        {orders_csv: "order_id,amount\n1,10.5\n"},
    )
    monkeypatch.setattr("fabric_rlm.lakehouse._get_fs", lambda: fs)
    monkeypatch.setattr(
        "fabric_rlm.lakehouse._read_delta_columns",
        lambda path: [["id", "BIGINT"], ["source", path]],
    )

    resolved = LakehouseSource(root, files="Files/data").resolve()

    assert resolved.is_resolved
    assert [entry["name"] for entry in resolved.catalog] == [
        "dbo.orders",
        "files.data.archive.README",
        "files.data.orders",
        "products",
    ]
    assert resolved.catalog[0]["kind"] == "delta"
    assert resolved.catalog[0]["columns"][0] == ["id", "BIGINT"]
    csv_entry = next(entry for entry in resolved.catalog if entry["kind"] == "csv")
    assert csv_entry["columns"] == [
        ["order_id", "UNKNOWN"],
        ["amount", "UNKNOWN"],
    ]


def test_specific_delta_table_scope_resolves_without_sibling_discovery(
    monkeypatch,
) -> None:
    table = (
        "abfss://ws@onelake.dfs.fabric.microsoft.com/"
        "lh/Tables/dbo/orders"
    )
    fs = _FakeFS({table: [_Item(f"{table}/_delta_log", is_dir=True)]})
    monkeypatch.setattr("fabric_rlm.lakehouse._get_fs", lambda: fs)
    monkeypatch.setattr(
        "fabric_rlm.lakehouse._read_delta_columns",
        lambda _path: [["order_id", "BIGINT"]],
    )

    resolved = LakehouseSource(table).resolve()

    assert resolved.catalog == (
        {
            "kind": "delta",
            "name": "dbo.orders",
            "path": table,
            "columns": [["order_id", "BIGINT"]],
        },
    )


def test_auto_discovery_fails_when_scope_is_inaccessible(monkeypatch) -> None:
    source = LakehouseSource(
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh/Tables"
    )
    monkeypatch.setattr(
        "fabric_rlm.lakehouse._get_fs",
        lambda: _FakeFS({}),
    )

    with pytest.raises(LakehouseDiscoveryError, match="could not be listed"):
        source.resolve()


def test_auto_discovery_fails_instead_of_truncating_catalog(monkeypatch) -> None:
    root = "abfss://ws@onelake.dfs.fabric.microsoft.com/lh"
    tables = f"{root}/Tables"
    first = f"{tables}/first"
    second = f"{tables}/second"
    fs = _FakeFS(
        {
            tables: [_Item(first, is_dir=True), _Item(second, is_dir=True)],
            first: [_Item(f"{first}/_delta_log", is_dir=True)],
            second: [_Item(f"{second}/_delta_log", is_dir=True)],
        }
    )
    monkeypatch.setattr("fabric_rlm.lakehouse._get_fs", lambda: fs)
    monkeypatch.setattr(
        "fabric_rlm.lakehouse._read_delta_columns",
        lambda _path: [],
    )

    with pytest.raises(LakehouseDiscoveryError, match="max_sources=1"):
        LakehouseSource(root, max_sources=1).resolve()


def test_auto_discovery_stops_when_source_limit_is_exceeded(monkeypatch) -> None:
    root = "abfss://ws@onelake.dfs.fabric.microsoft.com/lh"
    tables = f"{root}/Tables"
    first = f"{tables}/first"
    second = f"{tables}/second"
    unread = f"{tables}/must-not-be-read"
    fs = _FakeFS(
        {
            tables: [
                _Item(first, is_dir=True),
                _Item(second, is_dir=True),
                _Item(unread, is_dir=True),
            ],
            first: [_Item(f"{first}/_delta_log", is_dir=True)],
            second: [_Item(f"{second}/_delta_log", is_dir=True)],
        }
    )
    monkeypatch.setattr("fabric_rlm.lakehouse._get_fs", lambda: fs)
    monkeypatch.setattr("fabric_rlm.lakehouse._read_delta_columns", lambda _path: [])

    with pytest.raises(LakehouseDiscoveryError, match="max_sources=1"):
        LakehouseSource(root, max_sources=1).resolve()


def test_file_discovery_fails_instead_of_silently_truncating_depth(
    monkeypatch,
) -> None:
    root = "abfss://ws@onelake.dfs.fabric.microsoft.com/lh"
    scope = f"{root}/Files"
    listings = {}
    current = scope
    for depth in range(10):
        child = f"{current}/level{depth}"
        listings[current] = [_Item(child, is_dir=True)]
        current = child
    monkeypatch.setattr(
        "fabric_rlm.lakehouse._get_fs",
        lambda: _FakeFS(listings),
    )

    with pytest.raises(LakehouseDiscoveryError, match="maximum depth"):
        LakehouseSource(f"{root}/Files").resolve()


def test_delta_discovery_fails_instead_of_silently_truncating_depth(
    monkeypatch,
) -> None:
    root = "abfss://ws@onelake.dfs.fabric.microsoft.com/lh"
    scope = f"{root}/Tables"
    listings = {}
    current = scope
    for depth in range(5):
        child = f"{current}/level{depth}"
        listings[current] = [_Item(child, is_dir=True)]
        current = child
    monkeypatch.setattr(
        "fabric_rlm.lakehouse._get_fs",
        lambda: _FakeFS(listings),
    )

    with pytest.raises(LakehouseDiscoveryError, match="maximum depth"):
        LakehouseSource(f"{root}/Tables").resolve()


def test_extensionless_file_under_dotted_folder_keeps_full_name(monkeypatch) -> None:
    root = "abfss://ws@onelake.dfs.fabric.microsoft.com/lh"
    scope = f"{root}/Files/data.v1"
    readme = f"{scope}/README"
    monkeypatch.setattr(
        "fabric_rlm.lakehouse._get_fs",
        lambda: _FakeFS({scope: [_Item(readme, is_dir=False)]}),
    )

    resolved = LakehouseSource(scope).resolve()

    assert resolved.catalog[0]["name"] == "files.data.v1.README"


def test_nested_lakehouse_inputs_resolve_in_parent(monkeypatch) -> None:
    unresolved = LakehouseSource(
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh"
    )
    resolved = LakehouseSource(
        unresolved.root,
        catalog=[{"kind": "delta", "name": "orders", "path": "abfss://orders"}],
    )
    monkeypatch.setattr(LakehouseSource, "resolve", lambda self: resolved)

    inputs = {"primary": unresolved, "others": [unresolved]}
    output = resolve_lakehouse_inputs(inputs)

    assert output == {"primary": resolved, "others": [resolved]}


def test_rlm_resolves_lakehouse_before_worker_binding(monkeypatch) -> None:
    unresolved = LakehouseSource(
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh"
    )
    resolved = LakehouseSource(
        unresolved.root,
        catalog=[
            {
                "kind": "delta",
                "name": "dbo.orders",
                "path": "abfss://orders",
                "columns": [["order_id", "BIGINT"]],
            }
        ],
    )
    calls = []

    def resolve(source):
        calls.append(source)
        return resolved

    monkeypatch.setattr(LakehouseSource, "resolve", resolve)

    class ScriptedLM:
        def __call__(self, *, messages):
            return (
                "```python\n"
                "SUBMIT(answer={"
                "'type': type(source).__name__, "
                "'resolved': source.is_resolved, "
                "'count': len(source.catalog)"
                "})\n```"
            )

    result = RLM.task(
        task="Inspect the Lakehouse source.",
        inputs={"source": unresolved},
        outputs={"answer": dict},
        lm=ScriptedLM(),
        max_turns=1,
        timeout=10,
    ).run()

    assert calls == [unresolved]
    assert result.outputs["answer"] == {
        "type": "LakehouseSource",
        "resolved": True,
        "count": 1,
    }
