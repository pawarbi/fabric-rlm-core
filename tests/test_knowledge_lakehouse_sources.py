from __future__ import annotations

import builtins
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from fabric_rlm.artifacts import File
from fabric_rlm.knowledge import KnowledgePackage, canonical_json
from fabric_rlm.knowledge_preflight import preflight_knowledge
from fabric_rlm.knowledge_sources import ProfileLimits, profile_sources
from fabric_rlm.knowledge_store import save_knowledge_package
from fabric_rlm.lakehouse import LakehouseSource


def _module():
    import fabric_rlm.knowledge_lakehouse_sources as module

    return module


def _delta_directory(tmp_path: Path, marker: str = "00000000000000000001.json") -> Path:
    path = tmp_path / "orders"
    log = path / "_delta_log"
    log.mkdir(parents=True)
    (log / marker).write_text("", encoding="utf-8")
    return path


class _Schema:
    def __init__(self, fields: list[dict[str, object]] | None = None) -> None:
        self._fields = fields or [
            {
                "name": "order_id",
                "type": "long",
                "nullable": False,
                "metadata": {},
            },
            {
                "name": "amount",
                "type": "double",
                "nullable": True,
                "metadata": {},
            },
        ]

    def to_json(self) -> str:
        return json.dumps({"type": "struct", "fields": self._fields})


def _install_delta(
    monkeypatch: pytest.MonkeyPatch,
    states: list[tuple[int, str, _Schema]],
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    class DeltaTable:
        def __init__(self, path, **kwargs):
            calls.append({"path": path, **kwargs})
            self._version, self._identity, self._schema = states.pop(0)

        def version(self):
            return self._version

        def metadata(self):
            return SimpleNamespace(
                id=self._identity,
                name="orders",
                partition_columns=["region"],
            )

        def schema(self):
            return self._schema

        def files(self):
            raise AssertionError("Delta profiling must not enumerate files")

        def file_uris(self):
            raise AssertionError("Delta profiling must not enumerate files")

        def to_pyarrow_table(self):
            raise AssertionError("Delta profiling must not materialize rows")

    monkeypatch.setitem(sys.modules, "deltalake", SimpleNamespace(DeltaTable=DeltaTable))
    return calls


@pytest.mark.parametrize(
    "marker",
    [
        "_last_checkpoint",
        "1.json",
        "00000000000000000001.json",
        "1.checkpoint.parquet",
        "00000000000000000001.checkpoint.parquet",
    ],
)
@pytest.mark.parametrize("wrapper", [lambda path: path, str, File])
def test_delta_adapter_matches_only_recognized_direct_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
    wrapper,
) -> None:
    path = _delta_directory(tmp_path, marker)
    calls = _install_delta(
        monkeypatch,
        [(7, "raw-table-uuid", _Schema()), (7, "raw-table-uuid", _Schema())],
    )

    profile = profile_sources(
        {"sales": wrapper(path)},
        registry=_module().fabric_source_registry(),
    )[0]

    assert profile.family == "delta"
    assert profile.diagnostics["snapshot_exact"] is True
    assert profile.diagnostics["committed_version"] == 7
    assert all(call["without_files"] is True for call in calls)
    assert "raw-table-uuid" not in canonical_json(profile.to_dict())
    assert str(path) not in canonical_json(profile.to_dict())
    assert profile.locator.startswith("delta/v1/")


@pytest.mark.parametrize(
    "marker",
    ["not-delta.txt", "checkpoint.parquet", "1.parquet", "abc.json"],
)
def test_delta_adapter_fails_closed_for_arbitrary_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
) -> None:
    path = _delta_directory(tmp_path, marker)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "deltalake":
            raise AssertionError("unsupported directories must not import deltalake")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ValueError, match="regular file"):
        profile_sources(
            {"sales": path},
            registry=_module().fabric_source_registry(),
        )


def test_delta_optional_dependency_is_lazy_and_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _delta_directory(tmp_path)
    module = _module()
    real_import = builtins.__import__

    def missing_delta(name, *args, **kwargs):
        if name == "deltalake":
            raise ModuleNotFoundError("No module named 'deltalake'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_delta)

    with pytest.raises(ValueError, match=r"fabric-rlm\[analytics\].*deltalake"):
        profile_sources({"sales": path}, registry=module.fabric_source_registry())


def test_delta_profile_uses_structural_schema_and_exact_table_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _delta_directory(tmp_path)
    _install_delta(
        monkeypatch,
        [(11, "table-a", _Schema()), (11, "table-a", _Schema())],
    )

    profile = profile_sources(
        {"sales": path},
        registry=_module().fabric_source_registry(),
    )[0]

    assert profile.schema == {
        "amount": {
            "delta_type": "double",
            "nullable": True,
            "type": "number",
        },
        "order_id": {
            "delta_type": "long",
            "nullable": False,
            "type": "integer",
        },
    }
    assert profile.diagnostics["partition_column_count"] == 1
    assert profile.diagnostics["table_identity_fingerprint"]
    assert "orders" not in canonical_json(profile.to_dict())


def test_delta_profile_can_be_persisted_as_knowledge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _delta_directory(tmp_path)
    _install_delta(
        monkeypatch,
        [(11, "table-a", _Schema()), (11, "table-a", _Schema())],
    )
    profile = profile_sources(
        {"sales": path},
        registry=_module().fabric_source_registry(),
    )[0]

    save_knowledge_package(
        tmp_path / "knowledge.json",
        KnowledgePackage(package_id="delta.persistence.v1", sources=(profile,)),
    )

    assert (tmp_path / "knowledge.json").is_file()


def test_delta_locator_changes_with_stable_table_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = _delta_directory(tmp_path / "first")
    second_path = _delta_directory(tmp_path / "second")
    _install_delta(
        monkeypatch,
        [
            (1, "table-a", _Schema()),
            (1, "table-a", _Schema()),
            (1, "table-b", _Schema()),
            (1, "table-b", _Schema()),
        ],
    )

    first = profile_sources(
        {"sales": first_path},
        registry=_module().fabric_source_registry(),
    )[0]
    second = profile_sources(
        {"sales": second_path},
        registry=_module().fabric_source_registry(),
    )[0]

    assert first.locator != second.locator


def test_delta_profile_rejects_schema_over_profile_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _delta_directory(tmp_path)
    _install_delta(
        monkeypatch,
        [(1, "table-a", _Schema()), (1, "table-a", _Schema())],
    )

    with pytest.raises(ValueError, match="max_fields"):
        profile_sources(
            {"sales": path},
            limits=ProfileLimits(max_fields=1),
            registry=_module().fabric_source_registry(),
        )


def test_delta_profile_rejects_schema_over_nesting_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _delta_directory(tmp_path)
    nested = _Schema(
        [
            {
                "name": "outer",
                "type": {
                    "type": "struct",
                    "fields": [
                        {
                            "name": "inner",
                            "type": {
                                "type": "struct",
                                "fields": [
                                    {
                                        "name": "value",
                                        "type": "long",
                                        "nullable": False,
                                        "metadata": {},
                                    }
                                ],
                            },
                            "nullable": True,
                            "metadata": {},
                        }
                    ],
                },
                "nullable": True,
                "metadata": {},
            }
        ]
    )
    _install_delta(
        monkeypatch,
        [(1, "table-a", nested), (1, "table-a", nested)],
    )

    with pytest.raises(ValueError, match="max_nesting_depth"):
        profile_sources(
            {"sales": path},
            limits=ProfileLimits(max_nesting_depth=2),
            registry=_module().fabric_source_registry(),
        )


@pytest.mark.parametrize(
    "states",
    [
        [(3, "table-a", _Schema()), (4, "table-a", _Schema())],
        [(3, "table-a", _Schema()), (3, "table-b", _Schema())],
    ],
)
def test_delta_profile_rejects_mutation_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    states: list[tuple[int, str, _Schema]],
) -> None:
    path = _delta_directory(tmp_path)
    _install_delta(monkeypatch, states)

    with pytest.raises(ValueError, match="changed during profiling"):
        profile_sources(
            {"sales": path},
            registry=_module().fabric_source_registry(),
        )


def _lakehouse(
    *,
    root: str = (
        "abfss://workspace-secret@onelake.dfs.fabric.microsoft.com/"
        "lakehouse-secret.Lakehouse"
    ),
    catalog: list[dict[str, object]] | None = None,
) -> LakehouseSource:
    return LakehouseSource(root, catalog=catalog)


def test_lakehouse_adapter_resolves_metadata_without_querying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _lakehouse(catalog=None)
    resolved = _lakehouse(
        catalog=[
            {
                "kind": "delta",
                "name": "dbo.orders",
                "path": "abfss://private/Tables/dbo/orders",
                "columns": [["amount", "Float64"], ["order_id", "Int64"]],
                "version": 8,
                "table_id": "raw-table-uuid",
                "credential": "private-credential",
            }
        ]
    )
    resolve_calls: list[LakehouseSource] = []

    def resolve(self):
        resolve_calls.append(self)
        return resolved

    def query(*args, **kwargs):
        raise AssertionError("knowledge profiling must never call LakehouseSource.query")

    monkeypatch.setattr(LakehouseSource, "resolve", resolve)
    monkeypatch.setattr(LakehouseSource, "query", query)

    profile = profile_sources(
        {"sales": source},
        registry=_module().fabric_source_registry(),
    )[0]

    assert resolve_calls == [source]
    assert profile.family == "lakehouse"
    assert profile.diagnostics["snapshot_exact"] is True
    assert profile.schema == {
        "dbo.orders": {
            "columns": {
                "amount": {"lakehouse_type": "Float64", "type": "number"},
                "order_id": {"lakehouse_type": "Int64", "type": "integer"},
            },
            "kind": "delta",
        }
    }
    encoded = canonical_json(profile.to_dict())
    for secret in (
        source.root,
        "workspace-secret",
        "lakehouse-secret",
        "abfss://private/Tables/dbo/orders",
        "raw-table-uuid",
        "private-credential",
    ):
        assert secret not in encoded
    assert profile.locator.startswith("lakehouse/v1/")


def test_lakehouse_files_make_snapshot_inexact() -> None:
    source = _lakehouse(
        catalog=[
            {
                "kind": "csv",
                "name": "files.targets",
                "path": "abfss://private/Files/targets.csv",
                "columns": [["region", "VARCHAR"]],
            }
        ]
    )

    profile = profile_sources(
        {"sales": source},
        registry=_module().fabric_source_registry(),
    )[0]

    assert profile.diagnostics["snapshot_exact"] is False
    assert profile.diagnostics["inexact_entry_count"] == 1


def test_lakehouse_template_role_is_rejected_before_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _lakehouse(catalog=None)

    def forbidden_resolve(self):
        raise AssertionError("role validation must happen before Lakehouse I/O")

    monkeypatch.setattr(LakehouseSource, "resolve", forbidden_resolve)

    with pytest.raises(ValueError, match="template.*lakehouse"):
        profile_sources(
            {"sales": source},
            roles={"sales": "template"},
            registry=_module().fabric_source_registry(),
        )


def test_lakehouse_catalog_is_sorted_bounded_and_value_free() -> None:
    source = _lakehouse(
        catalog=[
            {
                "kind": "delta",
                "name": "z_orders",
                "path": "abfss://private/z",
                "columns": [["z", "STRING"]],
                "version": 1,
                "table_id": "z-id",
            },
            {
                "kind": "delta",
                "name": "a_orders",
                "path": "abfss://private/a",
                "columns": [["a", "LONG"]],
                "version": 1,
                "table_id": "a-id",
            },
        ]
    )
    registry = _module().fabric_source_registry()

    profile = profile_sources({"sales": source}, registry=registry)[0]

    assert list(profile.schema) == ["a_orders", "z_orders"]
    assert profile.diagnostics["catalog_entry_count"] == 2
    assert "private" not in canonical_json(profile.to_dict())
    with pytest.raises(ValueError, match="max_records"):
        profile_sources(
            {"sales": source},
            limits=ProfileLimits(max_records=1),
            registry=registry,
        )
    with pytest.raises(ValueError, match="max_fields"):
        profile_sources(
            {"sales": source},
            limits=ProfileLimits(max_fields=1),
            registry=registry,
        )


def test_lakehouse_catalog_drift_flows_through_preflight_registry_injection() -> None:
    registry = _module().fabric_source_registry()
    learned_source = _lakehouse(
        catalog=[
            {
                "kind": "delta",
                "name": "dbo.orders",
                "path": "abfss://private/orders",
                "columns": [["id", "BIGINT"]],
                "version": 1,
                "table_id": "table-a",
            }
        ]
    )
    current_source = _lakehouse(
        catalog=[
            {
                "kind": "delta",
                "name": "dbo.orders",
                "path": "abfss://private/orders",
                "columns": [["id", "BIGINT"]],
                "version": 2,
                "table_id": "table-a",
            }
        ]
    )
    learned = replace(
        profile_sources({"sales": learned_source}, registry=registry)[0],
        status="active",
    )
    package = KnowledgePackage(package_id="sales", sources=(learned,))

    result = preflight_knowledge(
        package,
        {"sales": current_source},
        registry=registry,
    )

    assert result.drift == {"sales": "snapshot"}
    assert result.package.sources[0].status == "stale"


def test_fabric_source_registry_is_additive_without_changing_default_registry(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text("id\n1\n", encoding="utf-8")

    profile = profile_sources(
        {"sales": csv_path},
        registry=_module().fabric_source_registry(),
    )[0]

    assert profile.family == "csv"
