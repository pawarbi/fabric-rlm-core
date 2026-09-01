from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from fabric_rlm.knowledge import (
    KnowledgePackage,
    RegisteredOperation,
    Relationship,
    SourceProfile,
)
from fabric_rlm.knowledge_preflight import preflight_knowledge
from fabric_rlm.knowledge_sources import (
    ProfileLimits,
    SourceAdapterRegistry,
    profile_sources,
)
from fabric_rlm.knowledge_store import (
    SourceBinding,
    SourceBindingDescriptor,
    load_knowledge_package,
    save_knowledge_package,
)


def _write_sources(tmp_path: Path) -> dict[str, Path]:
    orders = tmp_path / "orders.csv"
    customers = tmp_path / "customers.csv"
    inventory = tmp_path / "inventory.csv"
    orders.write_text("order_id,customer_id,total\n1,10,5\n", encoding="utf-8")
    customers.write_text("customer_id,region\n10,west\n", encoding="utf-8")
    inventory.write_text("sku,quantity\nA,3\n", encoding="utf-8")
    return {
        "orders": orders,
        "customers": customers,
        "inventory": inventory,
    }


def _learned_package(sources: dict[str, Path]) -> KnowledgePackage:
    profiles = tuple(
        replace(profile, status="active")
        for profile in profile_sources(
            sources,
            roles={
                "orders": "numeric_evidence",
                "customers": "lookup",
                "inventory": "numeric_evidence",
            },
        )
    )
    relationship = Relationship(
        relationship_id="orders_customers",
        left_source="orders",
        right_source="customers",
        key="customer_id",
        cardinality="many_to_one",
        left_coverage=1.0,
        left_key_unique=False,
        right_key_unique=True,
        max_right_rows_per_key=1,
        status="active",
    )
    orders_operation = RegisteredOperation(
        operation_id="orders_total",
        operation="aggregate",
        required_sources=("orders", "customers"),
        required_relationships=("orders_customers",),
        output_schema={"total": "number"},
        grain="all",
        host_implementation_id="aggregate_v1",
        status="active",
    )
    inventory_operation = RegisteredOperation(
        operation_id="inventory_total",
        operation="aggregate",
        required_sources=("inventory",),
        output_schema={"quantity": "number"},
        grain="all",
        host_implementation_id="aggregate_v1",
        status="active",
    )
    return KnowledgePackage(
        package_id="sales_knowledge",
        sources=profiles,
        relationships=(relationship,),
        operations=(orders_operation, inventory_operation),
    )


def test_preflight_requires_exact_aliases_before_profiling(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path)
    package = _learned_package(sources)
    sources["orders"].unlink()

    with pytest.raises(ValueError, match="missing aliases: customers"):
        preflight_knowledge(
            package,
            {"orders": sources["orders"], "inventory": sources["inventory"]},
        )

    with pytest.raises(ValueError, match="extra aliases: extra"):
        preflight_knowledge(package, {**sources, "extra": tmp_path / "missing.csv"})


@pytest.mark.parametrize(
    "roles",
    [
        {"orders": "numeric_evidence", "customers": "lookup"},
        {
            "orders": "numeric_evidence",
            "customers": "lookup",
            "inventory": "numeric_evidence",
            "extra": "lookup",
        },
    ],
)
def test_preflight_rejects_missing_or_extra_role_aliases_before_file_io(
    tmp_path: Path, roles: dict[str, str]
) -> None:
    sources = _write_sources(tmp_path)
    package = _learned_package(sources)
    for path in sources.values():
        path.unlink()

    with pytest.raises(ValueError, match="exact source aliases"):
        preflight_knowledge(package, sources, roles=roles)


def test_preflight_rejects_role_changes_before_file_io(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path)
    package = _learned_package(sources)
    for path in sources.values():
        path.unlink()

    with pytest.raises(ValueError, match="role mismatch"):
        preflight_knowledge(
            package,
            sources,
            roles={
                "orders": "lookup",
                "customers": "lookup",
                "inventory": "numeric_evidence",
            },
        )


def test_preflight_detects_snapshot_drift_for_non_path_handle() -> None:
    @dataclass(frozen=True)
    class FakeHandle:
        version: int

    class FakeAdapter:
        family = "fake"
        allowed_roles = frozenset({"context_only"})
        default_role = "context_only"

        def matches(self, value: object) -> bool:
            return isinstance(value, FakeHandle)

        def profile(self, source_id, value, role, limits):
            return SourceProfile(
                source_id=source_id,
                family=self.family,
                locator=f"fake/{source_id}",
                snapshot_fingerprint=f"snapshot-{value.version}",
                schema_fingerprint="schema-stable",
                diagnostics={"snapshot_exact": True},
                role=role,
            )

    registry = SourceAdapterRegistry((FakeAdapter(),))
    learned = replace(
        profile_sources(
            {"model": FakeHandle(1)},
            registry=registry,
        )[0],
        status="active",
    )
    package = KnowledgePackage(package_id="handles", sources=(learned,))

    result = preflight_knowledge(
        package,
        {"model": FakeHandle(2)},
        registry=registry,
    )

    assert result.drift == {"model": "snapshot"}
    assert result.current_profiles["model"].snapshot_fingerprint == "snapshot-2"
    assert result.package.sources[0].status == "stale"


def test_preflight_rejects_persisted_handle_role_before_adapter_profile() -> None:
    class FakeHandle:
        pass

    class FakeAdapter:
        family = "fake"
        allowed_roles = frozenset({"context_only"})
        default_role = "context_only"
        profile_called = False

        def matches(self, value: object) -> bool:
            return isinstance(value, FakeHandle)

        def profile(self, source_id, value, role, limits):
            self.profile_called = True
            return SourceProfile(
                source_id=source_id,
                family=self.family,
                locator=f"fake/{source_id}",
                snapshot_fingerprint="snapshot",
                schema_fingerprint="schema",
                diagnostics={"snapshot_exact": True},
                role=role,
            )

    adapter = FakeAdapter()
    registry = SourceAdapterRegistry((adapter,))
    package = KnowledgePackage(
        package_id="handles",
        sources=(
            SourceProfile(
                source_id="model",
                family="fake",
                locator="fake/model",
                snapshot_fingerprint="snapshot",
                schema_fingerprint="schema",
                diagnostics={"snapshot_exact": True},
                role="context_only",
                status="active",
            ),
        ),
    )

    with pytest.raises(ValueError, match="role mismatch"):
        preflight_knowledge(
            package,
            {"model": FakeHandle()},
            roles={"model": "numeric_evidence"},
            registry=registry,
        )

    assert adapter.profile_called is False


def test_unchanged_sources_return_current_without_mutating_package(
    tmp_path: Path,
) -> None:
    sources = _write_sources(tmp_path)
    package = _learned_package(sources)

    first = preflight_knowledge(package, sources)
    second = preflight_knowledge(package, sources)

    assert first == second
    assert first.is_current is True
    assert first.package == package
    assert first.drift == {}
    with pytest.raises(TypeError):
        first.current_profiles["orders"] = package.sources[0]


def test_unchanged_large_inexact_snapshot_never_returns_current(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large.csv"
    path.write_text("id,note\n" + "1,x\n" * 500, encoding="utf-8")
    limits = ProfileLimits(max_input_bytes=64, read_chunk_bytes=8)
    learned = replace(
        profile_sources(
            {"large": path},
            roles={"large": "numeric_evidence"},
            limits=limits,
        )[0],
        status="active",
    )
    relationship = Relationship(
        relationship_id="large_self",
        left_source="large",
        right_source="large",
        key="id",
        cardinality="one_to_one",
        left_coverage=1.0,
        left_key_unique=True,
        right_key_unique=True,
        max_right_rows_per_key=1,
        status="active",
    )
    operation = RegisteredOperation(
        operation_id="large_total",
        operation="aggregate",
        required_sources=("large",),
        required_relationships=("large_self",),
        output_schema={"count": "integer"},
        grain="all",
        host_implementation_id="aggregate_v1",
        status="active",
    )
    package = KnowledgePackage(
        package_id="large",
        sources=(learned,),
        relationships=(relationship,),
        operations=(operation,),
    )

    result = preflight_knowledge(package, {"large": path}, limits=limits)

    assert result.is_current is False
    assert result.drift == {"large": "inexact"}
    assert result.package.sources[0].status == "stale"
    assert result.package.relationships[0].status == "stale"
    assert result.package.relationships[0].reason_code == "source_snapshot_inexact"
    assert result.package.operations[0].status == "stale"
    assert result.package.operations[0].reason_code == "source_snapshot_inexact"
    assert result.package.events[0].reason_code == "source_snapshot_inexact"


def test_same_size_middle_only_large_file_mutation_is_inexact_not_current(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large.bin"
    original = b"A" * 64 + b"B" * 128 + b"C" * 64
    path.write_bytes(original)
    limits = ProfileLimits(max_input_bytes=64, read_chunk_bytes=8)
    learned = replace(
        profile_sources(
            {"large": path},
            roles={"large": "context_only"},
            limits=limits,
        )[0],
        status="active",
    )
    package = KnowledgePackage(package_id="large", sources=(learned,))
    path.write_bytes(b"A" * 64 + b"D" * 128 + b"C" * 64)

    result = preflight_knowledge(package, {"large": path}, limits=limits)

    assert result.current_profiles["large"].snapshot_fingerprint == (
        learned.snapshot_fingerprint
    )
    assert result.drift == {"large": "inexact"}
    assert result.is_current is False


def test_snapshot_drift_marks_only_dependent_knowledge_stale(
    tmp_path: Path,
) -> None:
    sources = _write_sources(tmp_path)
    package = _learned_package(sources)
    learned_orders = next(
        source for source in package.sources if source.source_id == "orders"
    )
    sources["orders"].write_text(
        "order_id,customer_id,total\n1,10,9\n",
        encoding="utf-8",
    )

    result = preflight_knowledge(package, sources)
    updated_sources = {source.source_id: source for source in result.package.sources}
    updated_operations = {
        operation.operation_id: operation
        for operation in result.package.operations
    }

    assert result.is_current is False
    assert result.drift == {"orders": "snapshot"}
    assert updated_sources["orders"].status == "stale"
    assert updated_sources["orders"].snapshot_fingerprint == (
        learned_orders.snapshot_fingerprint
    )
    assert result.current_profiles["orders"].snapshot_fingerprint != (
        learned_orders.snapshot_fingerprint
    )
    assert result.package.relationships[0].status == "stale"
    assert result.package.relationships[0].reason_code == "source_snapshot_drift"
    assert updated_operations["orders_total"].status == "stale"
    assert updated_operations["orders_total"].reason_code == (
        "source_snapshot_drift"
    )
    assert updated_operations["inventory_total"].status == "active"


def test_schema_drift_is_distinguished_and_events_are_deduplicated(
    tmp_path: Path,
) -> None:
    sources = _write_sources(tmp_path)
    package = _learned_package(sources)
    sources["customers"].write_text(
        "customer_id,territory\n10,west\n",
        encoding="utf-8",
    )

    first = preflight_knowledge(package, sources)
    second = preflight_knowledge(first.package, sources)
    operations = {
        operation.operation_id: operation
        for operation in first.package.operations
    }

    assert first.drift == {"customers": "schema"}
    assert first.package.relationships[0].reason_code == "source_schema_drift"
    assert operations["orders_total"].reason_code == "source_schema_drift"
    assert [event.event_id for event in first.package.events] == [
        event.event_id for event in second.package.events
    ]
    assert [event.subject_type for event in first.package.events] == [
        "source",
        "relationship",
        "operation",
    ]


def test_terminal_lifecycle_and_reason_are_preserved_while_drift_propagates(
    tmp_path: Path,
) -> None:
    sources = _write_sources(tmp_path)
    package = _learned_package(sources)
    relationship = replace(
        package.relationships[0],
        status="quarantined",
        reason_code="quality_hold",
    )
    orders_operation = package.operations[0]
    retired_operation = replace(
        orders_operation,
        operation_id="retired_orders_total",
        status="retired",
        reason_code="superseded",
    )
    package = replace(
        package,
        relationships=(relationship,),
        operations=(orders_operation, retired_operation, package.operations[1]),
    )
    sources["customers"].write_text(
        "customer_id,territory\n10,west\n",
        encoding="utf-8",
    )

    result = preflight_knowledge(package, sources)
    operations = {item.operation_id: item for item in result.package.operations}

    assert result.package.relationships[0].status == "quarantined"
    assert result.package.relationships[0].reason_code == "quality_hold"
    assert operations["orders_total"].status == "stale"
    assert operations["orders_total"].reason_code == "source_schema_drift"
    assert operations["retired_orders_total"].status == "retired"
    assert operations["retired_orders_total"].reason_code == "superseded"


@pytest.mark.parametrize("terminal_status", ["quarantined", "retired"])
@pytest.mark.parametrize("subject_type", ["source", "relationship", "operation"])
def test_terminal_subjects_preserve_reason_without_new_stale_event(
    tmp_path: Path,
    terminal_status: str,
    subject_type: str,
) -> None:
    sources = _write_sources(tmp_path)
    package = _learned_package(sources)
    terminal_reason = "terminal_hold"
    terminal_id = ""

    if subject_type == "source":
        terminal_id = "customers"
        package = replace(
            package,
            sources=tuple(
                replace(
                    source,
                    status=terminal_status,
                )
                if source.source_id == terminal_id
                else source
                for source in package.sources
            ),
        )
    elif subject_type == "relationship":
        terminal_id = package.relationships[0].relationship_id
        package = replace(
            package,
            relationships=(
                replace(
                    package.relationships[0],
                    status=terminal_status,
                    reason_code=terminal_reason,
                ),
            ),
        )
    else:
        terminal_id = "orders_total"
        package = replace(
            package,
            operations=tuple(
                replace(
                    operation,
                    status=terminal_status,
                    reason_code=terminal_reason,
                )
                if operation.operation_id == terminal_id
                else operation
                for operation in package.operations
            ),
        )

    sources["customers"].write_text(
        "customer_id,territory\n10,west\n",
        encoding="utf-8",
    )
    result = preflight_knowledge(package, sources)
    records = {
        "source": {item.source_id: item for item in result.package.sources},
        "relationship": {
            item.relationship_id: item
            for item in result.package.relationships
        },
        "operation": {
            item.operation_id: item
            for item in result.package.operations
        },
    }
    terminal = records[subject_type][terminal_id]

    assert terminal.status == terminal_status
    if subject_type != "source":
        assert terminal.reason_code == terminal_reason
    assert not any(
        event.subject_type == subject_type and event.subject_id == terminal_id
        for event in result.package.events
    )


def test_preflight_propagates_source_mutation_failure(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path)
    package = _learned_package(sources)
    import fabric_rlm.knowledge_sources as knowledge_sources

    class MutatingCsvAdapter(knowledge_sources._LocalFileAdapter):
        family = "csv"
        allowed_roles = frozenset({"numeric_evidence", "lookup"})
        default_role = "numeric_evidence"

        def _matches_path(self, path: Path) -> bool:
            return path.suffix.lower() == ".csv"

        def _profile_file(self, source_id, path, role, limits, snapshot):
            path.write_text(path.read_text(encoding="utf-8") + "2,20,8\n")
            return next(
                profile
                for profile in package.sources
                if profile.source_id == source_id
            )

    registry = SourceAdapterRegistry((MutatingCsvAdapter(),))

    with pytest.raises(ValueError, match="orders.*changed.*profil"):
        preflight_knowledge(package, sources, registry=registry)


def test_preflight_stale_package_round_trips_lifecycle_and_events(
    tmp_path: Path,
) -> None:
    sources = _write_sources(tmp_path)
    package = _learned_package(sources)
    sources["orders"].write_text(
        "order_id,customer_id,total\n1,10,9\n",
        encoding="utf-8",
    )
    result = preflight_knowledge(package, sources)
    destination = tmp_path / "stale-knowledge.json"

    save_knowledge_package(destination, result.package)
    bindings = {
        source.source_id: SourceBinding(
            SourceBindingDescriptor(source.source_id, source.locator),
            sources[source.source_id],
        )
        for source in result.package.sources
    }
    loaded = load_knowledge_package(destination, bindings=bindings).package

    assert loaded == result.package
    assert loaded.sources[0].status in {"active", "stale"}
    assert any(source.status == "stale" for source in loaded.sources)
    assert loaded.relationships[0].reason_code == "source_snapshot_drift"
    assert loaded.events == result.package.events


@pytest.mark.parametrize("replacement", ["missing", "directory"])
def test_missing_or_non_regular_current_source_fails_closed(
    tmp_path: Path,
    replacement: str,
) -> None:
    sources = _write_sources(tmp_path)
    package = _learned_package(sources)
    sources["orders"].unlink()
    if replacement == "directory":
        sources["orders"].mkdir()

    with pytest.raises((FileNotFoundError, ValueError)):
        preflight_knowledge(package, sources)
