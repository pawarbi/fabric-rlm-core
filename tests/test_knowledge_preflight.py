from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from fabric_rlm.knowledge import (
    KnowledgePackage,
    RegisteredOperation,
    Relationship,
)
from fabric_rlm.knowledge_preflight import preflight_knowledge
from fabric_rlm.knowledge_sources import profile_sources


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
