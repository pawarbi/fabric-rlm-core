from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from fabric_rlm.knowledge import (
    KnowledgeEvent,
    KnowledgePackage,
    RegisteredOperation,
    Relationship,
    SourceProfile,
    canonical_json,
)


def _source(
    source_id: str = "sales.orders",
    *,
    locator: str = "lakehouse/sales/orders",
    schema: dict[str, object] | None = None,
) -> SourceProfile:
    return SourceProfile(
        source_id=source_id,
        family="delta_table",
        locator=locator,
        snapshot_fingerprint=f"snapshot-{source_id}",
        schema_fingerprint=f"schema-{source_id}",
        schema=schema or {"order_id": {"type": "integer", "nullable": False}},
    )


def _relationship() -> Relationship:
    return Relationship(
        relationship_id="orders.customers",
        left_source="sales.orders",
        right_source="sales.customers",
        key="customer_id",
        cardinality="many_to_one",
        left_coverage=0.99,
        left_key_unique=False,
        right_key_unique=True,
        max_right_rows_per_key=1,
        status="active",
    )


def _operation() -> RegisteredOperation:
    return RegisteredOperation(
        operation_id="sales.total_by_region",
        operation="aggregate",
        required_sources=("sales.orders", "sales.customers"),
        required_relationships=("orders.customers",),
        parameter_schema={
            "region": {"type": "string", "enum": ["east", "west"]},
            "include_tax": {"type": "boolean"},
        },
        parameter_defaults={"region": "east", "include_tax": False},
        output_schema={"region": "string", "total": "number"},
        max_output_rows=100,
        max_output_columns=2,
        grain="region",
        sql_template="SELECT region, SUM(total) AS total FROM orders GROUP BY region",
        status="active",
    )


def _package(*, events: tuple[KnowledgeEvent, ...] = ()) -> KnowledgePackage:
    return KnowledgePackage(
        package_id="sales.knowledge.v1",
        sources=(_source(), _source("sales.customers")),
        relationships=(_relationship(),),
        operations=(_operation(),),
        events=events,
    )


def test_records_are_deeply_immutable_and_serialization_is_detached() -> None:
    schema = {"order_id": {"type": "integer"}, "labels": ["new", "repeat"]}
    source = _source(schema=schema)
    schema["order_id"]["type"] = "string"
    schema["labels"].append("changed")

    assert source.schema["order_id"]["type"] == "integer"
    assert source.schema["labels"] == ("new", "repeat")
    with pytest.raises(TypeError):
        source.schema["order_id"]["type"] = "string"
    with pytest.raises(FrozenInstanceError):
        source.locator = "other"

    payload = source.to_dict()
    payload["schema"]["order_id"]["type"] = "string"
    assert source.schema["order_id"]["type"] == "integer"


@pytest.mark.parametrize(
    "identifier",
    ["", " ", "../orders", "sales/orders", r"sales\orders", ".hidden", "orders?x"],
)
def test_records_reject_unsafe_logical_identifiers(identifier: str) -> None:
    with pytest.raises(ValueError, match="source_id"):
        _source(identifier)


@pytest.mark.parametrize(
    "locator",
    [
        r"C:\data\orders",
        r"\\server\share\orders",
        "/var/data/orders",
        "../orders",
        "lakehouse/../orders",
        "lakehouse/%2e%2e/orders",
        "lakehouse/%5c%5cserver/share",
        "lakehouse/sales orders",
        "lakehouse/%252e%252e/orders",
        "file:///var/data/orders",
        "https://user:password@example.test/orders",
        "https://example.test/orders?token=secret",
        "https://example.test/orders#credentials",
    ],
)
def test_source_profile_rejects_unsafe_persisted_locators(locator: str) -> None:
    with pytest.raises(ValueError, match="locator"):
        _source(locator=locator)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "fact"),
        ("source_status", "approved"),
        ("status", "approved"),
        ("cardinality", "sometimes"),
        ("left_coverage", math.nan),
        ("left_coverage", 1.01),
        ("max_right_rows_per_key", -1),
    ],
)
def test_source_and_relationship_enums_and_bounds_are_validated(
    field: str,
    value: object,
) -> None:
    if field == "role":
        with pytest.raises(ValueError, match="role"):
            _source().__class__(
                **{**_source().to_dict(), "role": value},
            )
        return
    if field == "source_status":
        with pytest.raises(ValueError, match="status"):
            _source().__class__(
                **{**_source().to_dict(), "status": value},
            )
        return

    relationship = _relationship().to_dict()
    relationship[field] = value
    with pytest.raises(ValueError, match=field.replace("_", ".*")):
        Relationship(**relationship)


@pytest.mark.parametrize(
    ("parameter_schema", "defaults", "match"),
    [
        ({"filters": {"type": "object"}}, {}, "type"),
        ({"limit": {"type": "integer"}}, {"limit": True}, "limit"),
        (
            {"ratio": {"type": "number"}},
            {"ratio": float("inf")},
            "finite",
        ),
        (
            {"region": {"type": "string", "enum": []}},
            {},
            "enum",
        ),
        (
            {"region": {"type": "string", "enum": ["east", {"bad": "value"}]}},
            {},
            "scalar",
        ),
        (
            {"region": {"type": "string", "enum": ["east", "west"]}},
            {"region": "north"},
            "enum",
        ),
        ({"region": {"type": "string"}}, {"unknown": "east"}, "unknown"),
    ],
)
def test_registered_operation_rejects_unsafe_parameter_contracts(
    parameter_schema: dict[str, object],
    defaults: dict[str, object],
    match: str,
) -> None:
    operation = _operation().to_dict()
    operation["parameter_schema"] = parameter_schema
    operation["parameter_defaults"] = defaults

    with pytest.raises(ValueError, match=match):
        RegisteredOperation(**operation)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_schema", {}),
        ("output_schema", {"total": "decimal128"}),
        ("max_output_rows", 0),
        ("max_output_rows", True),
        ("max_output_columns", -1),
    ],
)
def test_registered_operation_validates_output_schema_and_bounds(
    field: str,
    value: object,
) -> None:
    operation = _operation().to_dict()
    operation[field] = value

    with pytest.raises(ValueError, match="output|max_output"):
        RegisteredOperation(**operation)


def test_package_rejects_duplicate_ids_and_unknown_graph_references() -> None:
    with pytest.raises(ValueError, match="duplicate source_id"):
        KnowledgePackage(
            package_id="duplicate.sources",
            sources=(_source(), _source()),
        )

    with pytest.raises(ValueError, match="unknown source"):
        KnowledgePackage(
            package_id="unknown.relationship.source",
            sources=(_source(),),
            relationships=(_relationship(),),
        )

    with pytest.raises(ValueError, match="unknown relationship"):
        KnowledgePackage(
            package_id="unknown.operation.relationship",
            sources=(_source(), _source("sales.customers")),
            operations=(_operation(),),
        )


def test_package_validates_event_subject_references() -> None:
    with pytest.raises(ValueError, match="unknown operation"):
        KnowledgePackage(
            package_id="unknown.event.subject",
            sources=(_source(),),
            events=(
                KnowledgeEvent(
                    event_id="event-1",
                    event_type="operation.activated",
                    subject_type="operation",
                    subject_id="missing.operation",
                    status="active",
                ),
            ),
        )


def test_canonical_fingerprints_ignore_mapping_and_graph_record_order() -> None:
    first = _package()
    reordered = KnowledgePackage(
        package_id=first.package_id,
        sources=tuple(reversed(first.sources)),
        relationships=first.relationships,
        operations=(
            RegisteredOperation(
                **{
                    **first.operations[0].to_dict(),
                    "parameter_schema": {
                        "include_tax": {"type": "boolean"},
                        "region": {"enum": ["east", "west"], "type": "string"},
                    },
                    "parameter_defaults": {
                        "include_tax": False,
                        "region": "east",
                    },
                    "output_schema": {"total": "number", "region": "string"},
                }
            ),
        ),
    )

    assert canonical_json(first.to_dict()) == canonical_json(reordered.to_dict())
    assert first.snapshot_fingerprint == reordered.snapshot_fingerprint
    assert first.schema_fingerprint == reordered.schema_fingerprint
    assert first.fingerprint == reordered.fingerprint


def test_package_fingerprint_preserves_event_sequence() -> None:
    first_event = KnowledgeEvent(
        event_id="event-1",
        event_type="source.activated",
        subject_type="source",
        subject_id="sales.orders",
        status="active",
    )
    second_event = KnowledgeEvent(
        event_id="event-2",
        event_type="operation.activated",
        subject_type="operation",
        subject_id="sales.total_by_region",
        status="active",
    )

    assert _package(events=(first_event, second_event)).fingerprint != _package(
        events=(second_event, first_event)
    ).fingerprint


def test_package_round_trips_and_rejects_unknown_fields() -> None:
    package = _package(
        events=(
            KnowledgeEvent(
                event_id="event-1",
                event_type="relationship.activated",
                subject_type="relationship",
                subject_id="orders.customers",
                status="active",
            ),
        )
    )

    assert KnowledgePackage.from_dict(package.to_dict()) == package

    payload = package.to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown field"):
        KnowledgePackage.from_dict(payload)

    source_payload = package.sources[0].to_dict()
    source_payload["credential"] = "secret"
    with pytest.raises(ValueError, match="unknown field"):
        SourceProfile.from_dict(source_payload)

    with pytest.raises(ValueError, match="string keys"):
        SourceProfile.from_dict({1: "not-a-field"})
