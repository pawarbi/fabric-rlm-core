from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Callable

import pytest

from fabric_rlm.knowledge import (
    KnowledgeEvent,
    KnowledgePackage,
    RegisteredOperation,
    Relationship,
    SourceProfile,
    canonical_json,
)
from fabric_rlm.knowledge_store import (
    MAX_PACKAGE_BYTES,
    KnowledgePersistenceError,
    PersistenceIntegrityError,
    SourceBinding,
    SourceBindingDescriptor,
    load_knowledge_package,
    save_knowledge_package,
)


def _package(
    *,
    diagnostics: dict[str, object] | None = None,
) -> KnowledgePackage:
    return KnowledgePackage(
        package_id="sales.knowledge.v1",
        sources=(
            SourceProfile(
                source_id="sales.orders",
                family="delta_table",
                locator="lakehouse/sales/orders",
                snapshot_fingerprint="snapshot-orders",
                schema_fingerprint="schema-orders",
                schema={"order_id": {"type": "integer", "nullable": False}},
                diagnostics=diagnostics or {},
            ),
        ),
    )


def _full_package() -> KnowledgePackage:
    return KnowledgePackage(
        package_id="sales.knowledge.v1",
        sources=(
            _package(
                diagnostics={
                    "state_code": "ready",
                    "response_time": 12,
                    "reasoning_score": 0.9,
                }
            ).sources[0],
            SourceProfile(
                source_id="sales.customers",
                family="delta_table",
                locator="lakehouse/sales/customers",
                snapshot_fingerprint="snapshot-customers",
                schema_fingerprint="schema-customers",
                schema={
                    "content": {"type": "string", "nullable": True},
                    "password": {"type": "string", "nullable": True},
                },
            ),
            SourceProfile(
                source_id="sales.regions",
                family="delta_table",
                locator="lakehouse/sales/regions",
                snapshot_fingerprint="snapshot-regions",
                schema_fingerprint="schema-regions",
                schema={"state": {"type": "string", "nullable": False}},
            ),
        ),
        relationships=(
            Relationship(
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
            ),
            Relationship(
                relationship_id="customers.regions",
                left_source="sales.customers",
                right_source="sales.regions",
                key="region_id",
                cardinality="many_to_one",
                left_coverage=1.0,
                left_key_unique=False,
                right_key_unique=True,
                max_right_rows_per_key=1,
                status="active",
            ),
        ),
        operations=(
            RegisteredOperation(
                operation_id="sales.total_by_region",
                operation="aggregate",
                required_sources=("sales.customers", "sales.orders"),
                required_relationships=("orders.customers",),
                parameter_schema={
                    "state_code": {"type": "string", "enum": ["ready", "held"]}
                },
                parameter_defaults={"state_code": "ready"},
                output_schema={"content": "string", "total": "number"},
                max_output_rows=100,
                max_output_columns=2,
                grain="region",
                host_implementation_id="registry.sales.total_by_region",
                operation_version="1",
                status="active",
            ),
            RegisteredOperation(
                operation_id="sales.customer_count",
                operation="aggregate",
                required_sources=("sales.regions", "sales.customers"),
                required_relationships=("customers.regions",),
                output_schema={"state": "string", "count": "integer"},
                max_output_rows=50,
                max_output_columns=2,
                grain="state",
                host_implementation_id="registry.sales.customer_count",
                status="active",
            ),
        ),
        events=(
            KnowledgeEvent(
                event_id="event-1",
                event_type="operation.activated",
                subject_type="operation",
                subject_id="sales.total_by_region",
                status="active",
                reason_code="validated",
            ),
        ),
    )


def _full_bindings() -> dict[str, SourceBinding]:
    return {
        source.source_id: SourceBinding(
            SourceBindingDescriptor(source.source_id, source.locator),
            object(),
        )
        for source in _full_package().sources
    }


def _bindings(value: object | None = None) -> dict[str, SourceBinding]:
    return {
        "sales.orders": SourceBinding(
            descriptor=SourceBindingDescriptor(
                source_id="sales.orders",
                locator="lakehouse/sales/orders",
            ),
            value=object() if value is None else value,
        )
    }


def _write_envelope(path: Path, envelope: dict[str, object]) -> None:
    path.write_text(canonical_json(envelope) + "\n", encoding="utf-8")


def test_save_is_canonical_utf8_and_load_binds_runtime_values(tmp_path: Path) -> None:
    package = _package()
    runtime_value = object()
    destination = tmp_path / "knowledge.json"

    save_knowledge_package(destination, package)

    expected = canonical_json(
        {
            "format_version": 1,
            "package": package.to_dict(),
            "package_fingerprint": package.fingerprint,
        }
    ).encode("utf-8") + b"\n"
    assert destination.read_bytes() == expected

    bound = load_knowledge_package(
        destination,
        bindings=_bindings(runtime_value),
    )
    assert bound.package == package
    assert bound.bindings["sales.orders"] is runtime_value
    with pytest.raises(TypeError):
        bound.bindings["sales.orders"] = object()


def test_save_does_not_overwrite_without_explicit_permission(tmp_path: Path) -> None:
    destination = tmp_path / "knowledge.json"
    destination.write_bytes(b"original")

    with pytest.raises(FileExistsError):
        save_knowledge_package(destination, _package())

    assert destination.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [destination]


def test_save_overwrites_only_when_explicitly_requested(tmp_path: Path) -> None:
    destination = tmp_path / "knowledge.json"
    destination.write_bytes(b"old")
    package = _full_package()
    expected = canonical_json(
        {
            "format_version": 1,
            "package": package.to_dict(),
            "package_fingerprint": package.fingerprint,
        }
    ).encode("utf-8") + b"\n"

    save_knowledge_package(destination, package, overwrite=True)

    assert destination.read_bytes() == expected
    assert load_knowledge_package(destination, bindings=_full_bindings()).package == package
    assert list(tmp_path.iterdir()) == [destination]


def test_failed_atomic_replace_preserves_valid_original_and_cleans_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fabric_rlm.knowledge_store as store

    destination = tmp_path / "knowledge.json"
    package_a = _package(diagnostics={"generation": 1})
    package_b = _package(diagnostics={"generation": 2})
    save_knowledge_package(destination, package_a)
    original = destination.read_bytes()
    real_replace = store.os.replace
    calls = 0

    def fail_replace(source: object, target: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            Path(target).unlink()
            raise OSError("simulated destructive replace failure")
        real_replace(source, target)

    monkeypatch.setattr(store.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        save_knowledge_package(destination, package_b, overwrite=True)

    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]


def test_restore_failure_reports_publication_and_integrity_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fabric_rlm.knowledge_store as store

    destination = tmp_path / "knowledge.json"
    save_knowledge_package(destination, _package(diagnostics={"generation": 1}))
    calls = 0

    def fail_replace(source: object, target: object) -> None:
        nonlocal calls
        calls += 1
        Path(target).unlink(missing_ok=True)
        raise OSError(f"replace failure {calls}")

    monkeypatch.setattr(store.os, "replace", fail_replace)

    with pytest.raises(PersistenceIntegrityError, match="publication.*restore") as error:
        save_knowledge_package(
            destination,
            _package(diagnostics={"generation": 2}),
            overwrite=True,
        )

    assert isinstance(error.value.__cause__, OSError)
    assert not list(tmp_path.glob(".*.tmp"))
    backups = list(tmp_path.glob(".*.bak"))
    assert len(backups) == 1
    assert load_knowledge_package(backups[0], bindings=_bindings()).package == _package(
        diagnostics={"generation": 1}
    )


def test_temporary_fsync_failure_cleans_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fabric_rlm.knowledge_store as store

    def fail_fsync(file_descriptor: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(store.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="fsync"):
        save_knowledge_package(tmp_path / "knowledge.json", _package())

    assert list(tmp_path.iterdir()) == []


def test_temporary_write_failure_cleans_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fabric_rlm.knowledge_store as store

    real_fdopen = store.os.fdopen

    class FailingWriter:
        def __init__(self, descriptor: int, mode: str) -> None:
            self.stream = real_fdopen(descriptor, mode)

        def __enter__(self) -> FailingWriter:
            return self

        def __exit__(self, *args: object) -> None:
            self.stream.close()

        def write(self, data: bytes) -> int:
            raise OSError("simulated write failure")

    monkeypatch.setattr(store.os, "fdopen", FailingWriter)

    with pytest.raises(OSError, match="write"):
        save_knowledge_package(tmp_path / "knowledge.json", _package())

    assert list(tmp_path.iterdir()) == []


def test_save_creates_missing_destination_parents(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "packages" / "knowledge.json"

    save_knowledge_package(destination, _package())

    assert destination.is_file()


def test_save_rejects_oversized_envelope_before_filesystem_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fabric_rlm.knowledge_store as store

    destination = tmp_path / "missing" / "knowledge.json"
    monkeypatch.setattr(
        store,
        "_envelope_bytes",
        lambda package: b"x" * (MAX_PACKAGE_BYTES + 1),
    )

    with pytest.raises(KnowledgePersistenceError, match="maximum"):
        save_knowledge_package(destination, _package())

    assert not destination.parent.exists()
    assert not destination.exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_no_overwrite_unsupported_publication_fails_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fabric_rlm.knowledge_store as store

    primitive = "rename" if os.name == "nt" else "link"

    def unsupported(*args: object, **kwargs: object) -> None:
        raise OSError(errno.ENOTSUP, "unsupported")

    monkeypatch.setattr(store.os, primitive, unsupported)

    with pytest.raises(KnowledgePersistenceError, match="atomic no-clobber"):
        save_knowledge_package(tmp_path / "knowledge.json", _package())

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "contents",
    [
        b"{not-json\n",
        b"\xff\n",
        b"[]\n",
    ],
)
def test_load_fails_closed_for_corrupt_json(
    tmp_path: Path,
    contents: bytes,
) -> None:
    destination = tmp_path / "knowledge.json"
    destination.write_bytes(contents)

    with pytest.raises(ValueError):
        load_knowledge_package(destination, bindings=_bindings())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda envelope: envelope.pop("format_version"),
        lambda envelope: envelope.update(format_version=2),
        lambda envelope: envelope.update(format_version=True),
        lambda envelope: envelope.update(unexpected=True),
        lambda envelope: envelope.pop("package_fingerprint"),
        lambda envelope: envelope.update(package_fingerprint="tampered"),
        lambda envelope: envelope["package"].update(unexpected=True),
        lambda envelope: envelope["package"].update(format_version=True),
        lambda envelope: envelope["package"].update(sources=[]),
    ],
)
def test_load_rejects_invalid_envelopes_and_package_contracts(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
) -> None:
    package = _package()
    envelope = {
        "format_version": 1,
        "package": package.to_dict(),
        "package_fingerprint": package.fingerprint,
    }
    mutation(envelope)
    destination = tmp_path / "knowledge.json"
    _write_envelope(destination, envelope)

    with pytest.raises(ValueError):
        load_knowledge_package(destination, bindings=_bindings())


@pytest.mark.parametrize("field", ["sql_template", "code", "dax", "python"])
def test_load_rejects_legacy_executable_operation_fields(
    tmp_path: Path,
    field: str,
) -> None:
    package = _full_package()
    payload = package.to_dict()
    payload["operations"][0][field] = "arbitrary executable text"
    destination = tmp_path / "knowledge.json"
    _write_envelope(
        destination,
        {
            "format_version": 1,
            "package": payload,
            "package_fingerprint": package.fingerprint,
        },
    )

    with pytest.raises(ValueError, match="malformed"):
        load_knowledge_package(destination, bindings=_full_bindings())


def test_load_requires_explicit_exact_source_alias_bindings(tmp_path: Path) -> None:
    destination = tmp_path / "knowledge.json"
    save_knowledge_package(destination, _package())
    valid = _bindings()

    with pytest.raises(TypeError):
        load_knowledge_package(destination)
    with pytest.raises(ValueError, match="bindings.*empty"):
        load_knowledge_package(destination, bindings={})
    with pytest.raises(ValueError, match="extra"):
        load_knowledge_package(
            destination,
            bindings={**valid, "sales.extra": valid["sales.orders"]},
        )
    with pytest.raises(ValueError, match="exact source aliases"):
        load_knowledge_package(
            destination,
            bindings={"renamed.orders": valid["sales.orders"]},
        )


def test_load_names_missing_source_aliases(tmp_path: Path) -> None:
    package = KnowledgePackage(
        package_id="sales.knowledge.v1",
        sources=(
            _package().sources[0],
            SourceProfile(
                source_id="sales.customers",
                family="delta_table",
                locator="lakehouse/sales/customers",
                snapshot_fingerprint="snapshot-customers",
                schema_fingerprint="schema-customers",
                schema={"customer_id": {"type": "integer", "nullable": False}},
            ),
        ),
    )
    destination = tmp_path / "knowledge.json"
    save_knowledge_package(destination, package)

    with pytest.raises(ValueError, match="missing aliases: sales.customers"):
        load_knowledge_package(destination, bindings=_bindings())


def test_load_only_opens_package_file_not_logical_source_locators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "knowledge.json"
    save_knowledge_package(destination, _package())
    opened: list[Path] = []
    original_open = Path.open

    def record_open(path: Path, *args: object, **kwargs: object) -> object:
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_open)

    load_knowledge_package(destination, bindings=_bindings())

    assert opened == [destination]


@pytest.mark.parametrize(
    "descriptor",
    [
        SourceBindingDescriptor(
            source_id="sales.other",
            locator="lakehouse/sales/orders",
        ),
        SourceBindingDescriptor(
            source_id="sales.orders",
            locator="lakehouse/sales/other",
        ),
    ],
)
def test_load_validates_explicit_binding_identity_and_locator(
    tmp_path: Path,
    descriptor: SourceBindingDescriptor,
) -> None:
    destination = tmp_path / "knowledge.json"
    save_knowledge_package(destination, _package())

    with pytest.raises(ValueError, match="descriptor"):
        load_knowledge_package(
            destination,
            bindings={
                "sales.orders": SourceBinding(
                    descriptor=descriptor,
                    value=object(),
                )
            },
        )


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "credential",
        "credentials",
        "secret",
        "secrets",
        "sas",
        "password",
        "api_key",
        "private_key",
        "authorization",
        "cookie",
        "connection_string",
        "access_token",
        "client_secret",
        "sas_url",
        "shared_access_signature",
    ],
)
def test_save_rejects_privacy_forbidden_fields_at_any_nesting(
    tmp_path: Path,
    forbidden_name: str,
) -> None:
    package = _package(diagnostics={"safe": [{forbidden_name: "sentinel"}]})

    with pytest.raises(ValueError, match="privacy-forbidden"):
        save_knowledge_package(tmp_path / "knowledge.json", package)


def test_privacy_scan_allows_precise_contract_names_and_reason_fields(
    tmp_path: Path,
) -> None:
    package = _package(
        diagnostics={
            "snapshot_fingerprint": "diagnostic-copy",
            "state_code": "ready",
            "content_type": "tabular",
            "responsiveness_score": 10,
            "response_time": 12,
            "reasoning_score": 0.75,
            "token_count": 42,
            "estimated_units": 20,
        }
    )

    save_knowledge_package(tmp_path / "knowledge.json", package)


def test_full_contract_package_round_trips_through_persistence(tmp_path: Path) -> None:
    package = _full_package()
    destination = tmp_path / "knowledge.json"

    save_knowledge_package(destination, package)

    assert load_knowledge_package(destination, bindings=_full_bindings()).package == package


def test_save_rejects_free_text_disguised_as_schema_metadata(
    tmp_path: Path,
) -> None:
    package = KnowledgePackage(
        package_id="sales.knowledge.v1",
        sources=(
            SourceProfile(
                source_id="sales.orders",
                family="delta_table",
                locator="lakehouse/sales/orders",
                snapshot_fingerprint="snapshot-orders",
                schema_fingerprint="schema-orders",
                schema={
                    "content": {
                        "type": "string",
                        "description": "raw customer content",
                    }
                },
            ),
        ),
    )

    with pytest.raises(ValueError, match="structural schema"):
        save_knowledge_package(tmp_path / "knowledge.json", package)


@pytest.mark.parametrize("field", ["password", "api_key", "client_secret"])
def test_save_rejects_credential_named_operation_parameters(
    tmp_path: Path,
    field: str,
) -> None:
    package = _full_package()
    operation = package.operations[0].to_dict()
    operation["parameter_schema"] = {field: {"type": "string"}}
    operation["parameter_defaults"] = {field: "otherwise-neutral"}
    unsafe = KnowledgePackage(
        package_id=package.package_id,
        sources=package.sources,
        relationships=package.relationships,
        operations=(
            RegisteredOperation.from_dict(operation),
            package.operations[1],
        ),
        events=package.events,
    )

    with pytest.raises(ValueError, match="privacy-forbidden"):
        save_knowledge_package(tmp_path / "knowledge.json", unsafe)


def test_load_applies_privacy_scan_even_with_a_valid_fingerprint(
    tmp_path: Path,
) -> None:
    package = _package(diagnostics={"nested": {"access_token": "sentinel"}})
    destination = tmp_path / "knowledge.json"
    _write_envelope(
        destination,
        {
            "format_version": 1,
            "package": package.to_dict(),
            "package_fingerprint": package.fingerprint,
        },
    )

    with pytest.raises(ValueError, match="privacy-forbidden"):
        load_knowledge_package(destination, bindings=_bindings())


@pytest.mark.parametrize(
    "unsafe_value",
    [
        r"C:\private\orders.csv",
        r"C:orders.csv",
        r"C:.",
        "C%3Aorders.csv",
        "%43%3Aorders.csv",
        r"\\server\share\orders.csv",
        "/private/orders.csv",
        "file:///private/orders.csv",
        "data:text/plain,secret",
        "mailto:user@example.test",
        "jdbc:sqlserver://example.test;password=secret",
        "https://user:password@example.test/orders",
        "https://example.test/orders?sig=secret",
        "https://example.test/orders#private",
        "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----",
        "Bearer abc.def.ghi",
        "AKIAIOSFODNN7EXAMPLE",
        "sv=2024-01-01&sig=secret-signature",
        "Server=db;Database=sales;User Id=admin;Password=secret",
    ],
)
def test_save_rejects_unsafe_string_values_anywhere_in_package(
    tmp_path: Path,
    unsafe_value: str,
) -> None:
    package = _package(diagnostics={"sentinel": [{"value": unsafe_value}]})

    with pytest.raises(ValueError, match="unsafe string"):
        save_knowledge_package(tmp_path / "knowledge.json", package)


@pytest.mark.parametrize(
    "value",
    ["customer raw text may contain secrets", "secret-looking-code"],
)
def test_flexible_diagnostics_reject_arbitrary_free_text(
    tmp_path: Path,
    value: str,
) -> None:
    package = _package(diagnostics={"note": value})

    with pytest.raises(ValueError, match="bounded metadata"):
        save_knowledge_package(tmp_path / "knowledge.json", package)


def test_load_rejects_unsafe_drive_relative_flexible_values(tmp_path: Path) -> None:
    package = _package(diagnostics={"state_code": "C%3Aorders.csv"})
    destination = tmp_path / "knowledge.json"
    _write_envelope(
        destination,
        {
            "format_version": 1,
            "package": package.to_dict(),
            "package_fingerprint": package.fingerprint,
        },
    )

    with pytest.raises(ValueError, match="unsafe string"):
        load_knowledge_package(destination, bindings=_bindings())


@pytest.mark.parametrize(
    "locator",
    ["C:orders.csv", "C:.", "C%3Aorders.csv", "%43%3Aorders.csv"],
)
def test_load_rejects_drive_prefixed_logical_locators(
    tmp_path: Path,
    locator: str,
) -> None:
    package = _package()
    payload = package.to_dict()
    payload["sources"][0]["locator"] = locator
    destination = tmp_path / "knowledge.json"
    _write_envelope(
        destination,
        {
            "format_version": 1,
            "package": payload,
            "package_fingerprint": package.fingerprint,
        },
    )

    with pytest.raises(ValueError, match="unsafe string"):
        load_knowledge_package(destination, bindings=_bindings())


def test_load_rejects_duplicate_envelope_and_nested_package_keys(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "knowledge.json"
    destination.write_text(
        '{"format_version":1,"format_version":1,"package":{},'
        '"package_fingerprint":"x"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate field: format_version"):
        load_knowledge_package(destination, bindings=_bindings())

    destination.write_text(
        '{"format_version":1,"package":{"format_version":1,'
        '"package_id":"one","package_id":"two"},"package_fingerprint":"x"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate field: package_id"):
        load_knowledge_package(destination, bindings=_bindings())


def test_load_rejects_oversized_file_and_growth_during_bounded_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "knowledge.json"
    destination.write_bytes(b"x" * (MAX_PACKAGE_BYTES + 1))
    with pytest.raises(KnowledgePersistenceError, match="maximum"):
        load_knowledge_package(destination, bindings=_bindings())

    destination.write_bytes(b"{}")
    real_open = Path.open

    class GrowingReader:
        def __enter__(self) -> GrowingReader:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            return b"x" * (MAX_PACKAGE_BYTES + 1)

    def growing_open(path: Path, *args: object, **kwargs: object) -> object:
        if path == destination and args and args[0] == "rb":
            return GrowingReader()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", growing_open)
    with pytest.raises(KnowledgePersistenceError, match="maximum"):
        load_knowledge_package(destination, bindings=_bindings())


def test_binding_inputs_are_defensively_copied(tmp_path: Path) -> None:
    destination = tmp_path / "knowledge.json"
    save_knowledge_package(destination, _package())
    bindings = _bindings()

    bound = load_knowledge_package(destination, bindings=bindings)
    bindings.clear()

    assert tuple(bound.bindings) == ("sales.orders",)


def test_canonical_utf8_envelope_matches_independent_golden_bytes(
    tmp_path: Path,
) -> None:
    base = _package()
    package = KnowledgePackage(
        package_id=base.package_id,
        sources=base.sources,
        events=(
            KnowledgeEvent(
                event_id="event-1",
                event_type="source.activated",
                subject_type="source",
                subject_id="sales.orders",
                status="active",
                reason_code="validated",
            ),
        ),
    )
    destination = tmp_path / "knowledge.json"

    save_knowledge_package(destination, package)

    expected_package = (
        '{"events":[{"event_id":"event-1","event_type":"source.activated",'
        '"reason_code":"validated","status":"active","subject_id":"sales.orders",'
        '"subject_type":"source"}],"format_version":1,"operations":[],'
        '"package_id":"sales.knowledge.v1","relationships":[],"sources":'
        '[{"diagnostics":{},"family":"delta_table",'
        '"locator":"lakehouse/sales/orders","role":"numeric_evidence",'
        '"schema":{"order_id":{"nullable":false,"type":"integer"}},'
        '"schema_fingerprint":"schema-orders","sensitive_columns":[],'
        '"snapshot_fingerprint":"snapshot-orders","source_id":"sales.orders",'
        '"status":"candidate"}]}'
    )
    expected = (
        '{"format_version":1,"package":'
        + expected_package
        + ',"package_fingerprint":"'
        + "99f4dee01983512bc70ee8bd17cbba7318aedd7c9b6ec0e6372724d99ef32d06"
        + '"}\n'
    ).encode("utf-8")
    assert destination.read_bytes() == expected


def test_runtime_binding_objects_are_not_serialized_or_fingerprinted(
    tmp_path: Path,
) -> None:
    class RuntimeOnly:
        secret = "must-not-persist"

    package = _package()
    destination = tmp_path / "knowledge.json"
    save_knowledge_package(destination, package)
    before = destination.read_bytes()

    bound = load_knowledge_package(
        destination,
        bindings=_bindings(RuntimeOnly()),
    )

    assert bound.package.fingerprint == package.fingerprint
    assert destination.read_bytes() == before
    assert b"must-not-persist" not in before
