from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from fabric_rlm.knowledge import KnowledgePackage, SourceProfile, canonical_json
from fabric_rlm.knowledge_store import (
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

    save_knowledge_package(destination, _package(), overwrite=True)

    assert json.loads(destination.read_text(encoding="utf-8"))[
        "package_fingerprint"
    ] == _package().fingerprint


def test_failed_atomic_replace_preserves_valid_original_and_cleans_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fabric_rlm.knowledge_store as store

    destination = tmp_path / "knowledge.json"
    save_knowledge_package(destination, _package())
    original = destination.read_bytes()

    def fail_replace(source: object, target: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(store.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        save_knowledge_package(destination, _package(), overwrite=True)

    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]


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


def test_load_never_opens_the_persisted_locator(
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
        "raw_rows",
        "raw_content",
        "rows",
        "content",
        "credential",
        "credentials",
        "secret",
        "secrets",
        "token",
        "tokens",
        "sas",
        "code",
        "stdout",
        "stderr",
        "traceback",
        "prompt",
        "response",
        "reasoning",
        "chain_of_thought",
        "state",
        "access_token",
        "client_secret",
        "sas_url",
        "source_code",
        "captured_stdout",
        "model_prompt",
        "model_response",
        "execution_state",
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
            "reason": "contract-known explanation",
            "responsiveness_score": 10,
            "estimated_units": 20,
        }
    )

    save_knowledge_package(tmp_path / "knowledge.json", package)


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
        r"\\server\share\orders.csv",
        "/private/orders.csv",
        "file:///private/orders.csv",
        "https://user:password@example.test/orders",
        "https://example.test/orders?sig=secret",
        "https://example.test/orders#private",
    ],
)
def test_save_rejects_unsafe_string_values_anywhere_in_package(
    tmp_path: Path,
    unsafe_value: str,
) -> None:
    package = _package(diagnostics={"sentinel": [{"value": unsafe_value}]})

    with pytest.raises(ValueError, match="unsafe string"):
        save_knowledge_package(tmp_path / "knowledge.json", package)


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
