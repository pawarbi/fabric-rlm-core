from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from fabric_rlm.knowledge import KnowledgePackage, SourceProfile, canonical_json
from fabric_rlm.knowledge_store import (
    MAX_PACKAGE_BYTES,
    KnowledgePersistenceError,
    SourceBinding,
    SourceBindingDescriptor,
    _envelope_bytes,
    save_knowledge_package,
)
from fabric_rlm.onelake_knowledge_store import (
    AtomicRenameUnsupported,
    OneLakeKnowledgeLocation,
    OneLakeObjectStat,
    load_onelake_knowledge_package,
    save_onelake_knowledge_package,
)


ROOT = (
    "abfss://workspace@onelake.dfs.fabric.microsoft.com/"
    "lakehouse/Files"
)
LOGICAL_LOCATOR = "knowledge/sales.package.json"


def _package(*, package_id: str = "sales.knowledge.v1") -> KnowledgePackage:
    return KnowledgePackage(
        package_id=package_id,
        sources=(
            SourceProfile(
                source_id="sales.orders",
                family="delta_table",
                locator="lakehouse/sales/orders",
                snapshot_fingerprint="snapshot-orders",
                schema_fingerprint="schema-orders",
                schema={"order_id": {"type": "integer", "nullable": False}},
            ),
        ),
    )


def _bindings(value: object | None = None) -> dict[str, SourceBinding]:
    return {
        "sales.orders": SourceBinding(
            SourceBindingDescriptor(
                source_id="sales.orders",
                locator="lakehouse/sales/orders",
            ),
            object() if value is None else value,
        )
    }


class FakeOneLakeTransport:
    def __init__(
        self,
        files: dict[str, bytes] | None = None,
        *,
        conditional_supported: bool = True,
    ) -> None:
        self.files = files if files is not None else {}
        self.versions = {path: 1 for path in self.files}
        self.conditional_supported = conditional_supported
        self.calls: list[tuple[object, ...]] = []
        self.change_during_read: set[str] = set()
        self.tamper_upload = False
        self.change_before_rename = False
        self.fail_publication = False
        self.fail_restore = False

    def stat(self, path: str) -> OneLakeObjectStat | None:
        self.calls.append(("stat", path))
        if path not in self.files:
            return None
        return OneLakeObjectStat(
            size=len(self.files[path]),
            etag=f'"v{self.versions[path]}"',
        )

    def read(self, path: str, max_bytes: int) -> bytes:
        self.calls.append(("read", path, max_bytes))
        data = self.files[path][:max_bytes]
        if path in self.change_during_read:
            self.files[path] = data + b" "
            self.versions[path] += 1
            self.change_during_read.remove(path)
        return data

    def mkdirs(self, path: str) -> None:
        self.calls.append(("mkdirs", path))

    def upload(self, path: str, data: bytes) -> None:
        self.calls.append(("upload", path, data))
        self.files[path] = data + (b" " if self.tamper_upload else b"")
        self.versions[path] = self.versions.get(path, 0) + 1
        self.tamper_upload = False

    def rename_no_clobber(
        self,
        source: str,
        destination: str,
        *,
        source_etag: str | None,
    ) -> None:
        self.calls.append(("rename_no_clobber", source, destination))
        if not self.conditional_supported:
            raise AtomicRenameUnsupported
        self._check_source_etag(source, source_etag)
        if destination in self.files:
            raise FileExistsError
        self.files[destination] = self.files.pop(source)
        self.versions[destination] = self.versions.pop(source) + 1

    def rename_overwrite(
        self,
        source: str,
        destination: str,
        *,
        source_etag: str | None,
    ) -> None:
        self.calls.append(("rename_overwrite", source, destination))
        self._check_source_etag(source, source_etag)
        if self.fail_publication and ".tmp-" in source:
            self.files[destination] = b"partial"
            self.versions[destination] = self.versions.get(destination, 0) + 1
            self.fail_publication = False
            raise RuntimeError(f"publication failed at {destination}?sig=secret")
        if self.fail_restore and ".backup-" in source:
            raise RuntimeError(f"restore failed at {destination}?sig=secret")
        self.files[destination] = self.files.pop(source)
        self.versions[destination] = self.versions.pop(source) + 1

    def _check_source_etag(self, source: str, source_etag: str | None) -> None:
        if self.change_before_rename:
            self.files[source] += b" "
            self.versions[source] += 1
            self.change_before_rename = False
        if (
            source_etag is not None
            and source_etag != f'"v{self.versions[source]}"'
        ):
            raise RuntimeError("source etag changed?sig=secret")

    def delete(self, path: str) -> None:
        self.calls.append(("delete", path))
        self.files.pop(path, None)
        self.versions.pop(path, None)


def _location(root: str = ROOT) -> OneLakeKnowledgeLocation:
    return OneLakeKnowledgeLocation(root=root, locator=LOGICAL_LOCATOR)


def _target(root: str = ROOT) -> str:
    return f"{root}/{LOGICAL_LOCATOR}"


def test_onelake_save_matches_local_canonical_envelope_bytes(
    tmp_path: Path,
) -> None:
    package = _package()
    transport = FakeOneLakeTransport()
    local = tmp_path / "knowledge.json"

    save_knowledge_package(local, package)
    save_onelake_knowledge_package(_location(), package, transport=transport)

    assert transport.files[_target()] == local.read_bytes()


def test_location_is_immutable_and_runtime_root_is_not_fingerprinted() -> None:
    package = _package()
    first = _location()
    second = _location(
        "abfss://other-workspace@onelake.dfs.fabric.microsoft.com/"
        "other-lakehouse/Files"
    )

    with pytest.raises(FrozenInstanceError):
        first.locator = "other/package.json"

    assert first.locator == second.locator
    assert package.fingerprint == _package().fingerprint
    assert ROOT not in repr(first)


@pytest.mark.parametrize(
    "root",
    [
        "https://onelake.dfs.fabric.microsoft.com/lakehouse/Files",
        "abfss://workspace@other.example/lakehouse/Files",
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse/Tables",
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse/Files/..",
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse/Files?sig=x",
    ],
)
def test_location_requires_canonical_onelake_files_root(root: str) -> None:
    with pytest.raises(ValueError, match="Files root"):
        _location(root)


def test_abfss_root_and_transport_error_details_are_never_disclosed() -> None:
    transport = FakeOneLakeTransport()
    transport.fail_publication = True
    package = _package()

    with pytest.raises(Exception) as captured:
        save_onelake_knowledge_package(
            _location(),
            package,
            transport=transport,
            overwrite=True,
        )

    message = str(captured.value)
    assert ROOT not in message
    assert "secret" not in message
    assert captured.value.__cause__ is None
    assert ROOT.encode() not in _envelope_bytes(package)
    assert ROOT not in package.fingerprint


def test_load_reads_and_validates_before_requiring_exact_bindings() -> None:
    package = _package()
    transport = FakeOneLakeTransport({_target(): _envelope_bytes(package)})

    with pytest.raises(ValueError, match="bindings must not be empty"):
        load_onelake_knowledge_package(
            _location(),
            transport=transport,
            bindings={},
        )

    assert any(call[0] == "read" for call in transport.calls)


def test_no_overwrite_conflict_preserves_destination_and_cleans_temp() -> None:
    original = b"original"
    transport = FakeOneLakeTransport({_target(): original})

    with pytest.raises(FileExistsError):
        save_onelake_knowledge_package(
            _location(),
            _package(),
            transport=transport,
        )

    assert transport.files == {_target(): original}


def test_unsupported_conditional_rename_fails_explicitly_and_cleans_temp() -> None:
    transport = FakeOneLakeTransport(conditional_supported=False)

    with pytest.raises(
        KnowledgePersistenceError,
        match="atomic no-clobber publication is unsupported",
    ):
        save_onelake_knowledge_package(
            _location(),
            _package(),
            transport=transport,
        )

    assert transport.files == {}


def test_overwrite_publishes_new_bytes_and_cleans_backup_and_temp() -> None:
    old = _envelope_bytes(_package(package_id="old.package"))
    package = _package()
    transport = FakeOneLakeTransport({_target(): old})

    save_onelake_knowledge_package(
        _location(),
        package,
        transport=transport,
        overwrite=True,
    )

    assert transport.files == {_target(): _envelope_bytes(package)}


def test_failed_overwrite_restores_original_and_cleans_temporary_objects() -> None:
    original = _envelope_bytes(_package(package_id="old.package"))
    transport = FakeOneLakeTransport({_target(): original})
    transport.fail_publication = True

    with pytest.raises(KnowledgePersistenceError, match="publication failed"):
        save_onelake_knowledge_package(
            _location(),
            _package(),
            transport=transport,
            overwrite=True,
        )

    assert transport.files == {_target(): original}


def test_failed_overwrite_restore_reports_uncertain_integrity_and_keeps_backup() -> None:
    original = _envelope_bytes(_package(package_id="old.package"))
    transport = FakeOneLakeTransport({_target(): original})
    transport.fail_publication = True
    transport.fail_restore = True

    with pytest.raises(
        KnowledgePersistenceError,
        match="publication failed and restore also failed",
    ) as captured:
        save_onelake_knowledge_package(
            _location(),
            _package(),
            transport=transport,
            overwrite=True,
        )

    assert captured.value.__cause__ is None
    assert transport.files[_target()] == b"partial"
    backups = [path for path in transport.files if ".backup-" in path]
    assert len(backups) == 1
    assert transport.files[backups[0]] == original
    assert not any(".tmp-" in path for path in transport.files)


def test_save_rejects_oversized_envelope_before_remote_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeOneLakeTransport()
    monkeypatch.setattr(
        "fabric_rlm.onelake_knowledge_store._envelope_bytes",
        lambda package: b"x" * (MAX_PACKAGE_BYTES + 1),
    )

    with pytest.raises(KnowledgePersistenceError, match="maximum"):
        save_onelake_knowledge_package(
            _location(),
            _package(),
            transport=transport,
        )

    assert transport.calls == []


def test_load_rejects_oversized_object_before_remote_read() -> None:
    transport = FakeOneLakeTransport(
        {_target(): b"x" * (MAX_PACKAGE_BYTES + 1)}
    )

    with pytest.raises(KnowledgePersistenceError, match="maximum"):
        load_onelake_knowledge_package(
            _location(),
            transport=transport,
            bindings=_bindings(),
        )

    assert not any(call[0] == "read" for call in transport.calls)


def test_save_detects_tampered_temporary_upload_and_cleans_it() -> None:
    transport = FakeOneLakeTransport()
    transport.tamper_upload = True

    with pytest.raises(KnowledgePersistenceError, match="verification"):
        save_onelake_knowledge_package(
            _location(),
            _package(),
            transport=transport,
        )

    assert transport.files == {}


def test_save_conditions_publication_on_verified_temporary_etag() -> None:
    transport = FakeOneLakeTransport()
    transport.change_before_rename = True

    with pytest.raises(KnowledgePersistenceError, match="publication failed"):
        save_onelake_knowledge_package(
            _location(),
            _package(),
            transport=transport,
        )

    assert transport.files == {}


def test_load_rejects_tampered_package_fingerprint() -> None:
    package = _package()
    envelope = json.loads(_envelope_bytes(package))
    envelope["package"]["package_id"] = "tampered.package"
    data = (canonical_json(envelope) + "\n").encode()
    transport = FakeOneLakeTransport({_target(): data})

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_onelake_knowledge_package(
            _location(),
            transport=transport,
            bindings=_bindings(),
        )


def test_load_rejects_object_that_changes_during_read() -> None:
    package = _package()
    transport = FakeOneLakeTransport({_target(): _envelope_bytes(package)})
    transport.change_during_read.add(_target())

    with pytest.raises(KnowledgePersistenceError, match="changed during read"):
        load_onelake_knowledge_package(
            _location(),
            transport=transport,
            bindings=_bindings(),
        )


def test_separate_session_load_uses_new_runtime_location_and_binding_handles() -> None:
    package = _package()
    storage: dict[str, bytes] = {}
    first_transport = FakeOneLakeTransport(storage)
    save_onelake_knowledge_package(
        OneLakeKnowledgeLocation(ROOT, LOGICAL_LOCATOR),
        package,
        transport=first_transport,
    )

    runtime_handle = object()
    second_transport = FakeOneLakeTransport(storage)
    bound = load_onelake_knowledge_package(
        OneLakeKnowledgeLocation(
            "".join((ROOT[:-5], "Files")),
            "".join(("knowledge/", "sales.package.json")),
        ),
        transport=second_transport,
        bindings=_bindings(runtime_handle),
    )

    assert bound.package == package
    assert bound.bindings["sales.orders"] is runtime_handle
