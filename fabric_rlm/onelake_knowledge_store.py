"""Canonical knowledge-package persistence over a parent-authorized OneLake transport.

The transport boundary is intentionally injectable and credential-free. Fabric's
``notebookutils.fs.put`` and ``append`` APIs explicitly lack concurrent-write
atomicity, while ``mv`` does not document conditional destination creation:
https://learn.microsoft.com/en-us/fabric/data-engineering/notebookutils/notebookutils-file-system

Production transports should therefore implement rename with the documented
ADLS Gen2 Path Create/Rename REST operation, ``x-ms-rename-source``, and
``If-None-Match: *`` for no-clobber publication. When an ETag is available,
``source_etag`` maps to the documented ``x-ms-source-if-match`` condition:
https://learn.microsoft.com/en-us/rest/api/storageservices/datalakestoragegen2/path/create
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import hmac
from typing import Protocol
import uuid

from fabric_rlm.artifacts import (
    _normalize_destination_root,
    _safe_relative_path,
)
from fabric_rlm.knowledge import KnowledgePackage, _logical_locator
from fabric_rlm.knowledge_store import (
    MAX_PACKAGE_BYTES,
    BoundKnowledgePackage,
    KnowledgePersistenceError,
    PersistenceIntegrityError,
    SourceBinding,
    _bind_package,
    _envelope_bytes,
    _parse_envelope,
)


class AtomicRenameUnsupported(Exception):
    """The transport cannot conditionally rename without replacing a target."""


class ConcurrentWriteError(Exception):
    """A conditional OneLake operation rejected a concurrent change."""


@dataclass(frozen=True)
class OneLakeObjectStat:
    """Bounded metadata used to detect oversized or concurrently changed objects."""

    size: int
    etag: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, int):
            raise ValueError("remote object size must be an integer")
        if self.size < 0:
            raise ValueError("remote object size must not be negative")
        if self.etag is not None and not isinstance(self.etag, str):
            raise ValueError("remote object etag must be a string")


class OneLakeKnowledgeTransport(Protocol):
    """Operations supplied by a trusted parent that owns OneLake authorization.

    Rename implementations must be server-side and atomic. ``rename_no_clobber``
    must condition destination creation; it must raise
    :class:`AtomicRenameUnsupported` rather than emulate the condition with a
    racy existence check. A non-null ``source_etag`` must condition the source
    of the rename rather than being checked client-side.
    """

    def stat(self, path: str) -> OneLakeObjectStat | None: ...

    def read(self, path: str, max_bytes: int) -> bytes: ...

    def mkdirs(self, path: str) -> None: ...

    def upload(self, path: str, data: bytes) -> None: ...

    def rename_no_clobber(
        self,
        source: str,
        destination: str,
        *,
        source_etag: str | None,
    ) -> None: ...

    def rename_overwrite(
        self,
        source: str,
        destination: str,
        *,
        source_etag: str | None,
        destination_etag: str | None,
    ) -> None: ...

    def delete(self, path: str) -> None: ...


@dataclass(frozen=True)
class OneLakeKnowledgeLocation:
    """A safe logical package locator plus a runtime-only canonical Files root."""

    root: str = field(repr=False)
    locator: str

    def __post_init__(self) -> None:
        try:
            root = _normalize_destination_root(self.root)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "OneLake knowledge root must be a canonical OneLake ABFSS Files root"
            ) from exc
        logical = _logical_locator(self.locator)
        try:
            locator = _safe_relative_path(logical).as_posix()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "OneLake knowledge locator must be a safe logical relative path"
            ) from exc
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "locator", locator)

    @property
    def _target(self) -> str:
        return f"{self.root}/{self.locator}"

    @property
    def _parent(self) -> str:
        parent, _, _ = self._target.rpartition("/")
        return parent


def _safe_stat(
    transport: OneLakeKnowledgeTransport,
    path: str,
) -> OneLakeObjectStat | None:
    try:
        value = transport.stat(path)
    except Exception:
        raise KnowledgePersistenceError(
            "OneLake package metadata read failed"
        ) from None
    if value is not None and not isinstance(value, OneLakeObjectStat):
        raise KnowledgePersistenceError("OneLake package metadata is malformed")
    return value


def _read_stable(
    transport: OneLakeKnowledgeTransport,
    path: str,
) -> tuple[bytes, OneLakeObjectStat]:
    before = _safe_stat(transport, path)
    if before is None:
        raise FileNotFoundError
    if before.size > MAX_PACKAGE_BYTES:
        raise KnowledgePersistenceError("knowledge package exceeds maximum byte size")
    try:
        data = transport.read(path, MAX_PACKAGE_BYTES + 1)
    except Exception:
        raise KnowledgePersistenceError("OneLake package read failed") from None
    if not isinstance(data, bytes):
        raise KnowledgePersistenceError("OneLake package read returned invalid bytes")
    if len(data) > MAX_PACKAGE_BYTES:
        raise KnowledgePersistenceError("knowledge package exceeds maximum byte size")

    after = _safe_stat(transport, path)
    if (
        after is None
        or before.size != after.size
        or len(data) != before.size
        or (
            before.etag is not None
            and after.etag is not None
            and before.etag != after.etag
        )
    ):
        raise KnowledgePersistenceError("OneLake package changed during read")
    return data, after


def _upload_and_verify(
    transport: OneLakeKnowledgeTransport,
    path: str,
    data: bytes,
) -> OneLakeObjectStat:
    try:
        transport.upload(path, data)
    except Exception:
        raise KnowledgePersistenceError("OneLake temporary upload failed") from None
    uploaded, uploaded_stat = _read_stable(transport, path)
    expected_digest = hashlib.sha256(data).digest()
    actual_digest = hashlib.sha256(uploaded).digest()
    if not hmac.compare_digest(expected_digest, actual_digest) or uploaded != data:
        raise KnowledgePersistenceError("OneLake temporary upload verification failed")
    if not uploaded_stat.etag:
        raise KnowledgePersistenceError(
            "OneLake temporary upload did not return a usable ETag"
        )
    return uploaded_stat


def _temporary_path(target: str, kind: str) -> str:
    parent, _, name = target.rpartition("/")
    return f"{parent}/.{name}.{kind}-{uuid.uuid4().hex}"


def _delete_quietly(
    transport: OneLakeKnowledgeTransport,
    path: str | None,
) -> None:
    if path is None:
        return
    try:
        transport.delete(path)
    except Exception:
        pass


def save_onelake_knowledge_package(
    location: OneLakeKnowledgeLocation,
    package: KnowledgePackage,
    *,
    transport: OneLakeKnowledgeTransport,
    overwrite: bool = False,
) -> None:
    """Upload, verify, then publish canonical envelope bytes by server-side rename."""

    if not isinstance(location, OneLakeKnowledgeLocation):
        raise ValueError("location must be a OneLakeKnowledgeLocation")
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be boolean")
    data = _envelope_bytes(package)
    if len(data) > MAX_PACKAGE_BYTES:
        raise KnowledgePersistenceError("knowledge package exceeds maximum byte size")

    target = location._target
    temporary = _temporary_path(target, "tmp")
    backup: str | None = None
    retain_backup = False
    had_original = False
    destination_etag: str | None = None
    try:
        try:
            transport.mkdirs(location._parent)
        except Exception:
            raise KnowledgePersistenceError(
                "OneLake package parent creation failed"
            ) from None

        if overwrite:
            existing = _safe_stat(transport, target)
            if existing is not None:
                had_original = True
                original, original_stat = _read_stable(transport, target)
                if not original_stat.etag:
                    raise KnowledgePersistenceError(
                        "OneLake destination did not return a usable ETag"
                    )
                destination_etag = original_stat.etag
                backup = _temporary_path(target, "backup")
                backup_stat = _upload_and_verify(transport, backup, original)
            else:
                backup_stat = None

        temporary_stat = _upload_and_verify(transport, temporary, data)
        _parse_envelope(data)

        if overwrite:
            try:
                transport.rename_overwrite(
                    temporary,
                    target,
                    source_etag=temporary_stat.etag,
                    destination_etag=destination_etag,
                )
                temporary = ""
            except ConcurrentWriteError:
                retain_backup = backup is not None
                raise KnowledgePersistenceError(
                    "OneLake package publication rejected a concurrent change"
                ) from None
            except Exception:
                if backup is not None:
                    try:
                        failed_target = _safe_stat(transport, target)
                        transport.rename_overwrite(
                            backup,
                            target,
                            source_etag=(
                                backup_stat.etag
                                if backup_stat is not None
                                else None
                            ),
                            destination_etag=(
                                failed_target.etag
                                if failed_target is not None
                                else None
                            ),
                        )
                        backup = None
                    except Exception:
                        retain_backup = True
                        raise PersistenceIntegrityError(
                            "OneLake publication failed and restore also failed"
                        ) from None
                raise KnowledgePersistenceError(
                    "OneLake package publication failed; original restored"
                    if had_original
                    else "OneLake package publication failed"
                ) from None
        else:
            try:
                transport.rename_no_clobber(
                    temporary,
                    target,
                    source_etag=temporary_stat.etag,
                )
                temporary = ""
            except AtomicRenameUnsupported:
                raise KnowledgePersistenceError(
                    "atomic no-clobber publication is unsupported"
                ) from None
            except FileExistsError:
                raise FileExistsError from None
            except Exception:
                raise KnowledgePersistenceError(
                    "OneLake package publication failed"
                ) from None
    finally:
        _delete_quietly(transport, temporary or None)
        if not retain_backup:
            _delete_quietly(transport, backup)


def load_onelake_knowledge_package(
    location: OneLakeKnowledgeLocation,
    *,
    transport: OneLakeKnowledgeTransport,
    bindings: Mapping[str, SourceBinding],
) -> BoundKnowledgePackage:
    """Bounded-read and validate package integrity before explicit rebinding."""

    if not isinstance(location, OneLakeKnowledgeLocation):
        raise ValueError("location must be a OneLakeKnowledgeLocation")
    data, _ = _read_stable(transport, location._target)
    package = _parse_envelope(data)
    return _bind_package(package, bindings)
