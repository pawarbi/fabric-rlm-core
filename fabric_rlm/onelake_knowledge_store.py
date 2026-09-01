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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
import hmac
import sys
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import uuid

from fabric_rlm.artifacts import (
    _normalize_destination_root,
    _safe_relative_path,
)
from fabric_rlm.knowledge import KnowledgePackage, _logical_locator
from fabric_rlm.lakehouse import _storage_token
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


class _OneLakeHttpError(Exception):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"OneLake request failed with HTTP {status}")


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


class OneLakeRestTransport:
    """Parent-authorized OneLake transport using the ADLS Gen2 REST API.

    The caller supplies or inherits a Fabric storage-token provider. Tokens are
    used only in request headers and are never retained in package metadata.
    """

    _API_VERSION = "2023-08-03"
    _TIMEOUT_SECONDS = 60
    _UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024

    def __init__(
        self,
        *,
        token_provider: Callable[[], str] = _storage_token,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        if not callable(token_provider):
            raise TypeError("token_provider must be callable")
        if not callable(opener):
            raise TypeError("opener must be callable")
        self._token_provider = token_provider
        self._opener = opener

    def stat(self, path: str) -> OneLakeObjectStat | None:
        request = Request(
            self._url(path),
            headers=self._headers(),
            method="HEAD",
        )
        try:
            with self._open(request, expected_statuses={200}) as response:
                size_value = response.headers.get("Content-Length")
                etag = response.headers.get("ETag")
        except _OneLakeHttpError as error:
            if error.status == 404:
                return None
            raise KnowledgePersistenceError(
                "OneLake package metadata read failed"
            ) from None
        try:
            size = int(size_value)
        except (TypeError, ValueError):
            raise KnowledgePersistenceError(
                "OneLake package metadata is malformed"
            ) from None
        return OneLakeObjectStat(size=size, etag=etag)

    def read(self, path: str, max_bytes: int) -> bytes:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise TypeError("max_bytes must be an integer")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")
        request = Request(
            self._url(path),
            headers={
                **self._headers(),
                "Range": f"bytes=0-{max_bytes - 1}",
            },
            method="GET",
        )
        try:
            with self._open(request, expected_statuses={200, 206}) as response:
                data = response.read(max_bytes)
        except _OneLakeHttpError:
            raise KnowledgePersistenceError("OneLake package read failed") from None
        if not isinstance(data, bytes):
            raise KnowledgePersistenceError(
                "OneLake package read returned invalid bytes"
            )
        return data

    def mkdirs(self, path: str) -> None:
        workspace, parts = self._path_parts(path)
        try:
            files_index = parts.index("Files")
        except ValueError:
            raise ValueError(
                "OneLake directory must be beneath a canonical Files root"
            ) from None
        if files_index == 0:
            raise ValueError(
                "OneLake directory must include an item before Files"
            )
        for end in range(files_index + 2, len(parts) + 1):
            directory = self._abfss_path(workspace, parts[:end])
            request = Request(
                f"{self._url(directory)}?resource=directory",
                data=b"",
                headers={
                    **self._headers(),
                    "Content-Length": "0",
                    "If-None-Match": "*",
                },
                method="PUT",
            )
            try:
                with self._open(request, expected_statuses={201}):
                    pass
            except _OneLakeHttpError as error:
                if error.status == 409 and self.stat(directory) is not None:
                    continue
                raise KnowledgePersistenceError(
                    "OneLake package parent creation failed"
                ) from None

    def upload(self, path: str, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        url = self._url(path)
        create = Request(
            f"{url}?resource=file",
            data=b"",
            headers={
                **self._headers(),
                "Content-Length": "0",
                "If-None-Match": "*",
            },
            method="PUT",
        )
        try:
            with self._open(create, expected_statuses={201}):
                pass
            position = 0
            while position < len(data):
                chunk = data[
                    position : position + self._UPLOAD_CHUNK_BYTES
                ]
                append = Request(
                    f"{url}?action=append&position={position}",
                    data=chunk,
                    headers={
                        **self._headers(),
                        "Content-Length": str(len(chunk)),
                        "Content-Type": "application/octet-stream",
                    },
                    method="PATCH",
                )
                with self._open(append, expected_statuses={202}):
                    pass
                position += len(chunk)
            flush = Request(
                f"{url}?action=flush&position={len(data)}",
                data=b"",
                headers={
                    **self._headers(),
                    "Content-Length": "0",
                },
                method="PATCH",
            )
            with self._open(flush, expected_statuses={200}):
                pass
        except _OneLakeHttpError:
            raise KnowledgePersistenceError(
                "OneLake temporary upload failed"
            ) from None

    def rename_no_clobber(
        self,
        source: str,
        destination: str,
        *,
        source_etag: str | None,
    ) -> None:
        try:
            self._rename(
                source,
                destination,
                source_etag=source_etag,
                destination_etag=None,
            )
        except ConcurrentWriteError:
            raise FileExistsError from None

    def rename_overwrite(
        self,
        source: str,
        destination: str,
        *,
        source_etag: str | None,
        destination_etag: str | None,
    ) -> None:
        self._rename(
            source,
            destination,
            source_etag=source_etag,
            destination_etag=destination_etag,
        )

    def delete(self, path: str) -> None:
        request = Request(
            self._url(path),
            headers=self._headers(),
            method="DELETE",
        )
        try:
            with self._open(request, expected_statuses={200}):
                pass
        except _OneLakeHttpError as error:
            if error.status == 404:
                return
            raise KnowledgePersistenceError(
                "OneLake temporary cleanup failed"
            ) from None

    def _rename(
        self,
        source: str,
        destination: str,
        *,
        source_etag: str | None,
        destination_etag: str | None,
    ) -> None:
        if not source_etag:
            raise ValueError("source_etag is required for atomic rename")
        source_workspace, source_parts = self._path_parts(source)
        destination_workspace, _ = self._path_parts(destination)
        if source_workspace != destination_workspace:
            raise ValueError("OneLake rename must stay within one workspace")
        headers = {
            **self._headers(),
            "Content-Length": "0",
            "x-ms-rename-source": quote(
                f"/{source_workspace}/{'/'.join(source_parts)}",
                safe="/",
            ),
            "x-ms-source-if-match": source_etag,
        }
        if destination_etag is None:
            headers["If-None-Match"] = "*"
        else:
            headers["If-Match"] = destination_etag
        request = Request(
            self._url(destination),
            data=b"",
            headers=headers,
            method="PUT",
        )
        try:
            with self._open(request, expected_statuses={201}):
                pass
        except _OneLakeHttpError as error:
            if error.status in {409, 412}:
                raise ConcurrentWriteError from None
            raise KnowledgePersistenceError(
                "OneLake package rename failed"
            ) from None

    def _headers(self) -> dict[str, str]:
        try:
            token = self._token_provider()
        except Exception:
            raise KnowledgePersistenceError(
                "Fabric storage authorization failed"
            ) from None
        if (
            not isinstance(token, str)
            or not token
            or "\r" in token
            or "\n" in token
        ):
            raise KnowledgePersistenceError(
                "Fabric storage authorization returned an invalid token"
            )
        return {
            "Authorization": f"Bearer {token}",
            "x-ms-version": self._API_VERSION,
        }

    def _open(
        self,
        request: Request,
        *,
        expected_statuses: set[int],
    ) -> Any:
        try:
            response = self._opener(
                request,
                timeout=self._TIMEOUT_SECONDS,
            )
        except HTTPError as error:
            raise _OneLakeHttpError(error.code) from None
        except (OSError, URLError):
            raise KnowledgePersistenceError("OneLake request failed") from None
        status = getattr(response, "status", None)
        if status not in expected_statuses:
            close = getattr(response, "close", None)
            if callable(close):
                close()
            raise _OneLakeHttpError(int(status or 0))
        return response

    @staticmethod
    def _path_parts(path: str) -> tuple[str, tuple[str, ...]]:
        prefix = "abfss://"
        suffix = "@onelake.dfs.fabric.microsoft.com/"
        remainder = path[len(prefix) :] if path.startswith(prefix) else ""
        workspace, separator, object_path = remainder.partition(suffix)
        parts = tuple(object_path.split("/")) if object_path else ()
        if (
            not separator
            or not workspace
            or not parts
            or "\\" in path
            or any(ord(character) < 32 for character in path)
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("OneLake path must be a canonical ABFSS path")
        return workspace, parts

    @classmethod
    def _url(cls, path: str) -> str:
        workspace, parts = cls._path_parts(path)
        return (
            "https://onelake.dfs.fabric.microsoft.com/"
            f"{quote(workspace, safe='')}/"
            f"{quote('/'.join(parts), safe='/')}"
        )

    @staticmethod
    def _abfss_path(workspace: str, parts: tuple[str, ...]) -> str:
        return (
            f"abfss://{workspace}@onelake.dfs.fabric.microsoft.com/"
            f"{'/'.join(parts)}"
        )


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


def _delete_safely(
    transport: OneLakeKnowledgeTransport,
    path: str | None,
) -> bool:
    if path is None:
        return True
    try:
        transport.delete(path)
    except Exception:
        return False
    return True


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
        cleanup_failed = not _delete_safely(transport, temporary or None)
        if not retain_backup:
            cleanup_failed |= not _delete_safely(transport, backup)
        if cleanup_failed:
            active_error = sys.exc_info()[1]
            if active_error is not None:
                active_error.add_note("OneLake temporary cleanup failed")
            else:
                raise KnowledgePersistenceError(
                    "OneLake temporary cleanup failed"
                )


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


def read_onelake_knowledge_package(
    location: OneLakeKnowledgeLocation,
    *,
    transport: OneLakeKnowledgeTransport,
) -> KnowledgePackage:
    """Read and validate a OneLake package without attaching runtime bindings."""

    if not isinstance(location, OneLakeKnowledgeLocation):
        raise ValueError("location must be a OneLakeKnowledgeLocation")
    data, _ = _read_stable(transport, location._target)
    return _parse_envelope(data)
