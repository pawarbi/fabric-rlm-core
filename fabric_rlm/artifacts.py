"""Artifact helpers shared by the parent process and worker."""

from __future__ import annotations

import base64
import contextlib
import ctypes
import mimetypes
import os
import shutil
import stat
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from .lakehouse import LakehouseSource, _storage_token
from .semantic_model import SemanticModel


_HOST_FILE_TRANSPORT: Callable[..., dict[str, Any]] | None = None
_DEFAULT_MAX_FILE_BYTES = 256 * 1024 * 1024
_MFD_CLOEXEC = 0x0001
_MFD_ALLOW_SEALING = 0x0002
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_F_SEAL_SEAL = 0x0001
_F_SEAL_SHRINK = 0x0002
_F_SEAL_GROW = 0x0004
_F_SEAL_WRITE = 0x0008
_ONELAKE_API_VERSION = "2023-08-03"
_UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024
_UPLOAD_TIMEOUT_SECONDS = 60


def _configure_host_file_transport(
    transport: Callable[..., dict[str, Any]] | None,
) -> None:
    """Configure the worker-only transport used by ``FileDestination.publish``."""

    global _HOST_FILE_TRANSPORT
    _HOST_FILE_TRANSPORT = transport


@dataclass(frozen=True)
class File:
    """Lightweight file wrapper exposed inside the RLM worker namespace."""

    path: str

    def __init__(self, path: str | Path):
        object.__setattr__(self, "path", str(Path(path).expanduser()))

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def suffix(self) -> str:
        return Path(self.path).suffix

    def exists(self) -> bool:
        return Path(self.path).exists()

    def read_bytes(self) -> bytes:
        return Path(self.path).read_bytes()

    def read_text(self, encoding: str = "utf-8") -> str:
        return Path(self.path).read_text(encoding=encoding)

    def write_bytes(self, data: bytes) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def write_text(self, text: str, encoding: str = "utf-8") -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding)

    def as_data_uri(self, mime: str | None = None) -> str:
        mime = mime or mimetypes.guess_type(self.path)[0] or "application/octet-stream"
        encoded = base64.b64encode(self.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def toDict(self) -> dict[str, str]:
        return {"name": self.name, "path": self.path}

    def __frozen__(self) -> dict[str, str]:
        return self.toDict()

    def __repr__(self) -> str:
        return f"File({self.path!r})"

    def __fspath__(self) -> str:
        """Make File usable anywhere os.PathLike is accepted (open, os.path.exists,
        Path(...), shutil, etc.). Without this, models naturally write
        `os.path.exists(file)` and hit ``TypeError: stat: path should be string``.
        """
        return self.path


class LocalArtifactStore:
    """Run-scoped artifact store for local files or mounted Lakehouse paths."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def file(self, *parts: str) -> File:
        return File(self.path(*parts))

    def write_text(self, relative_path: str, text: str, encoding: str = "utf-8") -> File:
        file = self.file(relative_path)
        file.write_text(text, encoding=encoding)
        return file

    def write_bytes(self, relative_path: str, data: bytes) -> File:
        file = self.file(relative_path)
        file.write_bytes(data)
        return file

    def manifest_entry(self, file: File, **metadata: Any) -> dict[str, Any]:
        return {"path": file.path, "name": file.name, **metadata}


@dataclass(frozen=True)
class FileDestination:
    """A parent-published file destination with a private local staging area.

    Generated worker code writes ordinary files beneath ``staging_root`` and
    calls :meth:`publish`. The trusted parent then copies the staged file to
    ``root``. This keeps Fabric credentials out of the worker while supporting
    GUID- or name-based OneLake ``abfss://`` Files paths.
    """

    root: str
    max_bytes: int = _DEFAULT_MAX_FILE_BYTES
    staging_root: str = ""
    _owns_staging: bool = field(default=True, repr=False, compare=False)
    _staging_identity: tuple[int, int] = field(
        default=(0, 0),
        repr=False,
        compare=False,
    )
    _staged_paths: dict[str, str] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        root: str | Path,
        *,
        max_bytes: int = _DEFAULT_MAX_FILE_BYTES,
        _staging_root: str | Path | None = None,
        _owns_staging: bool = True,
        _staging_identity: tuple[int, int] | None = None,
    ) -> None:
        normalized_root = _normalize_destination_root(root)
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise TypeError("FileDestination max_bytes must be an int.")
        if max_bytes <= 0:
            raise ValueError("FileDestination max_bytes must be greater than zero.")
        staging_root = (
            Path(_staging_root).expanduser().resolve()
            if _staging_root is not None
            else Path(tempfile.mkdtemp(prefix="fabric-rlm-files-")).resolve()
        )
        staging_root.mkdir(parents=True, exist_ok=True)
        staging_stat = staging_root.lstat()
        staging_identity = _staging_identity or (
            staging_stat.st_dev,
            staging_stat.st_ino,
        )
        object.__setattr__(self, "root", normalized_root)
        object.__setattr__(self, "max_bytes", max_bytes)
        object.__setattr__(self, "staging_root", str(staging_root))
        object.__setattr__(self, "_owns_staging", _owns_staging)
        object.__setattr__(self, "_staging_identity", staging_identity)
        object.__setattr__(self, "_staged_paths", {})

    def stage(self, relative_path: str) -> File:
        """Return a local file path confined to this destination's staging area."""

        normalized = _safe_relative_path(relative_path)
        path = Path(self.staging_root) / f"{uuid.uuid4().hex}-{normalized.name}"
        self._staged_paths[str(path.resolve())] = normalized.as_posix()
        return File(path)

    def publish(
        self,
        file: File | str | Path,
        *,
        relative_path: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Publish a staged file through the trusted parent and return its manifest."""

        local_path = file.path if isinstance(file, File) else str(file)
        source = _validated_staged_source(self, local_path)
        if relative_path is None:
            relative = self._staged_paths.get(str(source))
            if relative is None:
                raise PermissionError(
                    "Published files must be created with this "
                    "FileDestination.stage() or provide relative_path explicitly."
                )
        else:
            relative = _safe_relative_path(relative_path).as_posix()
        if not isinstance(overwrite, bool):
            raise TypeError("FileDestination overwrite must be a bool.")
        if _HOST_FILE_TRANSPORT is not None:
            return _HOST_FILE_TRANSPORT(
                root=self.root,
                staging_root=self.staging_root,
                staging_identity=list(self._staging_identity),
                max_bytes=self.max_bytes,
                local_path=str(source),
                relative_path=relative,
                overwrite=overwrite,
            )
        return publish_file(
            self,
            local_path=str(source),
            relative_path=relative,
            overwrite=overwrite,
        )

    def close(self) -> None:
        """Remove this parent-owned staging directory; safe to call repeatedly."""

        if self._owns_staging:
            staging_root = Path(self.staging_root)
            if staging_root.exists():
                shutil.rmtree(staging_root)
            self._staged_paths.clear()

    def __enter__(self) -> "FileDestination":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __frozen__(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "max_bytes": self.max_bytes,
            "staging_root": self.staging_root,
            "staging_identity": list(self._staging_identity),
        }

    def __rlm_describe__(self) -> str:
        return (
            "FileDestination: create a staged local file with "
            ".stage(relative_path), write and verify it with normal Python file "
            "APIs, then call .publish(file, overwrite=False). Publication runs "
            "in the trusted parent; do not call notebookutils or write directly "
            "to the ABFSS root from the worker."
        )


def _normalize_destination_root(root: str | Path) -> str:
    supplied = str(root)
    value = supplied[:-1] if supplied.endswith("/") else supplied
    parts = urlsplit(value)
    raw_path_parts = parts.path.split("/")
    if (
        not value
        or supplied != supplied.strip()
        or any(ord(char) < 32 for char in supplied)
        or "%" in supplied
        or parts.scheme.lower() != "abfss"
        or parts.hostname is None
        or parts.hostname.lower() != "onelake.dfs.fabric.microsoft.com"
        or not parts.username
        or parts.password is not None
        or parts.port is not None
        or parts.query
        or parts.fragment
        or "\\" in value
        or not parts.path.startswith("/")
        or any(part in {"", ".", ".."} for part in raw_path_parts[1:])
        or value.endswith("/")
        or "//" in parts.path
        or len(raw_path_parts) < 3
        or raw_path_parts[2].lower() != "files"
    ):
        raise ValueError(
            "FileDestination root must be a canonical OneLake abfss:// Files scope."
        )
    workspace, _, host = parts.netloc.rpartition("@")
    return f"abfss://{workspace}@{host.lower()}{parts.path}"


def _safe_relative_path(relative_path: str) -> PurePosixPath:
    supplied = str(relative_path)
    value = supplied
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if (
        not value
        or supplied != supplied.strip()
        or any(ord(char) < 32 for char in supplied)
        or "%" in supplied
        or "\\" in value
        or "://" in value
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ValueError(
            "FileDestination requires a safe relative path without a URI, "
            "backslashes, or parent-directory segments."
        )
    return path


def _validated_staged_source(
    destination: FileDestination,
    local_path: str | Path,
) -> Path:
    source = Path(local_path).expanduser()
    try:
        staging_path = Path(destination.staging_root)
        staging_lstat = staging_path.lstat()
        if (
            stat.S_ISLNK(staging_lstat.st_mode)
            or (staging_lstat.st_dev, staging_lstat.st_ino)
            != destination._staging_identity
        ):
            raise PermissionError(
                "FileDestination staging directory identity changed."
            )
        source_lstat = source.lstat()
        if not stat.S_ISREG(source_lstat.st_mode):
            raise PermissionError("Published artifacts must be regular files.")
        staging_root = staging_path.resolve(strict=True)
        resolved = source.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PermissionError(
            "Published files must exist within this FileDestination staging directory."
        ) from exc
    if resolved != source.absolute() or resolved.parent != staging_root:
        raise PermissionError(
            "Published files must exist directly within this "
            "FileDestination staging directory."
        )
    return resolved


def _notebook_fs() -> Any:
    try:
        from notebookutils import fs
    except ImportError as exc:
        raise RuntimeError(
            "Publishing to abfss:// requires the Microsoft Fabric "
            "notebookutils runtime."
        ) from exc
    return fs


def publish_file(
    destination: FileDestination,
    *,
    local_path: str,
    relative_path: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy one authorized staged file to its bounded destination root."""

    source = _validated_staged_source(destination, local_path)
    relative = _safe_relative_path(relative_path)
    with _sealed_staged_snapshot(destination, source) as (
        snapshot_descriptor,
        size,
    ):
        target = f"{destination.root}/{relative.as_posix()}"
        fs = _notebook_fs()
        parent = target.rsplit("/", 1)[0]
        temporary = f"{parent}/.fabric-rlm-{uuid.uuid4().hex}-{relative.name}"
        if not fs.mkdirs(parent):
            raise RuntimeError(f"Could not create destination directory: {parent}")
        try:
            _upload_snapshot_to_onelake(
                snapshot_descriptor,
                temporary,
                size,
            )
            if not fs.exists(temporary):
                raise RuntimeError(f"Uploaded file is not visible at: {temporary}")
            uploaded_size = _remote_file_size(fs, temporary)
            if uploaded_size != size:
                raise RuntimeError(
                    f"Uploaded file size mismatch for {temporary}: "
                    f"expected {size}, got {uploaded_size}."
                )
            if not fs.mv(
                temporary,
                target,
                create_path=True,
                overwrite=overwrite,
            ):
                if fs.exists(target) and not overwrite:
                    raise FileExistsError(
                        f"Destination file already exists: {target}"
                    )
                raise RuntimeError(f"Could not publish staged file to: {target}")
            if not fs.exists(target):
                raise RuntimeError(f"Published file is not visible at: {target}")
        finally:
            active_error = sys.exc_info()[1]
            cleanup_failed = fs.exists(temporary) and not fs.rm(
                temporary,
                recurse=False,
            )
            if cleanup_failed:
                message = f"Could not remove temporary file: {temporary}"
                if active_error is not None:
                    active_error.add_note(message)
                else:
                    raise RuntimeError(message)

    return {
        "path": target,
        "name": relative.name,
        "size": size,
    }


@contextlib.contextmanager
def _sealed_staged_snapshot(
    destination: FileDestination,
    source: Path,
) -> Any:
    """Copy an authorized staged file into an immutable Linux memory file."""

    if (
        os.name != "posix"
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
    ):
        raise RuntimeError(
            "Secure FileDestination publication requires the Linux Fabric runtime."
        )

    import fcntl

    directory_descriptor: int | None = None
    source_descriptor: int | None = None
    snapshot_descriptor: int | None = None
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory_descriptor = os.open(destination.staging_root, directory_flags)
        directory_stat = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or (directory_stat.st_dev, directory_stat.st_ino)
            != destination._staging_identity
        ):
            raise PermissionError(
                "FileDestination staging directory identity changed."
            )

        before = os.stat(
            source.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PermissionError("Published artifacts must be regular files.")

        source_flags = os.O_RDONLY | os.O_NOFOLLOW
        source_descriptor = os.open(
            source.name,
            source_flags,
            dir_fd=directory_descriptor,
        )
        opened = os.fstat(source_descriptor)
        after = os.stat(
            source.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise PermissionError("Published artifacts must be stable regular files.")

        snapshot_descriptor = _create_sealable_memfd("fabric-rlm-publish")
        size = 0
        while chunk := os.read(source_descriptor, 1024 * 1024):
            size += len(chunk)
            if size > destination.max_bytes:
                raise ValueError(
                    f"Artifact size exceeds FileDestination "
                    f"max_bytes={destination.max_bytes}."
                )
            view = memoryview(chunk)
            while view:
                written = os.write(snapshot_descriptor, view)
                view = view[written:]

        final = os.fstat(source_descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(opened, key) != getattr(final, key) for key in stable_fields):
            raise PermissionError(
                "Staged artifact changed while it was being published."
            )

        _seal_memfd(snapshot_descriptor, fcntl)
        snapshot_stat = os.fstat(snapshot_descriptor)
        if snapshot_stat.st_size != size:
            raise RuntimeError("Immutable artifact snapshot size is inconsistent.")
        yield snapshot_descriptor, size
    finally:
        for descriptor in (
            snapshot_descriptor,
            source_descriptor,
            directory_descriptor,
        ):
            if descriptor is not None:
                os.close(descriptor)


def _create_sealable_memfd(name: str) -> int:
    """Create a sealable Linux memory file across Python runtime builds."""

    flags = _MFD_CLOEXEC | _MFD_ALLOW_SEALING
    python_memfd_create = getattr(os, "memfd_create", None)
    if python_memfd_create is not None:
        return python_memfd_create(name, flags=flags)

    try:
        libc_memfd_create = ctypes.CDLL(None, use_errno=True).memfd_create
    except AttributeError as exc:
        raise RuntimeError(
            "Secure FileDestination publication requires Linux memfd support."
        ) from exc

    libc_memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    libc_memfd_create.restype = ctypes.c_int
    descriptor = libc_memfd_create(name.encode("ascii"), flags)
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return descriptor


def _seal_memfd(descriptor: int, fcntl_module: Any) -> None:
    """Seal a Linux memory file even when Python omits the Linux constants."""

    seals = _F_SEAL_SEAL | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE
    fcntl_module.fcntl(descriptor, _F_ADD_SEALS, seals)
    applied = fcntl_module.fcntl(descriptor, _F_GET_SEALS)
    if applied & seals != seals:
        raise RuntimeError("Immutable artifact snapshot could not be sealed.")


def _upload_snapshot_to_onelake(
    descriptor: int,
    target: str,
    size: int,
) -> None:
    """Stream a sealed snapshot to a new OneLake file with the parent token."""

    url = _onelake_dfs_url(target)
    headers = {
        "Authorization": f"Bearer {_storage_token()}",
        "x-ms-version": _ONELAKE_API_VERSION,
    }
    _send_onelake_request(
        Request(
            f"{url}?resource=file",
            data=b"",
            headers={
                **headers,
                "Content-Length": "0",
                "If-None-Match": "*",
            },
            method="PUT",
        ),
        operation="create",
    )

    os.lseek(descriptor, 0, os.SEEK_SET)
    position = 0
    while position < size:
        chunk = os.read(descriptor, min(_UPLOAD_CHUNK_BYTES, size - position))
        if not chunk:
            raise RuntimeError("Immutable artifact snapshot ended unexpectedly.")
        _send_onelake_request(
            Request(
                f"{url}?action=append&position={position}",
                data=chunk,
                headers={
                    **headers,
                    "Content-Length": str(len(chunk)),
                    "Content-Type": "application/octet-stream",
                },
                method="PATCH",
            ),
            operation="append",
        )
        position += len(chunk)

    _send_onelake_request(
        Request(
            f"{url}?action=flush&position={size}",
            data=b"",
            headers={**headers, "Content-Length": "0"},
            method="PATCH",
        ),
        operation="flush",
    )


def _onelake_dfs_url(path: str) -> str:
    """Convert one canonical OneLake ABFSS path to its DFS API URL."""

    prefix = "abfss://"
    authority_suffix = "@onelake.dfs.fabric.microsoft.com/"
    workspace_and_path = path[len(prefix) :] if path.startswith(prefix) else ""
    workspace, separator, object_path = workspace_and_path.partition(
        authority_suffix
    )
    if (
        not separator
        or not workspace
        or not object_path
        or "\\" in path
        or any(ord(char) < 32 for char in path)
        or any(part in {"", ".", ".."} for part in object_path.split("/"))
    ):
        raise ValueError("OneLake upload target must be a canonical ABFSS path.")
    return (
        "https://onelake.dfs.fabric.microsoft.com/"
        f"{quote(workspace, safe='')}/{quote(object_path, safe='/')}"
    )


def _send_onelake_request(request: Request, *, operation: str) -> None:
    """Send one bounded OneLake upload request without exposing its token."""

    try:
        with urlopen(request, timeout=_UPLOAD_TIMEOUT_SECONDS) as response:
            status = response.status
    except HTTPError as exc:
        raise RuntimeError(
            f"OneLake upload failed during {operation}: HTTP {exc.code}."
        ) from None
    except (OSError, URLError) as exc:
        raise RuntimeError(
            f"OneLake upload failed during {operation}: "
            f"{type(exc).__name__}."
        ) from None
    if status < 200 or status >= 300:
        raise RuntimeError(
            f"OneLake upload failed during {operation}: HTTP {status}."
        )


def _remote_file_size(fs: Any, path: str) -> int:
    """Return the exact size of one uploaded temporary OneLake file."""

    entries = list(fs.ls(path))
    matching = [
        entry
        for entry in entries
        if str(getattr(entry, "path", "")).rstrip("/") == path.rstrip("/")
        or str(getattr(entry, "name", "")).rstrip("/") == path.rsplit("/", 1)[-1]
    ]
    if len(matching) != 1:
        raise RuntimeError(f"Could not verify uploaded file metadata: {path}")
    size = getattr(matching[0], "size", None)
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise RuntimeError(f"Uploaded file returned an invalid size: {path}")
    return size


def encode_for_worker(value: Any) -> Any:
    """Encode supported Python inputs into JSON-safe values for the worker."""

    if isinstance(value, File):
        return {"__fabric_rlm_file__": value.path}
    if isinstance(value, Path):
        return {"__fabric_rlm_path__": str(value)}
    if isinstance(value, FileDestination):
        return {
            "__fabric_rlm_file_destination__": {
                "root": value.root,
                "max_bytes": value.max_bytes,
                "staging_root": value.staging_root,
                "staging_identity": list(value._staging_identity),
            }
        }
    if isinstance(value, SemanticModel):
        # Only the coordinates cross the wire. Explicit notebook credentials
        # belong to the parent runtime; the worker uses SemPy's established
        # authentication path and does not revalidate the handle.
        return {"__fabric_rlm_semantic_model__": {
            "dataset": value.dataset,
            "workspace": value.workspace,
        }}
    if isinstance(value, LakehouseSource):
        return {
            "__fabric_rlm_lakehouse_source__": {
                "root": value.root,
                "tables": list(value.tables),
                "files": list(value.files),
                "catalog": [dict(item) for item in value.catalog or ()],
                "max_sources": value.max_sources,
            }
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [encode_for_worker(v) for v in value]
    if isinstance(value, list):
        return [encode_for_worker(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode_for_worker(v) for k, v in value.items()}
    raise TypeError(
        f"Unsupported input type for worker binding: {type(value).__name__}. "
        "Use primitives, dict/list containers, pathlib.Path, fabric_rlm.File, "
        "fabric_rlm.FileDestination, fabric_rlm.SemanticModel, or "
        "fabric_rlm.LakehouseSource."
    )


def decode_from_worker_wire(value: Any) -> Any:
    """Decode JSON-safe input values inside the worker process."""

    if isinstance(value, dict):
        if "__fabric_rlm_file__" in value:
            return File(value["__fabric_rlm_file__"])
        if "__fabric_rlm_path__" in value:
            return Path(value["__fabric_rlm_path__"])
        if "__fabric_rlm_file_destination__" in value:
            spec = value["__fabric_rlm_file_destination__"]
            return FileDestination(
                root=spec["root"],
                max_bytes=spec["max_bytes"],
                _staging_root=spec["staging_root"],
                _owns_staging=False,
                _staging_identity=tuple(spec["staging_identity"]),
            )
        if "__fabric_rlm_semantic_model__" in value:
            spec = value["__fabric_rlm_semantic_model__"]
            return SemanticModel(
                dataset=spec["dataset"],
                workspace=spec.get("workspace"),
                credential_provider=spec.get("credential_provider"),
                validate=False,
            )
        if "__fabric_rlm_lakehouse_source__" in value:
            spec = value["__fabric_rlm_lakehouse_source__"]
            return LakehouseSource(
                root=spec["root"],
                tables=spec.get("tables"),
                files=spec.get("files"),
                catalog=spec.get("catalog", []),
                max_sources=spec.get("max_sources", 200),
            )
        return {k: decode_from_worker_wire(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_from_worker_wire(v) for v in value]
    return value
