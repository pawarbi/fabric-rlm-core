"""Microsoft Fabric Lakehouse inputs for parent-side source discovery."""

from __future__ import annotations

import csv
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class LakehouseDiscoveryError(RuntimeError):
    """Raised when a Lakehouse scope cannot produce a complete catalog."""


def _split_lakehouse_scope(path: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    for segment, kind in (("Tables", "tables"), ("Files", "files")):
        marker = f"/{segment}"
        index = path.find(marker)
        if index < 0:
            continue
        end = index + len(marker)
        if end < len(path) and path[end] != "/":
            continue
        root = path[:index]
        scope = path[index + 1 :]
        if kind == "tables":
            return root, (scope,), ()
        return root, (), (scope,)
    return path, ("Tables",), ()


def _normalize_scopes(
    value: str | Sequence[str] | None,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if value is None:
        return default
    values = (value,) if isinstance(value, str) else tuple(value)
    normalized = tuple(str(item).strip().strip("/") for item in values)
    if any(not item for item in normalized):
        raise ValueError("LakehouseSource scopes must be non-empty paths.")
    if any(
        "\\" in item
        or ":" in item
        or any(part in {"", ".", ".."} for part in item.split("/"))
        for item in normalized
    ):
        raise ValueError(
            "LakehouseSource scopes must be safe relative paths without "
            "backslashes or parent-directory segments."
        )
    return normalized


def _normalize_catalog(
    catalog: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...] | None:
    if catalog is None:
        return None
    normalized: list[dict[str, Any]] = []
    for entry in catalog:
        item = dict(entry)
        if not all(str(item.get(key, "")).strip() for key in ("kind", "name", "path")):
            raise ValueError(
                "Each LakehouseSource catalog entry requires kind, name, and path."
            )
        normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True)
class LakehouseSource:
    """A Lakehouse scope that resolves to a metadata-only source catalog.

    ``root`` accepts a OneLake ABFSS Lakehouse root using names or GUIDs.
    ``tables`` and ``files`` are paths relative to that root. When ``catalog``
    is omitted, RLM resolves it in the parent process before starting the
    isolated worker. Supplying ``catalog`` bypasses discovery.
    """

    root: str
    tables: tuple[str, ...] = ("Tables",)
    files: tuple[str, ...] = ()
    catalog: tuple[dict[str, Any], ...] | None = field(default=None, repr=False)
    max_sources: int = 200

    def __init__(
        self,
        root: str,
        *,
        tables: str | Sequence[str] | None = None,
        files: str | Sequence[str] | None = None,
        catalog: Sequence[Mapping[str, Any]] | None = None,
        max_sources: int = 200,
    ) -> None:
        supplied_path = str(root).strip().rstrip("/")
        if not supplied_path:
            raise ValueError("LakehouseSource requires a non-empty root.")
        normalized_root, inferred_tables, inferred_files = _split_lakehouse_scope(
            supplied_path
        )
        if isinstance(max_sources, bool) or not isinstance(max_sources, int):
            raise TypeError("LakehouseSource max_sources must be an int.")
        if max_sources <= 0:
            raise ValueError("LakehouseSource max_sources must be greater than zero.")

        object.__setattr__(self, "root", normalized_root)
        object.__setattr__(
            self,
            "tables",
            _normalize_scopes(tables, default=inferred_tables),
        )
        object.__setattr__(
            self,
            "files",
            _normalize_scopes(files, default=inferred_files),
        )
        object.__setattr__(self, "catalog", _normalize_catalog(catalog))
        object.__setattr__(self, "max_sources", max_sources)

    @property
    def is_resolved(self) -> bool:
        """Whether this source already carries a caller- or parent-built catalog."""

        return self.catalog is not None

    def __frozen__(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "tables": list(self.tables),
            "files": list(self.files),
            "catalog": [dict(item) for item in self.catalog or ()],
        }

    def resolve(self) -> "LakehouseSource":
        """Build the catalog in the current process unless one was supplied."""

        if self.is_resolved:
            return self
        return build_lakehouse_catalog(self)

    def __repr__(self) -> str:
        state = f"{len(self.catalog)} sources" if self.is_resolved else "auto"
        return f"LakehouseSource(root={self.root!r}, catalog={state})"


def _get_fs() -> Any:
    try:
        from notebookutils import fs
    except ImportError:
        try:
            from notebookutils import mssparkutils
        except ImportError as exc:
            raise LakehouseDiscoveryError(
                "Automatic Lakehouse discovery requires the Microsoft Fabric "
                "notebookutils runtime. Supply catalog= explicitly outside Fabric."
            ) from exc
        return mssparkutils.fs
    return fs


def _storage_token() -> str:
    try:
        from notebookutils import credentials

        return credentials.getToken("storage")
    except ImportError:
        try:
            from notebookutils import mssparkutils
        except ImportError as exc:
            raise LakehouseDiscoveryError(
                "Delta schema discovery requires Fabric storage credentials."
            ) from exc
        return mssparkutils.credentials.getToken("storage")


def _read_delta_columns(path: str) -> list[list[str]]:
    try:
        from deltalake import DeltaTable
    except ImportError as exc:
        raise LakehouseDiscoveryError(
            "Automatic Delta schema discovery requires the analytics extra: "
            "pip install 'fabric-rlm[analytics]'."
        ) from exc

    options = None
    if "://" in path:
        options = {
            "bearer_token": _storage_token(),
            "use_fabric_endpoint": "true",
        }
    try:
        schema = DeltaTable(path, storage_options=options).schema().to_arrow()
    except Exception as exc:
        raise LakehouseDiscoveryError(
            f"Delta metadata at {path!r} could not be read: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return [[field.name, str(field.type)] for field in schema]


def _list(fs: Any, path: str) -> list[Any]:
    try:
        return list(fs.ls(path))
    except Exception as exc:
        raise LakehouseDiscoveryError(
            f"Lakehouse scope {path!r} could not be listed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _delta_name(path: str) -> str:
    relative = path.split("/Tables/", 1)[-1]
    return relative.replace("/", ".")


def _file_name(path: str) -> str:
    relative = path.split("/Files/", 1)[-1]
    parent, separator, name = relative.rpartition("/")
    stem = name.rsplit(".", 1)[0] if "." in name else name
    relative_stem = f"{parent}/{stem}" if separator else stem
    return f"files.{relative_stem.replace('/', '.')}"


def _discover_delta_entries(fs: Any, scope: str) -> Iterator[dict[str, Any]]:
    pending = deque([(scope, 0)])
    while pending:
        current, depth = pending.popleft()
        children = _list(fs, current)
        if any(item.isDir and item.name.rstrip("/") == "_delta_log" for item in children):
            yield {
                "kind": "delta",
                "name": _delta_name(current),
                "path": current,
                "columns": _read_delta_columns(current),
            }
            continue
        if depth >= 3:
            if any(item.isDir and not item.name.startswith("_") for item in children):
                raise LakehouseDiscoveryError(
                    f"Delta discovery reached its maximum depth at {current!r}. "
                    "Narrow the Tables scope."
                )
            continue
        pending.extend(
            (item.path.rstrip("/"), depth + 1)
            for item in children
            if item.isDir and not item.name.startswith("_")
        )


def _discover_file_entries(fs: Any, scope: str) -> Iterator[dict[str, Any]]:
    pending = deque([(scope, 0)])
    while pending:
        current, depth = pending.popleft()
        for item in _list(fs, current):
            path = item.path.rstrip("/")
            if item.isDir:
                if depth >= 8:
                    raise LakehouseDiscoveryError(
                        f"Files discovery reached its maximum depth at {path!r}. "
                        "Narrow the Files scope."
                    )
                pending.append((path, depth + 1))
                continue
            suffix = path.rsplit(".", 1)[-1].lower() if "." in path else "file"
            columns: list[list[str]] = []
            if suffix == "csv":
                try:
                    header = fs.head(path, 64 * 1024).splitlines()[0]
                    columns = [[name, "UNKNOWN"] for name in next(csv.reader([header]))]
                except Exception as exc:
                    raise LakehouseDiscoveryError(
                        f"CSV header at {path!r} could not be read: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            yield {
                "kind": suffix,
                "name": _file_name(path),
                "path": path,
                "columns": columns,
            }


def build_lakehouse_catalog(source: LakehouseSource) -> LakehouseSource:
    """Resolve a LakehouseSource into a bounded metadata-only catalog."""

    if source.is_resolved:
        return source
    fs = _get_fs()
    entries: dict[str, dict[str, Any]] = {}
    for scope in source.tables:
        discovered = _discover_delta_entries(fs, f"{source.root}/{scope}")
        for entry in discovered:
            entries.setdefault(entry["path"], entry)
            if len(entries) > source.max_sources:
                break
        if len(entries) > source.max_sources:
            break
    for scope in source.files:
        if len(entries) > source.max_sources:
            break
        discovered = _discover_file_entries(fs, f"{source.root}/{scope}")
        for entry in discovered:
            entries.setdefault(entry["path"], entry)
            if len(entries) > source.max_sources:
                break
    if len(entries) > source.max_sources:
        raise LakehouseDiscoveryError(
            f"Lakehouse catalog contains more than {source.max_sources} sources, exceeding "
            f"max_sources={source.max_sources}. Narrow the Tables/Files scope "
            "or raise the limit explicitly."
        )
    catalog = sorted(entries.values(), key=lambda item: item["name"])
    if not catalog:
        raise LakehouseDiscoveryError(
            "Lakehouse discovery found no Delta tables or files in the supplied scopes."
        )
    return LakehouseSource(
        source.root,
        tables=source.tables,
        files=source.files,
        catalog=catalog,
        max_sources=source.max_sources,
    )


def resolve_lakehouse_inputs(value: Any) -> Any:
    """Resolve Lakehouse sources recursively in the calling process."""

    if isinstance(value, LakehouseSource):
        return value.resolve()
    if isinstance(value, dict):
        return {key: resolve_lakehouse_inputs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_lakehouse_inputs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(resolve_lakehouse_inputs(item) for item in value)
    return value
