"""Microsoft Fabric Lakehouse inputs for parent-side source discovery."""

from __future__ import annotations

import csv
import datetime as dt
import decimal
import json
import re
from collections import deque
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


class LakehouseDiscoveryError(RuntimeError):
    """Raised when a Lakehouse scope cannot produce a complete catalog."""


_HOST_QUERY_TRANSPORT: Callable[..., dict[str, Any]] | None = None
_MAX_QUERY_ROWS = 10_000
_MAX_QUERY_CHARS = 100_000
_MAX_QUERY_RESULT_BYTES = 5 * 1024 * 1024
_SAFE_ALIAS = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UNSAFE_QUERY = re.compile(
    r"\b(?:"
    r"pragma|attach|detach|copy|install|load|export|import|call|"
    r"duckdb_[A-Za-z0-9_]*|glob|query_table|sniff_csv|"
    r"getenv|current_setting|[A-Za-z0-9_]*(?:secret|credential|token)[A-Za-z0-9_]*|"
    r"read_[A-Za-z0-9_]*|[A-Za-z0-9_]+_scan"
    r")\s*(?:\(|\b)",
    re.IGNORECASE,
)


def _configure_host_query_transport(
    transport: Callable[..., dict[str, Any]] | None,
) -> None:
    """Configure the worker-only transport used by ``LakehouseSource.query``."""

    global _HOST_QUERY_TRANSPORT
    _HOST_QUERY_TRANSPORT = transport


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

    def list_sources(self, *, kind: str | None = None) -> tuple[dict[str, Any], ...]:
        """Return catalog entries, optionally filtered by source kind."""

        catalog = self.resolve().catalog or ()
        normalized_kind = str(kind).strip().lower() if kind is not None else None
        return tuple(
            deepcopy(entry)
            for entry in catalog
            if normalized_kind is None
            or str(entry.get("kind", "")).lower() == normalized_kind
        )

    def find_sources(
        self,
        query: str,
        *,
        kind: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Find catalog entries by name, path, column, or data type."""

        normalized_query = str(query).strip().lower()
        if not normalized_query:
            raise ValueError("LakehouseSource find_sources query must be non-empty.")
        matches = []
        for entry in self.list_sources(kind=kind):
            searchable = " ".join(
                [
                    str(entry.get("name", "")),
                    str(entry.get("path", "")),
                    str(entry.get("columns", "")),
                ]
            ).lower()
            if normalized_query in searchable:
                matches.append(entry)
        return tuple(matches)

    def query(
        self,
        sql: str,
        *,
        sources: Mapping[str, str],
        max_rows: int = 1_000,
    ) -> dict[str, Any]:
        """Run bounded read-only SQL against named entries in this catalog.

        In an isolated RLM worker the query is transparently delegated to the
        trusted parent process, so Fabric credentials never enter generated
        code. Direct callers execute in the current process.
        """

        resolved = self.resolve()
        if _HOST_QUERY_TRANSPORT is not None:
            return _HOST_QUERY_TRANSPORT(
                root=resolved.root,
                catalog=list(resolved.catalog or ()),
                sql=sql,
                sources=dict(sources),
                max_rows=max_rows,
            )
        return execute_lakehouse_query(
            resolved,
            sql=sql,
            sources=sources,
            max_rows=max_rows,
        )

    def __rlm_describe__(self) -> str:
        state = (
            f"resolved metadata catalog with {len(self.catalog or ())} "
            f"source{'s' if len(self.catalog or ()) != 1 else ''}"
            if self.is_resolved
            else "unresolved source (resolved automatically before worker startup)"
        )
        return (
            f"LakehouseSource: {state}. Catalog entries are dictionaries with "
            "kind, name, path, and columns. Use .list_sources(kind=...) or "
            ".find_sources(query, kind=...) to choose relevant sources. Use "
            ".query(sql, sources={alias: catalog_name}) to analyze them through "
            "the parent process. Do not call notebookutils from the worker; the "
            "catalog is already resolved and credentials remain in the parent."
        )

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


def _quote_identifier(value: str) -> str:
    if not _SAFE_ALIAS.fullmatch(value):
        raise ValueError(
            f"Invalid source alias {value!r}; use letters, numbers, and underscores."
        )
    return f'"{value}"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _validate_catalog_query(sql: str) -> str:
    normalized = str(sql).strip()
    if (
        not normalized
        or len(normalized) > _MAX_QUERY_CHARS
        or any(marker in normalized for marker in ("--", "/*", "*/"))
        or not re.match(r"^(?:SELECT|WITH)\b", normalized, re.IGNORECASE)
    ):
        raise ValueError("LakehouseSource.query requires a read-only catalog query.")
    if _UNSAFE_QUERY.search(normalized):
        raise ValueError("LakehouseSource.query requires a read-only catalog query.")
    return normalized


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def execute_lakehouse_query(
    source: LakehouseSource,
    *,
    sql: str,
    sources: Mapping[str, str],
    max_rows: int = 1_000,
) -> dict[str, Any]:
    """Execute a catalog-bounded query in the trusted calling process."""

    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError("LakehouseSource.query max_rows must be a positive integer.")
    if max_rows > _MAX_QUERY_ROWS:
        raise ValueError(
            f"LakehouseSource.query max_rows must be at most {_MAX_QUERY_ROWS}."
        )
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("LakehouseSource.query requires at least one named source.")

    normalized_sql = _validate_catalog_query(sql)
    resolved = source.resolve()
    catalog = {str(entry["name"]): entry for entry in resolved.catalog or ()}
    selected: list[tuple[str, dict[str, Any]]] = []
    for alias, catalog_name in sources.items():
        alias_text = str(alias)
        _quote_identifier(alias_text)
        name_text = str(catalog_name)
        if name_text not in catalog:
            raise ValueError(
                f"Source {name_text!r} is not in this LakehouseSource catalog."
            )
        selected.append((alias_text, catalog[name_text]))

    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "LakehouseSource.query requires the analytics extra: "
            "pip install 'fabric-rlm[analytics]'."
        ) from exc

    con = duckdb.connect()
    token: str | None = None
    try:
        statements = con.extract_statements(normalized_sql)
        if (
            len(statements) != 1
            or str(statements[0].type).rsplit(".", 1)[-1] != "SELECT"
        ):
            raise ValueError("LakehouseSource.query requires a read-only catalog query.")

        kinds = {str(entry.get("kind", "")).lower() for _, entry in selected}
        remote = any("://" in str(entry["path"]) for _, entry in selected)
        if "delta" in kinds:
            con.sql("INSTALL delta; LOAD delta;")
        if remote:
            con.sql("INSTALL azure; LOAD azure;")
            token = _storage_token()
            escaped_token = token.replace("'", "''")
            con.execute(
                "CREATE SECRET onelake_tok "
                "(TYPE azure, PROVIDER access_token, "
                f"ACCESS_TOKEN '{escaped_token}', ACCOUNT_NAME 'onelake')"
            )

        for alias, entry in selected:
            kind = str(entry.get("kind", "")).lower()
            path = _quote_literal(str(entry["path"]))
            if kind == "delta":
                relation = f"delta_scan({path})"
            elif kind == "csv":
                relation = f"read_csv_auto({path}, header=true)"
            elif kind == "parquet":
                relation = f"read_parquet({path})"
            else:
                raise ValueError(
                    f"LakehouseSource.query does not support source kind {kind!r}."
                )
            con.execute(
                f"CREATE TEMP VIEW {_quote_identifier(alias)} AS "
                f"SELECT * FROM {relation}"
            )

        cursor = con.execute(
            f"SELECT * FROM ({normalized_sql}) AS __fabric_rlm_query "
            f"LIMIT {max_rows + 1}"
        )
        columns = [description[0] for description in cursor.description or ()]
        fetched = cursor.fetchall()
        result = {
            "columns": columns,
            "rows": [
                [_json_value(value) for value in row]
                for row in fetched[:max_rows]
            ],
            "truncated": len(fetched) > max_rows,
        }
        if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > _MAX_QUERY_RESULT_BYTES:
            raise ValueError(
                "LakehouseSource.query result exceeds the 5 MiB transfer limit. "
                "Aggregate further or select fewer columns."
            )
        return result
    except Exception as exc:
        message = str(exc)
        if token:
            message = message.replace(token, "[REDACTED]")
        if message != str(exc):
            raise RuntimeError(message) from None
        raise
    finally:
        con.close()


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
        delta_schema = DeltaTable(path, storage_options=options).schema()
        convert = getattr(delta_schema, "to_pyarrow", None)
        if convert is None:
            convert = delta_schema.to_arrow
        schema = convert()
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
