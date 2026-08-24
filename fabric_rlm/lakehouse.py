"""Microsoft Fabric Lakehouse inputs for parent-side source discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


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

    def __repr__(self) -> str:
        state = f"{len(self.catalog)} sources" if self.is_resolved else "auto"
        return f"LakehouseSource(root={self.root!r}, catalog={state})"
