"""Metadata-only knowledge adapters for Delta directories and Lakehouse handles.

Fabric exposes Lakehouse Tables and Files through distinct OneLake ABFSS
locations, and Tables are stored as Delta tables:
https://learn.microsoft.com/fabric/onelake/onelake-azure-databricks
https://learn.microsoft.com/fabric/onelake/onelake-open-access-quickstart
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
from typing import cast

from fabric_rlm.artifacts import File
from fabric_rlm.knowledge import (
    SourceProfile,
    SourceRole,
    _domain_fingerprint,
)
from fabric_rlm.knowledge_sources import (
    ProfileLimits,
    SourceAdapterRegistry,
    _is_safe_field,
    _sensitive_columns,
)
from fabric_rlm.lakehouse import LakehouseSource


_DELTA_MARKER = re.compile(
    r"^[0-9]{1,20}(?:\.json|\.checkpoint\.parquet)$"
)
_MAX_DELTA_LOG_ENTRIES_INSPECTED = 256
_LAKEHOUSE_ROLES = frozenset(
    {"numeric_evidence", "lookup", "context_only", "excluded"}
)
_TABULAR_ROLES = frozenset(
    {"numeric_evidence", "lookup", "context_only", "template", "excluded"}
)


def _path_descriptor(value: object) -> Path | None:
    if isinstance(value, File):
        return Path(value.path)
    if isinstance(value, (str, Path)):
        return Path(value)
    return None


def _has_delta_marker(path: Path) -> bool:
    """Recognize a Delta directory without reading transaction-log contents."""

    log = path / "_delta_log"
    try:
        if not path.is_dir() or not log.is_dir():
            return False
        checkpoint = log / "_last_checkpoint"
        if checkpoint.is_file() and not checkpoint.is_symlink():
            return True
        with os.scandir(log) as entries:
            for index, entry in enumerate(entries):
                if index >= _MAX_DELTA_LOG_ENTRIES_INSPECTED:
                    return False
                if (
                    entry.is_file(follow_symlinks=False)
                    and _DELTA_MARKER.fullmatch(entry.name)
                ):
                    return True
    except OSError:
        return False
    return False


def _portable_locator(domain: str, source_id: str, binding: object) -> str:
    fingerprint = _domain_fingerprint(
        f"{domain}-locator-v1",
        {"source_alias": source_id, "binding": binding},
    )
    return f"{domain}/v1/{fingerprint}"


def _primitive_type(type_name: str) -> str:
    normalized = type_name.casefold()
    if re.fullmatch(r"u?int(?:8|16|32|64)", normalized) or normalized in {
        "byte",
        "short",
        "integer",
        "int",
        "long",
        "tinyint",
        "smallint",
        "bigint",
        "utinyint",
        "usmallint",
        "uinteger",
        "ubigint",
        "unsigned byte",
        "unsigned short",
        "unsigned integer",
        "unsigned long",
    }:
        return "integer"
    if (
        re.fullmatch(r"float(?:16|32|64)", normalized)
        or normalized in {"float", "double", "real"}
        or normalized.startswith("decimal")
    ):
        return "number"
    if normalized in {"boolean", "bool"}:
        return "boolean"
    if normalized == "null":
        return "null"
    if normalized in {"array", "list"}:
        return "array"
    if normalized in {"map", "struct", "object"}:
        return "object"
    return "string"


def _normalize_delta_type(
    value: object,
    *,
    limits: ProfileLimits,
    depth: int,
    field_counter: list[int],
) -> tuple[str, object]:
    if depth > limits.max_nesting_depth:
        raise ValueError("Delta schema exceeds max_nesting_depth")
    if isinstance(value, str):
        return _primitive_type(value), value[:128]
    if not isinstance(value, Mapping):
        raise ValueError("Delta schema contains an unsupported type descriptor")

    type_name = str(value.get("type", "")).casefold()
    if type_name == "struct":
        fields = value.get("fields")
        if not isinstance(fields, list):
            raise ValueError("Delta struct schema must contain fields")
        normalized_fields: dict[str, object] = {}
        for field in fields:
            if not isinstance(field, Mapping):
                raise ValueError("Delta schema field is malformed")
            name = field.get("name")
            if not _is_safe_field(name):
                raise ValueError("Delta schema contains an unsafe field name")
            field_counter[0] += 1
            if field_counter[0] > limits.max_fields:
                raise ValueError("Delta schema exceeds max_fields")
            field_type, delta_type = _normalize_delta_type(
                field.get("type"),
                limits=limits,
                depth=depth + 1,
                field_counter=field_counter,
            )
            normalized_fields[cast(str, name)] = {
                "type": field_type,
                "delta_type": delta_type,
                "nullable": bool(field.get("nullable", True)),
            }
        return "object", {
            "type": "struct",
            "fields": dict(sorted(normalized_fields.items())),
        }
    if type_name == "array":
        element_type, delta_element = _normalize_delta_type(
            value.get("elementType"),
            limits=limits,
            depth=depth + 1,
            field_counter=field_counter,
        )
        return "array", {
            "type": "array",
            "element_type": element_type,
            "element_delta_type": delta_element,
            "contains_null": bool(value.get("containsNull", True)),
        }
    if type_name == "map":
        key_type, delta_key = _normalize_delta_type(
            value.get("keyType"),
            limits=limits,
            depth=depth + 1,
            field_counter=field_counter,
        )
        value_type, delta_value = _normalize_delta_type(
            value.get("valueType"),
            limits=limits,
            depth=depth + 1,
            field_counter=field_counter,
        )
        return "object", {
            "type": "map",
            "key_type": key_type,
            "key_delta_type": delta_key,
            "value_type": value_type,
            "value_delta_type": delta_value,
            "value_contains_null": bool(value.get("valueContainsNull", True)),
        }
    raise ValueError("Delta schema contains an unsupported nested type")


def _delta_schema(value: object, limits: ProfileLimits) -> dict[str, object]:
    to_json = getattr(value, "to_json", None)
    if not callable(to_json):
        raise ValueError("Delta schema does not expose metadata JSON")
    try:
        payload = json.loads(to_json())
    except (TypeError, ValueError) as error:
        raise ValueError("Delta schema metadata is malformed") from error
    field_counter = [0]
    field_type, normalized = _normalize_delta_type(
        payload,
        limits=limits,
        depth=1,
        field_counter=field_counter,
    )
    if field_type != "object" or not isinstance(normalized, Mapping):
        raise ValueError("Delta table schema must be a struct")
    fields = normalized.get("fields")
    if not isinstance(fields, Mapping):
        raise ValueError("Delta table schema must contain fields")
    return dict(fields)


def _delta_state(table: object) -> tuple[int, str, object, int]:
    try:
        version = int(table.version())
        metadata = table.metadata()
        identity = str(metadata.id)
        schema = table.schema()
        partition_count = len(tuple(metadata.partition_columns or ()))
    except Exception as error:
        raise ValueError("Delta transaction-log metadata could not be read") from error
    if not identity:
        raise ValueError("Delta transaction-log metadata has no table identity")
    return version, identity, schema, partition_count


class DeltaDirectoryAdapter:
    """Profile a local or mounted Delta directory from transaction metadata only.

    ``DeltaTable(..., without_files=True)`` intentionally avoids loading the
    active data-file set; only version, table metadata, and schema are used.
    """

    family = "delta"
    allowed_roles = _TABULAR_ROLES
    default_role: SourceRole = "numeric_evidence"

    def matches(self, value: object) -> bool:
        path = _path_descriptor(value)
        return path is not None and _has_delta_marker(path.expanduser())

    def profile(
        self,
        source_id: str,
        value: object,
        role: SourceRole,
        limits: ProfileLimits,
    ) -> SourceProfile:
        path = _path_descriptor(value)
        if path is None:
            raise TypeError("Delta sources must be str, Path, or File descriptors")
        path = path.expanduser()
        if not _has_delta_marker(path):
            raise ValueError(f"source alias {source_id} is not a recognized Delta table")

        try:
            from deltalake import DeltaTable
        except ImportError as error:
            raise ValueError(
                "Install fabric-rlm[analytics] to provide deltalake for "
                "Delta metadata profiling"
            ) from error

        try:
            first = DeltaTable(str(path), without_files=True)
            version, identity, raw_schema, partition_count = _delta_state(first)
            schema = _delta_schema(raw_schema, limits)
            second = DeltaTable(str(path), without_files=True)
            verified_version, verified_identity, _, _ = _delta_state(second)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(
                f"Delta metadata could not be opened for source alias {source_id}"
            ) from error

        if (verified_version, verified_identity) != (version, identity):
            raise ValueError(
                f"source alias {source_id} changed during profiling"
            )

        identity_fingerprint = _domain_fingerprint(
            "delta-table-identity-v1", identity
        )
        snapshot = _domain_fingerprint(
            "delta-table-snapshot-v1",
            {
                "table_identity_fingerprint": identity_fingerprint,
                "committed_version": version,
            },
        )
        return SourceProfile(
            source_id=source_id,
            family=self.family,
            locator=_portable_locator(
                "delta", source_id, identity_fingerprint
            ),
            snapshot_fingerprint=snapshot,
            schema_fingerprint=_domain_fingerprint(
                "delta-table-schema-v1", schema
            ),
            schema=schema,
            diagnostics={
                "format_code": "delta",
                "snapshot_exact": True,
                "committed_version": version,
                "table_identity_fingerprint": identity_fingerprint,
                "fields_inspected": len(schema),
                "partition_column_count": partition_count,
            },
            sensitive_columns=_sensitive_columns(tuple(schema)),
            role=role,
        )


def _catalog_column(column: object) -> tuple[str, str]:
    if isinstance(column, Mapping):
        name = column.get("name")
        type_name = column.get("type", column.get("data_type", "UNKNOWN"))
    elif (
        isinstance(column, Sequence)
        and not isinstance(column, (str, bytes))
        and len(column) >= 2
    ):
        name, type_name = column[0], column[1]
    else:
        raise ValueError("Lakehouse catalog column metadata is malformed")
    if not _is_safe_field(name):
        raise ValueError("Lakehouse catalog contains an unsafe column name")
    normalized_type = str(type_name).strip()
    if not normalized_type:
        normalized_type = "UNKNOWN"
    return cast(str, name), normalized_type[:128]


def _catalog_entry(
    entry: Mapping[str, object],
    *,
    limits: ProfileLimits,
    field_counter: list[int],
) -> tuple[str, dict[str, object], dict[str, object], bool]:
    name = entry.get("name")
    kind = str(entry.get("kind", "")).strip().casefold()
    if not _is_safe_field(name) or not kind:
        raise ValueError("Lakehouse catalog contains an unsafe source name or kind")
    columns_value = entry.get("columns", ())
    if not isinstance(columns_value, Sequence) or isinstance(
        columns_value, (str, bytes)
    ):
        raise ValueError("Lakehouse catalog columns must be a sequence")
    columns: dict[str, object] = {}
    for column in columns_value:
        column_name, type_name = _catalog_column(column)
        field_counter[0] += 1
        if field_counter[0] > limits.max_fields:
            raise ValueError("Lakehouse catalog schema exceeds max_fields")
        if column_name in columns:
            raise ValueError("Lakehouse catalog contains duplicate column names")
        columns[column_name] = {
            "type": _primitive_type(type_name),
            "lakehouse_type": type_name,
        }

    schema_entry = {
        "kind": kind[:64],
        "columns": dict(sorted(columns.items())),
    }
    snapshot_entry: dict[str, object] = {
        "name": name,
        "kind": kind[:64],
        "schema": schema_entry,
    }
    exact = False
    if kind == "delta":
        version = entry.get("version", entry.get("committed_version"))
        identity = entry.get(
            "table_identity",
            entry.get("table_id", entry.get("id")),
        )
        if type(version) is int and isinstance(identity, str) and identity:
            snapshot_entry["committed_version"] = version
            snapshot_entry["table_identity_fingerprint"] = _domain_fingerprint(
                "lakehouse-delta-table-identity-v1", identity
            )
            exact = True
    return cast(str, name), schema_entry, snapshot_entry, exact


class LakehouseSourceAdapter:
    """Profile a parent-resolved Fabric Lakehouse catalog without querying data."""

    family = "lakehouse"
    allowed_roles = _LAKEHOUSE_ROLES
    default_role: SourceRole = "numeric_evidence"

    def matches(self, value: object) -> bool:
        return isinstance(value, LakehouseSource)

    def profile(
        self,
        source_id: str,
        value: object,
        role: SourceRole,
        limits: ProfileLimits,
    ) -> SourceProfile:
        if not isinstance(value, LakehouseSource):
            raise TypeError("Lakehouse sources must be LakehouseSource handles")
        resolved = value.resolve()
        if not isinstance(resolved, LakehouseSource) or not resolved.is_resolved:
            raise ValueError("LakehouseSource.resolve() did not return a catalog")
        catalog = tuple(resolved.catalog or ())
        if len(catalog) > limits.max_records:
            raise ValueError("Lakehouse catalog exceeds max_records")

        root_identity_fingerprint = _domain_fingerprint(
            "lakehouse-root-identity-v1", resolved.root
        )
        schema: dict[str, object] = {}
        snapshot_entries: list[dict[str, object]] = []
        exact_flags: list[bool] = []
        field_counter = [0]
        for entry in sorted(
            catalog,
            key=lambda item: (
                str(item.get("name", "")).casefold(),
                str(item.get("name", "")),
            ),
        ):
            if not isinstance(entry, Mapping):
                raise ValueError("Lakehouse catalog entry is malformed")
            name, schema_entry, snapshot_entry, exact = _catalog_entry(
                entry,
                limits=limits,
                field_counter=field_counter,
            )
            if name in schema:
                raise ValueError("Lakehouse catalog contains duplicate source names")
            schema[name] = schema_entry
            snapshot_entries.append(snapshot_entry)
            exact_flags.append(exact)

        snapshot_exact = all(exact_flags)
        snapshot = {
            "root_identity_fingerprint": root_identity_fingerprint,
            "entries": snapshot_entries,
        }
        column_names = tuple(
            column
            for source_schema in schema.values()
            for column in cast(Mapping[str, object], source_schema)["columns"]
        )
        return SourceProfile(
            source_id=source_id,
            family=self.family,
            locator=_portable_locator(
                "lakehouse", source_id, root_identity_fingerprint
            ),
            snapshot_fingerprint=_domain_fingerprint(
                "lakehouse-catalog-snapshot-v1", snapshot
            ),
            schema_fingerprint=_domain_fingerprint(
                "lakehouse-catalog-schema-v1", schema
            ),
            schema=schema,
            diagnostics={
                "format_code": "lakehouse_catalog",
                "snapshot_exact": snapshot_exact,
                "root_identity_fingerprint": root_identity_fingerprint,
                "catalog_entry_count": len(catalog),
                "delta_entry_count": sum(
                    str(entry.get("kind", "")).casefold() == "delta"
                    for entry in catalog
                ),
                "inexact_entry_count": exact_flags.count(False),
                "fields_inspected": field_counter[0],
            },
            sensitive_columns=_sensitive_columns(column_names),
            role=role,
        )


def fabric_source_registry() -> SourceAdapterRegistry:
    """Return an opt-in registry with all Fabric and local adapters."""

    from fabric_rlm.knowledge_semantic_model import semantic_model_adapter

    defaults = SourceAdapterRegistry.default().adapters
    return SourceAdapterRegistry(
        (
            semantic_model_adapter(),
            LakehouseSourceAdapter(),
            DeltaDirectoryAdapter(),
            *defaults,
        )
    )


__all__ = [
    "DeltaDirectoryAdapter",
    "LakehouseSourceAdapter",
    "fabric_source_registry",
]
