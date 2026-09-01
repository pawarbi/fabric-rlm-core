"""Internal, dependency-minimal profiling for explicitly named local sources.

Large-file snapshot fingerprints cover only bounded head/tail observations.
Consequently, a mutation confined to an unobserved middle region can remain
undetected; ``snapshot_exact`` discloses whether the complete file was hashed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import stat
from typing import Protocol, cast, runtime_checkable

from fabric_rlm.artifacts import File
from fabric_rlm.knowledge import (
    SourceProfile,
    SourceRole,
    _SOURCE_ROLES,
    _domain_fingerprint,
    _logical_identifier,
    canonical_json,
)
from fabric_rlm.knowledge_store import _is_forbidden_field


_ROLES = frozenset(_SOURCE_ROLES)
_TABULAR_ROLES = frozenset(_ROLES)
_OPAQUE_ROLES = frozenset({"context_only", "template", "excluded"})
_SAFE_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class ProfileLimits:
    max_input_bytes: int = 1024 * 1024
    max_records: int = 1000
    max_fields: int = 256
    max_nesting_depth: int = 8
    max_diagnostic_bytes: int = 64 * 1024
    read_chunk_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        for name in (
            "max_input_bytes",
            "max_records",
            "max_fields",
            "max_nesting_depth",
            "max_diagnostic_bytes",
            "read_chunk_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class _Snapshot:
    fingerprint: str
    content_digest: str
    size_bytes: int
    observed_bytes: int
    exact: bool


@runtime_checkable
class KnowledgeSourceAdapter(Protocol):
    family: str
    allowed_roles: frozenset[str]
    default_role: SourceRole

    def matches(self, path: Path) -> bool: ...

    def profile(
        self,
        source_id: str,
        path: Path,
        role: SourceRole,
        limits: ProfileLimits,
        snapshot: _Snapshot,
    ) -> SourceProfile: ...


class SourceAdapterRegistry:
    """Ordered, explicit registry; the first matching adapter owns the source."""

    def __init__(self, adapters: Sequence[KnowledgeSourceAdapter]):
        self._adapters = tuple(adapters)
        if not self._adapters:
            raise ValueError("registry must contain at least one adapter")
        for adapter in self._adapters:
            if not isinstance(adapter, KnowledgeSourceAdapter):
                raise TypeError("registry entries must implement KnowledgeSourceAdapter")

    @property
    def adapters(self) -> tuple[KnowledgeSourceAdapter, ...]:
        return self._adapters

    def resolve(self, path: Path) -> KnowledgeSourceAdapter:
        for adapter in self._adapters:
            if adapter.matches(path):
                return adapter
        raise ValueError("no source adapter matched the regular file")

    @classmethod
    def default(cls) -> SourceAdapterRegistry:
        return cls(
            (
                _CsvAdapter(),
                _JsonAdapter(),
                _JsonLinesAdapter(),
                _ParquetAdapter(),
                _OpaqueAdapter(),
            )
        )


def _read_prefix(path: Path, limit: int, chunk_bytes: int) -> bytes:
    remaining = limit
    chunks: list[bytes] = []
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(remaining, chunk_bytes))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    return b"".join(chunks)


def _read_region(
    path: Path, *, offset: int, length: int, chunk_bytes: int
) -> bytes:
    remaining = length
    chunks: list[bytes] = []
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            chunk = handle.read(min(remaining, chunk_bytes))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    return b"".join(chunks)


def _snapshot(path: Path, size_bytes: int, limits: ProfileLimits) -> _Snapshot:
    budget = min(size_bytes, limits.max_input_bytes)
    exact = size_bytes <= limits.max_input_bytes
    if exact:
        observed = _read_prefix(path, budget, limits.read_chunk_bytes)
    else:
        head_length = (budget + 1) // 2
        tail_length = budget - head_length
        head = _read_region(
            path, offset=0, length=head_length, chunk_bytes=limits.read_chunk_bytes
        )
        tail = _read_region(
            path,
            offset=size_bytes - tail_length,
            length=tail_length,
            chunk_bytes=limits.read_chunk_bytes,
        )
        observed = head + tail
    digest = hashlib.sha256(observed).hexdigest()
    identity = {
        "size_bytes": size_bytes,
        "content_digest": digest,
        "snapshot_exact": exact,
        "observed_bytes": len(observed),
        "observation_code": "full" if exact else "head_tail",
    }
    return _Snapshot(
        fingerprint=_domain_fingerprint("local-source-snapshot-v1", identity),
        content_digest=digest,
        size_bytes=size_bytes,
        observed_bytes=len(observed),
        exact=exact,
    )


def _base_diagnostics(
    format_code: str,
    snapshot: _Snapshot,
    *,
    input_truncated: bool,
) -> dict[str, object]:
    return {
        "format_code": format_code,
        "size_bytes": snapshot.size_bytes,
        "snapshot_exact": snapshot.exact,
        "snapshot_observed_bytes": snapshot.observed_bytes,
        "snapshot_digest_fingerprint": snapshot.content_digest,
        "snapshot_observation_code": "full" if snapshot.exact else "head_tail",
        "input_truncated": input_truncated,
    }


def _decode_utf8(data: bytes, *, format_name: str, truncated: bool) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        if truncated and error.end == len(data):
            try:
                return data[: error.start].decode("utf-8-sig")
            except UnicodeDecodeError:
                pass
        raise ValueError(f"{format_name} input must be strict UTF-8") from error


def _complete_lines(data: bytes, *, truncated: bool) -> bytes:
    if not truncated or data.endswith((b"\n", b"\r")):
        return data
    newline = max(data.rfind(b"\n"), data.rfind(b"\r"))
    return data[: newline + 1] if newline >= 0 else b""


def _is_safe_field(name: object) -> bool:
    return (
        isinstance(name, str)
        and bool(_SAFE_FIELD.fullmatch(name))
        and not _is_forbidden_field(name)
    )


def _scalar_type(value: object) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    raise ValueError("JSON values must be JSON-compatible")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _load_json(text: str) -> object:
    return json.loads(text, parse_constant=_reject_json_constant)


def _csv_type(value: str) -> str:
    if value == "":
        return "null"
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return "boolean"
    try:
        int(value)
    except ValueError:
        pass
    else:
        return "integer"
    try:
        float(value)
    except ValueError:
        return "string"
    return "number"


def _merge_type(current: str | None, incoming: str) -> str:
    if current is None or current == "null":
        return incoming
    if incoming == "null" or incoming == current:
        return current
    if {current, incoming} <= {"integer", "number"}:
        return "number"
    return "string"


def _schema_from_states(
    states: Mapping[str, tuple[str | None, bool]]
) -> dict[str, object]:
    return {
        name: {
            "type": inferred_type or "null",
            "nullable": nullable or inferred_type is None,
        }
        for name, (inferred_type, nullable) in sorted(states.items())
    }


def _tabular_profile(
    *,
    source_id: str,
    family: str,
    role: SourceRole,
    snapshot: _Snapshot,
    schema: dict[str, object],
    diagnostics: dict[str, object],
) -> SourceProfile:
    return SourceProfile(
        source_id=source_id,
        family=family,
        locator=f"local/{source_id}",
        snapshot_fingerprint=snapshot.fingerprint,
        schema_fingerprint=_domain_fingerprint(
            "local-source-schema-v1", {"family": family, "schema": schema}
        ),
        schema=schema,
        diagnostics=diagnostics,
        role=role,
    )


class _CsvAdapter:
    family = "csv"
    allowed_roles = _TABULAR_ROLES
    default_role: SourceRole = "numeric_evidence"

    def matches(self, path: Path) -> bool:
        return path.suffix.lower() == ".csv"

    def profile(
        self,
        source_id: str,
        path: Path,
        role: SourceRole,
        limits: ProfileLimits,
        snapshot: _Snapshot,
    ) -> SourceProfile:
        input_truncated = snapshot.size_bytes > limits.max_input_bytes
        data = _read_prefix(path, limits.max_input_bytes, limits.read_chunk_bytes)
        text = _decode_utf8(
            _complete_lines(data, truncated=input_truncated),
            format_name="CSV",
            truncated=input_truncated,
        )
        try:
            reader = csv.reader(io.StringIO(text, newline=""), delimiter=",", strict=True)
            header = next(reader)
        except (csv.Error, StopIteration) as error:
            raise ValueError("CSV header is required and must be well formed") from error
        if (
            not header
            or len(header) > limits.max_fields
            or len(set(header)) != len(header)
            or any(not _is_safe_field(name) for name in header)
        ):
            raise ValueError("CSV header contains duplicate, unsafe, or excess fields")

        states: dict[str, tuple[str | None, bool]] = {
            name: (None, False) for name in header
        }
        inspected = 0
        has_extra_record = False
        try:
            for row in reader:
                if not row and len(header) != 0:
                    continue
                if inspected >= limits.max_records:
                    has_extra_record = True
                    break
                if len(row) != len(header):
                    raise ValueError("CSV records must match the header field count")
                for name, value in zip(header, row):
                    current, nullable = states[name]
                    incoming = _csv_type(value)
                    states[name] = (
                        _merge_type(current, incoming),
                        nullable or incoming == "null",
                    )
                inspected += 1
        except csv.Error as error:
            raise ValueError("CSV input is malformed") from error

        schema = _schema_from_states(states)
        diagnostics = _base_diagnostics(
            "csv", snapshot, input_truncated=input_truncated
        )
        diagnostics.update(
            {
                "records_inspected": inspected,
                "records_truncated": input_truncated or has_extra_record,
                "fields_inspected": len(header),
            }
        )
        return _tabular_profile(
            source_id=source_id,
            family=self.family,
            role=role,
            snapshot=snapshot,
            schema=schema,
            diagnostics=diagnostics,
        )


def _validate_nesting(value: object, maximum: int, depth: int = 1) -> None:
    if depth > maximum:
        raise ValueError("JSON nesting exceeds max_nesting_depth")
    if isinstance(value, Mapping):
        for item in value.values():
            if isinstance(item, (Mapping, list)):
                _validate_nesting(item, maximum, depth + 1)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (Mapping, list)):
                _validate_nesting(item, maximum, depth + 1)


def _profile_json_records(
    records: Sequence[object],
    *,
    limits: ProfileLimits,
) -> tuple[dict[str, object], int, bool]:
    states: dict[str, tuple[str | None, bool]] = {}
    inspected = 0
    for record in records:
        if inspected >= limits.max_records:
            break
        if not isinstance(record, Mapping):
            raise ValueError("JSON tabular records must be objects")
        _validate_nesting(record, limits.max_nesting_depth)
        names = list(record)
        if any(not _is_safe_field(name) for name in names):
            raise ValueError("JSON object contains an unsafe field name")
        if len(set(states).union(names)) > limits.max_fields:
            raise ValueError("JSON object union exceeds max_fields")
        prior_names = set(states)
        for name in prior_names - set(names):
            current, _ = states[name]
            states[name] = (current, True)
        for name in names:
            incoming = _scalar_type(record[name])
            if name not in states:
                states[name] = (None, inspected > 0)
            current, nullable = states[name]
            states[name] = (
                _merge_type(current, incoming),
                nullable or incoming == "null",
            )
        inspected += 1
    return _schema_from_states(states), inspected, len(records) > inspected


class _JsonAdapter:
    family = "json"
    allowed_roles = _TABULAR_ROLES
    default_role: SourceRole = "numeric_evidence"

    def matches(self, path: Path) -> bool:
        return path.suffix.lower() == ".json"

    def profile(
        self,
        source_id: str,
        path: Path,
        role: SourceRole,
        limits: ProfileLimits,
        snapshot: _Snapshot,
    ) -> SourceProfile:
        if snapshot.size_bytes > limits.max_input_bytes:
            raise ValueError("JSON input exceeds max_input_bytes")
        data = _read_prefix(path, limits.max_input_bytes, limits.read_chunk_bytes)
        text = _decode_utf8(data, format_name="JSON", truncated=False)
        try:
            value = _load_json(text)
        except (json.JSONDecodeError, RecursionError, ValueError) as error:
            raise ValueError("JSON input is malformed") from error
        records = value if isinstance(value, list) else [value]
        schema, inspected, records_truncated = _profile_json_records(
            records, limits=limits
        )
        diagnostics = _base_diagnostics(
            "json", snapshot, input_truncated=False
        )
        diagnostics.update(
            {
                "records_inspected": inspected,
                "records_truncated": records_truncated,
                "fields_inspected": len(schema),
            }
        )
        return _tabular_profile(
            source_id=source_id,
            family=self.family,
            role=role,
            snapshot=snapshot,
            schema=schema,
            diagnostics=diagnostics,
        )


class _JsonLinesAdapter:
    family = "jsonl"
    allowed_roles = _TABULAR_ROLES
    default_role: SourceRole = "numeric_evidence"

    def matches(self, path: Path) -> bool:
        return path.suffix.lower() == ".jsonl"

    def profile(
        self,
        source_id: str,
        path: Path,
        role: SourceRole,
        limits: ProfileLimits,
        snapshot: _Snapshot,
    ) -> SourceProfile:
        input_truncated = snapshot.size_bytes > limits.max_input_bytes
        data = _complete_lines(
            _read_prefix(path, limits.max_input_bytes, limits.read_chunk_bytes),
            truncated=input_truncated,
        )
        text = _decode_utf8(data, format_name="JSONL", truncated=input_truncated)
        records: list[object] = []
        has_extra_record = False
        for line in text.splitlines():
            if not line.strip():
                continue
            if len(records) >= limits.max_records:
                has_extra_record = True
                break
            try:
                records.append(_load_json(line))
            except (json.JSONDecodeError, RecursionError, ValueError) as error:
                raise ValueError("JSONL input is malformed") from error
        schema, inspected, records_truncated = _profile_json_records(
            records, limits=limits
        )
        diagnostics = _base_diagnostics(
            "jsonl", snapshot, input_truncated=input_truncated
        )
        diagnostics.update(
            {
                "records_inspected": inspected,
                "records_truncated": (
                    input_truncated or has_extra_record or records_truncated
                ),
                "fields_inspected": len(schema),
            }
        )
        return _tabular_profile(
            source_id=source_id,
            family=self.family,
            role=role,
            snapshot=snapshot,
            schema=schema,
            diagnostics=diagnostics,
        )


def _parquet_type(type_name: str) -> tuple[str, str]:
    normalized = type_name.casefold()
    if normalized == "boolean":
        return "boolean", "boolean"
    if any(
        normalized.startswith(prefix)
        for prefix in (
            "tinyint",
            "smallint",
            "integer",
            "bigint",
            "utinyint",
            "usmallint",
            "uinteger",
            "ubigint",
            "hugeint",
        )
    ):
        return "integer", normalized
    if any(
        normalized.startswith(prefix)
        for prefix in ("decimal", "float", "double", "real")
    ):
        return "number", normalized.split("(", 1)[0]
    if normalized.endswith("[]"):
        return "array", "array"
    if normalized.startswith(("struct", "map", "union")):
        return "object", normalized.split("(", 1)[0]
    if normalized.startswith("timestamp"):
        return "string", "timestamp"
    if normalized.startswith(("date", "time", "interval")):
        return "string", normalized.split("(", 1)[0]
    if normalized.startswith(("blob", "bit", "varint")):
        return "string", "binary"
    return "string", normalized.split("(", 1)[0]


class _ParquetAdapter:
    family = "parquet"
    allowed_roles = _TABULAR_ROLES
    default_role: SourceRole = "numeric_evidence"

    def matches(self, path: Path) -> bool:
        return path.suffix.lower() == ".parquet"

    def profile(
        self,
        source_id: str,
        path: Path,
        role: SourceRole,
        limits: ProfileLimits,
        snapshot: _Snapshot,
    ) -> SourceProfile:
        try:
            import duckdb
        except ModuleNotFoundError as error:
            raise ValueError(
                "Parquet profiling requires fabric-rlm[analytics]"
            ) from error

        connection = duckdb.connect(database=":memory:")
        try:
            columns = connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(path)],
            ).fetchall()
            metadata = connection.execute(
                """
                SELECT num_rows, num_row_groups
                FROM parquet_file_metadata(?)
                """,
                [str(path)],
            ).fetchone()
        except Exception as error:
            raise ValueError(
                f"Parquet metadata is malformed for source alias {source_id}"
            ) from error
        finally:
            connection.close()

        if len(columns) > limits.max_fields:
            raise ValueError("Parquet schema exceeds max_fields")
        schema: dict[str, object] = {}
        for column_name, type_name, nullable, *_ in columns:
            if not _is_safe_field(column_name):
                raise ValueError("Parquet schema contains an unsafe field name")
            field_type, logical_type = _parquet_type(str(type_name))
            schema[str(column_name)] = {
                "type": field_type,
                "logical_type": logical_type,
                "nullable": str(nullable).casefold() == "yes",
            }

        diagnostics = _base_diagnostics(
            "parquet",
            snapshot,
            input_truncated=snapshot.size_bytes > limits.max_input_bytes,
        )
        diagnostics.update(
            {
                "fields_inspected": len(schema),
                "row_count": int(metadata[0]) if metadata else 0,
                "row_group_count": int(metadata[1]) if metadata else 0,
            }
        )
        return _tabular_profile(
            source_id=source_id,
            family=self.family,
            role=role,
            snapshot=snapshot,
            schema=schema,
            diagnostics=diagnostics,
        )


class _OpaqueAdapter:
    family = "opaque"
    allowed_roles = _OPAQUE_ROLES
    default_role: SourceRole = "context_only"

    def matches(self, path: Path) -> bool:
        return True

    def profile(
        self,
        source_id: str,
        path: Path,
        role: SourceRole,
        limits: ProfileLimits,
        snapshot: _Snapshot,
    ) -> SourceProfile:
        suffix = path.suffix.lower().lstrip(".")
        suffix_code = suffix if suffix and suffix.isalnum() else "none"
        diagnostics = _base_diagnostics(
            "opaque",
            snapshot,
            input_truncated=snapshot.size_bytes > limits.max_input_bytes,
        )
        diagnostics["suffix_code"] = suffix_code[:64]
        return SourceProfile(
            source_id=source_id,
            family=self.family,
            locator=f"local/{source_id}",
            snapshot_fingerprint=snapshot.fingerprint,
            schema_fingerprint=_domain_fingerprint(
                "local-source-schema-v1",
                {"family": self.family, "suffix_code": suffix_code},
            ),
            schema={},
            diagnostics=diagnostics,
            role=role,
        )


def _path_for_source(value: object) -> Path:
    if isinstance(value, File):
        return Path(value.path)
    if isinstance(value, (str, Path)):
        return Path(value)
    raise TypeError("local sources must be str, Path, or fabric_rlm.artifacts.File")


def _validated_roles(
    source_ids: set[str], roles: Mapping[str, object] | None
) -> dict[str, SourceRole]:
    if roles is None:
        return {}
    if not isinstance(roles, Mapping):
        raise TypeError("roles must be a mapping from source alias to role")
    unknown = set(roles) - source_ids
    if unknown:
        raise ValueError(
            f"roles contain unknown source alias: {sorted(unknown)[0]}"
        )
    validated: dict[str, SourceRole] = {}
    for source_id, role in roles.items():
        if not isinstance(role, str) or role not in _ROLES:
            raise ValueError(f"role for {source_id} is not supported")
        validated[source_id] = cast(SourceRole, role)
    return validated


def profile_sources(
    sources: Mapping[str, object],
    roles: Mapping[str, object] | None = None,
    limits: ProfileLimits | None = None,
    registry: SourceAdapterRegistry | None = None,
) -> tuple[SourceProfile, ...]:
    """Profile explicitly aliased local files without importing optional readers."""

    if not isinstance(sources, Mapping):
        raise TypeError("sources must be a mapping from source alias to local file")
    normalized_sources: list[tuple[str, object]] = []
    for source_id, source in sources.items():
        normalized_sources.append(
            (_logical_identifier(source_id, "source alias"), source)
        )
    source_ids = {source_id for source_id, _ in normalized_sources}
    validated_roles = _validated_roles(source_ids, roles)
    active_limits = limits or ProfileLimits()
    if not isinstance(active_limits, ProfileLimits):
        raise TypeError("limits must be ProfileLimits")
    active_registry = registry or SourceAdapterRegistry.default()
    if not isinstance(active_registry, SourceAdapterRegistry):
        raise TypeError("registry must be SourceAdapterRegistry")

    profiles: list[SourceProfile] = []
    for source_id, source in normalized_sources:
        path = _path_for_source(source).expanduser()
        try:
            file_stat = path.stat()
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"source alias {source_id} does not exist"
            ) from error
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"source alias {source_id} must reference a regular file")
        adapter = active_registry.resolve(path)
        role = validated_roles.get(source_id, adapter.default_role)
        if role not in adapter.allowed_roles:
            raise ValueError(
                f"role {role} is not supported by the {adapter.family} adapter"
            )
        snapshot = _snapshot(path, file_stat.st_size, active_limits)
        profile = adapter.profile(
            source_id, path, role, active_limits, snapshot
        )
        encoded_size = len(canonical_json(profile.to_dict()).encode("utf-8"))
        if encoded_size > active_limits.max_diagnostic_bytes:
            raise ValueError(
                "canonical SourceProfile exceeds max_diagnostic_bytes"
            )
        profiles.append(profile)
    return tuple(profiles)


__all__: list[str] = []
