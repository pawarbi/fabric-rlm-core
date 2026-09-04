"""Immutable contracts for portable, rebindable knowledge packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import Literal
from urllib.parse import unquote, urlsplit

from fabric_rlm.experimental.analysis_reproducibility import (
    canonical_json as _canonical_json,
    fingerprint as _canonical_fingerprint,
)


Status = Literal["candidate", "active", "stale", "quarantined", "retired"]
SourceRole = Literal[
    "numeric_evidence",
    "lookup",
    "context_only",
    "template",
    "excluded",
]
Cardinality = Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]
SubjectType = Literal[
    "source", "relationship", "operation", "package", "lesson", "evidence"
]
LessonKind = Literal[
    "semantic_fact",
    "time_semantics",
    "context_requirement",
    "valid_grain",
    "expensive_grain",
    "invalid_path",
    "preferred_strategy",
    "metric_equivalence",
    "metric_non_equivalence",
    "relationship_path",
    "query_behavior",
    "cross_source_mapping",
]
Confidence = Literal["high", "medium", "low"]

_STATUSES = {"candidate", "active", "stale", "quarantined", "retired"}
_SOURCE_ROLES = {
    "numeric_evidence",
    "lookup",
    "context_only",
    "template",
    "excluded",
}
_CARDINALITIES = {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}
_SUBJECT_TYPES = {
    "source", "relationship", "operation", "package", "lesson", "evidence"
}
_LESSON_KINDS = {
    "semantic_fact",
    "time_semantics",
    "context_requirement",
    "valid_grain",
    "expensive_grain",
    "invalid_path",
    "preferred_strategy",
    "metric_equivalence",
    "metric_non_equivalence",
    "relationship_path",
    "query_behavior",
    "cross_source_mapping",
}
_CONFIDENCES = {"high", "medium", "low"}
_EVIDENCE_TYPES = {"execution", "structural", "trajectory", "verification"}
_OBSERVATION_TYPES = {
    "query_execution",
    "run_outcome",
    "turn_error",
    "strategy_sequence",
    "source_declaration",
    "verification",
    "integrity_finding",
}
_EXECUTION_STATUSES = {"success", "failure", "timeout", "rejected", "not_applicable"}
_VERIFIER_STATUSES = {"passed", "failed", "none"}
_INTEGRITY_STATUSES = {"passed", "unresolved", "failed", "off"}
# Structured rules and observations hold names and codes, never prose: one
# line, bounded length, and the whole record bounded when serialized.
_MAX_NAME_LENGTH = 256
_MAX_STRUCTURED_DEPTH = 6
_MAX_STRUCTURED_BYTES = 8 * 1024
_SCALAR_TYPES = {"string", "integer", "number", "boolean", "null"}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_JWT_LIKE = re.compile(
    r"^eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$"
)
_MAX_REASON_CODE_LENGTH = 64
_WINDOWS_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")
_MISSING = object()


def canonical_json(value: object) -> str:
    """Serialize using the repository's canonical JSON convention."""

    return _canonical_json(value)


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _logical_identifier(value: object, field_name: str) -> str:
    normalized = _required_text(value, field_name)
    if not _IDENTIFIER.fullmatch(normalized) or ".." in normalized:
        raise ValueError(f"{field_name} must be a safe logical identifier")
    return normalized


def _reason_code(value: object) -> str:
    normalized = _logical_identifier(value, "reason_code")
    if len(normalized) > _MAX_REASON_CODE_LENGTH:
        raise ValueError(
            f"reason_code must be at most {_MAX_REASON_CODE_LENGTH} characters"
        )
    if normalized.lower().startswith("sk-proj-") or _JWT_LIKE.fullmatch(normalized):
        raise ValueError("reason_code must not contain a secret-like value")
    return normalized


def _logical_locator(value: object) -> str:
    locator = _required_text(value, "locator")
    if any(character.isspace() or ord(character) < 32 for character in locator):
        raise ValueError("locator must not contain whitespace or control characters")
    decoded_locator = locator
    for _ in range(len(locator) + 1):
        decoded = unquote(decoded_locator)
        if decoded == decoded_locator:
            break
        decoded_locator = decoded
    if (
        any(
            character.isspace() or ord(character) < 32
            for character in decoded_locator
        )
        or decoded_locator.startswith(("/", "\\"))
        or _WINDOWS_DRIVE_PREFIX.match(decoded_locator)
        or "\\" in decoded_locator
    ):
        raise ValueError("locator must be logical, not an absolute filesystem path")

    parsed = urlsplit(decoded_locator)
    if parsed.scheme.lower() == "file":
        raise ValueError("locator must not use the file URI scheme")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("locator must not contain URL user-info")
    if parsed.query or parsed.fragment:
        raise ValueError("locator must not contain a query or fragment")

    path_segments = parsed.path.split("/")
    if any(segment == ".." for segment in path_segments):
        raise ValueError("locator must not contain traversal segments")
    return locator


def _freeze(value: object, path: str = "value") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON values")
        return value + 0.0
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} must have string object keys")
        return MappingProxyType(
            {
                key: _freeze(value[key], f"{path}.{key}")
                for key in sorted(value)
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(f"{path} must be JSON-compatible")


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _text_tuple(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence of identifiers")
    normalized = tuple(
        _logical_identifier(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return tuple(sorted(normalized))


def _strict_payload(
    value: object,
    *,
    record_name: str,
    fields: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{record_name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{record_name} must have string keys")
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValueError(f"{record_name} contains unknown field: {unknown[0]}")
    return dict(value)


def _status(value: object, field_name: str = "status") -> str:
    if value not in _STATUSES:
        raise ValueError(f"{field_name} is not a supported status")
    return str(value)


def _is_scalar(value: object) -> bool:
    return value is None or type(value) in {str, bool, int, float}


def _validate_scalar(
    value: object,
    *,
    field_name: str,
    expected_type: str | None = None,
) -> object:
    if not _is_scalar(value):
        raise ValueError(f"{field_name} must be a scalar value")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if expected_type == "string" and type(value) is not str:
        raise ValueError(f"{field_name} must match parameter type string")
    if expected_type == "integer" and type(value) is not int:
        raise ValueError(f"{field_name} must match parameter type integer")
    if expected_type == "number" and type(value) not in {int, float}:
        raise ValueError(f"{field_name} must match parameter type number")
    if expected_type == "boolean" and type(value) is not bool:
        raise ValueError(f"{field_name} must match parameter type boolean")
    if expected_type == "null" and value is not None:
        raise ValueError(f"{field_name} must match parameter type null")
    return value + 0.0 if isinstance(value, float) else value


def _parameter_contract(
    schema: object,
    defaults: object,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if not isinstance(schema, Mapping):
        raise ValueError("parameter_schema must be an object")
    if not isinstance(defaults, Mapping):
        raise ValueError("parameter_defaults must be an object")

    normalized_schema: dict[str, object] = {}
    for raw_name, raw_descriptor in schema.items():
        name = _logical_identifier(raw_name, "parameter_schema key")
        descriptor = _strict_payload(
            raw_descriptor,
            record_name=f"parameter_schema.{name}",
            fields={"type", "enum"},
        )
        parameter_type = descriptor.get("type")
        if parameter_type not in _SCALAR_TYPES:
            raise ValueError(
                f"parameter_schema.{name}.type must be a supported scalar type"
            )
        normalized: dict[str, object] = {"type": parameter_type}
        if "enum" in descriptor:
            enum = descriptor["enum"]
            if not isinstance(enum, (list, tuple)) or not enum:
                raise ValueError(
                    f"parameter_schema.{name}.enum must be a non-empty sequence"
                )
            normalized_enum = tuple(
                _validate_scalar(
                    item,
                    field_name=f"parameter_schema.{name}.enum[{index}]",
                    expected_type=str(parameter_type),
                )
                for index, item in enumerate(enum)
            )
            if len(set(normalized_enum)) != len(normalized_enum):
                raise ValueError(
                    f"parameter_schema.{name}.enum must not contain duplicates"
                )
            normalized["enum"] = normalized_enum
        normalized_schema[name] = MappingProxyType(normalized)

    unknown_defaults = sorted(set(defaults) - set(normalized_schema))
    if unknown_defaults:
        raise ValueError(
            f"parameter_defaults contains unknown parameter: {unknown_defaults[0]}"
        )
    normalized_defaults: dict[str, object] = {}
    for raw_name, value in defaults.items():
        name = str(raw_name)
        descriptor = normalized_schema[name]
        normalized = _validate_scalar(
            value,
            field_name=f"parameter_defaults.{name}",
            expected_type=str(descriptor["type"]),
        )
        enum = descriptor.get("enum")
        if enum is not None and normalized not in enum:
            raise ValueError(f"parameter_defaults.{name} must be present in enum")
        normalized_defaults[name] = normalized

    return (
        MappingProxyType(dict(sorted(normalized_schema.items()))),
        MappingProxyType(dict(sorted(normalized_defaults.items()))),
    )


def _output_contract(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("output_schema must be a non-empty object")
    normalized: dict[str, object] = {}
    for raw_name, raw_descriptor in value.items():
        name = _logical_identifier(raw_name, "output_schema key")
        if isinstance(raw_descriptor, str):
            output_type = raw_descriptor
            descriptor: object = output_type
        elif isinstance(raw_descriptor, Mapping):
            payload = _strict_payload(
                raw_descriptor,
                record_name=f"output_schema.{name}",
                fields={"type", "nullable"},
            )
            output_type = payload.get("type")
            nullable = payload.get("nullable", False)
            if type(nullable) is not bool:
                raise ValueError(f"output_schema.{name}.nullable must be boolean")
            descriptor = MappingProxyType(
                {"type": output_type, "nullable": nullable}
            )
        else:
            raise ValueError(f"output_schema.{name} must declare an output type")
        if output_type not in _SCALAR_TYPES:
            raise ValueError(f"output_schema.{name} has unsupported output type")
        normalized[name] = descriptor
    return MappingProxyType(dict(sorted(normalized.items())))


def _positive_bound(value: object, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _domain_fingerprint(domain: str, value: object) -> str:
    return _canonical_fingerprint({"domain": domain, "value": value})


def _bounded_name(value: object, field_name: str) -> str:
    """A measure, column, table or concept name: one line, bounded, no secret shape."""

    text = _required_text(value, field_name)
    if len(text) > _MAX_NAME_LENGTH:
        raise ValueError(f"{field_name} must be at most {_MAX_NAME_LENGTH} characters")
    if any(ord(character) < 32 for character in text):
        raise ValueError(f"{field_name} must be a single line without control characters")
    if text.lower().startswith("sk-proj-") or _JWT_LIKE.fullmatch(text):
        raise ValueError(f"{field_name} must not contain a secret-like value")
    return text


def _structured_value(value: object, path: str, depth: int = 0) -> object:
    """Freeze a structured rule or observation, rejecting prose.

    Strings are names or codes: a single bounded line. Anything longer, or
    with a line break, is free text and does not belong in durable
    knowledge; the runtime renders structured rules into language later.
    """
    if depth > _MAX_STRUCTURED_DEPTH:
        raise ValueError(f"{path} is nested too deeply")
    if isinstance(value, str):
        return _bounded_name(value, path) if value.strip() else value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON values")
        return value + 0.0
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} must have string object keys")
        return MappingProxyType(
            {
                key: _structured_value(value[key], f"{path}.{key}", depth + 1)
                for key in sorted(value)
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _structured_value(item, f"{path}[{index}]", depth + 1)
            for index, item in enumerate(value)
        )
    raise ValueError(f"{path} must be JSON-compatible")


def _structured_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    frozen = _structured_value(value, field_name)
    encoded = len(_canonical_json(_thaw(frozen)).encode("utf-8"))
    if encoded > _MAX_STRUCTURED_BYTES:
        raise ValueError(f"{field_name} exceeds {_MAX_STRUCTURED_BYTES} bytes")
    return frozen  # type: ignore[return-value]


def _fingerprint_mapping(value: object, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    normalized: dict[str, str] = {}
    for raw_key in sorted(value, key=str):
        key = _logical_identifier(raw_key, f"{field_name} key")
        normalized[key] = _required_text(value[raw_key], f"{field_name}.{key}")
    return MappingProxyType(normalized)


def _code_choice(value: object, field_name: str, choices: set[str]) -> str:
    if value not in choices:
        raise ValueError(f"{field_name} is not supported")
    return str(value)


@dataclass(frozen=True)
class SourceProfile:
    source_id: str
    family: str
    locator: str
    snapshot_fingerprint: str
    schema_fingerprint: str
    schema: Mapping[str, object] = field(default_factory=dict)
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    sensitive_columns: tuple[str, ...] = ()
    role: SourceRole = "numeric_evidence"
    status: Status = "candidate"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_id", _logical_identifier(self.source_id, "source_id")
        )
        object.__setattr__(self, "family", _logical_identifier(self.family, "family"))
        object.__setattr__(self, "locator", _logical_locator(self.locator))
        for name in ("snapshot_fingerprint", "schema_fingerprint"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if not isinstance(self.schema, Mapping):
            raise ValueError("schema must be an object")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be an object")
        object.__setattr__(self, "schema", _freeze(self.schema, "schema"))
        object.__setattr__(
            self, "diagnostics", _freeze(self.diagnostics, "diagnostics")
        )
        object.__setattr__(
            self,
            "sensitive_columns",
            _text_tuple(self.sensitive_columns, "sensitive_columns"),
        )
        if self.role not in _SOURCE_ROLES:
            raise ValueError("role is not a supported source role")
        object.__setattr__(self, "status", _status(self.status))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "family": self.family,
            "locator": self.locator,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "schema_fingerprint": self.schema_fingerprint,
            "schema": _thaw(self.schema),
            "diagnostics": _thaw(self.diagnostics),
            "sensitive_columns": list(self.sensitive_columns),
            "role": self.role,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: object) -> SourceProfile:
        payload = _strict_payload(
            value,
            record_name="SourceProfile",
            fields={
                "source_id",
                "family",
                "locator",
                "snapshot_fingerprint",
                "schema_fingerprint",
                "schema",
                "diagnostics",
                "sensitive_columns",
                "role",
                "status",
            },
        )
        return cls(**payload)


@dataclass(frozen=True)
class Relationship:
    relationship_id: str
    left_source: str
    right_source: str
    key: str
    cardinality: Cardinality
    left_coverage: float
    left_key_unique: bool
    right_key_unique: bool
    max_right_rows_per_key: int
    status: Status = "candidate"
    reason_code: str | None = None

    def __post_init__(self) -> None:
        for name in ("relationship_id", "left_source", "right_source", "key"):
            object.__setattr__(
                self, name, _logical_identifier(getattr(self, name), name)
            )
        if self.cardinality not in _CARDINALITIES:
            raise ValueError("cardinality is not supported")
        if (
            type(self.left_coverage) not in {int, float}
            or not math.isfinite(self.left_coverage)
            or not 0 <= self.left_coverage <= 1
        ):
            raise ValueError("left_coverage must be finite and in [0, 1]")
        object.__setattr__(self, "left_coverage", float(self.left_coverage) + 0.0)
        for name in ("left_key_unique", "right_key_unique"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if (
            type(self.max_right_rows_per_key) is not int
            or self.max_right_rows_per_key < 0
        ):
            raise ValueError("max_right_rows_per_key must be a non-negative integer")
        object.__setattr__(self, "status", _status(self.status))
        if self.reason_code is not None:
            object.__setattr__(self, "reason_code", _reason_code(self.reason_code))

    def to_dict(self) -> dict[str, object]:
        return {
            "relationship_id": self.relationship_id,
            "left_source": self.left_source,
            "right_source": self.right_source,
            "key": self.key,
            "cardinality": self.cardinality,
            "left_coverage": self.left_coverage,
            "left_key_unique": self.left_key_unique,
            "right_key_unique": self.right_key_unique,
            "max_right_rows_per_key": self.max_right_rows_per_key,
            "status": self.status,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, value: object) -> Relationship:
        payload = _strict_payload(
            value,
            record_name="Relationship",
            fields={
                "relationship_id",
                "left_source",
                "right_source",
                "key",
                "cardinality",
                "left_coverage",
                "left_key_unique",
                "right_key_unique",
                "max_right_rows_per_key",
                "status",
                "reason_code",
            },
        )
        return cls(**payload)


@dataclass(frozen=True)
class RegisteredOperation:
    operation_id: str
    operation: str
    required_sources: tuple[str, ...]
    required_relationships: tuple[str, ...] = ()
    parameter_schema: Mapping[str, object] = field(default_factory=dict)
    parameter_defaults: Mapping[str, object] = field(default_factory=dict)
    output_schema: Mapping[str, object] = field(default_factory=dict)
    max_output_rows: int = 1_000
    max_output_columns: int = 100
    grain: str = ""
    host_implementation_id: str = ""
    operation_version: str | None = None
    status: Status = "candidate"
    reason_code: str | None = None

    def __post_init__(self) -> None:
        for name in ("operation_id", "operation"):
            object.__setattr__(
                self, name, _logical_identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "required_sources",
            _text_tuple(
                self.required_sources,
                "required_sources",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "required_relationships",
            _text_tuple(self.required_relationships, "required_relationships"),
        )
        schema, defaults = _parameter_contract(
            self.parameter_schema,
            self.parameter_defaults,
        )
        object.__setattr__(self, "parameter_schema", schema)
        object.__setattr__(self, "parameter_defaults", defaults)
        object.__setattr__(self, "output_schema", _output_contract(self.output_schema))
        object.__setattr__(
            self,
            "max_output_rows",
            _positive_bound(self.max_output_rows, "max_output_rows"),
        )
        object.__setattr__(
            self,
            "max_output_columns",
            _positive_bound(self.max_output_columns, "max_output_columns"),
        )
        if len(self.output_schema) > self.max_output_columns:
            raise ValueError("output_schema exceeds max_output_columns")
        object.__setattr__(self, "grain", _required_text(self.grain, "grain"))
        object.__setattr__(
            self,
            "host_implementation_id",
            _logical_identifier(
                self.host_implementation_id,
                "host_implementation_id",
            ),
        )
        if self.operation_version is not None:
            object.__setattr__(
                self,
                "operation_version",
                _logical_identifier(self.operation_version, "operation_version"),
            )
        object.__setattr__(self, "status", _status(self.status))
        if self.reason_code is not None:
            object.__setattr__(self, "reason_code", _reason_code(self.reason_code))

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "operation": self.operation,
            "required_sources": list(self.required_sources),
            "required_relationships": list(self.required_relationships),
            "parameter_schema": _thaw(self.parameter_schema),
            "parameter_defaults": _thaw(self.parameter_defaults),
            "output_schema": _thaw(self.output_schema),
            "max_output_rows": self.max_output_rows,
            "max_output_columns": self.max_output_columns,
            "grain": self.grain,
            "host_implementation_id": self.host_implementation_id,
            "operation_version": self.operation_version,
            "status": self.status,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, value: object) -> RegisteredOperation:
        payload = _strict_payload(
            value,
            record_name="RegisteredOperation",
            fields={
                "operation_id",
                "operation",
                "required_sources",
                "required_relationships",
                "parameter_schema",
                "parameter_defaults",
                "output_schema",
                "max_output_rows",
                "max_output_columns",
                "grain",
                "host_implementation_id",
                "operation_version",
                "status",
                "reason_code",
            },
        )
        return cls(**payload)


@dataclass(frozen=True)
class KnowledgeEvent:
    event_id: str
    event_type: str
    subject_type: SubjectType
    subject_id: str
    status: Status
    reason_code: str | None = None

    def __post_init__(self) -> None:
        for name in ("event_id", "event_type", "subject_id"):
            object.__setattr__(
                self, name, _logical_identifier(getattr(self, name), name)
            )
        if self.subject_type not in _SUBJECT_TYPES:
            raise ValueError("subject_type is not supported")
        object.__setattr__(self, "status", _status(self.status))
        if self.reason_code is not None:
            object.__setattr__(self, "reason_code", _reason_code(self.reason_code))

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "status": self.status,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, value: object) -> KnowledgeEvent:
        payload = _strict_payload(
            value,
            record_name="KnowledgeEvent",
            fields={
                "event_id",
                "event_type",
                "subject_type",
                "subject_id",
                "status",
                "reason_code",
            },
        )
        return cls(**payload)


@dataclass(frozen=True)
class EvidenceRecord:
    """One typed observation about how a source behaved.

    Evidence is captured from runtime telemetry (a query's grain, its
    estimated group count, whether it ran or was rejected) and from run
    outcomes (verification and analytical-integrity status), never from
    agent prose. ``observation`` holds names, codes and numbers only.
    """

    evidence_id: str
    evidence_type: str
    source_ids: tuple[str, ...]
    observation_type: str
    observation: Mapping[str, object] = field(default_factory=dict)
    operation_ids: tuple[str, ...] = ()
    source_fingerprints: Mapping[str, str] = field(default_factory=dict)
    execution_status: str = "success"
    verifier_status: str | None = None
    analytical_integrity_status: str | None = None
    run_fingerprint: str | None = None
    turn: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evidence_id", _logical_identifier(self.evidence_id, "evidence_id")
        )
        object.__setattr__(
            self,
            "evidence_type",
            _code_choice(self.evidence_type, "evidence_type", _EVIDENCE_TYPES),
        )
        object.__setattr__(
            self,
            "source_ids",
            _text_tuple(self.source_ids, "source_ids", allow_empty=False),
        )
        object.__setattr__(
            self, "operation_ids", _text_tuple(self.operation_ids, "operation_ids")
        )
        object.__setattr__(
            self,
            "observation_type",
            _code_choice(self.observation_type, "observation_type", _OBSERVATION_TYPES),
        )
        object.__setattr__(
            self, "observation", _structured_mapping(self.observation, "observation")
        )
        object.__setattr__(
            self,
            "source_fingerprints",
            _fingerprint_mapping(self.source_fingerprints, "source_fingerprints"),
        )
        object.__setattr__(
            self,
            "execution_status",
            _code_choice(self.execution_status, "execution_status", _EXECUTION_STATUSES),
        )
        if self.verifier_status is not None:
            object.__setattr__(
                self,
                "verifier_status",
                _code_choice(self.verifier_status, "verifier_status", _VERIFIER_STATUSES),
            )
        if self.analytical_integrity_status is not None:
            object.__setattr__(
                self,
                "analytical_integrity_status",
                _code_choice(
                    self.analytical_integrity_status,
                    "analytical_integrity_status",
                    _INTEGRITY_STATUSES,
                ),
            )
        if self.run_fingerprint is not None:
            object.__setattr__(
                self,
                "run_fingerprint",
                _logical_identifier(self.run_fingerprint, "run_fingerprint"),
            )
        if self.turn is not None and (type(self.turn) is not int or self.turn < 1):
            raise ValueError("turn must be a positive integer")

    @property
    def trusted(self) -> bool:
        """Whether a success lesson may rest on this record.

        A strategy is only proven by a run whose answer passed verification
        and the analytical-integrity screen; typed execution failures teach
        regardless, because a timeout is a fact about the source.
        """
        return (
            self.execution_status == "success"
            and self.verifier_status in {"passed", None}
            and self.analytical_integrity_status in {"passed", "off", None}
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "source_ids": list(self.source_ids),
            "operation_ids": list(self.operation_ids),
            "observation_type": self.observation_type,
            "observation": _thaw(self.observation),
            "source_fingerprints": dict(self.source_fingerprints),
            "execution_status": self.execution_status,
            "verifier_status": self.verifier_status,
            "analytical_integrity_status": self.analytical_integrity_status,
            "run_fingerprint": self.run_fingerprint,
            "turn": self.turn,
        }

    @classmethod
    def from_dict(cls, value: object) -> EvidenceRecord:
        payload = _strict_payload(
            value,
            record_name="EvidenceRecord",
            fields={
                "evidence_id",
                "evidence_type",
                "source_ids",
                "operation_ids",
                "observation_type",
                "observation",
                "source_fingerprints",
                "execution_status",
                "verifier_status",
                "analytical_integrity_status",
                "run_fingerprint",
                "turn",
            },
        )
        return cls(**payload)


@dataclass(frozen=True)
class LearnedLesson:
    """A durable, structured rule about a source, tied to its evidence.

    ``structured_rule`` is what future runs receive, rendered into language
    at retrieval time; ``evidence_ids`` say why it is believed;
    ``source_fingerprints`` say what it depends on, so a changed measure or
    schema can stale it without discarding the rest of the package.
    """

    lesson_id: str
    kind: str
    subject: str
    structured_rule: Mapping[str, object]
    evidence_ids: tuple[str, ...] = ()
    confidence: str = "low"
    status: Status = "candidate"
    source_dependencies: tuple[str, ...] = ()
    operation_dependencies: tuple[str, ...] = ()
    source_fingerprints: Mapping[str, str] = field(default_factory=dict)
    basis: tuple[str, ...] = ()
    reason_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "lesson_id", _logical_identifier(self.lesson_id, "lesson_id")
        )
        object.__setattr__(self, "kind", _code_choice(self.kind, "kind", _LESSON_KINDS))
        object.__setattr__(self, "subject", _bounded_name(self.subject, "subject"))
        object.__setattr__(
            self,
            "structured_rule",
            _structured_mapping(self.structured_rule, "structured_rule"),
        )
        if not self.structured_rule:
            raise ValueError("structured_rule must not be empty")
        object.__setattr__(
            self, "evidence_ids", _text_tuple(self.evidence_ids, "evidence_ids")
        )
        object.__setattr__(
            self,
            "confidence",
            _code_choice(self.confidence, "confidence", _CONFIDENCES),
        )
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(
            self,
            "source_dependencies",
            _text_tuple(self.source_dependencies, "source_dependencies", allow_empty=False),
        )
        object.__setattr__(
            self,
            "operation_dependencies",
            _text_tuple(self.operation_dependencies, "operation_dependencies"),
        )
        object.__setattr__(
            self,
            "source_fingerprints",
            _fingerprint_mapping(self.source_fingerprints, "source_fingerprints"),
        )
        object.__setattr__(self, "basis", _text_tuple(self.basis, "basis"))
        if self.reason_code is not None:
            object.__setattr__(self, "reason_code", _reason_code(self.reason_code))

    def to_dict(self) -> dict[str, object]:
        return {
            "lesson_id": self.lesson_id,
            "kind": self.kind,
            "subject": self.subject,
            "structured_rule": _thaw(self.structured_rule),
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
            "status": self.status,
            "source_dependencies": list(self.source_dependencies),
            "operation_dependencies": list(self.operation_dependencies),
            "source_fingerprints": dict(self.source_fingerprints),
            "basis": list(self.basis),
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, value: object) -> LearnedLesson:
        payload = _strict_payload(
            value,
            record_name="LearnedLesson",
            fields={
                "lesson_id",
                "kind",
                "subject",
                "structured_rule",
                "evidence_ids",
                "confidence",
                "status",
                "source_dependencies",
                "operation_dependencies",
                "source_fingerprints",
                "basis",
                "reason_code",
            },
        )
        return cls(**payload)


@dataclass(frozen=True)
class KnowledgePackage:
    package_id: str
    sources: tuple[SourceProfile, ...]
    relationships: tuple[Relationship, ...] = ()
    operations: tuple[RegisteredOperation, ...] = ()
    events: tuple[KnowledgeEvent, ...] = ()
    evidence: tuple[EvidenceRecord, ...] = ()
    lessons: tuple[LearnedLesson, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "package_id", _logical_identifier(self.package_id, "package_id")
        )
        for field_name, record_type in (
            ("sources", SourceProfile),
            ("relationships", Relationship),
            ("operations", RegisteredOperation),
            ("events", KnowledgeEvent),
            ("evidence", EvidenceRecord),
            ("lessons", LearnedLesson),
        ):
            values = getattr(self, field_name)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"{field_name} must be a sequence")
            normalized = tuple(values)
            if any(not isinstance(item, record_type) for item in normalized):
                raise ValueError(
                    f"{field_name} must contain {record_type.__name__} values"
                )
            if field_name not in {"events", "evidence"}:
                identifier = {
                    "sources": "source_id",
                    "relationships": "relationship_id",
                    "operations": "operation_id",
                    "lessons": "lesson_id",
                }[field_name]
                normalized = tuple(
                    sorted(normalized, key=lambda item: getattr(item, identifier))
                )
            object.__setattr__(self, field_name, normalized)
        if not self.sources:
            raise ValueError("sources must not be empty")

        source_ids = self._unique_ids(self.sources, "source_id")
        relationship_ids = self._unique_ids(
            self.relationships, "relationship_id"
        )
        operation_ids = self._unique_ids(self.operations, "operation_id")
        self._unique_ids(self.events, "event_id")
        evidence_ids = self._unique_ids(self.evidence, "evidence_id")
        lesson_ids = self._unique_ids(self.lessons, "lesson_id")

        for record in self.evidence:
            unknown_sources = sorted(set(record.source_ids) - source_ids)
            if unknown_sources:
                raise ValueError(
                    f"evidence references unknown source: {unknown_sources[0]}"
                )
            unknown_operations = sorted(set(record.operation_ids) - operation_ids)
            if unknown_operations:
                raise ValueError(
                    f"evidence references unknown operation: {unknown_operations[0]}"
                )
        for lesson in self.lessons:
            unknown_sources = sorted(set(lesson.source_dependencies) - source_ids)
            if unknown_sources:
                raise ValueError(
                    f"lesson references unknown source: {unknown_sources[0]}"
                )
            unknown_operations = sorted(
                set(lesson.operation_dependencies) - operation_ids
            )
            if unknown_operations:
                raise ValueError(
                    f"lesson references unknown operation: {unknown_operations[0]}"
                )
            unknown_evidence = sorted(set(lesson.evidence_ids) - evidence_ids)
            if unknown_evidence:
                raise ValueError(
                    f"lesson references unknown evidence: {unknown_evidence[0]}"
                )

        for relationship in self.relationships:
            for source_id in (
                relationship.left_source,
                relationship.right_source,
            ):
                if source_id not in source_ids:
                    raise ValueError(
                        f"relationship references unknown source: {source_id}"
                    )
        for operation in self.operations:
            unknown_sources = sorted(set(operation.required_sources) - source_ids)
            if unknown_sources:
                raise ValueError(
                    f"operation references unknown source: {unknown_sources[0]}"
                )
            unknown_relationships = sorted(
                set(operation.required_relationships) - relationship_ids
            )
            if unknown_relationships:
                raise ValueError(
                    "operation references unknown relationship: "
                    f"{unknown_relationships[0]}"
                )

        subjects = {
            "source": source_ids,
            "relationship": relationship_ids,
            "operation": operation_ids,
            "package": {self.package_id},
            "lesson": lesson_ids,
            "evidence": evidence_ids,
        }
        for event in self.events:
            if event.subject_id not in subjects[event.subject_type]:
                raise ValueError(
                    f"event references unknown {event.subject_type}: "
                    f"{event.subject_id}"
                )

    @staticmethod
    def _unique_ids(records: Sequence[object], attribute: str) -> set[str]:
        values = [getattr(record, attribute) for record in records]
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {attribute}")
        return set(values)

    @property
    def snapshot_fingerprint(self) -> str:
        return _domain_fingerprint(
            "fabric-rlm.knowledge.snapshot.v1",
            {
                source.source_id: source.snapshot_fingerprint
                for source in self.sources
            },
        )

    @property
    def schema_fingerprint(self) -> str:
        return _domain_fingerprint(
            "fabric-rlm.knowledge.schema.v1",
            {
                source.source_id: source.schema_fingerprint
                for source in self.sources
            },
        )

    @property
    def fingerprint(self) -> str:
        return _domain_fingerprint(
            "fabric-rlm.knowledge.package.v1",
            self.to_dict(),
        )

    @property
    def format_version(self) -> int:
        """1 for a package without learning records, 2 with them.

        A package that carries no evidence and no lessons serializes exactly
        as before learning existed, so its fingerprint and its readers are
        unchanged; the learning records are the only thing that moves the
        format to 2.
        """
        return 2 if (self.evidence or self.lessons) else 1

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "format_version": self.format_version,
            "package_id": self.package_id,
            "sources": [
                item.to_dict()
                for item in sorted(self.sources, key=lambda item: item.source_id)
            ],
            "relationships": [
                item.to_dict()
                for item in sorted(
                    self.relationships,
                    key=lambda item: item.relationship_id,
                )
            ],
            "operations": [
                item.to_dict()
                for item in sorted(
                    self.operations,
                    key=lambda item: item.operation_id,
                )
            ],
            "events": [item.to_dict() for item in self.events],
        }
        if self.format_version == 2:
            payload["evidence"] = [item.to_dict() for item in self.evidence]
            payload["lessons"] = [
                item.to_dict()
                for item in sorted(self.lessons, key=lambda item: item.lesson_id)
            ]
        return payload

    @classmethod
    def from_dict(cls, value: object) -> KnowledgePackage:
        payload = _strict_payload(
            value,
            record_name="KnowledgePackage",
            fields={
                "format_version",
                "package_id",
                "sources",
                "relationships",
                "operations",
                "events",
                "evidence",
                "lessons",
            },
        )
        format_version = payload.pop("format_version", _MISSING)
        if type(format_version) is not int or format_version not in {1, 2}:
            raise ValueError("format_version must be 1 or 2")
        if format_version == 1 and ("evidence" in payload or "lessons" in payload):
            raise ValueError("format_version 1 packages carry no learning records")
        for field_name, record_type in (
            ("sources", SourceProfile),
            ("relationships", Relationship),
            ("operations", RegisteredOperation),
            ("events", KnowledgeEvent),
            ("evidence", EvidenceRecord),
            ("lessons", LearnedLesson),
        ):
            raw_values = payload.get(field_name, ())
            if not isinstance(raw_values, (list, tuple)):
                raise ValueError(f"{field_name} must be a sequence")
            payload[field_name] = tuple(
                record_type.from_dict(item) for item in raw_values
            )
        return cls(**payload)
