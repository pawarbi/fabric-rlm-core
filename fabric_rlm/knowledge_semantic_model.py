"""Metadata-only knowledge profiling for Power BI semantic models.

SemPy documents these metadata APIs on Microsoft Learn:
https://learn.microsoft.com/en-us/python/api/semantic-link-sempy/sempy.fabric?view=semantic-link-python#sempy-fabric-list-tables
https://learn.microsoft.com/en-us/python/api/semantic-link-sempy/sempy.fabric?view=semantic-link-python#sempy-fabric-list-columns
https://learn.microsoft.com/en-us/python/api/semantic-link-sempy/sempy.fabric?view=semantic-link-python#sempy-fabric-list-measures
https://learn.microsoft.com/en-us/python/api/semantic-link-sempy/sempy.fabric?view=semantic-link-python#sempy-fabric-list-relationships

The provider materializes each metadata result before this adapter can bound
it. The persisted profile is bounded, contains no evaluated rows, and retains
measure expressions and descriptions only as fingerprints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any, cast

from fabric_rlm.knowledge import (
    SourceProfile,
    SourceRole,
    _domain_fingerprint,
    canonical_json,
)
from fabric_rlm.knowledge_sources import ProfileLimits, SourceAdapterRegistry
from fabric_rlm.semantic_model import SemanticModel


_ALLOWED_ROLES = frozenset(
    {"numeric_evidence", "lookup", "context_only", "excluded"}
)
_CODE_CHARACTER = re.compile(r"[^a-z0-9_.-]+")
_METADATA_METHODS = ("tables", "columns", "measures", "relationships")

_ALIASES = {
    "tables": {
        "Name": "name",
        "Description": "description",
        "Type": "table_type",
    },
    "columns": {
        "Table Name": "table_name",
        "Column Name": "column_name",
        "Description": "description",
        "Data Type": "data_type",
    },
    "measures": {
        "Table Name": "table_name",
        "Measure Name": "measure_name",
        "Measure Expression": "measure_expression",
        "Measure Description": "measure_description",
        "Measure Display Folder": "measure_display_folder",
    },
    "relationships": {
        "From Table": "from_table",
        "From Column": "from_column",
        "To Table": "to_table",
        "To Column": "to_column",
        "Multiplicity": "multiplicity",
        "Cardinality": "cardinality",
        "Relationship Name": "relationship_name",
    },
}


def _normalized_field_name(value: object) -> str:
    text = str(value).strip().strip("[]")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "field"


def _records(value: object, family: str) -> list[dict[str, object]]:
    try:
        if hasattr(value, "to_dict"):
            raw = value.to_dict(orient="records")
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            raw = list(value)
        else:
            raise TypeError
    except Exception as error:
        raise ValueError(
            f"semantic model {family} metadata has an unsupported shape"
        ) from None
    if not isinstance(raw, Sequence):
        raise ValueError(
            f"semantic model {family} metadata has an unsupported shape"
        )
    records: list[dict[str, object]] = []
    aliases = _ALIASES[family]
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError(
                f"semantic model {family} metadata must contain records"
            )
        normalized: dict[str, object] = {}
        for name, field_value in item.items():
            normalized[aliases.get(str(name), _normalized_field_name(name))] = (
                field_value
            )
        records.append(normalized)
    return records


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if type(value) in {bool, int}:
        return str(value)
    if isinstance(value, float):
        return str(value) if math.isfinite(value) else type(value).__name__
    return str(value).strip()


def _required(record: Mapping[str, object], field: str, family: str) -> str:
    value = _text(record.get(field))
    if not value:
        raise ValueError(
            f"semantic model {family} metadata is missing required identities"
        )
    return value


def _code(value: object, default: str = "unknown") -> str:
    normalized = _CODE_CHARACTER.sub("_", _text(value).lower()).strip("_.-")
    if not normalized:
        return default
    if len(normalized) <= 64:
        return normalized
    return "fp_" + _domain_fingerprint(
        "semantic-model-metadata-code-v1", normalized
    )[:32]


def _value_fingerprint(domain: str, value: object) -> str:
    return _domain_fingerprint(domain, {"value": _text(value)})


def _record_fingerprint(family: str, record: Mapping[str, object]) -> str:
    normalized = {
        name: _value_fingerprint(
            "semantic-model-provider-field-v1",
            value,
        )
        for name, value in sorted(record.items())
    }
    return _domain_fingerprint(
        "semantic-model-provider-record-v1",
        {"family": family, "fields": normalized},
    )


def _normalize_tables(records: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    normalized = []
    for record in records:
        normalized.append(
            {
                "name": _required(record, "name", "tables"),
                "type": _code(record.get("table_type"), "table"),
                "description_fingerprint": _value_fingerprint(
                    "semantic-model-description-v1",
                    record.get("description"),
                ),
                "record_fingerprint": _record_fingerprint("tables", record),
            }
        )
    return sorted(normalized, key=canonical_json)


def _normalize_columns(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    normalized = []
    for record in records:
        normalized.append(
            {
                "table": _required(record, "table_name", "columns"),
                "name": _required(record, "column_name", "columns"),
                "type": _code(record.get("data_type")),
                "description_fingerprint": _value_fingerprint(
                    "semantic-model-description-v1",
                    record.get("description"),
                ),
                "record_fingerprint": _record_fingerprint("columns", record),
            }
        )
    return sorted(normalized, key=canonical_json)


def _normalize_measures(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    normalized = []
    for record in records:
        normalized.append(
            {
                "table": _required(record, "table_name", "measures"),
                "name": _required(record, "measure_name", "measures"),
                "expression_fingerprint": _value_fingerprint(
                    "semantic-model-measure-expression-v1",
                    record.get("measure_expression"),
                ),
                "description_fingerprint": _value_fingerprint(
                    "semantic-model-description-v1",
                    record.get("measure_description"),
                ),
                "display_folder_fingerprint": _value_fingerprint(
                    "semantic-model-display-folder-v1",
                    record.get("measure_display_folder"),
                ),
                "record_fingerprint": _record_fingerprint("measures", record),
            }
        )
    return sorted(normalized, key=canonical_json)


def _normalize_relationships(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    normalized = []
    for record in records:
        normalized.append(
            {
                "from_table": _required(
                    record, "from_table", "relationships"
                ),
                "from_column": _required(
                    record, "from_column", "relationships"
                ),
                "to_table": _required(record, "to_table", "relationships"),
                "to_column": _required(record, "to_column", "relationships"),
                "multiplicity": _code(record.get("multiplicity")),
                "cardinality": _code(record.get("cardinality")),
                "name_fingerprint": _value_fingerprint(
                    "semantic-model-relationship-name-v1",
                    record.get("relationship_name"),
                ),
                "record_fingerprint": _record_fingerprint(
                    "relationships", record
                ),
            }
        )
    return sorted(normalized, key=canonical_json)


def _capture(model: SemanticModel) -> dict[str, list[dict[str, str]]]:
    try:
        raw: dict[str, list[dict[str, object]]] = {}
        for method_name in _METADATA_METHODS:
            raw[method_name] = _records(
                getattr(model, method_name)(),
                method_name,
            )
        return {
            "tables": _normalize_tables(raw["tables"]),
            "columns": _normalize_columns(raw["columns"]),
            "measures": _normalize_measures(raw["measures"]),
            "relationships": _normalize_relationships(raw["relationships"]),
        }
    except Exception:
        raise ValueError(
            "semantic model metadata could not be read; verify SemPy "
            "availability and metadata permissions"
        ) from None


def _structural(capture: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, object]:
    return {
        "tables": [
            {"name": row["name"], "type": row["type"]}
            for row in capture["tables"]
        ],
        "columns": [
            {
                "table": row["table"],
                "name": row["name"],
                "type": row["type"],
            }
            for row in capture["columns"]
        ],
        "measures": [
            {"table": row["table"], "name": row["name"]}
            for row in capture["measures"]
        ],
        "relationships": [
            {
                "from_table": row["from_table"],
                "from_column": row["from_column"],
                "to_table": row["to_table"],
                "to_column": row["to_column"],
                "multiplicity": row["multiplicity"],
                "cardinality": row["cardinality"],
            }
            for row in capture["relationships"]
        ],
    }


@dataclass
class _TextRetainer:
    remaining_bytes: int
    truncated: bool = False

    def key(self, value: str, domain: str) -> str:
        encoded = value.encode("utf-8")
        if len(encoded) <= self.remaining_bytes:
            self.remaining_bytes -= len(encoded)
            return value
        self.truncated = True
        return "fp_" + _domain_fingerprint(domain, value)[:32]


def _descriptor(
    fields: Sequence[tuple[str, str]],
    limits: ProfileLimits,
) -> dict[str, str]:
    return dict(fields[: limits.max_fields])


def _retained_schema(
    capture: Mapping[str, Sequence[Mapping[str, str]]],
    limits: ProfileLimits,
) -> tuple[dict[str, object], bool, bool, bool]:
    text = _TextRetainer(limits.max_input_bytes)
    records_truncated = any(
        len(records) > limits.max_records for records in capture.values()
    )
    if limits.max_nesting_depth < 3:
        return {}, records_truncated, False, False
    fields_truncated = False
    schema: dict[str, dict[str, object]] = {
        "tables": {},
        "columns": {},
        "measures": {},
        "relationships": {},
    }

    for row in capture["tables"][: limits.max_records]:
        key = text.key(row["name"], "semantic-model-table-key-v1")
        fields = (("type", row["type"]),)
        fields_truncated |= len(fields) > limits.max_fields
        schema["tables"][key] = _descriptor(fields, limits)

    for row in capture["columns"][: limits.max_records]:
        identity = f"{row['table']}[{row['name']}]"
        key = text.key(identity, "semantic-model-column-key-v1")
        fields = (("type", row["type"]),)
        fields_truncated |= len(fields) > limits.max_fields
        schema["columns"][key] = _descriptor(fields, limits)

    for row in capture["measures"][: limits.max_records]:
        identity = f"{row['table']}[{row['name']}]"
        key = text.key(identity, "semantic-model-measure-key-v1")
        fields = (("type", "measure"),)
        fields_truncated |= len(fields) > limits.max_fields
        schema["measures"][key] = _descriptor(fields, limits)

    for row in capture["relationships"][: limits.max_records]:
        identity = (
            f"{row['from_table']}[{row['from_column']}]"
            f"->{row['to_table']}[{row['to_column']}]"
            f"|{row['multiplicity']}|{row['cardinality']}"
        )
        key = text.key(identity, "semantic-model-relationship-key-v1")
        fields = (("type", "relationship"),)
        fields_truncated |= len(fields) > limits.max_fields
        schema["relationships"][key] = _descriptor(fields, limits)

    return schema, records_truncated, fields_truncated, text.truncated


def _retained_fingerprints(
    capture: Mapping[str, Sequence[Mapping[str, str]]],
    limits: ProfileLimits,
) -> tuple[dict[str, str], bool]:
    fields = (
        (
            "expression_fingerprint",
            _domain_fingerprint(
                "semantic-model-retained-expressions-v1",
                [
                    row["expression_fingerprint"]
                    for row in capture["measures"]
                ],
            ),
        ),
        (
            "description_fingerprint",
            _domain_fingerprint(
                "semantic-model-retained-descriptions-v1",
                [
                    row["description_fingerprint"]
                    for family in ("tables", "columns", "measures")
                    for row in capture[family]
                ],
            ),
        ),
        (
            "complete_metadata_fingerprint",
            _domain_fingerprint(
                "semantic-model-retained-provider-metadata-v1",
                [
                    row["record_fingerprint"]
                    for family in _METADATA_METHODS
                    for row in capture[family]
                ],
            ),
        ),
    )
    return dict(fields[: limits.max_fields]), len(fields) > limits.max_fields


def _locator(model: SemanticModel) -> str:
    dataset = model.dataset
    workspace = model.workspace
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("semantic model logical coordinates are invalid")
    if workspace is not None and not isinstance(workspace, str):
        raise ValueError("semantic model logical coordinates are invalid")
    fingerprint = _domain_fingerprint(
        "semantic-model-locator-v1",
        {
            "dataset": dataset.strip(),
            "workspace": workspace.strip() if workspace else None,
        },
    )
    return f"semantic-model/v1/{fingerprint}"


@dataclass(frozen=True)
class SemanticModelKnowledgeAdapter:
    """Profile a ``SemanticModel`` through metadata APIs only."""

    family: str = "semantic_model"
    allowed_roles: frozenset[str] = _ALLOWED_ROLES
    default_role: SourceRole = "numeric_evidence"

    def matches(self, value: object) -> bool:
        return isinstance(value, SemanticModel)

    def profile(
        self,
        source_id: str,
        value: object,
        role: SourceRole,
        limits: ProfileLimits,
    ) -> SourceProfile:
        if role not in self.allowed_roles:
            raise ValueError(
                f"role {role} is not supported by the {self.family} adapter"
            )
        if not isinstance(value, SemanticModel):
            raise TypeError("semantic model adapter requires SemanticModel")

        model = cast(SemanticModel, value)
        locator = _locator(model)
        first = _capture(model)
        second = _capture(model)
        if locator != _locator(model) or first != second:
            raise ValueError("semantic model changed during profiling")

        structural = _structural(first)
        schema_fingerprint = _domain_fingerprint(
            "semantic-model-schema-v1",
            structural,
        )
        snapshot_fingerprint = _domain_fingerprint(
            "semantic-model-snapshot-v1",
            {"schema": structural, "metadata": first},
        )
        schema, records_truncated, fields_truncated, text_truncated = (
            _retained_schema(first, limits)
        )
        retained_fingerprints, fingerprint_fields_truncated = (
            _retained_fingerprints(first, limits)
        )
        total_records = sum(len(records) for records in first.values())
        return SourceProfile(
            source_id=source_id,
            family=self.family,
            locator=locator,
            snapshot_fingerprint=snapshot_fingerprint,
            schema_fingerprint=schema_fingerprint,
            schema=schema,
            diagnostics={
                "snapshot_exact": True,
                "provider_materialization_bounded": False,
                "records_inspected": total_records,
                "records_truncated": records_truncated,
                "fields_truncated": (
                    fields_truncated or fingerprint_fields_truncated
                ),
                "text_truncated": text_truncated,
                "nesting_truncated": limits.max_nesting_depth < 3,
                **retained_fingerprints,
            },
            role=role,
        )


def semantic_model_adapter() -> SemanticModelKnowledgeAdapter:
    """Return the opt-in adapter for later registry composition."""

    return SemanticModelKnowledgeAdapter()


def semantic_model_registry() -> SourceAdapterRegistry:
    """Return an explicit registry containing only the semantic-model adapter."""

    from fabric_rlm.knowledge_sources import SourceAdapterRegistry

    return SourceAdapterRegistry((semantic_model_adapter(),))


__all__ = [
    "SemanticModelKnowledgeAdapter",
    "semantic_model_adapter",
    "semantic_model_registry",
]
