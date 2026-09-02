"""Internal discovery of host-owned operations from approved source metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from itertools import combinations

from fabric_rlm.knowledge import RegisteredOperation, SourceProfile
from fabric_rlm.knowledge_sources import _normalized_field_tokens


_MAX_OPERATION_VALUES = 500
_MAX_METADATA_NAME_LENGTH = 256
_ANALYTICS_SENSITIVE_TOKENS = frozenset(
    {"email", "phone", "ssn", "socialsecuritynumber"}
)


def _safe_analytics_columns(
    columns: Mapping[str, object],
    sensitive_columns: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sensitive = set(sensitive_columns)
    safe_columns = tuple(
        sorted(
            name
            for name in columns
            if name not in sensitive
            and not _ANALYTICS_SENSITIVE_TOKENS.intersection(
                _normalized_field_tokens(name)
            )
        )
    )
    numeric_columns = tuple(
        name
        for name in safe_columns
        if isinstance(columns[name], Mapping)
        and columns[name].get("type") in {"integer", "number"}
    )
    return safe_columns, numeric_columns


def _records(value: object, family: str) -> list[Mapping[str, object]]:
    try:
        records = value.to_dict(orient="records")
    except Exception as exc:
        raise ValueError(
            f"semantic model {family} metadata could not be read for operations"
        ) from exc
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError(f"semantic model {family} metadata must be records")
    normalized: list[Mapping[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"semantic model {family} metadata must be records")
        normalized.append(record)
    return normalized


def _metadata_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"semantic model {field_name} must be a non-empty string")
    name = value.strip()
    if len(name) > _MAX_METADATA_NAME_LENGTH:
        raise ValueError(f"semantic model {field_name} is too long")
    if any(ord(character) < 32 for character in name):
        raise ValueError(f"semantic model {field_name} contains control characters")
    return name


def _is_visible(record: Mapping[str, object]) -> bool:
    hidden = next(
        (
            record[name]
            for name in ("Is Hidden", "IsHidden", "Hidden")
            if name in record
        ),
        False,
    )
    if isinstance(hidden, str):
        return hidden.strip().lower() not in {"1", "true", "yes"}
    return hidden is not True and hidden != 1


def _semantic_measure_operation(
    source_id: str,
    model: object,
) -> RegisteredOperation | None:
    measures = sorted(
        {
            _metadata_name(record.get("Measure Name"), "measure name")
            for record in _records(model.measures(), "measure")
            if _is_visible(record)
        }
    )
    columns = sorted(
        {
            (
                f"{_metadata_name(record.get('Table Name'), 'table name')}"
                f"[{_metadata_name(record.get('Column Name'), 'column name')}]"
            )
            for record in _records(model.columns(), "column")
            if _is_visible(record)
        }
    )
    if not measures:
        return None
    if len(measures) > _MAX_OPERATION_VALUES or len(columns) > _MAX_OPERATION_VALUES:
        return None
    allowed_columns = ("", *columns)
    return RegisteredOperation(
        operation_id=f"{source_id}.semantic_model.measure.v1",
        operation="semantic_model.measure",
        required_sources=(source_id,),
        parameter_schema={
            "measure": {"type": "string", "enum": measures},
            "groupby": {"type": "string", "enum": allowed_columns},
            "groupby_2": {"type": "string", "enum": allowed_columns},
            "filter_column": {"type": "string", "enum": allowed_columns},
            "filter_value": {"type": "string"},
            "filter_column_2": {"type": "string", "enum": allowed_columns},
            "filter_value_2": {"type": "string"},
            "filter_column_3": {"type": "string", "enum": allowed_columns},
            "filter_value_3": {"type": "string"},
        },
        parameter_defaults={
            "groupby": "",
            "groupby_2": "",
            "filter_column": "",
            "filter_value": "",
            "filter_column_2": "",
            "filter_value_2": "",
            "filter_column_3": "",
            "filter_value_3": "",
        },
        output_schema={
            "result_fingerprint": "string",
            "row_count": "integer",
        },
        max_output_rows=1_000,
        max_output_columns=20,
        grain="semantic_model_measure_result",
        host_implementation_id="semantic_model.measure.v1",
        operation_version="1",
        status="active",
    )


def _tabular_aggregate_operation(
    profile: SourceProfile,
) -> RegisteredOperation | None:
    if (
        profile.family not in {"csv", "parquet", "delta"}
        or profile.diagnostics.get("snapshot_exact") is not True
    ):
        return None
    safe_columns, numeric_columns = _safe_analytics_columns(
        profile.schema,
        profile.sensitive_columns,
    )
    if not safe_columns:
        return None
    allowed_columns = ("", *safe_columns)
    return RegisteredOperation(
        operation_id=f"{profile.source_id}.tabular.aggregate.v1",
        operation="tabular.aggregate",
        required_sources=(profile.source_id,),
        parameter_schema={
            "aggregate": {
                "type": "string",
                "enum": ("avg", "count_rows", "sum"),
            },
            "measure": {"type": "string", "enum": ("", *numeric_columns)},
            "groupby": {"type": "string", "enum": allowed_columns},
            "groupby_2": {"type": "string", "enum": allowed_columns},
            "filter_column": {"type": "string", "enum": allowed_columns},
            "filter_value": {"type": "string"},
            "filter_column_2": {"type": "string", "enum": allowed_columns},
            "filter_value_2": {"type": "string"},
        },
        parameter_defaults={
            "measure": "",
            "groupby": "",
            "groupby_2": "",
            "filter_column": "",
            "filter_value": "",
            "filter_column_2": "",
            "filter_value_2": "",
        },
        output_schema={
            "result_fingerprint": "string",
            "row_count": "integer",
        },
        max_output_rows=100,
        max_output_columns=20,
        grain="tabular_aggregate_result",
        host_implementation_id="tabular.aggregate.v1",
        operation_version="1",
        status="active",
    )


def _lakehouse_aggregate_operations(
    profile: SourceProfile,
) -> tuple[RegisteredOperation, ...]:
    if (
        profile.family != "lakehouse"
        or profile.diagnostics.get("snapshot_exact") is not True
    ):
        return ()
    operations: list[RegisteredOperation] = []
    for catalog_source, raw_entry in sorted(profile.schema.items()):
        if not isinstance(raw_entry, Mapping):
            continue
        raw_columns = raw_entry.get("columns")
        if not isinstance(raw_columns, Mapping):
            continue
        safe_columns, numeric_columns = _safe_analytics_columns(
            raw_columns,
            profile.sensitive_columns,
        )
        if not safe_columns:
            continue
        source_key = hashlib.sha256(catalog_source.encode("utf-8")).hexdigest()[:12]
        allowed_columns = ("", *safe_columns)
        operations.append(
            RegisteredOperation(
                operation_id=(
                    f"{profile.source_id}.lakehouse.aggregate.{source_key}.v1"
                ),
                operation="lakehouse.aggregate",
                required_sources=(profile.source_id,),
                parameter_schema={
                    "catalog_source": {
                        "type": "string",
                        "enum": (catalog_source,),
                    },
                    "aggregate": {
                        "type": "string",
                        "enum": ("avg", "count_rows", "sum"),
                    },
                    "measure": {
                        "type": "string",
                        "enum": ("", *numeric_columns),
                    },
                    "groupby": {
                        "type": "string",
                        "enum": allowed_columns,
                    },
                    "groupby_2": {
                        "type": "string",
                        "enum": allowed_columns,
                    },
                    "filter_column": {
                        "type": "string",
                        "enum": allowed_columns,
                    },
                    "filter_value": {"type": "string"},
                    "filter_column_2": {
                        "type": "string",
                        "enum": allowed_columns,
                    },
                    "filter_value_2": {"type": "string"},
                },
                parameter_defaults={
                    "measure": "",
                    "groupby": "",
                    "groupby_2": "",
                    "filter_column": "",
                    "filter_value": "",
                    "filter_column_2": "",
                    "filter_value_2": "",
                },
                output_schema={
                    "result_fingerprint": "string",
                    "row_count": "integer",
                },
                max_output_rows=100,
                max_output_columns=20,
                grain="lakehouse_aggregate_result",
                host_implementation_id="lakehouse.aggregate.v1",
                operation_version="1",
                status="active",
            )
        )
    return tuple(operations)


def _lakehouse_preaggregate_join_operations(
    profile: SourceProfile,
) -> tuple[RegisteredOperation, ...]:
    if (
        profile.family != "lakehouse"
        or profile.diagnostics.get("snapshot_exact") is not True
    ):
        return ()
    candidates: list[
        tuple[str, tuple[str, ...], tuple[str, ...]]
    ] = []
    for catalog_source, raw_entry in sorted(profile.schema.items()):
        if not isinstance(raw_entry, Mapping):
            continue
        raw_columns = raw_entry.get("columns")
        if not isinstance(raw_columns, Mapping):
            continue
        safe_columns, numeric_columns = _safe_analytics_columns(
            raw_columns,
            profile.sensitive_columns,
        )
        if safe_columns and numeric_columns:
            candidates.append((catalog_source, safe_columns, numeric_columns))

    operations: list[RegisteredOperation] = []
    for left, right in combinations(candidates, 2):
        left_source, left_columns, left_numeric = left
        right_source, right_columns, right_numeric = right
        common_columns = tuple(sorted(set(left_columns) & set(right_columns)))
        if not common_columns:
            continue
        pair_key = hashlib.sha256(
            f"{left_source}\0{right_source}".encode("utf-8")
        ).hexdigest()[:12]
        operations.append(
            RegisteredOperation(
                operation_id=(
                    f"{profile.source_id}.lakehouse.preaggregate_join."
                    f"{pair_key}.v1"
                ),
                operation="lakehouse.preaggregate_join",
                required_sources=(profile.source_id,),
                parameter_schema={
                    "left_catalog_source": {
                        "type": "string",
                        "enum": (left_source,),
                    },
                    "right_catalog_source": {
                        "type": "string",
                        "enum": (right_source,),
                    },
                    "left_measure": {
                        "type": "string",
                        "enum": left_numeric,
                    },
                    "right_measure": {
                        "type": "string",
                        "enum": right_numeric,
                    },
                    "join_key": {
                        "type": "string",
                        "enum": common_columns,
                    },
                    "join_key_2": {
                        "type": "string",
                        "enum": ("", *common_columns),
                    },
                    "scope": {
                        "type": "string",
                        "enum": ("all", "latest"),
                    },
                },
                parameter_defaults={
                    "join_key_2": "",
                    "scope": "all",
                },
                output_schema={
                    "result_fingerprint": "string",
                    "row_count": "integer",
                },
                max_output_rows=100,
                max_output_columns=20,
                grain="one_row_per_shared_join_key",
                host_implementation_id="lakehouse.preaggregate_join.v1",
                operation_version="1",
                status="active",
            )
        )
        if len(operations) >= 20:
            break
    return tuple(operations)


def discover_registered_operations(
    profiles: tuple[SourceProfile, ...],
    sources: Mapping[str, object],
) -> tuple[RegisteredOperation, ...]:
    """Return conservative operations supported by the current source bindings."""

    operations: list[RegisteredOperation] = []
    for profile in profiles:
        operations.extend(_lakehouse_aggregate_operations(profile))
        operations.extend(_lakehouse_preaggregate_join_operations(profile))
        operation = _tabular_aggregate_operation(profile)
        if profile.family == "semantic_model":
            model = sources[profile.source_id]
            if not callable(getattr(model, "measures", None)) or not callable(
                getattr(model, "columns", None)
            ):
                continue
            operation = _semantic_measure_operation(profile.source_id, model)
        if operation is not None:
            operations.append(operation)
    return tuple(sorted(operations, key=lambda item: item.operation_id))


__all__ = ["discover_registered_operations"]
