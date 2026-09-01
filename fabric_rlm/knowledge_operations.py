"""Internal discovery of host-owned operations from approved source metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fabric_rlm.knowledge import RegisteredOperation, SourceProfile


_MAX_OPERATION_VALUES = 500
_MAX_METADATA_NAME_LENGTH = 256


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
            "filter_column": {"type": "string", "enum": allowed_columns},
            "filter_value": {"type": "string"},
            "filter_column_2": {"type": "string", "enum": allowed_columns},
            "filter_value_2": {"type": "string"},
        },
        parameter_defaults={
            "groupby": "",
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
        grain="semantic_model_measure_result",
        host_implementation_id="semantic_model.measure.v1",
        operation_version="1",
        status="active",
    )


def discover_registered_operations(
    profiles: tuple[SourceProfile, ...],
    sources: Mapping[str, object],
) -> tuple[RegisteredOperation, ...]:
    """Return conservative operations supported by the current source bindings."""

    operations: list[RegisteredOperation] = []
    for profile in profiles:
        if profile.family != "semantic_model":
            continue
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
