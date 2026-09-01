"""Internal host execution for registered knowledge operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import math
import time

from fabric_rlm.knowledge import (
    RegisteredOperation,
    _domain_fingerprint,
)
from fabric_rlm.knowledge_api import Knowledge
from fabric_rlm.knowledge_preflight import preflight_knowledge


_MAX_PARAMETER_TEXT_LENGTH = 512


def _operation(
    knowledge: Knowledge,
    operation_id: str,
) -> RegisteredOperation:
    matches = [
        operation
        for operation in knowledge.package.operations
        if operation.operation_id == operation_id
    ]
    if not matches:
        raise ValueError(f"operation is not registered: {operation_id}")
    operation = matches[0]
    if operation.status != "active":
        raise ValueError(f"operation is not active: {operation_id}")
    if (
        operation.operation != "semantic_model.measure"
        or operation.host_implementation_id != "semantic_model.measure.v1"
    ):
        raise ValueError(f"operation host implementation is unavailable: {operation_id}")
    return operation


def _parameter_value(
    name: str,
    value: object,
    descriptor: Mapping[str, object],
) -> object:
    expected_type = descriptor["type"]
    if expected_type == "string" and type(value) is not str:
        raise ValueError(f"{name} must match parameter type string")
    if expected_type == "integer" and type(value) is not int:
        raise ValueError(f"{name} must match parameter type integer")
    if expected_type == "number" and type(value) not in {int, float}:
        raise ValueError(f"{name} must match parameter type number")
    if expected_type == "boolean" and type(value) is not bool:
        raise ValueError(f"{name} must match parameter type boolean")
    if expected_type == "null" and value is not None:
        raise ValueError(f"{name} must match parameter type null")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if isinstance(value, str):
        if len(value) > _MAX_PARAMETER_TEXT_LENGTH:
            raise ValueError(f"{name} is too long")
        if any(ord(character) < 32 for character in value):
            raise ValueError(f"{name} contains control characters")
    allowed = descriptor.get("enum")
    if allowed is not None and value not in allowed:
        raise ValueError(f"{name} is not an allowed value")
    return value


def _parameters(
    operation: RegisteredOperation,
    supplied: Mapping[str, object],
) -> dict[str, object]:
    unknown = sorted(set(supplied) - set(operation.parameter_schema))
    if unknown:
        raise ValueError(f"unknown parameter: {unknown[0]}")
    values = dict(operation.parameter_defaults)
    values.update(supplied)
    missing = sorted(set(operation.parameter_schema) - set(values))
    if missing:
        raise ValueError(f"missing required parameter: {missing[0]}")
    normalized = {
        name: _parameter_value(name, values[name], descriptor)
        for name, descriptor in operation.parameter_schema.items()
    }
    has_filter_column = bool(normalized["filter_column"])
    has_filter_value = bool(normalized["filter_value"])
    if has_filter_column != has_filter_value:
        raise ValueError(
            "filter_column and filter_value must both be provided or both omitted"
        )
    return normalized


def _scalar(value: object, field_name: str) -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"operation result {field_name} must be finite")
        return value + 0.0
    if isinstance(value, Decimal):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"operation result {field_name} must be finite")
        return normalized
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        return _scalar(item(), field_name)
    raise ValueError(f"operation result {field_name} must be a scalar value")


def _result_rows(
    value: object,
    operation: RegisteredOperation,
) -> tuple[dict[str, object], ...]:
    try:
        records = value.to_dict(orient="records")
    except Exception as exc:
        raise ValueError("operation result must provide tabular records") from exc
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("operation result must provide tabular records")
    if len(records) > operation.max_output_rows:
        raise ValueError("operation result exceeds row bound")

    columns: set[str] = set()
    normalized: list[dict[str, object]] = []
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("operation result rows must be objects")
        if any(not isinstance(name, str) or not name for name in record):
            raise ValueError("operation result columns must be non-empty strings")
        columns.update(record)
        if len(columns) > operation.max_output_columns:
            raise ValueError("operation result exceeds column bound")
        normalized.append(
            {
                name: _scalar(value, f"row {row_index} column {name}")
                for name, value in record.items()
            }
        )
    return tuple(normalized)


@dataclass(frozen=True)
class OperationExecutionResult:
    operation_id: str
    operation_version: str | None
    operation_fingerprint: str
    source_fingerprints: Mapping[str, str]
    result_fingerprint: str
    rows: tuple[dict[str, object], ...]
    elapsed_seconds: float
    audit_status: str = "passed"

    def to_packet(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "operation_version": self.operation_version,
            "operation_fingerprint": self.operation_fingerprint,
            "source_fingerprints": dict(self.source_fingerprints),
            "result_fingerprint": self.result_fingerprint,
            "audit_status": self.audit_status,
            "row_count": len(self.rows),
            "rows": [dict(row) for row in self.rows],
            "elapsed_seconds": self.elapsed_seconds,
        }


def execute_registered_operation(
    knowledge: Knowledge,
    *,
    operation_id: str,
    parameters: Mapping[str, object],
) -> OperationExecutionResult:
    """Validate and execute one registered operation through its host adapter."""

    if not isinstance(parameters, Mapping):
        raise TypeError("operation parameters must be a mapping")
    operation = _operation(knowledge, operation_id)
    preflight = preflight_knowledge(
        knowledge.package,
        knowledge.bindings,
        limits=knowledge._limits,
        registry=knowledge._registry,
    )
    if preflight.drift:
        raise ValueError(
            "stale knowledge sources detected: "
            + ", ".join(sorted(preflight.drift))
        )
    current_operation = next(
        item
        for item in preflight.package.operations
        if item.operation_id == operation.operation_id
    )
    if current_operation.status != "active":
        raise ValueError(f"operation is not active: {operation.operation_id}")

    normalized = _parameters(operation, parameters)
    source_id = operation.required_sources[0]
    model = knowledge.bindings[source_id]
    measure = getattr(model, "measure", None)
    if not callable(measure):
        raise TypeError("registered semantic model source cannot evaluate measures")
    groupby_value = str(normalized["groupby"])
    filter_column = str(normalized["filter_column"])
    filter_value = str(normalized["filter_value"])

    started = time.perf_counter()
    raw_result = measure(
        str(normalized["measure"]),
        groupby=[groupby_value] if groupby_value else None,
        filters={filter_column: [filter_value]} if filter_column else None,
    )
    elapsed_seconds = time.perf_counter() - started
    rows = _result_rows(raw_result, operation)
    source_fingerprints = {
        source.source_id: source.snapshot_fingerprint
        for source in preflight.package.sources
        if source.source_id in operation.required_sources
    }
    operation_fingerprint = _domain_fingerprint(
        "fabric-rlm.registered-operation.v1",
        operation.to_dict(),
    )
    result_fingerprint = _domain_fingerprint(
        "fabric-rlm.operation-result.v1",
        {
            "operation_id": operation.operation_id,
            "operation_version": operation.operation_version,
            "operation_fingerprint": operation_fingerprint,
            "source_fingerprints": source_fingerprints,
            "parameters": normalized,
            "rows": rows,
        },
    )
    return OperationExecutionResult(
        operation_id=operation.operation_id,
        operation_version=operation.operation_version,
        operation_fingerprint=operation_fingerprint,
        source_fingerprints=source_fingerprints,
        result_fingerprint=result_fingerprint,
        rows=rows,
        elapsed_seconds=elapsed_seconds,
    )


__all__ = ["OperationExecutionResult", "execute_registered_operation"]
