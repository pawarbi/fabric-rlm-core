"""Internal host execution for registered knowledge operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import json
import math
import re
import time

from fabric_rlm.knowledge import (
    RegisteredOperation,
    _domain_fingerprint,
    canonical_json,
)
from fabric_rlm.knowledge_api import Knowledge
from fabric_rlm.knowledge_preflight import preflight_knowledge


_MAX_PARAMETER_TEXT_LENGTH = 512
_MAX_PLAN_REASON_LENGTH = 256
_MAX_RESULT_BYTES = 256 * 1024
_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class OperationPlanError(ValueError):
    """A model-produced plan is incompatible with the registered contract."""


@dataclass(frozen=True)
class OperationPlan:
    operation_id: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class OperationPlanFallback:
    reason: str


def parse_operation_plan(text: str) -> OperationPlan | OperationPlanFallback:
    """Parse the planner's strict JSON response without accepting executable text."""

    if not isinstance(text, str):
        raise ValueError("operation plan response must be text")
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 3 or lines[0] not in {"```", "```json"}:
            raise ValueError("operation plan must be a JSON object")
        stripped = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(stripped)
    except Exception as exc:
        raise ValueError("operation plan must be valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("operation plan must be a JSON object")
    if payload.get("fallback") is True:
        unknown = sorted(set(payload) - {"fallback", "reason"})
        if unknown:
            raise ValueError(f"operation fallback contains unknown field: {unknown[0]}")
        reason = payload.get("reason", "")
        if not isinstance(reason, str) or len(reason) > _MAX_PLAN_REASON_LENGTH:
            raise ValueError("operation fallback reason is invalid")
        if any(ord(character) < 32 for character in reason):
            raise ValueError("operation fallback reason is invalid")
        return OperationPlanFallback(reason=reason)
    if set(payload) != {"operation_id", "parameters"}:
        raise ValueError(
            "operation plan must contain exactly operation_id and parameters"
        )
    operation_id = payload["operation_id"]
    parameters = payload["parameters"]
    if (
        not isinstance(operation_id, str)
        or not _OPERATION_ID.fullmatch(operation_id)
        or ".." in operation_id
    ):
        raise ValueError("operation_id must be a safe logical identifier")
    if not isinstance(parameters, Mapping):
        raise ValueError("operation parameters must be an object")
    return OperationPlan(
        operation_id=operation_id,
        parameters=dict(parameters),
    )


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
        raise OperationPlanError(f"operation is not registered: {operation_id}")
    operation = matches[0]
    if operation.status != "active":
        raise OperationPlanError(f"operation is not active: {operation_id}")
    supported = {
        ("semantic_model.measure", "semantic_model.measure.v1"),
        ("tabular.aggregate", "tabular.aggregate.v1"),
        ("lakehouse.aggregate", "lakehouse.aggregate.v1"),
        (
            "lakehouse.preaggregate_join",
            "lakehouse.preaggregate_join.v1",
        ),
    }
    if (operation.operation, operation.host_implementation_id) not in supported:
        raise OperationPlanError(
            f"operation host implementation is unavailable: {operation_id}"
        )
    return operation


def _parameter_value(
    name: str,
    value: object,
    descriptor: Mapping[str, object],
) -> object:
    expected_type = descriptor["type"]
    if expected_type == "string" and type(value) is not str:
        raise OperationPlanError(f"{name} must match parameter type string")
    if expected_type == "integer" and type(value) is not int:
        raise OperationPlanError(f"{name} must match parameter type integer")
    if expected_type == "number" and type(value) not in {int, float}:
        raise OperationPlanError(f"{name} must match parameter type number")
    if expected_type == "boolean" and type(value) is not bool:
        raise OperationPlanError(f"{name} must match parameter type boolean")
    if expected_type == "null" and value is not None:
        raise OperationPlanError(f"{name} must match parameter type null")
    if isinstance(value, float) and not math.isfinite(value):
        raise OperationPlanError(f"{name} must be finite")
    if isinstance(value, str):
        if len(value) > _MAX_PARAMETER_TEXT_LENGTH:
            raise OperationPlanError(f"{name} is too long")
        if any(ord(character) < 32 for character in value):
            raise OperationPlanError(f"{name} contains control characters")
    allowed = descriptor.get("enum")
    if allowed is not None and value not in allowed:
        raise OperationPlanError(f"{name} is not an allowed value")
    return value


def _parameters(
    operation: RegisteredOperation,
    supplied: Mapping[str, object],
) -> dict[str, object]:
    unknown = sorted(set(supplied) - set(operation.parameter_schema))
    if unknown:
        raise OperationPlanError(f"unknown parameter: {unknown[0]}")
    values = dict(operation.parameter_defaults)
    values.update(supplied)
    missing = sorted(set(operation.parameter_schema) - set(values))
    if missing:
        raise OperationPlanError(f"missing required parameter: {missing[0]}")
    normalized = {
        name: _parameter_value(name, values[name], descriptor)
        for name, descriptor in operation.parameter_schema.items()
    }
    filter_columns: list[str] = []
    for suffix in ("", "_2"):
        column_name = f"filter_column{suffix}"
        value_name = f"filter_value{suffix}"
        if column_name not in normalized:
            continue
        has_filter_column = bool(normalized[column_name])
        has_filter_value = bool(normalized[value_name])
        if has_filter_column != has_filter_value:
            raise OperationPlanError(
                f"{column_name} and {value_name} must both be provided or both omitted"
            )
        if has_filter_column:
            filter_columns.append(str(normalized[column_name]))
    if len(set(filter_columns)) != len(filter_columns):
        raise OperationPlanError("filter columns must not contain duplicates")
    grouping_dimensions = [
        str(normalized[name])
        for name in ("groupby", "groupby_2")
        if normalized.get(name)
    ]
    if len(set(grouping_dimensions)) != len(grouping_dimensions):
        raise OperationPlanError(
            "grouping dimensions must not contain duplicates"
        )
    aggregate = normalized.get("aggregate")
    measure = normalized.get("measure")
    if aggregate in {"sum", "avg"} and not measure:
        raise OperationPlanError(f"measure is required for {aggregate}")
    if aggregate == "count_rows" and measure:
        raise OperationPlanError("measure must be omitted for count_rows")
    if (
        normalized.get("join_key_2")
        and normalized.get("join_key_2") == normalized.get("join_key")
    ):
        raise OperationPlanError("join keys must not contain duplicates")
    return normalized


def _scalar(value: object, field_name: str) -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
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
    if isinstance(value, Mapping):
        columns = value.get("columns")
        raw_rows = value.get("rows")
        if (
            not isinstance(columns, Sequence)
            or isinstance(columns, (str, bytes))
            or not isinstance(raw_rows, Sequence)
            or isinstance(raw_rows, (str, bytes))
        ):
            raise ValueError("operation result must provide tabular records")
        records = [
            dict(zip(columns, row, strict=True))
            for row in raw_rows
            if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
        ]
        if len(records) != len(raw_rows):
            raise ValueError("operation result rows must be sequences")
    else:
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
    rows = tuple(normalized)
    if len(canonical_json(rows).encode("utf-8")) > _MAX_RESULT_BYTES:
        raise ValueError("operation result exceeds byte bound")
    return rows


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _bound_path(value: object) -> str:
    path = getattr(value, "path", value)
    try:
        return str(__import__("os").fspath(path))
    except TypeError as exc:
        raise TypeError("registered tabular source must be path-like") from exc


def _tabular_relation(
    connection: object,
    *,
    family: str,
    source: object,
) -> tuple[str, list[object]]:
    path = _bound_path(source)
    if family == "csv":
        return "read_csv_auto(?, header=true)", [path]
    if family == "parquet":
        return "read_parquet(?)", [path]
    if family == "delta":
        try:
            from deltalake import DeltaTable
        except ModuleNotFoundError as exc:
            raise ValueError(
                "Delta operation execution requires fabric-rlm[analytics]"
            ) from exc
        dataset = DeltaTable(path).to_pyarrow_dataset()
        connection.register("_fabric_rlm_delta", dataset)
        return "_fabric_rlm_delta", []
    raise ValueError(f"unsupported tabular source family: {family}")


def _execute_tabular_aggregate(
    operation: RegisteredOperation,
    profile: object,
    source: object,
    normalized: Mapping[str, object],
) -> object:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise ValueError(
            "tabular operation execution requires fabric-rlm[analytics]"
        ) from exc

    groupby = [
        str(normalized[name])
        for name in ("groupby", "groupby_2")
        if normalized[name]
    ]
    filter_pairs = [
        (str(normalized[column]), str(normalized[value]))
        for column, value in (
            ("filter_column", "filter_value"),
            ("filter_column_2", "filter_value_2"),
        )
        if normalized[column]
    ]
    aggregate = str(normalized["aggregate"])
    if aggregate == "count_rows":
        expression = "COUNT(*)"
    else:
        expression = (
            f"{aggregate.upper()}({_quoted_identifier(str(normalized['measure']))})"
        )
    selected = [
        *(_quoted_identifier(column) for column in groupby),
        f"{expression} AS value",
    ]
    where = " AND ".join(
        f"CAST({_quoted_identifier(column)} AS VARCHAR) = ?"
        for column, _value in filter_pairs
    )
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET memory_limit = '256MB'")
        connection.execute("SET threads = 1")
        relation, relation_parameters = _tabular_relation(
            connection,
            family=profile.family,
            source=source,
        )
        query = f"SELECT {', '.join(selected)} FROM {relation}"
        if where:
            query += f" WHERE {where}"
        if groupby:
            query += " GROUP BY " + ", ".join(
                _quoted_identifier(column) for column in groupby
            )
        query += " ORDER BY " + (
            ", ".join(_quoted_identifier(column) for column in groupby)
            if groupby
            else "value"
        )
        query += f" LIMIT {operation.max_output_rows + 1}"
        return connection.execute(
            query,
            [*relation_parameters, *(value for _column, value in filter_pairs)],
        ).fetchdf()
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("registered tabular aggregate execution failed") from exc
    finally:
        connection.close()


def _sql_string_literal(value: str) -> str:
    if "\x00" in value:
        raise ValueError("filter values must not contain NUL")
    return "'" + value.replace("'", "''") + "'"


def _execute_lakehouse_aggregate(
    operation: RegisteredOperation,
    source: object,
    normalized: Mapping[str, object],
) -> object:
    query = getattr(source, "query", None)
    if not callable(query):
        raise TypeError("registered Lakehouse source cannot execute queries")
    groupby = [
        str(normalized[name])
        for name in ("groupby", "groupby_2")
        if normalized[name]
    ]
    aggregate = str(normalized["aggregate"])
    if aggregate == "count_rows":
        expression = "COUNT(*)"
    else:
        expression = (
            f"{aggregate.upper()}({_quoted_identifier(str(normalized['measure']))})"
        )
    selected = [
        *(_quoted_identifier(column) for column in groupby),
        f"{expression} AS value",
    ]
    filters = [
        (
            str(normalized[column_name]),
            str(normalized[value_name]),
        )
        for column_name, value_name in (
            ("filter_column", "filter_value"),
            ("filter_column_2", "filter_value_2"),
        )
        if normalized[column_name]
    ]
    sql = f"SELECT {', '.join(selected)} FROM data"
    if filters:
        sql += " WHERE " + " AND ".join(
            f"CAST({_quoted_identifier(column)} AS VARCHAR) = "
            f"{_sql_string_literal(value)}"
            for column, value in filters
        )
    if groupby:
        sql += " GROUP BY " + ", ".join(
            _quoted_identifier(column) for column in groupby
        )
        sql += " ORDER BY " + ", ".join(
            _quoted_identifier(column) for column in groupby
        )
    catalog_source = str(normalized["catalog_source"])
    return query(
        sql,
        sources={"data": catalog_source},
        max_rows=operation.max_output_rows + 1,
    )


def _execute_lakehouse_preaggregate_join(
    operation: RegisteredOperation,
    source: object,
    normalized: Mapping[str, object],
) -> object:
    query = getattr(source, "query", None)
    if not callable(query):
        raise TypeError("registered Lakehouse source cannot execute queries")
    join_keys = [
        str(normalized["join_key"]),
        *(
            [str(normalized["join_key_2"])]
            if normalized["join_key_2"]
            else []
        ),
    ]
    left_keys = ", ".join(_quoted_identifier(key) for key in join_keys)
    right_keys = left_keys
    join_predicate = " AND ".join(
        f"left_agg.{_quoted_identifier(key)} = "
        f"right_agg.{_quoted_identifier(key)}"
        for key in join_keys
    )
    result_keys = ", ".join(
        f"COALESCE(left_agg.{_quoted_identifier(key)}, "
        f"right_agg.{_quoted_identifier(key)}) AS {_quoted_identifier(key)}"
        for key in join_keys
    )
    order_keys = ", ".join(
        f"{_quoted_identifier(key)} DESC" for key in join_keys
    )
    row_limit = 1 if normalized["scope"] == "latest" else (
        operation.max_output_rows + 1
    )
    sql = (
        "WITH left_agg AS ("
        f"SELECT {left_keys}, "
        f"SUM({_quoted_identifier(str(normalized['left_measure']))}) "
        "AS left_value FROM left_data "
        f"GROUP BY {left_keys}"
        "), right_agg AS ("
        f"SELECT {right_keys}, "
        f"SUM({_quoted_identifier(str(normalized['right_measure']))}) "
        "AS right_value FROM right_data "
        f"GROUP BY {right_keys}"
        ") "
        f"SELECT {result_keys}, left_value, right_value "
        "FROM left_agg FULL OUTER JOIN right_agg ON "
        f"{join_predicate} ORDER BY {order_keys} LIMIT {row_limit}"
    )
    return query(
        sql,
        sources={
            "left_data": str(normalized["left_catalog_source"]),
            "right_data": str(normalized["right_catalog_source"]),
        },
        max_rows=operation.max_output_rows + 1,
    )


@dataclass(frozen=True)
class OperationExecutionResult:
    operation_id: str
    operation_version: str | None
    operation_fingerprint: str
    source_fingerprints: Mapping[str, str]
    parameters: Mapping[str, object]
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
            "parameters": dict(self.parameters),
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
    started = time.perf_counter()
    source = knowledge.bindings[source_id]
    profile = next(
        item for item in preflight.package.sources if item.source_id == source_id
    )
    if operation.operation == "semantic_model.measure":
        measure = getattr(source, "measure", None)
        if not callable(measure):
            raise TypeError(
                "registered semantic model source cannot evaluate measures"
            )
        groupby = [
            str(normalized[name])
            for name in ("groupby", "groupby_2")
            if normalized[name]
        ]
        filters = {
            str(normalized[column_name]): [str(normalized[value_name])]
            for column_name, value_name in (
                ("filter_column", "filter_value"),
                ("filter_column_2", "filter_value_2"),
            )
            if normalized[column_name]
        }
        raw_result = measure(
            str(normalized["measure"]),
            groupby=groupby or None,
            filters=filters or None,
        )
    elif operation.operation == "tabular.aggregate":
        raw_result = _execute_tabular_aggregate(
            operation,
            profile,
            source,
            normalized,
        )
    elif operation.operation == "lakehouse.aggregate":
        raw_result = _execute_lakehouse_aggregate(
            operation,
            source,
            normalized,
        )
    else:
        raw_result = _execute_lakehouse_preaggregate_join(
            operation,
            source,
            normalized,
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
        parameters=normalized,
        result_fingerprint=result_fingerprint,
        rows=rows,
        elapsed_seconds=elapsed_seconds,
    )


__all__ = [
    "OperationExecutionResult",
    "OperationPlan",
    "OperationPlanError",
    "OperationPlanFallback",
    "execute_registered_operation",
    "parse_operation_plan",
]
