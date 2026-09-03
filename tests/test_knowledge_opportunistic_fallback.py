"""Regression coverage for opportunistic registered-operation use."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fabric_rlm import File, RLM
from fabric_rlm import knowledge_execution
from fabric_rlm.lakehouse import LakehouseSource
from fabric_rlm.runtime import _eligible_knowledge_operations
from fabric_rlm.semantic_model import SemanticModel


_BROAD_BUSINESS_REQUEST = (
    "Assess the state of the business and recommend where we should invest."
)
_GENERIC_OPERATION_NAMES = (
    "semantic_model.measure",
    "tabular.aggregate",
    "lakehouse.aggregate",
    "lakehouse.preaggregate_join",
)


class _Frame:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = [dict(row) for row in rows]

    def to_dict(self, orient: str = "dict") -> list[dict[str, object]]:
        assert orient == "records"
        return [dict(row) for row in self._rows]


class _SemanticModel(SemanticModel):
    def __init__(self) -> None:
        object.__setattr__(self, "dataset", "Sales Model")
        object.__setattr__(self, "workspace", "Analytics")
        object.__setattr__(self, "credential_provider", None)
        object.__setattr__(self, "validate", False)
        object.__setattr__(self, "_source_access_failed", False)
        object.__setattr__(self, "measure_calls", [])

    def tables(self) -> _Frame:
        return _Frame([{"Name": "Sales"}])

    def columns(self) -> _Frame:
        return _Frame(
            [
                {"Table Name": "Sales", "Column Name": "Month"},
                {"Table Name": "Geography", "Column Name": "Region"},
            ]
        )

    def measures(self) -> _Frame:
        return _Frame(
            [{"Table Name": "Measures", "Measure Name": "Net Revenue"}]
        )

    def relationships(self) -> _Frame:
        return _Frame([])

    def measure(self, measure, groupby=None, filters=None) -> _Frame:
        self.measure_calls.append(
            {
                "measure": measure,
                "groupby": groupby,
                "filters": filters,
            }
        )
        return _Frame(
            [
                {"Geography[Region]": "East", "[Net Revenue]": 125.0},
                {"Geography[Region]": "West", "[Net Revenue]": 75.0},
            ]
        )


class _SequenceLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []

    def __call__(self, *, messages: list[dict[str, str]]) -> str:
        self.messages.append([dict(message) for message in messages])
        response = self._responses[self.calls]
        self.calls += 1
        return response


def _code(source: str) -> str:
    return f"```python\n{source}\n```"


def _csv_source(tmp_path: Path) -> File:
    path = tmp_path / "orders.csv"
    path.write_text(
        "region,amount\nNorth,10.5\nSouth,20.0\nWest,30.0\n",
        encoding="utf-8",
    )
    return File(path)


def _parquet_source(tmp_path: Path) -> Path:
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    path = tmp_path / "orders.parquet"
    pandas.DataFrame(
        {"region": ["North", "South", "West"], "amount": [10.5, 20.0, 30.0]}
    ).to_parquet(path, index=False)
    return path


def _delta_source(tmp_path: Path) -> Path:
    deltalake = pytest.importorskip("deltalake")
    pyarrow = pytest.importorskip("pyarrow")
    path = tmp_path / "orders_delta"
    deltalake.write_deltalake(
        str(path),
        pyarrow.table(
            {"region": ["North", "South", "West"], "amount": [10.5, 20.0, 30.0]}
        ),
    )
    return path


def _lakehouse_source(tmp_path: Path) -> LakehouseSource:
    deltalake = pytest.importorskip("deltalake")
    pyarrow = pytest.importorskip("pyarrow")
    path = tmp_path / "orders_lakehouse_delta"
    deltalake.write_deltalake(
        str(path),
        pyarrow.table(
            {"region": ["North", "South"], "amount": [10.5, 25.0]}
        ),
    )
    returns_path = tmp_path / "returns_lakehouse_delta"
    deltalake.write_deltalake(
        str(returns_path),
        pyarrow.table(
            {"region": ["North", "South"], "amount": [1.0, 2.0]}
        ),
    )
    table = deltalake.DeltaTable(str(path), without_files=True)
    returns = deltalake.DeltaTable(str(returns_path), without_files=True)
    return LakehouseSource(
        "file:///knowledge-fallback-lakehouse",
        catalog=[
            {
                "kind": "delta",
                "name": "orders",
                "path": str(path),
                "version": table.version(),
                "table_id": table.metadata().id,
                "columns": [
                    ["region", "VARCHAR"],
                    ["amount", "DOUBLE"],
                ],
            },
            {
                "kind": "delta",
                "name": "returns",
                "path": str(returns_path),
                "version": returns.version(),
                "table_id": returns.metadata().id,
                "columns": [
                    ["region", "VARCHAR"],
                    ["amount", "DOUBLE"],
                ],
            },
        ],
    )


def _source_case(
    source_type: str,
    tmp_path: Path,
) -> tuple[str, object, str, str, dict[str, object], str, float]:
    if source_type == "semantic_model":
        return (
            "business_model",
            _SemanticModel(),
            "business_model.dataset",
            "Sales Model",
            {"measure": "Net Revenue", "groupby": "Geography[Region]"},
            "sum(row['[Net Revenue]'] for row in knowledge_result['rows'])",
            200.0,
        )
    if source_type == "csv":
        source = _csv_source(tmp_path)
        return (
            "orders",
            source,
            "orders.name",
            "orders.csv",
            {"aggregate": "sum", "measure": "amount"},
            "knowledge_result['rows'][0]['value']",
            60.5,
        )
    if source_type == "parquet":
        source = _parquet_source(tmp_path)
        return (
            "orders",
            source,
            "orders.name",
            "orders.parquet",
            {"aggregate": "sum", "measure": "amount"},
            "knowledge_result['rows'][0]['value']",
            60.5,
        )
    if source_type == "delta":
        source = _delta_source(tmp_path)
        return (
            "orders",
            source,
            "orders.name",
            "orders_delta",
            {"aggregate": "sum", "measure": "amount"},
            "knowledge_result['rows'][0]['value']",
            60.5,
        )
    if source_type == "lakehouse":
        source = _lakehouse_source(tmp_path)
        return (
            "orders_lakehouse",
            source,
            "orders_lakehouse.catalog[0]['name']",
            "orders",
            {
                "catalog_source": "orders",
                "aggregate": "sum",
                "measure": "amount",
            },
            "knowledge_result['rows'][0]['value']",
            35.5,
        )
    raise AssertionError(f"unknown source type: {source_type}")


def _aggregate_operation(knowledge, *, catalog_source: str | None = None):
    return next(
        operation
        for operation in knowledge.package.operations
        if operation.operation
        in {"semantic_model.measure", "tabular.aggregate", "lakehouse.aggregate"}
        and (
            catalog_source is None
            or operation.parameter_schema["catalog_source"]["enum"]
            == (catalog_source,)
        )
    )


def test_generic_operations_are_ineligible_for_multi_evidence_requests() -> None:
    operations = [{"operation": name} for name in _GENERIC_OPERATION_NAMES]
    specialized = {"operation": "business_health.evidence_plan"}

    broad, broad_reason = _eligible_knowledge_operations(
        _BROAD_BUSINESS_REQUEST,
        [*operations, specialized],
    )
    multi_metric, multi_metric_reason = _eligible_knowledge_operations(
        "What were revenue and order count in June 2025?",
        operations,
    )
    unresolved_time, time_reason = _eligible_knowledge_operations(
        "What was revenue last month?",
        operations,
    )
    narrow, narrow_reason = _eligible_knowledge_operations(
        "What was risk-adjusted revenue in June 2025?",
        operations,
    )
    no_task, no_task_reason = _eligible_knowledge_operations(None, operations)

    assert broad == [specialized]
    assert broad_reason == "generic_knowledge_operation_ineligible_for_broad_analysis"
    assert multi_metric == []
    assert multi_metric_reason == (
        "generic_knowledge_operation_ineligible_for_multi_metric_request"
    )
    assert unresolved_time == []
    assert time_reason == (
        "generic_knowledge_operation_ineligible_for_unresolved_time_scope"
    )
    assert narrow == operations
    assert narrow_reason is None
    assert no_task == operations
    assert no_task_reason is None


@pytest.mark.parametrize(
    "source_type",
    ("semantic_model", "csv", "parquet", "delta", "lakehouse"),
)
def test_broad_request_preserves_every_source_family_for_cold_fallback(
    source_type: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias, source, source_expression, expected, _parameters, _result_expression, _value = (
        _source_case(source_type, tmp_path)
    )
    knowledge = RLM.learn(sources={alias: source})
    if source_type == "lakehouse":
        assert any(
            operation.operation == "lakehouse.preaggregate_join"
            for operation in knowledge.package.operations
        )

    def unexpected_host_execution(*_args, **_kwargs) -> None:
        raise AssertionError("broad task must not execute a generic operation")

    monkeypatch.setattr(
        knowledge_execution,
        "execute_registered_operation",
        unexpected_host_execution,
    )
    lm = _SequenceLM(
        [_code(f"assert {source_expression}\nSUBMIT(answer={source_expression})")]
    )

    result = RLM.task(
        _BROAD_BUSINESS_REQUEST,
        knowledge=knowledge,
        outputs={"answer": str},
        lm=lm,
        max_turns=1,
        timeout=10,
    ).run()

    assert result.payload == {"answer": expected}
    assert lm.calls == 1
    metadata = result.trajectory.metadata
    assert metadata["knowledge_mode"] == "fallback_capability_insufficient"
    assert metadata["knowledge_fallback_reason"] == (
        "generic_knowledge_operation_ineligible_for_broad_analysis"
    )
    assert "operation_selection_lm_calls" not in metadata
    prompt = "\n".join(message["content"] for message in lm.messages[0])
    assert alias in prompt
    assert "Registered operations:" not in prompt
    assert "knowledge_result" not in prompt


@pytest.mark.parametrize(
    "source_type",
    ("semantic_model", "csv", "parquet", "delta", "lakehouse"),
)
def test_narrow_request_keeps_the_learned_fast_path_for_every_source_family(
    source_type: str,
    tmp_path: Path,
) -> None:
    alias, source, _source_expression, _expected, parameters, result_expression, value = (
        _source_case(source_type, tmp_path)
    )
    knowledge = RLM.learn(sources={alias: source})
    operation = _aggregate_operation(
        knowledge,
        **({"catalog_source": "orders"} if source_type == "lakehouse" else {}),
    )
    lm = _SequenceLM(
        [
            json.dumps(
                {
                    "operation_id": operation.operation_id,
                    "parameters": parameters,
                }
            ),
            _code(f"SUBMIT(answer={result_expression})"),
        ]
    )

    result = RLM.task(
        "What is the total recorded value in this approved snapshot?",
        knowledge=knowledge,
        outputs={"answer": float},
        lm=lm,
        max_turns=1,
        timeout=20,
    ).run()

    assert result.payload == {"answer": value}
    assert lm.calls == 2
    metadata = result.trajectory.metadata
    assert metadata["knowledge_mode"] == "registered_operation"
    assert metadata["knowledge_available"] is True
    assert metadata["knowledge_used"] is True
    assert metadata["operation_id"] == operation.operation_id
    assert metadata["operation_audit_status"] == "passed"
    assert metadata["operation_selection_lm_calls"] == 1
    synthesis_prompt = "\n".join(
        message["content"] for message in lm.messages[1]
    )
    assert "knowledge_result" in synthesis_prompt
    assert f"namespace ({alias}" not in synthesis_prompt


def test_question_input_is_considered_when_direct_rlm_has_no_inline_task() -> None:
    model = _SemanticModel()
    knowledge = RLM.learn(sources={"business_model": model})
    lm = _SequenceLM(
        [_code("SUBMIT(answer=business_model.dataset)")]
    )

    result = RLM(
        signature="question -> answer",
        knowledge=knowledge,
        lm=lm,
        max_turns=1,
        timeout=10,
    ).run({"question": _BROAD_BUSINESS_REQUEST})

    assert result.payload == {"answer": "Sales Model"}
    assert lm.calls == 1
    assert result.trajectory.metadata["knowledge_mode"] == (
        "fallback_capability_insufficient"
    )
