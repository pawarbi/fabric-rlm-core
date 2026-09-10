from __future__ import annotations

from pathlib import Path

import pytest

from fabric_rlm import File, RLM
from fabric_rlm.knowledge_execution import (
    OperationPlanError,
    OperationResultTooLarge,
    execute_registered_operation,
)


def _orders_csv(tmp_path: Path) -> Path:
    path = tmp_path / "orders.csv"
    path.write_text(
        "order_id,region,amount,customer_email\n"
        "1,North,10.5,first@example.com\n"
        "2,South,20.0,second@example.com\n"
        "3,West,30.0,third@example.com\n",
        encoding="utf-8",
    )
    return path


def test_learn_registers_bounded_tabular_aggregate_for_csv(
    tmp_path: Path,
) -> None:
    source = _orders_csv(tmp_path)

    knowledge = RLM.learn(sources={"orders": File(source)})

    operation = knowledge.package.operations[0]
    assert operation.operation_id == "orders.tabular.aggregate.v1"
    assert operation.operation == "tabular.aggregate"
    assert operation.host_implementation_id == "tabular.aggregate.v1"
    assert operation.parameter_schema["aggregate"]["enum"] == (
        "avg",
        "count_rows",
        "sum",
    )
    assert operation.parameter_schema["measure"]["enum"] == (
        "",
        "amount",
        "order_id",
    )
    assert operation.parameter_schema["groupby"]["enum"] == (
        "",
        "amount",
        "order_id",
        "region",
    )
    assert "customer_email" not in operation.parameter_schema["groupby"]["enum"]
    assert operation.max_output_rows == 100


def test_operation_allowlists_exclude_camelcase_pii_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "contacts.csv"
    path.write_text(
        "CustomerEmail,contactPhone,amount\n"
        "person@example.com,5550100,10.0\n",
        encoding="utf-8",
    )

    operation = RLM.learn(sources={"contacts": path}).package.operations[0]

    allowed = operation.parameter_schema["groupby"]["enum"]
    assert "CustomerEmail" not in allowed
    assert "contactPhone" not in allowed


def test_inexact_large_file_does_not_register_an_operation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large.csv"
    path.write_text(
        "amount\n" + ("1\n" * 600_000),
        encoding="utf-8",
    )

    knowledge = RLM.learn(sources={"large": path})

    assert knowledge.package.sources[0].diagnostics["snapshot_exact"] is False
    assert knowledge.package.operations == ()


def test_executes_compiler_owned_csv_aggregate(tmp_path: Path) -> None:
    source = _orders_csv(tmp_path)
    knowledge = RLM.learn(sources={"orders": source})

    result = execute_registered_operation(
        knowledge,
        operation_id="orders.tabular.aggregate.v1",
        parameters={
            "aggregate": "sum",
            "measure": "amount",
            "groupby": "region",
            "filter_column": "region",
            "filter_value": "West",
        },
    )

    assert result.to_packet()["rows"] == [
        {"region": "West", "value": 30.0},
    ]
    assert result.to_packet()["parameters"]["measure"] == "amount"
    assert result.audit_status == "passed"


def test_count_rows_requires_no_measure_and_sum_requires_numeric_measure(
    tmp_path: Path,
) -> None:
    source = _orders_csv(tmp_path)
    knowledge = RLM.learn(sources={"orders": source})

    count = execute_registered_operation(
        knowledge,
        operation_id="orders.tabular.aggregate.v1",
        parameters={"aggregate": "count_rows"},
    )
    assert count.to_packet()["rows"] == [{"value": 3}]

    with pytest.raises(ValueError, match="measure is required"):
        execute_registered_operation(
            knowledge,
            operation_id="orders.tabular.aggregate.v1",
            parameters={"aggregate": "sum"},
        )


def test_file_aggregate_rejects_drift_before_execution(tmp_path: Path) -> None:
    source = _orders_csv(tmp_path)
    knowledge = RLM.learn(sources={"orders": source})
    source.write_text(
        source.read_text(encoding="utf-8")
        + "4,North,40.0,fourth@example.com\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stale.*orders"):
        execute_registered_operation(
            knowledge,
            operation_id="orders.tabular.aggregate.v1",
            parameters={"aggregate": "sum", "measure": "amount"},
        )


def test_csv_query_values_are_bound_not_interpolated(tmp_path: Path) -> None:
    source = _orders_csv(tmp_path)
    knowledge = RLM.learn(sources={"orders": source})

    result = execute_registered_operation(
        knowledge,
        operation_id="orders.tabular.aggregate.v1",
        parameters={
            "aggregate": "sum",
            "measure": "amount",
            "filter_column": "region",
            "filter_value": "West' OR 1=1 --",
        },
    )

    assert result.to_packet()["rows"] == [{"value": None}]


def test_executes_parquet_aggregate(tmp_path: Path) -> None:
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    source = tmp_path / "orders.parquet"
    pandas.DataFrame(
        {
            "region": ["North", "South", "South"],
            "amount": [10.5, 20.0, 5.0],
        }
    ).to_parquet(source, index=False)
    knowledge = RLM.learn(sources={"orders": source})

    result = execute_registered_operation(
        knowledge,
        operation_id="orders.tabular.aggregate.v1",
        parameters={
            "aggregate": "sum",
            "measure": "amount",
            "groupby": "region",
        },
    )

    assert result.to_packet()["rows"] == [
        {"region": "North", "value": 10.5},
        {"region": "South", "value": 25.0},
    ]


def test_delta_aggregate_reads_only_current_table_state(tmp_path: Path) -> None:
    deltalake = pytest.importorskip("deltalake")
    pyarrow = pytest.importorskip("pyarrow")
    source = tmp_path / "orders_delta"
    deltalake.write_deltalake(
        str(source),
        pyarrow.table({"region": ["Old"], "amount": [1.0]}),
    )
    deltalake.write_deltalake(
        str(source),
        pyarrow.table({"region": ["Current"], "amount": [99.0]}),
        mode="overwrite",
    )
    knowledge = RLM.learn(sources={"orders": source})

    result = execute_registered_operation(
        knowledge,
        operation_id="orders.tabular.aggregate.v1",
        parameters={
            "aggregate": "sum",
            "measure": "amount",
            "groupby": "region",
        },
    )

    assert result.to_packet()["rows"] == [
        {"region": "Current", "value": 99.0},
    ]


# ------------------------------------------------------- typed filters --
def _ratings_csv(tmp_path: Path) -> Path:
    path = tmp_path / "ratings.csv"
    path.write_text(
        "product,rating,revenue,active\n"
        "A,1.0,100,true\n"
        "B,2.5,50,false\n"
        "C,1.0,25,true\n",
        encoding="utf-8",
    )
    return path


def _total(knowledge, *, measure: str, column: str, value: str) -> object:
    result = execute_registered_operation(
        knowledge,
        operation_id="ratings.tabular.aggregate.v1",
        parameters={
            "aggregate": "sum",
            "measure": measure,
            "filter_column": column,
            "filter_value": value,
        },
    )
    return result.to_packet()["rows"][0]["value"]


def test_csv_filters_compare_numbers_as_numbers(tmp_path: Path) -> None:
    # rating is 1.0 in the file: "1" and "1.0" are the same number, and the
    # aggregate must not depend on which spelling the planner chose.
    knowledge = RLM.learn(sources={"ratings": _ratings_csv(tmp_path)})
    assert _total(knowledge, measure="revenue", column="rating", value="1") == 125
    assert _total(knowledge, measure="revenue", column="rating", value="1.0") == 125
    assert _total(knowledge, measure="rating", column="revenue", value="100") == 1.0
    assert _total(knowledge, measure="rating", column="revenue", value="100.0") == 1.0
    assert _total(knowledge, measure="revenue", column="active", value="true") == 125
    assert _total(knowledge, measure="revenue", column="product", value="B") == 50
    # a value that does not fit the column is refused before any query runs
    with pytest.raises(OperationPlanError, match="not an integer for column revenue"):
        _total(knowledge, measure="rating", column="revenue", value="abc")
    with pytest.raises(OperationPlanError, match="not a number for column rating"):
        _total(knowledge, measure="revenue", column="rating", value="high")
    with pytest.raises(OperationPlanError, match="not a boolean for column active"):
        _total(knowledge, measure="revenue", column="active", value="yes")
    # the packet echoes the parameter as given; the typed value is binding-only
    result = execute_registered_operation(
        knowledge,
        operation_id="ratings.tabular.aggregate.v1",
        parameters={"aggregate": "sum", "measure": "revenue", "filter_column": "rating", "filter_value": "1"},
    )
    assert result.to_packet()["parameters"]["filter_value"] == "1"


def test_parquet_filters_bind_the_declared_column_type(tmp_path: Path) -> None:
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    path = tmp_path / "ratings.parquet"
    pandas.DataFrame(
        {"product": ["A", "B"], "rating": [1.0, 2.5], "revenue": [100, 50]}
    ).to_parquet(path, index=False)
    knowledge = RLM.learn(sources={"ratings": path})
    assert _total(knowledge, measure="revenue", column="rating", value="1") == 100
    assert _total(knowledge, measure="revenue", column="rating", value="1.0") == 100
    assert _total(knowledge, measure="rating", column="revenue", value="100.0") == 1.0
    with pytest.raises(OperationPlanError, match="not an integer for column revenue"):
        _total(knowledge, measure="rating", column="revenue", value="1.5")


def _many_orders_csv(tmp_path: Path, rows: int) -> Path:
    path = tmp_path / "many_orders.csv"
    lines = ["order_id,region,amount"]
    for index in range(rows):
        lines.append(f"{index},R{index % 3},{index}.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_group_by_high_cardinality_key_reports_a_bounded_plan_failure(
    tmp_path: Path,
) -> None:
    """A legal plan whose result overflows the row bound is a plan-level
    rejection, not a host-contract violation.

    ``groupby`` accepts ``order_id`` because it appears in the operation's own
    enum, and a declared grain lesson actively steers the model toward it. The
    result then exceeds ``max_output_rows``. That is a recoverable planning
    mistake, so it must be an OperationPlanError the runtime can fall back
    from, not a bare ValueError that aborts the whole run.
    """

    source = _many_orders_csv(tmp_path, rows=150)
    knowledge = RLM.learn(sources={"orders": source})
    operation = knowledge.package.operations[0]
    assert operation.max_output_rows == 100
    assert "order_id" in operation.parameter_schema["groupby"]["enum"]

    parameters = dict(operation.parameter_defaults)
    parameters.update({"aggregate": "sum", "measure": "amount", "groupby": "order_id"})

    with pytest.raises(OperationResultTooLarge, match="exceeds row bound"):
        execute_registered_operation(
            knowledge,
            operation_id=operation.operation_id,
            parameters=parameters,
        )


def test_bounded_group_by_still_executes(tmp_path: Path) -> None:
    source = _many_orders_csv(tmp_path, rows=150)
    knowledge = RLM.learn(sources={"orders": source})
    operation = knowledge.package.operations[0]

    parameters = dict(operation.parameter_defaults)
    parameters.update({"aggregate": "sum", "measure": "amount", "groupby": "region"})

    result = execute_registered_operation(
        knowledge,
        operation_id=operation.operation_id,
        parameters=parameters,
    )

    assert result.to_packet()["row_count"] == 3


def test_result_bound_error_is_recoverable_and_still_a_value_error() -> None:
    """The runtime falls back on OperationPlanError, and callers that predate
    this distinction catch ValueError. The new type must satisfy both."""

    assert issubclass(OperationResultTooLarge, OperationPlanError)
    assert issubclass(OperationResultTooLarge, ValueError)


def test_column_bound_overflow_stays_fatal(tmp_path: Path) -> None:
    """Column count is chosen by the host, not by the plan, so an overflow is a
    contract violation that must keep failing closed rather than degrading to
    ordinary execution."""

    import fabric_rlm.knowledge_execution as execution

    source = _many_orders_csv(tmp_path, rows=5)
    knowledge = RLM.learn(sources={"orders": source})
    operation = knowledge.package.operations[0]
    wide = [{f"column_{index}": index for index in range(operation.max_output_columns + 1)}]

    with pytest.raises(ValueError, match="exceeds column bound") as caught:
        execution._result_rows(
            {"columns": list(wide[0]), "rows": [list(wide[0].values())]},
            operation,
        )

    assert not isinstance(caught.value, OperationPlanError)


def test_truncated_host_result_is_recoverable(tmp_path: Path) -> None:
    """A host that signals overflow via ``truncated`` must be recoverable.

    The Lakehouse operation deliberately over-fetches by one row so that a
    period with more groups than the operation may return "is reported as
    truncated, never trimmed". That makes ``truncated`` the Lakehouse spelling
    of exactly the condition ``max_output_rows`` covers for in-memory results,
    and it follows from the model's choice of grain, so it is a recoverable
    planning mistake rather than a host-contract violation.

    Pinned because the in-memory row bound was made recoverable while this
    path kept raising a bare ValueError, which ``runtime.py`` re-raises after
    recording ``reason="audit_failed"`` -- aborting the whole run before the
    agent loop starts. That is the ``turns=None`` crash observed on real
    Fabric Delta data.
    """

    import fabric_rlm.knowledge_execution as execution

    source = _many_orders_csv(tmp_path, rows=5)
    knowledge = RLM.learn(sources={"orders": source})
    operation = knowledge.package.operations[0]

    with pytest.raises(ValueError, match="truncated") as caught:
        execution._result_rows(
            {"truncated": True, "columns": ["region"], "rows": [["R0"]]},
            operation,
        )

    # The type is the whole contract: runtime.py falls back on
    # OperationResultTooLarge and re-raises on a bare ValueError.
    assert isinstance(caught.value, OperationResultTooLarge)
    assert isinstance(caught.value, OperationPlanError)
