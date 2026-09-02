from __future__ import annotations

from pathlib import Path

import pytest

from fabric_rlm import File, RLM
from fabric_rlm.knowledge_execution import execute_registered_operation


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
