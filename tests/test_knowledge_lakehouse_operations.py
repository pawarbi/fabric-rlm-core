from __future__ import annotations

from pathlib import Path

import pytest

from fabric_rlm import RLM
from fabric_rlm.knowledge_execution import execute_registered_operation
from fabric_rlm.lakehouse import LakehouseSource


def _lakehouse(tmp_path: Path) -> LakehouseSource:
    deltalake = pytest.importorskip("deltalake")
    pyarrow = pytest.importorskip("pyarrow")
    orders = tmp_path / "orders_delta"
    deltalake.write_deltalake(
        str(orders),
        pyarrow.table(
            {
                "order_id": [1, 2, 3],
                "region": ["North", "South", "South"],
                "amount": [10.5, 20.0, 5.0],
            }
        ),
    )
    table = deltalake.DeltaTable(str(orders), without_files=True)
    return LakehouseSource(
        "file:///benchmark-lakehouse",
        catalog=[
            {
                "kind": "delta",
                "name": "orders",
                "path": str(orders),
                "version": table.version(),
                "table_id": table.metadata().id,
                "columns": [
                    ["order_id", "BIGINT"],
                    ["region", "VARCHAR"],
                    ["amount", "DOUBLE"],
                ],
            }
        ],
    )


def test_learn_registers_compiler_owned_lakehouse_aggregate(
    tmp_path: Path,
) -> None:
    knowledge = RLM.learn(sources={"sales": _lakehouse(tmp_path)})

    operation = knowledge.package.operations[0]
    assert operation.operation == "lakehouse.aggregate"
    assert operation.host_implementation_id == "lakehouse.aggregate.v1"
    assert operation.parameter_schema["catalog_source"]["enum"] == (
        "orders",
    )
    assert operation.parameter_schema["measure"]["enum"] == (
        "",
        "amount",
        "order_id",
    )


def test_executes_host_compiled_lakehouse_aggregate(tmp_path: Path) -> None:
    knowledge = RLM.learn(sources={"sales": _lakehouse(tmp_path)})
    operation = knowledge.package.operations[0]

    result = execute_registered_operation(
        knowledge,
        operation_id=operation.operation_id,
        parameters={
            "catalog_source": "orders",
            "aggregate": "sum",
            "measure": "amount",
            "groupby": "region",
        },
    )

    assert result.to_packet()["rows"] == [
        {"region": "North", "value": 10.5},
        {"region": "South", "value": 25.0},
    ]


def test_lakehouse_catalog_drift_is_rejected_before_query(
    tmp_path: Path,
) -> None:
    source = _lakehouse(tmp_path)
    knowledge = RLM.learn(sources={"sales": source})
    operation = knowledge.package.operations[0]
    assert source.catalog is not None
    source.catalog[0]["columns"].append(["unexpected", "VARCHAR"])

    with pytest.raises(ValueError, match="stale.*sales"):
        execute_registered_operation(
            knowledge,
            operation_id=operation.operation_id,
            parameters={
                "catalog_source": "orders",
                "aggregate": "sum",
                "measure": "amount",
            },
        )


def test_preaggregate_join_avoids_multi_fact_fanout(tmp_path: Path) -> None:
    deltalake = pytest.importorskip("deltalake")
    pyarrow = pytest.importorskip("pyarrow")
    indoor = tmp_path / "indoor"
    outdoor = tmp_path / "outdoor"
    deltalake.write_deltalake(
        str(indoor),
        pyarrow.table(
            {
                "month": ["2016-08", "2016-08"],
                "visits": [100, 200],
            }
        ),
    )
    deltalake.write_deltalake(
        str(outdoor),
        pyarrow.table(
            {
                "month": ["2016-08", "2016-08", "2016-08"],
                "visits": [10, 20, 30],
            }
        ),
    )
    indoor_table = deltalake.DeltaTable(str(indoor), without_files=True)
    outdoor_table = deltalake.DeltaTable(str(outdoor), without_files=True)
    source = LakehouseSource(
        "file:///tourism-lakehouse",
        catalog=[
            {
                "kind": "delta",
                "name": "indoor",
                "path": str(indoor),
                "version": indoor_table.version(),
                "table_id": indoor_table.metadata().id,
                "columns": [["month", "VARCHAR"], ["visits", "BIGINT"]],
            },
            {
                "kind": "delta",
                "name": "outdoor",
                "path": str(outdoor),
                "version": outdoor_table.version(),
                "table_id": outdoor_table.metadata().id,
                "columns": [["month", "VARCHAR"], ["visits", "BIGINT"]],
            },
        ],
    )
    knowledge = RLM.learn(sources={"tourism": source})
    operation = next(
        operation
        for operation in knowledge.package.operations
        if operation.operation == "lakehouse.preaggregate_join"
    )

    result = execute_registered_operation(
        knowledge,
        operation_id=operation.operation_id,
        parameters={
            "left_catalog_source": "indoor",
            "right_catalog_source": "outdoor",
            "left_measure": "visits",
            "right_measure": "visits",
            "join_key": "month",
            "scope": "latest",
        },
    )

    packet = result.to_packet()
    assert packet["rows"] == [
        {
            "month": "2016-08",
            "left_value": 300,
            "right_value": 60,
        }
    ]
    assert packet["parameters"]["scope"] == "latest"
    raw_join_left_sum = sum([100, 200]) * 3
    raw_join_right_sum = sum([10, 20, 30]) * 2
    assert raw_join_left_sum == 900
    assert raw_join_right_sum == 120
