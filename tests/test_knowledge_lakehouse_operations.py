from __future__ import annotations

from pathlib import Path

import pytest

from fabric_rlm import RLM
from fabric_rlm.knowledge_execution import (
    OperationPlanError,
    execute_registered_operation,
)
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


def test_underlying_delta_version_drift_is_rejected(
    tmp_path: Path,
) -> None:
    deltalake = pytest.importorskip("deltalake")
    pyarrow = pytest.importorskip("pyarrow")
    source = _lakehouse(tmp_path)
    knowledge = RLM.learn(sources={"sales": source})
    operation = knowledge.package.operations[0]
    assert source.catalog is not None
    path = str(source.catalog[0]["path"])
    deltalake.write_deltalake(
        path,
        pyarrow.table(
            {
                "order_id": [99],
                "region": ["Changed"],
                "amount": [99999.0],
            }
        ),
        mode="overwrite",
    )

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

    with pytest.raises(
        OperationPlanError,
        match="latest scope requires a temporal",
    ):
        execute_registered_operation(
            knowledge,
            operation_id=operation.operation_id,
            parameters={
                "left_catalog_source": "indoor",
                "right_catalog_source": "outdoor",
                "left_measure": "visits",
                "right_measure": "visits",
                "join_key": "visits",
                "scope": "latest",
            },
        )


def test_truncated_lakehouse_result_fails_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _lakehouse(tmp_path)
    knowledge = RLM.learn(sources={"sales": source})
    operation = knowledge.package.operations[0]

    monkeypatch.setattr(
        LakehouseSource,
        "query",
        lambda *_args, **_kwargs: {
            "columns": ["value"],
            "rows": [[30.5]],
            "truncated": True,
        },
    )

    with pytest.raises(ValueError, match="result was truncated"):
        execute_registered_operation(
            knowledge,
            operation_id=operation.operation_id,
            parameters={
                "catalog_source": "orders",
                "aggregate": "sum",
                "measure": "amount",
            },
        )


# --------------------------------------------------- latest scope semantics --
def _two_fact_lakehouse(tmp_path: Path, *, regions_in_latest: int = 2) -> LakehouseSource:
    deltalake = pytest.importorskip("deltalake")
    pyarrow = pytest.importorskip("pyarrow")
    latest_regions = (
        ["A", "B"]
        if regions_in_latest == 2
        else [f"R{index:03d}" for index in range(regions_in_latest)]
    )
    months = ["2025-01"] + ["2026-09"] * len(latest_regions)
    regions = ["Z"] + latest_regions
    years = [month[:4] for month in months]
    indoor = tmp_path / "indoor"
    outdoor = tmp_path / "outdoor"
    deltalake.write_deltalake(
        str(indoor),
        pyarrow.table(
            {
                "month": months,
                "region": regions,
                "year": years,
                "visits": [100] + [10 * (index + 1) for index in range(len(latest_regions))],
            }
        ),
    )
    deltalake.write_deltalake(
        str(outdoor),
        pyarrow.table(
            {
                "month": months,
                "region": regions,
                "year": years,
                "visits": [1] + [index + 1 for index in range(len(latest_regions))],
            }
        ),
    )
    columns = [
        ["month", "VARCHAR"],
        ["region", "VARCHAR"],
        ["year", "VARCHAR"],
        ["visits", "BIGINT"],
    ]
    entries = []
    for name, path in (("indoor", indoor), ("outdoor", outdoor)):
        table = deltalake.DeltaTable(str(path), without_files=True)
        entries.append(
            {
                "kind": "delta",
                "name": name,
                "path": str(path),
                "version": table.version(),
                "table_id": table.metadata().id,
                "columns": columns,
            }
        )
    return LakehouseSource("file:///tourism-lakehouse", catalog=entries)


def _join_parameters(**overrides):
    parameters = {
        "left_catalog_source": "indoor",
        "right_catalog_source": "outdoor",
        "left_measure": "visits",
        "right_measure": "visits",
        "scope": "latest",
    }
    parameters.update(overrides)
    return parameters


def test_latest_scope_returns_every_group_of_the_latest_period(tmp_path: Path) -> None:
    # Rows: Z in 2025-01, A and B in 2026-09. "Latest" is the latest period,
    # whichever position the period key was given in, and every group in it.
    knowledge = RLM.learn(sources={"tourism": _two_fact_lakehouse(tmp_path)})
    operation = next(
        op for op in knowledge.package.operations if op.operation == "lakehouse.preaggregate_join"
    )
    expected = [
        {"month": "2026-09", "region": "A", "left_value": 10, "right_value": 1},
        {"month": "2026-09", "region": "B", "left_value": 20, "right_value": 2},
    ]
    for keys in (
        {"join_key": "month", "join_key_2": "region"},
        {"join_key": "region", "join_key_2": "month"},
    ):
        result = execute_registered_operation(
            knowledge, operation_id=operation.operation_id, parameters=_join_parameters(**keys)
        )
        rows = [
            {name: row[name] for name in ("month", "region", "left_value", "right_value")}
            for row in result.to_packet()["rows"]
        ]
        assert rows == expected, keys
        assert result.audit_status == "passed"

    # "all" is untouched: every period, ordered as before
    everything = execute_registered_operation(
        knowledge,
        operation_id=operation.operation_id,
        parameters=_join_parameters(join_key="month", join_key_2="region", scope="all"),
    )
    assert [row["month"] for row in everything.to_packet()["rows"]] == ["2026-09", "2026-09", "2025-01"]

    # no period key, or two of them, cannot define "latest"
    with pytest.raises(OperationPlanError, match="latest scope requires a temporal"):
        execute_registered_operation(
            knowledge, operation_id=operation.operation_id, parameters=_join_parameters(join_key="region")
        )
    with pytest.raises(OperationPlanError, match="exactly one"):
        execute_registered_operation(
            knowledge,
            operation_id=operation.operation_id,
            parameters=_join_parameters(join_key="month", join_key_2="year"),
        )


def test_latest_period_with_too_many_groups_fails_the_row_bound(tmp_path: Path) -> None:
    knowledge = RLM.learn(sources={"tourism": _two_fact_lakehouse(tmp_path, regions_in_latest=101)})
    operation = next(
        op for op in knowledge.package.operations if op.operation == "lakehouse.preaggregate_join"
    )
    assert operation.max_output_rows == 100
    with pytest.raises(ValueError, match="exceeds row bound"):
        execute_registered_operation(
            knowledge,
            operation_id=operation.operation_id,
            parameters=_join_parameters(join_key="month", join_key_2="region"),
        )


# ------------------------------------------------ typed lakehouse filters --
def test_lakehouse_aggregate_filters_bind_the_declared_column_type(tmp_path: Path) -> None:
    knowledge = RLM.learn(sources={"sales": _lakehouse(tmp_path)})
    operation = knowledge.package.operations[0]

    def total(column: str, value: str, measure: str = "amount") -> object:
        result = execute_registered_operation(
            knowledge,
            operation_id=operation.operation_id,
            parameters={
                "catalog_source": "orders",
                "aggregate": "sum",
                "measure": measure,
                "filter_column": column,
                "filter_value": value,
            },
        )
        return result.to_packet()["rows"][0]["value"]

    # DOUBLE column: both spellings are the same number
    assert total("amount", "20") == 20.0
    assert total("amount", "20.0") == 20.0
    # BIGINT column: a decimal spelling of an integer is that integer
    assert total("order_id", "1") == 10.5
    assert total("order_id", "1.0") == 10.5
    # VARCHAR column: unchanged text comparison
    assert total("region", "North") == 10.5
    # a value that is not of the column's type is refused before any query
    with pytest.raises(OperationPlanError, match="not an integer for column order_id"):
        total("order_id", "1.5")
    with pytest.raises(OperationPlanError, match="not a number for column amount"):
        total("amount", "twenty")
