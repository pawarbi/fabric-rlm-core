from __future__ import annotations

from pathlib import Path

import pytest

from fabric_rlm import RLM, load_knowledge
from fabric_rlm.knowledge_execution import execute_registered_operation
from fabric_rlm.semantic_model import SemanticModel


class FakeFrame:
    def __init__(self, rows):
        self._rows = [dict(row) for row in rows]

    def to_dict(self, orient="dict"):
        assert orient == "records"
        return [dict(row) for row in self._rows]


class FakeSemanticModel(SemanticModel):
    def __init__(self):
        object.__setattr__(self, "dataset", "Sales Model")
        object.__setattr__(self, "workspace", "Analytics")
        object.__setattr__(self, "validate", False)
        self.measure_calls = []
        self.measure_rows = [
            {"Geography[Region]": "East", "[Net Revenue]": 125.0},
            {"Geography[Region]": "West", "[Net Revenue]": 75.0},
        ]

    def tables(self):
        return FakeFrame([{"Name": "Sales"}, {"Name": "Geography"}])

    def columns(self):
        return FakeFrame(
            [
                {"Table Name": "Sales", "Column Name": "Month"},
                {"Table Name": "Geography", "Column Name": "Region"},
            ]
        )

    def measures(self):
        return FakeFrame(
            [
                {"Table Name": "Measures", "Measure Name": "Net Revenue"},
                {"Table Name": "Measures", "Measure Name": "Order Count"},
            ]
        )

    def relationships(self):
        return FakeFrame([])

    def measure(self, measure, groupby=None, filters=None):
        self.measure_calls.append(
            {
                "measure": measure,
                "groupby": groupby,
                "filters": filters,
            }
        )
        return FakeFrame(self.measure_rows)


def test_learn_registers_bounded_semantic_model_measure_operation() -> None:
    knowledge = RLM.learn(sources={"sales": FakeSemanticModel()})

    assert len(knowledge.package.operations) == 1
    operation = knowledge.package.operations[0]
    assert operation.operation_id == "sales.semantic_model.measure.v1"
    assert operation.operation == "semantic_model.measure"
    assert operation.required_sources == ("sales",)
    assert operation.host_implementation_id == "semantic_model.measure.v1"
    assert operation.operation_version == "1"
    assert operation.status == "active"
    assert operation.parameter_schema["measure"]["enum"] == (
        "Net Revenue",
        "Order Count",
    )
    assert operation.parameter_schema["groupby"]["enum"] == (
        "",
        "Geography[Region]",
        "Sales[Month]",
    )
    assert operation.parameter_schema["filter_column"]["enum"] == (
        "",
        "Geography[Region]",
        "Sales[Month]",
    )
    assert operation.max_output_rows == 100
    assert operation.max_output_columns == 20


def test_semantic_measure_operation_survives_save_and_rebind(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "knowledge.json"
    learned = RLM.learn(
        sources={"sales": FakeSemanticModel()},
        store=destination,
    )

    loaded = load_knowledge(
        destination,
        sources={"sales": FakeSemanticModel()},
    )

    assert loaded.package.operations == learned.package.operations
    persisted = destination.read_text(encoding="utf-8")
    assert "semantic_model.measure.v1" in persisted
    assert "Net Revenue" in persisted
    assert "Geography[Region]" in persisted


def test_non_semantic_sources_do_not_register_measure_operations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "orders.csv"
    source.write_text("order_id,amount\n1,10.5\n", encoding="utf-8")

    knowledge = RLM.learn(sources={"orders": source})

    assert knowledge.package.operations == ()


def test_executes_registered_measure_with_validated_scalar_parameters() -> None:
    model = FakeSemanticModel()
    knowledge = RLM.learn(sources={"sales": model})

    result = execute_registered_operation(
        knowledge,
        operation_id="sales.semantic_model.measure.v1",
        parameters={
            "measure": "Net Revenue",
            "groupby": "Geography[Region]",
            "filter_column": "Sales[Month]",
            "filter_value": "2025-06",
        },
    )

    assert model.measure_calls == [
        {
            "measure": "Net Revenue",
            "groupby": ["Geography[Region]"],
            "filters": {"Sales[Month]": ["2025-06"]},
        }
    ]
    packet = result.to_packet()
    assert packet["operation_id"] == "sales.semantic_model.measure.v1"
    assert packet["operation_version"] == "1"
    assert packet["audit_status"] == "passed"
    assert packet["row_count"] == 2
    assert packet["rows"] == model.measure_rows
    assert packet["source_fingerprints"] == {
        "sales": knowledge.package.sources[0].snapshot_fingerprint,
    }
    assert len(packet["operation_fingerprint"]) == 64
    assert len(packet["result_fingerprint"]) == 64


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (
            {"measure": "Private Metric"},
            "measure.*allowed",
        ),
        (
            {
                "measure": "Net Revenue",
                "groupby": "Geography[Country]",
            },
            "groupby.*allowed",
        ),
        (
            {
                "measure": "Net Revenue",
                "filter_column": "Sales[Month]",
            },
            "filter_column and filter_value",
        ),
        (
            {
                "measure": "Net Revenue",
                "filter_column": "Sales[Month]",
                "filter_value": "",
            },
            "filter_column and filter_value",
        ),
        (
            {"measure": "Net Revenue", "unexpected": "value"},
            "unknown parameter",
        ),
    ],
)
def test_rejects_unregistered_or_incomplete_measure_parameters(
    parameters,
    message,
) -> None:
    knowledge = RLM.learn(sources={"sales": FakeSemanticModel()})

    with pytest.raises(ValueError, match=message):
        execute_registered_operation(
            knowledge,
            operation_id="sales.semantic_model.measure.v1",
            parameters=parameters,
        )


def test_rejects_unknown_or_non_active_operations() -> None:
    knowledge = RLM.learn(sources={"sales": FakeSemanticModel()})

    with pytest.raises(ValueError, match="not registered"):
        execute_registered_operation(
            knowledge,
            operation_id="sales.unknown.v1",
            parameters={"measure": "Net Revenue"},
        )


def test_rejects_measure_results_that_exceed_registered_bounds() -> None:
    model = FakeSemanticModel()
    model.measure_rows = [
        {f"column_{column}": column for column in range(21)}
    ]
    knowledge = RLM.learn(sources={"sales": model})

    with pytest.raises(ValueError, match="column bound"):
        execute_registered_operation(
            knowledge,
            operation_id="sales.semantic_model.measure.v1",
            parameters={"measure": "Net Revenue"},
        )


def test_result_fingerprint_is_deterministic_and_changes_with_values() -> None:
    first_model = FakeSemanticModel()
    first = RLM.learn(sources={"sales": first_model})
    first_result = execute_registered_operation(
        first,
        operation_id="sales.semantic_model.measure.v1",
        parameters={"measure": "Net Revenue"},
    )

    second_model = FakeSemanticModel()
    second = RLM.learn(sources={"sales": second_model})
    second_result = execute_registered_operation(
        second,
        operation_id="sales.semantic_model.measure.v1",
        parameters={"measure": "Net Revenue"},
    )
    assert second_result.result_fingerprint == first_result.result_fingerprint

    second_model.measure_rows[0]["[Net Revenue]"] = 126.0
    changed = execute_registered_operation(
        second,
        operation_id="sales.semantic_model.measure.v1",
        parameters={"measure": "Net Revenue"},
    )
    assert changed.result_fingerprint != first_result.result_fingerprint
