from __future__ import annotations

from pathlib import Path

from fabric_rlm import RLM, load_knowledge
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
    assert operation.max_output_rows == 1_000
    assert operation.max_output_columns == 100


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
