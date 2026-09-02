from __future__ import annotations

from pathlib import Path

import pytest

from fabric_rlm import RLM, load_knowledge
from fabric_rlm.knowledge_execution import (
    OperationPlan,
    OperationPlanFallback,
    execute_registered_operation,
    parse_operation_plan,
)
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
        self.column_rows = [
            {"Table Name": "Sales", "Column Name": "Month"},
            {"Table Name": "ARR Data", "Column Name": "IS_QUARTER"},
            {"Table Name": "Geography", "Column Name": "Region"},
        ]
        self.measure_metadata_rows = [
            {"Table Name": "Measures", "Measure Name": "Net Revenue"},
            {"Table Name": "Measures", "Measure Name": "Order Count"},
        ]
        self.measure_rows = [
            {"Geography[Region]": "East", "[Net Revenue]": 125.0},
            {"Geography[Region]": "West", "[Net Revenue]": 75.0},
        ]

    def tables(self):
        return FakeFrame([{"Name": "Sales"}, {"Name": "Geography"}])

    def columns(self):
        return FakeFrame(self.column_rows)

    def measures(self):
        return FakeFrame(self.measure_metadata_rows)

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


class SequenceLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages = []

    def __call__(self, *, messages):
        self.messages.append([dict(message) for message in messages])
        response = self.responses[self.calls]
        self.calls += 1
        if callable(response):
            response = response()
        return response


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
        "ARR Data[IS_QUARTER]",
        "Geography[Region]",
        "Sales[Month]",
    )
    assert operation.parameter_schema["groupby_2"]["enum"] == (
        "",
        "ARR Data[IS_QUARTER]",
        "Geography[Region]",
        "Sales[Month]",
    )
    assert operation.parameter_schema["filter_column"]["enum"] == (
        "",
        "ARR Data[IS_QUARTER]",
        "Geography[Region]",
        "Sales[Month]",
    )
    assert operation.parameter_schema["filter_column_2"]["enum"] == (
        "",
        "ARR Data[IS_QUARTER]",
        "Geography[Region]",
        "Sales[Month]",
    )
    assert operation.parameter_schema["filter_column_3"]["enum"] == (
        "",
        "ARR Data[IS_QUARTER]",
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


def test_hidden_semantic_model_fields_are_not_registered() -> None:
    model = FakeSemanticModel()
    model.measure_metadata_rows.append(
        {
            "Table Name": "Measures",
            "Measure Name": "Private Margin",
            "Is Hidden": True,
        }
    )
    model.column_rows.append(
        {
            "Table Name": "Owner",
            "Column Name": "Private Segment",
            "Is Hidden": True,
        }
    )

    operation = RLM.learn(sources={"sales": model}).package.operations[0]

    assert "Private Margin" not in operation.parameter_schema["measure"]["enum"]
    assert (
        "Owner[Private Segment]"
        not in operation.parameter_schema["groupby"]["enum"]
    )


def test_tabular_sources_register_aggregate_not_measure_operations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "orders.csv"
    source.write_text("order_id,amount\n1,10.5\n", encoding="utf-8")

    knowledge = RLM.learn(sources={"orders": source})

    assert tuple(
        (operation.operation, operation.host_implementation_id)
        for operation in knowledge.package.operations
    ) == (("tabular.aggregate", "tabular.aggregate.v1"),)


def test_executes_registered_measure_with_validated_scalar_parameters() -> None:
    model = FakeSemanticModel()
    knowledge = RLM.learn(sources={"sales": model})

    result = execute_registered_operation(
        knowledge,
        operation_id="sales.semantic_model.measure.v1",
        parameters={
            "measure": "Net Revenue",
            "groupby": "Geography[Region]",
            "groupby_2": "Sales[Month]",
            "filter_column": "Sales[Month]",
            "filter_value": "2025-06",
            "filter_column_2": "ARR Data[IS_QUARTER]",
            "filter_value_2": "1",
            "filter_column_3": "Geography[Region]",
            "filter_value_3": "North",
        },
    )

    assert model.measure_calls == [
        {
            "measure": "Net Revenue",
            "groupby": ["Geography[Region]", "Sales[Month]"],
            "filters": {
                "Sales[Month]": ["2025-06"],
                "ARR Data[IS_QUARTER]": ["1"],
                "Geography[Region]": ["North"],
            },
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
                "groupby": "Geography[Region]",
                "groupby_2": "Geography[Region]",
            },
            "grouping dimensions must not contain duplicates",
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
                "filter_column_2": "Sales[Month]",
            },
            "filter_column_2 and filter_value_2",
        ),
        (
            {
                "measure": "Net Revenue",
                "filter_column": "Sales[Month]",
                "filter_value": "2025-06",
                "filter_column_2": "Sales[Month]",
                "filter_value_2": "2025-07",
            },
            "filter columns must not contain duplicates",
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


def test_rejects_measure_results_that_exceed_packet_byte_bound() -> None:
    model = FakeSemanticModel()
    model.measure_rows = [{"[Net Revenue]": "x" * 300_000}]
    knowledge = RLM.learn(sources={"sales": model})

    with pytest.raises(ValueError, match="byte bound"):
        execute_registered_operation(
            knowledge,
            operation_id="sales.semantic_model.measure.v1",
            parameters={"measure": "Net Revenue"},
        )


def test_task_selects_executes_and_synthesizes_registered_measure() -> None:
    model = FakeSemanticModel()
    knowledge = RLM.learn(sources={"sales": model})
    lm = SequenceLM(
        [
            (
                '{"operation_id":"sales.semantic_model.measure.v1",'
                '"parameters":{"measure":"Net Revenue",'
                '"groupby":"Geography[Region]",'
                '"groupby_2":"Sales[Month]",'
                '"filter_column":"Sales[Month]",'
                '"filter_value":"2025-06",'
                '"filter_column_2":"ARR Data[IS_QUARTER]",'
                '"filter_value_2":"1"}}'
            ),
            (
                "```python\n"
                "SUBMIT(answer=sum(row['[Net Revenue]'] "
                "for row in knowledge_result['rows']))\n"
                "```"
            ),
        ]
    )

    result = RLM.task(
        "Return net revenue by region for June 2025.",
        knowledge=knowledge,
        outputs={"answer": float},
        lm=lm,
        max_turns=2,
        timeout=5,
    ).run()

    assert result.payload == {"answer": 200.0}
    assert lm.calls == 2
    assert model.measure_calls[-1] == {
        "measure": "Net Revenue",
        "groupby": ["Geography[Region]", "Sales[Month]"],
        "filters": {
            "Sales[Month]": ["2025-06"],
            "ARR Data[IS_QUARTER]": ["1"],
        },
    }
    metadata = result.trajectory.metadata
    assert metadata["knowledge_mode"] == "registered_operation"
    assert metadata["operation_id"] == "sales.semantic_model.measure.v1"
    assert metadata["operation_version"] == "1"
    assert metadata["operation_audit_status"] == "passed"
    assert len(metadata["operation_fingerprint"]) == 64
    assert len(metadata["operation_result_fingerprint"]) == 64
    assert metadata["operation_selection_lm_calls"] == 1
    synthesis_prompt = "\n".join(
        message["content"] for message in lm.messages[1]
    )
    assert "knowledge_result" in synthesis_prompt
    assert "SemanticModel dataset=" not in synthesis_prompt


def test_successful_operation_preserves_explicit_task_inputs() -> None:
    knowledge = RLM.learn(sources={"sales": FakeSemanticModel()})
    lm = SequenceLM(
        [
            (
                '{"operation_id":"sales.semantic_model.measure.v1",'
                '"parameters":{"measure":"Net Revenue"}}'
            ),
            (
                "```python\n"
                "SUBMIT(answer=context_note + ':' + "
                "str(sum(row['[Net Revenue]'] "
                "for row in knowledge_result['rows'])))\n"
                "```"
            ),
        ]
    )

    result = RLM.task(
        "Return net revenue with the supplied context label.",
        inputs={"context_note": "audited"},
        knowledge=knowledge,
        outputs={"answer": str},
        lm=lm,
        max_turns=1,
        timeout=5,
    ).run()

    assert result.payload == {"answer": "audited:200.0"}
    synthesis_prompt = "\n".join(
        message["content"] for message in lm.messages[1]
    )
    assert "context_note" in synthesis_prompt
    assert "SemanticModel dataset=" not in synthesis_prompt


def test_registered_operation_reserves_result_input_alias_before_planning() -> None:
    knowledge = RLM.learn(sources={"sales": FakeSemanticModel()})
    lm = SequenceLM([])

    with pytest.raises(ValueError, match="knowledge_result.*reserved"):
        RLM.task(
            "Return net revenue.",
            inputs={"knowledge_result": "caller value"},
            knowledge=knowledge,
            outputs={"answer": float},
            lm=lm,
            max_turns=1,
            timeout=5,
        ).run()

    assert lm.calls == 0


def test_invalid_operation_plan_falls_back_without_host_execution() -> None:
    model = FakeSemanticModel()
    knowledge = RLM.learn(sources={"sales": model})
    lm = SequenceLM(
        [
            (
                '{"operation_id":"sales.semantic_model.measure.v1",'
                '"parameters":{"measure":"Private Metric"}}'
            ),
            "```python\nSUBMIT(answer=sales.dataset)\n```",
        ]
    )

    result = RLM.task(
        "Return the model identity.",
        knowledge=knowledge,
        outputs={"answer": str},
        lm=lm,
        max_turns=1,
        timeout=5,
    ).run()

    assert result.payload == {"answer": "Sales Model"}
    assert model.measure_calls == []
    assert result.trajectory.metadata["knowledge_mode"] == (
        "fallback_operation_plan_rejected"
    )
    assert result.trajectory.metadata["operation_selection_lm_calls"] == 1


def test_planner_can_decline_incompatible_registered_operation() -> None:
    model = FakeSemanticModel()
    knowledge = RLM.learn(sources={"sales": model})
    lm = SequenceLM(
        [
            '{"fallback":true,"reason":"No compatible aggregate."}',
            "```python\nSUBMIT(answer=sales.dataset)\n```",
        ]
    )

    result = RLM.task(
        "Return the model identity.",
        knowledge=knowledge,
        outputs={"answer": str},
        lm=lm,
        max_turns=1,
        timeout=5,
    ).run()

    assert result.payload == {"answer": "Sales Model"}
    assert model.measure_calls == []
    assert result.trajectory.metadata["knowledge_mode"] == (
        "fallback_no_compatible_operation"
    )


def test_failed_host_audit_does_not_fall_back_to_ordinary_execution() -> None:
    model = FakeSemanticModel()
    model.measure_rows = [
        {f"column_{column}": column for column in range(21)}
    ]
    knowledge = RLM.learn(sources={"sales": model})
    lm = SequenceLM(
        [
            (
                '{"operation_id":"sales.semantic_model.measure.v1",'
                '"parameters":{"measure":"Net Revenue"}}'
            ),
        ]
    )

    with pytest.raises(ValueError, match="column bound"):
        RLM.task(
            "Return net revenue.",
            knowledge=knowledge,
            outputs={"answer": float},
            lm=lm,
            max_turns=1,
            timeout=5,
        ).run()

    assert lm.calls == 1


def test_source_drift_after_selection_fails_closed_before_host_execution() -> None:
    model = FakeSemanticModel()
    knowledge = RLM.learn(sources={"sales": model})

    def mutate_then_plan():
        model.measure_metadata_rows[0]["Measure Name"] = "Changed Revenue"
        return (
            '{"operation_id":"sales.semantic_model.measure.v1",'
            '"parameters":{"measure":"Net Revenue"}}'
        )

    lm = SequenceLM([mutate_then_plan])

    with pytest.raises(ValueError, match="stale.*sales"):
        RLM.task(
            "Return net revenue.",
            knowledge=knowledge,
            outputs={"answer": float},
            lm=lm,
            max_turns=1,
            timeout=5,
        ).run()

    assert lm.calls == 1
    assert model.measure_calls == []


def test_operation_selection_usage_is_included_in_result_totals() -> None:
    knowledge = RLM.learn(sources={"sales": FakeSemanticModel()})
    lm = SequenceLM(
        [
            {
                "content": (
                    '{"operation_id":"sales.semantic_model.measure.v1",'
                    '"parameters":{"measure":"Net Revenue"}}'
                ),
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
            },
            {
                "content": (
                    "```python\n"
                    "SUBMIT(answer=sum(row['[Net Revenue]'] "
                    "for row in knowledge_result['rows']))\n"
                    "```"
                ),
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "prompt_tokens_details": {"cached_tokens": 5},
                    "completion_tokens_details": {"reasoning_tokens": 1},
                },
            },
        ]
    )

    result = RLM.task(
        "Return net revenue.",
        knowledge=knowledge,
        outputs={"answer": float},
        lm=lm,
        max_turns=1,
        timeout=5,
    ).run()

    assert result.total_prompt_tokens == 30
    assert result.total_completion_tokens == 12
    assert result.total_cached_tokens == 8
    assert result.total_reasoning_tokens == 3


def test_parse_operation_plan_accepts_exact_plan_and_fallback_shapes() -> None:
    plan = parse_operation_plan(
        '```json\n{"operation_id":"sales.measure.v1",'
        '"parameters":{"measure":"Revenue"}}\n```'
    )
    fallback = parse_operation_plan(
        '{"fallback":true,"reason":"No compatible operation."}'
    )

    assert plan == OperationPlan(
        operation_id="sales.measure.v1",
        parameters={"measure": "Revenue"},
    )
    assert fallback == OperationPlanFallback(reason="No compatible operation.")


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "```python\n{}\n```",
        '{"operation_id":"sales.measure.v1"}',
        '{"operation_id":"sales.measure.v1","parameters":{},"extra":true}',
        '{"operation_id":3,"parameters":{}}',
        '{"operation_id":"sales.measure.v1","parameters":[]}',
        '{"fallback":true,"reason":"no","extra":true}',
        '{"fallback":true,"reason":3}',
        '{"fallback":true,"reason":"' + ("x" * 257) + '"}',
    ],
)
def test_parse_operation_plan_rejects_non_contract_responses(response) -> None:
    with pytest.raises(ValueError):
        parse_operation_plan(response)
