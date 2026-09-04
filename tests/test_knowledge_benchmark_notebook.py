import ast
from pathlib import Path
from types import SimpleNamespace

from fabric_rlm.knowledge_benchmark import (
    KnowledgeBenchmarkPlanValidators,
    KnowledgeBenchmarkTask,
    run_knowledge_benchmark,
)
from fabric_rlm.runtime import RLMResult
from fabric_rlm.trajectory import Trajectory, TurnRecord


NOTEBOOK = (
    Path(__file__).parents[1]
    / "examples"
    / "notebooks"
    / "development"
    / "rlm_knowledge_benchmark_matrix.py"
)


def _load_matrix_namespace() -> dict:
    source = NOTEBOOK.read_text(encoding="utf-8")
    matrix = source.split("def close_to", maxsplit=1)[1].split(
        "for task in TASKS:",
        maxsplit=1,
    )[0]
    namespace = {
        "KnowledgeBenchmarkPlanValidators": KnowledgeBenchmarkPlanValidators,
        "KnowledgeBenchmarkTask": KnowledgeBenchmarkTask,
        "EXPECTED_TOTAL_ARR": 237576169.6,
        "EXPECTED_CANADA_ARR": 12000000.0,
        "EXPECTED_TOP_REGIONS": {"Americas - Canada"},
        "EXPECTED_TOP_COMBINATIONS": {
            ("Americas - Canada", "Cloud"),
        },
        "EXPECTED_INDOOR_VISITS": 4448818,
        "EXPECTED_OUTDOOR_VISITS": 11798702,
        "top_region_value": 12000000.0,
        "top_combination_value": 8000000.0,
        "expected_order_value": 42.0,
        "expected_region_values": {"North": 42.0},
        "semantic_operation_id": "arr_model.semantic_model.measure.v1",
        "semantic_operation": SimpleNamespace(
            operation_id="arr_model.semantic_model.measure.v1",
            parameter_schema={
                name: {}
                for name in (
                    "measure",
                    "groupby",
                    "groupby_2",
                    "filter_column",
                    "filter_value",
                    "filter_column_2",
                    "filter_value_2",
                    "filter_column_3",
                    "filter_value_3",
                )
            },
        ),
        "lakehouse_operation": SimpleNamespace(
            operation_id="orders.lakehouse.aggregate.v1",
            parameter_schema={
                name: {}
                for name in (
                    "catalog_source",
                    "aggregate",
                    "measure",
                    "groupby",
                    "groupby_2",
                    "filter_column",
                    "filter_value",
                    "filter_column_2",
                    "filter_value_2",
                )
            },
        ),
        "csv_operation": SimpleNamespace(
            operation_id="orders_csv.tabular.aggregate.v1",
            parameter_schema={
                name: {}
                for name in (
                    "aggregate",
                    "measure",
                    "groupby",
                    "groupby_2",
                    "filter_column",
                    "filter_value",
                    "filter_column_2",
                    "filter_value_2",
                )
            },
        ),
        "parquet_operation": SimpleNamespace(
            operation_id="orders_parquet.tabular.aggregate.v1",
            parameter_schema={
                name: {}
                for name in (
                    "aggregate",
                    "measure",
                    "groupby",
                    "groupby_2",
                    "filter_column",
                    "filter_value",
                    "filter_column_2",
                    "filter_value_2",
                )
            },
        ),
        "tourism_operation": SimpleNamespace(
            operation_id="tourism.lakehouse.preaggregate_join.v1",
            parameter_schema={
                "left_catalog_source": {"enum": ("indoor",)},
                "right_catalog_source": {"enum": ("outdoor",)},
                "left_measure": {},
                "right_measure": {},
                "join_key": {},
                "join_key_2": {},
                "scope": {},
            },
        ),
        "quarter_filters": {
            "Period[YearQuarter]": ["2026/Q2"],
            "ARR Data[IS_QUARTER]": ["1"],
        },
        "region_column": "Geography[Owner Region]",
        "product_family_column": "Product[Family]",
        "amount_column": "amount",
        "region_column_orders": "region",
        "orders_entry": {"name": "dbo.knowledge_validation_orders"},
        "semantic_knowledge": object(),
        "lakehouse_knowledge": object(),
        "csv_knowledge": object(),
        "parquet_knowledge": object(),
        "tourism_knowledge": object(),
        "model": object(),
        "orders_lakehouse": object(),
        "tourism_lakehouse": object(),
        "csv_path": Path("orders.csv"),
        "parquet_path": Path("orders.parquet"),
        "File": lambda path: path,
    }
    exec(compile("def close_to" + matrix, str(NOTEBOOK), "exec"), namespace)
    return namespace


def _benchmark_result(*, metadata: dict, analysis: str) -> RLMResult:
    trajectory = Trajectory(metadata=metadata)
    trajectory.append(
        TurnRecord(
            turn=1,
            code="SUBMIT(value=1)",
            stdout="",
            stderr="",
            error=None,
            submitted=True,
            state={},
        )
    )
    return RLMResult(
        submitted=True,
        payload={"value": 1.0, "analysis": analysis},
        trajectory=trajectory,
        final_state={},
        max_turns=1,
    )


def test_benchmark_scopes_lakehouse_discovery_to_orders_table() -> None:
    source = NOTEBOOK.read_text(encoding="utf-8")
    setup = source.split(
        "# ## Prepare exact Delta, CSV, and Parquet benchmark sources",
        maxsplit=1,
    )[1].split("def pick_column", maxsplit=1)[0]
    tree = ast.parse(setup)

    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LakehouseSource"
    ]
    assert len(constructors) == 1
    assert ast.unparse(constructors[0].args[0]) == (
        "f'{LAKEHOUSE_ROOT}/Tables/dbo/knowledge_validation_orders'"
    )

    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    resolved = assignments["orders_lakehouse"]
    assert isinstance(resolved, ast.Call)
    assert isinstance(resolved.func, ast.Attribute)
    assert resolved.func.attr == "resolve"
    assert resolved.func.value is constructors[0]

    query = assignments["orders_packet"]
    assert isinstance(query, ast.Call)
    assert isinstance(query.func, ast.Attribute)
    assert isinstance(query.func.value, ast.Name)
    assert query.func.value.id == "orders_lakehouse"

    entry = assignments["orders_entry"]
    assert isinstance(entry, ast.Subscript)
    assert isinstance(entry.value, ast.Attribute)
    assert isinstance(entry.value.value, ast.Name)
    assert entry.value.value.id == "orders_lakehouse"


def test_benchmark_persists_failures_before_results_are_written() -> None:
    source = NOTEBOOK.read_text(encoding="utf-8")

    bootstrap_index = source.index("RUN_LOG_PATH =")
    install_index = source.index("%pip install")
    assert bootstrap_index < install_index

    assert "def persist_run_log(" in source
    assert "def persist_uncaught_exception(" in source
    assert "get_ipython().set_custom_exc(" in source
    assert source.count("install_uncaught_exception_logger()") >= 2
    assert source.count('        "failed",') >= 2
    assert "traceback.format_exception(" in source
    assert 'phase="trial"' in source
    assert "task_id=task.task_id" in source
    assert "arm=arm" in source
    assert 'phase="trial_completed"' in source
    assert "outputs=result.outputs" in source


def test_every_matrix_task_has_task_specific_stage_and_commentary_validators() -> None:
    namespace = _load_matrix_namespace()

    for task in namespace["TASKS"]:
        assert isinstance(task.plan_validators, KnowledgeBenchmarkPlanValidators)
        assert all(
            callable(getattr(task.plan_validators, field))
            for field in task.plan_validators.__dataclass_fields__
        )
        assert callable(task.is_commentary_valid)

    semantic_filtered = next(
        task for task in namespace["TASKS"] if task.task_id == "semantic_filtered"
    )
    matching = SimpleNamespace(
        operation_id=semantic_filtered.expected_operation_id,
        parameters={
            "measure": "ARR $",
            "groupby": "",
            "groupby_2": "",
            "filter_column": "Period[YearQuarter]",
            "filter_value": "2026/Q2",
            "filter_column_2": "ARR Data[IS_QUARTER]",
            "filter_value_2": "1",
            "filter_column_3": "Geography[Owner Region]",
            "filter_value_3": "Americas - Canada",
        },
    )
    validators = semantic_filtered.plan_validators
    assert validators is not None
    assert validators.measure(matching) is True
    assert validators.time_policy(matching) is True
    assert validators.groupby(matching) is True
    assert validators.filter_columns(matching) is True
    assert validators.filter_values(matching) is True
    assert validators.full_operation_plan(matching) is True

    wrong_measure = SimpleNamespace(
        operation_id=matching.operation_id,
        parameters={**matching.parameters, "measure": "Synthetic Correctness"},
    )
    assert validators.measure(wrong_measure) is False
    assert validators.full_operation_plan(wrong_measure) is False


def test_representative_real_matrix_tasks_emit_non_null_stage_metrics() -> None:
    namespace = _load_matrix_namespace()
    tasks = {
        task.task_id: task
        for task in namespace["TASKS"]
        if task.task_id in {"semantic_filtered", "fanout_two_fact"}
    }
    learned_parameters = {
        "semantic_filtered": {
            "measure": "ARR $",
            "groupby": "",
            "groupby_2": "",
            "filter_column": "Period[YearQuarter]",
            "filter_value": "2026/Q2",
            "filter_column_2": "ARR Data[IS_QUARTER]",
            "filter_value_2": "1",
            "filter_column_3": "Geography[Owner Region]",
            "filter_value_3": "Americas - Canada",
        },
        "fanout_two_fact": {
            "left_catalog_source": "indoor",
            "right_catalog_source": "outdoor",
            "left_measure": "visits",
            "right_measure": "visits",
            "join_key": "month",
            "join_key_2": "",
            "scope": "latest",
        },
    }

    for task_id, task in tasks.items():
        report = run_knowledge_benchmark(
            tasks=[task],
            repetitions=1,
            seed=0,
            make_lm=lambda **_kwargs: SimpleNamespace(
                kwargs={"cache": False},
                model="test/model",
            ),
            run=lambda _task, arm, _lm: _benchmark_result(
                metadata=(
                    {
                        "operation_id": task.expected_operation_id,
                        "operation_parameters": learned_parameters[task_id],
                        "operation_audit_status": "passed",
                    }
                    if arm == "learned"
                    else {}
                ),
                analysis="Grounded commentary.",
            ),
        )
        learned = next(trial for trial in report.trials if trial.arm == "learned")
        assert all(
            getattr(learned, field) is not None
            for field in (
                "measure_correct",
                "time_policy_correct",
                "groupby_correct",
                "filter_columns_correct",
                "filter_values_correct",
                "full_operation_plan_correct",
                "LLM_commentary_valid",
            )
        )
