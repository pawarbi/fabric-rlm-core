from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from fabric_rlm.experimental.analysis_contracts import (
    AnalysisBrief,
    AnalysisDAG,
    EvidenceEntry,
    OperatorNode,
    OperatorResult,
    RunBudget,
)


def test_analysis_brief_records_explicit_focus_and_safe_defaults() -> None:
    brief = AnalysisBrief(
        objective="Explain the decline in repeat purchase rate",
        focus_areas=("drivers", "cohorts"),
        target_metrics=("repeat_purchase_rate",),
    )

    assert brief.focus_areas == ("drivers", "cohorts")
    assert brief.interpretation_level == "descriptive"
    assert brief.exclusions == ()
    assert brief.to_dict()["objective"] == brief.objective


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"objective": ""}, "objective"),
        (
            {"objective": "Find drivers", "interpretation_level": "causal"},
            "interpretation_level",
        ),
        (
            {"objective": "Find drivers", "focus_areas": ("drivers", "drivers")},
            "focus_areas",
        ),
    ],
)
def test_analysis_brief_rejects_ambiguous_or_unsafe_values(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        AnalysisBrief(**kwargs)


def test_run_budget_requires_positive_bounded_work() -> None:
    budget = RunBudget(
        max_nodes=12,
        max_parallel_nodes=3,
        max_rows_per_node=100_000,
        timeout_seconds=600,
    )

    assert budget.to_dict() == {
        "max_nodes": 12,
        "max_parallel_nodes": 3,
        "max_rows_per_node": 100_000,
        "timeout_seconds": 600.0,
    }

    with pytest.raises(ValueError, match="max_parallel_nodes"):
        RunBudget(max_nodes=2, max_parallel_nodes=3)


def test_operator_node_is_deeply_immutable_and_json_compatible() -> None:
    node = OperatorNode(
        node_id="decompose-revenue",
        operator="kpi.additive.v1",
        source_ids=("orders",),
        seed=42,
        parameters={
            "metric": "revenue",
            "segments": ["region", "channel"],
            "tolerance": 1e-9,
        },
    )

    assert node.parameters["segments"] == ("region", "channel")
    with pytest.raises(TypeError):
        node.parameters["metric"] = "profit"
    with pytest.raises(FrozenInstanceError):
        node.seed = 7

    with pytest.raises(ValueError, match="JSON-compatible"):
        OperatorNode(
            node_id="bad-parameters",
            operator="kpi.additive.v1",
            source_ids=("orders",),
            seed=42,
            parameters={"columns": {"revenue", "profit"}},
        )

    with pytest.raises(ValueError, match="string object keys"):
        OperatorNode(
            node_id="bad-object-key",
            operator="kpi.additive.v1",
            source_ids=("orders",),
            seed=42,
            parameters={"metric": "revenue", 1: "invalid"},
        )


def test_operator_node_rejects_missing_identifiers_and_invalid_seed() -> None:
    with pytest.raises(ValueError, match="node_id"):
        OperatorNode(
            node_id="",
            operator="kpi.additive.v1",
            source_ids=("orders",),
            seed=42,
        )

    with pytest.raises(ValueError, match="seed"):
        OperatorNode(
            node_id="decompose-revenue",
            operator="kpi.additive.v1",
            source_ids=("orders",),
            seed=-1,
        )


def test_analysis_dag_serialization_is_stable_for_equivalent_inputs() -> None:
    first = AnalysisDAG(
        dag_id="revenue-drivers",
        root_seed=1234,
        budget=RunBudget(max_nodes=4),
        nodes=(
            OperatorNode(
                node_id="baseline",
                operator="kpi.baseline.v1",
                source_ids=("orders",),
                seed=10,
                parameters={"filters": {"status": "completed", "year": 2026}},
            ),
            OperatorNode(
                node_id="drivers",
                operator="kpi.additive.v1",
                source_ids=("orders",),
                seed=11,
                depends_on=("baseline",),
                execution_mode="parallel",
                parameters={"top_n": 10},
            ),
        ),
    )
    second = AnalysisDAG(
        dag_id="revenue-drivers",
        root_seed=1234,
        budget=RunBudget(max_nodes=4),
        nodes=first.nodes,
    )

    assert first.to_dict() == second.to_dict()
    assert json.dumps(first.to_dict(), sort_keys=True, allow_nan=False) == json.dumps(
        second.to_dict(),
        sort_keys=True,
        allow_nan=False,
    )


def test_analysis_dag_rejects_invalid_execution_mode() -> None:
    with pytest.raises(ValueError, match="execution_mode"):
        OperatorNode(
            node_id="drivers",
            operator="kpi.additive.v1",
            source_ids=("orders",),
            seed=11,
            execution_mode="async",
        )


def test_analysis_dag_rejects_duplicate_node_ids() -> None:
    node = OperatorNode(
        node_id="decompose",
        operator="kpi.additive.v1",
        source_ids=("orders",),
        seed=1,
    )

    with pytest.raises(ValueError, match="node_id"):
        AnalysisDAG(
            dag_id="duplicate-nodes",
            root_seed=1,
            budget=RunBudget(max_nodes=5),
            nodes=(node, node),
        )


def test_operator_result_and_evidence_entry_preserve_structured_failure() -> None:
    failed = OperatorResult(
        node_id="drivers",
        operator="kpi.additive.v1",
        status="failed",
        seed=11,
        sample_size=0,
        failure_code="insufficient_data",
        failure_message="No rows remained after the authorized filters",
    )
    evidence = EvidenceEntry(
        evidence_id="ev-drivers",
        node_id="drivers",
        state="failed",
        result=failed,
    )

    assert evidence.result.failure_code == "insufficient_data"
    assert evidence.to_dict()["result"]["status"] == "failed"


@pytest.mark.parametrize("state", ["planned", "running", "rejected"])
def test_nonterminal_evidence_states_have_no_result(state: str) -> None:
    evidence = EvidenceEntry(
        evidence_id=f"ev-{state}",
        node_id="drivers",
        state=state,
        supersedes="ev-prior",
    )

    assert evidence.result is None
    assert evidence.to_dict()["supersedes"] == "ev-prior"


def test_evidence_entry_rejects_self_supersession() -> None:
    with pytest.raises(ValueError, match="supersedes"):
        EvidenceEntry(
            evidence_id="ev-drivers",
            node_id="drivers",
            state="superseded",
            supersedes="ev-drivers",
        )


def test_completed_operator_result_requires_values_and_diagnostics() -> None:
    with pytest.raises(ValueError, match="values"):
        OperatorResult(
            node_id="drivers",
            operator="kpi.additive.v1",
            status="completed",
            seed=11,
            sample_size=100,
        )

    with pytest.raises(ValueError, match="diagnostics"):
        OperatorResult(
            node_id="drivers",
            operator="kpi.additive.v1",
            status="completed",
            seed=11,
            sample_size=100,
            values={"change": -0.05},
        )

    with pytest.raises(ValueError, match="failure_code"):
        OperatorResult(
            node_id="drivers",
            operator="kpi.additive.v1",
            status="failed",
            seed=11,
            sample_size=0,
        )
