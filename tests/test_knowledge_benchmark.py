from __future__ import annotations

from dataclasses import dataclass

import pytest

from fabric_rlm.knowledge_benchmark import (
    KnowledgeBenchmarkPlanValidators,
    KnowledgeBenchmarkSelectedPlan,
    KnowledgeBenchmarkTask,
    run_knowledge_benchmark,
)
from fabric_rlm.runtime import RLMResult
from fabric_rlm.trajectory import Trajectory, TurnRecord


@dataclass
class FakeLM:
    kwargs: dict
    model: str = "azure/gpt-5"
    history: list[dict] | None = None


def _result(
    *,
    answer: float,
    turns: int,
    metadata: dict | None = None,
    cached_tokens: int | None = None,
    analysis: str | None = None,
) -> RLMResult:
    trajectory = Trajectory(metadata=dict(metadata or {}))
    for turn in range(1, turns + 1):
        trajectory.append(
            TurnRecord(
                turn=turn,
                code="SUBMIT(answer=1)",
                stdout="",
                stderr="",
                error=None,
                submitted=turn == turns,
                state={},
                prompt_tokens=100,
                completion_tokens=20,
                lm_call_seconds=0.5,
                worker_execute_seconds=0.25,
            )
        )
    return RLMResult(
        submitted=True,
        payload={
            "answer": answer,
            **({"analysis": analysis} if analysis is not None else {}),
        },
        trajectory=trajectory,
        final_state={},
        total_prompt_tokens=100 * turns,
        total_completion_tokens=20 * turns,
        total_cached_tokens=cached_tokens,
        total_lm_seconds=0.5 * turns,
        total_worker_seconds=0.25 * turns,
        max_turns=turns,
    )


def test_repeated_benchmark_is_seeded_randomized_and_disables_cache() -> None:
    task = KnowledgeBenchmarkTask(
        task_id="revenue",
        question="What was total revenue?",
        expected_operation_id="sales.semantic_model.measure.v1",
        is_correct=lambda payload: bool(payload and payload.get("answer") == 30.5),
    )
    lm_calls = []
    run_calls = []

    def make_lm(*, task_id, repetition, arm, cache):
        lm_calls.append((task_id, repetition, arm, cache))
        return FakeLM(kwargs={"cache": cache})

    def run(task, arm, lm):
        run_calls.append((task.task_id, arm, lm.kwargs["cache"]))
        if arm == "cold":
            return _result(answer=30.5, turns=4)
        return _result(
            answer=30.5,
            turns=2,
            metadata={
                "knowledge_mode": "registered_operation",
                "operation_id": "sales.semantic_model.measure.v1",
                "operation_audit_status": "passed",
                "operation_host_seconds": 0.1,
            },
        )

    first = run_knowledge_benchmark(
        tasks=[task],
        repetitions=4,
        seed=17,
        make_lm=make_lm,
        run=run,
    )
    first_order = [(trial.repetition, trial.arm) for trial in first.trials]
    lm_calls.clear()
    run_calls.clear()
    second = run_knowledge_benchmark(
        tasks=[task],
        repetitions=4,
        seed=17,
        make_lm=make_lm,
        run=run,
    )

    assert [(trial.repetition, trial.arm) for trial in second.trials] == first_order
    assert {cache for *_prefix, cache in lm_calls} == {False}
    assert {cache for *_prefix, cache in run_calls} == {False}
    assert len(first.trials) == 8
    assert {tuple(first_order[index:index + 2]) for index in range(0, 8, 2)} <= {
        ((0, "cold"), (0, "learned")),
        ((0, "learned"), (0, "cold")),
        ((1, "cold"), (1, "learned")),
        ((1, "learned"), (1, "cold")),
        ((2, "cold"), (2, "learned")),
        ((2, "learned"), (2, "cold")),
        ((3, "cold"), (3, "learned")),
        ((3, "learned"), (3, "cold")),
    }


def test_benchmark_accepts_lm_with_public_cache_attribute() -> None:
    task = KnowledgeBenchmarkTask(
        task_id="revenue",
        question="What was total revenue?",
        expected_operation_id="sales.semantic_model.measure.v1",
        is_correct=lambda payload: bool(payload and payload.get("answer") == 30.5),
    )

    @dataclass
    class PublicCacheLM:
        cache: bool
        kwargs: dict

    report = run_knowledge_benchmark(
        tasks=[task],
        repetitions=1,
        seed=0,
        make_lm=lambda **_kwargs: PublicCacheLM(cache=False, kwargs={}),
        run=lambda *_args: _result(answer=30.5, turns=1),
    )

    assert len(report.trials) == 2


def test_benchmark_records_stage_correctness_and_complete_accounting() -> None:
    def parameters(plan: KnowledgeBenchmarkSelectedPlan) -> dict:
        return dict(plan.parameters)

    task = KnowledgeBenchmarkTask(
        task_id="revenue",
        question="What was total revenue?",
        expected_operation_id="sales.semantic_model.measure.v1",
        is_correct=lambda payload: bool(payload and payload.get("answer") == 30.5),
        plan_validators=KnowledgeBenchmarkPlanValidators(
            measure=lambda plan: parameters(plan).get("measure") == "Net Revenue",
            time_policy=lambda plan: parameters(plan).get("period") == "2025-06",
            groupby=lambda plan: parameters(plan).get("groupby")
            == "Geography[Region]",
            filter_columns=lambda plan: parameters(plan).get("filter_column")
            == "Sales[Month]",
            filter_values=lambda plan: parameters(plan).get("filter_value")
            == "2025-07",
            full_operation_plan=lambda plan: (
                plan.operation_id == "sales.semantic_model.measure.v1"
                and parameters(plan).get("filter_value") == "2025-07"
            ),
        ),
        is_commentary_valid=lambda payload: bool(
            payload
            and payload.get("analysis")
            == "Revenue was calculated from the governed result."
        ),
        is_deterministic_result_binding_correct=lambda payload, metadata: bool(
            payload
            and payload.get("answer")
            == metadata.get("trusted_host_packet", {}).get("answer")
        ),
    )

    def make_lm(**_kwargs):
        return FakeLM(kwargs={"cache": False}, history=[])

    def run(_task, arm, lm):
        assert lm.history is not None
        lm.history.extend([{"cost": 0.02}, {"cost": 0.03}])
        if arm == "cold":
            return _result(answer=30.5, turns=4, cached_tokens=120)
        return _result(
            answer=30.5,
            turns=1,
            cached_tokens=40,
            analysis="Revenue was calculated from the governed result.",
            metadata={
                "knowledge_fingerprint": "package-fingerprint",
                "knowledge_mode": "registered_operation",
                "operation_id": "sales.semantic_model.measure.v1",
                "operation_parameters": {
                    "measure": "Net Revenue",
                    "period": "2025-06",
                    "groupby": "Geography[Region]",
                    "filter_column": "Sales[Month]",
                    "filter_value": "2025-06",
                },
                "operation_audit_status": "passed",
                "operation_host_seconds": 0.2,
                "trusted_host_packet": {"answer": 30.5},
            },
        )

    report = run_knowledge_benchmark(
        tasks=[task],
        repetitions=1,
        seed=0,
        make_lm=make_lm,
        run=run,
    )
    rows = {trial.arm: trial.to_dict() for trial in report.trials}

    assert rows["cold"]["final_answer_correct"] is True
    assert rows["cold"]["operation_id_correct"] is None
    assert rows["learned"]["final_answer_correct"] is True
    assert rows["learned"]["operation_id_correct"] is True
    assert rows["learned"]["measure_correct"] is True
    assert rows["learned"]["time_policy_correct"] is True
    assert rows["learned"]["groupby_correct"] is True
    assert rows["learned"]["filter_columns_correct"] is True
    assert rows["learned"]["filter_values_correct"] is False
    assert rows["learned"]["full_operation_plan_correct"] is False
    assert rows["learned"]["host_execution_passed"] is True
    assert rows["learned"]["host_audit_passed"] is True
    assert rows["learned"]["deterministic_result_binding_correct"] is True
    assert rows["learned"]["LLM_commentary_valid"] is True
    assert rows["learned"]["turns"] == 1
    assert rows["learned"]["prompt_tokens"] == 100
    assert rows["learned"]["completion_tokens"] == 20
    assert rows["learned"]["cached_prompt_tokens"] == 40
    assert rows["learned"]["uncached_prompt_tokens"] == 60
    assert rows["learned"]["lm_seconds"] == 0.5
    assert rows["learned"]["provider"] == "azure"
    assert rows["learned"]["model"] == "gpt-5"
    assert rows["learned"]["provider_model_cost"] == 0.05
    assert rows["learned"]["worker_seconds"] == 0.25
    assert rows["learned"]["host_seconds"] == 0.2
    assert rows["learned"]["knowledge_fingerprint"] == "package-fingerprint"

    summary = report.summary()
    assert summary["cold"]["final_answer_correct_rate"] == 1.0
    assert summary["learned"]["final_answer_correct_rate"] == 1.0
    assert summary["learned"]["operation_id_correct_rate"] == 1.0
    assert summary["learned"]["filter_values_correct_rate"] == 0.0
    assert summary["learned"]["host_audit_passed_rate"] == 1.0
    assert summary["learned"]["LLM_commentary_valid_rate"] == 1.0
    assert summary["learned"]["mean_turns"] == 1.0
    assert summary["learned"]["mean_cached_prompt_tokens"] == 40.0
    assert summary["learned"]["mean_uncached_prompt_tokens"] == 60.0
    assert summary["learned"]["mean_provider_model_cost"] == 0.05
    assert summary["learned"]["numeric_correct_rate"] == 1.0
    assert summary["learned"]["operation_selection_accuracy"] == 1.0
    assert summary["learned"]["audit_pass_rate"] == 1.0
    assert rows["learned"]["numeric_correct"] is True
    assert rows["learned"]["operation_selection_correct"] is True
    assert rows["learned"]["audit_passed"] is True


def test_registered_operation_fingerprint_does_not_prove_result_binding() -> None:
    task = KnowledgeBenchmarkTask(
        task_id="revenue",
        question="What was total revenue?",
        expected_operation_id="sales.semantic_model.measure.v1",
        is_correct=lambda payload: bool(payload and payload.get("answer") == 30.5),
    )

    report = run_knowledge_benchmark(
        tasks=[task],
        repetitions=1,
        seed=0,
        make_lm=lambda **_kwargs: FakeLM(kwargs={"cache": False}),
        run=lambda _task, arm, _lm: _result(
            answer=999.0,
            turns=1,
            metadata={
                "knowledge_mode": "registered_operation",
                "operation_id": "sales.semantic_model.measure.v1",
                "operation_result_fingerprint": "host-result-fingerprint",
                "deterministic_result_binding_correct": True,
            }
            if arm == "learned"
            else {},
        ),
    )

    learned = next(trial for trial in report.trials if trial.arm == "learned")
    assert learned.host_execution_passed is True
    assert learned.operation_result_fingerprint == "host-result-fingerprint"
    assert learned.final_answer_correct is False
    assert learned.deterministic_result_binding_correct is None


@pytest.mark.parametrize(
    ("cost", "expected"),
    [
        (0.12, 0.12),
        ({"total_cost_usd": 0.12, "request_count": 7}, 0.12),
        ({"prompt_cost": 0.05, "completion_cost": 0.07}, 0.12),
        (
            {
                "total_cost": 0.12,
                "prompt_cost": 0.5,
                "completion_cost": 0.7,
            },
            0.12,
        ),
        ({"azure": {"total": 0.12, "latency_ms": 400}}, 0.12),
        (
            {
                "provider": {
                    "cost": {
                        "input": 0.05,
                        "output": 0.07,
                        "token_count": 1000,
                    }
                }
            },
            0.12,
        ),
        ({"total": True, "input_cost": float("nan"), "tokens": 1000}, None),
    ],
)
def test_provider_cost_normalization_uses_only_recognized_cost_fields(
    cost: object,
    expected: float | None,
) -> None:
    task = KnowledgeBenchmarkTask(
        task_id="revenue",
        question="What was total revenue?",
        expected_operation_id="sales.semantic_model.measure.v1",
        is_correct=lambda _payload: True,
    )

    def make_lm(**_kwargs):
        return FakeLM(kwargs={"cache": False}, history=[])

    def run(_task, _arm, lm):
        assert lm.history is not None
        lm.history.append({"cost": cost})
        return _result(answer=30.5, turns=1)

    report = run_knowledge_benchmark(
        tasks=[task],
        repetitions=1,
        seed=0,
        make_lm=make_lm,
        run=run,
    )

    for trial in report.trials:
        if expected is None:
            assert trial.provider_model_cost is None
        else:
            assert trial.provider_model_cost == pytest.approx(expected)


def test_stage_metrics_ignore_untrusted_synthetic_booleans() -> None:
    task = KnowledgeBenchmarkTask(
        task_id="revenue",
        question="What was total revenue?",
        expected_operation_id="sales.semantic_model.measure.v1",
        is_correct=lambda _payload: True,
        plan_validators=KnowledgeBenchmarkPlanValidators(
            measure=lambda _plan: True,
            full_operation_plan=lambda _plan: True,
        ),
    )

    report = run_knowledge_benchmark(
        tasks=[task],
        repetitions=1,
        seed=0,
        make_lm=lambda **_kwargs: FakeLM(kwargs={"cache": False}),
        run=lambda _task, arm, _lm: _result(
            answer=30.5,
            turns=1,
            metadata={
                "knowledge_mode": "registered_operation",
                "operation_id": "sales.semantic_model.measure.v1",
                "measure_correct": True,
                "full_operation_plan_correct": True,
            }
            if arm == "learned"
            else {},
        ),
    )

    learned = next(trial for trial in report.trials if trial.arm == "learned")
    assert learned.measure_correct is None
    assert learned.full_operation_plan_correct is None


@pytest.mark.parametrize(
    ("audit_status", "expected"),
    [
        (None, None),
        ("unknown", None),
        ("passed", True),
        ("failed", False),
    ],
)
def test_audit_status_only_reports_explicit_evidence(
    audit_status: str | None,
    expected: bool | None,
) -> None:
    task = KnowledgeBenchmarkTask(
        task_id="revenue",
        question="What was total revenue?",
        expected_operation_id="sales.semantic_model.measure.v1",
        is_correct=lambda _payload: True,
    )

    report = run_knowledge_benchmark(
        tasks=[task],
        repetitions=1,
        seed=0,
        make_lm=lambda **_kwargs: FakeLM(kwargs={"cache": False}),
        run=lambda _task, arm, _lm: _result(
            answer=30.5,
            turns=1,
            metadata={
                "operation_audit_status": audit_status,
            }
            if arm == "learned" and audit_status is not None
            else {},
        ),
    )

    learned = next(trial for trial in report.trials if trial.arm == "learned")
    assert learned.host_audit_passed is expected


def test_commentary_validity_requires_a_task_validator() -> None:
    without_validator = KnowledgeBenchmarkTask(
        task_id="unvalidated",
        question="What was total revenue?",
        expected_operation_id="sales.semantic_model.measure.v1",
        is_correct=lambda _payload: True,
    )
    with_validator = KnowledgeBenchmarkTask(
        task_id="validated",
        question="What was total revenue?",
        expected_operation_id="sales.semantic_model.measure.v1",
        is_correct=lambda _payload: True,
        is_commentary_valid=lambda payload: bool(
            payload and payload.get("analysis") == "grounded"
        ),
    )

    report = run_knowledge_benchmark(
        tasks=[without_validator, with_validator],
        repetitions=1,
        seed=0,
        make_lm=lambda **_kwargs: FakeLM(kwargs={"cache": False}),
        run=lambda task, _arm, _lm: _result(
            answer=30.5,
            turns=1,
            analysis="grounded" if task.task_id == "validated" else "nonblank",
        ),
    )

    by_task = {
        trial.task_id: trial.LLM_commentary_valid
        for trial in report.trials
        if trial.arm == "learned"
    }
    assert by_task == {"unvalidated": None, "validated": True}


def test_benchmark_rejects_non_natural_questions_and_cache_enabled_lms() -> None:
    with pytest.raises(ValueError, match="natural business language"):
        KnowledgeBenchmarkTask(
            task_id="bad",
            question="Run semantic_model.measure.v1 with Period[Month].",
            expected_operation_id="sales.semantic_model.measure.v1",
            is_correct=lambda _payload: True,
        )

    task = KnowledgeBenchmarkTask(
        task_id="revenue",
        question="What was total revenue?",
        expected_operation_id="sales.semantic_model.measure.v1",
        is_correct=lambda _payload: True,
    )

    with pytest.raises(ValueError, match="cache=False"):
        run_knowledge_benchmark(
            tasks=[task],
            repetitions=1,
            seed=0,
            make_lm=lambda **_kwargs: FakeLM(kwargs={"cache": True}),
            run=lambda *_args: _result(answer=30.5, turns=1),
        )
