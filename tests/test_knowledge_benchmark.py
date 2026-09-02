from __future__ import annotations

from dataclasses import dataclass

import pytest

from fabric_rlm.knowledge_benchmark import (
    KnowledgeBenchmarkTask,
    run_knowledge_benchmark,
)
from fabric_rlm.runtime import RLMResult
from fabric_rlm.trajectory import Trajectory, TurnRecord


@dataclass
class FakeLM:
    kwargs: dict


def _result(
    *,
    answer: float,
    turns: int,
    metadata: dict | None = None,
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
        payload={"answer": answer},
        trajectory=trajectory,
        final_state={},
        total_prompt_tokens=100 * turns,
        total_completion_tokens=20 * turns,
        total_lm_seconds=0.5 * turns,
        total_worker_seconds=0.25 * turns,
        max_turns=turns,
    )


def test_repeated_benchmark_is_seeded_randomized_and_disables_cache() -> None:
    task = KnowledgeBenchmarkTask(
        task_id="revenue",
        question="What was total revenue?",
        expected_operation_id="sales.semantic_model.measure.v1",
        is_correct=lambda payload: payload == {"answer": 30.5},
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


def test_benchmark_records_correctness_selection_audit_and_performance() -> None:
    task = KnowledgeBenchmarkTask(
        task_id="revenue",
        question="What was total revenue?",
        expected_operation_id="sales.semantic_model.measure.v1",
        is_correct=lambda payload: payload == {"answer": 30.5},
    )

    def make_lm(**_kwargs):
        return FakeLM(kwargs={"cache": False})

    def run(_task, arm, _lm):
        if arm == "cold":
            return _result(answer=30.5, turns=4)
        return _result(
            answer=30.5,
            turns=1,
            metadata={
                "knowledge_fingerprint": "package-fingerprint",
                "knowledge_mode": "registered_operation",
                "operation_id": "sales.semantic_model.measure.v1",
                "operation_audit_status": "passed",
                "operation_host_seconds": 0.2,
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

    assert rows["cold"]["numeric_correct"] is True
    assert rows["cold"]["operation_selection_correct"] is None
    assert rows["learned"]["numeric_correct"] is True
    assert rows["learned"]["operation_selection_correct"] is True
    assert rows["learned"]["audit_passed"] is True
    assert rows["learned"]["turns"] == 1
    assert rows["learned"]["prompt_tokens"] == 100
    assert rows["learned"]["completion_tokens"] == 20
    assert rows["learned"]["lm_seconds"] == 0.5
    assert rows["learned"]["worker_seconds"] == 0.25
    assert rows["learned"]["host_seconds"] == 0.2
    assert rows["learned"]["knowledge_fingerprint"] == "package-fingerprint"

    summary = report.summary()
    assert summary["cold"]["numeric_correct_rate"] == 1.0
    assert summary["learned"]["numeric_correct_rate"] == 1.0
    assert summary["learned"]["operation_selection_accuracy"] == 1.0
    assert summary["learned"]["audit_pass_rate"] == 1.0
    assert summary["learned"]["mean_turns"] == 1.0


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
