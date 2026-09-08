from __future__ import annotations

from dataclasses import dataclass
import sys
from types import SimpleNamespace

from evaluation.generalization.runner import (
    build_schedule,
    make_openrouter_lm,
    normalize_answer,
    result_metrics,
    summarize_trials,
)


def test_schedule_is_seeded_balanced_and_keeps_all_three_arms() -> None:
    questions = [
        {"question_id": "q1", "domain": "inventory"},
        {"question_id": "q2", "domain": "service"},
    ]

    first = build_schedule(questions, repetitions=3, seed=17)
    second = build_schedule(questions, repetitions=3, seed=17)

    assert first == second
    assert len(first) == 18
    assert {
        (row["question_id"], row["repetition"], row["arm"])
        for row in first
    } == {
        (question["question_id"], repetition, arm)
        for question in questions
        for repetition in range(3)
        for arm in ("A", "B", "C")
    }


def test_openrouter_lm_constructor_is_self_contained(monkeypatch) -> None:
    calls = []

    class FakeLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "dspy", SimpleNamespace(LM=FakeLM))

    make_openrouter_lm("openai/gpt-4.1-mini")

    assert calls == [
        {
            "model": "openrouter/openai/gpt-4.1-mini",
            "api_base": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "max_tokens": 4096,
            "cache": False,
            "temperature": 1.0,
        }
    ]


def test_normalize_answer_does_not_turn_failures_into_answers() -> None:
    assert normalize_answer(None, submitted=False, failure_reason="timeout") == {
        "status": "timeout"
    }
    assert normalize_answer("42", submitted=True, failure_reason=None) == {
        "status": "answered",
        "value": "42",
    }
    assert normalize_answer(
        {"status": "abstain", "reason": "missing definition"},
        submitted=True,
        failure_reason=None,
    ) == {"status": "abstain", "reason": "missing definition"}


@dataclass
class FakeResult:
    submitted: bool = True
    failure_reason: str | None = None
    n_turns: int = 2
    max_turns: int = 6
    total_prompt_tokens: int = 100
    total_completion_tokens: int = 20
    total_cached_tokens: int = 40
    total_reasoning_tokens: int = 5
    total_lm_seconds: float = 1.5
    total_worker_seconds: float = 0.4
    integrity_ok: bool = True
    trajectory: object = None


def test_result_metrics_separate_verification_source_calls_and_cost() -> None:
    result = FakeResult()
    result.trajectory = type(
        "Trajectory",
        (),
        {
            "metadata": {
                "verifier_execution": {"verified": False, "failed": 1},
                "source_call_summary": {
                    "source_calls": 3,
                    "failed_source_calls": 1,
                    "source_seconds": 0.7,
                },
                "knowledge_lessons_injected": ["lesson.1"],
            }
        },
    )()

    metrics = result_metrics(result, wall_seconds=2.1, provider_cost_usd=None)

    assert metrics["verification_outcome"] == "not_verified"
    assert metrics["source_calls"] == 3
    assert metrics["failed_source_calls"] == 1
    assert metrics["provider_cost_usd"] is None
    assert metrics["lessons_injected"] == 1


def test_summary_exposes_per_question_regressions_and_learning_harm() -> None:
    trials = [
        {"domain": "inventory", "question_id": "q1", "arm": "A", "grade": {"outcome": "correct"}},
        {"domain": "inventory", "question_id": "q1", "arm": "B", "grade": {"outcome": "confident_wrong"}},
        {"domain": "inventory", "question_id": "q1", "arm": "C", "grade": {"outcome": "incomplete"}},
    ]

    summary = summarize_trials(trials)

    assert summary["per_question"]["q1"]["A"]["correct"] == 1
    assert summary["per_question"]["q1"]["B"]["confident_wrong"] == 1
    assert summary["learning_hurts"] == [
        {"question_id": "q1", "baseline_arm": "A", "worse_arm": "B"},
        {"question_id": "q1", "baseline_arm": "A", "worse_arm": "C"},
    ]
