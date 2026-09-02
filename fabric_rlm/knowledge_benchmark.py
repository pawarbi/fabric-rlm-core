"""Seeded cold-versus-learned evaluation for knowledge operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import random
import re
import statistics
import time
from typing import Any, Literal

from fabric_rlm.runtime import RLMResult


BenchmarkArm = Literal["cold", "learned"]
_IMPLEMENTATION_DETAIL = re.compile(
    r"semantic_model\.|operation_id|evaluate_measure|"
    r"\b(?:dax|sql|python)\b|[A-Za-z][A-Za-z0-9 ]*\[[^\]]+\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class KnowledgeBenchmarkTask:
    task_id: str
    question: str
    expected_operation_id: str
    is_correct: Callable[[Mapping[str, Any] | None], bool]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        if _IMPLEMENTATION_DETAIL.search(self.question):
            raise ValueError(
                "benchmark questions must use natural business language"
            )
        if (
            not isinstance(self.expected_operation_id, str)
            or not self.expected_operation_id.strip()
        ):
            raise ValueError("expected_operation_id must be a non-empty string")
        if not callable(self.is_correct):
            raise TypeError("is_correct must be callable")


@dataclass(frozen=True)
class KnowledgeBenchmarkTrial:
    task_id: str
    repetition: int
    arm: BenchmarkArm
    numeric_correct: bool
    operation_selection_correct: bool | None
    audit_passed: bool | None
    submitted: bool
    turns: int
    prompt_tokens: int | None
    completion_tokens: int | None
    lm_seconds: float | None
    worker_seconds: float | None
    host_seconds: float | None
    wall_seconds: float
    knowledge_fingerprint: str | None
    knowledge_mode: str | None
    operation_id: str | None
    operation_result_fingerprint: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class KnowledgeBenchmarkReport:
    seed: int
    repetitions: int
    trials: tuple[KnowledgeBenchmarkTrial, ...]

    def summary(self) -> dict[str, dict[str, float | int | None]]:
        summary: dict[str, dict[str, float | int | None]] = {}
        for arm in ("cold", "learned"):
            trials = [trial for trial in self.trials if trial.arm == arm]
            selection = [
                trial.operation_selection_correct
                for trial in trials
                if trial.operation_selection_correct is not None
            ]
            audits = [
                trial.audit_passed
                for trial in trials
                if trial.audit_passed is not None
            ]
            summary[arm] = {
                "trials": len(trials),
                "numeric_correct_rate": _rate(
                    [trial.numeric_correct for trial in trials]
                ),
                "operation_selection_accuracy": _rate(selection),
                "audit_pass_rate": _rate(audits),
                "mean_turns": _mean([trial.turns for trial in trials]),
                "mean_prompt_tokens": _mean_optional(
                    [trial.prompt_tokens for trial in trials]
                ),
                "mean_completion_tokens": _mean_optional(
                    [trial.completion_tokens for trial in trials]
                ),
                "mean_lm_seconds": _mean_optional(
                    [trial.lm_seconds for trial in trials]
                ),
                "mean_worker_seconds": _mean_optional(
                    [trial.worker_seconds for trial in trials]
                ),
                "mean_host_seconds": _mean_optional(
                    [trial.host_seconds for trial in trials]
                ),
                "mean_wall_seconds": _mean(
                    [trial.wall_seconds for trial in trials]
                ),
            }
        return summary


def _rate(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean(values: Sequence[int | float]) -> float | None:
    return statistics.fmean(values) if values else None


def _mean_optional(values: Sequence[int | float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _cache_disabled(lm: object) -> bool:
    kwargs = getattr(lm, "kwargs", None)
    return isinstance(kwargs, Mapping) and kwargs.get("cache") is False


def _trial(
    task: KnowledgeBenchmarkTask,
    *,
    repetition: int,
    arm: BenchmarkArm,
    lm: object,
    run: Callable[[KnowledgeBenchmarkTask, BenchmarkArm, object], RLMResult],
) -> KnowledgeBenchmarkTrial:
    started = time.perf_counter()
    result = run(task, arm, lm)
    wall_seconds = time.perf_counter() - started
    if not isinstance(result, RLMResult):
        raise TypeError("benchmark run must return RLMResult")
    metadata = result.trajectory.metadata
    learned = arm == "learned"
    operation_id = metadata.get("operation_id")
    audit_status = metadata.get("operation_audit_status")
    payload = result.payload if isinstance(result.payload, Mapping) else None
    correct = bool(result.submitted and task.is_correct(payload))
    return KnowledgeBenchmarkTrial(
        task_id=task.task_id,
        repetition=repetition,
        arm=arm,
        numeric_correct=correct,
        operation_selection_correct=(
            operation_id == task.expected_operation_id if learned else None
        ),
        audit_passed=(audit_status == "passed" if learned else None),
        submitted=result.submitted,
        turns=result.n_turns,
        prompt_tokens=result.total_prompt_tokens,
        completion_tokens=result.total_completion_tokens,
        lm_seconds=result.total_lm_seconds,
        worker_seconds=result.total_worker_seconds,
        host_seconds=_finite_optional(metadata.get("operation_host_seconds")),
        wall_seconds=wall_seconds,
        knowledge_fingerprint=_text_optional(
            metadata.get("knowledge_fingerprint")
        ),
        knowledge_mode=_text_optional(metadata.get("knowledge_mode")),
        operation_id=_text_optional(operation_id),
        operation_result_fingerprint=_text_optional(
            metadata.get("operation_result_fingerprint")
        ),
    )


def _finite_optional(value: object) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("benchmark timing metadata must be finite")
    return float(value)


def _text_optional(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def run_knowledge_benchmark(
    *,
    tasks: Sequence[KnowledgeBenchmarkTask],
    repetitions: int,
    seed: int,
    make_lm: Callable[..., object],
    run: Callable[[KnowledgeBenchmarkTask, BenchmarkArm, object], RLMResult],
) -> KnowledgeBenchmarkReport:
    """Run seeded, cache-disabled cold and learned trials."""

    if not tasks:
        raise ValueError("tasks must not be empty")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task IDs must be unique")

    rng = random.Random(seed)
    trials: list[KnowledgeBenchmarkTrial] = []
    for repetition in range(repetitions):
        task_order = list(tasks)
        rng.shuffle(task_order)
        for task in task_order:
            arms: list[BenchmarkArm] = ["cold", "learned"]
            rng.shuffle(arms)
            for arm in arms:
                lm = make_lm(
                    task_id=task.task_id,
                    repetition=repetition,
                    arm=arm,
                    cache=False,
                )
                if not _cache_disabled(lm):
                    raise ValueError("benchmark LM must expose cache=False")
                trials.append(
                    _trial(
                        task,
                        repetition=repetition,
                        arm=arm,
                        lm=lm,
                        run=run,
                    )
                )
    return KnowledgeBenchmarkReport(
        seed=seed,
        repetitions=repetitions,
        trials=tuple(trials),
    )


__all__ = [
    "KnowledgeBenchmarkReport",
    "KnowledgeBenchmarkTask",
    "KnowledgeBenchmarkTrial",
    "run_knowledge_benchmark",
]
