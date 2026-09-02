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
PlanValidator = Callable[["KnowledgeBenchmarkSelectedPlan"], bool]
_IMPLEMENTATION_DETAIL = re.compile(
    r"semantic_model\.|operation_id|evaluate_measure|"
    r"\b(?:dax|sql|python)\b|[A-Za-z][A-Za-z0-9 ]*\[[^\]]+\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class KnowledgeBenchmarkSelectedPlan:
    """Host-validated operation selection evidence exposed to benchmark validators."""

    operation_id: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class KnowledgeBenchmarkPlanValidators:
    """Optional task-owned validators for the requested plan-stage metrics."""

    measure: PlanValidator | None = None
    time_policy: PlanValidator | None = None
    groupby: PlanValidator | None = None
    filter_columns: PlanValidator | None = None
    filter_values: PlanValidator | None = None
    full_operation_plan: PlanValidator | None = None

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            validator = getattr(self, field_name)
            if validator is not None and not callable(validator):
                raise TypeError(f"{field_name} plan validator must be callable")


@dataclass(frozen=True)
class KnowledgeBenchmarkTask:
    task_id: str
    question: str
    expected_operation_id: str
    is_correct: Callable[[Mapping[str, Any] | None], bool]
    plan_validators: KnowledgeBenchmarkPlanValidators | None = None
    is_commentary_valid: (
        Callable[[Mapping[str, Any] | None], bool] | None
    ) = None

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
        if (
            self.plan_validators is not None
            and not isinstance(
                self.plan_validators,
                KnowledgeBenchmarkPlanValidators,
            )
        ):
            raise TypeError(
                "plan_validators must be KnowledgeBenchmarkPlanValidators"
            )
        if (
            self.is_commentary_valid is not None
            and not callable(self.is_commentary_valid)
        ):
            raise TypeError("is_commentary_valid must be callable")


@dataclass(frozen=True)
class KnowledgeBenchmarkTrial:
    task_id: str
    repetition: int
    arm: BenchmarkArm
    operation_id_correct: bool | None
    measure_correct: bool | None
    time_policy_correct: bool | None
    groupby_correct: bool | None
    filter_columns_correct: bool | None
    filter_values_correct: bool | None
    full_operation_plan_correct: bool | None
    host_execution_passed: bool | None
    host_audit_passed: bool | None
    deterministic_result_binding_correct: bool | None
    LLM_commentary_valid: bool | None
    final_answer_correct: bool
    submitted: bool
    turns: int
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_prompt_tokens: int | None
    uncached_prompt_tokens: int | None
    lm_seconds: float | None
    provider: str | None
    model: str | None
    provider_model_cost: float | None
    worker_seconds: float | None
    host_seconds: float | None
    wall_seconds: float
    knowledge_fingerprint: str | None
    knowledge_mode: str | None
    operation_id: str | None
    operation_result_fingerprint: str | None

    def to_dict(self) -> dict[str, object]:
        serialized = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }
        serialized.update(
            {
                "numeric_correct": self.numeric_correct,
                "operation_selection_correct": self.operation_selection_correct,
                "audit_passed": self.audit_passed,
            }
        )
        return serialized

    @property
    def numeric_correct(self) -> bool:
        """Deprecated compatibility alias for ``final_answer_correct``."""
        return self.final_answer_correct

    @property
    def operation_selection_correct(self) -> bool | None:
        """Deprecated compatibility alias for ``operation_id_correct``."""
        return self.operation_id_correct

    @property
    def audit_passed(self) -> bool | None:
        """Deprecated compatibility alias for ``host_audit_passed``."""
        return self.host_audit_passed


@dataclass(frozen=True)
class KnowledgeBenchmarkReport:
    seed: int
    repetitions: int
    trials: tuple[KnowledgeBenchmarkTrial, ...]

    def summary(self) -> dict[str, dict[str, float | int | None]]:
        summary: dict[str, dict[str, float | int | None]] = {}
        for arm in ("cold", "learned"):
            trials = [trial for trial in self.trials if trial.arm == arm]
            summary[arm] = {
                "trials": len(trials),
                **{
                    f"{field}_rate": _rate_optional(
                        [getattr(trial, field) for trial in trials]
                    )
                    for field in (
                        "operation_id_correct",
                        "measure_correct",
                        "time_policy_correct",
                        "groupby_correct",
                        "filter_columns_correct",
                        "filter_values_correct",
                        "full_operation_plan_correct",
                        "host_execution_passed",
                        "host_audit_passed",
                        "deterministic_result_binding_correct",
                        "LLM_commentary_valid",
                    )
                },
                "final_answer_correct_rate": _rate(
                    [trial.final_answer_correct for trial in trials]
                ),
                "numeric_correct_rate": _rate(
                    [trial.numeric_correct for trial in trials]
                ),
                "operation_selection_accuracy": _rate_optional(
                    [trial.operation_selection_correct for trial in trials]
                ),
                "audit_pass_rate": _rate_optional(
                    [trial.audit_passed for trial in trials]
                ),
                "mean_turns": _mean([trial.turns for trial in trials]),
                "mean_prompt_tokens": _mean_optional(
                    [trial.prompt_tokens for trial in trials]
                ),
                "mean_completion_tokens": _mean_optional(
                    [trial.completion_tokens for trial in trials]
                ),
                "mean_cached_prompt_tokens": _mean_optional(
                    [trial.cached_prompt_tokens for trial in trials]
                ),
                "mean_uncached_prompt_tokens": _mean_optional(
                    [trial.uncached_prompt_tokens for trial in trials]
                ),
                "mean_lm_seconds": _mean_optional(
                    [trial.lm_seconds for trial in trials]
                ),
                "mean_provider_model_cost": _mean_optional(
                    [trial.provider_model_cost for trial in trials]
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


def _rate_optional(values: Sequence[bool | None]) -> float | None:
    return _rate([value for value in values if value is not None])


def _mean(values: Sequence[int | float]) -> float | None:
    return statistics.fmean(values) if values else None


def _mean_optional(values: Sequence[int | float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _cache_disabled(lm: object) -> bool:
    cache = getattr(lm, "cache", None)
    if cache is not None:
        return cache is False
    kwargs = getattr(lm, "kwargs", None)
    return isinstance(kwargs, Mapping) and kwargs.get("cache") is False


def _selected_plan(
    metadata: Mapping[str, Any],
    *,
    learned: bool,
) -> KnowledgeBenchmarkSelectedPlan | None:
    if not learned:
        return None
    operation_id = metadata.get("operation_id")
    parameters = metadata.get("operation_parameters")
    if not isinstance(operation_id, str) or not operation_id:
        return None
    if not isinstance(parameters, Mapping):
        return None
    return KnowledgeBenchmarkSelectedPlan(
        operation_id=operation_id,
        parameters=dict(parameters),
    )


def _stage_valid(
    plan: KnowledgeBenchmarkSelectedPlan | None,
    validator: PlanValidator | None,
) -> bool | None:
    if plan is None or validator is None:
        return None
    result = validator(plan)
    if not isinstance(result, bool):
        raise TypeError("benchmark plan validators must return bool")
    return result


def _commentary_valid(
    payload: Mapping[str, Any] | None,
    validator: Callable[[Mapping[str, Any] | None], bool] | None,
) -> bool | None:
    if validator is None:
        return None
    result = validator(payload)
    if not isinstance(result, bool):
        raise TypeError("is_commentary_valid must return bool")
    return result


def _audit_passed(value: object, *, learned: bool) -> bool | None:
    if not learned:
        return None
    if value == "passed":
        return True
    if value == "failed":
        return False
    return None


def _uncached_prompt_tokens(
    prompt_tokens: int | None,
    cached_tokens: int | None,
) -> int | None:
    if prompt_tokens is None or cached_tokens is None:
        return None
    if cached_tokens < 0 or cached_tokens > prompt_tokens:
        raise ValueError("cached prompt tokens must be between zero and prompt tokens")
    return prompt_tokens - cached_tokens


def _provider_and_model(lm: object) -> tuple[str | None, str | None]:
    raw_model = getattr(lm, "model", None)
    if not isinstance(raw_model, str) or not raw_model:
        kwargs = getattr(lm, "kwargs", None)
        raw_model = kwargs.get("model") if isinstance(kwargs, Mapping) else None
    if not isinstance(raw_model, str) or not raw_model:
        return None, None
    provider = getattr(lm, "provider", None)
    if not isinstance(provider, str) or not provider:
        provider, separator, model = raw_model.partition("/")
        if separator:
            return provider, model or None
        return None, raw_model
    return provider, raw_model.split("/", 1)[-1]


def _cost_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, Mapping):
        costs = [
            float(item)
            for item in value.values()
            if not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(item)
        ]
        return sum(costs) if costs else None
    return None


def _provider_model_cost(
    lm: object,
    *,
    history_start: int | None,
    result: RLMResult,
) -> float | None:
    history = getattr(lm, "history", None)
    if history_start is not None and isinstance(history, list):
        costs = [
            cost
            for entry in history[history_start:]
            if isinstance(entry, Mapping)
            for cost in [_cost_value(entry.get("cost"))]
            if cost is not None
        ]
        if costs:
            return sum(costs)
    for value in (
        getattr(result, "total_cost", None),
        getattr(result, "total_cost_usd", None),
        result.trajectory.metadata.get("provider_model_cost"),
        result.trajectory.metadata.get("total_cost"),
    ):
        cost = _cost_value(value)
        if cost is not None:
            return cost
    return None


def _trial(
    task: KnowledgeBenchmarkTask,
    *,
    repetition: int,
    arm: BenchmarkArm,
    lm: object,
    run: Callable[[KnowledgeBenchmarkTask, BenchmarkArm, object], RLMResult],
) -> KnowledgeBenchmarkTrial:
    history = getattr(lm, "history", None)
    history_start = len(history) if isinstance(history, list) else None
    started = time.perf_counter()
    result = run(task, arm, lm)
    wall_seconds = time.perf_counter() - started
    if not isinstance(result, RLMResult):
        raise TypeError("benchmark run must return RLMResult")
    metadata = result.trajectory.metadata
    learned = arm == "learned"
    operation_id = metadata.get("operation_id")
    audit_status = metadata.get("operation_audit_status")
    selected_plan = _selected_plan(metadata, learned=learned)
    validators = task.plan_validators or KnowledgeBenchmarkPlanValidators()
    payload = result.payload if isinstance(result.payload, Mapping) else None
    correct = bool(result.submitted and task.is_correct(payload))
    prompt_tokens = result.total_prompt_tokens
    cached_prompt_tokens = result.total_cached_tokens
    provider, model = _provider_and_model(lm)
    host_execution_passed = None
    if learned:
        explicit_host_execution = metadata.get("host_execution_passed")
        if isinstance(explicit_host_execution, bool):
            host_execution_passed = explicit_host_execution
    if learned and host_execution_passed is None:
        host_execution_passed = metadata.get("knowledge_mode") == "registered_operation"
    deterministic_binding = None
    if learned:
        explicit_binding = metadata.get(
            "deterministic_result_binding_correct"
        )
        if isinstance(explicit_binding, bool):
            deterministic_binding = explicit_binding
    if learned and deterministic_binding is None:
        deterministic_binding = bool(
            metadata.get("knowledge_mode") == "registered_operation"
            and _text_optional(metadata.get("operation_result_fingerprint"))
        )
    return KnowledgeBenchmarkTrial(
        task_id=task.task_id,
        repetition=repetition,
        arm=arm,
        operation_id_correct=(
            operation_id == task.expected_operation_id if learned else None
        ),
        measure_correct=_stage_valid(selected_plan, validators.measure),
        time_policy_correct=_stage_valid(
            selected_plan, validators.time_policy
        ),
        groupby_correct=_stage_valid(selected_plan, validators.groupby),
        filter_columns_correct=_stage_valid(
            selected_plan, validators.filter_columns
        ),
        filter_values_correct=_stage_valid(
            selected_plan, validators.filter_values
        ),
        full_operation_plan_correct=_stage_valid(
            selected_plan, validators.full_operation_plan
        ),
        host_execution_passed=host_execution_passed,
        host_audit_passed=_audit_passed(audit_status, learned=learned),
        deterministic_result_binding_correct=deterministic_binding,
        LLM_commentary_valid=_commentary_valid(
            payload,
            task.is_commentary_valid,
        ),
        final_answer_correct=correct,
        submitted=result.submitted,
        turns=result.n_turns,
        prompt_tokens=prompt_tokens,
        completion_tokens=result.total_completion_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        uncached_prompt_tokens=_uncached_prompt_tokens(
            prompt_tokens,
            cached_prompt_tokens,
        ),
        lm_seconds=result.total_lm_seconds,
        provider=provider,
        model=model,
        provider_model_cost=_provider_model_cost(
            lm,
            history_start=history_start,
            result=result,
        ),
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
    "KnowledgeBenchmarkPlanValidators",
    "KnowledgeBenchmarkReport",
    "KnowledgeBenchmarkSelectedPlan",
    "KnowledgeBenchmarkTask",
    "KnowledgeBenchmarkTrial",
    "run_knowledge_benchmark",
]
