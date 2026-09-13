"""Reshape recorded evaluation trials into calibration inputs.

A trial file records one row per (question, arm, repetition). Calibration needs
two different views of that: the *within-arm* view, where repetitions of an
unchanged configuration are compared against each other to measure noise, and
the *between-arm* view, where configurations are compared to each other. Keeping
both in one place makes it hard to accidentally compute one and report the
other.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "Trial",
    "TrialSet",
    "load_trials",
    "classify_answer",
    "ANSWERED",
    "NO_VALUE",
    "UNSERIALIZABLE",
    "CORRECTNESS_SCORERS",
]


@dataclass(frozen=True)
class Trial:
    question_id: str
    arm: str
    rep: int
    correct: bool
    abstained: bool
    turns: float
    seconds: float
    prompt_tokens: float
    completion_tokens: float
    # Why this trial carries no usable answer, if it does not. Set at load time
    # because attribution needs the raw payload, which is discarded here.
    answer_state: str = "answered"

    @property
    def total_tokens(self) -> float:
        return self.prompt_tokens + self.completion_tokens


ANSWERED = "answered"
NO_VALUE = "no_value"
UNSERIALIZABLE = "unserializable"


def classify_answer(record: dict[str, Any]) -> str:
    """Describe whether a recorded trial carries a usable answer value.

    Distinguishes two mechanical failures from a real answer:

    ``unserializable``
        A value was computed but frozen as an opaque marker — the number was
        right, the transport lost it.
    ``no_value``
        The run submitted without a value at all.
    """
    answer = record.get("answer")
    if not isinstance(answer, dict):
        return NO_VALUE if answer is None else ANSWERED
    value = answer.get("value")
    if isinstance(value, dict) and value.get("__serializable__") is False:
        return UNSERIALIZABLE
    if value is None:
        return NO_VALUE
    return ANSWERED



def _analytic_correct(grade: dict[str, Any]) -> bool:
    return bool(grade.get("analytic_correct"))


def _contract_correct(grade: dict[str, Any]) -> bool:
    return bool(grade.get("contract_correct"))


def _fully_correct(grade: dict[str, Any]) -> bool:
    """Correct on the number *and* on the reporting contract, without a hazard."""
    return (
        bool(grade.get("analytic_correct"))
        and bool(grade.get("contract_correct"))
        and not grade.get("hit_hazard")
    )


CORRECTNESS_SCORERS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "analytic": _analytic_correct,
    "contract": _contract_correct,
    "full": _fully_correct,
}


class TrialSet:
    """A collection of trials, queryable by arm, question and repetition."""

    def __init__(self, trials: list[Trial]) -> None:
        self._trials = trials

    def __len__(self) -> int:
        return len(self._trials)

    @property
    def arms(self) -> list[str]:
        return sorted({trial.arm for trial in self._trials})

    @property
    def reps(self) -> list[int]:
        return sorted({trial.rep for trial in self._trials})

    @property
    def question_ids(self) -> list[str]:
        return sorted({trial.question_id for trial in self._trials})

    def for_arm(self, arm: str) -> list[Trial]:
        return [trial for trial in self._trials if trial.arm == arm]

    def accuracy_by_rep(self, arm: str) -> list[float]:
        """Accuracy in percentage points, one value per repetition.

        Each value is a complete pass over every question, so the variation
        between them is the run-to-run noise of an unchanged configuration.
        """
        by_rep: dict[int, list[Trial]] = defaultdict(list)
        for trial in self.for_arm(arm):
            by_rep[trial.rep].append(trial)
        return [
            100.0 * sum(t.correct for t in trials) / len(trials)
            for _, trials in sorted(by_rep.items())
            if trials
        ]

    def outcomes_by_question(self, arm: str) -> dict[str, list[bool]]:
        """Per-question outcomes across repetitions of one unchanged arm."""
        outcomes: dict[str, list[bool]] = defaultdict(list)
        for trial in sorted(self.for_arm(arm), key=lambda t: t.rep):
            outcomes[trial.question_id].append(trial.correct)
        return dict(outcomes)

    def metric_by_rep(self, arm: str, attribute: str) -> list[float]:
        """A per-question-mean of ``attribute``, one value per repetition."""
        by_rep: dict[int, list[float]] = defaultdict(list)
        for trial in self.for_arm(arm):
            by_rep[trial.rep].append(float(getattr(trial, attribute)))
        return [
            sum(values) / len(values)
            for _, values in sorted(by_rep.items())
            if values
        ]

    def abstention_rate_by_rep(self, arm: str) -> list[float]:
        by_rep: dict[int, list[Trial]] = defaultdict(list)
        for trial in self.for_arm(arm):
            by_rep[trial.rep].append(trial)
        return [
            100.0 * sum(t.abstained for t in trials) / len(trials)
            for _, trials in sorted(by_rep.items())
            if trials
        ]


def load_trials(path: Path, *, scorer: str = "analytic") -> TrialSet:
    """Read a recorded trial file into a :class:`TrialSet`.

    ``scorer`` selects what "correct" means; an evaluation that reports one
    number should still be able to show whether the conclusion depends on that
    choice.
    """
    if scorer not in CORRECTNESS_SCORERS:
        raise ValueError(
            f"unknown scorer {scorer!r}; expected one of "
            f"{sorted(CORRECTNESS_SCORERS)}"
        )
    score = CORRECTNESS_SCORERS[scorer]

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    records = document["trials"] if isinstance(document, dict) else document

    trials: list[Trial] = []
    for record in records:
        grade = record.get("grade") or {}
        trials.append(
            Trial(
                question_id=str(record["question_id"]),
                arm=str(record["arm"]),
                rep=int(record["rep"]),
                correct=score(grade),
                abstained=bool(grade.get("abstained")),
                turns=float(record.get("turns") or 0),
                seconds=float(record.get("wall_seconds") or 0),
                prompt_tokens=float(record.get("prompt_tokens") or 0),
                completion_tokens=float(record.get("completion_tokens") or 0),
                answer_state=classify_answer(record),
            )
        )
    return TrialSet(trials)
