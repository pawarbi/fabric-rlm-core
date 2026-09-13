"""Tests for reshaping recorded trials into calibration inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.calibration.trials import load_trials


def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "trials.json"
    path.write_text(
        json.dumps({"model": "test", "trials": records}), encoding="utf-8"
    )
    return path


def _record(question: str, arm: str, rep: int, **grade) -> dict:
    return {
        "question_id": question,
        "arm": arm,
        "rep": rep,
        "turns": grade.pop("turns", 4),
        "wall_seconds": grade.pop("wall_seconds", 10.0),
        "prompt_tokens": grade.pop("prompt_tokens", 100),
        "completion_tokens": grade.pop("completion_tokens", 50),
        "grade": {
            "analytic_correct": grade.get("analytic_correct", True),
            "contract_correct": grade.get("contract_correct", True),
            "hit_hazard": grade.get("hit_hazard", False),
            "abstained": grade.get("abstained", False),
        },
    }


def test_accuracy_by_rep_gives_one_value_per_repetition(tmp_path) -> None:
    """Each repetition is a full pass, so its accuracy is a single sample."""
    path = _write(
        tmp_path,
        [
            _record("q1", "A", 0, analytic_correct=True),
            _record("q2", "A", 0, analytic_correct=False),
            _record("q1", "A", 1, analytic_correct=True),
            _record("q2", "A", 1, analytic_correct=True),
        ],
    )

    trials = load_trials(path)

    assert trials.accuracy_by_rep("A") == [50.0, 100.0]
    assert trials.reps == [0, 1]


def test_outcomes_by_question_are_ordered_by_repetition(tmp_path) -> None:
    path = _write(
        tmp_path,
        [
            _record("q1", "A", 2, analytic_correct=True),
            _record("q1", "A", 0, analytic_correct=False),
            _record("q1", "A", 1, analytic_correct=False),
        ],
    )

    assert load_trials(path).outcomes_by_question("A") == {
        "q1": [False, False, True]
    }


def test_the_full_scorer_requires_contract_and_no_hazard(tmp_path) -> None:
    """A right number reported on the wrong basis is not a correct answer."""
    path = _write(
        tmp_path,
        [
            _record("q1", "A", 0, analytic_correct=True, contract_correct=False),
            _record("q2", "A", 0, analytic_correct=True, hit_hazard=True),
        ],
    )

    assert load_trials(path, scorer="analytic").accuracy_by_rep("A") == [100.0]
    assert load_trials(path, scorer="full").accuracy_by_rep("A") == [0.0]


def test_arms_are_kept_apart(tmp_path) -> None:
    path = _write(
        tmp_path,
        [
            _record("q1", "A", 0, analytic_correct=True),
            _record("q1", "B", 0, analytic_correct=False),
        ],
    )
    trials = load_trials(path)

    assert trials.arms == ["A", "B"]
    assert trials.accuracy_by_rep("A") == [100.0]
    assert trials.accuracy_by_rep("B") == [0.0]


def test_token_metric_sums_prompt_and_completion(tmp_path) -> None:
    path = _write(
        tmp_path,
        [_record("q1", "A", 0, prompt_tokens=900, completion_tokens=100)],
    )

    assert load_trials(path).metric_by_rep("A", "total_tokens") == [1000.0]


def test_an_unknown_scorer_is_rejected(tmp_path) -> None:
    path = _write(tmp_path, [_record("q1", "A", 0)])

    with pytest.raises(ValueError, match="unknown scorer"):
        load_trials(path, scorer="vibes")


def test_a_missing_grade_counts_as_incorrect_not_a_crash(tmp_path) -> None:
    """A trial that died before grading is a failure, not missing data."""
    record = _record("q1", "A", 0)
    record["grade"] = None
    path = _write(tmp_path, [record])

    assert load_trials(path).accuracy_by_rep("A") == [0.0]
