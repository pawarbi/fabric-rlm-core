from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.calibration.attribution import (
    REASONING,
    flip_causes,
    mechanical_share,
)
from evaluation.calibration.trials import (
    ANSWERED,
    NO_VALUE,
    UNSERIALIZABLE,
    classify_answer,
    load_trials,
)


# --- classify_answer ----------------------------------------------------

def test_plain_value_is_answered() -> None:
    assert classify_answer({"answer": {"value": 334}}) == ANSWERED


def test_zero_is_answered_not_missing() -> None:
    # 0 and False are falsy but are perfectly good answers.
    assert classify_answer({"answer": {"value": 0}}) == ANSWERED
    assert classify_answer({"answer": {"value": False}}) == ANSWERED


def test_null_value_is_no_value() -> None:
    assert classify_answer({"answer": {"value": None}}) == NO_VALUE


def test_missing_answer_is_no_value() -> None:
    assert classify_answer({"answer": None}) == NO_VALUE


def test_opaque_marker_is_unserializable() -> None:
    record = {"answer": {"value": {
        "__type__": "int64",
        "__repr__": "np.int64(334)",
        "__serializable__": False,
    }}}
    assert classify_answer(record) == UNSERIALIZABLE


def test_ordinary_dict_value_is_answered() -> None:
    # A dict answer is legitimate; only the opaque marker is not.
    assert classify_answer({"answer": {"value": {"north": 1}}}) == ANSWERED


# --- flip_causes --------------------------------------------------------

def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "trials.json"
    path.write_text(json.dumps({"trials": records}), encoding="utf-8")
    return path


def _record(qid: str, rep: int, correct: bool, value: object) -> dict:
    return {
        "question_id": qid, "arm": "A", "rep": rep,
        "answer": {"value": value},
        "grade": {"analytic_correct": correct},
    }


def test_stable_question_is_not_counted(tmp_path: Path) -> None:
    path = _write(tmp_path, [_record("q1", r, True, 5) for r in range(3)])
    assert flip_causes(load_trials(path), "A") == {}


def test_consistently_wrong_question_is_not_noise(tmp_path: Path) -> None:
    path = _write(tmp_path, [_record("q1", r, False, 5) for r in range(3)])
    assert flip_causes(load_trials(path), "A") == {}


def test_flip_with_values_everywhere_is_reasoning(tmp_path: Path) -> None:
    path = _write(tmp_path, [
        _record("q1", 0, True, 5),
        _record("q1", 1, True, 5),
        _record("q1", 2, False, 9),
    ])
    assert flip_causes(load_trials(path), "A") == {REASONING: 1}


def test_flip_involving_a_missing_value_is_mechanical(tmp_path: Path) -> None:
    path = _write(tmp_path, [
        _record("q1", 0, True, 5),
        _record("q1", 1, True, 5),
        _record("q1", 2, False, None),
    ])
    assert flip_causes(load_trials(path), "A") == {NO_VALUE: 1}


def test_flip_involving_an_opaque_marker_is_mechanical(tmp_path: Path) -> None:
    marker = {"__type__": "int64", "__repr__": "np.int64(5)",
              "__serializable__": False}
    path = _write(tmp_path, [
        _record("q1", 0, False, marker),
        _record("q1", 1, True, 5),
        _record("q1", 2, True, 5),
    ])
    assert flip_causes(load_trials(path), "A") == {UNSERIALIZABLE: 1}


def test_a_question_hitting_both_causes_is_reported_as_both(tmp_path: Path) -> None:
    marker = {"__type__": "int64", "__repr__": "np.int64(5)",
              "__serializable__": False}
    path = _write(tmp_path, [
        _record("q1", 0, False, marker),
        _record("q1", 1, False, None),
        _record("q1", 2, True, 5),
    ])
    assert flip_causes(load_trials(path), "A") == {"no_value+unserializable": 1}


# --- mechanical_share ---------------------------------------------------

def test_mechanical_share_of_nothing_is_zero() -> None:
    from collections import Counter
    assert mechanical_share(Counter()) == 0.0


def test_mechanical_share_splits_causes() -> None:
    from collections import Counter
    causes = Counter({NO_VALUE: 9, UNSERIALIZABLE: 3, REASONING: 5})
    assert mechanical_share(causes) == pytest.approx(12 / 17)


def test_mechanical_share_is_one_when_all_mechanical() -> None:
    from collections import Counter
    assert mechanical_share(Counter({NO_VALUE: 4})) == 1.0


def test_mechanical_share_is_zero_when_all_reasoning() -> None:
    from collections import Counter
    assert mechanical_share(Counter({REASONING: 4})) == 0.0
