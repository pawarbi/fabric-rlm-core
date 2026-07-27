"""A truncated run must say so.

Hitting the turn cap produces a result that looks like a wrong answer, and the
fix is a config change the caller can only make if they know to make it. How
many turns a model needs for the same task varies widely, so a cap tuned
against one model can silently starve another -- which is exactly what makes
this invisible in practice.
"""

from __future__ import annotations

import logging

import pytest

from fabric_rlm import RLM


class _NeverSubmits:
    """An LM that always writes a harmless code block and never calls SUBMIT."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt=None, *, messages=None, **_):
        self.calls += 1
        return ["Working on it.\n\n```python\nx = 1\nprint(x)\n```"]


def _run(max_turns: int):
    rlm = RLM.from_task(
        task="Do something that is never finished.",
        inputs={},
        outputs=["answer"],
        lm=_NeverSubmits(),
        max_turns=max_turns,
        timeout=60.0,
    )
    return rlm.run()


def test_exhaustion_sets_failure_reason():
    result = _run(2)
    assert result.submitted is False
    assert result.failure_reason == "max_turns"


def test_exhaustion_logs_a_warning_naming_the_cap(caplog):
    with caplog.at_level(logging.WARNING, logger="fabric_rlm.runtime"):
        _run(2)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("ran out of turns" in m for m in warnings), warnings
    hit = next(m for m in warnings if "ran out of turns" in m)
    # The message has to carry the number to raise, not just say it failed.
    assert "max_turns" in hit
    assert "2" in hit


def test_result_reports_the_cap_it_was_given():
    result = _run(3)
    assert result.max_turns == 3
    assert len(result.turns) == 3, "a run that never submits should use the whole budget"


def test_turns_used_vs_cap_is_derivable():
    """The pair (len(turns), max_turns) is what tells a caller they are at the ceiling."""

    result = _run(2)
    assert result.max_turns is not None
    assert len(result.turns) == result.max_turns


def test_no_warning_when_the_run_submits(caplog):
    class _Submits:
        def __call__(self, prompt=None, *, messages=None, **_):
            return ['```python\nSUBMIT(answer="done")\n```']

    rlm = RLM.from_task(
        task="Finish immediately.",
        inputs={},
        outputs=["answer"],
        lm=_Submits(),
        max_turns=5,
        timeout=60.0,
    )
    with caplog.at_level(logging.WARNING, logger="fabric_rlm.runtime"):
        result = rlm.run()
    assert result.submitted is True
    assert not any("ran out of turns" in r.getMessage() for r in caplog.records)


def test_successful_run_still_reports_the_cap():
    class _Submits:
        def __call__(self, prompt=None, *, messages=None, **_):
            return ['```python\nSUBMIT(answer="done")\n```']

    rlm = RLM.from_task(
        task="Finish immediately.",
        inputs={},
        outputs=["answer"],
        lm=_Submits(),
        max_turns=7,
        timeout=60.0,
    )
    result = rlm.run()
    assert result.submitted is True
    assert result.max_turns == 7
