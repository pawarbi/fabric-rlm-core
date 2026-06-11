"""Failure-time truncation hint (reviewer item, validated by trace analysis).

When a turn prints more than ``STDOUT_FEEDBACK_LIMIT`` characters, the model
only sees the head/tail of its own output; a model that over-prints then guesses
the items it could not see. The hint fires at exactly that moment with a
concrete recovery instruction: aggregate in Python, or chunk + ``await
predict(...)``.

Gated by ``FABRIC_RLM_TRUNCATION_HINT`` (on|off), **default off**. An A/B on real
tasks showed deployed-class models self-limit printing (they sample rather than
dump), so the trigger rarely fires on counting/classification workloads; the
hint is retained as an opt-in safety net for genuine large-output dumps. Only
the non-submitting continuation feedback carries it (``_format_feedback``); a
turn that SUBMITs never reaches that path.
"""

from __future__ import annotations

import pytest

from fabric_rlm import RLM
from fabric_rlm.interpreter import ExecResult
from fabric_rlm.runtime import STDOUT_FEEDBACK_LIMIT, _TRUNCATION_HINT_MARKER


def _make_rlm() -> RLM:
    def _lm(*, messages):  # never called in these tests
        raise AssertionError("LM should not be called")

    return RLM.from_task(task="dummy", inputs={}, outputs=["answer"], lm=_lm, max_turns=3)


def _result(stdout: str, *, ok: bool = True) -> ExecResult:
    return ExecResult(
        ok=ok, submitted=False, stdout=stdout, stderr="",
        state={"x": 1}, error=None if ok else "ValueError: boom",
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FABRIC_RLM_TRUNCATION_HINT", raising=False)
    yield


def test_hint_fires_when_stdout_truncated(monkeypatch):
    monkeypatch.setenv("FABRIC_RLM_TRUNCATION_HINT", "on")
    rlm = _make_rlm()
    big = "x" * (STDOUT_FEEDBACK_LIMIT + 5000)
    fb = rlm._format_feedback(_result(big), turn=1)
    assert _TRUNCATION_HINT_MARKER in fb
    # reports the real size so the model grasps how much it is missing
    assert str(len(big)) in fb


def test_hint_mentions_both_python_aggregate_and_predict(monkeypatch):
    monkeypatch.setenv("FABRIC_RLM_TRUNCATION_HINT", "on")
    rlm = _make_rlm()
    fb = rlm._format_feedback(_result("y" * (STDOUT_FEEDBACK_LIMIT + 1)), turn=1)
    low = fb.lower()
    assert "python" in low          # aggregate-in-code path
    assert "predict(" in fb         # chunk + sub-LM path


def test_no_hint_when_stdout_fits(monkeypatch):
    monkeypatch.setenv("FABRIC_RLM_TRUNCATION_HINT", "on")
    rlm = _make_rlm()
    fb = rlm._format_feedback(_result("small output"), turn=1)
    assert _TRUNCATION_HINT_MARKER not in fb


def test_hint_disabled_via_env(monkeypatch):
    monkeypatch.setenv("FABRIC_RLM_TRUNCATION_HINT", "off")
    rlm = _make_rlm()
    fb = rlm._format_feedback(_result("z" * (STDOUT_FEEDBACK_LIMIT + 5000)), turn=1)
    assert _TRUNCATION_HINT_MARKER not in fb


def test_hint_off_by_default():
    # Unset env -> default off -> no hint even when stdout is truncated.
    rlm = _make_rlm()
    fb = rlm._format_feedback(_result("w" * (STDOUT_FEEDBACK_LIMIT + 10)), turn=1)
    assert _TRUNCATION_HINT_MARKER not in fb


def test_hint_also_fires_on_error_turn(monkeypatch):
    # Truncated stdout on a turn that also errored still warns about unseen output.
    monkeypatch.setenv("FABRIC_RLM_TRUNCATION_HINT", "on")
    rlm = _make_rlm()
    fb = rlm._format_feedback(_result("e" * (STDOUT_FEEDBACK_LIMIT + 10), ok=False), turn=2)
    assert _TRUNCATION_HINT_MARKER in fb

