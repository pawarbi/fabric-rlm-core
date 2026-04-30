"""Tests for visible stdout/stderr truncation in LM feedback."""

from __future__ import annotations

import importlib

import pytest

from fabric_rlm import RLM
from fabric_rlm import runtime as runtime_mod
from fabric_rlm.interpreter import ExecResult
from fabric_rlm.runtime import (
    STDERR_FEEDBACK_LIMIT,
    STDOUT_FEEDBACK_LIMIT,
    _truncate_for_feedback,
)


class ScriptedLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.messages: list[list[dict]] = []

    def __call__(self, *, messages):
        self.messages.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("No scripted responses left")
        return self.responses.pop(0)


class FakeInterpreter:
    """Stand-in for the real subprocess interpreter; returns scripted ExecResults."""

    def __init__(self, results: list[ExecResult]):
        self._results = list(results)
        self.executed: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def configure_lm(self, spec):
        return {}

    def set_inputs(self, inputs):
        return None

    def execute(self, code: str) -> ExecResult:
        self.executed.append(code)
        if not self._results:
            raise AssertionError("No scripted ExecResults left")
        return self._results.pop(0)


def _install_fake_interpreter(monkeypatch, results: list[ExecResult]) -> FakeInterpreter:
    fake = FakeInterpreter(results)

    def factory(*args, **kwargs):
        return fake

    monkeypatch.setattr(runtime_mod, "Interpreter", factory)
    return fake


def test_short_stdout_unchanged() -> None:
    text = "hello world"
    assert _truncate_for_feedback(text, 5000) == text


def test_long_stdout_truncated_with_marker() -> None:
    text = "a" * 10000
    out = _truncate_for_feedback(text, 5000)
    marker = "\n... (truncated 5000 more chars)"
    assert out == "a" * 5000 + marker
    assert len(out) == 5000 + len(marker)
    assert out.endswith("(truncated 5000 more chars)")


def test_truncate_handles_none() -> None:
    assert _truncate_for_feedback(None, 5000) == ""  # type: ignore[arg-type]


def test_env_var_override(monkeypatch) -> None:
    """Re-importing the runtime with the env var set picks up the new limit."""
    monkeypatch.setenv("FABRIC_RLM_STDOUT_LIMIT", "100")
    monkeypatch.setenv("FABRIC_RLM_STDERR_LIMIT", "100")
    reloaded = importlib.reload(runtime_mod)
    try:
        assert reloaded.STDOUT_FEEDBACK_LIMIT == 100
        assert reloaded.STDERR_FEEDBACK_LIMIT == 100
        text = "x" * 250
        out = reloaded._truncate_for_feedback(text, reloaded.STDOUT_FEEDBACK_LIMIT)
        assert out.startswith("x" * 100)
        assert out.endswith("(truncated 150 more chars)")
    finally:
        # Restore module-level constants for the rest of the test session.
        monkeypatch.delenv("FABRIC_RLM_STDOUT_LIMIT", raising=False)
        monkeypatch.delenv("FABRIC_RLM_STDERR_LIMIT", raising=False)
        importlib.reload(runtime_mod)


def test_full_stdout_preserved_in_trajectory(monkeypatch) -> None:
    big = "Z" * 10000
    results = [
        ExecResult(ok=True, submitted=False, stdout=big, stderr="", state={}),
        ExecResult(
            ok=True,
            submitted=True,
            stdout="",
            stderr="",
            state={},
            submit_payload={"answer": 1},
        ),
    ]
    _install_fake_interpreter(monkeypatch, results)

    lm = ScriptedLM(
        [
            "```python\nprint('big')\n```",
            "```python\nSUBMIT(answer=1)\n```",
        ]
    )
    rlm = RLM.from_task(
        "Return 1.", outputs=["answer"], lm=lm, max_turns=3, timeout=5, enable_reflection=False
    )
    result = rlm.run()

    assert result.submitted
    # Trajectory keeps the FULL stdout.
    assert result.trajectory[0].stdout == big
    assert len(result.trajectory[0].stdout) == 10000

    # Second LM call sees the feedback for turn 1; it must be truncated with marker.
    second_call_messages = lm.messages[1]
    feedback = second_call_messages[-1]["content"]
    assert "REPL output from turn 1" in feedback
    assert "(truncated" in feedback and "more chars)" in feedback
    # The literal full payload must NOT appear in the LM-visible feedback.
    assert big not in feedback
    expected_extra = 10000 - STDOUT_FEEDBACK_LIMIT
    assert f"(truncated {expected_extra} more chars)" in feedback


def test_stderr_same_treatment(monkeypatch) -> None:
    big_err = "E" * 10000
    results = [
        ExecResult(ok=True, submitted=False, stdout="ok", stderr=big_err, state={}),
        ExecResult(
            ok=True,
            submitted=True,
            stdout="",
            stderr="",
            state={},
            submit_payload={"answer": 1},
        ),
    ]
    _install_fake_interpreter(monkeypatch, results)

    lm = ScriptedLM(
        [
            "```python\nimport sys; sys.stderr.write('noise')\n```",
            "```python\nSUBMIT(answer=1)\n```",
        ]
    )
    rlm = RLM.from_task(
        "Return 1.", outputs=["answer"], lm=lm, max_turns=3, timeout=5, enable_reflection=False
    )
    result = rlm.run()

    assert result.submitted
    # Trajectory keeps full stderr.
    assert result.trajectory[0].stderr == big_err

    feedback = lm.messages[1][-1]["content"]
    assert "stderr:" in feedback
    expected_extra = 10000 - STDERR_FEEDBACK_LIMIT
    assert f"(truncated {expected_extra} more chars)" in feedback
    assert big_err not in feedback


def test_truncate_with_explicit_limit_arg() -> None:
    text = "y" * 200
    out = _truncate_for_feedback(text, 50)
    assert out.startswith("y" * 50)
    assert out.endswith("(truncated 150 more chars)")
