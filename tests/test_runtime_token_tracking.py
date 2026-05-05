"""Tests for per-turn token usage and timing breakdown captured by RLM.run()."""

from __future__ import annotations

import time

import fabric_rlm.runtime as runtime_module
from fabric_rlm import RLM
from fabric_rlm.runtime import _aggregate_trajectory_metrics
from fabric_rlm.trajectory import Trajectory, TurnRecord


class _LMResponse:
    """Minimal stand-in for an LM response object that carries usage."""

    def __init__(self, content: str, usage: dict | None = None):
        self.content = content
        if usage is not None:
            self.usage = usage


class UsageScriptedLM:
    """Scripted LM whose responses optionally include a ``usage`` dict."""

    def __init__(self, responses):
        # responses is a list of (text, usage_or_None)
        self.responses = list(responses)
        self.delay = 0.0

    def __call__(self, *, messages):
        if not self.responses:
            raise AssertionError("No scripted responses left")
        text, usage = self.responses.pop(0)
        if self.delay:
            time.sleep(self.delay)
        return _LMResponse(text, usage)


class _FakeExecResult:
    def __init__(self, *, submitted=True, payload=None, state=None, stdout="ok"):
        self.stdout = stdout
        self.stderr = ""
        self.error = None
        self.submitted = submitted
        self.submit_payload = payload if payload is not None else {"answer": 1}
        self.state: dict = state or {}
        self.ok = True


class FakeInterpreter:
    def __init__(self, *, result=None, delay=0.0, **_kwargs):
        self._result = result or _FakeExecResult()
        self.delay = delay
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def configure_lm(self, _spec):
        pass

    def set_inputs(self, _inputs):
        pass

    def execute(self, _code):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return self._result


def _install_fake_interpreter(monkeypatch, fake):
    monkeypatch.setattr(runtime_module, "Interpreter", lambda **kwargs: fake)


def test_tokens_captured_when_lm_returns_usage(monkeypatch) -> None:
    _install_fake_interpreter(monkeypatch, FakeInterpreter())
    lm = UsageScriptedLM(
        [
            (
                "```python\nSUBMIT(answer=1)\n```",
                {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            ),
        ]
    )
    rlm = RLM.from_task("Return one.", outputs=["answer"], lm=lm, max_turns=3, timeout=5)

    result = rlm.run()

    assert result.submitted
    assert len(result.trajectory) == 1
    first = result.trajectory[0]
    assert first.prompt_tokens == 100
    assert first.completion_tokens == 50
    assert first.total_tokens == 150
    assert first.token_usage == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    assert result.total_prompt_tokens == 100
    assert result.total_completion_tokens == 50


def test_tokens_none_when_lm_omits_usage(monkeypatch) -> None:
    _install_fake_interpreter(monkeypatch, FakeInterpreter())
    lm = UsageScriptedLM(
        [
            ("```python\nSUBMIT(answer=1)\n```", None),
            ("REFLECTION_OK", None),
        ]
    )
    rlm = RLM.from_task("Return one.", outputs=["answer"], lm=lm, max_turns=3, timeout=5)

    result = rlm.run()

    assert result.submitted
    for turn in result.trajectory:
        assert turn.prompt_tokens is None
        assert turn.completion_tokens is None
        assert turn.total_tokens is None
    # Aggregates use None to distinguish "unknown" from "zero".
    assert result.total_prompt_tokens is None
    assert result.total_completion_tokens is None
    # Timing aggregates are still populated since timing is always measured.
    assert result.total_lm_seconds is not None and result.total_lm_seconds >= 0
    assert result.total_worker_seconds is not None and result.total_worker_seconds >= 0


def test_timing_split_recorded(monkeypatch) -> None:
    fake = FakeInterpreter(delay=0.2)
    _install_fake_interpreter(monkeypatch, fake)
    lm = UsageScriptedLM([("```python\nSUBMIT(answer=1)\n```", None)])
    lm.delay = 0.1
    rlm = RLM.from_task(
        "Return one.",
        outputs=["answer"],
        lm=lm,
        max_turns=2,
        timeout=5,
    )

    result = rlm.run()

    assert result.submitted
    turn = result.trajectory[0]
    assert turn.lm_call_seconds is not None
    assert turn.worker_execute_seconds is not None
    # Allow generous slack for scheduler jitter on slower CI machines.
    assert turn.lm_call_seconds >= 0.08
    assert turn.lm_call_seconds < 0.5
    assert turn.worker_execute_seconds >= 0.18
    assert turn.worker_execute_seconds < 0.6
    assert result.total_lm_seconds == turn.lm_call_seconds
    assert result.total_worker_seconds == turn.worker_execute_seconds


def test_aggregates_skip_none_turns() -> None:
    trajectory = Trajectory()
    trajectory.append(
        TurnRecord(
            turn=1,
            code="",
            stdout="",
            stderr="",
            error=None,
            submitted=False,
            state={},
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
            lm_call_seconds=0.1,
            worker_execute_seconds=0.05,
        )
    )
    trajectory.append(
        TurnRecord(
            turn=2,
            code="",
            stdout="",
            stderr="",
            error=None,
            submitted=False,
            state={},
            # No token data on this turn.
            lm_call_seconds=0.2,
            worker_execute_seconds=0.1,
        )
    )
    trajectory.append(
        TurnRecord(
            turn=3,
            code="",
            stdout="",
            stderr="",
            error=None,
            submitted=True,
            state={},
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            lm_call_seconds=0.3,
            worker_execute_seconds=0.15,
        )
    )

    aggregates = _aggregate_trajectory_metrics(trajectory)

    assert aggregates["total_prompt_tokens"] == 15
    assert aggregates["total_completion_tokens"] == 5
    assert abs(aggregates["total_lm_seconds"] - 0.6) < 1e-9
    assert abs(aggregates["total_worker_seconds"] - 0.3) < 1e-9
