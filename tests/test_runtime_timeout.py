"""Tests for RLM.run() capturing worker timeouts/errors as trajectory turns."""

from __future__ import annotations

import fabric_rlm.runtime as runtime_module
from fabric_rlm import RLM
from fabric_rlm.interpreter import WorkerTimeout


class ScriptedLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.messages: list[list[dict]] = []

    def __call__(self, *, messages):
        self.messages.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("No scripted responses left")
        return self.responses.pop(0)


class _FakeExecResult:
    def __init__(self) -> None:
        self.stdout = "ok"
        self.stderr = ""
        self.error = None
        self.submitted = True
        self.submit_payload = {"answer": 1}
        self.state: dict = {}
        self.ok = True


class FakeInterpreter:
    """Minimal Interpreter stand-in supporting the context-manager protocol."""

    def __init__(self, *, exc_factory=None, result=None, **_kwargs):
        self._exc_factory = exc_factory
        self._result = result or _FakeExecResult()
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
        if self._exc_factory is not None:
            raise self._exc_factory()
        return self._result


def _install_fake_interpreter(monkeypatch, fake):
    monkeypatch.setattr(runtime_module, "Interpreter", lambda **kwargs: fake)


def test_worker_timeout_captured_as_turn(monkeypatch) -> None:
    fake = FakeInterpreter(exc_factory=lambda: WorkerTimeout("Worker timed out after 360s"))
    _install_fake_interpreter(monkeypatch, fake)

    lm = ScriptedLM(["```python\nprint('hello')\n```"])
    rlm = RLM.from_task("Say hi.", outputs=["answer"], lm=lm, max_turns=3, timeout=5)

    result = rlm.run()

    assert not result.submitted
    assert result.failure_reason == "worker_timeout"
    assert len(result.trajectory) == 1
    last = result.trajectory[-1]
    assert "WorkerTimeout" in (last.error or "")
    assert "timed out" in (last.error or "")
    assert "print('hello')" in last.code
    # Should not have retried after a timeout.
    assert fake.calls == 1
    # Trajectory entry should round-trip through serialization.
    assert "error" in last.to_dict()


def test_generic_worker_error_captured_as_turn(monkeypatch) -> None:
    fake = FakeInterpreter(exc_factory=lambda: RuntimeError("kaboom"))
    _install_fake_interpreter(monkeypatch, fake)

    lm = ScriptedLM(["```python\nx = 1\n```"])
    rlm = RLM.from_task("Set x.", outputs=["answer"], lm=lm, max_turns=3, timeout=5)

    result = rlm.run()

    assert not result.submitted
    assert result.failure_reason == "worker_error"
    assert len(result.trajectory) == 1
    last = result.trajectory[-1]
    assert "RuntimeError" in (last.error or "")
    assert "kaboom" in (last.error or "")
    assert "x = 1" in last.code
    assert fake.calls == 1


def test_happy_path_with_fake_interpreter_still_submits(monkeypatch) -> None:
    fake = FakeInterpreter()
    _install_fake_interpreter(monkeypatch, fake)

    lm = ScriptedLM(["```python\nSUBMIT(answer=1)\n```"])
    rlm = RLM.from_task("Return one.", outputs=["answer"], lm=lm, max_turns=2, timeout=5, enable_reflection=False)

    result = rlm.run()

    assert result.submitted
    assert result.payload == {"answer": 1}
    assert len(result.trajectory) == 1
    assert result.trajectory[-1].error is None
