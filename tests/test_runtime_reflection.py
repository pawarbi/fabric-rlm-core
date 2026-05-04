"""Tests for the reflect-before-submit turn in RLM.run()."""

from __future__ import annotations

import fabric_rlm.runtime as runtime_mod
from fabric_rlm import RLM
from fabric_rlm.interpreter import ExecResult


class ScriptedLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.messages: list[list[dict]] = []

    def __call__(self, *, messages):
        self.messages.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("No scripted LM responses left")
        return self.responses.pop(0)


class FakeInterpreter:
    """Stand-in for the real subprocess interpreter; returns scripted ExecResults.

    Supports either pre-built ExecResult instances or a callable raising an
    exception (used to simulate reflection-time worker failures).
    """

    def __init__(self, results: list):
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

    def execute(self, code: str):
        self.executed.append(code)
        if not self._results:
            raise AssertionError("No scripted ExecResults left")
        item = self._results.pop(0)
        if callable(item):
            return item()
        return item


def _install_fake_interpreter(monkeypatch, results: list) -> FakeInterpreter:
    fake = FakeInterpreter(results)
    monkeypatch.setattr(runtime_mod, "Interpreter", lambda **kwargs: fake)
    return fake


def _submit(payload: dict, stdout: str = "") -> ExecResult:
    return ExecResult(
        ok=True,
        submitted=True,
        stdout=stdout,
        stderr="",
        state={},
        submit_payload=payload,
    )


def _ran(stdout: str = "") -> ExecResult:
    return ExecResult(
        ok=True,
        submitted=False,
        stdout=stdout,
        stderr="",
        state={},
    )


def test_reflection_confirms_submit(monkeypatch) -> None:
    """Reflection prints REFLECTION_OK and emits no new SUBMIT -> keep original payload."""
    _install_fake_interpreter(
        monkeypatch,
        [
            _submit({"output": 42}),
            _ran("REFLECTION_OK: looks good"),
        ],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(output=42)\n```",
            "```python\nprint('REFLECTION_OK: looks good')\n```",
        ]
    )
    rlm = RLM.from_task("Return 42.", outputs=["output"], lm=lm, max_turns=4, timeout=5)

    result = rlm.run()

    assert result.submitted is True
    assert result.failure_reason is None
    assert result.payload == {"output": 42}
    assert result.output == 42
    assert result.reflection_used is True
    assert len(result.trajectory) == 2
    assert result.trajectory[0].turn_type == "normal"
    assert result.trajectory[0].submitted is True
    assert result.trajectory[1].turn_type == "reflection"
    assert result.trajectory[1].submitted is False
    # The reflection prompt should have been delivered to the LM on the second call.
    second_messages = lm.messages[1]
    assert any(
        "Final gate before this SUBMIT is finalized" in m["content"]
        for m in second_messages
        if m["role"] == "user"
    )


def test_reflection_corrects_submit(monkeypatch) -> None:
    """Reflection emits a new SUBMIT with a corrected payload -> use the corrected one."""
    _install_fake_interpreter(
        monkeypatch,
        [
            _submit({"output": 42}),
            _submit({"output": 43}, stdout="corrected"),
        ],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(output=42)\n```",
            "```python\nSUBMIT(output=43)\n```",
        ]
    )
    rlm = RLM.from_task("Return the right number.", outputs=["output"], lm=lm, max_turns=4, timeout=5)

    result = rlm.run()

    assert result.submitted is True
    assert result.failure_reason is None
    assert result.payload == {"output": 43}
    assert result.output == 43
    assert result.reflection_used is True
    assert len(result.trajectory) == 2
    turn_types = [t.turn_type for t in result.trajectory]
    assert turn_types == ["normal", "reflection"]
    assert result.trajectory[1].submitted is True


def test_reflection_raises_triggers_repair(monkeypatch) -> None:
    """Reflection raises -> failure recorded as a reflection turn, then a normal repair runs."""

    def _boom() -> ExecResult:
        raise AssertionError("Q5 negative")

    _install_fake_interpreter(
        monkeypatch,
        [
            _submit({"output": -1}),
            _boom,
            _submit({"output": 7}, stdout="repaired"),
        ],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(output=-1)\n```",
            "```python\nassert output >= 0\n```",
            "```python\nSUBMIT(output=7)\n```",
        ]
    )
    rlm = RLM.from_task("Return a non-negative number.", outputs=["output"], lm=lm, max_turns=5, timeout=5)

    result = rlm.run()

    assert result.submitted is True
    assert result.payload == {"output": 7}
    assert result.reflection_used is True
    assert len(result.trajectory) == 3
    assert result.trajectory[0].turn_type == "normal"
    assert result.trajectory[0].submitted is True
    assert result.trajectory[1].turn_type == "reflection"
    assert result.trajectory[1].submitted is False
    assert "AssertionError" in (result.trajectory[1].error or "")
    assert "Q5 negative" in (result.trajectory[1].error or "")
    # The third turn is the repair: a normal LM turn that re-SUBMITs.
    assert result.trajectory[2].submitted is True
    assert result.trajectory[2].turn_type == "validation_repair"
    # Repair feedback referencing the reflection error should reach the LM.
    third_messages = lm.messages[2]
    assert any(
        "Reflection turn raised" in m["content"]
        for m in third_messages
        if m["role"] == "user"
    )


def test_reflection_disabled(monkeypatch) -> None:
    """With enable_reflection=False the runtime preserves the legacy single-turn flow."""
    _install_fake_interpreter(
        monkeypatch,
        [_submit({"output": 99})],
    )
    lm = ScriptedLM(["```python\nSUBMIT(output=99)\n```"])
    rlm = RLM.from_task(
        "Return 99.",
        outputs=["output"],
        lm=lm,
        max_turns=4,
        timeout=5,
        enable_reflection=False,
    )

    result = rlm.run()

    assert result.submitted is True
    assert result.payload == {"output": 99}
    assert result.reflection_used is False
    assert len(result.trajectory) == 1
    assert result.trajectory[0].turn_type == "normal"
    # Only one LM call should have been made (no reflection prompt delivered).
    assert len(lm.messages) == 1


def test_reflection_hard_cap(monkeypatch) -> None:
    """Even if the reflection-SUBMIT is itself invalid, NO second reflection runs.

    Flow: original SUBMIT (valid) -> reflection SUBMIT (invalid: blank) ->
    validation/repair SUBMIT (valid) -> done. Three turns; only one reflection.
    """
    _install_fake_interpreter(
        monkeypatch,
        [
            _submit({"output": 42}),
            _submit({"output": "   "}),  # blank string -> validation failure
            _submit({"output": 100}),
        ],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(output=42)\n```",
            "```python\nSUBMIT(output='   ')\n```",  # reflection produces invalid payload
            "```python\nSUBMIT(output=100)\n```",  # repair turn
        ]
    )
    rlm = RLM.from_task("Return a non-blank value.", outputs=["output"], lm=lm, max_turns=5, timeout=5)

    result = rlm.run()

    assert result.submitted is True
    assert result.payload == {"output": 100}
    assert result.reflection_used is True
    assert len(result.trajectory) == 3
    turn_types = [t.turn_type for t in result.trajectory]
    # First normal SUBMIT, then reflection (invalid payload), then validation_repair SUBMIT.
    assert turn_types == ["normal", "reflection", "validation_repair"]
    # Only one reflection turn ever appears in the trajectory (hard cap).
    assert sum(1 for t in result.trajectory if t.turn_type == "reflection") == 1
    # The middle (reflection) turn carries the validation error.
    assert result.trajectory[1].validation_errors
    assert "blank string" in result.trajectory[1].validation_errors[0]
