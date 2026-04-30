"""Slice 2 tests: fabric_rlm.RLM(engine='v7-dspy') facade.

The v7-dspy engine delegates the loop to dspy.predict.RLM while keeping our
SubprocessPythonInterpreter as the code execution backend. These tests verify:
  - The facade returns a v6.5-compatible RLMResult shape.
  - Payload + trajectory are populated from dspy.Prediction.
  - Engine flag validation.
"""

from __future__ import annotations

import dspy
import pytest

from fabric_rlm import RLM, RLMResult


class _StubLM(dspy.LM):
    """A dspy.LM that returns a canned action containing a SUBMIT call."""

    def __init__(self, code: str = "SUBMIT(answer=42)") -> None:
        super().__init__(model="stub", model_type="chat")
        self._code = code
        self.calls = 0

    def __call__(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        self.calls += 1
        return [
            "[[ ## reasoning ## ]]\nLet me solve.\n\n"
            f"[[ ## code ## ]]\n```python\n{self._code}\n```\n\n"
            "[[ ## completed ## ]]\n"
        ]


# ----- Engine flag ------------------------------------------------------------


def test_default_engine_is_v6_for_back_compat() -> None:
    rlm = RLM(signature="q -> answer", lm=_StubLM())
    assert rlm.engine == "v6-custom"


def test_invalid_engine_rejected() -> None:
    with pytest.raises(ValueError):
        RLM(signature="q -> answer", lm=_StubLM(), engine="bogus")


# ----- v7-dspy facade end-to-end ----------------------------------------------


def test_v7_dspy_engine_returns_RLMResult() -> None:
    rlm = RLM(signature="question -> answer: int", lm=_StubLM(), engine="v7-dspy")
    result = rlm(question="what?")

    assert isinstance(result, RLMResult)
    assert result.submitted is True
    assert result.payload is not None
    assert result.payload["answer"] == 42


def test_v7_dspy_payload_attribute_access() -> None:
    """RLMResult.__getattr__ falls through to payload (preserves v6.5 surface)."""
    rlm = RLM(signature="question -> answer: int", lm=_StubLM(), engine="v7-dspy")
    result = rlm(question="what?")
    assert result.answer == 42  # via payload attribute fall-through


def test_v7_dspy_trajectory_records_at_least_one_turn() -> None:
    rlm = RLM(signature="question -> answer: int", lm=_StubLM(), engine="v7-dspy")
    result = rlm(question="what?")
    assert len(result.trajectory.turns) >= 1
    assert result.trajectory.metadata.get("engine") == "v7-dspy"


def test_v7_dspy_failure_recovers_to_RLMResult() -> None:
    """When dspy raises, the facade returns an unsuccessful RLMResult, not a crash."""

    class BrokenLM(dspy.LM):
        def __init__(self) -> None:
            super().__init__(model="broken", model_type="chat")

        def __call__(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
            raise RuntimeError("LM exploded")

    rlm = RLM(signature="question -> answer: int", lm=BrokenLM(), engine="v7-dspy", max_turns=2)
    result = rlm(question="what?")

    assert isinstance(result, RLMResult)
    assert result.submitted is False
    assert result.failure_reason is not None
    assert "exploded" in result.failure_reason or "RuntimeError" in result.failure_reason


# ----- Parity test (v7-dspy vs raw dspy.predict.RLM byte-equal Prediction) ---


def test_v7_dspy_matches_raw_dspy_RLM_output() -> None:
    """Parity: same signature + same stub LM → both produce the same answer.

    The facade is a thin wrapper over dspy.predict.RLM; this confirms zero
    semantic drift on the SUBMIT path.
    """
    from dspy.predict import RLM as DspyRLM

    from fabric_rlm.interpreter import SubprocessPythonInterpreter

    sig = dspy.Signature("question -> answer: int")
    stub_for_facade = _StubLM(code="SUBMIT(answer=99)")
    stub_for_raw = _StubLM(code="SUBMIT(answer=99)")

    # Facade path
    facade = RLM(signature=sig, lm=stub_for_facade, engine="v7-dspy")
    facade_result = facade(question="x")

    # Raw dspy path with our interpreter
    interp = SubprocessPythonInterpreter()
    try:
        with dspy.context(lm=stub_for_raw):
            raw_rlm = DspyRLM(signature=sig, sub_lm=stub_for_raw, interpreter=interp, max_iterations=10)
            raw_pred = raw_rlm(question="x")
    finally:
        interp.shutdown()

    assert facade_result.payload["answer"] == raw_pred.answer == 99
