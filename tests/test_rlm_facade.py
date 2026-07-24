"""Slice 2 tests: fabric_rlm.RLM(engine='v7-dspy') facade.

The v7-dspy engine delegates the loop to dspy.predict.RLM while keeping our
SubprocessPythonInterpreter as the code execution backend. These tests verify:
  - The facade returns a v6.5-compatible RLMResult shape.
  - Payload + trajectory are populated from dspy.Prediction.
  - Engine flag validation.
"""

from __future__ import annotations

from typing import Any

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


def test_submit_byte_limit_is_configurable() -> None:
    rlm = RLM(signature="q -> answer", lm=_StubLM(), max_submit_bytes=1234)

    assert rlm.max_submit_bytes == 1234


@pytest.mark.parametrize("limit", [0, -1])
def test_rlm_rejects_nonpositive_submit_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="max_submit_bytes must be greater than zero"):
        RLM(signature="q -> answer", lm=_StubLM(), max_submit_bytes=limit)


def test_adaptive_engine_propagates_submit_byte_limit() -> None:
    with pytest.warns(UserWarning, match="engine='adaptive' is experimental"):
        rlm = RLM(
            signature="q -> answer",
            lm=_StubLM(),
            engine="adaptive",
            max_submit_bytes=1234,
        )

    assert rlm._adaptive_inner_kwargs["max_submit_bytes"] == 1234


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


# ----- tools= kwarg plumbing (v7-dspy only) ----------------------------------


def test_tools_kwarg_accepted_on_v7_engine() -> None:
    """tools= is stored on the RLM instance when engine='v7-dspy'."""

    def my_tool(x: int) -> int:
        return x + 1

    rlm = RLM(
        signature="q -> a",
        lm=_StubLM(),
        engine="v7-dspy",
        tools=[my_tool],
    )
    assert rlm.tools == [my_tool]


def test_tools_kwarg_defaults_to_empty_list() -> None:
    """When tools= is omitted, self.tools is an empty list (not None)."""
    rlm = RLM(signature="q -> a", lm=_StubLM(), engine="v7-dspy")
    assert rlm.tools == []


def test_tools_kwarg_rejected_on_v6_custom_engine() -> None:
    """The legacy Interpreter has no tool-call protocol; refuse loudly."""

    def my_tool() -> str:
        return "ok"

    with pytest.raises(NotImplementedError, match="v7-dspy"):
        RLM(
            signature="q -> a",
            lm=_StubLM(),
            engine="v6-custom",
            tools=[my_tool],
        )


def test_tools_kwarg_rejected_on_adaptive_engine() -> None:
    """Adaptive wrapper doesn't carry tools through; refuse rather than drop."""

    def my_tool() -> str:
        return "ok"

    with pytest.raises(NotImplementedError, match="v7-dspy"):
        RLM(
            signature="q -> a",
            lm=_StubLM(),
            engine="adaptive",
            tools=[my_tool],
        )


def test_tools_kwarg_passes_through_to_dspy_rlm(monkeypatch) -> None:
    """The tools list reaches dspy.predict.RLM as a kwarg.

    This is the critical regression guard: if the seam in runtime.py
    drops the tools list, the LM never learns about the tools.
    """
    captured_kwargs: dict[str, Any] = {}

    class _SpyDspyRLM:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        def __call__(self, **inputs: Any):
            class _P:
                answer = 1
            return _P()

    import dspy.predict as _dspy_predict
    monkeypatch.setattr(_dspy_predict, "RLM", _SpyDspyRLM)

    def my_tool(x: int) -> int:
        return x

    rlm = RLM(
        signature="q -> a",
        lm=_StubLM(),
        engine="v7-dspy",
        tools=[my_tool],
    )
    rlm(q="hi")

    assert "tools" in captured_kwargs, (
        f"DspyRLM was called without 'tools' kwarg. Got: {sorted(captured_kwargs)}"
    )
    assert captured_kwargs["tools"] == [my_tool]


def test_no_tools_means_no_tools_kwarg_to_dspy(monkeypatch) -> None:
    """When no tools are registered, we omit the kwarg entirely."""
    captured_kwargs: dict[str, Any] = {}

    class _SpyDspyRLM:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        def __call__(self, **inputs: Any):
            class _P:
                answer = 1
            return _P()

    import dspy.predict as _dspy_predict
    monkeypatch.setattr(_dspy_predict, "RLM", _SpyDspyRLM)

    rlm = RLM(signature="q -> a", lm=_StubLM(), engine="v7-dspy")
    rlm(q="hi")

    assert "tools" not in captured_kwargs, (
        f"DspyRLM was called with 'tools' kwarg even though none were registered: "
        f"{captured_kwargs.get('tools')!r}"
    )


def test_tools_kwarg_rejects_non_callable() -> None:
    """A non-callable item in tools= should fail at construction, not later."""
    with pytest.raises(TypeError, match=r"tools\[1\] is not callable"):
        RLM(
            signature="q -> a",
            lm=_StubLM(),
            engine="v7-dspy",
            tools=[lambda: None, "not_a_function"],  # type: ignore[list-item]
        )


def test_tools_kwarg_accepts_generator() -> None:
    """An iterable (generator) is snapshotted to a list once."""

    def my_tool() -> str:
        return "ok"

    rlm = RLM(
        signature="q -> a",
        lm=_StubLM(),
        engine="v7-dspy",
        tools=(t for t in [my_tool]),
    )
    assert rlm.tools == [my_tool]


# Note: a real-path tool round-trip test (live dspy.RLM + SubprocessPythonInterpreter
# + tool callback) is not included here because dspy switches to a tool-call
# response format when tools= is set, which the canned _StubLM cannot mimic.
# End-to-end validation across three scenarios with openai/gpt-4.1-mini lives in
# experiments/tools_e2e/RESULTS.md on the tools-e2e-exploration branch.
