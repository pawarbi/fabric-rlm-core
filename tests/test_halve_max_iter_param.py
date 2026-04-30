"""Tests for the halve_max_iter_on_retry parameter (default True)."""
from __future__ import annotations

import dspy
import pytest

from fabric_rlm import RLM


class _CountingScriptedLM(dspy.LM):
    """Captures every call so we can later inspect what max_iterations dspy was given.

    We don't observe max_iter directly; instead we use the fact that on rejection
    the verifier loop re-instantiates DspyRLM. We monkeypatch DspyRLM in the test
    to capture max_iterations.
    """

    def __init__(self, scripted_codes: list[str]) -> None:
        super().__init__(model="scripted", model_type="chat")
        self._codes = list(scripted_codes)
        self.calls = 0

    def __call__(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        if self._codes:
            code = self._codes.pop(0)
        else:
            code = "SUBMIT(answer=999)"
        self.calls += 1
        return [
            "[[ ## reasoning ## ]]\nT.\n\n"
            f"[[ ## code ## ]]\n```python\n{code}\n```\n\n"
            "[[ ## completed ## ]]\n"
        ]


def _validator_must_be_42(payload: dict) -> None:
    assert payload.get("answer") == 42, "answer must be 42"


def _patched_rlm_factory(monkeypatch, captured_max_iter: list[int]):
    """Wrap DspyRLM so we record the max_iterations on each construction."""
    from dspy.predict import RLM as DspyRLM

    real_init = DspyRLM.__init__

    def spy_init(self, *args, max_iterations=10, **kwargs):
        captured_max_iter.append(max_iterations)
        return real_init(self, *args, max_iterations=max_iterations, **kwargs)

    monkeypatch.setattr(DspyRLM, "__init__", spy_init)


def test_halving_default_behavior(monkeypatch) -> None:
    """Default: max_iter halves on each rejection: 8 -> 4 -> 2."""
    captured: list[int] = []
    _patched_rlm_factory(monkeypatch, captured)

    lm = _CountingScriptedLM([
        "SUBMIT(answer=1)",
        "SUBMIT(answer=2)",
        "SUBMIT(answer=3)",
    ])
    rlm = RLM(
        signature="question -> answer: int",
        lm=lm,
        engine="v7-dspy",
        max_turns=8,
        output_validator=_validator_must_be_42,
    )
    rlm(question="x")

    # 3 attempts (initial + 2 retries); halving each time.
    # 8 -> ceil(8/2)=4 -> ceil(4/2)=2
    assert captured == [8, 4, 2], f"expected halving 8->4->2, got {captured}"


def test_no_halving_when_disabled(monkeypatch) -> None:
    """With halve_max_iter_on_retry=False, max_iter stays constant across retries."""
    captured: list[int] = []
    _patched_rlm_factory(monkeypatch, captured)

    lm = _CountingScriptedLM([
        "SUBMIT(answer=1)",
        "SUBMIT(answer=2)",
        "SUBMIT(answer=3)",
    ])
    rlm = RLM(
        signature="question -> answer: int",
        lm=lm,
        engine="v7-dspy",
        max_turns=8,
        output_validator=_validator_must_be_42,
        halve_max_iter_on_retry=False,
    )
    rlm(question="x")

    assert captured == [8, 8, 8], f"expected constant 8 with halving off, got {captured}"


def test_no_halving_recovers_when_default_would_starve(monkeypatch) -> None:
    """End-to-end: with halving off, retries get full budget and can succeed.

    This mirrors the MFMC_hard_2 scenario where halving (10->5->2->1) starved
    the model. With halving off, the second attempt has the same compute
    budget as the first.
    """
    captured: list[int] = []
    _patched_rlm_factory(monkeypatch, captured)

    # First SUBMIT rejected, second accepted.
    lm = _CountingScriptedLM(["SUBMIT(answer=7)", "SUBMIT(answer=42)"])
    rlm = RLM(
        signature="question -> answer: int",
        lm=lm,
        engine="v7-dspy",
        max_turns=10,
        output_validator=_validator_must_be_42,
        halve_max_iter_on_retry=False,
    )
    result = rlm(question="x")

    assert result.submitted is True, f"expected success; reason={result.failure_reason}"
    assert result.payload["answer"] == 42
    assert captured == [10, 10], f"expected [10, 10] with halving off, got {captured}"
