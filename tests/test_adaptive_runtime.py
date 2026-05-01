"""Integration tests for ``engine='adaptive'`` on :class:`RLM`.

These are construction- and routing-level tests; full LM-backed escalation is
exercised in the bench harness, not unit tests.
"""

from __future__ import annotations

import warnings

import pytest

from fabric_rlm import RLM
from fabric_rlm.experimental.adaptive_policy import (
    Budget,
    LadderPolicy,
    ValidationVerdict,
)

pytestmark = pytest.mark.experimental


def test_adaptive_construction_does_not_resolve_lm() -> None:
    # No env var, no API key, no skill loading — must still construct.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rlm = RLM(
            signature="question -> answer",
            lm="gpt-4.1-mini",
            engine="adaptive",
        )
    assert rlm.engine == "adaptive"
    assert rlm.outer_lm is None
    assert rlm.adaptive_config == {}


def test_adaptive_emits_experimental_warning() -> None:
    with pytest.warns(UserWarning, match="experimental"):
        RLM(
            signature="q -> a",
            lm="gpt-4.1-mini",
            engine="adaptive",
        )


def test_inner_engine_adaptive_rejected() -> None:
    with pytest.raises(ValueError, match="inner_engine"):
        RLM(
            signature="q -> a",
            lm="gpt-4.1-mini",
            engine="adaptive",
            inner_engine="adaptive",
        )


def test_invalid_engine_rejected() -> None:
    with pytest.raises(ValueError, match="engine must be"):
        RLM(signature="q -> a", lm="gpt-4.1-mini", engine="bogus")


def test_adaptive_routes_run_through_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``AdaptiveRunner`` to confirm ``RLM.run()`` routes through it."""
    from fabric_rlm.experimental import adaptive_runner as ar_mod

    seen_factories: list = []
    seen_inputs: list = []

    class FakeAdaptiveResult:
        def __init__(self):
            class _R:
                class trajectory:
                    metadata: dict = {}
                payload = {"answer": "stub"}
                submitted = True
                failure_reason = None
            self.result = _R()
            self.passed = True
            self.attempts = []
            self.winner = None
            self.stop_reason = "ok"

    class FakeRunner:
        def __init__(self, *, rlm_factory, **kw):
            seen_factories.append(rlm_factory)

        def run(self, inputs, **kw):
            seen_inputs.append(inputs)
            return FakeAdaptiveResult()

    monkeypatch.setattr(ar_mod, "AdaptiveRunner", FakeRunner)
    # Also patch the symbol where _run_adaptive imports from (relative import)
    from fabric_rlm import runtime as rt_mod
    # _run_adaptive imports inside the method, so the monkeypatch above
    # on ar_mod.AdaptiveRunner takes effect at lookup time.

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rlm = RLM(signature="q -> a", lm="gpt-4.1-mini", engine="adaptive")
    out = rlm.run({"q": "hello"})
    assert seen_inputs == [{"q": "hello"}]
    assert out.payload == {"answer": "stub"}


def test_adaptive_failed_result_converted_to_rlmresult(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: when every attempt raises, the runner returns a sentinel
    ``_FailedResult``. ``RLM(engine='adaptive').run()`` must convert it to a
    real :class:`RLMResult` so callers can use ``.payload`` / ``.trajectory``
    without ``AttributeError``."""
    from fabric_rlm.experimental import adaptive_runner as ar_mod
    from fabric_rlm.experimental.adaptive_runner import _FailedResult
    from fabric_rlm.experimental.adaptive_policy import (
        AttemptConfig,
        AttemptRecord,
        ValidationVerdict,
    )
    from fabric_rlm.runtime import RLMResult

    failed = _FailedResult(reason="boom")
    cfg = AttemptConfig(rung=4, max_turns=4, parallel_rollouts=1)
    rec = AttemptRecord(
        rung=4,
        rollout_index=0,
        config=cfg,
        result=failed,
        verdict=ValidationVerdict(passed=False, feedback="boom"),
        elapsed_seconds=0.1,
        turns_used=0,
    )

    class FakeAdaptiveResult:
        result = failed
        attempts = [rec]
        winner = rec
        passed = False
        stop_reason = "exhausted: every attempt failed"
        elapsed_seconds = 1.5

    class FakeRunner:
        def __init__(self, **_kw):
            pass

        def run(self, inputs, **_kw):
            return FakeAdaptiveResult()

    monkeypatch.setattr(ar_mod, "AdaptiveRunner", FakeRunner)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rlm = RLM(
            signature="q -> a",
            lm="gpt-4.1-mini",
            engine="adaptive",
            adaptive={"validator": lambda r: True},
        )
    out = rlm.run({"q": "hi"})

    assert isinstance(out, RLMResult), f"expected RLMResult, got {type(out).__name__}"
    assert out.submitted is False
    assert out.payload is None
    assert out.failure_reason == "boom"
    assert out.trajectory is not None
    adaptive_meta = out.trajectory.metadata.get("adaptive", {})
    assert adaptive_meta.get("all_attempts_failed") is True
    assert adaptive_meta.get("winner_rung") == 4
    assert adaptive_meta.get("stop_reason") == "exhausted: every attempt failed"


def test_failed_result_carries_trajectory() -> None:
    """``_FailedResult`` should always have a usable ``trajectory`` so
    ``_make_result`` can attach the adaptive metadata block, and downstream
    callers can read ``result.trajectory.metadata`` without ``AttributeError``.
    """
    from fabric_rlm.experimental.adaptive_runner import _FailedResult

    fr = _FailedResult(reason="oops")
    assert fr.payload is None
    assert fr.failure_reason == "oops"
    assert fr.submitted is False
    assert fr.trajectory is not None
    assert fr.trajectory.metadata["failed"] is True


def test_adaptive_warns_on_missing_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    from fabric_rlm.experimental import adaptive_runner as ar_mod

    class FakeRunner:
        def __init__(self, **_kw):
            pass

        def run(self, inputs, **kw):
            class _AR:
                class result:
                    class trajectory:
                        metadata: dict = {}
                    payload = {}
                    submitted = True
                    failure_reason = None
                passed = True
                attempts = []
                winner = None
                stop_reason = "ok"
            return _AR()

    monkeypatch.setattr(ar_mod, "AdaptiveRunner", FakeRunner)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rlm = RLM(signature="q -> a", lm="gpt-4.1-mini", engine="adaptive")
        rlm.run({"q": "hi"})
    msgs = [str(rec.message) for rec in w]
    assert any("validator" in m for m in msgs), msgs


def test_adaptive_no_warning_when_validator_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fabric_rlm.experimental import adaptive_runner as ar_mod

    class FakeRunner:
        def __init__(self, **_kw):
            pass

        def run(self, inputs, **kw):
            class _AR:
                class result:
                    class trajectory:
                        metadata: dict = {}
                    payload = {}
                    submitted = True
                    failure_reason = None
                passed = True
                attempts = []
                winner = None
                stop_reason = "ok"
            return _AR()

    monkeypatch.setattr(ar_mod, "AdaptiveRunner", FakeRunner)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rlm = RLM(
            signature="q -> a",
            lm="gpt-4.1-mini",
            engine="adaptive",
            adaptive={"validator": lambda r: True},
        )
        rlm.run({"q": "hi"})
    msgs = [str(rec.message) for rec in w]
    # Construction warning is fine; missing-validator warning must NOT appear.
    assert not any("validator" in m for m in msgs), msgs
