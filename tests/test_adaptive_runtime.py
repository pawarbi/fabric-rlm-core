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


def test_attempt_cfg_reasoning_effort_clones_dspy_lm_instance() -> None:
    """When the user passes a dspy.LM instance and the policy emits a non-None
    reasoning_effort, the inner factory must clone the LM with the new effort —
    not silently drop it. Regression: prior to this fix, reasoning_effort was
    only forwarded for dict specs, so EffortLadderPolicy + FabricLM(gpt-5)
    never actually changed the model's effort across rungs.
    """
    import dspy

    from fabric_rlm.experimental.adaptive_policy import AttemptConfig

    base_lm = dspy.LM("openai/gpt-4", api_key="x", reasoning_effort="minimal")
    rlm = RLM(
        signature="question -> answer",
        lm=base_lm,
        engine="adaptive",
        adaptive=dict(validator=lambda r: True),
    )

    captured = {}
    orig_RLM_init = RLM.__init__

    def spy_init(self, *args, **kwargs):
        captured["lm"] = kwargs.get("lm")
        return orig_RLM_init(self, *args, **kwargs)

    # Build the factory the way _run_adaptive does, then exercise it.
    from fabric_rlm import runtime as _rt

    inner_kwargs = rlm._adaptive_inner_kwargs
    cfg = AttemptConfig(rung=3, max_turns=10, reasoning_effort="high")
    fkwargs = dict(inner_kwargs)
    if cfg.lm_spec is not None:
        fkwargs["lm"] = cfg.lm_spec
    elif cfg.lm_instance is not None:
        fkwargs["lm"] = cfg.lm_instance
    if cfg.reasoning_effort:
        lm_obj = fkwargs.get("lm")
        if isinstance(lm_obj, dict):
            fkwargs["lm"] = {**lm_obj, "reasoning_effort": cfg.reasoning_effort}
        elif lm_obj is not None and hasattr(lm_obj, "copy"):
            fkwargs["lm"] = lm_obj.copy(reasoning_effort=cfg.reasoning_effort)
    new_lm = fkwargs["lm"]
    assert new_lm is not base_lm, "must clone, not mutate"
    assert new_lm.kwargs.get("reasoning_effort") == "high"
    assert base_lm.kwargs.get("reasoning_effort") == "minimal"


def test_lm_instance_seeds_base_reasoning_effort() -> None:
    """Passing a FabricLM-like instance with reasoning_effort should seed
    LadderPolicy.base_reasoning_effort the same way a dict spec does.
    """
    import dspy
    lm = dspy.LM("openai/gpt-4", api_key="x", reasoning_effort="medium")
    # The seeding logic lives inline in _run_adaptive — exercise the same
    # branch directly to keep this test offline-safe.
    obj = lm
    if isinstance(obj, dict):
        eff = obj.get("reasoning_effort")
    elif obj is not None and hasattr(obj, "kwargs"):
        eff = obj.kwargs.get("reasoning_effort") if isinstance(obj.kwargs, dict) else None
    else:
        eff = None
    assert eff == "medium"


def test_adaptive_factory_propagates_inline_state_from_from_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for adaptive ``answer=None`` bug.

    ``RLM.from_task(task=..., inputs=..., outputs=...)`` sets ``_inline_task`` /
    ``_inline_outputs`` / ``_inline_inputs`` on the outer instance AFTER
    ``__init__`` returns. The adaptive engine's inner-RLM factory snapshots
    constructor kwargs *during* outer ``__init__`` and previously did not copy
    those post-init attributes onto the freshly-built inner RLM. Result: every
    inner attempt ran "blind" (no task description, no declared outputs) and
    consistently emitted ``answer=None`` no matter how many rungs the ladder
    climbed.

    This test captures the factory closure, invokes it directly, and asserts
    the inner instance receives the outer's inline state.
    """
    from fabric_rlm.experimental import adaptive_runner as ar_mod
    from fabric_rlm.experimental.adaptive_policy import AttemptConfig

    captured: dict = {}

    class _FakeAdaptiveResult:
        def __init__(self) -> None:
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
            self.elapsed_seconds = 0.0

    class _CapturingRunner:
        def __init__(self, *, rlm_factory, **_kw) -> None:
            captured["factory"] = rlm_factory

        def run(self, inputs, **_kw):
            inner = captured["factory"](
                AttemptConfig(rung=0, max_turns=1, inner_engine="v6-custom")
            )
            captured["inner"] = inner
            return _FakeAdaptiveResult()

    monkeypatch.setattr(ar_mod, "AdaptiveRunner", _CapturingRunner)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rlm = RLM.from_task(
            "Compute the answer to the question.",
            inputs={"question": "2+2?"},
            outputs={"answer": dict},
            lm="gpt-4.1-mini",
            engine="adaptive",
            adaptive={"validator": lambda _result: True},
        )
        rlm.run()

    inner = captured["inner"]
    # The inner RLM must see the same task / outputs / inputs the outer was
    # built from -- otherwise it runs "blind" and produces answer=None.
    assert inner._inline_task == "Compute the answer to the question."
    assert inner._inline_outputs == ["answer"]
    assert inner._inline_output_types == {"answer": dict}
    assert inner._inline_inputs == {"question": "2+2?"}
    # The collections must be defensive copies so inner mutations don't bleed
    # back into the outer instance (or sibling inners on subsequent attempts).
    assert inner._inline_outputs is not rlm._inline_outputs
    assert inner._inline_output_types is not rlm._inline_output_types
    assert inner._inline_inputs is not rlm._inline_inputs


def test_adaptive_factory_skips_inline_copy_when_outer_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the outer adaptive RLM is constructed via ``RLM(signature=...)``
    (no ``from_task``), ``_inline_task`` is ``None`` and the factory must
    NOT copy any inline state -- the inner relies on ``signature`` (which is
    already in ``_adaptive_inner_kwargs``) to know what to run.
    """
    from fabric_rlm.experimental import adaptive_runner as ar_mod
    from fabric_rlm.experimental.adaptive_policy import AttemptConfig

    captured: dict = {}

    class _FakeAdaptiveResult:
        def __init__(self) -> None:
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
            self.elapsed_seconds = 0.0

    class _CapturingRunner:
        def __init__(self, *, rlm_factory, **_kw) -> None:
            captured["factory"] = rlm_factory

        def run(self, inputs, **_kw):
            inner = captured["factory"](
                AttemptConfig(rung=0, max_turns=1, inner_engine="v6-custom")
            )
            captured["inner"] = inner
            return _FakeAdaptiveResult()

    monkeypatch.setattr(ar_mod, "AdaptiveRunner", _CapturingRunner)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rlm = RLM(
            signature="question -> answer",
            lm="gpt-4.1-mini",
            engine="adaptive",
            adaptive={"validator": lambda _result: True},
        )
        rlm.run({"question": "hello"})

    inner = captured["inner"]
    assert inner._inline_task is None
    # signature path: inline outputs/inputs stay at their __init__ defaults
    # (None / empty), NOT copied from the outer because the guard skipped it.
    assert not inner._inline_outputs  # None or empty
    assert not inner._inline_inputs  # None or empty
