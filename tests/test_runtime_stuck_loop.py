"""Stuck-loop early-termination tests (NEW-H).

Trace mining of v4 SSB runs surfaced trajectories where the LM emits the SAME
failing code with the SAME error type 3+ times in a row, then exhausts the
turn budget. This adds a configurable circuit-breaker:

* ``stuck_loop_threshold`` (default 3) — stop the run early when the last N
  *failed* turns share the same normalized code AND the same error type.
* ``failure_reason`` is set to ``"stuck_loop"`` so callers can distinguish
  loop-aborted runs from ``"max_turns"`` budget exhaustion.
* Submit/non-error turns reset the loop history.
* ``stuck_loop_threshold=None`` disables the check entirely.
* Threshold of 1 or 0 raises ValueError (nonsensical).

These tests exercise the public RLM API end-to-end with a scripted LM and
the real Interpreter, so the same NameError text the LM sees in the wild is
what the detector hashes.
"""

from __future__ import annotations

import pytest

from fabric_rlm import RLM
from fabric_rlm.runtime import _normalize_code_for_loop_detection, _loop_signature
from fabric_rlm.trajectory import TurnRecord


class ScriptedLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def __call__(self, *, messages):
        self.messages.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("No scripted responses left")
        return self.responses.pop(0)


def _broken(code: str) -> str:
    return f"```python\n{code}\n```"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_normalize_code_strips_trailing_whitespace_per_line() -> None:
    assert (
        _normalize_code_for_loop_detection("x = 1   \ny = 2  ")
        == _normalize_code_for_loop_detection("x = 1\ny = 2")
    )


def test_normalize_code_drops_blank_lines() -> None:
    assert (
        _normalize_code_for_loop_detection("x = 1\n\n\ny = 2")
        == _normalize_code_for_loop_detection("x = 1\ny = 2")
    )


def test_normalize_code_distinguishes_genuinely_different_code() -> None:
    assert (
        _normalize_code_for_loop_detection("x = 1")
        != _normalize_code_for_loop_detection("x = 2")
    )


def test_normalize_code_handles_empty_and_none() -> None:
    assert _normalize_code_for_loop_detection("") == ""
    assert _normalize_code_for_loop_detection(None) == ""  # type: ignore[arg-type]


def test_loop_signature_extracts_error_type_and_message() -> None:
    """Same error type AND same message → same signature."""
    t1 = TurnRecord(turn=1, code="foo()", stdout="", stderr="",
                    error="NameError: name 'foo' is not defined",
                    submitted=False, state={})
    t2 = TurnRecord(turn=2, code="foo()", stdout="", stderr="",
                    error="NameError: name 'foo' is not defined",
                    submitted=False, state={})
    assert _loop_signature(t1) == _loop_signature(t2)


def test_loop_signature_distinguishes_different_messages_same_type() -> None:
    """Same type but different message → DIFFERENT signature.

    Critical false-positive guard for stateful runs: if identical code yields
    a different error message between attempts, state has advanced — not stuck.
    """
    t1 = TurnRecord(turn=1, code="foo()", stdout="", stderr="",
                    error="NameError: name 'foo' is not defined",
                    submitted=False, state={})
    t2 = TurnRecord(turn=2, code="foo()", stdout="", stderr="",
                    error="NameError: name 'bar' is not defined",
                    submitted=False, state={})
    assert _loop_signature(t1) != _loop_signature(t2)


def test_loop_signature_distinguishes_different_error_types() -> None:
    t1 = TurnRecord(turn=1, code="foo()", stdout="", stderr="",
                    error="NameError: x", submitted=False, state={})
    t2 = TurnRecord(turn=2, code="foo()", stdout="", stderr="",
                    error="TypeError: x", submitted=False, state={})
    assert _loop_signature(t1) != _loop_signature(t2)


def test_loop_signature_returns_none_error_for_no_error() -> None:
    t = TurnRecord(turn=1, code="x = 1", stdout="ok", stderr="", error=None,
                   submitted=False, state={})
    sig = _loop_signature(t)
    assert sig[1] is None and sig[2] is None


def test_loop_signature_handles_multiline_traceback() -> None:
    """Real Python tracebacks have the type+msg on the LAST line."""
    err = (
        "Traceback (most recent call last):\n"
        '  File "<sandbox>", line 1, in <module>\n'
        "ValueError: something bad happened"
    )
    t = TurnRecord(turn=1, code="x()", stdout="", stderr="", error=err,
                   submitted=False, state={})
    sig = _loop_signature(t)
    assert sig[1] == "ValueError"
    assert sig[2] == "something bad happened"


# ---------------------------------------------------------------------------
# End-to-end via RLM.run()
# ---------------------------------------------------------------------------


def test_stuck_loop_terminates_after_threshold_identical_failures() -> None:
    """3 identical (code, error) → stop with failure_reason='stuck_loop'."""
    code = "broken_call()"
    lm = ScriptedLM([_broken(code)] * 6)  # extra responses in case it doesn't stop
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=10, timeout=5,
                        stuck_loop_threshold=3)
    result = rlm.run()

    assert not result.submitted
    assert result.failure_reason == "stuck_loop", (
        f"expected stuck_loop, got {result.failure_reason!r}"
    )
    assert len(result.trajectory) == 3, (
        f"loop should stop AFTER the 3rd identical failure, "
        f"got {len(result.trajectory)} turns"
    )


def test_stuck_loop_does_not_trigger_below_threshold() -> None:
    """2 identical failures with threshold=3 → keep going until success."""
    code = "broken_call()"
    lm = ScriptedLM(
        [_broken(code), _broken(code), _broken("SUBMIT(answer=1)")]
    )
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=10, timeout=5,
                        stuck_loop_threshold=3)
    result = rlm.run()

    assert result.submitted
    assert result.failure_reason is None
    assert len(result.trajectory) == 3


def test_stuck_loop_disabled_by_none() -> None:
    """stuck_loop_threshold=None → never trigger; run goes to max_turns."""
    code = "broken_call()"
    lm = ScriptedLM([_broken(code)] * 5)
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=5, timeout=5,
                        stuck_loop_threshold=None)
    result = rlm.run()

    assert not result.submitted
    assert result.failure_reason == "max_turns"
    assert len(result.trajectory) == 5


def test_stuck_loop_different_codes_does_not_trigger() -> None:
    """Same error type but different code each turn → keep going."""
    lm = ScriptedLM(
        [
            _broken("aaa()"),
            _broken("bbb()"),
            _broken("ccc()"),
            _broken("SUBMIT(answer=1)"),
        ]
    )
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=10, timeout=5,
                        stuck_loop_threshold=3)
    result = rlm.run()

    assert result.submitted
    assert len(result.trajectory) == 4


def test_stuck_loop_different_error_types_does_not_trigger() -> None:
    """Same code structure but different error each turn → keep going."""
    # Each line raises a different exception type when executed.
    lm = ScriptedLM(
        [
            _broken("undefined_thing()"),       # NameError
            _broken("(1).nonexistent()"),       # AttributeError
            _broken("int('not a number')"),     # ValueError
            _broken("SUBMIT(answer=1)"),
        ]
    )
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=10, timeout=5,
                        stuck_loop_threshold=3)
    result = rlm.run()

    assert result.submitted
    assert len(result.trajectory) == 4


def test_stuck_loop_normalization_treats_whitespace_variants_as_same() -> None:
    """Whitespace-only differences must count as the SAME code."""
    lm = ScriptedLM(
        [
            _broken("broken_call()"),
            _broken("broken_call()  "),       # trailing spaces
            _broken("broken_call()\n\n"),     # blank lines
            _broken("broken_call()"),         # extras in case
        ]
    )
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=10, timeout=5,
                        stuck_loop_threshold=3)
    result = rlm.run()

    assert not result.submitted
    assert result.failure_reason == "stuck_loop"
    assert len(result.trajectory) == 3


def test_stuck_loop_progress_print_resets_consecutive_chain() -> None:
    """A non-erroring non-submit turn breaks the consecutive failure chain."""
    code = "broken_call()"
    lm = ScriptedLM(
        [
            _broken(code),                      # fail 1
            _broken(code),                      # fail 2
            _broken("print('thinking')"),       # OK, not submit -> resets chain
            _broken(code),                      # fail again, but only 1 consecutive
            _broken("SUBMIT(answer=1)"),
        ]
    )
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=10, timeout=5,
                        stuck_loop_threshold=3)
    result = rlm.run()

    assert result.submitted, f"got failure_reason={result.failure_reason}"
    assert len(result.trajectory) == 5


def test_stuck_loop_threshold_two() -> None:
    """Aggressive threshold=2: stop after 2 identical failures."""
    code = "broken_call()"
    lm = ScriptedLM([_broken(code)] * 5)
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=10, timeout=5,
                        stuck_loop_threshold=2)
    result = rlm.run()

    assert not result.submitted
    assert result.failure_reason == "stuck_loop"
    assert len(result.trajectory) == 2


def test_stuck_loop_threshold_invalid_raises() -> None:
    """Threshold of 1 or 0 doesn't make sense — should raise ValueError."""
    lm = ScriptedLM([])
    with pytest.raises(ValueError, match="stuck_loop_threshold"):
        RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=5, timeout=5,
                      stuck_loop_threshold=1)
    with pytest.raises(ValueError, match="stuck_loop_threshold"):
        RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=5, timeout=5,
                      stuck_loop_threshold=0)


def test_stuck_loop_threshold_non_int_raises() -> None:
    """Non-int threshold (float, str, bool) → TypeError at construction."""
    lm = ScriptedLM([])
    for bad in (2.5, "3", True, False, [3]):
        with pytest.raises(TypeError, match="stuck_loop_threshold"):
            RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=5, timeout=5,
                          stuck_loop_threshold=bad)


def test_stuck_loop_stateful_progress_not_flagged() -> None:
    """Identical code whose error MESSAGE changes is NOT stuck (state advancing).

    This is the key false-positive guard rubber-duck flagged: a stateful run
    can re-execute the same cell, mutate persistent state, and keep raising
    the same exception TYPE with different MESSAGES — that's progress, not
    a loop, and must not be aborted.
    """
    # Each turn raises NameError but for a different undefined name —
    # the message text differs even though the code is the same.
    lm = ScriptedLM(
        [
            _broken("undefined_a"),
            _broken("undefined_b"),
            _broken("undefined_c"),
            _broken("SUBMIT(answer=1)"),
        ]
    )
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=10, timeout=5,
                        stuck_loop_threshold=3)
    result = rlm.run()
    assert result.submitted, (
        f"3 NameErrors with different messages must not trigger stuck_loop "
        f"(got failure_reason={result.failure_reason})"
    )
    assert len(result.trajectory) == 4


def test_stuck_loop_threshold_propagates_to_adaptive_inner_kwargs() -> None:
    """``engine='adaptive'`` must propagate stuck_loop_threshold to inner v6 RLMs.

    Without explicit propagation the adaptive factory builds inner RLMs with
    the default (3) regardless of what the outer RLM was constructed with —
    silently breaking ``stuck_loop_threshold=None`` (intent: disable) and
    ``stuck_loop_threshold=2`` (intent: aggressive).
    """
    lm = ScriptedLM([])
    rlm = RLM(
        signature=None,
        lm=lm,
        engine="adaptive",
        inner_engine="v6-custom",
        adaptive={},
        stuck_loop_threshold=None,
    )
    assert rlm.stuck_loop_threshold is None
    assert "stuck_loop_threshold" in rlm._adaptive_inner_kwargs
    assert rlm._adaptive_inner_kwargs["stuck_loop_threshold"] is None

    rlm2 = RLM(
        signature=None,
        lm=lm,
        engine="adaptive",
        inner_engine="v6-custom",
        adaptive={},
        stuck_loop_threshold=2,
    )
    assert rlm2._adaptive_inner_kwargs["stuck_loop_threshold"] == 2


def test_stuck_loop_default_is_active() -> None:
    """Default stuck_loop_threshold (3) triggers without explicit opt-in."""
    code = "broken_call()"
    lm = ScriptedLM([_broken(code)] * 6)
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=10, timeout=5)
    # No stuck_loop_threshold passed -> default is active.
    result = rlm.run()
    assert result.failure_reason == "stuck_loop"
    assert len(result.trajectory) == 3


def test_stuck_loop_preserves_aggregated_metrics() -> None:
    """Early termination should still produce aggregated trajectory metrics."""
    code = "broken_call()"
    lm = ScriptedLM([_broken(code)] * 6)
    rlm = RLM.from_task("x", outputs=["answer"], lm=lm, max_turns=10, timeout=5,
                        stuck_loop_threshold=3)
    result = rlm.run()

    assert result.failure_reason == "stuck_loop"
    # Aggregated metrics should still be present (e.g., total_worker_seconds).
    assert hasattr(result, "total_worker_seconds")
    assert result.total_worker_seconds is not None
    assert len(result.trajectory) == 3
