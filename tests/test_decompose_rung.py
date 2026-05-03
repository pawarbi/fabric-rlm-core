"""Unit tests for :mod:`fabric_rlm.experimental.decompose_rung`.

The function is a total, error-swallowing combinator. We stub the LMs and
assert the contract on calls, parsing tolerance, error handling, and
parallelization.
"""

from __future__ import annotations

import threading
import time

import pytest

from fabric_rlm.experimental.decompose_rung import (
    DecomposeResult,
    EFFORT_LADDER_WITH_DECOMPOSE,
    _parse_sub_problems,
    decompose_then_synthesize,
    extended_effort_rung_cost,
)


# ----------------------------------------------------------------------------
# Stub LM
# ----------------------------------------------------------------------------


class _ScriptedLM:
    """Returns scripted responses in order; records every prompt it sees."""

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, prompt, **_kw):
        with self._lock:
            self.calls.append(prompt)
            if not self.responses:
                return "DEFAULT"
            return self.responses.pop(0)


# ----------------------------------------------------------------------------
# _parse_sub_problems
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1. one\n2. two\n3. three", ["one", "two", "three"]),
        ("1) one\n2) two", ["one", "two"]),
        ("- one\n- two\n- three", ["one", "two", "three"]),
        ("Sure! Here you go:\n1. alpha\n2. beta", ["alpha", "beta"]),
        ("nothing here", []),
        ("", []),
    ],
)
def test_parse_sub_problems(raw, expected):
    assert _parse_sub_problems(raw, max_subs=10) == expected


def test_parse_sub_problems_respects_max():
    raw = "\n".join(f"{i}. item {i}" for i in range(1, 11))
    assert len(_parse_sub_problems(raw, max_subs=3)) == 3


def test_parse_sub_problems_falls_back_on_plain_lines():
    raw = "Compute the sum of A.\nCompute the sum of B.\nCombine."
    out = _parse_sub_problems(raw, max_subs=5)
    assert len(out) == 3


# ----------------------------------------------------------------------------
# decompose_then_synthesize — happy path
# ----------------------------------------------------------------------------


def test_happy_path_two_subs():
    lm = _ScriptedLM("1. solve A\n2. solve B", "FINAL: 42")
    sub_lm = _ScriptedLM("answer A", "answer B")
    result = decompose_then_synthesize("Big question", lm, sub_lm, parallel=False)
    assert isinstance(result, DecomposeResult)
    assert result.rung_failure is False
    assert result.error is None
    assert result.sub_problems == ["solve A", "solve B"]
    assert result.sub_answers == ["answer A", "answer B"]
    assert result.final_answer == "FINAL: 42"
    assert result.llm_calls == 1 + 2 + 1
    assert len(lm.calls) == 2  # decompose + synthesize
    assert len(sub_lm.calls) == 2


def test_default_sub_lm_is_lm():
    lm = _ScriptedLM(
        "1. a\n2. b",
        "ans-a", "ans-b",  # sub_lm is lm, so phase B uses lm
        "FINAL",
    )
    result = decompose_then_synthesize("q", lm, sub_lm=None, parallel=False)
    assert result.final_answer == "FINAL"
    assert result.llm_calls == 4
    assert len(lm.calls) == 4


def test_synthesis_prompt_includes_subanswers():
    lm = _ScriptedLM("1. one\n2. two", "DONE")
    sub_lm = _ScriptedLM("first", "second")
    decompose_then_synthesize("Q", lm, sub_lm, parallel=False)
    synth_prompt = lm.calls[-1]
    assert "first" in synth_prompt and "second" in synth_prompt
    assert "Q" in synth_prompt


def test_max_subs_is_respected():
    lm = _ScriptedLM(
        "1. a\n2. b\n3. c\n4. d\n5. e\n6. f",
        "FINAL",
    )
    sub_lm = _ScriptedLM(*[f"ans{i}" for i in range(10)])
    result = decompose_then_synthesize("Q", lm, sub_lm, max_subs=3, parallel=False)
    assert len(result.sub_problems) == 3
    assert result.llm_calls == 1 + 3 + 1


# ----------------------------------------------------------------------------
# Degenerate decomposition
# ----------------------------------------------------------------------------


def test_degenerate_one_sub_yields_rung_failure():
    lm = _ScriptedLM("1. only one")
    result = decompose_then_synthesize("Q", lm, parallel=False)
    assert result.rung_failure is True
    assert "degenerate" in (result.error or "")
    assert result.final_answer == ""
    # No synthesis call was made
    assert len(lm.calls) == 1


def test_zero_sub_yields_rung_failure():
    lm = _ScriptedLM("I refuse")
    result = decompose_then_synthesize("Q", lm, parallel=False, min_subs=2)
    assert result.rung_failure is True
    assert result.sub_problems == []


# ----------------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------------


def test_decompose_error_does_not_raise():
    class Boom:
        def __call__(self, *_a, **_kw):
            raise RuntimeError("boom")

    result = decompose_then_synthesize("Q", Boom())
    assert result.rung_failure is True
    assert "boom" in (result.error or "")
    assert result.final_answer == ""


def test_synth_error_marks_rung_failure_but_keeps_partial_state():
    class FailOnSynth:
        def __init__(self):
            self.n = 0

        def __call__(self, prompt, **_kw):
            self.n += 1
            if self.n == 1:
                return "1. a\n2. b"
            raise RuntimeError("synth boom")

    lm = FailOnSynth()
    sub_lm = _ScriptedLM("ans-a", "ans-b")
    result = decompose_then_synthesize("Q", lm, sub_lm, parallel=False)
    assert result.rung_failure is True
    assert "synth" in (result.error or "")
    assert result.sub_problems == ["a", "b"]
    assert result.sub_answers == ["ans-a", "ans-b"]


def test_sub_solve_error_does_not_kill_run():
    """A single failing sub-solve must not abort the whole rung."""

    class HalfFail:
        def __init__(self):
            self.n = 0

        def __call__(self, prompt, **_kw):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("sub boom")
            return f"ans-{self.n}"

    lm = _ScriptedLM("1. a\n2. b\n3. c", "FINAL")
    result = decompose_then_synthesize("Q", lm, HalfFail(), parallel=False)
    assert result.rung_failure is False
    assert "<sub-solve error" in result.sub_answers[1]
    assert result.final_answer == "FINAL"


def test_empty_question_returns_failure_without_calls():
    lm = _ScriptedLM("never")
    result = decompose_then_synthesize("", lm)
    assert result.rung_failure is True
    assert lm.calls == []


def test_none_lm_returns_failure():
    result = decompose_then_synthesize("Q", None)
    assert result.rung_failure is True


def test_max_subs_lt_min_subs_returns_failure():
    result = decompose_then_synthesize("Q", _ScriptedLM("x"), max_subs=1, min_subs=2)
    assert result.rung_failure is True


# ----------------------------------------------------------------------------
# on_event hook
# ----------------------------------------------------------------------------


def test_on_event_called_at_each_phase():
    events: list[tuple[str, dict]] = []

    lm = _ScriptedLM("1. a\n2. b", "FINAL")
    sub_lm = _ScriptedLM("xa", "xb")
    decompose_then_synthesize(
        "Q", lm, sub_lm, parallel=False,
        on_event=lambda name, payload: events.append((name, payload)),
    )
    names = [n for n, _ in events]
    assert names == ["decompose_begin", "decompose_end", "solve_end", "synthesize_end"]


# ----------------------------------------------------------------------------
# Parallelism
# ----------------------------------------------------------------------------


def test_parallel_phase_b_runs_concurrently():
    """If parallel=True, total wall time should be roughly one slow call."""

    class SlowLM:
        def __init__(self, delay):
            self.delay = delay
            self.calls = 0
            self._lock = threading.Lock()

        def __call__(self, prompt, **_kw):
            with self._lock:
                self.calls += 1
                n = self.calls
            time.sleep(self.delay)
            return f"ans-{n}"

    decompose_lm = _ScriptedLM("1. a\n2. b\n3. c", "FINAL")
    sub_lm = SlowLM(0.10)
    t0 = time.perf_counter()
    res = decompose_then_synthesize("Q", decompose_lm, sub_lm, parallel=True)
    elapsed = time.perf_counter() - t0
    assert res.rung_failure is False
    # Three 100ms calls in parallel + tiny overhead — well under 300ms (sequential)
    assert elapsed < 0.28, f"phase B did not parallelize: {elapsed:.2f}s"


# ----------------------------------------------------------------------------
# extended_effort_rung_cost
# ----------------------------------------------------------------------------


def test_extended_cost_adds_one_rung_at_2x():
    base = {0: 1.0, 1: 3.0, 2: 8.0, 3: 25.0, 4: 75.0}
    out = extended_effort_rung_cost(base)
    assert set(out.keys()) == {0, 1, 2, 3, 4, 5}
    assert out[5] == 150.0
    # Base unchanged (we returned a copy)
    assert 5 not in base


def test_extended_cost_default_uses_real_table():
    out = extended_effort_rung_cost()
    assert 5 in out
    assert out[5] > out[4]


def test_default_ladder_has_six_entries():
    assert len(EFFORT_LADDER_WITH_DECOMPOSE) == 6


# ----------------------------------------------------------------------------
# Generalization smoke — non-CS prompts produce sensible structure
# ----------------------------------------------------------------------------


def test_works_on_generic_prompt():
    """The function does not bake in any CS-puzzle-specific assumptions."""

    lm = _ScriptedLM(
        "1. Identify the slowest stage.\n2. Quantify its time share.",
        "The bottleneck is the shuffle stage at 62% of total time.",
    )
    sub_lm = _ScriptedLM("Stage 3", "62%")
    result = decompose_then_synthesize(
        "Diagnose the bottleneck in this Spark job log: ...",
        lm, sub_lm, parallel=False,
    )
    assert result.rung_failure is False
    assert "shuffle" in result.final_answer.lower()
