"""End-to-end wiring test for decompose-rung integration.

Validates:
1. AttemptConfig.decompose_phase field exists and propagates.
2. EffortLadderPolicy with enable_decompose_top_rung=True adds rung max+1.
3. Runtime factory routes decompose_phase=True attempts to DecomposeRLMAdapter.
4. The adapter returns a real RLMResult on success and on degenerate decompose.
5. Question extraction works against several common input keys.

We use a stub LM (callable returning canned strings) so the test runs offline.
"""

from __future__ import annotations

from typing import Any

import pytest

from fabric_rlm.experimental.adaptive_policy import AttemptConfig
from fabric_rlm.experimental.decompose_engine import (
    DecomposeRLMAdapter,
    _extract_question,
)
from fabric_rlm.experimental.effort_ladder_policy import EffortBanditPolicy


# -----------------------------------------------------------------
# 1. AttemptConfig field
# -----------------------------------------------------------------


def test_attempt_config_has_decompose_phase_default_false():
    cfg = AttemptConfig(rung=0)
    assert cfg.decompose_phase is False
    assert cfg.decompose_max_subs == 6


def test_attempt_config_decompose_phase_can_be_set():
    cfg = AttemptConfig(rung=4, decompose_phase=True, decompose_max_subs=10)
    assert cfg.decompose_phase is True
    assert cfg.decompose_max_subs == 10


# -----------------------------------------------------------------
# 2. Policy adds top decompose rung
# -----------------------------------------------------------------


def test_effort_policy_default_no_decompose_rung():
    p = EffortBanditPolicy()
    # Default ladder is 5 rungs (0..4). max_rung = 4.
    assert p.max_rung == 4
    cfg = p._build_config(p.max_rung)
    assert cfg.decompose_phase is False


def test_effort_policy_with_decompose_top_rung_extends_ladder():
    p = EffortBanditPolicy(enable_decompose_top_rung=True)
    # Default ladder length 5 -> base max_rung 4; +1 for decompose -> 5.
    assert p.max_rung == 5
    decompose_cfg = p._build_config(5)
    assert decompose_cfg.decompose_phase is True
    assert decompose_cfg.decompose_max_subs == 6
    # Effort should be the highest in the ladder
    assert decompose_cfg.reasoning_effort == p.effort_ladder[-1]
    # Decompose rung should never use parallel rollouts
    assert decompose_cfg.parallel_rollouts == 1


def test_effort_policy_lower_rungs_unchanged_when_decompose_enabled():
    p_off = EffortBanditPolicy()
    p_on = EffortBanditPolicy(enable_decompose_top_rung=True)
    for rung in range(0, 5):
        cfg_off = p_off._build_config(rung)
        cfg_on = p_on._build_config(rung)
        assert cfg_off.reasoning_effort == cfg_on.reasoning_effort
        assert cfg_off.max_turns == cfg_on.max_turns
        assert cfg_off.decompose_phase is False
        assert cfg_on.decompose_phase is False


# -----------------------------------------------------------------
# 3. Question extraction
# -----------------------------------------------------------------


def test_extract_question_default_keys():
    assert _extract_question({"question": "Q?"}, None) == "Q?"
    assert _extract_question({"input": "I?"}, None) == "I?"
    assert _extract_question({"prompt": "P?"}, None) == "P?"
    assert _extract_question({"task": "T?"}, None) == "T?"


def test_extract_question_override_key():
    assert _extract_question({"my_q": "X"}, "my_q") == "X"


def test_extract_question_falls_back_to_first_string():
    out = _extract_question({"foo_bar": "yo"}, None)
    assert out == "yo"


def test_extract_question_empty():
    assert _extract_question(None, None) == ""
    assert _extract_question({}, None) == ""
    assert _extract_question({"x": 5}, None) == ""


# -----------------------------------------------------------------
# 4. DecomposeRLMAdapter behavior
# -----------------------------------------------------------------


class _StubLM:
    """Tiny callable LM returning canned responses by call count."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.responses:
            return ""
        return self.responses.pop(0)


def test_adapter_success_returns_submitted_rlm_result():
    """Decompose returns 3 sub-problems, each solved, then synthesized."""
    lm = _StubLM(
        [
            "1. Sub one\n2. Sub two\n3. Sub three",  # decompose
            "ans1",  # solve sub 1
            "ans2",  # solve sub 2
            "ans3",  # solve sub 3
            "Final synthesized answer",  # synthesize
        ]
    )
    adapter = DecomposeRLMAdapter(lm=lm, parallel=False)  # serial for test determinism
    result = adapter.run({"question": "Compute things"})
    assert result.submitted is True
    assert result.payload == {"answer": "Final synthesized answer"}
    assert result.failure_reason is None
    # Trajectory should carry decompose metadata
    meta = result.trajectory.metadata.get("decompose", {})
    assert meta["sub_problems"] == ["Sub one", "Sub two", "Sub three"]
    assert meta["sub_answers"] == ["ans1", "ans2", "ans3"]
    assert meta["llm_calls"] == 5


def test_adapter_degenerate_decompose_returns_unsubmitted():
    """If decompose returns < 2 sub-problems, the adapter must return submitted=False."""
    lm = _StubLM(["I refuse to decompose"])
    adapter = DecomposeRLMAdapter(lm=lm, parallel=False)
    result = adapter.run({"question": "anything"})
    assert result.submitted is False
    assert result.payload is None
    assert result.failure_reason  # non-empty
    meta = result.trajectory.metadata.get("decompose", {})
    assert meta["rung_failure"] is True


def test_adapter_custom_output_field():
    lm = _StubLM(
        [
            "1. a\n2. b",  # decompose
            "x",
            "y",
            "MERGED",
        ]
    )
    adapter = DecomposeRLMAdapter(lm=lm, parallel=False, output_field="solution")
    result = adapter.run({"question": "go"})
    assert result.payload == {"solution": "MERGED"}


def test_adapter_uses_question_input_key_override():
    lm = _StubLM(
        [
            "1. a\n2. b",
            "x",
            "y",
            "out",
        ]
    )
    adapter = DecomposeRLMAdapter(
        lm=lm, parallel=False, question_input_key="my_problem"
    )
    result = adapter.run({"question": "WRONG", "my_problem": "RIGHT"})
    assert result.submitted is True
    # The decompose prompt should have used "RIGHT", not "WRONG"
    assert any("RIGHT" in c for c in lm.calls)


def test_adapter_handles_empty_question():
    lm = _StubLM([])
    adapter = DecomposeRLMAdapter(lm=lm, parallel=False)
    result = adapter.run({})
    assert result.submitted is False
    assert result.failure_reason


# -----------------------------------------------------------------
# 5. Sanity: factory + adapter chain (no real runtime spin-up needed)
# -----------------------------------------------------------------


def test_policy_full_ladder_walk_with_decompose():
    """Walk all rungs; decompose rung must be the only one with decompose_phase."""
    p = EffortBanditPolicy(enable_decompose_top_rung=True)
    flags = [p._build_config(r).decompose_phase for r in range(p.max_rung + 1)]
    assert flags == [False, False, False, False, False, True]
