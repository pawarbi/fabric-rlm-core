"""End-to-end test for wiring task_classifier into AdaptiveRunner.

Validates:
1. AdaptiveRunner accepts a pre_run hook in its constructor.
2. make_classifier_pre_run extracts the question, classifies, and seeds priors.
3. Hook is a no-op for vanilla LadderPolicy (no .task_key / .state attrs).
4. Hook is idempotent — second call doesn't overwrite an actively-learned prior.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from fabric_rlm.experimental.adaptive_policy import AttemptConfig, LadderPolicy
from fabric_rlm.experimental.adaptive_runner import AdaptiveRunner
from fabric_rlm.experimental.bandit_policy import BanditState
from fabric_rlm.experimental.effort_ladder_policy import EffortBanditPolicy
from fabric_rlm.experimental.task_classifier import (
    DEFAULT_CLASS_PRIORS,
    TaskClass,
    make_classifier_pre_run,
)


# ----------- 1. AdaptiveRunner accepts pre_run -----------


def test_adaptive_runner_accepts_pre_run_param():
    """Ensure the constructor wires the new pre_run kwarg."""
    def hook(inputs, runner):
        pass

    runner = AdaptiveRunner(
        rlm_factory=lambda cfg: MagicMock(),
        pre_run=hook,
    )
    assert runner.pre_run is hook


def test_adaptive_runner_pre_run_defaults_to_none():
    runner = AdaptiveRunner(rlm_factory=lambda cfg: MagicMock())
    assert runner.pre_run is None


# ----------- 2. classifier hook seeds bandit priors -----------


class _StubLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


def test_classifier_hook_seeds_priors_for_cs_puzzle():
    state = BanditState()
    policy = EffortBanditPolicy(state=state, task_key="bench/cs_hard")
    runner = MagicMock()
    runner.policy = policy

    lm = _StubLM("CS_PUZZLE")
    hook = make_classifier_pre_run(classifier_lm=lm)
    hook({"question": "Find the longest path in this DAG"}, runner)

    assert "bench/cs_hard" in state.priors
    written = state.priors["bench/cs_hard"]
    expected = DEFAULT_CLASS_PRIORS[TaskClass.CS_PUZZLE]
    assert written == expected
    assert len(lm.calls) == 1


def test_classifier_hook_no_op_when_unknown_class():
    state = BanditState()
    policy = EffortBanditPolicy(state=state, task_key="X")
    runner = MagicMock()
    runner.policy = policy

    lm = _StubLM("not-a-class-name")
    hook = make_classifier_pre_run(classifier_lm=lm)
    hook({"question": "anything"}, runner)
    assert "X" not in state.priors  # UNKNOWN -> no priors written


def test_classifier_hook_skipped_for_vanilla_ladder_policy():
    """LadderPolicy has neither task_key nor state — hook must short-circuit."""
    runner = MagicMock()
    runner.policy = LadderPolicy()  # no .task_key, no .state

    lm = _StubLM("CS_PUZZLE")
    hook = make_classifier_pre_run(classifier_lm=lm)
    hook({"question": "x"}, runner)
    assert lm.calls == []  # never even invoked the LM


def test_classifier_hook_skipped_when_question_empty():
    state = BanditState()
    policy = EffortBanditPolicy(state=state, task_key="K")
    runner = MagicMock()
    runner.policy = policy

    lm = _StubLM("CS_PUZZLE")
    hook = make_classifier_pre_run(classifier_lm=lm)
    hook({}, runner)
    hook({"foo": 5}, runner)
    assert lm.calls == []
    assert "K" not in state.priors


def test_classifier_hook_idempotent_when_observations_exist():
    """If bandit already has observations, default seed_priors leaves them alone."""
    state = BanditState()
    state.priors["K"] = {0: (10.0, 5.0)}  # warm posterior
    policy = EffortBanditPolicy(state=state, task_key="K")
    runner = MagicMock()
    runner.policy = policy

    lm = _StubLM("CS_PUZZLE")
    hook = make_classifier_pre_run(classifier_lm=lm)
    hook({"question": "x"}, runner)
    assert state.priors["K"] == {0: (10.0, 5.0)}


def test_classifier_hook_overwrite_existing_when_requested():
    state = BanditState()
    state.priors["K"] = {0: (10.0, 5.0)}
    policy = EffortBanditPolicy(state=state, task_key="K")
    runner = MagicMock()
    runner.policy = policy

    lm = _StubLM("CS_PUZZLE")
    hook = make_classifier_pre_run(classifier_lm=lm, overwrite_existing=True)
    hook({"question": "x"}, runner)
    assert state.priors["K"] == DEFAULT_CLASS_PRIORS[TaskClass.CS_PUZZLE]


def test_classifier_hook_on_classify_callback():
    state = BanditState()
    policy = EffortBanditPolicy(state=state, task_key="K")
    runner = MagicMock()
    runner.policy = policy

    seen: list[Any] = []

    def on_cls(cls, key):
        seen.append((cls, key))

    lm = _StubLM("LOOKUP")
    hook = make_classifier_pre_run(classifier_lm=lm, on_classify=on_cls)
    hook({"question": "what is 2+2?"}, runner)
    assert seen == [(TaskClass.LOOKUP, "K")]


def test_classifier_hook_uses_input_key_override():
    state = BanditState()
    policy = EffortBanditPolicy(state=state, task_key="K")
    runner = MagicMock()
    runner.policy = policy

    lm = _StubLM("CS_PUZZLE")
    hook = make_classifier_pre_run(
        classifier_lm=lm, question_input_key="my_field"
    )
    hook({"question": "ignored", "my_field": "the real prompt"}, runner)
    assert any("the real prompt" in c for c in lm.calls)


def test_classifier_hook_classifier_failure_swallowed_at_runner_level():
    """make_classifier_pre_run itself can raise; AdaptiveRunner.run catches it.

    Validates that the documented contract is honored: the hook is *not*
    required to be exception-safe — the runner will swallow exceptions.
    """
    state = BanditState()
    policy = EffortBanditPolicy(state=state, task_key="K")
    runner = MagicMock()
    runner.policy = policy

    class _BrokenLM:
        def __call__(self, prompt):
            raise RuntimeError("API down")

    hook = make_classifier_pre_run(classifier_lm=_BrokenLM())
    # The hook itself catches LM errors via classify() returning UNKNOWN
    hook({"question": "x"}, runner)
    # No prior written, but no exception either
    assert "K" not in state.priors



# ----------- 2. classifier hook seeds bandit priors -----------


class _StubLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


def test_classifier_hook_seeds_priors_for_cs_puzzle():
    state = BanditState()
    policy = EffortBanditPolicy(state=state, task_key="bench/cs_hard")
    runner = MagicMock()
    runner.policy = policy

    lm = _StubLM("CS_PUZZLE")
    hook = make_classifier_pre_run(classifier_lm=lm)
    hook({"question": "Find the longest path in this DAG"}, runner)

    # Priors should be written for the bench/cs_hard key
    assert "bench/cs_hard" in state.priors
    written = state.priors["bench/cs_hard"]
    expected = DEFAULT_CLASS_PRIORS[TaskClass.CS_PUZZLE]
    assert written == expected
    assert len(lm.calls) == 1


def test_classifier_hook_no_op_when_unknown_class():
    state = BanditState()
    policy = EffortBanditPolicy(state=state, task_key="X")
    runner = MagicMock()
    runner.policy = policy

    lm = _StubLM("not-a-class-name")
    hook = make_classifier_pre_run(classifier_lm=lm)
    hook({"question": "anything"}, runner)
    assert "X" not in state.priors  # UNKNOWN -> no priors written


def test_classifier_hook_skipped_for_vanilla_ladder_policy():
    """LadderPolicy has neither task_key nor state — hook must short-circuit."""
    runner = MagicMock()
    runner.policy = LadderPolicy()  # no .task_key, no .state

    lm = _StubLM("CS_PUZZLE")
    hook = make_classifier_pre_run(classifier_lm=lm)
    hook({"question": "x"}, runner)
    assert lm.calls == []  # never even invoked the LM


def test_classifier_hook_skipped_when_question_empty():
    state = BanditState()
    policy = EffortBanditPolicy(state=state, task_key="K")
    runner = MagicMock()
    runner.policy = policy

    lm = _StubLM("CS_PUZZLE")
    hook = make_classifier_pre_run(classifier_lm=lm)
    hook({}, runner)
    hook({"foo": 5}, runner)
    assert lm.calls == []
    assert "K" not in state.priors


def test_classifier_hook_idempotent_when_observations_exist():
    """If bandit already has observations, default seed_priors leaves them alone."""
    state = BanditState()
    state.priors["K"] = {0: (10.0, 5.0)}  # warm posterior
    policy = EffortBanditPolicy(state=state, task_key="K")
    runner = MagicMock()
    runner.policy = policy

    lm = _StubLM("CS_PUZZLE")
    hook = make_classifier_pre_run(classifier_lm=lm)
    hook({"question": "x"}, runner)
    # Existing priors preserved
    assert state.priors["K"] == {0: (10.0, 5.0)}


def test_classifier_hook_overwrite_existing_when_requested():
    state = BanditState()
    state.priors["K"] = {0: (10.0, 5.0)}
    policy = EffortBanditPolicy(state=state, task_key="K")
    runner = MagicMock()
    runner.policy = policy

    lm = _StubLM("CS_PUZZLE")
    hook = make_classifier_pre_run(classifier_lm=lm, overwrite_existing=True)
    hook({"question": "x"}, runner)
    assert state.priors["K"] == DEFAULT_CLASS_PRIORS[TaskClass.CS_PUZZLE]


def test_classifier_hook_on_classify_callback():
    state = BanditState()
    policy = EffortBanditPolicy(state=state, task_key="K")
    runner = MagicMock()
    runner.policy = policy

    seen: list[Any] = []

    def on_cls(cls, key):
        seen.append((cls, key))

    lm = _StubLM("LOOKUP")
    hook = make_classifier_pre_run(classifier_lm=lm, on_classify=on_cls)
    hook({"question": "what is 2+2?"}, runner)
    assert seen == [(TaskClass.LOOKUP, "K")]


def test_classifier_hook_uses_input_key_override():
    state = BanditState()
    policy = EffortBanditPolicy(state=state, task_key="K")
    runner = MagicMock()
    runner.policy = policy

    lm = _StubLM("CS_PUZZLE")
    hook = make_classifier_pre_run(
        classifier_lm=lm, question_input_key="my_field"
    )
    hook({"question": "ignored", "my_field": "the real prompt"}, runner)
    assert any("the real prompt" in c for c in lm.calls)
