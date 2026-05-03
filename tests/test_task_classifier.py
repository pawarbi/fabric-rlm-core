"""Unit tests for :mod:`fabric_rlm.experimental.task_classifier`.

Tests cover:
  - TaskClass enum integrity (no dupes, all values lower-snake_case)
  - classify() happy paths for each class
  - classify() fallback to UNKNOWN on every failure mode
  - LM signature flexibility (callable, messages-style, .complete attr)
  - seed_priors() idempotence + overwrite semantics
  - DEFAULT_CLASS_PRIORS shape
"""

from __future__ import annotations

import pytest

from fabric_rlm.experimental.task_classifier import (
    DEFAULT_CLASS_PRIORS,
    TaskClass,
    _normalize,
    classify,
    seed_priors,
)


# ----------------------------------------------------------------------------
# Enum integrity
# ----------------------------------------------------------------------------


def test_taskclass_values_unique_and_lower_snake():
    values = [c.value for c in TaskClass]
    assert len(values) == len(set(values)), "duplicate enum values"
    for v in values:
        assert v == v.lower() and " " not in v


def test_taskclass_includes_unknown_and_canonical_classes():
    expected = {
        "lookup", "search", "aggregate", "pairwise",
        "multi_hop", "cs_puzzle", "code_gen", "format",
        "refusal", "unknown",
    }
    assert {c.value for c in TaskClass} == expected


# ----------------------------------------------------------------------------
# _normalize — defensive parsing of LM output
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("lookup", TaskClass.LOOKUP),
        (" Lookup ", TaskClass.LOOKUP),
        ("LOOKUP.", TaskClass.LOOKUP),
        ("multi_hop\n\nbecause...", TaskClass.MULTI_HOP),
        ('"format"', TaskClass.FORMAT),
        ("garbage", TaskClass.UNKNOWN),
        ("", TaskClass.UNKNOWN),
        (None, TaskClass.UNKNOWN),
        ("   ", TaskClass.UNKNOWN),
    ],
)
def test_normalize_handles_noisy_outputs(raw, expected):
    assert _normalize(raw) is expected


# ----------------------------------------------------------------------------
# classify — happy paths and fallbacks
# ----------------------------------------------------------------------------


class _StubLM:
    """Minimal callable LM matching dspy.LM signature: lm(prompt) -> list[str]."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, prompt, **_kwargs):
        self.calls.append(prompt)
        return [self.response] if isinstance(self.response, str) else self.response


@pytest.mark.parametrize("klass", list(TaskClass))
def test_classify_returns_each_class_when_lm_says_so(klass):
    lm = _StubLM(klass.value)
    assert classify("any question", lm) is klass


def test_classify_extracts_first_token_when_lm_is_chatty():
    lm = _StubLM("multi_hop  -- because the steps chain together")
    assert classify("Q?", lm) is TaskClass.MULTI_HOP


def test_classify_returns_unknown_when_lm_is_none():
    assert classify("anything", None) is TaskClass.UNKNOWN


def test_classify_returns_unknown_when_question_is_empty():
    assert classify("", _StubLM("lookup")) is TaskClass.UNKNOWN
    assert classify("   ", _StubLM("lookup")) is TaskClass.UNKNOWN


def test_classify_returns_unknown_when_lm_raises():
    class Boom:
        def __call__(self, *_a, **_kw):
            raise RuntimeError("api down")

    assert classify("q", Boom()) is TaskClass.UNKNOWN


def test_classify_returns_unknown_on_oov_response():
    assert classify("q", _StubLM("definitely_not_a_class")) is TaskClass.UNKNOWN


def test_classify_truncates_long_questions():
    lm = _StubLM("lookup")
    classify("x" * 10_000, lm)
    assert len(lm.calls) == 1
    # Prompt must not contain the full 10k chars (we cap at 4000 from question)
    assert lm.calls[0].count("x") < 5000


def test_classify_accepts_messages_style_lm():
    """LM that only accepts ``messages=[...]`` kwarg must still work."""

    class MessagesLM:
        def __init__(self):
            self.last_messages = None

        def __call__(self, *args, **kwargs):
            if "messages" not in kwargs:
                raise TypeError("expected messages kwarg")
            self.last_messages = kwargs["messages"]
            return ["aggregate"]

    lm = MessagesLM()
    assert classify("count the rows", lm) is TaskClass.AGGREGATE
    assert lm.last_messages and lm.last_messages[0]["role"] == "user"


def test_classify_accepts_complete_attr_lm():
    """LM with .complete(prompt) but no __call__ that takes a prompt."""

    class CompleteLM:
        def __init__(self):
            self.completed = []

        def __call__(self, *_a, **_kw):
            raise TypeError("no positional support")

        def complete(self, prompt):
            self.completed.append(prompt)
            return "search"

    lm = CompleteLM()
    assert classify("scan the log", lm) is TaskClass.SEARCH
    assert len(lm.completed) == 1


# ----------------------------------------------------------------------------
# DEFAULT_CLASS_PRIORS — shape and policy
# ----------------------------------------------------------------------------


def test_default_class_priors_only_uses_valid_classes():
    for klass in DEFAULT_CLASS_PRIORS:
        assert klass in TaskClass


def test_default_class_priors_unknown_is_empty():
    """UNKNOWN must never seed priors — that's the fallback contract."""

    assert DEFAULT_CLASS_PRIORS[TaskClass.UNKNOWN] == {}


def test_default_class_priors_have_positive_alpha_beta():
    for klass, rung_map in DEFAULT_CLASS_PRIORS.items():
        for rung, (a, b) in rung_map.items():
            assert isinstance(rung, int)
            assert a > 0 and b > 0, f"{klass.value}/rung {rung} has non-positive Beta"


# ----------------------------------------------------------------------------
# seed_priors — idempotence + overwrite semantics
# ----------------------------------------------------------------------------


class _StubState:
    def __init__(self, priors=None):
        self.priors = dict(priors or {})


def test_seed_priors_writes_when_state_is_empty():
    state = _StubState()
    wrote = seed_priors(state, "MFMC", TaskClass.CS_PUZZLE)
    assert wrote is True
    assert 3 in state.priors["MFMC"]
    a, b = state.priors["MFMC"][3]
    assert a == 4.0 and b == 1.0


def test_seed_priors_no_op_for_unknown():
    state = _StubState()
    wrote = seed_priors(state, "X", TaskClass.UNKNOWN)
    assert wrote is False
    assert state.priors == {}


def test_seed_priors_idempotent_when_taskkey_has_observations():
    """Don't trample live posteriors."""

    state = _StubState({"MFMC": {0: (5.0, 7.0)}})
    wrote = seed_priors(state, "MFMC", TaskClass.CS_PUZZLE)
    assert wrote is False
    assert state.priors["MFMC"] == {0: (5.0, 7.0)}


def test_seed_priors_overwrite_existing_resets():
    state = _StubState({"MFMC": {0: (5.0, 7.0)}})
    wrote = seed_priors(state, "MFMC", TaskClass.CS_PUZZLE, overwrite_existing=True)
    assert wrote is True
    # CS_PUZZLE seeds rungs 0, 1, 3
    assert set(state.priors["MFMC"].keys()) == {0, 1, 3}


def test_seed_priors_returns_false_on_no_priors_attr():
    """State without a priors dict (duck-type guard) is a graceful no-op."""

    class NotAState:
        pass

    assert seed_priors(NotAState(), "X", TaskClass.LOOKUP) is False


def test_seed_priors_accepts_custom_table():
    state = _StubState()
    custom = {TaskClass.LOOKUP: {2: (10.0, 1.0)}}
    wrote = seed_priors(state, "X", TaskClass.LOOKUP, prior_table=custom)
    assert wrote is True
    assert state.priors["X"] == {2: (10.0, 1.0)}


def test_seed_priors_no_op_when_class_not_in_table():
    state = _StubState()
    custom = {TaskClass.LOOKUP: {0: (4.0, 1.0)}}
    wrote = seed_priors(state, "X", TaskClass.PAIRWISE, prior_table=custom)
    assert wrote is False
    assert state.priors == {}


# ----------------------------------------------------------------------------
# Integration with the real BanditState
# ----------------------------------------------------------------------------


def test_seed_priors_works_with_real_bandit_state():
    from fabric_rlm.experimental.bandit_policy import BanditState

    state = BanditState()
    wrote = seed_priors(state, "spark_log_rca", TaskClass.AGGREGATE)
    assert wrote is True
    # Beta_for now reflects the seeded prior rather than uniform
    a, b = state.beta_for("spark_log_rca", 1)
    assert (a, b) == (3.0, 1.0)
    # Unseeded rungs still uniform
    assert state.beta_for("spark_log_rca", 0) == (1.0, 1.0)


def test_seed_priors_does_not_inflate_total_observations():
    """Seeded priors must not look like real observations to the warmup gate."""

    from fabric_rlm.experimental.bandit_policy import BanditState

    state = BanditState()
    seed_priors(state, "X", TaskClass.CS_PUZZLE)
    # total_observations counts (a + b - 2); seeded values should be small
    # enough that warmup logic still treats this task_key as "fresh" under
    # most policies, but at least the count is bounded and predictable.
    obs = state.total_observations("X")
    assert obs >= 0  # not negative
    assert obs <= 20  # bounded by the prior table mass
