"""Tests that verifier-rejection history is threaded into the reflection prompt."""

from __future__ import annotations

from fabric_rlm import RLM
from fabric_rlm.prompts import REFLECTION_HISTORY_HEADER

from test_runtime_verifier import (
    FakeInterpreter,
    ScriptedLM,
    _install_fake_interpreter,
    _make_skill,
    _ran,
    _stub_skill_loader,
    _submit,
    _verifier_assert,
    _verifier_pass,
)


def _reflection_prompt_text(lm: ScriptedLM) -> str:
    """Return the user-message prompt sent into the reflection LM call.

    The reflection turn is the call where the most-recent user message starts
    with the reflection scaffold ("You are about to submit...").
    """
    for messages in lm.messages:
        last_user = messages[-1]
        if last_user["role"] == "user" and "ATTACK your own answer" in last_user["content"]:
            return last_user["content"]
    raise AssertionError("No reflection-turn LM call captured")


def test_reflection_sees_verifier_history(monkeypatch) -> None:
    """A verifier rejection followed by a passing SUBMIT threads the assertion into reflection."""

    skill = _make_skill(
        "mcm_stub",
        "def verify(payload):\n    assert payload['Q5'] == (payload['Q4']-payload['Q3'])*payload['Q2']\n",
    )
    loader = _stub_skill_loader({"mcm_stub": skill})

    bad_payload = {"Q1": "(M_1*M_2)", "Q2": 1399039928, "Q3": 1, "Q4": 3, "Q5": -2798079856}
    good_payload = {"Q1": "(M_1*M_2)", "Q2": 1399039928, "Q3": 1, "Q4": 3, "Q5": -1399039928}

    _install_fake_interpreter(
        monkeypatch,
        results=[
            _submit(bad_payload),
            _submit(good_payload),
            _ran("REFLECTION_OK"),
        ],
        verifier_results=[
            _verifier_assert("Q5 must equal (Q4-Q3)*Q2 = -1399039928, got -2798079856"),
            _verifier_pass(),
        ],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(Q1='(M_1*M_2)', Q2=1399039928, Q3=1, Q4=3, Q5=-2798079856)\n```",
            "```python\nSUBMIT(Q1='(M_1*M_2)', Q2=1399039928, Q3=1, Q4=3, Q5=-1399039928)\n```",
            "```python\nprint('REFLECTION_OK')\n```",
        ]
    )
    rlm = RLM.from_task(
        "Solve MCM.",
        outputs=["Q1", "Q2", "Q3", "Q4", "Q5"],
        lm=lm,
        max_turns=6,
        timeout=5,
        skills=["mcm_stub"],
        skill_loader=loader,
    )

    result = rlm.run()
    assert result.submitted is True
    assert result.reflection_used is True

    prompt = _reflection_prompt_text(lm)
    assert REFLECTION_HISTORY_HEADER in prompt
    assert "mcm_stub" in prompt
    assert "Q5 must equal" in prompt
    # The summary should pull out only the offending field, not the whole payload.
    assert "Q5=-2798079856" in prompt
    # Sanity: the current (good) payload is still shown verbatim too.
    assert "-1399039928" in prompt


def test_reflection_no_history_when_first_submit_passes(monkeypatch) -> None:
    """Clean SUBMIT path: reflection prompt is unchanged from pre-feedback behavior."""

    skill = _make_skill(
        "mcm_stub",
        "def verify(payload):\n    assert payload['Q5'] == (payload['Q4']-payload['Q3'])*payload['Q2']\n",
    )
    loader = _stub_skill_loader({"mcm_stub": skill})

    _install_fake_interpreter(
        monkeypatch,
        results=[
            _submit({"Q1": "(M_1*M_2)", "Q2": 10, "Q3": 1, "Q4": 2, "Q5": 10}),
            _ran("REFLECTION_OK"),
        ],
        verifier_results=[_verifier_pass()],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(Q1='(M_1*M_2)', Q2=10, Q3=1, Q4=2, Q5=10)\n```",
            "```python\nprint('REFLECTION_OK')\n```",
        ]
    )
    rlm = RLM.from_task(
        "Solve MCM.",
        outputs=["Q1", "Q2", "Q3", "Q4", "Q5"],
        lm=lm,
        max_turns=4,
        timeout=5,
        skills=["mcm_stub"],
        skill_loader=loader,
    )

    result = rlm.run()
    assert result.submitted is True

    prompt = _reflection_prompt_text(lm)
    assert REFLECTION_HISTORY_HEADER not in prompt


def test_history_preserved_across_multiple_rejections(monkeypatch) -> None:
    """Two consecutive verifier rejections both surface in the eventual reflection prompt."""

    skill = _make_skill(
        "mcm_stub",
        "def verify(payload):\n    assert payload['Q5'] == (payload['Q4']-payload['Q3'])*payload['Q2']\n",
    )
    loader = _stub_skill_loader({"mcm_stub": skill})

    _install_fake_interpreter(
        monkeypatch,
        results=[
            _submit({"Q1": "(M_1*M_2)", "Q2": 10, "Q3": 1, "Q4": 2, "Q5": -1}),
            _submit({"Q1": "(M_1*M_2)", "Q2": 10, "Q3": 1, "Q4": 2, "Q5": -2}),
            _submit({"Q1": "(M_1*M_2)", "Q2": 10, "Q3": 1, "Q4": 2, "Q5": 10}),
            _ran("REFLECTION_OK"),
        ],
        verifier_results=[
            _verifier_assert("Q5 must equal (Q4-Q3)*Q2 = 10, got -1"),
            _verifier_assert("Q5 must equal (Q4-Q3)*Q2 = 10, got -2"),
            _verifier_pass(),
        ],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(Q1='(M_1*M_2)', Q2=10, Q3=1, Q4=2, Q5=-1)\n```",
            "```python\nSUBMIT(Q1='(M_1*M_2)', Q2=10, Q3=1, Q4=2, Q5=-2)\n```",
            "```python\nSUBMIT(Q1='(M_1*M_2)', Q2=10, Q3=1, Q4=2, Q5=10)\n```",
            "```python\nprint('REFLECTION_OK')\n```",
        ]
    )
    rlm = RLM.from_task(
        "Solve MCM.",
        outputs=["Q1", "Q2", "Q3", "Q4", "Q5"],
        lm=lm,
        max_turns=8,
        timeout=5,
        skills=["mcm_stub"],
        skill_loader=loader,
    )

    result = rlm.run()
    assert result.submitted is True

    prompt = _reflection_prompt_text(lm)
    assert REFLECTION_HISTORY_HEADER in prompt
    # Both rejected Q5 values should appear in the history block.
    assert "Q5=-1" in prompt
    assert "Q5=-2" in prompt
    # And both assertion messages.
    assert "got -1" in prompt
    assert "got -2" in prompt
    # Numbered entries.
    assert "1. Skill `mcm_stub`" in prompt
    assert "2. Skill `mcm_stub`" in prompt
