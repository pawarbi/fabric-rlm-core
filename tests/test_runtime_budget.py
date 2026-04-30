"""Tests for budget management (reserve_finalize_turns, digest_after_turn)."""

from __future__ import annotations

import pytest

from fabric_rlm import RLM
from fabric_rlm.skill_loader import Skill, SkillLoader

from test_runtime_verifier import (
    ScriptedLM,
    _install_fake_interpreter,
    _ran,
    _submit,
    _verifier_pass,
)


def _bigskill(name: str, verifier_source: str | None = None) -> Skill:
    big_body = "# " + name + "\nSummary: stub\n\n" + ("LOREM IPSUM " * 200)
    return Skill(
        name=name,
        title=name,
        summary="stub",
        content=big_body,
        dependencies=(),
        verifier_source=verifier_source,
        applies_when_keywords=("matchword",),
        applies_when_output_fields=(),
        excludes=(),
        specificity="domain",
        verifier_present=verifier_source is not None,
    )


def _budget_loader(skills: dict[str, Skill]) -> SkillLoader:
    loader = SkillLoader()
    loader.load = lambda name: skills[name]  # type: ignore[method-assign]
    loader.list_skills = lambda: sorted(skills)  # type: ignore[method-assign]
    loader.format_index = lambda: ""  # type: ignore[method-assign]
    return loader


def test_reserve_finalize_turns_injects_budget_warning(monkeypatch) -> None:
    """When budget is low, the next user turn must include a `[BUDGET]` urgency hint."""

    fake = _install_fake_interpreter(
        monkeypatch,
        results=[
            _ran("noop1"),  # turn 1
            _ran("noop2"),  # turn 2 — at this point budget_remaining = max_turns - 2 = 1
            _submit({"X": 1}),  # turn 3 with [BUDGET] prefix on user message
            _ran("REFLECTION_OK"),
        ],
        verifier_results=[],
    )
    lm = ScriptedLM(
        [
            "```python\nprint('noop1')\n```",
            "```python\nprint('noop2')\n```",
            "```python\nSUBMIT(X=1)\n```",
            "```python\nprint('REFLECTION_OK')\n```",
        ]
    )
    rlm = RLM.from_task(
        "task",
        outputs=["X"],
        lm=lm,
        max_turns=3,
        timeout=5,
        reserve_finalize_turns=1,
    )
    rlm.run()

    # Inspect the messages the LM saw on its 3rd call: the *user* message
    # should now carry the [BUDGET] prefix.
    third_call_msgs = lm.messages[2]
    last_user = next(m for m in reversed(third_call_msgs) if m["role"] == "user")
    assert "[BUDGET]" in last_user["content"]


def test_digest_after_turn_replaces_active_skill_body(monkeypatch) -> None:
    """After ``digest_after_turn`` turns, the system message swaps the full skill body for a digest."""

    skills = {
        "fat": _bigskill("fat", verifier_source="def verify(p): assert True\n"),
    }
    loader = _budget_loader(skills)

    fake = _install_fake_interpreter(
        monkeypatch,
        results=[
            _ran("turn1"),
            _ran("turn2"),
            _submit({"X": 1}),
            _ran("REFLECTION_OK"),
        ],
        verifier_results=[_verifier_pass()],
    )
    lm = ScriptedLM(
        [
            "```python\nprint('turn1')\n```",
            "```python\nprint('turn2')\n```",
            "```python\nSUBMIT(X=1)\n```",
            "```python\nprint('REFLECTION_OK')\n```",
        ]
    )
    rlm = RLM.from_task(
        "matchword task",
        outputs=["X"],
        lm=lm,
        max_turns=5,
        timeout=5,
        skill_loader=loader,
        enable_router=True,
        digest_after_turn=2,
    )
    rlm.run(inputs={"q": "matchword question text"})
    sys_t1 = lm.messages[0][0]["content"]
    assert "LOREM IPSUM" in sys_t1
    assert "## Skill: fat" in sys_t1

    # Turn 3 (after digest_after_turn=2 reached): digest header replaces full body.
    sys_t3 = lm.messages[2][0]["content"]
    assert "LOREM IPSUM" not in sys_t3
    assert "## Skill (digest): fat" in sys_t3
