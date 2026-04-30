"""Tests for runtime + SkillRouter integration."""

from __future__ import annotations

import pytest

import fabric_rlm.runtime as runtime_mod
from fabric_rlm import RLM
from fabric_rlm.skill_loader import Skill, SkillLoader

from test_runtime_verifier import (
    FakeInterpreter,
    ScriptedLM,
    _install_fake_interpreter,
    _ran,
    _submit,
    _verifier_assert,
    _verifier_pass,
)


def _router_skill(
    name: str,
    *,
    keywords=(),
    output_fields=(),
    depends_on=(),
    specificity: str = "domain",
    verifier_source: str | None = None,
) -> Skill:
    return Skill(
        name=name,
        title=name,
        summary=f"{name} stub",
        content=f"# {name}\nSummary: {name} stub\n",
        dependencies=tuple(depends_on),
        verifier_source=verifier_source,
        applies_when_keywords=tuple(keywords),
        applies_when_output_fields=tuple(output_fields),
        excludes=(),
        specificity=specificity,
        verifier_present=verifier_source is not None,
    )


def _routing_loader(skills: dict[str, Skill]) -> SkillLoader:
    loader = SkillLoader()
    loader.load = lambda name: skills[name]  # type: ignore[method-assign]
    loader.list_skills = lambda: sorted(skills)  # type: ignore[method-assign]
    loader.format_index = lambda: ""  # type: ignore[method-assign]
    return loader


def test_router_runs_only_active_skill_verifier(monkeypatch) -> None:
    """When the router activates `mcm_stub` only, `mfmc_stub`'s verifier must NOT run."""

    mcm_verifier = "def verify(payload):\n    assert payload.get('Q1'), 'Q1 missing'\n"
    mfmc_verifier = "def verify(payload):\n    assert False, 'mfmc verifier should never run here'\n"
    skills = {
        "mcm_stub": _router_skill(
            "mcm_stub", keywords=["matrix chain"], output_fields=["Q1"], verifier_source=mcm_verifier
        ),
        "mfmc_stub": _router_skill(
            "mfmc_stub", keywords=["max-flow"], verifier_source=mfmc_verifier
        ),
    }
    loader = _routing_loader(skills)

    fake = _install_fake_interpreter(
        monkeypatch,
        results=[
            _submit({"Q1": "(M_1*M_2)"}),
            _ran("REFLECTION_OK"),
        ],
        verifier_results=[_verifier_pass()],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(Q1='(M_1*M_2)')\n```",
            "```python\nprint('REFLECTION_OK')\n```",
        ]
    )
    rlm = RLM.from_task(
        "Solve a matrix chain question.",
        outputs=["Q1"],
        lm=lm,
        max_turns=4,
        timeout=5,
        skill_loader=loader,
        enable_router=True,
        max_active_skills=2,
    )
    result = rlm.run(inputs={"q": "matrix chain question"})

    assert result.submitted is True
    # mcm verifier ran exactly once; mfmc verifier MUST NOT have run.
    assert len(fake.verifier_executed) == 1
    code = fake.verifier_executed[0]
    assert "Q1 missing" in code
    assert "mfmc verifier should never run here" not in code
    # Router metadata should reflect what was active.
    assert "mcm_stub" in result.trajectory.metadata.get("router_active", [])


def test_activate_skill_marker_enables_verifier_mid_run(monkeypatch) -> None:
    """Worker stdout `[FABRIC_RLM_ACTIVATE]:name` adds the skill to the active set."""

    bonus_verifier = "def verify(payload):\n    assert payload.get('B') == 1, 'B must be 1'\n"
    skills = {
        "core": _router_skill("core", specificity="core"),
        "bonus": _router_skill("bonus", keywords=["unmatched-keyword"], verifier_source=bonus_verifier),
    }
    loader = _routing_loader(skills)

    # Turn 1: LM "explores" + prints the activation marker.
    # Turn 2: LM submits (which now triggers bonus verifier — and it passes).
    # Turn 3: reflection.
    fake = _install_fake_interpreter(
        monkeypatch,
        results=[
            _ran(stdout="[FABRIC_RLM_ACTIVATE]:bonus\n"),
            _submit({"B": 1}),
            _ran("REFLECTION_OK"),
        ],
        verifier_results=[_verifier_pass()],
    )
    lm = ScriptedLM(
        [
            "```python\nactivate_skill('bonus')\n```",
            "```python\nSUBMIT(B=1)\n```",
            "```python\nprint('REFLECTION_OK')\n```",
        ]
    )
    rlm = RLM.from_task(
        "Solve an unrelated task.",
        outputs=["B"],
        lm=lm,
        max_turns=5,
        timeout=5,
        skill_loader=loader,
        enable_router=True,
    )
    result = rlm.run()

    assert result.submitted is True
    # bonus verifier ran exactly once (after activation, on the SUBMIT turn).
    assert len(fake.verifier_executed) == 1
    assert "bonus" in rlm._activated_skills


def test_router_disabled_preserves_legacy_behavior(monkeypatch) -> None:
    """`enable_router=False` (default) routes solely off `skills=[...]`."""

    skill_v = _router_skill(
        "stub",
        verifier_source="def verify(payload):\n    assert True\n",
    )
    loader = _routing_loader({"stub": skill_v})

    fake = _install_fake_interpreter(
        monkeypatch,
        results=[
            _submit({"X": 1}),
            _ran("REFLECTION_OK"),
        ],
        verifier_results=[_verifier_pass()],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(X=1)\n```",
            "```python\nprint('REFLECTION_OK')\n```",
        ]
    )
    rlm = RLM.from_task(
        "task",
        outputs=["X"],
        lm=lm,
        max_turns=4,
        timeout=5,
        skills=["stub"],
        skill_loader=loader,
        # enable_router defaults to False
    )
    result = rlm.run()

    assert result.submitted is True
    assert len(fake.verifier_executed) == 1  # legacy path still runs verifiers
