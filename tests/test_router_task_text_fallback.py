"""Routing falls back to the task text when bound inputs carry no signal.

Fixes the ``from_task`` blind spot behind the historic router-activation bug:
``RLM.from_task(task="<the actual question>", inputs={"pdf_path": ...})``
used to route on the file path alone, score nothing, and elect the same
always-on bundle for every question.

The fallback is gated on *zero* input-keyword signal, so the original
protection still holds: when bound inputs match any skill keyword (the
benchmark shape where the task text enumerates every template), the task
text is never consulted.
"""

from __future__ import annotations

import pytest

import fabric_rlm.runtime as runtime_mod
from fabric_rlm import RLM
from fabric_rlm.runtime import _route_with_task_fallback
from fabric_rlm.skill_router import SkillRouter

from test_runtime_router_integration import _router_skill, _routing_loader
from test_runtime_verifier import (
    ScriptedLM,
    _install_fake_interpreter,
    _ran,
    _submit,
    _verifier_pass,
)


def _two_domain_skills():
    return {
        "core": _router_skill("core", specificity="core"),
        "pdf_stub": _router_skill("pdf_stub", keywords=["pdf", "contract", "clause"]),
        "logs_stub": _router_skill("logs_stub", keywords=["spark log", "OOM"]),
    }


def _router(skills) -> SkillRouter:
    return SkillRouter(skills.values(), max_active_skills=2)


# ----- helper-level behavior ---------------------------------------------------


def test_fallback_used_when_inputs_have_no_signal() -> None:
    skills = _two_domain_skills()
    decision, used_fallback = _route_with_task_fallback(
        _router(skills),
        "Summarize the termination clause of the contract PDF.",
        {"pdf_path": "/lakehouse/default/Files/agreement_2024_final.bin"},
        explicit_skills=None,
    )
    assert used_fallback is True
    assert "pdf_stub" in decision.active


def test_no_fallback_when_inputs_match_a_keyword() -> None:
    """Menu-inflation protection: inputs matched, so task text stays excluded
    even though it name-drops the other skill."""
    skills = _two_domain_skills()
    decision, used_fallback = _route_with_task_fallback(
        _router(skills),
        "Menu: choose between pdf analysis and spark log analysis (OOM).",
        {"question": "Find the failed job in the spark log."},
        explicit_skills=None,
    )
    assert used_fallback is False
    assert "logs_stub" in decision.active
    assert "pdf_stub" not in decision.active


def test_no_fallback_without_task_text() -> None:
    skills = _two_domain_skills()
    decision, used_fallback = _route_with_task_fallback(
        _router(skills),
        None,
        {"pdf_path": "/data/file.bin"},
        explicit_skills=None,
    )
    assert used_fallback is False
    assert decision.scores == {}


# ----- end-to-end through RLM.from_task ----------------------------------------


def test_from_task_question_in_task_routes_domain_skill(monkeypatch) -> None:
    """The canonical bug shape: question in task=, inputs are just a path."""
    skills = _two_domain_skills()
    loader = _routing_loader(skills)

    _install_fake_interpreter(
        monkeypatch,
        results=[
            _submit({"output": "done"}),
            _ran("REFLECTION_OK"),
        ],
        verifier_results=[_verifier_pass()],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(output='done')\n```",
            "```python\nprint('REFLECTION_OK')\n```",
        ]
    )
    rlm = RLM.from_task(
        "Summarize the parties, term, and termination clause of the contract.",
        inputs={"pdf_path": "/lakehouse/default/Files/agreement.bin"},
        outputs=["output"],
        lm=lm,
        max_turns=4,
        timeout=5,
        skill_loader=loader,
        enable_router=True,
        max_active_skills=2,
    )
    result = rlm.run()

    active = result.trajectory.metadata.get("router_active", [])
    assert "pdf_stub" in active, f"expected pdf_stub in router_active, got {active!r}"
    assert result.trajectory.metadata.get("router_used_task_text_fallback") is True


def test_from_task_question_in_inputs_does_not_use_fallback(monkeypatch) -> None:
    """Existing behavior pinned: question bound as input -> inputs-only routing."""
    skills = _two_domain_skills()
    loader = _routing_loader(skills)

    _install_fake_interpreter(
        monkeypatch,
        results=[
            _submit({"output": "done"}),
            _ran("REFLECTION_OK"),
        ],
        verifier_results=[_verifier_pass()],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(output='done')\n```",
            "```python\nprint('REFLECTION_OK')\n```",
        ]
    )
    rlm = RLM.from_task(
        "Solve the question provided as the bound input `question`.",
        outputs=["output"],
        lm=lm,
        max_turns=4,
        timeout=5,
        skill_loader=loader,
        enable_router=True,
        max_active_skills=2,
    )
    result = rlm.run(inputs={"question": "Count OOM events in the spark log."})

    active = result.trajectory.metadata.get("router_active", [])
    assert "logs_stub" in active
    assert result.trajectory.metadata.get("router_used_task_text_fallback") is False
