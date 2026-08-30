"""Routing must score skills against bound inputs, not just task description.

Pre-fix bug: when the task description was generic, the router never saw the
actual question supplied as a bound input, so domain skills never activated
correctly. This test pins the universal fix so routing stays sensitive to
input content for any signature.
"""

from __future__ import annotations

import pytest

import fabric_rlm.runtime as runtime_mod
from fabric_rlm import RLM
from fabric_rlm.skill_loader import SkillLoader

from test_runtime_router_integration import _router_skill, _routing_loader
from test_runtime_verifier import (
    FakeInterpreter,
    ScriptedLM,
    _install_fake_interpreter,
    _ran,
    _submit,
    _verifier_pass,
)


GENERIC_TASK = (
    "Solve the algorithm question provided as the bound input "
    "`question` and SUBMIT(output=...)."
)


def _five_domain_skills():
    return {
        "core": _router_skill("core", specificity="core"),
        "domain_mfmc": _router_skill(
            "domain_mfmc",
            keywords=["max-flow", "min-cut", "FLOW GAUNTLET"],
        ),
        "domain_backprop": _router_skill(
            "domain_backprop",
            keywords=["backprop", "backpropagation", "gradient"],
        ),
        "domain_mcm": _router_skill(
            "domain_mcm",
            keywords=["matrix chain", "MCM", "parenthesization"],
        ),
        "domain_distmem": _router_skill(
            "domain_distmem",
            keywords=["distributed memory", "DistMem", "barrier"],
        ),
        "domain_vliw": _router_skill(
            "domain_vliw",
            keywords=["VLIW", "issue width"],
        ),
    }


@pytest.mark.parametrize(
    "question_text, expected_skill",
    [
        ("FLOW GAUNTLET: compute the max-flow from s to t.", "domain_mfmc"),
        ("Compute the gradient of the loss via backpropagation.", "domain_backprop"),
        ("Find the optimal parenthesization for this matrix chain (MCM).", "domain_mcm"),
        ("With distributed memory and a barrier between phases, ...", "domain_distmem"),
        ("VLIW scheduling with issue width 4: pack the bundles.", "domain_vliw"),
    ],
    ids=["mfmc", "backprop", "mcm", "distmem", "vliw"],
)
def test_router_activates_correct_template_skill_from_bound_input(
    monkeypatch, question_text, expected_skill
) -> None:
    """Routing must pick the template-specific skill based on the question text
    bound as an input, even when the task description is generic."""

    skills = _five_domain_skills()
    loader = _routing_loader(skills)

    fake = _install_fake_interpreter(
        monkeypatch,
        results=[
            _submit({"output": "solution = 1"}),
            _ran("REFLECTION_OK"),
        ],
        verifier_results=[_verifier_pass()],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(output='solution = 1')\n```",
            "```python\nprint('REFLECTION_OK')\n```",
        ]
    )
    rlm = RLM.from_task(
        GENERIC_TASK,
        outputs=["output"],
        lm=lm,
        max_turns=4,
        timeout=5,
        skill_loader=loader,
        enable_router=True,
        max_active_skills=2,
    )
    result = rlm.run(inputs={"question": question_text})

    active = result.trajectory.metadata.get("router_active", [])
    assert (
        expected_skill in active
    ), f"expected {expected_skill!r} in router_active, got {active!r}"


def test_routing_text_helper_uses_inputs_only() -> None:
    """The routing-text helper ignores ``task_text`` (which describes the
    template menu) and routes purely on bound inputs."""

    text = runtime_mod._build_routing_text(
        "generic task description that mentions max-flow and backprop and VLIW",
        {"q": "FLOW GAUNTLET", "extra": 42, "skip": None},
    )
    # task_text deliberately excluded so menu words don't inflate every score.
    assert "generic task" not in text
    assert "max-flow" not in text
    assert "backprop" not in text
    assert "FLOW GAUNTLET" in text
    assert "42" in text  # non-string coerced
    # None values must be ignored, not stringified as 'None'.


def test_routing_text_helper_caps_per_input_length() -> None:
    """Long inputs are truncated to keep routing cheap."""

    big = "x" * 10_000
    text = runtime_mod._build_routing_text("t", {"big": big}, per_input_chars=128)
    assert len(text) < 200
