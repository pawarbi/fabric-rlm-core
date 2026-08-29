"""Tests for ``fabric_rlm.skill_router.SkillRouter``."""

from __future__ import annotations

import pytest

from fabric_rlm.skill_loader import Skill
from fabric_rlm.skill_router import RouteDecision, SkillRouter


def _skill(
    name: str,
    *,
    keywords=(),
    output_fields=(),
    excludes=(),
    depends_on=(),
    specificity: str = "domain",
    verifier_present: bool = False,
    summary: str = "",
) -> Skill:
    return Skill(
        name=name,
        title=name,
        summary=summary,
        content=f"# {name}\n",
        dependencies=tuple(depends_on),
        verifier_source="def verify(p): pass" if verifier_present else None,
        applies_when_keywords=tuple(keywords),
        applies_when_output_fields=tuple(output_fields),
        excludes=tuple(excludes),
        specificity=specificity,
        verifier_present=verifier_present,
    )


def test_route_keyword_match_activates_skill() -> None:
    skills = [
        _skill("core", specificity="core"),
        _skill("mcm", keywords=["matrix chain", "MCM"], output_fields=["A1"], verifier_present=True),
        _skill("mfmc", keywords=["max-flow"], verifier_present=True),
    ]
    router = SkillRouter(skills, max_active_skills=2)

    decision = router.route("Solve this matrix chain problem with output A1..A5.")

    assert "core" in decision.active
    assert "mcm" in decision.active
    assert "mfmc" not in decision.active
    assert decision.scores["mcm"] >= 2  # keyword + output field


def test_route_caps_active_skills_and_promotes_remainder_to_cards() -> None:
    skills = [
        _skill("core", specificity="core"),
        _skill("a", keywords=["alpha"]),
        _skill("b", keywords=["beta"]),
        _skill("c", keywords=["gamma"]),
    ]
    router = SkillRouter(skills, max_active_skills=1)

    decision = router.route("alpha beta gamma")

    # core always-on + 1 cap = 2 active total
    assert "core" in decision.active
    non_core_active = [n for n in decision.active if n != "core"]
    assert len(non_core_active) == 1
    # Remaining matched skills surface as cards.
    assert set(decision.cards) == {"a", "b", "c"} - set(non_core_active)


def test_route_honors_excludes() -> None:
    skills = [
        _skill("a", keywords=["alpha"], excludes=("b",)),
        _skill("b", keywords=["alpha"]),
    ]
    router = SkillRouter(skills, max_active_skills=2)

    decision = router.route("alpha alpha")

    assert "a" in decision.active
    assert "b" not in decision.active


def test_route_pulls_in_dependencies() -> None:
    skills = [
        _skill("util", specificity="utility"),
        _skill("dom", keywords=["foo"], depends_on=["util"], verifier_present=True),
    ]
    router = SkillRouter(skills, max_active_skills=1)

    decision = router.route("foo task")

    assert "dom" in decision.active
    assert "util" in decision.active  # dependency pulled in


def test_decoy_stress_no_false_activation() -> None:
    """20 unrelated decoys must not be activated for an MCM question."""

    decoys = [_skill(f"decoy_{i}", keywords=[f"decoyword{i}"]) for i in range(20)]
    skills = [
        _skill("core", specificity="core"),
        _skill("mcm", keywords=["matrix chain"], output_fields=["A1"], verifier_present=True),
        *decoys,
    ]
    router = SkillRouter(skills, max_active_skills=2)

    decision = router.route("Solve a matrix chain with outputs A1..A5.")

    activated = set(decision.active)
    decoy_names = {d.name for d in decoys}
    assert activated.isdisjoint(decoy_names)
    assert decision.cards == ()  # no decoys scored at all


def test_explicit_skills_override_routing() -> None:
    skills = [
        _skill("core", specificity="core"),
        _skill("a", keywords=["alpha"]),
        _skill("forced", keywords=["zzz_unused"]),
    ]
    router = SkillRouter(skills, max_active_skills=1)

    decision = router.route("alpha", explicit_skills=["forced"])

    assert "forced" in decision.active
    assert "core" in decision.active


# ----- Lean-router (predict-rlm-style) configuration -----


def test_baseline_skill_names_overrides_core_specificity() -> None:
    """When `baseline_skill_names=[]` is passed, no always-on skills load."""

    skills = [
        _skill("core", specificity="core"),
        _skill("secondary_core", specificity="core"),
        _skill("mcm", keywords=["matrix chain"], verifier_present=True),
    ]
    router = SkillRouter(
        skills,
        max_active_skills=1,
        baseline_skill_names=[],  # explicit empty: no always-on
    )

    decision = router.route("Solve a matrix chain problem.")

    assert "core" not in decision.active
    assert "secondary_core" not in decision.active
    assert "mcm" in decision.active


def test_candidate_specificities_filters_ranked_picks() -> None:
    """Only domain skills can be auto-elected when restricted."""

    skills = [
        _skill("util", keywords=["alpha"], specificity="utility"),
        _skill("dom", keywords=["alpha"], specificity="domain"),
    ]
    router = SkillRouter(
        skills,
        max_active_skills=2,
        baseline_skill_names=[],
        candidate_specificities=("domain",),
    )

    decision = router.route("alpha")

    assert "dom" in decision.active
    assert "util" not in decision.active


def test_include_dependencies_false_skips_dep_closure() -> None:
    """Lean mode does NOT pull in dependency closure of an active skill."""

    skills = [
        _skill("util", specificity="utility"),
        _skill("base", specificity="core"),
        _skill(
            "dom",
            keywords=["foo"],
            depends_on=["base", "util"],
            verifier_present=True,
        ),
    ]
    router = SkillRouter(
        skills,
        max_active_skills=1,
        baseline_skill_names=[],
        candidate_specificities=("domain",),
        include_dependencies=False,
    )

    decision = router.route("foo task")

    assert decision.active == ("dom",)
    assert "base" not in decision.active
    assert "util" not in decision.active


def test_lean_router_no_keyword_match_picks_nothing() -> None:
    """No domain match + no baseline => empty active set (bare-RLM-equivalent)."""

    skills = [
        _skill("core", specificity="core"),
        _skill("dom", keywords=["alpha"], specificity="domain"),
    ]
    router = SkillRouter(
        skills,
        max_active_skills=1,
        baseline_skill_names=[],
        candidate_specificities=("domain",),
        include_dependencies=False,
    )

    decision = router.route("nothing here matches")

    assert decision.active == ()


def test_default_behavior_unchanged_when_kwargs_omitted() -> None:
    """Sanity: omitting all new kwargs preserves legacy core+deps behavior."""

    skills = [
        _skill("core", specificity="core"),
        _skill("secondary_core", specificity="core"),
        _skill(
            "dom",
            keywords=["foo"],
            depends_on=["secondary_core"],
            verifier_present=True,
        ),
    ]
    router = SkillRouter(skills, max_active_skills=1)

    decision = router.route("foo")

    # Both core skills always-on + dom + dependency closure.
    assert "core" in decision.active
    assert "secondary_core" in decision.active
    assert "dom" in decision.active
