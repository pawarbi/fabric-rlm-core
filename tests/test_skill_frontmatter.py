"""Tests for YAML frontmatter parsing in skill_loader (Track 5″)."""

from __future__ import annotations

import textwrap

import pytest

from fabric_rlm.skill_loader import SkillLoader


def _write(tmp_path, name: str, body: str) -> None:
    (tmp_path / f"{name}.md").write_text(textwrap.dedent(body), encoding="utf-8")


def test_parses_full_frontmatter(tmp_path) -> None:
    _write(
        tmp_path,
        "demo",
        """\
        ---
        applies_when:
          keywords: ["FLOW GAUNTLET", "max-flow"]
          output_fields: ["final_capacities", "saturated_edges"]
        excludes: ["other_skill"]
        depends_on: ["base_skill"]
        specificity: domain
        ---
        # demo
        Summary: a demo skill.
        Dependencies: legacy_dep

        ## Required verifier

        ```python
        def verify(payload):
            assert True
        ```
        """,
    )
    skill = SkillLoader(skill_dir=tmp_path).load("demo")
    assert skill.applies_when_keywords == ("FLOW GAUNTLET", "max-flow")
    assert skill.applies_when_output_fields == ("final_capacities", "saturated_edges")
    assert skill.excludes == ("other_skill",)
    # Frontmatter `depends_on` overrides legacy `Dependencies:` line.
    assert skill.dependencies == ("base_skill",)
    assert skill.specificity == "domain"
    assert skill.verifier_present is True


def test_no_frontmatter_uses_defaults_and_legacy_deps(tmp_path) -> None:
    _write(
        tmp_path,
        "legacy",
        """\
        # legacy
        Summary: no frontmatter at all.
        Dependencies: foo, bar
        """,
    )
    skill = SkillLoader(skill_dir=tmp_path).load("legacy")
    assert skill.applies_when_keywords == ()
    assert skill.applies_when_output_fields == ()
    assert skill.excludes == ()
    assert skill.dependencies == ("foo", "bar")
    assert skill.specificity == "domain"
    assert skill.verifier_present is False


def test_malformed_yaml_is_graceful(tmp_path) -> None:
    _write(
        tmp_path,
        "broken",
        """\
        ---
        applies_when: [this: is not] valid yaml at all]
        ---
        # broken
        Summary: bad frontmatter, body still loads.
        Dependencies: base
        """,
    )
    skill = SkillLoader(skill_dir=tmp_path).load("broken")
    # Falls back to defaults; the legacy Dependencies line is still parsed
    # because we strip the frontmatter region before reading metadata.
    assert skill.applies_when_keywords == ()
    assert skill.specificity == "domain"
    assert skill.dependencies == ("base",)


def test_specificity_validation(tmp_path) -> None:
    _write(
        tmp_path,
        "weird",
        """\
        ---
        specificity: bogus
        ---
        # weird
        Summary: bad specificity.
        """,
    )
    skill = SkillLoader(skill_dir=tmp_path).load("weird")
    assert skill.specificity == "domain"  # defaulted


def test_verifier_present_auto_derived(tmp_path) -> None:
    _write(
        tmp_path,
        "withv",
        """\
        ---
        specificity: domain
        ---
        # withv
        Summary: has verifier.

        ## Required verifier

        ```python
        def verify(payload):
            pass
        ```
        """,
    )
    _write(
        tmp_path,
        "without",
        """\
        ---
        specificity: utility
        ---
        # without
        Summary: no verifier.
        """,
    )
    loader = SkillLoader(skill_dir=tmp_path)
    assert loader.load("withv").verifier_present is True
    assert loader.load("without").verifier_present is False
    assert loader.load("without").specificity == "utility"


def test_string_keywords_split_on_comma(tmp_path) -> None:
    _write(
        tmp_path,
        "csv",
        """\
        ---
        applies_when:
          keywords: "alpha, beta, gamma"
        ---
        # csv
        Summary: keywords as a CSV string.
        """,
    )
    skill = SkillLoader(skill_dir=tmp_path).load("csv")
    assert skill.applies_when_keywords == ("alpha", "beta", "gamma")
