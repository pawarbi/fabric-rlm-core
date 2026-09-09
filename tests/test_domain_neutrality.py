"""Nothing the model reads may teach it one company's vocabulary.

The bundled skills are delivered verbatim into the prompt, and they are
routed on generic analytical keywords ("ranking", "materiality"), so a
manufacturing or logistics task activates the same text a SaaS revenue task
does. Illustrative examples written in one domain's nouns are therefore not
harmless: they prime the model toward measures the data does not have.

This guard covers text that reaches the model. Developer-facing docstrings
and comments are deliberately out of scope; they are read by people who
benefit from a concrete example.
"""

from __future__ import annotations

import re

import pytest

from fabric_rlm.skill_loader import SkillLoader

# Nouns belonging to one line of business rather than to analysis itself.
# "churn" and "retention" appear in the discovery skill as named examples of
# a metric that must not be inferred from a status snapshot, which is a
# statement about evidence, so they are matched only as bare measure names.
_DOMAIN_TERMS = (
    r"\bARR\b",
    r"\bMRR\b",
    r"\bannual recurring\b",
    r"\bmonthly recurring\b",
    r"\bCustomer Group\b",
    r"\bmrr\b",
    r"\barr\b",
)
_PATTERN = re.compile("|".join(_DOMAIN_TERMS))


def _model_facing_skills() -> list[str]:
    loader = SkillLoader()
    return sorted(loader.list_skills())


@pytest.mark.parametrize("name", _model_facing_skills())
def test_bundled_skill_text_names_no_single_domain_measure(name: str) -> None:
    text = SkillLoader().load_text(name)
    found = sorted({match.group(0) for match in _PATTERN.finditer(text)})
    assert not found, (
        f"skill {name!r} is delivered to the model and names {found}; "
        "use a domain-neutral example such as 'the measure' or 'amount'"
    )


def test_the_guard_would_catch_a_regression() -> None:
    """The pattern is not vacuous."""
    assert _PATTERN.search("report ARR by region")
    assert _PATTERN.search("SUM(s.mrr) AS active_mrr")
    assert not _PATTERN.search("SUM(s.amount) AS active_amount")
    assert not _PATTERN.search("rank the production lines by defect rate")
