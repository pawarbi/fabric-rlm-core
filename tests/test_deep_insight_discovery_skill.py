"""Contract tests for the source-agnostic deep insight discovery skill."""

from __future__ import annotations

import pytest

from fabric_rlm.skill_loader import SkillLoader
from fabric_rlm.skill_router import SkillRouter

SKILL = "deep_insight_discovery"


def _verify(payload: dict) -> None:
    source = SkillLoader().load(SKILL).verifier_source
    assert source is not None
    namespace: dict = {}
    exec(source, namespace)
    namespace["verify"](payload)


def _strong_insight() -> dict:
    return {
        "title": "Enterprise retention diverged from the aggregate",
        "statement": (
            "Enterprise 90-day retention fell from 91% to 78%, while overall "
            "retention remained near 86%."
        ),
        "interpretation": (
            "The aggregate masks deterioration in a high-value subgroup. "
            "Implementation quality is a hypothesis, not an established cause."
        ),
        "competing_explanations": [
            "Enterprise cohort mix changed.",
            "Instrumentation coverage declined.",
        ],
        "action": {
            "owner": "Customer Success",
            "segment": "Enterprise customers activated in Q4",
            "decision": "Review onboarding and assign recovery plans",
            "target": "Restore 90-day retention above 85%",
            "time_horizon": "Next quarter",
        },
        "priority": {"impact": "high", "urgency": "high", "rank": 1},
        "confidence": {"level": "high", "reason": "Four cohorts and 412 accounts"},
        "limitations": ["Observational data does not establish causality."],
        "verification": {
            "method": "sql",
            "expression": (
                "SELECT segment, AVG(retained_90d) AS metric_value "
                "FROM customer_cohorts WHERE cohort_quarter >= '2024-Q1' "
                "GROUP BY segment"
            ),
            "sources": {"customer_cohorts": "analytics.customer_cohorts"},
        },
    }


def test_skill_is_packaged_and_has_tiered_verifier() -> None:
    loader = SkillLoader()
    assert SKILL in loader.list_skills()

    skill = loader.load(SKILL)
    assert skill.title == SKILL
    assert skill.specificity == "domain"
    assert skill.verifier_present
    assert "source-agnostic" in skill.summary.lower()


@pytest.mark.parametrize(
    ("question", "source_skill"),
    [
        (
            "Find deep business insights and hidden subgroups in this Excel workbook",
            "excel_extract",
        ),
        (
            "Find deep business insights hidden in this Power BI semantic model",
            "semantic_model",
        ),
        (
            "Find deep insights and emerging trends in this Delta Lakehouse",
            "delta_lakehouse",
        ),
        (
            "Find non-obvious insights, cohorts, and anomalies in this JSON dataset",
            "data_exploration",
        ),
    ],
)
def test_insight_questions_compose_with_source_skill(
    question: str, source_skill: str
) -> None:
    decision = SkillRouter.from_loader(SkillLoader()).route(question)
    assert SKILL in decision.active, decision.scores
    assert source_skill in decision.active, decision.scores


def test_excel_insight_request_does_not_activate_excel_modify() -> None:
    decision = SkillRouter.from_loader(SkillLoader()).route(
        "Find deep business insights and hidden subgroups in this Excel workbook"
    )
    assert "excel_extract" in decision.active
    assert "excel_modify" not in decision.active


def test_skill_requires_systematic_discovery_and_synthesis() -> None:
    content = SkillLoader().load(SKILL).content.lower()
    for requirement in (
        "business model",
        "kpi tree",
        "computable",
        "cohort",
        "subgroup",
        "change point",
        "interaction",
        "sample size",
        "materiality",
        "non-redund",
        "causal",
        "synthesis",
        "competing explanation",
    ):
        assert requirement in content


def test_skill_follows_playbook_contract_heading_order() -> None:
    content = SkillLoader().load(SKILL).content
    headings = (
        "## Purpose",
        "## Contract: output fields",
        "## Required verifier",
        "## Tripwires",
        "## Invariants",
        "## Procedure",
    )
    positions = [content.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_verifier_accepts_source_derived_insight() -> None:
    _verify({"insights": [_strong_insight()]})


def test_verifier_requires_insights_payload() -> None:
    with pytest.raises(AssertionError, match="insights"):
        _verify({})


@pytest.mark.parametrize(
    "expression",
    [
        "SELECT 0.78 AS metric_value FROM customer_cohorts",
        "SELECT segment, 0.78 AS metric_value FROM customer_cohorts",
        "SELECT 0.78 metric_value FROM customer_cohorts",
        "SELECT 78.0 / 100 AS metric_value FROM customer_cohorts",
        (
            "WITH metric AS (SELECT 0.78 AS metric_value) "
            "SELECT metric_value FROM metric JOIN customer_cohorts ON TRUE"
        ),
        "SELECT ROUND(0.78, 2) AS metric_value FROM customer_cohorts",
        "SELECT ABS(-0.78) AS metric_value FROM customer_cohorts",
        "SELECT COALESCE(0.78, 0) AS metric_value FROM customer_cohorts",
        (
            "SELECT AVG(retained_90d) * 0 + 0.78 AS metric_value "
            "FROM customer_cohorts"
        ),
        (
            "SELECT MIN(retained_90d) - MIN(retained_90d) + 0.78 "
            "AS metric_value FROM customer_cohorts"
        ),
        (
            "SELECT CASE WHEN COUNT(*) > 0 THEN 0.78 ELSE 0.78 END "
            "AS metric_value FROM customer_cohorts"
        ),
    ],
)
def test_verifier_rejects_constant_only_sql(expression: str) -> None:
    insight = _strong_insight()
    insight["verification"] = {
        "method": "sql",
        "expression": expression,
        "sources": {"customer_cohorts": "analytics.customer_cohorts"},
    }

    with pytest.raises(AssertionError, match="recompute"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize(
    ("method", "expression"),
    [
        ("python", "metric_value = round(0.78, 2)\nprint(metric_value)"),
        ("dax", 'EVALUATE ROW("metric_value", ROUND(0.78, 2))'),
        ("api", "metric_value = float(78 / 100)"),
    ],
)
def test_verifier_rejects_constant_only_non_sql(
    method: str, expression: str
) -> None:
    insight = _strong_insight()
    insight["verification"] = {
        "method": method,
        "expression": expression,
        "sources": {"customer_cohorts": "analytics.customer_cohorts"},
    }

    with pytest.raises(AssertionError, match="recompute"):
        _verify({"insights": [insight]})


def test_verifier_rejects_source_name_only_in_comment() -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        "SELECT AVG(retained_90d) AS metric_value FROM unrelated "
        "/* customer_cohorts */"
    )

    with pytest.raises(AssertionError, match="declared source"):
        _verify({"insights": [insight]})


def test_verifier_accepts_bound_source_identifier() -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        "SELECT AVG(retained_90d) AS metric_value "
        "FROM analytics.customer_cohorts"
    )
    _verify({"insights": [insight]})


@pytest.mark.parametrize(
    "expression",
    [
        "SELECT metric_value FROM customer_cohorts",
        (
            'SELECT AVG(retained_90d) AS "metric_value" '
            "FROM analytics.customer_cohorts"
        ),
        (
            "SELECT AVG(retained_90d) AS [metric_value] "
            "FROM analytics.customer_cohorts"
        ),
    ],
)
def test_verifier_accepts_valid_metric_projection_forms(expression: str) -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = expression
    _verify({"insights": [insight]})


def test_verifier_rejects_cte_shadowing_declared_source() -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        "WITH customer_cohorts AS ("
        "SELECT AVG(x) AS metric_value FROM unrelated"
        ") SELECT metric_value FROM customer_cohorts"
    )

    with pytest.raises(AssertionError, match="declared source"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize(
    "expression",
    [
        (
            "WITH customer_cohorts(metric_value) AS ("
            "SELECT AVG(x) FROM unrelated"
            ") SELECT metric_value FROM customer_cohorts"
        ),
        (
            "WITH RECURSIVE customer_cohorts AS ("
            "SELECT AVG(x) AS metric_value FROM unrelated"
            ") SELECT metric_value FROM customer_cohorts"
        ),
    ],
)
def test_verifier_rejects_extended_cte_shadowing(expression: str) -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = expression
    with pytest.raises(AssertionError, match="declared source"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize(
    "expression",
    [
        (
            "WITH unused AS (SELECT 1 FROM customer_cohorts) "
            "SELECT AVG(x) AS metric_value FROM unrelated"
        ),
        (
            "SELECT AVG(x) AS metric_value FROM unrelated; "
            "SELECT 1 FROM customer_cohorts"
        ),
    ],
)
def test_verifier_ties_lineage_to_metric_query(expression: str) -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = expression
    with pytest.raises(AssertionError, match="declared source"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize(
    "expression",
    [
        (
            "SELECT AVG(x) AS metric_value FROM unrelated "
            "WHERE EXISTS (SELECT 1 FROM customer_cohorts)"
        ),
        (
            "SELECT AVG(x) AS metric_value FROM customer_cohorts; "
            "WITH spoof(metric_value) AS (SELECT AVG(y) FROM unrelated) "
            "SELECT metric_value FROM spoof"
        ),
    ],
)
def test_verifier_rejects_nested_or_cross_statement_lineage(
    expression: str,
) -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = expression
    with pytest.raises(AssertionError, match="declared source"):
        _verify({"insights": [insight]})


def test_verifier_rejects_undeclared_join_relation() -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        "SELECT AVG(u.x) AS metric_value "
        "FROM unrelated u CROSS JOIN customer_cohorts c"
    )
    with pytest.raises(AssertionError, match="declared source"):
        _verify({"insights": [insight]})


def test_verifier_rejects_undeclared_comma_join_relation() -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        "SELECT AVG(u.x) AS metric_value "
        "FROM customer_cohorts c, unrelated u"
    )
    with pytest.raises(AssertionError, match="comma joins"):
        _verify({"insights": [insight]})


def test_verifier_rejects_comma_join_after_keyword_in_relation_name() -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        "SELECT AVG(s.x) AS metric_value FROM somewhere s, unrelated u"
    )
    insight["verification"]["sources"] = {"somewhere": "somewhere"}
    with pytest.raises(AssertionError, match="comma joins"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize("relation", ["analytics.where", "analytics . where"])
def test_verifier_rejects_comma_join_after_qualified_keyword_relation(
    relation: str,
) -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        f"SELECT AVG(w.x) AS metric_value FROM {relation} w, unrelated u"
    )
    insight["verification"]["sources"] = {
        "qualified_keyword_relation": "analytics.where"
    }
    with pytest.raises(AssertionError, match="comma joins"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize(
    "alias",
    [
        "AS window",
        "window",
        "AS window INDEXED BY ix",
        "AS window NOT INDEXED",
        "AS offset INDEXED BY ix",
        "AS offset NOT INDEXED",
        "AS fetch INDEXED BY ix",
        "AS fetch NOT INDEXED",
        "AS qualify INDEXED BY ix",
        "AS qualify NOT INDEXED",
    ],
)
def test_verifier_rejects_comma_join_after_keyword_alias(alias: str) -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        f"SELECT SUM(window.retained_90d) AS metric_value "
        f"FROM customer_cohorts {alias}, unrelated u"
    )
    with pytest.raises(AssertionError, match="comma joins"):
        _verify({"insights": [insight]})


def test_verifier_allows_commas_after_the_from_clause() -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        "SELECT segment, region, AVG(retained_90d) AS metric_value "
        "FROM customer_cohorts GROUP BY segment, region "
        "ORDER BY segment, region"
    )
    _verify({"insights": [insight]})


@pytest.mark.parametrize(
    ("method", "expression"),
    [
        ("dax", 'EVALUATE ROW("metric_value", AVERAGE(unrelated[x]))'),
        (
            "python",
            'metric_value = unrelated.mean()\nlabel = "customer_cohorts"',
        ),
        (
            "api",
            'metric_value = sum(unrelated)\nlabel = "customer_cohorts"',
        ),
    ],
)
def test_verifier_rejects_unrelated_non_sql_lineage(
    method: str, expression: str
) -> None:
    insight = _strong_insight()
    insight["verification"] = {
        "method": method,
        "expression": expression,
        "sources": {"customer_cohorts": "analytics.customer_cohorts"},
    }
    with pytest.raises(AssertionError, match="declared source"):
        _verify({"insights": [insight]})


def test_verifier_rejects_unsupported_causal_language() -> None:
    insight = _strong_insight()
    insight["statement"] = "Poor onboarding caused enterprise retention to fall."

    with pytest.raises(AssertionError, match="causal"):
        _verify({"insights": [insight]})


def test_verifier_allows_explicit_causal_disclaimer() -> None:
    insight = _strong_insight()
    insight["interpretation"] = (
        "The data cannot establish whether poor onboarding caused the decline."
    )
    _verify({"insights": [insight]})


def test_verifier_rejects_meaningless_causal_evidence() -> None:
    insight = _strong_insight()
    insight["statement"] = "Poor onboarding caused enterprise retention to fall."
    insight["causal_evidence"] = "none"

    with pytest.raises(AssertionError, match="causal evidence"):
        _verify({"insights": [insight]})


def test_verifier_rejects_meaningless_structured_causal_evidence() -> None:
    insight = _strong_insight()
    insight["statement"] = "Poor onboarding caused enterprise retention to fall."
    insight["causal_evidence"] = {
        "design": "none",
        "result": "n/a",
        "limitations": "unknown",
    }

    with pytest.raises(AssertionError, match="causal evidence"):
        _verify({"insights": [insight]})


def test_verifier_does_not_apply_unrelated_disclaimer() -> None:
    insight = _strong_insight()
    insight["statement"] = "Poor onboarding caused enterprise retention to fall."
    insight["interpretation"] = (
        "The data cannot establish whether pricing caused acquisition to decline."
    )

    with pytest.raises(AssertionError, match="causal"):
        _verify({"insights": [insight]})


def test_verifier_does_not_apply_same_sentence_unrelated_disclaimer() -> None:
    insight = _strong_insight()
    insight["statement"] = (
        "Poor onboarding caused retention to fall, although the data cannot "
        "establish whether pricing caused acquisition to decline."
    )
    with pytest.raises(AssertionError, match="causal"):
        _verify({"insights": [insight]})


def test_verifier_allows_explicitly_negated_causal_claim() -> None:
    insight = _strong_insight()
    insight["statement"] = "The decline was not due to poor onboarding."
    _verify({"insights": [insight]})


@pytest.mark.parametrize(
    "field",
    [
        "interpretation",
        "competing_explanations",
        "action",
        "priority",
        "confidence",
        "limitations",
        "verification",
    ],
)
def test_verifier_requires_decision_quality_fields(field: str) -> None:
    insight = _strong_insight()
    del insight[field]

    with pytest.raises(AssertionError, match=field):
        _verify({"insights": [insight]})


@pytest.mark.parametrize("value", [[], ["valid", 1], "not-a-list"])
def test_verifier_requires_nonempty_text_competing_explanations(value) -> None:
    insight = _strong_insight()
    insight["competing_explanations"] = value

    with pytest.raises(AssertionError, match="competing_explanations"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", 123),
        ("statement", ["text"]),
        ("interpretation", {"text": "value"}),
        ("competing_explanations", ["none"]),
        ("limitations", [1]),
        ("limitations", ["n/a"]),
        ("priority", {"impact": "high", "urgency": "high", "rank": "first"}),
        (
            "action",
            {
                "owner": 1,
                "segment": "Enterprise",
                "decision": "Review",
                "target": "85%",
                "time_horizon": "Q4",
            },
        ),
    ],
)
def test_verifier_enforces_exact_decision_field_types(field: str, value) -> None:
    insight = _strong_insight()
    insight[field] = value
    with pytest.raises(AssertionError):
        _verify({"insights": [insight]})


def test_verifier_rejects_duplicate_insights() -> None:
    insight = _strong_insight()

    with pytest.raises(AssertionError, match="duplicate"):
        _verify({"insights": [insight, dict(insight)]})


def test_verifier_rejects_same_statement_under_new_title() -> None:
    first = _strong_insight()
    second = _strong_insight()
    second["title"] = "A different title"
    with pytest.raises(AssertionError, match="duplicate"):
        _verify({"insights": [first, second]})
