"""Contract tests for the source-agnostic deep insight discovery skill."""

from __future__ import annotations

import pytest

from fabric_rlm.skill_loader import SkillLoader
from fabric_rlm.skill_router import SkillRouter

SKILL = "deep_insight_discovery"


def _verify_raw(payload: dict) -> None:
    source = SkillLoader().load(SKILL).verifier_source
    assert source is not None
    namespace: dict = {}
    exec(source, namespace)
    namespace["verify"](payload)


def _strong_plan() -> dict:
    return {
        "business_context": "Organizations progress through an observed lifecycle.",
        "kpi_map": [
            {
                "kpi": "90-day retention",
                "computability": "computable",
                "reason": "Observed cohort and retained-state fields are available.",
            },
            {
                "kpi": "Customer advocacy",
                "computability": "not_computable",
                "reason": "No survey or advocacy measure is available.",
            },
        ],
        "search_space": {
            "dimensions_available": ["segment", "cohort_quarter"],
            "dimensions_deferred": [],
            "time_grains_available": ["quarter"],
            "populations": ["Customers activated since 2024-Q1"],
        },
    }


def _strong_candidates(insights: list[dict]) -> list[dict]:
    return [
        {
            "candidate": insight["title"],
            "dimensions_tested": list(
                insight.get("discovery", {}).get("dimensions_tested", [])
            ),
            "disposition": "promoted",
            "reason": "Material, persistent, and decision-relevant.",
            "promoted_as": insight["title"],
        }
        for insight in insights
    ]


def _quantitative_rejection_evidence() -> dict:
    return {
        "effect_value": 0.01,
        "baseline_value": 0.50,
        "sample_size": 412,
        "verification": {
            "method": "sql",
            "expression": (
                "SELECT AVG(retained_90d) AS metric_value "
                "FROM customer_cohorts"
            ),
            "sources": {"customer_cohorts": "analytics.customer_cohorts"},
            "components": [
                {
                    "name": "effect_value",
                    "expected_value": 0.01,
                    "expression": (
                        "SELECT AVG(retained_90d) AS metric_value "
                        "FROM customer_cohorts WHERE segment = 'Enterprise'"
                    ),
                    "sources": {
                        "customer_cohorts": "analytics.customer_cohorts"
                    },
                },
                {
                    "name": "baseline_value",
                    "expected_value": 0.50,
                    "expression": (
                        "SELECT AVG(retained_90d) AS metric_value "
                        "FROM customer_cohorts"
                    ),
                    "sources": {
                        "customer_cohorts": "analytics.customer_cohorts"
                    },
                },
                {
                    "name": "sample_size",
                    "expected_value": 412,
                    "expression": (
                        "SELECT COUNT(customer_id) AS metric_value "
                        "FROM customer_cohorts"
                    ),
                    "sources": {
                        "customer_cohorts": "analytics.customer_cohorts"
                    },
                },
            ],
        },
    }


def _verify(payload: dict) -> None:
    expanded = dict(payload)
    expanded.setdefault("analysis_plan", _strong_plan())
    expanded.setdefault(
        "candidates",
        _strong_candidates(expanded.get("insights", [])),
    )
    _verify_raw(expanded)


def _strong_insight() -> dict:
    return {
        "title": "Enterprise retention diverged from the aggregate",
        "statement": "Enterprise 90-day retention fell from 91% to 78%.",
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
        "evidence_tier": "associational",
        "limitations": ["Observational data does not establish causality."],
        "supporting_claims": [],
        "discovery": {
            "pattern_type": "subgroup",
            "dimensions_tested": ["segment", "cohort_quarter"],
            "population": "Customers activated since 2024-Q1",
            "sample_size": 412,
            "robustness_checks": [
                "Compared four activation cohorts.",
                "Checked cohort denominator and composition stability.",
                "Recomputed with and without incomplete cohorts.",
            ],
        },
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


def _metric_component(
    name: str,
    role: str,
    expected_value: float,
    *,
    method: str = "sql",
) -> dict:
    if method == "python":
        expression = f"metric_value = {name}.sum()"
        sources = {name: name}
    else:
        expression = (
            f"SELECT SUM({name}) AS metric_value FROM customer_cohorts"
        )
        sources = {"customer_cohorts": "analytics.customer_cohorts"}
    return {
        "name": name,
        "role": role,
        "expected_value": expected_value,
        "verification": {
            "method": method,
            "expression": expression,
            "sources": sources,
        },
    }


def _derived_metric_spec(
    metric_type: str,
    expected_value: float,
    components: list[dict],
) -> dict:
    return {
        "type": metric_type,
        "expected_value": expected_value,
        "components": components,
    }


def _diagnostic_assessment(
    *,
    measurable: bool,
    disposition: str,
    decision_readiness: str,
    method: str = "sql",
) -> dict:
    explanation = "Enterprise cohort mix changed."
    item = {
        "explanation": explanation,
        "measurable": measurable,
        "disposition": disposition,
    }
    if measurable and disposition in {"ruled_out", "weakened"}:
        item["expected_value"] = 0.01
        item["verification"] = _metric_component(
            "mix_shift", "value", 0.01, method=method
        )["verification"]
    if not measurable:
        item["limitation"] = (
            "The source has no account-level implementation-quality measure."
        )
    return {
        "decision_readiness": decision_readiness,
        "explanations": [
            item,
            {
                "explanation": "Instrumentation coverage declined.",
                "measurable": False,
                "disposition": "not_measurable",
                "limitation": (
                    "Historical instrumentation coverage metadata is unavailable."
                ),
            },
        ],
    }


def test_verifier_accepts_source_agnostic_analysis_plan() -> None:
    insights = [_strong_insight()]
    _verify_raw(
        {
            "analysis_plan": {
                "business_context": "Devices emit operational events.",
                "kpi_map": [
                    {
                        "kpi": "Peak processing latency",
                        "computability": "partially_computable",
                        "reason": "Latency exists but maintenance windows are unavailable.",
                    }
                ],
                "search_space": {
                    "dimensions_available": [],
                    "dimensions_deferred": [],
                    "time_grains_available": [],
                    "populations": ["All observed devices"],
                },
            },
            "candidates": _strong_candidates(insights),
            "insights": insights,
        }
    )


def test_verifier_requires_analysis_plan() -> None:
    with pytest.raises(AssertionError, match="analysis_plan is required"):
        _verify_raw({"insights": [_strong_insight()]})


@pytest.mark.parametrize(
    "plan",
    [
        {},
        {
            "business_context": "Observed process",
            "kpi_map": [],
            "search_space": {
                "dimensions_available": [],
                "time_grains_available": [],
                "populations": ["All records"],
            },
        },
        {
            "business_context": "Observed process",
            "kpi_map": [
                {
                    "kpi": "Cycle time",
                    "computability": "estimated",
                    "reason": "Timestamps are available.",
                }
            ],
            "search_space": {
                "dimensions_available": [],
                "time_grains_available": [],
                "populations": ["All records"],
            },
        },
    ],
)
def test_verifier_rejects_invalid_analysis_plan(plan: dict) -> None:
    with pytest.raises(AssertionError, match="analysis_plan|kpi computability"):
        _verify_raw({"analysis_plan": plan, "insights": [_strong_insight()]})


def test_verifier_requires_candidate_ledger() -> None:
    with pytest.raises(AssertionError, match="candidates ledger is required"):
        _verify_raw(
            {
                "analysis_plan": _strong_plan(),
                "insights": [_strong_insight()],
            }
        )


def test_verifier_requires_every_insight_to_be_promoted() -> None:
    with pytest.raises(AssertionError, match="not promoted"):
        _verify_raw(
            {
                "analysis_plan": _strong_plan(),
                "candidates": [
                    {
                        "candidate": "Another finding",
                        "dimensions_tested": ["segment"],
                        "disposition": "rejected",
                        "reason": "The effect was not persistent.",
                        "rejection_type": "not_computable",
                        "missing_fields": ["retention_observation_window"],
                        "promoted_as": None,
                    }
                ],
                "insights": [_strong_insight()],
            }
        )


def test_verifier_rejects_candidate_with_invalid_disposition() -> None:
    insights = [_strong_insight()]
    candidates = _strong_candidates(insights)
    candidates[0]["disposition"] = "interesting"

    with pytest.raises(AssertionError, match="disposition is invalid"):
        _verify_raw(
            {
                "analysis_plan": _strong_plan(),
                "candidates": candidates,
                "insights": insights,
            }
        )


def test_verifier_requires_evidence_for_quantitative_candidate_rejection() -> None:
    insight = _strong_insight()
    candidates = _strong_candidates([insight])
    candidates.append(
        {
            "candidate": "Retention does not differ by region",
            "dimensions_tested": ["segment"],
            "disposition": "rejected",
            "reason": "The observed difference was negligible.",
            "rejection_type": "quantitative",
            "promoted_as": None,
        }
    )

    with pytest.raises(AssertionError, match="quantitative rejection evidence"):
        _verify(
            {
                "candidates": candidates,
                "insights": [insight],
            }
        )


def test_verifier_accepts_evidence_backed_quantitative_rejection() -> None:
    insight = _strong_insight()
    candidates = _strong_candidates([insight])
    candidates.append(
        {
            "candidate": "Retention does not differ by region",
            "dimensions_tested": ["segment"],
            "disposition": "rejected",
            "reason": "The measured effect was negligible relative to baseline.",
            "rejection_type": "quantitative",
            "rejection_evidence": _quantitative_rejection_evidence(),
            "promoted_as": None,
        }
    )

    _verify({"candidates": candidates, "insights": [insight]})


def test_verifier_accepts_nested_quantitative_rejection_component_verification() -> None:
    insight = _strong_insight()
    evidence = _quantitative_rejection_evidence()
    for component in evidence["verification"]["components"]:
        component["verification"] = {
            "method": evidence["verification"]["method"],
            "expression": component.pop("expression"),
            "sources": component.pop("sources"),
        }
    candidates = _strong_candidates([insight])
    candidates.append(
        {
            "candidate": "Retention does not differ by region",
            "dimensions_tested": ["segment"],
            "disposition": "rejected",
            "reason": "The measured effect was negligible relative to baseline.",
            "rejection_type": "quantitative",
            "rejection_evidence": evidence,
            "promoted_as": None,
        }
    )

    _verify({"candidates": candidates, "insights": [insight]})


def test_verifier_accepts_not_computable_rejection_with_missing_field() -> None:
    insight = _strong_insight()
    candidates = _strong_candidates([insight])
    candidates.append(
        {
            "candidate": "Advocacy predicts retention",
            "dimensions_tested": [],
            "disposition": "rejected",
            "reason": "The source has no advocacy outcome.",
            "rejection_type": "not_computable",
            "missing_fields": ["advocacy_score"],
            "promoted_as": None,
        }
    )

    _verify({"candidates": candidates, "insights": [insight]})


def test_verifier_rejects_not_computable_claim_for_available_dimension() -> None:
    insight = _strong_insight()
    candidates = _strong_candidates([insight])
    candidates.append(
        {
            "candidate": "Retention does not differ by segment",
            "dimensions_tested": [],
            "disposition": "rejected",
            "reason": "The source has no segment field.",
            "rejection_type": "not_computable",
            "missing_fields": ["segment"],
            "promoted_as": None,
        }
    )

    with pytest.raises(AssertionError, match="available field"):
        _verify({"candidates": candidates, "insights": [insight]})


def test_verifier_rejects_inflated_insight_dimension_provenance() -> None:
    insight = _strong_insight()
    insight["discovery"]["dimensions_tested"].append("region")
    candidates = _strong_candidates([insight])
    candidates[0]["dimensions_tested"] = ["segment", "cohort_quarter"]

    with pytest.raises(AssertionError, match="dimensions do not match promoted candidate"):
        _verify_raw(
            {
                "analysis_plan": _strong_plan(),
                "candidates": candidates,
                "insights": [insight],
            }
        )


def test_verifier_accepts_no_rejected_candidates_for_small_search_space() -> None:
    insight = _strong_insight()
    insight["discovery"]["dimensions_tested"] = []
    plan = _strong_plan()
    plan["search_space"]["dimensions_available"] = []

    _verify_raw(
        {
            "analysis_plan": plan,
            "candidates": _strong_candidates([insight]),
            "insights": [insight],
        }
    )


def test_verifier_requires_available_dimensions_to_be_tested_or_deferred() -> None:
    insight = _strong_insight()
    plan = _strong_plan()
    plan["search_space"]["dimensions_available"].append("region")

    with pytest.raises(AssertionError, match="unsearched dimension"):
        _verify_raw(
            {
                "analysis_plan": plan,
                "candidates": _strong_candidates([insight]),
                "insights": [insight],
            }
        )


def test_verifier_accepts_dimension_deferred_with_reason() -> None:
    insight = _strong_insight()
    plan = _strong_plan()
    plan["search_space"]["dimensions_available"].append("region")
    plan["search_space"]["dimensions_deferred"] = [
        {
            "dimension": "region",
            "reason": "Region is redacted in this extract.",
        }
    ]

    _verify_raw(
        {
            "analysis_plan": plan,
            "candidates": _strong_candidates([insight]),
            "insights": [insight],
        }
    )


def test_verifier_rejects_dimension_deferred_without_reason() -> None:
    insight = _strong_insight()
    plan = _strong_plan()
    plan["search_space"]["dimensions_available"].append("region")
    plan["search_space"]["dimensions_deferred"] = [
        {"dimension": "region", "reason": ""}
    ]

    with pytest.raises(AssertionError, match="deferred dimension.*reason"):
        _verify_raw(
            {
                "analysis_plan": plan,
                "candidates": _strong_candidates([insight]),
                "insights": [insight],
            }
        )


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


def test_current_contract_requires_typed_diagnostics_for_high_confidence_insight() -> None:
    with pytest.raises(AssertionError, match="current contract.*diagnostic"):
        _verify(
            {
                "contract_version": 2,
                "insights": [_strong_insight()],
            }
        )


def test_legacy_contract_omission_remains_compatible() -> None:
    _verify({"insights": [_strong_insight()]})


def test_verifier_rejects_unsupported_contract_version() -> None:
    with pytest.raises(AssertionError, match="contract_version is unsupported"):
        _verify(
            {
                "contract_version": 99,
                "insights": [_strong_insight()],
            }
        )


def test_current_contract_accepts_complete_typed_diagnostics() -> None:
    insight = _strong_insight()
    insight["diagnostic_measurability"] = "mixed"
    insight["diagnostic_assessment"] = _diagnostic_assessment(
        measurable=True,
        disposition="ruled_out",
        decision_readiness="act_ready",
    )
    insight["action"]["kind"] = "program"

    _verify({"contract_version": 2, "insights": [insight]})


def _add_v3_closure_fields(assessment: dict, status: str = "ruled_out") -> None:
    measurable = assessment["explanations"][0]
    measurable.update(
        {
            "explanation_id": "cohort-mix",
            "closure_status": status,
            "required_check": "Compare cohort composition across periods.",
        }
    )
    non_measurable = assessment["explanations"][1]
    non_measurable.update(
        {
            "explanation_id": "instrumentation-coverage",
            "closure_status": "unresolvable",
        }
    )


def test_evidence_closure_contract_accepts_resolved_explanations() -> None:
    insight = _strong_insight()
    insight["diagnostic_measurability"] = "mixed"
    assessment = _diagnostic_assessment(
        measurable=True,
        disposition="ruled_out",
        decision_readiness="act_ready",
    )
    _add_v3_closure_fields(assessment)
    insight["diagnostic_assessment"] = assessment
    insight["action"]["kind"] = "program"

    _verify({"contract_version": 3, "insights": [insight]})


def test_evidence_closure_contract_requires_stable_explanation_ids() -> None:
    insight = _strong_insight()
    insight["confidence"]["level"] = "medium"
    insight["diagnostic_measurability"] = "mixed"
    assessment = _diagnostic_assessment(
        measurable=True,
        disposition="unresolved",
        decision_readiness="investigate_first",
    )
    _add_v3_closure_fields(assessment, status="pending")
    del assessment["explanations"][0]["explanation_id"]
    insight["diagnostic_assessment"] = assessment
    insight["action"]["kind"] = "diagnostic"

    with pytest.raises(AssertionError, match="explanation_id"):
        _verify({"contract_version": 3, "insights": [insight]})


def test_evidence_closure_pending_or_supported_explanations_gate_action() -> None:
    for status in ("pending", "supported"):
        insight = _strong_insight()
        insight["confidence"]["level"] = "medium"
        insight["diagnostic_measurability"] = "mixed"
        assessment = _diagnostic_assessment(
            measurable=True,
            disposition="unresolved" if status == "pending" else "supported",
            decision_readiness="act_ready",
        )
        _add_v3_closure_fields(assessment, status=status)
        if status == "supported":
            assessment["explanations"][0]["expected_value"] = 0.25
            assessment["explanations"][0]["verification"] = _metric_component(
                "mix_shift",
                "value",
                0.25,
            )["verification"]
        insight["diagnostic_assessment"] = assessment
        insight["action"]["kind"] = "program"

        with pytest.raises(AssertionError, match="requires investigate_first"):
            _verify({"contract_version": 3, "insights": [insight]})


def test_verifier_accepts_one_aggregate_over_row_derived_source_values() -> None:
    insight = _strong_insight()
    insight["diagnostic_measurability"] = "mixed"
    assessment = _diagnostic_assessment(
        measurable=True,
        disposition="weakened",
        decision_readiness="act_ready",
    )
    assessment["explanations"][0]["verification"] = {
        "method": "sql",
        "expression": (
            "SELECT ROUND(AVG(DATE_DIFF('day', o.promised_at, "
            "o.delivered_at)), 2) AS metric_value "
            "FROM orders o JOIN reviews r ON r.order_id = o.order_id"
        ),
        "sources": {"orders": "orders", "reviews": "reviews"},
    }
    insight["diagnostic_assessment"] = assessment
    insight["action"]["kind"] = "program"

    _verify({"contract_version": 2, "insights": [insight]})


def test_verifier_rejects_metric_arithmetic_across_multiple_aggregates() -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        "SELECT SUM(retained_90d) / COUNT(customer_id) AS metric_value "
        "FROM customer_cohorts"
    )

    with pytest.raises(AssertionError, match="derived metrics as components"):
        _verify({"insights": [insight]})


def test_verifier_accepts_ruled_out_diagnostic_with_python_verification() -> None:
    insight = _strong_insight()
    insight["diagnostic_measurability"] = "mixed"
    insight["diagnostic_assessment"] = _diagnostic_assessment(
        measurable=True,
        disposition="ruled_out",
        decision_readiness="act_ready",
        method="python",
    )
    insight["action"]["kind"] = "program"

    _verify({"insights": [insight]})


def test_verifier_accepts_unresolved_measurable_investigation_gate() -> None:
    insight = _strong_insight()
    insight["confidence"]["level"] = "medium"
    insight["diagnostic_measurability"] = "mixed"
    insight["diagnostic_assessment"] = _diagnostic_assessment(
        measurable=True,
        disposition="unresolved",
        decision_readiness="investigate_first",
    )
    insight["action"]["kind"] = "diagnostic"
    insight["action"]["decision"] = "Test cohort mix and instrumentation coverage"

    _verify({"insights": [insight]})


def test_verifier_accepts_not_measurable_diagnostic_with_limitation() -> None:
    insight = _strong_insight()
    insight["diagnostic_measurability"] = "not_measurable"
    assessment = _diagnostic_assessment(
        measurable=False,
        disposition="not_measurable",
        decision_readiness="act_ready",
    )
    assessment["explanations"][0]["limitation"] = (
        "Historical cohort-composition fields are unavailable."
    )
    insight["diagnostic_assessment"] = assessment
    insight["action"]["kind"] = "program"

    _verify({"insights": [insight]})


@pytest.mark.parametrize("field", ["confidence", "urgency"])
def test_verifier_rejects_high_or_critical_with_unresolved_measurable_alternative(
    field: str,
) -> None:
    insight = _strong_insight()
    insight["confidence"]["level"] = "medium"
    insight["diagnostic_measurability"] = "mixed"
    insight["diagnostic_assessment"] = _diagnostic_assessment(
        measurable=True,
        disposition="unresolved",
        decision_readiness="investigate_first",
    )
    insight["action"]["kind"] = "diagnostic"
    if field == "confidence":
        insight["confidence"]["level"] = "high"
    else:
        insight["priority"]["urgency"] = "critical"

    with pytest.raises(AssertionError, match="unresolved measurable"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize("disposition", ["ruled_out", "weakened"])
def test_verifier_requires_verification_for_tested_diagnostic(
    disposition: str,
) -> None:
    insight = _strong_insight()
    insight["diagnostic_measurability"] = "mixed"
    assessment = _diagnostic_assessment(
        measurable=True,
        disposition=disposition,
        decision_readiness="act_ready",
    )
    del assessment["explanations"][0]["verification"]
    insight["diagnostic_assessment"] = assessment
    insight["action"]["kind"] = "program"

    with pytest.raises(AssertionError, match="verification"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize(
    ("measurable", "disposition"),
    [(False, "unresolved"), (True, "not_measurable")],
)
def test_verifier_rejects_not_measurable_disposition_inconsistency(
    measurable: bool,
    disposition: str,
) -> None:
    insight = _strong_insight()
    insight["diagnostic_measurability"] = "mixed"
    assessment = _diagnostic_assessment(
        measurable=measurable,
        disposition=disposition,
        decision_readiness="act_ready",
    )
    insight["diagnostic_assessment"] = assessment
    insight["action"]["kind"] = "program"

    with pytest.raises(AssertionError, match="measurable|not_measurable"):
        _verify({"insights": [insight]})


def test_verifier_rejects_investigate_first_with_program_action() -> None:
    insight = _strong_insight()
    insight["confidence"]["level"] = "medium"
    insight["diagnostic_measurability"] = "mixed"
    insight["diagnostic_assessment"] = _diagnostic_assessment(
        measurable=True,
        disposition="unresolved",
        decision_readiness="investigate_first",
    )
    insight["action"]["kind"] = "program"

    with pytest.raises(AssertionError, match="diagnostic action"):
        _verify({"insights": [insight]})


def test_verifier_requires_typed_diagnostics_when_measurability_is_declared() -> None:
    insight = _strong_insight()
    insight["diagnostic_measurability"] = "measurable"

    with pytest.raises(AssertionError, match="diagnostic_assessment"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize("metric_type", ["value", "count", "amount", "average"])
def test_verifier_accepts_first_class_simple_metric_spec(metric_type: str) -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        metric_type,
        78,
        [_metric_component("observed_value", "value", 78)],
    )
    if metric_type == "count":
        insight["metric_spec"]["comparison"] = {"kind": "none"}

    _verify({"insights": [insight]})


def test_verifier_requires_structured_comparison_metadata_for_count_spec() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "count",
        78,
        [_metric_component("observed_value", "value", 78)],
    )

    with pytest.raises(AssertionError, match="count comparison metadata"):
        _verify({"insights": [insight]})


def test_verifier_accepts_source_agnostic_python_rate_components() -> None:
    insight = _strong_insight()
    del insight["verification"]
    insight["metric_spec"] = _derived_metric_spec(
        "rate",
        837 / 2491,
        [
            _metric_component("low_csat_tickets", "numerator", 837, method="python"),
            _metric_component("all_tickets", "denominator", 2491, method="python"),
        ],
    )

    _verify({"insights": [insight]})


def test_verifier_recomputes_supporting_claim_metric_spec() -> None:
    insight = _strong_insight()
    insight["supporting_claims"] = [
        {
            "claim": "Low-CSAT tickets were 33.6% of all tickets.",
            "expected_value": 837 / 2491,
            "metric_spec": _derived_metric_spec(
                "rate",
                837 / 2491,
                [
                    _metric_component("low_csat_tickets", "numerator", 837),
                    _metric_component("all_tickets", "denominator", 2491),
                ],
            ),
        }
    ]

    _verify({"insights": [insight]})


@pytest.mark.parametrize("invalid_value", [True, float("inf")])
def test_verifier_rejects_invalid_component_expected_value(
    invalid_value: float,
) -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "rate",
        0.25,
        [
            _metric_component("part", "numerator", invalid_value),
            _metric_component("whole", "denominator", 100),
        ],
    )

    with pytest.raises(AssertionError, match="finite numeric"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize(
    ("metric_type", "expected_value", "components"),
    [
        (
            "share",
            25 / 100,
            [
                _metric_component("part", "numerator", 25),
                _metric_component("whole", "denominator", 100),
            ],
        ),
        (
            "delta",
            13,
            [
                _metric_component("current", "current", 78),
                _metric_component("comparison", "comparison", 65),
            ],
        ),
        (
            "rate_of_change",
            0.20,
            [
                _metric_component("current", "current", 120),
                _metric_component("comparison", "comparison", 100),
            ],
        ),
    ],
)
def test_verifier_recomputes_supported_derived_metric_types(
    metric_type: str,
    expected_value: float,
    components: list[dict],
) -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        metric_type, expected_value, components
    )

    _verify({"insights": [insight]})


def _correlation_metric_spec(expected_value: float = 1.0) -> dict:
    variables = {
        "x": "properties.base_price",
        "y": "reviews.average_rating",
    }
    population = "Properties with a non-null price and average rating"
    spec = _derived_metric_spec(
        "correlation",
        expected_value,
        [
            _metric_component("pair_count", "pair_count", 3),
            _metric_component("sum_price", "sum_x", 6),
            _metric_component("sum_rating", "sum_y", 12),
            _metric_component("sum_price_squared", "sum_x_squared", 14),
            _metric_component("sum_rating_squared", "sum_y_squared", 56),
            _metric_component("sum_price_rating", "sum_xy", 28),
        ],
    )
    for component in spec["components"]:
        component["variables"] = variables
        component["population"] = population
        component["pairwise_missing_policy"] = "complete_cases"
    spec["variables"] = variables
    spec["population"] = population
    spec["pairwise_missing_policy"] = "complete_cases"
    return spec


def test_verifier_recomputes_correlation_from_sufficient_statistics() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _correlation_metric_spec()

    _verify({"insights": [insight]})


def test_verifier_rejects_model_supplied_correlation_value() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _correlation_metric_spec(-0.00125)

    with pytest.raises(AssertionError, match="does not reconcile"):
        _verify({"insights": [insight]})


def test_verifier_requires_correlation_population_and_missing_policy() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _correlation_metric_spec()
    del insight["metric_spec"]["pairwise_missing_policy"]

    with pytest.raises(AssertionError, match="complete-case"):
        _verify({"insights": [insight]})


def test_verifier_rejects_correlation_with_zero_variance() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _correlation_metric_spec()
    for component in insight["metric_spec"]["components"]:
        if component["role"] == "sum_x_squared":
            component["expected_value"] = 12

    with pytest.raises(AssertionError, match="positive variance"):
        _verify({"insights": [insight]})


def test_verifier_accepts_large_offset_correlation_statistics() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _correlation_metric_spec()
    offset = 10**12
    values = {
        "pair_count": 3,
        "sum_x": 3 * offset + 6,
        "sum_y": 6 * offset + 12,
        "sum_x_squared": 3 * offset**2 + 12 * offset + 14,
        "sum_y_squared": 12 * offset**2 + 48 * offset + 56,
        "sum_xy": 6 * offset**2 + 24 * offset + 28,
    }
    for component in insight["metric_spec"]["components"]:
        component["expected_value"] = values[component["role"]]
    insight["metric_spec"]["expected_value"] = 1.0

    _verify({"insights": [insight]})


def test_verifier_rejects_inconsistent_correlation_component_population() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _correlation_metric_spec()
    insight["metric_spec"]["components"][0]["population"] = "All properties"

    with pytest.raises(AssertionError, match="same population"):
        _verify({"insights": [insight]})


def test_verifier_rejects_inconsistent_correlation_component_variables() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _correlation_metric_spec()
    insight["metric_spec"]["components"][-1]["variables"] = {
        "x": "properties.base_price",
        "y": "reviews.review_id",
    }

    with pytest.raises(AssertionError, match="same variables"):
        _verify({"insights": [insight]})


def test_verifier_rejects_zero_denominator_for_derived_metric() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "rate",
        0,
        [
            _metric_component("part", "numerator", 25),
            _metric_component("whole", "denominator", 0),
        ],
    )

    with pytest.raises(AssertionError, match="zero denominator"):
        _verify({"insights": [insight]})


def test_verifier_rejects_mismatched_derived_arithmetic() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "delta",
        14,
        [
            _metric_component("current", "current", 78),
            _metric_component("comparison", "comparison", 65),
        ],
    )

    with pytest.raises(AssertionError, match="does not reconcile"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize("invalid_value", [True, float("inf"), float("-inf"), float("nan")])
def test_verifier_rejects_bool_and_nonfinite_metric_values(
    invalid_value: float,
) -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "rate",
        invalid_value,
        [
            _metric_component("part", "numerator", 25),
            _metric_component("whole", "denominator", 100),
        ],
    )

    with pytest.raises(AssertionError, match="finite numeric"):
        _verify({"insights": [insight]})


def test_verifier_rejects_cross_period_count_without_denominator_integrity() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "count",
        837,
        [
            _metric_component("current_count", "current", 837),
            _metric_component("comparison_count", "comparison", 424),
        ],
    )
    insight["metric_spec"]["comparison"] = {"kind": "cross_period"}

    with pytest.raises(AssertionError, match="denominator integrity"):
        _verify({"insights": [insight]})


def test_verifier_accepts_cross_period_count_with_denominators_and_rates() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "count",
        837,
        [
            _metric_component("current_count", "current", 837),
            _metric_component("comparison_count", "comparison", 424),
            _metric_component("current_total", "current_denominator", 2491),
            _metric_component("comparison_total", "comparison_denominator", 1256),
        ],
    )
    insight["metric_spec"]["comparison"] = {
        "kind": "cross_period",
        "population": "variable",
        "current_rate": 837 / 2491,
        "comparison_rate": 424 / 1256,
    }

    _verify({"insights": [insight]})


def test_verifier_rejects_cross_period_count_with_wrong_rate() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "count",
        837,
        [
            _metric_component("current_count", "current", 837),
            _metric_component("comparison_count", "comparison", 424),
            _metric_component("current_total", "current_denominator", 2491),
            _metric_component("comparison_total", "comparison_denominator", 1256),
        ],
    )
    insight["metric_spec"]["comparison"] = {
        "kind": "cross_period",
        "population": "variable",
        "current_rate": 837 / 2491,
        "comparison_rate": 0.50,
    }

    with pytest.raises(AssertionError, match="comparison rate"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize("population", ["stable", "exhaustive"])
def test_verifier_accepts_equal_verified_stable_denominators(
    population: str,
) -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "count",
        837,
        [
            _metric_component("current_count", "current", 837),
            _metric_component("comparison_count", "comparison", 424),
            _metric_component("current_total", "current_denominator", 2491),
            _metric_component("comparison_total", "comparison_denominator", 2491),
        ],
    )
    insight["metric_spec"]["comparison"] = {
        "kind": "cross_period",
        "population": population,
    }

    _verify({"insights": [insight]})


def test_verifier_rejects_model_supplied_stable_flag() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "count",
        837,
        [
            _metric_component("current_count", "current", 837),
            _metric_component("comparison_count", "comparison", 424),
            _metric_component("stable_check", "denominator_stable", 1),
        ],
    )
    insight["metric_spec"]["comparison"] = {
        "kind": "cross_period",
        "population": "stable",
    }

    with pytest.raises(AssertionError, match="current_denominator"):
        _verify({"insights": [insight]})


def test_verifier_rejects_unequal_stable_denominators() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "count",
        837,
        [
            _metric_component("current_count", "current", 837),
            _metric_component("comparison_count", "comparison", 424),
            _metric_component("current_total", "current_denominator", 2491),
            _metric_component("comparison_total", "comparison_denominator", 1256),
        ],
    )
    insight["metric_spec"]["comparison"] = {
        "kind": "cross_period",
        "population": "stable",
    }

    with pytest.raises(AssertionError, match="denominators must be equal"):
        _verify({"insights": [insight]})


def test_verifier_reconciles_decomposition_and_explicit_residual() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "decomposition",
        13,
        [
            _metric_component("total_delta", "total_delta", 13),
            _metric_component("volume_effect", "contribution", 8),
            _metric_component("mix_effect", "contribution", 4),
            _metric_component("residual", "residual", 1),
        ],
    )

    _verify({"insights": [insight]})


def test_verifier_rejects_decomposition_without_explicit_residual() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "decomposition",
        13,
        [
            _metric_component("total_delta", "total_delta", 13),
            _metric_component("volume_effect", "contribution", 8),
            _metric_component("mix_effect", "contribution", 4),
        ],
    )

    with pytest.raises(AssertionError, match="residual"):
        _verify({"insights": [insight]})


def test_verifier_rejects_unreconciled_decomposition() -> None:
    insight = _strong_insight()
    insight["metric_spec"] = _derived_metric_spec(
        "decomposition",
        13,
        [
            _metric_component("total_delta", "total_delta", 13),
            _metric_component("volume_effect", "contribution", 8),
            _metric_component("mix_effect", "contribution", 4),
            _metric_component("residual", "residual", 0),
        ],
    )

    with pytest.raises(AssertionError, match="decomposition"):
        _verify({"insights": [insight]})


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


def test_verifier_accepts_cte_aggregate_over_declared_source() -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        "WITH segment_totals AS ("
        "SELECT segment, SUM(retained_90d) AS retained_total "
        "FROM customer_cohorts GROUP BY segment"
        ") "
        "SELECT COUNT(*) AS metric_value FROM segment_totals "
        "WHERE retained_total > 10"
    )

    _verify({"insights": [insight]})


def test_verifier_accepts_cast_count_over_source_derived_subquery() -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        "SELECT COUNT(*)::DOUBLE AS metric_value FROM ("
        "SELECT customer_id FROM customer_cohorts "
        "GROUP BY customer_id HAVING SUM(retained_90d) > 10"
        ")"
    )

    _verify({"insights": [insight]})


def test_verifier_accepts_quantile_over_source_derived_expression() -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        "SELECT quantile_cont(retained_90d - activated_30d, 0.5) "
        "AS metric_value FROM customer_cohorts"
    )

    _verify({"insights": [insight]})


def test_verifier_rejects_quantile_over_constants() -> None:
    insight = _strong_insight()
    insight["verification"]["expression"] = (
        "SELECT quantile_cont(10, 0.5) AS metric_value FROM customer_cohorts"
    )

    with pytest.raises(AssertionError, match="recompute metric_value"):
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


def test_verifier_accepts_nested_dax_metric_from_declared_measure() -> None:
    insight = _strong_insight()
    insight["verification"] = {
        "method": "dax",
        "expression": """
EVALUATE
ROW(
    "metric_value",
    DIVIDE(
        CALCULATE(
            AVERAGEX(VALUES('Period'[MonthNumber]), [ARR $]),
            'Period'[Year] = 2030
        ),
        CALCULATE(
            AVERAGEX(VALUES('Period'[MonthNumber]), [ARR $]),
            'Period'[Year] = 2026
        )
    )
)
""",
        "sources": {
            "arr": "[ARR $]",
            "period": "Period",
        },
    }

    _verify({"insights": [insight]})


def test_verifier_rejects_nested_dax_metric_from_undeclared_measure() -> None:
    insight = _strong_insight()
    insight["verification"] = {
        "method": "dax",
        "expression": """
EVALUATE
ROW(
    "metric_value",
    DIVIDE(
        CALCULATE([Unrelated Amount], 'Unrelated'[Year] = 2024),
        CALCULATE([Unrelated Count], 'Unrelated'[Year] = 2024)
    )
)
""",
        "sources": {
            "arr": "[ARR $]",
            "active_customers": "[Active Customers #]",
            "period": "Period",
        },
    }

    with pytest.raises(AssertionError, match="declared source"):
        _verify({"insights": [insight]})


def test_verifier_rejects_partial_dax_measure_name_match() -> None:
    insight = _strong_insight()
    insight["verification"] = {
        "method": "dax",
        "expression": 'EVALUATE ROW("metric_value", [ARR Growth])',
        "sources": {"arr": "[ARR $]"},
    }

    with pytest.raises(AssertionError, match="declared source"):
        _verify({"insights": [insight]})


def test_verifier_rejects_declared_dax_filter_with_unrelated_metric() -> None:
    insight = _strong_insight()
    insight["verification"] = {
        "method": "dax",
        "expression": """
EVALUATE
ROW(
    "metric_value",
    CALCULATE([Unrelated Amount], 'Period'[Year] = 2024)
)
""",
        "sources": {
            "period_year": "Period[Year]",
            "arr": "[ARR $]",
        },
    }

    with pytest.raises(AssertionError, match="declared source"):
        _verify({"insights": [insight]})


def test_verifier_rejects_undeclared_dax_measure_hidden_behind_var() -> None:
    insight = _strong_insight()
    insight["verification"] = {
        "method": "dax",
        "expression": """
VAR hidden = [Unrelated Amount]
EVALUATE
ROW(
    "metric_value",
    CALCULATE(hidden, 'Period'[Year] = 2024)
)
""",
        "sources": {
            "period_year": "Period[Year]",
            "arr": "[ARR $]",
        },
    }

    with pytest.raises(AssertionError, match="declared source"):
        _verify({"insights": [insight]})


def _temporal_context(status: str = "current_change") -> dict:
    return {
        "time_basis": "customer_cohorts.cohort_quarter",
        "timezone": "UTC",
        "requested_as_of": "2026-08-31",
        "data_as_of": "2026-08-15T23:59:59Z",
        "trustworthy_through": "2026-06-30T23:59:59Z",
        "latest_complete_period": {
            "grain": "quarter",
            "start": "2026-04-01",
            "end": "2026-06-30",
        },
        "current_window": {
            "grain": "quarter",
            "start": "2026-04-01",
            "end": "2026-06-30",
            "periods": ["2026-Q2"],
        },
        "comparators": [
            {
                "kind": "same_period_prior_year",
                "start": "2025-04-01",
                "end": "2025-06-30",
                "periods": ["2025-Q2"],
            }
        ],
        "partial_period_policy": "exclude",
        "completeness_basis": (
            "calendar_complete_and_source_marked_trustworthy"
        ),
        "recency_status": status,
        "supports_current_action": status in {
            "current_change",
            "current_level",
            "persistent",
        },
    }


def test_verifier_requires_temporal_context_for_current_claim_title() -> None:
    insight = _strong_insight()
    insight["title"] = "Current enterprise retention is declining"

    with pytest.raises(AssertionError, match="temporal_context"):
        _verify({"insights": [insight]})


def test_verifier_accepts_current_change_with_comparable_temporal_context() -> None:
    insight = _strong_insight()
    insight["title"] = "Recent enterprise retention decline"
    insight["temporal_context"] = _temporal_context()

    _verify({"insights": [insight]})


def test_verifier_rejects_stale_temporal_context_for_program_action() -> None:
    insight = _strong_insight()
    insight["temporal_context"] = _temporal_context("stale")
    insight["temporal_context"]["supports_current_action"] = False
    insight["action"]["kind"] = "program"

    with pytest.raises(AssertionError, match="stale.*program action"):
        _verify({"insights": [insight]})


def test_verifier_rejects_current_change_without_comparator() -> None:
    insight = _strong_insight()
    insight["temporal_context"] = _temporal_context()
    insight["temporal_context"]["comparators"] = []

    with pytest.raises(AssertionError, match="current_change.*comparator"):
        _verify({"insights": [insight]})


def test_verifier_rejects_unsupported_causal_language() -> None:
    insight = _strong_insight()
    insight["statement"] = "Poor onboarding caused enterprise retention to fall."

    with pytest.raises(AssertionError, match="causal"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize(
    "interpretation",
    [
        "The decline is a leading indicator of revenue softness.",
        "The decline signals churn risk.",
        "The decline implies customers are failing before renewal.",
        "The usage contraction confirms weakening demand.",
        "Usage is flowing into lower invoiced revenue.",
        "Product friction is behind the usage decline.",
    ],
)
def test_verifier_rejects_implied_causal_or_predictive_language(
    interpretation: str,
) -> None:
    insight = _strong_insight()
    insight["interpretation"] = interpretation

    with pytest.raises(AssertionError, match="causal"):
        _verify({"insights": [insight]})


def test_verifier_rejects_multiple_claims_packed_into_primary_statement() -> None:
    insight = _strong_insight()
    insight["statement"] = (
        "Enterprise retention fell from 91% to 78%; "
        "quarterly active users fell from 4,105 to 2,156."
    )

    with pytest.raises(AssertionError, match="one primary"):
        _verify({"insights": [insight]})


def test_verifier_rejects_comma_while_packed_primary_claim() -> None:
    insight = _strong_insight()
    insight["statement"] = (
        "Enterprise retention fell from 91% to 78%, while overall retention "
        "remained near 86%."
    )

    with pytest.raises(AssertionError, match="one primary"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize("connector", ["as", "with", "alongside"])
def test_verifier_rejects_other_packed_claim_connectors(
    connector: str,
) -> None:
    insight = _strong_insight()
    insight["statement"] = (
        f"Enterprise retention fell to 78% {connector} active users dropped "
        "to 2,156."
    )

    with pytest.raises(AssertionError, match="one primary"):
        _verify({"insights": [insight]})


def test_verifier_rejects_quantitative_fact_hidden_in_interpretation() -> None:
    insight = _strong_insight()
    insight["interpretation"] = (
        "Quarterly active users fell from 4,105 to 2,156, increasing renewal risk."
    )

    with pytest.raises(AssertionError, match="supporting_claims"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize(
    "interpretation",
    [
        "Half the recurring-revenue base sits in roughly two dozen accounts.",
        "One in three customers reported a poor experience.",
        "A quarter of orders account for most delayed deliveries.",
    ],
)
def test_verifier_rejects_word_number_fact_hidden_in_interpretation(
    interpretation: str,
) -> None:
    insight = _strong_insight()
    insight["interpretation"] = interpretation

    with pytest.raises(AssertionError, match="supporting_claims"):
        _verify({"insights": [insight]})


@pytest.mark.parametrize(
    "interpretation",
    [
        "The 2024 enterprise cohort is the most exposed segment.",
        "The Q4 enterprise cohort is the most exposed segment.",
    ],
)
def test_verifier_allows_period_labels_in_interpretation(
    interpretation: str,
) -> None:
    insight = _strong_insight()
    insight["interpretation"] = interpretation

    _verify({"insights": [insight]})


def test_verifier_does_not_mistake_four_digit_count_for_year() -> None:
    insight = _strong_insight()
    insight["interpretation"] = (
        "The affected population includes 2024 enterprise accounts."
    )

    with pytest.raises(AssertionError, match="supporting_claims"):
        _verify({"insights": [insight]})


def test_verifier_accepts_independently_verified_supporting_claim() -> None:
    insight = _strong_insight()
    insight["supporting_claims"] = [
        {
            "claim": "Quarterly active users fell from 4,105 to 2,156.",
            "expected_value": 2156,
            "verification": {
                "method": "sql",
                "expression": (
                    "SELECT COUNT(DISTINCT user_id) AS metric_value "
                    "FROM customer_cohorts WHERE cohort_quarter = '2024-Q4'"
                ),
                "sources": {
                    "customer_cohorts": "analytics.customer_cohorts"
                },
            },
        }
    ]

    _verify({"insights": [insight]})


def test_verifier_rejects_unverified_supporting_claim() -> None:
    insight = _strong_insight()
    insight["supporting_claims"] = [
        {
            "claim": "Quarterly active users fell from 4,105 to 2,156.",
            "expected_value": 2156,
            "verification": {
                "method": "sql",
                "expression": (
                    "SELECT 2156 AS metric_value FROM customer_cohorts"
                ),
                "sources": {
                    "customer_cohorts": "analytics.customer_cohorts"
                },
            },
        }
    ]

    with pytest.raises(AssertionError, match="recompute"):
        _verify({"insights": [insight]})


def test_verifier_requires_supporting_claim_expected_value() -> None:
    insight = _strong_insight()
    insight["supporting_claims"] = [
        {
            "claim": "Quarterly active users fell to 2,156.",
            "verification": {
                "method": "sql",
                "expression": (
                    "SELECT COUNT(DISTINCT user_id) AS metric_value "
                    "FROM customer_cohorts WHERE cohort_quarter = '2024-Q4'"
                ),
                "sources": {
                    "customer_cohorts": "analytics.customer_cohorts"
                },
            },
        }
    ]

    with pytest.raises(AssertionError, match="expected_value"):
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
        "supporting_claims",
        "discovery",
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


@pytest.mark.parametrize(
    "discovery",
    [
        {
            "pattern_type": "unknown",
            "dimensions_tested": ["segment"],
            "population": "Customers",
            "sample_size": 10,
            "robustness_checks": ["Compared groups."],
        },
        {
            "pattern_type": "subgroup",
            "dimensions_tested": [],
            "population": "Customers",
            "sample_size": 10,
            "robustness_checks": ["Compared groups."],
        },
        {
            "pattern_type": "subgroup",
            "dimensions_tested": ["segment"],
            "population": "Customers",
            "sample_size": 0,
            "robustness_checks": ["Compared groups."],
        },
        {
            "pattern_type": "subgroup",
            "dimensions_tested": ["segment"],
            "population": "Customers",
            "sample_size": 10,
            "robustness_checks": [],
        },
    ],
)
def test_verifier_requires_discovery_provenance(discovery: dict) -> None:
    insight = _strong_insight()
    insight["discovery"] = discovery

    with pytest.raises(AssertionError, match="discovery"):
        _verify({"insights": [insight]})


def test_verifier_requires_pattern_diversity_for_deep_analysis() -> None:
    insights = []
    for index in range(8):
        insight = _strong_insight()
        insight["title"] = f"Portfolio trend {index}"
        insight["statement"] = f"Metric {index} fell by {index + 1}%."
        insight["priority"]["rank"] = index + 1
        insight["discovery"]["pattern_type"] = "portfolio_trend"
        insights.append(insight)

    with pytest.raises(AssertionError, match="pattern diversity"):
        _verify({"insights": insights})


def test_verifier_rejects_vacuous_confidence_reason() -> None:
    insight = _strong_insight()
    insight["confidence"]["reason"] = "Based on the available data."

    with pytest.raises(AssertionError, match="evidence"):
        _verify({"insights": [insight]})


def test_verifier_accepts_specific_observational_confidence_reason() -> None:
    insight = _strong_insight()
    insight["confidence"]["reason"] = (
        "Twelve months of daily observations across all accounts."
    )

    _verify({"insights": [insight]})


def test_verifier_allows_negated_implied_causal_language() -> None:
    insight = _strong_insight()
    insight["interpretation"] = (
        "The subgroup gap does not signal churn risk."
    )

    _verify({"insights": [insight]})


@pytest.mark.parametrize(
    "interpretation",
    [
        "Implementation delays drove the decline.",
        "Implementation delays triggered the decline.",
        "Retention fell because onboarding weakened.",
        "Implementation delays led to the decline.",
    ],
)
def test_verifier_rejects_common_causal_verbs(
    interpretation: str,
) -> None:
    insight = _strong_insight()
    insight["interpretation"] = interpretation

    with pytest.raises(AssertionError, match="causal"):
        _verify({"insights": [insight]})


def test_verifier_requires_denominator_check_for_cohort_transition() -> None:
    insight = _strong_insight()
    insight["discovery"]["pattern_type"] = "cohort_transition"
    insight["discovery"]["robustness_checks"] = [
        "Compared four activation cohorts.",
        "Excluded incomplete periods.",
    ]

    with pytest.raises(AssertionError, match="denominator"):
        _verify({"insights": [insight]})


def test_verifier_requires_two_dimensions_for_interaction() -> None:
    insight = _strong_insight()
    insight["discovery"]["pattern_type"] = "interaction"
    insight["discovery"]["dimensions_tested"] = ["plan_tier"]
    insight["discovery"]["interaction_evidence"] = {
        "cells": [
            {"cell": "basic|north", "effect": -0.02, "sample_size": 80},
            {"cell": "basic|south", "effect": 0.08, "sample_size": 75},
        ],
        "heterogeneity": "The effect changes sign across regions.",
        "baseline_effect": -0.01,
    }

    with pytest.raises(AssertionError, match="two dimensions"):
        _verify({"insights": [insight]})


def test_verifier_requires_effect_heterogeneity_for_interaction() -> None:
    insight = _strong_insight()
    insight["discovery"]["pattern_type"] = "interaction"

    with pytest.raises(AssertionError, match="effect heterogeneity"):
        _verify({"insights": [insight]})


def test_verifier_rejects_uniform_interaction_cell_effects() -> None:
    insight = _strong_insight()
    insight["discovery"]["pattern_type"] = "interaction"
    insight["discovery"]["interaction_evidence"] = {
        "cells": [
            {"cell": "enterprise|north", "effect": 0.05, "sample_size": 80},
            {"cell": "enterprise|south", "effect": 0.05, "sample_size": 75},
        ],
        "heterogeneity": "Effects were compared across cells.",
        "baseline_effect": 0.05,
    }

    with pytest.raises(AssertionError, match="effects must differ"):
        _verify({"insights": [insight]})


def test_verifier_accepts_genuine_interaction_effect_evidence() -> None:
    insight = _strong_insight()
    insight["discovery"]["pattern_type"] = "interaction"
    insight["discovery"]["interaction_evidence"] = {
        "cells": [
            {"cell": "enterprise|north", "effect": -0.02, "sample_size": 80},
            {"cell": "enterprise|south", "effect": 0.08, "sample_size": 75},
        ],
        "heterogeneity": "The effect changes sign across regions.",
        "baseline_effect": 0.01,
    }

    _verify({"insights": [insight]})


def test_verifier_requires_recognized_evidence_tier() -> None:
    insight = _strong_insight()
    insight["evidence_tier"] = "predictive-ish"

    with pytest.raises(AssertionError, match="evidence tier is invalid"):
        _verify({"insights": [insight]})


def test_verifier_rejects_causal_tier_without_causal_evidence() -> None:
    insight = _strong_insight()
    insight["evidence_tier"] = "causal"

    with pytest.raises(AssertionError, match="causal evidence"):
        _verify({"insights": [insight]})


def test_verifier_rejects_overclaiming_language_below_causal_tier() -> None:
    insight = _strong_insight()
    insight["interpretation"] = (
        "This conclusively proves that implementation quality is responsible."
    )

    with pytest.raises(AssertionError, match="evidence tier"):
        _verify({"insights": [insight]})


def test_verifier_accepts_descriptive_evidence_tier() -> None:
    insight = _strong_insight()
    insight["evidence_tier"] = "descriptive"
    insight["interpretation"] = (
        "The measured subgroup differs from the aggregate during the period."
    )

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
