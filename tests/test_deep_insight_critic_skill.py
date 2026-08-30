"""Contract tests for the adversarial deep-insight critic skill."""

from __future__ import annotations

import copy

import pytest

from fabric_rlm.skill_loader import SkillLoader
from fabric_rlm.skill_router import SkillRouter

SKILL = "deep_insight_critic"
TAXONOMY = [
    "obviousness",
    "cross_domain_depth",
    "contradiction",
    "denominator_integrity",
    "metric_definition",
    "alternative_explanation",
    "target_basis",
    "benchmark_basis",
    "causal_overclaim",
    "grain_or_join",
    "headline_consistency",
    "actionability",
]


def _verify(payload: dict) -> None:
    source = SkillLoader().load(SKILL).verifier_source
    assert source is not None
    namespace: dict = {}
    exec(compile(source, "<deep_insight_critic verifier>", "exec"), namespace)
    namespace["verify"](payload)


def _check(category: str) -> dict:
    return {
        "type": category,
        "status": "tested",
        "rationale": f"Independently tested {category} against the source ledger.",
        "evidence_refs": [f"notebook://critic/checks/{category}"],
    }


def _valid_payload() -> dict:
    return {
        "critic_version": 1,
        "source_contract_version": 2,
        "source_fingerprint": "sha256:6d22b45a",
        "source_inventory": [
            {
                "title": "Retention weakened in enterprise",
                "rank": 1,
                "action_kind": "program",
                "decision_readiness": "act_ready",
            },
            {
                "title": "Low-CSAT volume increased",
                "rank": 2,
                "action_kind": "program",
                "decision_readiness": "act_ready",
            },
        ],
        "reviewed_insights": [
            {
                "title": "Retention weakened in enterprise",
                "rank": 1,
                "verdict": "approve",
                "decision_effect": (
                    "Preserve the enterprise-retention intervention decision."
                ),
                "challenges": [
                    {
                        "id": "insight-1-denominator",
                        "type": "denominator_integrity",
                        "assessment": (
                            "Cohort denominators reconcile across both periods."
                        ),
                        "severity": "material",
                        "evidence_refs": [
                            "discovery.insights[0].metric_spec.components",
                            "python://critic/recompute_retention_denominators",
                        ],
                    }
                ],
                "required_changes": [],
                "synthesis_eligible": True,
                "resolutions": [
                    {
                        "challenge_index": 0,
                        "challenge_type": "denominator_integrity",
                        "status": "resolved",
                        "rationale": (
                            "Independent denominator recomputation matched."
                        ),
                        "evidence_refs": [
                            "python://critic/recompute_retention_denominators"
                        ],
                    }
                ],
            },
            {
                "title": "Low-CSAT volume increased",
                "rank": 2,
                "verdict": "revise",
                "decision_effect": (
                    "Replace the rollout recommendation with instrumentation diagnosis."
                ),
                "challenges": [
                    {
                        "id": "insight-2-denominator",
                        "type": "denominator_integrity",
                        "assessment": (
                            "The count rose while the low-CSAT share remained flat."
                        ),
                        "severity": "blocking",
                        "evidence_refs": [
                            "discovery.insights[1].metric_spec",
                            "csv://independent_checks/csat_rates.csv#row=4",
                        ],
                    }
                ],
                "required_changes": [
                    {
                        "change": (
                            "Reframe as diagnostic-only until response-volume mix "
                            "is reconciled."
                        ),
                        "gate": "investigate_first",
                    }
                ],
                "synthesis_eligible": True,
                "resolutions": [
                    {
                        "challenge_index": 0,
                        "challenge_type": "denominator_integrity",
                        "status": "gated",
                        "rationale": (
                            "The count cannot support a program action before rate "
                            "and response-mix diagnostics complete."
                        ),
                        "evidence_refs": [
                            "csv://independent_checks/csat_rates.csv#row=4"
                        ],
                    }
                ],
            },
        ],
        "portfolio_challenges": [
            {
                "id": "portfolio-count-rate-tension",
                "type": "cross_insight_tension",
                "assessment": (
                    "The retention rate and CSAT count use different populations, "
                    "so their apparent direction cannot be combined."
                ),
                "severity": "material",
                "evidence_refs": [
                    "discovery.insights[0].discovery.population",
                    "discovery.insights[1].discovery.population",
                ],
                "affected_insight_titles": [
                    "Retention weakened in enterprise",
                    "Low-CSAT volume increased",
                ],
            }
        ],
        "checks_performed": [_check(category) for category in TAXONOMY],
        "synthesis_manifest": {
            "approved": ["Retention weakened in enterprise"],
            "revised": ["Low-CSAT volume increased"],
            "rejected": [],
            "program_action_titles": ["Retention weakened in enterprise"],
            "diagnostic_only_titles": ["Low-CSAT volume increased"],
        },
        "quality_summary": {
            "process_rigor": 8.5,
            "analytical_depth": 7,
            "decision_quality": 8,
            "overall_assessment": (
                "One finding can drive action; one requires denominator diagnosis."
            ),
            "blocking_issues": [
                {
                    "challenge_id": "insight-2-denominator",
                    "summary": (
                        "Low-CSAT count cannot justify a program action while its "
                        "share is flat."
                    ),
                    "evidence_refs": [
                        "csv://independent_checks/csat_rates.csv#row=4"
                    ],
                }
            ],
        },
    }


def test_skill_is_packaged_and_routes_adversarial_review_requests() -> None:
    loader = SkillLoader()
    assert SKILL in loader.list_skills()
    skill = loader.load(SKILL)
    assert skill.title == SKILL
    assert skill.specificity == "domain"
    assert skill.verifier_present
    assert "adversarial" in skill.summary.lower()

    decision = SkillRouter.from_loader(loader).route(
        "Run an adversarial analytics critic review of these insights"
    )
    assert SKILL in decision.active, decision.scores


def test_skill_documents_separate_critic_procedure_and_limits() -> None:
    content = SkillLoader().load(SKILL).content.lower()
    ordered = [
        "inspect the source payload",
        "independently test",
        "challenge obviousness",
        "submit",
    ]
    positions = [content.index(item) for item in ordered]
    assert positions == sorted(positions)
    assert "does not synthesize prose" in content
    assert "cannot repair evidence by assertion" in content
    assert "source_fingerprint" in content


def test_verifier_accepts_source_agnostic_non_sql_evidence() -> None:
    _verify(_valid_payload())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("critic_version", 2),
        ("source_contract_version", 1),
        ("source_fingerprint", ""),
    ],
)
def test_verifier_rejects_unsupported_versions_or_missing_binding(
    field: str, value: object
) -> None:
    payload = _valid_payload()
    payload[field] = value
    with pytest.raises(AssertionError):
        _verify(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("source_inventory", 0, "rank"), True),
        (("reviewed_insights", 0, "rank"), True),
        (("quality_summary", "process_rigor"), True),
    ],
)
def test_verifier_rejects_bools_as_ranks_or_scores(
    path: tuple, value: object
) -> None:
    payload = _valid_payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AssertionError):
        _verify(payload)


def test_verifier_requires_exact_exhaustive_inventory_coverage() -> None:
    payload = _valid_payload()
    payload["reviewed_insights"].pop()
    with pytest.raises(AssertionError, match="coverage"):
        _verify(payload)


@pytest.mark.parametrize("collection", ["source_inventory", "reviewed_insights"])
@pytest.mark.parametrize("field", ["title", "rank"])
def test_verifier_rejects_duplicate_titles_or_ranks(
    collection: str, field: str
) -> None:
    payload = _valid_payload()
    payload[collection][1][field] = payload[collection][0][field]
    with pytest.raises(AssertionError):
        _verify(payload)


def test_verifier_rejects_manifest_drift() -> None:
    payload = _valid_payload()
    payload["synthesis_manifest"]["approved"] = []
    with pytest.raises(AssertionError, match="manifest"):
        _verify(payload)


def test_verifier_rejects_default_approval_without_challenge() -> None:
    payload = _valid_payload()
    payload["reviewed_insights"][0]["challenges"] = []
    payload["reviewed_insights"][0]["resolutions"] = []
    with pytest.raises(AssertionError, match="challenge"):
        _verify(payload)


def test_verifier_rejects_approval_with_unresolved_material_challenge() -> None:
    payload = _valid_payload()
    payload["reviewed_insights"][0]["resolutions"] = []

    with pytest.raises(AssertionError, match="approve.*material"):
        _verify(payload)


@pytest.mark.parametrize(
    "text", ["", "TBD", "unknown", "none", "n/a", "not assessed", "no issues", "looks good"]
)
def test_verifier_rejects_empty_or_boilerplate_assessments(text: str) -> None:
    payload = _valid_payload()
    payload["reviewed_insights"][0]["challenges"][0]["assessment"] = text
    with pytest.raises(AssertionError):
        _verify(payload)


def test_verifier_rejects_unsupported_taxonomy() -> None:
    payload = _valid_payload()
    payload["reviewed_insights"][0]["challenges"][0]["type"] = "style"
    with pytest.raises(AssertionError, match="type"):
        _verify(payload)


def test_verifier_rejects_duplicate_evidence_refs() -> None:
    payload = _valid_payload()
    refs = payload["reviewed_insights"][0]["challenges"][0]["evidence_refs"]
    refs.append(refs[0])
    with pytest.raises(AssertionError, match="evidence"):
        _verify(payload)


def test_verifier_rejects_unresolved_blocking_eligibility() -> None:
    payload = _valid_payload()
    payload["reviewed_insights"][1]["resolutions"] = []
    payload["reviewed_insights"][1]["required_changes"][0]["gate"] = "none"
    with pytest.raises(AssertionError, match="blocking"):
        _verify(payload)


def test_verifier_rejects_blocking_challenge_downgrade() -> None:
    payload = _valid_payload()
    payload["reviewed_insights"][1]["resolutions"][0]["status"] = "downgraded"
    with pytest.raises(AssertionError, match="downgrad"):
        _verify(payload)


def test_verifier_rejects_duplicate_or_mismatched_resolutions() -> None:
    payload = _valid_payload()
    resolution = payload["reviewed_insights"][1]["resolutions"][0]
    payload["reviewed_insights"][1]["resolutions"].append(copy.deepcopy(resolution))
    with pytest.raises(AssertionError, match="resolution"):
        _verify(payload)

    payload = _valid_payload()
    payload["reviewed_insights"][1]["resolutions"][0]["challenge_index"] = 8
    with pytest.raises(AssertionError, match="resolution"):
        _verify(payload)


def test_verifier_rejects_gated_program_action() -> None:
    payload = _valid_payload()
    manifest = payload["synthesis_manifest"]
    manifest["diagnostic_only_titles"] = []
    manifest["program_action_titles"].append("Low-CSAT volume increased")
    with pytest.raises(AssertionError, match="gated"):
        _verify(payload)


def test_required_change_gate_also_forces_diagnostic_only() -> None:
    payload = _valid_payload()
    payload["reviewed_insights"][1]["resolutions"] = []
    manifest = payload["synthesis_manifest"]
    manifest["diagnostic_only_titles"] = []
    manifest["program_action_titles"].append("Low-CSAT volume increased")
    with pytest.raises(AssertionError, match="gated"):
        _verify(payload)


def test_verifier_enforces_source_action_kind_and_readiness() -> None:
    payload = _valid_payload()
    payload["source_inventory"][0]["action_kind"] = "diagnostic"
    with pytest.raises(AssertionError, match="action"):
        _verify(payload)

    payload = _valid_payload()
    payload["source_inventory"][0]["decision_readiness"] = "investigate_first"
    with pytest.raises(AssertionError, match="action"):
        _verify(payload)


def test_verifier_allows_all_insights_to_be_rejected() -> None:
    payload = _valid_payload()
    for insight in payload["reviewed_insights"]:
        insight["verdict"] = "reject"
        insight["decision_effect"] = "No decision survives the evidence challenge."
        insight["synthesis_eligible"] = False
        insight["required_changes"] = [
            {
                "change": "Reject because the asserted decision has no stable basis.",
                "gate": "none",
            }
        ]
        insight["resolutions"] = []
    payload["synthesis_manifest"] = {
        "approved": [],
        "revised": [],
        "rejected": [
            "Retention weakened in enterprise",
            "Low-CSAT volume increased",
        ],
        "program_action_titles": [],
        "diagnostic_only_titles": [],
    }
    payload["quality_summary"]["blocking_issues"] = [
        {
            "challenge_id": "insight-2-denominator",
            "summary": "The rejected CSAT finding retains a blocking denominator defect.",
            "evidence_refs": ["csv://independent_checks/csat_rates.csv#row=4"],
        }
    ]
    _verify(payload)


def test_verifier_requires_portfolio_challenge_for_multiple_insights() -> None:
    payload = _valid_payload()
    payload["portfolio_challenges"] = []
    with pytest.raises(AssertionError, match="portfolio"):
        _verify(payload)


def test_portfolio_blocking_challenge_prevents_program_action() -> None:
    payload = _valid_payload()
    challenge = payload["portfolio_challenges"][0]
    challenge["severity"] = "blocking"
    payload["quality_summary"]["blocking_issues"].append(
        {
            "challenge_id": challenge["id"],
            "summary": "The cross-insight population conflict blocks program action.",
            "evidence_refs": challenge["evidence_refs"],
        }
    )

    with pytest.raises(AssertionError, match="portfolio blocking"):
        _verify(payload)


def test_verifier_rejects_unknown_affected_title() -> None:
    payload = _valid_payload()
    payload["portfolio_challenges"][0]["affected_insight_titles"] = ["Missing"]
    with pytest.raises(AssertionError, match="affected"):
        _verify(payload)


def test_verifier_rejects_check_coverage_gaps() -> None:
    payload = _valid_payload()
    payload["checks_performed"].pop()
    with pytest.raises(AssertionError, match="checks_performed"):
        _verify(payload)


def test_deferred_material_check_prevents_program_action() -> None:
    payload = _valid_payload()
    check = payload["checks_performed"][0]
    check.update(
        {
            "status": "deferred",
            "severity": "material",
            "affected_insight_titles": ["Retention weakened in enterprise"],
            "rationale": "The baseline decision log is not available to test obviousness.",
            "evidence_refs": ["orchestration://missing/baseline-decision-log"],
        }
    )
    with pytest.raises(AssertionError, match="deferred"):
        _verify(payload)


def test_verifier_accepts_substantive_not_applicable_check() -> None:
    payload = _valid_payload()
    check = payload["checks_performed"][7]
    check.update(
        {
            "status": "not_applicable",
            "rationale": "No external benchmark is asserted by either source insight.",
            "evidence_refs": ["discovery.insights[*].statement"],
        }
    )
    _verify(payload)


@pytest.mark.parametrize(
    "challenge_type",
    [
        "contradiction",
        "denominator_integrity",
        "grain_or_join",
        "headline_consistency",
        "causal_overclaim",
    ],
)
def test_high_risk_challenge_cannot_be_minor(challenge_type: str) -> None:
    payload = _valid_payload()
    challenge = payload["reviewed_insights"][0]["challenges"][0]
    challenge.update({"type": challenge_type, "severity": "minor"})
    payload["reviewed_insights"][0]["resolutions"][0][
        "challenge_type"
    ] = challenge_type
    with pytest.raises(AssertionError, match="minor"):
        _verify(payload)


def test_verifier_requires_exact_blocking_id_coverage() -> None:
    payload = _valid_payload()
    payload["quality_summary"]["blocking_issues"] = []
    with pytest.raises(AssertionError, match="blocking"):
        _verify(payload)


def test_revise_requires_changes_and_reject_cannot_be_eligible() -> None:
    payload = _valid_payload()
    payload["reviewed_insights"][1]["required_changes"] = []
    with pytest.raises(AssertionError, match="required_changes"):
        _verify(payload)

    payload = _valid_payload()
    payload["reviewed_insights"][1]["verdict"] = "reject"
    with pytest.raises(AssertionError, match="reject"):
        _verify(payload)
