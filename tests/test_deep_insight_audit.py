"""Tests for internal deep-insight host audit traversal."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fabric_rlm._deep_insight_audit import (
    DeepInsightAuditError,
    audit_deep_insight,
)


def _verification(name: str) -> dict:
    return {
        "method": "python",
        "expression": f"metric_value = {name}.sum()",
        "sources": {name: name},
    }


def _component(name: str, expected: float) -> dict:
    return {
        "name": name,
        "role": "value",
        "expected_value": expected,
        "verification": _verification(name),
    }


def _mixed_payload() -> dict:
    return {
        "contract_version": 2,
        "insights": [
            {
                "expected_value": 999,
                "verification": _verification("skip_primary_parent"),
                "metric_spec": {
                    "type": "rate",
                    "expected_value": 0.5,
                    "components": [
                        _component("primary_numerator", 5),
                        _component("primary_denominator", 10),
                    ],
                },
                "supporting_claims": [
                    {
                        "expected_value": 999,
                        "verification": _verification("skip_supporting_parent"),
                        "metric_spec": {
                            "type": "delta",
                            "expected_value": 2,
                            "components": [
                                _component("supporting_current", 7),
                                _component("supporting_comparison", 5),
                            ],
                        },
                    },
                    {
                        "expected_value": 3,
                        "verification": _verification("legacy_supporting"),
                    },
                ],
                "diagnostic_assessment": {
                    "explanations": [
                        {
                            "explanation": "Mix changed.",
                            "measurable": True,
                            "disposition": "ruled_out",
                            "expected_value": 0.1,
                            "verification": _verification("diagnostic_ruled_out"),
                        },
                        {
                            "explanation": "Coverage weakened.",
                            "measurable": True,
                            "disposition": "weakened",
                            "expected_value": 0.2,
                            "verification": _verification("diagnostic_weakened"),
                        },
                        {
                            "explanation": "Pricing changed.",
                            "measurable": True,
                            "disposition": "unresolved",
                        },
                        {
                            "explanation": "Demand shifted.",
                            "measurable": True,
                            "disposition": "supported",
                            "expected_value": 0.3,
                            "verification": _verification("diagnostic_supported"),
                        },
                    ]
                },
            },
            {
                "expected_value": 11,
                "verification": _verification("legacy_primary_expected"),
                "supporting_claims": [],
            },
            {
                "current_value": 12,
                "verification": _verification("legacy_primary_current"),
                "supporting_claims": [],
            },
        ],
        "candidates": [
            {
                "rejection_type": "quantitative",
                "rejection_evidence": {
                    "verification": {
                        "method": "sql",
                        "expression": "SELECT AVG(x) AS metric_value FROM source",
                        "sources": {"source": "source"},
                        "components": [
                            {
                                "name": "effect_value",
                                "expected_value": 0.01,
                                "expression": (
                                    "SELECT AVG(effect) AS metric_value FROM source"
                                ),
                                "sources": {"source": "source"},
                            },
                            {
                                "name": "baseline_value",
                                "expected_value": 0.5,
                                "expression": (
                                    "SELECT AVG(baseline) AS metric_value FROM source"
                                ),
                                "sources": {"source": "source"},
                            },
                            {
                                "name": "sample_size",
                                "expected_value": 100,
                                "expression": (
                                    "SELECT COUNT(id) AS metric_value FROM source"
                                ),
                                "sources": {"source": "source"},
                            },
                        ],
                    }
                },
            }
        ],
    }


def test_audit_traverses_every_numeric_check_without_parent_duplicates() -> None:
    seen = []

    def executor(check):
        seen.append(check)
        return [[check.expected]]

    report = audit_deep_insight(_mixed_payload(), executor)

    assert report.total_checks == 13
    assert len(seen) == 13
    assert len({check.path for check in seen}) == 13
    assert not any("skip_" in check.verification["expression"] for check in seen)
    assert {result.path for result in report.checks} == {
        "insights[0].metric_spec.components[0]",
        "insights[0].metric_spec.components[1]",
        "insights[0].supporting_claims[0].metric_spec.components[0]",
        "insights[0].supporting_claims[0].metric_spec.components[1]",
        "insights[0].supporting_claims[1]",
        "insights[0].diagnostic_assessment.explanations[0]",
        "insights[0].diagnostic_assessment.explanations[1]",
        "insights[0].diagnostic_assessment.explanations[3]",
        "insights[1]",
        "insights[2]",
        "candidates[0].rejection_evidence.verification.components[0]",
        "candidates[0].rejection_evidence.verification.components[1]",
        "candidates[0].rejection_evidence.verification.components[2]",
    }


def test_audit_accepts_contract_v3_evidence_closure_payloads() -> None:
    payload = _mixed_payload()
    payload["contract_version"] = 3

    report = audit_deep_insight(payload, lambda check: check.expected)

    assert report.total_checks == 13


def test_audit_mismatch_reports_actionable_path() -> None:
    def executor(check):
        if check.path == "insights[0].metric_spec.components[1]":
            return 9
        return check.expected

    with pytest.raises(
        DeepInsightAuditError,
        match=r"insights\[0\]\.metric_spec\.components\[1\].*expected 10.*actual 9",
    ):
        audit_deep_insight(_mixed_payload(), executor)


def test_audit_mismatch_reports_round_trip_safe_large_values() -> None:
    payload = {
        "contract_version": 2,
        "insights": [
            {
                "expected_value": 1258681.0,
                "verification": {"method": "sql"},
            }
        ],
    }

    with pytest.raises(
        DeepInsightAuditError,
        match=r"expected 1258681\.0, actual 1258680\.4",
    ):
        audit_deep_insight(payload, lambda check: 1258680.4)


def test_audit_stabilizes_binary_aggregate_noise_before_reporting() -> None:
    payload = {
        "contract_version": 2,
        "insights": [
            {
                "expected_value": 1258681.0,
                "verification": {"method": "sql"},
            }
        ],
    }

    with pytest.raises(
        DeepInsightAuditError,
        match=r"expected 1258681\.0, actual 1258681\.34",
    ):
        audit_deep_insight(payload, lambda check: 1258681.3399999682)


@pytest.mark.parametrize(
    "result",
    [
        None,
        True,
        "1",
        [],
        [1, 2],
        [[1], [2]],
        {"first": 1, "second": 2},
    ],
)
def test_audit_rejects_invalid_executor_result_shapes(result) -> None:
    with pytest.raises(
        DeepInsightAuditError,
        match=r"insights\[0\]\.metric_spec\.components\[0\].*one finite numeric scalar",
    ):
        audit_deep_insight(_mixed_payload(), lambda check: result)


@pytest.mark.parametrize("result", [float("nan"), float("inf"), float("-inf")])
def test_audit_rejects_nonfinite_executor_result(result: float) -> None:
    with pytest.raises(DeepInsightAuditError, match="one finite numeric scalar"):
        audit_deep_insight(_mixed_payload(), lambda check: result)


def test_audit_wraps_callback_failure_with_path_and_preserves_cause() -> None:
    failure = LookupError("source unavailable")

    def executor(check):
        raise failure

    with pytest.raises(
        DeepInsightAuditError,
        match=r"executor failed at insights\[0\]\.metric_spec\.components\[0\]",
    ) as captured:
        audit_deep_insight(_mixed_payload(), executor)

    assert captured.value.__cause__ is failure


def test_audit_uses_configurable_tolerance_and_returns_immutable_report() -> None:
    report = audit_deep_insight(
        _mixed_payload(),
        lambda check: check.expected + 0.005,
        rel_tol=0,
        abs_tol=0.01,
    )

    assert report.total_checks == 13
    assert isinstance(report.checks, tuple)
    with pytest.raises(FrozenInstanceError):
        report.checks = ()


def test_audit_honors_precision_implied_by_decimal_expected_value() -> None:
    payload = {
        "contract_version": 2,
        "insights": [
            {
                "metric_spec": {
                    "components": [
                        {
                            "expected_value": -4.0606,
                            "verification": {"method": "sql"},
                        }
                    ]
                }
            }
        ],
    }

    report = audit_deep_insight(payload, lambda check: -4.06058)

    assert report.total_checks == 1


def test_audit_accepts_exact_decimal_half_unit_rounding_boundary() -> None:
    payload = {
        "contract_version": 2,
        "insights": [
            {
                "expected_value": 6597.88,
                "verification": {"method": "sql"},
            }
        ],
    }

    report = audit_deep_insight(payload, lambda check: 6597.875)

    assert report.total_checks == 1


def test_audit_rejects_difference_beyond_implied_decimal_precision() -> None:
    payload = {
        "contract_version": 2,
        "insights": [
            {
                "metric_spec": {
                    "components": [
                        {
                            "expected_value": -4.0606,
                            "verification": {"method": "sql"},
                        }
                    ]
                }
            }
        ],
    }

    with pytest.raises(DeepInsightAuditError, match="expected -4.0606"):
        audit_deep_insight(payload, lambda check: -4.06054)


def test_audit_requires_contract_version_two() -> None:
    with pytest.raises(DeepInsightAuditError, match="contract_version 2"):
        audit_deep_insight({"insights": []}, lambda check: 1)
