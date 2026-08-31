from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "olist_staged_deep_insight_benchmark.py"
)
CANONICAL_FILES = (
    "customers.csv",
    "geolocation.csv",
    "order_items.csv",
    "order_payments.csv",
    "order_reviews.csv",
    "orders.csv",
    "product_category_name_translation.csv",
    "products.csv",
    "sellers.csv",
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "olist_staged_deep_insight_benchmark", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bundle(data_dir: Path) -> None:
    data_dir.mkdir()
    for name in CANONICAL_FILES:
        (data_dir / name).write_text("header\n", encoding="utf-8")


def valid_research() -> dict:
    return {
        "analysis_plan": {"grain": "order"},
        "join_map": [{"left": "orders", "matched": 10, "unmatched": 1}],
        "method_applicability": {"decomposition": "applicable"},
        "candidates": [{"finding": "candidate"}],
    }


def test_research_prompt_names_sources_and_demands_depth_without_final_contract(
    tmp_path: Path,
) -> None:
    bench = load_module()
    make_bundle(tmp_path / "olist")
    sources = bench.discover_sources(tmp_path / "olist")

    prompt = bench.build_research_prompt(sources)
    lowered = " ".join(prompt.lower().split())

    for identity, path in sources.items():
        assert f"{identity}:" in prompt
        assert str(path) in prompt
    assert "research ledger" in lowered
    assert "not the final deep-insight contract" in lowered
    assert "6-10" in prompt and "quality" in lowered
    for phrase in (
        "schema",
        "grain",
        "join map",
        "coverage",
        "unmatched",
        "fan-out",
        "cross-domain",
        "rejected candidates",
        "diagnostic alternatives",
        "metric-definition sensitivities",
        "benchmark/target basis",
        "self-contained",
        "canonical source aliases",
        "no raw records",
        "review text",
    ):
        assert phrase in lowered
    for method in (
        "decomposition",
        "instrumentation",
        "change points",
        "cohorts",
        "interactions",
        "drivers",
        "concentration",
        "clustering",
        "classification",
        "regression",
    ):
        assert method in lowered


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("", "empty"),
        ("not-json", "valid JSON"),
        ("[]", "object"),
        (json.dumps({"analysis_plan": {}, "join_map": [], "method_applicability": {}, "candidates": []}), "non-empty"),
        (json.dumps({"analysis_plan": {"x": 1}}), "missing"),
    ],
)
def test_parse_research_json_rejects_malformed_or_incomplete_ledgers(
    value: str, match: str
) -> None:
    bench = load_module()

    with pytest.raises(ValueError, match=match):
        bench.parse_research_json(value)


def test_parse_research_json_accepts_complete_object() -> None:
    bench = load_module()
    research = valid_research()

    assert bench.parse_research_json(json.dumps(research)) == research


def test_scaffold_and_insight_prompts_embed_deterministic_inputs_and_contract_rules(
    tmp_path: Path,
) -> None:
    bench = load_module()
    make_bundle(tmp_path / "olist")
    sources = bench.discover_sources(tmp_path / "olist")
    research = valid_research()

    scaffold_prompt = bench.build_contract_scaffold_prompt(sources, research)
    scaffold = {
        "analysis_plan": {"grain": "order"},
        "candidates": [
            {
                "candidate": "Delivery delay",
                "dimensions_tested": ["seller_state"],
                "disposition": "promoted",
                "promoted_as": "Delivery delay",
            }
        ],
    }
    insight_prompt = bench.build_insights_prompt(sources, research, scaffold)
    compact_research = json.dumps(
        research, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    compact_scaffold = json.dumps(
        scaffold, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    scaffold_lowered = " ".join(scaffold_prompt.lower().split())
    insight_lowered = " ".join(insight_prompt.lower().split())

    assert compact_research in scaffold_prompt
    assert compact_research in insight_prompt
    assert compact_scaffold in insight_prompt
    assert "analysis_plan" in scaffold_lowered and "candidates" in scaffold_lowered
    assert "quantitative rejection evidence" in scaffold_lowered
    assert "dimensions_available" in scaffold_prompt
    assert "dimensions_deferred" in scaffold_prompt
    assert "promoted titles" in scaffold_lowered
    assert "3-5" in scaffold_prompt
    assert "exactly match" in insight_lowered
    assert "contract v2 diagnostics" in insight_lowered
    assert "metric specs" in insight_lowered
    assert "no broad exploration" in insight_lowered
    assert "exact alias->source" in insight_lowered
    assert "self-contained sql" in insight_lowered
    for identity, path in sources.items():
        assert f"{identity}: {path}" in scaffold_prompt
        assert f"{identity}: {path}" in insight_prompt


def _closure_ready_payload() -> dict:
    return {
        "contract_version": 2,
        "analysis_plan": {"business_context": "Generic business lifecycle"},
        "candidates": [],
        "insights": [
            {
                "title": "Observed group difference",
                "confidence": {"level": "medium"},
                "priority": {"urgency": "high"},
                "action": {"kind": "diagnostic"},
                "competing_explanations": [
                    "Population exposure differs.",
                    "A required field is unavailable.",
                ],
                "diagnostic_measurability": "mixed",
                "diagnostic_assessment": {
                    "decision_readiness": "investigate_first",
                    "explanations": [
                        {
                            "explanation": "Population exposure differs.",
                            "measurable": True,
                            "disposition": "unresolved",
                        },
                        {
                            "explanation": "A required field is unavailable.",
                            "measurable": False,
                            "disposition": "not_measurable",
                            "limitation": "The source does not contain the field.",
                        },
                    ],
                },
            }
        ],
    }


def test_evidence_closure_prompt_targets_only_measurable_unresolved_explanations(
    tmp_path: Path,
) -> None:
    bench = load_module()
    sources = {"facts": tmp_path / "facts.csv"}
    payload = _closure_ready_payload()

    prompt = bench.build_evidence_closure_prompt(sources, payload)
    lowered = " ".join(prompt.lower().split())

    assert "insight-1-explanation-1" in prompt
    assert "Population exposure differs." in prompt
    assert "A required field is unavailable." not in prompt
    assert "aggregate" in lowered
    assert "no raw records" in lowered
    assert "personal identifiers" in lowered
    assert "free-text" in lowered
    assert "exactly one" in lowered
    assert "one source column or one aggregate" in lowered
    assert "do not divide or combine aggregates" in lowered


def test_validate_evidence_closure_plan_accepts_exact_bounded_targets(
    tmp_path: Path,
) -> None:
    bench = load_module()
    sources = {"facts": tmp_path / "facts.csv"}
    payload = _closure_ready_payload()
    plan = {
        "closure_plans": [
            {
                "explanation_id": "insight-1-explanation-1",
                "required_check": "Compare exposure-normalized group rates.",
                "disposition": "weakened",
                "expected_value": 0.04,
                "verification": {
                    "method": "sql",
                    "expression": "SELECT AVG(exposure_rate) FROM facts",
                    "sources": {"f": "facts"},
                },
            }
        ]
    }

    assert bench.validate_evidence_closure_plan(payload, plan, sources) == plan


def test_validate_evidence_closure_plan_normalizes_exact_authorized_filename(
    tmp_path: Path,
) -> None:
    bench = load_module()
    sources = {"facts": tmp_path / "facts.csv"}
    payload = _closure_ready_payload()
    plan = {
        "closure_plans": [
            {
                "explanation_id": "insight-1-explanation-1",
                "required_check": "Compare exposure-normalized group rates.",
                "disposition": "weakened",
                "expected_value": 0.04,
                "verification": {
                    "method": "sql",
                    "expression": "SELECT AVG(exposure_rate) FROM facts",
                    "sources": {"f": "facts.csv"},
                },
            }
        ]
    }

    validated = bench.validate_evidence_closure_plan(payload, plan, sources)

    assert validated["closure_plans"][0]["verification"]["sources"] == {
        "f": "facts"
    }
    assert plan["closure_plans"][0]["verification"]["sources"] == {
        "f": "facts.csv"
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda plan: plan["closure_plans"][0].update(
                explanation_id="unknown-target"
            ),
            "exact pending target",
        ),
        (
            lambda plan: plan["closure_plans"][0]["verification"]["sources"].update(
                f="unknown_source"
            ),
            "authoritative source",
        ),
        (
            lambda plan: plan["closure_plans"][0].update(
                disposition="unresolved"
            ),
            "disposition",
        ),
        (
            lambda plan: plan["closure_plans"].append(
                dict(plan["closure_plans"][0])
            ),
            "exactly one",
        ),
    ],
)
def test_validate_evidence_closure_plan_rejects_scope_or_source_broadening(
    tmp_path: Path,
    mutate,
    match: str,
) -> None:
    bench = load_module()
    sources = {"facts": tmp_path / "facts.csv"}
    payload = _closure_ready_payload()
    plan = {
        "closure_plans": [
            {
                "explanation_id": "insight-1-explanation-1",
                "required_check": "Compare exposure-normalized group rates.",
                "disposition": "weakened",
                "expected_value": 0.04,
                "verification": {
                    "method": "sql",
                    "expression": "SELECT AVG(exposure_rate) FROM facts",
                    "sources": {"f": "facts"},
                },
            }
        ]
    }
    mutate(plan)

    with pytest.raises(ValueError, match=match):
        bench.validate_evidence_closure_plan(payload, plan, sources)


def test_merge_evidence_closure_plan_upgrades_contract_and_only_targeted_leaf(
    tmp_path: Path,
) -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    original = deepcopy(payload)
    plan = {
        "closure_plans": [
            {
                "explanation_id": "insight-1-explanation-1",
                "required_check": "Compare exposure-normalized group rates.",
                "disposition": "ruled_out",
                "expected_value": 0.01,
                "verification": {
                    "method": "sql",
                    "expression": "SELECT AVG(exposure_rate) FROM facts",
                    "sources": {"f": "facts"},
                },
            }
        ]
    }

    merged = bench.merge_evidence_closure_plan(
        payload,
        plan,
        {"facts": tmp_path / "facts.csv"},
    )

    explanation = merged["insights"][0]["diagnostic_assessment"]["explanations"][0]
    assert merged["contract_version"] == 3
    assert explanation == {
        "explanation": "Population exposure differs.",
        "measurable": True,
        "disposition": "ruled_out",
        "explanation_id": "insight-1-explanation-1",
        "closure_status": "ruled_out",
        "required_check": "Compare exposure-normalized group rates.",
        "expected_value": 0.01,
        "verification": plan["closure_plans"][0]["verification"],
    }
    assert merged["insights"][0]["diagnostic_assessment"]["explanations"][1][
        "closure_status"
    ] == "unresolvable"
    assert payload == original


def test_mechanical_normalizer_gates_supported_closure_evidence() -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    payload["contract_version"] = 3
    insight = payload["insights"][0]
    insight["confidence"]["level"] = "high"
    insight["priority"]["urgency"] = "critical"
    insight["action"]["kind"] = "program"
    insight["diagnostic_assessment"]["decision_readiness"] = "act_ready"
    explanation = insight["diagnostic_assessment"]["explanations"][0]
    explanation.update(
        {
            "disposition": "supported",
            "closure_status": "supported",
            "explanation_id": "insight-1-explanation-1",
            "required_check": "Compare exposure-normalized group rates.",
            "expected_value": 0.25,
            "verification": {
                "method": "sql",
                "expression": "SELECT AVG(exposure_rate) FROM facts",
                "sources": {"f": "facts"},
            },
        }
    )
    insight["diagnostic_assessment"]["explanations"][1].update(
        {
            "explanation_id": "insight-1-explanation-2",
            "closure_status": "unresolvable",
        }
    )

    normalized, changes = bench.normalize_mechanical_contract(payload, ["facts"])

    assert normalized["insights"][0]["diagnostic_assessment"][
        "decision_readiness"
    ] == "investigate_first"
    assert normalized["insights"][0]["action"]["kind"] == "diagnostic"
    assert normalized["insights"][0]["confidence"]["level"] == "medium"
    assert normalized["insights"][0]["priority"]["urgency"] == "high"
    assert len(changes) == 5


def test_mechanical_normalizer_marks_fully_closed_evidence_act_ready() -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    payload["contract_version"] = 3
    insight = payload["insights"][0]
    explanation = insight["diagnostic_assessment"]["explanations"][0]
    explanation.update(
        {
            "disposition": "ruled_out",
            "closure_status": "ruled_out",
            "explanation_id": "insight-1-explanation-1",
            "required_check": "Compare exposure-normalized group rates.",
            "expected_value": 0.01,
            "verification": {
                "method": "sql",
                "expression": "SELECT AVG(exposure_rate) FROM facts",
                "sources": {"f": "facts"},
            },
        }
    )
    insight["diagnostic_assessment"]["explanations"][1].update(
        {
            "explanation_id": "insight-1-explanation-2",
            "closure_status": "unresolvable",
        }
    )

    normalized, changes = bench.normalize_mechanical_contract(payload, ["facts"])

    assert normalized["insights"][0]["diagnostic_assessment"][
        "decision_readiness"
    ] == "act_ready"
    assert normalized["insights"][0]["action"]["kind"] == "diagnostic"
    assert changes == (
        "$.insights[0].diagnostic_assessment.explanations[0].verification.sources.facts",
        "$.insights[0].diagnostic_assessment.decision_readiness",
    )


def test_run_evidence_closure_verifies_and_audits_before_checkpoint(
    tmp_path: Path,
) -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    sources = {"facts": tmp_path / "facts.csv"}
    checkpoint = tmp_path / "closure.checkpoint.json"
    events = []
    plan = {
        "closure_plans": [
            {
                "explanation_id": "insight-1-explanation-1",
                "required_check": "Compare exposure-normalized group rates.",
                "disposition": "weakened",
                "expected_value": 0.04,
                "verification": {
                    "method": "sql",
                    "expression": "SELECT AVG(exposure_rate) FROM facts",
                    "sources": {"f": "facts"},
                },
            }
        ]
    }

    class Result:
        payload = plan
        trajectory = "closure-trajectory"

    class RLM:
        @classmethod
        def from_task(cls, **kwargs):
            events.append(("model", kwargs))
            return type("Runner", (), {"run": lambda self: Result()})()

    class Executor:
        def __init__(self, actual_sources):
            assert actual_sources == sources

        def __enter__(self):
            events.append("executor")
            return self

        def __exit__(self, *args):
            return None

    def verify(actual):
        assert actual["contract_version"] == 3
        events.append("verify")

    def audit(actual, executor):
        assert actual["contract_version"] == 3
        events.append("audit")
        return FakeAudit((FakeCheck("closure", 0.04, 0.04),))

    record = bench.run_evidence_closure(
        payload,
        sources,
        lm=object(),
        rlm_type=RLM,
        executor_type=Executor,
        audit_function=audit,
        summarize_trajectory=lambda value: {"trajectory": value, "turns": 2},
        verify_function=verify,
        max_turns=5,
        timeout=60,
        checkpoint_path=checkpoint,
    )

    assert events[0][0] == "model"
    assert events[0][1]["outputs"] == bench.EVIDENCE_CLOSURE_OUTPUTS
    assert events[0][1]["max_turns"] == 5
    assert events[1:] == ["verify", "executor", "audit"]
    assert record["payload"]["contract_version"] == 3
    assert record["audit"].total_checks == 1
    assert record["summary"] == {
        "cached": False,
        "submitted": True,
        "trajectory": "closure-trajectory",
        "turns": 2,
    }
    assert checkpoint.is_file()


def test_run_evidence_closure_defers_numeric_audit_mismatch_to_outer_repair(
    tmp_path: Path,
) -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    sources = {"facts": tmp_path / "facts.csv"}
    checkpoint = tmp_path / "closure.checkpoint.json"
    plan = {
        "closure_plans": [
            {
                "explanation_id": "insight-1-explanation-1",
                "required_check": "Compare exposure-normalized group rates.",
                "disposition": "weakened",
                "expected_value": 0.04,
                "verification": {
                    "method": "sql",
                    "expression": "SELECT AVG(exposure_rate) FROM facts",
                    "sources": {"f": "facts"},
                },
            }
        ]
    }

    class NumericAuditError(Exception):
        pass

    class Result:
        payload = plan
        trajectory = "closure-trajectory"

    class RLM:
        @classmethod
        def from_task(cls, **kwargs):
            return type("Runner", (), {"run": lambda self: Result()})()

    class Executor:
        def __init__(self, actual_sources):
            assert actual_sources == sources

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def audit(actual, executor):
        raise NumericAuditError(
            "insights[0].metric_spec.components[0]: expected 1, actual 2"
        )

    record = bench.run_evidence_closure(
        payload,
        sources,
        lm=object(),
        rlm_type=RLM,
        executor_type=Executor,
        audit_function=audit,
        summarize_trajectory=lambda value: {"turns": 2},
        verify_function=lambda actual: None,
        max_turns=5,
        timeout=60,
        checkpoint_path=checkpoint,
        deferred_audit_error_type=NumericAuditError,
    )

    assert record["payload"]["contract_version"] == 3
    assert record["audit"] is None
    assert not checkpoint.exists()


def test_run_evidence_closure_repairs_invalid_plan_before_execution(
    tmp_path: Path,
) -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    sources = {"facts": tmp_path / "facts.csv"}
    valid_item = {
        "explanation_id": "insight-1-explanation-1",
        "required_check": "Compare exposure-normalized group rates.",
        "disposition": "weakened",
        "expected_value": 0.04,
        "verification": {
            "method": "sql",
            "expression": "SELECT AVG(exposure_rate) FROM facts",
            "sources": {"f": "facts"},
        },
    }
    results = [
        {
            "closure_plans": [
                {**valid_item, "unsupported": "remove this field"}
            ]
        },
        {"closure_plans": [valid_item]},
    ]
    tasks = []

    class RLM:
        @classmethod
        def from_task(cls, **kwargs):
            tasks.append(kwargs["task"])
            result = type(
                "Result",
                (),
                {"payload": results.pop(0), "trajectory": f"turn-{len(tasks)}"},
            )()
            return type("Runner", (), {"run": lambda self: result})()

    class Executor:
        def __init__(self, actual_sources):
            assert actual_sources == sources

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    record = bench.run_evidence_closure(
        payload,
        sources,
        lm=object(),
        rlm_type=RLM,
        executor_type=Executor,
        audit_function=lambda actual, executor: FakeAudit(()),
        summarize_trajectory=lambda value: {"trajectory": value},
        verify_function=lambda actual: None,
        max_turns=5,
        timeout=60,
        max_plan_repairs=1,
    )

    assert len(tasks) == 2
    assert "unsupported fields" in tasks[1]
    assert "CURRENT INVALID CLOSURE PLAN" in tasks[1]
    assert record["summary"]["plan_repairs"] == 1


def test_run_evidence_closure_persists_repaired_cached_plan(
    tmp_path: Path,
) -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    sources = {"facts": tmp_path / "facts.csv"}
    checkpoint = tmp_path / "closure.checkpoint.json"
    valid_item = {
        "explanation_id": "insight-1-explanation-1",
        "required_check": "Compare exposure-normalized group rates.",
        "disposition": "weakened",
        "expected_value": 0.04,
        "verification": {
            "method": "sql",
            "expression": "SELECT AVG(exposure_rate) FROM facts",
            "sources": {"f": "facts"},
        },
    }
    invalid_plan = {
        "closure_plans": [{**valid_item, "unsupported": "remove this field"}]
    }
    valid_plan = {"closure_plans": [valid_item]}
    fingerprint = bench._input_fingerprint(
        payload,
        {"sources": {name: str(path) for name, path in sources.items()}},
    )
    bench._write_synthesis_checkpoint(checkpoint, fingerprint, invalid_plan)
    calls = 0

    class RLM:
        @classmethod
        def from_task(cls, **kwargs):
            nonlocal calls
            calls += 1
            result = type(
                "Result",
                (),
                {"payload": valid_plan, "trajectory": "repair"},
            )()
            return type("Runner", (), {"run": lambda self: result})()

    class Executor:
        def __init__(self, actual_sources):
            assert actual_sources == sources

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    kwargs = {
        "payload": payload,
        "sources": sources,
        "lm": object(),
        "rlm_type": RLM,
        "executor_type": Executor,
        "audit_function": lambda actual, executor: FakeAudit(()),
        "summarize_trajectory": lambda value: {},
        "verify_function": lambda actual: None,
        "max_turns": 5,
        "timeout": 60,
        "checkpoint_path": checkpoint,
        "max_plan_repairs": 1,
    }

    bench.run_evidence_closure(**kwargs)
    bench.run_evidence_closure(**kwargs)

    assert calls == 1
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["partial"] == valid_plan


def test_run_evidence_closure_repairs_plan_rejected_by_portable_verifier(
    tmp_path: Path,
) -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    sources = {"facts": tmp_path / "facts.csv"}
    base_item = {
        "explanation_id": "insight-1-explanation-1",
        "required_check": "Compare exposure-normalized group rates.",
        "disposition": "weakened",
        "expected_value": 0.04,
        "verification": {
            "method": "sql",
            "expression": "SELECT SUM(a) / SUM(b) AS metric_value FROM facts",
            "sources": {"f": "facts"},
        },
    }
    repaired_item = deepcopy(base_item)
    repaired_item["verification"]["expression"] = (
        "SELECT AVG(exposure_rate) AS metric_value FROM facts"
    )
    results = [
        {"closure_plans": [base_item]},
        {"closure_plans": [repaired_item]},
    ]

    class RLM:
        @classmethod
        def from_task(cls, **kwargs):
            result = type(
                "Result",
                (),
                {"payload": results.pop(0), "trajectory": "closure"},
            )()
            return type("Runner", (), {"run": lambda self: result})()

    class Executor:
        def __init__(self, actual_sources):
            assert actual_sources == sources

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def verify(actual):
        expression = actual["insights"][0]["diagnostic_assessment"][
            "explanations"
        ][0]["verification"]["expression"]
        assert " / " not in expression, (
            "diagnostic verification must use one aggregate"
        )

    record = bench.run_evidence_closure(
        payload,
        sources,
        lm=object(),
        rlm_type=RLM,
        executor_type=Executor,
        audit_function=lambda actual, executor: FakeAudit(()),
        summarize_trajectory=lambda value: {},
        verify_function=verify,
        max_turns=5,
        timeout=60,
        max_plan_repairs=1,
    )

    assert record["summary"]["plan_repairs"] == 1
    assert not results


def test_run_evidence_closure_skips_model_and_migration_without_targets(
    tmp_path: Path,
) -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    payload["insights"][0]["diagnostic_assessment"]["explanations"][0][
        "disposition"
    ] = "ruled_out"
    payload["insights"][0]["diagnostic_assessment"]["explanations"][0].update(
        {
            "expected_value": 0.01,
            "verification": {
                "method": "sql",
                "expression": "SELECT AVG(exposure_rate) FROM facts",
                "sources": {"f": "facts"},
            },
        }
    )

    record = bench.run_evidence_closure(
        payload,
        {"facts": tmp_path / "facts.csv"},
        lm=object(),
        rlm_type=object(),
        executor_type=object(),
        audit_function=lambda *args: None,
        summarize_trajectory=lambda value: {},
        verify_function=lambda value: None,
        max_turns=5,
        timeout=60,
        checkpoint_path=tmp_path / "closure.checkpoint.json",
    )

    assert record == {
        "payload": payload,
        "audit": None,
        "summary": {
            "cached": True,
            "submitted": True,
            "turns": 0,
            "skipped": True,
        },
    }


def _gated_critic() -> dict:
    return {
        "reviewed_insights": [
            {
                "title": "Observed group difference",
                "challenges": [
                    {
                        "id": "challenge-1",
                        "type": "denominator_integrity",
                        "assessment": "Exposure time may explain the difference.",
                        "severity": "blocking",
                    },
                    {
                        "id": "challenge-2",
                        "type": "obviousness",
                        "assessment": "The finding may be expected.",
                        "severity": "minor",
                    },
                ],
                "required_changes": [
                    {
                        "change": "Measure exposure-normalized rates.",
                        "gate": "investigate_first",
                    }
                ],
                "resolutions": [
                    {
                        "challenge_index": 0,
                        "challenge_type": "denominator_integrity",
                        "status": "gated",
                    },
                    {
                        "challenge_index": 1,
                        "challenge_type": "obviousness",
                        "status": "resolved",
                    },
                ],
            }
        ]
    }


def test_critic_closure_prompt_targets_only_gated_material_or_blocking_challenges(
    tmp_path: Path,
) -> None:
    bench = load_module()

    prompt = bench.build_critic_closure_prompt(
        {"facts": tmp_path / "facts.csv"},
        _closure_ready_payload(),
        _gated_critic(),
    )

    assert "challenge-1" in prompt
    assert "challenge-2" not in prompt
    assert "insight-1-explanation-1" in prompt
    assert "insight-1-explanation-2" in prompt
    assert "do not weaken" in prompt.lower()
    assert "aggregate" in prompt.lower()


def test_merge_critic_closure_plan_reopens_exact_existing_explanation(
    tmp_path: Path,
) -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    payload["insights"][0]["diagnostic_assessment"]["explanations"][0].update(
        {
            "measurable": False,
            "disposition": "not_measurable",
            "limitation": "The original discovery did not test exposure.",
        }
    )
    plan = {
        "critic_closure_plans": [
            {
                "challenge_id": "challenge-1",
                "explanation_id": "insight-1-explanation-1",
                "required_check": "Measure exposure-normalized rates.",
                "disposition": "weakened",
                "expected_value": 0.04,
                "verification": {
                    "method": "sql",
                    "expression": "SELECT AVG(exposure_rate) FROM facts",
                    "sources": {"f": "facts"},
                },
            }
        ]
    }

    merged = bench.merge_critic_closure_plan(
        payload,
        _gated_critic(),
        plan,
        {"facts": tmp_path / "facts.csv"},
    )

    explanation = merged["insights"][0]["diagnostic_assessment"]["explanations"][0]
    assert explanation["measurable"] is True
    assert explanation["disposition"] == "weakened"
    assert explanation["closure_status"] == "weakened"
    assert explanation["critic_challenge_id"] == "challenge-1"
    assert "limitation" not in explanation
    assert merged["contract_version"] == 3


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda plan: plan["critic_closure_plans"][0].update(
                challenge_id="challenge-2"
            ),
            "eligible gated challenge",
        ),
        (
            lambda plan: plan["critic_closure_plans"][0].update(
                explanation_id="unknown-explanation"
            ),
            "existing explanation",
        ),
        (
            lambda plan: plan["critic_closure_plans"][0]["verification"][
                "sources"
            ].update(f="unknown"),
            "authoritative source",
        ),
    ],
)
def test_merge_critic_closure_plan_rejects_scope_broadening(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    bench = load_module()
    plan = {
        "critic_closure_plans": [
            {
                "challenge_id": "challenge-1",
                "explanation_id": "insight-1-explanation-1",
                "required_check": "Measure exposure-normalized rates.",
                "disposition": "weakened",
                "expected_value": 0.04,
                "verification": {
                    "method": "sql",
                    "expression": "SELECT AVG(exposure_rate) FROM facts",
                    "sources": {"f": "facts"},
                },
            }
        ]
    }
    mutation(plan)

    with pytest.raises(ValueError, match=match):
        bench.merge_critic_closure_plan(
            _closure_ready_payload(),
            _gated_critic(),
            plan,
            {"facts": tmp_path / "facts.csv"},
        )


def test_run_critic_evidence_closure_executes_before_persisting(
    tmp_path: Path,
) -> None:
    bench = load_module()
    sources = {"facts": tmp_path / "facts.csv"}
    checkpoint = tmp_path / "critic-closure.checkpoint.json"
    events = []
    plan = {
        "critic_closure_plans": [
            {
                "challenge_id": "challenge-1",
                "explanation_id": "insight-1-explanation-1",
                "required_check": "Measure exposure-normalized rates.",
                "disposition": "weakened",
                "expected_value": 0.04,
                "verification": {
                    "method": "sql",
                    "expression": "SELECT AVG(exposure_rate) FROM facts",
                    "sources": {"f": "facts"},
                },
            }
        ]
    }

    class Result:
        payload = plan
        trajectory = "critic-closure-trajectory"

    class RLM:
        @classmethod
        def from_task(cls, **kwargs):
            events.append(("model", kwargs))
            return type("Runner", (), {"run": lambda self: Result()})()

    class Executor:
        def __init__(self, actual_sources):
            assert actual_sources == sources

        def __enter__(self):
            events.append("executor")
            return self

        def __exit__(self, *args):
            return None

    record = bench.run_critic_evidence_closure(
        _closure_ready_payload(),
        _gated_critic(),
        sources,
        lm=object(),
        rlm_type=RLM,
        executor_type=Executor,
        audit_function=lambda actual, executor: events.append("audit")
        or FakeAudit((FakeCheck("closure", 0.04, 0.04),)),
        summarize_trajectory=lambda value: {"trajectory": value, "turns": 2},
        verify_function=lambda actual: events.append("verify"),
        max_turns=8,
        timeout=60,
        checkpoint_path=checkpoint,
    )

    assert events[0][0] == "model"
    assert events[0][1]["outputs"] == bench.CRITIC_CLOSURE_OUTPUTS
    assert events[1:] == ["verify", "executor", "audit"]
    assert record["payload"]["contract_version"] == 3
    assert record["audit"].total_checks == 1
    assert checkpoint.is_file()


def _approved_action_critic() -> dict:
    return {
        "reviewed_insights": [
            {
                "title": "Observed group difference",
                "verdict": "approve",
                "decision_effect": "The evidence changes a bounded operating decision.",
                "challenges": [
                    {
                        "type": "denominator_integrity",
                        "severity": "material",
                    }
                ],
                "required_changes": [],
                "synthesis_eligible": True,
                "resolutions": [
                    {
                        "challenge_index": 0,
                        "challenge_type": "denominator_integrity",
                        "status": "resolved",
                    }
                ],
            }
        ],
        "portfolio_challenges": [],
        "checks_performed": [],
        "synthesis_manifest": {
            "program_action_titles": [],
            "diagnostic_only_titles": ["Observed group difference"],
        },
    }


def test_action_synthesis_targets_only_approved_act_ready_diagnostics() -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    payload["contract_version"] = 3
    payload["insights"][0]["diagnostic_assessment"][
        "decision_readiness"
    ] = "act_ready"

    prompt = bench.build_action_synthesis_prompt(
        payload,
        _approved_action_critic(),
    )

    assert "Observed group difference" in prompt
    assert "exactly one action update" in prompt.lower()
    assert "kind must be program" in prompt.lower()


@pytest.mark.parametrize(
    "critic_mutation",
    [
        lambda critic: (
            critic["portfolio_challenges"].append(
                {
                    "challenge_id": "portfolio-1",
                    "severity": "blocking",
                    "affected_insight_titles": ["Observed group difference"],
                }
            ),
        ),
        lambda critic: (
            critic["checks_performed"].append(
                {
                    "check_id": "check-1",
                    "severity": "material",
                    "status": "deferred",
                    "affected_insight_titles": ["Observed group difference"],
                }
            ),
        ),
        lambda critic: (
            critic["reviewed_insights"][0]["challenges"].append(
                {
                    "type": "scope_boundary",
                    "severity": "minor",
                }
            ),
            critic["reviewed_insights"][0]["resolutions"].append(
                {
                    "challenge_index": 1,
                    "challenge_type": "scope_boundary",
                    "status": "gated",
                }
            ),
        ),
    ],
)
def test_action_synthesis_respects_critic_manifest_program_bars(
    critic_mutation,
) -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    payload["contract_version"] = 3
    payload["insights"][0]["diagnostic_assessment"][
        "decision_readiness"
    ] = "act_ready"
    critic = _approved_action_critic()
    critic_mutation(critic)

    assert bench._action_synthesis_targets(payload, critic) == []


def test_merge_action_synthesis_updates_only_exact_action() -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    payload["contract_version"] = 3
    payload["insights"][0]["diagnostic_assessment"][
        "decision_readiness"
    ] = "act_ready"
    original = deepcopy(payload)
    updates = {
        "action_updates": [
            {
                "title": "Observed group difference",
                "action": {
                    "owner": "Operations",
                    "segment": "Eligible backlog",
                    "decision": "Launch a bounded recovery program",
                    "target": "Reduce unresolved backlog",
                    "time_horizon": "Next operating cycle",
                    "kind": "program",
                },
            }
        ]
    }

    merged = bench.merge_action_synthesis(
        payload,
        _approved_action_critic(),
        updates,
    )

    assert merged["insights"][0]["action"] == updates["action_updates"][0]["action"]
    unchanged = deepcopy(merged)
    unchanged["insights"][0]["action"] = original["insights"][0]["action"]
    assert unchanged == original


@pytest.mark.parametrize(
    ("critic_mutation", "update_mutation", "match"),
    [
        (
            lambda critic: critic["reviewed_insights"][0].update(
                verdict="revise"
            ),
            lambda updates: None,
            "eligible",
        ),
        (
            lambda critic: None,
            lambda updates: updates["action_updates"][0].update(title="Other"),
            "exact eligible title",
        ),
        (
            lambda critic: None,
            lambda updates: updates["action_updates"][0]["action"].update(
                kind="diagnostic"
            ),
            "kind must be program",
        ),
    ],
)
def test_merge_action_synthesis_rejects_ineligible_or_broadened_updates(
    critic_mutation,
    update_mutation,
    match: str,
) -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    payload["contract_version"] = 3
    payload["insights"][0]["diagnostic_assessment"][
        "decision_readiness"
    ] = "act_ready"
    critic = _approved_action_critic()
    updates = {
        "action_updates": [
            {
                "title": "Observed group difference",
                "action": {
                    "owner": "Operations",
                    "segment": "Eligible backlog",
                    "decision": "Launch a bounded recovery program",
                    "target": "Reduce unresolved backlog",
                    "time_horizon": "Next operating cycle",
                    "kind": "program",
                },
            }
        ]
    }
    critic_mutation(critic)
    update_mutation(updates)

    with pytest.raises(ValueError, match=match):
        bench.merge_action_synthesis(payload, critic, updates)


def test_run_action_synthesis_verifies_before_checkpoint(tmp_path: Path) -> None:
    bench = load_module()
    payload = _closure_ready_payload()
    payload["contract_version"] = 3
    payload["insights"][0]["diagnostic_assessment"][
        "decision_readiness"
    ] = "act_ready"
    checkpoint = tmp_path / "action.checkpoint.json"
    events = []
    updates = {
        "action_updates": [
            {
                "title": "Observed group difference",
                "action": {
                    "owner": "Operations",
                    "segment": "Eligible backlog",
                    "decision": "Launch a bounded recovery program",
                    "target": "Reduce unresolved backlog",
                    "time_horizon": "Next operating cycle",
                    "kind": "program",
                },
            }
        ]
    }

    class Result:
        payload = updates
        trajectory = "action-trajectory"

    class RLM:
        @classmethod
        def from_task(cls, **kwargs):
            events.append(("model", kwargs))
            return type("Runner", (), {"run": lambda self: Result()})()

    record = bench.run_action_synthesis(
        payload,
        _approved_action_critic(),
        lm=object(),
        rlm_type=RLM,
        summarize_trajectory=lambda value: {"turns": 2},
        verify_function=lambda actual: events.append("verify"),
        max_turns=6,
        timeout=60,
        checkpoint_path=checkpoint,
    )

    assert events[0][1]["outputs"] == bench.ACTION_SYNTHESIS_OUTPUTS
    assert events[1] == "verify"
    assert record["payload"]["insights"][0]["action"]["kind"] == "program"
    assert checkpoint.is_file()


def test_assemble_contract_is_strict_detached_and_does_not_mutate_inputs() -> None:
    bench = load_module()
    scaffold = {
        "analysis_plan": {"dimensions": ["seller_state"]},
        "candidates": [{"candidate": "Delivery delay"}],
    }
    insights = {"insights": [{"title": "Delivery delay"}]}
    original_scaffold = json.loads(json.dumps(scaffold))
    original_insights = json.loads(json.dumps(insights))

    payload = bench.assemble_contract(scaffold, insights)

    assert payload == {
        "contract_version": 2,
        "analysis_plan": scaffold["analysis_plan"],
        "candidates": scaffold["candidates"],
        "insights": insights["insights"],
    }
    assert scaffold == original_scaffold
    assert insights == original_insights
    payload["analysis_plan"]["dimensions"].append("month")
    payload["insights"][0]["title"] = "changed"
    assert scaffold == original_scaffold
    assert insights == original_insights


def test_assemble_contract_preserves_explicit_closure_contract_version() -> None:
    bench = load_module()

    payload = bench.assemble_contract(
        {"analysis_plan": {}, "candidates": []},
        {"insights": []},
        contract_version=3,
    )

    assert payload["contract_version"] == 3


def test_cached_closed_insights_remain_contract_v3_on_next_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {}, "candidates": []}
    insights = {
        "insights": [
            {
                "diagnostic_assessment": {
                    "explanations": [
                        {
                            "closure_status": "supported",
                            "disposition": "supported",
                            "expected_value": 1,
                            "explanation": "Measured alternative.",
                            "explanation_id": "insight-1-explanation-1",
                            "measurable": True,
                            "required_check": "Measure the alternative.",
                            "verification": {
                                "method": "sql",
                                "expression": "SELECT COUNT(*) AS metric_value FROM orders",
                                "sources": {"orders": "orders"},
                            },
                        }
                    ]
                }
            }
        ]
    }
    caches = write_all_caches(tmp_path, research, scaffold, insights)
    calls = install_synthesis_fakes(bench, monkeypatch, [])

    first = bench.run_staged_benchmark(
        data_dir,
        enable_evidence_closure=True,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
        max_insight_repairs=0,
        max_scaffold_repairs=0,
    )
    second = bench.run_staged_benchmark(
        data_dir,
        enable_evidence_closure=True,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
        max_insight_repairs=0,
        max_scaffold_repairs=0,
    )

    assert first["payload"]["contract_version"] == 3
    assert second["payload"]["contract_version"] == 3
    assert calls == []


@pytest.mark.parametrize(
    ("scaffold", "insights", "match"),
    [
        (
            {"analysis_plan": {}, "candidates": [], "extra": True},
            {"insights": []},
            "exactly",
        ),
        ({"analysis_plan": [], "candidates": []}, {"insights": []}, "analysis_plan"),
        ({"analysis_plan": {}, "candidates": {}}, {"insights": []}, "candidates"),
        (
            {"analysis_plan": {}, "candidates": []},
            {"insights": [], "extra": True},
            "exactly",
        ),
        ({"analysis_plan": {}, "candidates": []}, {"insights": {}}, "insights"),
    ],
)
def test_assemble_contract_rejects_malformed_or_extraneous_partials(
    scaffold: dict, insights: dict, match: str
) -> None:
    bench = load_module()

    with pytest.raises(ValueError, match=match):
        bench.assemble_contract(scaffold, insights)


def mechanical_insight() -> dict:
    return {
        "title": "Generic finding",
        "competing_explanations": ["A", "B"],
        "diagnostic_measurability": "mixed",
        "diagnostic_assessment": {
            "decision_readiness": "act_ready",
            "explanations": [
                {
                    "explanation": "A",
                    "measurable": True,
                    "disposition": "unresolved",
                },
                {
                    "explanation": "B",
                    "measurable": True,
                    "disposition": "weakened",
                    "expected_value": 1,
                    "verification": {
                        "method": "SQL",
                        "expression": (
                            "SELECT COUNT(*) FROM order_reviews r "
                            "JOIN orders o ON o.id = r.id"
                        ),
                        "sources": {"r": "order_reviews"},
                    },
                },
            ],
        },
        "action": {"kind": "program"},
        "confidence": {"level": "high"},
        "priority": {"urgency": "critical"},
    }


def test_mechanical_normalizer_repairs_sql_and_diagnostics_without_mutation() -> None:
    bench = load_module()
    payload = {"insights": [mechanical_insight()]}
    original = json.loads(json.dumps(payload))

    normalized, changes = bench.normalize_mechanical_contract(
        payload, {"order_reviews", "orders"}
    )

    assert payload == original
    assert normalized is not payload
    insight = normalized["insights"][0]
    verification = insight["diagnostic_assessment"]["explanations"][1][
        "verification"
    ]
    assert verification["sources"] == {
        "r": "order_reviews",
        "order_reviews": "order_reviews",
        "orders": "orders",
    }
    assert insight["diagnostic_measurability"] == "measurable"
    assert insight["diagnostic_assessment"]["decision_readiness"] == (
        "investigate_first"
    )
    assert insight["action"]["kind"] == "diagnostic"
    assert insight["confidence"]["level"] == "medium"
    assert insight["priority"]["urgency"] == "high"
    assert changes == (
        "$.insights[0].diagnostic_assessment.explanations[1].verification.sources.order_reviews",
        "$.insights[0].diagnostic_assessment.explanations[1].verification.sources.orders",
        "$.insights[0].diagnostic_measurability",
        "$.insights[0].diagnostic_assessment.decision_readiness",
        "$.insights[0].action.kind",
        "$.insights[0].confidence.level",
        "$.insights[0].priority.urgency",
    )


@pytest.mark.parametrize(
    ("measured_states", "expected"),
    [
        ([True, True], "measurable"),
        ([False, False], "not_measurable"),
        ([True, False], "mixed"),
    ],
)
def test_mechanical_normalizer_derives_each_diagnostic_measurability_state(
    measured_states: list[bool], expected: str
) -> None:
    bench = load_module()
    payload = {
        "insights": [
            {
                "diagnostic_measurability": "wrong",
                "diagnostic_assessment": {
                    "explanations": [
                        {
                            "measurable": measurable,
                            "disposition": (
                                "weakened" if measurable else "not_measurable"
                            ),
                        }
                        for measurable in measured_states
                    ]
                },
            }
        ]
    }

    normalized, changes = bench.normalize_mechanical_contract(payload, [])

    assert normalized["insights"][0]["diagnostic_measurability"] == expected
    assert changes == ("$.insights[0].diagnostic_measurability",)


@pytest.mark.parametrize(
    ("metric_type", "components", "expected"),
    [
        (
            "delta",
            [
                {"role": "current", "expected_value": -4.06},
                {"role": "comparison", "expected_value": -13.39},
            ],
            9.33,
        ),
        (
            "rate",
            [
                {"role": "numerator", "expected_value": 25},
                {"role": "denominator", "expected_value": 100},
            ],
            0.25,
        ),
        (
            "share",
            [
                {"role": "numerator", "expected_value": 1},
                {"role": "denominator", "expected_value": 8},
            ],
            0.125,
        ),
        (
            "rate_of_change",
            [
                {"role": "current", "expected_value": 120},
                {"role": "comparison", "expected_value": 100},
            ],
            0.2,
        ),
    ],
)
def test_mechanical_normalizer_reconciles_exact_derived_metric_arithmetic(
    metric_type: str, components: list[dict], expected: float
) -> None:
    bench = load_module()
    payload = {
        "metric_spec": {
            "type": metric_type,
            "expected_value": -999,
            "components": components,
        }
    }

    normalized, changes = bench.normalize_mechanical_contract(payload, [])

    assert normalized["metric_spec"]["expected_value"] == pytest.approx(expected)
    assert changes == ("$.metric_spec.expected_value",)


@pytest.mark.parametrize(
    "metric_spec",
    [
        {
            "type": "rate",
            "expected_value": 3,
            "components": [
                {"role": "numerator", "expected_value": 3},
                {"role": "denominator", "expected_value": 0},
            ],
        },
        {
            "type": "delta",
            "expected_value": 3,
            "components": [
                {"role": "current", "expected_value": "3"},
                {"role": "comparison", "expected_value": 1},
            ],
        },
        {
            "type": "custom",
            "expected_value": 3,
            "components": [],
        },
    ],
)
def test_mechanical_normalizer_does_not_guess_invalid_or_custom_metric_arithmetic(
    metric_spec: dict,
) -> None:
    bench = load_module()
    payload = {"metric_spec": metric_spec}

    normalized, changes = bench.normalize_mechanical_contract(payload, [])

    assert normalized == payload
    assert changes == ()


def test_mechanical_normalizer_removes_quantitative_interpretation_sentences() -> None:
    bench = load_module()
    payload = {
        "insights": [
            {
                "interpretation": (
                    "A sub-1% seller tail moves a fifth of marketplace value. "
                    "Retention attention on this tail is a material risk lever. "
                    "Ten dozen sellers would still be a concentrated group."
                ),
                "statement": "The seller tail accounts for 19.8% of GMV.",
                "supporting_claims": [{"claim": "18 sellers exceed the threshold."}],
            }
        ]
    }

    normalized, changes = bench.normalize_mechanical_contract(payload, [])

    assert normalized["insights"][0]["interpretation"] == (
        "Retention attention on this tail is a material risk lever."
    )
    assert normalized["insights"][0]["statement"] == payload["insights"][0]["statement"]
    assert normalized["insights"][0]["supporting_claims"] == payload["insights"][0][
        "supporting_claims"
    ]
    assert changes == ("$.insights[0].interpretation",)


def test_mechanical_normalizer_preserves_interpretation_if_every_sentence_is_quantitative() -> None:
    bench = load_module()
    payload = {"interpretation": "The segment accounts for 19.8% of GMV."}

    normalized, changes = bench.normalize_mechanical_contract(payload, [])

    assert normalized == payload
    assert changes == ()


def test_mechanical_normalizer_rephrases_dates_without_dropping_interpretation() -> None:
    bench = load_module()
    payload = {
        "interpretation": (
            "The post-break floor is durable, so 2018 benchmarks should use "
            "the new baseline. Planning should treat Nov-2017 as the change point."
        )
    }

    normalized, changes = bench.normalize_mechanical_contract(payload, [])

    interpretation = normalized["interpretation"]
    assert interpretation == (
        "The post-break floor is durable, so benchmarks should use the new "
        "baseline. Planning should treat the observed break as the change point."
    )
    assert not any(character.isdigit() for character in interpretation)
    assert changes == ("$.interpretation",)


def test_mechanical_normalizer_restores_uniquely_unmatched_diagnostic_label() -> None:
    bench = load_module()
    payload = {
        "competing_explanations": ["Seasonality", "Supply expansion"],
        "diagnostic_assessment": {
            "explanations": [
                {"explanation": "Seasonality with an invented numeric detail"},
                {"explanation": "Supply expansion"},
            ]
        },
    }

    normalized, changes = bench.normalize_mechanical_contract(payload, [])

    assert normalized["diagnostic_assessment"]["explanations"][0][
        "explanation"
    ] == "Seasonality"
    assert changes == (
        "$.diagnostic_assessment.explanations[0].explanation",
    )


def test_mechanical_normalizer_does_not_guess_multiple_unmatched_diagnostic_labels() -> None:
    bench = load_module()
    payload = {
        "competing_explanations": ["Seasonality", "Supply expansion"],
        "diagnostic_assessment": {
            "explanations": [
                {"explanation": "Changed first"},
                {"explanation": "Changed second"},
            ]
        },
    }

    normalized, changes = bench.normalize_mechanical_contract(payload, [])

    assert normalized == payload
    assert changes == ()


def test_mechanical_normalizer_is_conservative_for_untrusted_sql_text() -> None:
    bench = load_module()
    payload = {
        "checks": [
            {
                "method": "sql",
                "expression": (
                    "WITH orders AS (SELECT 1) "
                    "SELECT 'FROM secret JOIN hidden' AS note "
                    "FROM orders -- JOIN comments_only\n"
                    "JOIN known k ON true /* FROM block_only */"
                ),
                "sources": {"k": "known", "known": "must-not-overwrite"},
            },
            {
                "method": "SQL",
                "expression": "SELECT * FROM (SELECT * FROM nested) q",
                "sources": {},
            },
        ]
    }

    normalized, changes = bench.normalize_mechanical_contract(
        payload,
        {
            "orders",
            "known",
            "secret",
            "hidden",
            "comments_only",
            "block_only",
        },
    )

    assert normalized["checks"][0]["sources"] == {
        "k": "known",
        "known": "must-not-overwrite",
    }
    assert normalized["checks"][1]["sources"] == {}
    assert changes == ()


def test_mechanical_normalizer_never_adds_unknown_or_fuzzy_relations() -> None:
    bench = load_module()
    payload = {
        "verification": {
            "method": "sql",
            "expression": (
                "SELECT * FROM authorized a "
                "JOIN unauthorized u ON true JOIN authorized_backup b ON true"
            ),
            "sources": {},
        }
    }

    normalized, changes = bench.normalize_mechanical_contract(
        payload, ["authorized"]
    )

    assert normalized["verification"]["sources"] == {
        "authorized": "authorized"
    }
    assert changes == ("$.verification.sources.authorized",)


def test_mechanical_normalizer_binds_direct_relation_even_when_used_as_source_value() -> None:
    bench = load_module()
    payload = {
        "verification": {
            "method": "sql",
            "expression": "SELECT COUNT(*) FROM authorized",
            "sources": {"a": "authorized"},
        }
    }

    normalized, changes = bench.normalize_mechanical_contract(
        payload,
        ["authorized"],
    )

    assert normalized["verification"]["sources"] == {
        "a": "authorized",
        "authorized": "authorized",
    }
    assert changes == ("$.verification.sources.authorized",)


def test_mechanical_normalizer_rewrites_exact_authorized_paths_to_identities(
    tmp_path: Path,
) -> None:
    bench = load_module()
    authorized_path = tmp_path / "orders.csv"
    payload = {
        "verification": {
            "method": "sql",
            "expression": "SELECT COUNT(*) FROM orders",
            "sources": {"orders": str(authorized_path)},
        }
    }

    normalized, changes = bench.normalize_mechanical_contract(
        payload,
        {"orders": authorized_path},
    )

    assert normalized["verification"]["sources"] == {"orders": "orders"}
    assert changes == ("$.verification.sources.orders",)


def test_mechanical_normalizer_rewrites_unique_authorized_filenames_to_identities(
    tmp_path: Path,
) -> None:
    bench = load_module()
    payload = {
        "verification": {
            "method": "sql",
            "expression": "SELECT SUM(amount) AS metric_value FROM sales",
            "sources": {"sales": "sales.csv"},
        }
    }

    normalized, changes = bench.normalize_mechanical_contract(
        payload,
        {"sales": tmp_path / "sales.csv"},
    )

    assert normalized["verification"]["sources"] == {"sales": "sales"}
    assert changes == ("$.verification.sources.sales",)
    assert payload["verification"]["sources"] == {"sales": "sales.csv"}


def test_mechanical_normalizer_supports_only_exact_authorized_qualification() -> None:
    bench = load_module()
    payload = {
        "verification": {
            "method": "sql",
            "expression": "SELECT * FROM analytics.orders o JOIN orders x ON true",
            "sources": {},
        }
    }

    normalized, _ = bench.normalize_mechanical_contract(
        payload, ["analytics.orders"]
    )

    assert normalized["verification"]["sources"] == {
        "analytics.orders": "analytics.orders"
    }


@pytest.mark.parametrize(
    "source_names",
    [
        ["orders;drop"],
        ["../orders"],
        ["orders alias"],
        [""],
        ["analytics..orders"],
        ["9orders"],
        "orders",
    ],
)
def test_mechanical_normalizer_rejects_invalid_source_names(source_names) -> None:
    bench = load_module()

    with pytest.raises(ValueError, match="source_names"):
        bench.normalize_mechanical_contract({}, source_names)


def test_portable_verifier_wraps_only_assertions_and_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = load_module()

    class Skill:
        verifier_source = "def verify(payload):\n    assert payload['ok'], 'not ok'\n"

    class Loader:
        def load(self, name):
            assert name == "deep_insight_discovery"
            return Skill()

    monkeypatch.setattr(bench, "_load_skill_loader", lambda: Loader)
    bench.verify_portable_contract({"ok": True})

    with pytest.raises(
        AssertionError, match="portable deep-insight verification failed: not ok"
    ) as captured:
        bench.verify_portable_contract({"ok": False})
    assert isinstance(captured.value.__cause__, AssertionError)

    Skill.verifier_source = "def verify(payload):\n    raise RuntimeError('boom')\n"
    with pytest.raises(RuntimeError, match="boom"):
        bench.verify_portable_contract({"ok": True})


def test_portable_verifier_treats_double_quoted_sql_columns_as_source_data() -> None:
    bench = load_module()
    loader_type = bench._load_skill_loader()
    source = loader_type().load("deep_insight_discovery").verifier_source
    tree = ast.parse(source)
    verify = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "verify"
    )
    helpers = [
        deepcopy(node)
        for node in verify.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"constant_expression", "simple_source_metric"}
    ]
    module = ast.fix_missing_locations(
        ast.Module(
            body=[ast.Import(names=[ast.alias(name="re")]), *helpers],
            type_ignores=[],
        )
    )
    namespace: dict[str, object] = {}
    exec(compile(module, "<verifier helpers>", "exec"), namespace)

    assert namespace["simple_source_metric"](
        'COUNT(DISTINCT "Minor category code")'
    )
    assert namespace["simple_source_metric"]('SUM("Sales Amount")')
    assert not namespace["simple_source_metric"]("SUM(42)")


@dataclass(frozen=True)
class FakeCheck:
    path: str
    expected: float
    actual: float


@dataclass(frozen=True)
class FakeAudit:
    checks: tuple[FakeCheck, ...]

    @property
    def total_checks(self) -> int:
        return len(self.checks)


def synthesis_fingerprint(*inputs: dict) -> str:
    digest = hashlib.sha256()
    for value in inputs:
        digest.update(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        )
    return digest.hexdigest()


def write_checkpoint(path: Path, fingerprint: str, partial: dict) -> None:
    path.write_text(
        json.dumps({"input_fingerprint": fingerprint, "partial": partial}),
        encoding="utf-8",
    )


def install_synthesis_fakes(
    bench,
    monkeypatch: pytest.MonkeyPatch,
    payloads: list[dict],
    events: list[str] | None = None,
) -> list[dict]:
    calls: list[dict] = []

    class FakeDspy:
        @staticmethod
        def LM(*args, **kwargs):
            return object()

        @staticmethod
        def configure(**kwargs):
            pass

    class FakeRLM:
        @classmethod
        def from_task(cls, **kwargs):
            calls.append(kwargs)
            payload = payloads[len(calls) - 1]
            result = type(
                "Result",
                (),
                {"payload": payload, "trajectory": f"trajectory-{len(calls)}"},
            )()
            return type("Instance", (), {"run": lambda self: result})()

    class FakeExecutor:
        def __init__(self, sources):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            pass

    monkeypatch.setattr(
        bench,
        "_load_runtime_dependencies",
        lambda: (
            FakeDspy,
            FakeRLM,
            FakeExecutor,
            lambda payload, executor: FakeAudit(()),
            lambda trajectory: {"submitted": True, "turns": 3},
        ),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")
    monkeypatch.setattr(
        bench,
        "verify_portable_contract",
        lambda payload: events.append("verify") if events is not None else None,
    )
    return calls


def install_cached_repair_fakes(
    bench,
    monkeypatch: pytest.MonkeyPatch,
    repairs: list[dict],
    verifier,
    events: list[str] | None = None,
) -> tuple[list[dict], object]:
    calls: list[dict] = []
    lm = object()

    class FakeDspy:
        @staticmethod
        def LM(*args, **kwargs):
            return lm

        @staticmethod
        def configure(**kwargs):
            pass

    class FakeRLM:
        @classmethod
        def from_task(cls, **kwargs):
            calls.append(kwargs)
            result = type(
                "Result",
                (),
                {
                    "payload": repairs[len(calls) - 1],
                    "trajectory": f"repair-{len(calls)}",
                },
            )()
            return type("Instance", (), {"run": lambda self: result})()

    class FakeExecutor:
        def __init__(self, sources):
            if events is not None:
                events.append("audit-open")

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            pass

    def audit(payload, executor):
        if events is not None:
            events.append("audit")
        return FakeAudit(())

    monkeypatch.setattr(
        bench,
        "_load_runtime_dependencies",
        lambda: (
            FakeDspy,
            FakeRLM,
            FakeExecutor,
            audit,
            lambda trajectory: {"trajectory": trajectory, "turns": 2},
        ),
    )
    monkeypatch.setattr(bench, "verify_portable_contract", verifier)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")
    return calls, lm


def write_all_caches(
    tmp_path: Path, research: dict, scaffold: dict, insights: dict
) -> tuple[Path, Path, Path]:
    research_cache = tmp_path / "research.json"
    scaffold_cache = tmp_path / "scaffold.json"
    insights_cache = tmp_path / "insights.json"
    research_cache.write_text(json.dumps(research), encoding="utf-8")
    write_checkpoint(scaffold_cache, synthesis_fingerprint(research), scaffold)
    write_checkpoint(
        insights_cache, synthesis_fingerprint(research, scaffold), insights
    )
    return research_cache, scaffold_cache, insights_cache


def test_insight_repair_prompt_embeds_latest_inputs_and_exact_verifier_error(
    tmp_path: Path,
) -> None:
    bench = load_module()
    make_bundle(tmp_path / "olist")
    sources = bench.discover_sources(tmp_path / "olist")
    research = valid_research()
    scaffold = {"analysis_plan": {"grain": "order"}, "candidates": []}
    insights = {"insights": [{"title": "Delivery state", "evidence": "kept"}]}
    error = "insight 1 mixed diagnostic measurability requires both states"

    prompt = bench.build_insight_repair_prompt(
        sources, research, scaffold, insights, error
    )
    lowered = " ".join(prompt.lower().split())
    compact_sources = json.dumps(
        {name: str(path) for name, path in sources.items()},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    assert compact_sources in prompt
    for value in (research, scaffold, insights):
        assert json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ) in prompt
    assert error in prompt
    assert "return exactly" in lowered and "insights: list" in lowered
    assert "minimum contract correction" in lowered
    for phrase in (
        "verified facts",
        "titles",
        "dimensions",
        "sql aliases",
        "evidence",
        "unless implicated",
        "do not broaden exploration",
        "do not invent evidence",
        "call submit immediately",
    ):
        assert phrase in lowered


@pytest.mark.parametrize(
    ("error", "insight_count", "expected"),
    [
        ("insight 3 statement must contain one primary measured claim", 4, 2),
        ("INSIGHT 1 is invalid", 3, 0),
        ("prefix Insight   2 suffix", 2, 1),
        ("supporting claim 2 must be measurable", 4, None),
        ("supporting claim 2 for insight 3 is invalid", 4, 2),
        ("insight 0 is invalid", 4, None),
        ("insight 5 is invalid", 4, None),
        ("insight x is invalid", 4, None),
        ("", 4, None),
    ],
)
def test_extract_insight_index_is_case_insensitive_and_bounds_checked(
    error: str, insight_count: int, expected: int | None
) -> None:
    bench = load_module()

    assert bench.extract_insight_index(error, insight_count) == expected


def test_targeted_insight_repair_prompt_is_compact_and_scoped(
    tmp_path: Path,
) -> None:
    bench = load_module()
    make_bundle(tmp_path / "olist")
    sources = bench.discover_sources(tmp_path / "olist")
    scaffold = {
        "analysis_plan": {"unrelated_research": "must not appear"},
        "candidates": [
            {
                "candidate_id": "c1",
                "disposition": "promoted",
                "promoted_as": "Unrelated delivery title",
                "dimensions_tested": ["state"],
            },
            {
                "candidate_id": "c2",
                "disposition": "promoted",
                "promoted_as": "Seller concentration",
                "dimensions_tested": ["seller_id"],
            },
        ],
    }
    current = {
        "title": "Seller concentration",
        "discovery": {"dimensions_tested": ["seller_id"]},
        "statement": "Five sellers above 1% account for 20 of 100 orders (20%).",
        "metric_spec": {"expected_value": 0.2},
    }
    error = "insight 2 statement must contain one primary measured claim"

    prompt = bench.build_targeted_insight_repair_prompt(
        sources, scaffold, current, error
    )
    lowered = " ".join(prompt.lower().split())

    assert error in prompt
    assert bench._compact_json(current) in prompt
    assert bench._compact_json(scaffold["candidates"][1]) in prompt
    assert "Unrelated delivery title" not in prompt
    assert "unrelated_research" not in prompt
    assert "STAGE 1 RESEARCH" not in prompt
    assert all(identity in prompt for identity in sources)
    assert "insight: dict" in lowered
    assert "exact title" in lowered
    assert "dimensions" in lowered and "preserved" in lowered
    assert "minimum correction" in lowered
    assert "no invented evidence" in lowered
    assert "metric_spec.expected_value" in prompt
    assert "threshold" in lowered
    assert "numerator" in lowered
    assert "denominator" in lowered
    assert "supporting_claims" in prompt
    assert "submit immediately" in lowered


def test_targeted_interaction_repair_prompt_explains_required_evidence_shape(
    tmp_path: Path,
) -> None:
    bench = load_module()
    make_bundle(tmp_path / "olist")
    sources = bench.discover_sources(tmp_path / "olist")
    current = {
        "title": "Promotion response differs by category",
        "discovery": {
            "dimensions_tested": ["month", "category"],
            "pattern_type": "interaction",
        },
    }
    error = "insight 3 interaction requires effect heterogeneity evidence"

    prompt = bench.build_targeted_insight_repair_prompt(
        sources,
        {"candidates": []},
        current,
        error,
    )
    lowered = " ".join(prompt.lower().split())

    assert "interaction_evidence" in prompt
    assert "cells" in lowered
    assert "cell" in lowered
    assert "effect" in lowered
    assert "sample_size" in lowered
    assert "heterogeneity" in lowered
    assert "baseline_effect" in lowered
    assert "at least two" in lowered
    assert "change pattern_type" in prompt
    assert "no invented evidence" in lowered


def test_targeted_metric_component_repair_prompt_requires_source_recomputation(
    tmp_path: Path,
) -> None:
    bench = load_module()
    make_bundle(tmp_path / "olist")
    sources = bench.discover_sources(tmp_path / "olist")
    error = (
        "insight 5 metric component 'denominator' verification must recompute "
        "metric_value from source data"
    )

    prompt = bench.build_targeted_insight_repair_prompt(
        sources,
        {"candidates": []},
        {
            "title": "Concentration",
            "discovery": {
                "dimensions_tested": ["product_id"],
                "pattern_type": "subgroup",
            },
        },
        error,
    )
    lowered = " ".join(prompt.lower().split())

    assert "denominator" in prompt
    assert "metric_value" in prompt
    assert "declared source" in lowered
    assert "literal expected value" in lowered
    assert "one source column or one aggregate" in lowered


def test_targeted_statement_repair_prompt_requests_only_the_failing_field() -> None:
    bench = load_module()
    insight = {
        "title": "Seller concentration",
        "statement": "18 sellers account for 19.8% of GMV.",
        "metric_spec": {"type": "share", "expected_value": 0.198},
        "supporting_claims": [{"claim": "18 sellers exceed the threshold"}],
        "interpretation": "must not be regenerated",
    }
    error = "insight 3 statement must contain one primary measured claim"

    prompt = bench.build_targeted_statement_repair_prompt(insight, error)
    lowered = " ".join(prompt.lower().split())

    assert error in prompt
    assert insight["statement"] not in prompt
    assert insight["title"] in prompt
    assert '"expected_value":0.198' in prompt
    assert '"type":"share"' in prompt
    assert "18 sellers exceed the threshold" not in prompt
    assert "must not be regenerated" not in prompt
    assert "statement: str" in lowered
    assert "one measured fact" in lowered
    assert "exactly one numeric literal" in lowered
    assert "component values" in lowered
    assert "do not repeat" in lowered
    assert len(prompt) < 6000


@pytest.mark.parametrize(
    "error",
    [
        "insight 3 statement must contain one primary measured claim",
    ],
)
def test_statement_errors_route_to_field_only_repair(error: str) -> None:
    bench = load_module()

    assert bench.is_targeted_statement_error(error)


@pytest.mark.parametrize(
    "error",
    [
        "insight 3 supporting claim must be measurable",
        "insight 3 metric_spec expected_value does not reconcile",
        "insight 5 quantitative facts belong in verified supporting_claims",
        "candidate 3 quantitative facts belong in verified supporting_claims",
    ],
)
def test_non_statement_errors_do_not_route_to_field_only_repair(error: str) -> None:
    bench = load_module()

    assert not bench.is_targeted_statement_error(error)


def test_interpretation_error_routes_to_field_only_repair() -> None:
    bench = load_module()

    assert bench.is_targeted_interpretation_error(
        "INSIGHT 5 QUANTITATIVE FACTS BELONG IN VERIFIED SUPPORTING_CLAIMS"
    )
    assert not bench.is_targeted_interpretation_error(
        "candidate 5 quantitative facts belong in verified supporting_claims"
    )


def test_targeted_interpretation_prompt_requests_only_unmeasured_synthesis() -> None:
    bench = load_module()
    insight = {
        "title": "Installment depth",
        "interpretation": "Credit cards have 76,795 rows averaging 163 BRL.",
        "statement": "Average payment value is 303 BRL higher.",
        "supporting_claims": [{"claim": "Credit cards have 76,795 rows."}],
        "metric_spec": {"type": "delta", "expected_value": 303},
    }
    error = "insight 5 quantitative facts belong in verified supporting_claims"

    prompt = bench.build_targeted_interpretation_repair_prompt(insight, error)
    lowered = " ".join(prompt.lower().split())

    assert error in prompt
    assert insight["interpretation"] in prompt
    assert bench._compact_json(insight["statement"]) in prompt
    assert bench._compact_json(insight["supporting_claims"]) in prompt
    assert "metric_spec" not in prompt
    assert "interpretation: str" in lowered
    assert "no quantitative facts" in lowered
    assert "no digits" in lowered
    assert "number words" in lowered
    assert "preserve" in lowered


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            "insight 3 supporting claim 2 verification is invalid",
            {
                "title": "stable",
                "statement": "stable statement",
                "supporting_claims": [{"id": "old-1"}, {"id": "new-2"}],
                "metric_spec": {"id": "old-metric"},
            },
        ),
        (
            "insight 3 metric_spec expected_value is invalid",
            {
                "title": "stable",
                "statement": "stable statement",
                "supporting_claims": [{"id": "old-1"}, {"id": "old-2"}],
                "metric_spec": {"id": "new-metric"},
            },
        ),
        (
            "insight 3 diagnostic explanation 2 is invalid",
            {
                "title": "stable",
                "statement": "stable statement",
                "supporting_claims": [{"id": "old-1"}, {"id": "old-2"}],
                "metric_spec": {"id": "old-metric"},
                "diagnostic_assessment": {
                    "explanations": [{"id": "old-e1"}, {"id": "new-e2"}]
                },
            },
        ),
    ],
)
def test_targeted_merge_replaces_only_verifier_identified_leaf(
    error: str, expected: dict
) -> None:
    bench = load_module()
    current = {
        "title": "stable",
        "statement": "stable statement",
        "supporting_claims": [{"id": "old-1"}, {"id": "old-2"}],
        "metric_spec": {"id": "old-metric"},
    }
    repaired = {
        "title": "regressed",
        "statement": "regressed statement with 1 and 2",
        "supporting_claims": [{"id": "new-1"}, {"id": "new-2"}],
        "metric_spec": {"id": "new-metric"},
        "diagnostic_assessment": {
            "explanations": [{"id": "new-e1"}, {"id": "new-e2"}]
        },
    }
    if "diagnostic explanation" in error:
        current["diagnostic_assessment"] = {
            "explanations": [{"id": "old-e1"}, {"id": "old-e2"}]
        }

    merged = bench.merge_targeted_insight_repair(current, repaired, error)

    assert merged == expected


def test_targeted_merge_falls_back_to_full_replacement_for_unaddressed_error() -> None:
    bench = load_module()
    repaired = {"title": "replacement"}

    assert bench.merge_targeted_insight_repair(
        {"title": "current"}, repaired, "insight 1 is invalid"
    ) == repaired


def test_targeted_merge_replaces_only_named_metric_component() -> None:
    bench = load_module()
    current = {
        "title": "stable",
        "metric_spec": {
            "expected_value": 0.25,
            "components": [
                {"name": "numerator", "expected_value": 25},
                {"name": "denominator", "expected_value": 100},
            ],
        },
    }
    repaired = {
        "title": "regressed",
        "metric_spec": {
            "expected_value": 0.5,
            "components": [
                {"name": "numerator", "expected_value": 50},
                {
                    "name": "denominator",
                    "expected_value": 100,
                    "verification": {"expression": "SELECT SUM(value) AS metric_value"},
                },
            ],
        },
    }

    merged = bench.merge_targeted_insight_repair(
        current,
        repaired,
        "insight 5 metric component 'denominator' verification must recompute "
        "metric_value from source data",
    )

    assert merged["title"] == "stable"
    assert merged["metric_spec"]["expected_value"] == 0.25
    assert merged["metric_spec"]["components"][0] == {
        "name": "numerator",
        "expected_value": 25,
    }
    assert merged["metric_spec"]["components"][1] == repaired["metric_spec"][
        "components"
    ][1]


@pytest.mark.parametrize(
    ("error", "target"),
    [
        (
            "candidate 7 quantitative rejection verification must recompute "
            "metric_value from source data",
            "scaffold",
        ),
        ("candidates ledger has an invalid rejection", "scaffold"),
        ("analysis_plan.search_space is missing a population", "scaffold"),
        ("kpi_map must name its source field", "scaffold"),
        ("dimensions_available conflicts with dimensions_deferred", "scaffold"),
        ("deferred dimension Specification/model is not available", "scaffold"),
        ("promotion lineage for candidate 2 is incomplete", "scaffold"),
        ("insight 1 must include a diagnostic", "insights"),
        ("metric_spec denominator is not measurable", "insights"),
        ("supporting claim statement lacks confidence", "insights"),
        ("interpretation action priority is incomplete", "insights"),
        ("discovery dimensions conflict with causal language", "insights"),
        ("unknown portable contract problem", "insights"),
        ("", "insights"),
    ],
)
def test_classify_repair_target_routes_contract_errors(
    error: str, target: str
) -> None:
    bench = load_module()

    assert bench.classify_repair_target(error) == target


def test_scaffold_repair_prompt_embeds_inputs_and_quantitative_repair_rules(
    tmp_path: Path,
) -> None:
    bench = load_module()
    make_bundle(tmp_path / "olist")
    sources = bench.discover_sources(tmp_path / "olist")
    research = valid_research()
    scaffold = {
        "analysis_plan": {"search_space": {"dimensions_available": ["state"]}},
        "candidates": [{"candidate": "Delivery", "disposition": "promoted"}],
    }
    insights = {"insights": [{"title": "Delivery"}]}
    error = (
        "candidate 7 quantitative rejection verification must recompute "
        "metric_value from source data"
    )

    prompt = bench.build_scaffold_repair_prompt(
        sources, research, scaffold, insights, error
    )
    lowered = " ".join(prompt.lower().split())

    for value in (research, scaffold, insights):
        assert bench._compact_json(value) in prompt
    assert error in prompt
    assert "return exactly" in lowered
    assert "analysis_plan" in lowered and "candidates" in lowered
    assert "minimum contract correction" in lowered
    assert "preserve valid promotions" in lowered
    assert "titles" in lowered and "dimensions" in lowered
    assert "unless implicated" in lowered
    assert "derived effects" in lowered
    assert "source-derived checks" in lowered
    assert "honestly reclassified" in lowered
    assert "allowed rejection type" in lowered
    assert "never substitute an unrelated metric" in lowered
    assert "call submit immediately" in lowered


def test_cached_insights_repair_persists_before_reverify_then_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {"grain": "order"}, "candidates": []}
    original = {"insights": [{"title": "original"}]}
    repaired = {"insights": [{"title": "repaired"}]}
    caches = write_all_caches(tmp_path, research, scaffold, original)
    events: list[str] = []
    verify_count = 0

    def verifier(payload):
        nonlocal verify_count
        verify_count += 1
        events.append(f"verify:{payload['insights'][0]['title']}")
        if verify_count == 1:
            raise AssertionError("exact portable failure")

    calls, lm = install_cached_repair_fakes(
        bench, monkeypatch, [repaired], verifier, events
    )
    original_atomic_json = bench._atomic_json

    def recording_atomic_json(path, value):
        if Path(path) == caches[2]:
            events.append("persist:insights")
        original_atomic_json(path, value)

    monkeypatch.setattr(bench, "_atomic_json", recording_atomic_json)

    record = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
        repair_turns=5,
        max_insight_repairs=4,
    )

    assert events == [
        "verify:original",
        "persist:insights",
        "verify:repaired",
        "audit-open",
        "audit",
    ]
    assert record["payload"]["insights"] == repaired["insights"]
    assert record["repairs"] == [
        {
            "target": "insights",
            "attempt": 1,
            "mode": "full",
            "insights": {"trajectory": "repair-1", "turns": 2},
        }
    ]
    assert len(calls) == 1
    assert calls[0]["outputs"] == bench.INSIGHT_OUTPUTS
    assert calls[0]["lm"] is lm
    assert calls[0]["skills"] == list(bench.SYNTHESIS_SKILLS)
    assert calls[0]["enable_verifier"] is False
    assert calls[0]["block_network"] is True
    assert calls[0]["verbose"] is False
    assert calls[0]["max_turns"] == 5
    assert calls[0]["reserve_finalize_turns"] == 3
    assert json.loads(caches[2].read_text(encoding="utf-8")) == {
        "input_fingerprint": synthesis_fingerprint(research, scaffold),
        "partial": repaired,
    }

    resumed_calls, _ = install_cached_repair_fakes(
        bench, monkeypatch, [], lambda payload: None
    )
    resumed = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
    )
    assert resumed_calls == []
    assert resumed["repairs"] == []
    assert resumed["payload"]["insights"] == repaired["insights"]


def test_multiple_repairs_use_latest_output_and_verifier_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {"grain": "order"}, "candidates": []}
    original = {"insights": [{"title": "v0"}]}
    repair_one = {"insights": [{"title": "v1"}]}
    repair_two = {"insights": [{"title": "v2"}]}
    caches = write_all_caches(tmp_path, research, scaffold, original)
    errors = iter(("first exact error", "second exact error", None))

    def verifier(payload):
        error = next(errors)
        if error:
            raise AssertionError(error)

    calls, _ = install_cached_repair_fakes(
        bench, monkeypatch, [repair_one, repair_two], verifier
    )

    record = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
    )

    assert "first exact error" in calls[0]["task"]
    assert bench._compact_json(original) in calls[0]["task"]
    assert "second exact error" in calls[1]["task"]
    assert bench._compact_json(repair_one) in calls[1]["task"]
    assert record["payload"]["insights"] == repair_two["insights"]
    assert len(record["repairs"]) == 2


def test_targeted_repair_replaces_only_index_persists_before_verify_and_reuses_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {
        "analysis_plan": {"grain": "order"},
        "candidates": [
            {
                "disposition": "promoted",
                "promoted_as": title,
                "dimensions_tested": ["seller_id"],
            }
            for title in ("First title", "Target title", "Third title")
        ],
    }
    original = {
        "insights": [
            {"title": "First title", "version": "untouched-a"},
            {
                "title": "Target title",
                "version": "v0",
                "interpretation": "20 of 100 orders are concentrated.",
                "discovery": {"dimensions_tested": ["seller_id"]},
            },
            {"title": "Third title", "version": "untouched-c"},
        ]
    }
    targeted_interpretation = {
        "interpretation": "Orders are materially concentrated.",
    }
    targeted_two = {
        "insight": {
            "title": "Target title",
            "statement": "Target metric is 20%.",
            "version": "v2",
            "discovery": {"dimensions_tested": ["seller_id"]},
        }
    }
    original_snapshot = json.loads(json.dumps(original))
    caches = write_all_caches(tmp_path, research, scaffold, original)
    events: list[str] = []
    errors = iter(
        (
            "insight 2 quantitative facts belong in verified supporting_claims",
            "INSIGHT 2 supporting claim must be measurable",
            None,
        )
    )

    def verifier(payload):
        events.append(
            "verify:" + ",".join(item["version"] for item in payload["insights"])
        )
        error = next(errors)
        if error:
            raise AssertionError(error)

    calls, _ = install_cached_repair_fakes(
        bench, monkeypatch, [targeted_interpretation, targeted_two], verifier, events
    )
    original_atomic_json = bench._atomic_json

    def recording_atomic_json(path, value):
        if Path(path) == caches[2]:
            persisted = value["partial"]["insights"]
            events.append("persist:" + ",".join(item["version"] for item in persisted))
        original_atomic_json(path, value)

    monkeypatch.setattr(bench, "_atomic_json", recording_atomic_json)

    record = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
    )

    assert events[:5] == [
        "verify:untouched-a,v0,untouched-c",
        "persist:untouched-a,v0,untouched-c",
        "verify:untouched-a,v0,untouched-c",
        "persist:untouched-a,v2,untouched-c",
        "verify:untouched-a,v2,untouched-c",
    ]
    assert [call["outputs"] for call in calls] == [
        bench.TARGETED_INTERPRETATION_OUTPUTS,
        bench.TARGETED_INSIGHT_OUTPUTS,
    ]
    assert targeted_interpretation["interpretation"] in calls[1]["task"]
    assert "First title" not in calls[0]["task"]
    assert "Third title" not in calls[0]["task"]
    assert original == original_snapshot
    assert record["payload"]["insights"] == [
        original["insights"][0],
        targeted_two["insight"],
        original["insights"][2],
    ]
    assert record["repairs"] == [
        {
            "target": "insights",
            "attempt": 1,
            "mode": "targeted-interpretation",
            "insight_index": 2,
            "insights": {"trajectory": "repair-1", "turns": 2},
        },
        {
            "target": "insights",
            "attempt": 2,
            "mode": "targeted",
            "insight_index": 2,
            "insights": {"trajectory": "repair-2", "turns": 2},
        },
    ]
    checkpoint = json.loads(caches[2].read_text(encoding="utf-8"))
    assert checkpoint["partial"]["insights"] == record["payload"]["insights"]


def test_repair_exhaustion_names_target_attempts_and_preserves_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {}, "candidates": []}
    original = {"insights": []}
    caches = write_all_caches(tmp_path, research, scaffold, original)
    errors = iter(("initial", "after one", "final verifier message"))

    def verifier(payload):
        raise AssertionError(next(errors))

    calls, _ = install_cached_repair_fakes(
        bench,
        monkeypatch,
        [{"insights": [{"title": "v1"}]}, {"insights": [{"title": "v2"}]}],
        verifier,
    )

    with pytest.raises(
        AssertionError,
        match="insights.*2 repair attempts.*final verifier message",
    ) as captured:
        bench.run_staged_benchmark(
            data_dir,
            research_cache_path=caches[0],
            scaffold_cache_path=caches[1],
            insights_cache_path=caches[2],
            max_insight_repairs=2,
        )

    assert len(calls) == 2
    assert isinstance(captured.value.__cause__, AssertionError)
    assert str(captured.value.__cause__) == "final verifier message"


def test_candidate_error_repairs_scaffold_then_regenerates_insights_before_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    old_scaffold = {"analysis_plan": {"grain": "old"}, "candidates": []}
    old_insights = {"insights": [{"title": "old"}]}
    new_scaffold = {
        "analysis_plan": {"grain": "order"},
        "candidates": [{"candidate": "new"}],
    }
    new_insights = {"insights": [{"title": "new"}]}
    caches = write_all_caches(tmp_path, research, old_scaffold, old_insights)
    events: list[str] = []
    verify_count = 0

    def verifier(payload):
        nonlocal verify_count
        verify_count += 1
        events.append(f"verify:{payload['insights'][0]['title']}")
        if verify_count == 1:
            raise AssertionError(
                "candidate 7 quantitative rejection verification must "
                "recompute metric_value from source data"
            )

    calls, _ = install_cached_repair_fakes(
        bench, monkeypatch, [new_scaffold, new_insights], verifier, events
    )
    original_atomic_json = bench._atomic_json

    def recording_atomic_json(path, value):
        if Path(path) == caches[1]:
            events.append("persist:scaffold")
        elif Path(path) == caches[2]:
            events.append("persist:insights")
        original_atomic_json(path, value)

    monkeypatch.setattr(bench, "_atomic_json", recording_atomic_json)

    record = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
    )

    assert events == [
        "verify:old",
        "persist:scaffold",
        "persist:insights",
        "verify:new",
        "audit-open",
        "audit",
    ]
    assert [call["outputs"] for call in calls] == [
        bench.SCAFFOLD_OUTPUTS,
        bench.INSIGHT_OUTPUTS,
    ]
    assert "Repair only the contract scaffold" in calls[0]["task"]
    assert "Create only the insights portion" in calls[1]["task"]
    assert "Repair only the insights portion" not in calls[1]["task"]
    assert record["repairs"] == [
        {
            "target": "scaffold",
            "attempt": 1,
            "mode": "full",
            "scaffold": {"trajectory": "repair-1", "turns": 2},
            "insights": {"trajectory": "repair-2", "turns": 2},
        }
    ]
    assert json.loads(caches[1].read_text(encoding="utf-8")) == {
        "input_fingerprint": synthesis_fingerprint(research),
        "partial": new_scaffold,
    }
    assert json.loads(caches[2].read_text(encoding="utf-8")) == {
        "input_fingerprint": synthesis_fingerprint(research, new_scaffold),
        "partial": new_insights,
    }

    resumed_calls, _ = install_cached_repair_fakes(
        bench, monkeypatch, [], lambda payload: None
    )
    resumed = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
    )
    assert resumed_calls == []
    assert resumed["payload"]["insights"] == new_insights["insights"]


def test_repair_target_transitions_use_separate_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {}, "candidates": []}
    insights = {"insights": []}
    caches = write_all_caches(tmp_path, research, scaffold, insights)
    repaired_insights = {"insights": [{"title": "insight repair"}]}
    repaired_scaffold = {"analysis_plan": {"grain": "order"}, "candidates": []}
    regenerated_insights = {"insights": [{"title": "regenerated"}]}
    errors = iter(
        (
            "insight 1 diagnostic is incomplete",
            "candidate 2 rejection is invalid",
            None,
        )
    )

    def verifier(payload):
        error = next(errors)
        if error:
            raise AssertionError(error)

    calls, _ = install_cached_repair_fakes(
        bench,
        monkeypatch,
        [repaired_insights, repaired_scaffold, regenerated_insights],
        verifier,
    )

    record = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
        max_insight_repairs=1,
        max_scaffold_repairs=1,
    )

    assert [repair["target"] for repair in record["repairs"]] == [
        "insights",
        "scaffold",
    ]
    assert [repair["attempt"] for repair in record["repairs"]] == [1, 1]
    assert [call["outputs"] for call in calls] == [
        bench.INSIGHT_OUTPUTS,
        bench.SCAFFOLD_OUTPUTS,
        bench.INSIGHT_OUTPUTS,
    ]


def test_non_assertion_portable_failure_does_not_start_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {}, "candidates": []}
    caches = write_all_caches(tmp_path, research, scaffold, {"insights": []})

    def verifier(payload):
        raise RuntimeError("verifier infrastructure failed")

    calls, _ = install_cached_repair_fakes(bench, monkeypatch, [], verifier)

    with pytest.raises(RuntimeError, match="verifier infrastructure failed"):
        bench.run_staged_benchmark(
            data_dir,
            research_cache_path=caches[0],
            scaffold_cache_path=caches[1],
            insights_cache_path=caches[2],
        )

    assert calls == []


def test_run_staged_benchmark_wires_three_fresh_rlms_verifies_then_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    calls: dict = {"from_task": [], "instances": [], "summaries": [], "events": []}

    class FakeDspy:
        @staticmethod
        def LM(model, **kwargs):
            calls["lm"] = (model, kwargs)
            return object()

        @staticmethod
        def configure(**kwargs):
            calls["configure"] = kwargs

    research = valid_research()
    scaffold = {
        "analysis_plan": {"grain": "order"},
        "candidates": [],
    }
    insights = {"insights": []}

    class FakeResult:
        def __init__(self, result_payload, trajectory):
            self.payload = result_payload
            self.trajectory = trajectory

    results = [
        FakeResult({"research_json": json.dumps(research)}, "research-trajectory"),
        FakeResult(scaffold, "scaffold-trajectory"),
        FakeResult(insights, "insights-trajectory"),
    ]

    class FakeRLMInstance:
        def __init__(self, result):
            self.result = result

        def run(self):
            return self.result

    class FakeRLM:
        @classmethod
        def from_task(cls, **kwargs):
            if len(calls["instances"]) == 1:
                checkpoint = tmp_path / "research-cache.json"
                assert json.loads(checkpoint.read_text(encoding="utf-8")) == research
            calls["from_task"].append(kwargs)
            instance = FakeRLMInstance(results[len(calls["instances"])])
            calls["instances"].append(instance)
            return instance

    class FakeExecutor:
        def __init__(self, sources):
            calls["events"].append("executor")
            calls["executor_sources"] = sources

        def __enter__(self):
            calls["executor_entered"] = True
            return self

        def __exit__(self, *exc_info):
            calls["executor_closed"] = True

    def fake_audit(actual_payload, executor):
        calls["events"].append("audit")
        calls["audit"] = (actual_payload, executor)
        return FakeAudit((FakeCheck("insights[0]", 1, 1),))

    def summarize(trajectory):
        calls["summaries"].append(trajectory)
        return {"turns": {"research-trajectory": 7, "scaffold-trajectory": 5, "insights-trajectory": 6}[trajectory]}

    monkeypatch.setattr(
        bench,
        "_load_runtime_dependencies",
        lambda: (FakeDspy, FakeRLM, FakeExecutor, fake_audit, summarize),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")
    monkeypatch.setattr(
        bench,
        "verify_portable_contract",
        lambda payload: calls["events"].append("portable"),
    )

    record = bench.run_staged_benchmark(
        data_dir,
        model="openrouter/test",
        api_base="https://example.invalid/v1",
        research_turns=17,
        scaffold_turns=9,
        insight_turns=13,
        timeout=900,
        research_cache_path=tmp_path / "research-cache.json",
    )

    assert len(calls["instances"]) == 3
    assert len({id(instance) for instance in calls["instances"]}) == 3
    research_call, scaffold_call, insight_call = calls["from_task"]
    assert research_call["outputs"] == {"research_json": str}
    assert research_call["skills"] == ["data_exploration"]
    assert research_call["enable_verifier"] is False
    assert research_call["block_network"] is True
    assert research_call["max_turns"] == 17
    assert research_call["reserve_finalize_turns"] == 4
    assert research_call["verbose"] is False
    assert scaffold_call["outputs"] == {
        "analysis_plan": dict,
        "candidates": list,
    }
    assert scaffold_call["skills"] == [
        "deep_insight_discovery",
        "data_exploration",
    ]
    assert scaffold_call["enable_verifier"] is False
    assert scaffold_call["block_network"] is True
    assert scaffold_call["max_turns"] == 9
    assert scaffold_call["reserve_finalize_turns"] == 4
    assert scaffold_call["verbose"] is False
    assert insight_call["outputs"] == {"insights": list}
    assert insight_call["skills"] == [
        "deep_insight_discovery",
        "data_exploration",
    ]
    assert insight_call["enable_verifier"] is False
    assert insight_call["block_network"] is True
    assert insight_call["max_turns"] == 13
    assert insight_call["reserve_finalize_turns"] == 6
    assert insight_call["verbose"] is False
    assert all(call["timeout"] == 900 for call in calls["from_task"])
    assert research_call["lm"] is scaffold_call["lm"] is insight_call["lm"]
    assert calls["lm"][1] == {
        "api_key": "test-only-secret",
        "api_base": "https://example.invalid/v1",
        "max_tokens": 20000,
        "temperature": 0,
        "cache": False,
        "reasoning": {"max_tokens": 4096, "exclude": True},
    }
    assert calls["executor_entered"] is True
    assert calls["executor_closed"] is True
    assert calls["events"] == ["portable", "executor", "audit"]
    assert calls["audit"][0] is record["payload"]
    assert calls["summaries"] == [
        "research-trajectory",
        "scaffold-trajectory",
        "insights-trajectory",
    ]
    assert record["research"] == research
    assert record["model"] == "openrouter/test"
    assert record["stage_skills"] == {
        "research": ("data_exploration",),
        "contract_scaffold": ("deep_insight_discovery", "data_exploration"),
        "insights": ("deep_insight_discovery", "data_exploration"),
    }
    assert record["repairs"] == []
    assert json.loads(
        (tmp_path / "research-cache.json").read_text(encoding="utf-8")
    ) == research


def test_cached_research_skips_research_rlm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    cache = tmp_path / "research.json"
    cache.write_text(json.dumps(valid_research()), encoding="utf-8")
    calls = {"from_task": []}

    class FakeDspy:
        @staticmethod
        def LM(*args, **kwargs):
            return object()

        @staticmethod
        def configure(**kwargs):
            pass

    class FakeRLM:
        @classmethod
        def from_task(cls, **kwargs):
            calls["from_task"].append(kwargs)
            index = len(calls["from_task"])
            payload = (
                {"analysis_plan": {}, "candidates": []}
                if index == 1
                else {"insights": []}
            )
            result = type(
                "Result",
                (),
                {"payload": payload, "trajectory": f"downstream-{index}"},
            )()
            return type("Instance", (), {"run": lambda self: result})()

    class FakeExecutor:
        def __init__(self, sources):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            pass

    monkeypatch.setattr(
        bench,
        "_load_runtime_dependencies",
        lambda: (
            FakeDspy,
            FakeRLM,
            FakeExecutor,
            lambda payload, executor: FakeAudit(()),
            lambda trajectory: {"turns": 4},
        ),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")
    monkeypatch.setattr(bench, "verify_portable_contract", lambda payload: None)

    record = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=cache,
    )

    assert len(calls["from_task"]) == 2
    assert [call["outputs"] for call in calls["from_task"]] == [
        {"analysis_plan": dict, "candidates": list},
        {"insights": list},
    ]
    assert record["research"] == valid_research()
    assert record["trajectories"]["research"] == {
        "cached": True,
        "submitted": True,
        "turns": 0,
    }
    assert set(record["trajectories"]) == {
        "research",
        "contract_scaffold",
        "insights",
    }


def test_synthesis_checkpoints_are_written_before_portable_verifier_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    research_cache = tmp_path / "research.json"
    research_cache.write_text(json.dumps(research), encoding="utf-8")
    scaffold = {"analysis_plan": {"grain": "order"}, "candidates": []}
    insights = {"insights": [{"title": "incomplete"}]}
    scaffold_cache = tmp_path / "contract_scaffold.checkpoint.json"
    insights_cache = tmp_path / "insights.checkpoint.json"
    events: list[str] = []
    install_synthesis_fakes(
        bench, monkeypatch, [scaffold, insights], events
    )
    original_atomic_json = bench._atomic_json

    def recording_atomic_json(path, value):
        events.append(f"persist:{Path(path).name}")
        original_atomic_json(path, value)

    monkeypatch.setattr(bench, "_atomic_json", recording_atomic_json)

    def fail_verifier(payload):
        events.append("verify")
        raise AssertionError("measured claim")

    monkeypatch.setattr(bench, "verify_portable_contract", fail_verifier)

    with pytest.raises(AssertionError, match="measured claim"):
        bench.run_staged_benchmark(
            data_dir,
            research_cache_path=research_cache,
            scaffold_cache_path=scaffold_cache,
            insights_cache_path=insights_cache,
            max_insight_repairs=0,
        )

    assert json.loads(scaffold_cache.read_text(encoding="utf-8")) == {
        "input_fingerprint": synthesis_fingerprint(research),
        "partial": scaffold,
    }
    assert json.loads(insights_cache.read_text(encoding="utf-8")) == {
        "input_fingerprint": synthesis_fingerprint(research, scaffold),
        "partial": insights,
    }
    assert events == [
        "persist:contract_scaffold.checkpoint.json",
        "persist:insights.checkpoint.json",
        "verify",
    ]


def test_matching_synthesis_caches_skip_all_rlm_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {"grain": "order"}, "candidates": []}
    insights = {"insights": []}
    research_cache = tmp_path / "research.json"
    scaffold_cache = tmp_path / "scaffold.json"
    insights_cache = tmp_path / "insights.json"
    research_cache.write_text(json.dumps(research), encoding="utf-8")
    write_checkpoint(scaffold_cache, synthesis_fingerprint(research), scaffold)
    write_checkpoint(
        insights_cache, synthesis_fingerprint(research, scaffold), insights
    )
    calls = install_synthesis_fakes(bench, monkeypatch, [])

    record = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=research_cache,
        scaffold_cache_path=scaffold_cache,
        insights_cache_path=insights_cache,
    )

    assert calls == []
    cached = {"cached": True, "submitted": True, "turns": 0}
    assert record["trajectories"] == {
        "research": cached,
        "contract_scaffold": cached,
        "insights": cached,
    }
    assert record["repairs"] == []


def test_cached_mechanical_repairs_persist_before_verify_and_resume_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {"grain": "order"}, "candidates": []}
    insights = {"insights": [mechanical_insight()]}
    caches = write_all_caches(tmp_path, research, scaffold, insights)
    events: list[str] = []
    calls = install_synthesis_fakes(bench, monkeypatch, [], events)
    original_atomic_json = bench._atomic_json

    def recording_atomic_json(path, value):
        if Path(path) == caches[2]:
            events.append("persist:insights")
        original_atomic_json(path, value)

    monkeypatch.setattr(bench, "_atomic_json", recording_atomic_json)

    record = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
    )

    assert calls == []
    assert events[:2] == ["persist:insights", "verify"]
    assert record["repairs"] == []
    assert record["mechanical_repairs"]["count"] == 7
    assert record["mechanical_repairs"]["paths"] == [
        "$.insights[0].diagnostic_assessment.explanations[1].verification.sources.order_reviews",
        "$.insights[0].diagnostic_assessment.explanations[1].verification.sources.orders",
        "$.insights[0].diagnostic_measurability",
        "$.insights[0].diagnostic_assessment.decision_readiness",
        "$.insights[0].action.kind",
        "$.insights[0].confidence.level",
        "$.insights[0].priority.urgency",
    ]
    cached_insights = json.loads(caches[2].read_text(encoding="utf-8"))["partial"]
    assert cached_insights == {"insights": record["payload"]["insights"]}

    resumed_calls = install_synthesis_fakes(bench, monkeypatch, [])
    resumed = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
    )
    assert resumed_calls == []
    assert resumed["mechanical_repairs"] == {"count": 0, "paths": []}


def test_cached_scaffold_normalization_migrates_dependent_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {
        "analysis_plan": {"grain": "order"},
        "candidates": [
            {
                "verification": {
                    "method": "sql",
                    "expression": "SELECT COUNT(*) FROM orders",
                    "sources": {},
                }
            }
        ],
    }
    insights = {"insights": []}
    caches = write_all_caches(tmp_path, research, scaffold, insights)
    events: list[str] = []
    calls = install_synthesis_fakes(bench, monkeypatch, [], events)
    original_atomic_json = bench._atomic_json

    def recording_atomic_json(path, value):
        if Path(path) == caches[1]:
            events.append("persist:scaffold")
        elif Path(path) == caches[2]:
            events.append("persist:insights")
        original_atomic_json(path, value)

    monkeypatch.setattr(bench, "_atomic_json", recording_atomic_json)

    record = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
    )

    assert calls == []
    assert events[:3] == ["persist:scaffold", "persist:insights", "verify"]
    normalized_scaffold = {
        "analysis_plan": scaffold["analysis_plan"],
        "candidates": [
            {
                "verification": {
                    "method": "sql",
                    "expression": "SELECT COUNT(*) FROM orders",
                    "sources": {"orders": "orders"},
                }
            }
        ],
    }
    assert record["payload"]["candidates"] == normalized_scaffold["candidates"]
    assert json.loads(caches[1].read_text(encoding="utf-8"))["partial"] == (
        normalized_scaffold
    )
    assert json.loads(caches[2].read_text(encoding="utf-8")) == {
        "input_fingerprint": synthesis_fingerprint(research, normalized_scaffold),
        "partial": insights,
    }


def test_stale_scaffold_reruns_scaffold_and_dependent_insights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    old_scaffold = {"analysis_plan": {"grain": "old"}, "candidates": []}
    new_scaffold = {"analysis_plan": {"grain": "order"}, "candidates": []}
    new_insights = {"insights": []}
    research_cache = tmp_path / "research.json"
    scaffold_cache = tmp_path / "scaffold.json"
    insights_cache = tmp_path / "insights.json"
    research_cache.write_text(json.dumps(research), encoding="utf-8")
    write_checkpoint(scaffold_cache, "stale", old_scaffold)
    write_checkpoint(
        insights_cache,
        synthesis_fingerprint(research, old_scaffold),
        {"insights": [{"title": "old"}]},
    )
    calls = install_synthesis_fakes(
        bench, monkeypatch, [new_scaffold, new_insights]
    )

    record = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=research_cache,
        scaffold_cache_path=scaffold_cache,
        insights_cache_path=insights_cache,
    )

    assert [call["outputs"] for call in calls] == [
        bench.SCAFFOLD_OUTPUTS,
        bench.INSIGHT_OUTPUTS,
    ]
    assert record["trajectories"]["contract_scaffold"]["turns"] == 3
    assert record["trajectories"]["insights"]["turns"] == 3


def test_stale_insights_reruns_only_insights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {"grain": "order"}, "candidates": []}
    insights = {"insights": []}
    research_cache = tmp_path / "research.json"
    scaffold_cache = tmp_path / "scaffold.json"
    insights_cache = tmp_path / "insights.json"
    research_cache.write_text(json.dumps(research), encoding="utf-8")
    write_checkpoint(scaffold_cache, synthesis_fingerprint(research), scaffold)
    write_checkpoint(insights_cache, "stale", {"insights": [{"title": "old"}]})
    calls = install_synthesis_fakes(bench, monkeypatch, [insights])

    record = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=research_cache,
        scaffold_cache_path=scaffold_cache,
        insights_cache_path=insights_cache,
    )

    assert [call["outputs"] for call in calls] == [bench.INSIGHT_OUTPUTS]
    assert record["trajectories"]["contract_scaffold"] == {
        "cached": True,
        "submitted": True,
        "turns": 0,
    }
    assert record["trajectories"]["insights"]["turns"] == 3


def test_malformed_matching_checkpoint_fails_instead_of_rerunning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    research_cache = tmp_path / "research.json"
    scaffold_cache = tmp_path / "scaffold.json"
    research_cache.write_text(json.dumps(research), encoding="utf-8")
    write_checkpoint(
        scaffold_cache,
        synthesis_fingerprint(research),
        {"analysis_plan": {}, "candidates": "not-a-list"},
    )
    calls = install_synthesis_fakes(bench, monkeypatch, [])

    with pytest.raises(ValueError, match="candidates must be list"):
        bench.run_staged_benchmark(
            data_dir,
            research_cache_path=research_cache,
            scaffold_cache_path=scaffold_cache,
        )

    assert calls == []


def test_main_uses_named_synthesis_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    captured = {}
    output_dir = tmp_path / "artifacts"
    monkeypatch.setattr(
        bench,
        "run_staged_benchmark",
        lambda data_dir, **kwargs: captured.update(kwargs)
        or {"audit": FakeAudit(())},
    )
    monkeypatch.setattr(bench, "write_staged_artifacts", lambda output, record: {})

    assert bench.main(
        ["--data-dir", str(tmp_path / "olist"), "--output-dir", str(output_dir)]
    ) == 0

    assert captured["scaffold_cache_path"] == (
        output_dir / "contract_scaffold.checkpoint.json"
    )
    assert captured["insights_cache_path"] == (
        output_dir / "insights.checkpoint.json"
    )
    assert captured["repair_turns"] == 6
    assert captured["max_insight_repairs"] == 4
    assert captured["max_scaffold_repairs"] == 3
    assert captured["max_audit_repairs"] == 6


def test_api_key_is_required_before_runtime_dependencies_are_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    make_bundle(tmp_path / "olist")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def unexpected_import():
        raise AssertionError("runtime dependencies loaded before key validation")

    monkeypatch.setattr(bench, "_load_runtime_dependencies", unexpected_import)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        bench.run_staged_benchmark(tmp_path / "olist")


def test_artifacts_are_deterministic_aggregate_only_and_credential_safe(
    tmp_path: Path,
) -> None:
    bench = load_module()
    record = {
        "research": valid_research(),
        "payload": {
            "contract_version": 2,
            "analysis_plan": {},
            "candidates": [{"title": "candidate"}],
            "insights": [{"title": "finding"}],
        },
        "audit": FakeAudit((FakeCheck("insights[0]", 1, 1),)),
        "trajectories": {
            "research": {"turns": 8, "submitted": True},
            "contract_scaffold": {"turns": 4, "submitted": True},
            "insights": {"turns": 6, "submitted": True},
        },
        "model": "openrouter/test",
        "stage_skills": {
            "research": ("data_exploration",),
            "contract_scaffold": (
                "deep_insight_discovery",
                "data_exploration",
            ),
            "insights": ("deep_insight_discovery", "data_exploration"),
        },
        "repairs": [
            {
                "target": "insights",
                "attempt": 1,
                "mode": "targeted",
                "insight_index": 1,
                "insights": {"turns": 2, "submitted": True},
            },
            {
                "target": "scaffold",
                "attempt": 1,
                "mode": "full",
                "scaffold": {"turns": 1, "submitted": True},
                "insights": {"turns": 3, "submitted": True},
            },
        ],
        "mechanical_repairs": {
            "count": 2,
            "paths": [
                "$.insights[0].verification.sources.orders",
                "$.insights[0].confidence.level",
            ],
        },
    }

    paths = bench.write_staged_artifacts(tmp_path / "artifacts", record)

    assert set(paths) == {"research", "payload", "audit", "run"}
    assert {path.name for path in paths.values()} == {
        "research.json",
        "payload.json",
        "audit.json",
        "run.json",
    }
    run_data = json.loads(paths["run"].read_text(encoding="utf-8"))
    assert run_data == {
        "status": "success",
        "model": "openrouter/test",
        "stage_skills": {
            "research": ["data_exploration"],
            "contract_scaffold": [
                "deep_insight_discovery",
                "data_exploration",
            ],
            "insights": ["deep_insight_discovery", "data_exploration"],
        },
        "turn_summaries": record["trajectories"],
        "repair_summaries": record["repairs"],
        "audit_repair_summaries": [],
        "mechanical_repairs": record["mechanical_repairs"],
        "counts": {
            "research_candidates": 1,
            "contract_candidates": 1,
            "insights": 1,
            "audit_checks": 1,
            "repairs": 2,
            "insight_repairs": 1,
            "scaffold_repairs": 1,
            "audit_repairs": 0,
            "mechanical_repairs": 2,
        },
    }
    combined = "".join(path.read_text(encoding="utf-8") for path in paths.values())
    lowered = combined.lower()
    assert "api_key" not in lowered
    assert "openrouter_api_key" not in lowered
    assert "trajectory\"" not in lowered
    assert not list((tmp_path / "artifacts").glob("*.tmp"))
    first = {name: path.read_bytes() for name, path in paths.items()}
    second = bench.write_staged_artifacts(tmp_path / "artifacts", record)
    assert {name: path.read_bytes() for name, path in second.items()} == first


def test_cli_defaults_are_staged_and_bounded() -> None:
    bench = load_module()

    args = bench.parse_args(["--data-dir", "olist"])

    assert args.research_turns == 18
    assert args.scaffold_turns == 10
    assert args.insight_turns == 14
    assert args.repair_turns == 6
    assert args.max_insight_repairs == 4
    assert args.max_scaffold_repairs == 3
    assert args.max_audit_repairs == 6
    assert args.timeout == 3600
    assert args.output_dir == Path("_local") / "olist_staged_deep_insight_benchmark"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "insights[1].supporting_claims[0].metric_spec.components[0]: "
            "expected 6555, actual 6597.88",
            (
                "insights[1].supporting_claims[0].metric_spec.components[0]",
                6555.0,
                6597.88,
            ),
        ),
        ("insights[0].metric_spec: expected -1.2e3, actual +4.5E-2", (
            "insights[0].metric_spec", -1200.0, 0.045
        )),
        ("prefix insights[0].metric_spec: expected 1, actual 2", None),
        ("insights[-1].metric_spec: expected 1, actual 2", None),
        ("insights[0].metric spec: expected 1, actual 2", None),
        ("insights[0].metric_spec: expected nan, actual 2", None),
        ("insights[0].metric_spec: expected 1, actual inf", None),
        ("executor failed at insights[0].metric_spec: boom", None),
        ("insights[0].metric_spec: expected 1, actual 2 trailing", None),
    ],
)
def test_parse_host_audit_mismatch_accepts_only_safe_finite_exact_format(
    message: str, expected: tuple[str, float, float] | None
) -> None:
    bench = load_module()

    parsed = bench.parse_host_audit_mismatch(message)

    if expected is None:
        assert parsed is None
    else:
        assert (parsed.path, parsed.expected, parsed.actual) == expected


def test_merge_host_audit_repair_replaces_only_supporting_claim_and_preserves_sql(
) -> None:
    bench = load_module()
    current = {
        "title": "Stable",
        "statement": "Stable primary statement.",
        "supporting_claims": [
            {
                "statement": "Expected 10 orders.",
                "expected_value": 10,
                "verification": {
                    "expression": "SELECT COUNT(*) FROM orders",
                    "sources": {"orders": "orders"},
                },
            },
            {"statement": "untouched sibling"},
        ],
    }
    replacement = {
        "statement": "Expected 12 orders.",
        "expected_value": 12,
        "verification": {
            "expression": "SELECT COUNT(*) FROM orders",
            "sources": {"orders": "orders"},
        },
    }
    mismatch = bench.parse_host_audit_mismatch(
        "insights[0].supporting_claims[0]: expected 10, actual 12"
    )

    merged = bench.merge_host_audit_repair(
        current, mismatch, {"supporting_claim": replacement}
    )

    assert merged["supporting_claims"] == [
        {
            **replacement,
            "statement": "Expected 12 orders.",
        },
        current["supporting_claims"][1],
    ]
    assert merged["title"] == current["title"]
    assert merged["statement"] == current["statement"]
    assert current["supporting_claims"][0]["expected_value"] == 10


def test_host_audit_repair_prompt_preserves_round_trip_actual_precision() -> None:
    bench = load_module()
    insight = {
        "title": "GMV",
        "supporting_claims": [
            {"expected_value": 1258681.0, "claim": "Rounded GMV"}
        ],
    }
    mismatch = bench.HostAuditMismatch(
        "insights[0].supporting_claims[0]",
        1258681.0,
        1258681.3399999682,
    )

    prompt = bench.build_host_audit_repair_prompt(
        insight,
        mismatch,
        ("supporting_claims", 0),
        "supporting_claim",
    )

    assert "1258681.3399999682" in prompt
    assert "1.25868e+06" not in prompt


def test_host_audit_metric_component_uses_exact_actual_and_preserves_siblings(
) -> None:
    bench = load_module()
    numerator_sql = {
        "expression": "SELECT SUM(amount) AS metric_value FROM sales",
        "sources": {"sales": "sales"},
    }
    denominator_sql = {
        "expression": "SELECT SUM(total) AS metric_value FROM sales",
        "sources": {"sales": "sales"},
    }
    current = {
        "title": "Stable share",
        "statement": "The measured share was 40.0%.",
        "metric_spec": {
            "type": "share",
            "expected_value": 0.4,
            "components": [
                {
                    "name": "numerator",
                    "role": "numerator",
                    "expected_value": 100.0,
                    "verification": numerator_sql,
                },
                {
                    "name": "denominator",
                    "role": "denominator",
                    "expected_value": 250.0,
                    "verification": denominator_sql,
                },
            ],
        },
    }
    repaired_metric = deepcopy(current["metric_spec"])
    repaired_metric["expected_value"] = 0.5
    repaired_metric["components"][0]["expected_value"] = 101
    repaired_metric["components"][0]["name"] = "denominator"
    repaired_metric["components"][0]["role"] = "denominator"
    repaired_metric["components"][1]["expected_value"] = 999
    repaired_metric["components"][1]["name"] = "numerator"
    repaired_metric["components"][1]["role"] = "numerator"
    repaired_metric["components"][1]["unit"] = "mutated"
    mismatch = bench.HostAuditMismatch(
        "insights[0].metric_spec.components[0]",
        100.0,
        100.66,
    )

    merged = bench.merge_host_audit_repair(
        current,
        mismatch,
        {"metric_spec": repaired_metric},
    )

    assert merged["metric_spec"]["expected_value"] == pytest.approx(100.66 / 250)
    assert merged["metric_spec"]["components"][0]["expected_value"] == 100.66
    assert merged["metric_spec"]["components"][1]["expected_value"] == 250.0
    assert merged["metric_spec"]["components"][0]["name"] == "numerator"
    assert merged["metric_spec"]["components"][0]["role"] == "numerator"
    assert merged["metric_spec"]["components"][1]["name"] == "denominator"
    assert merged["metric_spec"]["components"][1]["role"] == "denominator"
    assert "unit" not in merged["metric_spec"]["components"][1]
    assert merged["metric_spec"]["components"][0]["verification"] == numerator_sql
    assert merged["metric_spec"]["components"][1]["verification"] == denominator_sql


def test_host_audit_metric_component_updates_only_exact_equal_valued_sibling() -> None:
    bench = load_module()
    current = {
        "metric_spec": {
            "components": [
                {"name": "audited", "label": "Audited 10", "expected_value": 10},
                {"name": "sibling", "label": "Sibling 10", "expected_value": 10},
            ]
        }
    }

    merged = bench.merge_host_audit_repair(
        current,
        bench.HostAuditMismatch(
            "insights[0].metric_spec.components[0]",
            10,
            12,
        ),
        {
            "metric_spec": {
                "components": [
                    {"name": "audited", "label": "Audited 12", "expected_value": 12},
                    {"name": "sibling", "label": "Sibling 10", "expected_value": 10},
                ]
            }
        },
    )

    assert merged["metric_spec"]["components"] == [
        {"name": "audited", "label": "Audited 12", "expected_value": 12},
        {"name": "sibling", "label": "Sibling 10", "expected_value": 10},
    ]


def test_exact_numeric_replacement_formats_percentage_with_decimal() -> None:
    bench = load_module()

    replaced = bench._replace_exact_numeric_value(
        "The measured share was 40.0%.",
        0.4,
        0.29,
    )

    assert replaced == "The measured share was 29%."
    assert bench._contains_exact_numeric_value(replaced, 0.29)


def test_exact_numeric_replacement_updates_every_matching_token() -> None:
    bench = load_module()

    assert bench._replace_exact_numeric_value(
        "From 10 to 10 orders.",
        10,
        12,
    ) == "From 12 to 12 orders."
    assert bench._replace_exact_numeric_value(
        "From 1,000 to 1,000.",
        1000,
        1250,
    ) == "From 1,250 to 1,250."
    assert bench._replace_exact_numeric_value(
        "Both rates were 10% and 10%.",
        0.1,
        0.12,
    ) == "Both rates were 12% and 12%."


@pytest.mark.parametrize(
    ("text", "number", "expected"),
    [
        ("Volume was 100 orders.", 10.0, False),
        ("Volume was 10.5 orders.", 10.0, False),
        ("Malformed 10abc token.", 10.0, False),
        ("Malformed 10.0.5 token.", 10.0, False),
        ("Malformed grouped value 1,00.", 100.0, False),
        ("Malformed percentage 10%%.", 0.1, False),
        ("Malformed sign --10.", -10.0, False),
        ("Malformed sign ++10.", 10.0, False),
        ("Malformed decimal 10..", 10.0, False),
        ("Large value 1000000000001.", 1000000000000.0, False),
        ("The rate was 10%.", 0.1, True),
        ("Revenue was 1,000.5.", 1000.5, True),
    ],
)
def test_exact_numeric_matching_uses_complete_tokens(
    text: str,
    number: float,
    expected: bool,
) -> None:
    bench = load_module()

    assert bench._contains_exact_numeric_value(text, number) is expected


def test_audit_target_does_not_match_expected_value_as_statement_substring() -> None:
    bench = load_module()
    insight = {
        "statement": "Volume was 100 orders.",
        "metric_spec": {
            "expected_value": 0.1,
            "components": [{"expected_value": 10}],
        },
    }

    _, _, _, outputs = bench._audit_target(
        {"insights": [insight]},
        bench.HostAuditMismatch(
            "insights[0].metric_spec.components[0]",
            10,
            12,
        ),
    )

    assert outputs == bench.AUDIT_METRIC_SPEC_OUTPUTS


def test_host_audit_repair_corrects_model_rounded_statement_value() -> None:
    bench = load_module()
    verification = {
        "expression": "SELECT SUM(amount) AS metric_value FROM sales",
        "sources": {"sales": "sales"},
    }
    current = {
        "statement": "Window revenue was 105895.4.",
        "metric_spec": {
            "expected_value": 105895.4,
            "components": [
                {
                    "expected_value": 105895.4,
                    "verification": verification,
                }
            ],
        },
    }
    mismatch = bench.HostAuditMismatch(
        "insights[0].metric_spec.components[0]",
        105895.4,
        105909.66,
    )

    merged = bench.merge_host_audit_repair(
        current,
        mismatch,
        {
            "metric_spec": {
                "expected_value": 105910,
                "components": [
                    {
                        "expected_value": 105910,
                        "verification": verification,
                    }
                ],
            },
            "statement": "Window revenue was 105910.",
        },
    )

    assert merged["statement"] == "Window revenue was 105909.66."
    assert merged["metric_spec"]["expected_value"] == 105909.66
    assert merged["metric_spec"]["components"][0]["expected_value"] == 105909.66


def test_host_audit_repair_reconciles_derived_supporting_claim_percentage() -> None:
    bench = load_module()
    numerator_sql = {
        "expression": "SELECT SUM(amount) AS metric_value FROM sales",
        "sources": {"sales": "sales"},
    }
    denominator_sql = {
        "expression": "SELECT SUM(total) AS metric_value FROM sales",
        "sources": {"sales": "sales"},
    }
    current = {
        "supporting_claims": [
            {
                "claim": "The segment contributed 43.4% of revenue.",
                "expected_value": 0.434,
                "metric_spec": {
                    "type": "share",
                    "expected_value": 0.434,
                    "components": [
                        {
                            "role": "numerator",
                            "expected_value": 67.8342,
                            "verification": numerator_sql,
                        },
                        {
                            "role": "denominator",
                            "expected_value": 156.2131,
                            "verification": denominator_sql,
                        },
                    ],
                },
            }
        ]
    }
    replacement = deepcopy(current["supporting_claims"][0])
    mismatch = bench.HostAuditMismatch(
        "insights[0].supporting_claims[0].metric_spec.components[0]",
        67.8342,
        48.29481,
    )

    merged = bench.merge_host_audit_repair(
        current,
        mismatch,
        {"supporting_claim": replacement},
    )

    claim = merged["supporting_claims"][0]
    expected_share = 48.29481 / 156.2131
    assert claim["expected_value"] == pytest.approx(expected_share)
    assert claim["metric_spec"]["expected_value"] == pytest.approx(expected_share)
    assert bench._contains_exact_numeric_value(claim["claim"], expected_share)
    assert "43.4%" not in claim["claim"]


def test_host_audit_numeric_repair_preserves_diagnostic_identity_fields() -> None:
    bench = load_module()
    verification = {
        "expression": "SELECT AVG(cnt) FROM orders",
        "sources": {"orders": "orders"},
    }
    current = {
        "competing_explanations": ["Seasonality may explain the break."],
        "diagnostic_assessment": {
            "explanations": [
                {
                    "explanation": "Seasonality may explain the break.",
                    "measurable": True,
                    "disposition": "weakened",
                    "expected_value": 10,
                    "verification": verification,
                }
            ]
        },
    }
    replacement = {
        "explanation": "Renamed and invalid hypothesis.",
        "measurable": False,
        "disposition": "ruled_out",
        "expected_value": 12,
        "verification": verification,
    }
    mismatch = bench.parse_host_audit_mismatch(
        "insights[0].diagnostic_assessment.explanations[0]: "
        "expected 10, actual 12"
    )

    merged = bench.merge_host_audit_repair(
        current, mismatch, {"audit_leaf": replacement}
    )

    explanation = merged["diagnostic_assessment"]["explanations"][0]
    assert explanation == {
        "explanation": "Seasonality may explain the break.",
        "measurable": True,
        "disposition": "weakened",
        "expected_value": 12,
        "verification": verification,
    }


def test_top_level_metric_component_repair_updates_primary_numeric_statement(
) -> None:
    bench = load_module()
    verification = {
        "expression": "SELECT COUNT(*) FROM orders",
        "sources": {"orders": "orders"},
    }
    current = {
        "title": "Stable title",
        "statement": "Primary volume was 6,555 orders.",
        "metric_spec": {
            "expected_value": 6555,
            "verification": verification,
            "components": [{
                "expected_value": 6555,
                "verification": verification,
            }],
        },
    }
    repaired_metric = {
        "expected_value": 6597.88,
        "verification": verification,
        "components": [{
            "expected_value": 6597.88,
            "verification": verification,
        }],
    }
    mismatch = bench.parse_host_audit_mismatch(
        "insights[0].metric_spec.components[0]: "
        "expected 6555, actual 6597.88"
    )

    merged = bench.merge_host_audit_repair(
        current,
        mismatch,
        {
            "metric_spec": repaired_metric,
            "statement": "Primary volume was 6,597.88 orders.",
        },
    )

    assert merged == {
        "title": "Stable title",
        "statement": "Primary volume was 6,597.88 orders.",
        "metric_spec": repaired_metric,
    }


def _install_host_audit_fakes(
    bench,
    monkeypatch: pytest.MonkeyPatch,
    repair_payloads: list[dict],
    audit_outcomes: list[object],
    events: list[str],
) -> list[dict]:
    calls: list[dict] = []

    class FakeHostAuditError(RuntimeError):
        pass

    class FakeDspy:
        @staticmethod
        def LM(*args, **kwargs):
            return object()

        @staticmethod
        def configure(**kwargs):
            pass

    class FakeRLM:
        @classmethod
        def from_task(cls, **kwargs):
            calls.append(kwargs)
            index = len(calls) - 1
            result = type(
                "Result",
                (),
                {
                    "payload": repair_payloads[index],
                    "trajectory": f"audit-repair-{index + 1}",
                },
            )()

            class Instance:
                def run(self):
                    events.append("model-repair")
                    return result

            return Instance()

    class FakeExecutor:
        def __init__(self, sources):
            events.append("audit-open")

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            events.append("audit-close")

    def audit(payload, executor):
        events.append("audit")
        outcome = audit_outcomes.pop(0)
        if isinstance(outcome, str):
            raise FakeHostAuditError(outcome)
        return outcome

    monkeypatch.setattr(
        bench,
        "_load_runtime_dependencies",
        lambda: (
            FakeDspy,
            FakeRLM,
            FakeExecutor,
            audit,
            lambda trajectory: {"trajectory": trajectory, "turns": 2},
            FakeHostAuditError,
        ),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")
    return calls


def _audit_claim(expected: float, statement: str) -> dict:
    return {
        "statement": statement,
        "expected_value": expected,
        "verification": {
            "method": "sql",
            "expression": "SELECT COUNT(*) FROM orders",
            "sources": {"orders": "orders"},
        },
    }


def _rejected_candidate(expected: float = 14703) -> dict:
    return {
        "candidate_id": "rejected-volume",
        "disposition": "rejected",
        "rejection_type": "quantitative",
        "rejection_reason": "Observed 14,703 rows, below the threshold.",
        "rejection_evidence": {
            "effect_value": expected,
            "verification": {
                "method": "sql",
                "components": [
                    {
                        "label": "Observed 14,703 rows.",
                        "name": "effect_value",
                        "expected_value": expected,
                        "method": "sql",
                        "expression": "SELECT COUNT(*) FROM orders",
                        "sources": {"orders": "orders"},
                    },
                    {
                        "label": "Untouched sibling",
                        "expected_value": 20,
                        "verification": {
                            "method": "sql",
                            "expression": "SELECT 20",
                            "sources": {"orders": "orders"},
                        },
                    },
                ],
            }
        },
    }


def test_candidate_audit_repair_merges_only_rejected_quantitative_component() -> None:
    bench = load_module()
    candidate = _rejected_candidate()
    scaffold = {
        "analysis_plan": {"grain": "order"},
        "candidates": [candidate],
    }
    insights = {"insights": [{"title": "Promoted insight", "statement": "Stable."}]}
    mismatch = bench.HostAuditMismatch(
        "candidates[0].rejection_evidence.verification.components[0]",
        14703,
        14575,
    )

    repaired = bench.merge_host_candidate_audit_repair(
        scaffold,
        mismatch,
        {
            "rejection_component": {
            "label": "Observed 14,575 rows.",
                "name": "renamed_effect",
                "expected_value": 14575,
                "unit": "mutated",
            }
        },
    )

    repaired_candidate = repaired["candidates"][0]
    repaired_component = (
        repaired_candidate["rejection_evidence"]["verification"]["components"][0]
    )
    assert repaired_component == {
        "label": "Observed 14,575 rows.",
        "name": "effect_value",
        "expected_value": 14575,
        "method": "sql",
        "expression": "SELECT COUNT(*) FROM orders",
        "sources": {"orders": "orders"},
    }
    assert repaired_candidate["candidate_id"] == candidate["candidate_id"]
    assert repaired_candidate["disposition"] == candidate["disposition"]
    assert repaired_candidate["rejection_type"] == candidate["rejection_type"]
    assert repaired_candidate["rejection_reason"] == candidate["rejection_reason"]
    assert repaired_candidate["rejection_evidence"]["effect_value"] == 14575
    assert (
        repaired_candidate["rejection_evidence"]["verification"]["components"][1]
        == candidate["rejection_evidence"]["verification"]["components"][1]
    )
    assert repaired["analysis_plan"] == scaffold["analysis_plan"]
    assert insights == {
        "insights": [{"title": "Promoted insight", "statement": "Stable."}]
    }
    assert scaffold["candidates"][0] == candidate


@pytest.mark.parametrize(
    ("path", "candidate_update", "repair", "match"),
    [
        (
            "candidates[0].rejection_evidence.verification.components",
            {},
            {"rejection_component": {"expected_value": 14575}},
            "exact rejected-candidate component",
        ),
        (
            "candidates[0].rejection_evidence.verification.components[9]",
            {},
            {"rejection_component": {"expected_value": 14575}},
            "component index is out of range",
        ),
        (
            "candidates[0].rejection_evidence.verification.components[0]",
            {"disposition": "promoted"},
            {"rejection_component": {"expected_value": 14575}},
            "promoted candidate",
        ),
        (
            "candidates[0].rejection_evidence.verification.components[0]",
            {"rejection_type": "qualitative"},
            {"rejection_component": {"expected_value": 14575}},
            "not quantitative",
        ),
        (
            "candidates[0].rejection_evidence.verification.components[0]",
            {},
            {
                "rejection_component": {
                    "label": "Observed 14,575 rows.",
                    "expected_value": 14575,
                    "expression": "SELECT 0",
                    "sources": {"orders": "orders"},
                }
            },
            "immutable",
        ),
        (
            "candidates[0].rejection_evidence.verification.components[0]",
            {},
            {
                "rejection_component": {
                    "label": "Arbitrary changed meaning with 14,575 rows.",
                    "expected_value": 14575,
                }
            },
            "numeric explanatory prose",
        ),
    ],
)
def test_candidate_audit_repair_rejects_unsafe_targets_and_mutations(
    path: str, candidate_update: dict, repair: dict, match: str
) -> None:
    bench = load_module()
    candidate = _rejected_candidate()
    candidate.update(candidate_update)
    scaffold = {"analysis_plan": {}, "candidates": [candidate]}
    original = deepcopy(scaffold)
    mismatch = bench.HostAuditMismatch(path, 14703, 14575)

    with pytest.raises(ValueError, match=match):
        bench.merge_host_candidate_audit_repair(scaffold, mismatch, repair)

    assert scaffold == original


def test_candidate_audit_repair_verifies_before_persisting_and_resumes_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    original_candidate = _rejected_candidate()
    scaffold = {
        "analysis_plan": {"grain": "order"},
        "candidates": [original_candidate],
    }
    insights = {"insights": [{"title": "Stable", "statement": "Unchanged."}]}
    caches = write_all_caches(tmp_path, research, scaffold, insights)
    repaired_component = {
        "label": "Observed 14,575 rows.",
        "expected_value": 14575,
    }
    mismatch_path = (
        "candidates[0].rejection_evidence.verification.components[0]"
    )
    events: list[str] = []
    calls = _install_host_audit_fakes(
        bench,
        monkeypatch,
        [{"rejection_component": repaired_component}],
        [
            f"{mismatch_path}: expected 14703.0, actual 14575.0",
            FakeAudit((FakeCheck(mismatch_path, 14575, 14575),)),
        ],
        events,
    )
    monkeypatch.setattr(
        bench,
        "verify_portable_contract",
        lambda payload: events.append("portable"),
    )
    original_normalize = bench.normalize_mechanical_contract

    def recording_normalize(payload, sources):
        events.append("normalize")
        return original_normalize(payload, sources)

    monkeypatch.setattr(bench, "normalize_mechanical_contract", recording_normalize)
    original_atomic_json = bench._atomic_json

    def recording_atomic_json(path, value):
        if Path(path) == caches[1]:
            events.append("persist:scaffold")
        elif Path(path) == caches[2]:
            events.append("persist:insights")
        original_atomic_json(path, value)

    monkeypatch.setattr(bench, "_atomic_json", recording_atomic_json)

    record = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
        max_audit_repairs=1,
    )

    repair_start = events.index("model-repair") - 3
    assert events[repair_start:] == [
        "audit-open",
        "audit",
        "audit-close",
        "model-repair",
        "normalize",
        "normalize",
        "portable",
        "persist:scaffold",
        "persist:insights",
        "audit-open",
        "audit",
        "audit-close",
    ]
    assert calls[0]["outputs"] == {"rejection_component": dict}
    assert mismatch_path in calls[0]["task"]
    assert "14703.0" in calls[0]["task"]
    assert "14575.0" in calls[0]["task"]
    assert "SELECT COUNT" not in calls[0]["task"]
    assert record["payload"]["insights"] == insights["insights"]
    assert record["audit_repairs"][0]["target_path"] == mismatch_path

    scaffold_envelope = json.loads(caches[1].read_text(encoding="utf-8"))
    repaired_scaffold = scaffold_envelope["partial"]
    assert scaffold_envelope["input_fingerprint"] == synthesis_fingerprint(research)
    insights_envelope = json.loads(caches[2].read_text(encoding="utf-8"))
    assert insights_envelope["input_fingerprint"] == synthesis_fingerprint(
        research, repaired_scaffold
    )
    assert insights_envelope["partial"] == insights

    resumed_events: list[str] = []
    resumed_calls = _install_host_audit_fakes(
        bench,
        monkeypatch,
        [],
        [FakeAudit((FakeCheck(mismatch_path, 14575, 14575),))],
        resumed_events,
    )
    monkeypatch.setattr(bench, "verify_portable_contract", lambda payload: None)
    resumed = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
    )
    assert resumed_calls == []
    assert resumed["payload"] == record["payload"]


def test_invalid_portable_candidate_repair_does_not_persist_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {}, "candidates": [_rejected_candidate()]}
    insights = {"insights": [{"title": "Stable"}]}
    caches = write_all_caches(tmp_path, research, scaffold, insights)
    before = [path.read_bytes() for path in caches[1:]]
    mismatch_path = (
        "candidates[0].rejection_evidence.verification.components[0]"
    )
    events: list[str] = []
    _install_host_audit_fakes(
        bench,
        monkeypatch,
        [{
            "rejection_component": {
                "label": "Observed 14,575 rows.",
                "expected_value": 14575,
            }
        }],
        [f"{mismatch_path}: expected 14703, actual 14575"],
        events,
    )
    verification_count = 0

    def verifier(payload):
        nonlocal verification_count
        verification_count += 1
        if verification_count > 1:
            raise AssertionError("candidate repair is invalid")

    monkeypatch.setattr(bench, "verify_portable_contract", verifier)

    with pytest.raises(AssertionError, match="candidate repair is invalid"):
        bench.run_staged_benchmark(
            data_dir,
            research_cache_path=caches[0],
            scaffold_cache_path=caches[1],
            insights_cache_path=caches[2],
        )

    assert [path.read_bytes() for path in caches[1:]] == before


def test_host_audit_mismatch_repairs_persists_portably_verifies_and_reaudits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {}, "candidates": []}
    original_claim = _audit_claim(10, "Expected 10 orders.")
    insights = {
        "insights": [{
            "title": "Orders",
            "statement": "Order volume is measured.",
            "supporting_claims": [original_claim, {"statement": "sibling"}],
        }]
    }
    caches = write_all_caches(tmp_path, research, scaffold, insights)
    repaired_claim = _audit_claim(12, "Expected 12 orders.")
    events: list[str] = []
    calls = _install_host_audit_fakes(
        bench,
        monkeypatch,
        [{"supporting_claim": repaired_claim}],
        [
            "insights[0].supporting_claims[0]: expected 10, actual 12",
            FakeAudit((FakeCheck("insights[0].supporting_claims[0]", 12, 12),)),
        ],
        events,
    )
    monkeypatch.setattr(
        bench,
        "verify_portable_contract",
        lambda payload: events.append("portable"),
    )
    original_normalize = bench.normalize_mechanical_contract

    def recording_normalize(payload, sources):
        events.append("normalize")
        return original_normalize(payload, sources)

    monkeypatch.setattr(bench, "normalize_mechanical_contract", recording_normalize)
    original_atomic_json = bench._atomic_json

    def recording_atomic_json(path, value):
        if Path(path) == caches[2]:
            events.append("persist")
        original_atomic_json(path, value)

    monkeypatch.setattr(bench, "_atomic_json", recording_atomic_json)

    record = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
        max_audit_repairs=2,
    )

    mismatch_index = events.index("model-repair") - 3
    assert events[mismatch_index:] == [
        "audit-open",
        "audit",
        "audit-close",
        "model-repair",
        "normalize",
        "portable",
        "persist",
        "audit-open",
        "audit",
        "audit-close",
    ]
    assert record["payload"]["insights"][0]["supporting_claims"] == [
        repaired_claim,
        {"statement": "sibling"},
    ]
    assert record["audit_repairs"] == [{
        "target_path": "insights[0].supporting_claims[0]",
        "attempt": 1,
        "expected": 10.0,
        "actual": 12.0,
        "trajectory": {"trajectory": "audit-repair-1", "turns": 2},
    }]
    assert calls[0]["outputs"] == {"supporting_claim": dict}
    assert calls[0]["max_turns"] == 6
    assert calls[0]["timeout"] == 3600
    assert bench._compact_json({
        "statement": "Expected 10 orders.",
        "expected_value": 10,
    }) in calls[0]["task"]
    assert "SELECT COUNT" not in calls[0]["task"]
    assert "actual 12" in calls[0]["task"].lower()

    resumed_events: list[str] = []
    resumed_calls = _install_host_audit_fakes(
        bench,
        monkeypatch,
        [],
        [FakeAudit((FakeCheck("insights[0].supporting_claims[0]", 12, 12),))],
        resumed_events,
    )
    monkeypatch.setattr(bench, "verify_portable_contract", lambda payload: None)
    resumed = bench.run_staged_benchmark(
        data_dir,
        research_cache_path=caches[0],
        scaffold_cache_path=caches[1],
        insights_cache_path=caches[2],
    )
    assert resumed_calls == []
    assert resumed["payload"]["insights"][0]["supporting_claims"][0] == repaired_claim


@pytest.mark.parametrize(
    "message",
    [
        "executor failed at insights[0].supporting_claims[0]: boom",
        "insights[0].supporting_claims[0]: expected NaN, actual 12",
        "some malformed audit failure",
    ],
)
def test_non_mismatch_host_audit_failures_are_not_model_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {}, "candidates": []}
    insights = {"insights": [{"supporting_claims": [_audit_claim(10, "10")]}]}
    caches = write_all_caches(tmp_path, research, scaffold, insights)
    events: list[str] = []
    calls = _install_host_audit_fakes(
        bench, monkeypatch, [], [message], events
    )
    monkeypatch.setattr(bench, "verify_portable_contract", lambda payload: None)

    with pytest.raises(RuntimeError, match=re.escape(message)):
        bench.run_staged_benchmark(
            data_dir,
            research_cache_path=caches[0],
            scaffold_cache_path=caches[1],
            insights_cache_path=caches[2],
        )

    assert calls == []
    assert "model-repair" not in events


def test_host_audit_repair_has_separate_budget_and_rejects_bad_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {}, "candidates": []}
    insights = {"insights": [{"supporting_claims": [_audit_claim(10, "10")]}]}
    caches = write_all_caches(tmp_path, research, scaffold, insights)
    mismatch = "insights[0].supporting_claims[0]: expected 10, actual 12"
    events: list[str] = []
    calls = _install_host_audit_fakes(
        bench,
        monkeypatch,
        [{"supporting_claim": _audit_claim(11, "Wrong 11.")}],
        [mismatch],
        events,
    )
    monkeypatch.setattr(bench, "verify_portable_contract", lambda payload: None)

    with pytest.raises(ValueError, match="authoritative actual 12"):
        bench.run_staged_benchmark(
            data_dir,
            research_cache_path=caches[0],
            scaffold_cache_path=caches[1],
            insights_cache_path=caches[2],
            max_insight_repairs=0,
            max_scaffold_repairs=0,
            max_audit_repairs=1,
        )
    assert len(calls) == 1
    assert len(events) == events.index("model-repair") + 1


def test_host_audit_repair_exhaustion_is_actionable_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    research = valid_research()
    scaffold = {"analysis_plan": {}, "candidates": []}
    insights = {"insights": [{"supporting_claims": [_audit_claim(10, "10")]}]}
    caches = write_all_caches(tmp_path, research, scaffold, insights)
    events: list[str] = []
    calls = _install_host_audit_fakes(
        bench,
        monkeypatch,
        [{"supporting_claim": _audit_claim(12, "12")}],
        [
            "insights[0].supporting_claims[0]: expected 10, actual 12",
            "insights[0].supporting_claims[0]: expected 12, actual 13",
        ],
        events,
    )
    monkeypatch.setattr(bench, "verify_portable_contract", lambda payload: None)

    with pytest.raises(RuntimeError, match="after 1 audit repair attempts"):
        bench.run_staged_benchmark(
            data_dir,
            research_cache_path=caches[0],
            scaffold_cache_path=caches[1],
            insights_cache_path=caches[2],
            max_audit_repairs=1,
        )

    assert len(calls) == 1


def test_run_artifact_records_audit_repair_metadata_without_claim_or_sql(
    tmp_path: Path,
) -> None:
    bench = load_module()
    record = {
        "research": valid_research(),
        "payload": {
            "contract_version": 2,
            "analysis_plan": {},
            "candidates": [],
            "insights": [],
        },
        "audit": FakeAudit(()),
        "trajectories": {
            "research": {},
            "contract_scaffold": {},
            "insights": {},
        },
        "model": "test",
        "stage_skills": {
            "research": (),
            "contract_scaffold": (),
            "insights": (),
        },
        "repairs": [],
        "audit_repairs": [{
            "target_path": "insights[0].supporting_claims[0]",
            "attempt": 1,
            "expected": 10.0,
            "actual": 12.0,
            "trajectory": {"turns": 2},
        }],
        "mechanical_repairs": {"count": 0, "paths": []},
    }

    paths = bench.write_staged_artifacts(tmp_path / "artifacts", record)
    run = json.loads(paths["run"].read_text(encoding="utf-8"))

    assert run["audit_repair_summaries"] == record["audit_repairs"]
    assert run["counts"]["audit_repairs"] == 1
    serialized = json.dumps(run["audit_repair_summaries"]).lower()
    assert "sql" not in serialized
    assert "payload" not in serialized
