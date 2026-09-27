"""A verified procedure can transfer to an unrelated CSV, with fail-closed binding."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fabric_rlm import RLM
from fabric_rlm.experimental.transferable_procedure import bind_csv, extract_procedure
from fabric_rlm.knowledge import EvidenceRecord
from fabric_rlm.knowledge_lessons import promote_lessons


TARGET = Path(__file__).parent / "fixtures" / "no_customer_production.csv"


def _learned_procedure(tmp_path: Path):
    training = tmp_path / "other_domain.csv"
    training.write_text(
        "division,channel,sku,amount\nA,X,one,10\nA,X,two,20\n",
        encoding="utf-8",
    )
    knowledge = RLM.learn(sources={"training": training})
    fingerprint = knowledge.package.sources[0].schema_fingerprint
    records = [
        EvidenceRecord(
            evidence_id=f"evidence.training.{index}",
            evidence_type="trajectory",
            source_ids=("training",),
            observation_type="strategy_sequence",
            observation={
                "strategy": "candidate_drilldown",
                "candidate_source_grain": ["division", "channel"],
                "drilldown_grain": ["division", "channel", "sku"],
                "candidate_identity_preserved": True,
            },
            source_fingerprints={"training": fingerprint},
            execution_status="success",
            verifier_status="passed",
            analytical_integrity_status="passed",
            run_fingerprint=f"independent-run-{index}",
        )
        for index in (1, 2)
    ]
    enriched = promote_lessons(knowledge.package, records)
    lesson = next(l for l in enriched.lessons if l.kind == "preferred_strategy")
    return extract_procedure(lesson), lesson


def test_verified_source_procedure_binds_to_csv_without_training_fields(tmp_path: Path) -> None:
    procedure, lesson = _learned_procedure(tmp_path)
    assert procedure is not None and procedure.evidence_count == 2
    target = RLM.learn(sources={"production": TARGET}).package.sources[0]
    question = "Which region produced the most units? Within that region, which machine contributed the most?"
    plan = bind_csv(procedure, target, question)
    assert plan is not None
    assert (plan.coarse, plan.detail, plan.measure) == ("region", "machine", "produced_units")
    assert not {"division", "channel", "sku", "customer"} & set(plan.guidance().split())

    renamed = tmp_path / "renamed.csv"
    renamed.write_text(
        TARGET.read_text(encoding="utf-8")
        .replace("region,line,machine,produced_units", "site,line,station,throughput"),
        encoding="utf-8",
    )
    other = RLM.learn(sources={"operations": renamed}).package.sources[0]
    rebound = bind_csv(
        procedure, other,
        "Which site has the most throughput? Within that site, which station contributed the most?",
    )
    assert rebound is not None
    assert (rebound.coarse, rebound.detail, rebound.measure) == ("site", "station", "throughput")
    assert extract_procedure(replace(lesson, status="candidate")) is None
    assert extract_procedure(replace(lesson, basis=("verified_runs",))) is None


def test_binding_refuses_missing_ambiguous_and_unrelated_questions(tmp_path: Path) -> None:
    procedure, _ = _learned_procedure(tmp_path)
    target = RLM.learn(sources={"production": TARGET}).package.sources[0]
    assert procedure is not None
    for question in (
        "Which customer has the most units? Within that customer, which machine leads?",
        "Which region has the most units? Within that region, which line or machine leads?",
        "How many units were produced?",
    ):
        assert bind_csv(procedure, target, question) is None
    assert bind_csv(procedure, replace(target, diagnostics={"snapshot_exact": False}),
                    "Which region has the most units? Within that region, which machine leads?") is None
