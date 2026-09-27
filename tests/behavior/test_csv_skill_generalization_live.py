"""Live Fabric RLM test: bind a verified procedure to a different CSV.

Training evidence below is a controlled fixture. This exercises extraction,
binding, and model execution, not automatic discovery of the training trace.

Run: OPENROUTER_API_KEY=... python -m pytest -q -s \
    tests/behavior/test_csv_skill_generalization_live.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from fabric_rlm import RLM
from fabric_rlm.experimental.transferable_procedure import bind_csv, extract_procedure
from fabric_rlm.knowledge import EvidenceRecord
from fabric_rlm.knowledge_lessons import promote_lessons
from .test_behavior_baseline import _PRIMARY_MODEL
from .runner import make_lm


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "fixtures" / "no_customer_production.csv"

def _training_procedure(tmp_path: Path):
    training = tmp_path / "training.csv"
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
    package = promote_lessons(knowledge.package, records)
    return extract_procedure(next(l for l in package.lessons if l.kind == "preferred_strategy"))


def _expected(path: Path, group: str, detail: str, measure: str) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = {r[group] for r in rows}
    winner = max(sorted(groups), key=lambda g: sum(int(r[measure]) for r in rows if r[group] == g))
    selected = [r for r in rows if r[group] == winner]
    details = {r[detail] for r in selected}
    top = max(sorted(details), key=lambda d: sum(int(r[measure]) for r in selected if r[detail] == d))
    return {
        "group": winner,
        "group_total": sum(int(r[measure]) for r in selected),
        "detail": top,
        "detail_total": sum(int(r[measure]) for r in selected if r[detail] == top),
    }


@pytest.mark.parametrize("with_transfer", [False, True], ids=["cold", "transferred_procedure"])
@pytest.mark.parametrize("renamed", [False, True], ids=["original_columns", "renamed_columns"])
def test_model_infers_csv_fields_and_executes(
    tmp_path: Path, with_transfer: bool, renamed: bool
) -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        if os.getenv("BEHAVIOR_CI_REQUIRED") == "1":
            pytest.fail("OPENROUTER_API_KEY is required for the live CSV generalization test")
        pytest.skip("Live model run needs OPENROUTER_API_KEY")
    if renamed:
        path = tmp_path / "operations.csv"
        path.write_text(
            SOURCE.read_text(encoding="utf-8")
            .replace("region,line,machine,produced_units", "site,line,station,throughput"),
            encoding="utf-8",
        )
        question = "Which site produced the most throughput? Within that site, which station contributed the most?"
        expected = _expected(path, "site", "station", "throughput")
    else:
        path = SOURCE
        question = "Which region produced the most units? Within that region, which machine contributed the most?"
        expected = _expected(path, "region", "machine", "produced_units")

    def validate(payload: dict) -> None:
        assert payload.get("answer") == expected, f"Expected {expected}, got {payload.get('answer')}"

    plan = None
    if with_transfer:
        procedure = _training_procedure(tmp_path)
        assert procedure is not None
        profile = RLM.learn(sources={"target": path}).package.sources[0]
        plan = bind_csv(procedure, profile, question)
        assert plan is not None, "The experimental transfer declined this applicable task"

    result = RLM.task(
        question + " Return answer as group, group_total, detail, detail_total."
        + ("\n" + plan.guidance() if plan else ""),
        inputs={"source": path},
        outputs={"answer": dict},
        lm=make_lm(_PRIMARY_MODEL),
        output_validator=validate,
        max_turns=8,
        timeout=120,
    ).run()
    print(json.dumps({
        "case": "renamed" if renamed else "original",
        "arm": "transferred_procedure" if with_transfer else "cold",
        "binding": {"group": plan.coarse, "detail": plan.detail, "measure": plan.measure} if plan else None,
        "expected": expected,
        "submitted": result.submitted,
        "payload": result.payload,
        "failure_reason": result.failure_reason,
        "turns": len(result.turns),
    }, default=str))
    assert result.submitted and result.payload == {"answer": expected}, result.failure_reason
