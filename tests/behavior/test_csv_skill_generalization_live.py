"""Live Fabric RLM test: model must discover fields in a CSV without customers.

Run: OPENROUTER_API_KEY=... python -m pytest -q -s \
    tests/behavior/test_csv_skill_generalization_live.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from fabric_rlm import RLM, SkillLoader
from .test_behavior_baseline import _PRIMARY_MODEL
from .runner import make_lm


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "fixtures" / "no_customer_production.csv"
SKILLS = Path(__file__).parent / "skills"


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


@pytest.mark.parametrize("with_skill", [False, True], ids=["cold", "generic_skill"])
@pytest.mark.parametrize("renamed", [False, True], ids=["original_columns", "renamed_columns"])
def test_model_infers_csv_fields_and_executes(
    tmp_path: Path, with_skill: bool, renamed: bool
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

    result = RLM.task(
        question + " Return answer as group, group_total, detail, detail_total.",
        inputs={"source": path},
        outputs={"answer": dict},
        lm=make_lm(_PRIMARY_MODEL),
        skill_loader=SkillLoader(SKILLS),
        skills=["coarse_to_detail"] if with_skill else [],
        skills_as_cards=False,
        output_validator=validate,
        max_turns=8,
        timeout=120,
    ).run()
    print(json.dumps({
        "case": "renamed" if renamed else "original",
        "arm": "generic_skill" if with_skill else "cold",
        "expected": expected,
        "submitted": result.submitted,
        "payload": result.payload,
        "failure_reason": result.failure_reason,
        "turns": len(result.turns),
    }, default=str))
    assert result.submitted and result.payload == {"answer": expected}, result.failure_reason
