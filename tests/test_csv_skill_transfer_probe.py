"""Offline probe: transfer a coarse-to-detail procedure to a non-customer CSV.

This tests the data operation and applicability boundary, not LM autonomy.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

from fabric_rlm import RLM
from fabric_rlm.knowledge import KnowledgePackage, LearnedLesson, SourceProfile
from fabric_rlm.knowledge_retrieval import retrieve_lessons


CSV = Path(__file__).parent / "fixtures" / "no_customer_production.csv"
TASK = "Find the region with the most produced units, then its top machine."


def _transferable_drilldown(
    path: Path, *, coarse: str, detail: str, measure: str
) -> tuple[str, int, str, int]:
    """Apply the reusable operation after checking the new source's columns."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        if len({coarse, detail, measure}) != 3 or not {coarse, detail, measure} <= columns:
            raise ValueError("source does not have the required distinct dimensions and measure")
        rows = list(reader)
    if not rows:
        raise ValueError("source has no rows")
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        if not row[coarse] or not row[detail]:
            raise ValueError("dimension value is missing")
        totals[row[coarse]] += int(row[measure])
    winner = max(sorted(totals), key=totals.get)
    by_detail: dict[str, int] = defaultdict(int)
    for row in rows:
        if row[coarse] == winner:
            by_detail[row[detail]] += int(row[measure])
    top_detail = max(sorted(by_detail), key=by_detail.get)
    return winner, totals[winner], top_detail, by_detail[top_detail]


def test_existing_source_bound_lesson_does_not_transfer_to_no_customer_csv() -> None:
    knowledge = RLM.learn(sources={"production": CSV})
    source = knowledge.package.sources[0]
    assert "customer" not in CSV.read_text(encoding="utf-8").lower()
    assert "customer" not in str(source.schema).lower()

    old_source = SourceProfile(
        source_id="old_model", family="semantic_model", locator="semantic-model/v1/old",
        snapshot_fingerprint="old-snapshot", schema_fingerprint="old-schema",
        schema={"tables": {}, "columns": {}, "measures": {}, "relationships": {}},
    )
    old_lesson = LearnedLesson(
        lesson_id="old.drilldown", kind="preferred_strategy",
        subject="coarse to candidate drilldown", status="active", confidence="high",
        structured_rule={"coarse_grain": ["product", "region"],
                         "drilldown_grain": ["product", "region", "customer_group"],
                         "strategy": "coarse_to_candidate_drilldown"},
        source_dependencies=("old_model",), source_fingerprints={"old_model": "old-schema"},
    )
    combined = KnowledgePackage(
        package_id="transfer_probe", sources=(old_source, source), lessons=(old_lesson,)
    )
    assert retrieve_lessons(combined, TASK, source_ids=["production"]) == ()


def test_generic_procedure_transfers_to_actual_csv_and_reconciles() -> None:
    result = _transferable_drilldown(
        CSV, coarse="region", detail="machine", measure="produced_units"
    )
    assert result == ("East", 260, "M1", 220)
    with CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert result[1] == sum(int(row["produced_units"]) for row in rows if row["region"] == result[0])
    assert result[3] == sum(int(row["produced_units"]) for row in rows if row["region"] == result[0] and row["machine"] == result[2])


def test_generic_procedure_rejects_unavailable_customer_dimension() -> None:
    with pytest.raises(ValueError, match="required distinct dimensions"):
        _transferable_drilldown(
            CSV, coarse="region", detail="customer_group", measure="produced_units"
        )


def test_same_procedure_works_after_unrelated_column_renames(tmp_path: Path) -> None:
    renamed = tmp_path / "renamed.csv"
    renamed.write_text(
        "site,station,throughput\nA,S1,10\nA,S1,20\nA,S2,7\nB,S3,30\n",
        encoding="utf-8",
    )
    assert _transferable_drilldown(
        renamed, coarse="site", detail="station", measure="throughput"
    ) == ("A", 37, "S1", 30)
