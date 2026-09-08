from __future__ import annotations

import json
from pathlib import Path

from evaluation.generalization.fixtures import generate_fixtures
from evaluation.generalization.grader import grade_answer
from evaluation.generalization.references import calculate_references


def test_seeded_fixtures_and_references_are_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = generate_fixtures(first, seed=20260908, large_rows=2_000)
    second_manifest = generate_fixtures(second, seed=20260908, large_rows=2_000)

    assert first_manifest["content_sha256"] == second_manifest["content_sha256"]
    assert calculate_references(first) == calculate_references(second)


def test_naming_variants_preserve_reference_answers(tmp_path: Path) -> None:
    generate_fixtures(tmp_path, seed=20260908, large_rows=2_000)

    references = calculate_references(tmp_path)

    for domain in ("inventory", "manufacturing", "service"):
        assert references[domain]["descriptive"] == references[domain]["abbreviated"]
        assert references[domain]["descriptive"] == references[domain]["camel"]


def test_reference_answers_cover_domain_specific_failure_modes(tmp_path: Path) -> None:
    generate_fixtures(tmp_path, seed=20260908, large_rows=2_000)

    references = calculate_references(tmp_path)

    inventory = references["inventory"]["descriptive"]
    assert inventory["inventory_available_units"]["value"] == 186
    assert inventory["inventory_open_order_units"]["value"] == 48
    assert inventory["inventory_top_customer"]["entity_id"] == "C-002"
    assert inventory["inventory_join_safe_value"]["value"] == 2250.0

    manufacturing = references["manufacturing"]["descriptive"]
    assert manufacturing["manufacturing_complete_units"]["value"] == 1800
    assert manufacturing["manufacturing_weighted_defect_rate"]["value"] == 0.025
    assert manufacturing["manufacturing_worst_line"]["entity_id"] == "LN-B"

    service = references["service"]["descriptive"]
    assert service["service_first_response_sla_rate"]["value"] == 2 / 3
    assert service["service_reopened_tickets"]["value"] == 1
    assert service["service_ambiguous_root_cause"]["expected_status"] == "abstain"


def test_large_fixture_is_bigger_than_prompt_budget(tmp_path: Path) -> None:
    manifest = generate_fixtures(tmp_path, seed=20260908, large_rows=50_000)

    large_file = tmp_path / manifest["large_fixture"]["path"]

    assert large_file.stat().st_size > manifest["large_fixture"]["prompt_budget_bytes"]


def test_grader_separates_wrong_incomplete_and_unsupported_claims() -> None:
    expected = {
        "value": 0.025,
        "units": "ratio",
        "grain": "complete production periods",
        "period": "2026-01",
        "expected_status": "answered",
    }

    correct = grade_answer(
        {
            "status": "answered",
            "value": 0.025,
            "units": "ratio",
            "grain": "complete production periods",
            "period": "2026-01",
            "claims": [],
        },
        expected,
    )
    wrong = grade_answer(
        {
            "status": "answered",
            "value": 0.04,
            "units": "ratio",
            "grain": "complete production periods",
            "period": "2026-01",
            "claims": [],
        },
        expected,
    )
    incomplete = grade_answer({"status": "timeout"}, expected)
    unsupported = grade_answer(
        {
            "status": "answered",
            "value": 0.025,
            "units": "ratio",
            "grain": "complete production periods",
            "period": "2026-01",
            "claims": [{"text": "Line B caused the defects", "supported": False}],
        },
        expected,
    )

    assert correct["outcome"] == "correct"
    assert wrong["outcome"] == "confident_wrong"
    assert incomplete["outcome"] == "incomplete"
    assert unsupported["outcome"] == "unsupported_claim"


def test_definitions_include_mappings_and_insufficient_metadata_case(
    tmp_path: Path,
) -> None:
    generate_fixtures(tmp_path, seed=20260908, large_rows=2_000)

    definitions = json.loads(
        (tmp_path / "definitions.json").read_text(encoding="utf-8")
    )
    questions = json.loads(
        (tmp_path / "questions.json").read_text(encoding="utf-8")
    )

    assert definitions["variants"]["abbreviated"]["inventory"]["field_mappings"]
    assert definitions["variants"]["camel"]["manufacturing"]["field_mappings"]
    ambiguous = next(
        question
        for question in questions
        if question["question_id"] == "service_ambiguous_root_cause"
    )
    assert ambiguous["expected_behavior"] == "abstain_or_request_definition"
    assert "root cause" not in ambiguous["provided_definitions"]
