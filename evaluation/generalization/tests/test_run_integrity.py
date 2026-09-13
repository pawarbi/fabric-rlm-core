from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.generalization import runner
from evaluation.generalization.insufficient_metadata import classify, extract_answer
from evaluation.generalization.insufficient_metadata import ambiguity_cases
from evaluation.generalization.fixtures import generate_fixtures


def test_run_artifacts_are_unique_and_refuse_reuse(tmp_path):
    first = runner.reserve_artifacts(tmp_path / "first.json")
    second = runner.reserve_artifacts(tmp_path / "second.json")
    assert first != second
    (first / "retained.txt").write_text("original")
    with pytest.raises(FileExistsError):
        runner.reserve_artifacts(tmp_path / "first.json")
    assert (first / "retained.txt").read_text() == "original"


def test_existing_result_cannot_be_overwritten(tmp_path):
    output = tmp_path / "done.json"
    output.write_text('{"status":"complete"}')
    with pytest.raises(FileExistsError):
        runner.reserve_artifacts(output)
    assert output.read_text() == '{"status":"complete"}'


def test_saved_package_has_actual_content_and_detects_mutation(tmp_path):
    package = SimpleNamespace(
        fingerprint="before", to_dict=lambda: {"fingerprint": "before", "lessons": []}
    )
    knowledge = SimpleNamespace(package=package)
    snapshot = runner.snapshot_package(tmp_path / "package.json", knowledge)
    assert Path(snapshot["path"]).read_text().find('"lessons"') >= 0
    runner.check_package(knowledge, snapshot)
    package.fingerprint = "after"
    with pytest.raises(RuntimeError, match="package changed"):
        runner.check_package(knowledge, snapshot)


def test_missing_metadata_reads_result_outputs_not_nonexistent_payload():
    result = SimpleNamespace(
        outputs={"answer": {"status": "needs_definition", "value": "definition missing"}},
        submitted=True, failure_reason=None,
    )
    assert extract_answer(result)["status"] == "needs_definition"


def test_generic_caveat_does_not_count_as_missing_definition_abstention():
    grade = classify(
        {"status": "success", "value": 123, "caveats": "rounding",
         "claims": [{"text": "unrelated", "supported": False}]},
        {"value": 123},
    )
    assert grade["outcome"] != "abstained"


def test_ambiguity_cases_have_distinct_hidden_counterfactual_answers(tmp_path):
    generate_fixtures(tmp_path, large_rows=20)
    cases = ambiguity_cases(tmp_path)
    assert {case["domain"] for case in cases} == {"inventory", "manufacturing", "service"}
    for case in cases:
        assert case["witness"]["answer_a"] != case["witness"]["answer_b"]
        assert case["witness"]["definition_a"] not in case["text"]
