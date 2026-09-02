from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "examples"
    / "notebooks"
    / "rlm_learn_semantic_model_value.py"
)


def test_user_facing_learn_notebook_compares_the_same_semantic_question() -> None:
    source = NOTEBOOK.read_text(encoding="utf-8")

    assert source.startswith("# Fabric notebook source")
    assert "WORKSPACE_ID =" in source
    assert "MODEL_ID =" in source
    assert 'MEASURE = "ARR $"' in source
    assert "expected_value = float(" in source
    assert "model.measure(MEASURE)" in source
    assert "knowledge = RLM.learn(" in source
    assert "inputs={\"business_model\": model}" in source
    assert "knowledge=knowledge" in source
    assert source.count("QUESTION,") >= 2
    assert source.count("cache=False") >= 2
    assert 'assert is_correct(cold_result)' in source
    assert "operation_audit_status" in source
    assert "operation_result_fingerprint" in source
    assert "mssparkutils" not in source
