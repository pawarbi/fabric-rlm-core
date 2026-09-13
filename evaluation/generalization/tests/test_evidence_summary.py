import json

from evaluation.generalization.evidence_summary import analyze, summarize


def test_nested_strict_false_is_not_truthy_success():
    row = {
        "answer": {"status": "success", "value": 186},
        "expected": {"value": 186},
        "grade": {"correct": False, "outcome": "confident_wrong",
                  "checks": {"value": True, "period": False},
                  "strict": {"correct": False}},
        "metrics": {"verification_outcome": "verified", "provider_cost_usd": None},
    }
    result = summarize([row])
    assert result["strict_correct"] == 0
    assert result["value_identity_correct"] == 1
    assert result["framing_only_mismatch"] == 1
    assert result["wrong_value_or_identity"] == 0
    assert result["cost_known_trials"] == 0
    assert result["provider_cost_usd"] is None


def test_expected_abstention_is_behavior_success_but_incomplete_task():
    row = {
        "answer": {"status": "abstain", "value": "no cause recorded"},
        "expected": {"expected_status": "abstain"},
        "grade": {"correct": True, "outcome": "correct", "strict": {"correct": True}},
        "metrics": {},
    }
    result = summarize([row])
    assert result["expected_abstention_detected"] == 1
    assert result["incomplete_tasks"] == 1
    assert result["answer_correct"] == 0


def test_wrong_identity_is_not_hidden_by_right_number():
    row = {
        "answer": {"status": "success", "value": 10, "entity_id": "wrong"},
        "expected": {"value": 10, "entity_id": "correct"},
        "grade": {"correct": False, "outcome": "confident_wrong",
                  "checks": {"value": True, "entity_id": False}},
        "metrics": {"provider_cost_usd": 0.1},
    }
    result = summarize([row])
    assert result["value_identity_correct"] == 0
    assert result["wrong_value_or_identity"] == 1
    assert result["provider_cost_usd"] == 0.1


def test_analysis_preserves_the_recorded_repetition_setting(tmp_path):
    path = tmp_path / "one-repetition.json"
    path.write_text(json.dumps({
        "status": "complete",
        "repetitions": 1,
        "trials": [{
            "domain": "inventory", "variant": "descriptive",
            "question_id": "example", "arm": "A", "repetition": 0,
            "answer": {"status": "success", "value": 1},
            "expected": {"value": 1},
            "grade": {"correct": True, "outcome": "correct", "checks": {"value": True}},
            "metrics": {},
        }],
    }), encoding="utf-8")

    result = analyze(path)

    assert result["repetitions"] == 1
    assert "Three repetitions" not in result["interpretation"]["sampling"]
