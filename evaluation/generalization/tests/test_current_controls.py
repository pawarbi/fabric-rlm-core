from copy import deepcopy
from pathlib import Path
import io
import json

import fabric_rlm
import pytest

from evaluation.generalization import runner
from evaluation.generalization.grader import grade_answer
from evaluation.generalization import nonregression


def test_live_import_must_match_the_harness_checkout(monkeypatch, tmp_path):
    assert runner.runtime_checkout() / "fabric_rlm" == Path(fabric_rlm.__file__).resolve().parent
    monkeypatch.setattr(fabric_rlm, "__file__", str(tmp_path / "foreign" / "__init__.py"))
    with pytest.raises(RuntimeError, match="checkout"):
        runner.runtime_checkout()


def _document():
    expected = {"expected_status": "answered", "value": 12, "units": "units", "period": "2026"}
    answer = {"status": "success", "value": 12, "units": "units", "period": "2026"}
    schedule = runner.build_schedule(
        [{"question_id": name, "domain": "inventory"} for name in ("one", "two")],
        repetitions=3, seed=7,
    )
    return {
        "status": "complete", "core_freeze_mismatches": [], "fixture_changes": [],
        "planned_schedule": schedule,
        "trials": [
            {**entry, "expected": deepcopy(expected), "answer": deepcopy(answer),
             "grade": grade_answer(answer, expected), "metrics": {"submitted": True}}
            for entry in schedule
        ],
    }


@pytest.mark.parametrize("case", ["empty", "whole_question_missing", "duplicate", "partial", "drift"])
def test_live_gate_rejects_incomplete_or_invalid_evidence(case):
    document = _document()
    if case == "empty":
        document["trials"] = []
    elif case == "whole_question_missing":
        document["trials"] = [r for r in document["trials"] if r["question_id"] != "two"]
    elif case == "duplicate":
        document["trials"].append(document["trials"][0])
    elif case == "partial":
        document["status"] = "partial"
    else:
        document["fixture_changes"] = ["a changed table"]
    assert nonregression.evaluate(document)["passed"] is False


def test_live_gate_does_not_hide_a_question_regression_or_a_faster_abstention():
    document = _document()
    for row in document["trials"]:
        if row["arm"] == "B" and row["question_id"] == "one":
            row["answer"] = {"status": "abstain"}
            row["grade"] = grade_answer(row["answer"], row["expected"])
    gate = nonregression.evaluate(document)
    assert gate["passed"] is False
    assert gate["coverage_ok"] is True
    assert gate["regressions"][0]["question_id"] == "one"
    assert "completion" in gate["regressions"][0]["reasons"]


def test_complete_equal_arms_pass_the_observed_gate():
    assert nonregression.evaluate(_document())["passed"] is True


@pytest.mark.parametrize("budget", [float("nan"), float("inf")])
def test_direct_live_runner_rejects_nonfinite_budget_before_paid_work(monkeypatch, tmp_path, budget):
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test")
    monkeypatch.setattr(runner, "account_usage", lambda: pytest.fail("paid path reached"))
    with pytest.raises(ValueError, match="max_cost_usd"):
        runner.run_live(
            fixtures=tmp_path, output=tmp_path / "result.json", model="unused",
            repetitions=1, seed=1, variants=("descriptive",), max_live_calls=1,
            max_turns=1, timeout=1, smoke=True, max_cost_usd=budget,
        )


@pytest.mark.parametrize("usage", [float("nan"), float("inf"), True, -1])
def test_unusable_provider_usage_fails_closed(monkeypatch, usage):
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test")
    monkeypatch.setattr(
        runner.urllib.request, "urlopen",
        lambda *_args, **_kwargs: io.StringIO(json.dumps({"data": {"usage": usage}})),
    )
    with pytest.raises(ValueError, match="usage"):
        runner.account_usage()
