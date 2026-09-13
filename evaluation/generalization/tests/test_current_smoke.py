import json
import sys

import pytest

from evaluation.generalization import current_smoke


def _setup(monkeypatch, tmp_path, *, usages, budget="2"):
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "current_smoke", "--fixtures", str(tmp_path / "data"),
        "--output", str(output), "--max-cost-usd", budget, "--repetitions", "3",
    ])
    values = iter(usages)
    monkeypatch.setattr(current_smoke, "account_usage", lambda: next(values))
    monkeypatch.setattr(current_smoke, "version", lambda _name: "test")
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return {"status": "complete", "trials": [{}] * 27, "gate": {"passed": False}}

    monkeypatch.setattr(current_smoke, "run_live", run)
    return output, calls


def test_batch_shares_one_ceiling_and_keeps_execution_separate_from_parity(monkeypatch, tmp_path):
    output, calls = _setup(monkeypatch, tmp_path, usages=[0, 0.1, 0.3, 0.7, 0.8])

    assert current_smoke.main() == 0
    assert [call["max_cost_usd"] for call in calls] == pytest.approx([1.9, 1.7, 1.3])
    assert all(call["max_live_calls"] == 33 for call in calls)
    assert {call["representation"] for call in calls} == {"csv", "parquet", "lakehouse"}
    manifest = json.loads((output / "batch.json").read_text())
    assert manifest["status"] == "complete"
    assert all(run["gate_passed"] is False for run in manifest["runs"])
    with pytest.raises(FileExistsError):
        current_smoke.main()


def test_batch_names_budget_exhaustion_without_starting_another_source(monkeypatch, tmp_path):
    output, calls = _setup(monkeypatch, tmp_path, usages=[0, 1.1, 1.1])

    assert current_smoke.main() == 2
    assert not calls
    manifest = json.loads((output / "batch.json").read_text())
    assert manifest["stop_reason"] == "budget_reserve"


@pytest.mark.parametrize("budget", ["nan", "inf"])
def test_batch_rejects_a_nonfinite_ceiling_before_paid_work(monkeypatch, tmp_path, budget):
    _output, calls = _setup(
        monkeypatch, tmp_path, usages=[0, 0, 0, 0, 0], budget=budget,
    )
    with pytest.raises(SystemExit) as error:
        current_smoke.main()
    assert error.value.code == 2
    assert not calls
