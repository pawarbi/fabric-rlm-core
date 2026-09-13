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


def _frozen_runs(tmp_path):
    root = tmp_path / "frozen"
    for source in ("csv", "parquet", "lakehouse"):
        directory = root / f"{source}.artifacts" / "packages"
        directory.mkdir(parents=True)
        for domain in ("inventory", "manufacturing", "service"):
            for arm in ("B", "C"):
                (directory / f"{domain}__descriptive__{arm}.json").write_text(
                    '{"snapshot": "test"}', encoding="utf-8",
                )
    return root


def test_policy_comparison_reuses_packages_and_alternates_order(monkeypatch, tmp_path):
    output, calls = _setup(
        monkeypatch, tmp_path, usages=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
    )
    frozen = _frozen_runs(tmp_path)
    sys.argv.extend(["--frozen-runs", str(frozen), "--knowledge-executions", "auto", "context_only"])

    assert current_smoke.main() == 0
    assert len(calls) == 6
    pairs = [calls[index:index + 2] for index in range(0, 6, 2)]
    for pair in pairs:
        assert pair[0]["representation"] == pair[1]["representation"]
        assert pair[0]["frozen_packages"] == pair[1]["frozen_packages"]
        assert {call["knowledge_execution"] for call in pair} == {"auto", "context_only"}
        assert all(call["max_live_calls"] == 27 for call in pair)
    assert pairs[0][0]["knowledge_execution"] != pairs[1][0]["knowledge_execution"]
    assert len({call["output"] for call in calls}) == 6
    manifest = json.loads((output / "batch.json").read_text())
    assert len(manifest["frozen_package_hashes"]) == 18
    assert manifest["status"] == "complete"


def test_policy_comparison_requires_identical_frozen_inputs(monkeypatch, tmp_path):
    _output, calls = _setup(monkeypatch, tmp_path, usages=[])
    sys.argv.extend(["--knowledge-executions", "auto", "context_only"])
    with pytest.raises(SystemExit) as error:
        current_smoke.main()
    assert error.value.code == 2
    assert not calls


@pytest.mark.parametrize("change", ["append", "delete"])
def test_batch_stops_if_original_frozen_packages_change(monkeypatch, tmp_path, change):
    output, calls = _setup(monkeypatch, tmp_path, usages=[0, 0, 0.01])
    frozen = _frozen_runs(tmp_path)
    sys.argv.extend(["--frozen-runs", str(frozen), "--knowledge-executions", "auto", "context_only"])
    original = current_smoke.run_live

    def mutate(**kwargs):
        result = original(**kwargs)
        path = kwargs["frozen_packages"] / "inventory__descriptive__B.json"
        if change == "delete":
            path.unlink()
        else:
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(current_smoke, "run_live", mutate)
    assert current_smoke.main() == 2
    assert len(calls) == 1
    manifest = json.loads((output / "batch.json").read_text())
    assert manifest["stop_reason"] == "frozen_package_drift"
