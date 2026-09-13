from dataclasses import replace
from types import SimpleNamespace

import fabric_rlm
import pytest

from evaluation.generalization import runner
from evaluation.generalization.fixtures import generate_fixtures
from fabric_rlm.trajectory import Trajectory


def _save_pair(directory, domain, sources):
    learned = fabric_rlm.RLM.learn(sources=sources)
    expected = {}
    for arm in ("B", "C"):
        package = replace(learned.package, package_id=f"test-{domain}-{arm}")
        knowledge = SimpleNamespace(package=package)
        runner.snapshot_package(directory / f"{domain}__descriptive__{arm}.json", knowledge)
        expected[arm] = package.fingerprint
    return expected


def test_frozen_loading_preserves_both_packages_and_never_relearns(monkeypatch, tmp_path):
    source = tmp_path / "stock.csv"
    source.write_text("location,units\nA,3\nB,4\n", encoding="utf-8")
    sources = {"stock": fabric_rlm.File(source)}
    expected = _save_pair(tmp_path, "inventory", sources)
    monkeypatch.setattr(fabric_rlm.RLM, "learn", lambda **_: pytest.fail("relearning"))
    monkeypatch.setattr(fabric_rlm.RLM, "enrich", lambda *_a, **_k: pytest.fail("reenriching"))

    packages, snapshots = runner._load_frozen_packages(
        tmp_path, domain="inventory", variant="descriptive", sources=sources,
    )

    assert {arm: item.package.fingerprint for arm, item in packages.items()} == expected
    assert expected["B"] != expected["C"]
    for arm, item in packages.items():
        runner.check_package(item, snapshots[arm])
        assert set(item.bindings) == {"stock"}
    path = tmp_path / "inventory__descriptive__C.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="saved evaluation package changed"):
        runner.check_package(packages["C"], snapshots["C"])


def test_frozen_loading_rejects_changed_data(tmp_path):
    source = tmp_path / "stock.csv"
    source.write_text("units\n3\n", encoding="utf-8")
    sources = {"stock": fabric_rlm.File(source)}
    _save_pair(tmp_path, "inventory", sources)
    source.write_text("units\n4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stale knowledge"):
        runner._load_frozen_packages(
            tmp_path, domain="inventory", variant="descriptive", sources=sources,
        )


@pytest.mark.parametrize("policy", ["auto", "context_only"])
def test_run_rlm_forwards_only_the_requested_policy(monkeypatch, policy):
    captured = {}
    sentinel = object()
    lm = object()
    monkeypatch.setattr(runner, "make_openrouter_lm", lambda _model: lm)

    def construct(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(run=lambda: sentinel)

    monkeypatch.setattr(fabric_rlm.RLM, "from_task", construct)
    result, used_lm, _wall = runner._run_rlm(
        model="test", question={"domain": "inventory", "text": "Compute a total."},
        definitions={"variants": {"descriptive": {"inventory": {}}}},
        variant="descriptive", inputs={"stock": "stock.csv"}, knowledge=None,
        max_turns=6, timeout=120, knowledge_execution=policy,
    )
    assert result is sentinel and used_lm is lm
    assert captured["knowledge_execution"] == policy
    assert captured["max_turns"] == 6 and captured["timeout"] == 120
    assert captured["skills"] == [] and captured["enable_skill_autoloading"] is False


def test_live_frozen_run_uses_no_development_budget(monkeypatch, tmp_path):
    fixtures = tmp_path / "fixtures"
    generate_fixtures(fixtures, large_rows=8)
    directory = tmp_path / "packages"
    for domain in ("inventory", "manufacturing", "service"):
        _save_pair(directory, domain, runner._domain_sources(fixtures, domain, "descriptive"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test")
    monkeypatch.setattr(runner, "_development_results", lambda **_: pytest.fail("development ran"))
    monkeypatch.setattr(fabric_rlm.RLM, "learn", lambda **_: pytest.fail("relearning"))
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        result = SimpleNamespace(
            submitted=True, outputs={"answer": {"status": "abstain", "value": "test double"}},
            failure_reason=None, trajectory=Trajectory(),
        )
        return result, SimpleNamespace(history=[]), 0.01

    monkeypatch.setattr(runner, "_run_rlm", run)
    document = runner.run_live(
        fixtures=fixtures, output=tmp_path / "result.json", model="test",
        repetitions=1, seed=7, variants=("descriptive",), max_live_calls=9,
        max_turns=6, timeout=120, smoke=True, representation="csv",
        frozen_packages=directory, knowledge_execution="context_only",
    )

    assert document["status"] == "complete"
    assert len(document["trials"]) == 9 and document["remaining_live_calls"] == 0
    assert all(call["knowledge_execution"] == "context_only" for call in calls)
    assert all(item["development_runs"] == 0 for item in document["packages"].values())
    assert all(item["learn_seconds"] is None for item in document["packages"].values())
    assert len(document["frozen_package_inputs"]) == 6


def test_live_cli_records_policy_and_frozen_package_directory(tmp_path):
    args = runner._parser().parse_args([
        "live", "--fixtures", str(tmp_path), "--output", str(tmp_path / "result.json"),
        "--knowledge-execution", "context_only", "--frozen-packages", str(tmp_path / "packages"),
    ])
    assert args.knowledge_execution == "context_only"
    assert args.frozen_packages == tmp_path / "packages"


def test_git_sha_preserves_empty_configuration_after_environment_restore(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "credential.helper")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "")

    sha = runner._git_sha(runner.runtime_checkout())

    assert len(sha) == 40 and all(character in "0123456789abcdef" for character in sha)
