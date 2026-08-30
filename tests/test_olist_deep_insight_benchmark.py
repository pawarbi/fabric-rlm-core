from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "olist_deep_insight_benchmark.py"
)
CANONICAL_FILES = (
    "customers.csv",
    "geolocation.csv",
    "order_items.csv",
    "order_payments.csv",
    "order_reviews.csv",
    "orders.csv",
    "product_category_name_translation.csv",
    "products.csv",
    "sellers.csv",
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "olist_deep_insight_benchmark", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bundle(data_dir: Path) -> None:
    data_dir.mkdir()
    for name in CANONICAL_FILES:
        (data_dir / name).write_text("header\n", encoding="utf-8")


def test_source_discovery_returns_deterministic_canonical_map_and_ignores_extras(
    tmp_path: Path,
) -> None:
    bench = load_module()
    make_bundle(tmp_path / "caller-data")
    (tmp_path / "caller-data" / "order_reviews.fabric.csv").write_text(
        "ignored\n", encoding="utf-8"
    )

    sources = bench.discover_sources(tmp_path / "caller-data")

    assert list(sources) == [Path(name).stem for name in CANONICAL_FILES]
    assert sources["orders"] == tmp_path / "caller-data" / "orders.csv"
    assert "order_reviews.fabric" not in sources


def test_source_discovery_lists_every_missing_canonical_file(tmp_path: Path) -> None:
    bench = load_module()
    data_dir = tmp_path / "incomplete"
    data_dir.mkdir()
    (data_dir / "orders.csv").write_text("header\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as captured:
        bench.discover_sources(data_dir)

    message = str(captured.value)
    for missing in set(CANONICAL_FILES) - {"orders.csv"}:
        assert missing in message
    assert "orders.csv" not in message


def test_prompt_contains_paths_contract_grain_controls_and_analysis_gates(
    tmp_path: Path,
) -> None:
    bench = load_module()
    make_bundle(tmp_path / "portable-bundle")
    sources = bench.discover_sources(tmp_path / "portable-bundle")

    prompt = bench.build_task_prompt(sources)
    lowered = " ".join(prompt.lower().split())

    for identity, path in sources.items():
        assert identity in prompt
        assert str(path) in prompt
    assert "contract_version: 2" in prompt
    assert "deep-insight skill contract" in lowered
    assert "3-5" in prompt and "quality" in lowered
    assert "at least two" in lowered and "cross-domain" in lowered
    assert "join map" in lowered and "coverage" in lowered
    assert "order_items" in lowered and "before joining to orders" in lowered
    assert "zip prefix" in lowered and "pre-aggregate" in lowered
    for method in (
        "decomposition",
        "instrumentation diagnostics",
        "change points",
        "cohorts",
        "interactions",
        "driver analysis",
        "concentration",
        "clustering",
        "classification",
        "regression",
    ):
        assert method in lowered
    for guardrail in (
        "count-vs-rate",
        "causal",
        "rejected candidates",
        "sensitivity",
        "benchmarked",
        "headline",
    ):
        assert guardrail in lowered
    implementation = MODULE_PATH.read_text(encoding="utf-8")
    assert "C:\\Users\\sandeeppawar" not in implementation


def test_default_model_is_glm_5_3_flash() -> None:
    bench = load_module()

    assert bench.DEFAULT_MODEL == "openrouter/z-ai/glm-5.3-flash"


def test_payload_normalization_accepts_object_and_json_string() -> None:
    bench = load_module()
    expected = {"contract_version": 2, "insights": []}

    assert bench.normalize_payload(expected) == expected
    assert bench.normalize_payload(json.dumps(expected)) == expected


@pytest.mark.parametrize("value", [None, "", "not json", "[]", "42", []])
def test_payload_normalization_rejects_malformed_or_non_object_output(value) -> None:
    bench = load_module()

    with pytest.raises(ValueError, match="payload"):
        bench.normalize_payload(value)


def test_extract_payload_reads_rlm_result_payload_insights_style() -> None:
    bench = load_module()

    class Result:
        payload = {"contract_version": 2, "insights": [{"title": "Finding"}]}

    assert bench.extract_payload(Result())["insights"][0]["title"] == "Finding"


@dataclass(frozen=True)
class FakeCheck:
    path: str
    expected: float
    actual: float


@dataclass(frozen=True)
class FakeAudit:
    checks: tuple[FakeCheck, ...]

    @property
    def total_checks(self) -> int:
        return len(self.checks)


def test_artifacts_are_json_deterministic_shaped_and_exclude_credentials(
    tmp_path: Path,
) -> None:
    bench = load_module()
    payload = {
        "contract_version": 2,
        "insights": [{"title": "Aggregate-only finding"}],
        "candidates": [],
    }
    audit = FakeAudit((FakeCheck("insights[0]", 3.0, 3.0),))
    record = {
        "payload": payload,
        "audit": audit,
        "trajectory": {"turns": 4, "submitted": True},
        "model": "openrouter/test-model",
        "skills": ("data_exploration", "deep_insight_discovery"),
    }

    paths = bench.write_artifacts(tmp_path / "artifacts", record)

    assert set(paths) == {"payload", "audit", "run"}
    payload_data = json.loads(paths["payload"].read_text(encoding="utf-8"))
    audit_data = json.loads(paths["audit"].read_text(encoding="utf-8"))
    run_data = json.loads(paths["run"].read_text(encoding="utf-8"))
    assert payload_data == payload
    assert audit_data["status"] == "passed"
    assert audit_data["total_checks"] == 1
    assert run_data["status"] == "success"
    assert run_data["counts"] == {
        "audit_checks": 1,
        "candidates": 0,
        "insights": 1,
        "trajectory_turns": 4,
    }
    combined = "".join(path.read_text(encoding="utf-8") for path in paths.values())
    assert "api_key" not in combined.lower()
    assert "OPENROUTER_API_KEY" not in combined
    assert not list((tmp_path / "artifacts").glob("*.tmp"))
    first = paths["run"].read_bytes()
    bench.write_artifacts(tmp_path / "artifacts", record)
    assert paths["run"].read_bytes() == first


def test_run_benchmark_wires_skills_verifier_network_block_and_host_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    data_dir = tmp_path / "olist"
    make_bundle(data_dir)
    calls = {}

    class FakeDspy:
        @staticmethod
        def LM(model, **kwargs):
            calls["lm"] = (model, kwargs)
            return "configured-lm"

        @staticmethod
        def configure(**kwargs):
            calls["configure"] = kwargs

    class FakeResult:
        payload = {"contract_version": 2, "insights": [], "candidates": []}
        trajectory = object()

    class FakeRLMInstance:
        def run(self):
            calls["run"] = True
            return FakeResult()

    class FakeRLM:
        @classmethod
        def from_task(cls, **kwargs):
            calls["from_task"] = kwargs
            return FakeRLMInstance()

    class FakeExecutor:
        def __init__(self, sources):
            calls["executor_sources"] = sources

        def __enter__(self):
            calls["executor_entered"] = True
            return self

        def __exit__(self, *exc_info):
            calls["executor_closed"] = True

    def fake_audit(payload, executor):
        calls["audit"] = (payload, executor)
        return FakeAudit(())

    monkeypatch.setattr(
        bench,
        "_load_runtime_dependencies",
        lambda: (
            FakeDspy,
            FakeRLM,
            FakeExecutor,
            fake_audit,
            lambda trajectory: {"turns": 3, "submitted": True},
        ),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-for-test")

    record = bench.run_benchmark(
        data_dir,
        model="openrouter/test-model",
        api_base="https://example.invalid/v1",
        max_turns=21,
        timeout=1200,
    )

    kwargs = calls["from_task"]
    assert kwargs["skills"] == ["data_exploration", "deep_insight_discovery"]
    assert kwargs["enable_verifier"] is True
    assert kwargs["block_network"] is True
    assert kwargs["max_turns"] == 21
    assert kwargs["timeout"] == 1200
    assert kwargs["reserve_finalize_turns"] == 6
    assert calls["run"] is True
    assert calls["lm"][1]["max_tokens"] == 20000
    assert calls["lm"][1]["reasoning"] == {
        "max_tokens": 4096,
        "exclude": True,
    }
    assert calls["executor_sources"] == bench.discover_sources(data_dir)
    assert calls["executor_entered"] is True
    assert calls["executor_closed"] is True
    assert calls["audit"][0] is record["payload"]
    assert record["trajectory"] == {"turns": 3, "submitted": True}
    assert record["model"] == "openrouter/test-model"
    assert record["skills"] == bench.DEFAULT_SKILLS


def test_run_benchmark_requires_openrouter_key_before_runtime_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = load_module()
    make_bundle(tmp_path / "olist")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        bench,
        "_load_runtime_dependencies",
        lambda: pytest.fail("runtime dependencies should not be loaded"),
    )

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        bench.run_benchmark(tmp_path / "olist")
