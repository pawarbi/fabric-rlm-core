from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "manifest_staged_deep_insight_benchmark.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "manifest_staged_deep_insight_benchmark",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_manifest(tmp_path: Path) -> Path:
    entries = []
    for identity in ("accounts", "opportunities"):
        csv_path = tmp_path / f"{identity}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(
                [["id", "status"], ["1", "Closed"]]
            )
        payload = csv_path.read_bytes()
        entries.append(
            {
                "table": identity,
                "source_db": "crm.db",
                "source_table": identity.title(),
                "file_path": str(csv_path),
                "row_count": 1,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "selection_rationale": "CRM source",
            }
        )
    manifest_path = tmp_path / "export_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "selection_rationale": "Unseen cross-domain CRM benchmark",
                "normalization_note": "Identifiers and timestamps normalized",
                "tables": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_manifest_research_prompt_is_source_agnostic_and_privacy_bounded(
    tmp_path: Path,
) -> None:
    bench = load_module()
    manifest_path = make_manifest(tmp_path)

    prompt = bench.build_manifest_research_prompt(manifest_path)
    lowered = " ".join(prompt.lower().split())

    assert "unseen cross-domain crm benchmark" in lowered
    assert "identifiers and timestamps normalized" in lowered
    assert "olist" not in lowered
    assert "source-agnostic" in lowered
    assert "no raw records" in lowered
    assert "free-text" in lowered
    assert "personal identifiers" in lowered
    assert "aggregate" in lowered
    for identity in ("accounts", "opportunities"):
        assert f"{identity}: {tmp_path / f'{identity}.csv'}" in prompt


def test_run_manifest_benchmark_forwards_verified_sources_and_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bench = load_module()
    manifest_path = make_manifest(tmp_path)
    captured = {}

    def fake_run(data_dir, **kwargs):
        captured["data_dir"] = data_dir
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(bench._STAGED, "run_staged_benchmark", fake_run)

    result = bench.run_manifest_benchmark(
        manifest_path,
        model="openrouter/test",
        output_dir=tmp_path / "output",
    )

    assert result == {"status": "complete"}
    assert captured["data_dir"] == tmp_path
    assert captured["manifest_path"] == manifest_path
    assert captured["research_cache_path"] == tmp_path / "output" / "research.json"
    assert captured["scaffold_cache_path"] == (
        tmp_path / "output" / "contract_scaffold.checkpoint.json"
    )
    assert captured["insights_cache_path"] == (
        tmp_path / "output" / "insights.checkpoint.json"
    )


def test_main_writes_staged_artifacts_for_manifest_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bench = load_module()
    manifest_path = make_manifest(tmp_path)
    output_dir = tmp_path / "artifacts"
    captured = {}
    record = {"audit": type("Audit", (), {"total_checks": 7})()}

    monkeypatch.setattr(
        bench,
        "run_manifest_benchmark",
        lambda manifest, **kwargs: captured.update(
            {"manifest": manifest, **kwargs}
        )
        or record,
    )
    monkeypatch.setattr(
        bench._STAGED,
        "write_staged_artifacts",
        lambda output, actual: {
            "payload": Path(output) / "payload.json",
            "audit": Path(output) / "audit.json",
        },
    )

    assert bench.main(
        [
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    assert captured["manifest"] == manifest_path
    assert captured["output_dir"] == output_dir
