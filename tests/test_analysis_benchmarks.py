from __future__ import annotations

import json
from pathlib import Path

import pytest

from fabric_rlm.experimental import (
    load_synthetic_benchmark,
    write_decomposition_benchmark,
    write_time_series_benchmark,
)
from fabric_rlm.experimental.analysis_reproducibility import derive_seed


def _bundle_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_decomposition_benchmark_is_byte_reproducible_for_same_seed(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    write_decomposition_benchmark(first_root, root_seed=20260830)
    write_decomposition_benchmark(second_root, root_seed=20260830)

    assert _bundle_bytes(first_root) == _bundle_bytes(second_root)


def test_decomposition_benchmark_changes_with_root_seed(tmp_path: Path) -> None:
    first = write_decomposition_benchmark(tmp_path / "first", root_seed=1)
    second = write_decomposition_benchmark(tmp_path / "second", root_seed=2)

    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))

    assert first_manifest["root_seed"] == 1
    assert second_manifest["root_seed"] == 2
    assert first_manifest["dataset_fingerprint"] != second_manifest[
        "dataset_fingerprint"
    ]
    assert _bundle_bytes(first.manifest_path.parent) != _bundle_bytes(
        second.manifest_path.parent
    )


def test_decomposition_benchmark_separates_worker_data_from_hidden_truth(
    tmp_path: Path,
) -> None:
    bundle = write_decomposition_benchmark(tmp_path, root_seed=17)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))

    assert manifest["dataset_id"] == "decomposition-ground-truth-v1"
    assert manifest["generator_version"] == "2"
    assert manifest["random_engine"] == "python.random.Random"
    assert {entry["path"] for entry in manifest["sources"]} == {
        "additive.csv",
        "rate.csv",
        "volume_rate_mix.csv",
    }
    assert "truth.json" not in bundle.worker_source_paths
    assert {case["task"] for case in truth["cases"]} == {
        "additive",
        "rate",
        "volume_rate_mix",
    }


def test_load_synthetic_benchmark_verifies_hashes_rows_and_fingerprint(
    tmp_path: Path,
) -> None:
    written = write_decomposition_benchmark(tmp_path, root_seed=17)

    loaded = load_synthetic_benchmark(written.manifest_path)

    assert loaded.dataset_id == "decomposition-ground-truth-v1"
    assert loaded.root_seed == 17
    assert loaded.worker_source_paths == written.worker_source_paths
    assert loaded.dataset_fingerprint == written.dataset_fingerprint

    additive_path = loaded.worker_source_paths["additive"]
    corrupted = bytearray(additive_path.read_bytes())
    corrupted[-2] = ord("0") if corrupted[-2] != ord("0") else ord("1")
    additive_path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="SHA-256 mismatch.*additive"):
        load_synthetic_benchmark(written.manifest_path)


def test_load_synthetic_benchmark_rejects_tampered_truth(tmp_path: Path) -> None:
    written = write_decomposition_benchmark(tmp_path, root_seed=17)
    truth = json.loads(written.truth_path.read_text(encoding="utf-8"))
    truth["cases"][0]["expected"]["observed_change"] += 1
    written.truth_path.write_text(
        json.dumps(truth, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="truth SHA-256 mismatch"):
        load_synthetic_benchmark(written.manifest_path)


@pytest.mark.parametrize(
    "writer",
    [write_decomposition_benchmark, write_time_series_benchmark],
)
@pytest.mark.parametrize("bad_seed", [-1, True, False, 1.5, "42", None])
def test_synthetic_benchmark_writers_reject_invalid_root_seed(
    tmp_path: Path,
    writer,
    bad_seed: object,
) -> None:
    with pytest.raises(ValueError, match="root_seed"):
        writer(tmp_path / "bundle", root_seed=bad_seed)


def test_load_synthetic_benchmark_names_missing_required_source(
    tmp_path: Path,
) -> None:
    written = write_decomposition_benchmark(tmp_path, root_seed=17)
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    manifest["sources"] = [
        source
        for source in manifest["sources"]
        if source["identity"] != "rate"
    ]
    written.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="missing required source: rate"):
        load_synthetic_benchmark(written.manifest_path)


def test_load_synthetic_benchmark_rejects_duplicate_source_identity(
    tmp_path: Path,
) -> None:
    written = write_decomposition_benchmark(tmp_path, root_seed=17)
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    manifest["sources"].append(dict(manifest["sources"][0]))
    written.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="duplicate source identity: additive"):
        load_synthetic_benchmark(written.manifest_path)


def test_load_synthetic_benchmark_rejects_stale_generator_version(
    tmp_path: Path,
) -> None:
    written = write_decomposition_benchmark(tmp_path, root_seed=17)
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    manifest["generator_version"] = "1"
    written.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="unsupported generator_version: '1'"):
        load_synthetic_benchmark(written.manifest_path)


def test_decomposition_benchmark_truth_contains_exact_reconciliation_targets(
    tmp_path: Path,
) -> None:
    bundle = write_decomposition_benchmark(tmp_path, root_seed=31)
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))
    cases = {case["task"]: case for case in truth["cases"]}

    assert cases["additive"]["expected"]["reconciliation_residual"] == 0.0
    assert cases["rate"]["expected"]["reconciliation_residual"] == pytest.approx(
        0.0,
        abs=1e-15,
    )
    # VRM combines three floating-point effects; observed residuals are ~1e-13.
    assert cases["volume_rate_mix"]["expected"][
        "reconciliation_residual"
    ] == pytest.approx(0.0, abs=1e-10)


def test_time_series_benchmark_is_byte_reproducible_for_same_seed(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    write_time_series_benchmark(first_root, root_seed=20260830)
    write_time_series_benchmark(second_root, root_seed=20260830)

    assert _bundle_bytes(first_root) == _bundle_bytes(second_root)


def test_time_series_benchmark_records_known_structure_and_hidden_truth(
    tmp_path: Path,
) -> None:
    bundle = write_time_series_benchmark(tmp_path, root_seed=41)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))
    series = {entry["series_id"]: entry for entry in truth["series"]}

    assert manifest["dataset_id"] == "time-series-ground-truth-v1"
    assert manifest["derived_seeds"] == {
        "noise": derive_seed(
            41,
            dataset_id="time-series-ground-truth-v1",
            operator_id="generate.time_series.noise.v1",
        ),
        "structure": derive_seed(
            41,
            dataset_id="time-series-ground-truth-v1",
            operator_id="generate.time_series.structure.v1",
        ),
    }
    assert {source["identity"] for source in manifest["sources"]} == {
        "time_series"
    }
    assert "truth.json" not in bundle.worker_source_paths
    assert series["seasonal_shift"]["base_level"] == 100
    assert series["seasonal_shift"]["seasonal_period"] == 12
    assert series["seasonal_shift"]["trend_per_period"] == 2
    assert series["seasonal_shift"]["level_shift_index"] == 72
    assert series["seasonal_shift"]["level_shift_magnitude"] == 40
    assert series["anomaly_missing"]["base_level"] == 200
    assert series["anomaly_missing"]["total_period_count"] == 120
    assert series["anomaly_missing"]["observed_period_count"] == 116
    assert series["anomaly_missing"]["anomaly_free_ranges"] == [
        [0, 11],
        [108, 119],
    ]
    assert len(series["anomaly_missing"]["anomaly_indices"]) == 5
    assert len(series["anomaly_missing"]["missing_indices"]) == 4
    assert not (
        set(series["anomaly_missing"]["anomaly_indices"])
        & set(series["anomaly_missing"]["missing_indices"])
    )


def test_time_series_benchmark_loader_verifies_bundle(tmp_path: Path) -> None:
    written = write_time_series_benchmark(tmp_path, root_seed=41)

    loaded = load_synthetic_benchmark(written.manifest_path)

    assert loaded.dataset_id == "time-series-ground-truth-v1"
    assert loaded.worker_source_paths == written.worker_source_paths
    assert loaded.dataset_fingerprint == written.dataset_fingerprint


def test_time_series_benchmark_loader_rejects_tampered_source(
    tmp_path: Path,
) -> None:
    written = write_time_series_benchmark(tmp_path, root_seed=41)
    source_path = written.worker_source_paths["time_series"]
    corrupted = bytearray(source_path.read_bytes())
    corrupted[-2] = ord("0") if corrupted[-2] != ord("0") else ord("1")
    source_path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="SHA-256 mismatch.*time_series"):
        load_synthetic_benchmark(written.manifest_path)


def test_time_series_seed_changes_anomaly_and_missing_patterns(
    tmp_path: Path,
) -> None:
    first = write_time_series_benchmark(tmp_path / "first", root_seed=1)
    second = write_time_series_benchmark(tmp_path / "second", root_seed=2)
    first_truth = json.loads(first.truth_path.read_text(encoding="utf-8"))
    second_truth = json.loads(second.truth_path.read_text(encoding="utf-8"))

    assert first.dataset_fingerprint != second.dataset_fingerprint
    assert first_truth["series"][1]["anomaly_indices"] != second_truth["series"][1][
        "anomaly_indices"
    ]
    assert first_truth["series"][1]["missing_indices"] != second_truth["series"][1][
        "missing_indices"
    ]
