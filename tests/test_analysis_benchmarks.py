from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from fabric_rlm.experimental import (
    load_synthetic_benchmark,
    write_clustered_benchmark,
    write_correlated_benchmark,
    write_decomposition_benchmark,
    write_panel_benchmark,
    write_shift_benchmark,
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
    [
        write_decomposition_benchmark,
        write_clustered_benchmark,
        write_correlated_benchmark,
        write_panel_benchmark,
        write_shift_benchmark,
        write_time_series_benchmark,
    ],
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


def test_panel_benchmark_is_byte_reproducible_for_same_seed(
    tmp_path: Path,
) -> None:
    write_panel_benchmark(tmp_path / "first", root_seed=20260830)
    write_panel_benchmark(tmp_path / "second", root_seed=20260830)

    assert _bundle_bytes(tmp_path / "first") == _bundle_bytes(tmp_path / "second")


def test_panel_benchmark_is_byte_reproducible_across_hash_seeds(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from fabric_rlm.experimental import write_panel_benchmark

        write_panel_benchmark(Path(sys.argv[1]), root_seed=73)
        """
    )
    for directory, hash_seed in (("first", "1"), ("second", "2")):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / directory)],
            check=True,
            env=environment,
        )

    assert _bundle_bytes(tmp_path / "first") == _bundle_bytes(tmp_path / "second")


def test_panel_benchmark_records_cohort_funnel_and_censoring_truth(
    tmp_path: Path,
) -> None:
    bundle = write_panel_benchmark(tmp_path, root_seed=73)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))
    cohorts = {entry["cohort"]: entry for entry in truth["cohorts"]}

    assert manifest["dataset_id"] == "panel-ground-truth-v1"
    assert {source["identity"] for source in manifest["sources"]} == {
        "customers",
        "events",
    }
    assert cohorts["2026-01"]["eligible_for_day_90_retention"] is True
    assert cohorts["2026-01"]["day_90_retained"] == 24
    assert cohorts["2026-01"]["day_90_retention_denominator"] == 30
    assert cohorts["2026-01"]["day_90_retention_rate"] == pytest.approx(24 / 30)
    assert cohorts["2026-03"]["day_90_retained"] == 12
    assert cohorts["2026-04"]["eligible_for_day_90_retention"] is False
    assert cohorts["2026-04"]["day_90_retained"] is None
    assert cohorts["2026-04"]["day_90_retention_denominator"] is None
    assert cohorts["2026-04"]["day_90_retention_rate"] is None
    assert truth["funnel"]["signup"] == 240
    assert truth["funnel"]["activated"] == 175
    assert truth["funnel"]["converted"] == 98
    assert truth["funnel"]["eligible_day_90_retained"] == 56
    assert truth["funnel"]["counting_rule"] == "distinct customer_id"
    assert truth["censoring"]["observation_cutoff_day"] == 180
    assert truth["censoring"]["minimum_retention_exposure_days"] == 90
    assert truth["censoring"]["censored_cohorts"] == ["2026-04"]

    with bundle.worker_source_paths["events"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        events = list(csv.DictReader(handle))
    april_retention = [
        event
        for event in events
        if event["cohort"] == "2026-04"
        and event["event_type"] == "retained_day_90"
    ]
    assert april_retention == []

    events_by_customer: dict[str, set[str]] = {}
    for event in events:
        events_by_customer.setdefault(event["customer_id"], set()).add(
            event["event_type"]
        )
    assert all(
        "converted" in event_types
        for event_types in events_by_customer.values()
        if "retained_day_90" in event_types
    )
    distinct_customers_by_event = {
        event_type: {
            event["customer_id"]
            for event in events
            if event["event_type"] == event_type
        }
        for event_type in ("signup", "activated", "converted", "retained_day_90")
    }
    assert len(distinct_customers_by_event["signup"]) == truth["funnel"]["signup"]
    assert len(distinct_customers_by_event["activated"]) == truth["funnel"][
        "activated"
    ]
    assert len(distinct_customers_by_event["converted"]) == truth["funnel"][
        "converted"
    ]
    assert len(distinct_customers_by_event["retained_day_90"]) == truth["funnel"][
        "eligible_day_90_retained"
    ]


def test_panel_benchmark_records_duplicate_and_leakage_traps(
    tmp_path: Path,
) -> None:
    bundle = write_panel_benchmark(tmp_path, root_seed=73)
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))

    assert len(truth["data_quality"]["duplicate_event_ids"]) == 8
    assert truth["data_quality"]["duplicate_event_rows"] == 8
    assert truth["leakage"]["prohibited_feature_columns"] == [
        "future_converted_label",
        "future_retained_day_90_label",
    ]
    assert truth["leakage"]["reason"] == "post-outcome information"


def test_panel_seed_changes_assignments_but_not_aggregate_truth(
    tmp_path: Path,
) -> None:
    first = write_panel_benchmark(tmp_path / "first", root_seed=1)
    second = write_panel_benchmark(tmp_path / "second", root_seed=2)
    first_truth = json.loads(first.truth_path.read_text(encoding="utf-8"))
    second_truth = json.loads(second.truth_path.read_text(encoding="utf-8"))

    assert first.dataset_fingerprint != second.dataset_fingerprint
    assert first.worker_source_paths["customers"].read_bytes() != second.worker_source_paths[
        "customers"
    ].read_bytes()
    assert first_truth["cohorts"] == second_truth["cohorts"]
    assert first_truth["funnel"] == second_truth["funnel"]


def test_panel_benchmark_loader_verifies_sources_and_truth(tmp_path: Path) -> None:
    written = write_panel_benchmark(tmp_path, root_seed=73)

    loaded = load_synthetic_benchmark(written.manifest_path)

    assert loaded.dataset_id == "panel-ground-truth-v1"
    assert loaded.worker_source_paths == written.worker_source_paths
    assert loaded.dataset_fingerprint == written.dataset_fingerprint


def _pearson_correlation(left: list[float], right: list[float]) -> float:
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    covariance = math.fsum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_variance = math.fsum((value - left_mean) ** 2 for value in left)
    right_variance = math.fsum((value - right_mean) ** 2 for value in right)
    return covariance / math.sqrt(left_variance * right_variance)


def test_correlated_benchmark_is_byte_reproducible_across_hash_seeds(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from fabric_rlm.experimental import write_correlated_benchmark

        write_correlated_benchmark(Path(sys.argv[1]), root_seed=91)
        """
    )
    for directory, hash_seed in (("first", "1"), ("second", "2")):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / directory)],
            check=True,
            env=environment,
        )

    assert _bundle_bytes(tmp_path / "first") == _bundle_bytes(tmp_path / "second")


def test_correlated_benchmark_records_structure_and_validation_truth(
    tmp_path: Path,
) -> None:
    bundle = write_correlated_benchmark(tmp_path, root_seed=91)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))

    assert manifest["dataset_id"] == "correlated-tabular-ground-truth-v1"
    assert manifest["sources"][0]["identity"] == "observations"
    assert manifest["sources"][0]["row_count"] == 480
    assert truth["structure"] == {
        "group_count": 120,
        "observations_per_group": 4,
        "row_count": 480,
    }
    assert truth["validation"]["group_key"] == "group_id"
    assert truth["validation"]["identifier_columns"] == ["row_id", "group_id"]
    assert truth["validation"]["required_strategy"] == "grouped"
    assert (
        truth["validation"]["model_selection_strategy"]
        == "nested_grouped_cross_validation"
    )
    assert truth["validation"]["preprocessing_scope"] == "training_fold_only"
    assert truth["validation"]["untouched_final_holdout_required"] is True
    assert truth["data_generating_process"]["direct_terms"] == {
        "confounder": 2.0,
        "interaction_left_x_right": 4.0,
        "linear_signal": 3.0,
        "missing_signal": 1.5,
        "nonlinear_signal_squared": -2.0,
    }
    assert truth["data_generating_process"]["non_driver_features"] == [
        "correlated_linear_proxy",
        "confounder_proxy",
        "nuisance",
    ]


def test_correlated_benchmark_contains_known_collinearity_and_missingness(
    tmp_path: Path,
) -> None:
    bundle = write_correlated_benchmark(tmp_path, root_seed=91)
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))
    with bundle.worker_source_paths["observations"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert {
            "group_effect",
            "missingness_score",
            "outcome_noise",
        }.isdisjoint(reader.fieldnames)

    group_counts: dict[str, int] = {}
    for row in rows:
        group_counts[row["group_id"]] = group_counts.get(row["group_id"], 0) + 1
    assert set(group_counts.values()) == {4}

    linear = [float(row["linear_signal"]) for row in rows]
    linear_proxy = [float(row["correlated_linear_proxy"]) for row in rows]
    confounder = [float(row["confounder"]) for row in rows]
    confounder_proxy = [float(row["confounder_proxy"]) for row in rows]
    assert _pearson_correlation(linear, linear_proxy) > 0.99
    assert _pearson_correlation(confounder, confounder_proxy) > 0.98

    missing_row_ids = {
        row["row_id"] for row in rows if row["missing_signal"] == ""
    }
    assert len(missing_row_ids) == 72
    assert missing_row_ids == set(truth["missingness"]["missing_row_ids"])
    assert truth["missingness"]["mechanism"] == "MAR"
    assert truth["missingness"]["selection_feature"] == "confounder"


def test_correlated_benchmark_target_reconciles_to_hidden_components(
    tmp_path: Path,
) -> None:
    bundle = write_correlated_benchmark(tmp_path, root_seed=91)
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))
    row_truth = truth["row_truth"]
    with bundle.worker_source_paths["observations"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        components = row_truth[row["row_id"]]
        reconstructed = truth["data_generating_process"]["intercept"] + math.fsum(
            components.values()
        )
        assert float(row["target"]) == pytest.approx(reconstructed, abs=1e-12)


def test_correlated_benchmark_seed_changes_rows_but_not_ground_truth(
    tmp_path: Path,
) -> None:
    first = write_correlated_benchmark(tmp_path / "first", root_seed=1)
    second = write_correlated_benchmark(tmp_path / "second", root_seed=2)
    first_truth = json.loads(first.truth_path.read_text(encoding="utf-8"))
    second_truth = json.loads(second.truth_path.read_text(encoding="utf-8"))

    assert first.dataset_fingerprint != second.dataset_fingerprint
    assert first.worker_source_paths["observations"].read_bytes() != second.worker_source_paths[
        "observations"
    ].read_bytes()
    assert first_truth["structure"] == second_truth["structure"]
    assert (
        first_truth["data_generating_process"]
        == second_truth["data_generating_process"]
    )


def test_clustered_benchmark_is_byte_reproducible_across_hash_seeds(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from fabric_rlm.experimental import write_clustered_benchmark

        write_clustered_benchmark(Path(sys.argv[1]), root_seed=104)
        """
    )
    for directory, hash_seed in (("first", "1"), ("second", "2")):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / directory)],
            check=True,
            env=environment,
        )

    assert _bundle_bytes(tmp_path / "first") == _bundle_bytes(tmp_path / "second")


def test_clustered_benchmark_records_cluster_and_contamination_truth(
    tmp_path: Path,
) -> None:
    bundle = write_clustered_benchmark(tmp_path, root_seed=104)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))

    assert manifest["dataset_id"] == "clustered-contaminated-ground-truth-v1"
    assert manifest["sources"][0]["identity"] == "observations"
    assert manifest["sources"][0]["row_count"] == 360
    assert truth["structure"]["row_count"] == 360
    assert truth["structure"]["batch_count"] == 12
    assert truth["structure"]["rows_per_batch"] == 30
    assert truth["clusters"]["expected_counts"] == {
        "cluster_a": 180,
        "cluster_b": 120,
        "cluster_c": 60,
    }
    assert truth["clusters"]["scoring_scope"] == "non_point_anomalies"
    assert truth["contamination"]["point_anomaly_count"] == 18
    assert truth["contamination"]["displacement_per_axis"] == 8.0
    assert truth["batch_anomalies"]["anomalous_batches"] == [
        "batch-003",
        "batch-009",
    ]
    assert truth["validation"]["scaling_required"] is True
    assert truth["validation"]["identifier_columns"] == ["row_id", "batch_id"]


def test_clustered_benchmark_exercises_scale_and_batch_anomalies(
    tmp_path: Path,
) -> None:
    bundle = write_clustered_benchmark(tmp_path, root_seed=104)
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))
    with bundle.worker_source_paths["observations"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert {"cluster_label", "is_point_anomaly"}.isdisjoint(reader.fieldnames)

    batch_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        batch_rows.setdefault(row["batch_id"], []).append(row)
    assert set(len(entries) for entries in batch_rows.values()) == {30}

    cluster_x = [float(row["cluster_x"]) for row in rows]
    large_scale = [float(row["large_scale_nuisance"]) for row in rows]
    cluster_x_mean = math.fsum(cluster_x) / len(cluster_x)
    large_scale_mean = math.fsum(large_scale) / len(large_scale)
    cluster_x_sd = math.sqrt(
        math.fsum((value - cluster_x_mean) ** 2 for value in cluster_x)
        / len(cluster_x)
    )
    large_scale_sd = math.sqrt(
        math.fsum((value - large_scale_mean) ** 2 for value in large_scale)
        / len(large_scale)
    )
    assert large_scale_sd > 100 * cluster_x_sd

    batch_means = {
        batch_id: math.fsum(float(row["batch_signal"]) for row in entries)
        / len(entries)
        for batch_id, entries in batch_rows.items()
    }
    assert batch_means["batch-003"] > 4.0
    assert batch_means["batch-009"] < -4.0
    assert all(
        abs(mean) < 1e-12
        for batch_id, mean in batch_means.items()
        if batch_id not in truth["batch_anomalies"]["anomalous_batches"]
    )


def test_clustered_benchmark_point_anomalies_are_far_from_hidden_centers(
    tmp_path: Path,
) -> None:
    bundle = write_clustered_benchmark(tmp_path, root_seed=104)
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))
    with bundle.worker_source_paths["observations"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = {
            row["row_id"]: row
            for row in csv.DictReader(handle)
        }

    anomaly_distances: list[float] = []
    clean_distances: list[float] = []
    for row_id, labels in truth["row_truth"].items():
        row = rows[row_id]
        center = truth["clusters"]["centers"][labels["cluster_label"]]
        distance = math.hypot(
            float(row["cluster_x"]) - center[0],
            float(row["cluster_y"]) - center[1],
        )
        if labels["is_point_anomaly"]:
            anomaly_distances.append(distance)
        else:
            clean_distances.append(distance)

    assert len(anomaly_distances) == 18
    assert min(anomaly_distances) > 6.0
    assert max(clean_distances) < 2.5


def test_clustered_benchmark_seed_changes_assignments_not_design(
    tmp_path: Path,
) -> None:
    first = write_clustered_benchmark(tmp_path / "first", root_seed=1)
    second = write_clustered_benchmark(tmp_path / "second", root_seed=2)
    first_truth = json.loads(first.truth_path.read_text(encoding="utf-8"))
    second_truth = json.loads(second.truth_path.read_text(encoding="utf-8"))

    assert first.dataset_fingerprint != second.dataset_fingerprint
    assert first.worker_source_paths["observations"].read_bytes() != second.worker_source_paths[
        "observations"
    ].read_bytes()
    assert first_truth["structure"] == second_truth["structure"]
    assert first_truth["clusters"] == second_truth["clusters"]
    assert first_truth["contamination"] == second_truth["contamination"]
    assert first_truth["batch_anomalies"] == second_truth["batch_anomalies"]


def test_shift_benchmark_is_byte_reproducible_across_hash_seeds(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from fabric_rlm.experimental import write_shift_benchmark

        write_shift_benchmark(Path(sys.argv[1]), root_seed=205)
        """
    )
    for directory, hash_seed in (("first", "1"), ("second", "2")):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / directory)],
            check=True,
            env=environment,
        )

    assert _bundle_bytes(tmp_path / "first") == _bundle_bytes(tmp_path / "second")


def test_shift_benchmark_records_holdout_and_subgroup_truth(
    tmp_path: Path,
) -> None:
    bundle = write_shift_benchmark(tmp_path, root_seed=205)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))

    assert manifest["dataset_id"] == "distribution-shift-ground-truth-v1"
    assert manifest["sources"][0]["row_count"] == 1000
    assert truth["partitions"] == {
        "development": {
            "row_count": 700,
            "subgroup_counts": {"majority": 560, "minority": 112, "rare": 28},
        },
        "final_holdout": {
            "row_count": 300,
            "subgroup_counts": {"majority": 240, "minority": 48, "rare": 12},
        },
    }
    assert truth["validation"]["partition_column"] == "partition"
    assert truth["validation"]["final_holdout_value"] == "final_holdout"
    assert truth["validation"]["subgroup_column"] == "subgroup"
    assert truth["validation"]["minimum_subgroup_support"] == 30
    assert truth["validation"]["preprocessing_scope"] == "training_fold_only"
    assert truth["validation"]["required_metrics"] == [
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "calibration",
    ]


def test_shift_benchmark_contains_covariate_shift_and_mnar_missingness(
    tmp_path: Path,
) -> None:
    bundle = write_shift_benchmark(tmp_path, root_seed=205)
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))
    with bundle.worker_source_paths["observations"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert {
            "complete_risk_marker",
            "label_probability",
            "missingness_score",
        }.isdisjoint(reader.fieldnames)

    by_partition = {
        partition: [row for row in rows if row["partition"] == partition]
        for partition in ("development", "final_holdout")
    }
    shift_means = {
        partition: math.fsum(float(row["shift_feature"]) for row in entries)
        / len(entries)
        for partition, entries in by_partition.items()
    }
    assert shift_means["final_holdout"] - shift_means["development"] > 1.2

    missing_counts = {
        partition: sum(row["risk_marker"] == "" for row in entries)
        for partition, entries in by_partition.items()
    }
    assert missing_counts == {"development": 70, "final_holdout": 90}
    assert truth["missingness"]["mechanism"] == "MNAR"
    assert truth["missingness"]["development_rate"] == pytest.approx(0.1)
    assert truth["missingness"]["final_holdout_rate"] == pytest.approx(0.3)
    assert set(truth["missingness"]["missing_row_ids"]) == {
        row["row_id"] for row in rows if row["risk_marker"] == ""
    }


def test_shift_benchmark_hidden_probabilities_match_declared_logit(
    tmp_path: Path,
) -> None:
    bundle = write_shift_benchmark(tmp_path, root_seed=205)
    truth = json.loads(bundle.truth_path.read_text(encoding="utf-8"))
    with bundle.worker_source_paths["observations"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = {
            row["row_id"]: row
            for row in csv.DictReader(handle)
        }

    for row_id, hidden in truth["row_truth"].items():
        expected_probability = 1.0 / (1.0 + math.exp(-hidden["logit"]))
        assert hidden["label_probability"] == pytest.approx(
            expected_probability,
            abs=1e-15,
        )
        assert rows[row_id]["target"] in {"0", "1"}
        assert hidden["is_mnar_missing"] == (rows[row_id]["risk_marker"] == "")


def test_shift_benchmark_seed_changes_rows_not_design(
    tmp_path: Path,
) -> None:
    first = write_shift_benchmark(tmp_path / "first", root_seed=1)
    second = write_shift_benchmark(tmp_path / "second", root_seed=2)
    first_truth = json.loads(first.truth_path.read_text(encoding="utf-8"))
    second_truth = json.loads(second.truth_path.read_text(encoding="utf-8"))

    assert first.dataset_fingerprint != second.dataset_fingerprint
    assert first.worker_source_paths["observations"].read_bytes() != second.worker_source_paths[
        "observations"
    ].read_bytes()
    assert first_truth["partitions"] == second_truth["partitions"]
    assert first_truth["data_generating_process"] == second_truth[
        "data_generating_process"
    ]
    assert first_truth["covariate_shift"] == second_truth["covariate_shift"]
