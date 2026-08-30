"""Seeded, hash-verified local benchmark bundles for analysis operators."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from types import MappingProxyType

from fabric_rlm.experimental.analysis_reproducibility import (
    derive_seed,
    fingerprint,
)


_DECOMPOSITION_DATASET_ID = "decomposition-ground-truth-v1"
_CORRELATED_DATASET_ID = "correlated-tabular-ground-truth-v1"
_CORRELATED_GROUP_COUNT = 120
_CORRELATED_OBSERVATIONS_PER_GROUP = 4
_PANEL_DATASET_ID = "panel-ground-truth-v1"
_PANEL_COHORT_SIZE = 60
_TIME_SERIES_DATASET_ID = "time-series-ground-truth-v1"
_GENERATOR_VERSION = "2"
_RANDOM_ENGINE = "python.random.Random"
_DATASET_SOURCE_FILES = {
    _CORRELATED_DATASET_ID: {
        "observations": "observations.csv",
    },
    _DECOMPOSITION_DATASET_ID: {
        "additive": "additive.csv",
        "rate": "rate.csv",
        "volume_rate_mix": "volume_rate_mix.csv",
    },
    _PANEL_DATASET_ID: {
        "customers": "customers.csv",
        "events": "events.csv",
    },
    _TIME_SERIES_DATASET_ID: {
        "time_series": "time_series.csv",
    },
}
_DATASET_SEED_NAMES = {
    _CORRELATED_DATASET_ID: {
        "measurement_noise",
        "missingness",
        "outcome_noise",
        "structure",
    },
    _DECOMPOSITION_DATASET_ID: {
        "additive",
        "rate",
        "volume_rate_mix",
    },
    _PANEL_DATASET_ID: {"attributes", "structure"},
    _TIME_SERIES_DATASET_ID: {"noise", "structure"},
}


@dataclass(frozen=True)
class SyntheticBenchmark:
    """Verified paths and identity for one local synthetic benchmark bundle."""

    dataset_id: str
    root_seed: int
    manifest_path: Path
    truth_path: Path
    worker_source_paths: Mapping[str, Path]
    dataset_fingerprint: str


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _file_record(root: Path, identity: str, path: Path, row_count: int) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "identity": identity,
        "path": path.relative_to(root).as_posix(),
        "row_count": row_count,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _dataset_fingerprint(
    *,
    dataset_id: str,
    root_seed: int,
    derived_seeds: Mapping[str, int],
    sources: list[dict[str, object]],
) -> str:
    return fingerprint(
        {
            "dataset_id": dataset_id,
            "generator_version": _GENERATOR_VERSION,
            "random_engine": _RANDOM_ENGINE,
            "root_seed": root_seed,
            "derived_seeds": derived_seeds,
            "sources": sources,
        }
    )


def _additive_case(rng: random.Random) -> tuple[list[tuple[object, ...]], dict[str, object]]:
    segments = ("enterprise", "midmarket", "smb")
    before = {segment: rng.randint(50, 200) for segment in segments}
    after = {
        segment: before[segment] + rng.randint(-30, 50)
        for segment in segments
    }
    rows = [
        ("additive-001", period, segment, values[segment])
        for period, values in (("before", before), ("after", after))
        for segment in segments
    ]
    contributions = {
        segment: after[segment] - before[segment] for segment in segments
    }
    observed_change = sum(after.values()) - sum(before.values())
    return rows, {
        "case_id": "additive-001",
        "task": "additive",
        "expected": {
            "before_total": sum(before.values()),
            "after_total": sum(after.values()),
            "observed_change": observed_change,
            "contributions": contributions,
            "reconciliation_residual": observed_change
            - sum(contributions.values()),
        },
    }


def _rate_case(rng: random.Random) -> tuple[list[tuple[object, ...]], dict[str, object]]:
    before_denominator = rng.randint(100, 300)
    after_denominator = rng.randint(100, 300)
    before_numerator = rng.randint(10, before_denominator - 10)
    after_numerator = rng.randint(10, after_denominator - 10)
    rows = [
        ("rate-001", "before", before_numerator, before_denominator),
        ("rate-001", "after", after_numerator, after_denominator),
    ]
    before_rate = before_numerator / before_denominator
    after_rate = after_numerator / after_denominator
    numerator_effect = 0.5 * (
        (after_numerator / before_denominator - before_rate)
        + (after_rate - before_numerator / after_denominator)
    )
    denominator_effect = 0.5 * (
        (before_numerator / after_denominator - before_rate)
        + (after_rate - after_numerator / before_denominator)
    )
    observed_change = after_rate - before_rate
    return rows, {
        "case_id": "rate-001",
        "task": "rate",
        "expected": {
            "before_rate": before_rate,
            "after_rate": after_rate,
            "observed_change": observed_change,
            "numerator_effect": numerator_effect,
            "denominator_effect": denominator_effect,
            "reconciliation_residual": observed_change
            - math.fsum((numerator_effect, denominator_effect)),
        },
    }


def _volume_rate_mix_case(
    rng: random.Random,
) -> tuple[list[tuple[object, ...]], dict[str, object]]:
    segments = ("enterprise", "smb")
    before = {
        segment: (rng.randint(30, 100), rng.randint(20, 120))
        for segment in segments
    }
    after = {
        segment: (
            before[segment][0] + rng.randint(-20, 30),
            before[segment][1] + rng.randint(-15, 25),
        )
        for segment in segments
    }
    rows = [
        ("volume-rate-mix-001", period, segment, values[segment][0], values[segment][1])
        for period, values in (("before", before), ("after", after))
        for segment in segments
    ]
    before_volume = sum(volume for volume, _ in before.values())
    after_volume = sum(volume for volume, _ in after.values())
    before_total = sum(volume * rate for volume, rate in before.values())
    after_total = sum(volume * rate for volume, rate in after.values())
    before_average_rate = before_total / before_volume
    after_average_rate = after_total / after_volume
    average_volume = 0.5 * (before_volume + after_volume)
    volume_effect = (after_volume - before_volume) * 0.5 * (
        before_average_rate + after_average_rate
    )
    rate_effect = math.fsum(
        average_volume
        * 0.5
        * (before[segment][0] / before_volume + after[segment][0] / after_volume)
        * (after[segment][1] - before[segment][1])
        for segment in segments
    )
    mix_effect = math.fsum(
        average_volume
        * 0.5
        * (after[segment][0] / after_volume - before[segment][0] / before_volume)
        * (before[segment][1] + after[segment][1])
        for segment in segments
    )
    observed_change = after_total - before_total
    return rows, {
        "case_id": "volume-rate-mix-001",
        "task": "volume_rate_mix",
        "expected": {
            "before_total": before_total,
            "after_total": after_total,
            "observed_change": observed_change,
            "volume_effect": volume_effect,
            "rate_effect": rate_effect,
            "mix_effect": mix_effect,
            "reconciliation_residual": observed_change
            - math.fsum((volume_effect, rate_effect, mix_effect)),
        },
    }


def write_decomposition_benchmark(
    output_dir: str | Path,
    *,
    root_seed: int,
) -> SyntheticBenchmark:
    """Write a byte-reproducible decomposition dataset and hidden truth."""

    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"output_dir must be empty: {root}")

    derived_seeds = {
        name: derive_seed(
            root_seed,
            dataset_id=_DECOMPOSITION_DATASET_ID,
            operator_id=f"generate.{name}.v1",
        )
        for name in ("additive", "rate", "volume_rate_mix")
    }
    additive_rows, additive_truth = _additive_case(
        random.Random(derived_seeds["additive"])
    )
    rate_rows, rate_truth = _rate_case(random.Random(derived_seeds["rate"]))
    vrm_rows, vrm_truth = _volume_rate_mix_case(
        random.Random(derived_seeds["volume_rate_mix"])
    )

    source_files = _DATASET_SOURCE_FILES[_DECOMPOSITION_DATASET_ID]
    additive_path = root / source_files["additive"]
    rate_path = root / source_files["rate"]
    vrm_path = root / source_files["volume_rate_mix"]
    _write_csv(
        additive_path,
        ("case_id", "period", "segment", "value"),
        additive_rows,
    )
    _write_csv(
        rate_path,
        ("case_id", "period", "numerator", "denominator"),
        rate_rows,
    )
    _write_csv(
        vrm_path,
        ("case_id", "period", "segment", "volume", "rate"),
        vrm_rows,
    )

    sources = [
        _file_record(root, "additive", additive_path, len(additive_rows)),
        _file_record(root, "rate", rate_path, len(rate_rows)),
        _file_record(root, "volume_rate_mix", vrm_path, len(vrm_rows)),
    ]
    dataset_fingerprint = _dataset_fingerprint(
        dataset_id=_DECOMPOSITION_DATASET_ID,
        root_seed=root_seed,
        derived_seeds=derived_seeds,
        sources=sources,
    )
    manifest_path = root / "manifest.json"
    truth_path = root / "truth.json"
    _write_json(
        truth_path,
        {
            "dataset_id": _DECOMPOSITION_DATASET_ID,
            "dataset_fingerprint": dataset_fingerprint,
            "generator_version": _GENERATOR_VERSION,
            "root_seed": root_seed,
            "cases": [additive_truth, rate_truth, vrm_truth],
        },
    )
    truth_bytes = truth_path.read_bytes()
    _write_json(
        manifest_path,
        {
            "dataset_id": _DECOMPOSITION_DATASET_ID,
            "dataset_fingerprint": dataset_fingerprint,
            "generator_version": _GENERATOR_VERSION,
            "random_engine": _RANDOM_ENGINE,
            "root_seed": root_seed,
            "derived_seeds": derived_seeds,
            "sources": sources,
            "truth_integrity": {
                "sha256": hashlib.sha256(truth_bytes).hexdigest(),
                "size_bytes": len(truth_bytes),
            },
        },
    )
    return SyntheticBenchmark(
        dataset_id=_DECOMPOSITION_DATASET_ID,
        root_seed=root_seed,
        manifest_path=manifest_path,
        truth_path=truth_path,
        worker_source_paths=MappingProxyType(
            {
                "additive": additive_path,
                "rate": rate_path,
                "volume_rate_mix": vrm_path,
            }
        ),
        dataset_fingerprint=dataset_fingerprint,
    )


def write_time_series_benchmark(
    output_dir: str | Path,
    *,
    root_seed: int,
) -> SyntheticBenchmark:
    """Write deterministic seasonal, shift, anomaly, and missingness series."""

    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"output_dir must be empty: {root}")

    derived_seeds = {
        name: derive_seed(
            root_seed,
            dataset_id=_TIME_SERIES_DATASET_ID,
            operator_id=f"generate.time_series.{name}.v1",
        )
        for name in ("noise", "structure")
    }
    structure_rng = random.Random(derived_seeds["structure"])
    noise_rng = random.Random(derived_seeds["noise"])
    seasonal_pattern = (0, 5, 9, 12, 15, 10, 4, -2, -8, -12, -9, -4)
    periods = 120
    shift_index = 72
    missing_indices = tuple(sorted(structure_rng.sample(range(12, 108), 4)))
    anomaly_candidates = [
        index for index in range(12, 108) if index not in missing_indices
    ]
    anomaly_indices = tuple(sorted(structure_rng.sample(anomaly_candidates, 5)))
    anomaly_values = [60, -55, 70, -65, 80]
    structure_rng.shuffle(anomaly_values)
    anomaly_magnitudes = {
        index: magnitude
        for index, magnitude in zip(
            anomaly_indices,
            anomaly_values,
        )
    }

    rows: list[tuple[object, ...]] = []
    for index in range(periods):
        seasonal_value = (
            100
            + 2 * index
            + seasonal_pattern[index % len(seasonal_pattern)]
            + (40 if index >= shift_index else 0)
            + noise_rng.randint(-2, 2)
        )
        rows.append(("seasonal_shift", index, seasonal_value))

        if index not in missing_indices:
            anomaly_value = (
                200
                + seasonal_pattern[index % len(seasonal_pattern)]
                + noise_rng.randint(-3, 3)
                + anomaly_magnitudes.get(index, 0)
            )
            rows.append(("anomaly_missing", index, anomaly_value))

    source_path = root / _DATASET_SOURCE_FILES[_TIME_SERIES_DATASET_ID][
        "time_series"
    ]
    _write_csv(
        source_path,
        ("series_id", "time_index", "value"),
        rows,
    )
    sources = [
        _file_record(root, "time_series", source_path, len(rows)),
    ]
    dataset_fingerprint = _dataset_fingerprint(
        dataset_id=_TIME_SERIES_DATASET_ID,
        root_seed=root_seed,
        derived_seeds=derived_seeds,
        sources=sources,
    )
    truth_path = root / "truth.json"
    _write_json(
        truth_path,
        {
            "dataset_id": _TIME_SERIES_DATASET_ID,
            "dataset_fingerprint": dataset_fingerprint,
            "generator_version": _GENERATOR_VERSION,
            "root_seed": root_seed,
            "series": [
                {
                    "series_id": "seasonal_shift",
                    "base_level": 100,
                    "total_period_count": periods,
                    "observed_period_count": periods,
                    "seasonal_pattern": list(seasonal_pattern),
                    "seasonal_period": len(seasonal_pattern),
                    "trend_per_period": 2,
                    "level_shift_index": shift_index,
                    "level_shift_magnitude": 40,
                    "noise_bounds": [-2, 2],
                },
                {
                    "series_id": "anomaly_missing",
                    "anomaly_free_ranges": [[0, 11], [108, 119]],
                    "base_level": 200,
                    "total_period_count": periods,
                    "observed_period_count": periods - len(missing_indices),
                    "seasonal_pattern": list(seasonal_pattern),
                    "seasonal_period": len(seasonal_pattern),
                    "trend_per_period": 0,
                    "anomaly_indices": list(anomaly_indices),
                    "anomaly_magnitudes": {
                        str(index): anomaly_magnitudes[index]
                        for index in anomaly_indices
                    },
                    "missing_indices": list(missing_indices),
                    "noise_bounds": [-3, 3],
                },
            ],
        },
    )
    truth_bytes = truth_path.read_bytes()
    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "dataset_id": _TIME_SERIES_DATASET_ID,
            "dataset_fingerprint": dataset_fingerprint,
            "generator_version": _GENERATOR_VERSION,
            "random_engine": _RANDOM_ENGINE,
            "root_seed": root_seed,
            "derived_seeds": derived_seeds,
            "sources": sources,
            "truth_integrity": {
                "sha256": hashlib.sha256(truth_bytes).hexdigest(),
                "size_bytes": len(truth_bytes),
            },
        },
    )
    return SyntheticBenchmark(
        dataset_id=_TIME_SERIES_DATASET_ID,
        root_seed=root_seed,
        manifest_path=manifest_path,
        truth_path=truth_path,
        worker_source_paths=MappingProxyType({"time_series": source_path}),
        dataset_fingerprint=dataset_fingerprint,
    )


def write_correlated_benchmark(
    output_dir: str | Path,
    *,
    root_seed: int,
) -> SyntheticBenchmark:
    """Write grouped tabular data with known drivers and correlated proxies."""

    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"output_dir must be empty: {root}")

    derived_seeds = {
        name: derive_seed(
            root_seed,
            dataset_id=_CORRELATED_DATASET_ID,
            operator_id=f"generate.correlated.{name}.v1",
        )
        for name in (
            "measurement_noise",
            "missingness",
            "outcome_noise",
            "structure",
        )
    }
    structure_rng = random.Random(derived_seeds["structure"])
    measurement_rng = random.Random(derived_seeds["measurement_noise"])
    outcome_rng = random.Random(derived_seeds["outcome_noise"])
    missingness_rng = random.Random(derived_seeds["missingness"])

    generated_rows: list[dict[str, object]] = []
    row_truth: dict[str, dict[str, float]] = {}
    for group_index in range(1, _CORRELATED_GROUP_COUNT + 1):
        group_id = f"group-{group_index:03d}"
        group_effect = structure_rng.gauss(0.0, 1.5)
        for observation_index in range(1, _CORRELATED_OBSERVATIONS_PER_GROUP + 1):
            row_id = f"{group_id}-row-{observation_index}"
            linear_signal = structure_rng.gauss(0.0, 1.0)
            nonlinear_signal = structure_rng.gauss(0.0, 1.0)
            interaction_left = structure_rng.gauss(0.0, 1.0)
            interaction_right = structure_rng.gauss(0.0, 1.0)
            confounder = structure_rng.gauss(0.0, 1.0)
            missing_signal = structure_rng.gauss(0.0, 1.0)
            components = {
                "confounder": 2.0 * confounder,
                "group_effect": group_effect,
                "interaction_left_x_right": (
                    4.0 * interaction_left * interaction_right
                ),
                "linear_signal": 3.0 * linear_signal,
                "missing_signal": 1.5 * missing_signal,
                "nonlinear_signal_squared": -2.0 * nonlinear_signal**2,
                "outcome_noise": outcome_rng.gauss(0.0, 1.0),
            }
            target = 10.0 + math.fsum(components.values())
            generated_rows.append(
                {
                    "row_id": row_id,
                    "group_id": group_id,
                    "observation_index": observation_index,
                    "linear_signal": linear_signal,
                    "correlated_linear_proxy": (
                        linear_signal + measurement_rng.gauss(0.0, 0.05)
                    ),
                    "nonlinear_signal": nonlinear_signal,
                    "interaction_left": interaction_left,
                    "interaction_right": interaction_right,
                    "confounder": confounder,
                    "confounder_proxy": (
                        confounder + measurement_rng.gauss(0.0, 0.1)
                    ),
                    "nuisance": structure_rng.gauss(0.0, 1.0),
                    "missing_signal": missing_signal,
                    "missingness_score": (
                        confounder + missingness_rng.gauss(0.0, 0.5)
                    ),
                    "target": target,
                }
            )
            row_truth[row_id] = components

    missing_count = int(len(generated_rows) * 0.15)
    missing_row_ids = {
        str(row["row_id"])
        for row in sorted(
            generated_rows,
            key=lambda row: (-float(row["missingness_score"]), str(row["row_id"])),
        )[:missing_count]
    }
    rows = [
        (
            row["row_id"],
            row["group_id"],
            row["observation_index"],
            row["linear_signal"],
            row["correlated_linear_proxy"],
            row["nonlinear_signal"],
            row["interaction_left"],
            row["interaction_right"],
            row["confounder"],
            row["confounder_proxy"],
            row["nuisance"],
            None if row["row_id"] in missing_row_ids else row["missing_signal"],
            row["target"],
        )
        for row in generated_rows
    ]

    source_path = root / _DATASET_SOURCE_FILES[_CORRELATED_DATASET_ID][
        "observations"
    ]
    _write_csv(
        source_path,
        (
            "row_id",
            "group_id",
            "observation_index",
            "linear_signal",
            "correlated_linear_proxy",
            "nonlinear_signal",
            "interaction_left",
            "interaction_right",
            "confounder",
            "confounder_proxy",
            "nuisance",
            "missing_signal",
            "target",
        ),
        rows,
    )
    sources = [
        _file_record(root, "observations", source_path, len(rows)),
    ]
    dataset_fingerprint = _dataset_fingerprint(
        dataset_id=_CORRELATED_DATASET_ID,
        root_seed=root_seed,
        derived_seeds=derived_seeds,
        sources=sources,
    )
    truth_path = root / "truth.json"
    _write_json(
        truth_path,
        {
            "dataset_id": _CORRELATED_DATASET_ID,
            "dataset_fingerprint": dataset_fingerprint,
            "generator_version": _GENERATOR_VERSION,
            "root_seed": root_seed,
            "structure": {
                "group_count": _CORRELATED_GROUP_COUNT,
                "observations_per_group": _CORRELATED_OBSERVATIONS_PER_GROUP,
                "row_count": len(rows),
            },
            "validation": {
                "group_key": "group_id",
                "identifier_columns": ["row_id", "group_id"],
                "model_selection_strategy": "nested_grouped_cross_validation",
                "preprocessing_scope": "training_fold_only",
                "required_strategy": "grouped",
                "untouched_final_holdout_required": True,
            },
            "data_generating_process": {
                "intercept": 10.0,
                "direct_terms": {
                    "confounder": 2.0,
                    "interaction_left_x_right": 4.0,
                    "linear_signal": 3.0,
                    "missing_signal": 1.5,
                    "nonlinear_signal_squared": -2.0,
                },
                "group_effect_standard_deviation": 1.5,
                "outcome_noise_standard_deviation": 1.0,
                "non_driver_features": [
                    "correlated_linear_proxy",
                    "confounder_proxy",
                    "nuisance",
                ],
            },
            "multicollinearity": {
                "pairs": [
                    {
                        "feature": "linear_signal",
                        "proxy": "correlated_linear_proxy",
                        "proxy_noise_standard_deviation": 0.05,
                    },
                    {
                        "feature": "confounder",
                        "proxy": "confounder_proxy",
                        "proxy_noise_standard_deviation": 0.1,
                    },
                ],
                "interpretation": "proxies are associated but have no direct term",
            },
            "missingness": {
                "mechanism": "MAR",
                "missing_count": missing_count,
                "missing_rate": missing_count / len(rows),
                "missing_row_ids": sorted(missing_row_ids),
                "selection_feature": "confounder",
                "selection_rule": "highest confounder plus independent noise scores",
            },
            "row_truth": row_truth,
        },
    )
    truth_bytes = truth_path.read_bytes()
    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "dataset_id": _CORRELATED_DATASET_ID,
            "dataset_fingerprint": dataset_fingerprint,
            "generator_version": _GENERATOR_VERSION,
            "random_engine": _RANDOM_ENGINE,
            "root_seed": root_seed,
            "derived_seeds": derived_seeds,
            "sources": sources,
            "truth_integrity": {
                "sha256": hashlib.sha256(truth_bytes).hexdigest(),
                "size_bytes": len(truth_bytes),
            },
        },
    )
    return SyntheticBenchmark(
        dataset_id=_CORRELATED_DATASET_ID,
        root_seed=root_seed,
        manifest_path=manifest_path,
        truth_path=truth_path,
        worker_source_paths=MappingProxyType({"observations": source_path}),
        dataset_fingerprint=dataset_fingerprint,
    )


def write_panel_benchmark(
    output_dir: str | Path,
    *,
    root_seed: int,
) -> SyntheticBenchmark:
    """Write a customer-event panel with cohort, funnel, and leakage truth."""

    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"output_dir must be empty: {root}")

    derived_seeds = {
        name: derive_seed(
            root_seed,
            dataset_id=_PANEL_DATASET_ID,
            operator_id=f"generate.panel.{name}.v1",
        )
        for name in ("attributes", "structure")
    }
    attribute_rng = random.Random(derived_seeds["attributes"])
    structure_rng = random.Random(derived_seeds["structure"])
    # cohort, signup_day, activated, converted, retained among converted
    cohort_specs = (
        ("2026-01", 0, 48, 30, 24),
        ("2026-02", 30, 45, 28, 20),
        ("2026-03", 60, 42, 20, 12),
        ("2026-04", 120, 40, 20, 12),
    )
    observation_cutoff = 180
    retention_exposure = 90
    channels = ("direct", "partner", "paid", "organic")

    customer_rows: list[tuple[object, ...]] = []
    event_rows: list[tuple[object, ...]] = []
    truth_cohorts: list[dict[str, object]] = []
    eligible_duplicate_rows: list[tuple[object, ...]] = []

    for cohort, signup_day, activated_count, converted_count, retained_count in cohort_specs:
        customer_ids = [
            f"{cohort}-{index:03d}"
            for index in range(1, _PANEL_COHORT_SIZE + 1)
        ]
        shuffled = list(customer_ids)
        structure_rng.shuffle(shuffled)
        activated = set(shuffled[:activated_count])
        converted_candidates = list(shuffled[:activated_count])
        structure_rng.shuffle(converted_candidates)
        converted = set(converted_candidates[:converted_count])
        retained_candidates = sorted(converted)
        structure_rng.shuffle(retained_candidates)
        retained = set(retained_candidates[:retained_count])
        eligible_for_retention = (
            observation_cutoff - signup_day >= retention_exposure
        )

        for customer_id in customer_ids:
            customer_rows.append(
                (
                    customer_id,
                    cohort,
                    signup_day,
                    channels[attribute_rng.randrange(len(channels))],
                    attribute_rng.randint(0, 12),
                    int(customer_id in converted),
                    int(customer_id in retained),
                )
            )
            signup_event = (
                f"evt-{customer_id}-signup",
                customer_id,
                cohort,
                "signup",
                signup_day,
            )
            event_rows.append(signup_event)
            if customer_id in activated:
                activated_event = (
                    f"evt-{customer_id}-activated",
                    customer_id,
                    cohort,
                    "activated",
                    signup_day + 7,
                )
                event_rows.append(activated_event)
                eligible_duplicate_rows.append(activated_event)
            if customer_id in converted:
                converted_event = (
                    f"evt-{customer_id}-converted",
                    customer_id,
                    cohort,
                    "converted",
                    signup_day + 30,
                )
                event_rows.append(converted_event)
                eligible_duplicate_rows.append(converted_event)
            if customer_id in retained and eligible_for_retention:
                retained_event = (
                    f"evt-{customer_id}-retained-day-90",
                    customer_id,
                    cohort,
                    "retained_day_90",
                    signup_day + retention_exposure,
                )
                event_rows.append(retained_event)

        truth_cohorts.append(
            {
                "cohort": cohort,
                "signup_day": signup_day,
                "signup": len(customer_ids),
                "activated": activated_count,
                "converted": converted_count,
                "eligible_for_day_90_retention": eligible_for_retention,
                "day_90_retained": (
                    retained_count if eligible_for_retention else None
                ),
                "day_90_retention_denominator": (
                    converted_count if eligible_for_retention else None
                ),
                "day_90_retention_rate": (
                    retained_count / converted_count
                    if eligible_for_retention
                    else None
                ),
            }
        )

    duplicate_rows = structure_rng.sample(eligible_duplicate_rows, 8)
    event_rows.extend(duplicate_rows)
    customer_rows.sort(key=lambda row: str(row[0]))
    event_rows.sort(key=lambda row: (str(row[0]), int(row[4])))

    source_files = _DATASET_SOURCE_FILES[_PANEL_DATASET_ID]
    customers_path = root / source_files["customers"]
    events_path = root / source_files["events"]
    _write_csv(
        customers_path,
        (
            "customer_id",
            "cohort",
            "signup_day",
            "acquisition_channel",
            "early_sessions",
            "future_converted_label",
            "future_retained_day_90_label",
        ),
        customer_rows,
    )
    _write_csv(
        events_path,
        ("event_id", "customer_id", "cohort", "event_type", "event_day"),
        event_rows,
    )
    sources = [
        _file_record(root, "customers", customers_path, len(customer_rows)),
        _file_record(root, "events", events_path, len(event_rows)),
    ]
    dataset_fingerprint = _dataset_fingerprint(
        dataset_id=_PANEL_DATASET_ID,
        root_seed=root_seed,
        derived_seeds=derived_seeds,
        sources=sources,
    )
    truth_path = root / "truth.json"
    _write_json(
        truth_path,
        {
            "dataset_id": _PANEL_DATASET_ID,
            "dataset_fingerprint": dataset_fingerprint,
            "generator_version": _GENERATOR_VERSION,
            "root_seed": root_seed,
            "cohorts": truth_cohorts,
            "funnel": {
                "signup": sum(_PANEL_COHORT_SIZE for _ in cohort_specs),
                "activated": sum(spec[2] for spec in cohort_specs),
                "converted": sum(spec[3] for spec in cohort_specs),
                "eligible_day_90_retained": sum(
                    spec[4]
                    for spec in cohort_specs
                    if observation_cutoff - spec[1] >= retention_exposure
                ),
                "counting_rule": "distinct customer_id",
            },
            "censoring": {
                "observation_cutoff_day": observation_cutoff,
                "minimum_retention_exposure_days": retention_exposure,
                "censored_cohorts": [
                    spec[0]
                    for spec in cohort_specs
                    if observation_cutoff - spec[1] < retention_exposure
                ],
            },
            "data_quality": {
                "duplicate_event_ids": sorted(
                    str(row[0]) for row in duplicate_rows
                ),
                "duplicate_event_rows": len(duplicate_rows),
            },
            "leakage": {
                "prohibited_feature_columns": [
                    "future_converted_label",
                    "future_retained_day_90_label",
                ],
                "reason": "post-outcome information",
            },
        },
    )
    truth_bytes = truth_path.read_bytes()
    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "dataset_id": _PANEL_DATASET_ID,
            "dataset_fingerprint": dataset_fingerprint,
            "generator_version": _GENERATOR_VERSION,
            "random_engine": _RANDOM_ENGINE,
            "root_seed": root_seed,
            "derived_seeds": derived_seeds,
            "sources": sources,
            "truth_integrity": {
                "sha256": hashlib.sha256(truth_bytes).hexdigest(),
                "size_bytes": len(truth_bytes),
            },
        },
    )
    return SyntheticBenchmark(
        dataset_id=_PANEL_DATASET_ID,
        root_seed=root_seed,
        manifest_path=manifest_path,
        truth_path=truth_path,
        worker_source_paths=MappingProxyType(
            {"customers": customers_path, "events": events_path}
        ),
        dataset_fingerprint=dataset_fingerprint,
    )


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV has no header: {path}") from exc
        return sum(1 for _ in reader)


def load_synthetic_benchmark(manifest_path: str | Path) -> SyntheticBenchmark:
    """Load and verify every worker-visible file in a synthetic bundle."""

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must contain an object")
    dataset_id = payload.get("dataset_id")
    source_files = _DATASET_SOURCE_FILES.get(dataset_id)
    if source_files is None:
        raise ValueError(f"unsupported dataset_id: {dataset_id}")
    if payload.get("generator_version") != _GENERATOR_VERSION:
        raise ValueError(
            "unsupported generator_version: "
            f"{payload.get('generator_version')!r}"
        )
    if payload.get("random_engine") != _RANDOM_ENGINE:
        raise ValueError(
            f"unsupported random_engine: {payload.get('random_engine')!r}"
        )
    root_seed = payload.get("root_seed")
    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    derived_seeds = payload.get("derived_seeds")
    if not isinstance(derived_seeds, dict):
        raise ValueError("derived_seeds must be an object")
    if set(derived_seeds) != _DATASET_SEED_NAMES[dataset_id]:
        raise ValueError("derived_seeds do not match the dataset generator")
    if any(type(seed) is not int or seed < 0 for seed in derived_seeds.values()):
        raise ValueError("derived_seeds must contain non-negative integers")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")

    root = path.parent
    verified: dict[str, Path] = {}
    normalized_sources: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"sources[{index}] must be an object")
        identity = source.get("identity")
        relative = source.get("path")
        if identity not in source_files or relative != source_files[identity]:
            raise ValueError(f"sources[{index}] has an invalid identity or path")
        if identity in verified:
            raise ValueError(f"duplicate source identity: {identity}")
        source_path = (root / str(relative)).resolve()
        try:
            source_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"source {identity} escapes the bundle") from exc
        if not source_path.is_file():
            raise FileNotFoundError(f"source {identity} does not exist")
        data = source_path.read_bytes()
        if len(data) != source.get("size_bytes"):
            raise ValueError(f"size mismatch for {identity}")
        digest = hashlib.sha256(data).hexdigest()
        if digest != source.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {identity}")
        if _csv_row_count(source_path) != source.get("row_count"):
            raise ValueError(f"row-count mismatch for {identity}")
        verified[identity] = source_path
        normalized_sources.append(source)

    missing_sources = sorted(set(source_files) - set(verified))
    if missing_sources:
        raise ValueError(f"missing required source: {missing_sources[0]}")
    expected_fingerprint = _dataset_fingerprint(
        dataset_id=dataset_id,
        root_seed=root_seed,
        derived_seeds=derived_seeds,
        sources=normalized_sources,
    )
    if payload.get("dataset_fingerprint") != expected_fingerprint:
        raise ValueError("dataset fingerprint mismatch")
    truth_path = root / "truth.json"
    if not truth_path.is_file():
        raise FileNotFoundError(f"truth file does not exist: {truth_path}")
    truth_integrity = payload.get("truth_integrity")
    if not isinstance(truth_integrity, dict):
        raise ValueError("manifest truth_integrity must be an object")
    truth_bytes = truth_path.read_bytes()
    if len(truth_bytes) != truth_integrity.get("size_bytes"):
        raise ValueError("truth size mismatch")
    if hashlib.sha256(truth_bytes).hexdigest() != truth_integrity.get("sha256"):
        raise ValueError("truth SHA-256 mismatch")
    truth_payload = json.loads(truth_bytes)
    if not isinstance(truth_payload, dict):
        raise ValueError("truth file must contain an object")
    if truth_payload.get("dataset_fingerprint") != expected_fingerprint:
        raise ValueError("truth dataset fingerprint does not match manifest")
    if truth_payload.get("root_seed") != root_seed:
        raise ValueError("truth root_seed does not match manifest")
    if truth_payload.get("generator_version") != _GENERATOR_VERSION:
        raise ValueError("truth generator_version does not match manifest")
    return SyntheticBenchmark(
        dataset_id=dataset_id,
        root_seed=root_seed,
        manifest_path=path,
        truth_path=truth_path,
        worker_source_paths=MappingProxyType(dict(sorted(verified.items()))),
        dataset_fingerprint=expected_fingerprint,
    )
