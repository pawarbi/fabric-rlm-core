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


_DATASET_ID = "decomposition-ground-truth-v1"
_GENERATOR_VERSION = "1"
_RANDOM_ENGINE = "python.random.Random"
_SOURCE_FILES = {
    "additive": "additive.csv",
    "rate": "rate.csv",
    "volume_rate_mix": "volume_rate_mix.csv",
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
    root_seed: int,
    sources: list[dict[str, object]],
) -> str:
    return fingerprint(
        {
            "dataset_id": _DATASET_ID,
            "generator_version": _GENERATOR_VERSION,
            "random_engine": _RANDOM_ENGINE,
            "root_seed": root_seed,
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

    additive_rows, additive_truth = _additive_case(
        random.Random(
            derive_seed(
                root_seed,
                dataset_id=_DATASET_ID,
                operator_id="generate.additive.v1",
            )
        )
    )
    rate_rows, rate_truth = _rate_case(
        random.Random(
            derive_seed(
                root_seed,
                dataset_id=_DATASET_ID,
                operator_id="generate.rate.v1",
            )
        )
    )
    vrm_rows, vrm_truth = _volume_rate_mix_case(
        random.Random(
            derive_seed(
                root_seed,
                dataset_id=_DATASET_ID,
                operator_id="generate.volume_rate_mix.v1",
            )
        )
    )

    additive_path = root / _SOURCE_FILES["additive"]
    rate_path = root / _SOURCE_FILES["rate"]
    vrm_path = root / _SOURCE_FILES["volume_rate_mix"]
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
        root_seed=root_seed,
        sources=sources,
    )
    manifest_path = root / "manifest.json"
    truth_path = root / "truth.json"
    _write_json(
        truth_path,
        {
            "dataset_id": _DATASET_ID,
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
            "dataset_id": _DATASET_ID,
            "dataset_fingerprint": dataset_fingerprint,
            "generator_version": _GENERATOR_VERSION,
            "random_engine": _RANDOM_ENGINE,
            "root_seed": root_seed,
            "sources": sources,
            "truth_integrity": {
                "sha256": hashlib.sha256(truth_bytes).hexdigest(),
                "size_bytes": len(truth_bytes),
            },
        },
    )
    return SyntheticBenchmark(
        dataset_id=_DATASET_ID,
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
    if payload.get("dataset_id") != _DATASET_ID:
        raise ValueError(f"unsupported dataset_id: {payload.get('dataset_id')}")
    root_seed = payload.get("root_seed")
    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
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
        if identity not in _SOURCE_FILES or relative != _SOURCE_FILES[identity]:
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

    missing_sources = sorted(set(_SOURCE_FILES) - set(verified))
    if missing_sources:
        raise ValueError(f"missing required source: {missing_sources[0]}")
    expected_fingerprint = _dataset_fingerprint(
        root_seed=root_seed,
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
        dataset_id=_DATASET_ID,
        root_seed=root_seed,
        manifest_path=path,
        truth_path=truth_path,
        worker_source_paths=MappingProxyType(dict(sorted(verified.items()))),
        dataset_fingerprint=expected_fingerprint,
    )
