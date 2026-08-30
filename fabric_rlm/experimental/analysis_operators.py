"""Deterministic exact operators for the experimental analysis DAG."""

from __future__ import annotations

from collections.abc import Mapping
import math

from fabric_rlm.experimental.analysis_contracts import OperatorResult
from fabric_rlm.experimental.analysis_reproducibility import fingerprint


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    return number + 0.0


def _positive_number(value: object, field_name: str) -> float:
    number = _finite_number(value, field_name)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _validated_segments(
    values: Mapping[str, object],
    field_name: str,
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    validated: dict[str, float] = {}
    for segment, value in values.items():
        if not isinstance(segment, str) or not segment.strip():
            raise ValueError(f"{field_name} segment names must be non-empty strings")
        identity = segment.strip()
        if identity in validated:
            raise ValueError(f"{field_name} contains duplicate segment {identity}")
        validated[identity] = _finite_number(value, f"{field_name}.{identity}")
    return validated


def _validated_volume_rate_period(
    values: Mapping[str, object],
    field_name: str,
) -> tuple[dict[str, tuple[float, float]], float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{field_name} must contain at least one segment")
    validated: dict[str, tuple[float, float]] = {}
    for segment, payload in values.items():
        if not isinstance(segment, str) or not segment.strip():
            raise ValueError(f"{field_name} segment names must be non-empty strings")
        identity = segment.strip()
        if identity in validated:
            raise ValueError(f"{field_name} contains duplicate segment {identity}")
        if not isinstance(payload, Mapping):
            raise ValueError(f"{field_name}.{identity} must be an object")
        missing = {"volume", "rate"} - payload.keys()
        if missing:
            raise ValueError(
                f"{field_name}.{identity} is missing {sorted(missing)[0]}"
            )
        unexpected = set(payload) - {"volume", "rate"}
        if unexpected:
            raise ValueError(
                f"{field_name}.{identity} contains unsupported field "
                f"{sorted(unexpected)[0]}"
            )
        volume = _finite_number(
            payload["volume"],
            f"{field_name}.{identity}.volume",
        )
        if volume < 0:
            raise ValueError(
                f"{field_name}.{identity}.volume must be non-negative"
            )
        rate = _finite_number(payload["rate"], f"{field_name}.{identity}.rate")
        validated[identity] = (volume, rate)
    total_volume = math.fsum(volume for volume, _ in validated.values())
    if total_volume <= 0:
        raise ValueError(f"{field_name} total volume must be positive")
    return validated, total_volume


def _reconciliation(
    observed_change: float,
    effects: tuple[float, ...],
    tolerance: float,
) -> dict[str, object]:
    residual = observed_change - math.fsum(effects)
    if residual == 0:
        # Keep deterministic JSON and fingerprints for IEEE signed zero.
        residual = 0.0
    return {
        "passed": abs(residual) <= tolerance,
        "residual": residual,
        "tolerance": tolerance,
    }


def additive_decomposition(
    *,
    node_id: str,
    before: Mapping[str, object],
    after: Mapping[str, object],
    seed: int,
    tolerance: float = 1e-12,
) -> OperatorResult:
    """Exactly attribute an additive KPI change to segment-level changes.

    ``seed`` is provenance metadata for the DAG and does not alter exact math.
    """

    before_values = _validated_segments(before, "before")
    after_values = _validated_segments(after, "after")
    segments = tuple(sorted(before_values.keys() | after_values.keys()))
    if not segments:
        raise ValueError("before and after must contain at least one segment")
    numeric_tolerance = _positive_number(tolerance, "tolerance")

    components = tuple(
        {
            "segment": segment,
            "before": before_values.get(segment, 0.0),
            "after": after_values.get(segment, 0.0),
            "contribution": (
                after_values.get(segment, 0.0)
                - before_values.get(segment, 0.0)
            ),
        }
        for segment in segments
    )
    before_total = math.fsum(before_values.values())
    after_total = math.fsum(after_values.values())
    observed_change = after_total - before_total
    contributions = tuple(
        component["contribution"] for component in components
    )
    input_values = {
        "before": before_values,
        "after": after_values,
        "tolerance": numeric_tolerance,
    }

    return OperatorResult(
        node_id=node_id,
        operator="kpi.additive.v1",
        status="completed",
        seed=seed,
        sample_size=len(segments),
        values={
            "before_total": before_total,
            "after_total": after_total,
            "observed_change": observed_change,
            "components": components,
        },
        diagnostics={
            "input_fingerprint": fingerprint(input_values),
            "method": "exact_additive_v1",
            "reconciliation": _reconciliation(
                observed_change,
                contributions,
                numeric_tolerance,
            ),
        },
    )


def rate_decomposition(
    *,
    node_id: str,
    before_numerator: object,
    before_denominator: object,
    after_numerator: object,
    after_denominator: object,
    seed: int,
    tolerance: float = 1e-12,
) -> OperatorResult:
    """Exactly decompose a rate change with symmetric two-factor attribution.

    ``seed`` is provenance metadata for the DAG and does not alter exact math.
    """

    before_num = _finite_number(before_numerator, "before_numerator")
    before_den = _positive_number(before_denominator, "before_denominator")
    after_num = _finite_number(after_numerator, "after_numerator")
    after_den = _positive_number(after_denominator, "after_denominator")
    numeric_tolerance = _positive_number(tolerance, "tolerance")

    before_rate = before_num / before_den
    after_rate = after_num / after_den
    observed_change = after_rate - before_rate
    numerator_effect = 0.5 * (
        (after_num / before_den - before_num / before_den)
        + (after_num / after_den - before_num / after_den)
    )
    denominator_effect = 0.5 * (
        (before_num / after_den - before_num / before_den)
        + (after_num / after_den - after_num / before_den)
    )
    input_values = {
        "before_numerator": before_num,
        "before_denominator": before_den,
        "after_numerator": after_num,
        "after_denominator": after_den,
        "tolerance": numeric_tolerance,
    }

    return OperatorResult(
        node_id=node_id,
        operator="kpi.rate.v1",
        status="completed",
        seed=seed,
        sample_size=2,
        values={
            "before_rate": before_rate,
            "after_rate": after_rate,
            "observed_change": observed_change,
            "numerator_effect": numerator_effect,
            "denominator_effect": denominator_effect,
        },
        diagnostics={
            "input_fingerprint": fingerprint(input_values),
            "method": "symmetric_two_factor",
            "reconciliation": _reconciliation(
                observed_change,
                (numerator_effect, denominator_effect),
                numeric_tolerance,
            ),
        },
    )


def volume_rate_mix_decomposition(
    *,
    node_id: str,
    before: Mapping[str, object],
    after: Mapping[str, object],
    seed: int,
    tolerance: float = 1e-12,
) -> OperatorResult:
    """Exactly decompose ``sum(volume * rate)`` into volume, rate, and mix.

    Segment rate changes and share changes use symmetric two-factor
    attribution. A segment absent from one period receives its observed rate
    from the other period, so its entry or exit is classified as mix rather
    than an invented rate change. A present segment with zero volume retains
    its supplied rate and is not treated as absent. ``seed`` is provenance
    metadata only.
    """

    before_values, before_volume = _validated_volume_rate_period(
        before,
        "before",
    )
    after_values, after_volume = _validated_volume_rate_period(after, "after")
    numeric_tolerance = _positive_number(tolerance, "tolerance")
    segments = tuple(sorted(before_values.keys() | after_values.keys()))

    aligned: dict[str, tuple[float, float, float, float]] = {}
    boundary_segments: list[dict[str, str]] = []
    for segment in segments:
        before_pair = before_values.get(segment)
        after_pair = after_values.get(segment)
        if before_pair is None:
            if after_pair is None:  # pragma: no cover - guarded by key union
                raise RuntimeError(
                    f"segment {segment!r} is missing from both periods"
                )
            before_pair = (0.0, after_pair[1])
            boundary_segments.append(
                {
                    "segment": segment,
                    "missing_period": "before",
                    "rate_convention": "carry_observed_rate",
                }
            )
        if after_pair is None:
            after_pair = (0.0, before_pair[1])
            boundary_segments.append(
                {
                    "segment": segment,
                    "missing_period": "after",
                    "rate_convention": "carry_observed_rate",
                }
            )
        aligned[segment] = (
            before_pair[0],
            before_pair[1],
            after_pair[0],
            after_pair[1],
        )

    before_total = math.fsum(
        volume_before * rate_before
        for volume_before, rate_before, _, _ in aligned.values()
    )
    after_total = math.fsum(
        volume_after * rate_after
        for _, _, volume_after, rate_after in aligned.values()
    )
    before_average_rate = before_total / before_volume
    after_average_rate = after_total / after_volume
    average_volume = 0.5 * (before_volume + after_volume)

    volume_effect = (after_volume - before_volume) * 0.5 * (
        before_average_rate + after_average_rate
    )
    components = tuple(
        {
            "segment": segment,
            "rate_effect": average_volume
            * 0.5
            * (
                volume_before / before_volume
                + volume_after / after_volume
            )
            * (rate_after - rate_before),
            "mix_effect": average_volume
            * 0.5
            * (
                volume_after / after_volume
                - volume_before / before_volume
            )
            * (rate_before + rate_after),
        }
        for segment, (
            volume_before,
            rate_before,
            volume_after,
            rate_after,
        ) in aligned.items()
    )
    rate_effect = math.fsum(
        component["rate_effect"] for component in components
    )
    mix_effect = math.fsum(component["mix_effect"] for component in components)
    observed_change = after_total - before_total
    input_values = {
        "before": before_values,
        "after": after_values,
        "tolerance": numeric_tolerance,
    }

    return OperatorResult(
        node_id=node_id,
        operator="kpi.volume_rate_mix.v1",
        status="completed",
        seed=seed,
        sample_size=len(segments),
        values={
            "before_total": before_total,
            "after_total": after_total,
            "observed_change": observed_change,
            "before_total_volume": before_volume,
            "after_total_volume": after_volume,
            "before_average_rate": before_average_rate,
            "after_average_rate": after_average_rate,
            "volume_effect": volume_effect,
            "rate_effect": rate_effect,
            "mix_effect": mix_effect,
            "components": components,
        },
        diagnostics={
            "boundary_segments": tuple(boundary_segments),
            "input_fingerprint": fingerprint(input_values),
            "method": "symmetric_volume_rate_mix_v1",
            "reconciliation": _reconciliation(
                observed_change,
                (volume_effect, rate_effect, mix_effect),
                numeric_tolerance,
            ),
        },
    )
