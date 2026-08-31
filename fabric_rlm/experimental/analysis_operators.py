"""Deterministic exact operators for the experimental analysis DAG."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone as datetime_timezone
import math
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fabric_rlm.experimental.analysis_contracts import OperatorResult
from fabric_rlm.experimental.analysis_reproducibility import fingerprint


_TIME_GRAINS = {"day", "week", "month", "quarter"}


def _parse_timestamp(
    value: object,
    field_name: str,
    *,
    source_timezone: ZoneInfo,
) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            raise ValueError(
                f"{field_name} must contain ISO dates or timestamps"
            ) from None
    else:
        raise ValueError(f"{field_name} must contain ISO dates or timestamps")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=source_timezone)
    return parsed.astimezone(datetime_timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _period_start(value: date, grain: str) -> date:
    if grain == "day":
        return value
    if grain == "week":
        return value - timedelta(days=value.weekday())
    if grain == "month":
        return value.replace(day=1)
    quarter_month = 3 * ((value.month - 1) // 3) + 1
    return value.replace(month=quarter_month, day=1)


def _next_period(value: date, grain: str) -> date:
    if grain == "day":
        return value + timedelta(days=1)
    if grain == "week":
        return value + timedelta(days=7)
    if grain == "month":
        year = value.year + (1 if value.month == 12 else 0)
        month = 1 if value.month == 12 else value.month + 1
        return date(year, month, 1)
    month = value.month + 3
    year = value.year + (month - 1) // 12
    return date(year, (month - 1) % 12 + 1, 1)


def _previous_period(value: date, grain: str) -> date:
    if grain == "day":
        return value - timedelta(days=1)
    if grain == "week":
        return value - timedelta(days=7)
    if grain == "month":
        year = value.year - (1 if value.month == 1 else 0)
        month = 12 if value.month == 1 else value.month - 1
        return date(year, month, 1)
    month = value.month - 3
    year = value.year
    if month <= 0:
        month += 12
        year -= 1
    return date(year, month, 1)


def _period_end(value: date, grain: str) -> date:
    return _next_period(value, grain) - timedelta(days=1)


def _period_label(value: date, grain: str) -> str:
    if grain == "day":
        return value.isoformat()
    if grain == "week":
        year, week, _ = value.isocalendar()
        return f"{year}-W{week:02d}"
    if grain == "month":
        return value.strftime("%Y-%m")
    return f"{value.year}-Q{((value.month - 1) // 3) + 1}"


def _period_payload(value: date, grain: str) -> dict[str, str]:
    return {
        "grain": grain,
        "start": value.isoformat(),
        "end": _period_end(value, grain).isoformat(),
    }


def profile_time_coverage(
    *,
    node_id: str,
    timestamps: object,
    seed: int,
    grain: str = "month",
    requested_as_of: object | None = None,
    source_watermark: object | None = None,
    trustworthy_through: object | None = None,
    timezone: str = "UTC",
) -> OperatorResult:
    """Profile source-time coverage without treating the event maximum as current."""

    if not isinstance(timezone, str) or not timezone.strip():
        raise ValueError("timezone must be a non-empty IANA timezone")
    try:
        source_timezone = ZoneInfo(timezone.strip())
    except ZoneInfoNotFoundError:
        raise ValueError("timezone must be a valid IANA timezone") from None
    if grain not in _TIME_GRAINS:
        raise ValueError("grain must be day, week, month, or quarter")
    if not isinstance(timestamps, (list, tuple)) or not timestamps:
        raise ValueError("timestamps must be a non-empty sequence")

    parsed = tuple(
        sorted(
            _parse_timestamp(
                value,
                f"timestamps[{index}]",
                source_timezone=source_timezone,
            )
            for index, value in enumerate(timestamps)
        )
    )
    event_min = parsed[0]
    event_max = parsed[-1]
    watermark = (
        _parse_timestamp(
            source_watermark,
            "source_watermark",
            source_timezone=source_timezone,
        )
        if source_watermark is not None
        else event_max
    )
    requested = (
        _parse_timestamp(
            requested_as_of,
            "requested_as_of",
            source_timezone=source_timezone,
        )
        if requested_as_of is not None
        else watermark
    )
    trustworthy = (
        _parse_timestamp(
            trustworthy_through,
            "trustworthy_through",
            source_timezone=source_timezone,
        )
        if trustworthy_through is not None
        else watermark
    )
    if watermark > requested + timedelta(days=1):
        raise ValueError("source_watermark must not be after requested_as_of")
    if trustworthy > watermark:
        raise ValueError("trustworthy_through must not exceed source_watermark")

    observed_starts = tuple(
        sorted({_period_start(value.date(), grain) for value in parsed})
    )
    all_starts: list[date] = []
    cursor = observed_starts[0]
    while cursor <= observed_starts[-1]:
        all_starts.append(cursor)
        cursor = _next_period(cursor, grain)
    missing_starts = tuple(
        value for value in all_starts if value not in set(observed_starts)
    )
    complete_starts = tuple(
        value
        for value in observed_starts
        if _period_end(value, grain) <= trustworthy.date()
    )
    latest_complete = (
        _period_payload(complete_starts[-1], grain)
        if complete_starts
        else None
    )
    final_start = observed_starts[-1]
    partial_final = (
        _period_payload(final_start, grain)
        if _period_end(final_start, grain) > trustworthy.date()
        else None
    )
    freshness_lag_days = max(0, (requested.date() - watermark.date()).days)
    freshness_status = (
        "current"
        if freshness_lag_days <= 7
        else "recent"
        if freshness_lag_days <= 45
        else "stale"
    )
    input_values = {
        "timestamps": tuple(_iso_utc(value) for value in parsed),
        "grain": grain,
        "requested_as_of": requested.date().isoformat(),
        "source_watermark": _iso_utc(watermark),
        "trustworthy_through": _iso_utc(trustworthy),
        "timezone": timezone.strip(),
    }

    return OperatorResult(
        node_id=node_id,
        operator="profile_time_coverage.v1",
        status="completed",
        seed=seed,
        sample_size=len(parsed),
        values={
            "event_time_min": _iso_utc(event_min),
            "event_time_max": _iso_utc(event_max),
            "source_watermark": _iso_utc(watermark),
            "trustworthy_through": _iso_utc(trustworthy),
            "observed_periods": tuple(
                _period_label(value, grain) for value in observed_starts
            ),
            "missing_periods": tuple(
                _period_label(value, grain) for value in missing_starts
            ),
            "latest_complete_period": latest_complete,
            "partial_final_period": partial_final,
            "freshness_lag_days": freshness_lag_days,
            "freshness_status": freshness_status,
        },
        diagnostics={
            "input_fingerprint": fingerprint(input_values),
            "method": "calendar_coverage_v1",
            "watermarks": {
                "requested_as_of": requested.date().isoformat(),
                "source_watermark": _iso_utc(watermark),
                "trustworthy_through": _iso_utc(trustworthy),
                "timezone": timezone.strip(),
            },
        },
    )


def select_latest_complete_window(
    *,
    node_id: str,
    coverage: object,
    seed: int,
    window_periods: int,
    comparator_kind: str = "none",
) -> OperatorResult:
    """Select a complete current window and require comparable history."""

    if (
        not isinstance(coverage, OperatorResult)
        or coverage.operator != "profile_time_coverage.v1"
        or coverage.status != "completed"
    ):
        raise ValueError(
            "coverage must be a completed profile_time_coverage.v1 result"
        )
    if type(window_periods) is not int or window_periods <= 0:
        raise ValueError("window_periods must be a positive integer")
    if comparator_kind not in {
        "none",
        "previous_window",
        "same_period_prior_year",
    }:
        raise ValueError(
            "comparator_kind must be none, previous_window, or "
            "same_period_prior_year"
        )

    latest_complete = coverage.values.get("latest_complete_period")
    if not isinstance(latest_complete, Mapping):
        raise ValueError("coverage must include a latest_complete_period")
    grain = latest_complete.get("grain")
    start_text = latest_complete.get("start")
    if grain not in _TIME_GRAINS or not isinstance(start_text, str):
        raise ValueError("coverage latest_complete_period is invalid")
    latest_start = date.fromisoformat(start_text)
    current_starts = [latest_start]
    for _ in range(window_periods - 1):
        current_starts.append(_previous_period(current_starts[-1], grain))
    current_starts.reverse()
    current_labels = tuple(
        _period_label(value, grain) for value in current_starts
    )
    observed = set(coverage.values.get("observed_periods", ()))
    missing_current = tuple(
        label for label in current_labels if label not in observed
    )
    base_diagnostics = {
        "coverage_fingerprint": coverage.diagnostics.get("input_fingerprint"),
        "input_fingerprint": fingerprint(
            {
                "coverage": coverage.to_dict(),
                "window_periods": window_periods,
                "comparator_kind": comparator_kind,
            }
        ),
        "method": "latest_complete_window_v1",
    }
    if missing_current:
        return OperatorResult(
            node_id=node_id,
            operator="select_latest_complete_window.v1",
            status="failed",
            seed=seed,
            sample_size=len(current_labels) - len(missing_current),
            diagnostics={
                **base_diagnostics,
                "missing_current_periods": missing_current,
            },
            limitations=(
                "The latest requested window contains missing periods.",
            ),
            failure_code="incomplete_current_window",
            failure_message=(
                "Current window is missing periods: "
                + ", ".join(missing_current)
            ),
        )

    current_window = {
        "grain": grain,
        "start": current_starts[0].isoformat(),
        "end": _period_end(current_starts[-1], grain).isoformat(),
        "periods": current_labels,
    }
    comparator = None
    if comparator_kind != "none":
        if comparator_kind == "previous_window":
            comparator_starts = []
            cursor = _previous_period(current_starts[0], grain)
            for _ in range(window_periods):
                comparator_starts.append(cursor)
                cursor = _previous_period(cursor, grain)
            comparator_starts.reverse()
        else:
            try:
                comparator_starts = [
                    value.replace(year=value.year - 1)
                    for value in current_starts
                ]
            except ValueError:
                return OperatorResult(
                    node_id=node_id,
                    operator="select_latest_complete_window.v1",
                    status="failed",
                    seed=seed,
                    sample_size=len(current_labels),
                    diagnostics=base_diagnostics,
                    limitations=(
                        "The selected calendar period has no exact "
                        "same-period-prior-year comparator.",
                    ),
                    failure_code="non_comparable_calendar_period",
                    failure_message=(
                        "Same-period-prior-year comparison is undefined for "
                        "the selected calendar window."
                    ),
                )
        comparator_labels = tuple(
            _period_label(value, grain) for value in comparator_starts
        )
        missing_comparator = tuple(
            label for label in comparator_labels if label not in observed
        )
        if missing_comparator:
            return OperatorResult(
                node_id=node_id,
                operator="select_latest_complete_window.v1",
                status="failed",
                seed=seed,
                sample_size=len(current_labels),
                diagnostics={
                    **base_diagnostics,
                    "missing_comparator_periods": missing_comparator,
                },
                limitations=(
                    "Current evidence cannot support a recent-change claim "
                    "without the requested comparable periods.",
                ),
                failure_code="insufficient_comparable_history",
                failure_message=(
                    "Comparator is missing periods: "
                    + ", ".join(missing_comparator)
                ),
            )
        comparator = {
            "kind": comparator_kind,
            "start": comparator_starts[0].isoformat(),
            "end": _period_end(comparator_starts[-1], grain).isoformat(),
            "periods": comparator_labels,
        }

    return OperatorResult(
        node_id=node_id,
        operator="select_latest_complete_window.v1",
        status="completed",
        seed=seed,
        sample_size=window_periods,
        values={
            "current_window": current_window,
            "comparator": comparator,
            "excluded_partial_period": coverage.values.get(
                "partial_final_period"
            ),
            "freshness_status": coverage.values.get("freshness_status"),
        },
        diagnostics=base_diagnostics,
    )


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
