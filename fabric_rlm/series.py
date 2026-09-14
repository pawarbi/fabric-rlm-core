"""Time-series arithmetic the reports share: moving averages, level shifts, year-over-year variance, classical decomposition, co-movement.

Every function here is plain arithmetic over figures the source produced,
with no parameter a reader cannot see: a centered moving average, binary
segmentation for level shifts, the average ratio of a calendar month to
the trend for seasonality, and a rank correlation for what moves
together. Each method states the minimum length it needs and stays silent
below it, and a flat series produces no story.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ChangePoint", "Decomposition", "Month", "SeriesStory", "analyse", "change_points", "comovement", "decompose", "moving_average", "usable_months", "yoy"]


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x_mean, y_mean = _mean(xs), _mean(ys)
    sx = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    sy = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if not sx or not sy:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / (sx * sy)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1  # ties share the average rank
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation: one enormous period cannot make every pair read r = 1.00."""
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _slope(values: Sequence[float]) -> float | None:
    """The least-squares slope per step of a series."""
    n = len(values)
    if n < 2:
        return None
    x_mean = (n - 1) / 2
    y_mean = _mean(values)
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if not denominator:
        return None
    return sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values)) / denominator


# --------------------------------------------------------------------------- #
# Level shifts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChangePoint:
    """A level shift: the first period of the new level, the average before and after, and how sharp the split is."""

    start: str
    before: float
    after: float
    statistic: float

    @property
    def pct(self) -> float | None:
        return (self.after - self.before) / abs(self.before) if self.before else None


def change_points(periods: Sequence[Any], *, min_size: int = 4, threshold: float = 3.0, limit: int = 2) -> list[ChangePoint]:
    """Level shifts by binary segmentation over periods with ``value`` and ``start``: the split that separates the means most, when it is sharp against the noise within the segments."""
    values = [p.value for p in periods]
    found: list[ChangePoint] = []

    def split(lo: int, hi: int) -> tuple[int, float] | None:
        best: tuple[int, float] | None = None
        for k in range(lo + min_size, hi - min_size + 1):
            left, right = values[lo:k], values[k:hi]
            pooled = math.sqrt(((len(left) - 1) * _std(left) ** 2 + (len(right) - 1) * _std(right) ** 2) / max(1, len(left) + len(right) - 2))
            pooled = max(pooled, 1e-9 * max(1.0, abs(_mean(values))))
            statistic = abs(_mean(left) - _mean(right)) / (pooled * math.sqrt(1 / len(left) + 1 / len(right)))
            if best is None or statistic > best[1]:
                best = (k, statistic)
        return best

    def line_beats_step(lo: int, hi: int, k: int) -> bool:
        # a smooth slope is not a step: one straight line fitting the segment at least as well as two flat levels, or the segments' own
        # slopes accounting for most of the gap between their means, means the series is trending rather than shifting
        segment = values[lo:hi]
        slope = _slope(segment) or 0.0
        intercept = _mean(segment) - slope * (len(segment) - 1) / 2
        sse_line = sum((v - (intercept + slope * i)) ** 2 for i, v in enumerate(segment))
        left, right = values[lo:k], values[k:hi]
        sse_step = sum((v - _mean(left)) ** 2 for v in left) + sum((v - _mean(right)) ** 2 for v in right)
        if sse_line <= sse_step * 1.05:
            return True
        gap = _mean(right) - _mean(left)
        within = 0.5 * ((_slope(left) or 0.0) + (_slope(right) or 0.0))
        distance = (k + (hi - k - 1) / 2) - (lo + (k - lo - 1) / 2)
        return abs(gap - within * distance) < 0.5 * abs(gap)

    def search(lo: int, hi: int, depth: int) -> None:
        if hi - lo < 2 * min_size or depth > 3 or len(found) >= limit:
            return
        best = split(lo, hi)
        if best is None or best[1] < threshold or line_beats_step(lo, hi, best[0]):
            return
        k = best[0]
        found.append(ChangePoint(periods[k].start, _mean(values[lo:k]), _mean(values[k:hi]), round(best[1], 1)))
        search(k, hi, depth + 1)  # the most recent shift matters most
        search(lo, k, depth + 1)

    if len(values) >= 2 * min_size:
        search(0, len(values), 0)
    return sorted(found, key=lambda p: p.start, reverse=True)[:limit]


# --------------------------------------------------------------------------- #
# Months
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Month:
    """One month of a series: the value and the rows behind it."""

    year: int
    month: int
    value: float
    rows: int = 0

    @property
    def start(self) -> str:
        return _dt.date(self.year, self.month, 1).isoformat()

    @property
    def label(self) -> str:
        return f"{calendar.month_abbr[self.month]} {self.year}"

    @property
    def index(self) -> int:
        return self.year * 12 + self.month - 1


def usable_months(points: Sequence[Any]) -> list[Month]:
    """The months of a series oldest first, with a trailing stub dropped: a last month holding under half the rows of a typical month before it has not fully arrived."""
    months = sorted((Month(int(p.year), int(p.month), float(p.value), int(getattr(p, "rows", 0) or 0)) for p in points), key=lambda m: m.index)
    while len(months) >= 4 and months[-1].rows:
        earlier = sorted(m.rows for m in months[-7:-1] if m.rows)
        if len(earlier) < 2 or months[-1].rows >= 0.5 * earlier[len(earlier) // 2]:
            break
        months = months[:-1]
    return months


def moving_average(values: Sequence[float], window: int) -> list[float | None]:
    """A centered moving average; an even window is the 2 x window average, so a 12-month window sits on a month rather than between two."""
    n = len(values)
    out: list[float | None] = [None] * n
    if window < 2 or n < window:
        return out
    if window % 2:
        half = window // 2
        for i in range(half, n - half):
            out[i] = _mean(values[i - half : i + half + 1])
        return out
    half = window // 2
    for i in range(half, n - half):
        first = _mean(values[i - half : i + half])
        second = _mean(values[i - half + 1 : i + half + 1])
        out[i] = (first + second) / 2
    return out


def yoy(months: Sequence[Month]) -> list[tuple[Month, float | None]]:
    """Each month against the same month a year earlier, as a share of that month; None when there is no such month or it is zero."""
    by_index = {m.index: m for m in months}
    out = []
    for m in months:
        prior = by_index.get(m.index - 12)
        out.append((m, (m.value - prior.value) / abs(prior.value) if prior is not None and prior.value else None))
    return out


# --------------------------------------------------------------------------- #
# Decomposition
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Decomposition:
    """Classical decomposition: trend as the centered 12-month average, a seasonal index per calendar month, the residual; multiplicative unless the series has zeros or negatives."""

    months: tuple[Month, ...]
    trend: tuple[float | None, ...]
    seasonal: Mapping[int, float]  # calendar month -> index (1.0 is an average month; 0.0 when additive)
    residual: tuple[float | None, ...]  # value / (trend x index), or value - trend - index when additive
    multiplicative: bool
    explained: float  # share of the variation around the trend that the seasonal index accounts for
    trend_per_year: float | None  # the fitted slope of the trend component, as a share of its level, per year
    sigma: float  # the spread of the residuals

    def departure(self, i: int) -> float | None:
        """How far a month sits from its trend and season: a share (multiplicative) or a share of the trend (additive)."""
        r, t = self.residual[i], self.trend[i]
        if r is None:
            return None
        if self.multiplicative:
            return r - 1.0
        return r / abs(t) if t else None


def decompose(months: Sequence[Month], *, least: int = 24) -> Decomposition | None:
    """Trend, season and residual for a monthly series of at least ``least`` months."""
    if len(months) < least:
        return None
    values = [m.value for m in months]
    trend = moving_average(values, 12)
    multiplicative = all(v > 0 for v in values) and all(t is None or t > 0 for t in trend)
    ratios: dict[int, list[float]] = {}
    for m, v, t in zip(months, values, trend):
        if t is None:
            continue
        ratios.setdefault(m.month, []).append(v / t if multiplicative else v - t)
    if len(ratios) < 12:
        return None
    raw = {month: _mean(r) for month, r in ratios.items()}
    level = _mean(list(raw.values()))
    seasonal = {month: (r / level if level else 1.0) if multiplicative else r - level for month, r in raw.items()}  # an average month is 1.0, or 0.0
    residual: list[float | None] = []
    detrended: list[float] = []
    fitted: list[float] = []
    for m, v, t in zip(months, values, trend):
        if t is None:
            residual.append(None)
            continue
        s = seasonal[m.month]
        if multiplicative:
            residual.append(v / (t * s) if t * s else None)
            detrended.append(v / t)
            fitted.append(s)
        else:
            residual.append(v - t - s)
            detrended.append(v - t)
            fitted.append(s)
    around = _std(detrended) ** 2
    left = _std([d - f for d, f in zip(detrended, fitted)]) ** 2
    explained = max(0.0, min(1.0, 1.0 - left / around)) if around else 0.0
    present = [t for t in trend if t is not None]
    slope = _slope(present[-24:]) if len(present) >= 6 else None
    trend_per_year = (slope * 12) / abs(_mean(present[-24:])) if slope is not None and _mean(present[-24:]) else None
    departures = [r for r in (residual[i] - 1.0 if multiplicative and residual[i] is not None else residual[i] for i in range(len(residual))) if r is not None]
    return Decomposition(tuple(months), tuple(trend), seasonal, tuple(residual), multiplicative, explained, trend_per_year, _std(departures))


# --------------------------------------------------------------------------- #
# The story of one series
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeriesStory:
    """What a monthly series says once the arithmetic is done, with the figures behind each sentence."""

    months: tuple[Month, ...]
    window: int  # the moving-average window used, 0 when the series is too short for one
    average: tuple[float | None, ...]
    shifts: tuple[ChangePoint, ...]
    yoy: tuple[tuple[Month, float | None], ...]
    decomposition: Decomposition | None
    trend_per_year: float | None
    sentences: tuple[str, ...] = field(default=())
    adjusted: bool = False  # level shifts were looked for on the seasonally adjusted series

    @property
    def methods(self) -> str:
        parts = []
        if self.window:
            parts.append(f"a centered {self.window}-month moving average")
        parts.append("binary segmentation for level shifts (segments of 4 months or more, a split at least 3 standard errors apart)")
        if self.decomposition is not None:
            parts.append(f"classical {'multiplicative' if self.decomposition.multiplicative else 'additive'} decomposition (trend as the 12-month average, a seasonal index per calendar month)")
        if self.adjusted:
            parts.append("level shifts on the seasonally adjusted series (three years or more of a season worth naming)")
        return "; ".join(parts)


def analyse(points: Sequence[Any], *, name: str = "the series") -> SeriesStory:
    """Moving average, level shifts, year-over-year variance and, with two years or more, the decomposition; each sentence carries its figures."""
    months = usable_months(points)
    values = [m.value for m in months]
    window = 12 if len(months) >= 18 else 3 if len(months) >= 6 else 0
    average = moving_average(values, window) if window else [None] * len(months)
    yearly = yoy(months)
    decomposition = decompose(months)
    adjusted = False
    basis: Sequence[Any] = months
    if decomposition is not None and len(months) >= 36 and decomposition.explained >= 0.2 and _swing(decomposition) >= 0.05:
        # a season observed at least twice per calendar month is taken out before looking for a step, so a peak is not a shift
        basis = [Month(m.year, m.month, (m.value / decomposition.seasonal[m.month]) if decomposition.multiplicative else (m.value - decomposition.seasonal[m.month]), m.rows) for m in months]
        adjusted = True
    shifts = change_points(basis) if len(months) >= 8 else []
    if decomposition is not None:
        trend_per_year = decomposition.trend_per_year
    else:
        present = [a for a in average if a is not None]
        slope = _slope(present) if len(present) >= 6 else None
        trend_per_year = (slope * 12) / abs(_mean(present)) if slope is not None and _mean(present) else None
    sentences: list[str] = []
    cap = name[0].upper() + name[1:] if name else "The series"
    for shift in shifts[:1]:
        when = next((m.label for m in months if m.start == shift.start), shift.start)
        sentences.append(f"{cap} has been running at a new level since {when}{' (seasonally adjusted)' if adjusted else ''}: {_num(shift.before)} to {_num(shift.after)} a month ({_pct(shift.pct)}).")
    if trend_per_year is not None and abs(trend_per_year) >= 0.03:
        span = min(24, len(months)) if decomposition is not None else len([a for a in average if a is not None])
        sentences.append(f"The trend runs {trend_per_year:+.0%} a year over the last {span} months.")
    with_prior = [(m, v) for m, v in yearly if v is not None][-12:]
    if len(with_prior) >= 3 and max(abs(v) for _m, v in with_prior) >= 0.01:
        ups = sum(1 for _m, v in with_prior if v > 0)
        last3 = [v for _m, v in with_prior[-3:]]
        sentences.append(f"Up on the same month a year earlier in {ups} of the last {len(with_prior)} months; the last {len(last3)} averaged {_mean(last3):+.0%}.")
    if decomposition is not None and decomposition.explained >= 0.2 and _swing(decomposition) >= 0.05:
        peak = max(decomposition.seasonal, key=lambda k: decomposition.seasonal[k])
        trough = min(decomposition.seasonal, key=lambda k: decomposition.seasonal[k])
        if decomposition.multiplicative:
            swing = f"{calendar.month_name[peak]} runs {decomposition.seasonal[peak] - 1:+.0%} against an average month, {calendar.month_name[trough]} {decomposition.seasonal[trough] - 1:+.0%}"
        else:
            swing = f"{calendar.month_name[peak]} runs {_num(decomposition.seasonal[peak])} above an average month, {calendar.month_name[trough]} {_num(decomposition.seasonal[trough])}"
        sentences.append(f"Seasonality explains {decomposition.explained:.0%} of the variation around the trend: {swing}.")
    if decomposition is not None and decomposition.sigma:
        largest = max((i for i in range(len(months)) if decomposition.departure(i) is not None), key=lambda i: abs(decomposition.departure(i) or 0), default=None)
        if largest is not None:
            departure = decomposition.departure(largest) or 0.0
            if abs(departure) >= 2 * decomposition.sigma and abs(departure) >= 0.05:
                sentences.append(f"The largest departure from trend and season was {months[largest].label} at {departure:+.0%}.")
    return SeriesStory(tuple(months), window, tuple(average), tuple(shifts), tuple(yearly), decomposition, trend_per_year, tuple(sentences), adjusted)


def expected_value(story: SeriesStory, year: int, month: int) -> tuple[float, str, float] | None:
    """What the trend, and with two years or more the season, implied for one month: (expected, the basis in words, the usual spread as a share).

    Inside the decomposition the expected value is the trend times the seasonal index (or their sum when additive).
    The centered average has no value for the last six months, so there the trend is extended from its last known
    month at the fitted rate. Without a decomposition the expected value comes from the fitted line through the
    moving average, and the spread from how far months usually sit from that average.
    """
    index = next((i for i, m in enumerate(story.months) if m.year == year and m.month == month), None)
    if index is None:
        return None
    d = story.decomposition
    if d is not None:
        trend = d.trend[index]
        basis = "trend and season"
        if trend is None:
            known = [(i, t) for i, t in enumerate(d.trend) if t is not None]
            if not known:
                return None
            k, anchor_value = known[-1] if index > known[-1][0] else known[0]
            per_month = (d.trend_per_year or 0.0) / 12
            trend = anchor_value * (1 + per_month) ** (index - k) if d.multiplicative else anchor_value + per_month * abs(anchor_value) * (index - k)
            basis = f"trend (extended from {story.months[k].label}) and season"
        season = d.seasonal.get(month, 1.0 if d.multiplicative else 0.0)
        expected = trend * season if d.multiplicative else trend + season
        return expected, basis, 2 * d.sigma
    present = [(i, a) for i, a in enumerate(story.average) if a is not None]
    if len(present) < 6:
        return None
    ys = [a for _i, a in present]
    slope = _slope(ys) or 0.0
    centre = sum(i for i, _a in present) / len(present)
    expected = _mean(ys) + slope * (index - centre)
    ratios = [m.value / a - 1 for m, a in zip(story.months, story.average) if a]
    return expected, "trend line", 2 * _std(ratios) if len(ratios) >= 3 else 0.0


def period_check(story: SeriesStory, year: int, month: int, *, name: str = "the series") -> str | None:
    """One sentence on whether a month was in line with the trend and season, with the figures behind it; None when the series does not cover the month."""
    found = expected_value(story, year, month)
    if found is None:
        return None
    expected, basis, spread = found
    spread = max(spread, 0.02) if spread else spread  # nothing sits exactly on its trend; under two percent is not a departure
    actual = next(m.value for m in story.months if m.year == year and m.month == month)
    label = f"{calendar.month_name[month]} {year}"
    cap = name[0].upper() + name[1:] if name else "The series"
    if not expected:
        return f"{cap}: the {basis} implied nothing for {label}."
    gap = actual / expected - 1
    if spread:
        verdict = ("within" if abs(gap) <= spread else "outside") + f" the usual spread of ±{spread:.0%}"
    else:
        verdict = "too few months to say whether that is usual"
    direction = "above" if gap > 0 else "below"
    sentence = f"{cap} came in at {_num(actual)} for {label}, {abs(gap):.0%} {direction} what the {basis} implied ({_num(expected)}): {verdict}."
    yearly = next((v for m, v in story.yoy if m.year == year and m.month == month), None)
    if yearly is not None and story.trend_per_year is not None:
        sentence += f" Against the same month a year earlier it is {yearly:+.0%}; the trend runs {story.trend_per_year:+.0%} a year."
    return sentence


def _swing(d: Decomposition) -> float:
    """How far the season swings around an average month, as a share of the level."""
    if not d.seasonal:
        return 0.0
    if d.multiplicative:
        return max(abs(v - 1.0) for v in d.seasonal.values())
    level = _mean([t for t in d.trend if t is not None])
    return max(abs(v) for v in d.seasonal.values()) / abs(level) if level else 0.0


def comovement(series_by_name: Mapping[str, Sequence[Any]], *, least: int = 12, strength: float = 0.6, limit: int = 6) -> list[str]:
    """Which monthly series moved together year over year, and which led by a month: rank correlations over the shared months, associations rather than causes."""
    changes: dict[str, dict[int, float]] = {}
    for name, points in series_by_name.items():
        changes[name] = {m.index: v for m, v in yoy(usable_months(points)) if v is not None}
    found: list[tuple[float, str]] = []
    names = list(changes)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(set(changes[a]) & set(changes[b]))
            if len(shared) < least:
                continue
            same = _spearman([changes[a][s] for s in shared], [changes[b][s] for s in shared])
            a_leads = _spearman([changes[a][s] for s in shared[:-1]], [changes[b][s] for s in shared[1:]])
            b_leads = _spearman([changes[b][s] for s in shared[:-1]], [changes[a][s] for s in shared[1:]])
            if same is not None and abs(same) >= strength:
                found.append((abs(same), f"{a[0].upper() + a[1:]} and {b} moved {'together' if same > 0 else 'in opposite directions'} year over year across {len(shared)} months (r = {same:.2f})."))
            if a_leads is not None and abs(a_leads) >= strength and abs(a_leads) > abs(same or 0) + 0.1:
                found.append((abs(a_leads), f"{a[0].upper() + a[1:]} led {b} by a month (r = {a_leads:.2f} at a one-month lag)."))
            if b_leads is not None and abs(b_leads) >= strength and abs(b_leads) > abs(same or 0) + 0.1:
                found.append((abs(b_leads), f"{b[0].upper() + b[1:]} led {a} by a month (r = {b_leads:.2f} at a one-month lag)."))
    lines = [text for _s, text in sorted(found, key=lambda item: -item[0])[:limit]]
    if lines:
        lines.append("These are associations in the source's own history; they say what moved with what, not what caused what.")
    return lines


def _num(value: float) -> str:
    if abs(value) >= 1000 or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.0%}"
