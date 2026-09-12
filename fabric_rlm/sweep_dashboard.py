"""The dashboard of a sweep: what moved, the trends, and for every finding the waterfall of drivers and the driver scatter.

Everything is inline HTML, CSS and SVG. Nothing is loaded from the
network, so the page renders inside a Fabric notebook (``displayHTML``),
as a file saved to the lakehouse, or in an email. Every number on the page
is a figure from the ledger, computed by the source; the queries behind
each finding are on the page.
"""

from __future__ import annotations

import calendar
from collections.abc import Sequence
from html import escape as esc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .sweep import Decomposition, Point, Sweep, SweepFinding

__all__ = ["document", "render"]

_STYLE = """
.wm{font:14px/1.45 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:#1f2937;max-width:1180px;margin:0 auto;padding:16px;background:#fff}
.wm *{box-sizing:border-box}
.wm h1{font-size:22px;margin:0 0 4px;font-weight:650}
.wm h2{font-size:16px;margin:22px 0 10px;font-weight:650;color:#111827}
.wm h3{font-size:15px;margin:0 0 6px;font-weight:650}
.wm .sub{color:#6b7280;margin-bottom:14px}
.wm .badge{display:inline-block;border-radius:999px;padding:2px 9px;font-size:12px;font-weight:600;vertical-align:middle}
.wm .badge.ok{background:#dcfce7;color:#166534}
.wm .badge.warn{background:#fee2e2;color:#991b1b}
.wm .badge.muted{background:#f3f4f6;color:#4b5563}
.wm .kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.wm .kpi{border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;background:#fff}
.wm .kpi .t{font-size:12px;color:#6b7280;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.wm .kpi .v{font-size:21px;font-weight:650;margin:2px 0}
.wm .kpi .d{font-size:13px}
.wm .up{color:#15803d}.wm .down{color:#b91c1c}.wm .flat{color:#6b7280}
.wm .card{border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin:0 0 16px;background:#fff}
.wm .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:820px){.wm .grid2{grid-template-columns:1fr}}
.wm .chart{width:100%;height:auto;display:block}
.wm .caption{font-size:12px;color:#6b7280;margin:2px 0 8px}
.wm table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}
.wm th,.wm td{padding:5px 8px;border-bottom:1px solid #f1f5f9;text-align:right;white-space:nowrap}
.wm th{color:#6b7280;font-weight:600;font-size:12px}
.wm th:first-child,.wm td:first-child{text-align:left;white-space:normal}
.wm .bar{display:inline-block;height:8px;border-radius:2px;background:#93c5fd;vertical-align:middle;margin-right:6px}
.wm .bar.neg{background:#fca5a5}
.wm .chip{display:inline-block;border:1px solid #e5e7eb;border-radius:999px;padding:1px 9px;margin:2px 4px 2px 0;font-size:12px;color:#374151;background:#f9fafb}
.wm .chip b{font-weight:600}
.wm .note{color:#92400e;background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:6px 10px;margin:6px 0;font-size:13px}
.wm .story{margin:6px 0 10px}
.wm details{margin-top:8px}
.wm summary{cursor:pointer;color:#4b5563;font-size:13px}
.wm pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:#f8fafc;padding:8px;border-radius:8px;overflow:auto;margin:6px 0}
.wm .foot{color:#6b7280;font-size:12px;margin-top:18px}
.wm svg text{font-family:inherit}
"""

_PALETTE = ("#2563eb", "#9ca3af", "#f59e0b", "#10b981")
_UP, _DOWN, _TOTAL, _OTHER = "#16a34a", "#dc2626", "#475569", "#94a3b8"


# --------------------------------------------------------------------------- #
# Numbers and words
# --------------------------------------------------------------------------- #


def _compact(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{value / 1e9:.2f}B"
    if magnitude >= 1e6:
        return f"{value / 1e6:.2f}M"
    if magnitude >= 1e4:
        return f"{value / 1e3:.1f}k"
    if magnitude >= 100 or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _full(value: float) -> str:
    if abs(value) >= 1000 or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def _direction(delta: float) -> str:
    return "up" if delta > 0 else "down" if delta < 0 else "flat"


def _arrow(delta: float) -> str:
    return "▲" if delta > 0 else "▼" if delta < 0 else "–"


def _label(value: Any) -> str:
    return "(blank)" if value is None else str(value)


def _short(text: str, limit: int = 16) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _word(path: Any) -> str:
    from .data_agent_review import humanize_column

    return humanize_column(str(path["column"])) if path else "total"


def _concentration_words(concentration: str) -> str:
    return {
        "single": "one group carries it",
        "concentrated": "concentrated in a few groups",
        "proportional": "in proportion to size",
        "broad": "spread across groups",
        "offsetting": "offsetting moves",
        "none": "nothing to split",
    }.get(concentration, concentration)


def _nice_ticks(low: float, high: float, count: int = 4) -> list[float]:
    """Round tick values covering [low, high]."""
    if high <= low:
        high = low + 1.0
    raw = (high - low) / max(1, count)
    magnitude = 10 ** _floor_log10(raw)
    step = next(s * magnitude for s in (1, 2, 2.5, 5, 10) if s * magnitude >= raw)
    start = _floor_div(low, step) * step
    ticks = []
    value = start
    while value <= high + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _floor_log10(value: float) -> int:
    import math

    return int(math.floor(math.log10(value))) if value > 0 else 0


def _floor_div(value: float, step: float) -> int:
    import math

    return int(math.floor(value / step))


# --------------------------------------------------------------------------- #
# SVG charts
# --------------------------------------------------------------------------- #


def _trend_svg(points: Sequence["Point"], title: str, aggregate: str) -> str:
    """One line per year over the twelve months, the latest year strongest."""
    by_year: dict[int, dict[int, float]] = {}
    for p in points:
        by_year.setdefault(p.year, {})[p.month] = p.value
    years = sorted(by_year)
    if not years:
        return ""
    width, height, left, right, top, bottom = 640, 230, 64, 16, 28, 34
    values = [v for year in by_year.values() for v in year.values()]
    low, high = min(0.0, min(values)), max(values)
    ticks = _nice_ticks(low, high)
    low, high = min(ticks[0], low), max(ticks[-1], high)
    span = high - low or 1.0

    def x(month: int) -> float:
        return left + (month - 1) * (width - left - right) / 11

    def y(value: float) -> float:
        return top + (high - value) * (height - top - bottom) / span

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    parts.append(f'<text x="{left}" y="16" font-size="12" fill="#374151" font-weight="600">{esc(title)}</text>')
    for tick in ticks:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" font-size="10" fill="#6b7280" text-anchor="end">{esc(_compact(tick))}</text>')
    for month in range(1, 13):
        parts.append(f'<text x="{x(month):.1f}" y="{height - 14}" font-size="10" fill="#6b7280" text-anchor="middle">{calendar.month_abbr[month]}</text>')
    latest = years[-1]
    for index, year in enumerate(years):
        series = by_year[year]
        color = _PALETTE[0] if year == latest else _PALETTE[min(1 + (len(years) - 1 - index - 1), len(_PALETTE) - 1)] if len(years) > 1 else _PALETTE[0]
        if year != latest:
            color = _PALETTE[1] if index == len(years) - 2 else _PALETTE[2]
        stroke = 2.4 if year == latest else 1.6
        coordinates = [(x(m), y(series[m])) for m in range(1, 13) if m in series]
        if len(coordinates) >= 2:
            path = " ".join(f"{'M' if i == 0 else 'L'}{cx:.1f},{cy:.1f}" for i, (cx, cy) in enumerate(coordinates))
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linejoin="round"/>')
        for m in range(1, 13):
            if m in series:
                parts.append(f'<circle cx="{x(m):.1f}" cy="{y(series[m]):.1f}" r="2.6" fill="{color}"><title>{esc(f"{calendar.month_abbr[m]} {year}: {_full(series[m])}")}</title></circle>')
        legend_x = width - right - 56 * (len(years) - index)
        parts.append(f'<rect x="{legend_x}" y="8" width="10" height="10" fill="{color}" rx="2"/><text x="{legend_x + 14}" y="17" font-size="11" fill="#374151">{year}</text>')
    if aggregate == "avg":
        parts.append(f'<text x="{width - right}" y="{height - 2}" font-size="10" fill="#9ca3af" text-anchor="end">monthly average</text>')
    parts.append("</svg>")
    return "".join(parts)


def _waterfall_svg(d: "Decomposition", title: str) -> str:
    """Before, the largest groups in the direction of the change, the largest opposite ones, the rest, after."""
    parent = d.parent
    same = [g for g in d.groups if g.delta * parent.delta > 0][:5]
    opposite = [g for g in d.groups if g.delta * parent.delta < 0][:2]
    steps = [(_label(g.group), g.delta) for g in same] + [(_label(g.group), g.delta) for g in opposite]
    other = parent.delta - sum(delta for _n, delta in steps)
    if abs(other) > 1e-9 * max(1.0, abs(parent.delta)):
        steps.append(("all other", other))
    bars: list[tuple[str, float, float, str]] = [("before", 0.0 if parent.aggregate != "avg" else 0.0, parent.before_value, _TOTAL)]
    running = parent.before_value
    for name, delta in steps:
        bars.append((name, running, running + delta, _UP if delta > 0 else _DOWN if delta < 0 else _OTHER))
        running += delta
    bars.append(("after", 0.0, parent.after_value, _TOTAL))
    width, height, left, right, top, bottom = 640, 270, 64, 12, 30, 58
    extremes = [v for _n, a, b, _c in bars for v in (a, b)]
    low, high = min(extremes), max(extremes)
    if low > 0 and low < 0.25 * high:
        low = 0.0
    pad = (high - low) * 0.08 or 1.0
    low, high = (min(0.0, low - pad) if low <= 0 else low - pad), high + pad
    ticks = _nice_ticks(low, high)
    span = high - low or 1.0
    slot = (width - left - right) / max(1, len(bars))
    bar_width = min(46.0, slot * 0.66)

    def y(value: float) -> float:
        return top + (high - value) * (height - top - bottom) / span

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    parts.append(f'<text x="{left}" y="16" font-size="12" fill="#374151" font-weight="600">{esc(title)}</text>')
    for tick in ticks:
        if low <= tick <= high:
            parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="#eef2f7"/>')
            parts.append(f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" font-size="10" fill="#6b7280" text-anchor="end">{esc(_compact(tick))}</text>')
    previous_top: float | None = None
    for index, (name, start, end, color) in enumerate(bars):
        cx = left + slot * index + slot / 2
        y0, y1 = y(max(start, end)), y(min(start, end))
        if color == _TOTAL:
            y0, y1 = y(end), y(max(low, 0.0) if low <= 0 <= high else low)
        parts.append(f'<rect x="{cx - bar_width / 2:.1f}" y="{y0:.1f}" width="{bar_width:.1f}" height="{max(1.0, y1 - y0):.1f}" fill="{color}" rx="2"><title>{esc(f"{name}: {_full(end - start) if color != _TOTAL else _full(end)}")}</title></rect>')
        if previous_top is not None and color != _TOTAL:
            parts.append(f'<line x1="{cx - slot / 2 - bar_width / 2 + bar_width:.1f}" x2="{cx - bar_width / 2:.1f}" y1="{previous_top:.1f}" y2="{previous_top:.1f}" stroke="#cbd5e1" stroke-dasharray="3 3"/>')
        previous_top = y(end)
        value_text = _compact(end) if color == _TOTAL else f"{'+' if end - start > 0 else ''}{_compact(end - start)}"
        parts.append(f'<text x="{cx:.1f}" y="{y0 - 4:.1f}" font-size="10" fill="#374151" text-anchor="middle">{esc(value_text)}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{height - bottom + 14}" font-size="10" fill="#4b5563" text-anchor="middle">{esc(_short(name, 14))}</text>')
    parts.append(f'<text x="{left}" y="{height - 4}" font-size="10" fill="#9ca3af">{esc(parent.comparison.label)}; bars are the change each group contributed</text>')
    parts.append("</svg>")
    return "".join(parts)


def _scatter_svg(d: "Decomposition", title: str) -> str:
    """Share of the base against share of the change: above the diagonal a group moved more than its size."""
    points: list[tuple[str, float, float]] = []
    for g in d.groups:
        change, base = d.share_of_change(g), d.share_of_base(g)
        if change is not None and base is not None:
            points.append((_label(g.group), base, change))
    if not points:
        return ""
    width, height, left, right, top, bottom = 640, 270, 64, 16, 30, 46
    x_high = max(1.0, max(p[1] for p in points) * 1.12)
    y_low, y_high = min(0.0, min(p[2] for p in points) * 1.12), max(1.0, max(p[2] for p in points) * 1.12)

    def x(value: float) -> float:
        return left + value * (width - left - right) / x_high

    def y(value: float) -> float:
        return top + (y_high - value) * (height - top - bottom) / (y_high - y_low or 1.0)

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    parts.append(f'<text x="{left}" y="16" font-size="12" fill="#374151" font-weight="600">{esc(title)}</text>')
    for tick in [t for t in _nice_ticks(y_low, y_high) if y_low <= t <= y_high]:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" font-size="10" fill="#6b7280" text-anchor="end">{tick:.0%}</text>')
    for tick in [t for t in _nice_ticks(0.0, x_high) if 0 <= t <= x_high]:
        parts.append(f'<text x="{x(tick):.1f}" y="{height - bottom + 14}" font-size="10" fill="#6b7280" text-anchor="middle">{tick:.0%}</text>')
    diagonal_end = min(x_high, y_high)
    parts.append(f'<line x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(diagonal_end):.1f}" y2="{y(diagonal_end):.1f}" stroke="#cbd5e1" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{x(diagonal_end) - 4:.1f}" y="{y(diagonal_end) - 6:.1f}" font-size="10" fill="#9ca3af" text-anchor="end">moved in step with size</text>')
    if y_low < 0:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="#d1d5db"/>')
    for name, base, change in sorted(points, key=lambda p: -abs(p[2])):
        color = _UP if change > base + 0.1 else _DOWN if change < 0 else _OTHER
        parts.append(f'<circle cx="{x(base):.1f}" cy="{y(change):.1f}" r="6" fill="{color}" fill-opacity="0.85"><title>{esc(f"{name}: {change:.0%} of the change on {base:.0%} of the base")}</title></circle>')
        parts.append(f'<text x="{x(base) + 8:.1f}" y="{y(change) + 4:.1f}" font-size="10" fill="#374151">{esc(_short(name, 18))}</text>')
    parts.append(f'<text x="{width - right}" y="{height - 4}" font-size="10" fill="#9ca3af" text-anchor="end">share of the base (x) against share of the change (y)</text>')
    parts.append("</svg>")
    return "".join(parts)


_SERIES_PALETTE = ("#2563eb", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444", "#0891b2", "#a16207", "#6b7280")


def _multi_trend_svg(series_by_label: "dict[Any, Sequence[Point]]", title: str) -> str:
    """One line per group over the months covered, the largest groups first."""
    present = sorted({(p.year, p.month) for points in series_by_label.values() for p in points})
    if len(present) < 2:
        return ""
    months: list[tuple[int, int]] = []  # every month from the first to the last, so gaps keep their width
    year, month = present[0]
    while (year, month) <= present[-1]:
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    width, height, left, right, top, bottom = 640, 260, 64, 16, 28, 44
    values = [p.value for points in series_by_label.values() for p in points]
    low, high = min(0.0, min(values)), max(values)
    ticks = _nice_ticks(low, high)
    low, high = min(ticks[0], low), max(ticks[-1], high)
    span = high - low or 1.0
    index = {key: i for i, key in enumerate(months)}

    def x(key: tuple[int, int]) -> float:
        return left + index[key] * (width - left - right) / max(1, len(months) - 1)

    def y(value: float) -> float:
        return top + (high - value) * (height - top - bottom) / span

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    parts.append(f'<text x="{left}" y="16" font-size="12" fill="#374151" font-weight="600">{esc(title)}</text>')
    for tick in ticks:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" font-size="10" fill="#6b7280" text-anchor="end">{esc(_compact(tick))}</text>')
    step = max(1, round(len(months) / 6))
    for i, key in enumerate(months):
        if i % step == 0 or (i == len(months) - 1 and (len(months) - 1) % step >= step / 2):
            parts.append(f'<text x="{x(key):.1f}" y="{height - bottom + 14}" font-size="10" fill="#6b7280" text-anchor="middle">{calendar.month_abbr[key[1]]} {key[0]}</text>')
    labels = sorted(series_by_label, key=lambda k: -sum(p.value for p in series_by_label[k]))
    for i, label in enumerate(labels[:8]):
        color = _SERIES_PALETTE[i % len(_SERIES_PALETTE)]
        points = sorted(series_by_label[label], key=lambda p: (p.year, p.month))
        coordinates = [(x((p.year, p.month)), y(p.value)) for p in points]
        if len(coordinates) >= 2:
            path = " ".join(f"{'M' if j == 0 else 'L'}{cx:.1f},{cy:.1f}" for j, (cx, cy) in enumerate(coordinates))
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round"><title>{esc(_label(label))}</title></path>')
        for p, (cx, cy) in zip(points, coordinates):
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.2" fill="{color}"><title>{esc(f"{_label(label)}, {calendar.month_abbr[p.month]} {p.year}: {_full(p.value)}")}</title></circle>')
        legend_y = height - 8
        legend_x = left + i * 78
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 9}" width="9" height="9" fill="{color}" rx="2"/><text x="{legend_x + 12}" y="{legend_y}" font-size="10" fill="#374151">{esc(_short(_label(label), 11))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _bars_svg(items: Sequence[tuple[str, float, float | None]], title: str, color: str) -> str:
    """Horizontal bars for a ranked list: label, change, and the relative change when there is a base."""
    if not items:
        return f'<div class="caption">{esc(title)}: none</div>'
    width, left, right, row_height, top = 640, 150, 90, 24, 26
    height = top + row_height * len(items) + 10
    biggest = max(abs(v) for _l, v, _p in items) or 1.0
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    parts.append(f'<text x="0" y="16" font-size="12" fill="#374151" font-weight="600">{esc(title)}</text>')
    for i, (label, value, pct) in enumerate(items):
        y = top + i * row_height
        length = (width - left - right) * abs(value) / biggest
        parts.append(f'<text x="{left - 8}" y="{y + 15}" font-size="11" fill="#374151" text-anchor="end">{esc(_short(label, 22))}</text>')
        parts.append(f'<rect x="{left}" y="{y + 4}" width="{max(1.0, length):.1f}" height="15" fill="{color}" rx="2"><title>{esc(f"{label}: {_full(value)}")}</title></rect>')
        text = f"{'+' if value > 0 else ''}{_compact(value)}" + (f" ({pct:+.0%})" if pct is not None else "")
        parts.append(f'<text x="{left + max(1.0, length) + 6:.1f}" y="{y + 15}" font-size="11" fill="#4b5563">{esc(text)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# HTML pieces
# --------------------------------------------------------------------------- #


def _groups_table(d: "Decomposition") -> str:
    rows = []
    biggest = max((abs(d.share_of_change(g) or 0) for g in d.groups), default=0) or 1.0
    for g in d.groups:
        change, base = d.share_of_change(g), d.share_of_base(g)
        width = 0 if change is None else int(60 * min(1.0, abs(change) / biggest))
        bar = f'<span class="bar{" neg" if (change or 0) < 0 else ""}" style="width:{width}px"></span>' if change is not None else ""
        rows.append(
            f"<tr><td>{esc(_label(g.group))}</td><td>{esc(_full(g.before_value))}</td><td>{esc(_full(g.after_value))}</td>"
            f'<td class="{_direction(g.delta)}">{esc(_full(g.delta))}</td><td>{bar}{"" if change is None else f"{change:.0%}"}</td><td>{"" if base is None else f"{base:.0%}"}</td></tr>'
        )
    return f"<table><tr><th>{esc(_word(d.path))}</th><th>Before</th><th>After</th><th>Change</th><th>Share of change</th><th>Share of base</th></tr>{''.join(rows)}</table>"


def _kpi_cards(result: "Sweep") -> str:
    cards = []
    flagged = {id(f.movement): f.flags for f in result.findings}
    for m in result.ledger:
        if m.path is not None or f"{m.fact}|{m.measure}" in result.collapsed:
            continue
        direction = _direction(m.delta)
        flags = flagged.get(id(m), ())
        cards.append(
            f'<div class="kpi"><div class="t" title="{esc(result.phrase(m))}">{esc(result.phrase(m).capitalize())}</div>'
            f'<div class="v">{esc(_compact(m.after_value))}</div>'
            f'<div class="d {direction}">{_arrow(m.delta)} {esc(_pct(m.pct))} <span class="flat">{esc(m.comparison.label)}, from {esc(_compact(m.before_value))}</span></div>'
            + (f'<div class="flat" style="font-size:12px">{esc(flags[0])}</div>' if flags else "")
            + "</div>"
        )
    return f'<div class="kpis">{"".join(cards)}</div>' if cards else ""


def _trends(result: "Sweep") -> str:
    parts = []
    for key, points in result.series.items():
        if len(points) < 2 or key in result.collapsed:
            continue
        fact, measure = key.split("|", 1)
        aggregate = next((m.aggregate for m in result.ledger if m.fact == fact and m.measure == measure), "sum")
        phrase = result.words.get(key, key).capitalize()
        parts.append(_trend_svg(points, f"{phrase} by month", aggregate))
    return f'<div class="grid2">{"".join(parts)}</div>' if parts else ""


def _finding(result: "Sweep", finding: "SweepFinding", index: int, *, others_as_tables: int = 0) -> str:
    from .sweep import _concentration_sentence

    m = finding.movement
    best = finding.best
    parts = [f'<div class="card"><h3>{index}. {esc(result.headline(m))}</h3>']
    for flag in finding.flags:
        parts.append(f'<div class="note">{esc(flag)}</div>')
    if best is not None:
        parts.append(f'<div class="story"><b>By {esc(_word(best.path))}:</b> {esc(_concentration_sentence(best))}</div>')
        parts.append('<div class="grid2">' + _waterfall_svg(best, f"What moved {m.comparison.label}, by {_word(best.path)}") + _scatter_svg(best, f"Who moved more than their size, by {_word(best.path)}") + "</div>")
        parts.append(_groups_table(best))
        lead = finding.lead_drill
        if lead is not None:
            parts.append(f'<div class="story"><b>Within {esc(_label(lead.parent.group))}, by {esc(_word(lead.path))}:</b> {esc(_concentration_sentence(lead))}</div>')
            parts.append('<div class="grid2">' + _waterfall_svg(lead, f"Within {_label(lead.parent.group)}, by {_word(lead.path)}") + _scatter_svg(lead, f"Within {_label(lead.parent.group)}: who moved more than their size") + "</div>")
            parts.append(_groups_table(lead))
        elif finding.drill:
            tried = ", ".join(_word(d.path) for d in finding.drill)
            parts.append(f'<div class="story">Within {esc(_label(finding.drill[0].parent.group))}, nothing stands out by {esc(tried)}.</div>')
        others = finding.decompositions[1:]
        for d in others[:others_as_tables]:
            parts.append(f'<div class="story"><b>By {esc(_word(d.path))}:</b> {esc(_concentration_sentence(d))}</div>')
            parts.append(_groups_table(d))
        if others[others_as_tables:]:
            chips = "".join(f'<span class="chip">by <b>{esc(_word(d.path))}</b>: {esc(_concentration_words(d.concentration))}</span>' for d in others[others_as_tables:])
            parts.append(f'<div class="caption">Other groupings tried: {chips}</div>')
    queries = [("Measured by", m.query), ("Recomputed by (after)", m.verification.get("after", "")), ("Recomputed by (before)", m.verification.get("before", ""))]
    if best is not None and best.verification:
        queries.append((f"Groups by {_word(best.path)}, recomputed by (after)", best.verification.get("after", "")))
    parts.append("<details><summary>Queries behind these figures</summary>" + "".join(f'<div class="caption">{esc(label)}</div><pre>{esc(text)}</pre>' for label, text in queries if text) + "</details>")
    parts.append("</div>")
    return "".join(parts)


def render(result: "Sweep") -> str:
    """The dashboard as an HTML fragment with its own styles, ready for ``displayHTML``."""
    kind = "semantic model" if result.kind == "semantic_model" else result.kind
    years = f"{result.years[0]} to {result.years[-1]}" if result.years else "the data's years"
    parts = [f"<style>{_STYLE}</style>", '<div class="wm">']
    parts.append(f"<h1>What moved in {esc(result.source)}</h1>")
    parts.append(f'<div class="sub">{esc(kind)}, {esc(years)}: {len(result.findings)} material movement(s) from {len(result.ledger)} figures measured by the source in {result.queries} of {result.budget} queries{esc(_seconds(result.elapsed))}. {_badge(result)}</div>')
    kpis = _kpi_cards(result)
    if kpis:
        parts.append("<h2>Headline movements</h2>")
        parts.append(kpis)
    trends = _trends(result)
    if trends:
        parts.append("<h2>Trends</h2>")
        parts.append(trends)
    if result.findings:
        parts.append("<h2>Driver analysis</h2>")
        parts.append('<div class="caption">Each material movement is split by every grouping the joins reach. The waterfall shows which groups carried the change; the scatter shows whether they moved more than their size (above the diagonal) or merely in proportion to it.</div>')
        for index, finding in enumerate(result.findings, start=1):
            parts.append(_finding(result, finding, index))
    else:
        parts.append('<div class="card">No material movement to decompose.</div>')
    if result.mismatches:
        parts.append("<h2>Figures that did not recompute</h2>")
        parts.append("".join(f'<div class="note">{esc(text)}</div>' for text in result.mismatches[:10]))
    if result.notes:
        parts.append("<h2>Notes</h2>")
        parts.append("".join(f'<div class="caption">{esc(note)}</div>' for note in result.notes))
    parts.append('<div class="foot">Every figure on this page was computed by the source with the query shown under its finding, and recomputed by an independent per-period query. No language model was involved in producing the numbers.</div>')
    parts.append("</div>")
    return "\n".join(parts)


def document(result: "Sweep") -> str:
    """The dashboard as a complete HTML page."""
    return _page(f"What moved in {result.source}", render(result))


def _page(title: str, body: str) -> str:
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)}</title></head><body style="margin:0;background:#f8fafc">{body}</body></html>'


# --------------------------------------------------------------------------- #
# Reports of one kind
# --------------------------------------------------------------------------- #


def _badge(result: "Sweep") -> str:
    if result.verified and result.recomputed:
        return f'<span class="badge {"ok" if not result.mismatches else "warn"}">{result.recomputed} figures recomputed, {len(result.mismatches)} mismatch{"es" if len(result.mismatches) != 1 else ""}</span>'
    if result.verified:
        return '<span class="badge muted">nothing to recompute</span>'
    return '<span class="badge muted">not recomputed</span>'


def _seconds(value: float) -> str:
    return "" if not value else (f", {value:.1f} s" if value < 10 else f", {value:.0f} s")


def _reading(spec: Any) -> str:
    items = "".join(f"<li>{esc(line)}</li>" for line in spec.reading)
    request = f'<div class="caption">Request: “{esc(spec.request)}”</div>' if spec.request else ""
    return f'<div class="card" style="padding:10px 16px">{request}<div class="caption" style="margin-top:4px">How the request was read</div><ul style="margin:4px 0 4px 18px;padding:0;font-size:13px">{items}</ul></div>'


def _trend_report(report: Any) -> list[str]:
    result = report.sweep
    parts = []
    kpis = _kpi_cards(result)
    if kpis:
        parts.append("<h2>Latest movements</h2>")
        parts.append(kpis)
    trends = _trends(result)
    if trends:
        parts.append("<h2>Trend by month</h2>")
        parts.append(trends)
    for key, by_label in report.grouped_series.items():
        fact, measure, column = key.split("|", 2)
        phrase = result.words.get(f"{fact}|{measure}", measure)
        parts.append(f"<h2>{esc(phrase.capitalize())} by {esc(_word({'column': column}))}</h2>")
        parts.append(_multi_trend_svg(by_label, f"{phrase.capitalize()} by month, largest {_word({'column': column})} groups"))
        years = sorted({p.year for points in by_label.values() for p in points})
        if len(years) >= 2:
            first, last = years[-2], years[-1]
            rows = []
            for label, points in sorted(by_label.items(), key=lambda item: -sum(p.value for p in item[1])):
                a = sum(p.value for p in points if p.year == first)
                b = sum(p.value for p in points if p.year == last)
                pct = (b - a) / abs(a) if a else None
                rows.append(f'<tr><td>{esc(_label(label))}</td><td>{esc(_full(a))}</td><td>{esc(_full(b))}</td><td class="{_direction(b - a)}">{esc(_full(b - a))}</td><td class="{_direction(b - a)}">{esc(_pct(pct))}</td></tr>')
            parts.append(f"<table><tr><th>{esc(_word({'column': column}))}</th><th>{first}</th><th>{last}</th><th>Change</th><th>%</th></tr>{''.join(rows)}</table>")
    return parts


def _top_movers_report(report: Any) -> list[str]:
    from .reports import top_movers

    result = report.sweep
    parts = []
    kpis = _kpi_cards(result)
    if kpis:
        parts.append("<h2>Headline movements</h2>")
        parts.append(kpis)
    blocks = top_movers(result, top=report.spec.top)
    if not blocks:
        parts.append('<div class="card">No grouping could be measured within the budget.</div>')
    for phrase, grouping, risers, fallers in blocks:
        parts.append(f'<div class="card"><h3>{esc(phrase.capitalize())}, by {esc(grouping)}</h3>')
        parts.append('<div class="grid2">' + _bars_svg([(_label(m.group), m.delta, m.pct) for m in risers], "Rose most", _UP) + _bars_svg([(_label(m.group), m.delta, m.pct) for m in fallers], "Fell most", _DOWN) + "</div>")
        rows = "".join(f'<tr><td>{esc(_label(m.group))}</td><td>{esc(_full(m.before_value))}</td><td>{esc(_full(m.after_value))}</td><td class="{_direction(m.delta)}">{esc(_full(m.delta))}</td><td class="{_direction(m.delta)}">{esc(_pct(m.pct))}</td></tr>' for m in risers + fallers)
        parts.append(f"<table><tr><th>{esc(grouping)}</th><th>Before</th><th>After</th><th>Change</th><th>%</th></tr>{rows}</table></div>")
    return parts


def render_report(report: Any) -> str:
    """A report of one kind as an HTML fragment with its own styles."""
    spec, result = report.spec, report.sweep
    kind = "semantic model" if result.kind == "semantic_model" else result.kind
    parts = [f"<style>{_STYLE}</style>", '<div class="wm">']
    parts.append(f"<h1>{esc(spec.title.capitalize())}: {esc(result.source)}</h1>")
    parts.append(f'<div class="sub">{esc(kind)}, {esc(f"{result.years[0]} to {result.years[-1]}" if result.years else "")}: {len(result.ledger)} figures measured by the source in {report.queries} of {result.budget} queries{esc(_seconds(report.elapsed))}. {_badge(result)}</div>')
    parts.append(_reading(spec))
    if spec.kind == "trend":
        parts.extend(_trend_report(report))
    elif spec.kind == "top_movers":
        parts.extend(_top_movers_report(report))
    elif spec.kind == "root_cause":
        if result.findings:
            for index, finding in enumerate(result.findings, start=1):
                parts.append(_finding(result, finding, index, others_as_tables=3))
        else:
            parts.append('<div class="card">The movement could not be measured; see the notes.</div>')
        trends = _trends(result)
        if trends:
            parts.append("<h2>The measure by month</h2>")
            parts.append(trends)
    else:
        kpis = _kpi_cards(result)
        if kpis:
            parts.append("<h2>Headline movements</h2>")
            parts.append(kpis)
        trends = _trends(result)
        if trends:
            parts.append("<h2>Trends</h2>")
            parts.append(trends)
        if result.findings:
            parts.append("<h2>Driver analysis</h2>")
            for index, finding in enumerate(result.findings, start=1):
                parts.append(_finding(result, finding, index))
        else:
            parts.append('<div class="card">No material movement to decompose.</div>')
    if result.mismatches:
        parts.append("<h2>Figures that did not recompute</h2>")
        parts.append("".join(f'<div class="note">{esc(text)}</div>' for text in result.mismatches[:10]))
    if result.notes:
        parts.append("<h2>Notes</h2>")
        parts.append("".join(f'<div class="caption">{esc(note)}</div>' for note in result.notes))
    parts.append('<div class="foot">Every figure on this page was computed by the source with the query shown under its finding, and recomputed by an independent per-period query. No language model was involved in producing the numbers.</div>')
    parts.append("</div>")
    return "\n".join(parts)


def document_report(report: Any) -> str:
    return _page(f"{report.spec.title.capitalize()}: {report.sweep.source}", render_report(report))


# --------------------------------------------------------------------------- #
# The Monday Morning Brief
# --------------------------------------------------------------------------- #


def _spark_svg(values: Sequence[float], width: int = 160, height: int = 36) -> str:
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    step = (width - 4) / (len(values) - 1)
    points = [(2 + i * step, 2 + (high - v) * (height - 4) / span) for i, v in enumerate(values)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
    last_x, last_y = points[-1]
    return f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" aria-hidden="true"><path d="{path}" fill="none" stroke="#94a3b8" stroke-width="1.5"/><circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3" fill="#2563eb"/></svg>'


def _weekly_svg(metric: Any, title: str) -> str:
    """The last 52 weeks with the briefed week marked, level shifts as dashed lines, and the expectation for the week as a hollow marker."""
    weeks = list(metric.weeks)
    target = metric.target
    if target is None or len(weeks) < 2:
        return ""
    index = next((i for i, w in enumerate(weeks) if w.start == target.start), len(weeks) - 1)
    window = weeks[max(0, index - 51) : index + 1]
    width, height, left, right, top, bottom = 640, 240, 64, 16, 28, 40
    values = [w.value for w in window]
    expected = metric.context.get("expected")
    low, high = min(0.0, min(values)), max(values + ([expected] if expected else []))
    ticks = _nice_ticks(low, high)
    low, high = min(ticks[0], low), max(ticks[-1], high)
    span = high - low or 1.0

    def x(i: int) -> float:
        return left + i * (width - left - right) / max(1, len(window) - 1)

    def y(v: float) -> float:
        return top + (high - v) * (height - top - bottom) / span

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    parts.append(f'<text x="{left}" y="16" font-size="12" fill="#374151" font-weight="600">{esc(title)}</text>')
    for tick in ticks:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" font-size="10" fill="#6b7280" text-anchor="end">{esc(_compact(tick))}</text>')
    step = max(1, round(len(window) / 6))
    for i, w in enumerate(window):
        if i % step == 0:
            parts.append(f'<text x="{x(i):.1f}" y="{height - bottom + 14}" font-size="10" fill="#6b7280" text-anchor="middle">{esc(_short_date(w.start))}</text>')
    starts = {w.start: i for i, w in enumerate(window)}
    for point in metric.change_points:
        if point.start in starts:
            px = x(starts[point.start])
            parts.append(f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{top}" y2="{height - bottom}" stroke="#f59e0b" stroke-dasharray="4 3"><title>{esc(f"level shift: {_full(point.before)} to {_full(point.after)}")}</title></line>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    parts.append(f'<path d="{path}" fill="none" stroke="#2563eb" stroke-width="2" stroke-linejoin="round"/>')
    for i, w in enumerate(window):
        parts.append(f'<circle cx="{x(i):.1f}" cy="{y(w.value):.1f}" r="2.2" fill="#2563eb"><title>{esc(f"{w.label}: {_full(w.value)}")}</title></circle>')
    last = len(window) - 1
    if expected:
        parts.append(f'<circle cx="{x(last):.1f}" cy="{y(expected):.1f}" r="5" fill="none" stroke="#6b7280" stroke-width="1.5"><title>{esc(f"expected: {_full(expected)}")}</title></circle>')
    parts.append(f'<circle cx="{x(last):.1f}" cy="{y(values[-1]):.1f}" r="5" fill="#dc2626"><title>{esc(f"this week: {_full(values[-1])}")}</title></circle>')
    parts.append(f'<text x="{width - right}" y="{height - 4}" font-size="10" fill="#9ca3af" text-anchor="end">red: this week; hollow: the expectation; dashed: a level shift</text>')
    parts.append("</svg>")
    return "".join(parts)


def _short_date(iso: str) -> str:
    import datetime as _dt

    day = _dt.date.fromisoformat(iso)
    return f"{day.day} {calendar.month_abbr[day.month]} {str(day.year)[2:]}"


def _weekday_svg(shares: Any, title: str) -> str:
    if not shares or not any(v for v, _u in shares.values()):
        return ""
    width, height, left, top, bottom = 640, 170, 40, 26, 30
    slot = (width - left - 10) / 7
    high = max(max(this, usual) for this, usual in shares.values()) or 1.0

    def y(v: float) -> float:
        return top + (high - v) * (height - top - bottom) / high

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    parts.append(f'<text x="{left}" y="16" font-size="12" fill="#374151" font-weight="600">{esc(title)}</text>')
    for weekday in range(7):
        this, usual = shares.get(weekday, (0.0, 0.0))
        cx = left + slot * weekday + slot / 2
        parts.append(f'<rect x="{cx - 22:.1f}" y="{y(usual):.1f}" width="20" height="{max(0.0, y(0) - y(usual)):.1f}" fill="#cbd5e1"><title>{esc(f"usual: {usual:.0%}")}</title></rect>')
        parts.append(f'<rect x="{cx + 2:.1f}" y="{y(this):.1f}" width="20" height="{max(0.0, y(0) - y(this)):.1f}" fill="#2563eb"><title>{esc(f"this week: {this:.0%}")}</title></rect>')
        parts.append(f'<text x="{cx:.1f}" y="{height - bottom + 14}" font-size="10" fill="#6b7280" text-anchor="middle">{calendar.day_abbr[weekday]}</text>')
    parts.append(f'<text x="{width - 10}" y="{height - 4}" font-size="10" fill="#9ca3af" text-anchor="end">blue: this week; grey: the usual share over the twelve weeks before</text>')
    parts.append("</svg>")
    return "".join(parts)


def _context_table(metric: Any) -> str:
    c = metric.context
    rows = [("This week", c.get("value"), None)]
    if c.get("previous") is not None:
        rows.append(("Week before", c["previous"], c.get("wow_pct")))
    if c.get("avg4") is not None:
        rows.append(("Average of the last 4 weeks", c["avg4"], c.get("vs_avg4_pct")))
    if c.get("avg13") is not None:
        rows.append(("Average of the last 13 weeks", c["avg13"], c.get("vs_avg13_pct")))
    if c.get("prior_year") is not None:
        rows.append(("Same week last year", c["prior_year"], c.get("yoy_pct")))
    if c.get("expected"):
        rows.append(("Expected for this week", c["expected"], c.get("vs_expected_pct")))
    body = "".join(f"<tr><td>{esc(label)}</td><td>{esc(_full(value)) if value is not None else ''}</td><td class=\"{_direction(pct or 0) if pct is not None else 'flat'}\">{esc(_pct(pct)) if pct is not None else ''}</td></tr>" for label, value, pct in rows)
    return f"<table><tr><th>Context</th><th>Value</th><th>This week against it</th></tr>{body}</table>"


def _metric_section(metric: Any, index: int) -> str:
    from .sweep import _concentration_sentence

    c = metric.context
    chips = []
    if c.get("verdict"):
        chips.append(f'<span class="chip">{esc(c["verdict"])}</span>')
    if c.get("trend_note"):
        chips.append(f'<span class="chip">{esc(c["trend_note"])}</span>')
    if c.get("rank_note"):
        chips.append(f'<span class="chip">{esc(c["rank_note"])}</span>')
    if c.get("streak_note"):
        chips.append(f'<span class="chip">{esc(c["streak_note"])}</span>')
    parts = [f'<div class="card"><h3>{index}. {esc(metric.name.capitalize())}</h3>', f'<div class="story">{esc(metric.headline)}</div>', "".join(chips)]
    from .brief import _week_label

    parts.append(_weekly_svg(metric, f"{metric.name.capitalize()} by week"))
    for point in metric.change_points:
        parts.append(f'<div class="note">Level shift the {esc(_week_label(point.start))}: the weekly average went from {esc(_full(point.before))} to {esc(_full(point.after))} ({esc(_pct(point.pct))}).</div>')
    parts.append(_context_table(metric))
    finding = metric.finding
    if finding is not None:
        parts.append("<h3 style=\"margin-top:12px\">Why it moved</h3>")
        for flag in finding.flags:
            parts.append(f'<div class="note">{esc(flag)}</div>')
        for text in metric.explanations:
            parts.append(f'<div class="story">{esc(text)}</div>')
        best = finding.best
        if best is not None and best.groups:
            parts.append('<div class="grid2">' + _waterfall_svg(best, f"Week over week, by {_word(best.path)}") + _scatter_svg(best, f"Who moved more than their size, by {_word(best.path)}") + "</div>")
            parts.append(_groups_table(best))
            lead = finding.lead_drill
            if lead is not None:
                parts.append(_groups_table(lead))
        others = finding.decompositions[1:]
        if others:
            parts.append('<div class="caption">Other groupings tried: ' + "".join(f'<span class="chip">by <b>{esc(_word(d.path))}</b>: {esc(_concentration_words(d.concentration))}</span>' for d in others) + "</div>")
    prior = metric.prior_year_finding
    if prior is not None and prior.best is not None and prior.best.groups:
        parts.append(f'<div class="story"><b>Against the same week last year, by {esc(_word(prior.best.path))}:</b> {esc(_concentration_sentence(prior.best))}</div>')
    if metric.pattern or any(v for v, _u in metric.weekday_shares.values()):
        parts.append("<h3 style=\"margin-top:12px\">Pattern within the week</h3>")
        if metric.pattern:
            parts.append(f'<div class="story">{esc(metric.pattern)}</div>')
        parts.append(_weekday_svg(metric.weekday_shares, "Share of the week by day"))
    for note in metric.notes:
        parts.append(f'<div class="caption">{esc(note)}</div>')
    if finding is not None:
        m = finding.movement
        parts.append("<details><summary>Queries behind these figures</summary>" + "".join(f'<div class="caption">{esc(label)}</div><pre>{esc(text)}</pre>' for label, text in (("Measured by", m.query), ("Recomputed by (this week)", m.verification.get("after", "")), ("Recomputed by (week before)", m.verification.get("before", ""))) if text) + "</details>")
    parts.append("</div>")
    return "".join(parts)


def render_brief(brief: Any) -> str:
    """The brief as an HTML fragment with its own styles: one look, the watch list, every metric, what moves together."""
    kind = "semantic model" if brief.kind == "semantic_model" else brief.kind
    parts = [f"<style>{_STYLE}</style>", '<div class="wm">']
    parts.append(f"<h1>{esc(brief.title)}</h1>")
    badge = (
        f'<span class="badge {"ok" if not brief.mismatches else "warn"}">{brief.recomputed} figures recomputed, {len(brief.mismatches)} mismatch{"es" if len(brief.mismatches) != 1 else ""}</span>'
        if brief.verified and brief.recomputed
        else '<span class="badge muted">nothing to recompute</span>' if brief.verified else '<span class="badge muted">not recomputed</span>'
    )
    parts.append(f'<div class="sub">Week of {esc(brief.week_label)}. {esc(kind)}, {len(brief.metrics)} metric(s), {brief.queries} of {brief.budget} queries{esc(_seconds(brief.elapsed))}. {badge}</div>')
    if brief.metrics:
        parts.append("<h2>In one look</h2>")
        cards = []
        for metric in brief.metrics:
            c = metric.context
            values = [w.value for w in metric.weeks[-13:]]
            chips = "".join(f'<span class="{_direction(v)}">{esc(label)} {esc(_pct(v))}</span> ' for label, v in (("wow", c.get("wow_pct")), ("yoy", c.get("yoy_pct")), ("vs 13w", c.get("vs_avg13_pct"))) if v is not None)
            cards.append(f'<div class="kpi"><div class="t" title="{esc(metric.name)}">{esc(metric.name.capitalize())}</div><div class="v">{esc(_compact(c.get("value", 0.0)))}</div><div class="d">{chips}</div>{_spark_svg(values)}</div>')
        parts.append(f'<div class="kpis">{"".join(cards)}</div>')
    if brief.watch:
        parts.append("<h2>Watch</h2>")
        parts.append("".join(f'<div class="note">{esc(text)}</div>' for text in brief.watch))
    for index, metric in enumerate(brief.metrics, start=1):
        parts.append(_metric_section(metric, index))
    if brief.comovement:
        parts.append("<h2>Across metrics</h2>")
        parts.append("".join(f'<div class="story">{esc(text)}</div>' for text in brief.comovement))
    if brief.mismatches:
        parts.append("<h2>Figures that did not recompute</h2>")
        parts.append("".join(f'<div class="note">{esc(text)}</div>' for text in brief.mismatches[:10]))
    if brief.notes:
        parts.append("<h2>Notes</h2>")
        parts.append("".join(f'<div class="caption">{esc(note)}</div>' for note in brief.notes))
    parts.append('<div class="foot">Every figure was computed by the source and, where it is reported as a movement, recomputed by an independent query. The seasonal expectation, level shifts, patterns and associations are computed from the source\'s own weekly history; no language model wrote a number.</div>')
    parts.append("</div>")
    return "\n".join(parts)


def document_brief(brief: Any) -> str:
    return _page(brief.title, render_brief(brief))
