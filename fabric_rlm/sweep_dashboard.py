"""The dashboard of a sweep: what moved, the trends, and for every finding the bridge of drivers, the variance chart, the driver scatter and the Pareto view.

Everything is inline HTML, CSS and SVG. Nothing is loaded from the
network, so the page renders inside a Fabric notebook (``displayHTML``),
as a file saved to the lakehouse, or in an email. Every number on the page
is a figure from the ledger, computed by the source; the queries behind
each finding are on the page.

The notation follows IBCS (International Business Communication
Standards, version 2): a title states the message, the current period is
dark and the period compared with is grey, a rise is green and a fall is
red (a colour pair that colour-blind readers can tell apart, with the sign
and a hatch as a second cue), time runs left to right, categories are
listed as horizontal bars, every chart of one measure on a page shares
its scale, and the unit sits in the title. Text and marks keep at least
the WCAG AA contrast against white.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import re
from collections.abc import Sequence
from html import escape as esc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .sweep import Decomposition, Point, Sweep, SweepFinding

__all__ = ["document", "document_brief", "document_report", "render", "render_brief", "render_report"]

# --------------------------------------------------------------------------- #
# Palette and type: WCAG AA on white, and a rise/fall pair colour-blind readers can tell apart
# --------------------------------------------------------------------------- #

_INK = "#111827"  # text, 17:1
_INK2 = "#374151"  # secondary text, 10:1
_MUTED = "#4b5563"  # captions and axis labels, 7.6:1
_GRID = "#e5e7eb"
_RULE = "#9ca3af"
_AC = "#1f2937"  # this period (IBCS: actual), near black
_PY = "#8b95a3"  # the period compared with (IBCS: previous), grey, 3.3:1 for marks
_OLD = "#c3c9d2"  # an older period, dashed
_UP = "#00875a"  # a rise: bluish green, 4.6:1
_DOWN = "#c9500a"  # a fall: vermilion, 4.5:1; hatched as a second cue
_UP_TEXT = "#0b6e4f"
_DOWN_TEXT = "#a63d00"
_MARK = "#0072b2"  # the one highlight: this week, this point
_SHIFT = "#6d28d9"  # a level shift
_SERIES = ("#0072b2", "#e69f00", "#009e73", "#cc79a7", "#d55e00", "#56b4e9", "#111827", "#8b95a3")  # Okabe and Ito, with a dash pattern each
_DASHES = ("", "7 4", "2 3", "9 3 2 3", "", "7 4", "2 3", "9 3 2 3")
_HATCH = '<defs><pattern id="wmhatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#ffffff" stroke-width="2.2" stroke-opacity="0.6"/></pattern></defs>'

_STYLE = f"""
.wm{{font:14px/1.5 "Segoe UI",-apple-system,BlinkMacSystemFont,Inter,Roboto,"Helvetica Neue",Arial,sans-serif;color:{_INK};max-width:1180px;margin:0 auto;padding:20px 18px 28px;background:#fff;font-variant-numeric:tabular-nums}}
.wm *{{box-sizing:border-box}}
.wm h1{{font-size:24px;line-height:1.25;margin:0 0 6px;font-weight:650;letter-spacing:-0.01em}}
.wm h2{{font-size:17px;margin:26px 0 10px;font-weight:650;color:{_INK}}}
.wm h3{{font-size:15px;line-height:1.4;margin:0 0 6px;font-weight:650}}
.wm .sub{{color:{_MUTED};margin-bottom:14px}}
.wm a{{color:#1d4ed8}}
.wm a:focus-visible,.wm summary:focus-visible{{outline:2px solid #1d4ed8;outline-offset:2px;border-radius:3px}}
.wm .badge{{display:inline-block;border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600;vertical-align:middle}}
.wm .badge.ok{{background:#dcfce7;color:#14532d}}
.wm .badge.warn{{background:#fee2e2;color:#7f1d1d}}
.wm .badge.muted{{background:#f3f4f6;color:{_INK2}}}
.wm .kpis{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}}
.wm .kpi{{border:1px solid {_GRID};border-radius:10px;padding:10px 12px;background:#fff}}
.wm .kpi .t{{font-size:12.5px;color:{_MUTED};white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.wm .kpi .v{{font-size:22px;font-weight:650;margin:2px 0}}
.wm .kpi .d{{font-size:13px}}
.wm .up{{color:{_UP_TEXT}}}.wm .down{{color:{_DOWN_TEXT}}}.wm .flat{{color:{_MUTED}}}
.wm .card{{border:1px solid {_GRID};border-radius:12px;padding:14px 16px;margin:0 0 16px;background:#fff}}
.wm .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:820px){{.wm .grid2{{grid-template-columns:1fr}}}}
.wm .chart{{width:100%;height:auto;display:block}}
.wm .caption{{font-size:12.5px;color:{_MUTED};margin:2px 0 8px}}
.wm table{{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}}
.wm th,.wm td{{padding:5px 8px;border-bottom:1px solid #eef0f3;text-align:right;white-space:nowrap}}
.wm th{{color:{_INK2};font-weight:600;font-size:12.5px}}
.wm th:first-child,.wm td:first-child{{text-align:left;white-space:normal}}
.wm .bar{{display:inline-block;height:8px;border-radius:2px;background:{_UP};vertical-align:middle;margin-right:6px}}
.wm .bar.neg{{background:{_DOWN}}}
.wm .chip{{display:inline-block;border:1px solid #d1d5db;border-radius:999px;padding:1px 9px;margin:2px 4px 2px 0;font-size:12.5px;color:{_INK2};background:#f9fafb}}
.wm .chip b{{font-weight:600}}
.wm .note{{color:{_INK};background:#f9fafb;border-left:3px solid #6b7280;border-radius:6px;padding:7px 10px;margin:6px 0;font-size:13px}}
.wm .note.warn{{border-left-color:{_DOWN};background:#fff7f2}}
.wm .note.shift{{border-left-color:{_SHIFT};background:#f8f5ff}}
.wm .story{{margin:6px 0 10px}}
.wm details{{margin-top:8px}}
.wm summary{{cursor:pointer;color:#1d4ed8;font-size:13px;font-weight:600}}
.wm details.aside,.wm details.about{{border:1px solid {_GRID};border-radius:12px;padding:10px 16px;margin:18px 0 0;background:#fafbfc}}
.wm details.aside summary,.wm details.about summary{{font-size:14px}}
.wm details.aside .card{{margin-top:12px}}
.wm dl{{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:8px 0;font-size:13px}}
.wm dt{{color:{_MUTED}}}
.wm dd{{margin:0;word-break:break-word}}
.wm pre{{white-space:pre-wrap;word-break:break-word;font-size:12px;background:#f8fafc;padding:8px;border-radius:8px;overflow:auto;margin:6px 0;color:{_INK}}}
.wm .foot{{color:{_MUTED};font-size:12.5px;margin-top:18px}}
.wm .legend{{font-size:12.5px;color:{_MUTED};margin:0 0 8px}}
.wm .legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:-1px;margin:0 4px 0 10px}}
.wm svg text{{font-family:inherit}}
@media print{{.wm{{max-width:none;padding:0}}.wm .card{{break-inside:avoid}}}}
"""


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


def _signed(value: float) -> str:
    return f"{'+' if value > 0 else ''}{_compact(value)}"


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
        "fragmented": "fragmented, too fine to explain",
        "none": "nothing to split",
    }.get(concentration, concentration)


def _scale_word(values: Sequence[float]) -> str:
    high = max((abs(v) for v in values), default=0.0)
    if high >= 1e9:
        return "in billions"
    if high >= 1e6:
        return "in millions"
    if high >= 1e4:
        return "in thousands"
    return ""


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
# SVG pieces
# --------------------------------------------------------------------------- #


def _open(title: str, width: int, height: int, *, left: int = 60, subtitle: str = "") -> list[str]:
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">', _HATCH]
    parts.append(f'<text x="{left}" y="17" font-size="13" fill="{_INK}" font-weight="600">{esc(title)}</text>')
    if subtitle:
        parts.append(f'<text x="{left}" y="32" font-size="11.5" fill="{_MUTED}">{esc(subtitle)}</text>')
    return parts


def _bar(x: float, y: float, w: float, h: float, color: str, title: str, *, hatch: bool = False, rx: float = 1.5) -> str:
    box = f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0.6, w):.1f}" height="{max(0.6, h):.1f}" fill="{color}" rx="{rx}"><title>{esc(title)}</title></rect>'
    if hatch:
        box += f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0.6, w):.1f}" height="{max(0.6, h):.1f}" fill="url(#wmhatch)" rx="{rx}" pointer-events="none"/>'
    return box


def _gridlines(parts: list[str], ticks: Sequence[float], y: Any, left: int, right_x: float, fmt: Any) -> None:
    for tick in ticks:
        parts.append(f'<line x1="{left}" x2="{right_x:.1f}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="{_GRID}"/>')
        parts.append(f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" font-size="11" fill="{_MUTED}" text-anchor="end">{esc(fmt(tick))}</text>')


# --------------------------------------------------------------------------- #
# SVG charts
# --------------------------------------------------------------------------- #


def _trend_svg(points: Sequence["Point"], title: str, aggregate: str) -> str:
    """One line per year over the twelve months: the latest year dark, the year before grey, older years light and dashed, each labelled at its end."""
    by_year: dict[int, dict[int, float]] = {}
    for p in points:
        by_year.setdefault(p.year, {})[p.month] = p.value
    years = sorted(by_year)
    if not years:
        return ""
    width, height, left, right, top, bottom = 640, 240, 60, 46, 42, 34
    values = [v for year in by_year.values() for v in year.values()]
    low, high = min(0.0, min(values)), max(values)
    ticks = _nice_ticks(low, high)
    low, high = min(ticks[0], low), max(ticks[-1], high)
    span = high - low or 1.0

    def x(month: int) -> float:
        return left + (month - 1) * (width - left - right) / 11

    def y(value: float) -> float:
        return top + (high - value) * (height - top - bottom) / span

    subtitle = ", ".join(p for p in (_scale_word(values), "monthly average" if aggregate == "avg" else "", f"{years[0]} to {years[-1]}" if len(years) > 1 else str(years[0])) if p)
    parts = _open(title, width, height, left=left, subtitle=subtitle)
    _gridlines(parts, ticks, y, left, width - right, _compact)
    for month in range(1, 13):
        parts.append(f'<text x="{x(month):.1f}" y="{height - 14}" font-size="11" fill="{_MUTED}" text-anchor="middle">{calendar.month_abbr[month]}</text>')
    for index, year in enumerate(years):
        series = by_year[year]
        age = len(years) - 1 - index
        color, stroke, dash = (_AC, 2.4, "") if age == 0 else (_PY, 1.9, "") if age == 1 else (_OLD, 1.5, "5 4")
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        coordinates = [(x(m), y(series[m])) for m in range(1, 13) if m in series]
        if len(coordinates) >= 2:
            path = " ".join(f"{'M' if i == 0 else 'L'}{cx:.1f},{cy:.1f}" for i, (cx, cy) in enumerate(coordinates))
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linejoin="round"{dash_attr}/>')
        for m in range(1, 13):
            if m in series:
                parts.append(f'<circle cx="{x(m):.1f}" cy="{y(series[m]):.1f}" r="{2.8 if age == 0 else 2.2}" fill="{color}"><title>{esc(f"{calendar.month_abbr[m]} {year}: {_full(series[m])}")}</title></circle>')
        if coordinates:
            end_x, end_y = coordinates[-1]
            parts.append(f'<text x="{end_x + 6:.1f}" y="{end_y + 4:.1f}" font-size="11" font-weight="{600 if age == 0 else 400}" fill="{_INK if age == 0 else _MUTED}">{year}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _waterfall_svg(d: "Decomposition", title: str) -> str:
    """The bridge: the period before (grey), the largest groups in the direction of the change, the largest opposite ones, the rest, the period after (dark)."""
    parent = d.parent
    same = [g for g in d.groups if g.delta * parent.delta > 0][:5]
    opposite = [g for g in d.groups if g.delta * parent.delta < 0][:2]
    steps = [(_label(g.group), g.delta) for g in same] + [(_label(g.group), g.delta) for g in opposite]
    other = parent.delta - sum(delta for _n, delta in steps)
    if abs(other) > 1e-9 * max(1.0, abs(parent.delta)):
        steps.append(("all other", other))
    from .sweep import _period_label

    before_name, after_name = _period_label(parent.comparison.before), _period_label(parent.comparison.after)
    bars: list[tuple[str, float, float, str]] = [(before_name, 0.0, parent.before_value, _PY)]
    running = parent.before_value
    for name, delta in steps:
        bars.append((name, running, running + delta, _UP if delta > 0 else _DOWN if delta < 0 else _RULE))
        running += delta
    bars.append((after_name, 0.0, parent.after_value, _AC))
    width, height, left, right, top, bottom = 640, 290, 60, 12, 44, 60
    extremes = [v for _n, a, b, _c in bars for v in (a, b)]
    low, high = min(extremes), max(extremes)
    if low > 0 and low < 0.25 * high:
        low = 0.0
    pad = (high - low) * 0.1 or 1.0
    low, high = (min(0.0, low - pad) if low <= 0 else low - pad), high + pad
    ticks = [t for t in _nice_ticks(low, high) if low <= t <= high]
    span = high - low or 1.0
    slot = (width - left - right) / max(1, len(bars))
    bar_width = min(46.0, slot * 0.66)

    def y(value: float) -> float:
        return top + (high - value) * (height - top - bottom) / span

    parts = _open(title, width, height, left=left, subtitle=", ".join(p for p in (_scale_word(extremes), f"{before_name} to {after_name}") if p))
    _gridlines(parts, ticks, y, left, width - right, _compact)
    previous_top: float | None = None
    for index, (name, start, end, color) in enumerate(bars):
        cx = left + slot * index + slot / 2
        total = color in (_PY, _AC)
        y0, y1 = y(max(start, end)), y(min(start, end))
        if total:
            y0, y1 = y(end), y(max(low, 0.0) if low <= 0 <= high else low)
        parts.append(_bar(cx - bar_width / 2, y0, bar_width, y1 - y0, color, f"{name}: {_full(end) if total else _full(end - start)}", hatch=color == _DOWN))
        if previous_top is not None and not total:
            parts.append(f'<line x1="{cx - slot + bar_width / 2:.1f}" x2="{cx - bar_width / 2:.1f}" y1="{previous_top:.1f}" y2="{previous_top:.1f}" stroke="{_RULE}" stroke-dasharray="3 3"/>')
        previous_top = y(end)
        value_text = _compact(end) if total else _signed(end - start)
        fill = _INK if total else _UP_TEXT if end - start > 0 else _DOWN_TEXT
        parts.append(f'<text x="{cx:.1f}" y="{y0 - 5:.1f}" font-size="11" font-weight="600" fill="{fill}" text-anchor="middle">{esc(value_text)}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{height - bottom + 15}" font-size="11" fill="{_INK2}" text-anchor="middle">{esc(_short(name, 14))}</text>')
    parts.append(f'<text x="{left}" y="{height - 6}" font-size="11" fill="{_MUTED}">grey: {esc(before_name)}; dark: {esc(after_name)}; green up, hatched red down: the change each group contributed</text>')
    parts.append("</svg>")
    return "".join(parts)


def _variance_svg(d: "Decomposition", title: str) -> str:
    """The IBCS variance chart: for each group the level before (grey) and after (dark), the absolute change, and the relative change as a pin."""
    rows = list(d.groups)[:8]
    if not rows:
        return ""
    width, left, top, row_h = 640, 136, 46, 24
    height = top + row_h * len(rows) + 14
    level_x, level_w = left, 178
    abs_x, abs_w = level_x + level_w + 26, 142
    pct_x, pct_w = abs_x + abs_w + 26, width - (abs_x + abs_w + 26) - 10
    max_level = max(max(abs(g.before_value), abs(g.after_value)) for g in rows) or 1.0
    max_abs = max(abs(g.delta) for g in rows) or 1.0
    pcts = [g.pct for g in rows]
    max_pct = max((abs(p) for p in pcts if p is not None), default=0.0) or 1.0
    from .sweep import _period_label

    before_name, after_name = _period_label(d.parent.comparison.before), _period_label(d.parent.comparison.after)
    parts = _open(title, width, height, left=left, subtitle=_scale_word([g.after_value for g in rows] + [g.before_value for g in rows]))
    for x0, w, text in ((level_x, level_w, f"{_short(before_name, 12)} (grey) and {_short(after_name, 12)} (dark)"), (abs_x, abs_w, "change"), (pct_x, pct_w, "change in %")):
        parts.append(f'<text x="{x0 + w / 2:.1f}" y="{top - 8}" font-size="11" fill="{_MUTED}" text-anchor="middle">{esc(text)}</text>')
    abs_zero, pct_zero = abs_x + abs_w / 2, pct_x + pct_w / 2
    parts.append(f'<line x1="{abs_zero:.1f}" x2="{abs_zero:.1f}" y1="{top - 2}" y2="{height - 12}" stroke="{_RULE}"/>')
    parts.append(f'<line x1="{pct_zero:.1f}" x2="{pct_zero:.1f}" y1="{top - 2}" y2="{height - 12}" stroke="{_RULE}"/>')
    for i, g in enumerate(rows):
        y = top + i * row_h
        name = _label(g.group)
        parts.append(f'<text x="{left - 8}" y="{y + 15}" font-size="11.5" fill="{_INK2}" text-anchor="end">{esc(_short(name, 20))}</text>')
        before_len = level_w * 0.82 * abs(g.before_value) / max_level
        after_len = level_w * 0.82 * abs(g.after_value) / max_level
        parts.append(_bar(level_x, y + 4, before_len, 6, _PY, f"{name}, {before_name}: {_full(g.before_value)}"))
        parts.append(_bar(level_x, y + 12, after_len, 7, _AC, f"{name}, {after_name}: {_full(g.after_value)}"))
        parts.append(f'<text x="{level_x + max(before_len, after_len) + 5:.1f}" y="{y + 18}" font-size="10.5" fill="{_MUTED}">{esc(_compact(g.after_value))}</text>')
        length = (abs_w / 2 - 4) * abs(g.delta) / max_abs
        up = g.delta > 0
        parts.append(_bar(abs_zero if up else abs_zero - length, y + 6, length, 12, _UP if up else _DOWN if g.delta < 0 else _RULE, f"{name}: {_signed(g.delta)}", hatch=g.delta < 0))
        anchor_x = abs_zero + length + 4 if up else abs_zero - length - 4
        parts.append(f'<text x="{anchor_x:.1f}" y="{y + 16}" font-size="10.5" font-weight="600" fill="{_UP_TEXT if up else _DOWN_TEXT if g.delta < 0 else _MUTED}" text-anchor="{"start" if up else "end"}">{esc(_signed(g.delta))}</text>')
        if g.pct is not None:
            pin = (pct_w / 2 - 6) * min(1.0, abs(g.pct) / max_pct)
            px = pct_zero + pin if g.pct > 0 else pct_zero - pin
            color = _UP if g.pct > 0 else _DOWN
            parts.append(f'<line x1="{pct_zero:.1f}" x2="{px:.1f}" y1="{y + 12}" y2="{y + 12}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<circle cx="{px:.1f}" cy="{y + 12}" r="4" fill="{color}"><title>{esc(f"{name}: {_pct(g.pct)}")}</title></circle>')
            parts.append(f'<text x="{px + (7 if g.pct > 0 else -7):.1f}" y="{y + 16}" font-size="10.5" font-weight="600" fill="{_UP_TEXT if g.pct > 0 else _DOWN_TEXT}" text-anchor="{"start" if g.pct > 0 else "end"}">{esc(f"{g.pct:+.0%}")}</text>')
        else:
            parts.append(f'<text x="{pct_zero + 7:.1f}" y="{y + 16}" font-size="10.5" fill="{_MUTED}">new</text>')
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
    width, height, left, right, top, bottom = 640, 280, 60, 16, 42, 46
    x_high = max(1.0, max(p[1] for p in points) * 1.12)
    y_low, y_high = min(0.0, min(p[2] for p in points) * 1.12), max(1.0, max(p[2] for p in points) * 1.12)

    def x(value: float) -> float:
        return left + value * (width - left - right) / x_high

    def y(value: float) -> float:
        return top + (y_high - value) * (height - top - bottom) / (y_high - y_low or 1.0)

    parts = _open(title, width, height, left=left, subtitle="share of the base (across) against share of the change (up); above the dashed line a group moved more than its size")
    _gridlines(parts, [t for t in _nice_ticks(y_low, y_high) if y_low <= t <= y_high], y, left, width - right, lambda t: f"{t:.0%}")
    for tick in [t for t in _nice_ticks(0.0, x_high) if 0 <= t <= x_high]:
        parts.append(f'<text x="{x(tick):.1f}" y="{height - bottom + 15}" font-size="11" fill="{_MUTED}" text-anchor="middle">{tick:.0%}</text>')
    diagonal_end = min(x_high, y_high)
    parts.append(f'<line x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(diagonal_end):.1f}" y2="{y(diagonal_end):.1f}" stroke="{_RULE}" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{x(diagonal_end) - 4:.1f}" y="{y(diagonal_end) - 6:.1f}" font-size="11" fill="{_MUTED}" text-anchor="end">moved in step with size</text>')
    if y_low < 0:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="{_RULE}"/>')
    for name, base, change in sorted(points, key=lambda p: -abs(p[2])):
        color = _UP if change > base + 0.1 else _DOWN if change < 0 else _PY
        parts.append(f'<circle cx="{x(base):.1f}" cy="{y(change):.1f}" r="6" fill="{color}" stroke="#fff" stroke-width="1.5"><title>{esc(f"{name}: {change:.0%} of the change on {base:.0%} of the base")}</title></circle>')
        parts.append(f'<text x="{x(base) + 9:.1f}" y="{y(change) + 4:.1f}" font-size="11" fill="{_INK2}">{esc(_short(name, 18))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _pareto_svg(pareto: dict[str, Any], word: str, title: str) -> str:
    """Cumulative share of the base and of the change against the share of groups, with the 80% line and where each curve crosses it."""
    n = int(pareto["n"])
    curve_base, curve_change = pareto["curve_base"], pareto["curve_change"]
    if n < 2 or not curve_base:
        return ""
    width, height, left, right, top, bottom = 640, 240, 56, 84, 42, 36

    def x(rank: int, total: int) -> float:
        return left + rank * (width - left - right) / max(1, total)

    def y(share: float) -> float:
        return top + (1.0 - share) * (height - top - bottom)

    parts = _open(title, width, height, left=left, subtitle=f"{word} groups ranked largest first; across: share of the {n:,} groups, up: cumulative share")
    _gridlines(parts, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], y, left, width - right, lambda t: f"{t:.0%}")
    for share in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        parts.append(f'<text x="{x(int(round(share * n)), n):.1f}" y="{height - bottom + 15}" font-size="11" fill="{_MUTED}" text-anchor="middle">{share:.0%}</text>')
    parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(0.8):.1f}" y2="{y(0.8):.1f}" stroke="{_SHIFT}" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{width - right + 4}" y="{y(0.8) + 4:.1f}" font-size="11" fill="{_SHIFT}">80%</text>')
    for curve, color, name, k in ((curve_base, _PY, "base", pareto["k_base"]), (curve_change, _AC, "change", pareto["k_change"])):
        total = len(curve)
        path = " ".join(f"{'M' if i == 0 else 'L'}{x(i, total):.1f},{y(c):.1f}" for i, c in enumerate([0.0] + list(curve)))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round"><title>{esc(name)}</title></path>')
        if 1 <= k <= total:
            parts.append(f'<circle cx="{x(k, total):.1f}" cy="{y(curve[k - 1]):.1f}" r="5" fill="{color}" stroke="#fff" stroke-width="1.5"><title>{esc(f"{k} of {total} carry {curve[k - 1]:.0%} of the {name}")}</title></circle>')
        parts.append(f'<text x="{width - right + 4}" y="{y(curve[-1]) + 4 + (12 if name == "change" and abs(curve[-1] - curve_base[-1]) < 0.06 else 0):.1f}" font-size="11" font-weight="{600 if name == "change" else 400}" fill="{_INK if name == "change" else _MUTED}">{name}: {k} of {total}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _multi_trend_svg(series_by_label: "dict[Any, Sequence[Point]]", title: str) -> str:
    """One line per group over the months covered, the largest groups first, each with its own colour and dash."""
    present = sorted({(p.year, p.month) for points in series_by_label.values() for p in points})
    if len(present) < 2:
        return ""
    months: list[tuple[int, int]] = []  # every month from the first to the last, so gaps keep their width
    year, month = present[0]
    while (year, month) <= present[-1]:
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    width, height, left, right, top, bottom = 640, 270, 60, 16, 42, 48
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

    parts = _open(title, width, height, left=left, subtitle=_scale_word(values))
    _gridlines(parts, ticks, y, left, width - right, _compact)
    step = max(1, round(len(months) / 6))
    for i, key in enumerate(months):
        if i % step == 0 or (i == len(months) - 1 and (len(months) - 1) % step >= step / 2):
            parts.append(f'<text x="{x(key):.1f}" y="{height - bottom + 15}" font-size="11" fill="{_MUTED}" text-anchor="middle">{calendar.month_abbr[key[1]]} {key[0]}</text>')
    labels = sorted(series_by_label, key=lambda k: -sum(p.value for p in series_by_label[k]))
    for i, label in enumerate(labels[:8]):
        color, dash = _SERIES[i % len(_SERIES)], _DASHES[i % len(_DASHES)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        points = sorted(series_by_label[label], key=lambda p: (p.year, p.month))
        coordinates = [(x((p.year, p.month)), y(p.value)) for p in points]
        if len(coordinates) >= 2:
            path = " ".join(f"{'M' if j == 0 else 'L'}{cx:.1f},{cy:.1f}" for j, (cx, cy) in enumerate(coordinates))
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"{dash_attr}><title>{esc(_label(label))}</title></path>')
        for p, (cx, cy) in zip(points, coordinates):
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.2" fill="{color}"><title>{esc(f"{_label(label)}, {calendar.month_abbr[p.month]} {p.year}: {_full(p.value)}")}</title></circle>')
        legend_y = height - 8
        legend_x = left + i * 78
        parts.append(f'<line x1="{legend_x}" x2="{legend_x + 16}" y1="{legend_y - 4}" y2="{legend_y - 4}" stroke="{color}" stroke-width="2.5"{dash_attr}/><text x="{legend_x + 20}" y="{legend_y}" font-size="11" fill="{_INK2}">{esc(_short(_label(label), 10))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _bars_svg(items: Sequence[tuple[str, float, float | None]], title: str, color: str) -> str:
    """Horizontal bars for a ranked list: label, change, and the relative change when there is a base."""
    if not items:
        return f'<div class="caption">{esc(title)}: none</div>'
    width, left, right, row_height, top = 640, 150, 96, 24, 30
    height = top + row_height * len(items) + 10
    biggest = max(abs(v) for _l, v, _p in items) or 1.0
    parts = _open(title, width, height, left=0)
    for i, (label, value, pct) in enumerate(items):
        y = top + i * row_height
        length = (width - left - right) * abs(value) / biggest
        parts.append(f'<text x="{left - 8}" y="{y + 15}" font-size="11.5" fill="{_INK2}" text-anchor="end">{esc(_short(label, 22))}</text>')
        parts.append(_bar(left, y + 4, length, 15, color, f"{label}: {_full(value)}", hatch=value < 0))
        text = _signed(value) + (f" ({pct:+.0%})" if pct is not None else "")
        parts.append(f'<text x="{left + max(1.0, length) + 6:.1f}" y="{y + 15}" font-size="11.5" fill="{_INK2}">{esc(text)}</text>')
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
            f'<td class="{_direction(g.delta)}">{_arrow(g.delta)} {esc(_full(g.delta))}</td><td class="{_direction(g.delta)}">{esc(_pct(g.pct)) if g.pct is not None else "new"}</td>'
            f'<td>{bar}{"" if change is None else f"{change:.0%}"}</td><td>{"" if base is None else f"{base:.0%}"}</td></tr>'
        )
    return f"<table><tr><th>{esc(_word(d.path))}</th><th>Before</th><th>After</th><th>Change</th><th>%</th><th>Share of change</th><th>Share of base</th></tr>{''.join(rows)}</table>"


def _flag_text(flag: str) -> str:
    """The flag without its machine prefix (``incomplete:``, ``coverage:``, ``volume:``, ``small base:``)."""
    return flag.split(": ", 1)[1] if ": " in flag and flag.split(": ", 1)[0] in {"incomplete", "coverage", "volume", "small base"} else flag


def _measure_cards(result: "Sweep") -> str:
    """One card per measure: its comparisons as rows, a thin period marked, nothing repeated."""
    groups: dict[str, list[Any]] = {}
    for m in result.ledger:
        if m.path is not None or f"{m.fact}|{m.measure}" in result.collapsed:
            continue
        groups.setdefault(f"{m.fact}|{m.measure}", []).append(m)
    cards = []
    for key, movements in groups.items():
        latest = next((m for m in movements if m.comparison.kind == "year"), movements[0])
        rows = []
        for m in movements:
            marker = ' <span class="badge warn" title="' + esc(next((_flag_text(f) for f in m.flags if f.startswith(("incomplete:", "coverage:"))), "")) + '">incomplete period</span>' if not m.trusted else ""
            rows.append(f'<div class="d {_direction(m.delta) if m.trusted else "flat"}">{_arrow(m.delta)} {esc(_pct(m.pct))} <span class="flat">{esc(m.comparison.label)}, {esc(_compact(m.before_value))} to {esc(_compact(m.after_value))}</span>{marker}</div>')
        cards.append(f'<div class="kpi"><div class="t" title="{esc(result.phrase(latest))}">{esc(result.phrase(latest).capitalize())}</div><div class="v">{esc(_compact(latest.after_value))} <span class="flat" style="font-size:12.5px;font-weight:400">{esc(_period_word(latest))}</span></div>{"".join(rows)}</div>')
    return f'<div class="kpis">{"".join(cards)}</div>' if cards else ""


def _period_word(m: Any) -> str:
    from .sweep import _period_label

    return _period_label(m.comparison.after)


def _takeaways(result: "Sweep") -> str:
    takeaways = result.takeaways()
    if not takeaways and not result.narrative():
        return ""
    parts = []
    if takeaways:
        parts.append(f"<h2>{'Three things to know' if len(takeaways) >= 3 else 'To know'}</h2>")
        items = []
        for t in takeaways:
            link = f' <a href="#{t.anchor}" style="text-decoration:none;font-size:12.5px">detail</a>' if t.anchor else ""
            items.append(f"<li>{esc(t.text)}{link}</li>")
        parts.append(f'<ol style="margin:0 0 8px 20px;padding:0;font-size:15px;line-height:1.55">{"".join(items)}</ol>')
    parts.append(f'<div class="card" style="padding:10px 16px"><div class="caption" style="margin:0 0 4px">The picture</div><div>{esc(result.narrative())}</div></div>')
    return "".join(parts)


def _aside_section(result: "Sweep", start_index: int) -> str:
    """The movements on incomplete periods, collapsed at the bottom: listed so nothing is hidden, read as coverage rather than business change."""
    aside = result.set_aside()
    untrusted = [f for f in result.findings if not f.trusted]
    if not aside and not untrusted:
        return ""
    count = max(len(aside), len(untrusted))
    items = "".join(f"<li>{esc(t.text)}</li>" for t in aside)
    cards = "".join(_finding(result, f, start_index + i) for i, f in enumerate(untrusted))
    return (
        f'<details class="aside"><summary>Set aside, not read as business change ({count})</summary>'
        f'<div class="caption" style="margin-top:8px">Each of these involves a period that is incomplete by its end date or by its row coverage. They are kept here so nothing is hidden; read them as coverage, not as business change.</div>'
        + (f'<ul style="margin:6px 0 4px 18px;padding:0;font-size:13px">{items}</ul>' if items else "")
        + cards
        + "</details>"
    )


def _notation() -> str:
    return (
        f'<div class="legend">Notation: <i style="background:{_AC}"></i>this period <i style="background:{_PY}"></i>the period compared with '
        f'<i style="background:{_UP}"></i>▲ rise <i style="background:{_DOWN}"></i>▼ fall (hatched) <i style="background:{_MARK}"></i>the point in question <i style="border-top:2px dashed {_SHIFT};height:0;border-radius:0"></i>a level shift or a guide</div>'
    )


def _about(result: Any, *, request: str, instructions: str, location: str, tables: Sequence[str], queries: str, statement: str, extra: Sequence[tuple[str, str]] = ()) -> str:
    """The reference, collapsed at the bottom: when the page was made, from what, with which request and instructions, and how it was checked."""
    from . import __version__

    when = _dt.datetime.now().astimezone()
    kind = "semantic model" if result.kind == "semantic_model" else str(result.kind)
    rows: list[tuple[str, str]] = [
        ("Generated", when.strftime("%A %d %B %Y, %H:%M ") + (when.tzname() or "")),
        ("Source", f"{result.source} ({kind})" + (f", {location}" if location else "")),
        ("Tables used", ", ".join(tables) if tables else "none"),
        ("Request", request or "none given; the defaults were used"),
        ("Queries", queries),
        ("Verification", statement),
        ("Made with", f"fabric-rlm {__version__}, deterministic rules over the source's own query engine; no language model wrote a number or a sentence"),
    ]
    rows.extend(extra)
    body = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows)
    given = f"<details><summary>Instructions given</summary><pre>{esc(instructions)}</pre></details>" if instructions else ""
    return f'<details class="about"><summary>About this page</summary><dl>{body}</dl>{given}</details>'


def _tables_of(result: "Sweep") -> list[str]:
    from .data_agent_review import _path_table

    facts: list[str] = []
    dimensions: list[str] = []
    for m in result.ledger:
        if m.fact not in facts:
            facts.append(m.fact)
        if m.path is not None:
            table = _path_table(m.path, m.fact)
            if table != m.fact and table not in dimensions:
                dimensions.append(table)
    return facts + sorted(dimensions)


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


def _driver_block(result: "Sweep", d: "Decomposition", word: str, heading: str, *, comparison_title: str) -> list[str]:
    """The charts for one decomposition: the bridge and the variance chart, then the scatter and the Pareto view when the grouping is fine enough."""
    from .sweep import _concentration_sentence

    parts = [f'<div class="story"><b>{esc(heading)}:</b> {esc(_concentration_sentence(d))}</div>']
    parts.append('<div class="grid2">' + _waterfall_svg(d, f"{comparison_title}, by {word}") + _variance_svg(d, f"Before and after, by {word}") + "</div>")
    parts.append(f'<div class="grid2">{_scatter_svg(d, f"Who moved more than their size, by {word}")}</div>')
    parts.append(_groups_table(d))
    return parts


def _pareto_block(result: "Sweep", finding: "SweepFinding", named: dict[int, str]) -> str:
    """The Pareto view of the finest grouping of a finding, when it has at least twelve members."""
    if not hasattr(result, "pareto_view"):
        return ""
    fine, view = result.pareto_view(finding)
    if fine is None or view is None:
        return ""
    word = named.get(id(fine.path), _word(fine.path))
    return f'<div class="story"><b>Pareto, by {esc(word)}:</b> {esc(result.pareto_sentence(fine))}</div><div class="grid2">{_pareto_svg(view, word, f"Pareto view, by {word}")}</div>'


def _finding(result: "Sweep", finding: "SweepFinding", index: int, *, others_as_tables: int = 0) -> str:
    from .sweep import _concentration_sentence, _qualified_words

    m = finding.movement
    best = finding.best
    named = _qualified_words([d.path for d in finding.decompositions], m.fact)
    subtitle = ", ".join(p for p in (result.phrase(m), _scale_word([m.before_value, m.after_value]), m.comparison.label) if p)
    parts = [f'<div class="card" id="{esc(finding.anchor)}"><h3>{index}. {esc(result.headline(m))}</h3><div class="caption">{esc(subtitle)}</div>']
    if not finding.trusted:
        parts.append('<div class="note warn"><b>Not read as business change.</b> One of the periods is incomplete; the figures below are shown for completeness.</div>')
    for flag in finding.flags:
        parts.append(f'<div class="note">{esc(_flag_text(flag))}</div>')
    if best is not None:
        word = named.get(id(best.path), _word(best.path))
        parts.extend(_driver_block(result, best, word, f"By {word}", comparison_title=f"What moved {m.comparison.label}"))
        lead = finding.lead_drill
        if lead is not None:
            parts.extend(_driver_block(result, lead, _word(lead.path), f"Within {_label(lead.parent.group)}, by {_word(lead.path)}", comparison_title=f"Within {_label(lead.parent.group)}"))
        elif finding.drill:
            tried = ", ".join(_word(d.path) for d in finding.drill)
            parts.append(f'<div class="story">Within {esc(_label(finding.drill[0].parent.group))}, nothing stands out by {esc(tried)}.</div>')
        parts.append(_pareto_block(result, finding, named))
        others = finding.decompositions[1:]
        for d in others[:others_as_tables]:
            parts.append(f'<div class="story"><b>By {esc(named.get(id(d.path), _word(d.path)))}:</b> {esc(_concentration_sentence(d))}</div>')
            parts.append(_groups_table(d))
        if others[others_as_tables:]:
            chips = "".join(f'<span class="chip">by <b>{esc(named.get(id(d.path), _word(d.path)))}</b>: {esc(_concentration_words(d.concentration))}</span>' for d in others[others_as_tables:])
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
    parts.extend(_recap_body(result, request="what moved", instructions=getattr(result, "instructions", "")))
    parts.append("</div>")
    return "\n".join(parts)


def _recap_body(result: "Sweep", *, request: str = "", instructions: str = "") -> list[str]:
    """The answer first, then the supporting detail: takeaways and the picture, the measures, the trends, the driver analysis; then the notes, the movements set aside, the reference."""
    parts = [_takeaways(result)]
    cards = _measure_cards(result)
    if cards:
        parts.append("<h2>Movements by measure</h2>")
        parts.append(cards)
    trends = _trends(result)
    if trends:
        parts.append("<h2>Trends</h2>")
        parts.append(trends)
    trusted = [f for f in result.findings if f.trusted]
    if trusted:
        parts.append("<h2>Driver analysis</h2>")
        parts.append(_notation())
        parts.append('<div class="caption">Each material movement is split by every grouping the joins reach. The bridge shows the change each group contributed between the two periods; the variance chart shows every group before and after with its change; the scatter shows whether a group moved more than its size (above the diagonal) or in proportion to it; the Pareto view, for groupings with many members, shows how few groups carry most of the base and most of the change.</div>')
        for index, finding in enumerate(trusted, start=1):
            parts.append(_finding(result, finding, index))
    elif not result.findings:
        steady = result.steady()
        parts.append('<div class="card">No movement of 5% or more on a complete period, so nothing to decompose.' + ('<ul style="margin:6px 0 0 18px">' + "".join(f"<li>{esc(result.headline(m))}</li>" for m in steady) + "</ul>" if steady else "") + "</div>")
    if result.mismatches:
        parts.append("<h2>Figures that did not recompute</h2>")
        parts.append("".join(f'<div class="note warn">{esc(text)}</div>' for text in result.mismatches[:10]))
    if result.notes:
        parts.append("<h2>Notes</h2>")
        parts.append("".join(f'<div class="caption">{esc(note)}</div>' for note in result.notes))
    parts.append(_aside_section(result, len(trusted) + 1))
    parts.append(_about(result, request=request, instructions=instructions, location=getattr(result, "location", ""), tables=_tables_of(result), queries=f"{result.queries} of a budget of {result.budget}{_seconds(result.elapsed)}", statement=result.verification_statement()))
    parts.append(f'<div class="foot">{esc(result.verification_statement())} No language model was involved in producing the numbers or the sentences.</div>')
    return parts


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
    kpis = _measure_cards(result)
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
                rows.append(f'<tr><td>{esc(_label(label))}</td><td>{esc(_full(a))}</td><td>{esc(_full(b))}</td><td class="{_direction(b - a)}">{_arrow(b - a)} {esc(_full(b - a))}</td><td class="{_direction(b - a)}">{esc(_pct(pct))}</td></tr>')
            parts.append(f"<table><tr><th>{esc(_word({'column': column}))}</th><th>{first}</th><th>{last}</th><th>Change</th><th>%</th></tr>{''.join(rows)}</table>")
    return parts


def _top_movers_report(report: Any) -> list[str]:
    from .reports import top_movers

    result = report.sweep
    parts = []
    kpis = _measure_cards(result)
    if kpis:
        parts.append("<h2>Headline movements</h2>")
        parts.append(kpis)
    blocks = top_movers(result, top=report.spec.top)
    if not blocks:
        parts.append('<div class="card">No grouping could be measured within the budget.</div>')
    for phrase, grouping, risers, fallers in blocks:
        parts.append(f'<div class="card"><h3>{esc(phrase.capitalize())}, by {esc(grouping)}</h3>')
        parts.append('<div class="grid2">' + _bars_svg([(_label(m.group), m.delta, m.pct) for m in risers], "Rose most", _UP) + _bars_svg([(_label(m.group), m.delta, m.pct) for m in fallers], "Fell most", _DOWN) + "</div>")
        rows = "".join(f'<tr><td>{esc(_label(m.group))}</td><td>{esc(_full(m.before_value))}</td><td>{esc(_full(m.after_value))}</td><td class="{_direction(m.delta)}">{_arrow(m.delta)} {esc(_full(m.delta))}</td><td class="{_direction(m.delta)}">{esc(_pct(m.pct))}</td></tr>' for m in risers + fallers)
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
    instructions = getattr(result, "instructions", "")
    if spec.kind == "trend":
        parts.extend(_trend_report(report))
    elif spec.kind == "top_movers":
        parts.extend(_top_movers_report(report))
    elif spec.kind == "root_cause":
        parts.append(_takeaways(result))
        trusted = [f for f in result.findings if f.trusted]
        if trusted:
            parts.append(_notation())
            for index, finding in enumerate(trusted, start=1):
                parts.append(_finding(result, finding, index, others_as_tables=3))
        elif not result.findings:
            parts.append('<div class="card">The movement could not be measured; see the notes.</div>')
        trends = _trends(result)
        if trends:
            parts.append("<h2>The measure by month</h2>")
            parts.append(trends)
    else:
        parts.extend(_recap_body(result, request=spec.request, instructions=instructions))
        parts.append("</div>")
        return "\n".join(parts)
    if result.mismatches:
        parts.append("<h2>Figures that did not recompute</h2>")
        parts.append("".join(f'<div class="note warn">{esc(text)}</div>' for text in result.mismatches[:10]))
    if result.notes:
        parts.append("<h2>Notes</h2>")
        parts.append("".join(f'<div class="caption">{esc(note)}</div>' for note in result.notes))
    if spec.kind == "root_cause":
        parts.append(_aside_section(result, len([f for f in result.findings if f.trusted]) + 1))
    parts.append(_about(result, request=spec.request, instructions=instructions, location=getattr(result, "location", ""), tables=_tables_of(result), queries=f"{report.queries} of a budget of {result.budget}{_seconds(report.elapsed)}", statement=result.verification_statement()))
    parts.append(f'<div class="foot">{esc(result.verification_statement())} No language model was involved in producing the numbers or the sentences.</div>')
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
    return f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" aria-hidden="true"><path d="{path}" fill="none" stroke="{_PY}" stroke-width="1.6"/><circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3.2" fill="{_MARK}"/></svg>'


def _weekly_svg(metric: Any, title: str) -> str:
    """The last 52 weeks with the briefed week marked, level shifts as dashed lines, and the expectation for the week as a hollow marker."""
    weeks = list(metric.weeks)
    target = metric.target
    if target is None or len(weeks) < 2:
        return ""
    index = next((i for i, w in enumerate(weeks) if w.start == target.start), len(weeks) - 1)
    window = weeks[max(0, index - 51) : index + 1]
    width, height, left, right, top, bottom = 640, 250, 60, 16, 42, 40
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

    parts = _open(title, width, height, left=left, subtitle=", ".join(p for p in (_scale_word(values), f"{len(window)} weeks to the {target.label}") if p))
    _gridlines(parts, ticks, y, left, width - right, _compact)
    step = max(1, round(len(window) / 6))
    for i, w in enumerate(window):
        if i % step == 0:
            parts.append(f'<text x="{x(i):.1f}" y="{height - bottom + 15}" font-size="11" fill="{_MUTED}" text-anchor="middle">{esc(_short_date(w.start))}</text>')
    starts = {w.start: i for i, w in enumerate(window)}
    for point in metric.change_points:
        if point.start in starts:
            px = x(starts[point.start])
            parts.append(f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{top}" y2="{height - bottom}" stroke="{_SHIFT}" stroke-dasharray="5 4"><title>{esc(f"level shift: {_full(point.before)} to {_full(point.after)}")}</title></line>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    parts.append(f'<path d="{path}" fill="none" stroke="{_AC}" stroke-width="2" stroke-linejoin="round"/>')
    for i, w in enumerate(window):
        parts.append(f'<circle cx="{x(i):.1f}" cy="{y(w.value):.1f}" r="2.2" fill="{_AC}"><title>{esc(f"{w.label}: {_full(w.value)}")}</title></circle>')
    last = len(window) - 1
    if expected:
        parts.append(f'<circle cx="{x(last):.1f}" cy="{y(expected):.1f}" r="5.5" fill="none" stroke="{_MUTED}" stroke-width="1.8"><title>{esc(f"expected: {_full(expected)}")}</title></circle>')
    parts.append(f'<circle cx="{x(last):.1f}" cy="{y(values[-1]):.1f}" r="5.5" fill="{_MARK}" stroke="#fff" stroke-width="1.5"><title>{esc(f"this week: {_full(values[-1])}")}</title></circle>')
    parts.append(f'<text x="{width - right}" y="{height - 6}" font-size="11" fill="{_MUTED}" text-anchor="end">blue: this week; hollow: the expectation; dashed: a level shift</text>')
    parts.append("</svg>")
    return "".join(parts)


def _short_date(iso: str) -> str:
    day = _dt.date.fromisoformat(iso)
    return f"{day.day} {calendar.month_abbr[day.month]} {str(day.year)[2:]}"


def _lines_svg(series_by_label: "dict[str, Sequence[Any]]", title: str, *, markers: Sequence[str] = (), percent: bool = False) -> str:
    """Lines by date for two or three weekly series (each a sequence with ``start`` and ``value``), with dashed markers at the dates given."""
    starts = sorted({w.start for points in series_by_label.values() for w in points})
    if len(starts) < 2:
        return ""
    width, height, left, right, top, bottom = 640, 250, 60, 16, 42, 46
    values = [w.value for points in series_by_label.values() for w in points]
    low, high = min(0.0, min(values)), max(values)
    ticks = _nice_ticks(low, high)
    low, high = min(ticks[0], low), max(ticks[-1], high)
    span = high - low or 1.0
    index = {s: i for i, s in enumerate(starts)}

    def x(start: str) -> float:
        return left + index[start] * (width - left - right) / max(1, len(starts) - 1)

    def y(v: float) -> float:
        return top + (high - v) * (height - top - bottom) / span

    parts = _open(title, width, height, left=left, subtitle="" if percent else _scale_word(values))
    _gridlines(parts, ticks, y, left, width - right, (lambda t: f"{t:.0%}") if percent else _compact)
    step = max(1, round(len(starts) / 6))
    for i, s in enumerate(starts):
        if i % step == 0:
            parts.append(f'<text x="{x(s):.1f}" y="{height - bottom + 15}" font-size="11" fill="{_MUTED}" text-anchor="middle">{esc(_short_date(s))}</text>')
    for marker in markers:
        if marker in index:
            parts.append(f'<line x1="{x(marker):.1f}" x2="{x(marker):.1f}" y1="{top}" y2="{height - bottom}" stroke="{_SHIFT}" stroke-dasharray="5 4"/>')
    for i, (label, points) in enumerate(series_by_label.items()):
        color, dash = _SERIES[i % len(_SERIES)], _DASHES[i % len(_DASHES)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        ordered = sorted(points, key=lambda w: w.start)
        path = " ".join(f"{'M' if j == 0 else 'L'}{x(w.start):.1f},{y(w.value):.1f}" for j, w in enumerate(ordered))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"{dash_attr}><title>{esc(label)}</title></path>')
        legend_x = left + i * 170
        parts.append(f'<line x1="{legend_x}" x2="{legend_x + 16}" y1="{height - 5}" y2="{height - 5}" stroke="{color}" stroke-width="2.5"{dash_attr}/><text x="{legend_x + 20}" y="{height - 1}" font-size="11" fill="{_INK2}">{esc(_short(label, 24))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _growth_svg(points: Sequence[Any], title: str) -> str:
    """Growth accounting by week: retained, new and resurrected stacked upward, churned downward."""
    if not points:
        return ""
    width, height, left, right, top, bottom = 640, 250, 60, 16, 42, 46
    high = max((p.new + p.retained + p.resurrected) for p in points) or 1.0
    low = -max(p.churned for p in points)
    ticks = _nice_ticks(low, high)
    low, high = min(ticks[0], low), max(ticks[-1], high)
    span = high - low or 1.0
    slot = (width - left - right) / max(1, len(points))
    bar = max(2.0, slot * 0.7)

    def y(v: float) -> float:
        return top + (high - v) * (height - top - bottom) / span

    parts = _open(title, width, height, left=left)
    _gridlines(parts, ticks, y, left, width - right, _compact)
    parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="{_RULE}"/>')
    step = max(1, round(len(points) / 6))
    palette = ((_PY, "retained"), (_MARK, "new"), (_UP, "resurrected"), (_DOWN, "churned"))
    for i, p in enumerate(points):
        cx = left + slot * i + slot / 2
        base = 0.0
        for value, (color, label) in zip((p.retained, p.new, p.resurrected), palette[:3]):
            if value:
                parts.append(_bar(cx - bar / 2, y(base + value), bar, y(base) - y(base + value), color, f"{label} {p.start}: {value:,}", rx=0))
                base += value
        if p.churned:
            parts.append(_bar(cx - bar / 2, y(0), bar, y(-p.churned) - y(0), _DOWN, f"churned {p.start}: {p.churned:,}", hatch=True, rx=0))
        if i % step == 0:
            parts.append(f'<text x="{cx:.1f}" y="{height - bottom + 15}" font-size="11" fill="{_MUTED}" text-anchor="middle">{esc(_short_date(p.start))}</text>')
    for i, (color, label) in enumerate(palette):
        parts.append(f'<rect x="{left + i * 96}" y="{height - 12}" width="10" height="10" fill="{color}" rx="2"/><text x="{left + i * 96 + 14}" y="{height - 3}" font-size="11" fill="{_INK2}">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _kpi_charts(metric: Any) -> str:
    extra = metric.extra or {}
    if metric.kind in {"new", "active", "retained", "resurrected", "churned"} and extra.get("growth"):
        return _growth_svg(extra["growth"], f"{extra['entity'].name.capitalize()} by week: who stayed, who joined, who came back, who left")
    if metric.kind == "ratio" and extra.get("numerator"):
        return _lines_svg({extra.get("numerator_name", "numerator"): extra["numerator"][-52:]}, f"{extra.get('numerator_name', 'numerator').capitalize()} by week") + _lines_svg({extra.get("denominator_name", "denominator"): extra["denominator"][-52:]}, f"{extra.get('denominator_name', 'denominator').capitalize()} by week")
    if metric.kind == "crossing" and extra.get("left"):
        return _lines_svg({extra.get("left_name", "left"): extra["left"][-52:], extra.get("right_name", "right"): extra["right"][-52:]}, "The two series by week; dashed: a crossing", markers=[when for when, _side in extra.get("crossings", ())])
    if metric.kind == "concentration" and extra.get("shares"):
        return _lines_svg({"share of the top groups": extra["shares"]}, "Share held by the top groups, by week", percent=True) if hasattr(extra["shares"][0], "value") else ""
    return ""


def _weekday_svg(shares: Any, title: str) -> str:
    if not shares or not any(v for v, _u in shares.values()):
        return ""
    width, height, left, top, bottom = 640, 180, 40, 42, 32
    slot = (width - left - 10) / 7
    high = max(max(this, usual) for this, usual in shares.values()) or 1.0

    def y(v: float) -> float:
        return top + (high - v) * (height - top - bottom) / high

    parts = _open(title, width, height, left=left, subtitle="grey: the usual share over the twelve weeks before; dark: this week")
    for weekday in range(7):
        this, usual = shares.get(weekday, (0.0, 0.0))
        cx = left + slot * weekday + slot / 2
        parts.append(_bar(cx - 22, y(usual), 20, y(0) - y(usual), _PY, f"usual: {usual:.0%}"))
        parts.append(_bar(cx + 2, y(this), 20, y(0) - y(this), _AC, f"this week: {this:.0%}"))
        parts.append(f'<text x="{cx:.1f}" y="{height - bottom + 15}" font-size="11" fill="{_MUTED}" text-anchor="middle">{calendar.day_abbr[weekday]}</text>')
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
    body = "".join(f"<tr><td>{esc(label)}</td><td>{esc(_full(value)) if value is not None else ''}</td><td class=\"{_direction(pct or 0) if pct is not None else 'flat'}\">{(_arrow(pct) + ' ' + esc(_pct(pct))) if pct is not None else ''}</td></tr>" for label, value, pct in rows)
    return f"<table><tr><th>Context</th><th>Value</th><th>This week against it</th></tr>{body}</table>"


def _metric_section(metric: Any, index: int) -> str:
    from .brief import _week_label
    from .sweep import _concentration_sentence

    c = metric.context
    chips = []
    for key in ("verdict", "trend_note", "rank_note", "streak_note"):
        if c.get(key):
            chips.append(f'<span class="chip">{esc(c[key])}</span>')
    parts = [f'<div class="card"><h3>{index}. {esc(metric.name.capitalize())}</h3>', f'<div class="story">{esc(metric.headline)}</div>']
    if metric.definition:
        parts.append(f'<div class="caption">Definition: {esc(metric.definition)}.</div>')
    parts.append("".join(chips))
    if metric.kind == "concentration":
        parts.append(_lines_svg({metric.name: list(metric.weeks)[-52:]}, f"{metric.name.capitalize()} by week", percent=True))
    else:
        parts.append(_weekly_svg(metric, f"{metric.name.capitalize()} by week"))
    parts.append(_kpi_charts(metric))
    if metric.kind != "measure":
        for text in metric.explanations:
            parts.append(f'<div class="story">{esc(text)}</div>')
    for point in metric.change_points:
        parts.append(f'<div class="note shift">Level shift the {esc(_week_label(point.start))}: the weekly average went from {esc(_full(point.before))} to {esc(_full(point.after))} ({esc(_pct(point.pct))}).</div>')
    parts.append(_context_table(metric))
    finding = metric.finding
    if finding is not None:
        parts.append("<h3 style=\"margin-top:12px\">Why it moved</h3>")
        for flag in finding.flags:
            parts.append(f'<div class="note">{esc(_flag_text(flag))}</div>')
        for text in metric.explanations:
            parts.append(f'<div class="story">{esc(text)}</div>')
        best = finding.best
        if best is not None and best.groups:
            against = "Week over week" if finding.movement.comparison.kind == "week" else "Against the same week last year"
            word = _word(best.path)
            parts.append('<div class="grid2">' + _waterfall_svg(best, f"{against}, by {word}") + _variance_svg(best, f"Before and after, by {word}") + "</div>")
            parts.append(f'<div class="grid2">{_scatter_svg(best, f"Who moved more than their size, by {word}")}</div>')
            parts.append(_groups_table(best))
            lead = finding.lead_drill
            if lead is not None:
                parts.append(_groups_table(lead))
            if metric.sweep is not None:
                parts.append(_pareto_block(metric.sweep, finding, {}))
        others = finding.decompositions[1:]
        if others:
            parts.append('<div class="caption">Other groupings tried: ' + "".join(f'<span class="chip">by <b>{esc(_word(d.path))}</b>: {esc(_concentration_words(d.concentration))}</span>' for d in others) + "</div>")
    prior = metric.prior_year_finding
    if prior is not None and prior is not metric.finding and prior.best is not None and prior.best.groups:
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


_RATE_NAME = re.compile(r"(rate|share|pct|percent|ratio|yield|margin)", re.IGNORECASE)


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
    narrative = brief.narrative()
    if narrative:
        parts.append(f'<div class="card" style="padding:10px 16px"><div class="caption" style="margin:0 0 4px">The week in short</div><div style="font-size:15px">{esc(narrative)}</div></div>')
    if brief.metrics:
        parts.append("<h2>In one look</h2>")
        cards = []
        for metric in brief.metrics:
            c = metric.context
            values = [w.value for w in metric.weeks[-13:]]
            chips = "".join(f'<span class="{_direction(v)}">{esc(label)} {_arrow(v)} {esc(_pct(v))}</span> ' for label, v in (("wow", c.get("wow_pct")), ("yoy", c.get("yoy_pct")), ("vs 13w", c.get("vs_avg13_pct"))) if v is not None)
            shown = c.get("value", 0.0)
            as_share = metric.kind == "concentration" or (metric.kind == "ratio" and abs(shown) < 1 and _RATE_NAME.search(metric.name))  # a share or a rate reads as a percentage, as the headline says it
            cards.append(f'<div class="kpi"><div class="t" title="{esc(metric.name)}">{esc(metric.name.capitalize())}</div><div class="v">{esc(f"{shown:.1%}" if as_share else _compact(shown))}</div><div class="d">{chips}</div>{_spark_svg(values)}</div>')
        parts.append(f'<div class="kpis">{"".join(cards)}</div>')
    if brief.watch:
        parts.append("<h2>Watch</h2>")
        parts.append("".join(f'<div class="note">{esc(text)}</div>' for text in brief.watch))
    if brief.definitions or brief.entity_choice:
        parts.append("<h2>Definitions</h2>")
        if brief.entity_choice:
            parts.append(f'<div class="caption">Entities: {esc(brief.entity_choice)}.</div>')
        parts.append("<ul style=\"margin:4px 0 8px 18px;padding:0;font-size:13px\">" + "".join(f"<li><b>{esc(name)}</b>: {esc(definition)}.</li>" for name, definition in brief.definitions) + "</ul>")
    if brief.metrics:
        parts.append(_notation())
    for index, metric in enumerate(brief.metrics, start=1):
        parts.append(_metric_section(metric, index))
    if brief.comovement:
        parts.append("<h2>Across metrics</h2>")
        parts.append("".join(f'<div class="story">{esc(text)}</div>' for text in brief.comovement))
    if brief.mismatches:
        parts.append("<h2>Figures that did not recompute</h2>")
        parts.append("".join(f'<div class="note warn">{esc(text)}</div>' for text in brief.mismatches[:10]))
    if brief.notes:
        parts.append("<h2>Notes</h2>")
        parts.append("".join(f'<div class="caption">{esc(note)}</div>' for note in brief.notes))
    tables: list[str] = []
    for metric in brief.metrics:
        if metric.fact and metric.fact not in tables:
            tables.append(metric.fact)
        if metric.sweep is not None:
            for table in _tables_of(metric.sweep):
                if table not in tables:
                    tables.append(table)
    parts.append(_about(brief, request=getattr(brief, "request", ""), instructions=getattr(brief, "instructions", ""), location=getattr(brief, "location", ""), tables=tables, queries=f"{brief.queries} of a budget of {brief.budget}{_seconds(brief.elapsed)}", statement=brief.verification_statement(), extra=(("Week", brief.week_label),)))
    parts.append(f'<div class="foot">{esc(brief.verification_statement())} The seasonal expectation, level shifts, patterns and associations are computed from the source\'s own weekly history. No language model wrote a number or a sentence.</div>')
    parts.append("</div>")
    return "\n".join(parts)


def document_brief(brief: Any) -> str:
    return _page(brief.title, render_brief(brief))
