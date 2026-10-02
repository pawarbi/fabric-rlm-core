"""``RLM.report``: a report from a question about a semantic model.

Four steps, and the model only does the two that need judgement:

1. Plan (model). A run reads the question and the semantic model and returns a typed plan: what kind of
   analysis (a change between two periods, or a trend), the model measures, the grouping columns and the
   periods. The plan is checked on the host before anything is computed: the names must exist in the model
   and the periods must be complete. A bad plan goes back to the run with the reason.
2. Compute (code). Every figure is computed here with DAX this module writes: each group's value in both
   periods, the totals, shares, counts, the period series. Nothing the narrative may cite comes from the model.
3. Narrative (model). A second run gets only the computed findings and the question, chooses the sections and
   the chart for each (or follows the sections the user asked for), and writes the words. Every number in the
   text must match a computed figure; a number that does not goes back to the run.
4. Render (code). HTML with charts drawn here, or Markdown with the same sections and tables. The plan, how the
   question was read, and how the numbers were checked are printed on the page.
"""

from __future__ import annotations

import html as _html
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .semantic_checks import period_bounds, period_key

BLOCKS = ("kpis", "gains_losses", "before_after", "mix", "trend", "period_changes", "table", "text")
VISUAL_BLOCKS = ("gains_losses", "before_after", "mix", "trend", "period_changes")
_GUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_COLUMN = re.compile(r"^\s*'?(?P<table>[^'\[]+?)'?\s*\[(?P<column>[^\]]+)\]\s*$")
_TOP = 5


# ----------------------------------------------------------------------------------------------- names
def _records(frame: Any) -> list[dict[str, Any]]:
    if frame is None:
        return []
    if hasattr(frame, "to_dict"):
        return list(frame.to_dict(orient="records"))
    return [dict(r) for r in frame]


def column_ref(text: str) -> tuple[str, str] | None:
    """``Table[Column]`` or ``'Table'[Column]`` as (table, column); None when it is not a column reference."""
    m = _COLUMN.match(str(text or ""))
    return (m.group("table").strip(), m.group("column").strip()) if m else None


def _quote(table: str, column: str) -> str:
    return "'" + table.replace("'", "''") + "'[" + column + "]"


def measure_name(text: str) -> str:
    return str(text or "").strip().strip("[]").strip()


@dataclass
class ModelNames:
    """The measures and columns a plan may use, from the model's own metadata."""

    measures: dict[str, dict[str, Any]]          # lower-case name -> record
    columns: dict[tuple[str, str], dict[str, Any]]  # (table, column) lower-case -> record

    @classmethod
    def read(cls, model: Any) -> "ModelNames":
        measures: dict[str, dict[str, Any]] = {}
        for r in _records(model.measures()):
            name = r.get("Measure Name") or r.get("measure_name") or r.get("Name")
            if name:
                measures[str(name).lower()] = r
        columns: dict[tuple[str, str], dict[str, Any]] = {}
        for r in _records(model.columns()):
            t = r.get("Table Name") or r.get("table_name")
            c = r.get("Column Name") or r.get("column_name")
            if t and c:
                columns[(str(t).lower(), str(c).lower())] = r
        return cls(measures, columns)

    def measure(self, text: str) -> str | None:
        r = self.measures.get(measure_name(text).lower())
        return (r.get("Measure Name") or r.get("measure_name") or r.get("Name")) if r else None

    def column(self, text: str) -> tuple[str, str] | None:
        ref = column_ref(text)
        if not ref:
            return None
        r = self.columns.get((ref[0].lower(), ref[1].lower()))
        if not r:
            return None
        return (str(r.get("Table Name") or r.get("table_name")), str(r.get("Column Name") or r.get("column_name")))

    def expression(self, measure: str) -> str:
        r = self.measures.get(measure.lower()) or {}
        return str(r.get("Measure Expression") or r.get("measure_expression") or "").strip()

    def suggest_measures(self, text: str, n: int = 8) -> list[str]:
        words = set(re.findall(r"[a-z]+", text.lower()))
        scored = sorted(self.measures.values(), key=lambda r: -len(words & set(re.findall(r"[a-z]+", str(r.get("Measure Name", "")).lower()))))
        return [str(r.get("Measure Name")) for r in scored[:n]]


# ----------------------------------------------------------------------------------------------- plan
@dataclass
class ReportPlan:
    kind: str                      # change | trend
    measures: list[str]
    group_by: list[tuple[str, str]]
    periods: list[str]             # change: [before, after]; trend: every period in order; [] with compare_measures
    compare_measures: list[str]    # [before measure, after measure] when the comparison lives in measures
    filters: list[tuple[tuple[str, str], list[Any]]]
    reading: str
    grain: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "measures": self.measures, "group_by": [_quote(*g) for g in self.group_by],
                "periods": self.periods, "compare_measures": self.compare_measures,
                "filters": {_quote(*c): v for c, v in self.filters}, "reading": self.reading, "grain": self.grain}


PLAN_OUTPUTS: dict[str, type] = {"kind": str, "measures": list, "group_by": list, "periods": list,
                                 "compare_measures": list, "filters": dict, "reading": str}

PLAN_RULES = """Plan rules:
- kind is "change" (how something moved between two periods, by group) or "trend" (a measure over consecutive periods).
- measures: the model's own measures, exact names, never a sum of a raw column when a measure exists. For a change,
  one measure; for a trend, one or two.
- group_by: up to two columns as 'Table'[Column] for a change (the dimension the question asks about); [] for a trend.
- periods: labels like 2018-Q2, 2018-07 or 2018. For a change, [before, after]. For a trend, every period in order.
  Use complete periods only; check them with {name}.period_coverage("<measure>", grain="month" or "quarter").
  "Latest" or "last" means the latest complete period, never a partial one. A period marked unknown (the first
  period, with no history to judge it by) counts as complete: keep it unless its row count is clearly short.
- compare_measures: when the model holds the comparison in two measures (for example [Last Year Sales] and
  [This Year Sales]), give them as [before, after] and leave periods empty. Otherwise [].
- filters: {{"'Table'[Column]": value or [values]}} only when the question narrows the data; otherwise {{}}.
- reading: two or three sentences on how you read the question, including anything in it the data contradicts
  (for example a question asking which group grew most when none grew)."""


def check_plan(payload: Mapping[str, Any], names: ModelNames, coverage: Any) -> ReportPlan:
    """The plan as a ReportPlan, or AssertionError with what to fix."""
    kind = str(payload.get("kind", "")).strip().lower()
    assert kind in ("change", "trend"), 'kind must be "change" or "trend".'
    measures: list[str] = []
    for m in payload.get("measures") or []:
        found = names.measure(str(m))
        assert found, (f"{m!r} is not a measure in the model. Measures that look relevant: "
                       f"{', '.join(names.suggest_measures(str(m)))}.")
        measures.append(found)
    compare = []
    for m in payload.get("compare_measures") or []:
        found = names.measure(str(m))
        assert found, f"compare_measures: {m!r} is not a measure in the model."
        compare.append(found)
    assert not compare or len(compare) == 2, "compare_measures needs exactly two measures: [before, after]."
    if compare and not measures:
        measures = [compare[1]]
    assert measures, "Give at least one measure."
    group_by = []
    for g in payload.get("group_by") or []:
        found = names.column(str(g))
        assert found, f"{g!r} is not a column in the model; write it as 'Table'[Column] with the exact names."
        group_by.append(found)
    filters = []
    for col, value in (payload.get("filters") or {}).items():
        found = names.column(str(col))
        assert found, f"filter {col!r} is not a column in the model."
        filters.append((found, list(value) if isinstance(value, (list, tuple)) else [value]))
    periods = [str(p).strip() for p in payload.get("periods") or []]
    grain = ""
    if kind == "change":
        assert len(measures) == 1, "A change report uses one measure."
        assert len(group_by) <= 2, "Use at most two group_by columns."
        assert group_by, "A change report needs a group_by column: the dimension the question asks about."
        if compare:
            assert not periods, "With compare_measures, leave periods empty; the measures hold the periods."
        else:
            assert len(periods) == 2, "A change report needs periods = [before, after]."
            grains = set()
            for p in periods:
                try:
                    grains.add(period_bounds(p)[2])
                except ValueError as exc:
                    raise AssertionError(f"{p!r} is not a period label; write 2018-Q2, 2018-07 or 2018.") from exc
            assert len(grains) == 1, "Both periods must be the same kind (two quarters, two months or two years)."
            grain = grains.pop()
            assert periods[0] != periods[1], "The two periods must differ."
            for p in periods:
                status = coverage(measures[0], grain).status(period_key(period_bounds(p)[0], grain))
                assert status in ("complete", "unknown"), (
                    f"{p} is {status} for [{measures[0]}]. Use complete periods; the latest complete {grain} is "
                    f"{coverage(measures[0], grain).latest_complete}.")
    else:
        assert not compare, "A trend does not use compare_measures."
        assert 1 <= len(measures) <= 2, "A trend uses one or two measures."
        assert len(periods) >= 3, "A trend needs at least three periods."
        grains = {period_bounds(p)[2] for p in periods}
        assert len(grains) == 1, "Every period in a trend must be the same kind."
        grain = grains.pop()
        keys = [period_key(period_bounds(p)[0], grain) for p in periods]
        assert keys == sorted(keys) and len(set(keys)) == len(keys), "List the trend periods in order, once each."
        for p in (periods[0], periods[-1]):
            status = coverage(measures[0], grain).status(period_key(period_bounds(p)[0], grain))
            assert status in ("complete", "unknown"), (
                f"{p} is {status} for [{measures[0]}]. End the trend at the latest complete {grain}, "
                f"{coverage(measures[0], grain).latest_complete}.")
        group_by = []
    reading = str(payload.get("reading") or "").strip()
    assert len(reading.split()) >= 8, "Write two or three sentences in reading on how you read the question."
    return ReportPlan(kind, measures, group_by, periods, compare, filters, reading, grain)


# ----------------------------------------------------------------------------------------------- compute
def _rows(result: Any) -> list[tuple[Any, ...]]:
    if result is None:
        return []
    if hasattr(result, "itertuples"):
        return [tuple(r) for r in result.itertuples(index=False)]
    return [tuple(r.values()) if isinstance(r, Mapping) else tuple(r) for r in result]


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


class DaxBackend:
    """The DAX this module writes, run through the semantic model handle."""

    def __init__(self, model: Any, date_column: str) -> None:
        self.model = model
        self.date_column = date_column
        self.queries: list[str] = []

    def _run(self, query: str) -> list[tuple[Any, ...]]:
        self.queries.append(query)
        return _rows(self.model.dax(query))

    def _period(self, label: str) -> str:
        start, end, _ = period_bounds(label)
        c = self.date_column
        return (f"{c} >= DATE({start.year},{start.month},{start.day}), "
                f"{c} < DATE({end.year},{end.month},{end.day})")

    @staticmethod
    def _filters(filters: Sequence[tuple[tuple[str, str], list[Any]]]) -> str:
        parts = []
        for (t, c), values in filters:
            vals = ", ".join(json.dumps(v) if isinstance(v, str) else str(v) for v in values)
            parts.append(f"TREATAS({{{vals}}}, {_quote(t, c)})")
        return "".join(", " + p for p in parts)

    def _pair(self, plan: ReportPlan) -> tuple[str, str]:
        f = self._filters(plan.filters)
        if plan.compare_measures:
            mb, ma = plan.compare_measures
            return f"CALCULATE([{mb}]{f})", f"CALCULATE([{ma}]{f})"
        m = plan.measures[0]
        return (f"CALCULATE([{m}], {self._period(plan.periods[0])}{f})",
                f"CALCULATE([{m}], {self._period(plan.periods[1])}{f})")

    def grouped(self, plan: ReportPlan, column: tuple[str, str]) -> list[tuple[str, float | None, float | None]]:
        b, a = self._pair(plan)
        rows = self._run(f'EVALUATE SUMMARIZECOLUMNS({_quote(*column)}, "b", {b}, "a", {a})')
        return [(_label(r[0]), _num(r[1]), _num(r[2])) for r in rows]

    def total(self, plan: ReportPlan) -> tuple[float | None, float | None]:
        b, a = self._pair(plan)
        r = self._run(f'EVALUATE ROW("b", {b}, "a", {a})')[0]
        return _num(r[0]), _num(r[1])

    def series(self, plan: ReportPlan, measure: str) -> list[tuple[str, float | None]]:
        f = self._filters(plan.filters)
        cols = ", ".join(f'"p{i}", CALCULATE([{measure}], {self._period(p)}{f})' for i, p in enumerate(plan.periods))
        r = self._run(f"EVALUATE ROW({cols})")[0]
        return [(p, _num(v)) for p, v in zip(plan.periods, r)]


def _label(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() in ("", "<NA>", "nan", "NaT", "None"):
        return "(blank)"
    return str(v)


def _is_ratio(measure: str, values: Sequence[float]) -> bool:
    named = bool(re.search(r"%|\brate\b|ratio|percent|pct|share|margin %|yield", measure, re.I))
    small = bool(values) and all(abs(v) <= 1.5 for v in values)
    return named and small


def compute_findings(plan: ReportPlan, backend: Any) -> dict[str, Any]:
    """Every figure the report may cite, computed with the backend's own queries."""
    out: dict[str, Any] = {"kind": plan.kind, "measures": plan.measures, "grain": plan.grain}
    if plan.kind == "change":
        before_label = plan.compare_measures[0] if plan.compare_measures else plan.periods[0]
        after_label = plan.compare_measures[1] if plan.compare_measures else plan.periods[1]
        out["before_label"], out["after_label"] = before_label, after_label
        tb, ta = backend.total(plan)
        out["total"] = _change(tb, ta)
        out["groupings"] = {}
        for col in plan.group_by:
            rows = [(lbl, b or 0.0, a or 0.0) for lbl, b, a in backend.grouped(plan, col) if (b or a)]
            sb, sa = sum(r[1] for r in rows), sum(r[2] for r in rows)
            additive = bool(tb and ta) and abs(sb - (tb or 0)) <= 0.01 * abs(tb or 1) and abs(sa - (ta or 0)) <= 0.01 * abs(ta or 1)
            groups = []
            for lbl, b, a in rows:
                g = {"group": lbl, **_change(b, a)}
                if additive:
                    g["share_before"] = 100 * b / tb if tb else None
                    g["share_after"] = 100 * a / ta if ta else None
                    g["share_change_points"] = (g["share_after"] - g["share_before"]) if tb and ta else None
                groups.append(g)
            groups.sort(key=lambda g: -(g["change"] or 0))
            ratio = _is_ratio(plan.measures[0], [v for r in rows for v in r[1:]] + [x for x in (tb, ta) if x is not None])
            grew = sum(1 for g in groups if (g["change"] or 0) > 0)
            fell = sum(1 for g in groups if (g["change"] or 0) < 0)
            total_change = (ta or 0) - (tb or 0)
            gains = [g for g in groups if (g["change"] or 0) > 0][:_TOP]
            losses = sorted([g for g in groups if (g["change"] or 0) < 0], key=lambda g: g["change"])[:_TOP]
            info: dict[str, Any] = {"column": _quote(*col), "additive": additive, "ratio": ratio, "groups": groups,
                                    "group_count": len(groups), "grew": grew, "fell": fell,
                                    "unchanged": len(groups) - grew - fell,
                                    "top_gains": [g["group"] for g in gains], "top_losses": [g["group"] for g in losses]}
            if additive and total_change:
                top3 = sum(g["change"] for g in groups[:3])
                info["top3_gains_share_of_total_change_pct"] = 100 * top3 / total_change
            out["groupings"][_quote(*col)] = info
        out["ratio"] = _is_ratio(plan.measures[0], [x for x in (tb, ta) if x is not None])
    else:
        out["series"] = {}
        for m in plan.measures:
            pts = backend.series(plan, m)
            vals = [v for _, v in pts if v is not None]
            points = [{"period": p, "value": v, "change_vs_previous": (v - pts[i - 1][1]) if i and v is not None and pts[i - 1][1] is not None else None}
                      for i, (p, v) in enumerate(pts)]
            stats: dict[str, Any] = {"points": points, "ratio": _is_ratio(m, vals)}
            if vals:
                stats.update(first=vals[0], last=vals[-1], minimum=min(vals), maximum=max(vals),
                             average=sum(vals) / len(vals),
                             max_period=pts[[v for _, v in pts].index(max(vals))][0],
                             min_period=pts[[v for _, v in pts].index(min(vals))][0],
                             last_vs_first=_change(vals[0], vals[-1]),
                             last_vs_previous=_change(vals[-2], vals[-1]) if len(vals) > 1 else None)
                stats["range_pct_of_average"] = 100 * (max(vals) - min(vals)) / abs(stats["average"]) if stats["average"] else None
            out["series"][m] = stats
    return out


def _change(b: float | None, a: float | None) -> dict[str, Any]:
    b0, a0 = b or 0.0, a or 0.0
    return {"before": b, "after": a, "change": a0 - b0, "change_pct": (100 * (a0 - b0) / abs(b0)) if b0 else None}


def _rounded(x: Any, key: str = "") -> Any:
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if key.endswith(("_pct", "_points", "points")):
            return round(x, 1)
        return round(x, 4) if abs(x) <= 1.5 else round(x, 0 if abs(x) >= 1000 else 2)
    if isinstance(x, Mapping):
        return {k: _rounded(v, str(k)) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rounded(v, key) for v in x]
    return x


def _as_rate(change: Mapping[str, Any]) -> dict[str, Any]:
    """A rate's change in percent and percentage points; its relative change misleads (26% to 2% is not -92%)."""
    out = {k: v for k, v in change.items() if k not in ("before", "after", "change", "change_pct")}
    pct = lambda v: None if v is None else 100 * v  # noqa: E731
    out.update(before_pct=pct(change.get("before")), after_pct=pct(change.get("after")),
               change_points=pct(change.get("change")))
    return out


def narrative_view(findings: Mapping[str, Any], per_side: int = 8) -> dict[str, Any]:
    """The findings the narrative run reads: rounded, display labels, and only the groups worth writing about.

    Rates are shown as percentages with changes in percentage points."""
    view = dict(findings)
    if findings.get("ratio") and findings.get("total"):
        view["total"] = _as_rate(findings["total"])
    if findings.get("groupings"):
        view["groupings"] = {}
        for key, g in findings["groupings"].items():
            groups = g["groups"]
            keep = {id(x): x for x in sorted(groups, key=lambda x: -(x["change"] or 0))[:per_side]}
            keep.update({id(x): x for x in sorted(groups, key=lambda x: x["change"] or 0)[:per_side]})
            keep.update({id(x): x for x in sorted(groups, key=lambda x: -(x["after"] or 0))[:per_side]})
            shown = sorted(keep.values(), key=lambda x: -(x["change"] or 0))
            shown = [_as_rate(x) if g.get("ratio") else x for x in shown]
            view["groupings"][key] = {**g, "groups": [{**x, "group": nice_label(x["group"])} for x in shown],
                                      "top_gains": [nice_label(x) for x in g["top_gains"]],
                                      "top_losses": [nice_label(x) for x in g["top_losses"]],
                                      "groups_shown": f"{len(shown)} of {g['group_count']} (largest gains, losses and sizes)"}
    if findings.get("series"):
        view["series"] = {}
        for m, st in findings["series"].items():
            if not st.get("ratio"):
                view["series"][m] = st
                continue
            pct = lambda v: None if v is None else 100 * v  # noqa: E731
            view["series"][m] = {
                "ratio": True, "unit": "percent",
                "points": [{"period": q["period"], "value_pct": pct(q["value"]), "change_vs_previous_points": pct(q["change_vs_previous"])}
                           for q in st["points"]],
                **{f"{k}_pct": pct(st.get(k)) for k in ("first", "last", "minimum", "maximum", "average")},
                "max_period": st.get("max_period"), "min_period": st.get("min_period"),
                "last_vs_first_points": pct((st.get("last_vs_first") or {}).get("change")),
                "range_points": pct((st.get("maximum") or 0) - (st.get("minimum") or 0)) if st.get("maximum") is not None else None,
            }
    return _rounded(view)


def figures(findings: Mapping[str, Any]) -> list[float]:
    """Every number in the findings, with percentages also as fractions and ratios also as percentages."""
    out: list[float] = []

    def walk(x: Any) -> None:
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
            out.append(float(x))
        elif isinstance(x, Mapping):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(findings)
    extra = [v * 100 for v in out if abs(v) <= 1.5] + [v / 100 for v in out]
    return out + extra


# ----------------------------------------------------------------------------------------------- narrative
_PERIOD_TEXT = re.compile(r"\b\d{4}[-/ ]?[Qq][1-4]\b|\b[Qq][1-4]\b|\b\d{4}-\d{2}\b|\bFY\s?\d{2,4}\b|\b(19|20)\d{2}\b")
_NUMBER = re.compile(r"(?<![\w.])[-+−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-+−]?\d+(?:\.\d+)?")
_SUFFIX = re.compile(r"\s?(k|K|thousand|m|M|mn|million|b|B|bn|billion)\b")


def untraceable_numbers(text: str, figs: Sequence[float]) -> list[str]:
    """Numbers in the text that no computed figure matches, allowing for rounding and k/M/B."""
    clean = _PERIOD_TEXT.sub(" ", text)
    bad = []
    for m in _NUMBER.finditer(clean):
        raw = m.group(0)
        digits = raw.replace(",", "").replace("−", "-").lstrip("+")
        try:
            v = abs(float(digits))
        except ValueError:
            continue
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        scale = 1.0
        sm = _SUFFIX.match(clean, m.end())
        if sm:
            scale = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}[sm.group(1).lower()]
        value = v * scale
        if scale == 1.0 and decimals == 0 and v <= 12:
            continue  # small counts and positions: top 5, 3 sections, 12 months
        tol = max(0.5 * 10 ** (-decimals) * scale, 0.005 * value)
        if not any(abs(abs(f) - value) <= tol for f in figs):
            bad.append(raw + (sm.group(0) if sm else ""))
    return bad


NARRATIVE_OUTPUTS: dict[str, type] = {"title": str, "summary": str, "sections": list, "insights": list}

NARRATIVE_RULES = """Report rules:
- Use only the figures in the findings; every number you write must be one of them. Do not compute new numbers.
  Write amounts with thousands separators (1,830,581) or with k/M/B and three or four significant digits
  (3.33M, 232.3k), never 0.0M; percentages with one decimal (121.9%). No currency symbols: the model does not
  name a currency. Write group names as they appear in the findings, and the blank group as "no value".
- sections: a list of {"title": ..., "block": ..., "grouping": ..., "text": ...}. block is one of
  kpis, gains_losses, before_after, mix, trend, period_changes, table, text. grouping is the findings key of the
  grouping the chart uses ("'Table'[Column]"), or "" for kpis, trend, period_changes and text. mix is only for
  groupings with additive true. Each chart appears once: never two sections with the same block and grouping.
- Every section has text: two to four plain sentences saying what the chart shows, the main number and why it
  matters. Between three and six sections, at least two of them with a chart. Pair each finding with the chart
  that shows it best: gains_losses for what rose and fell, before_after for levels in both periods, mix for
  shares, trend for a series, period_changes for the change from each period to the next. Write in the third
  person (no "I" or "we"), and do not mention tools, helpers or code.
- Rates (fields ending _pct with change_points): give the levels as percentages and changes in percentage
  points (fell from 26.0% to 2.0%, down 24.0 points), never a percent change of a rate.
- Say what the data shows, not causes it cannot show. If the question assumed something the findings
  contradict (for example growth when nothing grew), say so plainly in the summary.
- summary: three to five sentences for a business reader. insights: three to five short findings, each with
  its number. title: one line naming the measure and the periods."""


def check_narrative(payload: Mapping[str, Any], findings: Mapping[str, Any], requested: Sequence[str] | None) -> None:
    sections = payload.get("sections") or []
    assert isinstance(sections, list) and sections, "Give the sections as a list."
    assert 3 <= len(sections) <= 8, "Use three to six sections (or every section the user asked for)."
    groupings = findings.get("groupings") or {}
    for i, s in enumerate(sections, 1):
        assert isinstance(s, Mapping), f"Section {i} must be an object with title, block, grouping and text."
        block = str(s.get("block", "")).strip()
        assert block in BLOCKS, f"Section {i} block {block!r} is not one of {', '.join(BLOCKS)}."
        assert len(str(s.get("text", "")).split()) >= 15, f"Section {i} ({s.get('title')}) needs two to four sentences of text."
        if block in ("gains_losses", "before_after", "mix", "table"):
            g = str(s.get("grouping", "")).strip()
            assert g in groupings, f"Section {i} needs grouping set to one of {list(groupings)}."
            if block == "mix":
                assert groupings[g]["additive"], f"Section {i}: mix needs an additive measure; {g} is not. Use gains_losses or table."
        if block == "trend":
            assert findings.get("series"), f"Section {i}: there is no series in the findings for a trend chart."
        if block in ("gains_losses", "before_after", "mix", "table", "kpis"):
            assert findings.get("kind") == "change", f"Section {i}: {block} needs a change; this report is a trend. Use trend, period_changes or text."
        if block == "period_changes":
            assert findings.get("series"), f"Section {i}: period_changes needs a series; use gains_losses for groups."
    charts = [(str(s.get("block")), str(s.get("grouping", "")).strip()) for s in sections if s.get("block") in VISUAL_BLOCKS + ("table", "kpis")]
    repeated = sorted({c for c in charts if charts.count(c) > 1})
    assert not repeated, f"Each chart appears once; these repeat: {repeated}. Use a different block or a text section."
    visuals = sum(1 for s in sections if s.get("block") in VISUAL_BLOCKS)
    assert visuals >= (1 if requested else 2), "Use at least two sections with a chart (gains_losses, before_after, mix, trend, period_changes)."
    if requested:
        titles = [str(s.get("title", "")).lower() for s in sections]
        missing = [r for r in requested if not any(_same_section(r, t) for t in titles)]
        assert not missing, f"These requested sections are missing: {missing}. Use their titles."
    figs = figures(findings)
    words = " ".join([str(payload.get("title", "")), str(payload.get("summary", ""))] + [str(s.get("text", "")) for s in sections]
                     + [str(x) for x in payload.get("insights") or []])
    assert not re.search(r"[$€£¥]", words), "Write amounts without currency symbols; the model does not name a currency."
    assert not re.search(r"<NA>|\(blank\)|\bnan\b|\bNone\b", words), 'Call the blank group "no value".'
    long = re.findall(r"\d+\.\d{3,}", words)
    assert not long, f"Round these numbers for a reader: {long[:6]}."
    texts = [str(payload.get("title", "")), str(payload.get("summary", ""))] + [str(s.get("text", "")) for s in sections] + [str(x) for x in payload.get("insights") or []]
    bad = sorted(set(n for t in texts for n in untraceable_numbers(t, figs)))
    assert not bad, (f"These numbers are not in the findings: {bad[:12]}. Use only figures from `findings` "
                     "(rounded is fine), or leave the number out.")
    assert len(str(payload.get("summary", "")).split()) >= 25, "Write a summary of three to five sentences."
    assert 1 <= len(payload.get("insights") or []) <= 6, "Give three to five insights."


def _same_section(requested: str, title: str) -> bool:
    a = set(re.findall(r"[a-z]+", requested.lower())) - {"and", "the", "by", "of", "a"}
    b = set(re.findall(r"[a-z]+", title))
    return bool(a) and len(a & b) >= max(1, len(a) // 2 + (len(a) > 2))


# ----------------------------------------------------------------------------------------------- render
_CSS = """
:root{--bg:#fcfcfb;--fg:#1d2430;--muted:#5b5a56;--card:#f4f4f1;--line:#e2e1dc;--up:#1a7f5a;--down:#c2412d;--a:#2a78d6;--b:#9aa3ad}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#191918;--fg:#ecebe6;--muted:#b9b8b0;--card:#232322;--line:#383734;--up:#3cb37f;--down:#e0674f;--a:#4b93ea;--b:#6f7780}}
:root[data-theme="dark"]{--bg:#191918;--fg:#ecebe6;--muted:#b9b8b0;--card:#232322;--line:#383734;--up:#3cb37f;--down:#e0674f;--a:#4b93ea;--b:#6f7780}
body{background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;max-width:900px;margin:32px auto;padding:0 16px}
h1{font-size:26px;line-height:1.25;margin:0 0 6px}h2{font-size:19px;margin:34px 0 8px}.sub{color:var(--muted);margin:0 0 18px}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}.kpi{flex:1 1 200px;background:var(--card);border-radius:8px;padding:12px 14px}
.kpi .l{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.kpi .v{font-size:22px;font-weight:650}.kpi .d{font-size:13px;color:var(--muted)}
.up{color:var(--up)}.down{color:var(--down)}svg{max-width:100%;height:auto;display:block;margin:8px 0}svg text{fill:var(--fg);font:12px system-ui,sans-serif}
svg .muted{fill:var(--muted)}table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}details{margin-top:28px;color:var(--muted);font-size:14px}code{font-size:13px}
"""


def fmt(v: float | None, ratio: bool = False, full: bool = False) -> str:
    if v is None:
        return "n/a"
    if ratio:
        return f"{100 * v:.1f}%"
    a = abs(v)
    if full or a < 1_000:
        return f"{v:,.0f}" if a >= 100 else f"{v:,.2f}".rstrip("0").rstrip(".")
    for div, s in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= div:
            x = a / div
            return f"{v / div:,.{3 if x < 10 else 2 if x < 100 else 1}f}{s}"
    return f"{v:,.0f}"


def nice_label(label: str) -> str:
    if label == "(blank)":
        return "No value"
    text = label.replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text.islower() else text


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _text_width(s: str, size: float = 12) -> float:
    return len(s) * size * 0.56


def svg_bars(items: Sequence[tuple[str, float]], title: str, ratio: bool = False) -> str:
    """Horizontal bars around zero; labels on the left, values at the bar ends, room left for both."""
    items = [(nice_label(lbl)[:34], v) for lbl, v in items]
    if not items:
        return ""
    labw = min(260, max(_text_width(lbl) for lbl, _ in items) + 14)
    valw = max(_text_width(fmt(v, ratio)) for _, v in items) + 16
    width, rowh = 760, 28
    lo, hi = min(0, min(v for _, v in items)), max(0, max(v for _, v in items))
    span = (hi - lo) or 1
    plot_l = labw + (valw if lo < 0 else 8)
    plot_r = width - (valw if hi > 0 else 8)
    x = lambda v: plot_l + (v - lo) / span * (plot_r - plot_l)  # noqa: E731
    h = 34 + rowh * len(items) + 10
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{_esc(title)}"><text x="0" y="16" style="font-weight:600;font-size:13px">{_esc(title)}</text>']
    out.append(f'<line x1="{x(0):.1f}" y1="26" x2="{x(0):.1f}" y2="{h - 6}" stroke="var(--line)"/>')
    for i, (lbl, v) in enumerate(items):
        y = 30 + i * rowh
        x0, x1 = sorted((x(0), x(v)))
        color = "var(--up)" if v >= 0 else "var(--down)"
        out.append(f'<text x="{labw - 8:.1f}" y="{y + 15}" text-anchor="end">{_esc(lbl)}</text>')
        out.append(f'<rect x="{x0:.1f}" y="{y + 3}" width="{max(x1 - x0, 1.5):.1f}" height="{rowh - 9}" rx="3" fill="{color}"><title>{_esc(lbl)}: {fmt(v, ratio, full=True)}</title></rect>')
        if v >= 0:
            out.append(f'<text x="{x1 + 6:.1f}" y="{y + 15}">{fmt(v, ratio)}</text>')
        else:
            out.append(f'<text x="{x0 - 6:.1f}" y="{y + 15}" text-anchor="end">{fmt(v, ratio)}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_paired(items: Sequence[tuple[str, float, float]], title: str, names: tuple[str, str], ratio: bool = False, suffix: str = "") -> str:
    """Two bars per group (before lighter, after darker), values at the bar ends."""
    items = [(nice_label(lbl)[:34], b, a) for lbl, b, a in items]
    if not items:
        return ""
    show = (lambda v: f"{v:.1f}{suffix}") if suffix else (lambda v: fmt(v, ratio))  # noqa: E731
    labw = min(260, max(_text_width(lbl) for lbl, _, _ in items) + 14)
    valw = max(_text_width(show(v)) for _, b, a in items for v in (b, a)) + 14
    width, rowh = 760, 40
    hi = max(max(b, a) for _, b, a in items) or 1
    x = lambda v: labw + max(v, 0) / hi * (width - labw - valw)  # noqa: E731
    h = 52 + rowh * len(items) + 8
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{_esc(title)}"><text x="0" y="16" style="font-weight:600;font-size:13px">{_esc(title)}</text>']
    out.append(f'<rect x="{labw}" y="26" width="10" height="10" fill="var(--b)"/><text x="{labw + 14}" y="35" class="muted">{_esc(names[0])}</text>')
    out.append(f'<rect x="{labw + 24 + _text_width(names[0]):.0f}" y="26" width="10" height="10" fill="var(--a)"/><text x="{labw + 38 + _text_width(names[0]):.0f}" y="35" class="muted">{_esc(names[1])}</text>')
    for i, (lbl, b, a) in enumerate(items):
        y = 46 + i * rowh
        out.append(f'<text x="{labw - 8:.1f}" y="{y + 20}" text-anchor="end">{_esc(lbl)}</text>')
        for j, (v, color) in enumerate(((b, "var(--b)"), (a, "var(--a)"))):
            yy = y + 3 + j * 16
            out.append(f'<rect x="{labw}" y="{yy}" width="{max(x(v) - labw, 1.5):.1f}" height="13" rx="2" fill="{color}"><title>{_esc(lbl)}: {show(v)}</title></rect>')
            out.append(f'<text x="{x(v) + 5:.1f}" y="{yy + 11}" style="font-size:11px">{show(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_trend(series: Mapping[str, Sequence[tuple[str, float | None]]], title: str, ratio: bool = False) -> str:
    """One line per measure over the periods, first and last values labelled."""
    width, height, left, right, top, bottom = 760, 300, 64, 90, 34, 40
    vals = [v for pts in series.values() for _, v in pts if v is not None]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.15 or abs(hi) * 0.05 or 1
    lo, hi = lo - pad, hi + pad
    periods = list(next(iter(series.values())))
    n = len(periods)
    x = lambda i: left + i * (width - left - right) / max(n - 1, 1)  # noqa: E731
    y = lambda v: top + (hi - v) / (hi - lo) * (height - top - bottom)  # noqa: E731
    colors = ["var(--a)", "var(--down)"]
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}"><text x="0" y="16" style="font-weight:600;font-size:13px">{_esc(title)}</text>']
    for k in range(5):
        v = lo + k * (hi - lo) / 4
        out.append(f'<line x1="{left}" y1="{y(v):.1f}" x2="{width - right}" y2="{y(v):.1f}" stroke="var(--line)"/><text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end" class="muted">{fmt(v, ratio)}</text>')
    step = max(1, n // 8)
    for i, (p, _) in enumerate(periods):
        if i % step == 0 or i == n - 1:
            out.append(f'<text x="{x(i):.1f}" y="{height - 14}" text-anchor="middle" class="muted">{_esc(p)}</text>')
    for k, (name, pts) in enumerate(series.items()):
        c = colors[k % 2]
        coords = [(x(i), y(v)) for i, (_, v) in enumerate(pts) if v is not None]
        out.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.2" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in coords)}"/>')
        for i, (p, v) in enumerate(pts):
            if v is not None:
                out.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3" fill="{c}"><title>{_esc(p)}: {fmt(v, ratio, full=True)}</title></circle>')
        last_v = pts[-1][1]
        if last_v is not None:
            out.append(f'<text x="{x(n - 1) + 8:.1f}" y="{y(last_v) + 4 + 14 * k:.1f}" style="fill:{c}">{_esc(nice_label(name))[:16]}</text>')
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------------------------------- report
@dataclass
class BuiltReport:
    question: str
    source_name: str
    plan: ReportPlan | None
    findings: dict[str, Any]
    layout: dict[str, Any]
    names: ModelNames | None = None
    date_column: str = ""
    queries: list[str] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)
    plan_result: Any = None
    narrative_result: Any = None
    failure: str | None = None

    @property
    def verified(self) -> bool:
        return self.failure is None and bool(self.layout)

    # -- blocks ---------------------------------------------------------------------------------
    def _grouping(self, key: str) -> Mapping[str, Any]:
        return (self.findings.get("groupings") or {}).get(key) or next(iter((self.findings.get("groupings") or {}).values()), {})

    def _block_html(self, s: Mapping[str, Any]) -> str:
        block, f = s.get("block"), self.findings
        ratio = bool(f.get("ratio"))
        names = (self._period_name(f.get("before_label", "")), self._period_name(f.get("after_label", "")))
        if block == "kpis":
            t = f.get("total") or {}
            g = self._grouping("")
            cards = [(f"{nice_label(self.plan.measures[0]) if self.plan else 'Total'}, {names[1]}", fmt(t.get("after"), ratio, full=True),
                      _delta(t, ratio) + f" vs {names[0]}")]
            if g.get("groups"):
                top = g["groups"][0] if (g["groups"][0]["change"] or 0) > 0 else min(g["groups"], key=lambda x: x["change"] or 0)
                cards.append(("Biggest mover", nice_label(top["group"]), _delta(top, ratio)))
                cards.append(("Groups that rose / fell", f'{g["grew"]} / {g["fell"]}', f'of {g["group_count"]} in {_col_name(g["column"])}'))
            return '<div class="kpis">' + "".join(
                f'<div class="kpi"><div class="l">{_esc(lab)}</div><div class="v">{_esc(v)}</div><div class="d">{_esc(d)}</div></div>' for lab, v, d in cards) + "</div>"
        if block in ("gains_losses", "before_after", "mix", "table"):
            g = self._grouping(str(s.get("grouping", "")))
            groups = g.get("groups") or []
            ratio = bool(g.get("ratio"))
            gains = [x for x in groups if (x["change"] or 0) > 0][:_TOP]
            losses = sorted([x for x in groups if (x["change"] or 0) < 0], key=lambda x: x["change"])[:_TOP]
            col = _col_name(g.get("column", ""))
            if block == "gains_losses":
                items = [(x["group"], x["change"]) for x in gains] + [(x["group"], x["change"]) for x in reversed(losses)]
                return svg_bars(items, f"Change by {col}, {names[0]} to {names[1]}", ratio)
            if block == "before_after":
                movers = sorted(groups, key=lambda x: -abs(x["change"] or 0))[:6]
                return svg_paired([(x["group"], x["before"] or 0, x["after"] or 0) for x in movers], f"{col}: {names[0]} and {names[1]}", names, ratio)
            if block == "mix":
                top = sorted(groups, key=lambda x: -(x.get("share_after") or 0))[:8]
                return svg_paired([(x["group"], x.get("share_before") or 0, x.get("share_after") or 0) for x in top], f"Share of total by {col}", names, suffix="%")
            rows = "".join(f'<tr><td>{_esc(nice_label(x["group"]))}</td><td>{fmt(x["before"], ratio, True)}</td><td>{fmt(x["after"], ratio, True)}</td>'
                           f'<td class="{"up" if (x["change"] or 0) >= 0 else "down"}">{_signed(x["change"], ratio)}</td><td>{_pct(x["change_pct"])}</td></tr>'
                           for x in (gains + losses))
            return f'<table><tr><th>{_esc(col)}</th><th>{_esc(names[0])}</th><th>{_esc(names[1])}</th><th>Change</th><th>%</th></tr>{rows}</table>'
        if block == "period_changes":
            series = f.get("series") or {}
            m = next(iter(series), None)
            if m is None:
                return ""
            items = [(p["period"], p["change_vs_previous"]) for p in series[m]["points"] if p.get("change_vs_previous") is not None]
            return svg_bars(items, f"{nice_label(m)}: change from the previous {f.get('grain') or 'period'}", bool(series[m].get("ratio")))
        if block == "trend":
            series = {m: [(p["period"], p["value"]) for p in st["points"]] for m, st in (f.get("series") or {}).items()}
            ratio = any(st.get("ratio") for st in (f.get("series") or {}).values())
            return svg_trend(series, " and ".join(nice_label(m) for m in series) + " by " + (f.get("grain") or "period"), ratio)
        return ""

    def _period_name(self, label: str) -> str:
        return nice_label(label) if self.plan and self.plan.compare_measures else label

    # -- pages ----------------------------------------------------------------------------------
    def _parts(self) -> list[dict[str, Any]]:
        """The page in order. A section the user named Summary, Insights or How this was checked holds that
        content, so it is not repeated at the top or the end."""
        lay = self.layout
        role = lambda t: ("summary" if re.search(r"summary|overview", t, re.I) else  # noqa: E731
                          "insights" if re.search(r"insight|finding", t, re.I) else
                          "method" if re.search(r"checked|method", t, re.I) else "")
        sections = list(lay.get("sections") or [])
        roles = {role(str(s.get("title", ""))) for s in sections}
        parts: list[dict[str, Any]] = []
        if "summary" not in roles:
            parts.append({"title": None, "text": str(lay.get("summary", "")), "section": None})
        for s in sections:
            r = role(str(s.get("title", "")))
            text = str(lay.get("summary", "")) if r == "summary" else str(s.get("text", ""))
            parts.append({"title": str(s.get("title", "")), "text": text, "section": s,
                          "insights": r == "insights", "method": r == "method"})
        if "insights" not in roles and lay.get("insights"):
            parts.append({"title": "Insights", "text": "", "section": None, "insights": True})
        if "method" not in roles:
            parts.append({"title": "How this was checked", "text": "", "section": None, "method": True})
        return parts

    def _method_items(self) -> list[str]:
        p = self.plan
        if not p:
            return []
        items = [f"How the question was read: {p.reading}",
                 "Measures: " + "; ".join(f"[{m}]" + (f" = {self.names.expression(m)}" if self.names and self.names.expression(m) else "")
                                          for m in (p.compare_measures or p.measures))]
        if p.group_by:
            items.append("Grouped by " + ", ".join(_quote(*g) for g in p.group_by))
        if p.periods:
            items.append(f"Periods: {', '.join(p.periods)} ({p.grain}s), dated by {self.date_column}; each checked complete with period_coverage.")
        if p.filters:
            items.append("Filters: " + "; ".join(f"{_quote(*c)} in {v}" for c, v in p.filters))
        items.append(f"Every number on this page was computed by {len(self.queries)} DAX queries written by fabric-rlm, not by the "
                     "language model; the narrative was checked to cite only those figures.")
        return items

    def to_html(self) -> str:
        if not self.verified:
            return f"<p>No report: {_esc(self.failure)}</p>"
        lay = self.layout
        out = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
               f"<title>{_esc(lay.get('title'))}</title><style>{_CSS}</style></head><body>",
               f"<h1>{_esc(lay.get('title'))}</h1><p class='sub'>{_esc(self.source_name)}</p>"]
        for part in self._parts():
            if part["title"]:
                out.append(f"<h2>{_esc(part['title'])}</h2>")
            if part["text"]:
                out.append(f"<p>{_esc(part['text'])}</p>")
            if part["section"] is not None:
                out.append(self._block_html(part["section"]))
            if part.get("insights") and lay.get("insights"):
                out.append("<ul>" + "".join(f"<li>{_esc(i)}</li>" for i in lay["insights"]) + "</ul>")
            if part.get("method"):
                out.append("<ul class='method'>" + "".join(f"<li>{_esc(i)}</li>" for i in self._method_items()) + "</ul>")
        out.append("</body></html>")
        return "".join(out)

    def to_markdown(self, chart_dir: str | None = None) -> tuple[str, dict[str, str]]:
        """Markdown text and the chart files to write next to it ({relative path: svg})."""
        if not self.verified:
            return f"No report: {self.failure}\n", {}
        lay, files = self.layout, {}
        out = [f"# {lay.get('title')}", "", f"_{self.source_name}_", ""]
        for i, part in enumerate(self._parts(), 1):
            if part["title"]:
                out += [f"## {part['title']}", ""]
            if part["text"]:
                out += [part["text"], ""]
            s = part["section"]
            if s is not None:
                out += self._block_md(s)
                if chart_dir and s.get("block") in VISUAL_BLOCKS:
                    svg = self._block_html(s)
                    if svg:
                        path = f"{chart_dir}/chart_{i}.svg"
                        files[path] = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
                        out += [f"![{part['title']}]({path})", ""]
            if part.get("insights") and lay.get("insights"):
                out += [f"- {x}" for x in lay["insights"]] + [""]
            if part.get("method"):
                out += [f"- {x}" for x in self._method_items()] + [""]
        return "\n".join(out).rstrip() + "\n", files

    def _block_md(self, s: Mapping[str, Any]) -> list[str]:
        f = self.findings
        if s.get("block") in ("gains_losses", "before_after", "table", "mix"):
            g = self._grouping(str(s.get("grouping", "")))
            ratio = bool(g.get("ratio"))
            names = (self._period_name(f.get("before_label", "")), self._period_name(f.get("after_label", "")))
            groups = g.get("groups") or []
            if s.get("block") == "mix":
                rows = sorted(groups, key=lambda x: -(x.get("share_after") or 0))[:8]
                return [f"| {_col_name(g.get('column', ''))} | {names[0]} share | {names[1]} share |", "|---|---:|---:|"] + [
                    f"| {nice_label(x['group'])} | {x.get('share_before') or 0:.1f}% | {x.get('share_after') or 0:.1f}% |" for x in rows] + [""]
            gains = [x for x in groups if (x["change"] or 0) > 0][:_TOP]
            losses = sorted([x for x in groups if (x["change"] or 0) < 0], key=lambda x: x["change"])[:_TOP]
            return [f"| {_col_name(g.get('column', ''))} | {names[0]} | {names[1]} | Change | % |", "|---|---:|---:|---:|---:|"] + [
                f"| {nice_label(x['group'])} | {fmt(x['before'], ratio, True)} | {fmt(x['after'], ratio, True)} | {_signed(x['change'], ratio)} | {_pct(x['change_pct'])} |"
                for x in gains + losses] + [""]
        if s.get("block") in ("trend", "period_changes"):
            series = f.get("series") or {}
            ms = list(series)
            pts = list(zip(*[series[m]["points"] for m in ms]))
            return [f"| Period | {' | '.join(nice_label(m) for m in ms)} |", "|---|" + "---:|" * len(ms)] + [
                f"| {row[0]['period']} | {' | '.join(fmt(p['value'], series[ms[k]].get('ratio'), True) for k, p in enumerate(row))} |" for row in pts] + [""]
        if s.get("block") == "kpis":
            t = f.get("total") or {}
            return [f"**{fmt(t.get('after'), bool(f.get('ratio')), True)}** ({_delta(t, bool(f.get('ratio')))})", ""]
        return []

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in (".md", ".markdown"):
            text, files = self.to_markdown(chart_dir=f"{path.stem}_charts")
            for rel, svg in files.items():
                (path.parent / rel).parent.mkdir(parents=True, exist_ok=True)
                (path.parent / rel).write_text(_standalone_svg(svg), encoding="utf-8")
            path.write_text(text, encoding="utf-8")
        else:
            path.write_text(self.to_html(), encoding="utf-8")
        return path

    def _repr_html_(self) -> str:
        return self.to_html()


def _standalone_svg(svg: str) -> str:
    css = ("<style>text{font:12px sans-serif;fill:#1d2430}.muted{fill:#5b5a56}</style>")
    svg = re.sub(r"var\(--up\)", "#1a7f5a", svg)
    for k, v in {"down": "#c2412d", "a": "#2a78d6", "b": "#9aa3ad", "line": "#e2e1dc", "fg": "#1d2430", "muted": "#5b5a56"}.items():
        svg = svg.replace(f"var(--{k})", v)
    return svg.replace(">", ">" + css, 1)


def _col_name(column: str) -> str:
    ref = column_ref(column)
    return ref[1] if ref else column


def _signed(v: float | None, ratio: bool) -> str:
    if v is None:
        return "n/a"
    return ("+" if v > 0 else "") + (f"{100 * v:.1f} pts" if ratio else fmt(v, full=True))


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{'+' if v > 0 else ''}{v:.1f}%"


def _delta(change: Mapping[str, Any], ratio: bool) -> str:
    c, p = change.get("change"), change.get("change_pct")
    if ratio:
        return f"{_signed(c, True)}"
    return f"{_signed(c, False)}" + (f" ({_pct(p)})" if p is not None else "")


# ----------------------------------------------------------------------------------------------- build
def display_name(model: Any) -> str:
    """The model's name for the page: its display name, never a bare id when the name can be found."""
    name = str(getattr(model, "dataset", "") or "")
    if name and not _GUID.match(name):
        return name
    try:
        import sempy.fabric as fabric

        kw = {"workspace": model.workspace} if getattr(model, "workspace", None) else {}
        return str(fabric.resolve_dataset_name(name, **kw))
    except Exception:  # noqa: BLE001 - the page still renders without a name
        return "the semantic model"


def build_report(question: str, *, inputs: Mapping[str, Any], lm: Any, sections: Sequence[str] | None = None,
                 max_turns: int = 16, backend: Any = None, name: str | None = None, **rlm_kwargs: Any) -> BuiltReport:
    """Plan, compute, narrate and assemble a report for ``question`` (see the module docstring)."""
    from .runtime import RLM
    from .semantic_checks import period_coverage

    models = [(k, v) for k, v in inputs.items() if hasattr(v, "dax") and hasattr(v, "measures")]
    if len(models) != 1:
        raise ValueError("RLM.report needs exactly one semantic model among the inputs.")
    alias, model = models[0]
    names = ModelNames.read(model)
    covs: dict[tuple[str, str], Any] = {}

    def coverage(measure: str, grain: str) -> Any:
        if (measure, grain) not in covs:
            covs[(measure, grain)] = period_coverage(model, measure, grain=grain, find_as_of=False)
        return covs[(measure, grain)]

    report = BuiltReport(question, name or display_name(model), None, {}, {}, names)
    t0 = time.time()
    holder: dict[str, ReportPlan] = {}

    def plan_validator(payload: Mapping[str, Any]) -> None:
        try:
            holder["plan"] = check_plan(payload, names, coverage)
        except (ValueError, TypeError) as exc:  # a malformed field is the model's to fix, not a validator crash
            raise AssertionError(f"The plan could not be read: {exc}") from exc

    plan_validator.instructions = PLAN_RULES.format(name=alias)
    plan_task = (f"Plan a report that answers this question about the semantic model bound as `{alias}`:\n\n{question}\n\n"
                 f"Read the model first ({alias}.schema(), {alias}.measures()) and check periods with {alias}.period_coverage. "
                 "Do not compute the report's numbers; return the plan only.")
    plan_run = RLM.task(plan_task, inputs={alias: model}, outputs=PLAN_OUTPUTS, output_validator=plan_validator,
                        allow_empty={"group_by", "periods", "compare_measures", "filters"},
                        skills=["semantic_model"], lm=lm, max_turns=max_turns, **rlm_kwargs).run()
    report.plan_result, report.seconds["plan"] = plan_run, round(time.time() - t0, 1)
    if not plan_run.submitted or "plan" not in holder:
        report.failure = f"no checked plan ({plan_run.failure_reason or 'not submitted'})"
        return report
    plan = report.plan = holder["plan"]

    t1 = time.time()
    if backend is None:
        date_column = coverage(plan.measures[0], plan.grain).date_column if plan.periods else ""
        backend = DaxBackend(model, date_column)
    report.date_column = getattr(backend, "date_column", "")
    report.findings = compute_findings(plan, backend)
    report.queries = list(getattr(backend, "queries", []))
    report.seconds["compute"] = round(time.time() - t1, 1)

    t2 = time.time()
    requested = [s for s in (sections or []) if str(s).strip()]

    def narrative_validator(payload: Mapping[str, Any]) -> None:
        check_narrative(payload, report.findings, requested)

    narrative_validator.instructions = NARRATIVE_RULES
    view = narrative_view(report.findings)
    ask = (f"Write a report that answers this question: {question}\n\nHow the question was read: {plan.reading}\n\n"
           "The figures, computed from the semantic model, are below (also bound as `findings`). You do not need to "
           "query anything: write the report from these figures and SUBMIT it.\n\n"
           f"findings = {json.dumps(view, default=str)}\n\n"
           + (f"Use exactly these sections, in this order: {', '.join(requested)}. " if requested else
              "Choose the sections yourself: the findings worth a business reader's time, each with the chart that shows it best. ")
           + "Return title, summary, sections and insights.")
    narrative_run = RLM.task(ask, inputs={"findings": json.loads(json.dumps(view, default=str))},
                             outputs=NARRATIVE_OUTPUTS, output_validator=narrative_validator, lm=lm,
                             max_turns=max_turns, **rlm_kwargs).run()
    report.narrative_result, report.seconds["narrative"] = narrative_run, round(time.time() - t2, 1)
    report.seconds["total"] = round(time.time() - t0, 1)
    if not narrative_run.submitted:
        report.failure = f"no checked narrative ({narrative_run.failure_reason or 'not submitted'})"
        return report
    report.layout = dict(narrative_run.payload)
    return report
