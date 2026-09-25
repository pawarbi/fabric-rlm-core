"""Period coverage and claim checks for semantic-model runs.

Two failure classes kept recurring when RLM analysed a semantic model it had
never seen, and neither was about DAX skill:

* The headline period. Runs led with a 9-day month, a month four years in the
  future (forward-dated ARR), and a January-to-May "year", while their own
  caveats said the period might be incomplete. Knowing was not the problem;
  acting on it was.
* A number whose query silently lost a filter. A share "for 2017" was computed
  over every year because the year filter sat in SUMMARIZECOLUMNS and the
  share in an outer ADDCOLUMNS. Re-running the same query reproduces the same
  wrong number, so only a recomputation written independently catches it.

``period_coverage`` measures, instead of asking the model to reason, how much
data each period holds for a measure and which periods are complete.
``semantic_model_checks`` turns both into an ``output_validator``: a headline
period that is partial, in progress or in the future goes back for repair
unless it is labelled period-to-date, and every structured claim is recomputed
with DAX the harness writes itself.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "PeriodCoverage",
    "SemanticModelChecks",
    "period_coverage",
    "semantic_model_checks",
]

GRAINS = ("month", "quarter", "year", "week")
# A period holds "clearly fewer" days than usual below this share of the
# typical earlier period. Monthly-grain data (one row per month) has a typical
# count of 1 and is never partial by this rule; a weekday-only business has a
# typical count near 21 and a full month of it is complete.
PARTIAL_SHARE = 0.8
BASELINE_PERIODS = 6

_COLUMN_REF = re.compile(r"^\s*(?:'((?:[^']|'')+)'|([A-Za-z_][\w ]*?))\s*\[([^\]]+)\]\s*$")
_DATE_TABLE_HINT = re.compile(r"date|calendar|time|month|period|day", re.I)
_AS_OF_HINT = re.compile(r"latest|as[ _]?of|current|snapshot|refresh|data[ _]through|last[ _](?:data|load)", re.I)


# ---------------------------------------------------------------- helpers ---

def _quote_table(name: str) -> str:
    return "'" + str(name).replace("'", "''") + "'"


def _column_ref(ref: str) -> str:
    """Normalise ``Table[Column]`` or ``'Table'[Column]`` to a quoted DAX reference."""
    m = _COLUMN_REF.match(str(ref))
    if not m:
        raise ValueError(f"{ref!r} is not a column reference; write it as 'Table'[Column]")
    table = (m.group(1) or "").replace("''", "'") or m.group(2).strip()
    return f"{_quote_table(table)}[{m.group(3)}]"


def _dax_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE()" if value else "FALSE()"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"filter value {value!r} is not a finite number")
        return repr(value)
    if isinstance(value, (_dt.date, _dt.datetime)):
        return f"DATE({value.year},{value.month},{value.day})"
    return '"' + str(value).replace('"', '""') + '"'


def _as_date(value: Any) -> _dt.date | None:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    to_py = getattr(value, "to_pydatetime", None)
    if callable(to_py):
        try:
            return to_py().date()
        except Exception:  # noqa: BLE001 - NaT
            return None
    text = str(value).strip()
    if not text or text.lower() in {"nat", "none", "nan"}:
        return None
    try:
        return _dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def period_key(day: _dt.date, grain: str) -> str:
    if grain == "month":
        return f"{day.year:04d}-{day.month:02d}"
    if grain == "quarter":
        return f"{day.year:04d}-Q{(day.month - 1) // 3 + 1}"
    if grain == "year":
        return f"{day.year:04d}"
    if grain == "week":
        iso = day.isocalendar()
        return f"{iso[0]:04d}-W{iso[1]:02d}"
    raise ValueError(f"grain must be one of {GRAINS}, got {grain!r}")


def period_bounds(key: str) -> tuple[_dt.date, _dt.date, str]:
    """``(start, end_exclusive, grain)`` for a period label such as 2026-07, 2026-Q3, 2026 or 2026-W05."""
    text = str(key).strip()
    m = re.fullmatch(r"(\d{4})[-/ ]?[Qq]([1-4])", text)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        start = _dt.date(y, 3 * (q - 1) + 1, 1)
        end = _dt.date(y + (q == 4), 1 if q == 4 else 3 * q + 1, 1)
        return start, end, "quarter"
    m = re.fullmatch(r"(\d{4})-?W(\d{1,2})", text, re.I)
    if m:
        start = _dt.date.fromisocalendar(int(m.group(1)), int(m.group(2)), 1)
        return start, start + _dt.timedelta(days=7), "week"
    m = re.fullmatch(r"(?:FY)?(\d{4})", text, re.I)
    if m:
        y = int(m.group(1))
        return _dt.date(y, 1, 1), _dt.date(y + 1, 1, 1), "year"
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?(?:[T ].*)?", text)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if m.group(3) and int(m.group(3)) != 1:
            day = _dt.date(y, mo, int(m.group(3)))
            return day, day + _dt.timedelta(days=1), "day"
        return _dt.date(y, mo, 1), _dt.date(y + (mo == 12), 1 if mo == 12 else mo + 1, 1), "month"
    raise ValueError(
        f"period {key!r} is not recognised; use 2026-07 (month), 2026-Q3 (quarter), 2026 (year) or 2026-W05 (week)"
    )


def _frame_rows(frame: Any) -> list[list[Any]]:
    values = getattr(frame, "values", None)
    if values is not None and hasattr(values, "tolist"):
        return values.tolist()
    return [list(r) for r in frame]


def _records(frame: Any) -> list[dict[str, Any]]:
    to_dict = getattr(frame, "to_dict", None)
    if callable(to_dict):
        return to_dict("records")
    return list(frame)


# ------------------------------------------------------- date discovery ---

def discover_date_column(model: Any) -> tuple[str, str]:
    """Pick the date column that drives the model's time filters, and say why."""
    columns = _records(model.columns())
    dates = [c for c in columns if "date" in str(c.get("Data Type", "")).lower()]
    if not dates:
        raise ValueError("the model has no date or datetime column; pass date_column= explicitly")
    targets: set[tuple[str, str]] = set()
    try:
        for r in _records(model.relationships()):
            to_t = r.get("To Table") or r.get("to_table")
            to_c = r.get("To Column") or r.get("to_column")
            if to_t and to_c:
                targets.add((str(to_t), str(to_c)))
    except Exception:  # noqa: BLE001 - relationships are a hint, not a requirement
        pass

    def score(c: Mapping[str, Any]) -> tuple[int, int, int]:
        t, n = str(c.get("Table Name")), str(c.get("Column Name"))
        return (
            (t, n) in targets,
            bool(_DATE_TABLE_HINT.search(t)),
            n.lower() in {"date", "month", "day", "period"},
        )

    best = max(dates, key=score)
    t, n = str(best["Table Name"]), str(best["Column Name"])
    reasons = []
    s = score(best)
    if s[0]:
        reasons.append("fact tables filter through it (one side of a relationship)")
    if s[1]:
        reasons.append("its table is named like a calendar")
    if not reasons:
        reasons.append("the first date column found; pass date_column= to choose another")
    return f"{_quote_table(t)}[{n}]", "; ".join(reasons)


def as_of_candidates(model: Any, limit: int = 5) -> list[dict[str, Any]]:
    """Measures whose names suggest the model's own as-of date, with the value each returns."""
    found: list[dict[str, Any]] = []
    try:
        measures = _records(model.measures())
    except Exception:  # noqa: BLE001
        return found
    for m in measures:
        name = str(m.get("Measure Name", ""))
        if not _AS_OF_HINT.search(name):
            continue
        try:
            frame = model.dax(f'EVALUATE ROW("v", [{name}])')
            raw = _frame_rows(frame)[0][0]
        except Exception as exc:  # noqa: BLE001
            found.append({"measure": name, "value": None, "error": f"{type(exc).__name__}"})
            continue
        found.append({"measure": name, "value": raw, "date": _as_date(raw)})
        if len(found) >= limit:
            break
    return found


# ----------------------------------------------------------- coverage ---

@dataclass
class PeriodCoverage:
    """How much data each period holds for one expression, and which periods are complete."""

    expression: str
    date_column: str
    date_column_reason: str
    grain: str
    as_of: _dt.date
    as_of_source: str
    periods: list[dict[str, Any]]
    latest_complete: str | None
    latest_with_data: str | None
    first_data: _dt.date | None
    last_data: _dt.date | None
    future_periods_with_data: int
    as_of_measures: list[dict[str, Any]] = field(default_factory=list)

    def status(self, period: str) -> str:
        for p in self.periods:
            if p["period"] == period:
                return p["status"]
        return "no data"

    def table(self) -> Any:
        try:
            import pandas as pd
        except Exception:  # pragma: no cover
            return self.periods
        return pd.DataFrame(self.periods)

    def summary(self, last: int = 8) -> str:
        lines = [
            f"Coverage of {self.expression} by {self.grain}, dated by {self.date_column} ({self.date_column_reason}).",
            f"As of {self.as_of} ({self.as_of_source}). Data from {self.first_data} to {self.last_data}.",
            f"Latest complete {self.grain}: {self.latest_complete or 'none'}; latest with data: {self.latest_with_data or 'none'}.",
        ]
        current = [p for p in self.periods if p["status"] != "future"]
        data_idx = [i for i, p in enumerate(current) if p["days_with_data"]]
        stale = len(current) - 1 - data_idx[-1] if data_idx else 0
        if stale > 1:
            lines.append(
                f"No data in the {stale} {self.grain}s between {self.latest_with_data} and the as-of date: "
                "the data is stale, so say which period it describes instead of calling it current."
            )
        if self.future_periods_with_data:
            lines.append(
                f"{self.future_periods_with_data} period(s) after the as-of date hold data (forward-dated or planned values); "
                "they are marked future and are not the current period."
            )
        for m in self.as_of_measures:
            if m.get("date"):
                lines.append(f"The model's measure [{m['measure']}] returns {m['date']}; if that is the business as-of date, pass as_of=.")
        shown = current[: data_idx[-1] + 1] if data_idx else current
        if stale and current:
            shown = shown + [current[-1]]
        lines.append("period | dates with data | typical | status")
        for p in shown[-last:]:
            lines.append(f"{p['period']} | {p['days_with_data']} | {p['typical_days'] if p['typical_days'] is not None else '-'} | {p['status']}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # what the model sees when it prints the result
        return self.summary()


def _expression(measure: str | None, table: str | None, expression: str | None) -> str:
    given = [x for x in (measure, table, expression) if x]
    if len(given) != 1:
        raise ValueError("pass exactly one of measure=, table= or expression=")
    if measure:
        return "[" + str(measure).strip().strip("[]") + "]"
    if table:
        return f"COUNTROWS({_quote_table(str(table).strip().strip(chr(39)))})"
    return str(expression)


def period_coverage(
    model: Any,
    measure: str | None = None,
    *,
    table: str | None = None,
    expression: str | None = None,
    date_column: str | None = None,
    grain: str = "month",
    as_of: Any = None,
    periods: int | None = None,
    today: _dt.date | None = None,
    find_as_of: bool = True,
) -> PeriodCoverage:
    """Measure, per period, how many days hold data for ``measure`` and which periods are complete.

    A period is ``future`` when it starts after the as-of date, ``in progress``
    when the as-of date falls inside it, ``empty`` when it holds no data, and
    ``partial`` when it is the first or last period with data and holds
    clearly fewer days than the typical earlier period (below 80 percent of
    the median of up to six earlier periods). Low coverage in the middle of
    the series is ``low coverage``. Without enough history to judge,
    the status is ``unknown``. The as-of date is ``as_of`` if given, otherwise
    today; measures that look like the model's own as-of date are listed in
    the result but never applied silently.
    """
    if grain not in GRAINS:
        raise ValueError(f"grain must be one of {GRAINS}, got {grain!r}")
    expr = _expression(measure, table, expression)
    if date_column:
        col, why = _column_ref(date_column), "chosen by the caller"
    else:
        col, why = discover_date_column(model)
    today = today or _dt.date.today()
    as_of_date = _as_date(as_of) if as_of is not None else None
    if as_of is not None and as_of_date is None:
        raise ValueError(f"as_of={as_of!r} is not a date")
    as_of_source = "given as_of" if as_of_date else "today"
    as_of_date = as_of_date or today

    frame = model.dax(
        f'EVALUATE FILTER(ADDCOLUMNS(VALUES({col}), "v", {expr}), NOT ISBLANK([v]))'
    )
    days = sorted({d for d in (_as_date(r[0]) for r in _frame_rows(frame)) if d is not None})
    counts: dict[str, int] = {}
    for d in days:
        k = period_key(d, grain)
        counts[k] = counts.get(k, 0) + 1

    rows: list[dict[str, Any]] = []
    if days:
        # every period from the first with data to the later of the last data and the as-of date
        cursor, stop = days[0], max(days[-1], as_of_date)
        keys: list[str] = []
        while cursor <= stop:
            k = period_key(cursor, grain)
            if not keys or keys[-1] != k:
                keys.append(k)
            cursor = period_bounds(k)[1] if grain != "week" else cursor + _dt.timedelta(days=7)
        data_keys = [k for k in keys if counts.get(k)]
        first_k, last_k = (data_keys[0], data_keys[-1]) if data_keys else (None, None)
        history: list[int] = []
        for k in keys:
            start, end, _ = period_bounds(k)
            n = counts.get(k, 0)
            baseline = history[-BASELINE_PERIODS:]
            typical = statistics.median(baseline) if len(baseline) >= 2 else None
            if start > as_of_date:
                status = "future"
            elif end > as_of_date + _dt.timedelta(days=1):
                status = "in progress"
            elif n == 0:
                status = "empty"
            elif typical is None:
                status = "unknown" if k == first_k else "complete"
            elif n < PARTIAL_SHARE * typical:
                status = "partial" if k in (first_k, last_k) else "low coverage"
            else:
                status = "complete"
            rows.append({"period": k, "start": start.isoformat(), "end": (end - _dt.timedelta(days=1)).isoformat(),
                         "days_with_data": n, "typical_days": typical, "status": status})
            if n and status in {"complete", "unknown"}:
                history.append(n)
    complete = [r["period"] for r in rows if r["status"] == "complete"]
    with_data = [r["period"] for r in rows if r["days_with_data"] and r["status"] != "future"]
    future_with_data = sum(1 for r in rows if r["status"] == "future" and r["days_with_data"])
    return PeriodCoverage(
        expression=expr,
        date_column=col,
        date_column_reason=why,
        grain=grain,
        as_of=as_of_date,
        as_of_source=as_of_source,
        periods=rows,
        latest_complete=complete[-1] if complete else None,
        latest_with_data=with_data[-1] if with_data else None,
        first_data=days[0] if days else None,
        last_data=days[-1] if days else None,
        future_periods_with_data=future_with_data,
        as_of_measures=as_of_candidates(model) if (as_of is None and find_as_of) else [],
    )


# -------------------------------------------------------------- claims ---

_AGGREGATES = {"sum": "SUM", "count": "COUNT", "countrows": "COUNTROWS", "distinctcount": "DISTINCTCOUNT",
               "average": "AVERAGE", "avg": "AVERAGE", "min": "MIN", "max": "MAX"}


def _value_expression(spec: Mapping[str, Any]) -> str:
    if spec.get("measure"):
        return "[" + str(spec["measure"]).strip().strip("[]") + "]"
    agg = str(spec.get("aggregate", "")).lower()
    if agg in _AGGREGATES:
        if agg == "countrows":
            target = spec.get("table") or str(spec.get("column", "")).split("[")[0]
            return f"COUNTROWS({_quote_table(str(target).strip().strip(chr(39)))})"
        return f"{_AGGREGATES[agg]}({_column_ref(str(spec.get('column', '')))})"
    raise ValueError("each value needs 'measure', or 'aggregate' (sum, count, countrows, distinctcount, average, min, max) with 'column'")


def _filter_arguments(spec: Mapping[str, Any], date_column: str | None) -> list[str]:
    args: list[str] = []
    for ref, value in dict(spec.get("filters") or {}).items():
        col = _column_ref(ref)
        if isinstance(value, (list, tuple, set)):
            items = ", ".join(_dax_literal(v) for v in value)
            args.append(f"{col} IN {{{items}}}")
        else:
            args.append(f"{col} = {_dax_literal(value)}")
    if spec.get("period") not in (None, ""):
        if not date_column:
            raise ValueError("a claim with a period needs a date column; the model has none")
        start, end, _ = period_bounds(str(spec["period"]))
        col = _column_ref(spec.get("date_column") or date_column)
        args.append(f"FILTER(ALL({col}), {col} >= {_dax_literal(start)} && {col} < {_dax_literal(end)})")
    return args


def claim_expression(claim: Mapping[str, Any], date_column: str | None) -> str:
    """The DAX the harness writes for one claim: a CALCULATE, or a DIVIDE of two."""

    def one(spec: Mapping[str, Any], inherited: Mapping[str, Any]) -> str:
        merged = {**inherited, **spec, "filters": {**dict(inherited.get("filters") or {}), **dict(spec.get("filters") or {})}}
        args = _filter_arguments(merged, date_column)
        inner = _value_expression(merged)
        return f"CALCULATE({inner}{''.join(', ' + a for a in args)})"

    if claim.get("numerator") is not None or claim.get("denominator") is not None:
        shared = {k: claim[k] for k in ("filters", "period", "date_column") if k in claim}
        return f"DIVIDE({one(dict(claim['numerator']), shared)}, {one(dict(claim['denominator']), shared)})"
    return one(claim, {})


def _close(claimed: float, actual: float, rel: float) -> bool:
    tol = max(1e-9, rel * abs(actual))
    return abs(claimed - actual) <= tol or abs(claimed - 100 * actual) <= 100 * tol


# ------------------------------------------------------------- checks ---

_INSTRUCTIONS = """Checks on this run. The harness verifies two things when you SUBMIT, using the semantic model bound as `{name}`:
1. The headline period. Before choosing it, call `{name}.period_coverage("<measure>")` (optionally grain="quarter" or "year") and print it; it reports the days with data per period and the latest complete period. Submit headline_measure (the measure name), headline_period (such as 2026-07, 2026-Q3 or 2026) and period_to_date (true only when you deliberately report an unfinished period and say so in the text). A partial, in-progress or future headline period is sent back.
2. Your headline numbers. Submit claims: one entry per headline number, as {{"value": number, "measure": "<measure name>"}} or {{"value": number, "aggregate": "sum|count|countrows|distinctcount|average|min|max", "column": "'Table'[Column]"}}, plus "filters": {{"'Table'[Column]": value or [values]}} and "period": "2017" or "2017-07" where they apply. A share or rate is {{"value": number, "numerator": {{...}}, "denominator": {{...}}}}. The harness recomputes every claim with its own DAX; a claim that does not match is sent back with the recomputed value."""


@dataclass
class SemanticModelChecks:
    """An ``output_validator`` for semantic-model runs, with the outputs and instructions it expects."""

    model: Any
    name: str = "model"
    grain: str = "month"
    as_of: Any = None
    relative_tolerance: float = 0.005
    max_rejections: int = 3
    require_claims: bool = True
    today: _dt.date | None = None
    log: list[dict[str, Any]] = field(default_factory=list)
    _rejections: int = field(default=0, init=False, repr=False)
    _date_column: str | None = field(default=None, init=False, repr=False)

    @property
    def outputs(self) -> dict[str, type]:
        out: dict[str, type] = {"headline_measure": str, "headline_period": str, "period_to_date": bool}
        if self.require_claims:
            out["claims"] = list
        return out

    @property
    def instructions(self) -> str:
        return _INSTRUCTIONS.format(name=self.name)

    def date_column(self) -> str | None:
        if self._date_column is None:
            try:
                self._date_column = discover_date_column(self.model)[0]
            except Exception:  # noqa: BLE001
                self._date_column = ""
        return self._date_column or None

    def check_period(self, payload: Mapping[str, Any]) -> list[str]:
        measure = str(payload.get("headline_measure") or "").strip()
        period = str(payload.get("headline_period") or "").strip()
        if not measure or not period:
            return ["Submit headline_measure and headline_period so the headline period can be checked."]
        try:
            _, _, grain = period_bounds(period)
        except ValueError as exc:
            return [str(exc)]
        if grain == "day":
            return []
        cov = period_coverage(self.model, measure, grain=grain, as_of=self.as_of, today=self.today, find_as_of=False)
        key = period_key(period_bounds(period)[0], grain)
        status = cov.status(key)
        self.log.append({"check": "period", "measure": measure, "period": key, "status": status,
                         "latest_complete": cov.latest_complete})
        if status in {"complete", "unknown"}:
            return []
        if status in {"partial", "in progress"} and bool(payload.get("period_to_date")):
            return []
        detail = next((p for p in cov.periods if p["period"] == key), None)
        held = f" It has data on {detail['days_with_data']} dates against a typical {detail['typical_days']}." if detail and detail.get("typical_days") else ""
        return [
            f"headline_period {key} is {status} for [{measure}] (dated by {cov.date_column}, as of {cov.as_of}).{held} "
            f"The latest complete {grain} is {cov.latest_complete or 'not available'}. Lead with a complete period, "
            "or keep this one, set period_to_date to true and say in the text that it is unfinished."
        ]

    def check_claims(self, payload: Mapping[str, Any]) -> list[str]:
        claims = payload.get("claims")
        if claims in (None, []):
            return ["Submit claims: one entry per headline number, so each can be recomputed."] if self.require_claims else []
        if not isinstance(claims, list):
            return ["claims must be a list of objects."]
        problems: list[str] = []
        for i, claim in enumerate(claims, start=1):
            if not isinstance(claim, Mapping):
                problems.append(f"claim {i} is not an object")
                continue
            try:
                claimed = float(claim.get("value"))
            except (TypeError, ValueError):
                problems.append(f"claim {i} has no numeric value")
                continue
            try:
                expr = claim_expression(claim, self.date_column())
                frame = self.model.dax(f'EVALUATE ROW("v", {expr})')
                raw = _frame_rows(frame)[0][0]
                actual = None if raw is None or (isinstance(raw, float) and math.isnan(raw)) else float(raw)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"claim {i} could not be recomputed: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            ok = actual is not None and _close(claimed, actual, self.relative_tolerance)
            self.log.append({"check": "claim", "index": i, "claimed": claimed, "recomputed": actual, "ok": ok, "dax": expr})
            if not ok:
                problems.append(f"claim {i} says {claimed:g} but {expr} returns {actual!r}")
        return problems

    def __call__(self, payload: Mapping[str, Any]) -> None:
        problems = self.check_period(payload) + self.check_claims(payload)
        if not problems:
            return
        if self._rejections >= self.max_rejections:
            self.log.append({"check": "budget", "accepted_with_problems": problems})
            return
        self._rejections += 1
        raise AssertionError("\n".join(problems))


def semantic_model_checks(model: Any, *, name: str = "model", **options: Any) -> SemanticModelChecks:
    """Build the checks for a run over ``model`` (bound into the run as ``name``).

    Pass ``output_validator=checks``, add ``checks.outputs`` to the run's
    outputs and ``checks.instructions`` to the task text.
    """
    return SemanticModelChecks(model=model, name=name, **options)
