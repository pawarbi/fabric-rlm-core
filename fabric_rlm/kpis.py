"""KPIs built from the source's own structure: entities and their lifecycle, ratios, crossings, concentration.

Nothing here assumes customers, sales or regions. An *entity* is any key on
a fact that refers to something with repeat activity over time (customers,
users, devices, machines, patients, resellers, tickets); the candidates are
ranked by structure (cardinality, repeat rate, a name that reads like an
entity) and the choice is printed with its reasons, so a wrong guess is
visible and can be overridden with ``entity=``. From the chosen entity the
lifecycle counts follow for any domain: active, new (first activity in the
period), retained (active in the previous window too), resurrected (back
after a gap) and churned (active in the previous window, not now).

Ratios divide two measures per period (average order value, revenue per
active customer); crossings report when one series overtakes another or
a level; concentration tracks the share the top groups of a grouping hold
and who entered the top. Every series is computed by the source at day or
period grain and then analysed by the same machinery as any metric: context,
level shifts, the anomaly verdict, the watch list. Every definition is
printed next to its number.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .data_agent_review import _ORDER_ID_HINT, _TIME_COLUMN, _Joins, _is_key, _measure_columns, _q, _sql_literal, _time_join, attribute_paths, humanize_column
from .sweep import _dax_ref, _label

__all__ = ["Entity", "KpiSpec", "discover_entities", "parse_kpi"]

_ENTITY_HINT = re.compile(r"(customer|client|user|member|subscriber|account|consumer|buyer|shopper|visitor|player|patient|student|employee|driver|rider|guest|tenant|reseller|dealer|partner|vendor|supplier|merchant|store|shop|site|branch|device|machine|asset|vehicle|unit|sensor|ticket|case|household|company|organization|organisation)", re.IGNORECASE)
_LIFECYCLE = {
    "new": "new",
    "active": "active",
    "churned": "churned",
    "churn": "churned",
    "lost": "churned",
    "lapsed": "churned",
    "attrition": "churned",
    "retained": "retained",
    "returning": "retained",
    "resurrected": "resurrected",
    "reactivated": "resurrected",
    "won back": "resurrected",
}
_EPOCH_MONDAY = _dt.date(1970, 1, 5)


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Entity:
    """A key on a fact that can be new, active or gone, with the structure that ranked it."""

    column: str
    name: str  # the plural word for the thing: customers, devices
    entities: int
    rows: int
    repeaters: int
    score: float
    reasons: tuple[str, ...] = ()
    fact: str = ""  # the fact whose activity defines new, active and gone

    @property
    def repeat_rate(self) -> float:
        return self.repeaters / self.entities if self.entities else 0.0

    def describe(self) -> str:
        where = f" on {self.fact}" if self.fact else ""
        return f"{self.name} ({self.column}{where}: {self.entities:,} distinct, {self.repeat_rate:.0%} active in more than one month)"


def _entity_word(column: str, joins: Mapping[tuple[str, str], tuple[str, str]], table: str) -> str:
    target = joins.get((table, column))
    stem = target[0].rsplit(".", 1)[-1] if target else re.sub(r"[_ ]?(?:id|key)$", "", column, flags=re.IGNORECASE)
    stem = re.sub(r"^dim[_ ]?", "", stem, flags=re.IGNORECASE)
    word = humanize_column(stem).strip() or column
    if not word.endswith("s"):
        word = word + ("es" if word.endswith(("s", "x", "ch", "sh")) else "s")
    return word


def _candidate_keys(schema: Any, table: str, dialect: Any) -> list[str]:
    columns = schema.tables[table]
    own = table.rsplit(".", 1)[-1].casefold()
    axis = dialect.axis(table) or {}
    time_keys = {str(axis.get("column", "")), str(axis.get("via", ""))}
    out = []
    for column in columns:
        if column in time_keys or _TIME_COLUMN.search(column) or _ORDER_ID_HINT.search(column) or not _is_key(column):
            continue
        stem = re.sub(r"[_ ]?(?:id|key)$", "", column, flags=re.IGNORECASE).casefold()
        if stem and (stem == own or own.startswith(stem) and len(stem) >= 4 and stem not in {"fact"}):
            continue  # the fact's own key, or its line key
        out.append(column)
    return out


def discover_entities(probe: Any, dialect: Any, facts: Sequence[Mapping[str, Any]], joins: Mapping[tuple[str, str], tuple[str, str]], run: Any) -> list[Entity]:
    """Rank the keys of the facts as entities: how many there are, how many recur across months, and whether the name says entity."""
    found: list[Entity] = []
    for fact in facts:
        found.extend(_entities_of(probe, dialect, fact, joins, run))
    return sorted(found, key=lambda e: (-e.score, -e.entities, e.column))


def _entities_of(probe: Any, dialect: Any, fact: Mapping[str, Any], joins: Mapping[tuple[str, str], tuple[str, str]], run: Any) -> list[Entity]:
    schema = probe.schema
    table = fact["table"]
    found: list[Entity] = []
    for column in _candidate_keys(schema, table, dialect):
        query = _entity_stats_query(dialect, fact, column)
        try:
            rows = run(query)
        except Exception:  # noqa: BLE001 - a key the source cannot count is not an entity here
            continue
        row = rows[0] if rows else {}
        entities, rows_n, repeaters = int(row.get("entities") or 0), int(row.get("rows") or 0), int(row.get("repeaters") or 0)
        if entities < 2:
            continue
        reasons = []
        score = 0.0
        name = _entity_word(column, joins, table)
        if _ENTITY_HINT.search(column) or _ENTITY_HINT.search(name):
            score += 2
            reasons.append("named like an entity")
        if entities >= 1000:
            score += 1
            reasons.append(f"{entities:,} distinct")
        elif entities >= 100:
            score += 0.5
            reasons.append(f"{entities:,} distinct")
        else:
            reasons.append(f"only {entities:,} distinct")
        rate = repeaters / entities
        if 0.05 <= rate <= 0.95:
            score += 1
            reasons.append(f"{rate:.0%} recur across months")
        elif rate > 0.95:
            reasons.append(f"{rate:.0%} recur every month, more a dimension than a population")
        else:
            reasons.append(f"only {rate:.0%} recur, more an event than a population")
        found.append(Entity(column, name, entities, rows_n, repeaters, score, tuple(reasons), table))
    return found


def _month_index_sql(dialect: Any, fact: Mapping[str, Any]) -> str:
    day = dialect.day(fact)
    return f"(CAST(strftime({day}, '%Y') AS INTEGER) * 12 + CAST(strftime({day}, '%m') AS INTEGER))"


def _entity_stats_query(dialect: Any, fact: Mapping[str, Any], column: str) -> str:
    if dialect.kind == "lakehouse":
        join = _time_join(fact["date"])
        source = f"FROM {fact['table']} f" + (f" {join}" if join else "")
        return (
            f"SELECT COUNT(*) AS entities, SUM(n) AS rows, SUM(CASE WHEN months >= 2 THEN 1 ELSE 0 END) AS repeaters FROM "
            f"(SELECT f.{_q(column)} AS k, COUNT(*) AS n, COUNT(DISTINCT {_month_index_sql(dialect, fact)}) AS months {source} WHERE f.{_q(column)} IS NOT NULL GROUP BY 1) t"
        )
    key = _dax_ref(fact["table"], column)
    row_date = _dax_row_date(fact)
    months = f"COUNTROWS(DISTINCT(SELECTCOLUMNS(CALCULATETABLE('{fact['table']}'), \"__m\", YEAR({row_date}) * 12 + MONTH({row_date}))))"
    return (
        f'EVALUATE ROW("entities", DISTINCTCOUNT({key}), "rows", COUNTROWS(\'{fact["table"]}\'), '
        f'"repeaters", COUNTROWS(FILTER(ADDCOLUMNS(VALUES({key}), "__months", {months}), [__months] >= 2)))'
    )


def _dax_day_ref(fact: Mapping[str, Any]) -> str:
    dt = fact["date"]
    if dt.get("kind") != "date":
        raise ValueError("the model's date table has no day column, so activity cannot be placed in time")
    return _dax_ref(dt["table"], dt["column"])


def _dax_row_date(fact: Mapping[str, Any]) -> str:
    """The day of a fact row inside a row context: the fact's own column, or the related date table's day through RELATED."""
    dt = fact["date"]
    ref = _dax_day_ref(fact)
    return ref if dt["table"] == fact["table"] else f"RELATED({ref})"


# --------------------------------------------------------------------------- #
# KPI specifications
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KpiSpec:
    """What a KPI is: its kind, the entity or measures behind it, the window, and the definition in words."""

    kind: str  # new | active | churned | retained | resurrected | ratio | crossing | concentration
    name: str
    text: str = ""  # the request as written
    entity: str | None = None  # the entity column, for lifecycle kinds
    entity_words: str = ""  # what the request called it
    window: int = 1  # periods counted as "before" for retained and churned
    numerator: str = ""  # a metric phrase, for ratio
    denominator: str = ""  # a metric phrase or "rows", for ratio
    left: str = ""  # a metric phrase, for crossing
    right: str = ""  # a metric phrase or a number, for crossing
    measure_words: str = ""  # for concentration: the measure phrase
    grouping_words: str = ""  # for concentration: the grouping phrase
    top: int = 3
    definition: str = ""
    notes: tuple[str, ...] = ()


_RATIO = re.compile(r"^\s*(?:(?P<name>[^=]+?)\s*=\s*)?(?P<num>.+?)\s*(?:/|\bper\b|\bdivided by\b|\bover\b)\s*(?P<den>.+?)\s*$", re.IGNORECASE)
_CROSSING = re.compile(r"^\s*(?P<left>.+?)\s+(?:crossing|crosses|cross|overtaking|overtakes|vs\.?|versus|against)\s+(?P<right>.+?)\s*$", re.IGNORECASE)
_CONCENTRATION = re.compile(r"^\s*(?:(?:top\s+(?P<top>\d+)\s+)?(?:share|concentration)\s+of\s+)(?P<measure>.+?)\s+by\s+(?P<grouping>.+?)\s*$", re.IGNORECASE)
_LIFECYCLE_RE = re.compile(r"^\s*(?P<kind>new|active|churned|churn|lost|lapsed|attrition|retained|returning|resurrected|reactivated|won back)\s*(?:of\s+)?(?P<entity>[a-z][a-z _]*?)?\s*(?:over\s+(?P<n>\d+)\s+(?P<unit>weeks?|months?|periods?))?\s*$", re.IGNORECASE)


def parse_kpi(text: str) -> KpiSpec | None:
    """Read a KPI phrase: a lifecycle count, a ratio, a crossing, or a concentration; None when it is an ordinary metric."""
    clean = text.strip().rstrip(".")
    match = _CONCENTRATION.match(clean)
    if match:
        top = int(match.group("top") or 3)
        return KpiSpec("concentration", f"share of the top {top} {match.group('grouping').strip()} in {match.group('measure').strip()}", clean, measure_words=match.group("measure").strip(), grouping_words=match.group("grouping").strip(), top=top)
    match = _CROSSING.match(clean)
    if match and not re.search(r"\bby\b", match.group("right")) or (match and re.search(r"\bwhere\b", clean)):
        return KpiSpec("crossing", f"{match.group('left').strip()} against {match.group('right').strip()}", clean, left=match.group("left").strip(), right=match.group("right").strip())
    match = _LIFECYCLE_RE.match(clean)
    if match:
        kind = _LIFECYCLE[match.group("kind").casefold()]
        window = int(match.group("n")) if match.group("n") else 1
        words = (match.group("entity") or "").strip()
        return KpiSpec(kind, f"{kind} {words}".strip(), clean, entity_words=words, window=window)
    match = _RATIO.match(clean)
    if match and "/" in clean or (match and re.search(r"\b(per|divided by)\b", clean, re.IGNORECASE)):
        name = (match.group("name") or f"{match.group('num').strip()} per {match.group('den').strip()}").strip()
        return KpiSpec("ratio", name, clean, numerator=match.group("num").strip(), denominator=match.group("den").strip())
    return None


def resolve_entity(spec: KpiSpec, entities: Sequence[Entity], override: str | None) -> tuple[Entity | None, str]:
    """The entity a lifecycle KPI counts: the override, the one the words name, else the best ranked; and why."""
    if not entities:
        return None, "no key on the fact recurs across periods, so nothing can be new or churned"
    if override:
        wanted = override.casefold()
        chosen = next((e for e in entities if e.column.casefold() == wanted or e.name.casefold() == wanted or e.name.casefold().rstrip("s") == wanted.rstrip("s")), None)
        if chosen is not None:
            return chosen, f"entity {chosen.describe()}, as specified"
    words = spec.entity_words.casefold()
    if words:
        tokens = set(re.findall(r"[a-z]+", words))
        for e in entities:
            names = set(re.findall(r"[a-z]+", e.name.casefold())) | {e.column.casefold()}
            if tokens & names or any(t.rstrip("s") == n.rstrip("s") for t in tokens for n in names):
                return e, f"entity {e.describe()}, matching '{spec.entity_words}'"
    best = entities[0]
    others = ", ".join(e.describe() for e in entities[1:4])
    missing = f"nothing on any fact is called '{spec.entity_words}'; " if words else ""
    return best, f"{missing}entity {best.describe()}, ranked first because it is {', '.join(best.reasons)}" + (f"; other candidates: {others}" if others else "") + "; say entity=... to change"


# --------------------------------------------------------------------------- #
# Periods
# --------------------------------------------------------------------------- #


def period_index_sql(dialect: Any, fact: Mapping[str, Any], grain: str) -> str:
    day = dialect.day(fact)
    if grain == "week":
        return f"CAST(FLOOR((CAST({day} AS DATE) - DATE '{_EPOCH_MONDAY.isoformat()}') / 7) AS INTEGER)"
    return _month_index_sql(dialect, fact)


def period_start(index: int, grain: str) -> _dt.date:
    if grain == "week":
        return _EPOCH_MONDAY + _dt.timedelta(days=7 * index)
    year, month = divmod(index - 1, 12)
    return _dt.date(year, month + 1, 1)


def period_of(day: _dt.date, grain: str) -> int:
    if grain == "week":
        return (day - _EPOCH_MONDAY).days // 7
    return day.year * 12 + day.month


def period_range(index: int, grain: str) -> tuple[_dt.date, _dt.date]:
    start = period_start(index, grain)
    return start, period_start(index + 1, grain)


# --------------------------------------------------------------------------- #
# Lifecycle series
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LifecyclePoint:
    """One period of growth accounting for an entity: who was active, who was new, who stayed, who came back, who left."""

    index: int
    grain: str
    active: int
    new: int
    retained: int
    resurrected: int
    churned: int
    window_active: int = 0

    @property
    def start(self) -> str:
        return period_start(self.index, self.grain).isoformat()

    def value(self, kind: str) -> float:
        return float(getattr(self, kind))


def lifecycle_sql(dialect: Any, fact: Mapping[str, Any], column: str, grain: str, window: int, filters: Sequence[tuple[Mapping[str, Any], Any]] = ()) -> str:
    """Growth accounting in one query: per entity and period the activity, then per period the counts."""
    joins = _Joins()
    conditions = [dialect._filter(joins, p, v) for p, v in filters]
    time_join = _time_join(fact["date"])
    source = f"FROM {fact['table']} f" + (f" {time_join}" if time_join else "") + "".join(" " + c for c in joins.clauses)
    where = " AND ".join([f"f.{_q(column)} IS NOT NULL", *conditions])
    p = period_index_sql(dialect, fact, grain)
    return (
        f"WITH a AS (SELECT f.{_q(column)} AS k, {p} AS p {source} WHERE {where} GROUP BY 1, 2), "
        f"s AS (SELECT k, p, MIN(p) OVER (PARTITION BY k) AS first_p, LAG(p) OVER (PARTITION BY k ORDER BY p) AS prev_p FROM a), "
        f"per AS (SELECT p, COUNT(*) AS active, SUM(CASE WHEN p = first_p THEN 1 ELSE 0 END) AS new, "
        f"SUM(CASE WHEN p <> first_p AND prev_p >= p - {int(window)} THEN 1 ELSE 0 END) AS retained, "
        f"SUM(CASE WHEN p <> first_p AND prev_p < p - {int(window)} THEN 1 ELSE 0 END) AS resurrected FROM s GROUP BY p), "
        f"win AS (SELECT q.p, COUNT(DISTINCT a.k) AS window_active FROM (SELECT DISTINCT p FROM a) q JOIN a ON a.p BETWEEN q.p - {int(window)} AND q.p - 1 GROUP BY q.p) "
        f"SELECT per.p AS p, per.active, per.new, per.retained, per.resurrected, COALESCE(win.window_active, 0) AS window_active, "
        f"COALESCE(win.window_active, 0) - per.retained AS churned FROM per LEFT JOIN win ON win.p = per.p ORDER BY per.p"
    )


def lifecycle_dax(dialect: Any, fact: Mapping[str, Any], column: str, start: _dt.date, end: _dt.date, window_start: _dt.date, filters: Sequence[tuple[Mapping[str, Any], Any]] = ()) -> str:
    """Growth accounting for one period in one query: active, new, retained and the window's active count."""
    key = _dax_ref(fact["table"], column)
    date_ref = _dax_day_ref(fact)
    dt = fact["date"]
    narrow = [dialect._filter(fact, p, v) for p, v in filters]

    def between(a: _dt.date, b: _dt.date) -> str:
        return f"{date_ref} >= DATE({a.year},{a.month},{a.day}) && {date_ref} < DATE({b.year},{b.month},{b.day})"

    period, before = between(start, end), between(window_start, start)
    args = lambda *extra: ", ".join([*extra, *narrow])  # noqa: E731
    if dt["table"] != fact["table"]:
        # the entity filters the fact, not the date table, so the first activity is read off the fact's rows through RELATED
        first = f"CALCULATE(MINX('{fact['table']}', RELATED({date_ref})), ALL('{dt['table']}'))"
    else:
        first = f"CALCULATE(MIN({date_ref}), REMOVEFILTERS({date_ref}))"
    return (
        f'EVALUATE ROW("active", CALCULATE(DISTINCTCOUNT({key}), {args(period)}), '
        f'"window_active", CALCULATE(DISTINCTCOUNT({key}), {args(before)}), '
        f'"retained", COUNTROWS(INTERSECT(CALCULATETABLE(VALUES({key}), {args(period)}), CALCULATETABLE(VALUES({key}), {args(before)}))), '
        f'"new", COUNTROWS(FILTER(ADDCOLUMNS(CALCULATETABLE(VALUES({key}), {args(period)}), "__first", {first}), [__first] >= DATE({start.year},{start.month},{start.day}) && [__first] < DATE({end.year},{end.month},{end.day}))))'
    )


def lifecycle_series(probe: Any, dialect: Any, fact: Mapping[str, Any], column: str, grain: str, window: int, run: Any, *, filters: Sequence[tuple[Mapping[str, Any], Any]] = (), periods: Sequence[int] = ()) -> list[LifecyclePoint]:
    """The growth accounting series: one query for a lakehouse; one query per period for a model, over the periods given."""
    if dialect.kind == "lakehouse":
        rows = run(lifecycle_sql(dialect, fact, column, grain, window, filters))
        return [LifecyclePoint(int(r["p"]), grain, int(r.get("active") or 0), int(r.get("new") or 0), int(r.get("retained") or 0), int(r.get("resurrected") or 0), int(r.get("churned") or 0), int(r.get("window_active") or 0)) for r in rows if r.get("p") is not None]
    points: list[LifecyclePoint] = []
    skipped: list[str] = []
    for index in periods:
        start, end = period_range(index, grain)
        window_start = period_start(index - window, grain)
        query = lifecycle_dax(dialect, fact, column, start, end, window_start, filters)
        row: Mapping[str, Any] | None = None
        for attempt in range(2):
            try:
                row = (run(query) or [{}])[0]
                break
            except _Budget:
                raise
            except Exception as exc:  # noqa: BLE001 - one period the model would not answer twice is a gap, not the end of the brief
                if attempt == 1:
                    skipped.append(f"{start.isoformat()}: {type(exc).__name__}")
        if row is None:
            continue
        active, window_active, retained, new = (int(row.get(k) or 0) for k in ("active", "window_active", "retained", "new"))
        points.append(LifecyclePoint(index, grain, active, new, retained, max(0, active - new - retained), max(0, window_active - retained), window_active))
    if skipped:
        points.append(LifecyclePoint(-1, grain, 0, 0, 0, 0, 0))  # a marker the caller turns into a note; never a data point
        _SKIPPED[id(points)] = skipped
    return points


_SKIPPED: dict[int, list[str]] = {}


def skipped_periods(points: Sequence[LifecyclePoint]) -> tuple[list[LifecyclePoint], list[str]]:
    """Split the marker for periods the model would not answer from the real points."""
    real = [p for p in points if p.index >= 0]
    return real, _SKIPPED.pop(id(points), [])


# --------------------------------------------------------------------------- #
# Concentration
# --------------------------------------------------------------------------- #


def concentration_sql(dialect: Any, fact: Mapping[str, Any], path: Mapping[str, Any], grain: str, top: int, last: int) -> str:
    """Per period: the total, the sum of the top groups, and the number of groups; one row per period."""
    joins = _Joins()
    label = joins.ref(path)
    time_join = _time_join(fact["date"])
    source = f"FROM {fact['table']} f" + (f" {time_join}" if time_join else "") + "".join(" " + c for c in joins.clauses)
    p = period_index_sql(dialect, fact, grain)
    return (
        f"WITH t AS (SELECT {p} AS p, {label} AS label, SUM(f.{_q(fact['measure'])}) AS v {source} GROUP BY 1, 2), "
        f"r AS (SELECT p, label, v, ROW_NUMBER() OVER (PARTITION BY p ORDER BY v DESC NULLS LAST) AS rn FROM t) "
        f"SELECT p, SUM(v) AS total, SUM(CASE WHEN rn <= {int(top)} THEN v ELSE 0 END) AS top_value, COUNT(*) AS groups, "
        f"STRING_AGG(CASE WHEN rn <= {int(top)} THEN CAST(label AS VARCHAR) END, ' | ' ORDER BY rn) AS leaders FROM r GROUP BY p ORDER BY p DESC LIMIT {int(last)}"
    )


def concentration_dax(dialect: Any, fact: Mapping[str, Any], path: Mapping[str, Any], grain: str, first: _dt.date, last: _dt.date) -> str:
    """Group totals by day and group over a span; the periods are built in Python."""
    ref = dialect._ref(fact, path)
    date_ref = _dax_day_ref(fact)
    span = f"{date_ref} >= DATE({first.year},{first.month},{first.day}) && {date_ref} < DATE({last.year},{last.month},{last.day})"
    inner = f'SUMMARIZECOLUMNS({date_ref}, {ref}, "v", CALCULATE({dialect._expr(fact)}, {span}))'
    return f'EVALUATE SELECTCOLUMNS({inner}, "day", {date_ref}, "label", {ref}, "v", [v])'


@dataclass(frozen=True)
class SharePoint:
    index: int
    grain: str
    total: float
    top_value: float
    groups: int
    leaders: tuple[str, ...] = ()

    @property
    def start(self) -> str:
        return period_start(self.index, self.grain).isoformat()

    @property
    def share(self) -> float:
        return self.top_value / self.total if self.total else 0.0


def concentration_series(probe: Any, dialect: Any, fact: Mapping[str, Any], path: Mapping[str, Any], grain: str, top: int, run: Any, *, last: int = 60, until: _dt.date | None = None) -> list[SharePoint]:
    if dialect.kind == "lakehouse":
        rows = run(concentration_sql(dialect, fact, path, grain, top, last))
        return sorted((SharePoint(int(r["p"]), grain, float(r.get("total") or 0), float(r.get("top_value") or 0), int(r.get("groups") or 0), tuple(str(x) for x in str(r.get("leaders") or "").split(" | ") if x)) for r in rows if r.get("p") is not None), key=lambda s: s.index)
    end = until or _dt.date.today()
    start = period_start(period_of(end, grain) - last, grain)
    by_period: dict[int, dict[Any, float]] = {}
    for row in run(concentration_dax(dialect, fact, path, grain, start, end + _dt.timedelta(days=1))):
        iso = str(row.get("day") or "")[:10]
        if not iso or row.get("v") is None:
            continue
        index = period_of(_dt.date.fromisoformat(iso), grain)
        bucket = by_period.setdefault(index, {})
        bucket[row.get("label")] = bucket.get(row.get("label"), 0.0) + float(row.get("v") or 0)
    points = []
    for index in sorted(by_period):
        values = by_period[index]
        ranked = sorted(values.items(), key=lambda item: -item[1])
        points.append(SharePoint(index, grain, sum(values.values()), sum(v for _l, v in ranked[:top]), len(values), tuple(_label(label) for label, _v in ranked[:top])))
    return points


# --------------------------------------------------------------------------- #
# Crossings
# --------------------------------------------------------------------------- #


def crossings(left: Mapping[int, float], right: Mapping[int, float]) -> list[tuple[int, str]]:
    """Periods where the sign of left minus right changed: (index, "left overtook right" or "right overtook left")."""
    shared = sorted(set(left) & set(right))
    out: list[tuple[int, str]] = []
    previous: float | None = None
    for index in shared:
        gap = left[index] - right[index]
        if previous is not None and gap != 0 and previous != 0 and (gap > 0) != (previous > 0):
            out.append((index, "left overtook right" if gap > 0 else "right overtook left"))
        if gap != 0:
            previous = gap
    return out


def measure_phrase_filters(text: str) -> tuple[str, list[tuple[str, str]]]:
    """``order quantity where channel = Internet`` -> ("order quantity", [("channel", "Internet")])."""
    match = re.match(r"^(?P<metric>.+?)\s+(?:where|for|with)\s+(?P<filters>.+)$", text.strip(), re.IGNORECASE)
    if not match:
        return text.strip(), []
    pairs = []
    for clause in re.split(r"\s+and\s+|,", match.group("filters")):
        parts = re.split(r"\s*(?:=|==|\bis\b|\bequals\b|:)\s*", clause.strip(), maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            pairs.append((parts[0].strip(), parts[1].strip().strip("'\"")))
    return match.group("metric").strip(), pairs


def find_path(schema: Any, table: str, joins: Mapping[tuple[str, str], tuple[str, str]], words: str) -> Mapping[str, Any] | None:
    """The grouping path a word names: the column itself, its humanized name, or the closest by token overlap."""
    wanted = words.strip().casefold()
    paths = attribute_paths(schema, table, joins)
    for p in paths:
        column = str(p["column"])
        if column.casefold() == wanted or humanize_column(column).casefold() == wanted or column.casefold().replace("_", " ") == wanted:
            return p
    tokens = set(re.findall(r"[a-z0-9]+", wanted))
    scored = sorted(((len(tokens & set(re.findall(r"[a-z0-9]+", humanize_column(str(p["column"])).casefold()))), -len(p.get("hops") or ()), str(p["column"])) for p in paths), reverse=True)
    if scored and scored[0][0] > 0:
        return next(p for p in paths if str(p["column"]) == scored[0][2])
    return None


def literal_for(schema: Any, table: str, path: Mapping[str, Any], value: str) -> Any:
    """A filter value typed like its column when the column is numeric, else the text as given."""
    from .data_agent_review import _NUMERIC_TYPE, _path_table

    column_type = schema.column_type(_path_table(path, table), str(path["column"]))
    if _NUMERIC_TYPE.search(column_type):
        try:
            return int(value) if re.fullmatch(r"-?\d+", value) else float(value)
        except ValueError:
            return value
    return value


def sql_literal(value: Any) -> str:
    return _sql_literal(value)


def default_measure(schema: Any, table: str) -> str | None:
    columns = _measure_columns(schema, table)
    return columns[0] if columns else None


def kpi_definition(spec: KpiSpec, *, entity: Entity | None = None, grain: str = "week", how: str = "") -> str:
    """The definition printed next to the number, in words a reader can check."""
    unit = "week" if grain == "week" else "month"
    if spec.kind == "new" and entity:
        return f"{entity.name} whose first activity falls in the {unit} ({how})"
    if spec.kind == "active" and entity:
        return f"{entity.name} with any activity in the {unit} ({how})"
    if spec.kind == "retained" and entity:
        return f"{entity.name} active in the {unit} and in the {spec.window} {unit}{'s' if spec.window != 1 else ''} before ({how})"
    if spec.kind == "resurrected" and entity:
        return f"{entity.name} active in the {unit} after a gap of more than {spec.window} {unit}{'s' if spec.window != 1 else ''} ({how})"
    if spec.kind == "churned" and entity:
        return f"{entity.name} active in the {spec.window} {unit}{'s' if spec.window != 1 else ''} before and not in the {unit} ({how})"
    if spec.kind == "ratio":
        return f"{spec.numerator} divided by {spec.denominator}, both summed over the {unit}"
    if spec.kind == "crossing":
        return f"{spec.left} against {spec.right}, by {unit}; a crossing is the {unit} the sign of the gap changed"
    if spec.kind == "concentration":
        return f"the share of {spec.measure_words} held by the top {spec.top} {spec.grouping_words} in the {unit}"
    return spec.text


# --------------------------------------------------------------------------- #
# Building the KPI sections of a brief
# --------------------------------------------------------------------------- #

_LIFECYCLE_KINDS = frozenset({"new", "active", "retained", "resurrected", "churned"})
_ROW_WORDS = frozenset({"rows", "row", "records", "transactions", "count", "lines", "events"})


class _Budget(Exception):
    pass


def _side(phrase: str, probe: Any, joins: Mapping[tuple[str, str], tuple[str, str]], instructions: str, scope: str) -> tuple[dict[str, Any], list[tuple[Mapping[str, Any], Any]], str]:
    """A metric phrase with optional filters -> (metric spec, [(path, value)], what could not be matched)."""
    from .brief import _metric_specs

    words, pairs = measure_phrase_filters(phrase)
    spec = _metric_specs([words], probe, instructions, scope)[0]
    filters: list[tuple[Mapping[str, Any], Any]] = []
    unmatched = []
    for column_words, value in pairs:
        path = find_path(probe.schema, spec["fact"], joins, column_words) if spec.get("fact") else None
        if path is None:
            unmatched.append(column_words)
            continue
        filters.append((path, literal_for(probe.schema, spec["fact"], path, value)))
    return spec, filters, ", ".join(unmatched)


def _weeks_of(rows: Sequence[Mapping[str, Any]], value_key: str, max_day: _dt.date | None = None) -> tuple[list[Any], _dt.date | None]:
    from .brief import _weeks
    from .sweep import _iso_date

    daily = []
    for row in rows:
        iso = _iso_date(row.get("day"))
        if iso:
            daily.append((_dt.date.fromisoformat(iso), int(row.get("n") or 0), float(row.get(value_key) or 0)))
    if not daily:
        return [], None
    daily.sort()
    last = max_day or daily[-1][0]
    return _weeks(daily, "sum", last), last


def _kpi_headline(spec: KpiSpec, name: str, target: Any, c: Mapping[str, Any]) -> str:
    from .brief import _pct

    if spec.kind == "concentration":
        value = f"{target.value:.0%}"
    elif spec.kind == "ratio":
        value = f"{target.value:,.2f}"
    else:
        value = f"{target.value:,.0f}"
    text = f"{name.capitalize()} came in at {value} for the {target.label}"
    against = []
    if spec.kind == "concentration":
        if c.get("previous") is not None:
            against.append(f"{c['previous']:.0%} the week before")
        if c.get("prior_year") is not None:
            against.append(f"{c['prior_year']:.0%} the same week last year")
    else:
        if c.get("wow_pct") is not None:
            against.append(f"{_pct(c['wow_pct'])} on the week before")
        if c.get("yoy_pct") is not None:
            against.append(f"{_pct(c['yoy_pct'])} on the same week last year")
        if c.get("vs_avg13_pct") is not None:
            against.append(f"{_pct(c['vs_avg13_pct'])} against the 13-week average")
    text += (": " + ", ".join(against) if against else "") + "."
    if c.get("verdict") and c.get("expected_source"):
        text += f" Against {c['expected_source']}, this week is {c['verdict']}."
    return text


def _day_label(iso: str) -> str:
    day = _dt.date.fromisoformat(iso)
    return f"{day.day} {day.strftime('%b %Y')}"


def build_kpis(probe: Any, dialect: Any, joins: Mapping[tuple[str, str], tuple[str, str]], kpi_specs: Sequence[KpiSpec], metric_specs: Sequence[Mapping[str, Any]], *, entity: str | None, window: int, week: str | None, target_week: Any, history_weeks: int, instructions: str, scope: str, budget: int, per_kpi: int) -> tuple[list[Any], list[Entity], str, int, list[str], Any]:
    """Every KPI as a metric section: the series computed by the source, the same context and verdict as any metric, its definition in words."""
    from .brief import MetricBrief, Week, _change_points, _context

    schema = probe.schema
    text = "\n".join(p for p in (instructions, scope) if p)
    notes: list[str] = []
    spent = 0

    def run(query: str) -> list[dict[str, Any]]:
        nonlocal spent
        if spent >= budget:
            raise _Budget
        spent += 1
        return probe.run(query)

    fact_name = next((str(s["fact"]) for s in metric_specs if s.get("fact") in schema.tables), None) or next(iter(probe.facts(text)), None)
    if fact_name is None or dialect.axis(fact_name) is None:
        return [], [], "", spent, ["no fact with a time axis for the KPIs"], target_week
    axis = dialect.axis(fact_name)
    measure0 = default_measure(schema, fact_name) or ""
    fact = {"table": fact_name, "date": axis, "measure": measure0, "aggregate": "sum"}
    grain = "week"
    briefs: list[Any] = []
    entities: list[Entity] = []
    entity_choice = ""
    lifecycle_cache: dict[tuple[str, str, int], tuple[list[LifecyclePoint], list[str]]] = {}
    produced: dict[tuple[str, str, str, int], str] = {}
    facts_with_axis: list[dict[str, Any]] = []
    try:
        if target_week is None:
            rows = run(dialect.daily(fact, [measure0] if measure0 else []))
            weeks0, _max_day = _weeks_of(rows, "v0")
            if week:
                wanted = _dt.date.fromisoformat(str(week))
                monday = (wanted - _dt.timedelta(days=wanted.weekday())).isoformat()
                target_week = next((w for w in weeks0 if w.start == monday), weeks0[-1] if weeks0 else None)
            else:
                target_week = weeks0[-1] if weeks0 else None
        if target_week is None:
            return [], [], "", spent, ["no complete week for the KPIs"], None
        target_index = period_of(_dt.date.fromisoformat(target_week.start), grain)
        first_index = target_index - history_weeks + 1
        if any(s.kind in _LIFECYCLE_KINDS for s in kpi_specs):
            for table in [fact_name] + [t for t in probe.facts(text) if t != fact_name][:4]:
                table_axis = dialect.axis(table)
                if table_axis is not None:
                    facts_with_axis.append({"table": table, "date": table_axis, "measure": default_measure(schema, table) or "", "aggregate": "sum"})
            entities = discover_entities(probe, dialect, facts_with_axis, joins, run)
    except _Budget:
        return [], [], "", spent, ["the budget ran out before the KPIs could be measured"], target_week

    for spec in kpi_specs:
        try:
            name, definition, weeks, extra, explanations, kpi_notes = spec.name, "", [], {}, [], []
            if spec.kind in _LIFECYCLE_KINDS:
                chosen, why = resolve_entity(spec, entities, entity)
                if chosen is None:
                    notes.append(f"{spec.name}: {why}")
                    continue
                if not entity_choice:
                    entity_choice = why
                span = spec.window if spec.window else window
                same_count = (spec.kind, chosen.fact, chosen.column, span)
                if same_count in produced:
                    notes.append(f"{spec.name}: the same count as '{produced[same_count]}' ({why.split('; say entity')[0]}); not repeated")
                    continue
                produced[same_count] = spec.name
                entity_fact = next((f for f in facts_with_axis if f["table"] == chosen.fact), fact)
                cache_key = (chosen.fact, chosen.column, span)
                if cache_key in lifecycle_cache:
                    points, missing = lifecycle_cache[cache_key]  # new, active, retained and churned all come from the same growth accounting
                else:
                    periods = list(range(max(first_index, target_index - min(history_weeks, 60) + 1), target_index + 1))
                    if dialect.kind != "lakehouse":
                        # a model answers one period per query; keep the most recent weeks the budget can pay for
                        affordable = max(4, budget - spent - 1)
                        if affordable < len(periods):
                            kpi_notes.append(f"history limited to {affordable} weeks by the query budget")
                            periods = periods[-affordable:]
                    points, missing = skipped_periods(lifecycle_series(probe, dialect, entity_fact, chosen.column, grain, span, run, periods=periods))
                    lifecycle_cache[cache_key] = (points, missing)
                if missing:
                    kpi_notes.append(f"{len(missing)} week(s) the model would not answer were left out: {', '.join(missing[:3])}")
                weeks = [Week(p.start, p.value(spec.kind), p.active, 7) for p in points if first_index <= p.index <= target_index]
                name = f"{spec.kind} {chosen.name}"
                definition = kpi_definition(spec, entity=chosen, grain=grain, how=f"{chosen.column} on {chosen.fact}")
                extra = {"growth": [p for p in points if p.index <= target_index][-26:], "entity": chosen}
            elif spec.kind == "ratio":
                num, num_filters, missing_n = _side(spec.numerator, probe, joins, instructions, scope)
                den_words, _den_pairs = measure_phrase_filters(spec.denominator)
                rows_denominator = den_words.casefold() in _ROW_WORDS
                den, den_filters, missing_d = (None, [], "") if rows_denominator else _side(spec.denominator, probe, joins, instructions, scope)
                if not num.get("measure") or (den is not None and not den.get("measure")):
                    notes.append(f"{spec.name}: could not read the measures ({spec.numerator} / {spec.denominator})")
                    continue
                fact_n = {"table": num["fact"], "date": dialect.axis(num["fact"]), "measure": num["measure"], "aggregate": "sum"}
                if den is not None and den["fact"] == num["fact"] and den_filters == num_filters:
                    rows = run(dialect.daily(fact_n, [num["measure"], den["measure"]], num_filters))
                    weeks_n, max_day = _weeks_of(rows, "v0")
                    weeks_d, _ = _weeks_of(rows, "v1", max_day)
                else:
                    rows = run(dialect.daily(fact_n, [num["measure"]], num_filters))
                    weeks_n, max_day = _weeks_of(rows, "v0")
                    if rows_denominator or den is None:
                        weeks_d = [Week(w.start, float(w.rows), w.rows, w.days) for w in weeks_n]
                    else:
                        fact_d = {"table": den["fact"], "date": dialect.axis(den["fact"]), "measure": den["measure"], "aggregate": "sum"}
                        weeks_d, _ = _weeks_of(run(dialect.daily(fact_d, [den["measure"]], den_filters)), "v0", max_day)
                by_start = {w.start: w for w in weeks_d}
                weeks = [Week(w.start, (w.value / by_start[w.start].value) if by_start[w.start].value else 0.0, w.rows, w.days) for w in weeks_n if w.start in by_start]
                definition = kpi_definition(spec, grain=grain)
                extra = {"numerator": weeks_n, "denominator": weeks_d, "numerator_name": num["name"], "denominator_name": "rows" if rows_denominator else den["name"]}
                if missing_n or missing_d:
                    kpi_notes.append(f"no grouping matched {missing_n or missing_d}")
            elif spec.kind == "crossing":
                left, left_filters, missing_l = _side(spec.left, probe, joins, instructions, scope)
                if not left.get("measure"):
                    notes.append(f"{spec.name}: could not read {spec.left}")
                    continue
                threshold: float | None
                try:
                    threshold = float(spec.right.replace(",", ""))
                except ValueError:
                    threshold = None
                fact_l = {"table": left["fact"], "date": dialect.axis(left["fact"]), "measure": left["measure"], "aggregate": "sum"}
                weeks_l, max_day = _weeks_of(run(dialect.daily(fact_l, [left["measure"]], left_filters)), "v0")
                if not weeks_l:
                    notes.append(f"{spec.name}: no rows match {spec.left}; check the value against the source (case and spelling count)")
                    continue
                if threshold is not None:
                    weeks_r = [Week(w.start, threshold, w.rows, w.days) for w in weeks_l]
                    right_name = f"{threshold:,.0f}"
                else:
                    right, right_filters, missing_r = _side(spec.right, probe, joins, instructions, scope)
                    if not right.get("measure"):
                        notes.append(f"{spec.name}: could not read {spec.right}")
                        continue
                    fact_r = {"table": right["fact"], "date": dialect.axis(right["fact"]), "measure": right["measure"], "aggregate": "sum"}
                    weeks_r, _ = _weeks_of(run(dialect.daily(fact_r, [right["measure"]], right_filters)), "v0", max_day)
                    if not weeks_r:
                        notes.append(f"{spec.name}: no rows match {spec.right}; check the value against the source (case and spelling count)")
                        continue
                    right_name = spec.right
                    if missing_r:
                        kpi_notes.append(f"no grouping matched {missing_r}")
                if missing_l:
                    kpi_notes.append(f"no grouping matched {missing_l}")
                index_l = {period_of(_dt.date.fromisoformat(w.start), grain): w.value for w in weeks_l}
                index_r = {period_of(_dt.date.fromisoformat(w.start), grain): w.value for w in weeks_r}
                found = [(period_start(i, grain).isoformat(), "left" if "left overtook" in t else "right") for i, t in crossings(index_l, index_r) if i <= target_index]
                weeks = [w for w in weeks_l if period_of(_dt.date.fromisoformat(w.start), grain) <= target_index]
                name = f"{spec.left} against {right_name}"
                definition = kpi_definition(spec, grain=grain)
                gap = index_l.get(target_index, 0.0) - index_r.get(target_index, 0.0)
                if found:
                    when, side = found[-1]
                    who, whom = (spec.left, right_name) if side == "left" else (right_name, spec.left)
                    explanations.append(f"{who.capitalize()} overtook {whom} in the week of {_day_label(when)}; the gap this week is {gap:+,.0f}.")
                else:
                    explanations.append(f"No crossing in the last {len(weeks)} weeks; {spec.left} is {'above' if gap > 0 else 'below' if gap < 0 else 'level with'} {right_name} by {abs(gap):,.0f} this week.")
                extra = {"left": weeks_l, "right": weeks_r, "left_name": spec.left, "right_name": right_name, "crossings": found}
            elif spec.kind == "concentration":
                measure_spec, _f, _m = _side(spec.measure_words, probe, joins, instructions, scope)
                path = find_path(schema, measure_spec["fact"], joins, spec.grouping_words) if measure_spec.get("fact") else None
                if not measure_spec.get("measure") or path is None:
                    notes.append(f"{spec.name}: could not read the measure or the grouping")
                    continue
                fact_c = {"table": measure_spec["fact"], "date": dialect.axis(measure_spec["fact"]), "measure": measure_spec["measure"], "aggregate": "sum"}
                until = period_start(target_index + 1, grain) - _dt.timedelta(days=1)
                points = [p for p in concentration_series(probe, dialect, fact_c, path, grain, spec.top, run, last=max(60, min(history_weeks, 120)), until=until) if p.index <= target_index]
                weeks = [Week(p.start, p.share, p.groups, 7) for p in points]
                now = next((p for p in points if p.index == target_index), None)
                before = next((p for p in points if p.index == target_index - 52), None)
                entrants = [leader for leader in now.leaders if before is not None and leader not in before.leaders] if now else []
                name = f"share of the top {spec.top} {humanize_column(str(path['column']))} in {measure_spec['name']}"
                definition = kpi_definition(spec, grain=grain)
                if now:
                    explanations.append(f"The top {spec.top} ({', '.join(now.leaders)}) hold {now.share:.0%} of {measure_spec['name']} this week" + (f" against {before.share:.0%} a year ago." if before else ".") + (f" New in the top {spec.top} since a year ago: {', '.join(entrants)}." if entrants else ""))
                    if now.groups <= spec.top:
                        explanations.append(f"The grouping has only {now.groups} group{'s' if now.groups != 1 else ''} this week, so the top {spec.top} is all of it; a finer grouping would say more.")
                extra = {"shares": points[-52:], "leaders_now": now.leaders if now else (), "leaders_before": before.leaders if before else (), "entrants": tuple(entrants)}
            else:
                continue
            if not weeks:
                notes.append(f"{name}: no weekly series could be built")
                continue
            index = next((i for i, w in enumerate(weeks) if w.start == target_week.start), len(weeks) - 1)
            ctx = _context(weeks, index, "sum")
            shifts = _change_points(weeks[: index + 1])
            headline = _kpi_headline(spec, name, weeks[index], ctx)
            briefs.append(MetricBrief(name, fact_name, spec.kind, (), "sum", tuple(weeks), weeks[index], ctx, tuple(shifts), {}, "", None, {}, tuple(explanations), headline, tuple(kpi_notes), kind=spec.kind, definition=definition, extra=extra))
        except _Budget:
            notes.append(f"the budget of {budget} queries ran out before {spec.name}")
            break
        except ValueError as exc:
            notes.append(f"{spec.name}: {exc}")
    return briefs, entities, entity_choice, spent, notes, target_week
