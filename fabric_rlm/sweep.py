"""What moved: a budgeted, deterministic sweep over a source's facts, measures, groupings and periods.

Give :func:`what_moved` a lakehouse (``LakehouseSource``) or a semantic model
(``SemanticModel``) and it measures, with the source's own query engine, how
every measure of every fact moved between the periods the time axis
supports: the latest month against the month before, the same month a year
earlier, and the latest complete year against the previous one. Material
movements are decomposed by every grouping the joins or relationships
reach, each decomposition is classified (one group carries the change, a
few do, the groups moved in proportion to their size so the grouping
explains nothing, or groups moved both ways), and the leading group of the
most concentrated decomposition is drilled one level further.

The sweep runs in two phases so the budget goes where it matters: first
every total movement of every fact is measured, one query each, then the
material ones are decomposed largest relative change first until the
budget runs out. Two measures that move identically (an extended amount
that equals the sales amount) are collapsed. Every figure carries the
query that produced it and an independent per-period query that recomputes
it; :func:`verify_sweep` runs those. No model is involved: the numbers are
the source's own, and ``Sweep.to_html()`` renders them as a dashboard with
trend, waterfall and driver scatter charts. :mod:`fabric_rlm.reports`
builds a report of one kind (trend, root cause, recap, top movers) from a
request written in plain words.

Lakehouses are queried in SQL (DuckDB over OneLake, the joins written from
key columns and the instructions); semantic models in DAX (the model's
relationships do the joining, periods filter the related date table).
"""

from __future__ import annotations

import datetime as _dt
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .data_agent_review import (
    AgentDataSource,
    AgentSnapshot,
    LakehouseExecutor,
    ReviewContext,
    SourceSchema,
    _BUSINESS_DATE,
    _Joins,
    _LOCAL_EXCLUDED,
    _MONTH_COLUMN,
    _SECONDARY_DATE,
    _TIME_COLUMN,
    _TIME_TYPE,
    _YEAR_COLUMN,
    _channel_measure,
    _date_candidates,
    _date_join,
    _heuristic_joins,
    _is_time_column,
    _iso_date,
    _joins_from_instructions,
    _max_date_sql,
    _measure_columns,
    _month_expr,
    _month_name,
    _path_table,
    _paths_by_role,
    _period_condition,
    _q,
    _rows,
    _scoped_facts,
    _sql_literal,
    _stamp,
    _tables_named_in,
    _time_join,
    _year_expr,
    attribute_paths,
    build_vocabulary,
    excluded_terms,
    humanize_column,
    schema_from_tables,
)

__all__ = [
    "Comparison",
    "Decomposition",
    "LakehouseProbe",
    "Movement",
    "Point",
    "SemanticModelProbe",
    "Sweep",
    "SweepFinding",
    "Takeaway",
    "sweep",
    "verify_sweep",
    "what_moved",
]


class _Budget(Exception):
    """The query budget is spent."""


_AVERAGED = re.compile(r"(score|rating|_rate$|rate$|pct|percent|ratio|index|avg|average|mean|unit[_ ]?price|list[_ ]?price|^price$|unit[_ ]?cost|standard[_ ]?cost)", re.IGNORECASE)
_GROUP_LIMIT = 500  # groups a decomposition query returns, largest absolute change first
_INFORMATIVE = frozenset({"single", "concentrated", "offsetting", "broad"})
_COMPARISON_KINDS = ("year", "month", "same_month_prior_year")


def _aggregate_for(measure: str) -> str:
    """``avg`` for a score, rating, rate or percentage, which cannot be summed; ``sum`` for everything else."""
    return "avg" if _AVERAGED.search(measure) else "sum"


_PLACEHOLDER = re.compile(r"^\W*(n/?a|not applicable|not available|not assigned|not specified|unassigned|unspecified|unknown|none|null|blank|missing|other|others|\?+|-+)\W*$", re.IGNORECASE)
_GROUPING_HINT = re.compile(r"(name|country|region|group|category|segment|type|class|status|city|state|line|plant|channel|brand|tier|family|model|colou?r|method|source|kind|level|stage|priority|medium)", re.IGNORECASE)


def _is_placeholder(value: Any) -> bool:
    """A group value that stands for the absence of one: blank, [Not Applicable], Unknown, N/A."""
    return value is None or bool(_PLACEHOLDER.match(str(value)))


def _placeholder_lead(d: "Decomposition") -> bool:
    return bool(d.groups) and _is_placeholder(d.groups[0].group)


def _group_phrase(value: Any, path: Mapping[str, Any] | None) -> str:
    """How a reader hears a group: its value, or for a placeholder what the placeholder means (``rows with no business type recorded ([Not Applicable])``)."""
    if _is_placeholder(value):
        return f"rows with no {_word(path)} recorded ({_label(value)})"
    return _label(value)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


def _period_label(period: Mapping[str, Any]) -> str:
    """``December 2013``, ``2013``, or the label a date range carries (``week of 2 Dec 2013``)."""
    if period.get("label"):
        return str(period["label"])
    if period.get("start"):
        return f"{period['start']} to {period.get('end', '')}"
    return _month_name(period)


@dataclass(frozen=True)
class Comparison:
    """Two periods the time axis can express: ``kind`` is ``month``, ``same_month_prior_year``, ``year``, ``week``, ``same_week_prior_year`` or ``custom``.

    A period is ``{"year": 2013}``, ``{"year": 2013, "month": 12}``, or a
    date range ``{"start": "2013-12-02", "end": "2013-12-09", "label": ...}``
    with the end exclusive (a week, a quarter, any span with a day grain).
    """

    kind: str
    before: Mapping[str, Any]
    after: Mapping[str, Any]

    @property
    def label(self) -> str:
        return f"{_period_label(self.before)} to {_period_label(self.after)}"


@dataclass(frozen=True)
class Movement:
    """One measured change: a fact's measure between two periods, at the total level or for one group of one grouping path."""

    fact: str
    measure: str
    comparison: Comparison
    before_value: float
    after_value: float
    before_rows: int
    after_rows: int
    query: str
    verification: Mapping[str, str]  # independent per-period queries: {"before": query, "after": query}
    path: Mapping[str, Any] | None = None  # the grouping path; None at the total level
    group: Any = None  # the group's value; None at the total level
    parent: tuple[tuple[str, Any], ...] = ()  # (column, value) of the groups above this one in a drill
    aggregate: str = "sum"  # sum, or avg for a score or a rate
    flags: tuple[str, ...] = ()  # at the total level: coverage, incomplete period, small base

    @property
    def trusted(self) -> bool:
        """Whether both periods are complete enough for the movement to mean something about the business."""
        return not any(flag.startswith(("incomplete:", "coverage:")) for flag in self.flags)

    @property
    def delta(self) -> float:
        return self.after_value - self.before_value

    @property
    def pct(self) -> float | None:
        return self.delta / abs(self.before_value) if self.before_value else None

    @property
    def rows_pct(self) -> float | None:
        return (self.after_rows - self.before_rows) / self.before_rows if self.before_rows else None

    @property
    def column(self) -> str | None:
        return str(self.path["column"]) if self.path else None

    def material(self, min_pct: float) -> bool:
        return self.pct is not None and abs(self.pct) >= min_pct


@dataclass(frozen=True)
class Decomposition:
    """A movement split by one grouping path: the groups in the same direction first, largest first, then the largest opposite ones."""

    parent: Movement
    path: Mapping[str, Any]
    groups: tuple[Movement, ...]
    concentration: str  # single | concentrated | proportional | broad | offsetting | none
    explained: float  # share of the parent's change carried by the top three groups moving the same way
    verification: Mapping[str, str] = field(default_factory=dict)  # independent per-period queries recomputing every group in ``groups``
    size: int = 0  # groups the path has, as far as the query returned them

    def share_of_change(self, group: Movement) -> float | None:
        return group.delta / self.parent.delta if self.parent.delta else None

    def share_of_base(self, group: Movement) -> float | None:
        return group.before_value / self.parent.before_value if self.parent.before_value else None


@dataclass(frozen=True)
class SweepFinding:
    """A material movement with its decompositions (most concentrated first), the drills into its leading group (best first), and the flags a reader should see first."""

    movement: Movement
    decompositions: tuple[Decomposition, ...] = ()
    drill: tuple[Decomposition, ...] = ()
    flags: tuple[str, ...] = ()

    @property
    def best(self) -> Decomposition | None:
        return self.decompositions[0] if self.decompositions else None

    @property
    def lead_drill(self) -> Decomposition | None:
        """The drill worth showing: the best one when it says something, else None."""
        return self.drill[0] if self.drill and self.drill[0].concentration in _INFORMATIVE else None

    @property
    def trusted(self) -> bool:
        return not any(flag.startswith(("incomplete:", "coverage:")) for flag in self.flags)

    @property
    def anchor(self) -> str:
        m = self.movement
        return re.sub(r"[^a-z0-9]+", "-", f"f-{m.fact}-{m.measure}-{m.comparison.kind}-{m.comparison.after.get('year', '')}-{m.comparison.after.get('month', '')}".casefold()).strip("-")


@dataclass(frozen=True)
class Takeaway:
    """One sentence a reader can act on, with the movement behind it and whether its periods were complete."""

    text: str
    movement: Movement
    finding: SweepFinding | None = None
    trusted: bool = True

    @property
    def anchor(self) -> str | None:
        return self.finding.anchor if self.finding is not None else None


@dataclass(frozen=True)
class Point:
    """One month of a measure's series: the value (a sum, or an average for a score or a rate) and the rows behind it."""

    year: int
    month: int
    value: float
    rows: int


@dataclass(frozen=True)
class Sweep:
    """What moved in one source, with the ledger of every figure and the queries behind them."""

    source: str
    kind: str  # lakehouse or semantic_model
    findings: tuple[SweepFinding, ...]
    ledger: tuple[Movement, ...]
    queries: int
    budget: int
    years: tuple[int, ...] = ()  # the complete years the sweep compared
    notes: tuple[str, ...] = ()
    words: Mapping[str, str] = field(default_factory=dict)  # "fact|measure" -> the business phrase used in the narrative
    series: Mapping[str, tuple[Point, ...]] = field(default_factory=dict)  # "fact|measure" -> monthly points over the compared years
    mismatches: tuple[str, ...] = ()  # figures that did not recompute, after verify_sweep
    recomputed: int = 0  # figures recomputed by verify_sweep
    verified: bool = False
    elapsed: float = 0.0
    collapsed: tuple[str, ...] = ()  # "fact|measure" keys measured but not decomposed because they move like another measure

    def phrase(self, movement: Movement) -> str:
        return self.words.get(f"{movement.fact}|{movement.measure}", f"{movement.fact} {movement.measure}")

    def headline(self, movement: Movement) -> str:
        direction = "fell" if movement.delta < 0 else "rose"
        averaged = " on average" if movement.aggregate == "avg" else ""
        return f"{self.phrase(movement).capitalize()} {direction} {abs(movement.pct or 0):.1%}{averaged} {movement.comparison.label} ({_num(movement.before_value)} to {_num(movement.after_value)})."

    def lines(self) -> list[str]:
        """The findings as plain sentences, every number from the ledger."""
        lines: list[str] = []
        for finding in self.findings:
            lines.append(self.headline(finding.movement))
            for flag in finding.flags:
                lines.append(f"  Note: {flag}")
            best = finding.best
            if best is not None:
                lines.append(f"  By {_word(best.path)}: {_concentration_sentence(best)}")
                lead = finding.lead_drill
                if lead is not None:
                    lines.append(f"  Within {_label(lead.parent.group)}, by {_word(lead.path)}: {_concentration_sentence(lead)}")
                elif finding.drill:
                    lines.append(f"  Within {_label(finding.drill[0].parent.group)}, nothing stands out by {_words([d.path for d in finding.drill])}.")
            others = [d for d in finding.decompositions[1:] if d.concentration in {"single", "concentrated"} and d.groups]
            if others:
                lines.append("  Also concentrated by " + "; ".join(f"{_word(d.path)} ({_leader_sentence(d)})" for d in others[:2]))
        return lines

    def summary(self) -> str:
        text = f"{len(self.findings)} material movement(s) from {len(self.ledger)} measured figures in {self.queries} of {self.budget} queries"
        if self.verified:
            text += f"; {self.recomputed} figure(s) recomputed independently, {len(self.mismatches)} mismatch(es)"
        return text + "."

    def verification_statement(self) -> str:
        """One sentence that the header badge and the footer both follow, so they cannot disagree."""
        total = len(self.ledger)
        if not self.verified:
            return f"Every one of the {total:,} figures was computed by the source with the query kept next to it; none was recomputed in this run."
        rest = total - self.recomputed
        text = f"Every one of the {total:,} figures was computed by the source with the query kept next to it. {self.recomputed:,} of them, the reported movements and the groups of their leading decompositions, were recomputed by independent per-period queries"
        text += f" with {len(self.mismatches)} mismatch{'es' if len(self.mismatches) != 1 else ''}." if self.mismatches else " and all matched."
        if rest > 0:
            text += f" The other {rest:,} carry their recomputation query but were not rerun."
        return text

    def takeaways(self, limit: int = 3) -> list[Takeaway]:
        """The movements a reader should know first: complete periods only, the largest relative change per measure, years before months, one per measure."""
        by_finding = {id(f.movement): f for f in self.findings}
        candidates: list[tuple[tuple[Any, ...], Movement]] = []
        seen: set[str] = set()
        priority = {"year": 0, "same_month_prior_year": 1, "same_week_prior_year": 1, "custom": 1, "month": 2, "week": 2}
        for m in sorted((m for m in self.ledger if m.path is None and m.pct is not None and m.trusted and abs(m.pct) >= 0.05), key=lambda m: (priority.get(m.comparison.kind, 3), -abs(m.pct or 0))):
            key = f"{m.fact}|{m.measure}"
            if key in seen or key in self.collapsed:
                continue
            seen.add(key)
            candidates.append(((priority.get(m.comparison.kind, 3), any(flag.startswith("small base:") for flag in m.flags), -abs(m.pct or 0)), m))
        out: list[Takeaway] = []
        for _key, m in sorted(candidates, key=lambda item: item[0])[:limit]:
            finding = by_finding.get(id(m))
            out.append(Takeaway(self._takeaway_text(m, finding), m, finding, True))
        return out

    def set_aside(self) -> list[Takeaway]:
        """Movements whose periods were not complete: shown, never interpreted."""
        by_finding = {id(f.movement): f for f in self.findings}
        out: list[Takeaway] = []
        seen: set[str] = set()
        for m in self.ledger:
            if m.path is not None or m.trusted:
                continue
            key = f"{m.fact}|{m.measure}|{m.comparison.label}"
            if key in seen or f"{m.fact}|{m.measure}" in self.collapsed:
                continue
            seen.add(key)
            reason = next((flag for flag in m.flags if flag.startswith(("incomplete:", "coverage:"))), "")
            out.append(Takeaway(f"{self.phrase(m).capitalize()} {'fell' if m.delta < 0 else 'rose'} {abs(m.pct or 0):.0%} {m.comparison.label}, but {reason.split(': ', 1)[-1]}", m, by_finding.get(id(m)), False))
        return out

    def _takeaway_text(self, m: Movement, finding: SweepFinding | None) -> str:
        direction = "fell" if m.delta < 0 else "rose"
        averaged = " on average" if m.aggregate == "avg" else ""
        text = f"{self.phrase(m).capitalize()} {direction} {abs(m.pct or 0):.0%}{averaged} {m.comparison.label}, {_num(m.before_value)} to {_num(m.after_value)}"
        best = finding.best if finding is not None else None
        if best is not None and best.groups and m.aggregate == "sum":
            lead = best.groups[0]
            share, base = best.share_of_change(lead), best.share_of_base(lead)
            if best.concentration in {"single", "concentrated"} and share is not None and base is not None:
                who = _group_phrase(lead.group, best.path)
                text += f", led by {who} ({share:.0%} of the change, from almost no base before)" if base < 0.01 else f", led by {who} ({share:.0%} of the change on {base:.0%} of the base)"
            elif best.concentration == "proportional":
                text += f", spread across {_word(best.path)} groups in proportion to their size"
            elif best.concentration == "offsetting":
                opposite = [g for g in best.groups if g.delta * m.delta < 0]
                text += f", with {_group_phrase(lead.group, best.path)} moving one way and {_group_phrase(opposite[0].group, best.path) if opposite else 'others'} the other"
            elif best.concentration == "broad":
                text += f", spread broadly across {_word(best.path)} groups"
        if any(flag.startswith("volume:") for flag in m.flags):
            text += "; the row count moved as much as the value, so this is volume, not a change in rate"
        small = next((flag for flag in m.flags if flag.startswith("small base:")), None)
        if small:
            text += f"; on a small base ({small.split(': ', 1)[1]})"
        return text + "."

    def narrative(self) -> str:
        """The picture in a short paragraph: what was compared, what moved and where, what was set aside. Every number is from the ledger."""
        totals = [m for m in self.ledger if m.path is None and f"{m.fact}|{m.measure}" not in self.collapsed]
        measures = sorted({self.phrase(m) for m in totals})
        if not totals:
            return "No measure could be compared across the periods the time axis supports."
        trusted = [m for m in totals if m.trusted]
        material = [m for m in trusted if abs(m.pct or 0) >= 0.05]
        parts = [f"{len(measures)} measure{'s' if len(measures) != 1 else ''} ({', '.join(measures)}) were compared over {len({m.comparison.label for m in totals})} period pairs; {len(material)} of the {len(trusted)} comparisons on complete periods moved by 5% or more."]
        takeaways = self.takeaways(3)
        if takeaways:
            first = takeaways[0].movement
            parts.append(f"The largest is {self.phrase(first)}, {'down' if first.delta < 0 else 'up'} {abs(first.pct or 0):.0%} {first.comparison.label}.")
        concentrated = [f for f in self.findings if f.trusted and f.best is not None and f.best.concentration in {"single", "concentrated"} and f.best.groups]
        if concentrated:
            leaders = {f"{_group_phrase(f.best.groups[0].group, f.best.path)}" + ("" if _is_placeholder(f.best.groups[0].group) else f" ({_word(f.best.path)})") for f in concentrated[:3]}
            parts.append(f"Where a movement is concentrated, it sits with {', '.join(sorted(leaders))}.")
        aside = self.set_aside()
        if aside:
            parts.append(f"{len(aside)} movement{'s' if len(aside) != 1 else ''} involve{'s' if len(aside) == 1 else ''} an incomplete period and {'is' if len(aside) == 1 else 'are'} set aside rather than read as business change.")
        return " ".join(parts)

    def to_markdown(self) -> str:
        head = [self.summary()]
        takeaways = self.takeaways()
        if takeaways:
            head.append("")
            head.append("Three things to know:" if len(takeaways) >= 3 else "To know:")
            head.extend(f"{i}. {t.text}" for i, t in enumerate(takeaways, start=1))
        head.extend(["", self.narrative(), ""])
        body = "\n".join(("- " + line[2:] if line.startswith("  ") else f"- **{line}**") for line in self.lines())
        notes = "\n".join(f"- {n}" for n in self.notes)
        return "\n".join(part for part in ("\n".join(head), body, notes) if part)

    def to_html(self) -> str:
        """The dashboard: headline cards, trends, and for every finding a waterfall, a driver scatter, the tables and the queries."""
        from .sweep_dashboard import render

        return render(self)

    def save(self, path: str) -> str:
        """Write the dashboard as a complete HTML page; returns the path."""
        from .sweep_dashboard import document

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(document(self))
        return path


def _num(value: float) -> str:
    if abs(value) >= 1000 or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _label(value: Any) -> str:
    return "(blank)" if value is None else str(value)


def _word(path: Mapping[str, Any] | None) -> str:
    return humanize_column(str(path["column"])) if path else "total"


def _words(paths: Sequence[Mapping[str, Any]]) -> str:
    names = [_word(p) for p in paths]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


def _leader_sentence(d: Decomposition) -> str:
    if not d.groups:
        return "no groups moved"
    lead = d.groups[0]
    share = d.share_of_change(lead)
    return f"{_label(lead.group)} carries {share:.0%}" if share is not None else _label(lead.group)


def _shares(d: Decomposition, groups: Sequence[Movement]) -> str:
    parts: list[str] = []
    for index, g in enumerate(groups[:3]):
        change, base = d.share_of_change(g), d.share_of_base(g)
        if change is None or base is None:
            parts.append(_label(g.group))
        elif index == 0:
            parts.append(f"{_label(g.group)} carries {change:.0%} of the change on {base:.0%} of the base")
        else:
            parts.append(f"{_label(g.group)} ({change:.0%} on {base:.0%})")
    if len(parts) == 3:
        return f"{parts[0]}, then {parts[1]} and {parts[2]}"
    return ", then ".join(parts)


def _concentration_sentence(d: Decomposition) -> str:
    if not d.groups:
        return "no group moved the same way."
    same = [g for g in d.groups if g.delta * d.parent.delta > 0]
    opposite = [g for g in d.groups if g.delta * d.parent.delta < 0]
    if d.parent.aggregate == "avg":
        moves = "; ".join(f"{_label(g.group)} {_num(g.before_value)} to {_num(g.after_value)}" for g in d.groups[:3])
        return f"largest moves in the average: {moves}."
    shares = _shares(d, same)
    return {
        "single": f"one group: {shares}.",
        "concentrated": f"concentrated: {shares}; the top three explain {d.explained:.0%}.",
        "proportional": f"in proportion to size, nothing stands out: {shares}.",
        "broad": f"broad, spread across groups: {shares}; the top three explain {d.explained:.0%}.",
        "offsetting": f"offsetting moves: {shares}, while {', '.join(_label(o.group) for o in opposite[:2])} moved the other way.",
    }.get(d.concentration, shares + ".")


# --------------------------------------------------------------------------- #
# Query dialects: the same sweep rendered as SQL for a lakehouse and as DAX for a semantic model
# --------------------------------------------------------------------------- #

_Filters = Sequence[tuple[Mapping[str, Any], Any]]  # (grouping path, value) pairs that narrow a query to one group


class _Sql:
    """DuckDB SQL over a lakehouse catalog; the joins are written out from key columns."""

    kind = "lakehouse"

    def __init__(self, schema: SourceSchema, joins: Mapping[tuple[str, str], tuple[str, str]]) -> None:
        self.schema = schema
        self.joins = dict(joins)

    def axis(self, table: str) -> dict[str, Any] | None:
        return _date_join(self.schema, table, self.joins)

    def _from(self, fact: Mapping[str, Any], joins: _Joins) -> str:
        time_join = _time_join(fact["date"])
        return f"FROM {fact['table']} f" + (f" {time_join}" if time_join else "") + "".join(" " + c for c in joins.clauses)

    def _expr(self, fact: Mapping[str, Any]) -> str:
        m = f"f.{_q(fact['measure'])}"
        return f"AVG({m})" if fact["aggregate"] == "avg" else f"SUM({m})"

    def _case(self, fact: Mapping[str, Any], condition: str) -> str:
        m = f"f.{_q(fact['measure'])}"
        if fact["aggregate"] == "avg":
            return f"AVG(CASE WHEN {condition} THEN {m} END)"
        return f"SUM(CASE WHEN {condition} THEN {m} ELSE 0 END)"

    def day(self, fact: Mapping[str, Any]) -> str:
        """The time axis at day grain: the date table's day column, its yyyymmdd key, or the fact's own date or timestamp."""
        dt = fact["date"]
        if dt.get("period"):
            raise ValueError("a period written as text has no day grain")
        if dt.get("date_table") and not dt.get("timestamp"):
            column = _day_column(self.schema, str(dt["date_table"]))
            if column:
                return f"CAST(d.{_q(column)} AS DATE)"
            return f"CAST(strptime(CAST(d.{_q(dt['date_key'])} AS VARCHAR), '%Y%m%d') AS DATE)"
        return f"CAST({_stamp(dt, 'duckdb')} AS DATE)"

    def _period(self, fact: Mapping[str, Any], period: Mapping[str, Any]) -> str:
        if period.get("start"):
            day = self.day(fact)
            return f"{day} >= DATE '{period['start']}' AND {day} < DATE '{period['end']}'"
        return _period_condition(fact, period, "duckdb")

    def _span(self, fact: Mapping[str, Any], years: Sequence[int]) -> str:
        return f"{_year_expr(fact['date'], 'duckdb')} IN ({', '.join(str(int(y)) for y in years)})"

    def daily(self, fact: Mapping[str, Any], measures: Sequence[str]) -> str:
        """Rows and the sum of every measure by day over the whole fact."""
        day = self.day(fact)
        values = ", ".join(f"SUM(f.{_q(m)}) AS v{i}" for i, m in enumerate(measures))
        join = _time_join(fact["date"])
        return f"SELECT {day} AS day, COUNT(*) AS n, {values} FROM {fact['table']} f" + (f" {join}" if join else "") + " GROUP BY 1 ORDER BY 1"

    def _filter(self, joins: _Joins, path: Mapping[str, Any], value: Any) -> str:
        ref = joins.ref(path)
        return f"{ref} IS NULL" if value is None else f"{ref} = {_sql_literal(value)}"

    def _members(self, label: str, values: Sequence[Any]) -> str:
        members = [v for v in values if v is not None]
        parts = [f"{label} IN ({', '.join(_sql_literal(v) for v in members)})"] if members else []
        if any(v is None for v in values):
            parts.append(f"{label} IS NULL")
        return "(" + " OR ".join(parts) + ")"

    def _both(self, fact: Mapping[str, Any], comparison: Comparison) -> tuple[str, str, str]:
        before, after = self._period(fact, comparison.before), self._period(fact, comparison.after)
        return before, after, f"(({before}) OR ({after}))"

    def series(self, fact: Mapping[str, Any], measures: Sequence[str]) -> str:
        """Rows, and the sum of every measure, by year and month over the whole fact."""
        dt = fact["date"]
        year, month = _year_expr(dt, "duckdb"), _month_expr(dt, "duckdb")
        values = ", ".join(f"SUM(f.{_q(m)}) AS v{i}" for i, m in enumerate(measures))
        join = _time_join(dt)
        keys = f"{year}" + (f", {month}" if month else "")
        return f"SELECT {year} AS year, {month or 'NULL'} AS month, COUNT(*) AS n, {values} FROM {fact['table']} f" + (f" {join}" if join else "") + f" GROUP BY {keys} ORDER BY {keys}"

    def months(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [dict(r) for r in rows]

    def top_groups(self, fact: Mapping[str, Any], path: Mapping[str, Any], years: Sequence[int], limit: int) -> str:
        """The largest groups of a path by the measure over the compared years."""
        joins = _Joins()
        label = joins.ref(path)
        return f"SELECT {label} AS label, {self._expr(fact)} AS value {self._from(fact, joins)} WHERE {self._span(fact, years)} GROUP BY {label} ORDER BY ABS(value) DESC NULLS LAST LIMIT {limit}"

    def grouped_series(self, fact: Mapping[str, Any], path: Mapping[str, Any], years: Sequence[int], values: Sequence[Any]) -> str:
        """The measure by year, month and group, for the listed groups."""
        dt = fact["date"]
        joins = _Joins()
        label = joins.ref(path)
        year, month = _year_expr(dt, "duckdb"), _month_expr(dt, "duckdb")
        keys = f"{year}, {month}, {label}" if month else f"{year}, {label}"
        return f"SELECT {year} AS year, {month or 'NULL'} AS month, {label} AS label, COUNT(*) AS n, SUM(f.{_q(fact['measure'])}) AS v0 {self._from(fact, joins)} WHERE {self._span(fact, years)} AND {self._members(label, values)} GROUP BY {keys} ORDER BY {keys}"

    def max_date(self, fact: Mapping[str, Any]) -> str:
        return _max_date_sql(fact)

    def movement(self, fact: Mapping[str, Any], comparison: Comparison, filters: _Filters) -> str:
        joins = _Joins()
        conditions = [self._filter(joins, p, v) for p, v in filters]
        before, after, either = self._both(fact, comparison)
        where = " AND ".join([either, *conditions])
        return f"SELECT {self._case(fact, after)} AS after_value, {self._case(fact, before)} AS before_value, SUM(CASE WHEN {after} THEN 1 ELSE 0 END) AS after_rows, SUM(CASE WHEN {before} THEN 1 ELSE 0 END) AS before_rows {self._from(fact, joins)} WHERE {where}"

    def total(self, fact: Mapping[str, Any], period: Mapping[str, Any], filters: _Filters) -> str:
        joins = _Joins()
        conditions = [self._filter(joins, p, v) for p, v in filters]
        return f"SELECT {self._expr(fact)} AS value {self._from(fact, joins)} WHERE " + " AND ".join([f"({self._period(fact, period)})", *conditions])

    def grouped(self, fact: Mapping[str, Any], comparison: Comparison, path: Mapping[str, Any], filters: _Filters) -> str:
        joins = _Joins()
        conditions = [self._filter(joins, p, v) for p, v in filters]
        label = joins.ref(path)
        before, after, either = self._both(fact, comparison)
        where = " AND ".join([either, *conditions])
        return (
            f"SELECT {label} AS label, {self._case(fact, after)} AS after_value, {self._case(fact, before)} AS before_value, "
            f"SUM(CASE WHEN {after} THEN 1 ELSE 0 END) AS after_rows, SUM(CASE WHEN {before} THEN 1 ELSE 0 END) AS before_rows "
            f"{self._from(fact, joins)} WHERE {where} GROUP BY {label} ORDER BY ABS(COALESCE(after_value, 0) - COALESCE(before_value, 0)) DESC NULLS LAST LIMIT {_GROUP_LIMIT}"
        )

    def groups(self, fact: Mapping[str, Any], period: Mapping[str, Any], path: Mapping[str, Any], filters: _Filters, values: Sequence[Any]) -> str:
        joins = _Joins()
        conditions = [self._filter(joins, p, v) for p, v in filters]
        label = joins.ref(path)
        return f"SELECT {label} AS label, {self._expr(fact)} AS value {self._from(fact, joins)} WHERE " + " AND ".join([f"({self._period(fact, period)})", *conditions, self._members(label, values)]) + f" GROUP BY {label}"


def _dax_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE()" if value else "FALSE()"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (_dt.date, _dt.datetime)):
        return f"DATE({value.year},{value.month},{value.day})"
    return '"' + str(value).replace('"', '""') + '"'


def _dax_ref(table: str, column: str) -> str:
    return f"'{table}'[{column}]"


_NOT_A_DAY = re.compile(r"(month|quarter|year|week|period|start|end|key)", re.IGNORECASE)


def _day_column(schema: SourceSchema, table: str) -> str | None:
    """A day-grain date column of a date table: ``Date`` first, then ``Full Date``, then any other date-typed column that is not a month or a year."""
    columns = [c for c in schema.tables.get(table, ()) if _TIME_TYPE.search(schema.column_type(table, c)) and not _NOT_A_DAY.search(c)]
    if not columns and not schema.types.get(table):
        columns = [c for c in schema.tables.get(table, ()) if re.fullmatch(r"(full[_ ]?|calendar[_ ]?)?date", c, re.IGNORECASE)]
    return sorted(columns, key=lambda c: (c.casefold() != "date", "full" not in c.casefold(), c))[0] if columns else None


def _dax_axis(schema: SourceSchema, table: str, joins: Mapping[tuple[str, str], tuple[str, str]]) -> dict[str, Any] | None:
    """The time axis of a fact in a model: a related date table (a day column, else year and month columns), or a date column on the fact."""
    columns = schema.tables[table]
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    order = 0
    for column in _date_candidates(columns) + [c for c in columns if _is_time_column(schema, table, c) and not _TIME_COLUMN.search(c)]:
        order += 1
        score = (2 if _BUSINESS_DATE.search(column) else 0) - (2 if _SECONDARY_DATE.search(column) else 0)
        target = joins.get((table, column))
        if target is not None:
            dim, _key = target
            day = _day_column(schema, dim)
            if day:
                candidates.append((score + 4, order, {"kind": "date", "table": dim, "column": day, "via": column}))
                continue
            year = next((c for c in schema.tables[dim] if _YEAR_COLUMN.match(c)), None)
            if year:
                candidates.append((score + 3, order, {"kind": "parts", "table": dim, "year": year, "month": next((c for c in schema.tables[dim] if _MONTH_COLUMN.match(c)), None), "via": column}))
                continue
        if _TIME_TYPE.search(schema.column_type(table, column)):
            candidates.append((score, order, {"kind": "date", "table": table, "column": column}))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return candidates[0][2]


class _Dax:
    """DAX over a semantic model; the model's relationships do the joining."""

    kind = "semantic_model"

    def __init__(self, schema: SourceSchema, joins: Mapping[tuple[str, str], tuple[str, str]]) -> None:
        self.schema = schema
        self.joins = dict(joins)

    def axis(self, table: str) -> dict[str, Any] | None:
        return _dax_axis(self.schema, table, self.joins)

    def _expr(self, fact: Mapping[str, Any]) -> str:
        measure = str(fact["measure"])
        if measure.startswith("[") and measure.endswith("]"):
            return measure  # a model measure, evaluated as the model defines it
        function = "AVERAGE" if fact["aggregate"] == "avg" else "SUM"
        return f"{function}({_dax_ref(fact['table'], measure)})"

    def _sum(self, fact: Mapping[str, Any], measure: str) -> str:
        return measure if measure.startswith("[") else f"SUM({_dax_ref(fact['table'], measure)})"

    def _rows(self, fact: Mapping[str, Any]) -> str:
        return f"COUNTROWS('{fact['table']}')"

    def _period(self, fact: Mapping[str, Any], period: Mapping[str, Any]) -> str:
        dt = fact["date"]
        if period.get("start"):
            if dt["kind"] != "date":
                raise ValueError("the model's date table has no day column, so a date range cannot be expressed")
            ref = _dax_ref(dt["table"], dt["column"])
            start, end = _dt.date.fromisoformat(str(period["start"])), _dt.date.fromisoformat(str(period["end"]))
            return f"{ref} >= DATE({start.year},{start.month},{start.day}) && {ref} < DATE({end.year},{end.month},{end.day})"
        year = int(period["year"])
        month = period.get("month")
        if dt["kind"] == "date":
            ref = _dax_ref(dt["table"], dt["column"])
            if month:
                m = int(month)
                nxt = (year + 1, 1) if m == 12 else (year, m + 1)
                return f"{ref} >= DATE({year},{m},1) && {ref} < DATE({nxt[0]},{nxt[1]},1)"
            return f"{ref} >= DATE({year},1,1) && {ref} < DATE({year + 1},1,1)"
        year_ref = _dax_ref(dt["table"], dt["year"])
        if month and dt.get("month"):
            return "TREATAS({(" + f"{year}, {int(month)}" + ")}, " + f"{year_ref}, {_dax_ref(dt['table'], dt['month'])})"
        return "TREATAS({" + str(year) + "}, " + year_ref + ")"

    def _span(self, fact: Mapping[str, Any], years: Sequence[int]) -> str:
        dt = fact["date"]
        first, last = int(min(years)), int(max(years))
        if dt["kind"] == "date":
            ref = _dax_ref(dt["table"], dt["column"])
            return f"{ref} >= DATE({first},1,1) && {ref} < DATE({last + 1},1,1)"
        return "TREATAS({" + ", ".join(str(int(y)) for y in years) + "}, " + _dax_ref(dt["table"], dt["year"]) + ")"

    def _ref(self, fact: Mapping[str, Any], path: Mapping[str, Any]) -> str:
        return _dax_ref(_path_table(path, fact["table"]), str(path["column"]))

    def _filter(self, fact: Mapping[str, Any], path: Mapping[str, Any], value: Any) -> str:
        ref = self._ref(fact, path)
        if value is None:
            return f"FILTER(ALL({ref}), ISBLANK({ref}))"
        return "TREATAS({" + _dax_literal(value) + "}, " + ref + ")"

    def _members(self, ref: str, values: Sequence[Any]) -> str:
        members = [v for v in values if v is not None]
        listed = "{" + ", ".join(_dax_literal(v) for v in members) + "}"
        if any(v is None for v in values):
            return f"FILTER(ALL({ref}), {ref} IN {listed} || ISBLANK({ref}))" if members else f"FILTER(ALL({ref}), ISBLANK({ref}))"
        return f"TREATAS({listed}, {ref})"

    @staticmethod
    def _calc(expression: str, *arguments: str) -> str:
        return f"CALCULATE({expression}, {', '.join(arguments)})" if arguments else expression

    def _time_keys(self, fact: Mapping[str, Any]) -> tuple[str, str, str]:
        """The grouping columns, the output columns and the order clause of a series query for this axis."""
        dt = fact["date"]
        if dt["kind"] == "date" and dt["table"] != fact["table"]:
            ref = _dax_ref(dt["table"], dt["column"])
            return ref, f'"day", {ref}', "ORDER BY [day]"
        if dt["kind"] == "parts":
            year_ref = _dax_ref(dt["table"], dt["year"])
            month_ref = _dax_ref(dt["table"], dt["month"]) if dt.get("month") else None
            keys = year_ref + (f", {month_ref}" if month_ref else "")
            return keys, f'"year", {year_ref}, "month", {month_ref or "BLANK()"}', "ORDER BY [year]" + (", [month]" if month_ref else "")
        raise ValueError("a date column on the fact itself is grouped with GROUPBY")

    def series(self, fact: Mapping[str, Any], measures: Sequence[str]) -> str:
        dt = fact["date"]
        picked = ", ".join(f'"v{i}", [v{i}]' for i in range(len(measures)))
        if dt["kind"] == "date" and dt["table"] == fact["table"]:
            ref = _dax_ref(fact["table"], dt["column"])
            sums = ", ".join(f'"v{i}", SUMX(CURRENTGROUP(), {_dax_ref(fact["table"], m)})' for i, m in enumerate(measures))
            return f'EVALUATE SELECTCOLUMNS(GROUPBY(ADDCOLUMNS(\'{fact["table"]}\', "__y", YEAR({ref}), "__m", MONTH({ref})), [__y], [__m], "n", COUNTX(CURRENTGROUP(), 1), {sums}), "year", [__y], "month", [__m], "n", [n], {picked}) ORDER BY [year], [month]'
        keys, outputs, order = self._time_keys(fact)
        values = ", ".join(f'"v{i}", {self._sum(fact, m)}' for i, m in enumerate(measures))
        return f'EVALUATE SELECTCOLUMNS(SUMMARIZECOLUMNS({keys}, "n", {self._rows(fact)}, {values}), {outputs}, "n", [n], {picked}) {order}'

    def daily(self, fact: Mapping[str, Any], measures: Sequence[str]) -> str:
        """Rows and the sum of every measure by day: the date table's day column, or the fact's own date."""
        dt = fact["date"]
        if dt["kind"] != "date":
            raise ValueError("the model's date table has no day column, so there is no daily series")
        picked = ", ".join(f'"v{i}", [v{i}]' for i in range(len(measures)))
        if dt["table"] != fact["table"]:
            ref = _dax_ref(dt["table"], dt["column"])
            values = ", ".join(f'"v{i}", {self._sum(fact, m)}' for i, m in enumerate(measures))
            return f'EVALUATE SELECTCOLUMNS(SUMMARIZECOLUMNS({ref}, "n", {self._rows(fact)}, {values}), "day", {ref}, "n", [n], {picked}) ORDER BY [day]'
        ref = _dax_ref(fact["table"], dt["column"])
        sums = ", ".join(f'"v{i}", SUMX(CURRENTGROUP(), {_dax_ref(fact["table"], m)})' for i, m in enumerate(measures))
        return f'EVALUATE SELECTCOLUMNS(GROUPBY(ADDCOLUMNS(\'{fact["table"]}\', "__d", DATE(YEAR({ref}), MONTH({ref}), DAY({ref}))), [__d], "n", COUNTX(CURRENTGROUP(), 1), {sums}), "day", [__d], "n", [n], {picked}) ORDER BY [day]'

    def months(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not rows or "day" not in rows[0]:
            return [dict(r) for r in rows]
        by_month: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in rows:
            iso = _iso_date(row.get("day"))
            if not iso:
                continue
            key: tuple[Any, ...] = (int(iso[:4]), int(iso[5:7])) + ((row.get("label"),) if "label" in row else ())
            bucket = by_month.setdefault(key, {"year": key[0], "month": key[1], "n": 0, **({"label": row.get("label")} if "label" in row else {})})
            bucket["n"] += int(row.get("n") or 0)
            for name, value in row.items():
                if name.startswith("v"):
                    bucket[name] = float(bucket.get(name) or 0) + float(value or 0)
        return [by_month[k] for k in sorted(by_month, key=lambda k: (k[0], k[1], str(k[2:])))]

    def top_groups(self, fact: Mapping[str, Any], path: Mapping[str, Any], years: Sequence[int], limit: int) -> str:
        ref = self._ref(fact, path)
        inner = f'SUMMARIZECOLUMNS({ref}, "value", {self._calc(self._expr(fact), self._span(fact, years))})'
        return f'EVALUATE TOPN({limit}, SELECTCOLUMNS({inner}, "label", {ref}, "value", [value]), ABS([value]), DESC)'

    def grouped_series(self, fact: Mapping[str, Any], path: Mapping[str, Any], years: Sequence[int], values: Sequence[Any]) -> str:
        dt = fact["date"]
        ref = self._ref(fact, path)
        if dt["kind"] == "date" and dt["table"] == fact["table"]:
            date_ref = _dax_ref(fact["table"], dt["column"])
            table = f"CALCULATETABLE('{fact['table']}', {self._members(ref, values)}, {self._span(fact, years)})"
            return f'EVALUATE SELECTCOLUMNS(GROUPBY(ADDCOLUMNS({table}, "__y", YEAR({date_ref}), "__m", MONTH({date_ref}), "__g", {ref}), [__y], [__m], [__g], "n", COUNTX(CURRENTGROUP(), 1), "v0", SUMX(CURRENTGROUP(), {_dax_ref(fact["table"], fact["measure"])})), "year", [__y], "month", [__m], "label", [__g], "n", [n], "v0", [v0]) ORDER BY [year], [month]'
        keys, outputs, order = self._time_keys(fact)
        inner = f'SUMMARIZECOLUMNS({keys}, {ref}, {self._members(ref, values)}, "n", {self._calc(self._rows(fact), self._span(fact, years))}, "v0", {self._calc(self._sum(fact, fact["measure"]), self._span(fact, years))})'
        return f'EVALUATE SELECTCOLUMNS({inner}, {outputs}, "label", {ref}, "n", [n], "v0", [v0]) {order}'

    def max_date(self, fact: Mapping[str, Any]) -> str:
        dt = fact["date"]
        if dt["kind"] != "date":
            return 'EVALUATE ROW("value", BLANK())'
        ref = _dax_ref(dt["table"], dt["column"])
        if dt["table"] == fact["table"]:
            return f'EVALUATE ROW("value", MAX({ref}))'
        return f"EVALUATE ROW(\"value\", MAXX(SUMMARIZE('{fact['table']}', {ref}), {ref}))"

    def movement(self, fact: Mapping[str, Any], comparison: Comparison, filters: _Filters) -> str:
        narrow = [self._filter(fact, p, v) for p, v in filters]
        before, after = self._period(fact, comparison.before), self._period(fact, comparison.after)
        expr, rows = self._expr(fact), self._rows(fact)
        return (
            f'EVALUATE ROW("before_value", {self._calc(expr, before, *narrow)}, "after_value", {self._calc(expr, after, *narrow)}, '
            f'"before_rows", {self._calc(rows, before, *narrow)}, "after_rows", {self._calc(rows, after, *narrow)})'
        )

    def total(self, fact: Mapping[str, Any], period: Mapping[str, Any], filters: _Filters) -> str:
        narrow = [self._filter(fact, p, v) for p, v in filters]
        return f'EVALUATE ROW("value", {self._calc(self._expr(fact), self._period(fact, period), *narrow)})'

    def grouped(self, fact: Mapping[str, Any], comparison: Comparison, path: Mapping[str, Any], filters: _Filters) -> str:
        ref = self._ref(fact, path)
        narrow = "".join(", " + self._filter(fact, p, v) for p, v in filters)
        before, after = self._period(fact, comparison.before), self._period(fact, comparison.after)
        expr, rows = self._expr(fact), self._rows(fact)
        inner = f'SUMMARIZECOLUMNS({ref}{narrow}, "before_value", {self._calc(expr, before)}, "after_value", {self._calc(expr, after)}, "before_rows", {self._calc(rows, before)}, "after_rows", {self._calc(rows, after)})'
        return f'EVALUATE TOPN({_GROUP_LIMIT}, SELECTCOLUMNS({inner}, "label", {ref}, "before_value", [before_value], "after_value", [after_value], "before_rows", [before_rows], "after_rows", [after_rows]), ABS([after_value] - [before_value]), DESC)'

    def groups(self, fact: Mapping[str, Any], period: Mapping[str, Any], path: Mapping[str, Any], filters: _Filters, values: Sequence[Any]) -> str:
        ref = self._ref(fact, path)
        narrow = "".join(", " + self._filter(fact, p, v) for p, v in filters)
        inner = f'SUMMARIZECOLUMNS({ref}, {self._members(ref, values)}{narrow}, "value", {self._calc(self._expr(fact), self._period(fact, period))})'
        return f'EVALUATE SELECTCOLUMNS({inner}, "label", {ref}, "value", [value])'


# --------------------------------------------------------------------------- #
# Probes: a schema, a dialect and a way to run a query, for each kind of source
# --------------------------------------------------------------------------- #


def _records(frame: Any) -> list[dict[str, Any]]:
    if frame is None:
        return []
    if hasattr(frame, "to_dict"):
        return [dict(r) for r in frame.to_dict(orient="records")]
    return [dict(r) for r in frame if isinstance(r, Mapping)]


def _clean(value: Any) -> Any:
    if isinstance(value, float) and value != value:  # NaN from a pandas frame
        return None
    return value


def _plain_key(key: Any) -> str:
    text = str(key).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return text.rsplit("]", 1)[-1].lstrip("[") if "]" in text else text


def _first(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


class LakehouseProbe:
    """Runs the sweep's SQL through ``LakehouseSource.query``; the schema comes from the resolved catalog."""

    kind = "lakehouse"

    def __init__(self, source: Any, *, timeout: float = 600.0, name: str | None = None) -> None:
        handle = source.resolve() if callable(getattr(source, "resolve", None)) else source
        tables: dict[str, list[str]] = {}
        types: dict[str, dict[str, str]] = {}
        for entry in list(getattr(handle, "catalog", None) or ()):
            columns = entry.get("columns") or []
            if not columns:
                continue
            names: list[str] = []
            kinds: dict[str, str] = {}
            for column in columns:
                if isinstance(column, Mapping):
                    column_name, column_type = str(column.get("name", "")), str(column.get("type") or column.get("data_type") or "")
                else:
                    column_name, column_type = str(column[0]), (str(column[1]) if len(column) > 1 else "")
                if column_name:
                    names.append(column_name)
                    kinds[column_name] = column_type
            tables[str(entry["name"])] = names
            types[str(entry["name"])] = kinds
        self.name = name or str(getattr(handle, "root", "lakehouse")).rstrip("/").rsplit("/", 1)[-1]
        self.schema = schema_from_tables(self.name, tables, types=types)
        self._executor = LakehouseExecutor(handle.query, self.schema.tables, timeout=timeout)

    @classmethod
    def from_executor(cls, executor: Any, schema: SourceSchema, *, name: str = "lakehouse") -> "LakehouseProbe":
        """A probe over a ready executor (``run({"sql": ...})``) and schema; what the tests use."""
        probe = cls.__new__(cls)
        probe.name = name
        probe.schema = schema
        probe._executor = executor
        return probe

    def run(self, query: str) -> list[dict[str, Any]]:
        return self._executor.run({"kind": "sql", "sql": query})

    def joins(self, text: str) -> dict[tuple[str, str], tuple[str, str]]:
        joins = dict(_heuristic_joins(self.schema))
        joins.update(_joins_from_instructions(text, self.schema))
        return joins

    def facts(self, text: str) -> list[str]:
        return _scoped_facts(self.schema, text)

    def dialect(self, joins: Mapping[tuple[str, str], tuple[str, str]]) -> _Sql:
        return _Sql(self.schema, joins)


_AUTO_DATE_TABLE = re.compile(r"^(LocalDateTable_|DateTableTemplate_)", re.IGNORECASE)


def _model_facts(schema: SourceSchema, joins: Mapping[tuple[str, str], tuple[str, str]]) -> list[str]:
    """Tables with measures and a time axis, the ones with the most relationships out first."""
    facts = []
    for table in schema.tables:
        if re.search(r"(date|calendar)", table, re.IGNORECASE):
            continue
        if _measure_columns(schema, table) and _dax_axis(schema, table, joins):
            facts.append(table)
    return sorted(facts, key=lambda t: (-sum(1 for a, _c in joins if a == t), -len(_measure_columns(schema, t)), t))


class SemanticModelProbe:
    """Runs the sweep's DAX through ``SemanticModel.dax``; the schema comes from ``SemanticModel.metadata()``.

    Any object with ``dax(query)`` returning a DataFrame or row mappings and
    ``metadata()`` returning frames or row lists named ``tables``,
    ``columns``, ``measures`` and ``relationships`` (sempy's snake-case
    column names) works, so a REST client outside a notebook does too.
    """

    kind = "semantic_model"

    def __init__(self, model: Any, *, name: str | None = None) -> None:
        meta = model.metadata()
        frames = {key: _records(getattr(meta, key) if hasattr(meta, key) else meta.get(key)) for key in ("tables", "columns", "measures", "relationships")}
        hidden = {str(_first(r, "table_name", "name")) for r in frames["tables"] if _first(r, "hidden", "is_hidden") in (True, 1, "True", "true")}
        tables: dict[str, list[str]] = {}
        types: dict[str, dict[str, str]] = {}
        for row in frames["columns"]:
            table, column = str(_first(row, "table_name", "table") or ""), str(_first(row, "column_name", "name") or "")
            if not table or not column or table in hidden or _AUTO_DATE_TABLE.match(table) or column.startswith("RowNumber-"):
                continue
            tables.setdefault(table, []).append(column)
            types.setdefault(table, {})[column] = str(_first(row, "data_type", "type") or "")
        measures = tuple(str(_first(r, "measure_name", "name")) for r in frames["measures"] if _first(r, "measure_name", "name"))
        relationships: list[tuple[str, str, str, str]] = []
        for row in frames["relationships"]:
            if _first(row, "active", "is_active") in (False, 0, "False", "false"):
                continue
            parts = tuple(str(_first(row, key) or "") for key in ("from_table", "from_column", "to_table", "to_column"))
            if parts[0] in tables and parts[2] in tables and all(parts):
                relationships.append(parts)  # type: ignore[arg-type]
        self.name = name or str(getattr(model, "dataset", "") or "semantic model")
        self.schema = SourceSchema(self.name, "semantic_model", {t: tuple(c) for t, c in tables.items()}, measures, tuple(relationships), types)
        self._model = model

    def run(self, query: str) -> list[dict[str, Any]]:
        return [{_plain_key(k): _clean(v) for k, v in row.items()} for row in _rows(self._model.dax(query))]

    def joins(self, text: str) -> dict[tuple[str, str], tuple[str, str]]:
        return {(a, b): (c, d) for a, b, c, d in self.schema.relationships}

    def facts(self, text: str) -> list[str]:
        facts = _model_facts(self.schema, self.joins(text))
        named = [t for t in _tables_named_in(text, self.schema) if t in facts]
        return named or facts

    def dialect(self, joins: Mapping[tuple[str, str], tuple[str, str]]) -> _Dax:
        return _Dax(self.schema, joins)


def _probe_for(source: Any, *, timeout: float, name: str | None) -> Any:
    if isinstance(source, (LakehouseProbe, SemanticModelProbe)):
        return source
    if callable(getattr(source, "dax", None)) and callable(getattr(source, "metadata", None)):
        return SemanticModelProbe(source, name=name)
    if callable(getattr(source, "query", None)) and hasattr(source, "catalog"):
        return LakehouseProbe(source, timeout=timeout, name=name)
    raise TypeError("what_moved needs a LakehouseSource, a SemanticModel, or a probe built from one")


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #


def what_moved(
    source: Any,
    *,
    years: Sequence[int] | None = None,
    instructions: str = "",
    scope: str = "",
    facts: int | Sequence[str] = 2,
    measures: int | Sequence[str] = 2,
    paths: int | Sequence[str] = 8,
    comparisons: Sequence[Any] | None = None,
    budget: int = 60,
    min_pct: float = 0.05,
    top: int = 5,
    depth: int = 2,
    verify: bool = True,
    timeout: float = 600.0,
    name: str | None = None,
) -> Sweep:
    """What moved in a lakehouse or a semantic model, measured by the source and recomputed.

    ``years`` are the complete calendar years to compare (discovered from
    the data when omitted: every year, minus a last one that holds fewer
    than half the rows of the one before). ``instructions`` is free text
    about the source, the way a Data Agent's instructions read: tables to
    use, joins, definitions; ``scope`` says what to look at first. ``facts``,
    ``measures`` and ``paths`` are counts or explicit names (a model measure
    as ``"[Total Sales]"``); ``comparisons`` restricts the comparison kinds
    (``"month"``, ``"same_month_prior_year"``, ``"year"``) or supplies
    :class:`Comparison` objects; ``budget`` bounds the queries. ``verify``
    recomputes every reported figure with an independent query afterwards.
    """
    probe = _probe_for(source, timeout=timeout, name=name)
    started = time.monotonic()
    result = sweep(probe, years, instructions=instructions, scope=scope, facts=facts, measures=measures, paths=paths, comparisons=comparisons, budget=budget, min_pct=min_pct, top=top, depth=depth)
    if verify:
        result = verify_sweep(result, probe)
    return replace(result, elapsed=round(time.monotonic() - started, 1))


def sweep(
    probe: Any,
    years: Sequence[int] | None = None,
    *,
    instructions: str = "",
    scope: str = "",
    facts: int | Sequence[str] = 2,
    measures: int | Sequence[str] = 2,
    paths: int | Sequence[str] = 8,
    comparisons: Sequence[Any] | None = None,
    budget: int = 60,
    min_pct: float = 0.05,
    top: int = 5,
    depth: int = 2,
) -> Sweep:
    """The sweep over a probe: phase one measures every movement, phase two decomposes the material ones until the budget is spent."""
    schema: SourceSchema = probe.schema
    context = ReviewContext(scope=scope) if scope else None
    text = "\n".join(part for part in (instructions, context.text if context is not None else "") if part)
    snapshot = AgentSnapshot(agent_id="sweep", name=probe.name, instructions=instructions, datasources=(AgentDataSource(id=schema.source_id, kind=schema.kind, name=probe.name),))
    vocabulary = build_vocabulary(snapshot, schema, context)
    excluded = excluded_terms(text)
    terms = context.terms if context is not None else ()
    joins = probe.joins(text)
    dialect = probe.dialect(joins)
    ledger: list[Movement] = []
    findings: list[SweepFinding] = []
    notes: list[str] = []
    words: dict[str, str] = {}
    series: dict[str, tuple[Point, ...]] = {}
    collapsed: list[str] = []
    spent = 0

    def run(query: str) -> list[dict[str, Any]]:
        nonlocal spent
        if spent >= budget:
            raise _Budget
        spent += 1
        return probe.run(query)

    if isinstance(facts, int):
        fact_names = list(probe.facts(text))[:facts]
    else:
        lowered = {t.casefold(): t for t in schema.tables}
        fact_names = [lowered[str(f).casefold()] for f in facts if str(f).casefold() in lowered]
    if not fact_names:
        notes.append("no fact table recognised, nothing to sweep")

    totals: list[tuple[Movement, dict[str, Any], list[Mapping[str, Any]], tuple[str, ...]]] = []
    years_used: list[int] = [int(y) for y in years] if years else []
    exhausted = False
    try:
        for table in fact_names:
            axis = dialect.axis(table)
            candidates = _measure_candidates(schema, table, measures)
            if not axis or not candidates:
                notes.append(f"{table}: no time axis or no measure, not swept")
                continue
            base = {"table": table, "date": axis, "measure": candidates[0], "aggregate": _aggregate_for(candidates[0])}
            months = dialect.months(run(dialect.series(base, candidates)))
            fact_years = [int(y) for y in years] if years else _complete_years(months)
            if not years_used:
                years_used = list(fact_years)
            wanted_comparisons = _wanted_comparisons(months, fact_years, comparisons)
            if not wanted_comparisons:
                notes.append(f"{table}: no comparison the time axis supports for {fact_years}")
                continue
            try:
                rows = run(dialect.max_date(base))
                max_date = _iso_date(rows[0].get("value")) if rows and rows[0].get("value") is not None else None
            except _Budget:
                raise
            except Exception:  # noqa: BLE001 - only the incomplete-month flag needs it
                max_date = None
            chosen = _choose_paths(schema, table, joins, excluded, terms, paths)
            wanted = measures if isinstance(measures, int) else len(candidates)
            measured: list[tuple[str, list[Movement]]] = []
            for index, measure in enumerate(candidates):
                if len(measured) >= wanted:
                    break
                fact = {"table": table, "date": axis, "measure": measure, "aggregate": _aggregate_for(measure)}
                key = f"{table}|{measure}"
                words[key] = _channel_measure(vocabulary.table(table), vocabulary.measure(measure.strip("[]")))
                series[key] = _points(months, index, fact["aggregate"], fact_years)
                run_totals = [_movement(run, dialect, fact, comparison, ()) for comparison in wanted_comparisons]
                run_totals = [replace(t, flags=_flags(t, t.comparison, max_date, months)) for t in run_totals]
                ledger.extend(run_totals)
                twin = next((m for m, other in measured if _same_figures(other, run_totals)), None)
                if twin is not None:
                    collapsed.append(key)
                    notes.append(f"{words[key]} moves within 1% of {words[f'{table}|{twin}']} in every comparison, so it was not decomposed separately")
                    continue
                measured.append((measure, run_totals))
                totals.extend((t, fact, chosen, t.flags) for t in run_totals)
    except _Budget:
        exhausted = True
        notes.append(f"the budget of {budget} queries was spent before every movement was measured; nothing was decomposed")

    material = [item for item in totals if item[0].material(min_pct)]
    material.sort(key=lambda item: (not item[0].trusted, any("small base" in flag for flag in item[3]), -abs(item[0].pct or 0)))  # complete periods first: a thin month is measured, never explained first
    for index, (total, fact, chosen, flags) in enumerate(material):
        if exhausted:
            break
        decompositions: list[Decomposition] = []
        drill: list[Decomposition] = []
        try:
            for path in chosen:
                decompositions.append(_decompose(run, dialect, ledger, total, fact, path, (), top))
            decompositions.sort(key=lambda d: (-_rank(d), _placeholder_lead(d), -d.explained))  # a real leader beats a placeholder at the same rank
            best = decompositions[0] if decompositions else None
            if depth >= 2 and best is not None and best.concentration in {"single", "concentrated"} and best.groups:
                leader = best.groups[0]
                for path in chosen:
                    if _same_path(path, best.path, fact["table"]):
                        continue
                    drill.append(_decompose(run, dialect, ledger, leader, fact, path, ((best.path, leader.group),), top))
                    if len(drill) >= 3:
                        break
        except _Budget:
            exhausted = True
            left = len(material) - index - (1 if decompositions else 0)
            notes.append(f"the budget of {budget} queries was spent before the sweep finished; {left} material movement(s) were measured but not decomposed")
        if decompositions:
            decompositions.sort(key=lambda d: (-_rank(d), _placeholder_lead(d), -d.explained))  # a real leader beats a placeholder at the same rank
            drill.sort(key=lambda d: (-_rank(d), _placeholder_lead(d), -d.explained))
            findings.append(SweepFinding(total, tuple(decompositions), tuple(drill), flags))
    findings.sort(key=lambda f: (not f.trusted, any(flag.startswith("volume:") for flag in f.flags), -(_rank(f.best) if f.best else -1), -abs(f.movement.pct or 0)))
    return Sweep(probe.name, probe.kind, tuple(findings), tuple(ledger), spent, budget, tuple(years_used), tuple(notes), words, series, collapsed=tuple(collapsed))


def _wanted_comparisons(months: Sequence[Mapping[str, Any]], years: Sequence[int], comparisons: Sequence[Any] | None) -> list[Comparison]:
    """The comparisons the axis supports, narrowed to the kinds asked for, plus any explicit ones."""
    supported = _comparisons(months, years)
    if comparisons is None:
        return supported
    kinds = {str(c) for c in comparisons if isinstance(c, str)}
    explicit = [c for c in comparisons if isinstance(c, Comparison)]
    return [c for c in supported if c.kind in kinds] + explicit


def _measure_candidates(schema: SourceSchema, table: str, measures: int | Sequence[str]) -> list[str]:
    if isinstance(measures, int):
        additive_first = sorted(_measure_columns(schema, table), key=lambda c: _aggregate_for(c) == "avg")  # amounts and quantities before prices and scores, which only average
        return additive_first[: measures + 2]  # a couple of spares, in case two move identically
    lowered = {c.casefold(): c for c in schema.tables[table]}
    chosen: list[str] = []
    for measure in measures:
        text = str(measure)
        if text.startswith("[") and text.endswith("]"):
            chosen.append(text)
        elif text.casefold() in lowered:
            chosen.append(lowered[text.casefold()])
    return chosen


def _choose_paths(schema: SourceSchema, table: str, joins: Mapping[tuple[str, str], tuple[str, str]], excluded: Any, terms: Sequence[str], paths: int | Sequence[str]) -> list[Mapping[str, Any]]:
    """The grouping paths to sweep: the role paths first, then the rest, up to a count; or the named columns in the order named."""
    all_paths = [p for p in attribute_paths(schema, table, joins, excluded) if not _LOCAL_EXCLUDED.search(str(p["column"]))]  # a code, a SKU or a number is a label, not a grouping
    if not isinstance(paths, int):
        chosen_named: list[Mapping[str, Any]] = []
        for name in paths:
            wanted = str(name).casefold()
            match = next((p for p in all_paths if str(p["column"]).casefold() == wanted or humanize_column(str(p["column"])).casefold() == wanted), None)
            if match is not None and all(match is not c for c in chosen_named):
                chosen_named.append(match)
        return chosen_named
    roles = _paths_by_role(all_paths, terms)
    rest = sorted((p for p in all_paths if all(p is not r for r in roles.values())), key=lambda p: (not _GROUPING_HINT.search(str(p["column"])), len(p.get("hops") or ())))  # named like a grouping (channel, type, region, brand) first, nearest first
    ordered: list[Mapping[str, Any]] = list(roles.values()) + rest
    chosen: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in ordered:
        key = (_path_table(path, table), str(path["column"]))
        if key not in seen:
            seen.add(key)
            chosen.append(path)
        if len(chosen) >= paths:
            break
    return chosen


def _same_path(a: Mapping[str, Any], b: Mapping[str, Any], table: str) -> bool:
    return (_path_table(a, table), str(a["column"])) == (_path_table(b, table), str(b["column"]))


def _same_figures(a: Sequence[Movement], b: Sequence[Movement], tolerance: float = 0.01) -> bool:
    """Two measures move alike when every before and after figure agrees within the tolerance (an extended amount next to a sales amount)."""

    def near(x: float, y: float) -> bool:
        return abs(x - y) <= tolerance * max(1.0, abs(y))

    return len(a) == len(b) and all(near(x.before_value, y.before_value) and near(x.after_value, y.after_value) for x, y in zip(a, b))


def _complete_years(months: Sequence[Mapping[str, Any]]) -> list[int]:
    """Every year with rows, minus a first year that starts after January and a last year that ends before December.

    Without months to look at, a last year holding fewer than half the rows
    of the one before is taken as partial.
    """
    counts: dict[int, int] = {}
    present: dict[int, set[int]] = {}
    for row in months:
        if row.get("year") is None:
            continue
        year = int(row["year"])
        counts[year] = counts.get(year, 0) + int(row.get("n") or 0)
        if row.get("month") is not None and int(row.get("n") or 0) > 0:
            present.setdefault(year, set()).add(int(row["month"]))
    years = sorted(counts)
    if present:
        if years and 12 not in present.get(years[-1], set()):
            years = years[:-1]
        if years and 1 not in present.get(years[0], set()):
            years = years[1:]
        return years
    if len(years) >= 2 and counts[years[-1]] < 0.5 * counts[years[-2]]:
        years = years[:-1]
    return years


def _comparisons(months: Sequence[Mapping[str, Any]], years: Sequence[int]) -> list[Comparison]:
    """The latest complete year against the one before; the latest month with data against the month before and against the same month a year earlier."""
    comparisons: list[Comparison] = []
    if len(years) >= 2:
        comparisons.append(Comparison("year", {"year": int(years[-2])}, {"year": int(years[-1])}))
    present = sorted({(int(r["year"]), int(r["month"])) for r in months if r.get("year") is not None and r.get("month") is not None and int(r.get("n") or 0) > 0})
    if present:
        latest = present[-1]
        if len(present) >= 2:
            previous = present[-2]
            consecutive = (previous[0] == latest[0] and previous[1] == latest[1] - 1) or (previous[0] == latest[0] - 1 and previous[1] == 12 and latest[1] == 1)
            if consecutive:
                comparisons.append(Comparison("month", {"year": previous[0], "month": previous[1]}, {"year": latest[0], "month": latest[1]}))
        if (latest[0] - 1, latest[1]) in present:
            comparisons.append(Comparison("same_month_prior_year", {"year": latest[0] - 1, "month": latest[1]}, {"year": latest[0], "month": latest[1]}))
    return comparisons


def _points(months: Sequence[Mapping[str, Any]], index: int, aggregate: str, years: Sequence[int], value_key: str | None = None) -> tuple[Point, ...]:
    """The monthly series of one measure over the compared years and whatever follows them."""
    start = int(years[-2]) if len(years) >= 2 else (int(years[0]) if years else 0)
    key = value_key or f"v{index}"
    points: list[Point] = []
    for row in months:
        if row.get("year") is None or row.get("month") is None or int(row["year"]) < start:
            continue
        rows = int(row.get("n") or 0)
        value = float(row.get(key) or 0)
        if aggregate == "avg":
            value = value / rows if rows else 0.0
        points.append(Point(int(row["year"]), int(row["month"]), value, rows))
    return tuple(points)


def _movement(run: Callable[[str], list[dict[str, Any]]], dialect: Any, fact: Mapping[str, Any], comparison: Comparison, filters: _Filters, *, path: Mapping[str, Any] | None = None, group: Any = None) -> Movement:
    """One figure at the total level, or for one group when a drill needs a parent figure."""
    narrow = list(filters) + ([(path, group)] if path is not None else [])
    query = dialect.movement(fact, comparison, narrow)
    rows = run(query)
    row = rows[0] if rows else {}
    verification = {"before": dialect.total(fact, comparison.before, narrow), "after": dialect.total(fact, comparison.after, narrow)}
    return Movement(fact["table"], fact["measure"], comparison, float(row.get("before_value") or 0), float(row.get("after_value") or 0), int(row.get("before_rows") or 0), int(row.get("after_rows") or 0), query, verification, path, group, tuple((str(p["column"]), v) for p, v in filters), fact["aggregate"])


def _decompose(run: Callable[[str], list[dict[str, Any]]], dialect: Any, ledger: list[Movement], parent: Movement, fact: Mapping[str, Any], path: Mapping[str, Any], parents: _Filters, top: int) -> Decomposition:
    """Every group of one path in one query, classified; the groups join the ledger with their own recomputation queries."""
    comparison = parent.comparison
    query = dialect.grouped(fact, comparison, path, parents)
    parent_columns = tuple((str(p["column"]), v) for p, v in parents)
    groups: list[Movement] = []
    for row in run(query):
        group = row.get("label")
        narrow = [*parents, (path, group)]
        verification = {"before": dialect.total(fact, comparison.before, narrow), "after": dialect.total(fact, comparison.after, narrow)}
        groups.append(Movement(fact["table"], fact["measure"], comparison, float(row.get("before_value") or 0), float(row.get("after_value") or 0), int(row.get("before_rows") or 0), int(row.get("after_rows") or 0), query, verification, path, group, parent_columns, fact["aggregate"]))
    ledger.extend(groups)
    decomposition = _classify(parent, path, groups, top)
    values = [g.group for g in decomposition.groups]
    verification = {"before": dialect.groups(fact, comparison.before, path, parents, values), "after": dialect.groups(fact, comparison.after, path, parents, values)} if values else {}
    return replace(decomposition, verification=verification, size=len(groups))


def _classify(parent: Movement, path: Mapping[str, Any], groups: Sequence[Movement], top: int) -> Decomposition:
    """Classify how a change spreads over the groups of one path.

    ``single``: one group carries at least half of the change and more than
    its share of the base. ``concentrated``: the top three carry at least
    70 percent and the leader moved more than its size. ``proportional``:
    the top groups moved in step with their size, so the path explains
    nothing. ``offsetting``: groups moved both ways and the opposite side
    is at least half the net change. ``broad``: the rest. ``none``: an
    averaged measure, a single-group path or no change to split.
    """
    total = parent.delta
    ranked = tuple(sorted(groups, key=lambda g: -abs(g.delta))[:top])
    if not total or len(groups) < 2 or parent.aggregate != "sum":
        return Decomposition(parent, path, ranked, "none", 0.0)
    same = sorted((g for g in groups if g.delta * total > 0), key=lambda g: -abs(g.delta))
    opposite = sorted((g for g in groups if g.delta * total < 0), key=lambda g: -abs(g.delta))
    base = parent.before_value

    def share(g: Movement) -> float:
        return g.delta / total

    def excess(g: Movement) -> float:
        # how much more of the change a group carried than its size would predict: the driver signal
        return share(g) - (g.before_value / base if base else 0.0)

    explained = sum(sorted((share(g) for g in same), reverse=True)[:3])
    # the driver is the group that moved most against its size among those carrying a real part of the change,
    # not the largest contributor: a big group growing in step with the total is the base, not the story
    driver = max((g for g in same if share(g) >= 0.2), key=excess, default=None)
    if opposite and abs(sum(o.delta for o in opposite)) >= 0.5 * abs(total):
        concentration = "offsetting"
    elif driver is not None and share(driver) >= 0.5 and excess(driver) >= 0.1:
        concentration = "single"
    elif driver is not None and excess(driver) >= 0.1 and explained >= 0.7:
        concentration = "concentrated"
    elif same and all(abs(excess(g)) < 0.1 for g in same[:3]):
        concentration = "proportional"
    else:
        concentration = "broad"
    ordered = ([driver] + [g for g in same if g is not driver]) if driver is not None and concentration in {"single", "concentrated"} else same
    return Decomposition(parent, path, tuple(ordered[:top] + opposite[:2]), concentration, explained)


def _rank(d: Decomposition) -> int:
    return {"single": 4, "concentrated": 3, "offsetting": 2, "broad": 1, "proportional": 0, "none": -1}.get(d.concentration, 0)


def _flags(total: Movement, comparison: Comparison, max_date: str | None, months: Sequence[Mapping[str, Any]] = ()) -> tuple[str, ...]:
    """What a reader must see before the number: an incomplete period (by the data's end or by its row coverage), volume rather than rate, a small base.

    Flags start with ``incomplete:``, ``coverage:``, ``volume:`` or ``small base:`` so
    the callers can tell a period that must not be interpreted from a caveat.
    """
    flags: list[str] = []
    if comparison.after.get("month") is None and comparison.after.get("start") is None:
        thin = _thin_year(comparison.before, comparison.after, months)
        if thin:
            flags.append(thin)
    else:
        for period in (comparison.after, comparison.before):
            coverage = _coverage_flag(period, months)
            if coverage:
                flags.append(coverage)
    if max_date and comparison.after.get("month") is not None and not any(f.startswith("coverage:") for f in flags):
        try:
            end = _dt.date.fromisoformat(max_date)
        except ValueError:
            end = None
        if end is not None and end.year == int(comparison.after["year"]) and end.month == int(comparison.after["month"]) and end.day < 25:
            flags.append(f"incomplete: the data ends on {max_date}, so {_month_name(comparison.after)} is incomplete")
    pct, rows_pct = total.pct, total.rows_pct
    if total.aggregate == "sum" and pct is not None and rows_pct is not None and rows_pct * pct > 0 and abs(rows_pct) >= 0.5 * abs(pct):
        flags.append(f"volume: row counts moved {rows_pct:+.0%} against {pct:+.0%} in value, so this is volume or coverage rather than a change in rate")
    if total.before_rows and total.before_rows < 30:
        flags.append(f"small base: {total.before_rows} rows before, {total.after_rows} after")
    return tuple(flags)


def _coverage_flag(period: Mapping[str, Any], months: Sequence[Mapping[str, Any]]) -> str | None:
    """A month with fewer than half the rows of a typical month before it is not a complete period, whatever date the data runs to."""
    if not months or period.get("start") or period.get("month") is None:
        return None
    year, month = int(period["year"]), int(period["month"])
    rows = sum(int(r.get("n") or 0) for r in months if r.get("year") is not None and int(r["year"]) == year and r.get("month") is not None and int(r["month"]) == month)
    earlier = sorted(((int(r["year"]), int(r["month"]), int(r.get("n") or 0)) for r in months if r.get("year") is not None and r.get("month") is not None and (int(r["year"]), int(r["month"])) < (year, month) and int(r.get("n") or 0) > 0), reverse=True)[:6]
    if len(earlier) < 2:
        return None
    counts = sorted(n for _y, _m, n in earlier)
    typical = counts[len(counts) // 2]
    if typical and rows < 0.5 * typical:
        return f"coverage: {_month_name(period)} holds {rows:,} rows against a typical {typical:,} a month, so it looks incomplete and the movement is coverage, not business"
    return None


def _thin_year(before: Mapping[str, Any], after: Mapping[str, Any], months: Sequence[Mapping[str, Any]]) -> str | None:
    """Of two years compared, the one with fewer than half the other's months of data is not a complete year (a year that starts in December, a year that ends in January)."""
    if not months:
        return None

    def present(period: Mapping[str, Any]) -> int:
        year = int(period["year"])
        return len({int(r["month"]) for r in months if r.get("year") is not None and int(r["year"]) == year and r.get("month") is not None and int(r.get("n") or 0) > 0})

    a, b = present(before), present(after)
    if a and b and b < 0.5 * a:
        return f"coverage: {int(after['year'])} has data for {b} month{'s' if b != 1 else ''} against {a} in {int(before['year'])}, so it is not a complete year"
    if a and b and a < 0.5 * b:
        return f"coverage: {int(before['year'])} has data for {a} month{'s' if a != 1 else ''} against {b} in {int(after['year'])}, so it is not a complete year"
    return None


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def verify_sweep(result: Sweep, probe: Any, *, tolerance: float = 1e-6) -> Sweep:
    """Recompute every reported figure with its independent per-period query; the result carries the mismatches and the count."""
    mismatches: list[str] = []
    checked: set[str] = set()
    recomputed = 0

    def close(expected: float, actual: float) -> bool:
        return abs(actual - expected) <= tolerance * max(1.0, abs(expected))

    def figure(m: Movement, label: str) -> None:
        nonlocal recomputed
        key = m.verification["after"] + m.verification["before"]
        if key in checked:
            return
        checked.add(key)
        for side, expected in (("before", m.before_value), ("after", m.after_value)):
            rows = probe.run(m.verification[side])
            actual = float((rows[0] or {}).get("value") or 0) if rows else 0.0
            recomputed += 1
            if not close(expected, actual):
                mismatches.append(f"{label} {side}: expected {expected}, recomputed {actual}")

    def groups(d: Decomposition) -> None:
        nonlocal recomputed
        if not d.verification or not d.groups:
            return
        key = d.verification["after"] + d.verification["before"]
        if key in checked:
            return
        checked.add(key)
        for side in ("before", "after"):
            actual = {row.get("label"): float(row.get("value") or 0) for row in probe.run(d.verification[side])}
            for g in d.groups:
                expected = g.before_value if side == "before" else g.after_value
                got = actual.get(g.group, 0.0)
                recomputed += 1
                if not close(expected, got):
                    mismatches.append(f"{g.fact}.{g.measure} {g.comparison.label} {_word(g.path)}={_label(g.group)} {side}: expected {expected}, recomputed {got}")

    for finding in result.findings:
        figure(finding.movement, f"{finding.movement.fact}.{finding.movement.measure} {finding.movement.comparison.label}")
        for d in (*finding.decompositions[:1], *finding.drill[:1]):
            groups(d)
    return replace(result, mismatches=tuple(mismatches), recomputed=recomputed, verified=True)
