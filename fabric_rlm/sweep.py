"""What moved: a budgeted, deterministic sweep over a source's facts, measures, groupings and periods.

The sweep is the exhaustive version of the discovery step the review runs
before it writes questions. For every fact in scope it takes the measures
the schema exposes, the comparisons the time axis supports (the latest
month against the month before, the same month a year earlier, the latest
complete year against the previous one), and every grouping path the joins
reach. It computes each movement with one query per grouping, scores it by
absolute and relative change, decomposes the material ones by every path,
classifies the decomposition as single, concentrated, broad or offsetting,
drills the leading group one level further, and keeps a ledger of every
figure with the SQL that produced it and an independent query that
recomputes it. No model is involved: the numbers are the source's own, and
the narrative that a model may later write can only refer to them.

Everything is bounded by ``budget``, the number of queries the sweep may
run. What did not fit is recorded in the notes.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .data_agent_review import (
    AgentSnapshot,
    ReviewContext,
    SourceSchema,
    _Joins,
    _channel_measure,
    _date_join,
    _has_month,
    _heuristic_joins,
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
    _scoped_facts,
    _sql_literal,
    _time_join,
    _year_expr,
    attribute_paths,
    build_vocabulary,
    excluded_terms,
    humanize_column,
)

__all__ = ["Comparison", "Decomposition", "Movement", "Sweep", "SweepFinding", "sweep", "verify_sweep"]


class _Budget(Exception):
    """The query budget is spent."""


_AVERAGED = re.compile(r"(score|rating|_rate$|rate$|pct|percent|ratio|index|avg|average|mean)", re.IGNORECASE)


def _aggregate_for(measure: str) -> str:
    """``avg`` for a score, rating, rate or percentage, which cannot be summed; ``sum`` for everything else."""
    return "avg" if _AVERAGED.search(measure) else "sum"


@dataclass(frozen=True)
class Comparison:
    """Two periods the time axis can express: ``kind`` is ``month``, ``same_month_prior_year`` or ``year``."""

    kind: str
    before: Mapping[str, Any]
    after: Mapping[str, Any]

    @property
    def label(self) -> str:
        return f"{_month_name(self.before)} to {_month_name(self.after)}"


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
    sql: str
    verification: Mapping[str, str]  # independent per-period queries: {"before": sql, "after": sql}
    path: Mapping[str, Any] | None = None  # the grouping path; None at the total level
    group: Any = None  # the group's value; None at the total level
    parent: tuple[tuple[str, Any], ...] = ()  # (column, value) of the groups above this one in a drill
    aggregate: str = "sum"  # sum, or avg for a score or a rate

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
    concentration: str  # single | concentrated | broad | offsetting | none
    explained: float  # share of the parent's change carried by the top three groups moving the same way

    def share_of_change(self, group: Movement) -> float | None:
        return group.delta / self.parent.delta if self.parent.delta else None

    def share_of_base(self, group: Movement) -> float | None:
        return group.before_value / self.parent.before_value if self.parent.before_value else None


@dataclass(frozen=True)
class SweepFinding:
    """A material movement with its decompositions (most concentrated first), the drill into its leading group, and the flags a reader should see first."""

    movement: Movement
    decompositions: tuple[Decomposition, ...] = ()
    drill: tuple[Decomposition, ...] = ()
    flags: tuple[str, ...] = ()

    @property
    def best(self) -> Decomposition | None:
        return self.decompositions[0] if self.decompositions else None


@dataclass(frozen=True)
class Sweep:
    source_id: str
    findings: tuple[SweepFinding, ...]
    ledger: tuple[Movement, ...]
    queries: int
    budget: int
    notes: tuple[str, ...] = ()
    words: Mapping[str, str] = field(default_factory=dict)  # "fact|measure" -> the business phrase used in the narrative

    def phrase(self, movement: Movement) -> str:
        return self.words.get(f"{movement.fact}|{movement.measure}", f"{movement.fact} {movement.measure}")

    def lines(self) -> list[str]:
        """The findings as plain sentences, every number from the ledger."""
        lines: list[str] = []
        for finding in self.findings:
            m = finding.movement
            direction = "fell" if m.delta < 0 else "rose"
            averaged = " on average" if m.aggregate == "avg" else ""
            lines.append(f"{self.phrase(m).capitalize()} {direction} {abs(m.pct or 0):.1%}{averaged} {m.comparison.label} ({_num(m.before_value)} to {_num(m.after_value)}).")
            for flag in finding.flags:
                lines.append(f"  Note: {flag}")
            best = finding.best
            if best is not None:
                lines.append(f"  By {_word(best.path)}: {_concentration_sentence(best)}")
                for drill in finding.drill:
                    leader = drill.parent
                    lines.append(f"  Within {_label(leader.group)}, by {_word(drill.path)}: {_concentration_sentence(drill)}")
            others = [d for d in finding.decompositions[1:] if d.concentration in {"single", "concentrated"} and d.groups]
            if others:
                lines.append("  Also concentrated by " + "; ".join(f"{_word(d.path)} ({_leader_sentence(d)})" for d in others[:2]))
        return lines

    def to_markdown(self) -> str:
        head = f"{len(self.findings)} material movement(s) from {len(self.ledger)} measured figures in {self.queries} of {self.budget} queries."
        body = "\n".join(("- " + line[2:] if line.startswith("  ") else f"- **{line}**") for line in self.lines())
        notes = "\n".join(f"- {n}" for n in self.notes)
        return "\n".join(part for part in (head, body, notes) if part)

    def to_html(self) -> str:
        from html import escape as esc

        parts = [f'<div class="muted">{esc(f"{len(self.findings)} material movement(s) from {len(self.ledger)} measured figures in {self.queries} of {self.budget} queries.")}</div>']
        for finding in self.findings:
            m = finding.movement
            direction = "fell" if m.delta < 0 else "rose"
            parts.append(f'<div class="card"><div><b>{esc(self.phrase(m).capitalize())} {direction} {abs(m.pct or 0):.1%} {esc(m.comparison.label)}</b> <span class="muted">({esc(_num(m.before_value))} to {esc(_num(m.after_value))})</span></div>')
            for flag in finding.flags:
                parts.append(f'<div class="muted">{esc(flag)}</div>')
            best = finding.best
            if best is not None:
                parts.append(f"<div>By {esc(_word(best.path))}: {esc(_concentration_sentence(best))}</div>")
                parts.append(_groups_table(best, esc))
                for drill in finding.drill:
                    parts.append(f"<div>Within {esc(_label(drill.parent.group))}, by {esc(_word(drill.path))}: {esc(_concentration_sentence(drill))}</div>")
                    parts.append(_groups_table(drill, esc))
            parts.append(f"<details><summary>Queries</summary><pre>{esc(m.sql)}</pre><pre>{esc(m.verification.get('after', ''))}</pre></details></div>")
        for note in self.notes:
            parts.append(f'<div class="muted">{esc(note)}</div>')
        return "\n".join(parts)


def _num(value: float) -> str:
    if abs(value) >= 1000 or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _label(value: Any) -> str:
    return "(none)" if value is None else str(value)


def _word(path: Mapping[str, Any] | None) -> str:
    return humanize_column(str(path["column"])) if path else "total"


def _leader_sentence(d: Decomposition) -> str:
    if not d.groups:
        return "no groups moved"
    lead = d.groups[0]
    share = d.share_of_change(lead)
    return f"{_label(lead.group)} carries {share:.0%}" if share is not None else _label(lead.group)


def _concentration_sentence(d: Decomposition) -> str:
    if not d.groups:
        return "no group moved the same way."
    same = [g for g in d.groups if g.delta * d.parent.delta > 0]
    opposite = [g for g in d.groups if g.delta * d.parent.delta < 0]
    parts = []
    for g in same[:3]:
        change, base = d.share_of_change(g), d.share_of_base(g)
        parts.append(f"{_label(g.group)} {change:.0%} of the change on {base:.0%} of the base" if change is not None and base is not None else _label(g.group))
    if d.parent.aggregate == "avg":
        moves = "; ".join(f"{_label(g.group)} {_num(g.before_value)} to {_num(g.after_value)}" for g in d.groups[:3])
        return f"largest moves in the average: {moves}."
    text = {
        "single": f"one group carries it, {parts[0]}.",
        "concentrated": f"concentrated: {'; '.join(parts)} (top three explain {d.explained:.0%}).",
        "proportional": f"in proportion to size, nothing stands out: {'; '.join(parts)}.",
        "broad": f"broad, spread across groups: {'; '.join(parts)} (top three explain {d.explained:.0%}).",
        "offsetting": f"offsetting moves: {'; '.join(parts)}, while {', '.join(_label(o.group) for o in opposite[:2])} moved the other way.",
    }.get(d.concentration, "; ".join(parts) + ".")
    return text


def _groups_table(d: Decomposition, esc: Callable[[str], str]) -> str:
    rows = []
    for g in d.groups:
        change, base = d.share_of_change(g), d.share_of_base(g)
        rows.append(f"<tr><td>{esc(_label(g.group))}</td><td>{esc(_num(g.before_value))}</td><td>{esc(_num(g.after_value))}</td><td>{esc(_num(g.delta))}</td><td>{'' if change is None else f'{change:.0%}'}</td><td>{'' if base is None else f'{base:.0%}'}</td></tr>")
    return f"<table><tr><th>{esc(_word(d.path))}</th><th>Before</th><th>After</th><th>Change</th><th>Share of change</th><th>Share of base</th></tr>{''.join(rows)}</table>"


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #


def sweep(
    executor: Any,
    schema: SourceSchema,
    snapshot: AgentSnapshot,
    years: Sequence[int],
    *,
    context: ReviewContext | None = None,
    facts: int = 2,
    measures: int = 2,
    paths: int = 8,
    budget: int = 120,
    min_pct: float = 0.05,
    top: int = 5,
    depth: int = 2,
) -> Sweep:
    """What moved in a source, with every figure computed by the source and recomputable.

    ``facts``, ``measures`` and ``paths`` bound the search space per fact;
    ``budget`` bounds the queries; ``min_pct`` is the relative change a
    total movement needs to be decomposed; ``depth`` 2 drills the leading
    group of the most concentrated decomposition one level further.
    """
    source = next((s for s in snapshot.datasources if s.id == schema.source_id), None)
    instructions = (source.instructions if source else "") + "\n" + snapshot.instructions + ("\n" + context.text if context is not None else "")
    joins = dict(_heuristic_joins(schema))
    joins.update(_joins_from_instructions(instructions, schema))
    excluded = excluded_terms(instructions)
    terms = context.terms if context is not None else ()
    vocabulary = build_vocabulary(snapshot, schema, context)
    ledger: list[Movement] = []
    findings: list[SweepFinding] = []
    notes: list[str] = []
    words: dict[str, str] = {}
    spent = 0

    def run(sql: str) -> list[dict[str, Any]]:
        nonlocal spent
        if spent >= budget:
            raise _Budget
        spent += 1
        return executor.run({"kind": "sql", "sql": sql})

    fact_names = _scoped_facts(schema, instructions)[:facts]
    if not fact_names:
        notes.append("no fact table recognised, nothing to sweep")
    try:
        for table in fact_names:
            date = _date_join(schema, table, joins)
            measure_list = _measure_columns(schema, table)[:measures]
            if not date or not measure_list:
                notes.append(f"{table}: no time axis or no measure, not swept")
                continue
            base_fact = {"table": table, "date": date, "measure": measure_list[0]}
            comparisons = _comparisons(run, base_fact, years)
            if not comparisons:
                notes.append(f"{table}: no comparison the time axis supports for {list(years)}")
                continue
            try:
                rows = run(_max_date_sql(base_fact))
                max_date = _iso_date(rows[0]["value"]) if rows and rows[0].get("value") is not None else None
            except _Budget:
                raise
            except Exception:  # noqa: BLE001 - the flag that needs it is skipped
                max_date = None
            all_paths = attribute_paths(schema, table, joins, excluded)
            roles = _paths_by_role(all_paths, terms)
            ordered: list[Mapping[str, Any]] = list(roles.values()) + [p for p in all_paths if all(p is not r for r in roles.values())]
            chosen: list[Mapping[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for path in ordered:
                key = (_path_table(path, table), str(path["column"]))
                if key not in seen:
                    seen.add(key)
                    chosen.append(path)
                if len(chosen) >= paths:
                    break
            for measure in measure_list:
                fact = {"table": table, "date": date, "measure": measure}
                words[f"{table}|{measure}"] = _channel_measure(vocabulary.table(table), vocabulary.measure(measure))
                for comparison in comparisons:
                    total = _movement(run, fact, comparison, None, None, ())
                    ledger.append(total)
                    if not total.material(min_pct):
                        continue
                    decompositions: list[Decomposition] = []
                    for path in chosen:
                        groups = _grouped(run, fact, comparison, path, ())
                        ledger.extend(groups)
                        decompositions.append(_decompose(total, path, groups, top))
                    decompositions.sort(key=lambda d: (-_rank(d), -d.explained))
                    drill: list[Decomposition] = []
                    best = decompositions[0] if decompositions else None
                    if depth >= 2 and best is not None and best.concentration in {"single", "concentrated"} and best.groups:
                        leader = best.groups[0]
                        for path in chosen:
                            if (_path_table(path, table), str(path["column"])) == (_path_table(best.path, table), str(best.path["column"])):
                                continue
                            groups = _grouped(run, fact, comparison, path, ((best.path, leader.group),))
                            ledger.extend(groups)
                            drill.append(_decompose(leader, path, groups, top))
                            if len(drill) >= 3:
                                break
                        drill.sort(key=lambda d: (-_rank(d), -d.explained))
                    findings.append(SweepFinding(total, tuple(decompositions), tuple(drill[:1]), _flags(total, comparison, max_date)))
    except _Budget:
        notes.append(f"the budget of {budget} queries was spent before the sweep finished; later facts, measures or comparisons were not swept")
    findings.sort(key=lambda f: (any("coverage" in flag for flag in f.flags), -(_rank(f.best) if f.best else -1), -abs(f.movement.pct or 0)))
    return Sweep(schema.source_id, tuple(findings), tuple(ledger), spent, budget, tuple(notes), words)


def _rank(d: Decomposition) -> int:
    return {"single": 4, "concentrated": 3, "offsetting": 2, "broad": 1, "proportional": 0, "none": -1}.get(d.concentration, 0)


def _comparisons(run: Callable[[str], list[dict[str, Any]]], fact: Mapping[str, Any], years: Sequence[int]) -> list[Comparison]:
    """The comparisons the time axis supports: the latest complete year against the one before, the latest month with data against the month before and against the same month a year earlier."""
    years = [int(y) for y in years]
    comparisons: list[Comparison] = []
    if len(years) >= 2:
        comparisons.append(Comparison("year", {"year": years[-2]}, {"year": years[-1]}))
    dt = fact["date"]
    if years and _has_month(dt):
        year_expr, month_expr = _year_expr(dt, "duckdb"), _month_expr(dt, "duckdb")
        join = _time_join(dt)
        span = ", ".join(str(y) for y in years[-2:])
        sql = f"SELECT {year_expr} AS year, {month_expr} AS month, COUNT(*) AS n FROM {fact['table']} f" + (f" {join}" if join else "") + f" WHERE {year_expr} IN ({span}) GROUP BY {year_expr}, {month_expr} ORDER BY {year_expr}, {month_expr}"
        months = [(int(r["year"]), int(r["month"])) for r in run(sql) if r.get("year") is not None and r.get("month") is not None]
        if months:
            latest = months[-1]
            if len(months) >= 2:
                previous = months[-2]
                consecutive = (previous[0] == latest[0] and previous[1] == latest[1] - 1) or (previous[0] == latest[0] - 1 and previous[1] == 12 and latest[1] == 1)
                if consecutive:
                    comparisons.append(Comparison("month", {"year": previous[0], "month": previous[1]}, {"year": latest[0], "month": latest[1]}))
            if (latest[0] - 1, latest[1]) in months:
                comparisons.append(Comparison("same_month_prior_year", {"year": latest[0] - 1, "month": latest[1]}, {"year": latest[0], "month": latest[1]}))
    return comparisons


def _aggregates(fact: Mapping[str, Any], before: str, after: str) -> str:
    m = f"f.{_q(fact['measure'])}"
    if _aggregate_for(fact["measure"]) == "avg":
        return f"AVG(CASE WHEN {after} THEN {m} END) AS after_value, AVG(CASE WHEN {before} THEN {m} END) AS before_value, SUM(CASE WHEN {after} THEN 1 ELSE 0 END) AS after_rows, SUM(CASE WHEN {before} THEN 1 ELSE 0 END) AS before_rows"
    return f"SUM(CASE WHEN {after} THEN {m} ELSE 0 END) AS after_value, SUM(CASE WHEN {before} THEN {m} ELSE 0 END) AS before_value, SUM(CASE WHEN {after} THEN 1 ELSE 0 END) AS after_rows, SUM(CASE WHEN {before} THEN 1 ELSE 0 END) AS before_rows"


def _verification(fact: Mapping[str, Any], from_clause: str, before: str, after: str, filters: Sequence[str]) -> dict[str, str]:
    function = "AVG" if _aggregate_for(fact["measure"]) == "avg" else "SUM"
    m = f"f.{_q(fact['measure'])}"
    return {
        "before": f"SELECT {function}({m}) AS value {from_clause} WHERE " + " AND ".join([f"({before})", *filters]),
        "after": f"SELECT {function}({m}) AS value {from_clause} WHERE " + " AND ".join([f"({after})", *filters]),
    }


def _movement(run: Callable[[str], list[dict[str, Any]]], fact: Mapping[str, Any], comparison: Comparison, path: Mapping[str, Any] | None, group: Any, parent: tuple[tuple[Mapping[str, Any], Any], ...]) -> Movement:
    """The total-level movement (path None), or one filtered group when a drill needs a parent figure."""
    before, after = _period_condition(fact, comparison.before, "duckdb"), _period_condition(fact, comparison.after, "duckdb")
    joins = _Joins()
    filters = [_filter(joins, p, v) for p, v in parent]
    if path is not None:
        filters.append(_filter(joins, path, group))
    where = " AND ".join([f"(({before}) OR ({after}))", *filters])
    time_join = _time_join(fact["date"])
    from_clause = f"FROM {fact['table']} f" + (f" {time_join}" if time_join else "") + "".join(" " + c for c in joins.clauses)
    sql = f"SELECT {_aggregates(fact, before, after)} {from_clause} WHERE {where}"
    rows = run(sql)
    row = rows[0] if rows else {}
    verification = _verification(fact, from_clause, before, after, filters)
    return Movement(fact["table"], fact["measure"], comparison, float(row.get("before_value") or 0), float(row.get("after_value") or 0), int(row.get("before_rows") or 0), int(row.get("after_rows") or 0), sql, verification, path, group, tuple((str(p["column"]), v) for p, v in parent), _aggregate_for(fact["measure"]))


def _filter(joins: _Joins, path: Mapping[str, Any], value: Any) -> str:
    ref = joins.ref(path)
    return f"{ref} IS NULL" if value is None else f"{ref} = {_sql_literal(value)}"


def _grouped(run: Callable[[str], list[dict[str, Any]]], fact: Mapping[str, Any], comparison: Comparison, path: Mapping[str, Any], parent: tuple[tuple[Mapping[str, Any], Any], ...]) -> list[Movement]:
    """Every group of one path in one query, within the parent groups of a drill."""
    before, after = _period_condition(fact, comparison.before, "duckdb"), _period_condition(fact, comparison.after, "duckdb")
    joins = _Joins()
    filters = [_filter(joins, p, v) for p, v in parent]
    label = joins.ref(path)
    where = " AND ".join([f"(({before}) OR ({after}))", *filters])
    time_join = _time_join(fact["date"])
    from_clause = f"FROM {fact['table']} f" + (f" {time_join}" if time_join else "") + "".join(" " + c for c in joins.clauses)
    sql = f"SELECT {label} AS label, {_aggregates(fact, before, after)} {from_clause} WHERE {where} GROUP BY {label}"
    movements = []
    parent_columns = tuple((str(p["column"]), v) for p, v in parent)
    for row in run(sql):
        group = row.get("label")
        group_filters = [*filters, _filter(joins, path, group)]
        verification = _verification(fact, from_clause, before, after, group_filters)
        movements.append(Movement(fact["table"], fact["measure"], comparison, float(row.get("before_value") or 0), float(row.get("after_value") or 0), int(row.get("before_rows") or 0), int(row.get("after_rows") or 0), sql, verification, path, group, parent_columns, _aggregate_for(fact["measure"])))
    return movements


def _decompose(parent: Movement, path: Mapping[str, Any], groups: Sequence[Movement], top: int) -> Decomposition:
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
    shares = [g.delta / total for g in same]
    explained = sum(shares[:3])
    base = parent.before_value

    def excess(g: Movement, share: float) -> float:
        base_share = g.before_value / base if base else 0.0
        return share - base_share

    if opposite and abs(sum(o.delta for o in opposite)) >= 0.5 * abs(total):
        concentration = "offsetting"
    elif same and shares[0] >= 0.5 and excess(same[0], shares[0]) >= 0.1:
        concentration = "single"
    elif same and explained >= 0.7 and excess(same[0], shares[0]) >= 0.1:
        concentration = "concentrated"
    elif same and all(abs(excess(g, s)) < 0.1 for g, s in zip(same[:3], shares[:3])):
        concentration = "proportional"
    else:
        concentration = "broad"
    return Decomposition(parent, path, tuple(same[:top] + opposite[:2]), concentration, explained)


def _flags(total: Movement, comparison: Comparison, max_date: str | None) -> tuple[str, ...]:
    flags: list[str] = []
    pct, rows_pct = total.pct, total.rows_pct
    if total.aggregate == "sum" and pct is not None and rows_pct is not None and rows_pct * pct > 0 and abs(rows_pct) >= 0.5 * abs(pct):
        flags.append(f"row counts moved {rows_pct:+.0%} against {pct:+.0%} in value, so this is volume or coverage rather than a change in rate")
    if max_date and comparison.after.get("month") is not None:
        try:
            end = _dt.date.fromisoformat(max_date)
        except ValueError:
            end = None
        if end is not None and end.year == int(comparison.after["year"]) and end.month == int(comparison.after["month"]) and end.day < 25:
            flags.append(f"the data ends on {max_date}, so {_month_name(comparison.after)} is incomplete")
    if total.before_rows and total.before_rows < 30:
        flags.append(f"a small base: {total.before_rows} rows before, {total.after_rows} after")
    return tuple(flags)


def verify_sweep(sweep_result: Sweep, executor: Any, *, tolerance: float = 1e-6) -> list[str]:
    """Recompute every finding's headline and group figures with their independent per-period queries; the mismatches, as text."""
    mismatches: list[str] = []
    checked: set[str] = set()

    def check(m: Movement, label: str) -> None:
        key = m.verification["after"] + m.verification["before"]
        if key in checked:
            return
        checked.add(key)
        for side, expected in (("before", m.before_value), ("after", m.after_value)):
            rows = executor.run({"kind": "sql", "sql": m.verification[side]})
            actual = float((rows[0] or {}).get("value") or 0) if rows else 0.0
            if abs(actual - expected) > tolerance * max(1.0, abs(expected)):
                mismatches.append(f"{label} {side}: expected {expected}, recomputed {actual}")

    for finding in sweep_result.findings:
        check(finding.movement, f"{finding.movement.fact}.{finding.movement.measure} {finding.movement.comparison.label}")
        for d in (*finding.decompositions[:1], *finding.drill):
            for g in d.groups[:3]:
                check(g, f"{g.fact}.{g.measure} {g.comparison.label} {_word(g.path)}={_label(g.group)}")
    return mismatches
