"""Monday Morning Brief: the metrics you track, analysed over last week with the history as context.

Name the metrics, in plain words or as specs, and the brief takes the latest
complete Monday-to-Sunday week the source holds, measures each metric for
that week with the source's own engine, and puts it in context: the week
before, the same week a year earlier, the last four and thirteen weeks, the
seasonal expectation from prior years, the rank against the record. It
looks for level shifts in the weekly history, calls a week unusual when it
sits far from what the season and the recent level predicted, decomposes
the week's move by the groupings named (or the ones the joins reach),
splits it into volume and rate, states the counterfactual for the leading
group, reads the day-of-week pattern, and notes which metrics move
together, and which lead. Every figure is the source's own and
recomputable; no model writes a number. What it says about causes is what
the history supports: attribution, not proof.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .data_agent_review import AgentDataSource, AgentSnapshot, ReviewContext, _measure_columns, build_vocabulary, excluded_terms
from .sweep import Comparison, Sweep, SweepFinding, _aggregate_of, _concentration_sentence, _iso_date, _label, _num, _probe_for, _summed_columns, _word, sweep, verify_sweep

__all__ = ["Brief", "ChangePoint", "MetricBrief", "Week", "brief"]

_WEEKDAYS = tuple(calendar.day_name)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Week:
    """One Monday-to-Sunday week of a metric: the value (a sum, or an average for a score or a rate) and the rows behind it."""

    start: str  # ISO Monday
    value: float
    rows: int
    days: int  # days with rows

    @property
    def end(self) -> str:
        return (_dt.date.fromisoformat(self.start) + _dt.timedelta(days=7)).isoformat()

    @property
    def label(self) -> str:
        return f"week of {_day_label(self.start)}"

    @property
    def period(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "label": self.label}


@dataclass(frozen=True)
class ChangePoint:
    """A level shift in the weekly history: the first week of the new level, the average before and after, and how sharp the split is."""

    start: str
    before: float
    after: float
    statistic: float

    @property
    def pct(self) -> float | None:
        return (self.after - self.before) / abs(self.before) if self.before else None


@dataclass(frozen=True)
class MetricBrief:
    """One metric's week: the figures, the context, the analysis and the sentences."""

    name: str
    fact: str
    measure: str
    groupings: tuple[str, ...]
    aggregate: str
    weeks: tuple[Week, ...]  # complete weeks, oldest first
    target: Week | None  # the week briefed
    context: Mapping[str, Any] = field(default_factory=dict)
    change_points: tuple[ChangePoint, ...] = ()
    weekday_shares: Mapping[int, tuple[float, float]] = field(default_factory=dict)  # weekday -> (share this week, usual share)
    pattern: str = ""
    sweep: Sweep | None = None  # the week-over-week and same-week-prior-year movements, decomposed
    mix: Mapping[str, float] = field(default_factory=dict)  # volume, rate and interaction shares of the week-over-week change
    explanations: tuple[str, ...] = ()
    headline: str = ""
    notes: tuple[str, ...] = ()
    kind: str = "measure"  # measure, or a KPI kind: new, active, retained, resurrected, churned, ratio, crossing, concentration
    definition: str = ""  # what the number means, printed next to it
    extra: Mapping[str, Any] = field(default_factory=dict)  # the series behind a KPI: growth accounting, the two sides of a ratio or a crossing, the shares

    @property
    def finding(self) -> SweepFinding | None:
        """The week-over-week movement with its drivers; against an empty week before, the same week last year instead."""
        return _metric_finding(self.sweep, self.context)

    @property
    def prior_year_finding(self) -> SweepFinding | None:
        if self.sweep is None:
            return None
        return next((f for f in self.sweep.findings if f.movement.comparison.kind == "same_week_prior_year"), None)

    def lines(self) -> list[str]:
        lines = [self.headline]
        if self.definition:
            lines.append(f"  Definition: {self.definition}.")
        c = self.context
        if c.get("rank_note"):
            lines.append(f"  {c['rank_note']}")
        if c.get("streak_note"):
            lines.append(f"  {c['streak_note']}")
        if c.get("trend_note"):
            lines.append(f"  {c['trend_note']}")
        for point in self.change_points[:2]:
            lines.append(f"  Level shift the {_week_label(point.start)}: the weekly average went from {_num(point.before)} to {_num(point.after)} ({point.pct:+.0%})." if point.pct is not None else f"  Level shift the {_week_label(point.start)}.")
        lines.extend(f"  {text}" for text in self.explanations)
        if self.pattern:
            lines.append(f"  {self.pattern}")
        lines.extend(f"  Note: {note}" for note in self.notes)
        return lines


@dataclass(frozen=True)
class Brief:
    """The brief for one week over one source: every metric's analysis, what moves together, and what to watch."""

    source: str
    kind: str
    week: Week | None
    metrics: tuple[MetricBrief, ...]
    comovement: tuple[str, ...] = ()
    watch: tuple[str, ...] = ()
    queries: int = 0
    budget: int = 0
    notes: tuple[str, ...] = ()
    recomputed: int = 0
    mismatches: tuple[str, ...] = ()
    verified: bool = False
    elapsed: float = 0.0
    entities: tuple[Any, ...] = ()  # the entity candidates found on the fact, best first
    entity_choice: str = ""  # which entity the lifecycle KPIs count, and why

    @property
    def title(self) -> str:
        return f"Monday Morning Brief: {self.source}"

    @property
    def definitions(self) -> list[tuple[str, str]]:
        return [(m.name, m.definition) for m in self.metrics if m.definition]

    @property
    def week_label(self) -> str:
        if self.week is None:
            return "no complete week found"
        start = _dt.date.fromisoformat(self.week.start)
        end = start + _dt.timedelta(days=6)
        return f"{start.strftime('%A %d %B')} to {end.strftime('%A %d %B %Y')}"

    def lines(self) -> list[str]:
        lines: list[str] = []
        for metric in self.metrics:
            lines.extend(metric.lines())
        if self.comovement:
            lines.append("Across metrics:")
            lines.extend(f"  {text}" for text in self.comovement)
        if self.watch:
            lines.append("Watch:")
            lines.extend(f"  {text}" for text in self.watch)
        return lines

    def summary(self) -> str:
        text = f"{len(self.metrics)} metric(s) over the {self.week.label if self.week else 'latest week'}, {self.queries} of {self.budget} queries"
        if self.verified:
            text += f"; {self.recomputed} figure(s) recomputed independently, {len(self.mismatches)} mismatch(es)"
        return text + "."

    def verification_statement(self) -> str:
        """One sentence the badge and the footer both follow."""
        total = sum(len(m.sweep.ledger) for m in self.metrics if m.sweep is not None) + sum(len(m.weeks) for m in self.metrics)
        if not self.verified:
            return f"Every one of the {total:,} figures was computed by the source; none was recomputed in this run."
        rest = total - self.recomputed
        text = f"Every one of the {total:,} figures (the weekly series and the movements) was computed by the source. {self.recomputed:,} of them, the week-over-week and year-ago movements and the groups of their leading decompositions, were recomputed by independent per-period queries"
        text += f" with {len(self.mismatches)} mismatch{'es' if len(self.mismatches) != 1 else ''}." if self.mismatches else " and all matched."
        if rest > 0:
            text += f" The other {rest:,} come from the daily series query and were not rerun."
        return text

    def narrative(self) -> str:
        """The week in three sentences, every number from the metrics: the direction of each metric, the verdict that matters most, the driver of the largest move."""
        if not self.metrics or self.week is None:
            return ""
        moves = []
        for metric in self.metrics:
            pct = metric.context.get("wow_pct")
            if metric.kind == "concentration" and metric.target is not None:
                moves.append(f"{metric.name} held at {metric.target.value:.0%}" if pct is not None and abs(pct) < 0.02 else f"{metric.name} {'fell' if (pct or 0) < 0 else 'rose'} to {metric.target.value:.0%}")
            elif pct is not None:
                moves.append(f"{metric.name} {'fell' if pct < 0 else 'rose'} {abs(pct):.0%}")
        first = f"In the {self.week.label}, " + (", ".join(moves[:-1]) + (" and " if len(moves) > 1 else "") + moves[-1] + " on the week before." if moves else "no metric has a week before with rows to compare with.")
        sentences = [first[0].upper() + first[1:]]
        judged = [m for m in self.metrics if m.context.get("z") is not None and abs(m.context["z"]) >= 1.5]
        if judged:
            m = judged[0]
            sentences.append(f"{m.name.capitalize()} is {m.context['verdict']}, against {m.context.get('expected_source', 'the recent level')}.")
        else:
            steady = [m.name for m in self.metrics if m.context.get("verdict") == "within the usual range"]
            if steady:
                sentences.append(f"{', '.join(steady).capitalize()} {'is' if len(steady) == 1 else 'are'} within the usual range once the season and the recent level are taken into account.")
        largest = max((m for m in self.metrics if m.context.get("wow_pct") is not None and m.finding is not None and m.finding.best is not None and m.finding.best.groups), key=lambda m: abs(m.context["wow_pct"]), default=None)
        if largest is not None:
            best = largest.finding.best
            lead = best.groups[0]
            share, base = best.share_of_change(lead), best.share_of_base(lead)
            if best.concentration in {"single", "concentrated"} and share is not None and base is not None:
                sentences.append(f"The move in {largest.name} sits with {_label(lead.group)} ({_word(best.path)}), {share:.0%} of the change on {base:.0%} of the base.")
            elif best.concentration == "proportional":
                sentences.append(f"The move in {largest.name} is spread across {_word(best.path)} groups in proportion to their size, so no single group explains it.")
            elif best.concentration == "offsetting":
                sentences.append(f"The move in {largest.name} nets offsetting moves across {_word(best.path)} groups, {_label(lead.group)} the largest.")
        recent_shift = next((m for m in self.metrics for p in m.change_points if m.target and (_dt.date.fromisoformat(m.target.start) - _dt.date.fromisoformat(p.start)).days <= 42), None)
        if recent_shift is not None:
            point = next(p for p in recent_shift.change_points if (_dt.date.fromisoformat(recent_shift.target.start) - _dt.date.fromisoformat(p.start)).days <= 42)
            sentences.append(f"{recent_shift.name.capitalize()} has been running at a new level since the {_week_label(point.start)} ({_pct(point.pct)} on the average before).")
        return " ".join(sentences)

    def to_markdown(self) -> str:
        head = [f"# {self.title}", f"Week: {self.week_label}", "", self.summary(), "", self.narrative(), ""]
        if self.entity_choice:
            head.extend([f"Entities: {self.entity_choice}", ""])
        body = []
        for line in self.lines():
            body.append(("- " + line[2:]) if line.startswith("  ") else f"\n**{line}**\n")
        notes = [""] + [f"- {n}" for n in self.notes] if self.notes else []
        return "\n".join(head + body + notes)

    def to_html(self) -> str:
        from .sweep_dashboard import render_brief

        return render_brief(self)

    def save(self, path: str) -> str:
        from .sweep_dashboard import document_brief

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(document_brief(self))
        return path


def _day_label(iso: str) -> str:
    day = _dt.date.fromisoformat(iso)
    return f"{day.day} {day.strftime('%b %Y')}"


def _week_label(iso: str) -> str:
    return f"week of {_day_label(iso)}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


# --------------------------------------------------------------------------- #
# The metrics named
# --------------------------------------------------------------------------- #


def _metric_specs(metrics: Sequence[Any], probe: Any, instructions: str, scope: str) -> list[dict[str, Any]]:
    """Each metric as {name, fact, measure, groupings}: a spec mapping as given, plain words read against the source."""
    from .reports import _GROUPING_CLAUSE, _grouping_phrases, _match_facts, _match_groupings, _match_measures

    schema = probe.schema
    context = ReviewContext(scope=scope) if scope else None
    snapshot = AgentSnapshot(agent_id="brief", name=probe.name, instructions=instructions, datasources=(AgentDataSource(id=schema.source_id, kind=schema.kind, name=probe.name),))
    vocabulary = build_vocabulary(snapshot, schema, context)
    text = "\n".join(p for p in (instructions, context.text if context is not None else "") if p)
    joins = probe.joins(text)
    excluded = excluded_terms(text)
    terms = context.terms if context is not None else ()
    default_facts = list(probe.facts(text))
    specs: list[dict[str, Any]] = []
    for metric in metrics:
        if isinstance(metric, Mapping):
            fact = str(metric.get("fact") or (default_facts[0] if default_facts else ""))
            lowered = {t.casefold(): t for t in schema.tables}
            fact = lowered.get(fact.casefold(), fact)
            measure = str(metric.get("measure", ""))
            groupings = tuple(str(g) for g in (metric.get("by") or ()))
            name = str(metric.get("name") or f"{vocabulary.table(fact)} {vocabulary.measure(measure.strip('[]'))}" if fact in schema.tables else measure)
            specs.append({"name": name, "fact": fact, "measure": measure, "groupings": groupings})
            continue
        words = str(metric)
        facts = _match_facts(words, probe, vocabulary, text) or default_facts[:1]
        if not facts:
            specs.append({"name": words, "fact": "", "measure": "", "groupings": (), "note": "no fact table found for it"})
            continue
        fact = facts[0]
        measures, _matched = _match_measures(_GROUPING_CLAUSE.sub(" ", words), schema, fact, vocabulary)
        measure = measures[0] if measures else (_measure_columns(schema, fact)[:1] or [""])[0]
        groupings, unmatched = _match_groupings(_grouping_phrases(words), schema, fact, joins, excluded, terms) if _grouping_phrases(words) else ([], [])
        phrase = vocabulary.measure(measure.strip("[]")) if measure else words
        table_word = vocabulary.table(fact)
        name = phrase if phrase.split()[:1] == table_word.split()[-1:] or table_word in phrase else f"{table_word} {phrase}"
        specs.append({"name": name, "fact": fact, "measure": measure, "groupings": tuple(groupings), "note": (f"no grouping matched {', '.join(unmatched)}" if unmatched else "")})
    return specs


# --------------------------------------------------------------------------- #
# Weeks, context, change points, patterns
# --------------------------------------------------------------------------- #


def _weeks(daily: Sequence[tuple[_dt.date, int, float]], aggregate: str, max_day: _dt.date) -> list[Week]:
    """Complete Monday-to-Sunday weeks (the Sunday on or before the last day with data), oldest first."""
    by_week: dict[_dt.date, list[float]] = {}
    for day, rows, value in daily:
        monday = day - _dt.timedelta(days=day.weekday())
        bucket = by_week.setdefault(monday, [0.0, 0, 0])
        bucket[0] += value
        bucket[1] += rows
        bucket[2] += 1
    if not by_week:
        return []
    weeks = []
    monday = min(by_week)
    while monday + _dt.timedelta(days=6) <= max_day:  # only weeks whose Sunday is in the data
        value, rows, days = by_week.get(monday, [0.0, 0, 0])  # a week with no rows is a week of zero, not a missing week
        weeks.append(Week(monday.isoformat(), (value / rows if rows else 0.0) if aggregate == "avg" else value, int(rows), int(days)))
        monday += _dt.timedelta(days=7)
    return weeks


def _last_covered_week(weeks: Sequence[Week], *, back: int = 8) -> int:
    """The index of the latest week with real coverage: a trailing week holding under half the rows of a typical recent week is a stub."""
    index = len(weeks) - 1
    steps = 0
    while index > 0 and steps < back:
        recent = sorted(w.rows for w in weeks[max(0, index - 13) : index] if w.rows)
        if len(recent) < 4 or weeks[index].rows >= 0.5 * recent[len(recent) // 2]:
            break
        index -= 1
        steps += 1
    return index


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _slope_pct(values: Sequence[float]) -> float | None:
    """The fitted change over the window: where the fitted line ends against where it starts, a decline capped at -100%."""
    n = len(values)
    if n < 4:
        return None
    xs = list(range(n))
    x_mean, y_mean = _mean([float(x) for x in xs]), _mean(values)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if not denominator or not y_mean:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denominator
    start = y_mean - slope * x_mean
    end = y_mean + slope * (n - 1 - x_mean)
    if start <= 0 or start < 0.05 * abs(y_mean):
        return None  # a fitted line starting at or near zero has no base to state a change against
    return max(-1.0, (end - start) / start)


def _context(weeks: Sequence[Week], index: int, aggregate: str) -> dict[str, Any]:
    """This week against the week before, the same week a year earlier, the recent averages, the seasonal expectation and the record."""
    target = weeks[index]
    values = [w.value for w in weeks]
    c: dict[str, Any] = {"value": target.value, "rows": target.rows}
    # a series with rows in few of its weeks (monthly postings, a rare category) is at the wrong grain for a weekly comparison: say so instead of judging it
    history = weeks[max(0, index - 51) : index + 1]
    present = sum(1 for w in history if w.rows)
    sparse = len(history) >= 8 and present < 0.6 * len(history)
    c["sparse"] = f"rows in only {present} of the last {len(history)} weeks" if sparse else None
    previous = weeks[index - 1] if index >= 1 else None
    c["previous"] = previous.value if previous else None
    c["previous_empty"] = bool(previous is not None and not previous.rows)
    c["wow_pct"] = (target.value - previous.value) / abs(previous.value) if previous and previous.value else None
    starts = {w.start: i for i, w in enumerate(weeks)}
    year_ago = (_dt.date.fromisoformat(target.start) - _dt.timedelta(days=364)).isoformat()
    prior = weeks[starts[year_ago]] if year_ago in starts else None
    c["prior_year"] = prior.value if prior else None
    c["prior_year_start"] = prior.start if prior else None
    c["yoy_pct"] = (target.value - prior.value) / abs(prior.value) if prior and prior.value else None
    c["avg4"] = _mean(values[max(0, index - 4) : index]) if index >= 4 and not sparse else None
    c["vs_avg4_pct"] = (target.value - c["avg4"]) / abs(c["avg4"]) if c["avg4"] else None
    c["avg13"] = _mean(values[max(0, index - 13) : index]) if index >= 8 and not sparse else None
    c["vs_avg13_pct"] = (target.value - c["avg13"]) / abs(c["avg13"]) if c["avg13"] else None
    # the seasonal expectation: the recent level scaled by how this week of the year sat against its own recent level in prior years
    expected, index_values, source = _expectation(weeks, index) if not sparse else (None, [], None)
    c["expected"] = expected
    c["expected_source"] = source
    c["vs_expected_pct"] = (target.value - expected) / abs(expected) if expected else None
    residuals = []
    for i in range(max(8, index - 26), index if not sparse else 0):
        e, _iv, _s = _expectation(weeks, i)
        if e:
            residuals.append(weeks[i].value - e)
    sigma = _std(residuals)
    c["sigma"] = sigma
    deviation = (target.value - expected) if expected else None
    if deviation is not None and sigma:
        c["z"] = deviation / sigma
    elif deviation is not None and len(residuals) >= 8:
        # a history that never left its expectation: any deviation is out of range
        c["z"] = 0.0 if abs(deviation) <= 1e-9 * max(1.0, abs(expected)) else math.copysign(math.inf, deviation)
    else:
        c["z"] = None
    if sparse:
        c["verdict"] = f"the data has {c['sparse']}, so a weekly comparison says little about it; a monthly grain would fit it better"
    elif len(residuals) < 8 or c["z"] is None:
        c["verdict"] = "not enough history to say whether this is unusual"
    elif math.isinf(c["z"]):
        c["verdict"] = f"very unusual: {'above' if c['z'] > 0 else 'below'} an expectation the last {len(residuals)} weeks never deviated from"
    elif abs(c["z"]) < 1.5:
        c["verdict"] = "within the usual range"
    elif abs(c["z"]) < 2.5:
        c["verdict"] = f"unusual: {abs(c['z']):.1f} standard deviations {'above' if c['z'] > 0 else 'below'} the expectation"
    else:
        c["verdict"] = f"very unusual: {abs(c['z']):.1f} standard deviations {'above' if c['z'] > 0 else 'below'} the expectation"
    if c.get("z") is not None and abs(c["z"]) >= 1.5 and len(index_values) == 1 and c.get("yoy_pct") is not None and abs(c["yoy_pct"]) >= 1.0:
        c["verdict"] += ", though the seasonal pattern comes from a single prior year that ran at a very different level"
    # coverage: a week with far fewer rows than the weeks before it may be data that has not fully arrived
    recent_rows = sorted(w.rows for w in weeks[max(0, index - 13) : index] if w.rows)
    if recent_rows and len(recent_rows) >= 4:
        typical_rows = recent_rows[len(recent_rows) // 2]
        if target.rows < 0.5 * typical_rows:
            c["coverage_note"] = f"this week holds {target.rows:,} rows against a typical {typical_rows:,} a week, so the data may be incomplete; read the movement as coverage until it fills"
    # the record
    earlier = values[:index]
    if earlier:
        if target.value > max(earlier):
            c["rank_note"] = f"The highest week on record ({len(earlier)} weeks of history)."
        elif target.value < min(earlier):
            c["rank_note"] = f"The lowest week on record ({len(earlier)} weeks of history)."
        else:
            higher = [i for i in range(index) if values[i] > target.value]
            lower = [i for i in range(index) if values[i] < target.value]
            if higher and index - higher[-1] >= 8 and all(values[i] < target.value for i in range(higher[-1] + 1, index)):
                c["rank_note"] = f"The highest week since the {_week_label(weeks[higher[-1]].start)}."
            elif lower and index - lower[-1] >= 8 and all(values[i] > target.value for i in range(lower[-1] + 1, index)):
                c["rank_note"] = f"The lowest week since the {_week_label(weeks[lower[-1]].start)}."
        window = values[max(0, index - 52) : index + 1]
        c["percentile"] = sum(1 for v in window if v <= target.value) / len(window)
    # the streak
    streak = 0
    i = index
    while i >= 1 and (values[i] - values[i - 1]) * (values[index] - values[index - 1] if index >= 1 else 0) > 0:
        streak += 1
        i -= 1
    c["streak"] = streak
    if streak >= 3:
        direction = "rise" if values[index] > values[index - 1] else "decline"
        c["streak_note"] = f"The {_ordinal(streak)} consecutive weekly {direction}."
    # the trend
    trend13 = _slope_pct(values[max(0, index - 12) : index + 1])
    trend52 = _slope_pct(values[max(0, index - 51) : index + 1]) if index >= 20 else None
    c["trend13_pct"] = trend13
    c["trend52_pct"] = trend52
    if trend13 is not None and abs(trend13) >= 0.10:
        c["trend_note"] = f"Over the last 13 weeks the fitted trend is {'up' if trend13 > 0 else 'down'} {abs(trend13):.0%}" + (f"; over 52 weeks {'up' if trend52 > 0 else 'down'} {abs(trend52):.0%}." if trend52 is not None else ".")
    elif trend52 is not None and abs(trend52) >= 0.15:
        c["trend_note"] = f"Flat over the last 13 weeks; over 52 weeks the fitted trend is {'up' if trend52 > 0 else 'down'} {abs(trend52):.0%}."
    return c


def _expectation(weeks: Sequence[Week], index: int) -> tuple[float | None, list[float], str]:
    """What this week should have been: the average of the 13 weeks before it, scaled by the seasonal index of prior years when there are any."""
    if index < 4:
        return None, [], ""
    values = [w.value for w in weeks]
    level = _mean(values[max(0, index - 13) : index])
    if not level:
        return None, [], ""
    starts = {w.start: i for i, w in enumerate(weeks)}
    target = _dt.date.fromisoformat(weeks[index].start)
    ratios: list[float] = []
    for years_back in (1, 2, 3):
        anchor = (target - _dt.timedelta(days=364 * years_back)).isoformat()
        j = starts.get(anchor)
        if j is None or j < 4:
            continue
        neighbours = [values[k] for k in (j - 1, j, j + 1) if 0 <= k < index]
        prior_level = _mean(values[max(0, j - 13) : j])
        if neighbours and prior_level:
            ratios.append(_mean(neighbours) / prior_level)
    if ratios:
        return level * _mean(ratios), ratios, f"the average of the 13 weeks before, scaled by how this week of the year ran against its own recent level in {len(ratios)} prior year(s)"
    return level, [], "the average of the 13 weeks before (no prior year to take the season from)"


def _ordinal(n: int) -> str:
    words = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}
    if n in words:
        return words[n]
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _change_points(weeks: Sequence[Week], *, min_size: int = 4, threshold: float = 3.0, limit: int = 2) -> list[ChangePoint]:
    """Level shifts by binary segmentation: the split that separates the means most, when the separation is sharp against the noise within the segments."""
    values = [w.value for w in weeks]
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

    def search(lo: int, hi: int, depth: int) -> None:
        if hi - lo < 2 * min_size or depth > 3 or len(found) >= limit:
            return
        best = split(lo, hi)
        if best is None or best[1] < threshold:
            return
        k = best[0]
        found.append(ChangePoint(weeks[k].start, _mean(values[lo:k]), _mean(values[k:hi]), round(best[1], 1)))
        search(k, hi, depth + 1)  # the most recent shift matters most
        search(lo, k, depth + 1)

    if len(values) >= 2 * min_size:
        search(0, len(values), 0)
    return sorted(found, key=lambda p: p.start, reverse=True)[:limit]


def _weekday_pattern(daily: Sequence[tuple[_dt.date, int, float]], target: Week, weeks: Sequence[Week], index: int) -> tuple[dict[int, tuple[float, float]], str]:
    """How the week's value spread over its days against the usual spread of the twelve weeks before."""
    start = _dt.date.fromisoformat(target.start)
    usual_starts = {_dt.date.fromisoformat(w.start) for w in weeks[max(0, index - 12) : index]}
    this_week: dict[int, float] = {}
    usual: dict[_dt.date, dict[int, float]] = {}
    for day, _rows, value in daily:
        monday = day - _dt.timedelta(days=day.weekday())
        if monday == start:
            this_week[day.weekday()] = this_week.get(day.weekday(), 0.0) + value
        elif monday in usual_starts:
            usual.setdefault(monday, {})[day.weekday()] = usual.get(monday, {}).get(day.weekday(), 0.0) + value
    total = sum(this_week.values())
    shares: dict[int, tuple[float, float]] = {}
    usual_shares: dict[int, list[float]] = {}
    for week_values in usual.values():
        week_total = sum(week_values.values())
        if week_total:
            for weekday in range(7):
                usual_shares.setdefault(weekday, []).append(week_values.get(weekday, 0.0) / week_total)
    for weekday in range(7):
        shares[weekday] = ((this_week.get(weekday, 0.0) / total) if total else 0.0, _mean(usual_shares.get(weekday, [])) if usual_shares.get(weekday) else 0.0)
    if not total or not usual:
        return shares, ""
    active = sum(1 for v in this_week.values() if v)
    biggest = max(range(7), key=lambda d: abs(shares[d][0] - shares[d][1]))
    this, normal = shares[biggest]
    text = ""
    if abs(this - normal) >= 0.05:
        text = f"{_WEEKDAYS[biggest]} carried {this:.0%} of the week against {normal:.0%} usually."
    if active <= 3:
        text = (text + " " if text else "") + f"Only {active} day(s) of the week had activity."
    return shares, text


def _mix(finding: SweepFinding | None) -> dict[str, float]:
    """The week-over-week change split into volume (more or fewer rows at the old value per row), rate (a different value per row) and their interaction."""
    if finding is None or finding.movement.aggregate != "sum":
        return {}
    m = finding.movement
    if not m.before_rows or not m.after_rows or not m.delta:
        return {}
    rate0, rate1 = m.before_value / m.before_rows, m.after_value / m.after_rows
    volume = (m.after_rows - m.before_rows) * rate0
    rate = (rate1 - rate0) * m.before_rows
    interaction = m.delta - volume - rate
    return {"volume": volume / m.delta, "rate": rate / m.delta, "interaction": interaction / m.delta, "rate_before": rate0, "rate_after": rate1}


def _metric_finding(swept: Sweep | None, context: Mapping[str, Any]) -> SweepFinding | None:
    """The week-over-week finding; against an empty week before there is nothing to decompose, so the same week last year is the comparison left."""
    if swept is None:
        return None
    finding = next((f for f in swept.findings if f.movement.comparison.kind == "week"), None)
    if finding is None and context.get("previous_empty"):
        finding = next((f for f in swept.findings if f.movement.comparison.kind == "same_week_prior_year"), None)
    return finding


def _period_words(period: Mapping[str, Any]) -> str:
    return str(period.get("label") or (f"the week of {_week_label(period['start'])[8:]}" if period.get("start") else period))


def _explanations(name: str, finding: SweepFinding | None, mix: Mapping[str, float]) -> list[str]:
    lines: list[str] = []
    if finding is None:
        return lines
    m = finding.movement
    best = finding.best
    if best is not None and best.groups:
        lines.append(f"By {_word(best.path)}: {_concentration_sentence(best)}")
        lead = finding.lead_drill
        if lead is not None:
            lines.append(f"Within {_label(lead.parent.group)}, by {_word(lead.path)}: {_concentration_sentence(lead)}")
        if best.concentration in {"single", "concentrated", "offsetting"} and m.before_value and m.pct is not None:
            leader = best.groups[0]
            held = (m.delta - leader.delta) / abs(m.before_value)
            against = "the week before" if m.comparison.kind == "week" else "the same week last year" if m.comparison.kind == "same_week_prior_year" else _period_words(m.comparison.before)
            lines.append(f"Had {_label(leader.group)} held at {against}, {name} would have moved {held:+.1%} instead of {m.pct:+.1%}.")
    if mix:
        rows_word = "rows"
        lines.append(
            f"Volume explains {mix['volume']:.0%} of the move ({m.before_rows:,} to {m.after_rows:,} {rows_word}) and the value per row {mix['rate']:.0%} ({_num(mix['rate_before'])} to {_num(mix['rate_after'])})"
            + (f"; the rest is their interaction ({mix['interaction']:.0%})." if abs(mix["interaction"]) >= 0.05 else ".")
        )
    return lines


def _comovement(metrics: Sequence[MetricBrief]) -> list[str]:
    """Which metrics move together week over week, and which lead by a week: associations over the shared history, not causes."""
    found: list[tuple[float, str]] = []
    changes: dict[str, dict[str, float]] = {}
    for metric in metrics:
        series: dict[str, float] = {}
        thin = _thin_weeks(metric.weeks)  # a stub week at the tail would drag every series down together and fake a correlation
        level = sorted(abs(w.value) for w in metric.weeks if w.value)
        floor = 0.1 * level[len(level) // 2] if level else 0.0  # a change off a near-empty week is a spike, not a movement
        for previous, current in zip(metric.weeks, metric.weeks[1:]):
            if previous.value and abs(previous.value) >= floor and current.start not in thin and previous.start not in thin:
                series[current.start] = (current.value - previous.value) / abs(previous.value)
        changes[metric.name] = series
    names = [m.name for m in metrics]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(set(changes[a]) & set(changes[b]))
            if len(shared) < 12:
                continue
            same = _spearman([changes[a][s] for s in shared], [changes[b][s] for s in shared])
            a_leads = _spearman([changes[a][s] for s in shared[:-1]], [changes[b][s] for s in shared[1:]])
            b_leads = _spearman([changes[b][s] for s in shared[:-1]], [changes[a][s] for s in shared[1:]])
            if same is not None and abs(same) >= 0.5:
                found.append((abs(same), f"{a.capitalize()} and {b} moved {'together' if same > 0 else 'in opposite directions'} over the last {len(shared)} weeks (r = {same:.2f})."))
            if a_leads is not None and abs(a_leads) >= 0.5 and abs(a_leads) > abs(same or 0) + 0.1:
                found.append((abs(a_leads), f"{a.capitalize()} led {b} by a week (r = {a_leads:.2f} at a one-week lag)."))
            if b_leads is not None and abs(b_leads) >= 0.5 and abs(b_leads) > abs(same or 0) + 0.1:
                found.append((abs(b_leads), f"{b.capitalize()} led {a} by a week (r = {b_leads:.2f} at a one-week lag)."))
    lines = [text for _strength, text in sorted(found, key=lambda item: -item[0])[:6]]
    if lines:
        lines.append("These are associations in the source's own history; they say what moved with what, not what caused what.")
    return lines


def _thin_weeks(weeks: Sequence[Week]) -> set[str]:
    """Weeks holding under half the rows of a typical recent week: the tail of a data set that has not fully arrived."""
    thin: set[str] = set()
    for index in range(len(weeks) - 1, max(-1, len(weeks) - 10), -1):
        recent = sorted(w.rows for w in weeks[max(0, index - 13) : index] if w.rows)
        if len(recent) >= 4 and weeks[index].rows < 0.5 * recent[len(recent) // 2]:
            thin.add(weeks[index].start)
        else:
            break
    return thin


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation: one enormous week cannot make every pair read r = 1.00."""
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return _pearson(_ranks(xs), _ranks(ys))


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


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x_mean, y_mean = _mean(xs), _mean(ys)
    sx = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    sy = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if not sx or not sy:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / (sx * sy)


# --------------------------------------------------------------------------- #
# The brief
# --------------------------------------------------------------------------- #


def brief(
    source: Any,
    metrics: Sequence[Any] = (),
    *,
    kpis: Sequence[str] = (),
    entity: str | None = None,
    window: int = 1,
    week: str | None = None,
    history_weeks: int = 104,
    instructions: str = "",
    scope: str = "",
    paths: int = 4,
    budget: int = 90,
    verify: bool = True,
    timeout: float = 600.0,
    name: str | None = None,
) -> Brief:
    """The Monday Morning Brief over a lakehouse or a semantic model.

    ``metrics`` are plain words (``"revenue by product category"``,
    ``"order quantity"``) or mappings ``{"measure": ..., "fact": ...,
    "by": [...], "name": ...}``. ``kpis`` are built from the structure:
    lifecycle counts (``"new customers"``, ``"churned resellers over 4
    weeks"``, ``"active users"``), ratios (``"average order value = sales
    amount / order quantity"``, ``"revenue per rows"``), crossings
    (``"order quantity where channel = Internet vs order quantity where
    channel = Reseller"``) and concentration (``"top 3 share of revenue by
    reseller"``). ``entity`` overrides the entity the lifecycle counts use;
    ``window`` is how many weeks count as "before" for retained and churned.
    ``week`` is any date of the week to brief (default: the latest complete
    Monday-to-Sunday week the source holds); ``history_weeks`` bounds the
    weekly history used for context; ``paths`` the groupings tried per
    metric when none are named; ``budget`` the queries for the whole brief,
    shared out across the metrics and KPIs.
    """
    from .kpis import parse_kpi

    probe = _probe_for(source, timeout=timeout, name=name)
    started = time.monotonic()
    schema = probe.schema
    context = ReviewContext(scope=scope) if scope else None
    text = "\n".join(p for p in (instructions, context.text if context is not None else "") if p)
    joins = probe.joins(text)
    dialect = probe.dialect(joins)
    kpi_specs = []
    plain_metrics = list(metrics)
    for phrase in kpis:
        parsed = parse_kpi(str(phrase))
        if parsed is None:
            plain_metrics.append(phrase)  # an ordinary metric named among the KPIs
        else:
            kpi_specs.append(parsed)
    specs = _metric_specs(plain_metrics, probe, instructions, scope)
    notes: list[str] = []
    briefs: list[MetricBrief] = []
    spent = 0
    per_metric = max(8, budget // max(1, len(specs) + len(kpi_specs)))
    target_week: Week | None = None
    for spec in specs:
        fact_name, measure = spec["fact"], spec["measure"]
        if not fact_name or not measure or fact_name not in schema.tables:
            notes.append(f"{spec['name']}: {spec.get('note') or 'no measure found for it'}")
            continue
        axis = dialect.axis(fact_name)
        if axis is None:
            notes.append(f"{spec['name']}: {fact_name} has no time axis")
            continue
        aggregate = _aggregate_of(measure, _summed_columns(instructions))
        fact = {"table": fact_name, "date": axis, "measure": measure, "aggregate": aggregate}
        try:
            rows = probe.run(dialect.daily(fact, [measure]))
        except ValueError as exc:
            notes.append(f"{spec['name']}: {exc}")
            continue
        spent += 1
        daily: list[tuple[_dt.date, int, float]] = []
        for row in rows:
            iso = _iso_date(row.get("day"))
            if iso:
                daily.append((_dt.date.fromisoformat(iso), int(row.get("n") or 0), float(row.get("v0") or 0)))
        if not daily:
            notes.append(f"{spec['name']}: no rows with a date")
            continue
        daily.sort()
        max_day = daily[-1][0]
        weeks = _weeks(daily, aggregate, max_day)
        if week:
            wanted = _dt.date.fromisoformat(str(week))
            monday = (wanted - _dt.timedelta(days=wanted.weekday())).isoformat()
            index = next((i for i, w in enumerate(weeks) if w.start == monday), -1)
            if index < 0:
                notes.append(f"{spec['name']}: no complete week starting {monday}; the latest complete week is briefed")
                index = len(weeks) - 1
        elif target_week is not None:
            # every metric briefs the same week: the one the first metric settled on
            index = next((i for i, w in enumerate(weeks) if w.start == target_week.start), -1)
            if index < 0:
                notes.append(f"{spec['name']}: no complete week starting {target_week.start} in its data (it ends on {max_day.isoformat()}); left out")
                continue
        else:
            index = _last_covered_week(weeks)
            if index < len(weeks) - 1:
                typical = sorted(w.rows for w in weeks[max(0, index - 12) : index + 1] if w.rows)
                behind = len(weeks) - 1 - index
                notes.append(
                    f"the last {f'{behind} weeks hold' if behind > 1 else 'week holds'} far fewer rows than the weeks before {'them' if behind > 1 else 'it'} ({weeks[-1].rows:,} against a typical {typical[len(typical) // 2]:,}), "
                    f"which usually means the data has not fully arrived, so the brief covers the {weeks[index].label}; say week={weeks[-1].start} to brief the latest week anyway"
                )
        if index < 0:
            notes.append(f"{spec['name']}: no complete week in the data (it ends on {max_day.isoformat()})")
            continue
        dropped = max(0, index + 1 - history_weeks)  # the history window ends at the briefed week; the weeks after it stay for the charts
        weeks = weeks[dropped:]
        index -= dropped
        target = weeks[index]
        if target_week is None:
            target_week = target
        metric_notes = [spec["note"]] if spec.get("note") else []
        ctx = _context(weeks, index, aggregate)
        points = _change_points(weeks[: index + 1]) if not ctx.get("sparse") else []  # a level shift needs a series that has rows in most weeks
        shares, pattern = _weekday_pattern(daily, target, weeks, index)
        comparisons: list[Comparison] = []
        if index >= 1:
            comparisons.append(Comparison("week", weeks[index - 1].period, target.period))
        if ctx.get("prior_year_start"):
            comparisons.append(Comparison("same_week_prior_year", Week(ctx["prior_year_start"], 0.0, 0, 0).period, target.period))
        swept: Sweep | None = None
        if comparisons:
            swept = sweep(probe, None, instructions=instructions, scope=scope, facts=[fact_name], measures=[measure], paths=list(spec["groupings"]) if spec["groupings"] else paths, comparisons=comparisons, budget=per_metric, min_pct=0.0, depth=2)
            spent += swept.queries
            metric_notes.extend(n for n in swept.notes if "budget" in n)
        finding = _metric_finding(swept, ctx)
        mix = _mix(finding)
        explanations = _explanations(spec["name"], finding, mix)
        headline = _headline(spec["name"], target, ctx, aggregate)
        briefs.append(MetricBrief(spec["name"], fact_name, measure, tuple(spec["groupings"]), aggregate, tuple(weeks), target, ctx, tuple(points), shares, pattern, swept, mix, tuple(explanations), headline, tuple(n for n in metric_notes if n)))
    entities: list[Any] = []
    entity_choice = ""
    if kpi_specs:
        from .kpis import build_kpis

        kpi_briefs, entities, entity_choice, kpi_spent, kpi_notes, target_week = build_kpis(probe, dialect, joins, kpi_specs, specs, entity=entity, window=window, week=week, target_week=target_week, history_weeks=history_weeks, instructions=instructions, scope=scope, budget=max(8, budget - spent), per_kpi=per_metric)
        briefs.extend(kpi_briefs)
        spent += kpi_spent
        notes.extend(kpi_notes)
    comovement = _comovement(briefs)
    watch = _watch(briefs)
    result = Brief(probe.name, probe.kind, target_week, tuple(briefs), tuple(comovement), tuple(watch), spent, budget, tuple(notes), entities=tuple(entities), entity_choice=entity_choice)
    if verify:
        recomputed, mismatches = 0, []
        verified_metrics = []
        for metric in result.metrics:
            if metric.sweep is None:
                verified_metrics.append(metric)
                continue
            checked = verify_sweep(metric.sweep, probe)
            recomputed += checked.recomputed
            mismatches.extend(f"{metric.name}: {text}" for text in checked.mismatches)
            verified_metrics.append(replace(metric, sweep=checked))
        result = replace(result, metrics=tuple(verified_metrics), recomputed=recomputed, mismatches=tuple(mismatches), verified=True)
    return replace(result, elapsed=round(time.monotonic() - started, 1))


def _headline(name: str, target: Week, c: Mapping[str, Any], aggregate: str) -> str:
    parts = [f"{name.capitalize()} came in at {_num(target.value)}{' on average' if aggregate == 'avg' else ''} for the {target.label}"]
    against = []
    if c.get("wow_pct") is not None:
        against.append(f"{_pct(c['wow_pct'])} on the week before")
    elif c.get("previous_empty"):
        against.append("after a week before with no rows")
    if c.get("yoy_pct") is not None:
        against.append(f"{_pct(c['yoy_pct'])} on the same week last year")
    if c.get("vs_avg13_pct") is not None:
        against.append(f"{_pct(c['vs_avg13_pct'])} against the 13-week average")
    text = parts[0] + (": " + ", ".join(against) if against else "") + "."
    if c.get("verdict"):
        text += f" Against {c['expected_source']}, this week is {c['verdict']}." if c.get("expected_source") else f" {c['verdict'].capitalize()}."
    if c.get("coverage_note"):
        text += f" Caution: {c['coverage_note']}."
    return text


def _watch(metrics: Sequence[MetricBrief]) -> list[str]:
    watch: list[str] = []
    for metric in metrics:
        c = metric.context
        z = c.get("z")
        if c.get("sparse"):
            watch.append(f"{metric.name.capitalize()}: {c['sparse']}, so its week-to-week movements are not read as signal.")
            continue
        if c.get("coverage_note"):
            # a week that may not have fully arrived: one line, and the movements it would otherwise raise are read as coverage
            superseded = [f"{_pct(c['wow_pct'])} week over week"] if c.get("wow_pct") is not None and abs(c["wow_pct"]) >= 0.15 else []
            if z is not None and abs(z) >= 2:
                superseded.append(c["verdict"].split(":")[0])
            watch.append(f"{metric.name.capitalize()}: {c['coverage_note']}" + (f" ({', '.join(superseded)} read as coverage)." if superseded else "."))
            continue
        if z is not None and abs(z) >= 2:
            watch.append(f"{metric.name.capitalize()}: {c['verdict']}.")
        recent = [p for p in metric.change_points if metric.target and (_dt.date.fromisoformat(metric.target.start) - _dt.date.fromisoformat(p.start)).days <= 42]
        for point in recent:
            watch.append(f"{metric.name.capitalize()}: a level shift six weeks ago or less (the {_week_label(point.start)}, {_pct(point.pct)}).")
        if c.get("streak", 0) >= 3:
            watch.append(f"{metric.name.capitalize()}: {c['streak_note'][0].lower()}{c['streak_note'][1:]}")
        if c.get("wow_pct") is not None and abs(c["wow_pct"]) >= 0.15 and not (z is not None and abs(z) >= 2):
            watch.append(f"{metric.name.capitalize()}: {_pct(c['wow_pct'])} week over week.")
    return watch


def brief_request(text: str) -> list[str]:
    """The metrics named in a brief request: what follows the colon, ``for`` or ``on``, split on commas and semicolons."""
    body = re.split(r":|\bfor\b|\bon\b|\bof\b", text, maxsplit=1)
    tail = body[1] if len(body) > 1 else ""
    metrics: list[str] = []
    for part in re.split(r",|;", tail):
        # "revenue and orders" is two metrics; "revenue by region and country" is one metric with two groupings
        pieces = [part] if re.search(r"\bby\b", part) else re.split(r"\band\b", part)
        metrics.extend(p.strip() for p in pieces if p.strip())
    return metrics
