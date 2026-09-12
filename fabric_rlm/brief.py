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
from .sweep import Comparison, Sweep, SweepFinding, _aggregate_for, _concentration_sentence, _iso_date, _label, _num, _probe_for, _word, sweep, verify_sweep

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

    @property
    def finding(self) -> SweepFinding | None:
        """The week-over-week movement with its drivers."""
        if self.sweep is None:
            return None
        return next((f for f in self.sweep.findings if f.movement.comparison.kind == "week"), None)

    @property
    def prior_year_finding(self) -> SweepFinding | None:
        if self.sweep is None:
            return None
        return next((f for f in self.sweep.findings if f.movement.comparison.kind == "same_week_prior_year"), None)

    def lines(self) -> list[str]:
        lines = [self.headline]
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

    @property
    def title(self) -> str:
        return f"Monday Morning Brief: {self.source}"

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

    def to_markdown(self) -> str:
        head = [f"# {self.title}", f"Week: {self.week_label}", "", self.summary(), ""]
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
    weeks = []
    for monday in sorted(by_week):
        if monday + _dt.timedelta(days=6) > max_day:
            continue
        value, rows, days = by_week[monday]
        weeks.append(Week(monday.isoformat(), (value / rows if rows else 0.0) if aggregate == "avg" else value, int(rows), int(days)))
    return weeks


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _slope_pct(values: Sequence[float]) -> float | None:
    """The fitted change over the window as a share of the window's mean."""
    n = len(values)
    if n < 4:
        return None
    xs = list(range(n))
    x_mean, y_mean = _mean([float(x) for x in xs]), _mean(values)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if not denominator or not y_mean:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denominator
    return slope * (n - 1) / abs(y_mean)


def _context(weeks: Sequence[Week], index: int, aggregate: str) -> dict[str, Any]:
    """This week against the week before, the same week a year earlier, the recent averages, the seasonal expectation and the record."""
    target = weeks[index]
    values = [w.value for w in weeks]
    c: dict[str, Any] = {"value": target.value, "rows": target.rows}
    previous = weeks[index - 1] if index >= 1 else None
    c["previous"] = previous.value if previous else None
    c["wow_pct"] = (target.value - previous.value) / abs(previous.value) if previous and previous.value else None
    starts = {w.start: i for i, w in enumerate(weeks)}
    year_ago = (_dt.date.fromisoformat(target.start) - _dt.timedelta(days=364)).isoformat()
    prior = weeks[starts[year_ago]] if year_ago in starts else None
    c["prior_year"] = prior.value if prior else None
    c["prior_year_start"] = prior.start if prior else None
    c["yoy_pct"] = (target.value - prior.value) / abs(prior.value) if prior and prior.value else None
    c["avg4"] = _mean(values[max(0, index - 4) : index]) if index >= 4 else None
    c["vs_avg4_pct"] = (target.value - c["avg4"]) / abs(c["avg4"]) if c["avg4"] else None
    c["avg13"] = _mean(values[max(0, index - 13) : index]) if index >= 8 else None
    c["vs_avg13_pct"] = (target.value - c["avg13"]) / abs(c["avg13"]) if c["avg13"] else None
    # the seasonal expectation: the recent level scaled by how this week of the year sat against its own recent level in prior years
    expected, index_values, source = _expectation(weeks, index)
    c["expected"] = expected
    c["expected_source"] = source
    c["vs_expected_pct"] = (target.value - expected) / abs(expected) if expected else None
    residuals = []
    for i in range(max(8, index - 26), index):
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
    if len(residuals) < 8 or c["z"] is None:
        c["verdict"] = "not enough history to say whether this is unusual"
    elif math.isinf(c["z"]):
        c["verdict"] = f"very unusual: {'above' if c['z'] > 0 else 'below'} an expectation the last {len(residuals)} weeks never deviated from"
    elif abs(c["z"]) < 1.5:
        c["verdict"] = "within the usual range"
    elif abs(c["z"]) < 2.5:
        c["verdict"] = f"unusual: {abs(c['z']):.1f} standard deviations {'above' if c['z'] > 0 else 'below'} the expectation"
    else:
        c["verdict"] = f"very unusual: {abs(c['z']):.1f} standard deviations {'above' if c['z'] > 0 else 'below'} the expectation"
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
            if higher and index - higher[-1] >= 8:
                c["rank_note"] = f"The highest week since the {_week_label(weeks[higher[-1]].start)}."
            elif lower and index - lower[-1] >= 8:
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
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}.get(n, f"{n}th")


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
            lines.append(f"Had {_label(leader.group)} held at the week before, {name} would have moved {held:+.1%} instead of {m.pct:+.1%}.")
    if mix:
        rows_word = "rows"
        lines.append(
            f"Volume explains {mix['volume']:.0%} of the move ({m.before_rows:,} to {m.after_rows:,} {rows_word}) and the value per row {mix['rate']:.0%} ({_num(mix['rate_before'])} to {_num(mix['rate_after'])})"
            + (f"; the rest is their interaction ({mix['interaction']:.0%})." if abs(mix["interaction"]) >= 0.05 else ".")
        )
    return lines


def _comovement(metrics: Sequence[MetricBrief]) -> list[str]:
    """Which metrics move together week over week, and which lead by a week: associations over the shared history, not causes."""
    lines: list[str] = []
    changes: dict[str, dict[str, float]] = {}
    for metric in metrics:
        series: dict[str, float] = {}
        for previous, current in zip(metric.weeks, metric.weeks[1:]):
            if previous.value:
                series[current.start] = (current.value - previous.value) / abs(previous.value)
        changes[metric.name] = series
    names = [m.name for m in metrics]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(set(changes[a]) & set(changes[b]))
            if len(shared) < 12:
                continue
            same = _pearson([changes[a][s] for s in shared], [changes[b][s] for s in shared])
            a_leads = _pearson([changes[a][s] for s in shared[:-1]], [changes[b][s] for s in shared[1:]])
            b_leads = _pearson([changes[b][s] for s in shared[:-1]], [changes[a][s] for s in shared[1:]])
            if same is not None and abs(same) >= 0.5:
                lines.append(f"{a.capitalize()} and {b} moved {'together' if same > 0 else 'in opposite directions'} over the last {len(shared)} weeks (r = {same:.2f}).")
            if a_leads is not None and abs(a_leads) >= 0.5 and abs(a_leads) > abs(same or 0) + 0.1:
                lines.append(f"{a.capitalize()} led {b} by a week (r = {a_leads:.2f} at a one-week lag).")
            if b_leads is not None and abs(b_leads) >= 0.5 and abs(b_leads) > abs(same or 0) + 0.1:
                lines.append(f"{b.capitalize()} led {a} by a week (r = {b_leads:.2f} at a one-week lag).")
    if lines:
        lines.append("These are associations in the source's own history; they say what moved with what, not what caused what.")
    return lines


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
    metrics: Sequence[Any],
    *,
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
    "by": [...], "name": ...}``. ``week`` is any date of the week to brief
    (default: the latest complete Monday-to-Sunday week the source holds);
    ``history_weeks`` bounds the weekly history used for context; ``paths``
    the groupings tried per metric when none are named; ``budget`` the
    queries for the whole brief, shared out across the metrics.
    """
    probe = _probe_for(source, timeout=timeout, name=name)
    started = time.monotonic()
    schema = probe.schema
    context = ReviewContext(scope=scope) if scope else None
    text = "\n".join(p for p in (instructions, context.text if context is not None else "") if p)
    joins = probe.joins(text)
    dialect = probe.dialect(joins)
    specs = _metric_specs(metrics, probe, instructions, scope)
    notes: list[str] = []
    briefs: list[MetricBrief] = []
    spent = 0
    per_metric = max(8, budget // max(1, len(specs)))
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
        aggregate = _aggregate_for(measure)
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
        else:
            index = len(weeks) - 1
        if index < 0:
            notes.append(f"{spec['name']}: no complete week in the data (it ends on {max_day.isoformat()})")
            continue
        weeks = weeks[max(0, index + 1 - history_weeks) : index + 1] + weeks[index + 1 :]
        index = min(index, len(weeks) - 1)
        index = next(i for i, w in enumerate(weeks) if w.start == weeks[index].start)
        target = weeks[index]
        if target_week is None:
            target_week = target
        metric_notes = [spec["note"]] if spec.get("note") else []
        ctx = _context(weeks, index, aggregate)
        points = _change_points(weeks[: index + 1])
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
        finding = next((f for f in swept.findings if f.movement.comparison.kind == "week"), None) if swept else None
        mix = _mix(finding)
        explanations = _explanations(spec["name"], finding, mix)
        headline = _headline(spec["name"], target, ctx, aggregate)
        briefs.append(MetricBrief(spec["name"], fact_name, measure, tuple(spec["groupings"]), aggregate, tuple(weeks), target, ctx, tuple(points), shares, pattern, swept, mix, tuple(explanations), headline, tuple(n for n in metric_notes if n)))
    comovement = _comovement(briefs)
    watch = _watch(briefs)
    result = Brief(probe.name, probe.kind, target_week, tuple(briefs), tuple(comovement), tuple(watch), spent, budget, tuple(notes))
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
    if c.get("yoy_pct") is not None:
        against.append(f"{_pct(c['yoy_pct'])} on the same week last year")
    if c.get("vs_avg13_pct") is not None:
        against.append(f"{_pct(c['vs_avg13_pct'])} against the 13-week average")
    text = parts[0] + (": " + ", ".join(against) if against else "") + "."
    if c.get("verdict"):
        text += f" Against {c['expected_source']}, this week is {c['verdict']}." if c.get("expected_source") else f" {c['verdict'].capitalize()}."
    return text


def _watch(metrics: Sequence[MetricBrief]) -> list[str]:
    watch: list[str] = []
    for metric in metrics:
        c = metric.context
        z = c.get("z")
        if z is not None and abs(z) >= 2:
            watch.append(f"{metric.name.capitalize()}: {c['verdict']}.")
        recent = [p for p in metric.change_points if metric.target and (_dt.date.fromisoformat(metric.target.start) - _dt.date.fromisoformat(p.start)).days <= 42]
        for point in recent:
            watch.append(f"{metric.name.capitalize()}: a level shift six weeks ago or less (the {_week_label(point.start)}, {_pct(point.pct)}).")
        if c.get("streak", 0) >= 3:
            watch.append(f"{metric.name.capitalize()}: {c['streak_note']}")
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
