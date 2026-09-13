"""Reports from a request in plain words, over a lakehouse or a semantic model.

``report(source, "root cause of the drop in reseller sales in December 2013
by product and territory")`` reads the request against the source's own
vocabulary (its tables, measures, grouping columns and the instructions'
words for them), turns it into a :class:`ReportSpec`, runs the sweep the
spec needs, and renders one of four report kinds:

- ``trend``: every measure by month, and by the largest groups of each
  grouping asked for;
- ``root_cause``: one movement (a measure between two periods) decomposed
  by every grouping, classified, and drilled into its leading group;
- ``recap``: what moved across every fact and measure, the recap a weekly
  or monthly business review opens with;
- ``top_movers``: the groups with the largest changes, up and down, for
  every grouping.

The reading of the request is printed on the page, so a wrong guess is
visible and the spec can be passed back corrected. No model is involved:
the parser is a vocabulary match, the numbers are the source's own.
"""

from __future__ import annotations

import calendar
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .data_agent_review import AgentDataSource, AgentSnapshot, _is_key, _is_time_column, _measure_columns, _month_name, _tables_named_in, attribute_paths, build_vocabulary, excluded_terms, humanize_column
from .sweep import Comparison, Movement, Point, Sweep, _aggregate_for, _choose_paths, _points, _probe_for, sweep, verify_sweep

__all__ = ["Report", "ReportSpec", "parse_request", "report"]

_KINDS = {
    "brief": re.compile(r"\b(brief|monday morning|newsletter)\b", re.IGNORECASE),
    "root_cause": re.compile(r"\b(root cause|why|what drove|driver|drivers|drove|explain|cause|caused|reason)\b", re.IGNORECASE),
    "top_movers": re.compile(r"\b(top movers|movers|biggest|largest|winners|losers|risers|fallers|gainers|decliners|most improved)\b", re.IGNORECASE),
    "trend": re.compile(r"\b(trend|trends|over time|by month|monthly|seasonal|seasonality|month by month|time series|evolution)\b", re.IGNORECASE),
    "recap": re.compile(r"\b(recap|review|summary|summari[sz]e|what moved|overview|weekly|monthly business|wbr|mbr|highlights)\b", re.IGNORECASE),
}
_KIND_NAMES = {"brief": "Monday Morning Brief", "root_cause": "root cause analysis", "top_movers": "top movers", "trend": "trend analysis", "recap": "recap of what moved"}
_MONTHS = {name.casefold(): index for index, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.casefold(): index for index, name in enumerate(calendar.month_abbr) if name})
_PERIOD = re.compile(r"\b(?:(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?\s+)?((?:19|20)\d{2})\b", re.IGNORECASE)
_AGAINST = re.compile(r"\b(vs\.?|versus|against|compared (?:to|with)|over|from)\b", re.IGNORECASE)
_GENERIC = frozenset({"sales", "fact", "data", "table", "the", "of", "in", "by", "and", "for", "a", "an", "to", "on", "with", "amount", "total", "value"})
_SYNONYMS = {
    "revenue": ("amount", "revenue", "sales"),
    "sales": ("salesamount", "sales", "amount", "revenue"),
    "turnover": ("amount", "revenue"),
    "quantity": ("quantity", "qty", "units", "orderquantity"),
    "units": ("units", "quantity", "qty"),
    "volume": ("quantity", "units", "volume"),
    "orders": ("orderquantity", "orders"),
    "cost": ("cost",),
    "costs": ("cost",),
    "price": ("price",),
    "profit": ("profit", "margin"),
    "margin": ("margin", "profit"),
    "discount": ("discount",),
    "freight": ("freight",),
    "tax": ("tax",),
    "score": ("score", "rating"),
    "satisfaction": ("satisfaction", "score", "rating"),
    "payments": ("amount", "payment"),
    "spend": ("spend", "amount"),
}
_ROLE_WORDS = {
    "product": ("product",),
    "products": ("product",),
    "category": ("category",),
    "categories": ("category",),
    "territory": ("place",),
    "territories": ("place",),
    "region": ("place",),
    "regions": ("place",),
    "country": ("place",),
    "countries": ("place",),
    "geography": ("place",),
    "market": ("place",),
    "customer": ("entity",),
    "customers": ("entity",),
    "reseller": ("entity",),
    "resellers": ("entity",),
    "account": ("entity",),
    "accounts": ("entity",),
    "segment": ("category",),
    "segments": ("category",),
    "sector": ("category",),
    "sectors": ("category",),
}


@dataclass(frozen=True)
class ReportSpec:
    """What to report: the kind, the facts and measures, the groupings, the periods, and how the request was read."""

    kind: str = "recap"  # trend | root_cause | recap | top_movers
    facts: tuple[str, ...] = ()  # schema table names; empty means the probe's default facts
    measures: tuple[str, ...] = ()  # measure columns, or "[Model Measure]"; empty means the default measures
    groupings: tuple[str, ...] = ()  # grouping column names; empty means the default paths
    period: Mapping[str, Any] | None = None  # the period the request names ({"year": 2013, "month": 12}); the "after" side
    against: Mapping[str, Any] | None = None  # the period to compare with, when named
    comparisons: tuple[str, ...] = ()  # comparison kinds when no period is named: month, same_month_prior_year, year
    top: int = 5
    request: str = ""
    reading: tuple[str, ...] = ()  # how the request was read, printed on the page
    unmatched: tuple[str, ...] = ()  # parts of the request nothing in the source matched
    metrics: tuple[str, ...] = ()  # the metrics a brief tracks, in plain words

    @property
    def title(self) -> str:
        return _KIND_NAMES.get(self.kind, self.kind)


@dataclass(frozen=True)
class Report:
    """A report of one kind over one source: the sweep behind it and, for a trend, the series by group."""

    spec: ReportSpec
    sweep: Sweep
    grouped_series: Mapping[str, Mapping[Any, tuple[Point, ...]]] = field(default_factory=dict)  # "fact|measure|column" -> label -> monthly points
    queries: int = 0
    elapsed: float = 0.0

    @property
    def source(self) -> str:
        return self.sweep.source

    @property
    def notes(self) -> tuple[str, ...]:
        return self.sweep.notes

    @property
    def title(self) -> str:
        return f"{self.spec.title.capitalize()}: {self.sweep.source}"

    def summary(self) -> str:
        return self.sweep.summary()

    def lines(self) -> list[str]:
        return self.sweep.lines()

    def to_markdown(self) -> str:
        head = [f"# {self.spec.title.capitalize()}: {self.sweep.source}", ""]
        head.extend(f"- {line}" for line in self.spec.reading)
        head.append("")
        return "\n".join(head) + self.sweep.to_markdown()

    def to_html(self) -> str:
        from .sweep_dashboard import render_report

        return render_report(self)

    def save(self, path: str) -> str:
        from .sweep_dashboard import document_report

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(document_report(self))
        return path


# --------------------------------------------------------------------------- #
# Reading the request
# --------------------------------------------------------------------------- #


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.casefold())]


def _kind(text: str, has_period: bool) -> str:
    for kind in ("brief", "root_cause", "top_movers", "trend", "recap"):
        if _KINDS[kind].search(text):
            return kind
    return "root_cause" if has_period else "recap"


def _periods(text: str) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """The period named (after) and the one it is compared with (before), from month names and years in the request."""
    found: list[tuple[int, Mapping[str, Any]]] = []
    for match in _PERIOD.finditer(text):
        month, year = match.group(1), int(match.group(2))
        period: dict[str, Any] = {"year": year}
        if month:
            period["month"] = _MONTHS[month.casefold().rstrip(".")]
        found.append((match.start(), period))
    if not found:
        return None, None
    if len(found) == 1:
        return found[0][1], None
    first, second = found[0], found[1]
    between = text[first[0] : second[0]]
    if _AGAINST.search(between):
        return first[1], second[1]  # "December 2013 vs December 2012": the first is the period of interest
    later = max((first, second), key=lambda f: (int(f[1]["year"]), int(f[1].get("month") or 0)))
    earlier = first if later is second else second
    return later[1], earlier[1]


def _comparison_kinds(text: str, period: Mapping[str, Any] | None) -> tuple[str, ...]:
    lowered = text.casefold()
    kinds: list[str] = []
    if re.search(r"\b(year over year|yoy|y/y|prior year|last year|previous year|a year (ago|earlier))\b", lowered):
        kinds.append("same_month_prior_year" if (period and period.get("month")) or re.search(r"\bmonth\b", lowered) else "year")
    if re.search(r"\b(month over month|mom|m/m|previous month|prior month|last month|latest month|month before)\b", lowered):
        kinds.append("month")
    if re.search(r"\b(annual|yearly|full year|by year)\b", lowered) and "year" not in kinds:
        kinds.append("year")
    return tuple(dict.fromkeys(kinds))


def _explicit_comparisons(period: Mapping[str, Any] | None, against: Mapping[str, Any] | None) -> list[Comparison]:
    if period is None:
        return []
    if against is not None:
        return [Comparison("custom", dict(against), dict(period))]
    year = int(period["year"])
    month = period.get("month")
    if month is None:
        return [Comparison("year", {"year": year - 1}, {"year": year})]
    month = int(month)
    previous = {"year": year - 1, "month": 12} if month == 1 else {"year": year, "month": month - 1}
    return [Comparison("month", previous, dict(period)), Comparison("same_month_prior_year", {"year": year - 1, "month": month}, dict(period))]


def _match_facts(text: str, probe: Any, vocabulary: Any, instructions: str) -> list[str]:
    schema = probe.schema
    named = [t for t in _tables_named_in(text, schema)]
    facts = list(probe.facts(instructions))
    chosen = [t for t in named if t in facts]
    if chosen:
        return chosen
    tokens = set(_tokens(text))
    scored: list[tuple[int, int, str]] = []
    for index, table in enumerate(facts):
        words = {w for w in _tokens(vocabulary.table(table)) if w not in _GENERIC} | {w for w in _tokens(humanize_column(table.rsplit(".", 1)[-1])) if w not in _GENERIC}
        overlap = len(words & tokens)
        if overlap:
            scored.append((-overlap, index, table))
    scored.sort()
    return [t for _s, _i, t in scored] if scored else []


def _match_measures(text: str, schema: Any, table: str, vocabulary: Any) -> tuple[list[str], list[str]]:
    """Measure columns the request names, best first, and the request words that named them."""
    tokens = _tokens(text)
    lowered_text = " " + " ".join(tokens) + " "
    candidates = _measure_columns(schema, table)
    scored: list[tuple[int, int, str]] = []
    matched_words: list[str] = []
    for column in schema.tables[table]:
        # a column named outright is the measure, whether or not its name says it is one (orders, sessions, headcount)
        spelled = " ".join(_tokens(humanize_column(column)))
        if column not in candidates and spelled and f" {spelled} " in lowered_text and not _is_key(column) and not _is_time_column(schema, table, column):
            scored.append((-100, -1, column))
            matched_words.append(spelled)
    for index, column in enumerate(candidates):
        column_words = set(_tokens(humanize_column(column))) | {column.casefold().replace("_", "")}
        term = vocabulary.measure(column).casefold() if hasattr(vocabulary, "measure") else ""
        score = 0
        for token in tokens:
            if token in column_words or (term and token == term) or (term and token in _tokens(term) and token not in _GENERIC):
                score += 3
                matched_words.append(token)
            for hint in _SYNONYMS.get(token, ()):
                if hint in column.casefold().replace("_", "").replace(" ", ""):
                    score += 1
                    matched_words.append(token)
        if score:
            scored.append((-score, index, column))
    scored.sort()
    for measure in schema.measures:
        if measure.casefold() in text.casefold():
            scored.insert(0, (-99, -1, f"[{measure}]"))
            matched_words.append(measure)
    return [c for _s, _i, c in scored], matched_words


_GROUPING_CLAUSE = re.compile(r"\bby\s+(.+?)(?=\b(?:in|for|vs\.?|versus|against|compared|over|during|from|since|last|this|per|top|between)\b|[,.;?]|$)", re.IGNORECASE)


def _grouping_phrases(text: str) -> list[str]:
    """The words after ``by``: ``by product and territory`` gives product, territory."""
    phrases: list[str] = []
    for match in _GROUPING_CLAUSE.finditer(text):
        chunk = match.group(1)
        if re.fullmatch(r"\s*(month|year|week|quarter|day)s?\s*", chunk, re.IGNORECASE):
            continue  # "by month" is a trend, not a grouping
        for part in re.split(r",|\band\b|&", chunk, flags=re.IGNORECASE):
            words = part.strip().casefold()
            if words and words not in {"month", "year", "week", "quarter"}:
                phrases.append(words)
    return phrases


def _match_groupings(phrases: Sequence[str], schema: Any, table: str, joins: Mapping[tuple[str, str], tuple[str, str]], excluded: Any, terms: Sequence[str]) -> tuple[list[str], list[str]]:
    """Grouping columns for the phrases, in the order asked; the phrases nothing matched."""
    from .data_agent_review import _paths_by_role

    paths = attribute_paths(schema, table, joins, excluded)
    roles = _paths_by_role(paths, terms)
    chosen: list[str] = []
    unmatched: list[str] = []
    for phrase in phrases:
        words = _tokens(phrase)
        match = next((p for p in paths if str(p["column"]).casefold() == phrase.replace(" ", "").casefold() or humanize_column(str(p["column"])).casefold() == phrase), None)
        if match is None:
            scored = sorted(((len(set(_tokens(humanize_column(str(p["column"])))) & set(words)), -len(p.get("hops") or ()), str(p["column"])) for p in paths), reverse=True)
            if scored and scored[0][0] > 0:
                match = next(p for p in paths if str(p["column"]) == scored[0][2])
        if match is None:
            for word in words:
                for role in _ROLE_WORDS.get(word, ()):
                    if role in roles:
                        match = roles[role]
                        break
                if match is not None:
                    break
        if match is None:
            unmatched.append(phrase)
        elif str(match["column"]) not in chosen:
            chosen.append(str(match["column"]))
    return chosen, unmatched


def parse_request(request: str, probe: Any, *, instructions: str = "", scope: str = "") -> ReportSpec:
    """Read a request against the source: the kind of report, the fact, the measures, the groupings, the periods."""
    from .data_agent_review import ReviewContext

    text = request.strip()
    schema = probe.schema
    context = ReviewContext(scope=scope) if scope else None
    snapshot = AgentSnapshot(agent_id="report", name=probe.name, instructions=instructions, datasources=(AgentDataSource(id=schema.source_id, kind=schema.kind, name=probe.name),))
    vocabulary = build_vocabulary(snapshot, schema, context)
    joined_text = "\n".join(p for p in (instructions, context.text if context is not None else "") if p)
    period, against = _periods(text)
    kind = _kind(text, period is not None)
    reading = [f"report: {_KIND_NAMES[kind]}"]
    if kind == "brief":
        from .brief import brief_request

        metrics = brief_request(text)
        reading.append("metrics: " + ", ".join(metrics) if metrics else "metrics: the first fact's main measures (the request named none)")
        return ReportSpec(kind, request=text, reading=tuple(reading), metrics=tuple(metrics))
    facts = _match_facts(text, probe, vocabulary, joined_text)
    default_facts = list(probe.facts(joined_text))
    if facts:
        reading.append("fact: " + ", ".join(f"{t} ({vocabulary.table(t)})" for t in facts))
    elif default_facts:
        facts = default_facts[:1] if kind in {"root_cause", "trend", "top_movers"} else []
        reading.append(("fact: " + ", ".join(f"{t} ({vocabulary.table(t)})" for t in (facts or default_facts[:2]))) + " (the request named none)")
    table = facts[0] if facts else (default_facts[0] if default_facts else None)
    measures: list[str] = []
    without_groupings = _GROUPING_CLAUSE.sub(" ", text)  # "by product" names a grouping, not the product cost
    if table is not None:
        measures, _words = _match_measures(without_groupings, schema, table, vocabulary)
        if kind != "recap":
            measures = measures[:1] if measures else []
        if measures:
            reading.append("measure: " + ", ".join(f"{m} ({vocabulary.measure(m.strip('[]'))})" for m in measures))
        else:
            reading.append(f"measure: {', '.join(_measure_columns(schema, table)[:2]) or 'none found'} (the request named none)")
    groupings: list[str] = []
    unmatched: list[str] = []
    phrases = _grouping_phrases(text)
    if table is not None and phrases:
        joins = probe.joins(joined_text)
        groupings, unmatched = _match_groupings(phrases, schema, table, joins, excluded_terms(joined_text), context.terms if context is not None else ())
        if groupings:
            reading.append("by: " + ", ".join(humanize_column(g) for g in groupings))
        for phrase in unmatched:
            reading.append(f"no grouping in the source matched '{phrase}'")
    comparisons = _comparison_kinds(text, period)
    if period is not None:
        explicit = _explicit_comparisons(period, against)
        reading.append("periods: " + " and ".join(c.label for c in explicit))
    elif comparisons:
        reading.append("comparison: " + ", ".join({"month": "latest month against the month before", "same_month_prior_year": "latest month against the same month a year earlier", "year": "latest complete year against the year before"}[c] for c in comparisons))
    top_match = re.search(r"\btop\s+(\d{1,3})\b", text, re.IGNORECASE)
    top = int(top_match.group(1)) if top_match else 5
    if top_match:
        reading.append(f"top {top}")
    return ReportSpec(kind, tuple(facts), tuple(measures), tuple(groupings), period, against, comparisons, top, text, tuple(reading), tuple(unmatched))


# --------------------------------------------------------------------------- #
# Building the report
# --------------------------------------------------------------------------- #


def report(
    source: Any,
    request: str | ReportSpec,
    *,
    years: Sequence[int] | None = None,
    instructions: str = "",
    scope: str = "",
    budget: int = 60,
    verify: bool = True,
    timeout: float = 600.0,
    name: str | None = None,
) -> Report:
    """Build the report a request asks for over a lakehouse or a semantic model.

    ``request`` is plain words (``"trend of revenue by product category"``,
    ``"why did reseller sales fall in December 2013 by territory"``, ``"top
    movers by customer last month"``, ``"weekly recap"``) or a
    :class:`ReportSpec`. ``years``, ``instructions``, ``scope``, ``budget``
    and ``verify`` mean what they mean for :func:`fabric_rlm.sweep.what_moved`.
    """
    probe = _probe_for(source, timeout=timeout, name=name)
    started = time.monotonic()
    spec = request if isinstance(request, ReportSpec) else parse_request(request, probe, instructions=instructions, scope=scope)
    if spec.kind == "brief":
        from .brief import brief as build_brief

        metrics: list[Any] = list(spec.metrics)
        if not metrics:
            facts = list(probe.facts(instructions))
            metrics = [{"measure": m, "fact": facts[0]} for m in _measure_columns(probe.schema, facts[0])[:2]] if facts else []
        return build_brief(probe, metrics, instructions=instructions, scope=scope, budget=budget, verify=verify, timeout=timeout, name=name)  # type: ignore[return-value]
    explicit = _explicit_comparisons(spec.period, spec.against)
    comparisons: Sequence[Any] | None = explicit or (list(spec.comparisons) if spec.comparisons else None)
    facts: int | Sequence[str] = list(spec.facts) if spec.facts else 2
    measures: int | Sequence[str] = list(spec.measures) if spec.measures else 2
    paths: int | Sequence[str] = list(spec.groupings) if spec.groupings else 8
    common = dict(instructions=instructions, scope=scope, facts=facts, measures=measures, paths=paths, comparisons=comparisons, budget=budget, top=spec.top)
    if spec.kind == "root_cause":
        result = sweep(probe, years, min_pct=0.0, depth=2, **common)
    elif spec.kind == "top_movers":
        result = sweep(probe, years, min_pct=0.0, depth=1, **common)
    elif spec.kind == "trend":
        result = sweep(probe, years, min_pct=float("inf"), depth=1, **common)
    else:
        result = sweep(probe, years, min_pct=0.05, depth=2, **common)
    grouped: dict[str, dict[Any, tuple[Point, ...]]] = {}
    queries = result.queries
    if spec.kind == "trend" and result.years:
        grouped, extra, notes = _trend_by_group(probe, result, spec, instructions=instructions, scope=scope, budget=max(0, budget - result.queries))
        queries += extra
        if notes:
            result = replace(result, notes=result.notes + tuple(notes))
    if verify:
        result = verify_sweep(result, probe)
    result = replace(result, elapsed=round(time.monotonic() - started, 1))
    return Report(spec, result, grouped, queries, result.elapsed)


def _trend_by_group(probe: Any, result: Sweep, spec: ReportSpec, *, instructions: str, scope: str, budget: int) -> tuple[dict[str, dict[Any, tuple[Point, ...]]], int, list[str]]:
    """The first measure of the first fact by month for the largest groups of each grouping: one query to pick the groups, one for the series."""
    from .data_agent_review import ReviewContext

    schema = probe.schema
    context = ReviewContext(scope=scope) if scope else None
    text = "\n".join(p for p in (instructions, context.text if context is not None else "") if p)
    joins = probe.joins(text)
    dialect = probe.dialect(joins)
    grouped: dict[str, dict[Any, tuple[Point, ...]]] = {}
    notes: list[str] = []
    spent = 0
    totals = [m for m in result.ledger if m.path is None]
    if not totals:
        return grouped, spent, notes
    fact_name, measure = totals[0].fact, totals[0].measure
    axis = dialect.axis(fact_name)
    if axis is None:
        return grouped, spent, notes
    fact = {"table": fact_name, "date": axis, "measure": measure, "aggregate": _aggregate_for(measure)}
    paths = _choose_paths(schema, fact_name, joins, excluded_terms(text), context.terms if context is not None else (), list(spec.groupings) if spec.groupings else 3)
    years = list(result.years)[-2:] if len(result.years) >= 2 else list(result.years)
    for path in paths[:4]:
        if spent + 2 > budget:
            notes.append(f"the budget left no room for the series by {humanize_column(str(path['column']))}")
            break
        top_rows = probe.run(dialect.top_groups(fact, path, years, 6))
        spent += 1
        values = [row.get("label") for row in top_rows]
        if not values:
            continue
        rows = dialect.months(probe.run(dialect.grouped_series(fact, path, years, values)))
        spent += 1
        by_label: dict[Any, list[Mapping[str, Any]]] = {}
        for row in rows:
            by_label.setdefault(row.get("label"), []).append(row)
        grouped[f"{fact_name}|{measure}|{path['column']}"] = {label: _points(entries, 0, fact["aggregate"], years, "v0") for label, entries in by_label.items()}
    return grouped, spent, notes


def top_movers(result: Sweep, *, top: int = 5) -> list[tuple[str, str, list[Movement], list[Movement]]]:
    """For every (measure, comparison, grouping) in the ledger: the groups that rose most and fell most, by absolute change."""
    buckets: dict[tuple[str, str, str, str], list[Movement]] = {}
    for m in result.ledger:
        if m.path is None or m.parent:
            continue
        buckets.setdefault((m.fact, m.measure, m.comparison.label, str(m.path["column"])), []).append(m)
    order: dict[str, int] = {}
    for m in result.ledger:
        if m.path is None:
            order.setdefault(f"{m.fact}|{m.measure}", len(order))  # the measures in the order they were chosen, revenue before quantity
    out = []
    for (fact, measure, label, column), movements in sorted(buckets.items(), key=lambda item: (order.get(f"{item[0][0]}|{item[0][1]}", 99), item[0][2], item[0][3])):
        risers = sorted((m for m in movements if m.delta > 0), key=lambda m: -m.delta)[:top]
        fallers = sorted((m for m in movements if m.delta < 0), key=lambda m: m.delta)[:top]
        out.append((f"{result.words.get(f'{fact}|{measure}', measure)} {label}", humanize_column(column), risers, fallers))
    return out


def month_label(period: Mapping[str, Any]) -> str:
    return _month_name(period)
