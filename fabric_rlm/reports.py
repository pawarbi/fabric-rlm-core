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

from .source_model import _NUMERIC_TYPE, _TEXT_TYPE, AgentDataSource, AgentSnapshot, _is_key, _is_time_column, _measure_columns, _month_name, _path_table, _tables_named_in, attribute_paths, build_vocabulary, excluded_terms, humanize_column
from .sweep import _GROUPING_HINT, Comparison, Movement, Point, Sweep, _aggregate_for, _choose_paths, _filters_for, _points, _probe_for, sweep, verify_sweep

__all__ = ["Report", "ReportSpec", "parse_request", "report"]

_KINDS = {
    "brief": re.compile(r"\b(brief|monday morning|newsletter)\b", re.IGNORECASE),
    "root_cause": re.compile(r"\b(root cause|why|what drove|driver|drivers|drove|explain|cause|caused|reason)\b", re.IGNORECASE),
    "top_movers": re.compile(r"\b(top movers|movers|biggest|largest|winners|losers|risers|fallers|gainers|decliners|most improved)\b", re.IGNORECASE),
    "trend": re.compile(r"\b(trend|trends|over time|by month|monthly|seasonal|seasonality|month by month|time series|evolution)\b", re.IGNORECASE),
    "recap": re.compile(r"\b(recap|review|summary|summari[sz]e|what moved|overview|weekly|monthly business|wbr|mbr|highlights)\b", re.IGNORECASE),
}
_KIND_NAMES = {"brief": "Monday Morning Brief", "root_cause": "root cause analysis", "top_movers": "top movers", "trend": "trend analysis", "recap": "recap of what moved"}
_FOCUS = re.compile(r"\b(which|focus|moving|driving|behind|contribut\w*|responsible)\b", re.IGNORECASE)
# a question about whether a period was in line with the trend or the season: read as a check, not as a request for a trend page
_TREND_WORDS = r"(?:trend|trends|season\w*|pattern|history|usual|normal|expectation|expected)"
_CHECK_WORDS = r"(?:in\s?line|normal|expected|usual|typical|unusual|abnormal|an? (?:anomaly|outlier)|out of line|seasonal|anomal\w*|outliers?)"
# "in line with the trend", "against the usual pattern", "compared to the season": the phrase itself
_CHECK_PHRASE = re.compile(
    r"\b(?:in\s?line with|against|versus|vs\.?|compared? (?:to|with)|consistent with|expected (?:from|by|given)|explained by|match(?:es|ing)?|follow(?:s|ing)?|part of|fit(?:s|ting)?|normal for|typical for)"
    r"\s+(?:(?:the|its|a|an)\s+)?(?:(?:usual|normal|seasonal|historical|recent|long[- ]term|overall)\s+)?" + _TREND_WORDS + r"\b",
    re.IGNORECASE,
)
# "was it in line", "was August 2025 normal", "is that unusual": the question form, detected across the words in between
_CHECK_ASK = re.compile(r"\b(?:is|was|were|are)\b[^.?;]{0,40}?\b" + _CHECK_WORDS + r"\b", re.IGNORECASE)
_CHECK_ANY = re.compile(r"\b(?:anything|something|nothing)\s+(?:unusual|abnormal|odd|strange)\b|\banomal\w*\b|\boutliers?\b", re.IGNORECASE)
# the words that leave the text once a check is detected: the check words and their trend tail, never the measure or the period around them
_CHECK_STRIP = re.compile(r"\b" + _CHECK_WORDS + r"\b(?:\s+with\s+(?:(?:the|its|a|an)\s+)?" + _TREND_WORDS + r"\b)?", re.IGNORECASE)


def _trend_check(text: str) -> tuple[bool, str, set[str]]:
    """Whether the question asks how a period sits against the trend; the text without the words of that question; the words they used."""
    if not (_CHECK_PHRASE.search(text) or _CHECK_ASK.search(text) or _CHECK_ANY.search(text)):
        return False, text, set()
    used = _words_in(_CHECK_PHRASE, text) | _words_in(_CHECK_STRIP, text) | _words_in(_CHECK_ANY, text)
    stripped = _CHECK_ANY.sub(" ", _CHECK_STRIP.sub(" ", _CHECK_PHRASE.sub(" ", text)))
    return True, stripped, used
_YOY_WORDS = re.compile(r"\b(year over year|yoy|y/y|prior year|last year|previous year|a year (ago|earlier))\b", re.IGNORECASE)
_MOM_WORDS = re.compile(r"\b(month over month|mom|m/m|previous month|prior month|last month|latest month|month before)\b", re.IGNORECASE)
_ANNUAL_WORDS = re.compile(r"\b(annual|yearly|full year|by year)\b", re.IGNORECASE)
# words a request is made of that name nothing in a source: not reported as ignored
_FUNCTION_WORDS = frozenset("""
a an the of in on at to for from by with and or but if then than as is are was were be been being it its this that these those there here
what whats which who whom whose why how when where did do does done has have had having will would should could can may might must shall
me my we our us you your they them their i please show tell give get see look find want need like know help report analysis analyse analyze
explain summarize summarise summary recap review overview highlights focus moving moved move moves movement movements changed change changes
changing happened happen happening going went rise rose risen rising fall fell fallen falling drop dropped dropping increase increased increasing
decrease decreased decreasing grow grew grown growth growing decline declined declining up down over under time period periods month months year
years week weeks quarter quarters day days daily weekly monthly yearly annual latest last previous prior recent recently now current currently
today yesterday ago earlier later same versus vs against compared compare comparison trend trends trending line inline normal expected unusual
seasonal season seasonality pattern history usual typical anomaly anomalies outlier outliers top bottom biggest largest smallest most least more
less fewer much many all any some every each per across between within into out about around especially particularly only just also too very
really overall business performance performing doing numbers figures results drivers driver driving drove cause caused causes reason reasons
behind contributed contributing responsible higher lower well badly something anything nothing no not yes so rather still yet again back off
let lets make made take took put set based basis kind sort regarding since during data table fact
""".split())
_STOP_TOKENS = frozenset({"a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "and", "or", "is", "was", "were", "are", "it", "its", "this", "that", "what", "whats", "why", "how", "did", "do", "does", "per", "as", "vs", "with", "from"})  # up and down stay: a production log measures Down
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
    filters: tuple[tuple[Mapping[str, Any], Any], ...] = ()  # (grouping path, value) pairs every query is narrowed to; a value may be a tuple of members
    check: bool = False  # read the named period (or the latest month) against the trend and season of each measure
    lookups: int = 0  # queries spent resolving filter values against the source

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
    checks: tuple[str, ...] = ()  # the named period against the trend and season, one sentence per measure

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
        if self.checks:
            head.append("")
            head.append("Against the trend and season:")
            head.extend(f"- {line}" for line in self.checks)
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
    return "root_cause" if has_period or _FOCUS.search(text) else "recap"


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
    if _YOY_WORDS.search(lowered):
        kinds.append("same_month_prior_year" if (period and period.get("month")) or re.search(r"\bmonth\b", lowered) else "year")
    if _MOM_WORDS.search(lowered):
        kinds.append("month")
    if _ANNUAL_WORDS.search(lowered) and "year" not in kinds:
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
    if scored:
        return [t for _s, _i, t in scored]
    # a table named in full that the default facts left out ("support tickets") is the fact, if it has a time axis
    text_words = " ".join(_tokens(text))
    dialect = None
    for table in schema.tables:
        if table in facts:
            continue
        spelled = " ".join(_tokens(humanize_column(table.rsplit(".", 1)[-1])))
        if not spelled or f" {spelled} " not in f" {text_words} ":
            continue
        dialect = dialect or probe.dialect(probe.joins(instructions))
        if dialect.axis(table) is not None:
            return [table]
    return []


def _match_measures(text: str, schema: Any, table: str, vocabulary: Any) -> tuple[list[str], list[str]]:
    """Measure columns the request names, best first, and the request words that named them."""
    candidates = _measure_columns(schema, table, broad=True)
    # a word that names the table is not a measure word ("sessions by device" is not "sessions revenue"), unless a measure column carries it itself (SalesAmount for "sales")
    own_words = {w for c in candidates for w in _tokens(humanize_column(c))}
    table_words = (set(_tokens(vocabulary.table(table))) | set(_tokens(humanize_column(table.rsplit(".", 1)[-1])))) - own_words
    tokens = [t for t in _tokens(text) if t not in table_words and t not in _STOP_TOKENS]
    lowered_text = " " + " ".join(tokens) + " "
    scored: list[tuple[int, int, str]] = []
    matched_words: list[str] = []
    for column in schema.tables[table]:
        # a column named outright is the measure, whether or not its name says it is one (orders, sessions, headcount)
        spelled = " ".join(_tokens(humanize_column(column)))
        column_type = schema.column_type(table, column)
        if column not in candidates and spelled and f" {spelled} " in lowered_text and not _is_key(column) and not _is_time_column(schema, table, column) and not _TEXT_TYPE.search(column_type) and "bool" not in column_type:
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


_BROAD_GROUPING_CLAUSE = re.compile(
    r"\b(?:by|per|across|for each|for every|which|broken down by|split by|grouped by)\s+(.+?)"
    r"(?=\b(?:in|for|vs\.?|versus|against|compared|over|during|from|since|last|this|per|top|between|to|should|is|are|was|were|did|do|does|has|have|had|will|would|can|could|where|with|and (?:what|why|how|which|was|is|were|are|did))\b|[,.;?]|$)",
    re.IGNORECASE,
)


def _grouping_phrases(text: str, *, broad: bool = False) -> list[str]:
    """The words after ``by``: ``by product and territory`` gives product, territory. ``broad`` also reads per, across, for each and which."""
    phrases: list[str] = []
    for match in (_BROAD_GROUPING_CLAUSE if broad else _GROUPING_CLAUSE).finditer(text):
        chunk = match.group(1)
        if re.fullmatch(r"\s*(month|year|week|quarter|day)s?\s*", chunk, re.IGNORECASE):
            continue  # "by month" is a trend, not a grouping
        for part in re.split(r",|\band\b|&", chunk, flags=re.IGNORECASE):
            words = part.strip().casefold()
            if words and words not in {"month", "year", "week", "quarter"}:
                phrases.append(words)
    return phrases


def _same_word(a: str, b: str) -> bool:
    """The same word, or one an abbreviation of the other with at least four letters in common (cust, customer)."""
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))


def _word_overlap(words: Sequence[str], path: Mapping[str, Any]) -> int:
    """How many words of a phrase the column carries, in its humanized name or in its own name."""
    column = str(path["column"])
    own = set(_tokens(humanize_column(column))) | set(_tokens(column))
    return sum(1 for w in words if any(_same_word(w, o) for o in own))


def _match_groupings(phrases: Sequence[str], schema: Any, table: str, joins: Mapping[tuple[str, str], tuple[str, str]], excluded: Any, terms: Sequence[str]) -> tuple[list[str], list[str]]:
    """Grouping columns for the phrases, in the order asked; the phrases nothing matched."""
    from .source_model import _paths_by_role

    paths = attribute_paths(schema, table, joins, excluded)
    roles = _paths_by_role(paths, terms)
    chosen: list[str] = []
    unmatched: list[str] = []
    for phrase in phrases:
        words = _tokens(phrase)
        match = next((p for p in paths if str(p["column"]).casefold() == phrase.replace(" ", "").casefold() or humanize_column(str(p["column"])).casefold() == phrase), None)
        if match is None:
            scored = sorted(((_word_overlap(words, p), -len(p.get("hops") or ()), str(p["column"])) for p in paths), reverse=True)
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


_HARD_INTRO = frozenset({"where", "for", "with", "only", "just", "especially", "particularly", "regarding"})
_FILTER_CLAUSE = re.compile(
    r"\b(where|for|with|in|on|at|of|about|especially|particularly|only|just|regarding)\s+(.+?)"
    r"(?=\b(?:by|per|across|for each|for every|which|where|with|in|for|on|at|vs\.?|versus|against|compared|over|during|from|since|last|this|top|between|was|is|were|are|did|do|does|has|have|had|will|would|can|could|should)\b|[,.;?]|$)",
    re.IGNORECASE,
)
_CONDITION = re.compile(r"\s*(?:=|==|\bis\b|\bequals\b|:)\s*")
_EDGE_WORDS = frozenset({"the", "a", "an", "and", "or", "of", "to", "that", "those", "these", "this"})


def _words_in(pattern: re.Pattern[str], text: str) -> set[str]:
    return set(_tokens(" ".join(m.group(0) for m in pattern.finditer(text))))


def _trim_clause(clause: str) -> str:
    words = clause.strip().strip("\'\"").split()
    while words and words[0].casefold() in _EDGE_WORDS:
        words.pop(0)
    while words and words[-1].casefold() in _EDGE_WORDS:
        words.pop()
    return " ".join(words)


def _exact_path(paths: Sequence[Mapping[str, Any]], words: str) -> Mapping[str, Any] | None:
    wanted = words.casefold()
    return next((p for p in paths if str(p["column"]).casefold() == wanted.replace(" ", "") or humanize_column(str(p["column"])).casefold() == wanted or str(p["column"]).casefold().replace("_", " ") == wanted), None)


def _path_named(paths: Sequence[Mapping[str, Any]], column: str) -> Mapping[str, Any] | None:
    return next((p for p in paths if str(p["column"]) == column), None)


def _match_filters(residue: str, *, probe: Any, dialect: Any, schema: Any, table: str, joins: Mapping[tuple[str, str], tuple[str, str]], excluded: Any, terms: Sequence[str], vocabulary: Any, consumed: set[str]) -> tuple[list[tuple[Mapping[str, Any], Any]], list[str], int]:
    """The filters a request asks for, resolved against the source: (path, value) pairs, the reading lines, the lookup queries spent.

    A clause after where, for, with, only or especially names a column and a value ("channel = tiktok", "customer state SP",
    "the Night shift") or a value alone ("tiktok", "Ski Resort"); a clause after in, on, at, of or about counts only when it
    resolves. Every value is looked up in the source: an exact match filters on the stored spelling, one partial match on that
    value, up to five partial matches on all of them, and more than five or none leaves the report unfiltered and says so.
    """
    axis = dialect.axis(table)
    if axis is None:
        return [], [], 0, set()
    fact = {"table": table, "date": axis, "measure": "rows", "aggregate": "count"}
    noun = vocabulary.table(table)
    paths = list(attribute_paths(schema, table, joins, excluded))

    def numeric(path: Mapping[str, Any]) -> bool:
        return bool(_NUMERIC_TYPE.search(schema.column_type(_path_table(path, table), str(path["column"]))))

    # where a bare value is looked for: every text grouping the sweep could use, the ones named like a grouping (category, type, region, status) first
    candidates = sorted((p for p in _choose_paths(schema, table, joins, excluded, terms, 40) if not numeric(p)), key=lambda p: not _GROUPING_HINT.search(str(p["column"])))
    fact_words = set(_tokens(noun)) | set(_tokens(humanize_column(table.rsplit(".", 1)[-1])))
    table_words = {w for t in schema.tables for w in _tokens(humanize_column(t.rsplit(".", 1)[-1]))} | {w for t in schema.tables for w in _tokens(vocabulary.table(t))}
    filters: list[tuple[Mapping[str, Any], Any]] = []
    lines: list[str] = []
    spent = 0
    taken: set[str] = set()

    refused: list[str] = []

    def lookup(path: Mapping[str, Any], value: str) -> list[dict[str, Any]]:
        nonlocal spent
        spent += 1
        try:
            return [r for r in probe.run(dialect.lookup(fact, path, value)) if r.get("label") is not None]
        except Exception as exc:  # noqa: BLE001 - a grouping the source cannot search is not a match, but the refusal is said, never hidden
            refused.append(f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
            return []

    for match in _FILTER_CLAUSE.finditer(residue):
        hard = match.group(1).casefold() in _HARD_INTRO
        for raw in re.split(r"\s+and\s+|,", match.group(2), flags=re.IGNORECASE):
            clause = _trim_clause(raw)
            if not clause:
                continue
            parts = _CONDITION.split(clause, maxsplit=1)
            column_words: str | None = None
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                column_words, value = parts[0].strip(), parts[1].strip().strip("\'\"")
            else:
                kept = [w for w in clause.split() if w.casefold() not in fact_words and w.casefold() not in table_words and w.casefold() not in consumed and w.casefold() not in _FUNCTION_WORDS and w.casefold() not in _GENERIC]
                consumed.update(w.casefold() for w in clause.split() if w.casefold() in table_words)
                if not kept:
                    continue  # the clause named the fact, a measure or a period: not a filter
                value = " ".join(kept)
            splits: list[tuple[Mapping[str, Any], str]] = []
            if column_words is not None:
                matched, _unmatched = _match_groupings([column_words.casefold()], schema, table, joins, excluded, terms)
                path = _path_named(paths, matched[0]) if matched else None
                if path is None:
                    lines.append(f"no grouping in the source matched '{column_words}'; '{value}' was not applied")
                    consumed.update(_tokens(clause))
                    continue
                splits.append((path, value))
            else:
                words = value.split()
                named: list[tuple[int, Mapping[str, Any], str]] = []
                loose: list[tuple[int, Mapping[str, Any], str]] = []
                for k in range(1, len(words)):
                    for column_text, rest in ((" ".join(words[:k]), " ".join(words[k:])), (" ".join(words[-k:]), " ".join(words[:-k]))):
                        path = _exact_path(paths, column_text)
                        if path is not None:
                            named.append((k, path, rest))
                            continue
                        matched, _unmatched = _match_groupings([column_text.casefold()], schema, table, joins, excluded, terms)
                        path = _path_named(paths, matched[0]) if matched else None
                        if path is not None:
                            loose.append((k, path, rest))
                splits.extend((path, rest) for _k, path, rest in sorted(named, key=lambda item: -item[0]))
                splits.extend((path, rest) for _k, path, rest in sorted(loose, key=lambda item: item[0]))
                splits.extend((path, value) for path in candidates)
            seen: set[tuple[str, str]] = set()
            best: tuple[Mapping[str, Any], list[dict[str, Any]], str] | None = None
            exact_hits: list[tuple[int, Mapping[str, Any], Any]] = []
            for path, text_value in splits:
                key = (str(path["column"]), text_value.casefold())
                if key in seen or not text_value or str(path["column"]) in taken or len(seen) >= 20:
                    continue
                seen.add(key)
                if numeric(path) and not re.fullmatch(r"-?\d+(\.\d+)?", text_value):
                    continue
                rows = lookup(path, text_value)
                exact = [r for r in rows if str(r.get("label")).casefold() == text_value.casefold()]
                if exact:
                    exact_hits.append((int(exact[0].get("n") or 0), path, exact[0]["label"]))
                    if column_words is not None or len(exact_hits) >= 3:
                        break  # a named column is settled; a bare value stops after a few groupings hold it
                    continue
                if rows and (best is None or len(rows) < len(best[1])):
                    best = (path, rows, text_value)
            if exact_hits:
                count, path, label = max(exact_hits, key=lambda hit: hit[0])
                filters.append((dict(path, fact=table), label))
                taken.add(str(path["column"]))
                lines.append(f"only: {humanize_column(str(path['column']))} = {label} ({count:,} {noun})")
                consumed.update(set(_tokens(clause)) | set(_tokens(match.group(1))))
                continue
            if best is not None:
                path, rows, text_value = best
                word = humanize_column(str(path["column"]))
                labels = [r["label"] for r in rows]
                if len(labels) == 1:
                    filters.append((dict(path, fact=table), labels[0]))
                    taken.add(str(path["column"]))
                    lines.append(f"only: {word} = {labels[0]} (matched '{text_value}'; {int(rows[0].get('n') or 0):,} {noun})")
                elif len(labels) <= 5:
                    filters.append((dict(path, fact=table), tuple(labels)))
                    taken.add(str(path["column"]))
                    lines.append(f"only: {word} is one of {', '.join(str(v) for v in labels)} (matched '{text_value}')")
                else:
                    lines.append(f"'{text_value}' matches {len(labels)} or more different {word} values; say which one; not filtered")
                consumed.update(set(_tokens(clause)) | set(_tokens(match.group(1))))
            elif hard:
                if column_words is not None:
                    lines.append(f"no {humanize_column(str(splits[0][0]['column']))} like '{value}' in the source; not filtered")
                else:
                    lines.append(f"nothing reachable from {noun} is called '{value}'; not filtered")
                consumed.update(set(_tokens(clause)) | set(_tokens(match.group(1))))
    if refused and not filters:
        lines.append(f"the source refused {len(refused)} value lookup{'s' if len(refused) != 1 else ''} ({refused[0]}); not filtered")
    return filters, lines, spent, consumed


def _ignored_words(text: str, consumed: set[str]) -> list[str]:
    """Request words that named nothing: not a kind, a period, a fact, a measure, a grouping, a filter, or a function word."""
    out: list[str] = []
    for token in dict.fromkeys(_tokens(text)):
        if token in consumed or token in _FUNCTION_WORDS or token in _GENERIC or token in _MONTHS or token.isdigit() or len(token) < 2:
            continue
        out.append(token)
    return out


def parse_request(request: str, probe: Any, *, instructions: str = "", scope: str = "") -> ReportSpec:
    """Read a request against the source: the kind of report, the fact, the measures, the groupings, the periods, the filters, and what was not understood."""
    from .source_model import ReviewContext

    text = request.strip()
    schema = probe.schema
    context = ReviewContext(scope=scope) if scope else None
    snapshot = AgentSnapshot(agent_id="report", name=probe.name, instructions=instructions, datasources=(AgentDataSource(id=schema.source_id, kind=schema.kind, name=probe.name),))
    vocabulary = build_vocabulary(snapshot, schema, context)
    joined_text = "\n".join(p for p in (instructions, context.text if context is not None else "") if p)
    text_tokens = set(_tokens(text))
    consumed: set[str] = _words_in(_PERIOD, text) | _words_in(_AGAINST, text) | _words_in(_YOY_WORDS, text) | _words_in(_MOM_WORDS, text) | _words_in(_ANNUAL_WORDS, text)
    period, against = _periods(text)
    check, kind_text, check_words = _trend_check(text)  # "was it in line with the trend" asks for a check, not for the trend page
    consumed |= check_words
    kind = _kind(kind_text, period is not None)
    for pattern in _KINDS.values():
        consumed |= _words_in(pattern, kind_text)
    consumed |= _words_in(_FOCUS, kind_text)
    reading = [f"report: {_KIND_NAMES[kind]}"]
    if kind == "brief":
        from .brief import brief_request

        metrics = brief_request(text)
        reading.append("metrics: " + ", ".join(metrics) if metrics else "metrics: the first fact's main measures (the request named none)")
        return ReportSpec(kind, request=text, reading=tuple(reading), metrics=tuple(metrics))
    facts = _match_facts(kind_text, probe, vocabulary, joined_text)
    default_facts = list(probe.facts(joined_text))
    without_groupings = _BROAD_GROUPING_CLAUSE.sub(" ", kind_text)  # "by product" names a grouping, not the product cost
    derived = False
    if not facts and default_facts and kind in {"root_cause", "trend", "top_movers"}:
        # the fact is the one whose measure the request names: "why did down rise" is the production log, not the first fact
        facts = [next((f for f in default_facts if _match_measures(without_groupings, schema, f, vocabulary)[0]), default_facts[0])]
        derived = True
    if facts:
        reading.append("fact: " + ", ".join(f"{t} ({vocabulary.table(t)})" for t in facts) + (" (the fact with the measure named)" if derived else ""))
    elif default_facts:
        facts = default_facts[:1] if kind in {"root_cause", "trend", "top_movers"} else []
        reading.append(("fact: " + ", ".join(f"{t} ({vocabulary.table(t)})" for t in (facts or default_facts[:2]))) + " (the request named none)")
    for fact_name in facts:
        consumed |= (set(_tokens(vocabulary.table(fact_name))) | set(_tokens(humanize_column(fact_name.rsplit(".", 1)[-1])))) & text_tokens
    table = facts[0] if facts else (default_facts[0] if default_facts else None)
    measures: list[str] = []
    if table is not None:
        measures, measure_words = _match_measures(without_groupings, schema, table, vocabulary)
        if kind != "recap":
            measures = measures[:1] if measures else []
        if measures:
            reading.append("measure: " + ", ".join(f"{m} ({vocabulary.measure(m.strip('[]'))})" for m in measures))
            consumed |= set(_tokens(" ".join(measure_words)))
            for m in measures:
                consumed |= set(_tokens(humanize_column(m.strip("[]")))) & text_tokens
        else:
            reading.append(f"measure: {', '.join(_measure_columns(schema, table, broad=True)[:2]) or 'none found'} (the request named none)")
    groupings: list[str] = []
    unmatched: list[str] = []
    phrases = _grouping_phrases(kind_text, broad=True)
    joins = probe.joins(joined_text) if table is not None else {}
    excluded = excluded_terms(joined_text)
    terms = context.terms if context is not None else ()
    if table is not None and phrases:
        groupings, unmatched = _match_groupings(phrases, schema, table, joins, excluded, terms)
        if groupings:
            reading.append("by: " + ", ".join(humanize_column(g) for g in groupings))
        for phrase in unmatched:
            reading.append(f"no grouping in the source matched '{phrase}'")
        consumed |= _words_in(_BROAD_GROUPING_CLAUSE, kind_text)
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
        consumed |= set(_tokens(top_match.group(0)))
    filters: list[tuple[Mapping[str, Any], Any]] = []
    lookups = 0
    if table is not None:
        residue = _PERIOD.sub(" ", kind_text)
        residue = _BROAD_GROUPING_CLAUSE.sub(" ", residue)
        residue = re.sub(r"\btop\s+\d{1,3}\b", " ", residue, flags=re.IGNORECASE)
        residue = re.sub(r"\s+", " ", residue)
        for _ in range(3):  # an introducer left with nothing after it ("in", once the period is blanked) is dropped, so the next clause keeps its own introducer
            residue = re.sub(r"\b(?:where|for|with|in|on|at|of|about|especially|particularly|only|just|regarding)\s+(?=(?:where|for|with|in|on|at|of|about|especially|particularly|only|just|regarding|by|per|which|and|vs\.?|versus|against)\b|[,.;?]|$)", "", residue, flags=re.IGNORECASE)
        dialect = probe.dialect(joins)
        tried: list[str] = [table] if (facts and not derived) else list(dict.fromkeys([table, *default_facts[:4]]))
        filter_lines: list[str] = []
        first_lines: list[str] = []
        for candidate_table in tried:
            found, lines_here, spent_here, used_here = _match_filters(residue, probe=probe, dialect=dialect, schema=schema, table=candidate_table, joins=joins, excluded=excluded, terms=terms, vocabulary=vocabulary, consumed=set(consumed))
            lookups += spent_here
            first_lines = first_lines or lines_here
            if found or candidate_table == tried[-1]:
                filters, filter_lines = found, (lines_here if found else first_lines)
                consumed |= used_here  # the words of a clause that resolved, or that was explained, are not ignored words
                if found and candidate_table != table:
                    table = candidate_table
                    if kind != "recap":
                        facts = [candidate_table]
                        reading = [line for line in reading if not line.startswith(("fact:", "measure:"))]
                        reading.insert(1, f"fact: {candidate_table} ({vocabulary.table(candidate_table)}) (the fact that holds the value asked for)")
                        measures = _match_measures(without_groupings, schema, candidate_table, vocabulary)[0][:1]
                        if measures:
                            reading.insert(2, "measure: " + ", ".join(f"{m} ({vocabulary.measure(m.strip('[]'))})" for m in measures))
                break
        consumed |= {w for line in filter_lines for w in _tokens(line)}  # a value that failed is said once, as a filter, not again as an ignored word
        consumed |= {w for t in schema.tables for w in _tokens(humanize_column(t.rsplit(".", 1)[-1]))} & text_tokens  # a table named in passing ("boleto payments") is a table, not an ignored word
        reading.extend(filter_lines)
    if check and period is not None and not period.get("month"):
        reading.append(f"check: {_month_name(period)} is a whole year, which is not read against the season; name a month to check it")
    elif check:
        reading.append("check: " + (f"{_month_name(period)} against the trend and season of each measure" if period is not None else "the latest complete month against the trend and season of each measure"))
    ignored = _ignored_words(text, consumed)
    if ignored:
        reading.append(f"ignored: {', '.join(ignored)} (nothing in the source matched)")
    return ReportSpec(kind, tuple(facts), tuple(measures), tuple(groupings), period, against, comparisons, top, text, tuple(reading), tuple(unmatched), filters=tuple(filters), check=check, lookups=lookups)


def _period_checks(result: Sweep, spec: ReportSpec) -> list[str]:
    """One sentence per measure: the named month (or the latest complete month) against the trend and season of its own series."""
    from .series import analyse, period_check

    out: list[str] = []
    for key, points in result.series.items():
        if key in result.collapsed or len(points) < 6:
            continue
        name = result.words.get(key, key)
        story = analyse(points, name=name)
        if not story.months:
            continue
        if spec.period is not None and spec.period.get("month"):
            year, month = int(spec.period["year"]), int(spec.period["month"])
        elif spec.period is not None:
            out.append(f"{name[0].upper() + name[1:]}: a whole year is not read against the season; name a month to check it.")
            continue
        else:
            year, month = story.months[-1].year, story.months[-1].month
        sentence = period_check(story, year, month, name=name)
        out.append(sentence or f"{name[0].upper() + name[1:]}: the monthly series does not cover {calendar.month_name[month]} {year}.")
    return out


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
            metrics = [{"measure": m, "fact": facts[0]} for m in _measure_columns(probe.schema, facts[0], broad=True)[:2]] if facts else []
        return build_brief(probe, metrics, instructions=instructions, scope=scope, budget=budget, verify=verify, timeout=timeout, name=name)  # type: ignore[return-value]
    explicit = _explicit_comparisons(spec.period, spec.against)
    comparisons: Sequence[Any] | None = explicit or (list(spec.comparisons) if spec.comparisons else None)
    facts: int | Sequence[str] = list(spec.facts) if spec.facts else (4 if spec.kind == "recap" else 2)  # a recap covers the source; four facts fit a budget of 60 to 80
    measures: int | Sequence[str] = list(spec.measures) if spec.measures else 2
    paths: int | Sequence[str] = list(spec.groupings) if spec.groupings else 8
    common = dict(instructions=instructions, scope=scope, facts=facts, measures=measures, paths=paths, comparisons=comparisons, budget=budget, top=spec.top, filters=list(spec.filters))
    if spec.kind == "root_cause":
        result = sweep(probe, years, min_pct=0.0, depth=2, **common)
    elif spec.kind == "top_movers":
        result = sweep(probe, years, min_pct=0.0, depth=1, **common)
    elif spec.kind == "trend":
        result = sweep(probe, years, min_pct=float("inf"), depth=1, **common)
    else:
        result = sweep(probe, years, min_pct=0.05, depth=2, **common)
    grouped: dict[str, dict[Any, tuple[Point, ...]]] = {}
    queries = result.queries + spec.lookups
    if spec.kind == "trend" and result.years:
        grouped, extra, notes = _trend_by_group(probe, result, spec, instructions=instructions, scope=scope, budget=max(0, budget - result.queries))
        queries += extra
        if notes:
            result = replace(result, notes=result.notes + tuple(notes))
    if verify:
        result = verify_sweep(result, probe)
    if spec.period is not None:
        totals = [m for m in result.ledger if m.path is None]
        if totals and all(m.after_rows == 0 for m in totals):
            months = sorted({(p.year, p.month) for points in result.series.values() for p in points})
            span = f"; the data runs from {calendar.month_name[months[0][1]]} {months[0][0]} to {calendar.month_name[months[-1][1]]} {months[-1][0]}" if months else ""
            result = replace(result, notes=(f"the source has no rows in {_month_name(spec.period)}{span}, so there is nothing to compare", *result.notes))
    result = replace(result, elapsed=round(time.monotonic() - started, 1))
    if not spec.facts or not spec.measures:
        spec = replace(spec, reading=_reading_after(spec, result))  # the request named no fact or measure: say what was actually swept, not what was guessed
    checks = _period_checks(result, spec) if spec.check else []
    return Report(spec, result, grouped, queries, result.elapsed, checks=tuple(checks))


def _reading_after(spec: ReportSpec, result: Sweep) -> tuple[str, ...]:
    """The reading lines with the guessed fact and measure lines replaced by the facts and measures the sweep measured."""
    swept: dict[str, list[str]] = {}
    for m in result.ledger:
        if m.path is None and m.measure not in swept.setdefault(m.fact, []):
            swept[m.fact].append(m.measure)
    if not swept:
        return spec.reading
    kept = [line for line in spec.reading if not line.startswith(("fact:", "measure:")) or "the fact that holds" in line]  # the fact chosen for the value asked for stays explained
    said = "; ".join(f"{fact}: {', '.join(result.words.get(f'{fact}|{m}', m) for m in measures)}" for fact, measures in swept.items())
    kept.insert(1, f"swept: {said}" + (" (the request named none)" if not spec.facts and not spec.measures else ""))
    return tuple(kept)


def _trend_by_group(probe: Any, result: Sweep, spec: ReportSpec, *, instructions: str, scope: str, budget: int) -> tuple[dict[str, dict[Any, tuple[Point, ...]]], int, list[str]]:
    """The first measure of the first fact by month for the largest groups of each grouping: one query to pick the groups, one for the series."""
    from .source_model import ReviewContext

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
    narrowing = _filters_for(schema, fact_name, joins, excluded_terms(text), list(spec.filters)) or []
    paths = _choose_paths(schema, fact_name, joins, excluded_terms(text), context.terms if context is not None else (), list(spec.groupings) if spec.groupings else 3)
    years = list(result.years)[-2:] if len(result.years) >= 2 else list(result.years)
    for path in paths[:4]:
        if spent + 2 > budget:
            notes.append(f"the budget left no room for the series by {humanize_column(str(path['column']))}")
            break
        top_rows = probe.run(dialect.top_groups(fact, path, years, 6, narrowing))
        spent += 1
        values = [row.get("label") for row in top_rows]
        if not values:
            continue
        rows = dialect.months(probe.run(dialect.grouped_series(fact, path, years, values, narrowing)))
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
