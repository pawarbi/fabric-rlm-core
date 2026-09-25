"""The source as the tools see it.

Schemas and the tables an agent selected, the instructions that name tables and
exclude topics, keys and time columns, measure columns, date joins and period
expressions, the attribute paths a fact reaches through its joins, the words a
source uses for its tables and columns, SQL helpers, and the executor that runs
SQL against a lakehouse. The sweep, the brief, the reports and the KPIs build on
this module; so does the experimental Data Agent review. Nothing here calls a
model or a Data Agent.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# What an agent has: its sources, instructions and few-shots, and a schema of a source
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FewShot:
    id: str
    question: str
    query: str


@dataclass(frozen=True)
class AgentDataSource:
    id: str
    kind: str
    name: str = ""
    instructions: str = ""
    description: str = ""
    item_id: str | None = None
    workspace_id: str | None = None
    fewshots: tuple[FewShot, ...] = ()
    selected_tables: tuple[str, ...] = ()  # schema/table paths the agent has selected, when the elements are readable


@dataclass(frozen=True)
class AgentSnapshot:
    agent_id: str
    name: str
    instructions: str
    datasources: tuple[AgentDataSource, ...]
    description: str = ""
    stage: str = "published"
    workspace_id: str | None = None


_EMPHASIS = re.compile(r"(focus|focus(ed|es|ing)?|priorit(y|ies|ise|ize)|important|especially|mainly|primarily|key)", re.IGNORECASE)
_CONTEXT_STOP = {"focus", "identifying", "identify", "related", "issues", "issue", "under", "review", "agent", "questions", "question", "about", "with", "from", "that", "this", "kpis", "kpi", "their", "these", "those", "which", "what", "into", "only", "also", "such", "more", "most", "less", "than", "over", "between", "through", "during", "include", "including", "well"}


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


@dataclass(frozen=True)
class ReviewContext:
    """What the reviewer states about the agent beyond its configuration.

    ``scope`` says what the agent is for, in plain words; the tables it
    names are in scope for question generation, like tables the agent's
    instructions name. ``priorities`` are topics, measures or attributes to
    evaluate first: questions that mention them come first and survive the
    per-source limit. ``definitions`` are business terms the review
    declares to the RLM, and they override the agent's own where the names
    clash. ``questions`` are the reviewer's own evaluation cases as
    ``(question, expected)`` pairs, where ``expected`` is either a query the
    source can run (the reference is computed by executing it) or a prose
    answer whose figures are the reference; they come before generated
    questions, so supplied ground truth outranks generated ground truth.
    ``notes`` is free text for every RLM task: known quirks, partial
    periods, rules about personal data.
    """

    scope: str = ""
    priorities: tuple[str, ...] = ()
    definitions: Mapping[str, str] = field(default_factory=dict)
    questions: tuple[tuple[str, str], ...] = ()
    notes: str = ""

    @property
    def text(self) -> str:
        """Scope, priorities and notes as one text, for scoping tables and joins."""
        return "\n".join([self.scope, *self.priorities, self.notes])

    @property
    def terms(self) -> tuple[str, ...]:
        """Lowercase singular terms from the priorities and the scope, for ranking attributes and questions."""
        words: list[str] = []
        for text in (*self.priorities, self.scope):
            for word in re.findall(r"[A-Za-z][A-Za-z_]{3,}", text):
                lowered = word.casefold()
                if lowered in _STOPWORDS or lowered in _CONTEXT_STOP:
                    continue
                words.append(_singular(lowered))
        return tuple(dict.fromkeys(words))

    @property
    def ranking_terms(self) -> tuple[str, ...]:
        """The priorities when stated, else the scope's terms."""
        stated = tuple(p.strip().casefold() for p in self.priorities if p.strip())
        return stated or self.terms

    @property
    def emphasised(self) -> tuple[str, ...]:
        """Terms from the scope sentences that say focus, priority or important; they weigh double when ranking."""
        words: list[str] = []
        for sentence in self.scope.replace(chr(10), ". ").split("."):
            if not _EMPHASIS.search(sentence):
                continue
            for word in re.findall(r"[A-Za-z][A-Za-z_]{3,}", sentence):
                lowered = word.casefold()
                if lowered in _STOPWORDS or lowered in _CONTEXT_STOP or _EMPHASIS.fullmatch(lowered):
                    continue
                words.append(_singular(lowered))
        return tuple(dict.fromkeys(words))

    def as_prompt(self) -> str:
        """The context as a block for an RLM task prompt; empty when nothing was stated."""
        parts = []
        if self.scope.strip():
            parts.append(f"Scope of the data agent under review: {self.scope.strip()}")
        if self.priorities:
            parts.append("Priorities: " + "; ".join(p.strip() for p in self.priorities if p.strip()))
        if self.definitions:
            parts.append("Definitions: " + "; ".join(f"{k} = {v}" for k, v in self.definitions.items()))
        if self.notes.strip():
            parts.append(f"Notes: {self.notes.strip()}")
        return "\n".join(parts)

    def __bool__(self) -> bool:
        return bool(self.scope.strip() or self.priorities or self.definitions or self.questions or self.notes.strip())


@dataclass(frozen=True)
class SourceSchema:
    """What a source actually exposes, from a fabric-rlm profile or a catalog."""

    source_id: str
    kind: str
    tables: Mapping[str, tuple[str, ...]]
    measures: tuple[str, ...] = ()
    relationships: tuple[tuple[str, str, str, str], ...] = ()  # from_table, from_col, to_table, to_col
    types: Mapping[str, Mapping[str, str]] = field(default_factory=dict)  # table -> column -> type, when the profile knows

    def column_type(self, table: str, column: str) -> str:
        return str((self.types.get(table) or {}).get(column, "") or "").casefold()

    def table_names(self) -> set[str]:
        return {name.casefold() for name in self.tables}

    def column_names(self) -> set[str]:
        return {column.casefold() for columns in self.tables.values() for column in columns}

    def identifiers(self) -> set[str]:
        names = self.table_names() | self.column_names() | {m.casefold() for m in self.measures}
        for table, columns in self.tables.items():
            for column in columns:
                names.add(f"{table}.{column}".casefold())
                names.add(f"{table}[{column}]".casefold())
        return names


def schema_from_tables(
    source_id: str,
    tables: Mapping[str, Sequence[str]],
    *,
    kind: str = "lakehouse",
    measures: Sequence[str] = (),
    relationships: Sequence[tuple[str, str, str, str]] = (),
    types: Mapping[str, Mapping[str, str]] | None = None,
) -> SourceSchema:
    return SourceSchema(
        source_id=source_id,
        kind=kind,
        tables={str(name): tuple(str(c) for c in columns) for name, columns in tables.items()},
        measures=tuple(str(m) for m in measures),
        types={str(t): {str(c): str(v) for c, v in cols.items()} for t, cols in (types or {}).items()},
        relationships=tuple(tuple(str(x) for x in r) for r in relationships),  # type: ignore[misc]
    )


# --------------------------------------------------------------------------- #
# Reading instructions: the tables they name, the topics they exclude, the facts in scope
# --------------------------------------------------------------------------- #

_IDENTIFIER = re.compile(r"(?<![\w.])(?:dbo\.)?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)(?![\w.])")


_STOPWORDS = {"the", "and", "for", "with", "use", "only", "from", "not", "all", "when", "then", "sum", "count", "distinct", "avg", "min", "max", "as", "by", "in", "on", "or", "of", "to", "a", "an", "is", "are", "table", "tables", "dbo"}


def _lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _tables_named_in(text: str, schema: SourceSchema) -> list[str]:
    """Tables the instructions name, in the order they first appear."""
    lower = {t.casefold(): t for t in schema.tables}
    named: list[str] = []
    for match in _IDENTIFIER.finditer(text or ""):
        for segment in match.group(1).split("."):  # sales.custName names sales; dbo.sales names sales
            table = lower.get(segment.casefold())
            if table and table not in named:
                named.append(table)
    return named


_OUT_OF_SCOPE = re.compile(r"(out[- ]of[- ]scope|not in scope|not configured|do not claim|don't claim|do not answer|cannot answer|not available)", re.IGNORECASE)
_SCOPE_STOP = {"those", "tables", "table", "required", "business", "definitions", "definition", "because", "scope", "claim", "answer", "questions", "question", "about", "data", "source", "sources", "configured", "available", "these", "their", "other", "with", "from", "that", "this", "when", "such", "there", "what", "which", "into", "only", "also", "them", "they", "will", "have", "does", "your", "please", "respond", "request", "requests", "related", "remain", "remains", "level", "ask", "asking", "topics", "topic", "instead", "since", "were", "been", "being", "does", "cannot", "would", "should", "could", "agent", "model", "lakehouse", "warehouse"}


def excluded_terms(instructions: str) -> set[str]:
    """Topics the instructions declare out of scope, as lowercase singular terms.

    Lines such as "Do not claim order status, returns, inventory, promotions
    or quotas because those tables are not in scope" name topics the agent is
    told to decline; tables named after them are not asked about, and a
    declined question on them is graded as policy, not as a failure.
    """
    terms: set[str] = set()
    for line in _lines(instructions):
        if not _OUT_OF_SCOPE.search(line):
            continue
        for word in re.findall(r"[A-Za-z][A-Za-z_]{3,}", line):
            lowered = word.casefold()
            if lowered in _SCOPE_STOP or lowered in _STOPWORDS or _OUT_OF_SCOPE.search(lowered):
                continue
            terms.add(_singular(lowered))
    return terms


def _mentions_excluded(name: str, excluded: Collection[str]) -> bool:
    lowered = name.casefold()
    return any(term and term in lowered for term in excluded)


def _scoped_facts(schema: SourceSchema, instructions: str, *, broad: bool = False) -> list[str]:
    """Fact tables in the agent's declared scope first; all of them when it names none; never an out-of-scope one."""
    excluded = excluded_terms(instructions)
    facts = [t for t in _fact_tables(schema, broad=broad) if not _mentions_excluded(t, excluded)]
    named = _tables_named_in(instructions, schema)
    in_scope = [t for t in named if t in facts]
    return in_scope or facts


# --------------------------------------------------------------------------- #
# Columns: keys, time, measures, dates, joins and period expressions
# --------------------------------------------------------------------------- #

_MEASURE_HINT = re.compile(r"(amount|qty|quantity|units|revenue|sales|cost|price|margin|total|profit|value|hours|minutes|count|arr|mrr|usd|eur|gbp|calls|duration|balance|fee|charge|spend|volume|score|rating)", re.IGNORECASE)


_KEY_FORMS = re.compile(r"(?:^|[_ ])(?:[Ii][Dd]|[Kk][Ee][Yy])$|[a-z0-9](?:Id|ID|Key|KEY)$")
_KEY_STOPWORDS = frozenset({"paid", "unpaid", "prepaid", "valid", "invalid", "grid", "void", "avoid", "rapid", "solid", "liquid", "fluid", "acid", "hybrid", "said", "laid", "mid", "bid", "kid", "lid", "rid", "amid", "turkey", "monkey", "hockey", "jockey", "donkey", "whiskey", "journey", "period", "bandwidth"})
_DATE_KEY = re.compile(r"date(key)?$", re.IGNORECASE)
_YEAR_COLUMN = re.compile(r"^(calendar)?year$", re.IGNORECASE)
_ATTRIBUTE_HINT = re.compile(r"(name|country|region|group|category|segment|type|class|status|city|state|line|plant)", re.IGNORECASE)
_ORDER_ID_HINT = re.compile(r"(ordernumber|orderid|order_id|order_number|ticket_id|invoice|transaction)", re.IGNORECASE)
_LOCAL_EXCLUDED = re.compile(r"(number|nbr|code|sku|email|url|website|uri|hash|uuid|guid)$", re.IGNORECASE)  # never a grouping column on a fact
_FLAG_COLUMN = re.compile(r"^(?:is|has|was|can)_|flag$|^is[A-Z]", re.IGNORECASE)  # is_active, SalesPersonFlag: a yes or no, not a quantity
_JOIN_LINE = re.compile(r"(?:dbo\.)?([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*(?:->|→|to|=)\s*(?:dbo\.)?([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)")


def _joins_from_instructions(text: str, schema: SourceSchema) -> dict[tuple[str, str], tuple[str, str]]:
    """``fact.col -> dim.col`` lines, kept only when both sides exist."""
    joins: dict[tuple[str, str], tuple[str, str]] = {}
    lower = {t.casefold(): t for t in schema.tables}
    for match in _JOIN_LINE.finditer(text or ""):
        a, ac, b, bc = match.groups()
        if a.casefold() in lower and b.casefold() in lower:
            a, b = lower[a.casefold()], lower[b.casefold()]
            if ac.casefold() in {c.casefold() for c in schema.tables[a]} and bc.casefold() in {c.casefold() for c in schema.tables[b]}:
                joins[(a, ac)] = (b, bc)
    return joins


def _heuristic_joins(schema: SourceSchema) -> dict[tuple[str, str], tuple[str, str]]:
    """``<Name>Key``, ``<name>_id`` or ``id_<name>`` in a table to the table that carries the same column.

    The target is ``dim<name>``, ``<name>``, ``<name>s``, ``<name>es`` or
    ``<name>ies`` (company to companies), with the same schema prefix or
    none, or the one table whose name ends with ``_<those>``
    (``sales_customers`` for ``customerID``).
    """
    joins: dict[tuple[str, str], tuple[str, str]] = {}
    tables = {t.casefold(): t for t in schema.tables}
    for table, columns in schema.tables.items():
        prefix = table.rsplit(".", 1)[0] + "." if "." in table else ""
        for column in columns:
            lowered = column.casefold()
            if lowered in {"datekey", "id", "key"} or not _is_key(column):
                continue
            stem = re.sub(r"^id_|[_ ]?(?:id|key)$", "", lowered)
            if not stem:
                continue
            plural_forms = [stem, f"{stem}s", f"{stem}es"] + ([stem[:-1] + "ies"] if stem.endswith("y") else [])
            own = table.rsplit(".", 1)[-1].casefold()
            if own in plural_forms or own in {f"dim{stem}", f"dim_{stem}", f"dim{stem}s"}:
                continue  # the table's own primary key (pedidos.id_pedido), not a reference to another table
            found = None
            for candidate in [f"dim{stem}", f"dim_{stem}", f"dim{stem}s", *plural_forms]:
                for name in ((prefix + candidate, candidate) if prefix else (candidate,)):
                    target = tables.get(name)
                    if target and target != table and column in schema.tables[target]:
                        found = target
                        break
                if found:
                    break
            if not found:
                suffixed = [t for lowered_name, t in tables.items() if t != table and column in schema.tables[t] and any(lowered_name.rsplit(".", 1)[-1].endswith("_" + form) for form in plural_forms)]
                if len(suffixed) == 1:
                    found = suffixed[0]
            if found:
                joins[(table, column)] = (found, column)
    return joins


def _fact_tables(schema: SourceSchema, *, broad: bool = False) -> list[str]:
    facts = []
    joins = _heuristic_joins(schema)
    referenced = _referenced_tables(schema, joins)
    for table, columns in schema.tables.items():
        if re.match(r"^(?:[a-z0-9_]+\.)?dim[_a-z]", table, re.IGNORECASE) or re.search(r"(?:^|[._])(?:date|dates|calendar|time|dim_date)$", table, re.IGNORECASE):
            continue  # a dimension by name (dimproduct carries prices and a start date, and is still not a fact)
        measures = _measure_columns(schema, table, broad=broad)
        if measures and not any(_MEASURE_HINT.search(c) for c in measures) and table in referenced:
            continue  # numeric columns on a table others reference are attributes of a dimension, not measures
        dates = [c for c in columns if _is_time_column(schema, table, c) or (_PERIOD_COLUMN.search(c) and (not schema.types.get(table) or _TEXT_TYPE.search(schema.column_type(table, c))))]
        joined_time = any(
            _is_time_column(schema, target[0], other_column) and not other_column.casefold().endswith("key")
            for column in columns
            for target in [joins.get((table, column))]
            if target is not None
            for other_column in schema.tables[target[0]]
        )
        references_dimension = any(joins.get((table, c)) is not None for c in columns)
        countable = broad and not measures and bool(references_dimension or _local_attributes(schema, table))  # cases, tickets, events: no amount, but rows in time
        if (measures or countable) and (dates or joined_time) and (table.casefold().startswith("fact") or len(measures) >= 2 or references_dimension or _local_attributes(schema, table)):
            facts.append(table)

    def richness(t: str) -> tuple[bool, int, int, str]:
        # a fact named as one first; then the ones with the most grouping columns and measures
        return (not t.casefold().startswith("fact"), -len(attribute_paths(schema, t, joins)), -len(_measure_columns(schema, t, broad=broad)), t)

    return sorted(facts, key=richness)


def _measure_columns(schema: SourceSchema, table: str, *, broad: bool = False) -> list[str]:
    """The measure columns of a table: the ones named like measures, or the numeric ones when no name says so.

    ``broad`` adds every numeric non-key column the profile types allow (Good, Scrap, Down, Planned on a production log):
    what the driver tools sweep, where a column that is never asked about costs nothing.
    """
    preferred = ("salesamount", "revenue", "amount", "sales", "price", "total", "net", "gross", "totalproductcost", "orderquantity", "quantity", "qty", "units", "value")
    secondary = re.compile(r"(freight|tax|discount|shipping|fee|handling|cost)", re.IGNORECASE)
    def usable(c: str) -> bool:
        return (
            not _is_key(c)
            and not _is_time_column(schema, table, c)
            and not _PERIOD_COLUMN.search(c)
            and not _FLAG_COLUMN.search(c)
            and not _LOCAL_EXCLUDED.search(c)
            and not _ORDER_ID_HINT.search(c)
            and "bool" not in schema.column_type(table, c)
            and not _TEXT_TYPE.search(schema.column_type(table, c))
        )

    columns = [c for c in schema.tables[table] if _MEASURE_HINT.search(c) and usable(c)]
    if (not columns or broad) and schema.types.get(table):
        # names say nothing (or every number is wanted); the profile's types do
        typed = [c for c in schema.tables[table] if _NUMERIC_TYPE.search(schema.column_type(table, c)) and usable(c) and not _NOT_A_MEASURE.search(c) and not _NUMBERED.search(c)]
        columns = columns + [c for c in typed if c not in columns]
    return sorted(columns, key=lambda c: (next((i for i, p in enumerate(preferred) if p in c.casefold().replace(" ", "").replace("_", "")), 99) + (50 if secondary.search(c) else 0), c))


_NUMBERED = re.compile(r"(?:^|_| )no$|[a-z]No$|(?:^|_| )NO$|(?:^|_| )num$|[a-z]Num$")  # OrderNo, invoice_no, LineNum: a label, not a quantity
_NOT_A_MEASURE = re.compile(r"^(?:year|yr|month|mo|day|week|wk|quarter|qtr|fiscal[_ ]?year|period)$|(?:latitude|longitude|^lat$|^lng$|^lon$|zip|postal|phone|fax|ssn|sequence|^seq$|ordinal|^rank$|version)", re.IGNORECASE)  # a number that is a label, a place or a position


def _date_candidates(columns: Sequence[str]) -> list[str]:
    """Date columns of a fact in the order to try them: keys before dates, the order date before due or ship dates.

    A profile lists columns alphabetically, so the physical order cannot be
    relied on; ``DueDate`` must not win over ``OrderDateKey``.
    """
    matches = [c for c in columns if _TIME_COLUMN.search(c)]
    lowered = lambda c: c.casefold()  # noqa: E731
    return sorted(
        matches,
        key=lambda c: (
            not lowered(c).endswith("key"),
            not any(word in lowered(c) for word in ("order", "sales", "transaction", "invoice", "activity", "event")),
            any(word in lowered(c) for word in ("due", "ship", "delivery", "modified", "created", "updated")),
        ),
    )


_TIME_COLUMN = re.compile(r"(date[ _]?(key|id)|(date|timestamp|datetime|_at|_time|_on)([ _]?key)?)$", re.IGNORECASE)
_PERIOD_COLUMN = re.compile(r"(?:^|[_ ])(?:quarter|year_?quarter|fiscal_quarter|period|fiscal_period|year_?month|month)$", re.IGNORECASE)  # 2024/Q1, 2024-03: a period written as text
_TIME_TYPE = re.compile(r"(timestamp|datetime|date)", re.IGNORECASE)  # Delta names, SQL names or arrow ``DataType<Timestamp(...)>`` and ``Date32``
_NUMERIC_TYPE = re.compile(r"(int|long|double|float|decimal|numeric|real|number|short|byte)", re.IGNORECASE)
_TEXT_TYPE = re.compile(r"(string|varchar|char|text|utf8)", re.IGNORECASE)
_ZONED_TYPE = re.compile(r"(time ?zone|timestamptz|datetimeoffset)", re.IGNORECASE)  # a timestamp that carries an offset: read in UTC
_FREE_TEXT_HINT = re.compile(r"(comment|message|description|note|text|address|email|phone|url|zip|postal|guid|hash|token)", re.IGNORECASE)


_ID_PREFIX = re.compile(r"^id_", re.IGNORECASE)


def _is_key(column: str) -> bool:
    """``CustomerKey``, ``customerID``, ``customer_id``, ``Customer ID``, ``id_cliente`` or ``customerid``; never ``amount_paid``."""
    name = column.strip()
    lowered = name.casefold()
    if lowered in _KEY_STOPWORDS or lowered.rsplit("_", 1)[-1] in _KEY_STOPWORDS:
        return False
    if _ID_PREFIX.match(name) or _KEY_FORMS.search(name):
        return True
    return bool(re.fullmatch(r"[a-z]{3,}(?:id|key)", lowered)) and name == lowered


def _is_time_column(schema: SourceSchema, table: str, column: str) -> bool:
    """A time axis candidate: named like one, or typed as a date or timestamp."""
    return bool(_TIME_COLUMN.search(column)) or bool(_TIME_TYPE.search(schema.column_type(table, column)))


def _is_attribute(schema: SourceSchema, table: str, column: str) -> bool:
    """A grouping column: named like one, or a text column that is not a key or a time."""
    if _is_key(column) or _is_time_column(schema, table, column) or _PERIOD_COLUMN.search(column):
        return False  # keys and the time axis are never groupings
    if _PERSONAL_HINT.search(column) and not re.search(r"(gender|marital|sexo|genero)", column, re.IGNORECASE):
        return False
    if _ATTRIBUTE_HINT.search(column):
        return True
    return bool(_TEXT_TYPE.search(schema.column_type(table, column))) and not _FREE_TEXT_HINT.search(column)


def _referenced_tables(schema: SourceSchema, joins: Mapping[tuple[str, str], tuple[str, str]]) -> set[str]:
    return {target for (_table, _column), (target, _key) in joins.items()}
_BUSINESS_DATE = re.compile(r"(order|purchase|sale|sales|transaction|invoice|created|event|activity|booking|visit)", re.IGNORECASE)
_SECONDARY_DATE = re.compile(r"(ship|deliver|estimated|approved|limit|updated|modified|due|cancel|return|expir)", re.IGNORECASE)


def _date_join(schema: SourceSchema, table: str, joins: Mapping[tuple[str, str], tuple[str, str]]) -> dict[str, Any] | None:
    """The time axis of a fact, as the SQL renderers need it.

    A date dimension with a year column (``column``, ``date_table``,
    ``date_key``, ``year``, ``month``, ``quarter``), a timestamp on the fact
    itself (``timestamp``), or a timestamp on a joined table such as the
    order header for order lines (``timestamp`` plus ``timestamp_column`` on
    ``date_table``). A dimension outranks a timestamp, a business date (order,
    purchase, sale) outranks a secondary one (shipping, delivery, due).
    """
    columns = schema.tables[table]
    date_tables = {t for t in schema.tables if re.search(r"(date|calendar)", t, re.IGNORECASE)}
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    order = 0
    time_columns = _date_candidates(columns) + [c for c in columns if _is_time_column(schema, table, c) and not _TIME_COLUMN.search(c)]
    for column in time_columns:
        order += 1
        score = (2 if _BUSINESS_DATE.search(column) else 0) - (2 if _SECONDARY_DATE.search(column) else 0)
        column_type = schema.column_type(table, column)
        target = joins.get((table, column))
        if target is None and _DATE_KEY.search(column):
            for candidate in ("dimdate", "date", "dim_date", "calendar"):
                for name in schema.tables:
                    if name.casefold() == candidate and "DateKey" in schema.tables[name]:
                        target = (name, "DateKey")
                        break
                if target:
                    break
        if target is not None:
            date_table, date_key = target
            dim_columns = schema.tables[date_table]
            year = next((c for c in dim_columns if _YEAR_COLUMN.match(c)), None)
            if year:
                candidates.append((score + 4, order, {"column": column, "date_table": date_table, "date_key": date_key, "year": year, "month": next((c for c in dim_columns if _MONTH_COLUMN.match(c)), None), "quarter": next((c for c in dim_columns if _QUARTER_COLUMN.match(c)), None)}))
                continue
        if column.casefold().endswith("key") and not _DATE_KEY.search(column):
            continue
        entry: dict[str, Any] = {"column": column, "timestamp": True}
        if _ZONED_TYPE.search(column_type):
            entry["tz"] = True  # read in UTC, so a day does not move with the session's time zone
        if column.casefold().endswith("key") or _NUMERIC_TYPE.search(column_type):
            entry["stored"] = "yyyymmdd"  # an integer date: 20240131, the way a date key is written
            score -= 1
        elif _TEXT_TYPE.search(column_type):
            entry["stored"] = "text"
        candidates.append((score, order, entry))
    for column in columns:
        target = joins.get((table, column))
        if target is None or _TIME_COLUMN.search(column):
            continue
        other, key = target
        if other in date_tables:
            continue
        for stamp in schema.tables[other]:
            if _is_time_column(schema, other, stamp) and not stamp.casefold().endswith("key"):
                order += 1
                score = -3 + (2 if _BUSINESS_DATE.search(stamp) else 0) - (2 if _SECONDARY_DATE.search(stamp) else 0)  # a date on a joined table only when the fact has none worth using
                entry = {"column": column, "date_table": other, "date_key": key, "timestamp": True, "timestamp_column": stamp}
                stamp_type = schema.column_type(other, stamp)
                if _ZONED_TYPE.search(stamp_type):
                    entry["tz"] = True
                if _NUMERIC_TYPE.search(stamp_type):
                    entry["stored"] = "yyyymmdd"
                elif _TEXT_TYPE.search(stamp_type):
                    entry["stored"] = "text"
                candidates.append((score, order, entry))
    for column in columns:
        # a period written as text (2024/Q1, 2024-03) is a time axis of its own, with no day grain
        if _PERIOD_COLUMN.search(column) and not _is_time_column(schema, table, column) and (not schema.types.get(table) or _TEXT_TYPE.search(schema.column_type(table, column))):
            order += 1
            candidates.append((-1, order, {"column": column, "period": "month" if re.search(r"month|period", column, re.IGNORECASE) else "quarter"}))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return candidates[0][2]


def _time_join(dt: Mapping[str, Any]) -> str | None:
    if dt.get("date_table"):
        return f"JOIN {dt['date_table']} d ON f.{_q(dt['column'])} = d.{_q(dt['date_key'])}"
    return None


def _time_column(dt: Mapping[str, Any]) -> str:
    if dt.get("timestamp") and dt.get("date_table"):
        return f"d.{_q(dt['timestamp_column'])}"
    return f"f.{_q(dt['column'])}"


def _stamp(dt: Mapping[str, Any], dialect: str) -> str:
    """The time axis as a timestamp: a date or timestamp column as it is, an integer 20240131 or a text date converted."""
    column = _time_column(dt)
    stored = dt.get("stored")
    if stored == "yyyymmdd":
        return f"CONVERT(datetime, CAST({column} AS VARCHAR(8)), 112)" if dialect == "tsql" else f"CAST(strptime(CAST({column} AS VARCHAR), '%Y%m%d') AS TIMESTAMP)"
    if stored == "text":
        return f"TRY_CONVERT(datetime, {column})" if dialect == "tsql" else f"COALESCE(TRY_CAST({column} AS TIMESTAMP), TRY_STRPTIME({column}, '%Y%m%d'))"
    if dialect == "tsql":
        return column
    return f"CAST(timezone('UTC', {column}) AS TIMESTAMP)" if dt.get("tz") else f"CAST({column} AS TIMESTAMP)"


def _period_part(dt: Mapping[str, Any], part: str, dialect: str) -> str:
    """Year, quarter or month out of a period written as text (2024/Q1, 2024-Q1, 2024Q1, 2024-03, 202403)."""
    column = f"f.{_q(dt['column'])}"
    if part == "year":
        return f"CAST(SUBSTRING({column}, PATINDEX('%[12][0-9][0-9][0-9]%', {column}), 4) AS INT)" if dialect == "tsql" else f"CAST(regexp_extract({column}, '([12][0-9]{{3}})', 1) AS INTEGER)"
    if part == "quarter":
        return f"CAST(SUBSTRING({column}, PATINDEX('%[Qq][1-4]%', {column}) + 1, 1) AS INT)" if dialect == "tsql" else f"CAST(regexp_extract({column}, '[Qq]([1-4])', 1) AS INTEGER)"
    return f"CAST(RIGHT(REPLACE(REPLACE({column}, '-', ''), '/', ''), 2) AS INT)" if dialect == "tsql" else f"CAST(regexp_extract({column}, '[12][0-9]{{3}}[-/]?([01][0-9])', 1) AS INTEGER)"


def _year_expr(dt: Mapping[str, Any], dialect: str) -> str:
    if dt.get("year"):
        return f"d.{_q(dt['year'])}"
    if dt.get("period"):
        return _period_part(dt, "year", dialect)
    return f"YEAR({_stamp(dt, dialect)})" if dialect == "tsql" else f"year({_stamp(dt, dialect)})"


def _month_expr(dt: Mapping[str, Any], dialect: str) -> str | None:
    if dt.get("month"):
        return f"d.{_q(dt['month'])}"
    if dt.get("period") == "month":
        return _period_part(dt, "month", dialect)
    if dt.get("timestamp"):
        return f"MONTH({_stamp(dt, dialect)})" if dialect == "tsql" else f"month({_stamp(dt, dialect)})"
    return None


def _quarter_expr(dt: Mapping[str, Any], dialect: str) -> str | None:
    if dt.get("quarter"):
        return f"d.{_q(dt['quarter'])}"
    if dt.get("period") == "quarter":
        return _period_part(dt, "quarter", dialect)
    if dt.get("timestamp"):
        return f"DATEPART(QUARTER, {_stamp(dt, dialect)})" if dialect == "tsql" else f"quarter({_stamp(dt, dialect)})"
    return None


def _has_month(dt: Mapping[str, Any]) -> bool:
    return bool(dt.get("month") or dt.get("timestamp") or dt.get("period") == "month")


def _has_quarter(dt: Mapping[str, Any]) -> bool:
    return bool(dt.get("quarter") or dt.get("timestamp") or dt.get("period") == "quarter")


def _max_date_sql(fact: Mapping[str, Any]) -> str:
    dt = fact["date"]
    join = _time_join(dt)
    return f"SELECT MAX({_time_column(dt)}) AS value FROM {fact['table']} f" + (f" {join}" if join else "")


def _local_attributes(schema: SourceSchema, table: str) -> list[str]:
    """Grouping columns on the fact table itself: a flat sales file has its categories on the same rows."""
    return [
        c
        for c in schema.tables[table]
        if _is_attribute(schema, table, c) and not _ORDER_ID_HINT.search(c) and not _LOCAL_EXCLUDED.search(c) and not _PERSONAL_HINT.search(c) and not _MEASURE_HINT.search(c)
    ]


_LANGUAGE_VARIANT = re.compile(r"^(spanish|french|german|italian|portuguese|dutch|japanese|chinese)", re.IGNORECASE)


def _q(name: Any) -> str:
    """A column reference: bare when it is a plain identifier, double-quoted otherwise (DuckDB and T-SQL both accept that)."""
    text = str(name)
    return text if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text) else '"' + text.replace('"', '""') + '"'


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


# --------------------------------------------------------------------------- #
# Vocabulary: the words a source uses for its tables and columns
# --------------------------------------------------------------------------- #

_COMMON_WORDS = frozenset(
    """sales sale reseller resellers internet online web store stores retail wholesale customer customers client clients
    product products item items sku order orders territory territories region regions country countries state states
    city cities geography date dates calendar year years month months quarter quarters week day days time period
    amount quantity qty units unit price cost costs revenue margin profit tax freight discount promotion promotions
    category categories subcategory subcategories employee employees vendor vendors supplier suppliers account accounts
    invoice invoices ticket tickets line lines plant plants asset assets inventory stock shipment shipments return
    returns payment payments budget budgets quota quotas forecast forecasts actual actuals survey response responses
    call center calls campaign campaigns channel channels currency currencies scenario scenarios organization
    organizations department departments finance financial detail details header headers transaction transactions
    summary total totals daily weekly monthly yearly annual group groups class type types status name names key keys
    code codes description descriptions log logs event events session sessions user users visit visits page pages
    click clicks lead leads opportunity opportunities contract contracts subscription subscriptions plan plans
    location locations site sites warehouse warehouses machine machines device devices sensor sensors reading readings
    result results score scores rating ratings review reviews registry alias aliases color colors size sizes""".split()
)
_LANGUAGE_PREFIX = re.compile(r"^(english|spanish|french|german|italian|portuguese|dutch|japanese|chinese)", re.IGNORECASE)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TABLE_PREFIX = re.compile(r"^(fact|dim|tbl|vw|stg|v)_?(?=[a-z])", re.IGNORECASE)
_COLUMN_SUFFIX = re.compile(r"\s+(name|key|id|code)$", re.IGNORECASE)
_QUOTED_SYNONYMS = re.compile(r'((?:"[^"]+"\s*,?\s*(?:or|and)?\s*)+)(?:uses|use|refers? to|means?|maps? to)\s+(?:dbo\.)?([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
_USE_FOR = re.compile(r"\b([A-Z][A-Za-z0-9_]+)\s+for\s+([a-z][a-z ]{2,24}?)(?=,|\.|;| and |$)")
_SUM_DEF = re.compile(r"\b([A-Za-z][A-Za-z ]{2,30}?)\s*=\s*SUM\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", re.IGNORECASE)
_MEANS = re.compile(r"\b([A-Za-z][A-Za-z ]{1,30}?)\s+means\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE)
_ORDER_DEF = re.compile(r"\ban? ([a-z]+) is a distinct ([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_ABBREVIATION = re.compile(r"^[A-Z][A-Z0-9]{1,4}$")
_TABLE_DESC = re.compile(r"(?:dbo\.)?([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Z][A-Z0-9]{1,4})\b")


def _segment(run: str, words: Collection[str]) -> list[str]:
    """Split a lowercase run such as ``resellersales`` into known words, the fewest unknown letters first, then the fewest pieces."""
    n = len(run)
    best: list[tuple[int, int, list[str]] | None] = [None] * (n + 1)
    best[0] = (0, 0, [])
    for i in range(n):
        current = best[i]
        if current is None:
            continue
        unknown, pieces, parts = current
        for j in range(i + 3, n + 1):
            if run[i:j] in words:
                candidate = (unknown, pieces + 1, parts + [run[i:j]])
                if best[j] is None or candidate[:2] < best[j][:2]:
                    best[j] = candidate
        # one unknown letter, merged into the previous unknown piece
        merged = parts[:-1] + [parts[-1] + run[i]] if parts and parts[-1] not in words else parts + [run[i]]
        candidate = (unknown + 1, len(merged), merged)
        if best[i + 1] is None or candidate[:2] < best[i + 1][:2]:
            best[i + 1] = candidate
    final = best[n]
    return final[2] if final is not None else [run]


def humanize_table(name: str, words: Collection[str] = _COMMON_WORDS) -> str:
    """``factresellersales`` to ``reseller sales``, ``dimsalesterritory`` to ``sales territory``."""
    base = name.split(".")[-1]
    if len(base) > 5:
        base = _TABLE_PREFIX.sub("", base)
    tokens = [t for t in re.split(r"[_\s]+", _CAMEL.sub(" ", base)) if t]
    out: list[str] = []
    for token in tokens:
        lowered = token.casefold()
        if len(lowered) > 4 and lowered not in words:
            out.extend(_segment(lowered, words))
        else:
            out.append(lowered)
    return " ".join(out) or name


def humanize_column(name: str) -> str:
    """``EnglishProductName`` to ``product``, ``SalesTerritoryRegion`` to ``sales territory region``."""
    base = name.rsplit("[", 1)[-1].rstrip("]") if "[" in name else name.split(".")[-1]
    base = _LANGUAGE_PREFIX.sub("", base.strip("'"))
    text = " ".join(t for t in re.split(r"[_\s]+", _CAMEL.sub(" ", base)) if t).casefold()
    trimmed = _COLUMN_SUFFIX.sub("", text)
    return trimmed or text or name


def _plural(term: str) -> str:
    if term.endswith("y") and not term.endswith(("ay", "ey", "oy", "uy")):
        return term[:-1] + "ies"
    if term.endswith(("s", "x", "ch", "sh")):
        return term + "es"
    return term + "s"


@dataclass(frozen=True)
class Vocabulary:
    """Business words for schema names, taken from the instructions and the context, so questions read as a user would ask them."""

    tables: Mapping[str, str] = field(default_factory=dict)  # table -> phrase ("factresellersales" -> "reseller sales")
    measures: Mapping[str, str] = field(default_factory=dict)  # column -> term ("SalesAmount" -> "revenue")
    attributes: Mapping[str, str] = field(default_factory=dict)  # column -> term ("SalesTerritoryRegion" -> "territory")
    abbreviations: Mapping[str, str] = field(default_factory=dict)  # table -> abbreviation ("factresellersales" -> "B2B")
    orders: str = "orders"

    def table(self, name: str) -> str:
        return self.tables.get(name) or humanize_table(name)

    def measure(self, column: str) -> str:
        return self.measures.get(column) or humanize_column(column)

    def attribute(self, column: str) -> str:
        return self.attributes.get(column) or humanize_column(column)

    def lines(self) -> list[str]:
        """The vocabulary as brief lines for an RLM prompt."""
        out = [f"{phrase} = table {table}" for table, phrase in self.tables.items()]
        out += [f"{term} = column {column}" for column, term in {**self.measures, **self.attributes}.items()]
        out += [f"{abbreviation} = {self.table(table)}" for table, abbreviation in self.abbreviations.items()]
        return out


def build_vocabulary(snapshot: AgentSnapshot, schema: SourceSchema, context: ReviewContext | None = None) -> Vocabulary:
    """The words the agent's own instructions use for its tables and columns."""
    source = next((s for s in snapshot.datasources if s.id == schema.source_id), None)
    text = "\n".join([source.instructions if source else "", source.description if source else "", snapshot.instructions, context.text if context is not None else ""])
    words = set(_COMMON_WORDS) | {w.casefold() for w in re.findall(r"[A-Za-z]{3,}", text)}
    by_lower = {t.casefold(): t for t in schema.tables}
    columns = {c.casefold(): c for cols in schema.tables.values() for c in cols}
    tables = {t: humanize_table(t, words) for t in schema.tables}
    abbreviations: dict[str, str] = {}
    for match in _QUOTED_SYNONYMS.finditer(text):
        table = by_lower.get(match.group(2).casefold())
        if table is None:
            continue
        phrases = [p.strip() for p in re.findall(r'"([^"]+)"', match.group(1)) if p.strip()]
        if phrases:
            tables[table] = phrases[0].casefold()
        for phrase in phrases:
            token = next((w for w in phrase.split() if _ABBREVIATION.match(w)), None)
            if token and table not in abbreviations:
                abbreviations[table] = token
    for match in _TABLE_DESC.finditer(text):
        table = by_lower.get(match.group(1).casefold())
        if table is not None:
            abbreviations.setdefault(table, match.group(2))
    measures: dict[str, str] = {}
    for column, term in _USE_FOR.findall(text):
        if column.casefold() in columns:
            measures.setdefault(columns[column.casefold()], term.strip())
    for term, column in _SUM_DEF.findall(text):
        if column.casefold() in columns:
            measures.setdefault(columns[column.casefold()], term.strip().casefold())
    attributes: dict[str, str] = {}
    for term, column in _MEANS.findall(text):
        if column.casefold() in columns and term.strip():
            attributes.setdefault(columns[column.casefold()], term.strip().split()[-1].casefold())
    orders = "orders"
    order_definition = _ORDER_DEF.search(text)
    if order_definition:
        orders = _plural(order_definition.group(1).casefold())
    return Vocabulary(tables, measures, attributes, abbreviations, orders)


# --------------------------------------------------------------------------- #
# Attribute paths: how a fact reaches its groupings through the joins
# --------------------------------------------------------------------------- #

_MONTH_COLUMN = re.compile(r"^(month|monthnumber|monthnumberofyear|calendarmonth|month_number|monthofyear|monthno|month_no|month no|monthnum)$", re.IGNORECASE)
_QUARTER_COLUMN = re.compile(r"^(quarter|calendarquarter|quarter_number|quarterofyear)$", re.IGNORECASE)
_ENTITY_HINT = re.compile(r"(reseller|customer|vendor|supplier|account|store|client|dealer|partner|employee|company|organization|franchise|merchant|brand|manufacturer|seller)[_ ]?name$|^name$", re.IGNORECASE)
_PRODUCT_HINT = re.compile(r"(product|item|sku)[_ ]?name$|^(product|item|sku|plan|plan_name|feature)$", re.IGNORECASE)
_CATEGORY_HINT = re.compile(r"(category|subcategory|segment|class|group|tier|sector|industry|module|department|family)", re.IGNORECASE)
_PLACE_HINT = re.compile(r"(territory|region|country|city|state|geography|district|continent|market|area|zone|location|province)", re.IGNORECASE)
_PERSONAL_HINT = re.compile(
    r"(e_?mail|phone|mobile|fax|address|birth|\bdob\b|ssn|social_?security|passport|national_?id|tax_?id|salary|password|secret|token|credit|card_?number|iban|routing|first_?name|last_?name|middle_?name|full_?name|surname|gender|marital"
    r"|telefon|celular|endereco|direccion|adresse|nascimento|nacimiento|geburt|correo|courriel|\bcpf\b|\bcnpj\b|\bnif\b|\bdni\b|\brg\b|sexo|genero)",
    re.IGNORECASE,
)  # English first, then the tokens common in Portuguese, Spanish, French and German column names


def attribute_paths(schema: SourceSchema, table: str, joins: Mapping[tuple[str, str], tuple[str, str]], excluded: Collection[str] = (), *, max_depth: int = 3) -> list[dict[str, Any]]:
    """Every grouping column reachable from a fact through one or more dimension joins.

    A path is ``{"hops": [{"from_column", "table", "key"}, ...], "column", "alias"}``;
    the first hop leaves the fact, later hops follow key columns of the
    dimension just joined (product to subcategory to category). Date tables
    and out-of-scope tables are not entered.
    """
    paths: list[dict[str, Any]] = []
    date_tables = {t for t in schema.tables if re.search(r"(date|calendar|time)", t, re.IGNORECASE)}
    frontier: list[tuple[str, list[dict[str, str]]]] = [(table, [])]
    visited = {table}
    for _depth in range(max_depth):
        next_frontier: list[tuple[str, list[dict[str, str]]]] = []
        for current, hops in frontier:
            for column in schema.tables[current]:
                target = joins.get((current, column))
                if target is None or _DATE_KEY.search(column):
                    continue
                dim_table, dim_key = target
                if dim_table in visited or dim_table in date_tables or _mentions_excluded(dim_table, excluded):
                    continue
                visited.add(dim_table)
                new_hops = hops + [{"from_column": column, "table": dim_table, "key": dim_key}]
                for attribute in schema.tables[dim_table]:
                    if _is_attribute(schema, dim_table, attribute) and attribute != dim_key:
                        paths.append({"hops": new_hops, "column": attribute, "alias": attribute})
                next_frontier.append((dim_table, new_hops))
        frontier = next_frontier
        if not frontier:
            break
    for attribute in _local_attributes(schema, table):
        paths.append({"hops": [], "column": attribute, "alias": attribute})
    return paths


def _path_table(path: Mapping[str, Any], fact: str) -> str:
    """The table a grouping column lives on: the last hop's table, or the fact itself."""
    hops = path.get("hops") or ()
    return str(hops[-1]["table"]) if hops else fact


def _hops(attr: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if "hops" in attr:
        return list(attr["hops"] or ())
    if not attr.get("fact_key"):
        return []  # a column on the fact itself
    return [{"from_column": attr["fact_key"], "table": attr["dim_table"], "key": attr["dim_key"]}]


class _Joins:
    """JOIN clauses for a set of attribute paths, one alias per distinct hop chain."""

    def __init__(self) -> None:
        self.clauses: list[str] = []
        self._alias: dict[tuple[tuple[str, str, str], ...], str] = {}

    def seed(self, chain: tuple[tuple[str, str, str], ...], alias: str) -> None:
        """Name a hop chain the FROM clause already joins (the time join as ``d``), so a path through it reuses that alias."""
        self._alias[chain] = alias

    def ref(self, attr: Mapping[str, Any]) -> str:
        previous = "f"
        chain: tuple[tuple[str, str, str], ...] = ()
        for hop in _hops(attr):
            chain = chain + ((str(hop["from_column"]), str(hop["table"]), str(hop["key"])),)
            if chain not in self._alias:
                alias = f"a{len(self._alias) + 1}"
                self._alias[chain] = alias
                self.clauses.append(f"JOIN {hop['table']} {alias} ON {previous}.{_q(hop['from_column'])} = {alias}.{_q(hop['key'])}")
            previous = self._alias[chain]
        return f"{previous}.{_q(attr['column'])}"


def _period_condition(fact: Mapping[str, Any], period: Mapping[str, Any], dialect: str = "duckdb") -> str:
    dt = fact["date"]
    parts = [f"{_year_expr(dt, dialect)} = {int(period['year'])}"]
    if period.get("month") is not None and _has_month(dt):
        parts.append(f"{_month_expr(dt, dialect)} = {int(period['month'])}")
    if period.get("quarter") is not None and _has_quarter(dt):
        parts.append(f"{_quarter_expr(dt, dialect)} = {int(period['quarter'])}")
    return " AND ".join(parts)


def _paths_by_role(paths: Sequence[Mapping[str, Any]], terms: Collection[str] = ()) -> dict[str, Mapping[str, Any]]:
    """The path to use for each role a question needs: who (entity), what (product), group (category), where (place)."""

    def pick(pattern: re.Pattern[str], *, prefer_depth: int | None = None, exclude: re.Pattern[str] | None = None) -> Mapping[str, Any] | None:
        candidates = [p for p in paths if pattern.search(str(p["column"])) and not _LANGUAGE_VARIANT.match(str(p["column"])) and not (exclude and exclude.search(str(p["column"])))]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda p: (
                -sum(1 for term in terms if term and term in f"{p['hops'][-1]['table'] if p.get('hops') else ''} {p['column']}".casefold()),
                not str(p["column"]).casefold().startswith("english"),
                -len(p["hops"]) if prefer_depth == -1 else len(p["hops"]),
            ),
        )[0]

    roles: dict[str, Mapping[str, Any]] = {}
    for role, pattern, depth in (("entity", _ENTITY_HINT, None), ("product", _PRODUCT_HINT, None), ("category", _CATEGORY_HINT, -1), ("place", _PLACE_HINT, None)):
        chosen = pick(pattern, prefer_depth=depth, exclude=_PLACE_HINT if role == "category" else None)  # a territory group is a place, not a category
        if chosen is not None:
            roles[role] = chosen
    if not roles and paths:
        # no name gives a role away; the nearest grouping column still lets the driver questions ask what led a change
        roles["category"] = sorted(paths, key=lambda p: (len(p["hops"]), str(p["column"]).casefold()))[0]
    return roles


def _iso_date(value: Any) -> str | None:
    """A date as ISO text from a yyyymmdd key, a date, a datetime or ISO text."""
    if value is None:
        return None
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.date().isoformat() if isinstance(value, _dt.datetime) else value.isoformat()
    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    match = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    return match.group(1) if match else None


def _month_name(period: Mapping[str, Any]) -> str:
    month = period.get("month")
    return f"{calendar.month_name[int(month)]} {int(period['year'])}" if month else str(period["year"])


def _channel_measure(channel: str, measure: str) -> str:
    """``internet sales`` with ``sales amount`` reads ``internet sales amount``; with ``revenue`` it reads ``internet sales revenue``."""
    channel_words, measure_words = channel.split(), measure.split()
    if channel_words and measure_words and channel_words[-1] == measure_words[0]:
        return " ".join(channel_words + measure_words[1:])
    return f"{channel} {measure}"


# --------------------------------------------------------------------------- #
# Running SQL against a lakehouse
# --------------------------------------------------------------------------- #

class LakehouseExecutor:
    """Runs lakehouse question SQL through a ``LakehouseSource.query``-shaped callable.

    ``query(sql, sources={alias: catalog_name})`` must return a mapping with
    ``columns`` and ``rows`` (row lists) or a list of row mappings. The
    generated SQL names tables by their catalog names, so ``sources`` maps
    each name to itself.
    """

    def __init__(self, query: Callable[..., Any], tables: Iterable[str], *, timeout: float | None = None, retries: int = 1) -> None:
        self._query = query
        self._tables = list(tables)
        self._timeout = timeout  # passed to the query callable when set (LakehouseSource.query accepts it)
        self._retries = max(0, int(retries))  # a timed-out query is retried once: the file list is cached by then

    def run(self, execution: Mapping[str, Any]) -> list[dict[str, Any]]:
        sql = str(execution["sql"])
        sources: dict[str, str] = {}
        bare_names = {t.split(".")[-1].casefold(): t for t in self._tables if "." in t}
        for table in self._tables:
            alias = re.sub(r"[^A-Za-z0-9_]", "_", table)
            pattern = rf"(?<![\w.]){re.escape(table)}(?![\w])"
            if re.search(pattern, sql, re.IGNORECASE):
                if alias != table:
                    sql = re.sub(pattern, alias, sql, flags=re.IGNORECASE)
                sources[alias] = table
        for bare, table in bare_names.items():
            # the agent's own SQL names the table without its schema; the catalog names it with
            alias = re.sub(r"[^A-Za-z0-9_]", "_", table)
            if alias in sources or bare in {t.casefold() for t in self._tables}:
                continue
            pattern = rf"(?<![\w.]){re.escape(bare)}(?![\w])"
            if re.search(pattern, sql, re.IGNORECASE):
                sql = re.sub(pattern, alias, sql, flags=re.IGNORECASE)
                sources[alias] = table
        kwargs: dict[str, Any] = {"sources": sources}
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        attempt = 0
        while True:
            try:
                return _rows(self._query(sql, **kwargs))
            except TimeoutError:
                attempt += 1
                if attempt > self._retries:
                    raise

    def run_agent_sql(self, sql: str) -> list[dict[str, Any]]:
        """Run a query the agent generated (T-SQL) after the small translation DuckDB needs."""
        return self.run({"sql": _tsql_to_duckdb(sql)})


def _tsql_to_duckdb(sql: str) -> str:
    text = re.sub(r"\bdbo\.", "", sql)
    text = re.sub(r"\[([^\]]+)\]", r'"\1"', text)
    match = re.search(r"\bSELECT\s+TOP\s*\(?\s*(\d+)\s*\)?\s+", text, re.IGNORECASE)
    if match:
        text = text[: match.start()] + "SELECT " + text[match.end():]
        text = text.rstrip().rstrip(";") + f" LIMIT {match.group(1)}"
    return text.strip().rstrip(";")


def _rows(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, Mapping) and "rows" in result:
        columns = list(result.get("columns") or [])
        rows = result["rows"]
        if rows and isinstance(rows[0], Mapping):
            return [dict(r) for r in rows]
        return [dict(zip(columns, r)) for r in rows]
    if hasattr(result, "to_dict"):
        return [dict(r) for r in result.to_dict(orient="records")]
    if isinstance(result, Sequence):
        return [dict(r) for r in result if isinstance(r, Mapping)]
    return []
