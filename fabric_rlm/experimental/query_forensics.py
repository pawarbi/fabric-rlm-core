"""What the agent's query did differently from the reference query, and the line that closes the gap.

A Data Agent that answers with a wrong number has already told you why: the query it
ran is in its run steps. Compared against the reference query, which the review
executed to get the right number, the difference is not an opinion. The agent summed
a different column, left out a filter the reference applied, read a table the
reference never touches, or filtered the period on another date column. Each of
those differences is a fact about two pieces of SQL, and each one names the
instruction a person would otherwise have to discover by trial.

That is the whole idea here: the reference query is the teacher, the agent's query is
the student, and the diff between them is the lesson. Nothing in this module knows
anything about a particular schema; every instruction it proposes is built from the
identifiers the two queries actually used.

Parsing needs ``sqlglot``. Without it :func:`shape_of` returns ``None`` and
:func:`diverge` returns nothing, so a review that cannot parse simply reports no
forensics rather than guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "QueryShape",
    "Divergence",
    "shape_of",
    "diverge",
    "instruction_lines",
]

_AGGREGATES = {"SUM", "AVG", "MIN", "MAX", "COUNT", "COUNTIF", "STDDEV", "VARIANCE"}
_TIME_HINT = re.compile(r"(date|dt|day|time|stamp|period|month|year|created|ordered|shipped|closed)", re.IGNORECASE)
_PERIOD_LABEL = re.compile(r"^(?:fy|yr|year|qtr|quarter|mo|month|period|week|wk)\d*$", re.IGNORECASE)


def _sqlglot() -> Any:
    try:
        import sqlglot
    except ImportError:  # pragma: no cover - optional dependency
        return None
    return sqlglot


@dataclass(frozen=True)
class QueryShape:
    """What one query reads, measures, filters, groups and joins.

    Names are lowercased and unqualified: two queries written by different authors
    alias the same table differently, and the comparison is about which column of
    which table was used, not about the letter in front of it.
    """

    tables: frozenset[str] = frozenset()
    columns: frozenset[str] = frozenset()
    measures: frozenset[tuple[str, str]] = frozenset()  # (function, column)
    filters: frozenset[tuple[str, str, str]] = frozenset()  # (column, operator, literal)
    groupings: frozenset[str] = frozenset()
    distinct_counts: frozenset[str] = frozenset()
    aggregates_before_join: bool = False
    steering: frozenset[str] = frozenset()  # columns that choose a weight inside an aggregate: CASE cur WHEN 'EUR' THEN 1.1
    conversions: tuple[tuple[str, str], ...] = ()  # (column, the branches it chooses between)
    sql: str = ""

    @property
    def measure_columns(self) -> frozenset[str]:
        return frozenset(column for _function, column in self.measures)

    @property
    def filter_columns(self) -> frozenset[str]:
        return frozenset(column for column, _op, _value in self.filters)


@dataclass(frozen=True)
class Divergence:
    """One difference between the agent's query and the reference, with the line that closes it.

    ``kind`` is stable and machine-readable; ``instruction`` is the sentence proposed
    for the data source's instructions; ``evidence`` quotes what each query did, so a
    reviewer can check the claim without reading both queries in full.
    """

    kind: str
    detail: str
    instruction: str
    evidence: str = ""
    columns: tuple[str, ...] = ()


def _name(node: Any) -> str:
    text = getattr(node, "name", "") or ""
    return str(text).strip().casefold()


def _column_name(node: Any) -> str:
    """A column as ``table.column`` when the query said so, else ``column``.

    Aliases are resolved to the table they stand for, so ``d.amt2`` and
    ``sls_dtl.amt2`` compare equal.
    """

    return _name(node)


def shape_of(sql: str, *, dialect: str = "tsql") -> QueryShape | None:
    """Read one query into a :class:`QueryShape`, or ``None`` when it cannot be parsed."""

    text = (sql or "").strip()
    if not text:
        return None
    glot = _sqlglot()
    if glot is None:
        return None
    try:
        tree = glot.parse_one(text, dialect=dialect)
    except Exception:  # noqa: BLE001 - an unparsable query yields no forensics, never an error
        try:
            tree = glot.parse_one(text)
        except Exception:  # noqa: BLE001
            return None
    if tree is None:
        return None
    exp = glot.exp

    alias_to_table: dict[str, str] = {}
    tables: set[str] = set()
    for table in tree.find_all(exp.Table):
        name = _name(table)
        if not name:
            continue
        tables.add(name)
        alias = (table.alias or "").strip().casefold()
        if alias:
            alias_to_table[alias] = name
        alias_to_table.setdefault(name, name)

    def qualified(column: Any) -> str:
        bare = _name(column)
        table = (column.table or "").strip().casefold()
        if table:
            resolved = alias_to_table.get(table, table)
            return f"{resolved}.{bare}"
        return bare

    columns = {qualified(c) for c in tree.find_all(exp.Column) if _name(c)}
    # Names the query invents (a subquery's SUM(amt2) AS order_value) are not columns of
    # any table, so they must not be compared against one.
    # An Alias keeps its new name in ``alias``; ``name`` is the expression it renames.
    invented = {str(getattr(a, "alias", "") or _name(a)).strip().casefold() for a in tree.find_all(exp.Alias)} - {""}
    comparisons = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Is, exp.In, exp.Like)

    def measured(function: Any) -> list[str]:
        """The columns an aggregate measures, without the ones that only steer it.

        ``SUM(amt * CASE cur WHEN 'EUR' THEN 1.1 ELSE 1 END)`` measures ``amt``; ``cur``
        chooses the rate. Counting the second as a measure would report every currency
        conversion as the wrong column.
        """

        steering: set[int] = set()
        for case in function.find_all(exp.Case):
            subject = case.this
            if subject is not None:
                steering.update(id(c) for c in subject.find_all(exp.Column))
            for branch in case.args.get("ifs") or []:
                condition = getattr(branch, "this", None)
                if condition is not None:
                    steering.update(id(c) for c in condition.find_all(exp.Column))
        for node in function.find_all(*comparisons):
            steering.update(id(c) for c in node.find_all(exp.Column))
        return [
            qualified(c) for c in function.find_all(exp.Column)
            if _name(c) and id(c) not in steering and _name(c) not in invented
        ]

    def literal(node: Any) -> str:
        return node.sql(dialect=dialect).strip("'\" ") if node is not None else "?"

    steering_columns: set[str] = set()
    conversions: dict[str, str] = {}
    for function in tree.find_all(exp.AggFunc):
        for case in function.find_all(exp.Case):
            names = [qualified(c) for c in case.this.find_all(exp.Column) if _name(c)] if case.this is not None else []
            branches = []
            for branch in case.args.get("ifs") or []:
                condition = getattr(branch, "this", None)
                if condition is None:
                    continue
                if case.this is None:  # a searched CASE names its column in each condition
                    names += [qualified(c) for c in condition.find_all(exp.Column) if _name(c)]
                    shown = condition.sql(dialect=dialect)
                else:
                    shown = literal(condition)
                branches.append(f"{shown} -> {literal(branch.args.get('true'))}")
            if case.args.get("default") is not None:
                branches.append(f"else {literal(case.args.get('default'))}")
            for name in names:
                steering_columns.add(name)
                conversions.setdefault(name, ", ".join(branches))

    measures: set[tuple[str, str]] = set()
    distinct_counts: set[str] = set()
    for function in tree.find_all(exp.AggFunc):
        label = type(function).__name__.upper()
        if label not in _AGGREGATES:
            label = (function.sql_name() if hasattr(function, "sql_name") else label).upper()
        inner = measured(function)
        measures.update((label, column) for column in inner)
        if label == "COUNT" and function.find(exp.Distinct) is not None:
            distinct_counts.update(inner)

    filters: set[tuple[str, str, str]] = set()
    for where in tree.find_all(exp.Where):
        for predicate in where.find_all(exp.Binary):
            if isinstance(predicate, (exp.And, exp.Or)):
                continue
            left, right = predicate.left, predicate.right
            column = left if isinstance(left, exp.Column) else (right if isinstance(right, exp.Column) else None)
            literal = right if isinstance(left, exp.Column) else left
            if column is None or isinstance(literal, exp.Column):
                continue
            operator = {
                exp.EQ: "=", exp.NEQ: "<>", exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<=",
            }.get(type(predicate), type(predicate).__name__.upper())
            value = literal.sql(dialect=dialect) if hasattr(literal, "sql") else str(literal)
            filters.add((qualified(column), str(operator), value.strip("'\" ")))
        for isin in where.find_all(exp.In):
            column = isin.this
            if isinstance(column, exp.Column):
                members = ", ".join(sorted(e.sql(dialect=dialect).strip("'\" ") for e in isin.expressions))
                filters.add((qualified(column), "IN", members))

    groupings: set[str] = set()
    for group in tree.find_all(exp.Group):
        groupings.update(qualified(c) for c in group.find_all(exp.Column) if _name(c))

    # A reference that aggregates inside a subquery before joining is guarding against
    # a fan-out; a flat query over the same tables is not.
    nested = [s for s in tree.find_all(exp.Select) if s is not tree]
    aggregates_before_join = any(any(sub.find_all(exp.AggFunc)) for sub in nested)

    return QueryShape(
        tables=frozenset(tables),
        columns=frozenset(columns),
        measures=frozenset(measures),
        filters=frozenset(filters),
        groupings=frozenset(groupings),
        distinct_counts=frozenset(distinct_counts),
        aggregates_before_join=aggregates_before_join,
        steering=frozenset(steering_columns),
        conversions=tuple(sorted(conversions.items())),
        sql=text,
    )


def _bare(column: str) -> str:
    return column.rsplit(".", 1)[-1]


def _similar(a: str, b: str) -> bool:
    """Whether two table names look like versions of one another (sls_dtl and sls_dtl_v2)."""

    x, y = re.sub(r"[^a-z0-9]", "", a.casefold()), re.sub(r"[^a-z0-9]", "", b.casefold())
    if not x or not y or x == y:
        return x == y
    short, long = sorted((x, y), key=len)
    return long.startswith(short) and len(long) - len(short) <= 4


def diverge(
    agent: QueryShape | None,
    reference: QueryShape | None,
    *,
    measure_word: str = "this measure",
    table_rows: Mapping[str, int] | None = None,
) -> tuple[Divergence, ...]:
    """Every difference that changed the answer, as facts about the two queries.

    ``measure_word`` is what the question called the measure, so the proposed line
    reads in the user's words rather than in column names alone. ``table_rows``, when
    the profile has it, lets a stale-table finding say which table is the smaller one.
    """

    if agent is None or reference is None:
        return ()
    out: list[Divergence] = []

    # A table the agent read that the reference never touches, while the reference
    # reads one with a near-identical name: the classic leftover copy.
    extra_tables = agent.tables - reference.tables
    for table in sorted(extra_tables):
        twin = next((t for t in sorted(reference.tables) if _similar(table, t)), "")
        if not twin:
            continue
        sizes = ""
        if table_rows and table in table_rows and twin in table_rows:
            if table_rows[table] > table_rows[twin]:
                # the agent read the fuller table; the reference may be the stale one, and
                # naming the smaller table authoritative would write the mistake into the
                # configuration. Say nothing rather than the wrong thing.
                continue
            sizes = f" ({table_rows[table]:,} rows against {table_rows[twin]:,})"
        out.append(Divergence(
            kind="stale_table",
            detail=f"the agent read {table}, the reference reads {twin}{sizes}",
            instruction=f"Use {twin} for {measure_word}. {table} is not the table to read.",
            evidence=f"agent FROM {table}; reference FROM {twin}",
            columns=(table, twin),
        ))

    # A different column inside the same aggregate.
    def real_measure(column: str) -> bool:
        return not _PERIOD_LABEL.match(column) and not _TIME_HINT.search(column)

    ref_measures = {(f, _bare(c)) for f, c in reference.measures if real_measure(_bare(c))}
    agent_measures = {(f, _bare(c)) for f, c in agent.measures if real_measure(_bare(c))}
    for function, column in sorted(ref_measures - agent_measures):
        wrong = sorted({c for f, c in agent_measures if f == function and c != column})
        if not wrong:
            continue
        out.append(Divergence(
            kind="wrong_measure_column",
            detail=f"the agent aggregated {', '.join(wrong)} where the reference aggregates {column}",
            instruction=f"{measure_word.capitalize()} is {function}({column}); {' and '.join(wrong)} is a different column.",
            evidence=f"agent {function}({wrong[0]}); reference {function}({column})",
            columns=tuple([column, *wrong]),
        ))

    ref_columns = {c for _f, c in ref_measures}
    agent_columns_measured = {c for _f, c in agent_measures}
    if ref_columns and ref_columns <= agent_columns_measured:
        for function, column in sorted(agent_measures - ref_measures):
            if column in ref_columns:
                continue
            right = sorted(ref_columns)
            out.append(Divergence(
                kind="extra_measure_column",
                detail=f"the agent added {column} into {measure_word}; the reference measures {', '.join(right)} alone",
                instruction=f"{measure_word.capitalize()} is {', '.join(right)} alone. Do not add {column} into it.",
                evidence=f"agent {function}(... {column} ...); reference {function}({right[0]})",
                columns=(column, *right),
            ))

    # A column that chooses a weight inside the reference aggregate, and that the agent never reads.
    agent_columns = {_bare(c) for c in agent.columns}
    conversions = dict(reference.conversions)
    for column in sorted(reference.steering):
        if _bare(column) in agent_columns:
            continue
        branches = conversions.get(column, "")
        weighted = f"weight each row {branches}" if branches else "apply its value to each row"
        out.append(Divergence(
            kind="missing_conversion",
            detail=f"the reference weights {measure_word} by {column} ({branches}); the agent never reads {_bare(column)}",
            instruction=f"{measure_word.capitalize()} depends on {column}: {weighted} before adding it up.",
            evidence=f"reference CASE on {column}",
            columns=(column,),
        ))

    # A filter the reference applies and the agent never applies on that column at all.
    agent_filter_columns = {_bare(c) for c in agent.filter_columns}
    for column, operator, value in sorted(reference.filters):
        bare = _bare(column)
        if bare in agent_filter_columns:
            continue
        if _TIME_HINT.search(bare) and operator in {">=", ">", "<", "<="}:
            continue  # a period bound belongs to the question, not to the configuration
        readable = f"{column} {operator} {value}" if operator != "IN" else f"{column} in ({value})"
        table = column.rsplit(".", 1)[0] if "." in column else ""
        out.append(Divergence(
            kind="missing_filter",
            detail=f"the reference filters {readable}; the agent's query does not filter {bare} at all",
            instruction=f"Always filter {readable} when reading {table or 'this table'}; rows outside it are not part of {measure_word}.",
            evidence=f"reference WHERE {readable}",
            columns=(column,),
        ))

    # The period was filtered on another date column.
    def time_columns(shape: QueryShape) -> set[str]:
        return {_bare(c) for c, op, _v in shape.filters if _TIME_HINT.search(_bare(c)) and op in {">=", ">", "<", "<=", "=", "IN"}}

    ref_time, agent_time = time_columns(reference), time_columns(agent)
    if ref_time and agent_time and not (ref_time & agent_time):
        right, wrong = sorted(ref_time)[0], sorted(agent_time)[0]
        out.append(Divergence(
            kind="wrong_time_column",
            detail=f"the agent filtered the period on {wrong}, the reference on {right}",
            instruction=f"A period in a question means {right}. {wrong} is a different date and must not be used for it.",
            evidence=f"agent WHERE {wrong} ...; reference WHERE {right} ...",
            columns=(right, wrong),
        ))

    # A lookup the reference joins to turn codes into names.
    missing_joins = reference.tables - agent.tables
    for table in sorted(missing_joins):
        # Only what this table alone supplies: a code that also sits on the fact is no
        # reason to join anything.
        elsewhere = {_bare(c) for c in reference.columns if "." in c and not c.startswith(f"{table}.")}
        contributed = sorted({_bare(c) for c in reference.columns if c.startswith(f"{table}.")} - elsewhere)
        if not contributed or any(_similar(table, t) for t in agent.tables):
            continue
        picked = contributed[:3]
        named = ", ".join(picked)
        dates = [c for c in picked if _TIME_HINT.search(c)]
        others = [c for c in picked if not _TIME_HINT.search(c)]
        if dates and others:
            instruction = f"Join {table} to reach {', '.join(dates)}, the date {measure_word} is reported by, and to report {', '.join(others)}."
        elif dates:
            instruction = f"Join {table} to reach {', '.join(dates)}: it carries the date {measure_word} is reported by."
        else:
            instruction = f"Join {table} to report {', '.join(others)} rather than the raw code."
        out.append(Divergence(
            kind="missing_join",
            detail=f"the reference joins {table} for {named}; the agent's query never reads it",
            instruction=instruction,
            evidence=f"reference JOIN {table}",
            columns=(table,),
        ))

    # The reference aggregated before joining and the agent did not, over the same tables.
    if reference.aggregates_before_join and not agent.aggregates_before_join and len(agent.tables & reference.tables) > 1:
        out.append(Divergence(
            kind="fan_out_join",
            detail="the reference aggregates one side before joining; the agent joins first, so the joined side is counted once per matching row",
            instruction=f"Aggregate {measure_word} at its own grain before joining another table, or every row of the joined table multiplies it.",
            evidence="reference has an aggregate inside a subquery; the agent's query is flat",
        ))
    return tuple(out)


def instruction_lines(divergences: Iterable[Divergence]) -> tuple[str, ...]:
    """The proposed lines, in a stable order and without repeats.

    The same divergence turns up on every question it affects, and a data source's
    instructions should carry it once.
    """

    order = ["stale_table", "wrong_measure_column", "extra_measure_column", "missing_conversion", "missing_filter", "wrong_time_column", "missing_join", "fan_out_join"]
    seen: dict[str, str] = {}
    for divergence in divergences:
        seen.setdefault(divergence.instruction, divergence.kind)
    return tuple(sorted(seen, key=lambda line: (order.index(seen[line]) if seen[line] in order else len(order), line)))
