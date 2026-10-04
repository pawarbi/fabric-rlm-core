"""Column rules for a semantic model: which column or measure stands for a concept, and which must not.

A model often holds two columns for one idea: a translated and an original category name, a customer state and a
geography table joined by postcode, an order date and a delivery date. Saying which one to use in the task helps,
but a run can still read the right column and then type the other into its final query. These rules are checked
against what ran: every call on a ``SemanticModel`` records the columns, measures and tables its query referenced,
and when the run submits, the latest query that touches a concept must not use a column the rule forbids.

    rules = model.column_rules('''
        Product categories come from Products[Product Category English], not Products[Product Category].
        The buyer's state is Customers[Customer State], not the Customer Geography table.
    ''', lm=lm)
    print(rules)                       # review what was compiled, then
    rules.save("rules/sales.json")     # keep it with the model
    model = SemanticModel("Sales", rules="rules/sales.json")

The rules are also shown to the model at the start of the run. Names only are checked, never values: a rule can say
which column holds the category, not which category values to filter on.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REF = re.compile(r"^\s*'?(?P<table>[^'\[\]]+?)'?\s*\[(?P<column>[^\]]+)\]\s*$")
_MEASURE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*$")


def _key(text: str) -> str:
    return " ".join(str(text).split()).casefold()


@dataclass(frozen=True)
class Ref:
    """A column (``Table[Column]``), a measure (``[Measure]``) or a whole table (``Table``)."""

    kind: str  # column | measure | table
    table: str = ""
    name: str = ""

    @classmethod
    def parse(cls, text: str) -> "Ref":
        text = str(text).strip()
        m = _MEASURE.match(text)
        if m:
            return cls("measure", "", m.group("name").strip())
        m = _REF.match(text)
        if m:
            return cls("column", m.group("table").strip(), m.group("column").strip())
        return cls("table", text.strip("'").strip(), "")

    def __str__(self) -> str:
        if self.kind == "measure":
            return f"[{self.name}]"
        if self.kind == "column":
            table = f"'{self.table}'" if re.search(r"\W", self.table) else self.table
            return f"{table}[{self.name}]"
        return f"'{self.table}' (any column)"

    def matches(self, call: Mapping[str, Any]) -> bool:
        """Whether a recorded call referenced this name."""
        if self.kind == "measure":
            return any(_key(m) == _key(self.name) for m in call.get("measures") or [])
        columns = [Ref.parse(c) for c in call.get("columns") or []]
        if self.kind == "column":
            return any(c.kind == "column" and _key(c.table) == _key(self.table) and _key(c.name) == _key(self.name)
                       for c in columns)
        return (any(_key(t) == _key(self.table) for t in call.get("tables") or [])
                or any(c.kind == "column" and _key(c.table) == _key(self.table) for c in columns))


@dataclass
class ColumnRule:
    """For one concept: what to use and what never to use."""

    concept: str
    use: list[str]
    never: list[str]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"concept": self.concept, "use": list(self.use), "never": list(self.never)}
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class ColumnRules:
    """Rules for one semantic model. Build with ``SemanticModel.column_rules(text, lm=...)`` or from a dict/file."""

    rules: list[ColumnRule] = field(default_factory=list)
    model: str = ""
    unresolved: list[str] = field(default_factory=list)

    # -- construction -----------------------------------------------------------------------------------------
    @classmethod
    def from_value(cls, value: Any) -> "ColumnRules | None":
        """A ``ColumnRules``, a path to a saved file, a dict ``{"rules": [...]}`` or a list of rule dicts."""
        if value is None or isinstance(value, ColumnRules):
            return value
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.exists():
                raise FileNotFoundError(f"No rules file at {path}.")
            value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            items = value.get("rules") or []
            model = str(value.get("model") or "")
        elif isinstance(value, Sequence):
            items, model = list(value), ""
        else:
            raise TypeError("rules must be a ColumnRules, a path, a dict with 'rules', or a list of rules")
        rules = []
        for i, item in enumerate(items, 1):
            if not isinstance(item, Mapping) or not item.get("never"):
                raise ValueError(f"rule {i} needs at least one name under 'never' (and usually one under 'use').")
            rules.append(ColumnRule(str(item.get("concept") or f"rule {i}"), [str(x) for x in item.get("use") or []],
                                    [str(x) for x in item["never"]], str(item.get("note") or "")))
        return cls(rules, model)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "rules": [r.to_dict() for r in self.rules]}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ColumnRules":
        return cls.from_value(Path(path))  # type: ignore[return-value]

    # -- what the run sees and what is checked ---------------------------------------------------------------------
    def instructions(self, alias: str = "model") -> str:
        lines = [f"Column rules for `{alias}` (checked against the queries you run before an answer is accepted):"]
        for r in self.rules:
            use = ", ".join(str(Ref.parse(u)) for u in r.use) or "(see note)"
            never = ", ".join(str(Ref.parse(n)) for n in r.never)
            lines.append(f"- {r.concept}: use {use}; never {never}." + (f" {r.note}" if r.note else ""))
        return "\n".join(lines)

    def violations(self, calls: Iterable[Mapping[str, Any]]) -> list[str]:
        """Problems in the latest call that touches each rule's concept (names in ``use`` or ``never``)."""
        calls = list(calls)
        problems = []
        for r in self.rules:
            use, never = [Ref.parse(u) for u in r.use], [Ref.parse(n) for n in r.never]
            touching = [c for c in calls if any(ref.matches(c) for ref in use + never)]
            if not touching:
                continue
            bad = [ref for ref in never if ref.matches(touching[-1])]
            if bad:
                problems.append(
                    f"{r.concept}: your latest query that involves it used {', '.join(map(str, bad))}. "
                    f"Use {', '.join(map(str, use)) or 'the column the rule names'} instead and run the "
                    "calculation again." + (f" ({r.note})" if r.note else ""))
        return problems

    def __repr__(self) -> str:
        head = f"Rules for {self.model or 'the semantic model'} ({len(self.rules)})"
        lines = [head]
        width = max([len(r.concept) for r in self.rules] + [8])
        for i, r in enumerate(self.rules, 1):
            lines.append(f"  {i:<2} {r.concept:<{width}}  use   {', '.join(str(Ref.parse(u)) for u in r.use) or '-'}")
            lines.append(f"     {'':<{width}}  never {', '.join(str(Ref.parse(n)) for n in r.never)}")
            if r.note:
                lines.append(f"     {'':<{width}}  note  {r.note}")
        lines.append("Unresolved: " + ("; ".join(self.unresolved) if self.unresolved else "none") + ".")
        return "\n".join(lines)


# -- compiling plain words into rules ------------------------------------------------------------------------------
_COMPILE_PROMPT = """Turn the data owner's notes about a Power BI semantic model into column rules.

Model tables and columns (Table[Column]) and measures ([Measure]):
{catalog}

Notes:
{text}

Return only JSON: {{"rules": [{{"concept": "<short name>", "use": ["<names>"], "never": ["<names>"], "note": "<optional>"}}]}}
Rules:
- Use only names from the list above, spelled exactly. A whole table may be named in "never" as just its table name.
- One rule per concept the notes mention. "never" lists every name the notes rule out for that concept; "use" lists
  what the notes say to use. If the notes rule out a raw column in favour of a measure, put the measure in "use" and
  the column in "never".
- Do not invent rules the notes do not state. Leave out notes about values or periods; only names can be checked."""


def _catalog(model: Any) -> tuple[list[str], list[str], list[str]]:
    from .semantic_checks import _records  # noqa: PLC0415 - avoid a cycle at import time

    columns, measures, tables = [], [], set()
    for r in _records(model.columns()):
        t = r.get("Table Name") or r.get("table_name")
        c = r.get("Column Name") or r.get("column_name")
        if t and c and not str(c).startswith("RowNumber"):
            columns.append(f"{t}[{c}]")
            tables.add(str(t))
    for r in _records(model.measures()):
        n = r.get("Measure Name") or r.get("measure_name")
        if n:
            measures.append(f"[{n}]")
    return columns, measures, sorted(tables)


def _resolve(name: str, columns: list[str], measures: list[str], tables: list[str]) -> str | None:
    ref = Ref.parse(name)
    if ref.kind == "measure":
        hit = [m for m in measures if _key(m[1:-1]) == _key(ref.name)]
    elif ref.kind == "column":
        hit = [c for c in columns if (p := Ref.parse(c)) and _key(p.table) == _key(ref.table) and _key(p.name) == _key(ref.name)]
    else:
        hit = [t for t in tables if _key(t) == _key(ref.table)]
    return hit[0] if hit else None


def compile_rules(model: Any, text: str, lm: Any) -> ColumnRules:
    """One model call turns the notes into rules; every name is then checked against the model's own metadata."""
    from .lm import resolve_lm  # noqa: PLC0415
    from .runtime import _call_lm_text  # noqa: PLC0415

    columns, measures, tables = _catalog(model)
    catalog = "\n".join(columns + measures)
    reply = _call_lm_text(resolve_lm(lm), [{"role": "user", "content": _COMPILE_PROMPT.format(catalog=catalog, text=text.strip())}])
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        raise ValueError("The model did not return rules as JSON; write them as a dict instead (see ColumnRules).")
    raw = json.loads(m.group(0))
    rules, unresolved = [], []
    for item in raw.get("rules") or []:
        use, never = [], []
        for bucket, out in (("use", use), ("never", never)):
            for name in item.get(bucket) or []:
                hit = _resolve(str(name), columns, measures, tables)
                if hit:
                    out.append(hit)
                else:
                    unresolved.append(f"{item.get('concept', '?')}: {name}")
        if never:
            rules.append(ColumnRule(str(item.get("concept") or "rule"), use, never, str(item.get("note") or "")))
    return ColumnRules(rules, str(getattr(model, "dataset", "") or ""), unresolved)


def _model_inputs(inputs: Mapping[str, Any] | None) -> list[tuple[str, Any]]:
    """(alias, handle) for every bound SemanticModel-like input, up to one level of nesting."""
    found: list[tuple[str, Any]] = []

    def visit(name: str, value: Any, depth: int) -> None:
        if hasattr(value, "dax") and hasattr(type(value), "query_telemetry"):
            found.append((name, value))
        elif depth < 1 and isinstance(value, Mapping):
            for k, v in value.items():
                visit(f"{name}.{k}", v, depth + 1)

    for name, value in (inputs or {}).items():
        visit(str(name), value, 0)
    return found


def rules_for_inputs(inputs: Mapping[str, Any] | None) -> list[tuple[str, ColumnRules]]:
    """(alias, rules) for each bound SemanticModel input that carries rules."""
    return [(alias, m.rules) for alias, m in _model_inputs(inputs)
            if isinstance(getattr(m, "rules", None), ColumnRules) and m.rules.rules]


def _owner(call: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    """Which bound model a call record belongs to: its ``input`` name, or None when that name is not a bound alias."""
    name = str(call.get("input") or "")
    for alias in sorted(aliases, key=len, reverse=True):
        if name == alias or name.startswith(alias + ".") or name.startswith(alias + "["):
            return alias
    return None


def check_calls(inputs: Mapping[str, Any] | None, calls: Iterable[Mapping[str, Any]]) -> list[str]:
    """Rule violations across every bound model, each judged only on its own calls.

    A call is attributed by the input name the worker recorded. A call under a name that is not a bound alias (the
    run renamed the handle, ``m = model``) is attributed to the one bound model when there is only one; with several
    models it cannot be attributed and is left out rather than charged to the wrong model.
    """
    calls = list(calls)
    models = _model_inputs(inputs)
    aliases = [a for a, _ in models]
    problems: list[str] = []
    for alias, rules in rules_for_inputs(inputs):
        own = [c for c in calls if _owner(c, aliases) == alias
               or (len(models) == 1 and _owner(c, aliases) is None)]
        problems += [f"`{alias}` {p}" for p in rules.violations(own)]
    return problems
