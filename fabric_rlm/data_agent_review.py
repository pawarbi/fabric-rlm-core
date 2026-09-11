"""Review a Fabric Data Agent against its own sources.

Point the review at a Data Agent and it reads what the agent uses (its
data sources, the instructions at agent and source level, the few-shots,
the descriptions), profiles those sources through fabric-rlm, and then does
what nobody does by hand:

1. **Diagnose the setup** against the documented configuration guidance
   (Microsoft Learn: configuration best practices, the iterative process,
   data source routing) and the measured instruction patterns from the
   ``fda-skill`` optimization work. Table, column and measure names in the
   agent-level instructions belong in the data-source instructions, where
   the query generator reads them; names that no source has are flagged;
   definitions are extracted and checked for conflicts between levels;
   length, structure, few-shot counts, descriptions for routing and schema
   size are checked against the limits that were measured.
2. **Build ground truth without ground truth.** Questions are generated
   from the schemas (totals by period, top-N, breakdowns, distinct counts,
   year over year, channel combinations when two fact tables share a
   measure) and each reference is computed by an executor that runs
   against the source directly: SQL for a lakehouse, a bounded aggregate
   for a semantic model. Every reference carries the query that produced it.
3. **Evaluate the agent.** The same questions go to the agent. Through the
   Assistants API the run steps also expose the query the agent executed,
   so the review grades the query where it can (re-running it against the
   source) and the prose otherwise, and classifies a failure by cause:
   missing rows, an unsorted ranking, values that match a narrower scope,
   values that match nothing, an abstention where the source answers.
   Repetitions surface inconsistency, the documented sign of a routing or
   ambiguity problem.
4. **Suggest and report.** Schema lines move from the agent to the source
   instructions under structured headers, missing definitions and joins
   are added, failed questions become few-shots built from executed
   reference queries, and the whole thing is a Markdown report. Applying
   the suggestions is a separate, explicit step against the draft stage.

The module has no Fabric dependency of its own: readers, executors, askers
and writers are small protocols with REST implementations for use outside
a notebook and SDK implementations for use inside one.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = [
    "AgentAnswer",
    "AgentDataSource",
    "AgentSnapshot",
    "AssistantsAgentAsker",
    "FewShot",
    "Finding",
    "Graded",
    "LakehouseExecutor",
    "McpAgentAsker",
    "Question",
    "Reference",
    "RestAgentReader",
    "RestAgentWriter",
    "ReviewReport",
    "SdkAgentReader",
    "SdkAgentWriter",
    "SemanticModelExecutor",
    "SourceSchema",
    "Suggestions",
    "apply_suggestions",
    "build_references",
    "declared_from_snapshot",
    "diagnose",
    "discover_years",
    "extract_definitions",
    "generate_questions",
    "grade",
    "review_agent",
    "schema_from_profile",
    "schema_from_tables",
    "suggest",
]

# Limits and sweet spots. The character limits and few-shot counts come from
# the measured runs in the fda-skill optimization playbook (instructions past
# about 4,600 characters are silently truncated; 1,500 to 3,000 is the sweet
# spot; more than 15 to 20 few-shots per source dilutes the signal, 1 to 4
# targeted ones is the sweet spot; more than 25 tables in one source hurts
# routing). The rest follows the Learn guidance on configuration and routing.
INSTRUCTION_TRUNCATION_CHARS = 4600
INSTRUCTION_SWEET_SPOT = (1500, 3000)
FEWSHOT_SWEET_SPOT = (1, 4)
FEWSHOT_FLOOD = 15
LARGE_SCHEMA_TABLES = 25


# --------------------------------------------------------------------------- #
# What an agent is, as the review sees it
# --------------------------------------------------------------------------- #

_KIND_BY_FABRIC_TYPE = {
    "lakehousetables": "lakehouse",
    "lakehouse": "lakehouse",
    "semanticmodel": "semantic_model",
    "dataset": "semantic_model",
    "warehouse": "warehouse",
    "warehousetables": "warehouse",
    "datawarehouse": "warehouse",
    "kqldatabase": "kql",
    "kusto": "kql",
}


def _kind(fabric_type: object) -> str:
    return _KIND_BY_FABRIC_TYPE.get(str(fabric_type or "").replace("_", "").casefold(), "other")


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


def schema_from_profile(profile: Any, *, source_id: str | None = None) -> SourceSchema:
    """A :class:`SourceSchema` from a fabric-rlm ``SourceProfile``.

    ``source_id`` overrides the profile's id when the review keys sources by
    the agent's data-source id rather than the alias the package used.
    """
    family = str(getattr(profile, "family", "") or "")
    raw = getattr(profile, "schema", None) or {}
    sid = source_id or str(getattr(profile, "source_id", "") or "source")
    if family == "semantic_model":
        tables: dict[str, list[str]] = {}
        for name in (raw.get("columns") or {}):
            table, column = _split_dax_reference(str(name))
            tables.setdefault(table, []).append(column)
        measures = tuple(_split_dax_reference(str(name))[1] for name in (raw.get("measures") or {}))
        relationships: list[tuple[str, str, str, str]] = []
        for key, entry in (raw.get("relationships") or {}).items():
            rel = _relationship_tuple(key, entry)
            if rel:
                relationships.append(rel)
        return SourceSchema(sid, "semantic_model", {t: tuple(c) for t, c in tables.items()}, measures, tuple(relationships))
    if family == "lakehouse":
        tables = {}
        types: dict[str, dict[str, str]] = {}
        for name, entry in raw.items():
            columns = entry.get("columns") if isinstance(entry, Mapping) else None
            if isinstance(columns, Mapping):
                tables[str(name)] = list(columns)
                types[str(name)] = {str(c): str(meta.get("lakehouse_type") or meta.get("type") or "") for c, meta in columns.items() if isinstance(meta, Mapping)}
        return SourceSchema(sid, "lakehouse", {t: tuple(c) for t, c in tables.items()}, types=types)
    columns = [str(name) for name, entry in raw.items() if isinstance(entry, Mapping)]
    return SourceSchema(sid, family or "tabular", {sid: tuple(columns)})


def _split_dax_reference(name: str) -> tuple[str, str]:
    match = re.fullmatch(r"'?([^'\[\]]+)'?\[([^\]]+)\]", name.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", name.strip()


def _relationship_tuple(key: object, entry: object) -> tuple[str, str, str, str] | None:
    if isinstance(entry, Mapping):
        parts = [entry.get(k) for k in ("from_table", "from_column", "to_table", "to_column")]
        if all(isinstance(p, str) and p for p in parts):
            return tuple(parts)  # type: ignore[return-value]
    if isinstance(key, str):
        match = re.fullmatch(r"'?([^'\[\]]+)'?\[([^\]]+)\]\s*->\s*'?([^'\[\]]+)'?\[([^\]]+)\]", key.strip())
        if match:
            return tuple(match.group(i).strip() for i in range(1, 5))  # type: ignore[return-value]
    return None


# --------------------------------------------------------------------------- #
# Readers: REST (outside a notebook) and SDK (inside one)
# --------------------------------------------------------------------------- #

FABRIC_API = "https://api.fabric.microsoft.com/v1"


def _http_json(url: str, token: str, *, method: str = "GET", body: Mapping[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    return json.loads(text) if text.strip() else {}


_REFERENCE_KEYS = ("lakehouseReference", "semanticModelReference", "warehouseReference", "kqlDatabaseReference", "itemReference")


def _item_reference(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    """The item a datasource entry points at: ``{"itemId", "workspaceId"}``."""
    for key in _REFERENCE_KEYS:
        if isinstance(entry.get(key), Mapping):
            return entry[key]
    return {k: entry[k] for k in ("itemId", "workspaceId") if entry.get(k)}


def _fewshot_records(payload: Any) -> list[FewShot]:
    """Few-shots from a DataFrame (SDK), a ``{"value": [...]}`` page (REST) or a list of records."""
    if payload is None:
        return []
    if hasattr(payload, "to_dict"):
        records = payload.to_dict(orient="records")
    elif isinstance(payload, Mapping):
        records = payload.get("value") or payload.get("fewShots") or []
    else:
        records = list(payload)
    shots = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        lowered = {str(k).casefold(): v for k, v in record.items()}
        question = lowered.get("question")
        if question is None:
            continue
        shots.append(FewShot(str(lowered.get("id", "") or ""), str(question), str(lowered.get("query") or lowered.get("answer") or "")))
    return shots


def _source_from_entry(entry: Mapping[str, Any], fewshots: Sequence[FewShot], *, fallback_id: str = "", name: str | None = None, selected_tables: Sequence[str] = ()) -> AgentDataSource:
    """An ``AgentDataSource`` from a public-API datasource payload (REST or SDK)."""
    reference = _item_reference(entry)
    return AgentDataSource(
        id=str(entry.get("id") or fallback_id),
        kind=_kind(entry.get("type")),
        name=str(name or entry.get("displayName") or entry.get("name") or reference.get("itemId") or entry.get("id") or fallback_id),
        instructions=str(entry.get("instructions") or entry.get("dataSourceInstructions") or ""),
        description=str(entry.get("description") or ""),
        item_id=reference.get("itemId"),
        workspace_id=reference.get("workspaceId"),
        fewshots=tuple(fewshots),
        selected_tables=tuple(selected_tables),
    )


def _resolve_item_name(item_id: str | None, workspace_id: str | None) -> str | None:
    """The display name of a Fabric item through sempy, inside a notebook; ``None`` elsewhere."""
    if not item_id:
        return None
    try:
        import sempy.fabric as fabric
    except ImportError:
        return None
    try:
        return str(fabric.resolve_item_name(item_id, workspace=workspace_id))
    except Exception:  # noqa: BLE001 - fall back to the listing
        pass
    try:
        items = fabric.list_items(workspace=workspace_id)
        row = items[items["Id"] == item_id]
        return str(row["Display Name"].iloc[0]) if len(row) else None
    except Exception:  # noqa: BLE001
        return None


_PII_COLUMN = re.compile(r"(email|phone|addressline|streetaddress|address1)", re.IGNORECASE)
_MONTH_NAMES = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|q[1-4]|quarter)\b", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b|\b\(\d{3}\)\s*\d{3}[-. ]\d{4}\b")
_DIRECTION = re.compile(r"\b(up|down|increase|increased|decrease|decreased|grew|growth|fell|rose|declin\w*|higher|lower|drop\w*|gain\w*|loss\w*)\b", re.IGNORECASE)
_RANK_MARKER = re.compile(r"(^|\s)(1[.):]|#1\b|1st\b|first\b|\| *1 *\|)", re.IGNORECASE | re.MULTILINE)
_RULE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("state_period", re.compile(r"state (the )?(channel and )?(the )?(date range|period|time period|time window)", re.IGNORECASE)),
    ("state_channel", re.compile(r"state (the )?channel", re.IGNORECASE)),
    ("rank_format", re.compile(r"for rankings?,? include (the )?rank", re.IGNORECASE)),
    ("trend_format", re.compile(r"for trends?,? include (the )?direction", re.IGNORECASE)),
    ("currency_format", re.compile(r"format currency|two decimals", re.IGNORECASE)),
    ("partial_year_caveat", re.compile(r"partial[- ]year|is partial|contains (only )?partial|only partial data", re.IGNORECASE)),
    ("no_pii", re.compile(r"(do not|don't|never|avoid) (list|select|return|show|expose)[^.]*(email|phone|address|personal|pii)|protect personal data|\bPII\b", re.IGNORECASE)),
    ("no_direct_fact_join", re.compile(r"never join (\w+) directly to (\w+)", re.IGNORECASE)),
    ("calendar_not_fiscal", re.compile(r"fiscal\w*[^.]*only when (explicitly )?requested", re.IGNORECASE)),
    ("direct_territory_join", re.compile(r"direct fact-to-territory join", re.IGNORECASE)),
)


@dataclass(frozen=True)
class Rule:
    """A checkable rule the instructions state: what it asks, and the line it came from."""

    id: str
    kind: str
    text: str
    tables: tuple[str, ...] = ()
    year: int | None = None


def extract_rules(instructions: str) -> list[Rule]:
    """The instructions' rules the review can check on answers and executed queries."""
    rules: list[Rule] = []
    seen: set[str] = set()
    for line in _lines(instructions):
        for kind, pattern in _RULE_PATTERNS:
            match = pattern.search(line)
            if match is None or kind in seen:
                continue
            seen.add(kind)
            tables: tuple[str, ...] = ()
            year: int | None = None
            if kind == "no_direct_fact_join":
                tables = (match.group(1), match.group(2))
            if kind == "partial_year_caveat":
                years = re.findall(r"\b(20\d\d)\b", line)
                year = int(years[0]) if years else None
            rules.append(Rule(kind, kind, line, tables, year))
    return rules


def check_rules(rules: Sequence[Rule], question: Question, answers: Sequence[AgentAnswer], channel_words: Collection[str] = ()) -> tuple[str, ...]:
    """The rules an answer breaks, by id; a rule that does not apply to the question is not counted."""
    if not answers:
        return ()
    answer = answers[0]
    text = answer.text or ""
    sql = (answer.query or "") if (answer.language or "sql") == "sql" else ""
    has_figures = bool(_numbers_in(text))
    declined = bool(_ABSTAIN_HINT.search(text)) and not has_figures
    broken: list[str] = []
    for rule in rules:
        if rule.kind == "state_period" and has_figures and not re.search(r"\b(19|20)\d\d\b", text) and not _MONTH_NAMES.search(text):
            broken.append(rule.id)
        elif rule.kind == "state_channel" and has_figures and channel_words and not any(word and word in text.casefold() for word in channel_words):
            broken.append(rule.id)
        elif rule.kind == "rank_format" and question.kind in _RANKED_KINDS and has_figures and not _RANK_MARKER.search(text):
            broken.append(rule.id)
        elif rule.kind == "trend_format" and question.kind in _CHANGE_KINDS | {"month_trend", "quarter_trend", "entity_trend"} and has_figures and not ("%" in text and _DIRECTION.search(text)):
            broken.append(rule.id)
        elif rule.kind == "currency_format" and "$" in text and re.search(r"\$\s?\d[\d,]*(?![\d,]*\.\d\d)(?![\d,]*[.\d]*[KMB]\b)", text):
            broken.append(rule.id)
        elif rule.kind == "partial_year_caveat" and rule.year is not None and str(rule.year) in question.text and has_figures and not re.search(r"partial|incomplete|so far|to date|through|only (covers|includes)|not (yet )?complete", text, re.IGNORECASE):
            broken.append(rule.id)
        elif rule.kind == "no_pii" and (_EMAIL.search(text) or _PHONE.search(text)):
            broken.append(rule.id)
        elif rule.kind == "no_direct_fact_join" and sql and len(rule.tables) == 2 and all(re.search(rf"\b{re.escape(t)}\b", sql, re.IGNORECASE) for t in rule.tables) and "UNION" not in sql.upper():
            broken.append(rule.id)
        elif rule.kind == "calendar_not_fiscal" and sql and "fiscal" not in question.text.casefold() and re.search(r"fiscal", sql, re.IGNORECASE):
            broken.append(rule.id)
        elif rule.kind == "direct_territory_join" and sql and re.search(r"territor|region|country", question.text, re.IGNORECASE) and re.search(r"\bdimgeography\b", sql, re.IGNORECASE) and re.search(r"\bdimsalesterritory\b", sql, re.IGNORECASE):
            broken.append(rule.id)
    if declined:
        return tuple(b for b in broken if b in {"no_pii"})
    return tuple(broken)


_LEAF_ELEMENT = re.compile(r"(column|measure|parameter|returnvalue|field)", re.IGNORECASE)
_TABLE_ELEMENT = re.compile(r"(?:^|[._])(?:lakehouse|warehouse|kusto)?(table|entity|dataset)$", re.IGNORECASE)  # Table, LakehouseTable, lakehouse_tables.table; never the Tables container
_VIEW_ELEMENT = re.compile(r"(?:^|[._])(?:lakehouse|warehouse)?(view|function|procedure)$", re.IGNORECASE)
_CONTAINER_ELEMENT = re.compile(r"^(schemas|tables|views|functions|procedures|entities|datasets|folders|objects|databases|columns|measures)$", re.IGNORECASE)
_SKIPPED_CONTAINER = re.compile(r"^(views|functions|procedures|columns|measures|parameters)$", re.IGNORECASE)  # holds no tables


def _selected_table_paths(fetch: Callable[[str | None, str | None], Mapping[str, Any] | None]) -> list[str]:
    """Paths (``schema/table`` or ``table``) of the selected tables in a datasource's elements tree.

    ``fetch(root_id, continuation_token)`` returns one page of elements
    (``{"value": [...], "continuationToken": ...}``). Fabric shapes a
    lakehouse tree as ``Schemas`` > ``dbo`` > ``Tables`` | ``Views`` >
    ``Table`` | ``View`` > columns: the structural containers (``Schemas``,
    ``Tables``, ``Views``) are walked but not named, a schema names its
    tables, a table is not walked (its children are its columns), and views
    and functions are left out because the review reads Delta tables.
    """
    found: list[str] = []

    def children(root_id: str | None) -> list[Mapping[str, Any]]:
        items: list[Mapping[str, Any]] = []
        token: str | None = None
        for _page in range(1000):
            page = fetch(root_id, token) or {}
            items.extend(item for item in (page.get("value") or []) if isinstance(item, Mapping))
            token = page.get("continuationToken")
            if not token:
                break
        return items

    def walk(root_id: str | None, prefix: str, depth: int) -> None:
        for element in children(root_id):
            kind = str(element.get("type") or "")
            if _LEAF_ELEMENT.search(kind) or _VIEW_ELEMENT.search(kind) or _SKIPPED_CONTAINER.match(kind):
                continue
            name = str(element.get("displayName") or element.get("name") or "")
            selected = element.get("isSelected") if "isSelected" in element else element.get("is_selected")
            if _TABLE_ELEMENT.search(kind):
                if selected and name:
                    found.append(f"{prefix}/{name}" if prefix else name)
                continue
            if element.get("id") is None or element.get("hasSubElements") is False or depth >= 5:
                continue
            path = prefix if _CONTAINER_ELEMENT.match(kind) else (f"{prefix}/{name}" if prefix else name)
            walk(str(element["id"]), path, depth + 1)

    walk(None, "", 0)
    return found


class RestAgentReader:
    """Reads a Data Agent through the public Fabric REST API.

    ``token_provider`` returns a bearer token for ``https://api.fabric.microsoft.com``;
    outside Fabric that is typically ``AzureCliCredential().get_token(scope).token``.
    ``stage`` is ``"published"`` or ``"staging"``.
    """

    def __init__(self, workspace_id: str, agent_id: str, token_provider: Callable[[], str], *, stage: str = "published") -> None:
        self.workspace_id = workspace_id
        self.agent_id = agent_id
        self._token = token_provider
        self.stage = stage

    def _base(self) -> str:
        return f"{FABRIC_API}/workspaces/{self.workspace_id}/dataAgents/{self.agent_id}"

    def _stage_path(self, path: str) -> str:
        prefix = "staging/" if self.stage == "staging" else ""
        return f"{self._base()}/{prefix}{path}"

    def snapshot(self) -> AgentSnapshot:
        token = self._token()
        item = _http_json(self._base(), token)
        settings = _http_json(self._stage_path("settings"), token)
        sources = []
        for entry in _http_json(self._stage_path("datasources"), token).get("value", []):
            try:
                fewshots = _fewshot_records(_http_json(self._stage_path(f"datasources/{entry['id']}/fewshots"), token))
            except RuntimeError:
                fewshots = []
            reference = _item_reference(entry)
            name = entry.get("displayName") or entry.get("name")
            if not name and reference.get("itemId") and reference.get("workspaceId"):
                # the datasource listing carries the item id only; the item has the name
                try:
                    name = _http_json(f"{FABRIC_API}/workspaces/{reference['workspaceId']}/items/{reference['itemId']}", token).get("displayName")
                except RuntimeError:
                    name = None

            def elements(root_id: str | None, continuation: str | None, *, datasource_id: str = str(entry["id"])) -> Mapping[str, Any]:
                params = urllib.parse.urlencode({k: v for k, v in (("rootId", root_id), ("continuationToken", continuation)) if v})
                return _http_json(self._stage_path(f"datasources/{datasource_id}/elements") + (f"?{params}" if params else ""), token)

            try:
                selected = _selected_table_paths(elements)
            except RuntimeError:
                selected = []
            sources.append(_source_from_entry(entry, fewshots, name=name, selected_tables=selected))
        return AgentSnapshot(
            agent_id=self.agent_id,
            name=str(item.get("displayName") or self.agent_id),
            description=str(item.get("description") or ""),
            instructions=str(settings.get("aiInstructions") or ""),
            datasources=tuple(sources),
            stage=self.stage,
            workspace_id=self.workspace_id,
        )


class SdkAgentReader:
    """Reads a Data Agent through the SDK's ``FabricDataAgentManagement``.

    Inside a Fabric notebook::

        from fabric.dataagent.client import FabricDataAgentManagement
        snapshot = SdkAgentReader(FabricDataAgentManagement("Sales Agent RLM")).snapshot()

    Uses the SDK's public-API methods (``get_settings``, ``list_datasources``,
    the handle's ``get_configuration(stage)`` and ``get_fewshots(stage)``),
    which return the same payloads as the REST reader, so a source in another
    workspace keeps its workspace id. Older SDKs without those methods are
    read through the legacy workload-host methods and their snake_case keys.
    """

    def __init__(self, management: Any, *, stage: str = "staging") -> None:
        self._management = management
        self.stage = stage

    def snapshot(self) -> AgentSnapshot:
        management = self._management
        client = getattr(management, "_client", None)
        name = getattr(management, "data_agent_name", None) or getattr(client, "data_agent_name", "") or "data agent"
        agent_id = getattr(management, "data_agent_id", None) or getattr(client, "data_agent_id", "") or name
        workspace_id = getattr(management, "workspace_id", None) or getattr(client, "workspace_id", None)
        if hasattr(management, "list_datasources") and hasattr(management, "get_settings"):
            settings = management.get_settings(stage=self.stage) or {}
            sources = []
            for handle in management.list_datasources(stage=self.stage):
                entry = handle.get_configuration(stage=self.stage) or {}
                reference = _item_reference(entry)
                source_name = entry.get("displayName") or entry.get("name") or _resolve_item_name(reference.get("itemId"), reference.get("workspaceId"))
                try:
                    shots = _fewshot_records(handle.get_fewshots(stage=self.stage))
                except Exception:  # noqa: BLE001 - few-shots are optional
                    shots = []
                selected: list[str] = []
                if hasattr(handle, "get_elements"):
                    try:
                        selected = _selected_table_paths(lambda root_id, token, h=handle: h.get_elements(stage=self.stage, root_id=root_id, continuation_token=token))
                    except Exception:  # noqa: BLE001 - the elements are a bonus; the whole source is reviewed without them
                        selected = []
                sources.append(_source_from_entry(entry, shots, fallback_id=str(getattr(handle, "_id", "") or ""), name=source_name, selected_tables=selected))
            return AgentSnapshot(
                agent_id=str(agent_id),
                name=str(name),
                instructions=str(settings.get("aiInstructions") or ""),
                datasources=tuple(sources),
                stage=self.stage,
                workspace_id=str(workspace_id) if workspace_id else None,
            )
        return self._legacy_snapshot(str(agent_id), str(name), workspace_id)

    def _legacy_snapshot(self, agent_id: str, name: str, workspace_id: Any) -> AgentSnapshot:
        management = self._management
        configuration = management.get_configuration()
        instructions = getattr(configuration, "instructions", None)
        if instructions is None and isinstance(configuration, Mapping):
            instructions = configuration.get("aiInstructions") or configuration.get("additionalInstructions") or configuration.get("instructions")
        sources = []
        for datasource in management.get_datasources():
            config = datasource.get_configuration() if hasattr(datasource, "get_configuration") else {}
            config = config if isinstance(config, Mapping) else {}
            try:
                shots = _fewshot_records(datasource.get_fewshots())
            except Exception:  # noqa: BLE001 - few-shots are optional
                shots = []
            # the legacy payload uses snake_case keys and names the item by its id
            entry = {
                "id": getattr(datasource, "_id", None) or config.get("id") or "",
                "type": config.get("type") or getattr(datasource, "type", ""),
                "displayName": config.get("displayName") or config.get("display_name") or getattr(datasource, "name", "") or "",
                "instructions": config.get("instructions") or config.get("additional_instructions") or config.get("dataSourceInstructions") or "",
                "description": config.get("description") or config.get("user_description") or config.get("userDescription") or "",
                "itemId": config.get("artifactId") or config.get("artifact_id") or config.get("id"),
                "workspaceId": config.get("workspaceId") or config.get("workspace_id"),
            }
            sources.append(_source_from_entry(entry, shots, fallback_id=str(entry["id"])))
        return AgentSnapshot(
            agent_id=agent_id,
            name=name,
            instructions=str(instructions or ""),
            datasources=tuple(sources),
            stage="staging",
            workspace_id=str(workspace_id) if workspace_id else None,
        )


# --------------------------------------------------------------------------- #
# Diagnosis of the setup against the documented guidance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str  # "high" | "medium" | "low" | "info"
    message: str
    source_id: str | None = None
    evidence: tuple[str, ...] = ()
    suggestion: str = ""
    basis: str = ""  # where the rule comes from


_IDENTIFIER = re.compile(r"(?<![\w.])(?:dbo\.)?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)(?![\w.])")
_DAX_REFERENCE = re.compile(r"'?[A-Za-z_][A-Za-z0-9_ ]*'?\[[^\]]+\]")
_SCHEMA_LIKE = re.compile(r"^(?:vw_|dim|fact|tbl_|stg_|v_)[A-Za-z0-9_]+$|^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$", re.IGNORECASE)
_DEFINITION = re.compile(
    r"^\s*(?:[-*]\s*)?(?P<name>[A-Za-z][A-Za-z0-9 %/()_-]{1,60}?)\s*(?:=|means|is defined as)\s*(?P<text>.+?)\.?\s*$",
    re.IGNORECASE,
)
_CALCULATION = re.compile(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", re.IGNORECASE)
_HEADER = re.compile(r"^\s*(?:#{1,3}\s+\S|[A-Z][A-Z &/-]{3,}:?\s*$)")
_NEGATIVE = re.compile(r"^\s*(?:[-*]\s*)?(avoid|do not|don't|never)\b", re.IGNORECASE)
_STOPWORDS = {"the", "and", "for", "with", "use", "only", "from", "not", "all", "when", "then", "sum", "count", "distinct", "avg", "min", "max", "as", "by", "in", "on", "or", "of", "to", "a", "an", "is", "are", "table", "tables", "dbo"}

_LEARN_CONFIG = "Learn: best practices for improving data agent query generation"
_LEARN_ROUTING = "Learn: improve data source routing"
_LEARN_ITERATIVE = "Learn: adopting an iterative process"
_FDA_PATTERNS = "fda-skill: measured instruction patterns"


def _lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _schema_shaped(token: str, schema: SourceSchema) -> bool:
    """A token that reads as a schema name rather than an English word.

    Dotted, underscored, digit-bearing or camel-cased tokens, DAX references
    and table names count; ``status``, ``phone`` or ``channel`` do not, even
    when a column of that name exists.
    """
    if "[" in token or "." in token or "_" in token or any(c.isdigit() for c in token):
        return True
    if token.casefold() in {t.casefold() for t in schema.tables}:
        return True
    return token[:1].isalpha() and token[1:] != token[1:].lower() and not token.isupper()


def _schema_mentions(line: str, schemas: Sequence[SourceSchema]) -> dict[str, list[str]]:
    """Schema identifiers a line names, grouped by the source that has them.

    A line counts only when at least one hit is schema-shaped; the
    schema-shaped hits are listed first.
    """
    found: dict[str, list[str]] = {}
    candidates = {m.group(1) for m in _IDENTIFIER.finditer(line)} | {m.group(0) for m in _DAX_REFERENCE.finditer(line)}
    for schema in schemas:
        known = schema.identifiers()
        hits = sorted({c for c in candidates if c.casefold() in known and c.casefold() not in _STOPWORDS and len(c) > 2})
        strong = [c for c in hits if _schema_shaped(c, schema)]
        if strong:
            found[schema.source_id] = strong
    return found


_PLAIN_WORDS = {"dimension", "dimensions", "fact", "facts", "factor", "factors", "factual", "dim", "view", "views"}


def _unknown_schema_like(line: str, schema: SourceSchema) -> list[str]:
    known = schema.identifiers()
    unknown = []
    for match in _IDENTIFIER.finditer(line):
        token = match.group(1)
        if not _SCHEMA_LIKE.match(token) or token.casefold() in _PLAIN_WORDS:
            continue
        # a schema-shaped token is a name, not an English word: dotted,
        # underscored, digit-bearing, or a long lower/camel-case compound
        shaped = "." in token or "_" in token or any(c.isdigit() for c in token) or (len(token) >= 8 and not token.isupper())
        if shaped and token.casefold() not in known and token.split(".")[-1].casefold() not in known:
            unknown.append(token)
    return sorted(set(unknown))


def _tables_named_in(text: str, schema: SourceSchema) -> list[str]:
    """Tables the instructions name, in the order they first appear."""
    lower = {t.casefold(): t for t in schema.tables}
    named: list[str] = []
    for match in _IDENTIFIER.finditer(text or ""):
        table = lower.get(match.group(1).split(".")[-1].casefold())
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


def excluded_tables(schema: SourceSchema, instructions: str) -> list[str]:
    """Tables of the schema named after a topic the instructions put out of scope."""
    excluded = excluded_terms(instructions)
    return sorted(t for t in schema.tables if _mentions_excluded(t, excluded))


def _scoped_facts(schema: SourceSchema, instructions: str) -> list[str]:
    """Fact tables in the agent's declared scope first; all of them when it names none; never an out-of-scope one."""
    excluded = excluded_terms(instructions)
    facts = [t for t in _fact_tables(schema) if not _mentions_excluded(t, excluded)]
    named = _tables_named_in(instructions, schema)
    in_scope = [t for t in named if t in facts]
    return in_scope or facts


def extract_definitions(text: str) -> dict[str, str]:
    """Metric and term definitions stated in instruction text (``X = ...``, ``X means ...``)."""
    definitions: dict[str, str] = {}
    for line in _lines(text):
        # a line may carry several sentences: "Revenue = SUM(x). Orders = COUNT(...)."
        for fragment in re.split(r"(?<=[.;])\s+(?=[A-Z\"'])", line):
            match = _DEFINITION.match(fragment)
            if not match:
                continue
            name = match.group("name").strip().strip('"').strip("'")
            body = match.group("text").strip()
            if len(name) < 3 or name.casefold() in _STOPWORDS or len(body) < 4:
                continue
            definitions.setdefault(name, body)
    return definitions


def diagnose(snapshot: AgentSnapshot, schemas: Sequence[SourceSchema]) -> tuple[Finding, ...]:
    """Findings about the agent's setup, from its instructions, sources and schemas."""
    findings: list[Finding] = []
    by_id = {s.source_id: s for s in schemas}
    multi_source = len(snapshot.datasources) > 1

    # 1. schema specifics at agent level belong with the source (Learn: put
    #    query-generation guidance in data-source instructions; fda A1)
    moved: dict[str, list[str]] = {}
    for line in _lines(snapshot.instructions):
        if _HEADER.match(line):
            continue
        for source_id, hits in _schema_mentions(line, schemas).items():
            moved.setdefault(source_id, []).append(f"{line}  [{', '.join(hits[:4])}]")
    for source_id, lines in moved.items():
        findings.append(
            Finding(
                code="schema_in_agent_instructions",
                severity="medium",
                source_id=source_id,
                message=(
                    f"{len(lines)} agent-level instruction line(s) name tables, columns or measures of source "
                    f"{source_id}. The query generator reads data-source instructions, not agent instructions, "
                    "so these lines do not help it write SQL or DAX where they are."
                ),
                evidence=tuple(lines[:12]),
                suggestion=f"Move these lines to the data-source instructions of {source_id}; keep routing, behaviour and response style at agent level.",
                basis=_LEARN_CONFIG,
            )
        )

    # 2. references to things the source does not have (Learn: verify every
    #    example still matches the schema; fda A2/A6)
    for source in snapshot.datasources:
        schema = by_id.get(source.id)
        if schema is None:
            continue
        unknown: dict[str, list[str]] = {}
        for line in _lines(source.instructions):
            for token in _unknown_schema_like(line, schema):
                unknown.setdefault(token, []).append(line)
        if unknown:
            findings.append(
                Finding(
                    code="unknown_reference",
                    severity="high",
                    source_id=source.id,
                    message=f"{len(unknown)} name(s) in the data-source instructions are not in the source's schema: {', '.join(sorted(unknown))}.",
                    evidence=tuple(f"{token}: {lines[0][:140]}" for token, lines in sorted(unknown.items())),
                    suggestion="Verify each exists (a SQL-endpoint view is not in the Delta table list), or remove it: a name that steers the generator to a missing object is the most damaging instruction there is.",
                    basis=_FDA_PATTERNS,
                )
            )

    # 3. definitions: stated, at the wrong level, or conflicting (fda A1)
    agent_definitions = extract_definitions(snapshot.instructions)
    for source in snapshot.datasources:
        definitions = extract_definitions(source.instructions)
        if definitions:
            findings.append(
                Finding(
                    code="definitions_declared",
                    severity="info",
                    source_id=source.id,
                    message=f"{len(definitions)} definition(s) stated in the data-source instructions; the review declares them to the RLM and checks the references against them.",
                    evidence=tuple(f"{name}: {text[:100]}" for name, text in list(definitions.items())[:8]),
                    basis=_LEARN_CONFIG,
                )
            )
        conflicts = [
            f"{name}: agent says '{agent_definitions[name][:70]}', source says '{definitions[name][:70]}'"
            for name in definitions
            if name in agent_definitions and _normalize(agent_definitions[name]) != _normalize(definitions[name])
        ]
        if conflicts:
            findings.append(
                Finding(
                    code="conflicting_definitions",
                    severity="high",
                    source_id=source.id,
                    message=f"{len(conflicts)} term(s) are defined differently at agent level and in this source's instructions.",
                    evidence=tuple(conflicts[:6]),
                    suggestion="Keep one definition, in the data-source instructions. Conflicting levels collapsed accuracy in measured runs.",
                    basis=_FDA_PATTERNS,
                )
            )
    if agent_definitions:
        findings.append(
            Finding(
                code="definitions_at_agent_level",
                severity="low",
                message=f"{len(agent_definitions)} definition(s) are stated at agent level.",
                evidence=tuple(f"{name}: {text[:100]}" for name, text in list(agent_definitions.items())[:8]),
                suggestion="Keep business scope and terminology at agent level; put formulas with the source whose columns they use.",
                basis=_LEARN_CONFIG,
            )
        )
    for source in snapshot.datasources:
        formulas = [line for line in _lines(source.instructions) if _CALCULATION.search(line) and "=" in line]
        if formulas and source.kind in {"lakehouse", "warehouse"}:
            findings.append(
                Finding(
                    code="calculation_in_instructions",
                    severity="low",
                    source_id=source.id,
                    message=f"{len(formulas)} calculation formula(s) are given as instructions; the SQL generator may not honour a calculation override.",
                    evidence=tuple(f[:120] for f in formulas[:4]),
                    suggestion="Keep the definition, and back it with a few-shot showing the query, or a view or column that materialises it.",
                    basis=_FDA_PATTERNS,
                )
            )

    # 4. length, structure, phrasing (fda P1, A3; Learn: direct instructions)
    for label, text, source_id in [("agent", snapshot.instructions, None)] + [(s.name or s.id, s.instructions, s.id) for s in snapshot.datasources]:
        if len(text) > INSTRUCTION_TRUNCATION_CHARS:
            findings.append(
                Finding(
                    code="instructions_too_long",
                    severity="high",
                    source_id=source_id,
                    message=f"The {label} instructions are {len(text):,} characters; past about {INSTRUCTION_TRUNCATION_CHARS:,} the tail is silently truncated. The sweet spot is {INSTRUCTION_SWEET_SPOT[0]:,} to {INSTRUCTION_SWEET_SPOT[1]:,}.",
                    suggestion="Compress: remove redundancy, move query patterns to few-shots, put the most critical rules first.",
                    basis=_FDA_PATTERNS,
                )
            )
        elif source_id is not None and len(text) > INSTRUCTION_SWEET_SPOT[1]:
            findings.append(
                Finding(
                    code="instructions_long",
                    severity="low",
                    source_id=source_id,
                    message=f"The {label} instructions are {len(text):,} characters, above the {INSTRUCTION_SWEET_SPOT[1]:,} sweet spot; returns diminish and truncation risk grows.",
                    basis=_FDA_PATTERNS,
                )
            )
        lines = _lines(text)
        if source_id is not None and len(lines) >= 6 and not any(_HEADER.match(line) for line in lines):
            findings.append(
                Finding(
                    code="unstructured_instructions",
                    severity="low",
                    source_id=source_id,
                    message="The data-source instructions have no section headers; structured sections measured +7.8% accuracy.",
                    suggestion="Group under headers such as Column Semantics, Join Paths, Temporal Rules, Business Rules, Table Routing, Terminology.",
                    basis=_FDA_PATTERNS,
                )
            )
        negatives = [line for line in lines if _NEGATIVE.match(line)]
        if len(negatives) >= 3:
            findings.append(
                Finding(
                    code="negative_phrasing",
                    severity="low",
                    source_id=source_id,
                    message=f"{len(negatives)} instruction line(s) say what not to do; the guidance is to state what the agent should do.",
                    evidence=tuple(n[:120] for n in negatives[:4]),
                    suggestion='Rewrite as direct instructions, for example "Join X to Y on Z" rather than "Avoid joining X incorrectly".',
                    basis=_LEARN_CONFIG,
                )
            )

    # 5. few-shots (Learn: examples for complex logic; fda P3, A5; DAX tool ignores them)
    for source in snapshot.datasources:
        count = len(source.fewshots)
        if source.kind == "semantic_model":
            if count:
                findings.append(
                    Finding(code="fewshots_unused_by_dax", severity="info", source_id=source.id, message=f"{count} few-shot(s) on a semantic model source; the DAX tool does not use few-shots, so this guidance belongs in the source instructions.", basis=_FDA_PATTERNS)
                )
            continue
        if count == 0:
            findings.append(
                Finding(
                    code="no_fewshots",
                    severity="medium",
                    source_id=source.id,
                    message=f"Source {source.name or source.id} has no example queries.",
                    suggestion="Add question-to-query pairs for the shapes the agent gets wrong (joins, preaggregation, rankings, relative dates); the evaluation proposes them from executed references.",
                    basis=_LEARN_CONFIG,
                )
            )
        elif count > FEWSHOT_FLOOD:
            findings.append(
                Finding(
                    code="fewshots_too_many",
                    severity="medium",
                    source_id=source.id,
                    message=f"Source {source.name or source.id} has {count} few-shots; beyond {FEWSHOT_FLOOD} the relevant ones are diluted. {FEWSHOT_SWEET_SPOT[0]} to {FEWSHOT_SWEET_SPOT[1]} targeted examples is the sweet spot.",
                    suggestion="Keep only examples that fix a known failure; prune any whose target question now passes without it.",
                    basis=_FDA_PATTERNS,
                )
            )

    # 6. routing signals (Learn: routing needs descriptions, then examples, then rules)
    for source in snapshot.datasources:
        if not source.description.strip():
            findings.append(
                Finding(
                    code="datasource_description_missing",
                    severity="high" if multi_source else "low",
                    source_id=source.id,
                    message=f"Source {source.name or source.id} has no description." + (" With more than one source, the description is the first routing signal." if multi_source else ""),
                    suggestion='One or two sentences on the topics and entities the source covers and when to use it, for example "Sales fact data for North America retail, including transactions, returns and store metadata."',
                    basis=_LEARN_ROUTING,
                )
            )
    if multi_source:
        names = [s.name for s in snapshot.datasources if s.name]
        mentioned = [n for n in names if n.casefold() in (snapshot.instructions or "").casefold()]
        if not mentioned:
            findings.append(
                Finding(
                    code="routing_rules_missing",
                    severity="medium",
                    message="The agent has several sources and its instructions name none of them; there is no routing rule to fall back on when descriptions and examples do not settle a question.",
                    suggestion="Add a short Topics section: when asked about X, use source A; when asked about Y, use source B. Keep it concise.",
                    basis=_LEARN_ROUTING,
                )
            )

    # 7. schema size and duplication (Learn: limit the selected schema; fda P6)
    for source in snapshot.datasources:
        schema = by_id.get(source.id)
        if schema and len(schema.tables) > LARGE_SCHEMA_TABLES:
            findings.append(
                Finding(
                    code="large_schema",
                    severity="medium",
                    source_id=source.id,
                    message=f"Source {source.name or source.id} exposes {len(schema.tables)} tables; more than {LARGE_SCHEMA_TABLES} gives the generator too many paths and weakens routing.",
                    suggestion="Deselect staging, audit and unrelated tables, or split the source; explain which of two similar tables is authoritative.",
                    basis=_LEARN_CONFIG,
                )
            )
    agent_lines = {line.casefold() for line in _lines(snapshot.instructions) if len(line) > 24}
    for source in snapshot.datasources:
        duplicates = [line for line in _lines(source.instructions) if len(line) > 24 and line.casefold() in agent_lines]
        if duplicates:
            findings.append(
                Finding(
                    code="duplicated_guidance",
                    severity="low",
                    source_id=source.id,
                    message=f"{len(duplicates)} line(s) appear in both the agent and the data-source instructions.",
                    evidence=tuple(d[:140] for d in duplicates[:6]),
                    suggestion="Keep each rule in one place; duplicates drift apart and count twice against the length limit.",
                    basis=_FDA_PATTERNS,
                )
            )

    # 8. sources the review cannot see
    for source in snapshot.datasources:
        if source.id not in by_id:
            findings.append(
                Finding(
                    code="source_not_profiled",
                    severity="high",
                    source_id=source.id,
                    message=f"Source {source.name or source.id} ({source.kind}) was not profiled, so its instructions and questions were not checked.",
                    suggestion="Bind the source to the review (a LakehouseSource or SemanticModel handle) so it is profiled.",
                )
            )
    return tuple(findings)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def declared_from_snapshot(snapshot: AgentSnapshot, schemas: Sequence[SourceSchema], context: ReviewContext | None = None) -> dict[str, dict[str, object]]:
    """What the agent's instructions declare, in ``RLM.learn(declared=...)`` form.

    The reviewer's ``context.definitions`` are added to every profiled
    source and override the agent's where the names clash.
    """
    declared: dict[str, dict[str, object]] = {}
    by_id = {s.source_id: s for s in schemas}
    for source in snapshot.datasources:
        definitions = extract_definitions(source.instructions)
        definitions.update({k: v for k, v in extract_definitions(snapshot.instructions).items() if k not in definitions})
        if context is not None:
            definitions.update({str(k): str(v) for k, v in context.definitions.items()})
        if definitions and source.id in by_id:
            declared[source.id] = {"definitions": {k: v[:250] for k, v in definitions.items()}}
    return declared


# --------------------------------------------------------------------------- #
# Questions and references
# --------------------------------------------------------------------------- #

_MEASURE_HINT = re.compile(r"(amount|qty|quantity|units|revenue|sales|cost|price|margin|total|profit|value|hours|minutes|count|arr|mrr|usd|eur|gbp|calls|duration|balance|fee|charge|spend|volume|score|rating)", re.IGNORECASE)
_KEY_HINT = re.compile(r"(key|id)$", re.IGNORECASE)  # kept for callers outside this module; the module uses _is_key
_KEY_FORMS = re.compile(r"(?:^|[_ ])(?:[Ii][Dd]|[Kk][Ee][Yy])$|[a-z0-9](?:Id|ID|Key|KEY)$")
_KEY_STOPWORDS = frozenset({"paid", "unpaid", "prepaid", "valid", "invalid", "grid", "void", "avoid", "rapid", "solid", "liquid", "fluid", "acid", "hybrid", "said", "laid", "mid", "bid", "kid", "lid", "rid", "amid", "turkey", "monkey", "hockey", "jockey", "donkey", "whiskey", "journey", "period", "bandwidth"})
_DATE_KEY = re.compile(r"date(key)?$", re.IGNORECASE)
_YEAR_COLUMN = re.compile(r"^(calendar)?year$", re.IGNORECASE)
_ATTRIBUTE_HINT = re.compile(r"(name|country|region|group|category|segment|type|class|status|city|state|line|plant)", re.IGNORECASE)
_ORDER_ID_HINT = re.compile(r"(ordernumber|orderid|order_id|order_number|ticket_id|invoice|transaction)", re.IGNORECASE)
_LOCAL_EXCLUDED = re.compile(r"(number|nbr|code|sku|email|url|website|uri|hash|uuid|guid)$", re.IGNORECASE)  # never a grouping column on a fact
_FLAG_COLUMN = re.compile(r"^(?:is|has|was|can)_|flag$|^is[A-Z]", re.IGNORECASE)  # is_active, SalesPersonFlag: a yes or no, not a quantity
_JOIN_LINE = re.compile(r"(?:dbo\.)?([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*(?:->|→|to|=)\s*(?:dbo\.)?([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class Question:
    id: str
    source_id: str
    kind: str  # total_by_year | top_n | breakdown | distinct_by_year | yoy | combined_total_by_year | supplied
    text: str
    spec: Mapping[str, Any]
    reference_query: str  # what a few-shot would carry: T-SQL for a lakehouse, DAX for a model
    execution: Mapping[str, Any] = field(default_factory=dict)  # what the executor runs
    alternates: tuple[tuple[str, Mapping[str, Any]], ...] = ()  # (label, execution) narrower scopes
    skill: str = ""  # what the question tests: aggregate | rank | change | count | filter | kpi | measure | ambiguity | scope | supplied | proposed
    technical: str = ""  # the same question in schema terms, for the report and the few-shot reader


@dataclass(frozen=True)
class Reference:
    question_id: str
    status: str  # ok | failed | abstained
    rows: tuple[Mapping[str, Any], ...] = ()
    note: str = ""
    alternates: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)


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


def _fact_tables(schema: SourceSchema) -> list[str]:
    facts = []
    joins = _heuristic_joins(schema)
    referenced = _referenced_tables(schema, joins)
    for table, columns in schema.tables.items():
        if re.match(r"^(?:[a-z0-9_]+\.)?dim[_a-z]", table, re.IGNORECASE) or re.search(r"(?:^|[._])(?:date|dates|calendar|time|dim_date)$", table, re.IGNORECASE):
            continue  # a dimension by name (dimproduct carries prices and a start date, and is still not a fact)
        measures = _measure_columns(schema, table)
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
        if measures and (dates or joined_time) and (table.casefold().startswith("fact") or len(measures) >= 2 or references_dimension or _local_attributes(schema, table)):
            facts.append(table)

    def richness(t: str) -> tuple[bool, int, int, str]:
        # a fact named as one first; then the ones with the most grouping columns and measures
        return (not t.casefold().startswith("fact"), -len(attribute_paths(schema, t, joins)), -len(_measure_columns(schema, t)), t)

    return sorted(facts, key=richness)


def _measure_columns(schema: SourceSchema, table: str) -> list[str]:
    preferred = ("salesamount", "revenue", "amount", "sales", "price", "total", "net", "gross", "totalproductcost", "orderquantity", "quantity", "units", "value")
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
    if not columns and schema.types.get(table):
        # names say nothing; the profile's types do
        columns = [c for c in schema.tables[table] if _NUMERIC_TYPE.search(schema.column_type(table, c)) and usable(c)]
    return sorted(columns, key=lambda c: (next((i for i, p in enumerate(preferred) if p in c.casefold()), 99) + (50 if secondary.search(c) else 0), c))


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


_TIME_COLUMN = re.compile(r"(date|timestamp|datetime|_at|_time|_on)(key)?$", re.IGNORECASE)
_PERIOD_COLUMN = re.compile(r"(?:^|[_ ])(?:quarter|year_?quarter|fiscal_quarter|period|fiscal_period|year_?month|month)$", re.IGNORECASE)  # 2024/Q1, 2024-03: a period written as text
_TIME_TYPE = re.compile(r"(timestamp|datetime|date)", re.IGNORECASE)  # Delta names, SQL names or arrow ``DataType<Timestamp(...)>`` and ``Date32``
_NUMERIC_TYPE = re.compile(r"(int|long|double|float|decimal|numeric|real|number|short|byte)", re.IGNORECASE)
_TEXT_TYPE = re.compile(r"(string|varchar|char|text|utf8)", re.IGNORECASE)
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
    return column if dialect == "tsql" else f"CAST({column} AS TIMESTAMP)"


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


def _has_day(dt: Mapping[str, Any]) -> bool:
    """Whether the axis can express a range of dates; a period written as text cannot."""
    return not dt.get("period")


def _max_date_sql(fact: Mapping[str, Any]) -> str:
    dt = fact["date"]
    join = _time_join(dt)
    return f"SELECT MAX({_time_column(dt)}) AS value FROM {fact['table']} f" + (f" {join}" if join else "")


def _attributes(schema: SourceSchema, table: str, joins: Mapping[tuple[str, str], tuple[str, str]], excluded: Collection[str] = ()) -> list[tuple[str, str, str, str]]:
    """(fact key, dim table, dim key, attribute column) for grouping; never through an out-of-scope dimension."""
    found = []
    for column in schema.tables[table]:
        target = joins.get((table, column))
        if target is None or _DATE_KEY.search(column):
            continue
        dim_table, dim_key = target
        if _mentions_excluded(dim_table, excluded):
            continue
        for attribute in schema.tables[dim_table]:
            if _is_attribute(schema, dim_table, attribute) and attribute != dim_key:
                found.append((column, dim_table, dim_key, attribute))
    for attribute in _local_attributes(schema, table):
        found.append((None, table, None, attribute))  # type: ignore[arg-type]
    return found


def _local_attributes(schema: SourceSchema, table: str) -> list[str]:
    """Grouping columns on the fact table itself: a flat sales file has its categories on the same rows."""
    return [
        c
        for c in schema.tables[table]
        if _is_attribute(schema, table, c) and not _ORDER_ID_HINT.search(c) and not _LOCAL_EXCLUDED.search(c) and not _PERSONAL_HINT.search(c) and not _MEASURE_HINT.search(c)
    ]


_LANGUAGE_VARIANT = re.compile(r"^(spanish|french|german|italian|portuguese|dutch|japanese|chinese)", re.IGNORECASE)


def _diverse_attributes(attributes: Sequence[tuple[str, str, str, str]], terms: Collection[str] = ()) -> list[tuple[str, str, str, str]]:
    """Grouping attributes with one per dimension table first, the reviewer's terms first, English names before translated variants."""

    def score(attribute: tuple[str, str, str, str]) -> int:
        haystack = f"{attribute[1]} {attribute[3]}".casefold()
        return sum(1 for term in terms if term and term in haystack)

    ranked = sorted(attributes, key=lambda a: (-score(a), bool(_LANGUAGE_VARIANT.match(a[3])), not a[3].casefold().startswith("english")))
    chosen: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for attribute in ranked:
        if attribute[1] not in seen:
            chosen.append(attribute)
            seen.add(attribute[1])
    chosen.extend(a for a in ranked if a not in chosen)
    return chosen


def _order_column(schema: SourceSchema, table: str) -> str | None:
    """The column whose distinct values are the things counted: the table's own key (payment_id on payments) before any order-like column."""
    bare = table.rsplit(".", 1)[-1].casefold()
    for column in schema.tables[table]:
        stem = re.sub(r"^id_|[_ ]?(?:id|key)$", "", column.casefold())
        if _is_key(column) and stem and bare in {stem, f"{stem}s", f"{stem}es", stem[:-1] + "ies" if stem.endswith("y") else stem} | {f"fact{stem}", f"fact_{stem}", f"fact{stem}s"}:
            return column
    return next((c for c in schema.tables[table] if _ORDER_ID_HINT.search(c)), None)


def _counted(channel: str, word: str) -> str:
    """``reseller sales orders``, but ``invoices`` rather than ``invoices invoices``."""
    return channel if channel == word or channel.endswith(" " + word) or channel.endswith(word) else f"{channel} {word}"


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


def _sql(spec: Mapping[str, Any], *, dialect: str) -> str:
    """Render a lakehouse question spec as T-SQL (few-shots) or DuckDB (execution)."""
    kind = spec["kind"]
    if kind in _EXTENDED_KINDS:
        return _sql_extended(spec, dialect=dialect)
    top = spec.get("top")
    facts: list[Mapping[str, Any]] = spec["facts"]
    by_year = kind in {"total_by_year", "distinct_by_year", "yoy", "combined_total_by_year"}
    grouped = kind in {"top_n", "breakdown"}

    def per_fact(fact: Mapping[str, Any]) -> str:
        select: list[str] = []
        group: list[str] = []
        joins: list[str] = []
        conditions: list[str] = []
        dt = fact["date"]
        time_join = _time_join(dt)
        if time_join:
            joins.append(time_join)
        year_expr = _year_expr(dt, dialect)
        if by_year:
            select.append(f"{year_expr} AS year")
            group.append(year_expr)
        ref = "f"
        if grouped or kind == "filter":
            attr = fact["attribute"]
            if attr.get("fact_key"):  # a grouping column on a dimension; on the fact itself it needs no join
                joins.append(f"JOIN {attr['dim_table']} a ON f.{_q(attr['fact_key'])} = a.{_q(attr['dim_key'])}")
                ref = "a"
        if grouped:
            attr = fact["attribute"]
            select.append(f"{ref}.{_q(attr['column'])} AS {_q(attr['alias'])}")
            group.append(f"{ref}.{_q(attr['column'])}")
        if kind in {"distinct_by_year", "distinct_orders_year"}:
            select.append(f"COUNT(DISTINCT f.{_q(fact['order_column'])}) AS value")
        elif kind == "rowcount_year":
            select.append("COUNT(*) AS value")
        elif kind == "definition":
            select.append(f"{spec['expr']} AS value")
        else:
            select.append(f"SUM(f.{_q(fact['measure'])}) AS value")
        if by_year and spec.get("years"):
            conditions.append(f"{year_expr} IN ({', '.join(str(int(y)) for y in spec['years'])})")
        elif spec.get("year") is not None:
            conditions.append(f"{year_expr} = {int(spec['year'])}")
        if kind == "filter":
            conditions.append(f"{ref}.{_q(fact['attribute']['column'])} = {_sql_literal(spec['value'])}")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        group_by = f" GROUP BY {', '.join(group)}" if group else ""
        return f"SELECT {', '.join(select)} FROM {fact['table']} f {' '.join(joins)}{where}{group_by}"

    order = "ORDER BY value DESC" if grouped else ("ORDER BY year" if by_year else "")
    if len(facts) == 1:
        body = per_fact(facts[0])
    else:
        union = " UNION ALL ".join(f"({per_fact(fact)})" for fact in facts)
        keys = "year" if by_year else (facts[0]["attribute"]["alias"] if grouped else None)
        body = f"SELECT {keys}, SUM(value) AS value FROM ({union}) u GROUP BY {keys}" if keys else f"SELECT SUM(value) AS value FROM ({union}) u"
    if kind == "top_n" and top:
        if dialect == "tsql":
            return body.replace("SELECT ", f"SELECT TOP {int(top)} ", 1) + f" {order}"
        return f"{body} {order} LIMIT {int(top)}"
    return f"{body} {order}".rstrip()


def _dax(spec: Mapping[str, Any]) -> str:
    """A DAX rendering of a semantic-model question spec, for the report."""
    measure = spec["measure"]
    groupby = spec.get("groupby") or []
    filters = spec.get("filters") or {}
    columns = ", ".join(groupby)
    measure_expr = f'"{measure}", [{measure}]'
    if filters:
        parts = ", ".join(f"TREATAS({{{', '.join(repr(v) if isinstance(v, str) else str(v) for v in values)}}}, {column})" for column, values in filters.items())
        body = f"SUMMARIZECOLUMNS({columns}, {parts}, {measure_expr})" if columns else f"CALCULATE([{measure}], {parts})"
    else:
        body = f"SUMMARIZECOLUMNS({columns}, {measure_expr})" if columns else f"ROW({measure_expr})"
    if spec.get("top"):
        return f"EVALUATE TOPN({int(spec['top'])}, {body}, [{measure}], DESC)"
    return f"EVALUATE {body}"


_QUERY_SHAPED = re.compile(r"^\s*(SELECT|WITH|EVALUATE|DEFINE)\b", re.IGNORECASE)


def _supplied_questions(context: ReviewContext | None, source_id: str) -> list[Question]:
    """The reviewer's own evaluation cases as questions; a query is executed, prose figures are the reference."""
    if context is None:
        return []
    supplied: list[Question] = []
    for number, (text, expected) in enumerate(context.questions, start=1):
        expected_text = str(expected or "").strip()
        if _QUERY_SHAPED.match(expected_text):
            execution: dict[str, Any] = {"kind": "supplied_sql", "sql": expected_text}
            reference_query = expected_text
        else:
            execution = {"kind": "supplied_text", "text": expected_text}
            reference_query = ""
        supplied.append(Question(id=f"{source_id}.u{number}", source_id=source_id, kind="supplied", text=str(text), spec={"kind": "supplied", "expected": expected_text}, reference_query=reference_query, execution=execution, skill="supplied"))
    return supplied


def _priority_score(question: Question, terms: Sequence[str], emphasised: Sequence[str] = ()) -> int:
    haystack = f"{question.text} {json.dumps(question.spec, default=str)}".casefold()
    return sum(1 for term in terms if term and term in haystack) + sum(1 for term in emphasised if term and term in haystack)


def _prioritised(generated: Sequence[Question], context: ReviewContext | None, limit: int) -> list[Question]:
    """The questions to keep under the limit: every skill represented, the reviewer's terms first within each.

    Questions are scored by the terms they mention (emphasised terms count
    double), grouped by skill, and taken round-robin across the skills in
    score order, so a small limit still covers the matrix. The kept
    questions are renumbered in the order they are asked.
    """
    terms = context.ranking_terms if context is not None else ()
    emphasised = context.emphasised if context is not None and not context.priorities else ()
    ranked = sorted(enumerate(generated), key=lambda item: (-_priority_score(item[1], terms, emphasised) if terms else 0, item[0]))
    groups: dict[str, list[Question]] = {}
    for _index, question in ranked:
        groups.setdefault(question.skill or question.kind, []).append(question)
    chosen: list[Question] = []
    while len(chosen) < limit and any(groups.values()):
        for skill in list(groups):
            if groups[skill]:
                chosen.append(groups[skill].pop(0))
                if len(chosen) >= limit:
                    break
    return [replace(question, id=re.sub(r"\.q\d+$", f".q{number}", question.id)) for number, question in enumerate(chosen, start=1)]


def _with_table_prefix(sql: str, schema: SourceSchema, prefix: str | None) -> str:
    """Table names with the schema prefix the agent uses (``dbo.``), for few-shots that read like its own SQL."""
    if not prefix:
        return sql
    for table in schema.tables:
        sql = re.sub(rf"(?<![\w.]){re.escape(table)}(?![\w])", f"{prefix}.{table}", sql)
    return sql


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
_FORMULA_WORDS = frozenset({"sum", "count", "distinct", "avg", "min", "max", "nullif", "coalesce", "case", "when", "then", "else", "end", "and", "or", "not", "as", "cast", "float", "decimal", "int", "bigint", "double", "round", "abs"})
_SKILLS = {
    "total_by_year": "aggregate",
    "combined_total_by_year": "aggregate",
    "breakdown": "aggregate",
    "top_n": "rank",
    "yoy": "change",
    "distinct_by_year": "count",
    "distinct_orders_year": "count",
    "filter": "filter",
    "definition": "kpi",
    "total_year": "measure",
    "ambiguous_total": "ambiguity",
    "abbreviation": "ambiguity",
    "scope_out": "scope",
    "supplied": "supplied",
    "deep": "proposed",
    "month_trend": "trend",
    "quarter_trend": "trend",
    "drivers": "drivers",
    "entity_trend": "entity",
    "entity_value": "entity",
    "fuzzy_value": "fuzzy",
    "share": "share",
    "top_share": "share",
    "having_count": "threshold",
    "anti_join": "churn",
    "compare": "compare",
    "best_per_group": "leaders",
    "same_month_prior_year": "period",
    "month_vs_previous": "period",
    "relative_month": "relative period",
    "trailing_days": "relative period",
    "holiday_week": "holiday",
    "season": "season",
    "ytd": "year to date",
    "quarter_value": "quarter",
    "partial_year_rank": "instructions",
    "pii_probe": "instructions",
}


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


_MONTH_COLUMN = re.compile(r"^(month|monthnumber|monthnumberofyear|calendarmonth|month_number|monthofyear)$", re.IGNORECASE)
_QUARTER_COLUMN = re.compile(r"^(quarter|calendarquarter|quarter_number|quarterofyear)$", re.IGNORECASE)
_ENTITY_HINT = re.compile(r"(reseller|customer|vendor|supplier|account|store|client|dealer|partner|employee|company|organization|franchise|merchant|brand|manufacturer|seller)[_ ]?name$|^name$", re.IGNORECASE)
_PRODUCT_HINT = re.compile(r"(product|item|sku)[_ ]?name$|^(product|item|sku|plan|plan_name|feature)$", re.IGNORECASE)
_CATEGORY_HINT = re.compile(r"(category|subcategory|segment|class|group|tier|sector|industry|module|department|family)", re.IGNORECASE)
_PLACE_HINT = re.compile(r"(territory|region|country|city|state|geography|district|continent|market|area|zone|location|province)", re.IGNORECASE)


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


def _date_range_condition(fact: Mapping[str, Any], start: str, end: str, dialect: str) -> str:
    """``start`` and ``end`` (ISO dates, inclusive) on the time axis: a yyyymmdd key, a date column, or a timestamp."""
    dt = fact["date"]
    if dt.get("period"):
        raise ValueError("a period written as text has no day grain")
    if dt.get("timestamp"):
        if dt.get("stored") == "yyyymmdd" and not dt.get("date_table"):
            return f"{_time_column(dt)} BETWEEN {start.replace('-', '')} AND {end.replace('-', '')}"
        stamp = _stamp(dt, dialect)
        if dialect == "tsql":
            return f"CAST({stamp} AS DATE) BETWEEN '{start}' AND '{end}'"
        return f"CAST({stamp} AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'"
    column = f"f.{_q(dt['column'])}"
    if str(dt["column"]).casefold().endswith("key"):
        return f"{column} BETWEEN {start.replace('-', '')} AND {end.replace('-', '')}"
    return f"CAST({column} AS DATE) BETWEEN '{start}' AND '{end}'"


def _periods_condition(fact: Mapping[str, Any], periods: Sequence[Mapping[str, Any]], dialect: str) -> str:
    """OR of period conditions: each a year with an optional month or quarter, or a start and end date."""
    parts = []
    for period in periods:
        if "start" in period:
            parts.append(_date_range_condition(fact, str(period["start"]), str(period["end"]), dialect))
        else:
            parts.append(_period_condition(fact, period, dialect))
    return " OR ".join(f"({p})" for p in parts) if len(parts) > 1 else parts[0]


def _sql_extended(spec: Mapping[str, Any], *, dialect: str) -> str:
    """Render the analytical kinds: trends by month or quarter, drivers of a change, entities, shares, thresholds, churn, comparisons, leaders."""
    kind = spec["kind"]
    fact = spec["facts"][0]
    dt = fact["date"]
    measure = f"SUM(f.{_q(fact['measure'])})"
    joins = _Joins()
    base_joins = _time_join(dt) or ""
    year_expr = _year_expr(dt, dialect)
    conditions: list[str] = []
    if spec.get("years"):
        conditions.append(f"{year_expr} IN ({', '.join(str(int(y)) for y in spec['years'])})")
    elif spec.get("year") is not None:
        conditions.append(f"{year_expr} = {int(spec['year'])}")
    for item in spec.get("filters") or ():
        conditions.append(f"{joins.ref(item['attr'])} = {_sql_literal(item['value'])}")
    top = int(spec.get("top") or 10)

    def limit(sql: str, n: int) -> str:
        return sql.replace("SELECT ", f"SELECT TOP {n} ", 1) if dialect == "tsql" else f"{sql} LIMIT {n}"

    def where(extra: Sequence[str] = ()) -> str:
        all_conditions = [*conditions, *extra]
        return f" WHERE {' AND '.join(all_conditions)}" if all_conditions else ""

    def from_clause() -> str:
        return f"FROM {fact['table']} f" + (f" {base_joins}" if base_joins else "") + "".join(" " + c for c in joins.clauses)

    if kind in {"month_series", "month_trend", "quarter_trend"}:
        unit = _month_expr(dt, dialect) if kind != "quarter_trend" else _quarter_expr(dt, dialect)
        if unit is None:
            raise ValueError(f"{kind} needs a month or quarter on the time axis")
        label = "month" if kind != "quarter_trend" else "quarter"
        select = f"{year_expr} AS year, {unit} AS {label}, {measure} AS value" if kind == "month_series" else f"{unit} AS {label}, {measure} AS value"
        group = f"{year_expr}, {unit}" if kind == "month_series" else unit
        order = f"year, {label}" if kind == "month_series" else label
        return f"SELECT {select} {from_clause()}{where()} GROUP BY {group} ORDER BY {order}"
    if kind == "drivers":
        label = joins.ref(spec["attr"])
        before, after = _period_condition(fact, spec["before"], dialect), _period_condition(fact, spec["after"], dialect)
        select = f"{label} AS label, SUM(CASE WHEN {after} THEN f.{_q(fact['measure'])} ELSE 0 END) - SUM(CASE WHEN {before} THEN f.{_q(fact['measure'])} ELSE 0 END) AS value"
        order = "value ASC" if spec.get("direction", "drop") == "drop" else "value DESC"
        return limit(f"SELECT {select} {from_clause()}{where([f'(({before}) OR ({after}))'])} GROUP BY {label} ORDER BY {order}", top)
    if kind == "range_value":
        return f"SELECT {measure} AS value {from_clause()}{where([_periods_condition(fact, spec['periods'], dialect)])}"
    if kind == "two_periods":
        selects = []
        for label, period in spec["periods"]:
            selects.append(f"SELECT {_sql_literal(label)} AS period, {measure} AS value {from_clause()}{where([_periods_condition(fact, [period], dialect)])}")
        return " UNION ALL ".join(selects)
    if kind == "entity_trend":
        return f"SELECT {year_expr} AS year, {measure} AS value {from_clause()}{where()} GROUP BY {year_expr} ORDER BY year"
    if kind == "entity_value":
        return f"SELECT {measure} AS value {from_clause()}{where()}"
    if kind == "share":
        part = joins.ref(spec["attr"])
        return f"SELECT 100.0 * SUM(CASE WHEN {part} = {_sql_literal(spec['value'])} THEN f.{_q(fact['measure'])} ELSE 0 END) / {measure} AS value {from_clause()}{where()}"
    if kind == "top_share":
        label = joins.ref(spec["attr"])
        inner = limit(f"SELECT {measure} AS value {from_clause()}{where()} GROUP BY {label} ORDER BY value DESC", top)
        total = f"SELECT {measure} FROM {fact['table']} f" + (f" {base_joins}" if base_joins else "") + where()
        return f"SELECT 100.0 * (SELECT SUM(value) FROM ({inner}) t) / ({total}) AS value"
    if kind == "top_labels":
        label = joins.ref(spec["attr"])
        return limit(f"SELECT {label} AS label, {measure} AS value {from_clause()}{where()} GROUP BY {label} ORDER BY value DESC", top)
    if kind == "entity_orders":
        label = joins.ref(spec["attr"])
        return limit(f"SELECT {label} AS label, COUNT(DISTINCT f.{_q(fact['order_column'])}) AS value {from_clause()}{where()} GROUP BY {label} ORDER BY value DESC", top)
    if kind == "having_count":
        label = joins.ref(spec["attr"])
        inner = f"SELECT {label} AS label, COUNT(DISTINCT f.{_q(fact['order_column'])}) AS orders {from_clause()}{where()} GROUP BY {label} HAVING COUNT(DISTINCT f.{_q(fact['order_column'])}) > {int(spec['threshold'])}"
        return f"SELECT COUNT(*) AS value FROM ({inner}) t"
    if kind == "anti_join":
        label = joins.ref(spec["attr"])
        present = f"SELECT DISTINCT {label} {from_clause()} WHERE {year_expr} = {int(spec['other_year'])}"
        return f"SELECT DISTINCT {label} AS label {from_clause()} WHERE {year_expr} = {int(spec['year'])} AND {label} NOT IN ({present}) ORDER BY label"
    if kind == "compare":
        label = joins.ref(spec["attr"])
        values = ", ".join(_sql_literal(v) for v in spec["values"])
        return f"SELECT {label} AS label, {measure} AS value {from_clause()}{where([f'{label} IN ({values})'])} GROUP BY {label} ORDER BY value DESC"
    if kind == "best_per_group":
        group_ref, label_ref = joins.ref(spec["group"]), joins.ref(spec["attr"])
        inner = f"SELECT {label_ref} AS label, {group_ref} AS group_label, {measure} AS value, ROW_NUMBER() OVER (PARTITION BY {group_ref} ORDER BY {measure} DESC) AS rn {from_clause()}{where()} GROUP BY {group_ref}, {label_ref}"
        return f"SELECT label, group_label, value FROM ({inner}) t WHERE rn = 1 ORDER BY value DESC"
    raise ValueError(f"unknown question kind {kind!r}")


_EXTENDED_KINDS = frozenset({"month_series", "month_trend", "quarter_trend", "drivers", "entity_trend", "entity_value", "share", "top_share", "top_labels", "entity_orders", "having_count", "anti_join", "compare", "best_per_group", "range_value", "two_periods"})


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


def discover_drivers(executor: Any, schema: SourceSchema, snapshot: AgentSnapshot, years: Sequence[int], context: ReviewContext | None = None, *, facts: int = 2) -> dict[str, dict[str, Any]]:
    """Real names and periods from the data, per fact table, for the driver-based questions.

    A handful of small queries per fact: the top names for each role (who,
    what, category, where) in the latest complete year, the month series
    over the last two complete years (for the largest drop and rise between
    consecutive months), and the order counts per entity (for a threshold).
    Failures are recorded, never raised.
    """
    source = next((s for s in snapshot.datasources if s.id == schema.source_id), None)
    instructions = (source.instructions if source else "") + "\n" + snapshot.instructions + ("\n" + context.text if context is not None else "")
    joins = dict(_heuristic_joins(schema))
    joins.update(_joins_from_instructions(instructions, schema))
    excluded = excluded_terms(instructions)
    terms = context.terms if context is not None else ()
    found: dict[str, dict[str, Any]] = {}
    if not years:
        return found
    latest = max(years)
    for table in _scoped_facts(schema, instructions)[:facts]:
        date = _date_join(schema, table, joins)
        measures = _measure_columns(schema, table)
        if not date or not measures:
            continue
        fact = {"table": table, "measure": measures[0], "date": date, "order_column": _order_column(schema, table)}
        roles = _paths_by_role(attribute_paths(schema, table, joins, excluded), terms)
        entry: dict[str, Any] = {"fact": fact, "roles": roles, "top": {}, "drop": None, "rise": None, "threshold": None, "errors": []}

        def run(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
            try:
                return executor.run({"kind": "sql", "sql": _sql_extended(spec, dialect="duckdb")})
            except Exception as exc:  # noqa: BLE001 - discovery is best effort
                entry["errors"].append(f"{spec['kind']}: {type(exc).__name__}: {str(exc)[:160]}")
                return []

        for role, path in roles.items():
            rows = run({"kind": "top_labels", "facts": [fact], "attr": path, "year": latest, "top": 3})
            labels = [str(r["label"]) for r in rows if r.get("label") is not None]
            if labels:
                entry["top"][role] = labels
        series: list[tuple[int, int, float]] = []
        if _has_month(date):
            span = [y for y in years if y >= latest - 1]
            rows = run({"kind": "month_series", "facts": [fact], "years": span})
            series = [(int(r["year"]), int(r["month"]), float(r["value"])) for r in rows if r.get("value") is not None]
            best_drop, best_rise = None, None
            for (y1, m1, v1), (y2, m2, v2) in zip(series, series[1:]):
                consecutive = (y2 == y1 and m2 == m1 + 1) or (y2 == y1 + 1 and m1 == 12 and m2 == 1)
                if not consecutive or not v1:
                    continue
                change = {"before": {"year": y1, "month": m1}, "after": {"year": y2, "month": m2}, "before_value": v1, "after_value": v2, "delta": v2 - v1}
                if v2 < v1 and (best_drop is None or change["delta"] < best_drop["delta"]):
                    best_drop = change
                if v2 > v1 and (best_rise is None or change["delta"] > best_rise["delta"]):
                    best_rise = change
            entry["drop"], entry["rise"] = best_drop, best_rise
        try:
            rows = executor.run({"kind": "sql", "sql": _max_date_sql(fact)})
            entry["max_date"] = _iso_date(rows[0]["value"]) if rows and rows[0].get("value") is not None else None
        except Exception as exc:  # noqa: BLE001
            entry["errors"].append(f"max_date: {type(exc).__name__}: {str(exc)[:160]}")
            entry["max_date"] = None
        if _has_month(date):
            # the latest month of the latest complete year that also has data in the year before
            series_months = {(y, m) for y, m, _v in series}
            entry["prior_year_month"] = next((m for m in range(12, 0, -1) if (latest, m) in series_months and (latest - 1, m) in series_months), None)
        if fact["order_column"] and "entity" in roles:
            rows = run({"kind": "entity_orders", "facts": [fact], "attr": roles["entity"], "year": latest, "top": 20})
            counts = sorted((int(r["value"]) for r in rows if r.get("value") is not None), reverse=True)
            if len(counts) >= 3:
                tenth = counts[min(9, len(counts) - 1)]
                entry["threshold"] = max(1, tenth - 1) if tenth > 1 else None
        found[table] = entry
    return found


class _HintBudget(Exception):
    """The query budget for the hints is spent."""


def discover_hints(
    executor: Any,
    schema: SourceSchema,
    snapshot: AgentSnapshot,
    years: Sequence[int] | None = None,
    *,
    discovered: Mapping[str, Mapping[str, Any]] | None = None,
    context: ReviewContext | None = None,
    facts: int = 2,
    attributes: int = 6,
    budget: int = 40,
) -> list[Finding]:
    """Hints from the data itself, as findings with basis ``"data"``: what neither the agent's author nor the agent would think to state.

    Referential gaps (fact keys with no match, dimension keys that repeat),
    values spelled in several cases or with padding (a lakehouse compares
    case-sensitively), the vocabulary of small attributes (users say
    "bikes" for the Bikes category), several date columns on a fact, partial
    years and coverage, negative or missing measures, snowflaked join paths,
    personal data on joined tables, a measure name shared by several facts.
    Each finding's ``suggestion`` is the instruction line to add; the
    reviewer decides. At most ``budget`` queries run; the hints that need
    none always come. A single failed probe is skipped, never raised.
    """
    source = next((s for s in snapshot.datasources if s.id == schema.source_id), None)
    instructions = (source.instructions if source else "") + "\n" + snapshot.instructions + ("\n" + context.text if context is not None else "")
    joins = dict(_heuristic_joins(schema))
    joins.update(_joins_from_instructions(instructions, schema))
    excluded = excluded_terms(instructions)
    terms = context.terms if context is not None else ()
    date_tables = {t for t in schema.tables if re.search(r"(date|calendar)", t, re.IGNORECASE)}
    all_facts = _fact_tables(schema)
    fact_list = _scoped_facts(schema, instructions)[:facts]
    hints: list[Finding] = []
    seen: set[tuple[str, str, str]] = set()
    spent = 0

    def hint(code: str, severity: str, table: str, column: str, message: str, suggestion: str, *evidence: str) -> None:
        key = (code, table, column)
        if key in seen:
            return
        seen.add(key)
        hints.append(Finding(code, severity, message, schema.source_id, tuple(e[:300] for e in evidence), suggestion, basis="data"))

    def run(sql: str) -> list[dict[str, Any]]:
        nonlocal spent
        if spent >= budget:
            raise _HintBudget
        spent += 1
        return executor.run({"kind": "sql", "sql": sql})

    def attempt(sql: str) -> list[dict[str, Any]]:
        try:
            return run(sql)
        except _HintBudget:
            raise
        except Exception:  # noqa: BLE001 - one failed probe is not a finding
            return []

    plans: list[tuple[str, Mapping[str, Any] | None, list[str], list[dict[str, Any]], dict[str, Mapping[str, Any]]]] = []
    for table in fact_list:
        date = _date_join(schema, table, joins)
        measures = _measure_columns(schema, table)
        paths = attribute_paths(schema, table, joins, excluded)
        roles = _paths_by_role(paths, terms)
        plans.append((table, date, measures, paths, roles))
        columns = schema.tables[table]
        # hints that need no query
        time_columns = [c for c in columns if _is_time_column(schema, table, c)]
        if date and len(time_columns) > 1:
            others = [c for c in time_columns if c != date["column"]]
            hint("date_choice", "info", table, date["column"], f"{table} has {len(time_columns)} date columns ({', '.join(time_columns[:5])}); the review dates a row by {date['column']}.", f"A {humanize_table(table)} row is dated by {date['column']}; use {', '.join(others[:3])} only when the user asks about that date.")
        for column in measures[:2]:
            others = [t for t in all_facts if t != table and column in schema.tables[t]]
            if others:
                hint("shared_measure", "info", table, column, f"{column} exists in {table} and {', '.join(others)}; a total of {humanize_column(column)} is ambiguous between them.", f"'{humanize_column(column).capitalize()}' on its own means {humanize_table(table)}; say '{humanize_table(others[0])}' for {others[0]}, or combine them when the user asks for the total across both.")
        for path in roles.values():
            if len(path.get("hops") or ()) >= 2:
                chain = " -> ".join([table] + [str(h["table"]) for h in path["hops"]])
                hint("join_path", "info", table, str(path["column"]), f"{path['column']} is {len(path['hops'])} joins away from {table}: {chain}.", f"To group {humanize_table(table)} by {humanize_column(str(path['column']))}, join " + ", then ".join(f"{h['from_column']} to {h['table']}.{h['key']}" for h in path["hops"]) + ".")
        for dim in [table] + sorted({str(h["table"]) for p in paths for h in (p.get("hops") or ())}):
            personal = [c for c in schema.tables[dim] if _PERSONAL_HINT.search(c)]
            if personal:
                hint("personal_data", "low", dim, "", f"{dim} carries personal data: {', '.join(personal[:6])}.", f"Do not return {', '.join(personal[:6])} from {dim}; answer with counts and totals instead.")
        if date and not date.get("year") and not date.get("period"):
            stamp_table = str(date.get("date_table") or table)
            stamp = str(date.get("timestamp_column") or date["column"])
            if _TEXT_TYPE.search(schema.column_type(stamp_table, stamp)):
                hint("date_as_text", "low", stamp_table, stamp, f"{stamp_table}.{stamp} is stored as text, not as a date.", f"{stamp} is text; cast it to a date before filtering or grouping by period.")
    # hints that ask the data
    checked_dims: set[str] = set()
    checked_attributes: set[tuple[str, str]] = set()
    vocabularies = 0
    try:
        for table, date, measures, paths, roles in plans:
            columns = schema.tables[table]
            if date:
                year_expr = _year_expr(date, "duckdb")
                month_expr = _month_expr(date, "duckdb")
                join = _time_join(date)
                sql = f"SELECT MIN({year_expr}) AS first_year, MAX({year_expr}) AS last_year" + (f", MAX(CAST({year_expr} AS BIGINT) * 100 + CAST({month_expr} AS BIGINT)) AS last_month" if month_expr else "") + f" FROM {table} f" + (f" {join}" if join else "")
                rows = attempt(sql)
                if rows and rows[0].get("first_year") is not None and rows[0].get("last_year") is not None:
                    first, last = int(rows[0]["first_year"]), int(rows[0]["last_year"])
                    last_month = int(rows[0]["last_month"]) % 100 if rows[0].get("last_month") is not None else None
                    if last_month and last_month < 12:
                        hint("partial_year", "low", table, str(date["column"]), f"{table} covers {first} to {last}; {last} ends in {calendar.month_name[last_month]}.", f"Data for {last} is partial (through {calendar.month_name[last_month]}); say so when answering about {last}, and do not compare it with a full year.", sql)
                    else:
                        hint("coverage", "info", table, str(date["column"]), f"{table} covers {first} to {last}.", f"Data covers {first} to {last}; say so when a question asks about another period.", sql)
            if measures:
                m = measures[0]
                sql = f"SELECT COUNT(*) AS n, COUNT(*) FILTER (WHERE {_q(m)} IS NULL) AS nulls, COUNT(*) FILTER (WHERE {_q(m)} < 0) AS negatives FROM {table}"
                rows = attempt(sql)
                if rows:
                    n, nulls, negatives = (int(rows[0].get(k) or 0) for k in ("n", "nulls", "negatives"))
                    if negatives:
                        hint("negative_measure", "low", table, m, f"{m} is negative in {negatives:,} of {n:,} rows of {table}.", f"{humanize_column(m).capitalize()} has negative rows (returns or corrections); say whether totals include them.", sql)
                    if nulls:
                        hint("null_measure", "low", table, m, f"{m} is missing in {nulls:,} of {n:,} rows of {table}.", f"{humanize_column(m).capitalize()} is missing on some rows; say whether they are left out of counts and averages.", sql)
            checked = 0
            for column in columns:
                target = joins.get((table, column))
                if target is None or _DATE_KEY.search(column) or target[0] in date_tables or checked >= 6:
                    continue
                checked += 1
                dim, key = target
                sql = f"SELECT COUNT(*) AS n, COUNT(*) FILTER (WHERE NOT EXISTS (SELECT 1 FROM {dim} d WHERE d.{_q(key)} = f.{_q(column)})) AS orphans FROM {table} f"  # no fan-out from a dimension key that repeats
                rows = attempt(sql)
                if rows and int(rows[0].get("orphans") or 0):
                    n, orphans = int(rows[0].get("n") or 0), int(rows[0]["orphans"])
                    pct = 100.0 * orphans / n if n else 0.0
                    hint("join_orphans", "medium", table, column, f"{orphans:,} of {n:,} rows ({pct:.1f}%) of {table} have a {column} with no match in {dim}; an inner join drops them.", f"Join {table}.{column} to {dim}.{key} with a LEFT JOIN, or say that {pct:.1f}% of {humanize_table(table)} rows have no {humanize_table(dim)}.", sql)
                if dim not in checked_dims:
                    checked_dims.add(dim)
                    sql = f"SELECT COUNT(*) - COUNT(DISTINCT {_q(key)}) AS duplicates FROM {dim}"
                    rows = attempt(sql)
                    dup = int(rows[0].get("duplicates") or 0) if rows else 0
                    if dup:
                        hint("duplicate_keys", "medium", dim, key, f"{dim}.{key} repeats {dup:,} time{'s' if dup != 1 else ''}; joining {table} to it multiplies rows.", f"When joining {dim}, keep one row per {key} (say which: the current or the latest one).", sql)
            ordered = list(roles.values()) + [p for p in paths if all(p is not r for r in roles.values())]
            entry = (discovered or {}).get(table) if discovered else None
            top_category = list((entry.get("top") or {}).get("category") or []) if isinstance(entry, Mapping) else []
            count = 0
            for path in ordered:
                dim, column = _path_table(path, table), str(path["column"])
                if (dim, column) in checked_attributes:
                    continue
                if count >= attributes:
                    break
                checked_attributes.add((dim, column))
                count += 1
                q = _q(column)
                sql = f"SELECT COUNT(DISTINCT {q}) AS distinct_values, COUNT(DISTINCT lower(ltrim(rtrim({q})))) AS normalized, COUNT(*) FILTER (WHERE {q} <> ltrim(rtrim({q}))) AS padded FROM {dim}"  # ltrim(rtrim()) rather than trim(): the read-only validator rejects the keyword form
                rows = attempt(sql)
                if not rows:
                    continue
                distinct, normalized, padded = (int(rows[0].get(k) or 0) for k in ("distinct_values", "normalized", "padded"))
                if distinct > normalized:
                    example_sql = f"SELECT string_agg(v, ' | ' ORDER BY v) AS variants FROM (SELECT DISTINCT {q} AS v, lower(ltrim(rtrim({q}))) AS k FROM {dim} WHERE {q} IS NOT NULL) t GROUP BY k HAVING COUNT(*) > 1 ORDER BY k LIMIT 3"
                    examples = [str(r.get("variants")) for r in attempt(example_sql) if r.get("variants")]
                    hint("case_variants", "medium", dim, column, f"{dim}.{column} spells {distinct - normalized} value(s) in more than one case or spacing (for example {'; '.join(examples) or 'see the query'}); lakehouse SQL compares case-sensitively, so a filter on one spelling misses the others.", f"Compare {column} with lower(trim({column})) = lower('<value>'), or normalise the values in {dim}.", sql, example_sql)
                elif padded:
                    hint("padded_values", "low", dim, column, f"{dim}.{column} has {padded:,} value(s) with leading or trailing spaces; an exact filter misses them.", f"Compare {column} with trim({column}), or trim the values in {dim}.", sql)
                named_grouping = bool(_CATEGORY_HINT.search(column) or _PLACE_HINT.search(column) or re.search(r"(status|type|line|group|tier|level|channel|stage)", column, re.IGNORECASE))
                if 0 < distinct <= (30 if named_grouping else 12) and not _ENTITY_HINT.search(column) and not _PRODUCT_HINT.search(column) and vocabularies < 6:
                    values_sql = f"SELECT DISTINCT {q} AS value FROM {dim} WHERE {q} IS NOT NULL ORDER BY 1 LIMIT 30"
                    values = list(dict.fromkeys(str(r.get("value")).strip() for r in attempt(values_sql) if r.get("value") is not None and str(r.get("value")).strip()))
                    if values and any(re.search(r"[A-Za-z]", v) for v in values):  # a vocabulary of numbers helps nobody
                        vocabularies += 1
                        preferred = next((v for v in top_category if v in values and v.lower() != v), None)
                        sample = preferred or next((v for v in values if v.lower() != v and re.search(r"[A-Za-z]", v)), None)
                        example = f" (for example {sample.lower()} means {sample})" if sample else ""
                        hint("vocabulary", "info", dim, column, f"{dim}.{column} has {distinct} values: {', '.join(values)}.", f"{humanize_column(column).capitalize()} values are {', '.join(values)}; match a user's word to them case-insensitively and accept singular or plural{example}.", values_sql)
    except _HintBudget:
        pass
    return hints


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


def _thanksgiving(year: int) -> _dt.date:
    first = _dt.date(year, 11, 1)
    return first + _dt.timedelta(days=(3 - first.weekday()) % 7 + 21)


def _month_end(year: int, month: int) -> _dt.date:
    return _dt.date(year, month, calendar.monthrange(year, month)[1])


def _month_name(period: Mapping[str, Any]) -> str:
    month = period.get("month")
    return f"{calendar.month_name[int(month)]} {int(period['year'])}" if month else str(period["year"])


def _order_word(vocabulary: Vocabulary, column: str | None, *, singular: bool = False) -> str:
    """What the distinct-count column counts: the instructions' word, else the column's own (tickets, transactions, invoices), else orders."""
    word = vocabulary.orders
    if word == "orders" and column:
        lowered = column.casefold()
        word = next((plural for stem, plural in (("ticket", "tickets"), ("transaction", "transactions"), ("invoice", "invoices"), ("payment", "payments"), ("receipt", "receipts"), ("booking", "bookings"), ("visit", "visits")) if stem in lowered), "orders")
    return word[:-1] if singular and word.endswith("s") else word


def _attribute_word(vocabulary: Vocabulary, path: Mapping[str, Any], fact: str) -> str:
    """The business word for a grouping column; a column called plainly ``name`` takes its table's word (``franchise`` for ``sales_franchises.name``)."""
    column = str(path["column"])
    if column.casefold() == "name":
        table = _path_table(path, fact)
        word = humanize_table(table)
        return word[:-1] if word.endswith("s") and len(word) > 3 else word
    return vocabulary.attribute(column)


def _driver_questions(fact_entry: Mapping[str, Any], vocabulary: Vocabulary, schema: SourceSchema, schema_prefix: str | None, years: Sequence[int], top: int, add: Callable[..., None]) -> None:
    """The driver-based and analytical questions for one fact, from what discovery found."""
    fact = fact_entry["fact"]
    roles = fact_entry["roles"]
    tops = fact_entry["top"]
    latest = max(years)
    channel = vocabulary.table(fact["table"])
    cm = _channel_measure(channel, vocabulary.measure(fact["measure"]))
    words = {role: _attribute_word(vocabulary, path, fact["table"]) for role, path in roles.items()}
    base = {"facts": [fact]}
    if _has_month(fact["date"]):
        add("month_trend", f"How did {cm} move month by month in {latest}?", f"SUM({fact['measure']}) in {fact['table']} by month for {latest}", {**base, "kind": "month_trend", "year": latest})
    if _has_quarter(fact["date"]):
        add("quarter_trend", f"Which quarter of {latest} was the strongest for {cm}, and how did the quarters compare?", f"SUM({fact['measure']}) in {fact['table']} by quarter for {latest}", {**base, "kind": "quarter_trend", "year": latest})
    drop = fact_entry.get("drop")
    driver_role = "product" if "product" in roles else ("category" if "category" in roles else None)
    if drop and driver_role:
        add(
            "drivers",
            f"Why did {cm} drop from {_month_name(drop['before'])} to {_month_name(drop['after'])}, and which {_plural(words[driver_role])} drove the drop?",
            f"Change in SUM({fact['measure']}) in {fact['table']} between {_month_name(drop['before'])} and {_month_name(drop['after'])} by {roles[driver_role]['column']}, largest decreases first",
            {**base, "kind": "drivers", "attr": roles[driver_role], "before": drop["before"], "after": drop["after"], "direction": "drop", "top": 5},
        )
    entity = roles.get("entity")
    names = tops.get("entity") or []
    if entity and names:
        first = names[0]
        add("entity_trend", f"How has {first} performed year by year on {cm}?", f"SUM({fact['measure']}) in {fact['table']} by year for {roles['entity']['column']} = {first!r}", {**base, "kind": "entity_trend", "years": list(years), "filters": [{"attr": entity, "value": first}]})
        category_names = tops.get("category") or []
        if "category" in roles and category_names:
            add("entity_value", f"How is {first} doing on {category_names[0]} in {latest}?", f"SUM({fact['measure']}) in {fact['table']} for {roles['entity']['column']} = {first!r} and {roles['category']['column']} = {category_names[0]!r} in {latest}", {**base, "kind": "entity_value", "year": latest, "filters": [{"attr": entity, "value": first}, {"attr": roles["category"], "value": category_names[0]}]})
        add("top_share", f"What share of {latest} {cm} did the top 10 {_plural(words['entity'])} bring in?", f"Share of SUM({fact['measure']}) in {fact['table']} for {latest} from the top 10 {roles['entity']['column']}", {**base, "kind": "top_share", "attr": entity, "year": latest, "top": 10})
        if fact_entry.get("threshold") and fact.get("order_column"):
            add("having_count", f"How many {_plural(words['entity'])} placed more than {fact_entry['threshold']} {_order_word(vocabulary, fact.get('order_column'), singular=fact_entry['threshold'] == 1)} in {latest}?", f"Count of {roles['entity']['column']} with more than {fact_entry['threshold']} distinct {fact['order_column']} in {latest}", {**base, "kind": "having_count", "attr": entity, "year": latest, "threshold": fact_entry["threshold"]})
        if len(years) >= 2:
            add("anti_join", f"Which {_plural(words['entity'])} bought in {years[-2]} but not in {latest}?", f"{roles['entity']['column']} present in {years[-2]} and absent in {latest} in {fact['table']}", {**base, "kind": "anti_join", "attr": entity, "year": years[-2], "other_year": latest})
        if len(names) >= 2:
            add("compare", f"How did {names[0]} and {names[1]} compare on {cm} in {latest}?", f"SUM({fact['measure']}) in {fact['table']} for {roles['entity']['column']} in ({names[0]!r}, {names[1]!r}) in {latest}", {**base, "kind": "compare", "attr": entity, "year": latest, "values": names[:2]})
    category_names = tops.get("category") or []
    if "category" in roles and category_names:
        add("share", f"What share of {latest} {cm} came from {category_names[0]}?", f"Share of SUM({fact['measure']}) in {fact['table']} for {latest} where {roles['category']['column']} = {category_names[0]!r}", {**base, "kind": "share", "attr": roles["category"], "year": latest, "value": category_names[0]})
    if "category" in roles and category_names and category_names[0].lower() != category_names[0]:
        exact = category_names[0]
        add(
            "fuzzy_value",
            f"What was {cm} for {exact.lower()} in {latest}?",
            f"SUM({fact['measure']}) in {fact['table']} for {latest} where {roles['category']['column']} = {exact!r}; the question spells it {exact.lower()!r}, as a user would",
            {**base, "kind": "entity_value", "year": latest, "filters": [{"attr": roles["category"], "value": exact}]},
        )
    _period_questions(fact_entry, cm, years, top, roles, words, add)


def _period_questions(fact_entry: Mapping[str, Any], cm: str, years: Sequence[int], top: int, roles: Mapping[str, Mapping[str, Any]], words: Mapping[str, str], add: Callable[..., None]) -> None:
    """Time expressions a user writes, each with the reading the reference takes and the other readings that are acceptable when stated."""
    fact = fact_entry["fact"]
    latest = max(years)
    base = {"facts": [fact]}
    max_date = _dt.date.fromisoformat(fact_entry["max_date"]) if fact_entry.get("max_date") else None

    def ranged(kind: str, text: str, technical: str, periods: Sequence[Mapping[str, Any]], alternates: Sequence[tuple[str, Sequence[Mapping[str, Any]]]] = ()) -> None:
        if not _has_day(fact["date"]) and any("start" in p for p in periods):
            return  # a period written as text cannot express a range of dates
        spec = {**base, "kind": "range_value", "periods": list(periods)}
        add(kind, text, technical, spec, [(f"period:{label}", {"kind": "sql", "sql": _sql({**base, "kind": "range_value", "periods": list(other)}, dialect="duckdb")}) for label, other in alternates])

    month = _has_month(fact["date"])
    prior = fact_entry.get("prior_year_month")
    if month and prior:
        name = calendar.month_name[prior]
        add(
            "same_month_prior_year",
            f"How did {cm} in {name} {latest} compare with {name} {latest - 1}?",
            f"SUM({fact['measure']}) in {fact['table']} for {name} {latest - 1} and {name} {latest}",
            {**base, "kind": "two_periods", "periods": [(f"{name} {latest - 1}", {"year": latest - 1, "month": prior}), (f"{name} {latest}", {"year": latest, "month": prior})]},
        )
    drop = fact_entry.get("drop")
    if month and drop:
        after, before = drop["after"], drop["before"]
        add(
            "month_vs_previous",
            f"How did {cm} in {_month_name(after)} compare with the month before?",
            f"SUM({fact['measure']}) in {fact['table']} for {_month_name(before)} and {_month_name(after)}",
            {**base, "kind": "two_periods", "periods": [(_month_name(before), before), (_month_name(after), after)]},
        )
    if month and max_date:
        last = {"year": max_date.year, "month": max_date.month}
        previous_date = _dt.date(max_date.year, max_date.month, 1) - _dt.timedelta(days=1)
        previous = {"year": previous_date.year, "month": previous_date.month}
        ranged(
            "relative_month",
            f"What was {cm} last month?",
            f"SUM({fact['measure']}) in {fact['table']} for the last month with data, {_month_name(last)}; {_month_name(previous)} is an acceptable reading",
            [last],
            [(_month_name(previous), [previous])],
        )
    if max_date and max_date >= _dt.date(latest, 12, 5):
        thanksgiving = _thanksgiving(latest)
        week = {"start": (thanksgiving + _dt.timedelta(days=1)).isoformat(), "end": (thanksgiving + _dt.timedelta(days=7)).isoformat()}
        next_week = {"start": (thanksgiving + _dt.timedelta(days=4)).isoformat(), "end": (thanksgiving + _dt.timedelta(days=10)).isoformat()}
        ranged(
            "holiday_week",
            f"What was {cm} in the week after Thanksgiving {latest}?",
            f"SUM({fact['measure']}) in {fact['table']} for {week['start']} to {week['end']} (the seven days after Thanksgiving, {thanksgiving.isoformat()}); the following Monday to Sunday is an acceptable reading",
            [week],
            [(f"the week of {next_week['start']}", [next_week])],
        )
    if max_date:
        trailing = {"start": (max_date - _dt.timedelta(days=29)).isoformat(), "end": max_date.isoformat()}
        year_end = _dt.date(latest, 12, 31)
        alternate = {"start": (year_end - _dt.timedelta(days=29)).isoformat(), "end": year_end.isoformat()}
        ranged(
            "trailing_days",
            f"What was {cm} in the last 30 days of available data?",
            f"SUM({fact['measure']}) in {fact['table']} for {trailing['start']} to {trailing['end']}; the last 30 days of {latest} is an acceptable reading",
            [trailing],
            [(f"the last 30 days of {latest}", [alternate])] if alternate != trailing else [],
        )
    ranged(
        "ytd",
        f"What was {cm} year to date at the end of September {latest}?",
        f"SUM({fact['measure']}) in {fact['table']} for {latest}-01-01 to {latest}-09-30",
        [{"start": f"{latest}-01-01", "end": f"{latest}-09-30"}],
    )
    winter = {"start": f"{latest - 1}-12-01", "end": _month_end(latest, 2).isoformat()}
    calendar_winter = [{"start": f"{latest}-01-01", "end": _month_end(latest, 2).isoformat()}, {"start": f"{latest}-12-01", "end": f"{latest}-12-31"}]
    winter_alternates: list[tuple[str, Sequence[Mapping[str, Any]]]] = [(f"January, February and December {latest}", calendar_winter)]
    if max_date and max_date >= _month_end(latest + 1, 2):
        winter_alternates.append((f"winter {latest}/{str(latest + 1)[-2:]}", [{"start": f"{latest}-12-01", "end": _month_end(latest + 1, 2).isoformat()}]))
    ranged(
        "season",
        f"What was {cm} in the winter of {latest}?",
        f"SUM({fact['measure']}) in {fact['table']} for December {latest - 1} to February {latest}; other readings of winter are acceptable when stated",
        [winter],
        winter_alternates,
    )
    ranged(
        "quarter_value",
        f"What was {cm} in Q4 {latest}?",
        f"SUM({fact['measure']}) in {fact['table']} for {latest}-10-01 to {latest}-12-31",
        [{"start": f"{latest}-10-01", "end": f"{latest}-12-31"}],
    )
    # instruction triggers: a partial year, and personal data
    place = roles.get("place")
    hops = list(place.get("hops") or ()) if place else []
    if max_date and max_date.year > latest and place and len(hops) <= 1:
        partial = max_date.year
        attribute = {"fact_key": hops[0]["from_column"], "dim_table": hops[0]["table"], "dim_key": hops[0]["key"], "column": place["column"], "alias": place["column"]} if hops else {"fact_key": None, "dim_table": fact["table"], "dim_key": None, "column": place["column"], "alias": place["column"]}
        add(
            "partial_year_rank",
            f"Which {top} {_plural(words['place'])} had the highest {cm} in {partial} so far?",
            f"Top {top} {place['column']} by SUM({fact['measure']}) in {fact['table']} for {partial}, a partial year the answer must caveat",
            {**base, "kind": "top_n", "facts": [{**fact, "attribute": attribute}], "year": partial, "top": top},
        )
    if "place" in roles and "category" in roles:
        add("best_per_group", f"For each {words['place']}, which {words['category']} led {cm} in {latest}?", f"Top {roles['category']['column']} by SUM({fact['measure']}) per {roles['place']['column']} in {fact['table']} for {latest}", {**base, "kind": "best_per_group", "group": roles["place"], "attr": roles["category"], "year": latest})


def _channel_measure(channel: str, measure: str) -> str:
    """``internet sales`` with ``sales amount`` reads ``internet sales amount``; with ``revenue`` it reads ``internet sales revenue``."""
    channel_words, measure_words = channel.split(), measure.split()
    if channel_words and measure_words and channel_words[-1] == measure_words[0]:
        return " ".join(channel_words + measure_words[1:])
    return f"{channel} {measure}"


def _period(years: Sequence[int]) -> str:
    return f"in {years[0]}" if len(years) == 1 else f"from {years[0]} to {years[-1]}"


def _definition_expression(body: str, columns: Collection[str]) -> str | None:
    """A definition such as ``SUM(SalesAmount) / COUNT(DISTINCT SalesOrderNumber)`` as a qualified SQL expression, or None."""
    if not _CALCULATION.search(body) or re.search(r"\b(select|from|where)\b", body, re.IGNORECASE):
        return None
    known = {c.casefold(): c for c in columns}
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", body):
        if token.casefold() not in _FORMULA_WORDS and token.casefold() not in known:
            return None
    return re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", lambda m: f"f.{known[m.group(1).casefold()]}" if m.group(1).casefold() in known else m.group(1), body).strip().rstrip(".")


def generate_questions(
    snapshot: AgentSnapshot,
    schemas: Sequence[SourceSchema],
    *,
    years: Mapping[str, Sequence[int]] | None = None,
    top: int = 10,
    limit_per_source: int = 8,
    context: ReviewContext | None = None,
    discovered: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> tuple[Question, ...]:
    """Questions the sources can answer, phrased as a business user asks them, each with the query that answers it.

    ``discovered`` maps a source id to what :func:`discover_drivers` found
    per fact table (real names, the month the measure dropped, an order
    threshold); with it the set adds the driver-based questions: a monthly
    or quarterly trend, why the measure dropped and which products drove it,
    how a named entity performs over time and on a category, shares of the
    total, a count above a threshold, churned entities, a comparison of two
    entities, and the leading category per place.

    The generated set covers what a data agent's author needs to know:
    aggregation by period and by attribute, ranking, change between years,
    distinct counts, a KPI from the instructions' own definitions, the right
    measure column for a term the instructions map (units, cost, freight),
    an ambiguous total across channels, a channel abbreviation, and a topic
    the instructions put out of scope, where the right answer is to decline.
    ``years`` maps a source id to the complete years its data covers (see
    :func:`discover_years`); without it the period questions are skipped.
    ``context`` scopes the facts by the tables its text names, puts the
    questions that mention a priority first, and leads with the reviewer's
    own questions, which do not count against the limit.
    """
    questions: list[Question] = []
    sources = {s.id: s for s in snapshot.datasources}
    supplied_target = next((s.source_id for s in schemas if s.kind != "semantic_model"), schemas[0].source_id if schemas else None)
    # generate everything the schema supports and cut after ranking, so a source with many facts still gets its trend, driver and period questions
    generation_limit = 10_000
    for schema in schemas:
        source = sources.get(schema.source_id)
        source_years = list((years or {}).get(schema.source_id) or [])
        if schema.source_id == supplied_target:
            questions.extend(_supplied_questions(context, schema.source_id))
        start = len(questions)
        if schema.kind == "semantic_model":
            questions.extend(_prioritised(_semantic_questions(schema, source_years, top=top, limit=generation_limit), context, limit_per_source))
            continue
        instructions = (source.instructions if source else "") + "\n" + snapshot.instructions + ("\n" + context.text if context is not None else "")
        joins = dict(_heuristic_joins(schema))
        joins.update(_joins_from_instructions(instructions, schema))
        excluded = excluded_terms(instructions)
        schema_prefix = "dbo" if re.search(r"\bdbo\.", (source.instructions if source else "") + snapshot.instructions) else None
        vocabulary = build_vocabulary(snapshot, schema, context)
        facts = _scoped_facts(schema, instructions)
        per_fact = []
        for table in facts:
            date = _date_join(schema, table, joins)
            measures = _measure_columns(schema, table)
            if not date or not measures:
                continue
            per_fact.append({"table": table, "measure": measures[0], "date": date, "attributes": _attributes(schema, table, joins, excluded), "order_column": _order_column(schema, table)})
        if not per_fact or not source_years:
            continue
        count = 0
        shared = [f for f in per_fact if f["measure"] == per_fact[0]["measure"]]
        latest = max(source_years)
        span_words = _period(source_years)
        span_technical = f"{source_years[0]} to {source_years[-1]}" if len(source_years) > 1 else str(source_years[0])
        prefix = f"{schema.source_id}"

        def add(kind: str, text: str, technical: str, spec: Mapping[str, Any], alternates: Sequence[tuple[str, Mapping[str, Any]]] = ()) -> None:
            nonlocal count
            if count >= generation_limit:
                return
            count += 1
            questions.append(
                Question(
                    id=f"{prefix}.q{count}",
                    source_id=schema.source_id,
                    kind=kind,
                    text=text,
                    spec=spec,
                    reference_query=_with_table_prefix(_sql(spec, dialect="tsql"), schema, schema_prefix),
                    execution={"kind": "sql", "sql": _sql(spec, dialect="duckdb")},
                    alternates=tuple(alternates),
                    skill=_SKILLS.get(kind, kind),
                    technical=technical,
                )
            )

        measure_name = per_fact[0]["measure"]
        measure_word = vocabulary.measure(measure_name)
        if len(shared) > 1:
            bare = [{k: v for k, v in f.items() if k != "attributes"} for f in shared]
            channels = " and ".join(vocabulary.table(f["table"]) for f in shared)
            spec = {"kind": "combined_total_by_year", "facts": bare, "years": source_years}
            alternates = [(f["table"], {"kind": "sql", "sql": _sql({**spec, "facts": [f]}, dialect="duckdb")}) for f in bare]
            add(
                "combined_total_by_year",
                f"What was total {measure_word} by year {span_words}, {channels} combined?",
                f"What was total {measure_name} by year for {span_technical}, across {' and '.join(f['table'] for f in shared)} combined?",
                spec,
                alternates,
            )
            # the same total with the channel left unsaid: the instructions decide what "total" means
            single = {"kind": "combined_total_by_year", "facts": bare, "years": [latest]}
            add(
                "ambiguous_total",
                f"What was total {measure_word} in {latest}?",
                f"What was total {measure_name} in {latest} across {' and '.join(f['table'] for f in shared)}, channel unspecified?",
                single,
                [(f["table"], {"kind": "sql", "sql": _sql({**single, "facts": [f]}, dialect="duckdb")}) for f in bare],
            )
        for fact in per_fact:
            base = {k: v for k, v in fact.items() if k != "attributes"}
            channel = vocabulary.table(fact["table"])
            measure = vocabulary.measure(fact["measure"])
            cm = _channel_measure(channel, measure)
            add(
                "total_by_year",
                f"What was {cm} by year {span_words}?",
                f"What was total {fact['measure']} in {fact['table']} by year for {span_technical}?",
                {"kind": "total_by_year", "facts": [base], "years": source_years},
            )
            if fact["order_column"]:
                add(
                    "distinct_by_year",
                    f"How many {_counted(channel, _order_word(vocabulary, fact.get('order_column')))} were there per year {span_words}?",
                    f"How many distinct {fact['order_column']} values (orders) does {fact['table']} have per year for {span_technical}?",
                    {"kind": "distinct_by_year", "facts": [base], "years": source_years},
                )
                add(
                    "distinct_orders_year",
                    f"How many {_counted(channel, _order_word(vocabulary, fact.get('order_column')))} were placed in {latest}?",
                    f"How many distinct {fact['order_column']} values does {fact['table']} have for {latest} (not the row count)?",
                    {"kind": "distinct_orders_year", "facts": [base], "year": latest},
                    [("rowcount", {"kind": "sql", "sql": _sql({"kind": "rowcount_year", "facts": [base], "year": latest}, dialect="duckdb")})],
                )
            if len(source_years) >= 2:
                add(
                    "yoy",
                    f"How did {cm} change from {source_years[-2]} to {source_years[-1]}?",
                    f"What was the year-over-year change in total {fact['measure']} in {fact['table']} from {source_years[-2]} to {source_years[-1]}?",
                    {"kind": "yoy", "facts": [base], "years": source_years[-2:]},
                )
            for fact_key, dim_table, dim_key, attribute in _diverse_attributes(fact["attributes"], context.terms if context is not None else ())[:2]:
                attr = {"fact_key": fact_key, "dim_table": dim_table, "dim_key": dim_key, "column": attribute, "alias": attribute}
                word = _attribute_word(vocabulary, {"column": attribute, "hops": [{"from_column": fact_key, "table": dim_table, "key": dim_key}] if fact_key else []}, fact["table"])
                add(
                    "top_n",
                    f"Which {top} {_plural(word)} had the highest {cm} in {latest}?",
                    f"What are the top {top} {attribute} values by {fact['measure']} in {fact['table']} for {latest}?",
                    {"kind": "top_n", "facts": [{**base, "attribute": attr}], "year": latest, "top": top, "phrases": {"cm": cm, "attribute": word}},
                )
                add(
                    "breakdown",
                    f"What was {cm} by {word} in {latest}?",
                    f"What was total {fact['measure']} in {fact['table']} by {attribute} for {latest}?",
                    {"kind": "breakdown", "facts": [{**base, "attribute": attr}], "year": latest},
                )
            # the right measure column for a term the instructions map (units, cost, freight)
            other = next(((column, term) for column, term in vocabulary.measures.items() if column != fact["measure"] and column in schema.tables[fact["table"]] and column.casefold() != fact["measure"].casefold()), None)
            if other:
                column, term = other
                add(
                    "total_year",
                    f"What were total {term} for {channel} in {latest}?",
                    f"What was SUM({column}) in {fact['table']} for {latest}?",
                    {"kind": "total_year", "facts": [{**base, "measure": column}], "year": latest},
                    [(f"measure:{fact['measure']}", {"kind": "sql", "sql": _sql({"kind": "total_year", "facts": [base], "year": latest}, dialect="duckdb")})],
                )
            # a KPI the instructions define as a formula over this fact's columns
            for name, body in extract_definitions(instructions).items():
                expression = _definition_expression(body, schema.tables[fact["table"]])
                if expression and not re.fullmatch(r"(SUM|COUNT|AVG|MIN|MAX)\(\s*(?:DISTINCT\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\)", body.strip().rstrip("."), re.IGNORECASE):  # a single aggregate is a measure word, not a KPI
                    add(
                        "definition",
                        f"What was the {name.casefold()} for {channel} in {latest}?",
                        f"What is {body} over {fact['table']} for {latest}?",
                        {"kind": "definition", "facts": [base], "year": latest, "expr": expression, "term": name},
                    )
                    break
            # a channel abbreviation the instructions define
            abbreviation = vocabulary.abbreviations.get(fact["table"])
            if abbreviation:
                others = [f for f in per_fact if f["table"] != fact["table"]]
                add(
                    "abbreviation",
                    f"What was {abbreviation} {measure} in {latest}?",
                    f"What was total {fact['measure']} in {fact['table']} for {latest} ({abbreviation} means {fact['table']})?",
                    {"kind": "total_year", "facts": [base], "year": latest},
                    [(f"channel:{o['table']}", {"kind": "sql", "sql": _sql({"kind": "total_year", "facts": [{k: v for k, v in o.items() if k != 'attributes'}], "year": latest}, dialect="duckdb")}) for o in others],
                )
        for fact in per_fact:
            entry = ((discovered or {}).get(schema.source_id) or {}).get(fact["table"])
            if entry:
                _driver_questions(entry, vocabulary, schema, schema_prefix, source_years, top, add)
        pii_table = next((t for t, cols in schema.tables.items() if any(_PII_COLUMN.search(c) for c in cols) and not _mentions_excluded(t, excluded)), None)
        if pii_table and any(r.kind == "no_pii" for r in extract_rules(instructions)):
            who = humanize_table(pii_table)
            if count < generation_limit:
                count += 1
                questions.append(Question(id=f"{prefix}.q{count}", source_id=schema.source_id, kind="pii_probe", text=f"Who were the top 5 {_plural(who)} by {vocabulary.measure(measure_name)} in {latest}, and how can we contact them?", spec={"kind": "pii_probe", "table": pii_table}, reference_query="", execution={"kind": "expect_behaviour", "rule": "no_pii"}, skill="instructions", technical=f"A request for contact details from {pii_table}; the instructions forbid listing personal data, so the answer must carry no email, phone or street address."))
        # a topic the instructions put out of scope: the right answer is to decline
        for table in excluded_tables(schema, instructions)[:2]:
            phrase = humanize_table(table)
            text = f"What was the {phrase} for {latest}?" if table.casefold().startswith("fact") else f"Which {_plural(phrase)} were used most in {latest}?"
            if count >= generation_limit:
                break
            count += 1
            questions.append(Question(id=f"{prefix}.q{count}", source_id=schema.source_id, kind="scope_out", text=text, spec={"kind": "scope_out", "table": table, "year": latest}, reference_query="", execution={"kind": "expect_decline"}, skill="scope", technical=f"A question on {table}, which the instructions put out of scope; the agent should decline."))
        generated = questions[start:]
        del questions[start:]
        questions.extend(_prioritised(generated, context, limit_per_source))
    return tuple(questions)


def derive_filter_questions(questions: Sequence[Question], references: Sequence[Reference], *, per_source: int = 1) -> tuple[tuple[Question, ...], tuple[Reference, ...]]:
    """A filtered total per source from the first ranked reference: the top row's label becomes the filter, its value the reference.

    No query runs: the ranking already computed the value. The question
    tests whether the agent applies a literal filter on an attribute.
    """
    added_questions: list[Question] = []
    added_references: list[Reference] = []
    ref_by_id = {r.question_id: r for r in references}
    seen: dict[str, int] = {}
    for question in questions:
        reference = ref_by_id.get(question.id)
        if question.kind != "top_n" or reference is None or reference.status != "ok" or not reference.rows:
            continue
        if seen.get(question.source_id, 0) >= per_source:
            continue
        fact = dict(question.spec["facts"][0])
        attr = fact["attribute"]
        row = reference.rows[0]
        label = row.get(attr["alias"])
        raw = row.get("value")
        if label is None or raw is None or isinstance(raw, (bool, str)):
            continue
        try:
            value = float(raw)  # DuckDB sums a DECIMAL column to a Decimal
        except (TypeError, ValueError):
            continue
        seen[question.source_id] = seen.get(question.source_id, 0) + 1
        phrases = question.spec.get("phrases") or {}
        cm = phrases.get("cm") or humanize_column(fact["measure"])
        word = phrases.get("attribute") or humanize_column(attr["column"])
        spec = {"kind": "filter", "facts": [fact], "year": question.spec["year"], "value": label}
        prefix = "dbo" if "dbo." in question.reference_query else None
        schema_tables = {t: () for t in (fact["table"], attr.get("dim_table"), fact["date"].get("date_table")) if t}
        qid = f"{question.source_id}.f{seen[question.source_id]}"
        added_questions.append(
            Question(
                id=qid,
                source_id=question.source_id,
                kind="filter",
                text=f"What was {cm} for {word} {label} in {question.spec['year']}?",
                spec=spec,
                reference_query=_with_table_prefix(_sql(spec, dialect="tsql"), SourceSchema(question.source_id, "lakehouse", schema_tables), prefix),
                execution={"kind": "supplied_rows", "rows": [{"value": value}]},
                skill="filter",
                technical=f"SUM({fact['measure']}) in {fact['table']} where {attr['column']} = {label!r} for {question.spec['year']}",
            )
        )
        added_references.append(Reference(qid, "ok", ({"value": value},), "from the ranked reference"))
    return tuple(questions) + tuple(added_questions), tuple(references) + tuple(added_references)


def _semantic_questions(schema: SourceSchema, years: Sequence[int], *, top: int, limit: int) -> list[Question]:
    questions: list[Question] = []
    date_table = next((t for t in schema.tables if re.search(r"(date|calendar|period)", t, re.IGNORECASE)), None)
    year_column = next((c for c in schema.tables.get(date_table, ()) if _YEAR_COLUMN.match(c)), None) if date_table else None
    year_ref = f"'{date_table}'[{year_column}]" if date_table and year_column else None
    attributes = [
        f"'{t}'[{c}]"
        for t, columns in schema.tables.items()
        if t != date_table
        for c in columns
        if _ATTRIBUTE_HINT.search(c) and not _is_key(c)
    ][:3]
    count = 0
    for measure in list(schema.measures[:3]):
        if count >= limit:
            break
        if year_ref and years:
            spec = {"kind": "total_by_year", "measure": measure, "groupby": [year_ref], "filters": {year_ref: list(years)}}
            count += 1
            questions.append(Question(id=f"{schema.source_id}.q{count}", source_id=schema.source_id, kind="total_by_year", text=f"What was {measure} by year {_period(list(years))}?", spec=spec, reference_query=_dax(spec), execution={"kind": "aggregate", "measures": [measure], "groupby": [year_ref], "filters": {year_ref: list(years)}}, skill="aggregate", technical=f"What was {measure} by year for {years[0]} to {years[-1]}?"))
        for attribute in attributes[:2]:
            if count >= limit:
                break
            filters = {year_ref: [max(years)]} if year_ref and years else {}
            spec = {"kind": "top_n", "measure": measure, "groupby": [attribute], "filters": filters, "top": top}
            count += 1
            period = f" in {max(years)}" if years else ""
            word = humanize_column(attribute.split("[", 1)[-1])
            questions.append(Question(id=f"{schema.source_id}.q{count}", source_id=schema.source_id, kind="top_n", text=f"Which {top} {_plural(word)} had the highest {measure}{period}?", spec=spec, reference_query=_dax(spec), execution={"kind": "aggregate", "measures": [measure], "groupby": [attribute], "filters": filters, "order_by": measure, "top": top}, skill="rank", technical=f"What are the top {top} {attribute} by {measure}{period}?"))
    return questions


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


class SemanticModelExecutor:
    """Runs semantic-model question specs through ``SemanticModel.aggregate``."""

    def __init__(self, aggregate: Callable[..., Any]) -> None:
        self._aggregate = aggregate

    def run(self, execution: Mapping[str, Any]) -> list[dict[str, Any]]:
        frame = self._aggregate(
            list(execution["measures"]),
            groupby=list(execution.get("groupby") or []) or None,
            filters=dict(execution.get("filters") or {}) or None,
            order_by=execution.get("order_by"),
            top=execution.get("top"),
        )
        return _rows(frame)


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


def discover_years(executor: LakehouseExecutor, schema: SourceSchema, source: AgentDataSource | None = None, agent_instructions: str = "") -> list[int]:
    """The complete calendar years the first in-scope fact table covers."""
    instructions = (source.instructions if source else "") + "\n" + (agent_instructions or "")
    joins = dict(_heuristic_joins(schema))
    joins.update(_joins_from_instructions(instructions, schema))
    for table in _scoped_facts(schema, instructions):
        date = _date_join(schema, table, joins)
        if not date:
            continue
        year = _year_expr(date, "duckdb")
        join = _time_join(date)
        rows = executor.run({"sql": f"SELECT {year} AS year, COUNT(*) AS n FROM {table} f" + (f" {join}" if join else "") + f" GROUP BY {year} ORDER BY {year}"})
        counts = [(int(r["year"]), int(r["n"])) for r in rows if r.get("year") is not None]
        years = [y for y, _ in counts]
        if len(counts) >= 2 and counts[-1][1] < 0.5 * counts[-2][1]:
            # the last year is partial when it holds far fewer rows than the one before
            years = years[:-1]
        return years
    return []


def _source_label(snapshot: AgentSnapshot, source_id: str) -> str:
    source = next((s for s in snapshot.datasources if s.id == source_id), None)
    return (source.name or source.id) if source else source_id


def explain_no_questions(snapshot: AgentSnapshot, schemas: Sequence[SourceSchema], years: Mapping[str, Sequence[int]] | None = None) -> list[str]:
    """Why the generator produced nothing for each source: no fact table, no date join, no measure, no complete years."""
    notes: list[str] = []
    sources = {s.id: s for s in snapshot.datasources}
    for schema in schemas:
        label = _source_label(snapshot, schema.source_id)
        if schema.kind == "semantic_model":
            if not schema.measures:
                notes.append(f"{label}: the model exposes no measures, so no questions were generated")
            continue
        source = sources.get(schema.source_id)
        instructions = (source.instructions if source else "") + "\n" + snapshot.instructions
        joins = dict(_heuristic_joins(schema))
        joins.update(_joins_from_instructions(instructions, schema))
        facts = _scoped_facts(schema, instructions)
        if not facts:
            notes.append(f"{label}: no fact table recognised among {len(schema.tables)} tables (none named in the instructions, none fact-shaped)")
            continue
        for table in facts:
            if not _date_join(schema, table, joins):
                notes.append(f"{label}: {table} has no time axis (no date dimension with a year column, no date or timestamp column on it or on a joined table), so no period questions")
            if not _measure_columns(schema, table):
                notes.append(f"{label}: {table} has no numeric measure column")
        if not (years or {}).get(schema.source_id):
            notes.append(f"{label}: no complete years were discovered, so no questions were generated")
    return notes


def build_references(questions: Sequence[Question], executors: Mapping[str, Any]) -> tuple[Reference, ...]:
    """Execute every question's reference query; a failure is a status, not a guess."""
    references: list[Reference] = []
    for question in questions:
        executor = executors.get(question.source_id)
        if question.execution.get("kind") == "expect_decline":
            references.append(Reference(question.id, "decline", note="the instructions put this topic out of scope; the right answer declines"))
            continue
        if question.execution.get("kind") == "expect_behaviour":
            references.append(Reference(question.id, "behaviour", note=f"graded by the instructions' rule {question.execution.get('rule')}"))
            continue
        if question.execution.get("kind") == "supplied_rows":
            references.append(Reference(question.id, "ok", tuple(dict(r) for r in question.execution.get("rows") or ()), "from the ranked reference"))
            continue
        if question.execution.get("kind") == "supplied_text":
            figures = _reference_figures(str(question.execution.get("text", "")))
            if figures:
                references.append(Reference(question.id, "ok", tuple({"value": figure} for figure in figures), "figures of the supplied answer"))
            else:
                references.append(Reference(question.id, "failed", note="the supplied answer carries no figures; supply a query or a figure"))
            continue
        if executor is None:
            references.append(Reference(question.id, "abstained", note="no executor for the source"))
            continue
        try:
            if question.execution.get("kind") == "supplied_sql":
                sql = str(question.execution["sql"])
                rows = executor.run_agent_sql(sql) if isinstance(executor, LakehouseExecutor) else executor.run({"sql": sql})
            else:
                rows = executor.run(question.execution)
        except Exception as exc:  # noqa: BLE001 - a failed reference is recorded, never invented
            references.append(Reference(question.id, "failed", note=f"{type(exc).__name__}: {exc}"[:200]))
            continue
        alternates: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for label, execution in question.alternates:
            try:
                alternates[label] = tuple(executor.run(execution))
            except Exception:  # noqa: BLE001
                continue
        references.append(Reference(question.id, "ok" if rows else "abstained", tuple(rows), "" if rows else "the query returned no rows", alternates))
    return tuple(references)


# --------------------------------------------------------------------------- #
# Asking the agent
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AgentStep:
    """One tool step of an agent run: the function called, its arguments, its output."""

    kind: str  # function_call | function_call_output | code_interpreter_call | tool_call
    name: str = ""
    arguments: str = ""
    output: str = ""


@dataclass(frozen=True)
class AgentAnswer:
    text: str
    query: str | None = None  # the query the agent executed, when the run steps expose it
    language: str | None = None
    datasource: str | None = None
    seconds: float = 0.0
    steps: tuple[AgentStep, ...] = ()  # the run steps, for the insight section of the report
    status: str = "completed"
    executed: bool = True  # False when the steps show a query generated but no execution


def _executed(steps: Sequence[AgentStep]) -> bool:
    """Whether the run executed a query: an execute step exists, or the steps carry no names to tell."""
    names = [s.name for s in steps if s.name]
    if not names:
        return True
    return any(re.search(r"(execut|run_query|query_execution)", name, re.IGNORECASE) for name in names)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _response_text(items: Sequence[Any]) -> str:
    parts = []
    for item in items:
        if _field(item, "type") != "message":
            continue
        for content in _field(item, "content", []) or []:
            text = _field(content, "text")
            if text:
                parts.append(str(text))
    return "\n\n".join(parts)


def _response_steps(items: Sequence[Any]) -> tuple[AgentStep, ...]:
    """The tool steps of a Responses API output, in order."""
    steps = []
    for item in items:
        kind = str(_field(item, "type", "") or "")
        if kind == "function_call":
            steps.append(AgentStep(kind, str(_field(item, "name", "") or ""), _text(_field(item, "arguments", "")), ""))
        elif kind == "function_call_output":
            steps.append(AgentStep(kind, str(_field(item, "name", "") or ""), "", _text(_field(item, "output", ""))))
        elif kind == "code_interpreter_call":
            steps.append(AgentStep(kind, kind, _text(_field(item, "code") or _field(item, "input", "")), _text(_field(item, "output", ""))))
    return tuple(steps)


_FENCED = re.compile(r"```(sql|dax|kql|kusto)\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_QUERY_START = re.compile(r"^\s*(SELECT|WITH|EVALUATE|DEFINE|DECLARE|let\s|\w+\s*\|)", re.IGNORECASE)
_DATASOURCE_KEYS = ("datasource_name", "datasourceName", "data_source", "dataSource", "datasource", "artifact_name", "artifactName", "source")


def _language_of(query: str, tag: str | None = None) -> str:
    if tag:
        return "kql" if tag.casefold() == "kusto" else tag.casefold()
    if "EVALUATE" in query.upper():
        return "dax"
    if re.search(r"^\s*\w+\s*\|\s*(where|summarize|project|take|count|extend)\b", query, re.IGNORECASE | re.MULTILINE):
        return "kql"
    return "sql"


def _steps_summary(steps: Sequence[AgentStep]) -> tuple[str | None, str | None, str | None]:
    """The executed query, its language and the routed data source, from the run steps.

    A fenced ``sql``/``dax``/``kql`` block in a tool output is the query the
    service ran; failing that, a query-shaped value in the call arguments or
    the output. The data source comes from the call arguments.
    """
    query: str | None = None
    language: str | None = None
    datasource: str | None = None
    for step in steps:
        candidate: tuple[str, str] | None = None
        fenced = _FENCED.findall(step.output)
        if fenced:
            tag, body = fenced[-1]
            candidate = (body.strip(), _language_of(body, tag))
        else:
            for text in (step.arguments, step.output):
                found = _query_in(text)
                if found:
                    candidate = (found, _language_of(found))
                    break
        if candidate:
            query, language = candidate
        for key in _DATASOURCE_KEYS:
            match = re.search(rf'"{key}"\s*:\s*"([^"]+)"', step.arguments)
            if match:
                datasource = match.group(1)
                break
    return query, language, datasource


class McpAgentAsker:
    """Asks through the agent's MCP endpoint; returns the answer text only.

    Inside a Fabric notebook the token provider is
    ``lambda: notebookutils.credentials.getToken("pbi")``; the endpoint
    serves the published agent.
    """

    def __init__(self, workspace_id: str, agent_id: str, token_provider: Callable[[], str]) -> None:
        self.url = f"{FABRIC_API}/mcp/workspaces/{workspace_id}/dataagents/{agent_id}/agent"
        self._token = token_provider

    def __call__(self, question: str) -> AgentAnswer:
        import asyncio

        from mcp.client.session import ClientSession

        async def ask(streams: Any) -> str:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool = tools.tools[0]
                schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
                key = next(iter(schema.get("properties", {"userQuestion": 1})))
                result = await session.call_tool(tool.name, {key: question})
                return " ".join(getattr(c, "text", "") for c in result.content)

        async def run() -> str:
            headers = {"Authorization": f"Bearer {self._token()}"}
            try:
                from mcp.client.streamable_http import streamable_http_client as connect
            except ImportError:  # mcp releases before 1.24 spell it differently
                from mcp.client.streamable_http import streamablehttp_client as connect_legacy

                async with connect_legacy(self.url, headers=headers) as streams:
                    return await ask(streams)
            import httpx

            async with httpx.AsyncClient(headers=headers, timeout=240.0) as client:
                async with connect(self.url, http_client=client) as streams:
                    return await ask(streams)

        started = time.time()
        return AgentAnswer(text=asyncio.run(run()), seconds=round(time.time() - started, 1))


class ResponsesAgentAsker:
    """Asks through the agent's OpenAI-compatible Responses endpoint.

    Inside a notebook pass the SDK's ``FabricOpenAIResponses`` client (SDK
    0.1.28a0 and later; ``ai_skill_stage="sandbox"`` asks the draft
    configuration, ``"production"`` the published one). ``responses.create``
    returns the answer together with the output items: the function calls
    and their outputs, which carry the query the agent executed and the
    source it routed to. Any client with ``responses.create`` and
    ``responses.retrieve`` works.
    """

    TERMINAL = frozenset({"completed", "failed", "incomplete", "cancelled"})

    def __init__(self, client: Any, *, model: str | None = None) -> None:
        self.client = client
        self.model = model

    def __call__(self, question: str, *, timeout: float = 600.0, poll_seconds: float = 2.0) -> AgentAnswer:
        started = time.time()
        kwargs: dict[str, Any] = {"input": question}
        if self.model:
            kwargs["model"] = self.model
        response = self.client.responses.create(**kwargs)
        while str(_field(response, "status", "completed") or "completed").lower() not in self.TERMINAL:
            if time.time() - started > timeout:
                raise TimeoutError(f"response {_field(response, 'id', '')} did not finish within {timeout:.0f} s")
            time.sleep(poll_seconds)
            response = self.client.responses.retrieve(_field(response, "id"))
        items = list(_field(response, "output", []) or [])
        text = _response_text(items) or str(_field(response, "output_text", "") or "")
        steps = _response_steps(items)
        query, language, datasource = _steps_summary(steps)
        status = str(_field(response, "status", "completed") or "completed").lower()
        if status != "completed" and not text:
            text = f"ERROR: response ended with status {status}"
        return AgentAnswer(text=text, query=query, language=language, datasource=datasource, seconds=round(time.time() - started, 1), steps=steps, status=status, executed=_executed(steps))


class AssistantsAgentAsker:
    """Asks through the agent's OpenAI-compatible Assistants endpoint.

    The run steps expose the query the agent generated and executed and the
    data source it routed to, so the review can grade the query itself.
    Works outside a notebook with a token for
    ``https://api.fabric.microsoft.com`` and inside one through the SDK's
    ``FabricOpenAI`` client (pass it as ``client``). Prefer
    ``ResponsesAgentAsker`` where the SDK offers ``FabricOpenAIResponses``.
    """

    def __init__(self, workspace_id: str | None = None, agent_id: str | None = None, token_provider: Callable[[], str] | None = None, *, client: Any = None, api_version: str = "2024-05-01-preview") -> None:
        if client is None:
            if not (workspace_id and agent_id and token_provider):
                raise ValueError("workspace_id, agent_id and token_provider are required without a client")
            from openai import OpenAI

            token = token_provider()
            client = OpenAI(
                base_url=f"{FABRIC_API}/workspaces/{workspace_id}/dataagents/{agent_id}/aiassistant/openai",
                api_key=token,
                default_headers={"Authorization": f"Bearer {token}"},
                default_query={"api-version": api_version},
            )
        self.client = client
        self._assistant_id: str | None = None

    def _assistant(self) -> str:
        if self._assistant_id is None:
            self._assistant_id = self.client.beta.assistants.create(model="not used").id
        return self._assistant_id

    def __call__(self, question: str, *, timeout: float = 240.0) -> AgentAnswer:
        started = time.time()
        client = self.client
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(thread_id=thread.id, role="user", content=question)
        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=self._assistant())
        while run.status in {"queued", "in_progress", "requires_action"} and time.time() - started < timeout:
            time.sleep(2)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        text = ""
        for message in client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=5).data:
            if message.role == "assistant":
                text = " ".join(getattr(getattr(part, "text", None), "value", "") for part in message.content)
                break
        try:
            raw_steps = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id, order="asc", limit=100).data
        except Exception:  # noqa: BLE001 - steps are a bonus
            raw_steps = []
        steps: list[AgentStep] = []
        for step in raw_steps:
            if getattr(step, "type", "") != "tool_calls" or not getattr(step, "step_details", None):
                continue
            for call in getattr(step.step_details, "tool_calls", []) or []:
                function = getattr(call, "function", None)
                if function is None:
                    continue
                steps.append(AgentStep("tool_call", str(getattr(function, "name", "") or ""), _text(getattr(function, "arguments", "")), _text(getattr(function, "output", ""))))
        query, language, datasource = _steps_summary(steps)
        if run.status != "completed" and not text:
            text = f"ERROR: run ended with status {run.status}"
        return AgentAnswer(text=text, query=query, language=language, datasource=datasource, seconds=round(time.time() - started, 1), steps=tuple(steps), status=str(run.status), executed=_executed(steps))


def _query_in(text: str) -> str | None:
    if not text:
        return None
    def unfenced(value: str) -> str:
        blocks = re.findall(r"```(?:sql|kql|kusto|dax)?\s*(.*?)```", value, re.IGNORECASE | re.DOTALL)
        return blocks[-1].strip() if blocks else value.strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            for key in ("code", "sql", "query", "dax", "kql"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip() and _QUERY_START.search(unfenced(value)):
                    return unfenced(value)
    except ValueError:
        pass
    blocks = re.findall(r"```(?:sql|kql|kusto|dax)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    match = re.search(r"\b(SELECT|WITH|EVALUATE)\b.*", text, re.IGNORECASE | re.DOTALL)
    return match.group(0).strip()[:4000] if match else None


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

_TEXT_NUMBER = re.compile(r"(?<![A-Za-z_\d.])-?\$?\d[\d,]*(?:\.\d+)?")
_ABSTAIN_HINT = re.compile(
    r"(cannot|can't|unable|not available|not in scope|no data|no records|no rows|no results|not found|"
    r"does not contain|doesn't contain|missing|do not have|don't have|not possible)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Graded:
    question_id: str
    outcome: str  # correct | partial | wrong | incomplete | abstained | no_reference
    cause: str = ""
    detail: str = ""
    matched: int = 0
    expected: int = 0
    query_checked: bool = False
    violations: tuple[str, ...] = ()  # ids of the instructions' rules the answer broke, see extract_rules


def _reference_figures(text: str) -> list[float]:
    """The figures of a prose reference: its numbers without the years it names, unless years are all it has."""
    numbers = _numbers_in(text)
    figures = [n for n in numbers if not (float(n).is_integer() and 1900 <= n <= 2100)]
    return figures or numbers


def _numbers_in(text: str) -> list[float]:
    out = []
    for match in _TEXT_NUMBER.finditer(text or ""):
        try:
            out.append(float(match.group(0).replace("$", "").replace(",", "")))
        except ValueError:
            continue
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.005 * abs(b), 0.51)


def _reference_values(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    values = []
    for row in rows:
        for key, value in row.items():
            if key == "year" or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                values.append(float(value))
            else:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
    return values


def _labels(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    labels = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, str) and key != "year":
                labels.append(value)
                break
    return labels


def _rows_match(reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]) -> bool:
    expected = sorted(_reference_values(reference))
    got = sorted(_reference_values(candidate))
    return len(expected) == len(got) and all(_close(g, e) for g, e in zip(got, expected))


_RANKED_KINDS = frozenset({"top_n", "drivers", "partial_year_rank"})
_LABELLED_KINDS = frozenset({"top_n", "drivers", "anti_join", "best_per_group", "compare", "partial_year_rank"})
_CHANGE_KINDS = frozenset({"yoy", "same_month_prior_year", "month_vs_previous"})


_PERIOD_ASSUMPTION_KINDS = frozenset({"relative_month", "holiday_week", "trailing_days", "season"})


def _with_assumption(question: Question, graded: Graded, text: str) -> Graded:
    """A relative period read one way or another is right only when the answer says which way."""
    if question.kind in _PERIOD_ASSUMPTION_KINDS and not (re.search(r"\b(19|20)\d\d\b", text) or _MONTH_NAMES.search(text)):
        return replace(graded, outcome="partial", cause="assumption_not_stated", detail=graded.detail + "; the answer does not say which period it took")
    return graded


def _alternate_cause(label: str) -> tuple[str, str]:
    """The cause an alternate reference names: a narrower channel, the wrong measure, a row count, the wrong channel."""
    if label.startswith("measure:"):
        return "wrong_measure", f"the sum of {label.split(':', 1)[1]}, not the measure the question and the definitions name"
    if label == "rowcount":
        return "row_count_not_distinct", "the row count, not the distinct order count the definitions name"
    if label.startswith("channel:"):
        return "wrong_channel", f"{label.split(':', 1)[1]}, not the channel the abbreviation names"
    return "narrower_scope", f"{label} alone, not the combined scope asked for"


def _grade_change(question: Question, reference: Reference, text: str, agent_rows: Sequence[Mapping[str, Any]] | None) -> Graded | None:
    """A year-over-year answer is right when it carries the change (absolute or percent) or both yearly totals."""
    totals = [float(row["value"]) for row in reference.rows if isinstance(row.get("value"), (int, float)) and not isinstance(row.get("value"), bool)]
    if len(totals) != 2:
        return None
    first, last = totals
    observed = list(_numbers_in(text))
    for row in agent_rows or ():
        observed.extend(float(v) for v in row.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
    if not observed:
        return None
    delta = last - first
    candidates = [delta, -delta]
    if first:
        candidates.extend([delta / first, delta / first * 100, -delta / first, -delta / first * 100])
    if any(_close(n, c) for n in observed for c in candidates if c) or (any(_close(n, first) for n in observed) and any(_close(n, last) for n in observed)):
        return Graded(question.id, "correct", "", "the change or both yearly totals match the reference", 2, 2, agent_rows is not None)
    return None


def grade(question: Question, reference: Reference, answer: AgentAnswer | str, *, agent_rows: Sequence[Mapping[str, Any]] | None = None) -> Graded:
    """Grade an answer against its reference: by the executed query when it was run, else by the prose."""
    text = answer.text if isinstance(answer, AgentAnswer) else str(answer)
    if reference.status == "behaviour":
        # graded by the rule alone: any answer without the forbidden content is right
        if not text.strip():
            return Graded(question.id, "incomplete", "no_answer", "no answer")
        if _EMAIL.search(text) or _PHONE.search(text):
            return Graded(question.id, "wrong", "personal_data_listed", "the answer lists contact details the instructions protect")
        return Graded(question.id, "correct", "", "no personal data in the answer")
    if reference.status == "decline":
        # the instructions put the topic out of scope: declining is right, figures are wrong
        if _numbers_in(text):
            return Graded(question.id, "wrong", "answered_out_of_scope", "the instructions put this topic out of scope, yet the answer carries figures")
        if _ABSTAIN_HINT.search(text or ""):
            return Graded(question.id, "correct", "declined_as_instructed", "the topic is out of scope by the instructions and the agent declined")
        return Graded(question.id, "incomplete", "no_decline", "the topic is out of scope by the instructions; the answer neither declines nor gives figures")
    if reference.status != "ok":
        if reference.status == "abstained":
            # the source holds no rows for this question: saying so is right,
            # producing figures is an invented answer
            if _ABSTAIN_HINT.search(text or "") and not _numbers_in(text):
                return Graded(question.id, "correct", "abstain_expected", "the source has no rows for this question and the agent said so")
            if _numbers_in(text):
                return Graded(question.id, "wrong", "answered_without_data", "the source has no rows for this question, yet the answer carries figures")
            return Graded(question.id, "incomplete", "abstain_expected", "the source has no rows for this question; the answer neither says so nor gives figures")
        return Graded(question.id, "no_reference", "reference_failed", reference.note)
    if question.kind in _CHANGE_KINDS:
        verdict = _grade_change(question, reference, text, agent_rows)
        if verdict is not None:
            return verdict
        agent_rows = None  # a one-row change is not a missing row; the prose says how much
    if agent_rows is not None:
        if _rows_match(reference.rows, agent_rows):
            return Graded(question.id, "correct", "", "the agent's executed query returns the reference rows", len(reference.rows), len(reference.rows), True)
        for label, rows in reference.alternates.items():
            if _rows_match(rows, agent_rows):
                if label.startswith("period:"):
                    return _with_assumption(question, Graded(question.id, "correct", "", f"the agent read the period as {label[7:]}", len(rows), len(rows), True), text)
                cause, detail = _alternate_cause(label)
                return Graded(question.id, "wrong", cause, f"the agent's query returns {detail}", 0, len(reference.rows), True)
        if len(agent_rows) < len(reference.rows):
            return Graded(question.id, "partial" if agent_rows else "wrong", "missing_rows", f"the agent's query returns {len(agent_rows)} rows, the reference {len(reference.rows)}", len(agent_rows), len(reference.rows), True)
        # fall through to the prose: the query differs, the prose says how much
    numbers = _numbers_in(text)
    expected = _reference_values(reference.rows)
    if question.kind == "drivers":
        # a drop is reported as a positive amount as often as a negative one
        numbers = [abs(n) for n in numbers]
        expected = [abs(v) for v in expected]
    matched = sum(1 for value in expected if any(_close(n, value) for n in numbers))
    if question.kind in _LABELLED_KINDS and not expected:
        labels = _labels(reference.rows)
        present = sum(1 for label in labels if text.find(label) >= 0)
        if labels and present == len(labels):
            return Graded(question.id, "correct", "", f"all {len(labels)} names present", present, len(labels))
        if labels and present >= 0.5 * len(labels):
            return Graded(question.id, "partial", "missing_rows", f"{len(labels) - present} of {len(labels)} names absent", present, len(labels))
        if _ABSTAIN_HINT.search(text or "") and not present:
            return Graded(question.id, "abstained", "agent_abstained", "the agent declined a question the source answers", 0, len(labels))
        return Graded(question.id, "wrong", "values_differ", f"{present} of {len(labels)} names present", present, len(labels))
    if not numbers:
        if _ABSTAIN_HINT.search(text or ""):
            if question.kind == "fuzzy_value":
                return _fuzzy_failure(question, len(expected))
            return Graded(question.id, "abstained", "agent_abstained", "the agent declined a question the source answers", 0, len(expected))
        if question.kind == "fuzzy_value":
            return _fuzzy_failure(question, len(expected))
        return Graded(question.id, "incomplete", "no_numbers", "no figures in the answer", 0, len(expected))
    if expected and question.execution.get("kind") == "supplied_text":
        # a prose reference may say more than the question asked; the agent is
        # right when every figure it gave is in the reference
        agent_figures = [n for n in numbers if not (float(n).is_integer() and 1900 <= n <= 2100)] or numbers
        hits = sum(1 for n in agent_figures if any(_close(n, value) for value in expected))
        if hits == len(agent_figures):
            return Graded(question.id, "correct", "", f"every figure the agent gave ({hits}) is in the reference answer", hits, len(expected), False)
        if hits:
            return Graded(question.id, "partial", "values_partially_match", f"{hits} of the {len(agent_figures)} figures the agent gave are in the reference answer", hits, len(expected), False)
    if expected and matched == len(expected):
        cause, detail = "", ""
        if question.kind in _PERIOD_ASSUMPTION_KINDS and not (re.search(r"\b(19|20)\d\d\b", text) or _MONTH_NAMES.search(text)):
            return Graded(question.id, "partial", "assumption_not_stated", "the figure is right but the answer does not say which period it took", matched, len(expected), agent_rows is not None)
        if question.kind in _LABELLED_KINDS:
            labels = _labels(reference.rows)
            positions = [text.find(label) for label in labels]
            if any(p < 0 for p in positions):
                cause, detail = "missing_rows", f"{sum(1 for p in positions if p < 0)} of {len(labels)} named items absent"
            elif question.kind in _RANKED_KINDS and positions != sorted(positions):
                cause, detail = "unsorted_ranking", "ranked items appear out of order"
        return Graded(question.id, "correct" if not cause else "partial", cause, detail, matched, len(expected), agent_rows is not None)
    for label, rows in reference.alternates.items():
        alternate = _reference_values(rows)
        if alternate and sum(1 for v in alternate if any(_close(n, v) for n in numbers)) == len(alternate):
            if label.startswith("period:"):
                return _with_assumption(question, Graded(question.id, "correct", "", f"the agent read the period as {label[7:]}", len(alternate), len(alternate), agent_rows is not None), text)
            cause, detail = _alternate_cause(label)
            return Graded(question.id, "wrong", cause, f"the figures match {detail}", matched, len(expected), agent_rows is not None)
    if expected and matched >= 0.5 * len(expected):
        labels = _labels(reference.rows)
        if question.kind in _LABELLED_KINDS and labels and any(text.find(label) < 0 for label in labels):
            return Graded(question.id, "partial", "missing_rows", f"{matched} of {len(expected)} values present; some ranked items absent", matched, len(expected), agent_rows is not None)
        return Graded(question.id, "partial", "values_partially_match", f"{matched} of {len(expected)} values present", matched, len(expected), agent_rows is not None)
    return Graded(question.id, "wrong", "values_differ", f"{matched} of {len(expected)} reference values present; the answer's figures do not match the source", matched, len(expected), agent_rows is not None)


# --------------------------------------------------------------------------- #
# Suggestions and the report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Suggestions:
    agent_instructions: str
    datasource_instructions: Mapping[str, str]
    datasource_descriptions: Mapping[str, str]
    fewshots: Mapping[str, tuple[FewShot, ...]]
    notes: tuple[str, ...] = ()


_CAUSE_GUIDANCE = {
    "unsorted_ranking": "Rank by the requested metric after all aggregation and return rows in that order with the rank shown.",
    "missing_rows": "Return exactly the number of rows the question asks for; if fewer exist, say so.",
    "narrower_scope": "When a question covers more than one fact table or channel, aggregate each separately and combine before ranking or totalling; state the scope used.",
    "values_differ": "Compute every figure with a query against the selected tables; never estimate or carry a number from a previous answer.",
    "agent_abstained": "Answer questions the selected schema can answer; abstain only when a required definition or table is missing.",
    "answered_without_data": "When the query returns no rows, say that the source has no data for that scope; do not produce figures.",
    "inconsistent": "Give the same answer to the same question: prefer the authoritative table named in the source instructions.",
    "misrouted": "Choose the data source by topic: name which source answers which subjects, and keep each source's description specific to its subjects.",
    "wrong_measure": "Use the column the definitions name for each measure word (units, cost, tax, freight); never substitute revenue for a quantity or a count.",
    "row_count_not_distinct": "Count orders as distinct order numbers, never as rows.",
    "wrong_channel": "Map channel words and abbreviations to their fact table before querying, as the definitions state (for example B2B to reseller sales, B2C to internet sales).",
    "answered_out_of_scope": "Decline questions on the topics the instructions put out of scope; never produce figures for them.",
    "assumption_not_stated": "When a period is relative or ambiguous (last month, winter, the week after a holiday), say which dates were used.",
    "personal_data_listed": "Never list email addresses, phone numbers or street addresses; summarise customers instead.",
    "fuzzy_match_failed": "Match a user's word to the stored values case-insensitively and accept singular or plural (bikes means Bikes); the vocabulary in the data-source instructions lists the values.",
}


def _fuzzy_failure(question: Question, expected: int) -> Graded:
    """A fuzzy-value question answered without a figure: the agent did not map the user's word to the stored value."""
    exact = next((str(f.get("value")) for f in (question.spec.get("filters") or ()) if isinstance(f, Mapping)), "")
    return Graded(question.id, "wrong", "fuzzy_match_failed", f"no figure for a value spelled as a user would; the stored value is {exact!r}", 0, expected)


def suggest(snapshot: AgentSnapshot, schemas: Sequence[SourceSchema], findings: Sequence[Finding], questions: Sequence[Question], references: Sequence[Reference], graded: Sequence[Graded]) -> Suggestions:
    by_schema = {s.source_id: s for s in schemas}
    ref_by_id = {r.question_id: r for r in references}
    grade_by_id = {g.question_id: g for g in graded}
    moved: dict[str, list[str]] = {}
    remove: set[str] = set()
    for finding in findings:
        if finding.code == "schema_in_agent_instructions" and finding.source_id:
            for line in finding.evidence:
                original = line.rsplit("  [", 1)[0]
                moved.setdefault(finding.source_id, []).append(re.sub(r"^\s*[-*]\s+", "", original))
                remove.add(original.casefold())
    agent_lines = [line for line in (snapshot.instructions or "").splitlines() if line.strip().casefold() not in remove]
    causes = sorted({g.cause for g in graded if g.outcome in {"wrong", "partial", "abstained"} and g.cause in _CAUSE_GUIDANCE})
    if causes:
        agent_lines.append("")
        agent_lines.append("BEHAVIOUR (from evaluation)")
        agent_lines.extend(f"- {_CAUSE_GUIDANCE[c]}" for c in causes)
    if len(snapshot.datasources) > 1 and any(f.code == "routing_rules_missing" for f in findings):
        agent_lines.append("")
        agent_lines.append("## Topics")
        agent_lines.extend(f"- When asked about {s.description.split('.')[0].strip() or s.name}, use {s.name}." for s in snapshot.datasources)
    datasource_instructions: dict[str, str] = {}
    datasource_descriptions: dict[str, str] = {}
    fewshots: dict[str, tuple[FewShot, ...]] = {}
    notes: list[str] = []
    for source in snapshot.datasources:
        text = source.instructions or ""
        additions: list[str] = []
        if moved.get(source.id):
            additions.append("## Schema notes (moved from agent instructions)")
            additions.extend(f"- {line}" for line in moved[source.id])
        schema = by_schema.get(source.id)
        if schema and schema.relationships and "join" not in text.casefold():
            additions.append("## Join Paths")
            additions.extend(f"- Join '{a}'[{ac}] to '{b}'[{bc}]." for a, ac, b, bc in schema.relationships[:12])
        data_lines: list[str] = []
        for finding in findings:
            if finding.basis == "data" and finding.source_id == source.id and finding.suggestion and f"- {finding.suggestion}" not in data_lines:
                data_lines.append(f"- {finding.suggestion}")
        if data_lines:
            additions.append("## From the data (inferred by the review; confirm before applying)")
            additions.extend(data_lines[:20])
        definitions_in_source = extract_definitions(text)
        used: list[str] = []
        for question in questions:
            if question.source_id == source.id and question.kind == "distinct_by_year" and "orders" not in " ".join(definitions_in_source).casefold():
                col = question.spec["facts"][0].get("order_column")
                if col:
                    used.append(f"- Orders = COUNT(DISTINCT {col}), not a row count.")
        if used:
            additions.append("## Business Rules (used by the reference queries)")
            additions.extend(sorted(set(used)))
        shots = []
        for question in questions:
            if question.source_id != source.id or source.kind == "semantic_model":
                continue
            g = grade_by_id.get(question.id)
            r = ref_by_id.get(question.id)
            if g and r and r.status == "ok" and g.outcome in {"wrong", "partial", "abstained"}:
                shots.append(FewShot(id="", question=question.text, query=question.reference_query))
        if shots:
            fewshots[source.id] = tuple(shots[: FEWSHOT_SWEET_SPOT[1]])
            if len(shots) > FEWSHOT_SWEET_SPOT[1]:
                notes.append(f"{len(shots) - FEWSHOT_SWEET_SPOT[1]} further failed question(s) on {source.name or source.id} were left out of the few-shots to stay within the {FEWSHOT_SWEET_SPOT[1]}-example sweet spot; fix these first and re-run.")
        datasource_instructions[source.id] = text + ("\n\n" + "\n".join(additions) if additions else "")
        if not source.description.strip() and schema:
            facts = ", ".join(t for t in _fact_tables(schema)[:3]) or ", ".join(list(schema.tables)[:3])
            datasource_descriptions[source.id] = f"{source.kind.replace('_', ' ').title()} data covering {facts}. Use for questions about these tables."
        if not additions and not shots and source.id not in datasource_descriptions:
            notes.append(f"No change proposed for source {source.name or source.id}.")
    return Suggestions("\n".join(agent_lines).strip(), datasource_instructions, datasource_descriptions, fewshots, tuple(notes))


@dataclass(frozen=True)
class Analysis:
    """The RLM's explanation of one graded question and the change it proposes."""

    question_id: str
    question: str
    explanation: str
    proposed_change: str = ""


def _short_json(value: Any, limit: int = 240) -> str:
    try:
        text = json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


_PERSONAL_HINT = re.compile(
    r"(e_?mail|phone|mobile|fax|address|birth|\bdob\b|ssn|social_?security|passport|national_?id|tax_?id|salary|password|secret|token|credit|card_?number|iban|routing|first_?name|last_?name|middle_?name|full_?name|surname|gender|marital"
    r"|telefon|celular|endereco|direccion|adresse|nascimento|nacimiento|geburt|correo|courriel|\bcpf\b|\bcnpj\b|\bnif\b|\bdni\b|\brg\b|sexo|genero)",
    re.IGNORECASE,
)  # English first, then the tokens common in Portuguese, Spanish, French and German column names

_KNOWLEDGE_USE = (
    "How the review uses it: the profile (tables, column types, joins) is the schema behind every generated question and its reference; "
    "the registered operations let the RLM's second opinion aggregate any covered table without writing SQL; "
    "the lessons carry the definitions the instructions and the reviewer declared into every RLM task; "
    "the personal-data columns show what the agent could expose from the tables it can read."
)


def _time_axis_text(date: Mapping[str, Any] | None) -> str:
    if not date:
        return ""
    if date.get("year"):
        return f"{date['date_table']}.{date['year']} via {date['column']}"
    if date.get("date_table"):
        return f"{date['date_table']}.{date['timestamp_column']} via {date['column']}"
    return str(date["column"])


def _table_summaries(schema: SourceSchema, selected_paths: Sequence[str], covered: Collection[str]) -> list[dict[str, Any]]:
    """One row per table of a lakehouse profile, the agent's selected tables and the facts first."""
    joins = _heuristic_joins(schema)
    facts = set(_fact_tables(schema))
    selected = {p.replace("/", ".").casefold() for p in selected_paths} | {p.rsplit("/", 1)[-1].casefold() for p in selected_paths}
    rows: list[dict[str, Any]] = []
    for table, columns in schema.tables.items():
        names = {table.casefold(), table.rsplit(".", 1)[-1].casefold()}
        date = _date_join(schema, table, joins) if table in facts else None
        rows.append(
            {
                "table": table,
                "columns": len(columns),
                "selected": bool(names & selected) if selected else None,
                "fact": table in facts,
                "time_axis": _time_axis_text(date),
                "measures": _measure_columns(schema, table)[:3] if table in facts else [],
                "personal": [c for c in columns if _PERSONAL_HINT.search(c)][:6],
                "operation": bool(names & set(covered)),
            }
        )
    rows.sort(key=lambda r: (r["selected"] is not True, not r["fact"], r["table"].casefold()))
    return rows


def _lesson_dict(lesson: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(lesson, "lesson_id", "") or ""),
        "kind": str(getattr(lesson, "kind", "") or ""),
        "subject": str(getattr(lesson, "subject", "") or ""),
        "status": str(getattr(lesson, "status", "") or ""),
        "confidence": str(getattr(lesson, "confidence", "") or ""),
        "rule": _short_json(getattr(lesson, "structured_rule", None) or {}),
        "basis": [str(b) for b in (getattr(lesson, "basis", ()) or ())],
    }


def summarize_knowledge(knowledge: Any, schemas: Sequence[SourceSchema] | None = None, snapshot: AgentSnapshot | None = None) -> dict[str, Any]:
    """What ``RLM.learn`` recorded, as plain data for the report, and what the review makes of it.

    ``sources`` carry the profiles; a lakehouse profile also lists its
    ``tables`` as the generator sees them (the agent's selected tables and
    the facts first): whether the table is a fact, the time axis and the
    measures the questions rely on, personal-data columns, and whether a
    registered aggregate operation covers it. ``operation_kinds`` groups the
    registered operations by name; ``lessons`` are the facts every RLM task
    receives; ``learned`` and ``runs`` are filled by :func:`deepen` when the
    package learned from the RLM's own runs.
    """
    package = getattr(knowledge, "package", knowledge)
    schema_by_id = {s.source_id: s for s in (schemas or ())}
    selected_by_id = {s.id: tuple(s.selected_tables) for s in (snapshot.datasources if snapshot is not None else ())}
    operations = [
        {
            "operation": str(getattr(op, "operation", "") or ""),
            "objects": [str(v) for v in (((getattr(op, "parameter_schema", None) or {}).get("catalog_source") or {}).get("enum") or ())],
            "sources": [str(s) for s in (getattr(op, "required_sources", ()) or ())],
            "status": str(getattr(op, "status", "") or ""),
            "grain": str(getattr(op, "grain", "") or ""),
            "parameters": sorted(str(k) for k in (getattr(op, "parameter_schema", None) or {})),
        }
        for op in getattr(package, "operations", ()) or ()
    ]
    covered = {o.casefold() for op in operations for o in op["objects"]}
    kinds: dict[str, dict[str, Any]] = {}
    for op in operations:
        entry = kinds.setdefault(op["operation"], {"operation": op["operation"], "count": 0, "parameters": op["parameters"], "objects": 0, "sources": []})
        entry["count"] += 1
        entry["objects"] += len(op["objects"])
        entry["sources"].extend(s for s in op["sources"] if s not in entry["sources"])
    sources: list[dict[str, Any]] = []
    for profile in getattr(package, "sources", ()) or ():
        schema = getattr(profile, "schema", None) or {}
        family = str(getattr(profile, "family", "") or "")
        source_id = str(getattr(profile, "source_id", "") or "")
        if family == "lakehouse":
            counts = [len(entry.get("columns") or {}) for entry in schema.values() if isinstance(entry, Mapping)]
            shape = f"{len(counts)} tables, {sum(counts)} columns"
        elif family == "semantic_model":
            shape = f"{len(schema.get('columns') or {})} columns, {len(schema.get('measures') or {})} measures, {len(schema.get('relationships') or {})} relationships"
        else:
            shape = f"{sum(1 for entry in schema.values() if isinstance(entry, Mapping))} columns"
        diagnostics = getattr(profile, "diagnostics", None) or {}
        source_schema = schema_by_id.get(source_id)
        if source_schema is None and family == "lakehouse":
            try:
                source_schema = schema_from_profile(profile, source_id=source_id)
            except Exception:  # noqa: BLE001 - an odd profile still gets its summary line
                source_schema = None
        tables = _table_summaries(source_schema, selected_by_id.get(source_id, ()), covered) if source_schema is not None and source_schema.kind == "lakehouse" else []
        sources.append(
            {
                "source_id": source_id,
                "family": family,
                "status": str(getattr(profile, "status", "") or ""),
                "role": str(getattr(profile, "role", "") or ""),
                "shape": shape,
                "schema_fingerprint": str(getattr(profile, "schema_fingerprint", "") or "")[:12],
                "snapshot_fingerprint": str(getattr(profile, "snapshot_fingerprint", "") or "")[:12],
                "sensitive_columns": [str(c) for c in (getattr(profile, "sensitive_columns", ()) or ())],
                "diagnostics": {str(k): v for k, v in diagnostics.items() if isinstance(v, (str, int, float, bool))} if isinstance(diagnostics, Mapping) else {},
                "selected": len(selected_by_id.get(source_id, ())),
                "tables": tables,
            }
        )
    lessons = [_lesson_dict(lesson) for lesson in (getattr(package, "lessons", ()) or ())]
    events = Counter(str(getattr(event, "event_type", "") or "") for event in (getattr(package, "events", ()) or ()))
    return {
        "package_id": str(getattr(package, "package_id", "") or ""),
        "sources": sources,
        "operations": operations,
        "operation_kinds": sorted(kinds.values(), key=lambda k: (-k["count"], k["operation"])),
        "lessons": lessons,
        "learned": [],
        "runs": 0,
        "events": dict(sorted(events.items())),
        "evidence": len(getattr(package, "evidence", ()) or ()),
    }


def _shown_tables(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The agent's selected tables, or every table when the selection is unknown."""
    tables = list(source.get("tables") or [])
    selected = [t for t in tables if t.get("selected") is True]
    return selected or tables


def _cell(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "" if value is None else str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


_HTML_STYLE = """<style>
.rlm-review { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #1f2933; max-width: 1150px; line-height: 1.45; }
.rlm-review h1 { font-size: 1.6em; margin: 0 0 4px 0; }
.rlm-review h2 { font-size: 1.2em; margin: 22px 0 8px 0; border-bottom: 1px solid #d9dee3; padding-bottom: 4px; }
.rlm-review h3 { font-size: 1em; margin: 14px 0 6px 0; }
.rlm-review .muted { color: #616e7c; }
.rlm-review table { border-collapse: collapse; width: 100%; font-size: 0.92em; margin: 8px 0; }
.rlm-review th, .rlm-review td { border: 1px solid #d9dee3; padding: 5px 8px; text-align: left; vertical-align: top; }
.rlm-review th { background: #f0f3f5; }
.rlm-review .badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 0.8em; font-weight: 600; color: #fff; margin-right: 6px; }
.rlm-review .sev-high { background: #c0392b; } .rlm-review .sev-medium { background: #d68910; } .rlm-review .sev-low { background: #2e86c1; } .rlm-review .sev-info { background: #7f8c8d; }
.rlm-review .out-correct { background: #e6f4ea; } .rlm-review .out-partial { background: #fff4e0; } .rlm-review .out-wrong { background: #fdecea; }
.rlm-review .out-abstained, .rlm-review .out-incomplete, .rlm-review .out-no_reference { background: #f0f3f5; }
.rlm-review .chip { display: inline-block; padding: 2px 10px; border-radius: 12px; margin: 0 6px 6px 0; font-size: 0.9em; border: 1px solid #d9dee3; }
.rlm-review .card { border: 1px solid #d9dee3; border-left: 4px solid #7f8c8d; border-radius: 4px; padding: 8px 12px; margin: 8px 0; }
.rlm-review .card-high { border-left-color: #c0392b; } .rlm-review .card-medium { border-left-color: #d68910; } .rlm-review .card-low { border-left-color: #2e86c1; }
.rlm-review pre { background: #f6f8fa; border: 1px solid #d9dee3; border-radius: 4px; padding: 8px; overflow-x: auto; white-space: pre-wrap; font-size: 0.85em; margin: 4px 0 8px 0; }
.rlm-review code { background: #f0f3f5; padding: 1px 4px; border-radius: 3px; }
.rlm-review details { margin: 4px 0; } .rlm-review summary { cursor: pointer; }
.rlm-review ul { margin: 4px 0 4px 18px; padding: 0; }
</style>"""


@dataclass
class ReviewReport:
    snapshot: AgentSnapshot
    schemas: tuple[SourceSchema, ...]
    findings: tuple[Finding, ...]
    questions: tuple[Question, ...]
    references: tuple[Reference, ...]
    answers: Mapping[str, tuple[AgentAnswer, ...]]
    graded: tuple[Graded, ...]
    suggestions: Suggestions
    notes: tuple[str, ...] = ()  # diagnostics: year discovery, why no questions
    context: ReviewContext | None = None
    knowledge: Mapping[str, Any] | None = None  # summarize_knowledge(RLM.learn(...))
    analysis: tuple[Analysis, ...] = ()  # the RLM's explanations, from deepen()
    discovered: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None  # discover_drivers() per source
    rules: tuple[Rule, ...] = ()  # the instructions' checkable rules, see extract_rules
    learned_knowledge: Any = None  # the package after RLM.enrich over the deeper analysis runs, see deepen(); save it to reuse the lessons
    sweeps: tuple[Any, ...] = ()  # fabric_rlm.sweep.Sweep per lakehouse source, when review_agent ran with a sweep budget

    def compliance(self) -> list[dict[str, Any]]:
        """Per rule: how many answers it applied to, how many broke it, and which questions."""
        rows: list[dict[str, Any]] = []
        for rule in self.rules:
            broken = [g.question_id for g in self.graded if rule.id in g.violations]
            rows.append({"rule": rule.id, "text": rule.text, "answers": len(self.graded), "violations": len(broken), "questions": broken[:8]})
        return rows

    def score(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for g in self.graded:
            counts[g.outcome] = counts.get(g.outcome, 0) + 1
        return counts

    def by_skill(self) -> dict[str, dict[str, int]]:
        """Outcome counts per skill the questions test, in the order the skills first appear."""
        grade_by_id = {g.question_id: g for g in self.graded}
        table: dict[str, dict[str, int]] = {}
        for q in self.questions:
            skill = q.skill or q.kind
            counts = table.setdefault(skill, {"questions": 0, "correct": 0, "partial": 0, "wrong": 0, "other": 0})
            counts["questions"] += 1
            g = grade_by_id.get(q.id)
            if g is None:
                counts["other"] += 1
            elif g.outcome in counts:
                counts[g.outcome] += 1
            else:
                counts["other"] += 1
        return table

    def _knowledge_lines(self) -> list[str]:
        k = self.knowledge or {}
        runs = f", {k['runs']} run(s) of the RLM's own references" if k.get("runs") else ""
        learned = f", {len(k['learned'])} lesson(s) learned from them" if k.get("learned") else ""
        lines = [
            "## What the RLM learned",
            f"Package {k.get('package_id', '')}: {len(k.get('sources', []))} source(s) profiled, {len(k.get('operations', []))} registered operation(s), {len(k.get('lessons', []))} lesson(s), {k.get('evidence', 0)} evidence record(s){runs}{learned}.",
            "",
            _KNOWLEDGE_USE,
            "",
        ]
        for src in k.get("sources", []):
            sensitive = f"; sensitive columns: {', '.join(src['sensitive_columns'][:8])}" if src.get("sensitive_columns") else ""
            selected = f"; {src['selected']} tables selected for the agent" if src.get("selected") else ""
            lines.append(f"- {src['source_id']} ({src['family']}, {src['status']}, role {src['role']}): {src['shape']}{selected}; schema fingerprint {src['schema_fingerprint']}; snapshot {src['snapshot_fingerprint']}{sensitive}")
            shown = _shown_tables(src)
            if shown and not src.get("selected"):
                lines.append(f"  - the agent's selection was not readable; all {len(shown)} tables follow")
            for table in shown[:60]:
                role = "fact" if table["fact"] else "dimension"
                axis = f"; time axis {table['time_axis'] or 'none'}; measures {', '.join(table['measures']) or 'none'}" if table["fact"] else ""
                personal = f"; personal data: {', '.join(table['personal'])}" if table.get("personal") else ""
                lines.append(f"  - {table['table']}: {table['columns']} columns, {role}{axis}{personal}; aggregate operation {'registered' if table['operation'] else 'none'}")
        for kind in k.get("operation_kinds", []):
            lines.append(f"- operation {kind['operation']} x{kind['count']} covering {kind['objects']} object(s) on {', '.join(kind['sources'])}; parameters: {', '.join(kind['parameters']) or 'none'}")
        for lesson in k.get("lessons", []):
            lines.append(f"- lesson {lesson['kind']} on {lesson['subject']} ({lesson['status']}, {lesson['confidence']}): {lesson['rule']}")
        if k.get("runs"):
            if k.get("learned"):
                for lesson in k["learned"]:
                    lines.append(f"- learned from this review: {lesson['kind']} on {lesson['subject']} ({lesson['status']}, {lesson['confidence']}): {lesson['rule']}")
            else:
                lines.append(f"- learned from this review: nothing yet; the {k['runs']} run(s) left {k.get('evidence', 0)} evidence record(s) and no lesson met the promotion policy")
        if k.get("events"):
            lines.append("- events: " + ", ".join(f"{name} x{n}" for name, n in k["events"].items()))
        lines.append("")
        return lines

    def to_html(self) -> str:
        """The report as one self-contained HTML fragment (inline style), for a notebook or a file."""
        esc = html.escape
        s = self.snapshot
        by_source = {x.source_id: x for x in self.schemas}
        ref_by_id = {r.question_id: r for r in self.references}
        grade_by_id = {g.question_id: g for g in self.graded}
        parts = [_HTML_STYLE, '<div class="rlm-review">', f"<h1>Data Agent review: {esc(s.name)}</h1>"]
        parts.append(f'<div class="muted">Stage: {esc(s.stage)}. Agent instructions: {len(s.instructions):,} characters. Sources: {len(s.datasources)}.</div>')
        if self.context:
            parts.append("<h2>Scope and context (stated by the reviewer)</h2><ul>")
            parts.extend(f"<li>{esc(line)}</li>" for line in self.context.as_prompt().splitlines())
            if self.context.questions:
                parts.append(f"<li>Supplied questions: {len(self.context.questions)} (graded first)</li>")
            parts.append("</ul>")
        parts.append("<h2>Sources</h2><table><tr><th>Source</th><th>Kind</th><th>Profile</th><th>Selected for the agent</th><th>Few-shots</th><th>Instructions</th><th>Description</th></tr>")
        for source in s.datasources:
            schema = by_source.get(source.id)
            shape = f"{len(schema.tables)} tables" + (f", {len(schema.measures)} measures" if schema and schema.measures else "") if schema else "not profiled"
            parts.append(f"<tr><td>{esc(source.name or source.id)}</td><td>{esc(source.kind)}</td><td>{esc(shape)}</td><td>{len(source.selected_tables) if source.selected_tables else 'unknown'}</td><td>{len(source.fewshots)}</td><td>{len(source.instructions):,} chars</td><td>{'present' if source.description.strip() else 'missing'}</td></tr>")
        parts.append("</table>")
        parts.append("<h2>Findings</h2>")
        setup_findings = [f for f in self.findings if f.basis != "data"]
        if not setup_findings:
            parts.append('<div class="muted">No findings.</div>')
        for f in setup_findings:
            where = f' <span class="muted">({esc(_source_label(s, f.source_id))})</span>' if f.source_id else ""
            parts.append(f'<div class="card card-{esc(f.severity)}"><span class="badge sev-{esc(f.severity)}">{esc(f.severity)}</span><code>{esc(f.code)}</code>{where}<div>{esc(f.message)}</div>')
            if f.evidence:
                parts.append("<ul>" + "".join(f"<li>{esc(e)}</li>" for e in f.evidence[:12]) + "</ul>")
            if f.suggestion:
                parts.append(f"<div><b>Suggestion:</b> {esc(f.suggestion)}</div>")
            if f.basis:
                parts.append(f'<div class="muted">Basis: {esc(f.basis)}</div>')
            parts.append("</div>")
        data_hints = [f for f in self.findings if f.basis == "data"]
        if data_hints:
            parts.append("<h2>Hints from the data</h2>")
            parts.append('<div class="muted">Inferred from the data itself, not from the setup: referential gaps, spellings, vocabulary, coverage, personal data. The reviewer decides which become instructions or data fixes; the suggested data-source instructions carry them under "From the data".</div>')
            for f in sorted(data_hints, key=lambda x: ("high", "medium", "low", "info").index(x.severity)):
                where = f' <span class="muted">({esc(_source_label(s, f.source_id))})</span>' if f.source_id else ""
                parts.append(f'<div class="card card-{esc(f.severity)}"><span class="badge sev-{esc(f.severity)}">{esc(f.severity)}</span><code>{esc(f.code)}</code>{where}<div>{esc(f.message)}</div>')
                if f.evidence:
                    parts.append("<ul>" + "".join(f"<li><code>{esc(e)}</code></li>" for e in f.evidence[:4]) + "</ul>")
                if f.suggestion:
                    parts.append(f"<div><b>Proposed instruction:</b> {esc(f.suggestion)}</div>")
                parts.append("</div>")
        for swept in self.sweeps:
            parts.append(f"<h2>What moved ({esc(_source_label(s, swept.source_id))})</h2>")
            parts.append('<div class="muted">Every figure was computed by the source and recomputed by an independent query. The classification says whether a change is carried by one group, a few, or spread in proportion to size.</div>')
            parts.append(swept.to_html())
        parts.append("<h2>Evaluation</h2>")
        counts = self.score()
        parts.append("<div>" + ("".join(f'<span class="chip out-{esc(k)}">{esc(k)}: {v}</span>' for k, v in sorted(counts.items())) or '<span class="muted">No questions.</span>') + "</div>")
        if self.notes:
            parts.append("<div><b>Diagnostics</b><ul>" + "".join(f"<li>{esc(n)}</li>" for n in self.notes) + "</ul></div>")
        if self.questions:
            parts.append("<table><tr><th>#</th><th>Question</th><th>Outcome</th><th>Cause</th><th>Detail</th><th>Query</th><th>Routed to</th></tr>")
            for index, q in enumerate(self.questions, start=1):
                g = grade_by_id.get(q.id)
                answers = self.answers.get(q.id, ())
                query = next((a.query for a in answers if a.query), None)
                routed = next((a.datasource for a in answers if a.datasource), "")
                detail = g.detail if g else ref_by_id.get(q.id, Reference(q.id, "failed")).note
                outcome = g.outcome if g else ""
                broken = f'<div class="muted">broke {esc(", ".join(g.violations))}</div>' if g and g.violations else ""
                parts.append(f'<tr><td>{index}</td><td>{esc(q.text)}<div class="muted">{esc(q.skill or q.kind)}</div></td><td class="out-{esc(outcome)}">{esc(outcome)}</td><td>{esc(g.cause if g else "")}</td><td>{esc(detail)}{broken}</td><td>{"yes" if query else "no"}</td><td>{esc(routed)}</td></tr>')
            parts.append("</table>")
            if self.rules:
                parts.append("<h3>Instruction compliance</h3><table><tr><th>Rule</th><th>Broken by</th><th>Questions</th><th>The instruction</th></tr>")
                for row in self.compliance():
                    parts.append(f"<tr><td><code>{esc(row['rule'])}</code></td><td>{row['violations']} of {row['answers']}</td><td>{esc(', '.join(q.rsplit('.', 1)[-1] for q in row['questions']))}</td><td>{esc(row['text'][:160])}</td></tr>")
                parts.append("</table>")
            by_skill = self.by_skill()
            if by_skill:
                parts.append("<h3>Outcomes by skill</h3><table><tr><th>Skill</th><th>Questions</th><th>Correct</th><th>Partial</th><th>Wrong</th><th>Other</th></tr>")
                for skill, counts in by_skill.items():
                    parts.append(f"<tr><td>{esc(skill)}</td><td>{counts['questions']}</td><td>{counts['correct']}</td><td>{counts['partial']}</td><td>{counts['wrong']}</td><td>{counts['other']}</td></tr>")
                parts.append("</table>")
            for index, q in enumerate(self.questions, start=1):
                answers = self.answers.get(q.id, ())
                reference = ref_by_id.get(q.id)
                body: list[str] = []
                for attempt, a in enumerate(answers, start=1):
                    label = f" {attempt}" if len(answers) > 1 else ""
                    body.append(f"<div><b>Agent answer{label}</b> ({a.seconds}s):</div><pre>{esc(a.text[:3000])}</pre>")
                    if a.query:
                        body.append(f"<div><b>Agent query</b> ({esc(a.language or 'unknown')}{'' if a.executed else ', generated but not executed'}):</div><pre>{esc(a.query[:3000])}</pre>")
                    if a.steps:
                        step_counts = Counter(st.name for st in a.steps if st.name)
                        functions = ", ".join(f"{name} x{n}" if n > 1 else name for name, n in sorted(step_counts.items())) or "unnamed"
                        body.append(f'<div class="muted">{len(a.steps)} step(s): {esc(functions)}; routed to: {esc(a.datasource or "unknown")}; status: {esc(a.status)}</div>')
                if q.technical:
                    body.append(f'<div class="muted">In schema terms: {esc(q.technical)}</div>')
                if q.reference_query:
                    body.append(f"<div><b>Reference query:</b></div><pre>{esc(q.reference_query)}</pre>")
                if reference is not None and reference.rows:
                    columns = list(reference.rows[0].keys())
                    body.append("<div><b>Reference rows</b>" + (f' <span class="muted">(first 10 of {len(reference.rows)})</span>' if len(reference.rows) > 10 else "") + "</div>")
                    body.append("<table><tr>" + "".join(f"<th>{esc(str(c))}</th>" for c in columns) + "</tr>" + "".join("<tr>" + "".join(f"<td>{esc(_cell(row.get(c)))}</td>" for c in columns) + "</tr>" for row in reference.rows[:10]) + "</table>")
                elif reference is not None:
                    body.append(f'<div class="muted">Reference {esc(reference.status)}: {esc(reference.note)}</div>')
                parts.append(f"<details><summary>{index}. {esc(q.text)}</summary>" + "".join(body) + "</details>")
        if self.analysis:
            parts.append("<h2>Analysis (RLM)</h2>")
            for item in self.analysis:
                g = grade_by_id.get(item.question_id)
                verdict = f' <span class="chip out-{esc(g.outcome)}">{esc(g.outcome)}{", " + esc(g.cause) if g.cause else ""}</span>' if g else ""
                change = f"<div><b>Proposed change:</b> {esc(item.proposed_change)}</div>" if item.proposed_change else ""
                parts.append(f'<div class="card"><div><b>{esc(item.question)}</b>{verdict}</div><div>{esc(item.explanation)}</div>{change}</div>')
        if self.knowledge:
            k = self.knowledge
            parts.append("<h2>What the RLM learned</h2>")
            runs = f", {k['runs']} run(s) of the RLM's own references" if k.get("runs") else ""
            learned = f", {len(k['learned'])} lesson(s) learned from them" if k.get("learned") else ""
            parts.append(f'<div class="muted">Package {esc(str(k.get("package_id", "")))}: {len(k.get("sources", []))} source(s) profiled, {len(k.get("operations", []))} registered operation(s), {len(k.get("lessons", []))} lesson(s), {k.get("evidence", 0)} evidence record(s){esc(runs)}{esc(learned)}.</div>')
            parts.append(f"<p>{esc(_KNOWLEDGE_USE)}</p>")
            if k.get("sources"):
                parts.append("<table><tr><th>Source</th><th>Family</th><th>Status</th><th>Role</th><th>Profile</th><th>Selected for the agent</th><th>Schema fingerprint</th><th>Snapshot</th><th>Sensitive columns</th></tr>")
                for src in k["sources"]:
                    parts.append(f"<tr><td>{esc(src['source_id'])}</td><td>{esc(src['family'])}</td><td>{esc(src['status'])}</td><td>{esc(src['role'])}</td><td>{esc(src['shape'])}</td><td>{src.get('selected') or 'unknown'}</td><td><code>{esc(src['schema_fingerprint'])}</code></td><td><code>{esc(src['snapshot_fingerprint'])}</code></td><td>{esc(', '.join(src.get('sensitive_columns', [])[:8]))}</td></tr>")
                parts.append("</table>")
                for src in k["sources"]:
                    shown = _shown_tables(src)
                    if not shown:
                        continue
                    title = "Selected tables as the RLM sees them" if src.get("selected") else f"Tables as the RLM sees them (the agent's selection was not readable; all {len(shown)} shown)"
                    parts.append(f"<h3>{esc(title)}</h3><table><tr><th>Table</th><th>Columns</th><th>Role</th><th>Time axis</th><th>Measures</th><th>Personal data</th><th>Aggregate operation</th></tr>")
                    for table in shown[:60]:
                        role = "fact" if table["fact"] else "dimension"
                        axis = esc(table["time_axis"] or ("none" if table["fact"] else ""))
                        parts.append(f"<tr><td><code>{esc(table['table'])}</code></td><td>{table['columns']}</td><td>{role}</td><td>{axis}</td><td>{esc(', '.join(table['measures']))}</td><td>{esc(', '.join(table['personal']))}</td><td>{'registered' if table['operation'] else 'none'}</td></tr>")
                    parts.append("</table>")
                    facts = [t for t in shown if t["fact"]]
                    notes_ = []
                    if facts:
                        notes_.append(f"{len(facts)} of {len(shown)} tables are facts the generator can ask about")
                    missing_axis = [t["table"] for t in facts if not t["time_axis"]]
                    if missing_axis:
                        notes_.append("no time axis for " + ", ".join(missing_axis))
                    uncovered = [t["table"] for t in shown if not t["operation"]]
                    if uncovered:
                        notes_.append("no registered operation for " + ", ".join(uncovered[:8]) + (" and more" if len(uncovered) > 8 else ""))
                    if notes_:
                        parts.append(f'<div class="muted">{esc("; ".join(notes_))}.</div>')
            if k.get("operation_kinds"):
                parts.append("<h3>Registered operations</h3><table><tr><th>Operation</th><th>Count</th><th>Objects covered</th><th>Sources</th><th>Parameters</th></tr>")
                for kind in k["operation_kinds"]:
                    parts.append(f"<tr><td><code>{esc(kind['operation'])}</code></td><td>{kind['count']}</td><td>{kind['objects']}</td><td>{esc(', '.join(kind['sources']))}</td><td>{esc(', '.join(kind['parameters']))}</td></tr>")
                parts.append("</table>")
            if k.get("lessons"):
                parts.append("<h3>Lessons</h3><table><tr><th>Kind</th><th>Subject</th><th>Status</th><th>Confidence</th><th>Rule</th></tr>")
                for lesson in k["lessons"]:
                    parts.append(f"<tr><td>{esc(lesson['kind'])}</td><td>{esc(lesson['subject'])}</td><td>{esc(lesson['status'])}</td><td>{esc(lesson['confidence'])}</td><td><code>{esc(lesson['rule'])}</code></td></tr>")
                parts.append("</table>")
            if k.get("runs"):
                parts.append("<h3>Learned from this review</h3>")
                if k.get("learned"):
                    parts.append("<table><tr><th>Kind</th><th>Subject</th><th>Status</th><th>Confidence</th><th>Rule</th><th>Basis</th></tr>")
                    for lesson in k["learned"]:
                        parts.append(f"<tr><td>{esc(lesson['kind'])}</td><td>{esc(lesson['subject'])}</td><td>{esc(lesson['status'])}</td><td>{esc(lesson['confidence'])}</td><td><code>{esc(lesson['rule'])}</code></td><td>{esc(', '.join(lesson.get('basis', [])))}</td></tr>")
                    parts.append("</table>")
                    parts.append('<div class="muted">The enriched package is on the report as learned_knowledge; save it with a knowledge store to start the next review from these lessons.</div>')
                else:
                    parts.append(f'<div class="muted">Nothing yet: the {k["runs"]} run(s) left {k.get("evidence", 0)} evidence record(s) and no lesson met the promotion policy.</div>')
            if k.get("events"):
                parts.append('<div class="muted">Events: ' + esc(", ".join(f"{name} x{n}" for name, n in k["events"].items())) + "</div>")
        parts.append("<h2>Suggested changes</h2>")
        if self.suggestions.agent_instructions:
            parts.append(f"<h3>Agent instructions</h3><pre>{esc(self.suggestions.agent_instructions)}</pre>")
        for source in s.datasources:
            text = self.suggestions.datasource_instructions.get(source.id)
            description = self.suggestions.datasource_descriptions.get(source.id)
            shots = self.suggestions.fewshots.get(source.id, ())
            if not (text or description or shots):
                continue
            parts.append(f"<h3>{esc(source.name or source.id)}</h3>")
            if description:
                parts.append(f"<div><b>Description:</b> {esc(description)}</div>")
            if text:
                parts.append(f"<div><b>Data-source instructions:</b></div><pre>{esc(text)}</pre>")
            if shots:
                parts.append(f"<div><b>Few-shots ({len(shots)}):</b></div>")
                for shot in shots:
                    parts.append(f"<div>Q: {esc(shot.question)}</div><pre>{esc(shot.query)}</pre>")
        if self.suggestions.notes:
            parts.append("<ul>" + "".join(f"<li>{esc(n)}</li>" for n in self.suggestions.notes) + "</ul>")
        parts.append("<h2>Method</h2>")
        parts.append('<div class="muted">References were computed by executing generated queries against the sources, never by a model; supplied and RLM-proposed questions carry the reference the reviewer or the verified RLM solve gave. Grades compare the agent\'s executed query where the run steps exposed it, else its prose figures, with a 0.5% tolerance. Repeat the evaluation three times before believing a delta.</div>')
        parts.append("</div>")
        return "\n".join(parts)

    def to_markdown(self) -> str:
        s = self.snapshot
        lines = [f"# Data Agent review: {s.name}", ""]
        lines.append(f"Stage: {s.stage}. Agent instructions: {len(s.instructions):,} characters. Sources: {len(s.datasources)}.")
        lines.append("")
        if self.context:
            lines.append("## Scope and context (stated by the reviewer)")
            lines.extend(f"- {line}" for line in self.context.as_prompt().splitlines())
            if self.context.questions:
                lines.append(f"- Supplied questions: {len(self.context.questions)} (graded first; a supplied query is executed for the reference, a supplied answer's figures are the reference)")
            lines.append("")
        lines.append("## Sources")
        for source in s.datasources:
            schema = next((x for x in self.schemas if x.source_id == source.id), None)
            shape = f"{len(schema.tables)} tables" + (f", {len(schema.measures)} measures" if schema and schema.measures else "") if schema else "not profiled"
            selected = f"; {len(source.selected_tables)} tables selected for the agent" if source.selected_tables else ""
            lines.append(f"- {source.name or source.id} ({source.kind}): {shape}{selected}; {len(source.fewshots)} few-shots; {len(source.instructions):,} characters of instructions; description {'present' if source.description.strip() else 'missing'}.")
        lines.append("")
        lines.append("## Findings")
        setup_findings = [f for f in self.findings if f.basis != "data"]
        if not setup_findings:
            lines.append("- none")
        for f in sorted(setup_findings, key=lambda x: ("high", "medium", "low", "info").index(x.severity)):
            lines.append(f"- **{f.severity}** `{f.code}`" + (f" ({f.source_id})" if f.source_id else "") + f": {f.message}")
            for e in f.evidence[:6]:
                lines.append(f"    - {e}")
            if f.suggestion:
                lines.append(f"    - suggestion: {f.suggestion}")
            if f.basis:
                lines.append(f"    - basis: {f.basis}")
        lines.append("")
        data_hints = [f for f in self.findings if f.basis == "data"]
        if data_hints:
            lines.append("## Hints from the data")
            lines.append("Inferred from the data itself, not from the setup; the reviewer decides which become instructions or data fixes.")
            for f in sorted(data_hints, key=lambda x: ("high", "medium", "low", "info").index(x.severity)):
                lines.append(f"- [{f.severity}] {f.code}: {f.message}")
                if f.suggestion:
                    lines.append(f"    - proposed instruction: {f.suggestion}")
            lines.append("")
        for swept in self.sweeps:
            lines.append(f"## What moved ({_source_label(s, swept.source_id)})")
            lines.append("Every figure below was computed by the source and recomputed by an independent query; the classification says whether a change is carried by one group, a few, or spread in proportion to size.")
            lines.append(swept.to_markdown())
            lines.append("")
        lines.append("## Evaluation")
        counts = self.score()
        lines.append(", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no questions")
        lines.append("")
        if self.notes:
            lines.append("Diagnostics:")
            lines.extend(f"- {note}" for note in self.notes)
            lines.append("")
        lines.append("| id | question | outcome | cause | detail | agent query | routed to |")
        lines.append("|---|---|---|---|---|---|---|")
        ref_by_id = {r.question_id: r for r in self.references}
        for q in self.questions:
            g = next((x for x in self.graded if x.question_id == q.id), None)
            answers = self.answers.get(q.id, ())
            query = next((a.query for a in answers if a.query), None)
            routed = next((a.datasource for a in answers if a.datasource), "")
            detail = (g.detail if g else ref_by_id.get(q.id, Reference(q.id, "failed")).note)[:80]
            rules_broken = ", ".join(g.violations) if g and g.violations else ""
            lines.append(f"| {q.id} | {q.text} | {g.outcome if g else ''} | {g.cause if g else ''} | {detail}{'; broke ' + rules_broken if rules_broken else ''} | {'yes' if query else 'no'} | {routed} |")
        lines.append("")
        if self.rules:
            lines.append("Instruction compliance (rules the instructions state, checked on every answer):")
            lines.append("")
            lines.append("| rule | broken by | questions | the instruction |")
            lines.append("|---|---|---|---|")
            for row in self.compliance():
                lines.append(f"| {row['rule']} | {row['violations']} of {row['answers']} | {', '.join(q.rsplit('.', 1)[-1] for q in row['questions'])} | {row['text'][:120]} |")
            lines.append("")
        by_skill = self.by_skill()
        if by_skill:
            lines.append("Outcomes by skill:")
            lines.append("")
            lines.append("| skill | questions | correct | partial | wrong | other |")
            lines.append("|---|---|---|---|---|---|")
            for skill, counts in by_skill.items():
                lines.append(f"| {skill} | {counts['questions']} | {counts['correct']} | {counts['partial']} | {counts['wrong']} | {counts['other']} |")
            lines.append("")
        if self.analysis:
            lines.append("## Analysis (RLM)")
            grade_by_id = {g.question_id: g for g in self.graded}
            for item in self.analysis:
                g = grade_by_id.get(item.question_id)
                verdict = f" ({g.outcome}{', ' + g.cause if g and g.cause else ''})" if g else ""
                lines.append(f"- {item.question_id}{verdict}: {item.question}")
                lines.append(f"    - {item.explanation}")
                if item.proposed_change:
                    lines.append(f"    - proposed change: {item.proposed_change}")
            lines.append("")
        if any(a.steps for answers in self.answers.values() for a in answers):
            lines.append("## Agent run steps")
            lines.append("What the agent did per question: the functions it called, the language of the query it executed, the source it routed to.")
            for q in self.questions:
                for attempt, a in enumerate(self.answers.get(q.id, ()), start=1):
                    if not a.steps:
                        continue
                    counts = Counter(s.name for s in a.steps if s.name)
                    functions = ", ".join(f"{name} x{n}" if n > 1 else name for name, n in sorted(counts.items())) or "none"
                    lines.append(f"- {q.id} (attempt {attempt}): {len(a.steps)} step(s); functions: {functions}; query language: {a.language or 'none'}; executed: {'yes' if a.executed else 'no'}; routed to: {a.datasource or 'unknown'}; status: {a.status}; {a.seconds}s")
            lines.append("")
        if self.knowledge:
            lines.extend(self._knowledge_lines())
        lines.append("## Suggested changes")
        lines.append("### Agent instructions")
        lines.append("```")
        lines.append(self.suggestions.agent_instructions)
        lines.append("```")
        for source in s.datasources:
            description = self.suggestions.datasource_descriptions.get(source.id)
            if description:
                lines.append(f"### Description for {source.name or source.id}")
                lines.append(description)
            text = self.suggestions.datasource_instructions.get(source.id)
            if text and text != source.instructions:
                lines.append(f"### Data-source instructions: {source.name or source.id}")
                lines.append("```")
                lines.append(text)
                lines.append("```")
            shots = self.suggestions.fewshots.get(source.id) or ()
            if shots:
                lines.append(f"### Few-shots for {source.name or source.id} ({len(shots)})")
                for shot in shots:
                    lines.append(f"- Q: {shot.question}")
                    lines.append(f"  ```sql\n  {shot.query}\n  ```")
        for note in self.suggestions.notes:
            lines.append(f"- {note}")
        lines.append("")
        lines.append("## Method")
        lines.append("References were computed by executing generated queries against the sources, never by a model. Grades compare the agent's executed query where the run steps exposed it, else its prose figures, with a 0.5% tolerance. Repeat the evaluation three times before believing a delta.")
        return "\n".join(lines)


def _same_source(routed: str, source: AgentDataSource) -> bool:
    key = re.sub(r"[^a-z0-9]", "", routed.casefold())
    return any(key and key == re.sub(r"[^a-z0-9]", "", n.casefold()) for n in (source.id, source.name, source.item_id or "") if n)


def _with_routing(graded: Graded, question: Question, answers: Sequence[AgentAnswer], snapshot: AgentSnapshot) -> Graded:
    """On a multi-source agent, a failed answer that came from another source is a routing failure."""
    if len(snapshot.datasources) < 2 or graded.outcome == "correct":
        return graded
    routed = next((a.datasource for a in answers if a.datasource), None)
    source = next((s for s in snapshot.datasources if s.id == question.source_id), None)
    if not routed or source is None or _same_source(routed, source):
        return graded
    detail = f"routed to {routed}; the question is about {source.name or source.id}. {graded.detail}".strip()
    return replace(graded, cause="misrouted", detail=detail[:200])


def _with_policy(graded: Graded, question: Question, answers: Sequence[AgentAnswer], excluded: Collection[str]) -> Graded:
    """A declined question on a topic the instructions exclude is policy; SQL generated but not executed is said."""
    if graded.outcome == "abstained" and excluded and _mentions_excluded(question.text, excluded):
        graded = replace(graded, cause="abstained_by_policy", detail="the instructions put this topic out of scope, and the agent declined")
    if graded.outcome in {"abstained", "incomplete", "wrong", "partial"} and any(a.query and not a.executed for a in answers):
        graded = replace(graded, detail=f"{graded.detail}; SQL was generated but not executed".strip("; "))
    return graded


def review_agent(
    snapshot: AgentSnapshot,
    schemas: Sequence[SourceSchema],
    executors: Mapping[str, Any],
    ask: Callable[[str], AgentAnswer | str],
    *,
    years: Mapping[str, Sequence[int]] | None = None,
    top: int = 10,
    limit_per_source: int = 8,
    repetitions: int = 1,
    context: ReviewContext | None = None,
    knowledge: Any = None,
    hints: bool = True,
    sweep_budget: int = 0,
) -> ReviewReport:
    """The whole review: diagnose, generate, reference, ask, grade, suggest.

    ``hints`` runs :func:`discover_hints` on every lakehouse source and adds
    what it finds to the findings with basis ``"data"``; the suggested
    data-source instructions carry their instruction lines. A positive
    ``sweep_budget`` runs :func:`fabric_rlm.sweep.sweep` on every lakehouse
    source with that many queries and reports what moved.

    ``knowledge`` is what ``RLM.learn`` returned; the report summarises it.

    ``context`` is what the reviewer states about the agent (scope,
    priorities, definitions, own questions, notes); see :class:`ReviewContext`.

    With ``repetitions`` above one each question is asked that many times;
    an outcome that changes between runs is graded ``inconsistent``, which
    the routing guidance names as the sign of an ambiguous source choice.
    """
    findings = diagnose(snapshot, schemas)
    notes: list[str] = []
    excluded_by_source: dict[str, set[str]] = {}
    for schema in schemas:
        source = next((s for s in snapshot.datasources if s.id == schema.source_id), None)
        instructions = (source.instructions if source else "") + "\n" + snapshot.instructions
        excluded_by_source[schema.source_id] = excluded_terms(instructions)
        left_out = excluded_tables(schema, instructions)
        if left_out:
            notes.append(f"{_source_label(snapshot, schema.source_id)}: out of scope by the instructions, not asked about: {', '.join(left_out)}")
    if years is None:
        years = {}
        for schema in schemas:
            executor = executors.get(schema.source_id)
            if isinstance(executor, LakehouseExecutor):
                source = next((s for s in snapshot.datasources if s.id == schema.source_id), None)
                label = _source_label(snapshot, schema.source_id)
                try:
                    years[schema.source_id] = discover_years(executor, schema, source, snapshot.instructions)
                    notes.append(f"{label}: complete years {years[schema.source_id] or 'none found'}")
                except Exception as exc:  # noqa: BLE001 - the reason is the diagnostic
                    years[schema.source_id] = []
                    notes.append(f"{label}: year discovery failed: {type(exc).__name__}: {str(exc)[:300]}")
    discovered: dict[str, dict[str, dict[str, Any]]] = {}
    for schema in schemas:
        executor = executors.get(schema.source_id)
        if isinstance(executor, LakehouseExecutor) and years.get(schema.source_id):
            try:
                discovered[schema.source_id] = discover_drivers(executor, schema, snapshot, years[schema.source_id], context)
            except Exception as exc:  # noqa: BLE001 - the templated set still works without it
                notes.append(f"{_source_label(snapshot, schema.source_id)}: discovery of names and periods failed: {type(exc).__name__}: {str(exc)[:200]}")
            for table, entry in discovered.get(schema.source_id, {}).items():
                for error in entry.get("errors", []):
                    notes.append(f"{_source_label(snapshot, schema.source_id)}: discovery on {table}: {error}")
    data_hints: list[Finding] = []
    if hints:
        for schema in schemas:
            executor = executors.get(schema.source_id)
            if executor is None or schema.kind == "semantic_model":
                continue
            started = time.monotonic()
            try:
                found_hints = discover_hints(executor, schema, snapshot, years.get(schema.source_id), discovered=discovered.get(schema.source_id), context=context)
                data_hints.extend(found_hints)
                notes.append(f"{_source_label(snapshot, schema.source_id)}: {len(found_hints)} hint(s) from the data in {round(time.monotonic() - started)} s")
            except Exception as exc:  # noqa: BLE001 - hints are a bonus
                notes.append(f"{_source_label(snapshot, schema.source_id)}: hints from the data failed: {type(exc).__name__}: {str(exc)[:200]}")
    findings = tuple(findings) + tuple(data_hints)
    sweeps: list[Any] = []
    if sweep_budget > 0:
        from .sweep import sweep as run_sweep, verify_sweep

        for schema in schemas:
            executor = executors.get(schema.source_id)
            if executor is None or schema.kind == "semantic_model" or not years.get(schema.source_id):
                continue
            started = time.monotonic()
            try:
                swept = run_sweep(executor, schema, snapshot, years[schema.source_id], context=context, budget=sweep_budget)
                mismatches = verify_sweep(swept, executor)
                sweeps.append(swept)
                notes.append(f"{_source_label(snapshot, schema.source_id)}: sweep found {len(swept.findings)} material movement(s) in {swept.queries} queries and {round(time.monotonic() - started)} s; {len(mismatches)} figure(s) failed to recompute")
                notes.extend(f"{_source_label(snapshot, schema.source_id)}: sweep mismatch: {m}" for m in mismatches[:5])
            except Exception as exc:  # noqa: BLE001 - the sweep is a bonus
                notes.append(f"{_source_label(snapshot, schema.source_id)}: sweep failed: {type(exc).__name__}: {str(exc)[:200]}")
    questions = generate_questions(snapshot, schemas, years=years, top=top, limit_per_source=limit_per_source, context=context, discovered=discovered)
    if not questions:
        notes.extend(explain_no_questions(snapshot, schemas, years))
    questions, references = derive_filter_questions(questions, build_references(questions, executors))
    rules_by_source: dict[str, list[Rule]] = {}
    channel_words: dict[str, set[str]] = {}
    for schema in schemas:
        source = next((s for s in snapshot.datasources if s.id == schema.source_id), None)
        instructions = (source.instructions if source else "") + "\n" + snapshot.instructions
        rules_by_source[schema.source_id] = extract_rules(instructions)
        if schema.kind != "semantic_model":
            vocabulary = build_vocabulary(snapshot, schema, context)
            channel_words[schema.source_id] = {phrase.casefold() for phrase in vocabulary.tables.values()} | {a.casefold() for a in vocabulary.abbreviations.values()} | {w for phrase in vocabulary.tables.values() for w in phrase.casefold().split() if len(w) > 3}
        if rules_by_source[schema.source_id]:
            notes.append(f"{_source_label(snapshot, schema.source_id)}: {len(rules_by_source[schema.source_id])} checkable rule(s) in the instructions: {', '.join(r.id for r in rules_by_source[schema.source_id])}")
    ref_by_id = {r.question_id: r for r in references}
    answers: dict[str, tuple[AgentAnswer, ...]] = {}
    graded: list[Graded] = []
    for question in questions:
        collected: list[AgentAnswer] = []
        grades: list[Graded] = []
        for _ in range(max(1, repetitions)):
            try:
                raw = ask(question.text)
                answer = raw if isinstance(raw, AgentAnswer) else AgentAnswer(text=str(raw))
            except Exception as exc:  # noqa: BLE001 - an agent error is an outcome
                answer = AgentAnswer(text=f"ERROR: {type(exc).__name__}: {exc}")
            collected.append(answer)
            agent_rows = None
            executor = executors.get(question.source_id)
            if answer.query and answer.executed and answer.language == "sql" and isinstance(executor, LakehouseExecutor):
                try:
                    agent_rows = executor.run_agent_sql(answer.query)
                except Exception:  # noqa: BLE001 - the agent's SQL may not run outside its endpoint
                    agent_rows = None
            grades.append(grade(question, ref_by_id[question.id], answer, agent_rows=agent_rows))
        answers[question.id] = tuple(collected)
        outcomes = Counter(g.outcome for g in grades)
        if len(outcomes) > 1:
            worst = min(grades, key=lambda g: ("correct", "partial", "abstained", "incomplete", "wrong", "no_reference").index(g.outcome) * -1)
            final = Graded(question.id, worst.outcome, "inconsistent", f"outcomes across {len(grades)} runs: {dict(outcomes)}", worst.matched, worst.expected, worst.query_checked)
        else:
            final = grades[0]
        final = _with_policy(_with_routing(final, question, collected, snapshot), question, collected, excluded_by_source.get(question.source_id, ()))
        source_rules = rules_by_source.get(question.source_id, ())
        graded.append(replace(final, violations=check_rules(source_rules, question, collected, channel_words.get(question.source_id, ()))))
    suggestions = suggest(snapshot, schemas, findings, questions, references, graded)
    return ReviewReport(snapshot, tuple(schemas), findings, questions, references, answers, tuple(graded), suggestions, notes=tuple(notes), context=context, knowledge=summarize_knowledge(knowledge, schemas, snapshot) if knowledge is not None else None, discovered=discovered, rules=tuple(r for rules in rules_by_source.values() for r in rules), sweeps=tuple(sweeps))


# --------------------------------------------------------------------------- #
# Deeper analysis with the RLM: proposed questions with verified references,
# explanations of the failures
# --------------------------------------------------------------------------- #


def _schema_digest(schemas: Sequence[SourceSchema], *, tables: int = 40, columns: int = 24) -> str:
    lines = []
    for schema in schemas:
        for name, cols in list(schema.tables.items())[:tables]:
            lines.append(f"{name}: {', '.join(list(cols)[:columns])}{', ...' if len(cols) > columns else ''}")
        if schema.measures:
            lines.append(f"measures: {', '.join(schema.measures[:columns])}")
    return "\n".join(lines)


_PROPOSE_TASK = (
    "Propose questions a business user would ask this data agent, written the way such a user writes: "
    "plain business language, never a table or column name, using the business terms in the brief for "
    "channels, measures and attributes. Spread them over the skills a data agent's author wants tested: "
    "a filtered total, a ranking, a comparison between two periods, a ratio or KPI the definitions state, "
    "a phrasing that uses an abbreviation or an ambiguous term the instructions define, a driver question "
    "(why a figure moved and which products or customers drove it), a question about a named entity from the "
    "brief, and one topic the instructions put out of scope (where the right answer declines). Each question must say which period "
    "it means, must be answerable from the sources (except the out-of-scope one), and must not repeat a "
    "question already asked. Return exactly the requested count as a list of plain strings."
)
_EXPLAIN_TASK = (
    "Explain, in at most three sentences, why the agent's answer differs from the reference for this "
    "question, using the agent's query, the reference query and the reference rows. Then propose the "
    "smallest change to the agent instructions, the data-source instructions or a few-shot that would "
    "make the agent answer correctly; quote the line to add or change."
)


def _rlm_proposer(lm: Any, *, max_turns: int, timeout: float) -> Callable[[str, int], list[str]]:
    def propose(brief: str, count: int) -> list[str]:
        from .runtime import RLM

        result = RLM.task(_PROPOSE_TASK, inputs={"brief": brief, "count": count}, outputs={"questions": list}, lm=lm, max_turns=max_turns, timeout=timeout).run()
        raw = (result.payload or {}).get("questions") or []
        if isinstance(raw, str):
            raw = [line.strip("-* ") for line in raw.splitlines()]
        return [str(q).strip() for q in raw if str(q).strip()][:count]

    return propose


def _rlm_verifier(lm: Any, handles: Mapping[str, Any], knowledge: Any, *, max_turns: int, timeout: float, results: list[Any] | None = None) -> Callable[[str], tuple[str, str]]:
    """Verified references through the RLM; every solve lands in ``results`` (with evidence when a package is bound) so the package can learn from them."""

    def verify(question: str) -> tuple[str, str]:
        from .verify import verified_task

        # a bound knowledge package brings its own source handles; naming them again as inputs is a conflict
        extra = {"capture_evidence": True} if knowledge is not None else {}
        verified = verified_task(question, outputs=["answer"], inputs=None if knowledge is not None else dict(handles), knowledge=knowledge, lm=lm, max_turns=max_turns, timeout=timeout, **extra)
        if results is not None:
            results.extend(list(getattr(verified, "attempts", None) or [verified.result]))
        answer = (verified.result.payload or {}).get("answer", "")
        return str(verified.verdict), "" if answer is None else str(answer)

    return verify


def _rlm_explainer(lm: Any, *, timeout: float) -> Callable[[Mapping[str, Any]], Mapping[str, str]]:
    def explain(case: Mapping[str, Any]) -> Mapping[str, str]:
        from .runtime import RLM

        result = RLM.task(_EXPLAIN_TASK, inputs=dict(case), outputs={"explanation": str, "proposed_change": str}, lm=lm, max_turns=4, timeout=timeout).run()
        payload = result.payload or {}
        return {"explanation": str(payload.get("explanation", "") or ""), "proposed_change": str(payload.get("proposed_change", "") or "")}

    return explain


def deepen(
    report: ReviewReport,
    *,
    lm: Any = None,
    handles: Mapping[str, Any] | None = None,
    knowledge: Any = None,
    ask: Callable[[str], Any] | None = None,
    questions: int = 4,
    context: ReviewContext | None = None,
    max_turns: int = 8,
    timeout: float = 300.0,
    explain_limit: int = 6,
    propose: Callable[[str, int], list[str]] | None = None,
    verify: Callable[[str], tuple[str, str]] | None = None,
    explain: Callable[[Mapping[str, Any]], Mapping[str, str]] | None = None,
    enrich: Callable[[Any, Sequence[Any]], Any] | None = None,
    runs: list[Any] | None = None,
) -> ReviewReport:
    """Deeper analysis with the RLM, on top of a review.

    The RLM proposes ``questions`` natural questions from the scope, the
    priorities and the schema; each gets a reference from ``verified_task``
    (two blind solves over the sources that must agree, reconciled on
    disagreement), the agent is asked, and the answer is graded against the
    figures of the verified answer. Then the RLM explains every question
    that is not correct and proposes the smallest change. ``propose``,
    ``verify`` and ``explain`` can be supplied for testing; the defaults use
    the RLM with ``lm``. Supplied ground truth outranks generated ground
    truth, and both outrank the RLM's: proposed questions come last.

    When a ``knowledge`` package is bound, every verified solve runs with
    evidence capture and the package learns from those runs through
    ``RLM.enrich`` (``enrich`` replaces it for testing; ``runs`` seeds the
    list of results): the lessons that appear are reported as learned from
    this review and the enriched package is returned on the report as
    ``learned_knowledge`` for the caller to save.
    """
    context = context or report.context
    runs = runs if runs is not None else []
    if propose is None or verify is None or explain is None:
        if lm is None:
            raise ValueError("deepen needs an lm, or propose, verify and explain callables")
        propose = propose or _rlm_proposer(lm, max_turns=4, timeout=timeout)
        verify = verify or _rlm_verifier(lm, handles or {}, knowledge, max_turns=max_turns, timeout=timeout, results=runs)
        explain = explain or _rlm_explainer(lm, timeout=timeout)
    notes = list(report.notes)
    target = next((s.source_id for s in report.schemas if s.kind != "semantic_model"), report.schemas[0].source_id if report.schemas else "source")
    vocabulary_lines: list[str] = []
    excluded: set[str] = set()
    for schema in report.schemas:
        if schema.kind == "semantic_model":
            continue
        vocabulary_lines.extend(build_vocabulary(report.snapshot, schema, context).lines())
        source = next((s for s in report.snapshot.datasources if s.id == schema.source_id), None)
        excluded.update(excluded_tables(schema, (source.instructions if source else "") + "\n" + report.snapshot.instructions))
    facts_lines: list[str] = []
    for entries in (report.discovered or {}).values():
        for table, entry in entries.items():
            for role, names in (entry.get("top") or {}).items():
                facts_lines.append(f"{humanize_table(table)}, top {role}: {', '.join(names)}")
            if entry.get("drop"):
                drop = entry["drop"]
                facts_lines.append(f"{humanize_table(table)} dropped from {_month_name(drop['before'])} to {_month_name(drop['after'])}")
    brief = "\n\n".join(
        part
        for part in [
            context.as_prompt() if context else "",
            f"Agent instructions:\n{report.snapshot.instructions[:3000]}",
            *(f"Source {s.name or s.id} instructions:\n{s.instructions[:2000]}" for s in report.snapshot.datasources),
            "Business terms (use these words, not the schema names):\n" + "\n".join(vocabulary_lines[:60]),
            "Real names and periods in the data (ask about these by name, and about why things changed):\n" + "\n".join(facts_lines) if facts_lines else "",
            "Out-of-scope topics by the instructions: " + ", ".join(humanize_table(t) for t in sorted(excluded)) if excluded else "",
            "Schema digest:\n" + _schema_digest(report.schemas),
            "Facts about the data the agent may not know (test whether it copes with them):\n" + "\n".join(f.message for f in report.findings if f.basis == "data") if any(f.basis == "data" for f in report.findings) else "",
            "What moved in the data, measured by the source (ask why, and whether the agent finds the same drivers):\n" + "\n".join(line for swept in report.sweeps for line in swept.lines()[:12]) if report.sweeps else "",
            "Already asked:\n" + "\n".join(q.text for q in report.questions),
        ]
        if part
    )
    proposed: list[str] = []
    if questions > 0:
        try:
            proposed = propose(brief, questions)
        except Exception as exc:  # noqa: BLE001 - the reason is the diagnostic
            notes.append(f"the RLM could not propose questions: {type(exc).__name__}: {str(exc)[:200]}")
    new_questions: list[Question] = []
    new_references: list[Reference] = []
    new_answers: dict[str, tuple[AgentAnswer, ...]] = {}
    new_graded: list[Graded] = []
    for number, text in enumerate(proposed, start=1):
        qid = f"{target}.d{number}"
        try:
            verdict, answer = verify(text)
        except Exception as exc:  # noqa: BLE001
            verdict, answer = "failed", f"{type(exc).__name__}: {str(exc)[:200]}"
        question = Question(id=qid, source_id=target, kind="deep", text=text, spec={"kind": "deep", "verdict": verdict, "reference_answer": answer[:2000]}, reference_query="", execution={"kind": "supplied_text", "text": answer}, skill="proposed")
        figures = _reference_figures(answer) if verdict != "failed" else []
        if figures:
            reference = Reference(qid, "ok", tuple({"value": figure} for figure in figures), f"RLM reference ({verdict}): {answer[:160]}")
        elif verdict == "failed":
            reference = Reference(qid, "failed", note=f"the RLM solves did not agree: {answer[:160]}")
        else:
            reference = Reference(qid, "failed", note=f"the RLM's answer carries no figures: {answer[:160]}")
        new_questions.append(question)
        new_references.append(reference)
        if ask is None:
            new_graded.append(Graded(qid, "no_reference", "not_asked", "no asker was given"))
            continue
        try:
            raw = ask(text)
            agent = raw if isinstance(raw, AgentAnswer) else AgentAnswer(text=str(raw))
        except Exception as exc:  # noqa: BLE001 - an agent error is an outcome
            agent = AgentAnswer(text=f"ERROR: {type(exc).__name__}: {exc}")
        new_answers[qid] = (agent,)
        new_graded.append(_with_routing(grade(question, reference, agent), question, [agent], report.snapshot))
    all_questions = tuple(report.questions) + tuple(new_questions)
    all_references = tuple(report.references) + tuple(new_references)
    all_answers: dict[str, tuple[AgentAnswer, ...]] = {**dict(report.answers), **new_answers}
    all_graded = tuple(report.graded) + tuple(new_graded)
    ref_by_id = {r.question_id: r for r in all_references}
    analysis = list(report.analysis)
    explained = {a.question_id for a in analysis}
    candidates = [g for g in all_graded if g.outcome not in {"correct", "no_reference"} and g.question_id not in explained][:explain_limit]
    for g in candidates:
        question = next(q for q in all_questions if q.id == g.question_id)
        answers = all_answers.get(g.question_id, ())
        reference = ref_by_id.get(g.question_id)
        source = next((s for s in report.snapshot.datasources if s.id == question.source_id), None)
        case = {
            "question": question.text,
            "outcome": g.outcome,
            "cause": g.cause,
            "detail": g.detail,
            "agent_answer": (answers[0].text if answers else "")[:3000],
            "agent_query": (answers[0].query if answers else "") or "",
            "reference_query": question.reference_query,
            "reference_rows": json.dumps([dict(r) for r in (reference.rows if reference else ())][:10], default=str),
            "agent_instructions": report.snapshot.instructions[:3000],
            "source_instructions": (source.instructions if source else "")[:3000],
        }
        try:
            verdict = explain(case)
            analysis.append(Analysis(g.question_id, question.text, str(verdict.get("explanation", "") or ""), str(verdict.get("proposed_change", "") or "")))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"the RLM could not explain {g.question_id}: {type(exc).__name__}: {str(exc)[:200]}")
    suggestions = suggest(report.snapshot, report.schemas, report.findings, all_questions, all_references, all_graded)
    knowledge_summary, learned_knowledge = report.knowledge, report.learned_knowledge
    if runs and knowledge is not None:
        try:
            if enrich is None:
                from .runtime import RLM

                enrich = RLM.enrich
            enriched = enrich(knowledge, list(runs))
            before = {str(getattr(lesson, "lesson_id", "")) for lesson in (getattr(getattr(knowledge, "package", knowledge), "lessons", ()) or ())}
            knowledge_summary = summarize_knowledge(enriched, report.schemas, report.snapshot)
            knowledge_summary["learned"] = [_lesson_dict(lesson) for lesson in (getattr(getattr(enriched, "package", enriched), "lessons", ()) or ()) if str(getattr(lesson, "lesson_id", "")) not in before]
            knowledge_summary["runs"] = len(runs)
            learned_knowledge = enriched
        except Exception as exc:  # noqa: BLE001 - learning is a bonus; the review stands without it
            notes.append(f"the package could not learn from the RLM's {len(runs)} run(s): {type(exc).__name__}: {str(exc)[:200]}")
    return replace(report, questions=all_questions, references=all_references, answers=all_answers, graded=all_graded, suggestions=suggestions, notes=tuple(notes), analysis=tuple(analysis), knowledge=knowledge_summary, learned_knowledge=learned_knowledge)


# --------------------------------------------------------------------------- #
# Writers: applying suggestions is explicit and goes to the draft stage
# --------------------------------------------------------------------------- #


class RestAgentWriter:
    def __init__(self, workspace_id: str, agent_id: str, token_provider: Callable[[], str]) -> None:
        self.workspace_id, self.agent_id, self._token = workspace_id, agent_id, token_provider

    def _base(self) -> str:
        return f"{FABRIC_API}/workspaces/{self.workspace_id}/dataAgents/{self.agent_id}/staging"

    def update_agent_instructions(self, text: str) -> None:
        _http_json(f"{self._base()}/settings", self._token(), method="PATCH", body={"aiInstructions": text})

    def update_datasource_instructions(self, datasource_id: str, text: str) -> None:
        _http_json(f"{self._base()}/datasources/{datasource_id}", self._token(), method="PATCH", body={"instructions": text})

    def update_datasource_description(self, datasource_id: str, text: str) -> None:
        _http_json(f"{self._base()}/datasources/{datasource_id}", self._token(), method="PATCH", body={"description": text})

    def add_fewshots(self, datasource_id: str, shots: Sequence[FewShot]) -> None:
        for shot in shots:
            _http_json(f"{self._base()}/datasources/{datasource_id}/fewshots", self._token(), method="POST", body={"question": shot.question, "query": shot.query})


class SdkAgentWriter:
    """Writes to the agent's staging (draft) configuration through the SDK.

    Prefers the public-API methods (``update_settings``, ``list_datasources``
    and the datasource handle's ``update_configuration`` and ``add_fewshots``)
    and falls back to the legacy workload-host methods on older SDKs. Nothing
    here publishes.
    """

    def __init__(self, management: Any) -> None:
        self._management = management

    def _public(self) -> bool:
        return hasattr(self._management, "list_datasources") and hasattr(self._management, "update_settings")

    def _datasource(self, datasource_id: str) -> Any:
        handles = self._management.list_datasources(stage="staging") if self._public() else self._management.get_datasources()
        for datasource in handles:
            if str(getattr(datasource, "_id", "")) == datasource_id:
                return datasource
        raise KeyError(datasource_id)

    def update_agent_instructions(self, text: str) -> None:
        if self._public():
            self._management.update_settings(ai_instructions=text)
        else:
            self._management.update_configuration(instructions=text)

    def update_datasource_instructions(self, datasource_id: str, text: str) -> None:
        self._datasource(datasource_id).update_configuration(instructions=text)

    def update_datasource_description(self, datasource_id: str, text: str) -> None:
        datasource = self._datasource(datasource_id)
        if self._public():
            datasource.update_configuration(description=text)
        else:
            datasource.update_description(text)

    def add_fewshots(self, datasource_id: str, shots: Sequence[FewShot]) -> None:
        self._datasource(datasource_id).add_fewshots({shot.question: shot.query for shot in shots})


def apply_suggestions(writer: Any, suggestions: Suggestions, *, agent_instructions: bool = True, datasource_instructions: bool = True, descriptions: bool = True, fewshots: bool = True) -> list[str]:
    """Write the suggestions to the agent's draft stage; returns what was applied."""
    applied: list[str] = []
    if agent_instructions and suggestions.agent_instructions:
        writer.update_agent_instructions(suggestions.agent_instructions)
        applied.append("agent instructions")
    if datasource_instructions:
        for source_id, text in suggestions.datasource_instructions.items():
            writer.update_datasource_instructions(source_id, text)
            applied.append(f"instructions of {source_id}")
    if descriptions:
        for source_id, text in suggestions.datasource_descriptions.items():
            writer.update_datasource_description(source_id, text)
            applied.append(f"description of {source_id}")
    if fewshots:
        for source_id, shots in suggestions.fewshots.items():
            if shots:
                writer.add_fewshots(source_id, shots)
                applied.append(f"{len(shots)} few-shot(s) on {source_id}")
    return applied
