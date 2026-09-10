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

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
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


@dataclass(frozen=True)
class SourceSchema:
    """What a source actually exposes, from a fabric-rlm profile or a catalog."""

    source_id: str
    kind: str
    tables: Mapping[str, tuple[str, ...]]
    measures: tuple[str, ...] = ()
    relationships: tuple[tuple[str, str, str, str], ...] = ()  # from_table, from_col, to_table, to_col

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
) -> SourceSchema:
    return SourceSchema(
        source_id=source_id,
        kind=kind,
        tables={str(name): tuple(str(c) for c in columns) for name, columns in tables.items()},
        measures=tuple(str(m) for m in measures),
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
        for name, entry in raw.items():
            columns = entry.get("columns") if isinstance(entry, Mapping) else None
            if isinstance(columns, Mapping):
                tables[str(name)] = list(columns)
        return SourceSchema(sid, "lakehouse", {t: tuple(c) for t, c in tables.items()})
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


_LEAF_ELEMENT = re.compile(r"(column|measure|parameter|returnvalue|field)", re.IGNORECASE)
_TABLE_ELEMENT = re.compile(r"(table|view|function|entity|dataset)", re.IGNORECASE)


def _selected_table_paths(fetch: Callable[[str | None, str | None], Mapping[str, Any] | None]) -> list[str]:
    """Paths (``schema/table`` or ``table``) of the selected tables in a datasource's elements tree.

    ``fetch(root_id, continuation_token)`` returns one page of elements
    (``{"value": [...], "continuationToken": ...}``). Columns and measures
    are leaves; a container (a schema) is walked, a table is not.
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
            if _LEAF_ELEMENT.search(kind):
                continue
            name = str(element.get("displayName") or element.get("name") or "")
            path = f"{prefix}/{name}" if prefix else name
            selected = element.get("isSelected") if "isSelected" in element else element.get("is_selected")
            if selected and name:
                found.append(path)
            if not _TABLE_ELEMENT.search(kind) and element.get("id") is not None and depth < 4:
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
                sources.append(_source_from_entry(entry, shots, fallback_id=str(getattr(handle, "_id", "") or ""), selected_tables=selected))
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


def _schema_mentions(line: str, schemas: Sequence[SourceSchema]) -> dict[str, list[str]]:
    """Schema identifiers a line names, grouped by the source that has them."""
    found: dict[str, list[str]] = {}
    candidates = {m.group(1) for m in _IDENTIFIER.finditer(line)} | {m.group(0) for m in _DAX_REFERENCE.finditer(line)}
    for schema in schemas:
        known = schema.identifiers()
        hits = sorted({c for c in candidates if c.casefold() in known and c.casefold() not in _STOPWORDS and len(c) > 2})
        if hits:
            found[schema.source_id] = hits
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


def _scoped_facts(schema: SourceSchema, instructions: str) -> list[str]:
    """Fact tables in the agent's declared scope first; all of them when it names none."""
    facts = _fact_tables(schema)
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


def declared_from_snapshot(snapshot: AgentSnapshot, schemas: Sequence[SourceSchema]) -> dict[str, dict[str, object]]:
    """What the agent's instructions declare, in ``RLM.learn(declared=...)`` form."""
    declared: dict[str, dict[str, object]] = {}
    by_id = {s.source_id: s for s in schemas}
    for source in snapshot.datasources:
        definitions = extract_definitions(source.instructions)
        definitions.update({k: v for k, v in extract_definitions(snapshot.instructions).items() if k not in definitions})
        if definitions and source.id in by_id:
            declared[source.id] = {"definitions": {k: v[:250] for k, v in definitions.items()}}
    return declared


# --------------------------------------------------------------------------- #
# Questions and references
# --------------------------------------------------------------------------- #

_MEASURE_HINT = re.compile(r"(amount|qty|quantity|units|revenue|sales|cost|price|margin|total|profit|value|hours|minutes|count)", re.IGNORECASE)
_KEY_HINT = re.compile(r"(key|id)$", re.IGNORECASE)
_DATE_KEY = re.compile(r"date(key)?$", re.IGNORECASE)
_YEAR_COLUMN = re.compile(r"^(calendar)?year$", re.IGNORECASE)
_ATTRIBUTE_HINT = re.compile(r"(name|country|region|group|category|segment|type|class|status|city|state|line|plant)", re.IGNORECASE)
_ORDER_ID_HINT = re.compile(r"(ordernumber|orderid|order_id|order_number|ticket_id|invoice)", re.IGNORECASE)
_JOIN_LINE = re.compile(r"(?:dbo\.)?([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*(?:->|→|to|=)\s*(?:dbo\.)?([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class Question:
    id: str
    source_id: str
    kind: str  # total_by_year | top_n | breakdown | distinct_by_year | yoy | combined_total_by_year
    text: str
    spec: Mapping[str, Any]
    reference_query: str  # what a few-shot would carry: T-SQL for a lakehouse, DAX for a model
    execution: Mapping[str, Any] = field(default_factory=dict)  # what the executor runs
    alternates: tuple[tuple[str, Mapping[str, Any]], ...] = ()  # (label, execution) narrower scopes


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
    """``<Name>Key`` in a fact table to ``dim<name>.<Name>Key`` when that exists."""
    joins: dict[tuple[str, str], tuple[str, str]] = {}
    tables = {t.casefold(): t for t in schema.tables}
    for table, columns in schema.tables.items():
        for column in columns:
            if not column.casefold().endswith("key") or column.casefold() == "datekey":
                continue
            stem = column[:-3].casefold()
            for candidate in (f"dim{stem}", stem, f"dim_{stem}", f"{stem}s"):
                target = tables.get(candidate)
                if target and target != table and column in schema.tables[target]:
                    joins[(table, column)] = (target, column)
                    break
    return joins


def _fact_tables(schema: SourceSchema) -> list[str]:
    facts = []
    for table, columns in schema.tables.items():
        measures = [c for c in columns if _MEASURE_HINT.search(c) and not _KEY_HINT.search(c)]
        dates = [c for c in columns if _DATE_KEY.search(c)]
        if measures and dates and (table.casefold().startswith("fact") or len(measures) >= 2):
            facts.append(table)
    return sorted(facts, key=lambda t: (not t.casefold().startswith("fact"), t))


def _measure_columns(schema: SourceSchema, table: str) -> list[str]:
    preferred = ("salesamount", "revenue", "amount", "totalproductcost", "orderquantity", "quantity", "units")
    columns = [c for c in schema.tables[table] if _MEASURE_HINT.search(c) and not _KEY_HINT.search(c)]
    return sorted(columns, key=lambda c: (next((i for i, p in enumerate(preferred) if p in c.casefold()), 99), c))


def _date_join(schema: SourceSchema, table: str, joins: Mapping[tuple[str, str], tuple[str, str]]) -> tuple[str, str, str, str] | None:
    """(date column on the fact, date table, date key, year column)."""
    for column in schema.tables[table]:
        if not _DATE_KEY.search(column):
            continue
        target = joins.get((table, column))
        if target is None:
            for candidate in ("dimdate", "date", "dim_date", "calendar"):
                for name in schema.tables:
                    if name.casefold() == candidate and "DateKey" in schema.tables[name]:
                        target = (name, "DateKey")
                        break
                if target:
                    break
        if target is None:
            continue
        date_table, date_key = target
        year = next((c for c in schema.tables[date_table] if _YEAR_COLUMN.match(c)), None)
        if year:
            return column, date_table, date_key, year
    return None


def _attributes(schema: SourceSchema, table: str, joins: Mapping[tuple[str, str], tuple[str, str]]) -> list[tuple[str, str, str, str]]:
    """(fact key, dim table, dim key, attribute column) for grouping."""
    found = []
    for column in schema.tables[table]:
        target = joins.get((table, column))
        if target is None or _DATE_KEY.search(column):
            continue
        dim_table, dim_key = target
        for attribute in schema.tables[dim_table]:
            if _ATTRIBUTE_HINT.search(attribute) and not _KEY_HINT.search(attribute) and attribute != dim_key:
                found.append((column, dim_table, dim_key, attribute))
    return found


_LANGUAGE_VARIANT = re.compile(r"^(spanish|french|german|italian|portuguese|dutch|japanese|chinese)", re.IGNORECASE)


def _diverse_attributes(attributes: Sequence[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str]]:
    """Grouping attributes with one per dimension table first, English names before translated variants."""
    ranked = sorted(attributes, key=lambda a: (bool(_LANGUAGE_VARIANT.match(a[3])), not a[3].casefold().startswith("english")))
    chosen: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for attribute in ranked:
        if attribute[1] not in seen:
            chosen.append(attribute)
            seen.add(attribute[1])
    chosen.extend(a for a in ranked if a not in chosen)
    return chosen


def _order_column(schema: SourceSchema, table: str) -> str | None:
    return next((c for c in schema.tables[table] if _ORDER_ID_HINT.search(c)), None)


def _sql(spec: Mapping[str, Any], *, dialect: str) -> str:
    """Render a lakehouse question spec as T-SQL (few-shots) or DuckDB (execution)."""
    kind = spec["kind"]
    top = spec.get("top")
    facts: list[Mapping[str, Any]] = spec["facts"]
    by_year = kind in {"total_by_year", "distinct_by_year", "yoy", "combined_total_by_year"}

    def per_fact(fact: Mapping[str, Any]) -> str:
        select: list[str] = []
        group: list[str] = []
        joins: list[str] = []
        dt = fact["date"]
        joins.append(f"JOIN {dt['date_table']} d ON f.{dt['column']} = d.{dt['date_key']}")
        if by_year:
            select.append(f"d.{dt['year']} AS year")
            group.append(f"d.{dt['year']}")
        if kind in {"top_n", "breakdown"}:
            attr = fact["attribute"]
            joins.append(f"JOIN {attr['dim_table']} a ON f.{attr['fact_key']} = a.{attr['dim_key']}")
            select.append(f"a.{attr['column']} AS {attr['alias']}")
            group.append(f"a.{attr['column']}")
        if kind == "distinct_by_year":
            select.append(f"COUNT(DISTINCT f.{fact['order_column']}) AS value")
        else:
            select.append(f"SUM(f.{fact['measure']}) AS value")
        where = ""
        if by_year and spec.get("years"):
            where = f" WHERE d.{dt['year']} IN ({', '.join(str(int(y)) for y in spec['years'])})"
        elif spec.get("year") is not None:
            where = f" WHERE d.{dt['year']} = {int(spec['year'])}"
        return f"SELECT {', '.join(select)} FROM {fact['table']} f {' '.join(joins)}{where} GROUP BY {', '.join(group)}"

    order = "ORDER BY value DESC" if kind in {"top_n", "breakdown"} else "ORDER BY year"
    if len(facts) == 1:
        body = per_fact(facts[0])
    else:
        union = " UNION ALL ".join(f"({per_fact(fact)})" for fact in facts)
        keys = "year" if by_year else facts[0]["attribute"]["alias"]
        body = f"SELECT {keys}, SUM(value) AS value FROM ({union}) u GROUP BY {keys}"
    if kind == "top_n" and top:
        if dialect == "tsql":
            return body.replace("SELECT ", f"SELECT TOP {int(top)} ", 1) + f" {order}"
        return f"{body} {order} LIMIT {int(top)}"
    return f"{body} {order}"


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


def generate_questions(
    snapshot: AgentSnapshot,
    schemas: Sequence[SourceSchema],
    *,
    years: Mapping[str, Sequence[int]] | None = None,
    top: int = 10,
    limit_per_source: int = 8,
) -> tuple[Question, ...]:
    """Questions the sources can answer, each with the query that answers it.

    ``years`` maps a source id to the complete years its data covers (see
    :func:`discover_years`); without it the period questions are skipped.
    """
    questions: list[Question] = []
    sources = {s.id: s for s in snapshot.datasources}
    for schema in schemas:
        source = sources.get(schema.source_id)
        source_years = list((years or {}).get(schema.source_id) or [])
        if schema.kind == "semantic_model":
            questions.extend(_semantic_questions(schema, source_years, top=top, limit=limit_per_source))
            continue
        instructions = (source.instructions if source else "") + "\n" + snapshot.instructions
        joins = dict(_heuristic_joins(schema))
        joins.update(_joins_from_instructions(instructions, schema))
        facts = _scoped_facts(schema, instructions)
        per_fact = []
        for table in facts:
            date = _date_join(schema, table, joins)
            measures = _measure_columns(schema, table)
            if not date or not measures:
                continue
            per_fact.append({"table": table, "measure": measures[0], "date": {"column": date[0], "date_table": date[1], "date_key": date[2], "year": date[3]}, "attributes": _attributes(schema, table, joins), "order_column": _order_column(schema, table)})
        if not per_fact or not source_years:
            continue
        count = 0
        shared = [f for f in per_fact if f["measure"] == per_fact[0]["measure"]]
        latest = max(source_years)
        span = f"{source_years[0]} to {source_years[-1]}" if len(source_years) > 1 else str(source_years[0])
        prefix = f"{schema.source_id}"

        def add(kind: str, text: str, spec: Mapping[str, Any], alternates: Sequence[tuple[str, Mapping[str, Any]]] = ()) -> None:
            nonlocal count
            if count >= limit_per_source:
                return
            count += 1
            questions.append(Question(id=f"{prefix}.q{count}", source_id=schema.source_id, kind=kind, text=text, spec=spec, reference_query=_sql(spec, dialect="tsql"), execution={"kind": "sql", "sql": _sql(spec, dialect="duckdb")}, alternates=tuple(alternates)))

        measure_name = per_fact[0]["measure"]
        if len(shared) > 1:
            bare = [{k: v for k, v in f.items() if k != "attributes"} for f in shared]
            spec = {"kind": "combined_total_by_year", "facts": bare, "years": source_years}
            alternates = [(f["table"], {"kind": "sql", "sql": _sql({**spec, "facts": [f]}, dialect="duckdb")}) for f in bare]
            add("combined_total_by_year", f"What was total {measure_name} by year for {span}, across {' and '.join(f['table'] for f in shared)} combined?", spec, alternates)
        for fact in per_fact:
            base = {k: v for k, v in fact.items() if k != "attributes"}
            add("total_by_year", f"What was total {fact['measure']} in {fact['table']} by year for {span}?", {"kind": "total_by_year", "facts": [base], "years": source_years})
            if fact["order_column"]:
                add("distinct_by_year", f"How many distinct {fact['order_column']} values (orders) does {fact['table']} have per year for {span}?", {"kind": "distinct_by_year", "facts": [base], "years": source_years})
            if len(source_years) >= 2:
                add("yoy", f"What was the year-over-year change in total {fact['measure']} in {fact['table']} from {source_years[-2]} to {source_years[-1]}?", {"kind": "yoy", "facts": [base], "years": source_years[-2:]})
            for fact_key, dim_table, dim_key, attribute in _diverse_attributes(fact["attributes"])[:2]:
                attr = {"fact_key": fact_key, "dim_table": dim_table, "dim_key": dim_key, "column": attribute, "alias": attribute}
                add("top_n", f"What are the top {top} {attribute} values by {fact['measure']} in {fact['table']} for {latest}?", {"kind": "top_n", "facts": [{**base, "attribute": attr}], "year": latest, "top": top})
                add("breakdown", f"What was total {fact['measure']} in {fact['table']} by {attribute} for {latest}?", {"kind": "breakdown", "facts": [{**base, "attribute": attr}], "year": latest})
    return tuple(questions)


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
        if _ATTRIBUTE_HINT.search(c) and not _KEY_HINT.search(c)
    ][:3]
    count = 0
    for measure in list(schema.measures[:3]):
        if count >= limit:
            break
        if year_ref and years:
            spec = {"kind": "total_by_year", "measure": measure, "groupby": [year_ref], "filters": {year_ref: list(years)}}
            count += 1
            questions.append(Question(id=f"{schema.source_id}.q{count}", source_id=schema.source_id, kind="total_by_year", text=f"What was {measure} by year for {years[0]} to {years[-1]}?", spec=spec, reference_query=_dax(spec), execution={"kind": "aggregate", "measures": [measure], "groupby": [year_ref], "filters": {year_ref: list(years)}}))
        for attribute in attributes[:2]:
            if count >= limit:
                break
            filters = {year_ref: [max(years)]} if year_ref and years else {}
            spec = {"kind": "top_n", "measure": measure, "groupby": [attribute], "filters": filters, "top": top}
            count += 1
            period = f" in {max(years)}" if years else ""
            questions.append(Question(id=f"{schema.source_id}.q{count}", source_id=schema.source_id, kind="top_n", text=f"What are the top {top} {attribute} by {measure}{period}?", spec=spec, reference_query=_dax(spec), execution={"kind": "aggregate", "measures": [measure], "groupby": [attribute], "filters": filters, "order_by": measure, "top": top}))
    return questions


class LakehouseExecutor:
    """Runs lakehouse question SQL through a ``LakehouseSource.query``-shaped callable.

    ``query(sql, sources={alias: catalog_name})`` must return a mapping with
    ``columns`` and ``rows`` (row lists) or a list of row mappings. The
    generated SQL names tables by their catalog names, so ``sources`` maps
    each name to itself.
    """

    def __init__(self, query: Callable[..., Any], tables: Iterable[str]) -> None:
        self._query = query
        self._tables = list(tables)

    def run(self, execution: Mapping[str, Any]) -> list[dict[str, Any]]:
        sql = str(execution["sql"])
        used = [t for t in self._tables if re.search(rf"(?<![\w.]){re.escape(t)}(?![\w])", sql, re.IGNORECASE)]
        result = self._query(sql, sources={t: t for t in used})
        return _rows(result)

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
        column, date_table, date_key, year = date
        rows = executor.run({"sql": f"SELECT d.{year} AS year, COUNT(*) AS n FROM {table} f JOIN {date_table} d ON f.{column} = d.{date_key} GROUP BY d.{year} ORDER BY d.{year}"})
        counts = [(int(r["year"]), int(r["n"])) for r in rows if r.get("year") is not None]
        years = [y for y, _ in counts]
        if len(counts) >= 2 and counts[-1][1] < 0.5 * counts[-2][1]:
            # the last year is partial when it holds far fewer rows than the one before
            years = years[:-1]
        return years
    return []


def build_references(questions: Sequence[Question], executors: Mapping[str, Any]) -> tuple[Reference, ...]:
    """Execute every question's reference query; a failure is a status, not a guess."""
    references: list[Reference] = []
    for question in questions:
        executor = executors.get(question.source_id)
        if executor is None:
            references.append(Reference(question.id, "abstained", note="no executor for the source"))
            continue
        try:
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
        return AgentAnswer(text=text, query=query, language=language, datasource=datasource, seconds=round(time.time() - started, 1), steps=steps, status=status)


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
        return AgentAnswer(text=text, query=query, language=language, datasource=datasource, seconds=round(time.time() - started, 1), steps=tuple(steps), status=str(run.status))


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


def grade(question: Question, reference: Reference, answer: AgentAnswer | str, *, agent_rows: Sequence[Mapping[str, Any]] | None = None) -> Graded:
    """Grade an answer against its reference: by the executed query when it was run, else by the prose."""
    text = answer.text if isinstance(answer, AgentAnswer) else str(answer)
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
    if agent_rows is not None:
        if _rows_match(reference.rows, agent_rows):
            return Graded(question.id, "correct", "", "the agent's executed query returns the reference rows", len(reference.rows), len(reference.rows), True)
        for label, rows in reference.alternates.items():
            if _rows_match(rows, agent_rows):
                return Graded(question.id, "wrong", "narrower_scope", f"the agent's query returns {label} alone, not the combined scope asked for", 0, len(reference.rows), True)
        if len(agent_rows) < len(reference.rows):
            return Graded(question.id, "partial" if agent_rows else "wrong", "missing_rows", f"the agent's query returns {len(agent_rows)} rows, the reference {len(reference.rows)}", len(agent_rows), len(reference.rows), True)
        # fall through to the prose: the query differs, the prose says how much
    numbers = _numbers_in(text)
    expected = _reference_values(reference.rows)
    matched = sum(1 for value in expected if any(_close(n, value) for n in numbers))
    if not numbers:
        if _ABSTAIN_HINT.search(text or ""):
            return Graded(question.id, "abstained", "agent_abstained", "the agent declined a question the source answers", 0, len(expected))
        return Graded(question.id, "incomplete", "no_numbers", "no figures in the answer", 0, len(expected))
    if expected and matched == len(expected):
        cause, detail = "", ""
        if question.kind == "top_n":
            labels = _labels(reference.rows)
            positions = [text.find(label) for label in labels]
            if any(p < 0 for p in positions):
                cause, detail = "missing_rows", f"{sum(1 for p in positions if p < 0)} of {len(labels)} ranked items absent"
            elif positions != sorted(positions):
                cause, detail = "unsorted_ranking", "ranked items appear out of order"
        return Graded(question.id, "correct" if not cause else "partial", cause, detail, matched, len(expected), agent_rows is not None)
    for label, rows in reference.alternates.items():
        alternate = _reference_values(rows)
        if alternate and sum(1 for v in alternate if any(_close(n, v) for n in numbers)) == len(alternate):
            return Graded(question.id, "wrong", "narrower_scope", f"the figures match {label} alone, not the combined scope asked for", matched, len(expected), agent_rows is not None)
    if expected and matched >= 0.5 * len(expected):
        labels = _labels(reference.rows)
        if question.kind == "top_n" and labels and any(text.find(label) < 0 for label in labels):
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
}


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
                moved.setdefault(finding.source_id, []).append(original)
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

    def score(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for g in self.graded:
            counts[g.outcome] = counts.get(g.outcome, 0) + 1
        return counts

    def to_markdown(self) -> str:
        s = self.snapshot
        lines = [f"# Data Agent review: {s.name}", ""]
        lines.append(f"Stage: {s.stage}. Agent instructions: {len(s.instructions):,} characters. Sources: {len(s.datasources)}.")
        lines.append("")
        lines.append("## Sources")
        for source in s.datasources:
            schema = next((x for x in self.schemas if x.source_id == source.id), None)
            shape = f"{len(schema.tables)} tables" + (f", {len(schema.measures)} measures" if schema and schema.measures else "") if schema else "not profiled"
            selected = f"; {len(source.selected_tables)} tables selected for the agent" if source.selected_tables else ""
            lines.append(f"- {source.name or source.id} ({source.kind}): {shape}{selected}; {len(source.fewshots)} few-shots; {len(source.instructions):,} characters of instructions; description {'present' if source.description.strip() else 'missing'}.")
        lines.append("")
        lines.append("## Findings")
        if not self.findings:
            lines.append("- none")
        for f in sorted(self.findings, key=lambda x: ("high", "medium", "low", "info").index(x.severity)):
            lines.append(f"- **{f.severity}** `{f.code}`" + (f" ({f.source_id})" if f.source_id else "") + f": {f.message}")
            for e in f.evidence[:6]:
                lines.append(f"    - {e}")
            if f.suggestion:
                lines.append(f"    - suggestion: {f.suggestion}")
            if f.basis:
                lines.append(f"    - basis: {f.basis}")
        lines.append("")
        lines.append("## Evaluation")
        counts = self.score()
        lines.append(", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no questions")
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
            lines.append(f"| {q.id} | {q.text} | {g.outcome if g else ''} | {g.cause if g else ''} | {detail} | {'yes' if query else 'no'} | {routed} |")
        lines.append("")
        if any(a.steps for answers in self.answers.values() for a in answers):
            lines.append("## Agent run steps")
            lines.append("What the agent did per question: the functions it called, the language of the query it executed, the source it routed to.")
            for q in self.questions:
                for attempt, a in enumerate(self.answers.get(q.id, ()), start=1):
                    if not a.steps:
                        continue
                    names = sorted({s.name for s in a.steps if s.name})
                    lines.append(f"- {q.id} (attempt {attempt}): {len(a.steps)} step(s); functions: {', '.join(names) or 'none'}; query language: {a.language or 'none'}; routed to: {a.datasource or 'unknown'}; status: {a.status}; {a.seconds}s")
            lines.append("")
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
) -> ReviewReport:
    """The whole review: diagnose, generate, reference, ask, grade, suggest.

    With ``repetitions`` above one each question is asked that many times;
    an outcome that changes between runs is graded ``inconsistent``, which
    the routing guidance names as the sign of an ambiguous source choice.
    """
    findings = diagnose(snapshot, schemas)
    if years is None:
        years = {}
        for schema in schemas:
            executor = executors.get(schema.source_id)
            if isinstance(executor, LakehouseExecutor):
                source = next((s for s in snapshot.datasources if s.id == schema.source_id), None)
                try:
                    years[schema.source_id] = discover_years(executor, schema, source, snapshot.instructions)
                except Exception:  # noqa: BLE001
                    years[schema.source_id] = []
    questions = generate_questions(snapshot, schemas, years=years, top=top, limit_per_source=limit_per_source)
    references = build_references(questions, executors)
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
            if answer.query and answer.language == "sql" and isinstance(executor, LakehouseExecutor):
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
        graded.append(_with_routing(final, question, collected, snapshot))
    suggestions = suggest(snapshot, schemas, findings, questions, references, graded)
    return ReviewReport(snapshot, tuple(schemas), findings, questions, references, answers, tuple(graded), suggestions)


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
