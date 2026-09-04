"""A Power BI semantic model as an RLM input.

    from fabric_rlm import RLM, SemanticModel

    RLM.task(
        task="Which product line has the highest ARR?",
        inputs={"arr": SemanticModel("ARR Model SF (79)")},
        outputs=["answer"],
    ).run()

Inside the run, `arr` is a live handle: `arr.schema()`, `arr.measures()`,
`arr.dax("EVALUATE ...")`. The model does not have to know that sempy exists,
which is the point.

Measured on two semantic models and two LM families, a task that named a
semantic model but gave no entry point scored 7/19 and 5/15, with most
questions burning every available turn hunting for a way in. The same tasks
with the entry point supplied scored 18-19/19 and 13/15. Describing the entry
point in a skill worked, but cost ~2.4k characters resent on every turn.
Binding a handle costs one line in the input listing.
"""

from __future__ import annotations

import base64
import difflib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, ClassVar

_SEMPY_MISSING = (
    "sempy is not importable, so a SemanticModel cannot be queried here. "
    "sempy ships in the Microsoft Fabric notebook runtime; outside Fabric, "
    "install semantic-link (`pip install semantic-link`) and make sure you are "
    "authenticated. Note that `import fabric` is a different package (SSH "
    "automation) and is not what this needs."
)


_log = logging.getLogger("fabric_rlm.semantic_model")

# Guardrails for SemanticModel.aggregate(). One observed run asked for
# Sub Product Line x Region x Customer Group x Quarter with five measures, hit
# the 300s worker timeout, retried per quarter and hit it again. Nothing told
# the model the grain was too wide until the whole budget was gone. The
# cardinality preflight below answers that in seconds instead.
DEFAULT_MAX_GROUPS = 10_000
# The first live run measured 8.7s and 8.3s for preflights that counted 44 and
# 2,418 groups: the engine round trip on that model has an ~8s floor regardless
# of grain. A 10s budget would reject legitimate queries on any latency spike,
# so the default leaves headroom while still failing well inside the 300s
# worker timeout.
DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 30.0
MAX_GROUPS_ENV = "FABRIC_RLM_SEMANTIC_MAX_GROUPS"
PREFLIGHT_TIMEOUT_ENV = "FABRIC_RLM_SEMANTIC_PREFLIGHT_TIMEOUT"


class SemanticModelQueryError(RuntimeError):
    """A semantic-model query was rejected before or instead of running.

    Recoverable: the handle stays usable and a narrower query can follow.
    """


class SemanticModelQueryTooBroad(SemanticModelQueryError):
    """The cardinality preflight estimated more groups than the safe limit."""

    def __init__(
        self,
        message: str,
        *,
        estimated_groups: int,
        max_groups: int,
    ) -> None:
        super().__init__(message)
        self.estimated_groups = estimated_groups
        self.max_groups = max_groups


class SemanticModelQueryRiskUnknown(SemanticModelQueryError):
    """The cardinality preflight did not finish inside its short budget."""

    def __init__(self, message: str, *, timeout_seconds: float) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


_COLUMN_REF = re.compile(
    r"^\s*'?(?P<table>[^'\[\]]+?)'?\s*\[(?P<column>[^\[\]]+)\]\s*$"
)


def _split_column_ref(ref: Any) -> tuple[str, str]:
    match = _COLUMN_REF.match(str(ref))
    if not match:
        raise SemanticModelQueryError(
            f"Column references must look like Table[Column]; got {ref!r}."
        )
    return match.group("table").strip(), match.group("column").strip()


def _dax_column(table: str, column: str) -> str:
    return f"'{table}'[{column}]"


def _dax_literal(value: Any) -> str:
    if value is None:
        return "BLANK()"
    if isinstance(value, bool):
        return "TRUE()" if value else "FALSE()"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:
            return "BLANK()"
        return repr(value)
    if isinstance(value, datetime):
        return (
            f"DATE({value.year}, {value.month}, {value.day}) + "
            f"TIME({value.hour}, {value.minute}, {value.second})"
        )
    if isinstance(value, date):
        return f"DATE({value.year}, {value.month}, {value.day})"
    text = str(value).replace('"', '""')
    return f'"{text}"'


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SemanticModelQueryError(
            f"{name} must be a positive integer; got {value!r}."
        )
    return value


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        _log.warning("%s=%r is not an integer; using %s", name, raw, default)
        return default
    if value <= 0:
        _log.warning("%s=%r is not positive; using %s", name, raw, default)
        return default
    return value


def _env_positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        _log.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    if value <= 0:
        _log.warning("%s=%r is not positive; using %s", name, raw, default)
        return default
    return value


def _run_with_deadline(call: Any, timeout: float) -> Any:
    """Run ``call`` on a daemon thread and give up waiting after ``timeout``.

    sempy exposes no per-query timeout, so the only way to bound a preflight
    is to stop waiting for it. The thread is abandoned, not cancelled; the
    engine may still finish the query. Raises ``TimeoutError`` on expiry.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = call()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
            box["error"] = exc

    thread = threading.Thread(
        target=target,
        name="fabric-rlm-semantic-preflight",
        daemon=True,
    )
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"preflight exceeded {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _first_scalar(frame: Any) -> Any:
    try:
        return frame.iloc[0, 0]
    except Exception:
        pass
    records = frame.to_dict(orient="records")
    if not records:
        return None
    first = records[0]
    return next(iter(first.values()), None) if first else None


def _row_count(frame: Any) -> int | None:
    try:
        return int(len(frame))
    except Exception:
        return None


def _query_fingerprint(query: Any) -> str:
    from hashlib import sha256

    return sha256(str(query).encode("utf-8")).hexdigest()[:16]


def _measure_observations(frame: Any, plan: "_AggregatePlan") -> dict[str, Any]:
    """Value-free facts about the measures in an aggregate result.

    Which measure columns are identical across every row, and which are a
    constant zero, one or null. A derived measure that equals its base
    measure under an unfiltered context is the signature of a missing
    evaluation context (a previous-period measure with no period to be
    previous to); the observation records that shape without recording a
    single value, so it can travel into durable knowledge.
    """
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - pandas ships with sempy
        return {}
    try:
        available = set(str(c) for c in getattr(frame, "columns", ()))
        columns: dict[str, str] = {}
        for alias, name in plan.aliases.items():
            for candidate in (f"[{alias}]", alias):
                if candidate in available:
                    columns[name] = candidate
                    break
        if not columns or _row_count(frame) in {None, 0}:
            return {}
        numeric: dict[str, Any] = {}
        constants: dict[str, str] = {}
        for name, column in columns.items():
            series = frame[column]
            if series.isna().all():
                constants[name] = "null"
                continue
            values = pd.to_numeric(series, errors="coerce")
            if values.isna().any():
                continue
            numeric[name] = values
            if bool((values == 0).all()):
                constants[name] = "zero"
            elif bool((values == 1).all()):
                constants[name] = "one"
        identities: list[list[str]] = []
        names = list(numeric)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                scale = max(
                    float(numeric[left].abs().max()),
                    float(numeric[right].abs().max()),
                    1.0,
                )
                if float((numeric[left] - numeric[right]).abs().max()) <= 1e-9 * scale:
                    identities.append([left, right])
        observations: dict[str, Any] = {}
        if constants:
            observations["constant_measures"] = constants
        if identities:
            observations["measure_identities"] = identities
        return observations
    except Exception:
        return {}


@dataclass(frozen=True)
class _AggregatePlan:
    """A validated aggregate() request and the DAX built from it."""

    measures: tuple[str, ...]
    group_columns: tuple[tuple[str, str], ...]
    filters: tuple[tuple[tuple[str, str], tuple[Any, ...]], ...]
    order_by: str | None
    descending: bool
    top: int | None

    @property
    def aliases(self) -> dict[str, str]:
        """Output column alias -> requested measure name.

        Output columns are aliased ``__m0``, ``__m1``... so that TOPN and
        ORDER BY refer unambiguously to the summarized column rather than to
        a measure of the same name being re-evaluated per row.
        """
        return {f"__m{i}": name for i, name in enumerate(self.measures)}

    def _order_expression(self) -> str | None:
        if self.order_by is None:
            return None
        for alias, name in self.aliases.items():
            if name == self.order_by:
                return f"[{alias}]"
        for table, column in self.group_columns:
            if f"{table}[{column}]" == self.order_by:
                return _dax_column(table, column)
        return None

    def _summarize_arguments(self, *, with_measures: bool) -> list[str]:
        args = [_dax_column(t, c) for t, c in self.group_columns]
        for (table, column), values in self.filters:
            literals = ", ".join(_dax_literal(v) for v in values)
            args.append(f"TREATAS({{{literals}}}, {_dax_column(table, column)})")
        if with_measures:
            for alias, name in self.aliases.items():
                args.append(f'"{alias}", [{name}]')
        return args

    def preflight_query(self) -> str:
        inner = ",\n            ".join(self._summarize_arguments(with_measures=False))
        return (
            "EVALUATE\n"
            "ROW(\n"
            '    "group_count",\n'
            "    COUNTROWS(\n"
            "        SUMMARIZECOLUMNS(\n"
            f"            {inner}\n"
            "        )\n"
            "    )\n"
            ")"
        )

    def query(self) -> str:
        inner = ",\n        ".join(self._summarize_arguments(with_measures=True))
        summarize = f"SUMMARIZECOLUMNS(\n        {inner}\n    )"
        direction = "DESC" if self.descending else "ASC"
        order = self._order_expression()
        if self.top is not None:
            body = (
                "TOPN(\n"
                f"    {self.top},\n"
                f"    {summarize},\n"
                f"    {order}, {direction}\n"
                ")"
            )
        else:
            body = summarize
        text = f"EVALUATE\n{body}"
        if order is not None:
            text += f"\nORDER BY {order} {direction}"
        return text


@dataclass(frozen=True)
class SemanticModelMetadata:
    """Normalized semantic-model metadata with stable snake-case columns.

    The four frames are attributes (``meta.columns``) and also reachable the
    way generated code tends to assume a metadata container works:
    ``meta["columns"]``, ``meta.keys()``, ``meta.items()``, ``meta.values()``,
    ``meta.get("measures")`` and ``"relationships" in meta``. A trace showed
    the model writing ``for name, df in meta.items():`` on its first turn and
    spending the next one recovering from the AttributeError; that turn was
    pure overhead.

    This is deliberately not a ``collections.abc.Mapping`` subclass: the
    serializer and prompt code dispatch on dataclasses, and turning this into
    a Mapping would change how a metadata snapshot is frozen and described.
    Only the four public frames are exposed as keys, so a private field added
    later does not leak into ``keys()``.
    """

    tables: Any
    columns: Any
    measures: Any
    relationships: Any

    _KEYS: ClassVar[tuple[str, ...]] = (
        "tables",
        "columns",
        "measures",
        "relationships",
    )

    def __getitem__(self, key: str) -> Any:
        if not isinstance(key, str) or key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._KEYS

    def keys(self) -> tuple[str, ...]:
        """The four metadata frame names, in a stable order."""
        return self._KEYS

    def values(self) -> tuple[Any, ...]:
        return tuple(getattr(self, key) for key in self._KEYS)

    def items(self) -> tuple[tuple[str, Any], ...]:
        return tuple((key, getattr(self, key)) for key in self._KEYS)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


_METADATA_COLUMNS = {
    "tables": {
        "Name": "table_name",
        "Description": "description",
        "Type": "table_type",
    },
    "columns": {
        "Table Name": "table_name",
        "Column Name": "column_name",
        "Description": "description",
        "Data Type": "data_type",
    },
    "measures": {
        "Table Name": "table_name",
        "Measure Name": "measure_name",
        "Measure Expression": "measure_expression",
        "Measure Description": "measure_description",
        "Measure Display Folder": "measure_display_folder",
    },
    "relationships": {
        "From Table": "from_table",
        "From Column": "from_column",
        "To Table": "to_table",
        "To Column": "to_column",
        "Multiplicity": "multiplicity",
        "Cardinality": "cardinality",
        "Relationship Name": "relationship_name",
    },
}


def _normalized_column_name(value: Any) -> str:
    text = str(value).strip().strip("[]")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower() or "column"


def _plain_frame(frame: Any, aliases: dict[str, str] | None = None) -> Any:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - sempy includes pandas in Fabric
        raise RuntimeError(
            "Normalized semantic-model results require pandas."
        ) from exc

    if isinstance(frame, pd.DataFrame):
        result = pd.DataFrame(frame.copy())
    else:
        records = frame.to_dict(orient="records")
        result = pd.DataFrame.from_records(records, columns=list(frame.columns))
    aliases = aliases or {}
    assigned: set[str] = set()
    next_suffix: dict[str, int] = {}
    normalized: list[str] = []
    for column in result.columns:
        base = aliases.get(str(column), _normalized_column_name(column))
        candidate = base
        suffix = next_suffix.get(base, 2)
        while candidate in assigned:
            candidate = f"{base}_{suffix}"
            suffix += 1
        next_suffix[base] = suffix
        assigned.add(candidate)
        normalized.append(candidate)
    result.columns = normalized
    return result


def sempy_available() -> bool:
    """True when semantic link can be imported in this process."""
    try:
        import sempy.fabric  # noqa: F401
    except Exception:
        return False
    return True


class _NotebookUtilsPbiCredential:
    """Refresh Power BI tokens through the current Fabric notebook identity."""

    def get_token(self, *scopes: str, **kwargs: Any) -> Any:
        del scopes, kwargs
        try:
            import notebookutils
            from azure.core.credentials import AccessToken
        except Exception as exc:  # pragma: no cover - Fabric runtime dependent
            raise RuntimeError(
                "credential_provider='notebookutils' requires a Fabric "
                "notebook runtime with notebookutils and azure-core"
            ) from exc
        token = notebookutils.credentials.getToken("pbi")
        if not isinstance(token, str) or not token:
            raise RuntimeError("notebookutils returned an invalid Power BI token")
        try:
            payload = token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            expires_on = int(
                json.loads(base64.urlsafe_b64decode(payload))["exp"]
            )
        except Exception as exc:
            raise RuntimeError(
                "Power BI token expiry could not be decoded"
            ) from exc
        return AccessToken(token, expires_on)


@dataclass(frozen=True)
class SemanticModel:
    """A handle to a Power BI semantic model, bound into the run namespace.

    ``workspace`` may be a name or a GUID, and defaults to the workspace the
    notebook is attached to.

    ``validate`` controls the reachability check performed at construction:
    ``"auto"`` checks only when sempy is importable (so this class can be
    constructed and unit-tested anywhere), ``True`` always checks and raises if
    sempy is missing, ``False`` never checks.
    """

    # Marks an input that carries evidence; see prompts.is_evidence_source.
    __rlm_evidence_source__ = True

    dataset: str
    workspace: str | None = None
    credential_provider: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    validate: bool | str = field(default="auto", repr=False, compare=False)
    # Host-side ceiling for aggregate(); None means the environment default.
    # Set by whoever builds the handle. The LM-visible rejection never
    # mentions raising it, so a wide query gets narrowed rather than waved on.
    max_groups: int | None = field(default=None, repr=False, compare=False)
    _catalog: Any = field(default=None, init=False, repr=False, compare=False)
    _query_telemetry: Any = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not str(self.dataset).strip():
            raise ValueError("SemanticModel requires a dataset name or GUID.")
        if self.credential_provider not in {None, "notebookutils"}:
            raise ValueError(
                "credential_provider must be None or 'notebookutils'"
            )
        if self.max_groups is not None and (
            isinstance(self.max_groups, bool)
            or not isinstance(self.max_groups, int)
            or self.max_groups <= 0
        ):
            raise ValueError("max_groups must be a positive integer or None")
        if self.validate is False:
            return
        if self.validate == "auto" and not sempy_available():
            return
        self.check()

    # -- plumbing ---------------------------------------------------------

    @property
    def _fabric(self) -> Any:
        try:
            import sempy.fabric as fabric
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(_SEMPY_MISSING) from exc
        return fabric

    @property
    def _kw(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.workspace:
            kwargs["workspace"] = self.workspace
        if self.credential_provider == "notebookutils":
            kwargs["credential"] = _NotebookUtilsPbiCredential()
        return kwargs

    def __frozen__(self) -> dict[str, Any]:
        """What a namespace snapshot records for this handle.

        The name catalog and query telemetry are working state, not identity.
        Left to ``dataclasses.asdict`` they turned every turn's trajectory
        state into a dump of several hundred column names.
        """
        return {
            "dataset": self.dataset,
            "workspace": self.workspace,
            "credential_provider": self.credential_provider,
            "validate": self.validate,
            "max_groups": self.max_groups,
        }

    def check(self) -> "SemanticModel":
        """Confirm the model is reachable. Raises with a usable message."""
        try:
            self._fabric.list_tables(self.dataset, **self._kw)
        except RuntimeError:
            raise
        except Exception as exc:
            where = f" in workspace {self.workspace!r}" if self.workspace else ""
            raise ValueError(
                f"Semantic model {self.dataset!r}{where} could not be read: "
                f"{type(exc).__name__}: {exc}. Check the name (it is the model's "
                "display name, not the report name) and that you have access."
            ) from exc
        return self

    # -- metadata ---------------------------------------------------------

    def tables(self) -> Any:
        """Tables in the model, as a DataFrame."""
        return self._fabric.list_tables(self.dataset, **self._kw)

    def columns(self, table: str | None = None) -> Any:
        """Columns in the model, as a DataFrame. ``table`` narrows to one table.

        Generated code reaches for ``columns("ARR Data")`` unprompted, and
        sempy's ``list_columns`` takes the same argument, so pass it through.
        """
        kwargs: dict[str, Any] = dict(self._kw)
        if table is not None:
            kwargs["table"] = table
        return self._fabric.list_columns(self.dataset, **kwargs)

    def measures(self) -> Any:
        """Measures, with their DAX expressions and descriptions."""
        return self._fabric.list_measures(self.dataset, **self._kw)

    def relationships(self) -> Any:
        """Relationships between tables, as a DataFrame."""
        return self._fabric.list_relationships(self.dataset, **self._kw)

    def metadata(self) -> SemanticModelMetadata:
        """Return ordinary pandas metadata frames with stable column names.

        The raw metadata methods remain available when callers need SemPy's
        complete provider-specific columns.
        """
        return SemanticModelMetadata(
            tables=_plain_frame(self.tables(), _METADATA_COLUMNS["tables"]),
            columns=_plain_frame(self.columns(), _METADATA_COLUMNS["columns"]),
            measures=_plain_frame(self.measures(), _METADATA_COLUMNS["measures"]),
            relationships=_plain_frame(
                self.relationships(),
                _METADATA_COLUMNS["relationships"],
            ),
        )

    def schema(self, max_chars: int = 4000) -> str:
        """One call that answers "what is in this model".

        This exists so the first turn is a single call rather than a recipe the
        model has to be told. Includes measure expressions, because measure
        names routinely describe something other than what they compute, and
        descriptions, because that is where model authors put business meaning.
        """
        out: list[str] = [f"Semantic model: {self.dataset}"]
        if self.workspace:
            out.append(f"Workspace: {self.workspace}")

        def section(title: str, frame_fn: Any, cols: tuple[str, ...]) -> None:
            out.append(f"\n== {title} ==")
            try:
                df = frame_fn()
            except Exception as exc:
                out.append(f"(unavailable: {type(exc).__name__}: {exc})")
                return
            keep = [c for c in cols if c in getattr(df, "columns", [])]
            view = df[keep] if keep else df
            out.append(view.to_string()[:max_chars])

        section("Tables", self.tables, ("Name", "Description"))
        # sempy names this column "Measure Description", not "Description".
        # Asking for the latter drops every description without erroring, which
        # is how it went unnoticed: a trace showed the model retrying with the
        # right name after its own first guess failed.
        section("Measures", self.measures,
                ("Table Name", "Measure Name", "Measure Expression",
                 "Measure Description", "Measure Display Folder"))
        section("Relationships", self.relationships,
                ("From Table", "From Column", "To Table", "To Column",
                 "Multiplicity", "Cardinality"))

        # Columns, compactly. A DataFrame dump of a few hundred columns blows
        # the budget; grouped names do not. Omitting them entirely cost 16
        # separate .columns() calls in one 12-question run.
        out.append("\n== Columns ==")
        try:
            cdf = self.columns()
            tcol = "Table Name" if "Table Name" in cdf.columns else cdf.columns[0]
            ncol = "Column Name" if "Column Name" in cdf.columns else cdf.columns[1]
            grouped: dict[str, list[str]] = {}
            for _i, row in cdf.iterrows():
                grouped.setdefault(str(row[tcol]), []).append(str(row[ncol]))
            budget = max_chars
            for table, names in grouped.items():
                line = f"{table}: {', '.join(names)}"
                if len(line) > 600:
                    line = line[:600] + f" ... (+{len(names)} columns total)"
                if budget - len(line) < 0:
                    out.append(f"... ({len(grouped)} tables in total, listing truncated)")
                    break
                out.append(line)
                budget -= len(line)
        except Exception as exc:
            out.append(f"(unavailable: {type(exc).__name__}: {exc})")

        return "\n".join(out)

    # -- querying ---------------------------------------------------------

    def dax(self, query: str, *, normalize_columns: bool = False) -> Any:
        """Run a DAX query and return a DataFrame.

        Aggregate here rather than pulling rows out and aggregating in pandas.
        Set ``normalize_columns`` to return an ordinary pandas DataFrame with
        stable snake-case names instead of SemPy's bracketed result columns.

        Use :meth:`aggregate` for grouped measure analysis: it estimates the
        result grain before running and rejects queries likely to consume the
        worker timeout. Use ``dax`` for custom DAX that ``aggregate`` cannot
        express; it runs whatever it is given, with no size check.
        """
        started = time.monotonic()
        record: dict[str, Any] = {
            "query_type": "dax",
            "query_fingerprint": _query_fingerprint(query),
            "query_chars": len(str(query)),
            "executed": True,
        }
        try:
            result = self._evaluate(query)
        except Exception as exc:
            record.update(
                execution_seconds=round(time.monotonic() - started, 3),
                reason="execution_error",
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
            self._record_query(record)
            raise
        record.update(
            execution_seconds=round(time.monotonic() - started, 3),
            returned_rows=_row_count(result),
            total_seconds=round(time.monotonic() - started, 3),
        )
        self._record_query(record)
        return _plain_frame(result) if normalize_columns else result

    def _evaluate(self, query: str) -> Any:
        """Run DAX without recording telemetry; aggregate() records its own."""
        return self._fabric.evaluate_dax(self.dataset, query, **self._kw)

    def measure(
        self,
        measure: str | list[str],
        groupby: list[str] | None = None,
        filters: dict[str, list[str]] | None = None,
    ) -> Any:
        """Evaluate a model measure, optionally grouped and filtered.

        No DAX to author, so no DAX syntax to get wrong. ``groupby`` entries are
        fully qualified, e.g. ``["Owner[Owner Country]"]``.
        """
        kwargs: dict[str, Any] = dict(self._kw)
        if groupby:
            kwargs["groupby_columns"] = list(groupby)
        if filters:
            kwargs["filters"] = dict(filters)
        measures = [measure] if isinstance(measure, str) else [str(m) for m in measure]
        started = time.monotonic()
        record: dict[str, Any] = {
            "query_type": "measure",
            "measures": measures,
            "measure_count": len(measures),
            "groupby": [str(g) for g in (groupby or [])],
            "groupby_count": len(groupby or []),
            "filter_columns": [str(c) for c in (filters or {})],
            "filter_count": len(filters or {}),
            "executed": True,
        }
        try:
            result = self._fabric.evaluate_measure(self.dataset, measure, **kwargs)
        except Exception as exc:
            record.update(
                execution_seconds=round(time.monotonic() - started, 3),
                reason="execution_error",
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
            self._record_query(record)
            raise
        record.update(
            execution_seconds=round(time.monotonic() - started, 3),
            returned_rows=_row_count(result),
            total_seconds=round(time.monotonic() - started, 3),
        )
        self._record_query(record)
        return result

    def read_table(self, table: str, num_rows: int | None = None) -> Any:
        """Read a table. Use for small dimension tables only, never a fact table."""
        kwargs: dict[str, Any] = dict(self._kw)
        if num_rows is not None:
            kwargs["num_rows"] = num_rows
        return self._fabric.read_table(self.dataset, table, **kwargs)

    # -- bounded aggregation ------------------------------------------------

    def aggregate(
        self,
        measures: str | list[str],
        *,
        groupby: list[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        order_by: str | None = None,
        descending: bool = True,
        top: int | None = None,
        max_groups: int | None = None,
        preflight: bool = True,
        normalize_columns: bool = True,
    ) -> Any:
        """Evaluate measures by dimensions with a query-size guardrail.

        Names are validated against the model first, so a typo costs one
        message rather than a round trip to the engine. A short cardinality
        preflight then counts the groups the grouping columns and filters
        would produce (an upper bound: the cross join of the grouping columns
        after filters); if that exceeds ``max_groups``, or the preflight does
        not finish inside its budget, the expensive query never runs and the
        error explains how to narrow it. ``top``/``order_by`` bound what is
        returned but not what the engine has to evaluate, so they do not
        skip the preflight.

        ``max_groups`` defaults to the value set when the handle was built,
        then ``FABRIC_RLM_SEMANTIC_MAX_GROUPS``, then 10,000. The preflight
        budget comes from ``FABRIC_RLM_SEMANTIC_PREFLIGHT_TIMEOUT`` (default
        30 seconds). Returns an ordinary pandas DataFrame with snake-case
        columns unless ``normalize_columns=False``.
        """
        started = time.monotonic()
        record: dict[str, Any] = {"query_type": "aggregate", "executed": False}
        try:
            limit = self._effective_max_groups(max_groups)
            plan = self._plan_aggregate(
                measures,
                groupby=groupby,
                filters=filters,
                order_by=order_by,
                descending=descending,
                top=top,
            )
        except SemanticModelQueryError as exc:
            record.update(reason="validation", error=str(exc)[:300])
            self._record_query(record)
            raise
        record.update(
            measures=list(plan.measures),
            groupby=[f"{t}[{c}]" for t, c in plan.group_columns],
            groupby_count=len(plan.group_columns),
            measure_count=len(plan.measures),
            # Filter column names only: the values are data, and telemetry
            # feeds durable knowledge, which must never carry data values.
            filter_columns=[f"{t}[{c}]" for (t, c), _values in plan.filters],
            filter_count=len(plan.filters),
            top=plan.top,
            order_by=plan.order_by,
            max_groups=limit,
            preflight=bool(preflight and plan.group_columns),
        )
        if not plan.group_columns:
            record["estimated_groups"] = 1
        elif preflight:
            budget = _env_positive_float(
                PREFLIGHT_TIMEOUT_ENV, DEFAULT_PREFLIGHT_TIMEOUT_SECONDS
            )
            preflight_started = time.monotonic()
            try:
                estimated = _run_with_deadline(
                    lambda: self._evaluate(plan.preflight_query()),
                    budget,
                )
            except TimeoutError:
                record.update(
                    preflight_seconds=round(time.monotonic() - preflight_started, 3),
                    reason="preflight_timeout",
                )
                self._record_query(record)
                raise SemanticModelQueryRiskUnknown(
                    self._risk_unknown_message(plan, budget),
                    timeout_seconds=budget,
                ) from None
            except Exception as exc:
                record.update(
                    preflight_seconds=round(time.monotonic() - preflight_started, 3),
                    reason="preflight_error",
                    error=f"{type(exc).__name__}: {exc}"[:300],
                )
                self._record_query(record)
                raise
            record["preflight_seconds"] = round(
                time.monotonic() - preflight_started, 3
            )
            count = _first_scalar(estimated)
            try:
                estimated_groups = int(count) if count == count else 0
            except (TypeError, ValueError):
                estimated_groups = 0
            record["estimated_groups"] = estimated_groups
            if estimated_groups > limit:
                record["reason"] = "cardinality_limit"
                self._record_query(record)
                raise SemanticModelQueryTooBroad(
                    self._too_broad_message(plan, estimated_groups, limit),
                    estimated_groups=estimated_groups,
                    max_groups=limit,
                )

        query = plan.query()
        record["query"] = query
        execution_started = time.monotonic()
        try:
            frame = self._evaluate(query)
        except Exception as exc:
            record.update(
                executed=True,
                execution_seconds=round(time.monotonic() - execution_started, 3),
                reason="execution_error",
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
            self._record_query(record)
            raise
        record.update(
            executed=True,
            execution_seconds=round(time.monotonic() - execution_started, 3),
            returned_rows=_row_count(frame),
            total_seconds=round(time.monotonic() - started, 3),
            **_measure_observations(frame, plan),
        )
        self._record_query(record)
        return self._finish_aggregate_frame(frame, plan, normalize_columns)

    @property
    def query_telemetry(self) -> tuple[dict[str, Any], ...]:
        """Per-query records from :meth:`aggregate`, oldest first.

        Each record carries the grouping/measure counts, the estimated group
        count, preflight and execution seconds, whether the query executed
        and, when it did not, the reason (``cardinality_limit``,
        ``preflight_timeout``, ``validation``).
        """
        log = getattr(self, "_query_telemetry", None)
        return tuple(dict(item) for item in (log or ()))

    def _record_query(self, record: dict[str, Any]) -> None:
        log = getattr(self, "_query_telemetry", None)
        if log is None:
            log = []
            object.__setattr__(self, "_query_telemetry", log)
        log.append(dict(record))
        _log.debug("semantic model query: %s", record)

    def _effective_max_groups(self, override: int | None) -> int:
        if override is not None:
            return _positive_int(override, "max_groups")
        configured = getattr(self, "max_groups", None)
        if configured is not None:
            return int(configured)
        return _env_positive_int(MAX_GROUPS_ENV, DEFAULT_MAX_GROUPS)

    # -- name resolution --------------------------------------------------

    def _catalog_names(self) -> dict[str, dict[str, Any]]:
        """Case-insensitive lookups for measure names and Table[Column] refs.

        Fetched once per handle: two metadata calls, then cached. Model
        schemas do not change mid-run and each fetch is a network round trip.
        """
        cached = getattr(self, "_catalog", None)
        if cached is not None:
            return cached
        measures = _plain_frame(self.measures(), _METADATA_COLUMNS["measures"])
        columns = _plain_frame(self.columns(), _METADATA_COLUMNS["columns"])
        measure_names: dict[str, str] = {}
        for row in measures.to_dict(orient="records"):
            name = str(row.get("measure_name", "")).strip()
            if name:
                measure_names.setdefault(name.lower(), name)
        column_refs: dict[str, tuple[str, str]] = {}
        for row in columns.to_dict(orient="records"):
            table = str(row.get("table_name", "")).strip()
            column = str(row.get("column_name", "")).strip()
            if table and column:
                column_refs.setdefault(f"{table}[{column}]".lower(), (table, column))
        catalog = {"measures": measure_names, "columns": column_refs}
        object.__setattr__(self, "_catalog", catalog)
        return catalog

    def _resolve_measure(self, name: Any, catalog: dict[str, Any]) -> str:
        text = str(name).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if not text:
            raise SemanticModelQueryError("Measure names must not be empty.")
        hit = catalog["measures"].get(text.lower())
        if hit is not None:
            return hit
        known = list(catalog["measures"].values())
        close = difflib.get_close_matches(text, known, n=5, cutoff=0.4)
        lines = [f"Unknown semantic-model measure: {text}"]
        if close:
            lines.append("")
            lines.append("Available close matches:")
            lines.extend(f"- {m}" for m in close)
        elif known:
            lines.append("")
            lines.append("Some available measures:")
            lines.extend(f"- {m}" for m in known[:10])
        lines.append("")
        lines.append("Call .metadata().measures for the full list.")
        raise SemanticModelQueryError("\n".join(lines))

    def _resolve_column(
        self, ref: Any, catalog: dict[str, Any], role: str
    ) -> tuple[str, str]:
        table, column = _split_column_ref(ref)
        hit = catalog["columns"].get(f"{table}[{column}]".lower())
        if hit is not None:
            return hit
        known = [f"{t}[{c}]" for t, c in catalog["columns"].values()]
        close = difflib.get_close_matches(f"{table}[{column}]", known, n=5, cutoff=0.5)
        same_name = [
            k for k in known
            if k.lower().endswith(f"[{column.lower()}]") and k not in close
        ]
        lines = [f"Unknown semantic-model column ({role}): {table}[{column}]"]
        suggestions = close + same_name[:3]
        if suggestions:
            lines.append("")
            lines.append("Available close matches:")
            lines.extend(f"- {c}" for c in suggestions)
        lines.append("")
        lines.append("Call .metadata().columns for the full list.")
        raise SemanticModelQueryError("\n".join(lines))

    def _plan_aggregate(
        self,
        measures: str | list[str],
        *,
        groupby: list[str] | None,
        filters: Mapping[str, Any] | None,
        order_by: str | None,
        descending: bool,
        top: int | None,
    ) -> _AggregatePlan:
        if isinstance(measures, str):
            requested = [measures]
        elif isinstance(measures, (list, tuple)):
            requested = list(measures)
        else:
            raise SemanticModelQueryError(
                "measures must be a measure name or a list of measure names."
            )
        if not requested:
            raise SemanticModelQueryError("aggregate() needs at least one measure.")
        if groupby is not None and not isinstance(groupby, (list, tuple)):
            raise SemanticModelQueryError(
                "groupby must be a list of Table[Column] references."
            )
        if filters is not None and not isinstance(filters, Mapping):
            raise SemanticModelQueryError(
                "filters must be a mapping of Table[Column] -> value or list of values."
            )
        if top is not None:
            top = _positive_int(top, "top")

        catalog = self._catalog_names()
        resolved_measures: list[str] = []
        for name in requested:
            canonical = self._resolve_measure(name, catalog)
            if canonical not in resolved_measures:
                resolved_measures.append(canonical)
        group_columns: list[tuple[str, str]] = []
        for ref in groupby or ():
            canonical = self._resolve_column(ref, catalog, "groupby")
            if canonical not in group_columns:
                group_columns.append(canonical)
        resolved_filters: list[tuple[tuple[str, str], tuple[Any, ...]]] = []
        for ref, value in (filters or {}).items():
            canonical = self._resolve_column(ref, catalog, "filter")
            if isinstance(value, (list, tuple, set, frozenset)):
                values = tuple(value)
            else:
                values = (value,)
            if not values:
                raise SemanticModelQueryError(
                    f"Filter on {canonical[0]}[{canonical[1]}] has no values."
                )
            resolved_filters.append((canonical, values))

        canonical_order: str | None = None
        if order_by is not None:
            text = str(order_by).strip()
            bare = text[1:-1].strip() if text.startswith("[") and text.endswith("]") else text
            for name in resolved_measures:
                if name.lower() == bare.lower():
                    canonical_order = name
                    break
            if canonical_order is None and _COLUMN_REF.match(text):
                table, column = _split_column_ref(text)
                for t, c in group_columns:
                    if (t.lower(), c.lower()) == (table.lower(), column.lower()):
                        canonical_order = f"{t}[{c}]"
                        break
            if canonical_order is None:
                options = list(resolved_measures) + [f"{t}[{c}]" for t, c in group_columns]
                raise SemanticModelQueryError(
                    f"order_by must name one of the requested measures or groupby "
                    f"columns; got {order_by!r}. Options: {options}"
                )
        elif top is not None:
            canonical_order = resolved_measures[0]

        return _AggregatePlan(
            measures=tuple(resolved_measures),
            group_columns=tuple(group_columns),
            filters=tuple(resolved_filters),
            order_by=canonical_order,
            descending=bool(descending),
            top=top,
        )

    @staticmethod
    def _finish_aggregate_frame(
        frame: Any, plan: _AggregatePlan, normalize_columns: bool
    ) -> Any:
        aliases = plan.aliases
        if normalize_columns:
            mapping: dict[str, str] = {}
            for alias, name in aliases.items():
                normalized = _normalized_column_name(name)
                mapping[f"[{alias}]"] = normalized
                mapping[alias] = normalized
            return _plain_frame(frame, mapping)
        rename: dict[str, str] = {}
        for alias, name in aliases.items():
            rename[f"[{alias}]"] = f"[{name}]"
            rename[alias] = f"[{name}]"
        try:
            return frame.rename(columns=rename)
        except Exception:
            return frame

    @staticmethod
    def _describe_request(plan: _AggregatePlan) -> list[str]:
        lines = ["Requested grouping:"]
        if plan.group_columns:
            lines.extend(f"- {t}[{c}]" for t, c in plan.group_columns)
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("Requested measures:")
        lines.extend(f"- {m}" for m in plan.measures)
        lines.append("")
        lines.append("Filters:")
        if plan.filters:
            for (t, c), values in plan.filters:
                shown = ", ".join(str(v) for v in values[:5])
                if len(values) > 5:
                    shown += f", ... ({len(values)} values)"
                lines.append(f"- {t}[{c}] in [{shown}]")
        else:
            lines.append("- (none)")
        return lines

    @staticmethod
    def _repair_guidance(plan: _AggregatePlan) -> list[str]:
        widest = plan.group_columns[0] if plan.group_columns else None
        coarser = (
            f"1. Use a coarser dimension in place of {widest[0]}[{widest[1]}] "
            "(a parent level such as a line of business or a region)."
            if widest else
            "1. Group by a coarser dimension."
        )
        return [
            "Try one of:",
            coarser,
            "2. Remove one grouping dimension.",
            "3. Add a narrower period, region, or segment filter via filters={...}.",
            "4. Request TOP N with order_by=... and top=... at a coarser grain.",
            "5. Query one measure at the coarsest grain first, then drill into "
            "the highest-impact segments with the other measures.",
            "",
            "The model is still available; the full query did not run.",
        ]

    def _too_broad_message(
        self, plan: _AggregatePlan, estimated: int, limit: int
    ) -> str:
        lines = [
            f"Estimated result grain: up to ~{estimated:,} groups "
            "(cross join of the grouping columns after filters).",
            f"Configured safe limit: {limit:,} groups.",
            "",
            *self._describe_request(plan),
            "",
            *self._repair_guidance(plan),
        ]
        return "\n".join(lines)

    def _risk_unknown_message(self, plan: _AggregatePlan, budget: float) -> str:
        lines = [
            f"The grouping cardinality could not be estimated within {budget:g} seconds.",
            "The requested grain is likely expensive.",
            "",
            *self._describe_request(plan),
            "",
            "Reduce the grouping grain or apply a narrower filter before retrying.",
            *self._repair_guidance(plan)[1:],
        ]
        return "\n".join(lines)

    # -- presentation -----------------------------------------------------

    def __rlm_describe__(self) -> str:
        """How this input is listed in the system prompt.

        Deliberately compact. The whole point of binding a handle is that the
        entry point does not have to be paid for in prompt tokens every turn.
        """
        where = f" workspace={self.workspace!r}" if self.workspace else ""
        return (
            f"SemanticModel dataset={self.dataset!r}{where} - already connected. "
            ".schema() for text, .metadata() for DataFrames. Prefer "
            ".aggregate(measures=[...], groupby=[...], filters={...}, "
            "order_by=..., top=...) for measures by dimensions; it checks "
            "query size first. .dax(\"EVALUATE ...\", normalize_columns=True) "
            "for custom DAX; .measure(name, groupby=[...], filters={...}); "
            ".tables() .columns() .measures() .relationships()"
        )

    def __repr__(self) -> str:
        where = f", workspace={self.workspace!r}" if self.workspace else ""
        return f"SemanticModel({self.dataset!r}{where})"
