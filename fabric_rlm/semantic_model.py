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
import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

_SEMPY_MISSING = (
    "sempy is not importable, so a SemanticModel cannot be queried here. "
    "sempy ships in the Microsoft Fabric notebook runtime; outside Fabric, "
    "install semantic-link (`pip install semantic-link`) and make sure you are "
    "authenticated. Note that `import fabric` is a different package (SSH "
    "automation) and is not what this needs."
)


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

    dataset: str
    workspace: str | None = None
    credential_provider: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    validate: bool | str = field(default="auto", repr=False, compare=False)

    def __post_init__(self) -> None:
        if not str(self.dataset).strip():
            raise ValueError("SemanticModel requires a dataset name or GUID.")
        if self.credential_provider not in {None, "notebookutils"}:
            raise ValueError(
                "credential_provider must be None or 'notebookutils'"
            )
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

    def columns(self) -> Any:
        """Columns in the model, as a DataFrame."""
        return self._fabric.list_columns(self.dataset, **self._kw)

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
        """
        result = self._fabric.evaluate_dax(self.dataset, query, **self._kw)
        return _plain_frame(result) if normalize_columns else result

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
        return self._fabric.evaluate_measure(self.dataset, measure, **kwargs)

    def read_table(self, table: str, num_rows: int | None = None) -> Any:
        """Read a table. Use for small dimension tables only, never a fact table."""
        kwargs: dict[str, Any] = dict(self._kw)
        if num_rows is not None:
            kwargs["num_rows"] = num_rows
        return self._fabric.read_table(self.dataset, table, **kwargs)

    # -- presentation -----------------------------------------------------

    def __rlm_describe__(self) -> str:
        """How this input is listed in the system prompt.

        Deliberately compact. The whole point of binding a handle is that the
        entry point does not have to be paid for in prompt tokens every turn.
        """
        where = f" workspace={self.workspace!r}" if self.workspace else ""
        return (
            f"SemanticModel dataset={self.dataset!r}{where} - already connected. "
            "Call .schema() for formatted text or .metadata() for normalized "
            "DataFrames; .dax(\"EVALUATE ...\", normalize_columns=True); "
            ".measure(name, groupby=[...], filters={...}); "
            ".tables() .columns() .measures() .relationships()"
        )

    def __repr__(self) -> str:
        where = f", workspace={self.workspace!r}" if self.workspace else ""
        return f"SemanticModel({self.dataset!r}{where})"
