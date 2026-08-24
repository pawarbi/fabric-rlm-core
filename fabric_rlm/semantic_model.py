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

from dataclasses import dataclass, field
from typing import Any

_SEMPY_MISSING = (
    "sempy is not importable, so a SemanticModel cannot be queried here. "
    "sempy ships in the Microsoft Fabric notebook runtime; outside Fabric, "
    "install semantic-link (`pip install semantic-link`) and make sure you are "
    "authenticated. Note that `import fabric` is a different package (SSH "
    "automation) and is not what this needs."
)


def sempy_available() -> bool:
    """True when semantic link can be imported in this process."""
    try:
        import sempy.fabric  # noqa: F401
    except Exception:
        return False
    return True


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
    validate: bool | str = field(default="auto", repr=False, compare=False)

    def __post_init__(self) -> None:
        if not str(self.dataset).strip():
            raise ValueError("SemanticModel requires a dataset name or GUID.")
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
        return {"workspace": self.workspace} if self.workspace else {}

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

    def dax(self, query: str) -> Any:
        """Run a DAX query and return a DataFrame.

        Aggregate here rather than pulling rows out and aggregating in pandas.
        """
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
            "Call .schema() for tables, measures with their DAX, and "
            'relationships; .dax("EVALUATE ...") -> DataFrame; '
            ".measure(name, groupby=[...], filters={...}); "
            ".tables() .columns() .measures() .relationships()"
        )

    def __repr__(self) -> str:
        where = f", workspace={self.workspace!r}" if self.workspace else ""
        return f"SemanticModel({self.dataset!r}{where})"
