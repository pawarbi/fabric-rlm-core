"""Trajectory records and exports."""

from __future__ import annotations

import json
import ast
import re
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


@dataclass
class Issue:
    """A single diagnostic finding from :meth:`Trajectory.diagnose`.

    ``kind`` is a short stable identifier (e.g. ``"markdown_in_code"``)
    callers can switch on; ``message`` is a human-readable explanation.
    """

    turn: int
    kind: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Module-level regex: matches markdown prose lines that escape into code.
# Patterns we have actually seen emitted by models when they confuse PLAN/VERIFY
# blocks with executable Python: bullet list items (``- foo`` / ``* foo``),
# bold spans (``**foo**``), and a small allowlist of known prose label words
# (``Target:``, ``Output:``, ``Step:``, etc.). The label list is intentionally
# narrow: matching every CamelCase identifier followed by ``:`` would false-
# positive on valid Python type annotations like ``Result: dict[str, Any]``.
_MD_PROSE_RE = re.compile(
    r"""^\s*(
        -\ |\*\ |                                          # bullet list
        \*\*[^*]+\*\*|                                       # bold span at line start
        (?:Target|Output|Approach|Assumptions?|Sub-?problems?|Step|Steps|Rationale|Goal|Plan|Verify|Reflect|Notes?|Context):\s
    )""",
    re.VERBOSE,
)


def _last_error_line(error: str | None) -> str:
    lines = (error or "").strip().splitlines()
    return lines[-1].strip() if lines else ""


@dataclass
class TurnRecord:
    turn: int
    code: str
    stdout: str
    stderr: str
    error: str | None
    submitted: bool
    state: dict[str, Any]
    response_text: str = ""
    duration_s: float | None = None
    token_usage: dict[str, Any] = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)
    turn_type: str = "normal"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # Nested usage details from OpenAI-style responses. ``cached_tokens`` is a
    # subset of ``prompt_tokens`` billed at the cached-input rate (~10x cheaper
    # for gpt-5). ``reasoning_tokens`` is a subset of ``completion_tokens`` and
    # is the dominant cost driver for reasoning models. Both are optional and
    # default to ``None`` so older trajectories remain readable.
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    lm_call_seconds: float | None = None
    worker_execute_seconds: float | None = None
    # Populated when the worker called SUBMIT(...) on this turn — the literal
    # payload that was submitted. ``None`` for non-submit turns.
    submit_payload: dict[str, Any] | None = None

    @property
    def state_keys(self) -> list[str]:
        return list(self.state.keys())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Azure storage URI loading (Fabric / Synapse)
#
# The library is dependency-free. We import notebookutils / fsspec lazily so
# the only cost of supporting Lakehouse URIs is a try/except
# at call time. ``/lakehouse/default/Files/...`` paths are regular FUSE-
# mounted files and need no special handling — they go through Path.read_text.
# ---------------------------------------------------------------------------


_AZURE_STORAGE_SCHEMES = frozenset({"abfss", "abfs", "wasbs", "wasb"})


def _try_import(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError:
        return None


def _redact_uri(uri: str) -> str:
    """Strip query/fragment from a URI before including it in user-facing
    messages. Azure Storage URIs can contain SAS tokens in their query
    string and we must never echo those back to logs / notebook output."""
    from urllib.parse import urlsplit, urlunsplit

    try:
        parts = urlsplit(uri)
    except ValueError:
        return uri
    if not parts.query and not parts.fragment:
        return uri
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "<redacted>" if parts.query else "", ""))


def _iter_fabric_fs_candidates() -> Iterator[Any]:
    """Yield the current Fabric ``fs`` object when available.

    The deprecated compatibility namespaces are intentionally unsupported.
    """
    nu = _try_import("notebookutils")
    if nu is not None:
        fs = getattr(nu, "fs", None)
        if fs is not None:
            yield fs


def _read_via_notebookutils(uri: str) -> str | None:
    """Read a remote text file by copying through a local temp file.

    Returns ``None`` if no Fabric/Synapse ``fs`` API is importable (i.e.
    we're outside those runtimes). Uses ``fs.cp`` rather than ``fs.head``
    because head truncates to ~1MB by default. Copies into a fresh
    ``TemporaryDirectory`` so the destination path is guaranteed not to
    pre-exist (some ``cp()`` implementations reject existing destinations)
    and the temp file is cleaned up in all paths.
    """
    fs = next(_iter_fabric_fs_candidates(), None)
    if fs is None:
        return None
    with tempfile.TemporaryDirectory(prefix="fabric_rlm_trace_") as tmpdir:
        tmp_path = Path(tmpdir) / "trace.jsonl"
        # Fabric expects file URIs for the local destination.
        fs.cp(uri, f"file:{tmp_path}")
        return tmp_path.read_text(encoding="utf-8")


def _read_via_fsspec(uri: str) -> str | None:
    """Read a remote text file via fsspec (uses adlfs for abfs/abfss)."""
    fsspec = _try_import("fsspec")
    if fsspec is None:
        return None
    try:
        with fsspec.open(uri, "r", encoding="utf-8") as fh:
            return fh.read()
    except ImportError as exc:
        # fsspec raises ImportError when the protocol-specific backend
        # (adlfs for abfss) isn't installed. Surface that as actionable.
        raise ImportError(
            f"fsspec is installed but cannot read {_redact_uri(uri)!r}: {exc}. "
            "For Azure Storage URIs install adlfs (`pip install adlfs`)."
        ) from exc


def _read_azure_storage_text(uri: str) -> str:
    """Read text content from an Azure Storage URI.

    Tries Fabric/Synapse ``notebookutils`` first (handles AAD/MSI auth
    transparently inside Fabric), then falls back to ``fsspec`` /
    ``adlfs``. Raises ``ImportError`` if neither is available.

    A reader that *is* available but *fails* (e.g., permission denied,
    not found) aborts the chain — we do not silently retry under a
    different identity, because that would mask auth/permission errors.
    """
    safe_uri = _redact_uri(uri)
    for reader in (_read_via_notebookutils, _read_via_fsspec):
        try:
            text = reader(uri)
        except Exception as exc:
            raise RuntimeError(f"Failed to read {safe_uri!r}: {exc}") from exc
        if text is not None:
            return text
    raise ImportError(
        f"To read {safe_uri!r} you must run inside Fabric/Synapse (so "
        "``notebookutils`` is importable) or install "
        "``fsspec`` + ``adlfs``. Alternatively, read the bytes yourself and "
        "pass the text/iterable/dicts to Trajectory.from_jsonl/from_dicts."
    )


@dataclass
class Trajectory:
    turns: list[TurnRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, record: TurnRecord) -> None:
        self.turns.append(record)

    def __iter__(self) -> Iterable[TurnRecord]:
        return iter(self.turns)

    def __len__(self) -> int:
        return len(self.turns)

    def __bool__(self) -> bool:
        # Always truthy. Without this, an empty-turns trajectory would test
        # falsy due to ``__len__`` falling back as Python's truthiness source,
        # which masks adaptive metadata, error reasons, and other fields that
        # downstream code reads via ``if traj: ...`` patterns.
        return True

    def __getitem__(self, index: int) -> TurnRecord:
        return self.turns[index]

    def to_dict(self) -> dict[str, Any]:
        return {"metadata": self.metadata, "turns": [turn.to_dict() for turn in self.turns]}

    def to_jsonl(self) -> str:
        lines = [json.dumps({"metadata": self.metadata}, ensure_ascii=False)]
        lines.extend(json.dumps(turn.to_dict(), ensure_ascii=False) for turn in self.turns)
        return "\n".join(lines) + "\n"

    def write_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_jsonl(), encoding="utf-8")

    def to_markdown(self) -> str:
        parts = ["# fabric-rlm trajectory", ""]
        for turn in self.turns:
            parts.extend(
                [
                    f"## Turn {turn.turn}",
                    "",
                    "```python",
                    turn.code,
                    "```",
                    "",
                    "**stdout**",
                    "",
                    "```text",
                    turn.stdout,
                    "```",
                ]
            )
            if turn.error:
                parts.extend(["", "**error**", "", "```text", turn.error, "```"])
        return "\n".join(parts)

    def write_markdown(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_dicts(cls, records: Iterable[Any]) -> "Trajectory":
        """Build a Trajectory from an iterable of dict-like records.

        The first record may be a metadata envelope (``{"metadata": {...}}``);
        all remaining records are turn records. This is the format produced by
        :meth:`write_jsonl` and is what callers will get from an MLflow
        artifact download or a Fabric Lakehouse Spark/notebookutils read of
        the JSONL file.

        Records may be plain ``dict`` or any duck-typed object exposing
        ``.asDict()`` (e.g. PySpark ``Row``). This makes the loader usable
        directly from a Spark DataFrame: ``Trajectory.from_dicts(df.collect())``.

        Forward-compatible: unknown keys on a turn record are ignored, and
        missing optional fields fall back to their dataclass defaults so old
        trajectories load without errors after we add new TurnRecord fields.
        """

        def _to_dict(rec: Any) -> dict[str, Any]:
            if isinstance(rec, dict):
                return rec
            if hasattr(rec, "asDict"):
                try:
                    return rec.asDict(recursive=True)
                except TypeError:
                    return rec.asDict()
            raise TypeError(
                f"Trajectory record must be dict or have .asDict(), got {type(rec).__name__}"
            )

        records = [_to_dict(r) for r in records]
        metadata: dict[str, Any] = {}
        turn_records = records
        if (
            records
            and "metadata" in records[0]
            and "turn" not in records[0]
        ):
            metadata = dict(records[0]["metadata"])
            turn_records = records[1:]
        known = {f.name for f in fields(TurnRecord)}
        turns: list[TurnRecord] = []
        for raw in turn_records:
            clean = {k: v for k, v in raw.items() if k in known}
            clean.setdefault("stdout", "")
            clean.setdefault("stderr", "")
            clean.setdefault("error", None)
            clean.setdefault("submitted", False)
            clean.setdefault("state", {})
            clean.setdefault("code", "")
            clean.setdefault("turn", len(turns) + 1)
            turns.append(TurnRecord(**clean))
        return cls(turns=turns, metadata=metadata)

    @classmethod
    def from_jsonl(cls, source: str | Path | Any) -> "Trajectory":
        """Load a Trajectory from a JSONL source.

        ``source`` may be:

        * A local path (``str`` / ``Path``). Lakehouse-mounted paths like
          ``/lakehouse/default/Files/traces/x.jsonl`` are regular files on
          Fabric Spark and work without any extra setup. ``file://`` URIs
          pointing at the local machine are also accepted and converted
          back to a filesystem path.
        * An Azure storage URI (``abfss://``, ``abfs://``, ``wasbs://``,
          ``wasb://``). These are read via Fabric's ``notebookutils.fs``
          (preferred) or ``fsspec`` if installed. Both
          are imported lazily so the library stays dependency-free.
        * A file-like object with ``.read()``, or any iterable of
          bytes/str lines.

        Other remote schemes (``s3://``, ``https://``, ``gs://``) are not
        opened directly: read the bytes with your own client and pass the
        file-like, line iterable, or parsed dicts here.

        Note: any query/fragment in URIs is stripped from error messages
        so SAS tokens or other credentials in the URI are not echoed to
        notebook output or logs.
        """

        if isinstance(source, (str, Path)):
            from urllib.parse import urlsplit
            from urllib.request import url2pathname

            text_source = str(source)
            parts = urlsplit(text_source)
            # urlsplit treats Windows drive letters (e.g. ``C:\...``) as a
            # one-char scheme. Real URI schemes are always >1 character.
            scheme = parts.scheme.lower() if len(parts.scheme) > 1 else ""
            if scheme in _AZURE_STORAGE_SCHEMES:
                text = _read_azure_storage_text(text_source)
                lines: Iterable[str] = text.splitlines()
            elif scheme == "file":
                # ``file://`` URIs are local if netloc is empty or localhost.
                if parts.netloc not in ("", "localhost"):
                    raise ValueError(
                        f"file:// URI with non-local host "
                        f"{_redact_uri(text_source)!r} is not supported."
                    )
                local_path = url2pathname(parts.path)
                text = Path(local_path).read_text(encoding="utf-8")
                lines = text.splitlines()
            elif scheme:
                raise ValueError(
                    f"Remote URI {_redact_uri(text_source)!r} is not supported "
                    "directly. Read the bytes with your own client (fsspec / "
                    "mlflow.artifacts / Spark / requests) and pass the "
                    "file-like, line iterable, or parsed dicts to "
                    "Trajectory.from_jsonl() or Trajectory.from_dicts()."
                )
            else:
                text = Path(source).read_text(encoding="utf-8")
                lines = text.splitlines()
        elif hasattr(source, "read"):
            text = source.read()
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            lines = text.splitlines()
        else:
            decoded: list[str] = []
            for ln in source:
                if isinstance(ln, bytes):
                    ln = ln.decode("utf-8")
                decoded.append(ln)
            lines = decoded
        records = [json.loads(ln) for ln in lines if str(ln).strip()]
        return cls.from_dicts(records)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable rollup of the trajectory.

        Includes turn counts, submission status, total/cached/reasoning
        token usage, total LM call seconds, and a histogram of error
        classes (e.g. ``{"SyntaxError": 1}``). Token fields return
        ``None`` (not ``0``) when no turn carries that field — this lets
        dashboards distinguish "not instrumented" from "zero usage".
        Designed to be cheap and printable; safe to call on a 100-turn
        trajectory.
        """

        def _sum_or_none(attr: str) -> int | float | None:
            present = [getattr(t, attr) for t in self.turns if getattr(t, attr) is not None]
            if not present:
                return None
            return sum(present)

        sub = next((t for t in self.turns if t.submitted), None)
        errs = [t for t in self.turns if t.error]
        kinds: dict[str, int] = {}
        for t in errs:
            line = _last_error_line(t.error)
            kind = line.split(":", 1)[0].strip() or "Unknown"
            kinds[kind] = kinds.get(kind, 0) + 1
        lm_seconds = _sum_or_none("lm_call_seconds")
        return {
            "turns": len(self.turns),
            "submitted": sub is not None,
            "submit_turn": sub.turn if sub else None,
            "errors": len(errs),
            "error_kinds": kinds,
            "prompt_tokens": _sum_or_none("prompt_tokens"),
            "completion_tokens": _sum_or_none("completion_tokens"),
            "total_tokens": _sum_or_none("total_tokens"),
            "cached_tokens": _sum_or_none("cached_tokens"),
            "reasoning_tokens": _sum_or_none("reasoning_tokens"),
            "lm_seconds": round(float(lm_seconds), 3) if lm_seconds is not None else None,
            "metadata": dict(self.metadata),
        }

    def diagnose(self) -> list[Issue]:
        """Return a list of detected issues across detectors.

        Detectors are intentionally cheap and conservative — each catches
        a class of failure we have actually seen in production traces.
        New detectors should follow the same pattern: walk turns once,
        emit ``Issue`` entries, never raise.
        """

        issues: list[Issue] = []
        issues.extend(_detect_markdown_in_code(self.turns))
        issues.extend(_detect_repeated_error(self.turns))
        issues.extend(_detect_noop_turn(self.turns))
        issues.extend(_detect_token_cliff(self.turns))
        issues.extend(_detect_independent_dimension_filters(self.turns))
        issues.extend(_detect_ranking_drift(self.turns))
        issues.sort(key=lambda i: (i.turn, i.kind))
        return issues


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def _detect_markdown_in_code(turns: Sequence[TurnRecord]) -> Iterator[Issue]:
    """Catch turns whose code contains bare markdown prose.

    Background: the model occasionally emits PLAN/VERIFY blocks as bare
    markdown (bullets like ``- foo``, label lines like ``Target: ...``,
    bold spans like ``**foo**``) instead of wrapping each line in a
    Python comment. Because turn bodies are executed as Python, the
    first such non-comment prose line raises ``SyntaxError``. The current
    ``skills/core.md`` wording (commit 77bd9c3) reduces this drastically
    but does not eliminate it across all models — this detector flags
    every occurrence so regressions are obvious.

    A line that begins with ``#`` is treated as a Python comment and
    ignored, so legitimate ``# ## PLAN`` headings inside comments don't
    trip the detector.
    """

    for t in turns:
        for ln in (t.code or "").splitlines():
            stripped = ln.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            if _MD_PROSE_RE.match(ln):
                yield Issue(
                    turn=t.turn,
                    kind="markdown_in_code",
                    message=(
                        f"Code contains bare markdown line {stripped[:60]!r}. "
                        "Will raise SyntaxError. Wrap PLAN/VERIFY in Python comments."
                    ),
                )
                break  # one issue per turn


def _detect_repeated_error(turns: Sequence[TurnRecord]) -> Iterator[Issue]:
    """Flag any run of three-or-more consecutive turns sharing an error class.

    Three rather than two consecutive errors is the threshold because two
    in a row are common during normal recovery (a fix attempt that itself
    has a bug). We only emit one issue per streak — at the streak's last
    turn — and we include the full streak length and start/end turns so
    the user can see how long the loop ran.
    """

    def _kind(t: TurnRecord) -> str:
        return _last_error_line(t.error).split(":", 1)[0].strip()

    streak_kind: str | None = None
    streak_start: int = 0
    streak_len: int = 0
    last_turn_in_streak: int = 0
    issues: list[Issue] = []

    def _flush() -> None:
        if streak_kind and streak_len >= 3:
            issues.append(
                Issue(
                    turn=last_turn_in_streak,
                    kind="repeated_error",
                    message=(
                        f"Same error class {streak_kind!r} on {streak_len} consecutive turns "
                        f"(turns {streak_start}..{last_turn_in_streak}) -- likely stuck loop."
                    ),
                )
            )

    for t in turns:
        if not t.error:
            _flush()
            streak_kind = None
            streak_len = 0
            continue
        kind = _kind(t)
        if kind == streak_kind:
            streak_len += 1
            last_turn_in_streak = t.turn
        else:
            _flush()
            streak_kind = kind
            streak_start = t.turn
            last_turn_in_streak = t.turn
            streak_len = 1
    _flush()
    yield from issues


def _detect_noop_turn(turns: Sequence[TurnRecord]) -> Iterator[Issue]:
    """Flag turns that produced no output, no error, and did not submit.

    These are wasted LM calls — usually empty code blocks or pure
    whitespace. They almost always indicate a prompt-formatting bug or a
    model that's idling.
    """

    for t in turns:
        if t.submitted or t.error:
            continue
        if (t.stdout or "").strip():
            continue
        if (t.code or "").strip():
            continue
        yield Issue(
            turn=t.turn,
            kind="noop_turn",
            message="Turn produced no code, no stdout, no error, and did not submit.",
        )


def _detect_token_cliff(turns: Sequence[TurnRecord]) -> Iterator[Issue]:
    """Flag any single turn whose prompt_tokens >> mean of OTHER turns.

    Usually means an entire input blob got pasted back into the prompt,
    which is a strong signal the model is over-quoting context. Computing
    the baseline from the *other* turns prevents the cliff turn from
    diluting its own threshold (a true cliff at 50K with neighbours at
    1K should always trip, even if there are several cliffs). We use a
    3x-baseline threshold and require an absolute floor of 5K tokens to
    avoid noise on tiny trajectories.
    """

    counts = [(t, t.prompt_tokens) for t in turns if t.prompt_tokens]
    if len(counts) < 3:
        return
    total = sum(c for _, c in counts)
    n_others = len(counts) - 1
    for t, c in counts:
        baseline = (total - c) / n_others
        threshold = max(baseline * 3.0, 5000.0)
        if c > threshold:
            yield Issue(
                turn=t.turn,
                kind="token_cliff",
                message=(
                    f"prompt_tokens={c:,} is >3x baseline of other turns "
                    f"({baseline:,.0f}). Likely pasted-back input."
                ),
            )


# ---------------------------------------------------------------------------
# Analytical-integrity detectors
#
# These read the code the model wrote, not the data it saw, so they are the
# same for a CSV, a Lakehouse query and a semantic-model aggregate.
# ---------------------------------------------------------------------------

_PLAN_HEADING_RE = re.compile(r"^\s*#\s*#{1,3}\s*PLAN\b", re.IGNORECASE)
_PLAN_END_RE = re.compile(r"^\s*#\s*#{1,3}\s*(?!PLAN)\w+", re.IGNORECASE)


def extract_plan(turns: Sequence[TurnRecord]) -> str | None:
    """Return the text of the first ``## PLAN`` comment block, if any.

    The core skill asks for the plan as a Python comment block on the
    first turn. Nothing parsed it before; the ranking-drift detector needs
    the ranking concept the plan committed to.
    """
    for t in turns:
        lines = (t.code or "").splitlines()
        for index, line in enumerate(lines):
            if not _PLAN_HEADING_RE.match(line):
                continue
            body: list[str] = []
            for following in lines[index + 1:]:
                stripped = following.strip()
                if not stripped.startswith("#") or _PLAN_END_RE.match(following):
                    break
                body.append(stripped.lstrip("#").strip())
            return "\n".join(body).strip() or None
    return None


def _string_constants(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple)):
        out: list[str] = []
        for elt in node.elts:
            out.extend(_string_constants(elt))
        return out
    return []


def _parse(code: str | None) -> ast.Module | None:
    try:
        return ast.parse(code or "")
    except (SyntaxError, ValueError):
        return None


def _unique_list_source(value: ast.AST) -> tuple[str, str] | None:
    """``frame["col"].unique()`` (or ``.tolist()``/``list(...)``/``set(...)``) -> (frame, col)."""
    node = value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"list", "set", "sorted"} and node.args:
        node = node.args[0]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"tolist", "to_list"}:
        node = node.func.value
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    if node.func.attr not in {"unique", "drop_duplicates"}:
        return None
    target = node.func.value
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        cols = _string_constants(target.slice)
        if len(cols) == 1:
            return target.value.id, cols[0]
    return None


def _isin_names(node: ast.AST) -> list[str]:
    """Names passed to ``.isin(name)`` inside a ``&``-combined filter."""
    names: list[str] = []
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
        names.extend(_isin_names(node.left))
        names.extend(_isin_names(node.right))
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "isin":
        if node.args and isinstance(node.args[0], ast.Name):
            names.append(node.args[0].id)
    return names


def _detect_independent_dimension_filters(turns: Sequence[TurnRecord]) -> Iterator[Issue]:
    """Flag candidate tuples collapsed into independent per-dimension lists.

    Seen live: candidates chosen at Product x Region x Customer Group were
    turned into ``products = c["product"].unique()``, ``regions = ...``,
    ``groups = ...`` and a later frame was filtered with three ``.isin``
    calls joined by ``&``. That admits every combination of the three
    lists, including ones never selected. The fix is a tuple merge
    (``restrict_to_candidate_tuples``).
    """
    lists: dict[str, tuple[str, str]] = {}
    for t in turns:
        tree = _parse(t.code)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                source = _unique_list_source(node.value)
                if source is not None:
                    lists[node.targets[0].id] = source
        if not lists:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd)):
                continue
            names = [n for n in _isin_names(node) if n in lists]
            frames = {lists[n][0] for n in names}
            if len(names) >= 2 and len(frames) == 1:
                cols = [lists[n][1] for n in names]
                frame = frames.pop()
                yield Issue(
                    turn=t.turn,
                    kind="cartesian_candidate_filter",
                    message=(
                        f"Filters by independent lists {names} taken from {frame}"
                        f"[{cols}] admit every combination of those values, not only "
                        f"the candidates selected in {frame}. Keep the compound "
                        "identity: restrict_to_candidate_tuples(frame, candidates, "
                        f"keys={cols})."
                    ),
                )
                break


def _sort_fields(tree: ast.Module) -> list[str]:
    fields: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in {"sort_values", "nlargest", "nsmallest"}:
            continue
        for kw in node.keywords:
            if kw.arg in {"by", "columns"}:
                fields.extend(_string_constants(kw.value))
        if node.func.attr == "sort_values" and node.args:
            fields.extend(_string_constants(node.args[0]))
        elif node.func.attr in {"nlargest", "nsmallest"} and len(node.args) >= 2:
            fields.extend(_string_constants(node.args[1]))
    return fields


def _defined_columns(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    names.extend(_string_constants(target.slice))
                elif isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "assign":
            names.extend(kw.arg for kw in node.keywords if kw.arg)
    return names


def detect_ranking_drift(turns: Sequence[TurnRecord], request: Any) -> Issue | None:
    """The code sorted by something other than the requested ranking concept.

    ``request`` is a :class:`fabric_rlm.analytical_integrity.RankingRequest`.
    Drift is reported when the trajectory sorts, but no sort field and no
    column it defined relates to the concept. A defined column that names
    the concept (``summary["impact"] = ...``) is taken as the metric.
    """
    from .analytical_integrity import _stems_overlap, _tokens

    if request is None:
        return None
    sort_fields: list[tuple[int, str]] = []
    defined: list[str] = []
    for t in turns:
        tree = _parse(t.code)
        if tree is None:
            continue
        sort_fields.extend((t.turn, f) for f in _sort_fields(tree))
        defined.extend(_defined_columns(tree))
    if not sort_fields:
        return None
    concept = list(request.tokens)
    if any(_stems_overlap(concept, _tokens(f)) for _turn, f in sort_fields):
        return None
    if any(_stems_overlap(concept, _tokens(name)) for name in defined):
        return None
    last_turn, last_field = sort_fields[-1]
    return Issue(
        turn=last_turn,
        kind="ranking_drift",
        message=(
            f"The task asked to rank by {request.concept!r} but the code sorted by "
            f"{sorted({f for _t, f in sort_fields})} and never defined a metric for "
            f"{request.concept!r}. Derive that metric, sort by it, and show it."
        ),
    )


def _detect_ranking_drift(turns: Sequence[TurnRecord]) -> Iterator[Issue]:
    """Plan-to-execution drift: the PLAN promised a ranking the code did not do."""
    from .analytical_integrity import infer_requested_ranking

    plan = extract_plan(turns)
    if not plan:
        return
    issue = detect_ranking_drift(turns, infer_requested_ranking(plan))
    if issue is not None:
        yield issue
