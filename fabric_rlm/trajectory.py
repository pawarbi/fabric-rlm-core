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


# Calls that merely look at the lists rather than consuming them together.
_TRIVIAL_CONSUMERS = {"print", "len", "repr", "str", "display", "type", "isinstance", "set", "list", "sorted", "tuple"}


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _referenced_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _norm_column(name: Any) -> str:
    """Leaf column name, lowercase alphanumerics: ``Sold To[Region]`` -> ``region``."""
    text = str(name)
    match = re.match(r"^\s*'?([^'\[\]]+?)'?\s*\[([^\[\]]+)\]\s*$", text)
    if match:
        text = match.group(2)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _statements(turns: Sequence[TurnRecord]) -> list[tuple[int, ast.stmt]]:
    """Every top-level statement of every parsable turn, in program order."""
    out: list[tuple[int, ast.stmt]] = []
    for t in turns:
        tree = _parse(t.code)
        if tree is None:
            continue
        out.extend((t.turn, stmt) for stmt in tree.body)
    return out


def _call_keyword_strings(node: ast.Call, names: set[str], position: int | None = None) -> list[str]:
    fields: list[str] = []
    for kw in node.keywords:
        if kw.arg in names:
            fields.extend(_string_constants(kw.value))
    if position is not None and len(node.args) > position:
        fields.extend(_string_constants(node.args[position]))
    return fields


def _is_tuple_repair(statement: ast.stmt, columns: Sequence[str]) -> bool:
    """A restriction to compound identities on exactly these dimensions.

    ``restrict_to_candidate_tuples(..., keys=[...])`` or ``.merge(..., on=[...])``
    whose keys cover every flagged column. A merge on other keys (an
    unrelated lookup join) is not a repair.
    """
    wanted = {_norm_column(c) for c in columns}
    for node in ast.walk(statement):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name == "restrict_to_candidate_tuples":
            keys = _call_keyword_strings(node, {"keys"}, position=2)
            if keys and wanted <= {_norm_column(k) for k in keys}:
                return True
        elif name == "merge":
            keys = _call_keyword_strings(node, {"on", "left_on", "right_on"}, position=1)
            if keys and wanted <= {_norm_column(k) for k in keys}:
                return True
    return False


def _repaired_downstream(
    later: Sequence[tuple[int, ast.stmt]], columns: Sequence[str], outputs: set[str]
) -> bool:
    """Whether a later statement restores the compound identity of the
    expanded result itself.

    ``outputs`` are the variables the consumer wrote (``seg_3q`` for
    ``seg_3q = model.aggregate(...)``). A repair must reference one of them
    or an alias derived from them (``hist = seg_3q.rename(...)`` then
    ``restrict_to_candidate_tuples(hist, ...)``) and cover every flagged
    column; a merge on the same keys between unrelated frames is not a
    repair of this result.
    """
    expanded = set(outputs)
    if not expanded:
        return False
    for _turn, statement in later:
        reads = _referenced_names(statement)
        if reads & expanded and _is_tuple_repair(statement, columns):
            return True
        writes = _written_names(statement)
        if reads & expanded and writes:
            expanded |= writes
    return False


def _joint_consumer(
    statement: ast.stmt, lists: dict[str, tuple[str, str]]
) -> tuple[str, str, list[str], list[str]] | None:
    """(consumer, frame, list names, columns) when a statement consumes two or
    more lists from different columns of one frame together."""
    for node in ast.walk(statement):
        names: list[str] = []
        if isinstance(node, ast.Call):
            if _call_name(node) in _TRIVIAL_CONSUMERS:
                continue
            args = list(node.args) + [kw.value for kw in node.keywords]
            names = sorted({n for arg in args for n in _referenced_names(arg)} & set(lists))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            names = sorted({n for n in _isin_names(node) if n in lists})
        if len(names) < 2:
            continue
        by_frame: dict[str, list[str]] = {}
        for name in names:
            by_frame.setdefault(lists[name][0], []).append(name)
        for frame, group in by_frame.items():
            cols = sorted({lists[n][1] for n in group})
            if len(cols) >= 2:
                consumer = _call_name(node) if isinstance(node, ast.Call) else "isin filter"
                return consumer, frame, group, cols
    return None


def _detect_independent_dimension_filters(turns: Sequence[TurnRecord]) -> Iterator[Issue]:
    """Flag candidate tuples collapsed into independent per-dimension lists
    and not restored afterwards.

    The pattern, source-agnostic: two or more value lists are taken from
    different columns of the same frame (``c["product"].unique()``,
    ``c["region"].unique().tolist()``, ...) and a later operation consumes
    them together. Seen live as ``aggregate(..., filters={"Product":
    prod_vals, "Region": reg_vals, "Customer Group": cust_vals})`` on a
    semantic model, and as ``df[a.isin(x) & b.isin(y) & c.isin(z)]`` on a
    frame. Either admits every combination of the lists, including ones the
    candidate frame never contained.

    The issue is tied to its provenance: it is dropped only when a later
    statement (same turn or later) restores the compound identity of the
    expanded result itself, referencing the variable the consumer wrote or
    an alias derived from it, via ``restrict_to_candidate_tuples(...,
    keys=[...])`` or a ``merge(..., on=[...])`` covering those same
    dimensions. Retrieving a bounded superset and restricting it at once is
    therefore fine; a merge on other keys, or on the same keys between
    unrelated frames, does not clear it.
    """
    statements = _statements(turns)
    lists: dict[str, tuple[str, str]] = {}
    reported_turns: set[int] = set()
    for position, (turn, statement) in enumerate(statements):
        if lists and turn not in reported_turns:
            found = _joint_consumer(statement, lists)
            if found is not None:
                consumer, frame, group, cols = found
                repaired = _is_tuple_repair(statement, cols) or _repaired_downstream(
                    statements[position + 1:], cols, _written_names(statement)
                )
                if not repaired:
                    reported_turns.add(turn)
                    yield Issue(
                        turn=turn,
                        kind="cartesian_candidate_filter",
                        message=(
                            f"{consumer} consumes independent lists {group} taken from "
                            f"{frame}[{cols}] together, which admits every combination of "
                            f"those values, not only the candidates in {frame}. Keep the "
                            "compound identity: restrict_to_candidate_tuples(frame, "
                            f"candidates, keys={cols})."
                        ),
                    )
        for node in ast.walk(statement):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                source = _unique_list_source(node.value)
                if source is not None:
                    lists[node.targets[0].id] = source


# -- ranking drift -------------------------------------------------------------

_SORT_METHODS = {"sort_values", "nlargest", "nsmallest"}
# polars / pyspark / generic spellings; string arguments name the sort fields
_GENERIC_SORT_METHODS = {"sort", "orderBy", "order_by", "sort_by", "top_k"}
_SQL_ORDER_BY_RE = re.compile(
    r"\border\s+by\s+(?P<fields>.+?)(?:\s+(?:limit|offset|fetch)\b|;|$)",
    re.IGNORECASE | re.DOTALL,
)


def _lambda_fields(node: ast.AST) -> list[str]:
    """Fields a ``key=lambda r: r["impact"]`` or ``r.impact`` sorts on."""
    fields: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Subscript):
            fields.extend(_string_constants(sub.slice))
        elif isinstance(sub, ast.Attribute):
            fields.append(sub.attr)
    return fields


def _sort_fields_in_statement(statement: ast.stmt) -> list[str]:
    fields: list[str] = []
    for node in ast.walk(statement):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _SORT_METHODS:
                fields.extend(_call_keyword_strings(node, {"by", "columns"}))
                if name == "sort_values" and node.args:
                    fields.extend(_string_constants(node.args[0]))
                elif name in {"nlargest", "nsmallest"} and len(node.args) >= 2:
                    fields.extend(_string_constants(node.args[1]))
            elif name in _GENERIC_SORT_METHODS and isinstance(node.func, ast.Attribute):
                fields.extend(_call_keyword_strings(node, {"by", "columns"}))
                for arg in node.args:
                    fields.extend(_string_constants(arg))
            elif name == "sorted" and isinstance(node.func, ast.Name):
                for kw in node.keywords:
                    if kw.arg == "key":
                        fields.extend(_lambda_fields(kw.value))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "order" in node.value.lower():
            for match in _SQL_ORDER_BY_RE.finditer(node.value):
                for part in match.group("fields").split(","):
                    token = re.sub(r"\s+(asc|desc)\b.*$", "", part.strip(), flags=re.IGNORECASE)
                    token = token.strip().strip("\"'`[]")
                    if token:
                        fields.append(token.split(".")[-1].strip("\"'`[]"))
    return fields


_MUTATORS = {"append", "extend", "insert", "update", "add", "setdefault", "sort", "reverse", "pop", "remove", "clear"}


def _written_names(statement: ast.stmt) -> set[str]:
    """Names a statement assigns or mutates, including loop targets."""
    written: set[str] = set()

    def base(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            written.add(target.id)
        elif isinstance(target, (ast.Subscript, ast.Attribute)):
            base(target.value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                base(elt)
        elif isinstance(target, ast.Starred):
            base(target.value)

    for node in ast.walk(statement):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                base(target)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            base(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            base(node.target)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            base(node.optional_vars)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATORS:
            base(node.func.value)
    return written


def _answer_closure(statements: Sequence[tuple[int, ast.stmt]]) -> set[str]:
    """Names whose values reach the last SUBMIT, by a backward slice over
    statements: whenever a statement writes a name in the closure, every
    name it reads joins it. Empty when SUBMIT was given only literals."""
    seeds: set[str] = set()
    for _turn, statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "SUBMIT":
                args = list(node.args) + [kw.value for kw in node.keywords]
                seeds = {n for arg in args for n in _referenced_names(arg)}
    closure = set(seeds)
    for _ in range(50):
        grew = False
        for _turn, statement in statements:
            if _written_names(statement) & closure:
                new = _referenced_names(statement) - closure
                if new:
                    closure |= new
                    grew = True
        if not grew:
            break
    return closure


def _column_derivations(tree: ast.Module) -> dict[str, set[str]]:
    """Defined column or variable -> names and string columns its expression reads."""
    derived: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            reads = _referenced_names(node.value) | set(_string_constants_deep(node.value))
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    for col in _string_constants(target.slice):
                        derived.setdefault(col, set()).update(reads)
                elif isinstance(target, ast.Name):
                    derived.setdefault(target.id, set()).update(reads)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "assign":
            for kw in node.keywords:
                if kw.arg:
                    derived.setdefault(kw.arg, set()).update(
                        _referenced_names(kw.value) | set(_string_constants_deep(kw.value))
                    )
    return derived


def _string_constants_deep(node: ast.AST) -> list[str]:
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _defined_columns(tree: ast.Module) -> list[str]:
    return list(_column_derivations(tree))


def detect_ranking_drift(
    turns: Sequence[TurnRecord], request: Any, answer_text: str | None = None
) -> Issue | None:
    """The ranking that reaches the answer is not by the requested concept.

    ``request`` is a :class:`fabric_rlm.analytical_integrity.RankingRequest`.
    Sorts are recognised in pandas (``sort_values``, ``nlargest``,
    ``nsmallest``), polars and pyspark (``sort``, ``orderBy``), ``sorted(...,
    key=...)`` and SQL ``ORDER BY`` inside string constants; a best-effort
    set, not a parser for every engine.

    Which sort is "the ranking" follows the data, not the clock: a backward
    slice from the last SUBMIT finds the variables that reach the answer,
    and for each of them the last sort that wrote it counts. So ``ranked =
    df.sort_values("impact")`` followed by ``detail = hist.sort_values(
    "quarter")`` and ``SUBMIT(build(ranked, detail))`` is fine, while
    ``summary["impact"] = ...`` followed by ``summary =
    summary.sort_values("latest_arr")`` and an answer built from ``summary``
    is drift: defining the metric is not ranking by it. When no lineage can
    be established (SUBMIT was given literals) every sort is considered and
    any concept-related one satisfies the check. A field is related to the
    concept when its name overlaps the concept or it was derived, up to
    three assignments deep, from a field that does (``df["rank"] =
    df["impact"].rank()``), or when ``answer_text`` declares it as the
    ranking metric ("Ranking metric: absolute ARR loss" covers a sort by
    ``abs_loss``: every token of the field is accounted for by the
    declaration, see ``metric_matches_declaration``; it does not cover
    ``latest_arr``, so an answer cannot talk the check into accepting a sort
    it did not do). Two live runs sorted by a correctly derived loss column
    and were sent back only because it was not named "impact"; one of them
    then invented a stranger metric to satisfy the name. Whether a declared
    metric answers the concept is the disclosure check's question, not this
    one's.
    """
    from .analytical_integrity import (
        _stems_overlap,
        _tokens,
        declared_ranking_phrases,
        metric_matches_declaration,
    )

    if request is None:
        return None
    declared_phrases = [p for p in declared_ranking_phrases(answer_text) if _tokens(p)]
    statements = _statements(turns)
    sites: list[tuple[int, list[str], set[str]]] = []
    derived: dict[str, set[str]] = {}
    for turn, statement in statements:
        fields = _sort_fields_in_statement(statement)
        if fields:
            sites.append((turn, fields, _written_names(statement)))
        module = ast.Module(body=[statement], type_ignores=[])
        for name, reads in _column_derivations(module).items():
            derived.setdefault(name, set()).update(reads)
    if not sites:
        return None
    concept = list(request.tokens)

    def related(field: str, depth: int = 0) -> bool:
        field_tokens = _tokens(field)
        if _stems_overlap(concept, field_tokens):
            return True
        if any(metric_matches_declaration(field_tokens, phrase) for phrase in declared_phrases):
            return True
        if depth >= 3:
            return False
        return any(related(dep, depth + 1) for dep in derived.get(field, ()) if dep != field)

    closure = _answer_closure(statements)
    considered: list[tuple[int, list[str], set[str]]] = []
    if closure:
        last_by_variable: dict[str, tuple[int, list[str], set[str]]] = {}
        for site in sites:
            for name in site[2] & closure:
                last_by_variable[name] = site
        seen: set[int] = set()
        for site in last_by_variable.values():
            if id(site) not in seen:
                seen.add(id(site))
                considered.append(site)
    if not considered:
        considered = list(sites)
    if any(related(f) for _turn, fields, _w in considered for f in fields):
        return None
    from .analytical_integrity import concept_head

    head = concept_head(request.concept)
    concept_columns = [name for name in derived if head and _stems_overlap([head], _tokens(name))]
    defined_note = (
        f" A metric named {concept_columns[0]!r} was defined but the ranking that reaches the answer did not use it."
        if concept_columns
        else f" No metric for {request.concept!r} was defined."
    )
    last_turn, last_fields, _w = considered[-1]
    fields_shown = sorted({f for _t, fields, _w in considered for f in fields})
    return Issue(
        turn=last_turn,
        kind="ranking_drift",
        message=(
            f"The task asked to rank by {request.concept!r} but the ranking that reaches "
            f"the answer sorted by {fields_shown}.{defined_note} Derive the metric, sort by it, "
            "and show it."
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
