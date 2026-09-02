"""Trajectory records and exports."""

from __future__ import annotations

import json
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
            line = (t.error or "").strip().splitlines()[-1] if t.error else ""
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
        return t.error.strip().splitlines()[-1].split(":", 1)[0].strip() if t.error else ""

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
