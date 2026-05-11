"""Trajectory records and exports."""

from __future__ import annotations

import json
import re
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
# bold spans (``**foo**``), and label-style lines (``Target:``, ``Output:``).
# A single such line outside a string literal will raise SyntaxError on
# execution; we flag any code block containing one or more.
_MD_PROSE_RE = re.compile(
    r"""^\s*(
        -\ |\*\ |          # bullet list
        \*\*[^*]+\*\*|      # bold span at line start
        [A-Z][A-Za-z]+:\s   # label (Target: / Output: / Step: ...)
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
    def from_dicts(cls, records: Iterable[dict[str, Any]]) -> "Trajectory":
        """Build a Trajectory from an iterable of plain dicts.

        The first dict may be a metadata envelope (``{"metadata": {...}}``);
        all remaining dicts are turn records. This is the format produced by
        :meth:`write_jsonl` and is what callers will get from an MLflow
        artifact download or a Fabric Lakehouse Spark/notebookutils read of
        the JSONL file.

        Forward-compatible: unknown keys on a turn record are ignored, and
        missing optional fields fall back to their dataclass defaults so old
        trajectories load without errors after we add new TurnRecord fields.
        """

        records = list(records)
        metadata: dict[str, Any] = {}
        turn_records = records
        if (
            records
            and isinstance(records[0], dict)
            and "metadata" in records[0]
            and "turn" not in records[0]
        ):
            metadata = dict(records[0]["metadata"])
            turn_records = records[1:]
        known = {f.name for f in fields(TurnRecord)}
        turns: list[TurnRecord] = []
        for raw in turn_records:
            if not isinstance(raw, dict):
                raise TypeError(f"Trajectory record must be dict, got {type(raw).__name__}")
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

        ``source`` may be a path (``str`` / ``Path``), a file-like object
        with ``.read()``, or any iterable of bytes/str lines. For Fabric
        Lakehouse, read the file with notebookutils/fsspec/Spark first and
        pass the parsed dicts to :meth:`from_dicts`, or pass the line
        iterable here directly.
        """

        if isinstance(source, (str, Path)):
            text = Path(source).read_text(encoding="utf-8")
            lines: Iterable[str] = text.splitlines()
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
        classes (e.g. ``{"SyntaxError": 1}``). Designed to be cheap and
        printable; safe to call on a 100-turn trajectory.
        """

        def _sum(attr: str) -> int | float:
            return sum((getattr(t, attr) or 0) for t in self.turns)

        sub = next((t for t in self.turns if t.submitted), None)
        errs = [t for t in self.turns if t.error]
        kinds: dict[str, int] = {}
        for t in errs:
            line = (t.error or "").strip().splitlines()[-1] if t.error else ""
            kind = line.split(":", 1)[0].strip() or "Unknown"
            kinds[kind] = kinds.get(kind, 0) + 1
        return {
            "turns": len(self.turns),
            "submitted": sub is not None,
            "submit_turn": sub.turn if sub else None,
            "errors": len(errs),
            "error_kinds": kinds,
            "prompt_tokens": int(_sum("prompt_tokens")),
            "completion_tokens": int(_sum("completion_tokens")),
            "total_tokens": int(_sum("total_tokens")),
            "cached_tokens": int(_sum("cached_tokens")),
            "reasoning_tokens": int(_sum("reasoning_tokens")),
            "lm_seconds": round(float(_sum("lm_call_seconds")), 3),
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
    """Flag two-or-more consecutive turns with the same error class.

    A repeated error class usually indicates the model is stuck in a
    loop and not learning from feedback. We flag the second occurrence
    (the first is just a normal recoverable failure).
    """

    last_kind: str | None = None
    streak = 0
    for t in turns:
        if not t.error:
            last_kind = None
            streak = 0
            continue
        kind = t.error.strip().splitlines()[-1].split(":", 1)[0].strip()
        if kind == last_kind:
            streak += 1
            if streak == 1:
                yield Issue(
                    turn=t.turn,
                    kind="repeated_error",
                    message=f"Same error class {kind!r} on two consecutive turns — possible loop.",
                )
        else:
            last_kind = kind
            streak = 0


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
    """Flag any single turn whose prompt_tokens >> mean.

    Usually means an entire input blob got pasted back into the prompt,
    which is a strong signal the model is over-quoting context. We use
    a 3x-mean threshold and require an absolute floor of 5K tokens to
    avoid noise on tiny trajectories.
    """

    counts = [t.prompt_tokens for t in turns if t.prompt_tokens]
    if len(counts) < 3:
        return
    mean = sum(counts) / len(counts)
    threshold = max(mean * 3.0, 5000.0)
    for t in turns:
        if t.prompt_tokens and t.prompt_tokens > threshold:
            yield Issue(
                turn=t.turn,
                kind="token_cliff",
                message=(
                    f"prompt_tokens={t.prompt_tokens:,} is >3x mean "
                    f"({mean:,.0f}). Likely pasted-back input."
                ),
            )

