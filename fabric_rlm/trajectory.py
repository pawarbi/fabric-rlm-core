"""Trajectory records and exports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


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

