"""Dependency-free HTML inspection for RLM trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runtime import RLMResult
    from .trajectory import TurnRecord


_STYLES = """
<style>
.frlm-inspector {
  --frlm-bg: #ffffff;
  --frlm-surface: #f6f8fa;
  --frlm-border: #d0d7de;
  --frlm-text: #1f2328;
  --frlm-muted: #59636e;
  --frlm-accent: #0969da;
  --frlm-good: #1a7f37;
  --frlm-warn: #9a6700;
  --frlm-bad: #cf222e;
  color: var(--frlm-text);
  background: var(--frlm-bg);
  border: 1px solid var(--frlm-border);
  border-radius: 6px;
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  overflow: hidden;
}
.frlm-inspector-summary { cursor: pointer; display: flex; align-items: center; gap: 10px; padding: 14px 16px; }
.frlm-inspector-summary:hover { background: var(--frlm-surface); }
.frlm-inspector-title { font-size: 18px; font-weight: 600; margin-right: auto; }
.frlm-inspector-content { border-top: 1px solid var(--frlm-border); }
.frlm-header { padding: 16px; border-bottom: 1px solid var(--frlm-border); }
.frlm-subtitle { color: var(--frlm-muted); margin: 0; }
.frlm-summary { display: flex; flex-wrap: wrap; gap: 8px; padding: 12px 16px; }
.frlm-metric { background: var(--frlm-surface); border: 1px solid var(--frlm-border); border-radius: 4px; padding: 6px 10px; }
.frlm-metric strong { display: block; font-size: 16px; }
.frlm-metric span { color: var(--frlm-muted); font-size: 12px; }
.frlm-turns { border-top: 1px solid var(--frlm-border); }
.frlm-turn { border-bottom: 1px solid var(--frlm-border); }
.frlm-turn:last-child { border-bottom: 0; }
.frlm-turn > summary { cursor: pointer; display: flex; align-items: center; gap: 8px; padding: 12px 16px; }
.frlm-turn > summary:hover { background: var(--frlm-surface); }
.frlm-turn-title { font-weight: 600; margin-right: auto; }
.frlm-badge { border: 1px solid currentColor; border-radius: 999px; font-size: 11px; font-weight: 600; padding: 1px 7px; }
.frlm-good { color: var(--frlm-good); }
.frlm-warn { color: var(--frlm-warn); }
.frlm-bad { color: var(--frlm-bad); }
.frlm-neutral { color: var(--frlm-muted); }
.frlm-body { padding: 0 16px 16px; }
.frlm-section { margin-top: 10px; }
.frlm-section > summary { cursor: pointer; font-weight: 600; padding: 4px 0; }
.frlm-inspector pre { background: var(--frlm-surface); border: 1px solid var(--frlm-border); border-radius: 4px; margin: 6px 0 0; max-height: 420px; overflow: auto; padding: 10px; white-space: pre-wrap; word-break: break-word; }
.frlm-empty { color: var(--frlm-muted); padding: 20px 16px; }
@media (prefers-color-scheme: dark) {
  .frlm-inspector {
    --frlm-bg: #0d1117; --frlm-surface: #161b22; --frlm-border: #30363d;
    --frlm-text: #e6edf3; --frlm-muted: #8b949e; --frlm-accent: #58a6ff;
    --frlm-good: #3fb950; --frlm-warn: #d29922; --frlm-bad: #f85149;
  }
}
</style>
""".strip()


def _format_number(value: Any, *, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{value}{suffix}"


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


@dataclass(frozen=True)
class RunInspector:
    """Notebook-renderable, standalone-exportable view of an RLM run."""

    result: "RLMResult"
    max_chars: int = 20_000
    slow_turn_seconds: float = 10.0
    expanded: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_chars, bool)
            or not isinstance(self.max_chars, int)
            or self.max_chars <= 0
        ):
            raise ValueError("max_chars must be greater than zero.")
        if (
            isinstance(self.slow_turn_seconds, bool)
            or not isinstance(self.slow_turn_seconds, Real)
            or self.slow_turn_seconds <= 0
        ):
            raise ValueError("slow_turn_seconds must be greater than zero.")
        if not isinstance(self.expanded, bool):
            raise ValueError("expanded must be a bool.")

    def _text(self, value: Any) -> str:
        text = "" if value is None else str(value)
        if len(text) > self.max_chars:
            omitted = len(text) - self.max_chars
            text = f"{text[:self.max_chars]}\n\n... [{omitted:,} characters truncated]"
        return escape(text)

    def _section(self, title: str, value: Any, *, open_by_default: bool = False) -> str:
        if value in (None, "", [], {}):
            return ""
        text = _json_text(value) if isinstance(value, (dict, list, tuple)) else value
        opened = " open" if open_by_default else ""
        return (
            f'<details class="frlm-section"{opened}>'
            f"<summary>{escape(title)}</summary><pre>{self._text(text)}</pre></details>"
        )

    @staticmethod
    def _elapsed(turn: "TurnRecord") -> float | None:
        parts = (turn.lm_call_seconds, turn.worker_execute_seconds)
        if any(value is not None for value in parts):
            return sum(value or 0.0 for value in parts)
        return turn.duration_s

    def _turn_html(self, turn: "TurnRecord", *, recovered: bool = False) -> str:
        elapsed = self._elapsed(turn)
        badges: list[str] = []
        if turn.error:
            badges.append('<span class="frlm-badge frlm-bad">Error</span>')
        if turn.validation_errors:
            badges.append('<span class="frlm-badge frlm-warn">Validation</span>')
        if turn.turn_type != "normal":
            badges.append('<span class="frlm-badge frlm-warn">Repair</span>')
        if recovered:
            badges.append('<span class="frlm-badge frlm-good">Recovered</span>')
        if elapsed is not None and elapsed >= self.slow_turn_seconds:
            badges.append('<span class="frlm-badge frlm-warn">Slow</span>')
        if turn.submitted:
            badges.append('<span class="frlm-badge frlm-good">Submitted</span>')
        if not badges:
            badges.append('<span class="frlm-badge frlm-neutral">Completed</span>')

        metrics = {
            "elapsed_seconds": elapsed,
            "lm_seconds": turn.lm_call_seconds,
            "worker_seconds": turn.worker_execute_seconds,
            "prompt_tokens": turn.prompt_tokens,
            "completion_tokens": turn.completion_tokens,
            "reasoning_tokens": turn.reasoning_tokens,
            "cached_tokens": turn.cached_tokens,
            "state_keys": turn.state_keys,
        }
        sections = [
            self._section("Model response", turn.response_text),
            self._section("Code", turn.code, open_by_default=True),
            self._section("Output", turn.stdout, open_by_default=True),
            self._section("Stderr", turn.stderr),
            self._section("Error", turn.error, open_by_default=True),
            self._section("Validator feedback", turn.validation_errors, open_by_default=True),
            self._section("Submitted payload", turn.submit_payload, open_by_default=True),
            self._section("Metrics", metrics),
        ]
        opened = " open" if turn.error or turn.submitted else ""
        return (
            f'<details class="frlm-turn"{opened}><summary>'
            f'<span class="frlm-turn-title">Turn {turn.turn}</span>'
            f'{"".join(badges)}'
            f'<span class="frlm-neutral">{_format_number(elapsed, suffix="s")}</span>'
            f'</summary><div class="frlm-body">{"".join(sections)}</div></details>'
        )

    def to_html(self) -> str:
        """Return a safe HTML fragment suitable for notebook display."""

        facts = self.result.report(as_dict=True)
        status = "SUBMITTED" if self.result.submitted else "NOT SUBMITTED"
        status_class = "frlm-good" if self.result.submitted else "frlm-bad"
        metrics = (
            ("Status", status),
            ("Turns", facts.get("turns")),
            ("Errors", facts.get("errors")),
            ("LM time", _format_number(facts.get("lm_seconds"), suffix="s")),
            ("Worker time", _format_number(facts.get("worker_seconds"), suffix="s")),
            ("Prompt tokens", facts.get("prompt_tokens")),
            ("Reasoning tokens", facts.get("reasoning_tokens")),
        )
        cards = "".join(
            f'<div class="frlm-metric"><strong class="{status_class if label == "Status" else ""}">'
            f'{escape(_format_number(value))}</strong><span>{escape(label)}</span></div>'
            for label, value in metrics
        )
        if self.result.turns:
            turns = "".join(
                self._turn_html(
                    turn,
                    recovered=index > 0 and bool(self.result.turns[index - 1].error),
                )
                for index, turn in enumerate(self.result.turns)
            )
        else:
            turns = '<div class="frlm-empty">No executable turns were recorded.</div>'
        opened = " open" if self.expanded else ""
        return (
            f'{_STYLES}<details class="frlm-inspector"{opened}>'
            '<summary class="frlm-inspector-summary">'
            '<span class="frlm-inspector-title">RLM run inspector</span>'
            f'<span class="frlm-badge {status_class}">{escape(status)}</span>'
            f'<span class="frlm-neutral">{self.result.n_turns} turns</span>'
            "</summary><div class=\"frlm-inspector-content\">"
            "<header class=\"frlm-header\">"
            "<p class=\"frlm-subtitle\">Observable model responses, code, execution, "
            "validation, and performance by turn.</p></header>"
            f'<div class="frlm-summary">{cards}</div>'
            f'<div class="frlm-turns">{turns}</div></div></details>'
        )

    def _repr_html_(self) -> str:
        return self.to_html()

    def save_html(self, path: str | Path) -> Path:
        """Write a standalone inspector document and return its path."""

        destination = Path(path)
        document = (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>RLM run inspector</title></head><body>"
            f"{self.to_html()}</body></html>"
        )
        destination.write_text(document, encoding="utf-8")
        return destination
