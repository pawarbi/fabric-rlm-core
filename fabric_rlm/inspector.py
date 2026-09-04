"""Dependency-free HTML inspection for RLM trajectories."""

from __future__ import annotations

import ast
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
.frlm-inspector summary { list-style: none; }
.frlm-inspector summary::-webkit-details-marker { display: none; }
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
.frlm-turns {
  --frlm-turn-row-height: 3.25rem;
  border-top: 1px solid var(--frlm-border);
  max-height: calc(var(--frlm-visible-turns, 15) * var(--frlm-turn-row-height));
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-gutter: stable;
}
.frlm-turns:focus-visible { outline: 2px solid var(--frlm-accent); outline-offset: -2px; }
.frlm-turn > summary { border-bottom: 1px solid var(--frlm-border); box-sizing: border-box; cursor: pointer; display: flex; align-items: center; gap: 8px; min-height: var(--frlm-turn-row-height); padding: 12px 16px; }
.frlm-turn:last-child > summary { border-bottom: 0; }
.frlm-turn > summary:hover { background: var(--frlm-surface); }
.frlm-turn-title { font-weight: 600; white-space: nowrap; }
.frlm-turn-summary { color: var(--frlm-muted); margin-right: auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
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


def _humanize_identifier(value: str) -> str:
    text = value.strip().strip("[]").replace("_", " ")
    return " ".join(text.split())


def _literal_label(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _humanize_identifier(node.value)
    if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) == 1:
        return _literal_label(node.elts[0])
    return None


def _grouped_aggregation_summary(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Attribute):
        return None
    if call.func.attr not in {"sum", "mean", "median", "min", "max", "count", "nunique"}:
        return None

    value: ast.AST = call.func.value
    metric: str | None = None
    if isinstance(value, ast.Subscript):
        metric = _literal_label(value.slice)
        value = value.value
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "groupby"
        and value.args
    ):
        return None

    group = _literal_label(value.args[0])
    if not group:
        return None
    if call.func.attr in {"count", "nunique"} and not metric:
        return f"Counted records by {group}"
    target = metric or "values"
    return f"Aggregated {target} by {group}"


def _top_level_calls(tree: ast.Module, *, errored: bool) -> list[ast.Call]:
    class CallVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.calls: list[ast.Call] = []

        def visit_Call(self, node: ast.Call) -> None:
            self.visit(node.func)
            for argument in node.args:
                self.visit(argument)
            for keyword in node.keywords:
                self.visit(keyword.value)
            self.calls.append(node)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            return

        def visit_ListComp(self, node: ast.ListComp) -> None:
            return

        def visit_SetComp(self, node: ast.SetComp) -> None:
            return

        def visit_DictComp(self, node: ast.DictComp) -> None:
            return

        def visit_BoolOp(self, node: ast.BoolOp) -> None:
            return

        def visit_IfExp(self, node: ast.IfExp) -> None:
            return

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

    calls: list[ast.Call] = []
    for statement in tree.body:
        if isinstance(statement, ast.Raise):
            break
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not isinstance(
            statement,
            (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr),
        ):
            continue
        visitor = CallVisitor()
        visitor.visit(statement)
        calls.extend(visitor.calls)
        if errored:
            break
    return calls


def _call_action(call: ast.Call) -> str | None:
    summary = _grouped_aggregation_summary(call)
    if summary:
        return summary

    method_actions = {
        "schema": "Inspected the source schema",
        "metadata": "Inspected source metadata",
        "tables": "Inspected available tables",
        "columns": "Inspected available columns",
        "measures": "Inspected available measures",
        "relationships": "Inspected source relationships",
        "describe": "Profiled the data",
        "profile": "Profiled the data",
        "read_csv": "Loaded CSV data",
        "read_parquet": "Loaded Parquet data",
        "read_table": "Loaded a source table",
        "dax": "Queried the semantic model",
        "sql": "Queried the data source",
        "query": "Queried the data source",
        "merge": "Joined datasets",
        "join": "Joined datasets",
        "plot": "Created a visualization",
        "hist": "Created a distribution chart",
        "sort_values": "Ranked the results",
        "nlargest": "Selected the largest results",
        "nsmallest": "Selected the smallest results",
    }
    if isinstance(call.func, ast.Attribute):
        return method_actions.get(call.func.attr)
    if isinstance(call.func, ast.Name) and call.func.id != "SUBMIT":
        name = _humanize_identifier(call.func.id)
        if name not in {"print", "len", "list", "dict", "set", "tuple"}:
            return f"Ran {name}"
    return None


def _turn_action(code: str, *, errored: bool) -> str:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return "Executed Python code"

    for call in _top_level_calls(tree, errored=errored):
        action = _call_action(call)
        if action:
            return action
    return "Executed Python code"


def _turn_summary(turn: "TurnRecord", *, recovered: bool) -> str:
    action = _turn_action(turn.code, errored=bool(turn.error))
    if recovered:
        action = f"Recovered from the previous error; {action[:1].lower()}{action[1:]}"
    if turn.error:
        error_lines = turn.error.strip().splitlines()
        error_kind = (
            error_lines[-1].split(":", 1)[0].strip()
            if error_lines
            else "Unknown error"
        ) or "Unknown error"
        action += f" - error: {error_kind}"
    if turn.submitted:
        if action == "Executed Python code" and turn.code.strip().startswith("SUBMIT("):
            action = "Submitted the answer"
        else:
            action += "; submitted the answer"
    return action


@dataclass(frozen=True)
class RunInspector:
    """Notebook-renderable, standalone-exportable view of an RLM run."""

    result: "RLMResult"
    max_chars: int = 20_000
    slow_turn_seconds: float = 10.0
    expanded: bool = True
    visible_turns: int = 15

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
        if (
            isinstance(self.visible_turns, bool)
            or not isinstance(self.visible_turns, int)
            or self.visible_turns <= 0
        ):
            raise ValueError("visible_turns must be greater than zero.")

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
        summary = _turn_summary(turn, recovered=recovered)
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
        return (
            '<details class="frlm-turn"><summary>'
            f'<span class="frlm-turn-title">Turn {turn.turn}</span>'
            f'<span class="frlm-turn-summary">{escape(summary)}</span>'
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
                    recovered=(
                        index > 0
                        and bool(self.result.turns[index - 1].error)
                        and not bool(turn.error)
                    ),
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
            f'<div class="frlm-turns" role="region" aria-label="Run turns" '
            f'tabindex="0" style="--frlm-visible-turns: {self.visible_turns}">'
            f"{turns}</div></div></details>"
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
