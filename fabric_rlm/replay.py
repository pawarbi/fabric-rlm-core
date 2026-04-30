"""Inspect / replay a fabric-rlm trajectory JSONL file.

Usage examples
--------------

    # Pretty-print every turn in a trajectory
    python -m fabric_rlm.replay path/to/trajectory.jsonl

    # Only show the first 3 turns
    python -m fabric_rlm.replay path/to/trajectory.jsonl --turns 3

    # Dump the parsed trajectory as a single JSON document (pipe to jq)
    python -m fabric_rlm.replay path/to/trajectory.jsonl --json | jq '.turns | length'

    # Disable ANSI color (e.g. when redirecting to a file)
    python -m fabric_rlm.replay path/to/trajectory.jsonl --no-color > out.txt

    # Disable per-field truncation
    python -m fabric_rlm.replay path/to/trajectory.jsonl --full

The renderer is intentionally tolerant of schema drift: canonical TurnRecord
fields (``code``, ``stdout``, ``stderr``, ``error``, ``duration_s``,
``submitted``, ``validation_errors``, ``turn_type``) are looked up at the top
level of each record, and if missing, under a ``metadata`` sub-object.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Constants

_TRUNC_HEAD = 2000
_TRUNC_TAIL = 500
_TRUNC_THRESHOLD = 4000

_ANSI = {
    "reset": "\x1b[0m",
    "dim": "\x1b[2m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
    "bold": "\x1b[1m",
}

_CANONICAL_FIELDS = (
    "turn",
    "turn_index",
    "turn_type",
    "code",
    "stdout",
    "stderr",
    "error",
    "submitted",
    "duration_s",
    "validation_errors",
)


# ---------------------------------------------------------------------------
# Parsing


def load_trajectory(path: str | Path) -> dict[str, Any]:
    """Load a JSONL trajectory file, tolerating BOM and blank leading lines.

    Returns a dict with keys ``metadata`` (dict, possibly empty) and ``turns``
    (list of normalized turn dicts).
    """
    raw = Path(path).read_text(encoding="utf-8-sig")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    metadata: dict[str, Any] = {}
    turns: list[dict[str, Any]] = []

    for idx, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {idx + 1}: invalid JSON ({exc})") from exc
        if not isinstance(obj, dict):
            continue

        # Heuristic: a metadata-only record (the leading line written by
        # Trajectory.to_jsonl) has no canonical turn fields at the top level,
        # *and* the nested "metadata" dict itself has no canonical turn fields
        # — otherwise it's a per-turn record using the nested-schema variant.
        nested_md = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else None
        top_has_turn_field = any(k in obj for k in ("turn", "turn_index", "code", "stdout"))
        nested_has_turn_field = bool(
            nested_md and any(k in nested_md for k in ("turn", "turn_index", "code", "stdout"))
        )
        is_meta_only = nested_md is not None and not top_has_turn_field and not nested_has_turn_field
        if is_meta_only and idx == 0:
            metadata = nested_md or {}
            continue

        turns.append(_normalize_turn(obj, fallback_index=len(turns)))

    return {"metadata": metadata, "turns": turns}


def _normalize_turn(obj: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    """Return a turn dict with canonical fields lifted to the top level.

    Prefers top-level values; falls back to values nested under ``metadata``.
    Unknown extra fields are preserved at the top level.
    """
    nested = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}

    def pick(key: str, default: Any) -> Any:
        if key in obj and obj[key] is not None:
            return obj[key]
        if key in nested and nested[key] is not None:
            return nested[key]
        return default

    turn_index = pick("turn_index", None)
    if turn_index is None:
        turn_index = pick("turn", fallback_index)

    return {
        "turn_index": turn_index,
        "turn_type": pick("turn_type", "normal"),
        "code": pick("code", ""),
        "stdout": pick("stdout", ""),
        "stderr": pick("stderr", ""),
        "error": pick("error", None),
        "submitted": bool(pick("submitted", False)),
        "duration_s": pick("duration_s", None),
        "validation_errors": pick("validation_errors", []),
        "_raw": obj,
    }


# ---------------------------------------------------------------------------
# Rendering


def _color(text: str, code: str, enabled: bool) -> str:
    if not enabled or not code:
        return text
    return f"{_ANSI[code]}{text}{_ANSI['reset']}"


def _truncate(text: str, full: bool) -> str:
    if full or len(text) <= _TRUNC_THRESHOLD:
        return text
    omitted = len(text) - _TRUNC_HEAD - _TRUNC_TAIL
    return (
        text[:_TRUNC_HEAD]
        + f"\n... (truncated {omitted} chars) ...\n"
        + text[-_TRUNC_TAIL:]
    )


def _indent(text: str, prefix: str = "    ") -> str:
    if text == "":
        return prefix + "(empty)"
    return "\n".join(prefix + line for line in text.splitlines())


def _format_field_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return "\n".join(str(v) for v in value)
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def render_turn(turn: dict[str, Any], *, color: bool, full: bool) -> str:
    """Render a single normalized turn dict as a multi-line string."""
    duration = turn.get("duration_s")
    duration_str = f"{duration:.2f}s" if isinstance(duration, (int, float)) else "n/a"
    header = (
        f"=== Turn {turn['turn_index']} [{turn['turn_type']}] "
        f"dur={duration_str} submitted={turn['submitted']} ==="
    )
    parts = [_color(header, "yellow", color)]

    field_styles = {
        "code": "cyan",
        "stdout": "green",
        "stderr": "red",
        "error": "red",
        "validation_errors": "red",
    }

    for field in ("code", "stdout", "stderr", "error", "validation_errors"):
        raw_value = turn.get(field)
        formatted = _format_field_value(raw_value)
        body = _indent(_truncate(formatted, full=full))
        label = _color(f"-- {field} --", field_styles[field], color)
        parts.append(label)
        parts.append(body)

    parts.append(_color("-" * 60, "dim", color))
    return "\n".join(parts)


def render_trajectory(
    trajectory: dict[str, Any],
    *,
    color: bool,
    full: bool,
    limit: int | None,
) -> str:
    turns: Sequence[dict[str, Any]] = trajectory.get("turns", [])
    if limit is not None:
        turns = turns[:limit]

    out: list[str] = []
    metadata = trajectory.get("metadata") or {}
    if metadata:
        meta_header = _color("### trajectory metadata ###", "bold", color)
        out.append(meta_header)
        out.append(_indent(json.dumps(metadata, indent=2, default=str)))
        out.append(_color("-" * 60, "dim", color))

    out.append(
        _color(
            f"Loaded {len(trajectory.get('turns', []))} turn(s); rendering {len(turns)}.",
            "dim",
            color,
        )
    )
    for turn in turns:
        out.append(render_turn(turn, color=color, full=full))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI


def _strip_raw(turns: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in t.items() if k != "_raw"} for t in turns]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fabric_rlm.replay",
        description="Pretty-print a fabric-rlm trajectory JSONL file.",
    )
    parser.add_argument("trajectory_path", help="Path to a .jsonl trajectory file.")
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        metavar="N",
        help="Render only the first N turns.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the parsed trajectory as a single JSON document.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Disable per-field truncation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    path = Path(args.trajectory_path)
    if not path.exists():
        print(f"error: trajectory file not found: {path}", file=sys.stderr)
        return 2

    try:
        trajectory = load_trajectory(path)
    except ValueError as exc:
        print(f"error: failed to parse {path}: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        payload = {
            "metadata": trajectory.get("metadata", {}),
            "turns": _strip_raw(trajectory.get("turns", [])),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0

    color_enabled = (not args.no_color) and sys.stdout.isatty()
    print(
        render_trajectory(
            trajectory,
            color=color_enabled,
            full=args.full,
            limit=args.turns,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
