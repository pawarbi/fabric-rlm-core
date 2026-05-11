"""Command-line entry point for fabric-rlm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .runtime import RLM
from .trajectory import Trajectory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fabric-rlm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an inline task from a JSON file")
    run_parser.add_argument("task_file", help="JSON file with task, inputs, outputs, lm, and optional sub_lm")
    run_parser.add_argument("--output", help="Write result JSON to this path")
    run_parser.add_argument("--trajectory", help="Write trajectory JSONL to this path")
    run_parser.add_argument("--max-turns", type=int, default=None)
    run_parser.add_argument("--timeout", type=float, default=None)

    trace_parser = subparsers.add_parser("trace", help="Inspect a trajectory JSONL file")
    trace_sub = trace_parser.add_subparsers(dest="trace_command", required=True)
    inspect_p = trace_sub.add_parser("inspect", help="Print summary + diagnostics for a trajectory")
    inspect_p.add_argument("path", help="Path to trajectory JSONL (use '-' for stdin)")
    inspect_p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of pretty text",
    )
    inspect_p.add_argument(
        "--no-diagnose",
        action="store_true",
        help="Skip diagnostic checks (summary only)",
    )
    inspect_p.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit with non-zero status when diagnose() finds any issue (CI-friendly)",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        result = _run_task(args)
        print(json.dumps(result.payload if result.submitted else result.to_dict(), indent=2))
        return 0 if result.submitted else 2
    if args.command == "trace" and args.trace_command == "inspect":
        return _trace_inspect(args)
    return 1


def _run_task(args: argparse.Namespace) -> Any:
    config = json.loads(Path(args.task_file).read_text(encoding="utf-8"))
    kwargs = {
        "lm": config["lm"],
        "sub_lm": config.get("sub_lm"),
        "max_turns": args.max_turns or config.get("max_turns", 10),
        "timeout": args.timeout or config.get("timeout", 300.0),
        "skills": config.get("skills"),
        "enable_skill_autoloading": bool(config.get("enable_skill_autoloading", False)),
    }
    rlm = RLM.from_task(
        task=config["task"],
        inputs=config.get("inputs", {}),
        outputs=config.get("outputs", []),
        **kwargs,
    )
    result = rlm.run()
    if args.output:
        payload = {
            "submitted": result.submitted,
            "payload": result.payload,
            "failure_reason": result.failure_reason,
            "final_state": result.final_state,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.trajectory:
        result.trajectory.write_jsonl(args.trajectory)
    return result


def _trace_inspect(args: argparse.Namespace) -> int:
    if args.path == "-":
        import sys

        traj = Trajectory.from_jsonl(sys.stdin)
    else:
        traj = Trajectory.from_jsonl(args.path)
    summary = traj.summary()
    issues = [] if args.no_diagnose else traj.diagnose()
    if args.json:
        print(json.dumps({"summary": summary, "issues": [i.to_dict() for i in issues]}, indent=2))
    else:
        _print_human(args.path, summary, issues)
    if getattr(args, "fail_on_issues", False) and issues:
        return 1
    return 0


def _print_human(path: str, summary: dict[str, Any], issues: list[Any]) -> None:
    def _fmt(v: Any) -> str:
        return f"{v:,}" if isinstance(v, int) else "n/a"

    print(f"Trajectory: {path}")
    print(f"  turns         : {summary['turns']}")
    print(f"  submitted     : {summary['submitted']}" + (f" (turn {summary['submit_turn']})" if summary["submit_turn"] else ""))
    print(f"  errors        : {summary['errors']}" + (f" {summary['error_kinds']}" if summary["error_kinds"] else ""))
    print(
        f"  tokens        : prompt={_fmt(summary['prompt_tokens'])} "
        f"completion={_fmt(summary['completion_tokens'])} "
        f"total={_fmt(summary['total_tokens'])}"
    )
    if summary["cached_tokens"] or summary["reasoning_tokens"]:
        print(
            f"                  cached={_fmt(summary['cached_tokens'])} "
            f"reasoning={_fmt(summary['reasoning_tokens'])}"
        )
    print(f"  lm_seconds    : {summary['lm_seconds'] if summary['lm_seconds'] is not None else 'n/a'}")
    if summary["metadata"]:
        meta = summary["metadata"]
        skills = meta.get("skills") or []
        print(f"  skills        : {skills}")
    print()
    if not issues:
        print("Diagnostics: clean (0 issues)")
        return
    print(f"Diagnostics: {len(issues)} issue(s)")
    for i in issues:
        print(f"  [turn {i.turn}] {i.kind}: {i.message}")


if __name__ == "__main__":
    raise SystemExit(main())

