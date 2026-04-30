"""Command-line entry point for fabric-rlm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .runtime import RLM


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fabric-rlm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an inline task from a JSON file")
    run_parser.add_argument("task_file", help="JSON file with task, inputs, outputs, lm, and optional sub_lm")
    run_parser.add_argument("--output", help="Write result JSON to this path")
    run_parser.add_argument("--trajectory", help="Write trajectory JSONL to this path")
    run_parser.add_argument("--max-turns", type=int, default=None)
    run_parser.add_argument("--timeout", type=float, default=None)

    args = parser.parse_args(argv)
    if args.command == "run":
        result = _run_task(args)
        print(json.dumps(result.payload if result.submitted else result.to_dict(), indent=2))
        return 0 if result.submitted else 2
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


if __name__ == "__main__":
    raise SystemExit(main())

