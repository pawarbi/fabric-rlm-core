"""Run the same bounded regression smoke through three real local sources."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import sys

from .fixtures import SEED
from .runner import (
    DEFAULT_MODEL, _write_json, account_usage, run_live,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    args = parser.parse_args()
    if args.repetitions < 1 or not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 1:
        parser.error("positive repetitions and a budget greater than the $1 reserve are required")
    args.output.mkdir(parents=True, exist_ok=False)
    start_usage = account_usage()
    order = ["csv", "parquet", "lakehouse"]
    random.Random(args.seed).shuffle(order)
    manifest = {
        "status": "running", "model": args.model, "seed": args.seed,
        "repetitions": args.repetitions, "source_order": order,
        "max_cost_usd": args.max_cost_usd, "account_usage_start": start_usage,
        "python": sys.version,
        "versions": {name: version(name) for name in ("dspy", "duckdb", "pandas", "numpy", "pyarrow", "deltalake")},
        "scope": "Previously used synthetic regression questions; not a new blinded holdout.",
        "runs": [],
    }
    _write_json(args.output / "batch.json", manifest)
    for representation in order:
        remaining = args.max_cost_usd - (account_usage() - start_usage)
        if remaining <= 1:
            manifest["stop_reason"] = "budget_reserve"
            break
        output = args.output / f"{representation}.json"
        print(f"START {representation}", flush=True)
        result = run_live(
            fixtures=args.fixtures, output=output, model=args.model,
            repetitions=args.repetitions, seed=args.seed, variants=("descriptive",),
            max_live_calls=6 + 9 * args.repetitions, max_turns=6, timeout=120,
            smoke=True, max_cost_usd=remaining, representation=representation,
        )
        manifest["runs"].append({
            "representation": representation, "output": str(output),
            "status": result["status"], "trials": len(result.get("trials", [])),
            "gate_passed": result.get("gate", {}).get("passed", False),
        })
        _write_json(args.output / "batch.json", manifest)
        print(json.dumps(manifest["runs"][-1]), flush=True)
        if result["status"] != "complete":
            manifest["stop_reason"] = f"run_{result['status']}"
            break
    manifest["account_usage_end"] = account_usage()
    manifest["status"] = (
        "complete" if len(manifest["runs"]) == 3
        and all(run["status"] == "complete" for run in manifest["runs"])
        else "incomplete"
    )
    _write_json(args.output / "batch.json", manifest)
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
