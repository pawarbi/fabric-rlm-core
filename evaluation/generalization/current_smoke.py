"""Run the same bounded regression smoke through three real local sources."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import sys

from .fixtures import SEED
from .runner import (
    DEFAULT_MODEL, _redact_trace_text, _write_json, account_usage, run_live,
)


def _frozen_package_hashes(root: Path | None) -> dict[str, str]:
    if root is None:
        return {}
    paths = [
        root / f"{source}.artifacts" / "packages" / f"{domain}__descriptive__{arm}.json"
        for source in ("csv", "parquet", "lakehouse")
        for domain in ("inventory", "manufacturing", "service")
        for arm in ("B", "C")
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    parser.add_argument("--knowledge-executions", nargs="+", choices=("auto", "context_only"), default=["auto"])
    parser.add_argument("--frozen-runs", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1 or not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 1:
        parser.error("positive repetitions and a budget greater than the $1 reserve are required")
    if len(set(args.knowledge_executions)) != len(args.knowledge_executions):
        parser.error("knowledge execution policies must be unique")
    if len(args.knowledge_executions) > 1 and args.frozen_runs is None:
        parser.error("policy comparisons require --frozen-runs so packages are identical")
    frozen_hashes = _frozen_package_hashes(args.frozen_runs)
    args.output.mkdir(parents=True, exist_ok=False)
    start_usage = account_usage()
    order = ["csv", "parquet", "lakehouse"]
    rng = random.Random(args.seed)
    rng.shuffle(order)
    policies = list(args.knowledge_executions)
    if len(policies) > 1:
        rng.shuffle(policies)
    plan = [
        {"representation": source, "knowledge_execution": policy}
        for index, source in enumerate(order)
        for policy in (policies if index % 2 == 0 else list(reversed(policies)))
    ]
    manifest = {
        "status": "running", "model": args.model, "seed": args.seed,
        "repetitions": args.repetitions, "source_order": order,
        "max_cost_usd": args.max_cost_usd, "account_usage_start": start_usage,
        "python": sys.version,
        "versions": {name: version(name) for name in ("dspy", "duckdb", "pandas", "numpy", "pyarrow", "deltalake")},
        "scope": "Previously used synthetic regression questions; not a new blinded holdout.",
        "run_order": plan, "frozen_runs": str(args.frozen_runs) if args.frozen_runs else None,
        "frozen_package_hashes": frozen_hashes,
        "development_scope": "none: reusing frozen packages" if args.frozen_runs else "fresh development",
        "runs": [],
    }
    _write_json(args.output / "batch.json", manifest)
    for entry in plan:
        representation, policy = entry["representation"], entry["knowledge_execution"]
        remaining = args.max_cost_usd - (account_usage() - start_usage)
        if remaining <= 1:
            manifest["stop_reason"] = "budget_reserve"
            break
        name = representation if policies == ["auto"] else f"{representation}__{policy}"
        output = args.output / f"{name}.json"
        print(f"START {representation} {policy}", flush=True)
        package_dir = args.frozen_runs / f"{representation}.artifacts" / "packages" if args.frozen_runs else None
        try:
            try:
                package_drift = _frozen_package_hashes(args.frozen_runs) != frozen_hashes
            except OSError as exc:
                manifest["error"] = _redact_trace_text(f"{type(exc).__name__}: {exc}")
                print(manifest["error"], file=sys.stderr, flush=True)
                package_drift = True
            if package_drift:
                manifest["stop_reason"] = "frozen_package_drift"
                break
            result = run_live(
                fixtures=args.fixtures, output=output, model=args.model,
                repetitions=args.repetitions, seed=args.seed, variants=("descriptive",),
                max_live_calls=(0 if package_dir else 6) + 9 * args.repetitions,
                max_turns=6, timeout=120, smoke=True, max_cost_usd=remaining,
                representation=representation, knowledge_execution=policy,
                frozen_packages=package_dir,
            )
        except Exception as exc:
            manifest["stop_reason"] = "execution_error"
            manifest["error"] = _redact_trace_text(f"{type(exc).__name__}: {exc}")
            _write_json(args.output / "batch.json", manifest)
            print(manifest["error"], file=sys.stderr, flush=True)
            break
        manifest["runs"].append({
            "representation": representation, "output": str(output),
            "knowledge_execution": policy,
            "status": result["status"], "trials": len(result.get("trials", [])),
            "gate_passed": result.get("gate", {}).get("passed", False),
        })
        _write_json(args.output / "batch.json", manifest)
        print(json.dumps(manifest["runs"][-1]), flush=True)
        try:
            package_drift = _frozen_package_hashes(args.frozen_runs) != frozen_hashes
        except OSError as exc:
            manifest["error"] = _redact_trace_text(f"{type(exc).__name__}: {exc}")
            print(manifest["error"], file=sys.stderr, flush=True)
            package_drift = True
        if package_drift:
            manifest["stop_reason"] = "frozen_package_drift"
            break
        if result["status"] != "complete":
            manifest["stop_reason"] = f"run_{result['status']}"
            break
    manifest["account_usage_end"] = account_usage()
    manifest["status"] = (
        "complete" if len(manifest["runs"]) == len(plan)
        and all(run["status"] == "complete" for run in manifest["runs"])
        and not manifest.get("stop_reason")
        else "incomplete"
    )
    _write_json(args.output / "batch.json", manifest)
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
