"""Manifest-driven staged deep-insight transfer benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from fabric_rlm._benchmark_manifest import (
    build_source_agnostic_research_prompt,
    load_source_manifest,
)


def _load_staged_example():
    path = Path(__file__).with_name("olist_staged_deep_insight_benchmark.py")
    spec = importlib.util.spec_from_file_location(
        "_staged_deep_insight_benchmark",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load staged benchmark helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STAGED = _load_staged_example()
DEFAULT_MODEL = _STAGED.DEFAULT_MODEL
DEFAULT_API_BASE = _STAGED.DEFAULT_API_BASE


def build_manifest_research_prompt(manifest_path: str | Path) -> str:
    """Build the research task from a fully verified source manifest."""

    manifest = load_source_manifest(manifest_path)
    return build_source_agnostic_research_prompt(manifest)


def run_manifest_benchmark(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    research_turns: int = 18,
    scaffold_turns: int = 10,
    insight_turns: int = 14,
    repair_turns: int = 6,
    max_insight_repairs: int = 4,
    max_scaffold_repairs: int = 3,
    max_audit_repairs: int = 6,
    closure_turns: int = 10,
    timeout: float = 3600,
) -> dict[str, Any]:
    """Run the frozen staged pipeline against a verified manifest bundle."""

    manifest = load_source_manifest(manifest_path)
    output = Path(output_dir)
    return _STAGED.run_staged_benchmark(
        manifest.path.parent,
        manifest_path=manifest.path,
        enable_evidence_closure=True,
        model=model,
        api_base=api_base,
        research_turns=research_turns,
        scaffold_turns=scaffold_turns,
        insight_turns=insight_turns,
        repair_turns=repair_turns,
        max_insight_repairs=max_insight_repairs,
        max_scaffold_repairs=max_scaffold_repairs,
        max_audit_repairs=max_audit_repairs,
        closure_turns=closure_turns,
        timeout=timeout,
        research_cache_path=output / "research.json",
        scaffold_cache_path=output / "contract_scaffold.checkpoint.json",
        insights_cache_path=output / "insights.checkpoint.json",
        closure_cache_path=output / "evidence_closure.checkpoint.json",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manifest-driven staged deep-insight benchmark."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--research-turns", default=18, type=int)
    parser.add_argument("--scaffold-turns", default=10, type=int)
    parser.add_argument("--insight-turns", default=14, type=int)
    parser.add_argument("--repair-turns", default=6, type=int)
    parser.add_argument("--max-insight-repairs", default=4, type=int)
    parser.add_argument("--max-scaffold-repairs", default=3, type=int)
    parser.add_argument("--max-audit-repairs", default=6, type=int)
    parser.add_argument("--closure-turns", default=10, type=int)
    parser.add_argument("--timeout", default=3600, type=float)
    parser.add_argument(
        "--output-dir",
        default=Path("_local") / "manifest_staged_deep_insight_benchmark",
        type=Path,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = run_manifest_benchmark(
        args.manifest,
        output_dir=args.output_dir,
        model=args.model,
        api_base=args.api_base,
        research_turns=args.research_turns,
        scaffold_turns=args.scaffold_turns,
        insight_turns=args.insight_turns,
        repair_turns=args.repair_turns,
        max_insight_repairs=args.max_insight_repairs,
        max_scaffold_repairs=args.max_scaffold_repairs,
        max_audit_repairs=args.max_audit_repairs,
        closure_turns=args.closure_turns,
        timeout=args.timeout,
    )
    paths = _STAGED.write_staged_artifacts(args.output_dir, record)
    print(
        json.dumps(
            {
                "status": "success",
                "audit_checks": record["audit"].total_checks,
                "artifacts": {
                    name: str(path) for name, path in paths.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
