"""Local Olist deep-insight transfer benchmark.

The harness accepts only caller-provided canonical Olist CSV files, delegates
analysis to Fabric RLM, and independently re-executes every numeric evidence
check on the host before persisting aggregate artifacts.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any


CANONICAL_FILES = (
    "customers.csv",
    "geolocation.csv",
    "order_items.csv",
    "order_payments.csv",
    "order_reviews.csv",
    "orders.csv",
    "product_category_name_translation.csv",
    "products.csv",
    "sellers.csv",
)
DEFAULT_SKILLS = ("data_exploration", "deep_insight_discovery")
DEFAULT_MODEL = "openrouter/z-ai/glm-5.3-flash"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"


def discover_sources(data_dir: str | Path) -> dict[str, Path]:
    """Return the exact canonical bundle as a stable identity-to-path map."""

    root = Path(data_dir)
    missing = [name for name in CANONICAL_FILES if not (root / name).is_file()]
    if missing:
        formatted = ", ".join(missing)
        raise FileNotFoundError(
            f"canonical Olist bundle is incomplete under {root}; "
            f"missing files: {formatted}"
        )
    return {Path(name).stem: root / name for name in CANONICAL_FILES}


def build_task_prompt(sources: Mapping[str, Path]) -> str:
    """Build the analysis brief with explicit, caller-injected source paths."""

    source_lines = "\n".join(
        f"- {identity}: {path}" for identity, path in sources.items()
    )
    return f"""\
Perform a non-obvious, cross-domain business analysis of the canonical public
Olist data bundle below. Treat each listed identity as its authoritative source:
{source_lines}

OUTPUT CONTRACT
- Activate and follow the deep_insight_discovery deep-insight skill contract.
- SUBMIT one concise JSON object with contract_version: 2. The deep-insight
  skill owns the exact schema, so do not invent a parallel schema.
- Aim for 3-5 decision-grade findings, but quality wins over count.
- Quantitative rejected candidates must carry executable numeric evidence.
- Keep narrative and evidence concise.

DATA MODEL AND GRAIN SAFETY
- First measure a join map and join coverage (matched and unmatched populations)
  before attempting cross-domain analysis.
- When measured join coverage supports them, include at least two genuine
  cross-domain findings. Do not force them when coverage is inadequate.
- Aggregate order_items, order_payments, and order_reviews to order grain before joining to orders.
- Pre-aggregate geolocation by ZIP prefix before any join.
- State the population, unit/grain, numerator, denominator, time basis, and
  exclusions for every metric. Prevent fan-out and count-vs-rate errors.

ANALYTIC APPLICABILITY GATES
- Record an applicability decision for each of: decomposition,
  instrumentation diagnostics, change points, cohorts, interactions, driver
  analysis, concentration, clustering, classification, and regression.
- Use a method only when its target, sample size, leakage controls, holdout
  design, and decision relevance support it; otherwise explain why it is not
  applicable. Do not create unsupported targets.

QUALITY BAR
- Reject obvious raw-count restatements. Reconcile metric definitions and run
  sensitivity checks when alternative populations or definitions exist.
- Concentration and mix claims must be benchmarked against an explicit comparator.
- Require exact numeric consistency between every headline and comparison and
  their authoritative evidence.
- Avoid causal overclaiming from observational analysis.
- Do not recommend action while measurable alternative explanations remain
  unresolved; rule them out, weaken them with evidence, or label them unresolved.
- Prefer findings that change a business decision over generic descriptive facts.

All authoritative numeric evidence must use SQL verification compatible with
the host DuckDBAuditExecutor. Each verification sources mapping must map a
simple query alias to one source identity listed above, for example
{{"o": "orders"}}. Every expression must be a self-contained SELECT or WITH
query over those aliases; do not reference worker-created tables or views.
The worker has no network access.
"""


def normalize_payload(value: Any) -> dict[str, Any]:
    """Normalize a submitted object or JSON object string, rejecting ambiguity."""

    parsed = value
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("RLM payload is empty")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"RLM payload is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("RLM payload must be a JSON object")
    return parsed


def extract_payload(result: Any) -> dict[str, Any]:
    """Extract the native ``result.payload`` submission."""

    if not hasattr(result, "payload"):
        raise ValueError("RLM result has no payload")
    return normalize_payload(result.payload)


def audit_to_dict(report: Any) -> dict[str, Any]:
    """Serialize an immutable audit report without depending on its dataclasses."""

    checks = [
        {
            "actual": check.actual,
            "expected": check.expected,
            "path": check.path,
        }
        for check in report.checks
    ]
    return {
        "status": "passed",
        "total_checks": report.total_checks,
        "checks": checks,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_artifacts(output_dir: str | Path, record: Mapping[str, Any]) -> dict[str, Path]:
    """Persist deterministic aggregate-only JSON artifacts."""

    output = Path(output_dir)
    payload = normalize_payload(record["payload"])
    audit_data = audit_to_dict(record["audit"])
    trajectory = dict(record["trajectory"])
    insights = payload.get("insights")
    candidates = payload.get("candidates")
    counts = {
        "audit_checks": audit_data["total_checks"],
        "candidates": len(candidates) if isinstance(candidates, list) else 0,
        "insights": len(insights) if isinstance(insights, list) else 0,
        "trajectory_turns": trajectory.get("turns", 0),
    }
    run_data = {
        "status": "success",
        "model": record["model"],
        "skills": list(record["skills"]),
        "counts": counts,
        "trajectory": trajectory,
    }
    paths = {
        "payload": output / "payload.json",
        "audit": output / "audit.json",
        "run": output / "run.json",
    }
    _atomic_json(paths["payload"], payload)
    _atomic_json(paths["audit"], audit_data)
    _atomic_json(paths["run"], run_data)
    return paths


def _load_runtime_dependencies():
    import dspy
    from fabric_rlm import RLM
    from fabric_rlm._deep_insight_audit import audit_deep_insight
    from fabric_rlm._duckdb_audit import DuckDBAuditExecutor
    from fabric_rlm.metrics import summarize_trajectory

    return (
        dspy,
        RLM,
        DuckDBAuditExecutor,
        audit_deep_insight,
        summarize_trajectory,
    )


def run_benchmark(
    data_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    max_turns: int = 24,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Run RLM and host-audit all submitted numeric evidence."""

    sources = discover_sources(data_dir)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required to run the Olist benchmark"
        )

    (
        dspy,
        RLM,
        DuckDBAuditExecutor,
        audit_deep_insight,
        summarize_trajectory,
    ) = _load_runtime_dependencies()
    lm = dspy.LM(
        model,
        api_key=api_key,
        api_base=api_base,
        max_tokens=20000,
        temperature=0,
        reasoning={"max_tokens": 4096, "exclude": True},
    )
    dspy.configure(lm=lm)
    rlm = RLM.from_task(
        task=build_task_prompt(sources),
        lm=lm,
        skills=list(DEFAULT_SKILLS),
        enable_verifier=True,
        block_network=True,
        engine="default",
        max_turns=max_turns,
        reserve_finalize_turns=6,
        timeout=timeout,
        verbose=True,
    )
    result = rlm.run()
    payload = extract_payload(result)
    with DuckDBAuditExecutor(sources) as executor:
        audit = audit_deep_insight(payload, executor)
    return {
        "payload": payload,
        "audit": audit,
        "trajectory": summarize_trajectory(result.trajectory),
        "model": model,
        "skills": DEFAULT_SKILLS,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Olist deep-insight transfer benchmark."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--max-turns", default=24, type=int)
    parser.add_argument("--timeout", default=1800.0, type=float)
    parser.add_argument(
        "--output-dir",
        default=Path("_local") / "olist_deep_insight_benchmark",
        type=Path,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = run_benchmark(
        args.data_dir,
        model=args.model,
        api_base=args.api_base,
        max_turns=args.max_turns,
        timeout=args.timeout,
    )
    paths = write_artifacts(args.output_dir, record)
    print(
        json.dumps(
            {
                "status": "success",
                "audit_checks": record["audit"].total_checks,
                "artifacts": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
