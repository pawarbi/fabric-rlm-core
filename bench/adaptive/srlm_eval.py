"""SRLM bench CLI entry point (Phase 1).

Runs the canonical 32-question bench (or a 2-question smoke subset) across
one or more configs × seeds, writes per-rollout JSON, and emits a markdown
summary + JSON summary.

Usage::

    python bench/adaptive/srlm_eval.py --smoke --configs default,adaptive_current --seeds 1
    python bench/adaptive/srlm_eval.py --configs default,adaptive_current --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct ``python bench/adaptive/srlm_eval.py`` invocation by ensuring
# the bench/adaptive directory is on sys.path so ``_eval_lib`` resolves.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _eval_lib import (  # type: ignore  # noqa: E402
    CONFIG_NAMES,
    default_question_set,
    get_config,
    load_all_results,
    render_markdown_report,
    run_bench,
    summarize,
)


def _parse_configs(s: str) -> list[str]:
    out = [c.strip() for c in s.split(",") if c.strip()]
    for c in out:
        if c not in CONFIG_NAMES:
            raise SystemExit(f"unknown config: {c} (valid: {', '.join(CONFIG_NAMES)})")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="srlm_eval")
    p.add_argument("--model", default="openai/gpt-4.1")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (0..N-1)")
    p.add_argument(
        "--configs",
        default="default,adaptive_current",
        help=f"comma-separated config names ({', '.join(CONFIG_NAMES)})",
    )
    p.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "bench" / "adaptive" / "results" / "srlm_eval"),
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="smoke run: only the first 2 questions (typically the calibrated easy ones)",
    )
    p.add_argument(
        "--question-set", default="default",
        help="question set selector (only 'default' is supported in Phase 1)",
    )
    p.add_argument(
        "--summary-json",
        default=str(REPO_ROOT / "bench" / "adaptive" / "srlm_eval_summary.json"),
    )
    p.add_argument(
        "--summary-md",
        default=str(REPO_ROOT / "bench" / "adaptive" / "srlm_eval_report.md"),
    )
    p.add_argument(
        "--max-seconds", type=int, default=300,
        help="per-question wall-clock budget forwarded to RLM(timeout=...)",
    )
    args = p.parse_args(argv)

    if args.question_set != "default":
        raise SystemExit(f"--question-set={args.question_set!r} not supported in Phase 1")

    questions = default_question_set(REPO_ROOT)
    if args.smoke:
        questions = questions[:2]

    config_names = _parse_configs(args.configs)
    configs = [get_config(c) for c in config_names]
    seeds = list(range(args.seeds))
    results_dir = Path(args.results_dir)

    print(
        f"[srlm_eval] model={args.model} configs={config_names} "
        f"seeds={seeds} n_questions={len(questions)} -> {results_dir}",
        file=sys.stderr, flush=True,
    )

    aggregate = run_bench(
        questions, configs, seeds, args.model, results_dir,
        max_seconds=args.max_seconds,
    )
    print(f"[srlm_eval] {aggregate}", file=sys.stderr, flush=True)

    rows = load_all_results(results_dir)
    summary = summarize(rows)
    summary["aggregate"] = aggregate
    summary["model"] = args.model
    summary["configs"] = config_names
    summary["seeds"] = seeds

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    md_path = Path(args.summary_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown_report(summary), encoding="utf-8")

    print(f"[srlm_eval] wrote {summary_path}\n[srlm_eval] wrote {md_path}", file=sys.stderr)
    # human-readable to stdout
    for name, s in sorted(summary.get("per_config", {}).items()):
        print(
            f"  {name}: {s['n_passed']}/{s['n_total']} "
            f"acc={s['accuracy']:.3f} CI=[{s['ci_lo']:.3f},{s['ci_hi']:.3f}] "
            f"tokens={s['mean_total_tokens']:.0f} elapsed={s['mean_elapsed_s']:.2f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
