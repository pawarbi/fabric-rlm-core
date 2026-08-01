"""Windows-safe grading for AgenticDataBench pilot runs.

AgenticDataBench's evaluate.py substitutes file paths into eval_func strings
with re.sub, using the raw path as the replacement template. On Windows the
backslashes are parsed as escapes ("bad escape \\s"), so the stock evaluator
cannot run. This grader executes the same eval_func strings against the same
metric functions (da_agent.evaluators.metrics), substituting absolute paths
as Python literals instead of regex templates. Scores are therefore computed
by the benchmark's own comparators, unmodified.

Usage:
    python grade_pilot.py --testbed <testbed> --outdir <pilot outdir> \
        --tasks strategy_2,strategy_3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tasks", required=True)
    args = ap.parse_args()

    testbed = Path(args.testbed).resolve()
    outdir = Path(args.outdir).resolve()
    sys.path.insert(0, str(testbed))
    from da_agent.evaluators import metrics  # noqa: E402

    with open(testbed / "tasks" / "dev.jsonl", encoding="utf-8") as fh:
        all_tasks = {json.loads(line)["id"]: json.loads(line) for line in fh}

    results = []
    for tid in args.tasks.split(","):
        task = all_tasks[tid.strip()]
        task_dir = outdir / task["id"]
        gold_dir = testbed / "gold" / task["id"]
        scores = []
        # These three fields are lists on all but a handful of tasks
        # (entertainment_02 stores eval_func as a bare string). Iterating a
        # string yields characters, so normalize exactly as the stock
        # evaluator does before using them.
        def as_list(value):
            return value if isinstance(value, list) else [value]

        eval_funcs = as_list(task["eval_func"])
        gold_names = as_list(task["gold_file_name"])
        out_names = as_list(task["output_file_name"])

        for func in eval_funcs:
            exe = func
            # A function replacement sidesteps re.sub's template escaping,
            # which is what breaks the stock evaluator on Windows paths.
            for name in gold_names:
                path_literal = repr(str(gold_dir / Path(name).name))
                exe = re.sub(rf"(['\"]){re.escape(name)}\1",
                             lambda m, p=path_literal: p, exe)
            for name in out_names:
                path_literal = repr(str(task_dir / Path(name).name))
                exe = re.sub(rf"(['\"]){re.escape(name)}\1",
                             lambda m, p=path_literal: p, exe)
            try:
                out = eval(exe, {fn: getattr(metrics, fn) for fn in dir(metrics)})
                score = out.get("score", out) if isinstance(out, dict) else out
                errors = out.get("errors", []) if isinstance(out, dict) else []
            except Exception as exc:
                score, errors = 0.0, [f"{type(exc).__name__}: {exc}"]
            scores.append(score)
            if errors:
                print(f"[{task['id']}] errors: {errors[:3]}")
        total = sum(scores) / len(scores) if scores else 0.0
        print(f"[{task['id']}] score = {total:.3f}")
        results.append({"id": task["id"], "score": total})

    summary = {
        "num_results": len(results),
        "average_score": sum(r["score"] for r in results) / len(results)
        if results else 0.0,
        "results": results,
    }
    (outdir / "grades.json").write_text(json.dumps(summary, indent=2),
                                        encoding="utf-8")
    print(f"\naverage = {summary['average_score']:.3f}  "
          f"({outdir / 'grades.json'})")


if __name__ == "__main__":
    main()
