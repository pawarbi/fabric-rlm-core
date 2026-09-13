"""Calibrate an evaluation's noise floor, then re-judge its conclusions.

Run against a recorded trial file:

    python -m evaluation.calibration \
        evaluation/dbo_lakehouse/dbo_eval_results.json --baseline-arm A

The report has two halves. The first measures how much an *unchanged*
configuration varies between identical repetitions. The second re-runs the
arm-versus-arm comparison with that measurement as the floor, and states which
conclusions survive it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .noise import Metric, Verdict, compare, flip_rate, spread, spread_kind
from .trials import CORRECTNESS_SCORERS, TrialSet, load_trials

ACCURACY = Metric(
    key="accuracy",
    label="Accuracy",
    unit="pp",
    absolute_floor=2.0,
    relative_floor=0.02,
    lower_is_better=False,
)
ABSTENTION = Metric(
    key="abstention",
    label="Abstention rate",
    unit="pp",
    absolute_floor=2.0,
    relative_floor=0.02,
    lower_is_better=True,
)
TURNS = Metric(
    key="turns",
    label="Turns per question",
    unit="turns",
    absolute_floor=0.5,
    relative_floor=0.10,
)
SECONDS = Metric(
    key="seconds",
    label="Latency per question",
    unit="s",
    absolute_floor=1.0,
    relative_floor=0.10,
)
TOKENS = Metric(
    key="tokens",
    label="Tokens per question",
    unit="tok",
    absolute_floor=100.0,
    relative_floor=0.10,
)

SERIES = [
    (ACCURACY, lambda ts, arm: ts.accuracy_by_rep(arm)),
    (ABSTENTION, lambda ts, arm: ts.abstention_rate_by_rep(arm)),
    (TURNS, lambda ts, arm: ts.metric_by_rep(arm, "turns")),
    (SECONDS, lambda ts, arm: ts.metric_by_rep(arm, "seconds")),
    (TOKENS, lambda ts, arm: ts.metric_by_rep(arm, "total_tokens")),
]


def calibrate(trials: TrialSet, baseline_arm: str) -> dict:
    """Measure the baseline arm against itself, repetition by repetition."""
    floors, detail = {}, {}
    for metric, series in SERIES:
        values = series(trials, baseline_arm)
        floors[metric.key] = spread(values)
        detail[metric.key] = {
            "label": metric.label,
            "unit": metric.unit,
            "per_rep": values,
            "spread": spread(values),
            "spread_kind": spread_kind(len(values)),
        }
    outcomes = trials.outcomes_by_question(baseline_arm)
    unstable = sorted(
        qid for qid, results in outcomes.items() if len(set(results)) > 1
    )
    return {
        "arm": baseline_arm,
        "floors": floors,
        "detail": detail,
        "flip_rate": flip_rate(outcomes),
        "unstable_questions": unstable,
        "question_count": len(outcomes),
    }


def judge(
    trials: TrialSet, baseline_arm: str, candidate_arm: str, floors: dict
) -> list[dict]:
    """Compare two arms under three progressively honest noise models.

    ``point`` is what a report that quotes only its headline numbers concludes:
    floors on the size of the change, but no notion that a rerun might differ.
    ``naive`` uses the spread of the very runs being compared, which is better
    but conflates the difference with the noise. ``calibrated`` uses the floor
    measured by running the baseline against itself.
    """
    rows = []
    expected = len(trials.reps)
    for metric, series in SERIES:
        baseline = series(trials, baseline_arm)
        candidate = series(trials, candidate_arm)
        point = compare(
            metric,
            baseline,
            candidate,
            expected_samples=expected,
            calibrated_floor=0.0,
        )
        naive = compare(
            metric, baseline, candidate, expected_samples=expected
        )
        calibrated = compare(
            metric,
            baseline,
            candidate,
            expected_samples=expected,
            calibrated_floor=floors[metric.key],
        )
        rows.append(
            {
                "metric": metric.key,
                "label": metric.label,
                "unit": metric.unit,
                "baseline": naive.baseline,
                "candidate": naive.candidate,
                "delta": naive.delta,
                "relative_change": naive.relative_change,
                "point_verdict": point.verdict.value,
                "naive_verdict": naive.verdict.value,
                "calibrated_verdict": calibrated.verdict.value,
                "calibrated_floor": floors[metric.key],
                "overturned": (
                    point.is_significant and not calibrated.is_significant
                ),
                "survives": (
                    point.is_significant and calibrated.is_significant
                ),
            }
        )
    return rows


def render(report: dict) -> str:
    lines: list[str] = []
    cal = report["calibration"]
    add = lines.append

    add(f"Trial file    : {report['path']}")
    add(f"Scorer        : {report['scorer']}")
    add(
        f"Design        : {report['arms']} arms x "
        f"{report['question_count']} questions x {report['rep_count']} reps "
        f"= {report['trial_count']} trials"
    )
    add("")
    add(f"--- 1. Same-versus-same, arm {cal['arm']} against itself ---")
    add("")
    for key, entry in cal["detail"].items():
        per_rep = ", ".join(f"{value:.4g}" for value in entry["per_rep"])
        add(
            f"  {entry['label']:<24} reps [{per_rep}] {entry['unit']}"
            f"  -> noise floor {entry['spread']:.4g} "
            f"({entry['spread_kind']})"
        )
    add("")
    add(
        f"  Unstable questions       {len(cal['unstable_questions'])}"
        f"/{cal['question_count']} "
        f"({cal['flip_rate'] * 100:.1f}% flip between identical runs)"
    )
    if cal["unstable_questions"]:
        add(f"  {', '.join(cal['unstable_questions'])}")
    add("")
    add(
        f"--- 2. {report['baseline_arm']} vs {report['candidate_arm']}, "
        "before and after calibration ---"
    )
    add("")
    header = (
        f"  {'metric':<24}{'baseline':>10}{'candidate':>11}"
        f"{'delta':>10}   {'point':<16}{'calibrated':<16}"
    )
    add(header)
    add("  " + "-" * (len(header) - 2))
    for row in report["comparisons"]:
        flag = ""
        if row["overturned"]:
            flag = "  <-- OVERTURNED"
        elif row["survives"]:
            flag = "  <-- survives"
        add(
            f"  {row['label']:<24}{row['baseline']:>10.4g}"
            f"{row['candidate']:>11.4g}{row['delta']:>+10.4g}   "
            f"{row['point_verdict']:<16}{row['calibrated_verdict']:<16}{flag}"
        )
    add("")
    overturned = [row for row in report["comparisons"] if row["overturned"]]
    survives = [row for row in report["comparisons"] if row["survives"]]
    add(
        f"  {len(overturned)} of {len(report['comparisons'])} findings are "
        f"overturned by calibration; {len(survives)} survive."
    )
    if overturned:
        add("")
        for row in overturned:
            add(
                f"  OVERTURNED  {row['label']}: {row['delta']:+.4g} "
                f"{row['unit']} is within the {row['calibrated_floor']:.4g} "
                f"{row['unit']} this configuration moves on its own."
            )
    return "\n".join(lines)


def build_report(
    path: Path, baseline_arm: str, candidate_arm: str, scorer: str
) -> dict:
    trials = load_trials(path, scorer=scorer)
    arms = trials.arms
    for arm in (baseline_arm, candidate_arm):
        if arm not in arms:
            raise SystemExit(f"arm {arm!r} not in trial file; found {arms}")

    calibration = calibrate(trials, baseline_arm)
    return {
        "path": str(path),
        "scorer": scorer,
        "arms": len(arms),
        "question_count": len(trials.question_ids),
        "rep_count": len(trials.reps),
        "trial_count": len(trials),
        "baseline_arm": baseline_arm,
        "candidate_arm": candidate_arm,
        "calibration": calibration,
        "comparisons": judge(
            trials, baseline_arm, candidate_arm, calibration["floors"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.calibration",
        description="Measure an evaluation's noise floor and re-judge it.",
    )
    parser.add_argument("path", type=Path, help="recorded trial JSON")
    parser.add_argument("--baseline-arm", default="A")
    parser.add_argument("--candidate-arm", default="B")
    parser.add_argument(
        "--scorer", default="analytic", choices=sorted(CORRECTNESS_SCORERS)
    )
    parser.add_argument("--json", type=Path, help="also write the raw report")
    args = parser.parse_args(argv)

    report = build_report(
        args.path, args.baseline_arm, args.candidate_arm, args.scorer
    )
    print(render(report))
    if args.json:
        args.json.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"\nraw report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
