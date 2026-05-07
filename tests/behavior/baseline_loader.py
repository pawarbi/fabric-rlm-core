"""Loader/validator for behavior-CI baselines.json.

The baseline file pins the per-qid pass status for each model so PR CI can
detect regressions.  This module:

* Defines the file's expected schema.
* Validates a loaded baseline (raises ``BaselineSchemaError`` with a clear
  message when the file is malformed or stale).
* Resolves the gates: ``per-qid stable-passing must still pass`` and
  ``aggregate floor (>= baseline_passes - 1)``.

Schema (informal):

    {
      "suite_version": "behavior-v1",
      "calibrated_at": "2026-05-07T12:00:00Z",
      "calibrated_against_commit": "<git sha or 'unknown'>",
      "questions_sha256": "<hex sha256 of questions.py builders>",
      "max_turns": 8,
      "timeout_s": 120,
      "calibration_runs_per_qid": 5,
      "models": {
        "openai/gpt-4.1-mini": {
          "questions": {
            "C1_sum_squares_mod": {
              "baseline_pass_rate": 1.0,
              "passes": 5, "runs": 5,
              "expected_to_pass": true
            },
            ...
          },
          "aggregate": {"baseline_passes": 5, "min_passes": 4}
        }
      }
    }

A qid is in the *blocking suite* iff its ``expected_to_pass`` is ``true``
(set by the calibrator when ``passes >= ceil(0.8 * runs)``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUITE_VERSION = "behavior-v1"


class BaselineSchemaError(ValueError):
    """Raised when a baseline file is malformed, stale, or missing required fields."""


@dataclass(frozen=True)
class QuestionBaseline:
    qid: str
    baseline_pass_rate: float
    passes: int
    runs: int
    expected_to_pass: bool


@dataclass(frozen=True)
class ModelBaseline:
    model: str
    questions: dict[str, QuestionBaseline]
    baseline_passes: int  # aggregate count of expected-to-pass qids
    min_passes: int  # aggregate floor (typically baseline_passes - 1)

    def stable_qids(self) -> list[str]:
        """Qids that the PR is required to keep passing."""
        return [qid for qid, qb in self.questions.items() if qb.expected_to_pass]


@dataclass(frozen=True)
class Baseline:
    suite_version: str
    calibrated_at: str
    calibrated_against_commit: str
    questions_sha256: str
    max_turns: int
    timeout_s: int
    calibration_runs_per_qid: int
    models: dict[str, ModelBaseline]


_REQUIRED_TOP = (
    "suite_version",
    "calibrated_at",
    "calibrated_against_commit",
    "questions_sha256",
    "max_turns",
    "timeout_s",
    "calibration_runs_per_qid",
    "models",
)


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise BaselineSchemaError(f"{ctx}: missing required key {key!r}")
    return d[key]


def parse_baseline(payload: dict[str, Any]) -> Baseline:
    """Validate and parse a loaded baseline dict; raise BaselineSchemaError on issues."""
    for k in _REQUIRED_TOP:
        _require(payload, k, "baseline")

    suite_version = payload["suite_version"]
    if suite_version != SUITE_VERSION:
        raise BaselineSchemaError(
            f"suite_version mismatch: file={suite_version!r} expected={SUITE_VERSION!r}; "
            "recalibrate with the current runner."
        )

    raw_models = payload["models"]
    if not isinstance(raw_models, dict) or not raw_models:
        raise BaselineSchemaError("baseline: 'models' must be a non-empty mapping")

    models: dict[str, ModelBaseline] = {}
    for model_name, model_data in raw_models.items():
        ctx = f"models[{model_name!r}]"
        questions_data = _require(model_data, "questions", ctx)
        aggregate = _require(model_data, "aggregate", ctx)

        if not isinstance(questions_data, dict) or not questions_data:
            raise BaselineSchemaError(f"{ctx}.questions must be a non-empty mapping")

        qbs: dict[str, QuestionBaseline] = {}
        for qid, qdata in questions_data.items():
            qctx = f"{ctx}.questions[{qid!r}]"
            qbs[qid] = QuestionBaseline(
                qid=qid,
                baseline_pass_rate=float(_require(qdata, "baseline_pass_rate", qctx)),
                passes=int(_require(qdata, "passes", qctx)),
                runs=int(_require(qdata, "runs", qctx)),
                expected_to_pass=bool(_require(qdata, "expected_to_pass", qctx)),
            )

        baseline_passes = int(_require(aggregate, "baseline_passes", f"{ctx}.aggregate"))
        min_passes = int(_require(aggregate, "min_passes", f"{ctx}.aggregate"))
        if min_passes > baseline_passes:
            raise BaselineSchemaError(
                f"{ctx}.aggregate.min_passes ({min_passes}) > baseline_passes ({baseline_passes})"
            )
        # Cross-check aggregate vs per-qid expected_to_pass count.
        stable = sum(1 for qb in qbs.values() if qb.expected_to_pass)
        if stable != baseline_passes:
            raise BaselineSchemaError(
                f"{ctx}.aggregate.baseline_passes ({baseline_passes}) != "
                f"count of expected_to_pass qids ({stable}). Recalibrate or fix by hand."
            )

        models[model_name] = ModelBaseline(
            model=model_name,
            questions=qbs,
            baseline_passes=baseline_passes,
            min_passes=min_passes,
        )

    return Baseline(
        suite_version=suite_version,
        calibrated_at=str(payload["calibrated_at"]),
        calibrated_against_commit=str(payload["calibrated_against_commit"]),
        questions_sha256=str(payload["questions_sha256"]),
        max_turns=int(payload["max_turns"]),
        timeout_s=int(payload["timeout_s"]),
        calibration_runs_per_qid=int(payload["calibration_runs_per_qid"]),
        models=models,
    )


def load_baseline(path: Path | str) -> Baseline:
    """Read and parse the baselines.json file at ``path``.

    Raises ``FileNotFoundError`` (with a hint about how to calibrate) if the
    file is missing, and ``BaselineSchemaError`` if malformed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"baseline file not found: {p}. "
            "Run `python -m tests.behavior.runner --calibrate --model openai/gpt-4.1-mini` "
            "on main and commit the result."
        )
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineSchemaError(f"{p}: invalid JSON: {exc}") from exc
    return parse_baseline(payload)


@dataclass(frozen=True)
class GateOutcome:
    passed: bool
    reasons: list[str]


def evaluate_gates(
    model_baseline: ModelBaseline,
    pr_results: dict[str, bool],
) -> GateOutcome:
    """Apply the per-qid gate (primary) and the aggregate floor (secondary).

    ``pr_results`` is a ``{qid: passed_bool}`` mapping from the PR run.
    Returns a ``GateOutcome`` whose ``reasons`` is empty iff both gates pass.
    """
    reasons: list[str] = []

    # Per-qid gate: every stable baseline-passing qid must still pass.
    for qid in model_baseline.stable_qids():
        if qid not in pr_results:
            reasons.append(f"per-qid: {qid} missing from PR results (was expected to pass)")
        elif not pr_results[qid]:
            reasons.append(f"per-qid: {qid} regressed (baseline passing -> PR failing)")

    # Aggregate floor: count only over baseline qids to avoid future callers
    # accidentally inflating the pass count with non-baseline qids.
    baseline_qids = set(model_baseline.questions)
    pr_passes = sum(1 for qid in baseline_qids if pr_results.get(qid) is True)
    if pr_passes < model_baseline.min_passes:
        reasons.append(
            f"aggregate: PR passed {pr_passes} qids; floor is {model_baseline.min_passes} "
            f"(baseline {model_baseline.baseline_passes})"
        )

    return GateOutcome(passed=not reasons, reasons=reasons)
