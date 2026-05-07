"""Runner for behavior CI: executes a Question via fabric_rlm.RLM and grades the answer.

Two modes:

* ``run_question(q, model, ...)`` — single attempt with retry-once on infra
  errors (HTTP 429 / timeouts / provider 5xx).  Returns a structured
  ``QuestionRun`` that classifies the outcome.
* ``calibrate(model, runs)`` — runs each question ``runs`` times and writes a
  ``baselines.json`` capturing per-qid pass rates and aggregate floor.  Used
  to bootstrap the baseline file on ``main`` before enabling the workflow.

CLI:

    python -m tests.behavior.runner --calibrate \
        --model openai/gpt-4.1-mini --runs 5 \
        --out tests/behavior/baselines.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .baseline_loader import SUITE_VERSION
from .grader import GradeResult, grade
from .questions import QUESTIONS, Question, get_question, questions_sha256


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class QuestionRun:
    qid: str
    model: str
    passed: bool
    answer: Any = None
    expected: Any = None
    reason: str = ""
    error_class: str | None = None  # "infra" | "wrong_answer" | "runner_error" | None
    error_message: str | None = None
    n_turns: int | None = None
    elapsed_s: float | None = None
    attempts: int = 1


# ---------------------------------------------------------------------------
# LM construction
# ---------------------------------------------------------------------------

_REASONING_PREFIXES = ("openai/gpt-5", "openai/o1", "openai/o3", "openai/o4")


def _is_reasoning(model: str) -> bool:
    return model.lower().startswith(_REASONING_PREFIXES)


def make_lm(model: str) -> Any:
    """Construct a dspy.LM via OpenRouter with cache disabled.

    ``cache=False`` is required so each PR invocation hits the provider; we are
    measuring the model, not a cached response.  Reasoning models (gpt-5/o-series)
    skip ``temperature`` (the API rejects it).
    """
    import dspy  # imported lazily so offline tests don't pull dspy.

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")

    kwargs: dict[str, Any] = {
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": api_key,
        "max_tokens": 16_000 if _is_reasoning(model) else 4_096,
        "cache": False,
    }
    if not _is_reasoning(model):
        kwargs["temperature"] = 1.0  # behavior CI matches dspy's default chat sampling

    # OpenRouter routing: dspy uses litellm under the hood; the "openrouter/" prefix
    # tells litellm to dispatch to the OpenRouter endpoint.
    return dspy.LM(model=f"openrouter/{model}", **kwargs)


# ---------------------------------------------------------------------------
# Infra-error classification
# ---------------------------------------------------------------------------

# Heuristic — these substrings indicate something other than "model produced
# wrong answer".  Anything matching is retried once before being recorded as an
# infra failure.  Conservative on purpose: false negatives (treating an infra
# error as a wrong answer) are worse than false positives (one extra retry).
_INFRA_TOKENS = (
    "rate limit",
    "rate-limit",
    "rate_limit",
    "429",
    "timeout",
    "timed out",
    "connection",
    "connection reset",
    "read timed out",
    "service unavailable",
    "503",
    "502",
    "504",
    "bad gateway",
    "remote disconnected",
    "maxretry",
    "retries exceeded",
)


def _classify_error(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if any(tok in msg for tok in _INFRA_TOKENS):
        return "infra"
    return "runner_error"


# ---------------------------------------------------------------------------
# Single-question execution
# ---------------------------------------------------------------------------

def _run_once(q: Question, model: str, *, max_turns: int, timeout_s: float) -> tuple[Any, int | None, float, BaseException | None]:
    """Run a single attempt; return (answer, n_turns, elapsed_s, exception_or_None).

    Imports fabric_rlm lazily so the offline test suite doesn't depend on dspy.
    """
    from fabric_rlm import RLM

    lm = make_lm(model)
    t0 = time.time()
    try:
        rlm = RLM.from_task(
            task=q.task,
            inputs=q.inputs if q.inputs else None,
            outputs=["answer"],
            lm=lm,
            max_turns=max_turns,
            timeout=timeout_s,
        )
        result = rlm.run()
        elapsed = time.time() - t0
        ans = None
        if getattr(result, "outputs", None):
            ans = result.outputs.get("answer")
        return (ans, getattr(result, "n_turns", None), elapsed, None)
    except BaseException as exc:  # noqa: BLE001 — we classify and re-raise selectively
        return (None, None, time.time() - t0, exc)


def run_question(
    q: Question,
    model: str,
    *,
    max_turns: int = 8,
    timeout_s: float = 120.0,
    retry_on_infra: bool = True,
) -> QuestionRun:
    """Run ``q`` against ``model``; grade the answer; classify failures.

    On infra errors (429 / timeout / 5xx), retries once.  Wrong-answer outcomes
    are NEVER retried — that's the regression signal we're trying to detect.
    """
    ans, turns, elapsed, exc = _run_once(q, model, max_turns=max_turns, timeout_s=timeout_s)
    attempts = 1

    if exc is not None and retry_on_infra and _classify_error(exc) == "infra":
        # One retry on infra; brief sleep to let upstream cool off.
        time.sleep(2.0)
        ans, turns, elapsed, exc = _run_once(q, model, max_turns=max_turns, timeout_s=timeout_s)
        attempts = 2

    if exc is not None:
        klass = _classify_error(exc)
        return QuestionRun(
            qid=q.qid,
            model=model,
            passed=False,
            answer=None,
            expected=q.expected,
            reason=f"{type(exc).__name__}: {exc}",
            error_class=klass,
            error_message=str(exc),
            n_turns=turns,
            elapsed_s=elapsed,
            attempts=attempts,
        )

    g: GradeResult = grade(ans, q.expected, cmp=q.cmp)
    return QuestionRun(
        qid=q.qid,
        model=model,
        passed=g.passed,
        answer=ans,
        expected=q.expected,
        reason=g.reason,
        error_class=None if g.passed else "wrong_answer",
        error_message=None,
        n_turns=turns,
        elapsed_s=elapsed,
        attempts=attempts,
    )


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_suite(
    model: str,
    *,
    questions: Iterable[Question] = QUESTIONS,
    max_turns: int = 8,
    timeout_s: float = 120.0,
) -> list[QuestionRun]:
    return [run_question(q, model, max_turns=max_turns, timeout_s=timeout_s) for q in questions]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _utc_now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def calibrate(
    model: str,
    *,
    runs: int = 5,
    max_turns: int = 8,
    timeout_s: float = 120.0,
    pass_rate_threshold: float = 0.8,
    questions: Iterable[Question] = QUESTIONS,
) -> dict[str, Any]:
    """Run each question ``runs`` times; produce a baselines.json payload for ``model``.

    A qid's ``expected_to_pass`` is True iff its pass rate >= ``pass_rate_threshold``.
    Aggregate ``min_passes`` is set to ``baseline_passes - 1`` (allow one stable qid
    to flake to wrong-answer; per-qid gate is the primary check).
    """
    qlist = list(questions)
    per_qid_passes: dict[str, int] = {q.qid: 0 for q in qlist}
    per_qid_runs: dict[str, int] = {q.qid: 0 for q in qlist}

    for q in qlist:
        for i in range(runs):
            print(f"[calibrate] {q.qid} run {i + 1}/{runs}", file=sys.stderr, flush=True)
            res = run_question(q, model, max_turns=max_turns, timeout_s=timeout_s)
            per_qid_runs[q.qid] += 1
            if res.passed:
                per_qid_passes[q.qid] += 1

    questions_payload: dict[str, Any] = {}
    expected_count = 0
    for q in qlist:
        passes = per_qid_passes[q.qid]
        nruns = per_qid_runs[q.qid]
        rate = (passes / nruns) if nruns else 0.0
        expected = rate >= pass_rate_threshold
        if expected:
            expected_count += 1
        questions_payload[q.qid] = {
            "baseline_pass_rate": round(rate, 3),
            "passes": passes,
            "runs": nruns,
            "expected_to_pass": expected,
        }

    aggregate = {
        "baseline_passes": expected_count,
        "min_passes": max(0, expected_count - 1),
    }

    return {
        "suite_version": SUITE_VERSION,
        "calibrated_at": _utc_now_iso(),
        "calibrated_against_commit": _git_head_sha(),
        "questions_sha256": questions_sha256(),
        "max_turns": max_turns,
        "timeout_s": int(timeout_s),
        "calibration_runs_per_qid": runs,
        "models": {
            model: {
                "questions": questions_payload,
                "aggregate": aggregate,
            }
        },
    }


def merge_calibration(existing: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """Merge a new model's calibration into an existing baselines.json payload.

    Top-level metadata (calibrated_at, commit, sha) reflects the most recent
    calibration; ``models`` is a per-model union (new replaces old for the
    calibrated model).
    """
    if not existing:
        return new
    merged = dict(existing)
    merged_models = dict(existing.get("models", {}))
    for k, v in new.get("models", {}).items():
        merged_models[k] = v
    merged["models"] = merged_models
    merged["suite_version"] = new["suite_version"]
    merged["calibrated_at"] = new["calibrated_at"]
    merged["calibrated_against_commit"] = new["calibrated_against_commit"]
    merged["questions_sha256"] = new["questions_sha256"]
    merged["max_turns"] = new["max_turns"]
    merged["timeout_s"] = new["timeout_s"]
    merged["calibration_runs_per_qid"] = new["calibration_runs_per_qid"]
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_calibrate(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    existing: dict[str, Any] | None = None
    if out_path.exists() and not args.replace:
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"warning: {out_path} exists but is invalid JSON; replacing.", file=sys.stderr)
            existing = None

    print(
        f"[calibrate] model={args.model} runs={args.runs} "
        f"max_turns={args.max_turns} timeout_s={args.timeout_s}",
        file=sys.stderr,
    )
    payload = calibrate(
        args.model,
        runs=args.runs,
        max_turns=args.max_turns,
        timeout_s=args.timeout_s,
    )
    merged = merge_calibration(existing, payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"[calibrate] wrote {out_path}", file=sys.stderr)

    # Friendly summary to stdout.
    mb = merged["models"][args.model]
    print(json.dumps(mb, indent=2))
    return 0


def _cli_run_one(args: argparse.Namespace) -> int:
    q = get_question(args.qid)
    res = run_question(q, args.model, max_turns=args.max_turns, timeout_s=args.timeout_s)
    print(json.dumps(asdict(res), indent=2, default=str))
    return 0 if res.passed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="behavior-runner")
    sub = p.add_subparsers(dest="cmd")

    # Default action: --calibrate flag on the root parser, for ergonomics.
    p.add_argument("--calibrate", action="store_true", help="run calibration and write baselines.json")
    p.add_argument("--run-qid", default=None, help="run a single qid and print the structured result")
    p.add_argument("--model", default="openai/gpt-4.1-mini")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument(
        "--out",
        default=str(Path(__file__).parent / "baselines.json"),
        help="output baseline path",
    )
    p.add_argument("--replace", action="store_true", help="overwrite existing baselines.json instead of merging")

    args = p.parse_args(argv)
    if args.calibrate:
        return _cli_calibrate(args)
    if args.run_qid:
        args.qid = args.run_qid
        return _cli_run_one(args)
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
