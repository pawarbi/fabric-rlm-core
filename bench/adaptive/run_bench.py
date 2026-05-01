"""Adaptive bench runner — 4 modes × 3 buckets, deterministic and resumable.

Modes:
  baseline      - RLM(engine=inner, max_turns=10) once
  retry_only    - baseline × max_attempts (no escalation, validator-gated)
  adaptive      - RLM(engine='adaptive', adaptive=...)
  ceiling       - RLM(strong_lm, max_turns=20, reasoning_effort='high') once

Buckets:
  easy          - 12 hand-authored cases (exact / keyword match)
  longcot       - 20 hard CS puzzles (verify_response)
  spark         - 1 long-log RCA (field-level scoring 0..5)

Usage:
  python -m bench.adaptive.run_bench \\
      --output bench/adaptive/results-0.1.10.json \\
      --cheap-lm openai/gpt-4.1-mini \\
      --strong-lm openai/gpt-5 \\
      --modes baseline retry_only adaptive ceiling \\
      --buckets easy longcot spark
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Ensure bench/adaptive is importable when run as a script.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from longcot_adapter import (  # type: ignore  # noqa: E402
    LongCoTExample,
    format_question_only_prompt,
    load_jsonl_dataset,
    verify_response,
)


# ---------------------------------------------------------------------------
# Cost normalization
# ---------------------------------------------------------------------------

# Normalized Cost Unit: prompt + 4*completion (proxy for OpenAI 1:4 in:out
# pricing), times a model-family multiplier. Defaults are intentionally crude;
# the comparison between modes is what matters, not absolute USD.
NCU_MULTIPLIER_DEFAULTS = {
    "cheap": 1.0,
    "strong": 5.0,
}


def ncu(prompt_tokens: int, completion_tokens: int, family: str) -> float:
    mult = NCU_MULTIPLIER_DEFAULTS.get(family, 1.0)
    return mult * (prompt_tokens + 4 * completion_tokens)


# ---------------------------------------------------------------------------
# Per-bucket loaders + verifiers
# ---------------------------------------------------------------------------


@dataclass
class BenchCase:
    id: str
    bucket: str
    template: str
    inputs: dict[str, Any]
    verifier: Any  # Callable[[answer_text: str], (passed, score_0_to_1)]
    max_score: float = 1.0
    raw: dict[str, Any] = field(default_factory=dict)


def _exact_match(expected: str):
    expected_norm = expected.strip().lower()

    def verify(answer_text: str) -> tuple[bool, float]:
        if answer_text is None:
            return False, 0.0
        actual = str(answer_text).strip().lower()
        # Allow the answer to be embedded in a larger sentence; we only require
        # the canonical form to appear.
        passed = expected_norm in actual or actual == expected_norm
        return passed, 1.0 if passed else 0.0

    return verify


def _keyword_match(keyword: str):
    kw = keyword.strip().lower()

    def verify(answer_text: str) -> tuple[bool, float]:
        if answer_text is None:
            return False, 0.0
        passed = kw in str(answer_text).lower()
        return passed, 1.0 if passed else 0.0

    return verify


def load_easy_cases(path: Path) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row["validator"] == "exact_match":
            verifier = _exact_match(row["answer"])
        else:
            verifier = _keyword_match(row["answer"])
        cases.append(
            BenchCase(
                id=row["id"],
                bucket=row["bucket"],
                template=row["template"],
                inputs={"question": row["question"]},
                verifier=verifier,
                raw=row,
            )
        )
    return cases


def _verify_longcot(example: LongCoTExample):
    def verify(answer_text: str) -> tuple[bool, float]:
        try:
            result = verify_response(example, answer_text)
        except Exception:
            return False, 0.0
        passed = bool(result.supported and result.correct is True)
        return passed, 1.0 if passed else 0.0

    return verify


def load_longcot_cases(path: Path) -> list[BenchCase]:
    examples = load_jsonl_dataset(path)
    cases: list[BenchCase] = []
    for ex in examples:
        prompt = format_question_only_prompt(ex)
        cases.append(
            BenchCase(
                id=str(ex.id or ex.template or "longcot"),
                bucket="longcot",
                template=str(ex.template or "unknown"),
                inputs={"question": prompt},
                verifier=_verify_longcot(ex),
                raw={"id": ex.id, "template": ex.template, "difficulty": ex.difficulty},
            )
        )
    return cases


# Spark log scoring (ported from _spark_run_eval_reference.py:score()).
_SPARK_GROUND_TRUTH = {
    "failed_job_id": 17,
    "failed_stage_id": 42,
    "top_3_slow_tasks_ids": {"88001", "88002", "88003"},
    "oom_count": 50,
    "root_cause_keywords": ["outofmemory", "oom", "executor 7", "heap"],
}


def _verify_spark(answer_text: str) -> tuple[bool, float]:
    if not answer_text:
        return False, 0.0
    text = str(answer_text).lower()
    score = 0
    if "17" in text and "job" in text:
        score += 1
    if "42" in text and "stage" in text:
        score += 1
    if any(kw in text for kw in _SPARK_GROUND_TRUTH["root_cause_keywords"]):
        score += 1
    found_tasks = sum(
        1 for tid in _SPARK_GROUND_TRUTH["top_3_slow_tasks_ids"] if tid in text
    )
    if found_tasks >= 3:
        score += 1
    nums = [int(n) for n in re.findall(r"\b(\d{2,3})\b", str(answer_text))]
    if any(45 <= n <= 55 for n in nums):
        score += 1
    return score >= 4, score / 5.0


def load_spark_case(log_path: Path) -> BenchCase:
    prompt = (
        f"A Spark application log is on the local filesystem at:\n"
        f"  {log_path}\n\n"
        f"It is approximately 53 MB / 500,000 lines — too large to fit in your context. "
        f"You MUST read it using Python code. Use efficient streaming / grep-style "
        f"line filtering.\n\n"
        f"Find and report exactly these five things as a JSON dict in your final answer:\n"
        f"  - failed_job_id: int (e.g. 17)\n"
        f"  - failed_stage_id: int (e.g. 42)\n"
        f"  - root_cause: str (one short sentence)\n"
        f"  - top_3_slow_tasks: list of {{\"task_id\": str, \"duration_ms\": int}} sorted DESC by duration\n"
        f"  - oom_count: int (total ERROR lines mentioning OutOfMemoryError)\n\n"
        f"Hints: the fatal failure is at the END of the log. Slow tasks are 'Finished task ... in N ms' "
        f"with N > 100000. OOM lines contain 'OutOfMemoryError'.\n"
    )
    return BenchCase(
        id="spark-rca-1",
        bucket="spark",
        template="rca",
        inputs={"question": prompt},
        verifier=lambda t: _verify_spark(t),
        max_score=5.0,
        raw={"log_path": str(log_path)},
    )


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


@dataclass
class ModeResult:
    case_id: str
    bucket: str
    template: str
    mode: str
    passed: bool
    score: float
    turns_used: int
    wall_seconds: float
    prompt_tokens: int
    completion_tokens: int
    ncu: float
    attempts: int
    lm_calls: int
    answer_preview: str
    error: str | None = None


def _extract_answer(result: Any) -> str:
    """Best-effort extraction of the model's final answer.

    Order:
      1. Explicit payload field (``answer`` / ``output`` / ``result`` / ``report``)
      2. Concatenated stdout from all turns. The v6 code agent often calls
         ``submit()`` with no kwargs and prints intermediate results across
         multiple turns; for exact-match validators we want to see the union.
      3. Repr fallback.
    """
    if result is None:
        return ""
    payload = getattr(result, "payload", None) or {}
    for key in ("answer", "output", "result", "report"):
        if key in payload and payload[key] is not None:
            return str(payload[key])
    traj = getattr(result, "trajectory", None)
    turns = getattr(traj, "turns", None) if traj else None
    if turns:
        # Concatenate stdout AND response_text — for the v6 code agent the
        # final answer often appears only in the model's natural-language
        # response (e.g. just before a no-arg submit()), not in printed code.
        parts: list[str] = []
        for t in turns:
            txt = (getattr(t, "stdout", "") or "").strip()
            if txt:
                parts.append(txt)
            rt = (getattr(t, "response_text", "") or "").strip()
            if rt:
                parts.append(rt)
        joined = "\n".join(parts).strip()
        if joined:
            return joined
    return str(payload or "")


def _result_token_metrics(result: Any) -> tuple[int, int]:
    """Sum prompt+completion tokens. Falls back to summing per-turn fields.

    Some LM backends (e.g. dspy.LM via litellm wrapping OpenRouter) drop usage
    metadata, so the top-level ``total_*_tokens`` fields can be ``None`` even
    though individual turns recorded them. We sum the trajectory as a fallback.
    """
    p = getattr(result, "total_prompt_tokens", None)
    c = getattr(result, "total_completion_tokens", None)
    if p is None or c is None:
        traj = getattr(result, "trajectory", None)
        turns = getattr(traj, "turns", None) if traj else None
        if turns:
            sp = sum(int(getattr(t, "prompt_tokens", None) or 0) for t in turns)
            sc = sum(int(getattr(t, "completion_tokens", None) or 0) for t in turns)
            if sp or sc:
                return sp, sc
    return int(p or 0), int(c or 0)


def _turns_used(result: Any) -> int:
    traj = getattr(result, "trajectory", None)
    if traj is None:
        return 0
    return len(getattr(traj, "turns", []) or [])


_LM_BACKEND = "fabric"  # set by main() from --lm-backend


def _resolve_lm_spec(model: str) -> Any:
    """Build an LM spec for the active backend.

    - ``fabric``: pass the string through unchanged. Caller is expected to use a
      ``fabric/<model>`` prefix; routed via :func:`fabric_rlm.lm._fabric_factory`.
    - ``openrouter``: build a dict spec that targets ``openrouter.ai`` via the
      OpenAI-compatible API. Strips a leading ``fabric/`` if present so the same
      ``--cheap-lm fabric/gpt-4.1-mini`` works for both backends.
    """
    if _LM_BACKEND == "openrouter":
        bare = model.split("/", 1)[1] if model.startswith("fabric/") else model
        # openrouter expects openai/<name> for OpenAI models
        if "/" not in bare:
            bare = f"openai/{bare}"
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "lm_backend=openrouter but OPENROUTER_API_KEY is not set"
            )
        return {
            "model": f"openrouter/{bare}" if not bare.startswith("openrouter/") else bare,
            "api_key": key,
            "api_base": "https://openrouter.ai/api/v1",
            "cache": False,
        }
    # default: fabric — caller must already prefix correctly
    return model


def run_baseline(case: BenchCase, *, lm_spec: str, inner_engine: str) -> ModeResult:
    from fabric_rlm import RLM

    rlm = RLM(
        signature="question -> answer",
        lm=_resolve_lm_spec(lm_spec),
        engine=inner_engine,
        max_turns=10,
        enable_router=False,
        enable_verifier=False,
        enable_skill_autoloading=False,
        enable_reflection=False,
    )
    return _run_single(case, rlm, mode="baseline", family="cheap")


def run_retry_only(
    case: BenchCase,
    *,
    lm_spec: str,
    inner_engine: str,
    max_attempts: int,
) -> ModeResult:
    """Same baseline config × N attempts, no escalation."""
    from fabric_rlm import RLM

    t0 = time.time()
    best: ModeResult | None = None
    total_p = total_c = total_turns = 0
    last_answer = ""
    last_err: str | None = None
    attempts = 0
    for i in range(max_attempts):
        attempts += 1
        try:
            rlm = RLM(
                signature="question -> answer",
                lm=_resolve_lm_spec(lm_spec),
                engine=inner_engine,
                max_turns=10,
                enable_router=False,
                enable_verifier=False,
                enable_skill_autoloading=False,
                enable_reflection=False,
            )
            res = rlm.run(case.inputs)
            ans = _extract_answer(res)
            p, c = _result_token_metrics(res)
            total_p += p
            total_c += c
            total_turns += _turns_used(res)
            last_answer = ans
            passed, score = case.verifier(ans)
            if passed:
                best = ModeResult(
                    case_id=case.id, bucket=case.bucket, template=case.template,
                    mode="retry_only", passed=True, score=score,
                    turns_used=total_turns, wall_seconds=time.time() - t0,
                    prompt_tokens=total_p, completion_tokens=total_c,
                    ncu=ncu(total_p, total_c, "cheap"),
                    attempts=attempts, lm_calls=attempts,
                    answer_preview=str(ans)[:500],
                )
                return best
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            break
    # All retries failed
    passed, score = case.verifier(last_answer) if last_answer else (False, 0.0)
    return ModeResult(
        case_id=case.id, bucket=case.bucket, template=case.template,
        mode="retry_only", passed=passed, score=score,
        turns_used=total_turns, wall_seconds=time.time() - t0,
        prompt_tokens=total_p, completion_tokens=total_c,
        ncu=ncu(total_p, total_c, "cheap"),
        attempts=attempts, lm_calls=attempts,
        answer_preview=str(last_answer)[:500],
        error=last_err,
    )


def run_adaptive(
    case: BenchCase,
    *,
    cheap_lm: str,
    strong_lm: str,
    inner_engine: str,
    max_attempts: int,
) -> ModeResult:
    from fabric_rlm import RLM

    def validator(result):
        ans = _extract_answer(result)
        passed, _ = case.verifier(ans)
        return passed

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rlm = RLM(
            signature="question -> answer",
            lm=_resolve_lm_spec(cheap_lm),
            engine="adaptive",
            inner_engine=inner_engine,
            max_turns=10,
            enable_router=False,
            enable_verifier=False,
            enable_skill_autoloading=False,
            enable_reflection=False,
            adaptive=dict(
                strong_lm=_resolve_lm_spec(strong_lm),
                max_attempts=max_attempts,
                validator=validator,
                parallel_rollouts=3,
            ),
        )
    err: str | None = None
    answer = ""
    res = None
    try:
        res = rlm.run(case.inputs)
        answer = _extract_answer(res)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    p, c = _result_token_metrics(res) if res is not None else (0, 0)
    turns_used = _turns_used(res)
    meta = {}
    if res is not None and getattr(res, "trajectory", None):
        meta = (res.trajectory.metadata or {}).get("adaptive", {}) or {}
    attempt_list = meta.get("attempts") or []
    attempts = len(attempt_list) if attempt_list else 1
    rungs = [a.get("rung", 0) for a in attempt_list]
    used_strong = any(r >= 4 for r in rungs)
    family = "strong" if used_strong else "cheap"
    passed, score = case.verifier(answer) if answer else (False, 0.0)
    result_obj = ModeResult(
        case_id=case.id, bucket=case.bucket, template=case.template,
        mode="adaptive", passed=passed, score=score,
        turns_used=turns_used, wall_seconds=elapsed,
        prompt_tokens=p, completion_tokens=c,
        ncu=ncu(p, c, family),
        attempts=attempts, lm_calls=attempts,
        answer_preview=str(answer)[:500],
        error=err,
    )
    # Stash adaptive rung trace into the answer_preview prefix so it shows up
    # in the JSON without a schema change. (Used by signal-ab analysis.)
    if rungs:
        result_obj.answer_preview = (
            f"[rungs={rungs} stop={meta.get('stop_reason')}] " + result_obj.answer_preview
        )
    return result_obj


def run_ceiling(case: BenchCase, *, strong_lm: str, inner_engine: str) -> ModeResult:
    from fabric_rlm import RLM

    base_spec = _resolve_lm_spec(strong_lm)
    # When the spec is a plain string, wrap as a dict so we can attach
    # reasoning_effort. fabric_rlm.lm.resolve_lm handles dict specs natively.
    if isinstance(base_spec, str):
        spec = {"model": base_spec, "reasoning_effort": "high"}
    else:
        spec = {**base_spec, "reasoning_effort": "high"}
    rlm = RLM(
        signature="question -> answer",
        lm=spec,
        engine=inner_engine,
        max_turns=20,
        enable_router=False,
        enable_verifier=False,
        enable_skill_autoloading=False,
        enable_reflection=False,
    )
    return _run_single(case, rlm, mode="ceiling", family="strong")


def _run_single(
    case: BenchCase, rlm: Any, *, mode: str, family: str
) -> ModeResult:
    t0 = time.time()
    err: str | None = None
    answer = ""
    res = None
    try:
        res = rlm.run(case.inputs)
        answer = _extract_answer(res)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    p, c = _result_token_metrics(res) if res is not None else (0, 0)
    passed, score = case.verifier(answer) if answer else (False, 0.0)
    return ModeResult(
        case_id=case.id, bucket=case.bucket, template=case.template,
        mode=mode, passed=passed, score=score,
        turns_used=_turns_used(res), wall_seconds=elapsed,
        prompt_tokens=p, completion_tokens=c,
        ncu=ncu(p, c, family),
        attempts=1, lm_calls=1,
        answer_preview=str(answer)[:500],
        error=err,
    )


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------


def aggregate(results: list[ModeResult]) -> dict[str, Any]:
    by_mode_bucket: dict[str, dict[str, dict[str, Any]]] = {}
    for r in results:
        by_mode_bucket.setdefault(r.mode, {}).setdefault(
            r.bucket, {"passed": 0, "total": 0, "ncu": 0.0, "turns": [], "wall": 0.0}
        )
        cell = by_mode_bucket[r.mode][r.bucket]
        cell["passed"] += int(r.passed)
        cell["total"] += 1
        cell["ncu"] += r.ncu
        cell["turns"].append(r.turns_used)
        cell["wall"] += r.wall_seconds

    # Per-template breakdown
    by_mode_tpl: dict[str, dict[str, dict[str, Any]]] = {}
    for r in results:
        key = f"{r.bucket}/{r.template}"
        by_mode_tpl.setdefault(r.mode, {}).setdefault(
            key, {"passed": 0, "total": 0, "ncu": 0.0}
        )
        cell = by_mode_tpl[r.mode][key]
        cell["passed"] += int(r.passed)
        cell["total"] += 1
        cell["ncu"] += r.ncu

    # Median turns helper
    def _median(xs: list[int]) -> float:
        if not xs:
            return 0.0
        xs = sorted(xs)
        n = len(xs)
        if n % 2 == 1:
            return float(xs[n // 2])
        return (xs[n // 2 - 1] + xs[n // 2]) / 2.0

    bucket_totals = {}
    for mode, buckets in by_mode_bucket.items():
        bucket_totals[mode] = {
            b: {
                "passed": cell["passed"],
                "total": cell["total"],
                "ncu_total": cell["ncu"],
                "turns_median": _median(cell["turns"]),
                "wall_total_s": cell["wall"],
            }
            for b, cell in buckets.items()
        }

    return {
        "by_mode_bucket": bucket_totals,
        "by_mode_template": by_mode_tpl,
        "totals_by_mode": {
            mode: {
                "passed": sum(c["passed"] for c in buckets.values()),
                "total": sum(c["total"] for c in buckets.values()),
                "ncu_total": sum(c["ncu_total"] for c in buckets.values()),
                "wall_total_s": sum(c["wall_total_s"] for c in buckets.values()),
            }
            for mode, buckets in bucket_totals.items()
        },
    }


def print_table(agg: dict[str, Any]) -> None:
    print("\n=== Per-bucket results ===")
    modes = sorted(agg["by_mode_bucket"].keys())
    buckets = sorted({b for m in agg["by_mode_bucket"].values() for b in m.keys()})
    header = f"{'Bucket':<14}" + "".join(f"  {m:<22}" for m in modes)
    print(header)
    for b in buckets:
        row = f"{b:<14}"
        for m in modes:
            cell = agg["by_mode_bucket"].get(m, {}).get(b)
            if cell:
                row += f"  {cell['passed']}/{cell['total']:<3} ncu={cell['ncu_total']:<10.0f}"
            else:
                row += f"  {'-':<22}"
        print(row)
    print("\n=== Totals ===")
    for m in modes:
        t = agg["totals_by_mode"].get(m, {})
        print(
            f"  {m:<14} passed={t.get('passed', 0)}/{t.get('total', 0)}  "
            f"ncu={t.get('ncu_total', 0):.0f}  wall={t.get('wall_total_s', 0):.0f}s"
        )


# ---------------------------------------------------------------------------
# Win conditions
# ---------------------------------------------------------------------------


def evaluate_win_conditions(agg: dict[str, Any], *, hard_buckets=("longcot", "spark")) -> dict[str, Any]:
    """Score the 6 win conditions from the plan.

    Returns a dict with a per-condition pass/fail and a short rationale.
    Win conditions are advisory — the bench prints them but does not crash.
    """
    bb = agg["by_mode_bucket"]
    res: dict[str, Any] = {}

    # Hard-case totals
    def hard_passed(mode: str) -> int:
        return sum(bb.get(mode, {}).get(b, {}).get("passed", 0) for b in hard_buckets)

    def hard_total(mode: str) -> int:
        return sum(bb.get(mode, {}).get(b, {}).get("total", 0) for b in hard_buckets)

    base_p = hard_passed("baseline")
    base_total = hard_total("baseline")
    adapt_p = hard_passed("adaptive")
    retry_p = hard_passed("retry_only") if "retry_only" in bb else None
    base_failures = max(0, base_total - base_p)
    floor = max(3, -(-base_failures // 2))  # ceil(base_failures/2)

    # 1: absolute lift
    res["1_lift_on_hard"] = {
        "required": floor,
        "achieved_extra_wins": max(0, adapt_p - base_p),
        "passed": (adapt_p - base_p) >= floor,
    }
    # 2: beats retry_only
    if retry_p is not None:
        res["2_beats_retry"] = {
            "required_extra_over_retry": 2,
            "actual": adapt_p - retry_p,
            "passed": (adapt_p - retry_p) >= 2,
        }
    else:
        res["2_beats_retry"] = {"skipped": "retry_only mode not run"}
    # 3: easy non-regression + ≤1 hard regression vs baseline
    easy_base = bb.get("baseline", {}).get("easy", {})
    easy_adapt = bb.get("adaptive", {}).get("easy", {})
    if not easy_adapt:
        res["3_no_regression"] = {"skipped": "adaptive mode not run on easy bucket"}
    else:
        easy_ok = easy_adapt.get("passed", 0) >= easy_base.get("passed", 0)
        res["3_no_regression"] = {
            "easy_pass_rate_ok": easy_ok,
            "passed": easy_ok,
        }
    # 4: easy cost not blown up (median turns / NCU)
    if not easy_adapt or not easy_base:
        res["4_easy_cost_ok"] = {"skipped": "need both baseline and adaptive on easy"}
    else:
        easy_cost_ok = True
        if easy_base.get("turns_median", 1) > 0:
            ratio_turns = easy_adapt.get("turns_median", 0) / easy_base["turns_median"]
            easy_cost_ok = ratio_turns <= 1.25
        if easy_base.get("ncu_total", 1) > 0:
            ratio_ncu = easy_adapt.get("ncu_total", 0) / easy_base["ncu_total"]
            easy_cost_ok = easy_cost_ok and ratio_ncu <= 1.4
        res["4_easy_cost_ok"] = {"passed": easy_cost_ok}
    # 5: cheaper than ceiling at ≤5pt accuracy gap
    if "ceiling" in bb:
        ceil_total_pass = sum(c.get("passed", 0) for c in bb.get("ceiling", {}).values())
        ceil_total_n = sum(c.get("total", 0) for c in bb.get("ceiling", {}).values())
        adapt_total_pass = sum(c.get("passed", 0) for c in bb.get("adaptive", {}).values())
        adapt_total_n = sum(c.get("total", 0) for c in bb.get("adaptive", {}).values())
        ceil_acc = ceil_total_pass / ceil_total_n if ceil_total_n else 0
        adapt_acc = adapt_total_pass / adapt_total_n if adapt_total_n else 0
        ceil_ncu = sum(c.get("ncu_total", 0) for c in bb.get("ceiling", {}).values())
        adapt_ncu = sum(c.get("ncu_total", 0) for c in bb.get("adaptive", {}).values())
        ncu_ratio = (adapt_ncu / ceil_ncu) if ceil_ncu else float("inf")
        gap = ceil_acc - adapt_acc
        res["5_cheaper_than_ceiling"] = {
            "ncu_ratio_vs_ceiling": ncu_ratio,
            "accuracy_gap": gap,
            "passed": ncu_ratio < 0.75 and gap <= 0.05,
        }
    else:
        res["5_cheaper_than_ceiling"] = {"skipped": "ceiling mode not run"}

    # 6: ladder is actually used — needs per-attempt log; deferred to a separate
    # pass over the raw results JSON.
    res["6_ladder_used"] = {"deferred": "compute from per-case attempt logs"}
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def maybe_ensure_spark_log(generator: Path, target: Path, lines: int) -> Path | None:
    env_path = os.environ.get("SPARK_LOG_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    if target.exists():
        return target
    if not generator.exists():
        return None
    print(f"[setup] Generating Spark log ({lines} lines) at {target} ...", flush=True)
    import subprocess
    subprocess.run(
        [sys.executable, str(generator), str(lines)],
        check=True,
        cwd=str(generator.parent),
    )
    return target if target.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.adaptive.run_bench")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cheap-lm", default="fabric/gpt-4.1-mini")
    parser.add_argument("--strong-lm", default="fabric/gpt-5")
    parser.add_argument(
        "--lm-backend",
        choices=("fabric", "openrouter"),
        default="fabric",
        help="fabric uses fabric/<model> via Fabric notebook; openrouter routes "
        "via OPENROUTER_API_KEY (works on a dev box).",
    )
    parser.add_argument("--inner-engine", default="v6-custom")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "retry_only", "adaptive", "ceiling"],
        choices=["baseline", "retry_only", "adaptive", "ceiling"],
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=["easy", "longcot", "spark"],
        choices=["easy", "longcot", "spark"],
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=6,
        help="LadderPolicy budget. With parallel_rollouts=3 at rung 3, need ≥6 "
        "for rung 4 (strong LM) to be reachable.",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap cases per bucket")
    parser.add_argument("--spark-lines", type=int, default=200_000)
    args = parser.parse_args(argv)

    global _LM_BACKEND
    _LM_BACKEND = args.lm_backend
    print(f"[run] lm_backend={args.lm_backend}")

    cases: list[BenchCase] = []
    if "easy" in args.buckets:
        easy = load_easy_cases(HERE / "easy_cases.jsonl")
        if args.limit:
            easy = easy[: args.limit]
        cases += easy
    if "longcot" in args.buckets:
        lc = load_longcot_cases(HERE / "longcot_cs_hard_pilot20.jsonl")
        if args.limit:
            lc = lc[: args.limit]
        cases += lc
    if "spark" in args.buckets:
        log_path = maybe_ensure_spark_log(
            HERE / "spark_generate.py",
            HERE / "spark_app.log",
            args.spark_lines,
        )
        if log_path is None:
            print("[skip] spark bucket: no log available and no generator")
        else:
            cases.append(load_spark_case(log_path))

    print(f"[run] {len(cases)} cases × {len(args.modes)} modes")
    results: list[ModeResult] = []
    for ci, case in enumerate(cases, 1):
        print(f"\n--- case {ci}/{len(cases)}: {case.bucket}/{case.template}/{case.id}")
        for mode in args.modes:
            print(f"  mode={mode} ...", end="", flush=True)
            try:
                if mode == "baseline":
                    r = run_baseline(case, lm_spec=args.cheap_lm, inner_engine=args.inner_engine)
                elif mode == "retry_only":
                    r = run_retry_only(case, lm_spec=args.cheap_lm, inner_engine=args.inner_engine, max_attempts=args.max_attempts)
                elif mode == "adaptive":
                    r = run_adaptive(case, cheap_lm=args.cheap_lm, strong_lm=args.strong_lm, inner_engine=args.inner_engine, max_attempts=args.max_attempts)
                elif mode == "ceiling":
                    r = run_ceiling(case, strong_lm=args.strong_lm, inner_engine=args.inner_engine)
                else:
                    continue
            except Exception as e:
                tb = traceback.format_exc()[:600]
                print(f" ERROR: {e}\n{tb}")
                r = ModeResult(
                    case_id=case.id, bucket=case.bucket, template=case.template,
                    mode=mode, passed=False, score=0.0, turns_used=0,
                    wall_seconds=0.0, prompt_tokens=0, completion_tokens=0,
                    ncu=0.0, attempts=0, lm_calls=0, answer_preview="",
                    error=f"{type(e).__name__}: {e}",
                )
            else:
                tag = "OK" if r.passed else "FAIL"
                print(f" {tag} score={r.score:.2f} turns={r.turns_used} ncu={r.ncu:.0f} wall={r.wall_seconds:.1f}s")
            results.append(r)

    agg = aggregate(results)
    win = evaluate_win_conditions(agg)
    print_table(agg)
    print("\n=== Win conditions (advisory) ===")
    for k, v in win.items():
        print(f"  {k}: {v}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "cheap_lm": args.cheap_lm,
                    "strong_lm": args.strong_lm,
                    "inner_engine": args.inner_engine,
                    "modes": args.modes,
                    "buckets": args.buckets,
                    "max_attempts": args.max_attempts,
                    "spark_lines": args.spark_lines,
                },
                "results": [asdict(r) for r in results],
                "aggregates": agg,
                "win_conditions": win,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
