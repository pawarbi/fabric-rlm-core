from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audit import run_offline_audit
from .fixtures import SEED, VARIANTS, generate_fixtures
from .freeze import create_freeze_manifest, verify_freeze
from .grader import grade_answer
from .references import calculate_references


BASELINE_SHA = "b5226712a9aa41c3173d5f427e81244c333c0179"
DEFAULT_MODEL = "openai/gpt-4.1-mini"
ARMS = ("A", "B", "C")
RELIABILITY_TESTS = (
    "tests/test_knowledge_preflight.py",
    "tests/test_knowledge_api.py",
    "tests/test_knowledge_operations.py",
    "tests/test_knowledge_source_adapters.py",
    "tests/test_semantic_model_failure_integrity.py",
    "tests/test_runtime_timeout.py",
    "tests/test_runtime_verifier.py",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def build_schedule(
    questions: Sequence[Mapping[str, object]],
    *,
    repetitions: int,
    seed: int,
) -> list[dict[str, object]]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rng = random.Random(seed)
    schedule: list[dict[str, object]] = []
    for repetition in range(repetitions):
        ordered = [dict(question) for question in questions]
        rng.shuffle(ordered)
        for question in ordered:
            arms = list(ARMS)
            rng.shuffle(arms)
            for arm in arms:
                schedule.append(
                    {
                        "question_id": question["question_id"],
                        "domain": question["domain"],
                        "variant": question.get("variant", "descriptive"),
                        "repetition": repetition,
                        "arm": arm,
                    }
                )
    return schedule


def normalize_answer(
    answer: object,
    *,
    submitted: bool,
    failure_reason: str | None,
) -> dict[str, Any]:
    if not submitted:
        reason = (failure_reason or "").lower()
        return {"status": "timeout" if "timeout" in reason else "failed"}
    if isinstance(answer, Mapping):
        return dict(answer)
    return {"status": "answered", "value": answer}


def _verification_outcome(metadata: Mapping[str, object]) -> str:
    execution = metadata.get("verifier_execution")
    if isinstance(execution, Mapping) and execution.get("verified") is True:
        return "verified"
    return "not_verified"


def result_metrics(
    result: object,
    *,
    wall_seconds: float,
    provider_cost_usd: float | None,
) -> dict[str, object]:
    trajectory = getattr(result, "trajectory", None)
    metadata = getattr(trajectory, "metadata", None) or {}
    source = metadata.get("source_call_summary")
    source_summary = source if isinstance(source, Mapping) else {}
    injected = metadata.get("knowledge_lessons_injected")
    return {
        "submitted": bool(getattr(result, "submitted", False)),
        "failure_reason": getattr(result, "failure_reason", None),
        "turns": int(getattr(result, "n_turns", 0)),
        "max_turns": getattr(result, "max_turns", None),
        "prompt_tokens": getattr(result, "total_prompt_tokens", None),
        "completion_tokens": getattr(result, "total_completion_tokens", None),
        "cached_tokens": getattr(result, "total_cached_tokens", None),
        "reasoning_tokens": getattr(result, "total_reasoning_tokens", None),
        "provider_cost_usd": provider_cost_usd,
        "lm_seconds": getattr(result, "total_lm_seconds", None),
        "worker_seconds": getattr(result, "total_worker_seconds", None),
        "wall_seconds": wall_seconds,
        "source_calls": int(source_summary.get("source_calls", 0)),
        "failed_source_calls": int(source_summary.get("failed_source_calls", 0)),
        "source_seconds": float(source_summary.get("source_seconds", 0.0)),
        "verification_outcome": _verification_outcome(metadata),
        "integrity_ok": bool(getattr(result, "integrity_ok", True)),
        "knowledge_mode": metadata.get("knowledge_mode"),
        "knowledge_fingerprint": metadata.get("knowledge_fingerprint"),
        "lessons_injected": len(injected) if isinstance(injected, (list, tuple)) else 0,
        "operation_id": metadata.get("operation_id"),
        "operation_audit_status": metadata.get("operation_audit_status"),
    }


def _outcome_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {
        "correct": 0,
        "confident_wrong": 0,
        "incomplete": 0,
        "unsupported_claim": 0,
    }
    for row in rows:
        grade = row.get("grade")
        if isinstance(grade, Mapping):
            outcome = str(grade.get("outcome"))
            counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def summarize_trials(trials: Sequence[Mapping[str, object]]) -> dict[str, object]:
    per_domain: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    per_question: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for domain in sorted({str(row["domain"]) for row in trials}):
        for arm in ARMS:
            rows = [
                row for row in trials
                if row["domain"] == domain and row["arm"] == arm
            ]
            per_domain[domain][arm] = _outcome_counts(rows)
    for question_id in sorted({str(row["question_id"]) for row in trials}):
        for arm in ARMS:
            rows = [
                row for row in trials
                if row["question_id"] == question_id and row["arm"] == arm
            ]
            per_question[question_id][arm] = _outcome_counts(rows)
    learning_hurts: list[dict[str, str]] = []
    for question_id, arms in per_question.items():
        baseline = arms["A"]["correct"]
        for arm in ("B", "C"):
            if arms[arm]["correct"] < baseline:
                learning_hurts.append(
                    {
                        "question_id": question_id,
                        "baseline_arm": "A",
                        "worse_arm": arm,
                    }
                )
    return {
        "trials": len(trials),
        "per_domain": dict(per_domain),
        "per_question": dict(per_question),
        "learning_hurts": learning_hurts,
    }


def _provider_cost(lm: object) -> float | None:
    costs: list[float] = []
    for entry in getattr(lm, "history", ()) or ():
        response = entry.get("response") if isinstance(entry, Mapping) else None
        hidden = getattr(response, "_hidden_params", None)
        if isinstance(hidden, Mapping):
            cost = hidden.get("response_cost")
            if isinstance(cost, (int, float)):
                costs.append(float(cost))
    return sum(costs) if costs else None


def make_openrouter_lm(model: str) -> object:
    import dspy

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not available")
    reasoning = model.lower().startswith(
        ("openai/gpt-5", "openai/o1", "openai/o3", "openai/o4")
    )
    kwargs: dict[str, object] = {
        "model": f"openrouter/{model}",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": api_key,
        "max_tokens": 16_000 if reasoning else 4_096,
        "cache": False,
    }
    if not reasoning:
        kwargs["temperature"] = 1.0
    return dspy.LM(**kwargs)


def _domain_sources(fixtures: Path, domain: str, variant: str) -> dict[str, str]:
    return {
        path.stem: str(path)
        for path in sorted((fixtures / domain / variant).glob("*.csv"))
        if not path.name.endswith("_large.csv")
    }


def _task_text(
    question: Mapping[str, object],
    definitions: Mapping[str, object],
    variant: str,
) -> str:
    domain = str(question["domain"])
    domain_definitions = definitions["variants"][variant][domain]
    return (
        f"{question['text']}\n\n"
        f"Definitions and {variant} field mappings:\n"
        f"{json.dumps(domain_definitions, sort_keys=True)}\n\n"
        "Use only the supplied sources. Preserve reporting period, grain, units, "
        "and entity IDs. Do not infer causes from timing alone. If a required "
        "definition is missing, return status='needs_definition' or 'abstain'. "
        "Return answer as a dictionary with status, value, units, grain, period, "
        "entity_id or entity_ids when applicable, and claims. Each claim is a "
        "dictionary with text and supported."
    )


def _run_rlm(
    *,
    model: str,
    question: Mapping[str, object],
    definitions: Mapping[str, object],
    variant: str,
    inputs: Mapping[str, object] | None,
    knowledge: object | None,
    max_turns: int,
    timeout: float,
) -> tuple[object, object, float]:
    from fabric_rlm import RLM

    lm = make_openrouter_lm(model)
    rlm = RLM.from_task(
        task=_task_text(question, definitions, variant),
        inputs=dict(inputs or {}),
        outputs={"answer": dict},
        lm=lm,
        knowledge=knowledge,
        max_turns=max_turns,
        timeout=timeout,
        capture_evidence=True,
        enable_skill_autoloading=False,
        skills=[],
    )
    started = time.perf_counter()
    result = rlm.run()
    return result, lm, time.perf_counter() - started


def _development_results(
    *,
    model: str,
    domain: str,
    variant: str,
    definitions: Mapping[str, object],
    knowledge: object,
    max_turns: int,
    timeout: float,
    budget: list[int],
) -> list[object]:
    prompts = (
        {
            "domain": domain,
            "question_id": f"dev_{domain}_grain",
            "text": "Find one safe useful aggregation and report its grain without answering any evaluation question.",
        },
        {
            "domain": domain,
            "question_id": f"dev_{domain}_risk",
            "text": "Probe one potentially unsafe or expensive aggregation and report whether it executes.",
        },
    )
    results: list[object] = []
    for prompt in prompts:
        if budget[0] <= 0:
            break
        result, _lm, _wall = _run_rlm(
            model=model,
            question=prompt,
            definitions=definitions,
            variant=variant,
            inputs=None,
            knowledge=knowledge,
            max_turns=max_turns,
            timeout=timeout,
        )
        budget[0] -= 1
        results.append(result)
    return results


def run_live(
    *,
    fixtures: Path,
    output: Path,
    model: str,
    repetitions: int,
    seed: int,
    variants: Sequence[str],
    max_live_calls: int,
    max_turns: int,
    timeout: float,
    smoke: bool,
) -> dict[str, object]:
    if not os.environ.get("OPENROUTER_API_KEY"):
        result = {
            "status": "unmeasured",
            "reason": "OPENROUTER_API_KEY is not available",
            "trials": [],
        }
        _write_json(output, result)
        return result
    from fabric_rlm import RLM

    definitions = json.loads((fixtures / "definitions.json").read_text(encoding="utf-8"))
    questions = json.loads((fixtures / "questions.json").read_text(encoding="utf-8"))
    references = calculate_references(fixtures)
    selected: list[dict[str, object]] = []
    for variant in variants:
        domain_counts: dict[str, int] = defaultdict(int)
        for question in questions:
            domain = str(question["domain"])
            if smoke and domain_counts[domain] >= 1:
                continue
            selected.append({**question, "variant": variant})
            domain_counts[domain] += 1
    schedule = build_schedule(selected, repetitions=repetitions, seed=seed)
    budget = [max_live_calls]
    packages: dict[tuple[str, str, str], object | None] = {}
    package_summaries: dict[str, object] = {}
    for variant in variants:
        for domain in ("inventory", "manufacturing", "service"):
            sources = _domain_sources(fixtures, domain, variant)
            learned = RLM.learn(sources=sources)
            packages[(domain, variant, "A")] = None
            packages[(domain, variant, "B")] = learned
            development = _development_results(
                model=model,
                domain=domain,
                variant=variant,
                definitions=definitions,
                knowledge=learned,
                max_turns=max_turns,
                timeout=timeout,
                budget=budget,
            )
            enriched = RLM.enrich(learned, development) if development else learned
            packages[(domain, variant, "C")] = enriched
            package_summaries[f"{domain}:{variant}"] = {
                "learn_only_lessons": len(learned.package.lessons),
                "enriched_lessons": len(enriched.package.lessons),
                "development_runs": len(development),
                "B_fingerprint": learned.package.fingerprint,
                "C_fingerprint": enriched.package.fingerprint,
            }
    trials: list[dict[str, object]] = []
    questions_by_id = {str(question["question_id"]): question for question in selected}
    for trial in schedule:
        if budget[0] <= 0:
            break
        question = questions_by_id[str(trial["question_id"])]
        domain = str(trial["domain"])
        variant = str(trial["variant"])
        arm = str(trial["arm"])
        knowledge = packages[(domain, variant, arm)]
        inputs = _domain_sources(fixtures, domain, variant) if arm == "A" else None
        try:
            result, lm, wall = _run_rlm(
                model=model,
                question=question,
                definitions=definitions,
                variant=variant,
                inputs=inputs,
                knowledge=knowledge,
                max_turns=max_turns,
                timeout=timeout,
            )
            answer = normalize_answer(
                result.outputs.get("answer"),
                submitted=result.submitted,
                failure_reason=result.failure_reason,
            )
            metrics = result_metrics(
                result,
                wall_seconds=wall,
                provider_cost_usd=_provider_cost(lm),
            )
            error = None
        except Exception as exc:
            answer = {"status": "failed"}
            metrics = {
                "submitted": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "wall_seconds": None,
                "provider_cost_usd": None,
                "verification_outcome": "not_verified",
            }
            error = f"{type(exc).__name__}: {exc}"
        budget[0] -= 1
        expected = references[domain][variant][str(trial["question_id"])]
        grade = grade_answer(answer, expected)
        trials.append(
            {
                **trial,
                "model": model,
                "cache": False,
                "max_turns": max_turns,
                "timeout": timeout,
                "answer": answer,
                "grade": grade,
                "metrics": metrics,
                "error": error,
                "expected": expected,
                "workbook_correctness": "not_applicable",
                "evidence_coverage": sum(
                    1
                    for claim in answer.get("claims", ())
                    if isinstance(claim, Mapping) and claim.get("supported") is True
                ),
            }
        )
        _write_json(
            output,
            {
                "status": "partial",
                "baseline_sha": BASELINE_SHA,
                "model": model,
                "seed": seed,
                "repetitions": repetitions,
                "variants": list(variants),
                "max_live_calls": max_live_calls,
                "remaining_live_calls": budget[0],
                "packages": package_summaries,
                "trials": trials,
                "summary": summarize_trials(trials),
            },
        )
    final = {
        "status": "complete" if len(trials) == len(schedule) else "budget_limited",
        "baseline_sha": BASELINE_SHA,
        "model": model,
        "seed": seed,
        "repetitions": repetitions,
        "variants": list(variants),
        "max_live_calls": max_live_calls,
        "development_calls_included_in_budget": True,
        "remaining_live_calls": budget[0],
        "planned_trials": len(schedule),
        "packages": package_summaries,
        "trials": trials,
        "summary": summarize_trials(trials),
    }
    _write_json(output, final)
    return final


def run_offline(repo: Path, fixtures: Path, output_dir: Path) -> dict[str, object]:
    freeze_path = repo / "evaluation" / "generalization" / "frozen-baseline.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    mismatches = verify_freeze(repo, freeze)
    audit = run_offline_audit(repo, fixtures)
    command = [sys.executable, "-m", "pytest", *RELIABILITY_TESTS, "-q"]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    reliability = {
        "command": command,
        "exit_code": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    result = {
        "baseline_sha": BASELINE_SHA,
        "freeze_mismatches": mismatches,
        "audit": audit,
        "reliability_tests": reliability,
        "real_integrations": {
            "files": "tested",
            "lakehouse": "unavailable_no_credentials",
            "semantic_model": "unavailable_no_credentials",
        },
    }
    _write_json(output_dir / "offline-results.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--repo", type=Path, default=Path.cwd())
    freeze.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=SEED)
    prepare.add_argument("--large-rows", type=int, default=250_000)
    offline = sub.add_parser("offline")
    offline.add_argument("--repo", type=Path, default=Path.cwd())
    offline.add_argument("--fixtures", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    live = sub.add_parser("live")
    live.add_argument("--fixtures", type=Path, required=True)
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--model", default=DEFAULT_MODEL)
    live.add_argument("--repetitions", type=int, default=3)
    live.add_argument("--seed", type=int, default=SEED)
    live.add_argument("--variants", default=",".join(VARIANTS))
    live.add_argument("--max-live-calls", type=int, default=15)
    live.add_argument("--max-turns", type=int, default=6)
    live.add_argument("--timeout", type=float, default=120.0)
    live.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        _write_json(
            args.output,
            create_freeze_manifest(args.repo, baseline_sha=BASELINE_SHA),
        )
    elif args.command == "prepare":
        generate_fixtures(args.output, seed=args.seed, large_rows=args.large_rows)
        private = args.output / "private"
        _write_json(private / "references.json", calculate_references(args.output))
    elif args.command == "offline":
        result = run_offline(args.repo, args.fixtures, args.output)
        return 0 if not result["freeze_mismatches"] and result["reliability_tests"]["exit_code"] == 0 else 1
    elif args.command == "live":
        variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
        unknown = sorted(set(variants) - set(VARIANTS))
        if unknown:
            raise ValueError(f"unknown variants: {', '.join(unknown)}")
        run_live(
            fixtures=args.fixtures,
            output=args.output,
            model=args.model,
            repetitions=args.repetitions,
            seed=args.seed,
            variants=variants,
            max_live_calls=args.max_live_calls,
            max_turns=args.max_turns,
            timeout=args.timeout,
            smoke=args.smoke,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_schedule",
    "main",
    "make_openrouter_lm",
    "normalize_answer",
    "result_metrics",
    "run_live",
    "run_offline",
    "summarize_trials",
]
