"""Stage 3: budgeted live matrix over real Microsoft Fabric data.

Every table below was read from OneLake Delta in the user's own Fabric
workspaces and materialised locally, so the questions exercise production
data rather than seeded fixtures. Reference answers are computed with pandas,
entirely outside the library's query compiler, and never enter the agent's
inputs.

Each question carries a *hazard*: a plausible but wrong computation. Recording
the hazard value separately lets us distinguish "wrong" from "fell into the
specific trap the lesson is supposed to prevent".

Arms
    A  cold        no knowledge package
    B  declared    RLM.learn(..., declared={...}) frozen before evaluation

Usage
    python stage3_fabric_live.py --smoke
    python stage3_fabric_live.py --repetitions 3 --max-live-calls 400
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "stage3_data"
MODEL = "openai/gpt-4.1-mini"

# Real Fabric provenance, recorded so the run is reproducible.
PROVENANCE = {
    "olist_orders": "cbcb3fe8/361097a2/Tables/orders",
    "olist_order_items": "cbcb3fe8/361097a2/Tables/order_items",
    "olist_order_payments": "cbcb3fe8/361097a2/Tables/order_payments",
    "bakehouse_sales_transactions": "cbcb3fe8/550a3a96/Tables/sales_transactions",
    "callcenter_call_log": "ee90374b/b8a38cc5/Tables/call_log",
}

DOMAIN_TABLES = {
    "ecommerce": ["olist_orders", "olist_order_items", "olist_order_payments"],
    "food_retail": ["bakehouse_sales_transactions"],
    "service_ops": ["callcenter_call_log"],
}

# Source metadata. This is the escape hatch under test: it states grain and
# join hazards that no English name pattern could have revealed.
DECLARED = {
    "olist_orders": {
        "grain": ["order_id"],
        "notes": ["one row per order; order_id is unique in this table"],
    },
    "olist_order_items": {
        "grain": ["order_id", "order_item_id"],
        "notes": [
            "one row per item, not per order; joining this table to orders "
            "repeats every order-level value once per item",
            "count distinct order_id when counting orders",
        ],
    },
    "olist_order_payments": {
        "grain": ["order_id", "payment_sequential"],
        "notes": [
            "one row per payment instalment, not per order; an order can have "
            "several payment rows",
            "a per-order average must divide the summed payment_value by the "
            "number of distinct order_id, not by the number of rows",
        ],
    },
    "callcenter_call_log": {
        "grain": ["Call_ID"],
        "notes": [
            "one row per call; agents handle unequal numbers of calls, so an "
            "overall average must be computed across calls, not as the mean "
            "of per-agent averages",
        ],
    },
    "bakehouse_sales_transactions": {
        "grain": ["transactionID"],
        "notes": ["one row per transaction; totalPrice is already extended"],
    },
}


def _frames() -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(DATA / f"{name}.parquet") for name in PROVENANCE}


def references() -> dict[str, dict]:
    """Reference and hazard values, computed independently of the library."""
    f = _frames()
    orders, items = f["olist_orders"], f["olist_order_items"]
    pay, bake, calls = f["olist_order_payments"], f["bakehouse_sales_transactions"], f["callcenter_call_log"]
    top = bake.groupby("product").totalPrice.sum().sort_values(ascending=False)
    per_agent = calls.groupby("Agent_ID").Talk_Time.mean()
    return {
        "q_ecom_order_count": {
            "value": float(items.order_id.nunique()),
            "hazard": float(len(items)),
            "hazard_name": "counted item rows instead of distinct orders",
        },
        "q_ecom_avg_payment": {
            "value": round(float(pay.payment_value.sum() / pay.order_id.nunique()), 2),
            "hazard": round(float(pay.payment_value.mean()), 2),
            "hazard_name": "averaged payment rows instead of per order",
        },
        "q_retail_top_product": {
            "value": round(float(top.iloc[0]), 2),
            "entity": str(top.index[0]),
            "hazard": round(float(top.iloc[1]), 2),
            "hazard_name": "returned the runner-up product",
        },
        "q_service_avg_talk": {
            "value": round(float(calls.Talk_Time.mean()), 4),
            "hazard": round(float(per_agent.mean()), 4),
            "hazard_name": "mean of per-agent means (unweighted)",
        },
    }


QUESTIONS = [
    {
        "question_id": "q_ecom_order_count",
        "domain": "ecommerce",
        "text": "How many distinct orders have at least one order item? "
                "Report a single integer count of orders.",
    },
    {
        "question_id": "q_ecom_avg_payment",
        "domain": "ecommerce",
        "text": "What is the average total payment amount per order, across all "
                "orders that appear in the payments data? Report one number "
                "rounded to two decimals.",
    },
    {
        "question_id": "q_retail_top_product",
        "domain": "food_retail",
        "text": "Which single product generated the highest total revenue, and "
                "what was that revenue? Report the product name and the amount.",
    },
    {
        "question_id": "q_service_avg_talk",
        "domain": "service_ops",
        "text": "What is the overall average talk time across all calls in the "
                "log? Report one number rounded to four decimals.",
    },
]

TASK_SUFFIX = (
    "\n\nUse only the supplied sources. Report the grain your answer is at and "
    "the units. If a required definition is missing, abstain rather than guess. "
    "Return answer as a dictionary with keys: status, value, units, grain, and "
    "entity when a specific entity is requested."
)


def _sources_for(domain: str) -> dict[str, str]:
    return {name: str(DATA / f"{name}.csv") for name in DOMAIN_TABLES[domain]}


def ensure_csv() -> None:
    for name in PROVENANCE:
        target = DATA / f"{name}.csv"
        if not target.exists():
            pd.read_parquet(DATA / f"{name}.parquet").to_csv(target, index=False)


def make_limits():
    """Profiling limits sized for real analytics tables.

    The defaults (1 MB / 1000 records) truncate any realistic table, which
    makes the profile inexact and causes preflight to reject the package. See
    PR75_REVIEW.md F1.
    """
    from fabric_rlm.knowledge_sources import ProfileLimits

    return ProfileLimits(
        max_input_bytes=64 * 1024 * 1024,
        max_records=200_000,
        read_chunk_bytes=1024 * 1024,
    )


def make_lm():
    import dspy

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not available")
    return dspy.LM(
        model=f"openrouter/{MODEL}",
        api_base="https://openrouter.ai/api/v1",
        api_key=key,
        max_tokens=4096,
        temperature=1.0,
        cache=False,
    )


def _extract(payload) -> dict:
    if isinstance(payload, dict):
        for key in ("answer", "result", "output"):
            inner = payload.get(key)
            if isinstance(inner, dict):
                return inner
        return payload
    return {}


def _numeric(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("$", "").strip().rstrip("%")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def grade(question_id: str, answer: dict, refs: dict) -> dict:
    ref = refs[question_id]
    status = str(answer.get("status", "")).lower()
    value = _numeric(answer.get("value"))
    abstained = status in {"abstain", "abstained", "needs_definition", "unknown"}

    def close(a, b) -> bool:
        if a is None or b is None:
            return False
        return abs(a - b) <= max(0.005, abs(b) * 0.001)

    correct = close(value, ref["value"])
    hazard = close(value, ref["hazard"])
    entity_ok = None
    if "entity" in ref:
        text = f"{answer.get('entity','')} {answer.get('value','')}".lower()
        entity_ok = ref["entity"].lower() in text
        correct = correct and bool(entity_ok)
    return {
        "status": status,
        "value": value,
        "reference": ref["value"],
        "correct": bool(correct),
        "hit_hazard": bool(hazard),
        "hazard_name": ref["hazard_name"],
        "abstained": abstained,
        "entity_ok": entity_ok,
        "confidently_wrong": bool(not correct and not abstained and value is not None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-live-calls", type=int, default=400)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", default="stage3_results.json")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated question_ids to run (default: all).",
    )
    parser.add_argument(
        "--arms",
        default="A,B",
        help="Comma-separated arms to run (default: A,B).",
    )
    args = parser.parse_args()

    from fabric_rlm import RLM

    ensure_csv()
    refs = references()
    questions = QUESTIONS[:1] if args.smoke else QUESTIONS
    if args.only:
        wanted = {qid.strip() for qid in args.only.split(",") if qid.strip()}
        unknown = wanted - {q["question_id"] for q in QUESTIONS}
        if unknown:
            raise SystemExit(f"unknown question_id(s): {sorted(unknown)}")
        questions = [q for q in QUESTIONS if q["question_id"] in wanted]
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    reps = 1 if args.smoke else args.repetitions

    packages: dict[tuple[str, str], object] = {}
    summaries = {}
    for domain in {q["domain"] for q in questions}:
        sources = _sources_for(domain)
        declared = {name: DECLARED[name] for name in sources if name in DECLARED}
        learned = RLM.learn(sources=sources, declared=declared, limits=make_limits())
        packages[(domain, "A")] = None
        packages[(domain, "B")] = learned
        summaries[domain] = {
            "declared_lessons": len(learned.package.lessons),
            "active_lessons": sum(
                1 for lesson in learned.package.lessons if lesson.status == "active"
            ),
            "fingerprint": learned.package.fingerprint,
        }
        print(f"[learn] {domain}: {summaries[domain]}")

    schedule = [
        {"question_id": q["question_id"], "domain": q["domain"], "arm": arm, "rep": rep}
        for q in questions
        for arm in arms
        for rep in range(reps)
    ]
    random.Random(args.seed).shuffle(schedule)

    by_id = {q["question_id"]: q for q in questions}
    budget = args.max_live_calls
    trials = []
    for index, trial in enumerate(schedule, start=1):
        if budget <= 0:
            print("budget exhausted")
            break
        question = by_id[trial["question_id"]]
        knowledge = packages[(trial["domain"], trial["arm"])]
        inputs = _sources_for(trial["domain"]) if trial["arm"] == "A" else None
        lm = make_lm()
        started = time.perf_counter()
        record = {**trial, "model": MODEL}
        try:
            rlm = RLM.from_task(
                task=question["text"] + TASK_SUFFIX,
                inputs=dict(inputs or {}),
                outputs={"answer": dict},
                lm=lm,
                knowledge=knowledge,
                max_turns=args.max_turns,
                timeout=args.timeout,
                capture_evidence=True,
                enable_skill_autoloading=False,
                skills=[],
            )
            result = rlm.run()
            answer = _extract(result.payload)
            record.update(
                {
                    "ok": True,
                    "submitted": bool(result.submitted),
                    "failure_reason": result.failure_reason,
                    "turns": len(getattr(result.trajectory, "turns", []) or []),
                    "prompt_tokens": result.total_prompt_tokens,
                    "completion_tokens": result.total_completion_tokens,
                    "wall_seconds": round(time.perf_counter() - started, 2),
                    "answer": answer,
                    "grade": grade(question["question_id"], answer, refs),
                }
            )
            budget -= max(1, record["turns"])
        except Exception as error:
            message = str(error)
            key = os.environ.get("OPENROUTER_API_KEY")
            if key:
                message = message.replace(key, "[REDACTED]")
            record.update({"ok": False, "error": f"{type(error).__name__}: {message[:400]}"})
            budget -= 1
        trials.append(record)
        g = record.get("grade", {})
        print(
            f"[{index}/{len(schedule)}] {trial['question_id']} arm={trial['arm']} "
            f"rep={trial['rep']} ok={record.get('ok')} correct={g.get('correct')} "
            f"hazard={g.get('hit_hazard')} value={g.get('value')} budget={budget}"
        )

    report = {
        "model": MODEL,
        "seed": args.seed,
        "repetitions": reps,
        "provenance": PROVENANCE,
        "references": refs,
        "packages": summaries,
        "remaining_budget": budget,
        "trials": trials,
    }
    Path(args.output).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {args.output}; trials={len(trials)} remaining_budget={budget}")


if __name__ == "__main__":
    main()
