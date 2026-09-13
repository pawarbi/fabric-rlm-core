"""Does the system notice when a required definition is missing?

The evaluation's definitions document does more than name fields. It carries
the rules that disambiguate each question, and several of those rules are the
only thing standing between the model and a hazard deliberately built into the
fixtures:

    "customer identity": "Use customer_id; customer_name is not unique."
    "inventory value":   "... on_hand_units multiplied once by unit_cost."
    "available units":   "At the latest snapshot only, ..."

The fixtures contain repeated customer names, a join that multiplies values,
and multiple snapshots. So removing one definition does not merely make a
question harder -- it makes it *ambiguous in a way the data will not reveal*,
and it walks the model toward a specific wrong answer.

The expected behaviour, stated in the task text the runner sends, is to return
``status='needs_definition'`` or ``'abstain'``. Anything else is graded on what
it did instead, and the interesting outcome is not merely "wrong" but which of
these it was:

``abstained``        named the gap or declined -- the wanted behaviour
``guessed_wrong``    confidently produced the hazard's wrong answer
``guessed_right``    confidently produced the right answer anyway

``guessed_right`` is not a pass. Reaching the documented answer without the
document means the rule was assumed, and the same assumption applied to a
source whose convention differs would be silently wrong. It is reported
separately rather than folded into either bucket.

Run:

    python -m evaluation.generalization.insufficient_metadata \
        --fixtures evaluation/generalization/generated \
        --output evaluation/generalization/raw-results/insufficient-metadata.json \
        --repetitions 2 --max-live-calls 30
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .grader import values_match
from .runner import (
    DEFAULT_MODEL,
    _domain_sources,
    _require_value,
    make_openrouter_lm,
    normalize_answer,
    result_metrics,
    _provider_cost,
)

SEED = 20260908

ABSTAIN_STATUSES = frozenset({
    "abstain", "needs_definition", "uncertain", "insufficient_metadata",
    "needs_clarification", "unknown",
})


def strip_definitions(definitions: Mapping[str, Any], domain: str,
                      variant: str, drop: list[str]) -> dict[str, Any]:
    """Return the definitions document with ``drop`` removed for one domain.

    Field mappings are left intact on purpose. The point is a missing *rule*,
    not a missing schema: the model can still see and read every column, it
    simply has not been told which reading is intended. A test that also hid
    the fields would be measuring something easier.
    """
    doc = json.loads(json.dumps(definitions))
    block = doc["variants"][variant][domain]
    removed = [name for name in drop if name in block["definitions"]]
    for name in removed:
        del block["definitions"][name]
    return {"doc": doc, "removed": removed}


def _task_text_without(question: Mapping[str, Any], definitions: Mapping[str, Any],
                       variant: str) -> str:
    """The runner's task text, verbatim, over a reduced definitions document.

    Kept identical to ``runner._task_text`` so the only difference between this
    probe and the main matrix is the content of the definitions block. It still
    tells the model that abstaining is available, which is what makes silence
    a finding rather than an unfair trick.
    """
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


def classify(answer: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    status = answer.get("status")
    status = status.strip().lower() if isinstance(status, str) else None

    if status in {"timeout", "failed"} or not status:
        return {"outcome": "incomplete", "abstained": False}
    if status in ABSTAIN_STATUSES:
        return {"outcome": "abstained", "abstained": True}

    value = answer.get("value")
    if value is None:
        return {"outcome": "incomplete", "abstained": False}

    # An explicit uncertainty flag alongside a value still counts as naming the
    # gap: the caller is told the answer rests on an assumption.
    flagged = bool(answer.get("assumptions") or answer.get("caveats"))
    unsupported = [c for c in answer.get("claims", ())
                   if isinstance(c, Mapping) and c.get("supported") is False]
    if flagged or unsupported:
        return {"outcome": "abstained", "abstained": True,
                "note": "answered but flagged the assumption"}

    right = values_match(value, reference.get("value"))
    return {"outcome": "guessed_right" if right else "guessed_wrong",
            "abstained": False, "value": value,
            "reference_value": reference.get("value")}


def run(fixtures: Path, output: Path, *, model: str, repetitions: int,
        max_live_calls: int, max_turns: int, timeout: float,
        variant: str = "descriptive") -> dict[str, Any]:
    from fabric_rlm import RLM

    definitions = json.loads((fixtures / "definitions.json").read_text("utf-8"))
    questions = json.loads((fixtures / "questions.json").read_text("utf-8"))
    references = json.loads(
        (fixtures / "private" / "references.json").read_text("utf-8"))

    cases = [q for q in questions if q.get("expected_behavior") == "answer"
             and q.get("provided_definitions")]

    schedule = [(q, rep) for q in cases for rep in range(repetitions)]
    random.Random(SEED).shuffle(schedule)

    trials: list[dict[str, Any]] = []
    budget = max_live_calls
    for question, rep in schedule:
        if budget <= 0:
            break
        domain = str(question["domain"])
        qid = str(question["question_id"])
        reduced = strip_definitions(definitions, domain, variant,
                                    list(question["provided_definitions"]))
        if not reduced["removed"]:
            continue

        record: dict[str, Any] = {
            "question_id": qid, "domain": domain, "repetition": rep,
            "removed_definitions": reduced["removed"], "model": model,
        }
        started = time.perf_counter()
        try:
            lm = make_openrouter_lm(model)
            rlm = RLM.from_task(
                task=_task_text_without(question, reduced["doc"], variant),
                inputs=_domain_sources(fixtures, domain, variant),
                outputs={"answer": dict},
                lm=lm,
                knowledge=None,
                max_turns=max_turns,
                timeout=timeout,
                capture_evidence=True,
                enable_skill_autoloading=False,
                skills=[],
                output_validator=_require_value,
            )
            result = rlm.run()
            wall = time.perf_counter() - started
            payload = getattr(result, "payload", None)
            answer = normalize_answer(
                (payload or {}).get("answer") if isinstance(payload, Mapping) else payload,
                submitted=bool(getattr(result, "submitted", False)),
                failure_reason=getattr(result, "failure_reason", None),
            )
            record["answer"] = answer
            record.update(result_metrics(result, wall_seconds=wall,
                                         provider_cost_usd=_provider_cost(lm)))
            record["classification"] = classify(
                answer, references[domain][variant].get(qid, {}))
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            record["classification"] = {"outcome": "error", "abstained": False}
        budget -= 1
        trials.append(record)

    counts: dict[str, int] = {}
    by_domain: dict[str, dict[str, int]] = {}
    for t in trials:
        outcome = t["classification"]["outcome"]
        counts[outcome] = counts.get(outcome, 0) + 1
        by_domain.setdefault(t["domain"], {})
        by_domain[t["domain"]][outcome] = by_domain[t["domain"]].get(outcome, 0) + 1

    report = {
        "model": model, "variant": variant, "repetitions": repetitions,
        "seed": SEED, "trials": trials, "counts": counts,
        "by_domain": by_domain,
        "abstention_rate": (
            sum(1 for t in trials if t["classification"]["abstained"]) / len(trials)
            if trials else None
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--repetitions", type=int, default=2)
    ap.add_argument("--max-live-calls", type=int, default=30)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args(argv)

    report = run(args.fixtures, args.output, model=args.model,
                 repetitions=args.repetitions, max_live_calls=args.max_live_calls,
                 max_turns=args.max_turns, timeout=args.timeout)

    print(f"trials {len(report['trials'])}   "
          f"abstention rate {report['abstention_rate']}")
    for outcome, n in sorted(report["counts"].items()):
        print(f"  {outcome:16s} {n}")
    print("\nby domain")
    for domain, c in sorted(report["by_domain"].items()):
        print(f"  {domain:15s} {c}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
