from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SKILLS = ("excel_extract", "data_exploration")

EXACT_LOOKUPS = (
    {
        "cdid": "D7BT",
        "period": "2024 MAR",
        "value": 133.0,
        "title": "CPI INDEX 00: ALL ITEMS 2015=100",
    },
    {
        "cdid": "D7G7",
        "period": "2024 MAR",
        "value": 3.2,
        "title": "CPI ANNUAL RATE 00: ALL ITEMS 2015=100",
    },
    {
        "cdid": "CHAW",
        "period": "2024 MAR",
        "value": 383.0,
        "title": "RPI All Items Index: Jan 1987=100",
    },
    {
        "cdid": "L522",
        "period": "2024 MAR",
        "value": 131.6,
        "title": "CPIH INDEX 00: ALL ITEMS 2015=100",
    },
    {
        "cdid": "ZPTX",
        "period": "2025 JAN",
        "value": 1954.0,
        "title": "RPI: Ave price - Salmon fillets, per Kg",
    },
)

REASONING_QUESTIONS = (
    {
        "id": "q1_cpi_index_mar24",
        "question": "What was the headline CPI All Items index value in March 2024? (CPI, base 2015=100)",
        "expected_value": 133.0,
        "expected_title_contains": "CPI INDEX 00: ALL ITEMS 2015=100",
        "expected_cdid": "D7BT",
    },
    {
        "id": "q2_cpi_annual_rate_mar24",
        "question": "What was the annual CPI inflation rate (year-over-year %) for All Items in March 2024?",
        "expected_value": 3.2,
        "expected_title_contains": "CPI ANNUAL RATE 00: ALL ITEMS",
        "expected_cdid": "D7G7",
    },
    {
        "id": "q3_rpi_all_items_mar24",
        "question": "What was the RPI All Items Index in March 2024? (RPI, January 1987 = 100)",
        "expected_value": 383.0,
        "expected_title_contains": "RPI All Items Index: Jan 1987=100",
        "expected_cdid": "CHAW",
    },
    {
        "id": "q4_cpih_index_mar24",
        "question": "What was the CPIH All Items index in March 2024? (CPIH, base 2015=100)",
        "expected_value": 131.6,
        "expected_title_contains": "CPIH INDEX 00: ALL ITEMS",
        "expected_cdid": "L522",
    },
    {
        "id": "q5_salmon_price_jan25",
        "question": "What was the average retail price of salmon fillets per kilogram in January 2025?",
        "expected_value": 1954.0,
        "expected_title_contains": "Salmon fillets",
        "expected_cdid": "ZPTX",
    },
)


def build_exact_lookup_task() -> str:
    lookups = "\n".join(
        f"  - CDID={row['cdid']}, period={row['period']!r}" for row in EXACT_LOOKUPS
    )
    return f"""\
Extract specific values from a large Excel workbook.

FILE: {{workbook_path}}

Workbook layout:
- one sheet named data
- row 1 = human-readable series title
- row 2 = CDID code
- rows 3-7 = metadata
- row 8 onward = data
- column A holds the period label, such as '2024 MAR'

For each lookup, find the column whose row-2 cell equals the CDID, then the row
whose column-A cell equals the period. Return the value at that intersection.

Lookups:
{lookups}

Return one JSON object via SUBMIT with this exact shape:
{{
  "results": [
    {{"cdid": "...", "period": "...", "value": <number|string|null>}}
  ]
}}
"""


def build_reasoning_task() -> str:
    questions = "\n".join(
        f"  {idx}. [{row['id']}] {row['question']}"
        for idx, row in enumerate(REASONING_QUESTIONS, start=1)
    )
    return f"""\
You are given the UK ONS Consumer Price Inflation MM23 release as an Excel file.

FILE: {{workbook_path}}

Workbook layout:
- one sheet named data
- row 1 = human-readable series title
- row 2 = CDID code
- rows 3-7 = metadata
- row 8 onward = data
- column A holds the period label, such as '2024 MAR'
- there are about 4,053 series and 1,484 period rows

For each plain-English question, find the matching row-1 series title and row-2
CDID, disambiguating CPI vs CPIH vs RPI and index vs rate where needed. Then
look up the period mentioned in the question.

Questions:
{questions}

Return one JSON object via SUBMIT with this exact shape:
{{
  "answers": [
    {{
      "id": "<question id>",
      "matched_title": "<row-1 title>",
      "matched_cdid": "<row-2 CDID>",
      "period": "<column-A period label>",
      "value": <number|string|null>
    }}
  ]
}}
"""


def normalize_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _numeric_equal(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) < 0.01
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()


def score_exact_lookup_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("results") or []
    by_key = {
        (str(row.get("cdid", "")).upper(), row.get("period")): row.get("value")
        for row in rows
        if isinstance(row, dict)
    }
    details = []
    n_value_ok = 0
    for expected in EXACT_LOOKUPS:
        key = (expected["cdid"], expected["period"])
        got = by_key.get(key)
        value_ok = _numeric_equal(got, expected["value"])
        n_value_ok += int(value_ok)
        details.append(
            {
                "cdid": expected["cdid"],
                "period": expected["period"],
                "expected_value": expected["value"],
                "got_value": got,
                "value_ok": value_ok,
            }
        )
    return {
        "n_value_ok": n_value_ok,
        "n_total": len(EXACT_LOOKUPS),
        "details": details,
    }


def score_reasoning_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("answers") or []
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    details = []
    n_value_ok = 0
    n_title_ok = 0
    n_cdid_ok = 0
    for expected in REASONING_QUESTIONS:
        got = by_id.get(expected["id"]) or {}
        got_title = str(got.get("matched_title") or "")
        got_cdid = str(got.get("matched_cdid") or "").strip().upper()
        value_ok = _numeric_equal(got.get("value"), expected["expected_value"])
        title_ok = expected["expected_title_contains"].lower() in got_title.lower()
        cdid_ok = got_cdid == expected["expected_cdid"]
        n_value_ok += int(value_ok)
        n_title_ok += int(title_ok)
        n_cdid_ok += int(cdid_ok)
        details.append(
            {
                "id": expected["id"],
                "question": expected["question"],
                "expected_value": expected["expected_value"],
                "expected_cdid": expected["expected_cdid"],
                "got_value": got.get("value"),
                "got_cdid": got_cdid,
                "got_title": got_title,
                "value_ok": value_ok,
                "title_ok": title_ok,
                "cdid_ok": cdid_ok,
            }
        )
    return {
        "n_value_ok": n_value_ok,
        "n_title_ok": n_title_ok,
        "n_cdid_ok": n_cdid_ok,
        "n_total": len(REASONING_QUESTIONS),
        "details": details,
    }


def rlm_kwargs(task_text: str, *, lm: Any, max_turns: int, timeout: float) -> dict[str, Any]:
    return {
        "task": task_text,
        "outputs": ["payload_json"],
        "lm": lm,
        "skills": list(DEFAULT_SKILLS),
        "engine": "default",
        "max_turns": max_turns,
        "verbose": True,
        "timeout": timeout,
    }


def _extract_payload(result: Any) -> dict[str, Any]:
    outputs = getattr(result, "outputs", None) or getattr(result, "result", None)
    if isinstance(outputs, dict):
        return normalize_payload(outputs.get("payload_json", next(iter(outputs.values()), None)))
    return normalize_payload(outputs)


def run_rlm(
    workbook: Path,
    *,
    task: str,
    model: str,
    api_base: str,
    max_turns: int,
    timeout: float,
) -> dict[str, Any]:
    import dspy
    from fabric_rlm import RLM

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    task_text = build_exact_lookup_task() if task == "exact" else build_reasoning_task()
    lm = dspy.LM(
        model,
        api_key=api_key,
        api_base=api_base,
        max_tokens=4000,
        temperature=0,
    )
    dspy.configure(lm=lm)
    rlm = RLM.from_task(**rlm_kwargs(task_text, lm=lm, max_turns=max_turns, timeout=timeout))
    result = rlm.run({"workbook_path": str(workbook)})
    payload = _extract_payload(result)
    score = score_exact_lookup_payload(payload) if task == "exact" else score_reasoning_payload(payload)
    return {
        "task": task,
        "model": model,
        "skills": list(DEFAULT_SKILLS),
        "payload": payload,
        "score": score,
        "trajectory": str(getattr(result, "trajectory", "")),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONS CPI MM23 tasks through fabric-rlm.")
    parser.add_argument("--workbook", required=True, type=Path, help="Path to mm23.xlsx.")
    parser.add_argument("--task", choices=["exact", "reasoning"], default="reasoning")
    parser.add_argument("--model", default="openrouter/openai/gpt-4.1")
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output-dir", type=Path, default=Path("_local") / "ons_cpi_rlm_benchmark")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.workbook.exists():
        raise FileNotFoundError(args.workbook)

    record = run_rlm(
        args.workbook,
        task=args.task,
        model=args.model,
        api_base=args.api_base,
        max_turns=args.max_turns,
        timeout=args.timeout,
    )
    out = args.output_dir / args.task
    out.mkdir(parents=True, exist_ok=True)
    (out / "payload.json").write_text(json.dumps(record["payload"], indent=2), encoding="utf-8")
    (out / "score.json").write_text(json.dumps(record["score"], indent=2), encoding="utf-8")
    (out / "trajectory.txt").write_text(record["trajectory"], encoding="utf-8")
    print(json.dumps({"task": record["task"], "skills": record["skills"], "score": record["score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
