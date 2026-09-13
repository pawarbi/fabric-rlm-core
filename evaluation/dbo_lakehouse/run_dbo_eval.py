"""25-question live evaluation over da_agent_tests.Lakehouse/Tables/dbo.

Two arms, identical except for one argument:

  A  cold      RLM.from_task(..., knowledge=None, inputs=<10 csv sources>)
  B  learned   RLM.from_task(..., knowledge=<frozen RLM.learn package>)

Everything else -- model, temperature, max_turns, timeout, skills -- is pinned
identically, so nothing but the knowledge package can explain a difference.

The Excel workbook is rewritten after EVERY question rather than once at the
end. Q1 creates it; Q2..Q25 update it. That satisfies the requested incremental
workflow and means a crash at question 17 still leaves 16 questions of results
on disk instead of nothing.

Reference answers come from ground_truth.py, which uses pandas and DuckDB and
never imports fabric_rlm.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "cache"
CSV = HERE / "csv"
MODEL = "openai/gpt-4.1-mini"

LAKEHOUSE = ("abfss://sandeep_ws@onelake.dfs.fabric.microsoft.com/"
             "da_agent_tests.Lakehouse/Tables/dbo")

TABLES = [
    "companies", "dim_date", "features", "industries", "invoices",
    "payments", "subscriptions", "support_tickets", "usage_logs", "users",
]

# Source metadata. This is the "definitions document": grain, keys and the
# meaning of columns whose interpretation is not inferable from the name.
# It is declarative source metadata, not a domain rule baked into the library.
DECLARED = {
    "companies": {
        "grain": ["company_id"],
        "definitions": {
            "company_size": "One of startup, small, mid-market, enterprise.",
            "region": "Sales region the company belongs to.",
        },
    },
    "dim_date": {
        "grain": ["date_key"],
        "period_column": "date",
        "notes": ["Calendar spans 2021-07-01 to 2025-03-31."],
    },
    "features": {
        "grain": ["feature_id"],
        "definitions": {
            "is_premium": "1 marks a premium feature, 0 a standard one.",
            "module": "Coarser grouping of features.",
        },
    },
    "industries": {
        "grain": ["industry_id"],
        "definitions": {"sector": "Coarser grouping than industry_name."},
    },
    "invoices": {
        "grain": ["invoice_id"],
        "period_column": "invoice_date",
        "units": {"amount_due": "USD", "amount_paid": "USD"},
        "definitions": {
            "amount_due": "Amount billed on the invoice.",
            "amount_paid": "Settled amount recorded on the invoice row itself. "
                           "This is not the same as summing the payments table.",
            "status": "One of paid, void, overdue, sent.",
        },
        "notes": [
            "An invoice may have zero, one or many matching rows in payments. "
            "Joining invoices to payments multiplies invoice rows, so summing "
            "amount_due after that join double-counts.",
        ],
    },
    "payments": {
        "grain": ["payment_id"],
        "period_column": "payment_date",
        "units": {"amount": "USD"},
        "definitions": {
            "amount": "Cash actually collected in this transaction.",
            "method": "One of ach, bank_transfer, credit_card, wire.",
        },
        "notes": [
            "Many payments may settle one invoice: 9,204 payments cover 7,584 "
            "distinct invoices.",
        ],
    },
    "subscriptions": {
        "grain": ["sub_id"],
        "period_column": "start_date",
        "units": {"mrr": "USD per month"},
        "definitions": {
            "mrr": "Monthly recurring revenue for this subscription.",
            "status": "One of active, cancelled, expired. cancelled and expired "
                      "are distinct outcomes and must not be merged.",
            "end_date": "NULL for subscriptions that have not ended.",
            "annual_contract": "1 if the subscription is on an annual contract.",
        },
        "notes": ["A company may hold several subscriptions."],
    },
    "support_tickets": {
        "grain": ["ticket_id"],
        "period_column": "created_at",
        "definitions": {
            "resolved_at": "NULL for tickets never resolved (678 of 4,192).",
            "satisfaction_score": "1-5, NULL when the ticket was not resolved. "
                                  "Exclude nulls from means; do not treat as zero.",
            "first_response_at": "Populated for every ticket.",
        },
    },
    "usage_logs": {
        "grain": ["log_id"],
        "period_column": "usage_date",
        "units": {"compute_minutes": "minutes", "api_calls": "calls"},
        "notes": [
            "661,734 rows: too large to read exhaustively. Aggregate rather "
            "than inspect row by row.",
        ],
    },
    "users": {
        "grain": ["user_id"],
        "period_column": "created_at",
        "definitions": {
            "is_active": "Stored activity flag. NOT equivalent to having "
                         "appeared in usage_logs.",
        },
    },
}

TASK_SUFFIX = (
    "\n\nUse only the supplied sources. State the grain and units of your answer. "
    "If a required definition is genuinely missing, say so rather than guessing. "
    "Return answer as a dictionary with keys: status, value, units, grain, and "
    "sql (the query or pandas expression you used), and reasoning (one or two "
    "sentences on how you computed it). Put ONLY the bare number or the bare "
    "entity name in value -- no commentary, no units, no formatting. "
    "value must never be null: if you cannot compute it, set status to "
    "\"abstain\" and put a short phrase in value naming what is missing."
)


def require_value(payload) -> None:
    """Reject a SUBMIT whose answer.value is missing or null.

    Without this, a run that describes an approach instead of computing it is
    recorded as a success with ``failure_reason: None`` -- it grades wrong, but
    silently, and the trial is spent. Rejecting it costs one re-prompt inside
    the same run and keeps the grade attributable to reasoning rather than to
    an unenforced output contract.
    """
    answer = payload.get("answer") if isinstance(payload, dict) else None
    assert isinstance(answer, dict), (
        "SUBMIT payload must contain 'answer' as a dictionary with keys: "
        "status, value, units, grain, sql, reasoning."
    )
    assert answer.get("value") is not None, (
        "answer.value is missing or null. Compute the value and submit it. "
        "If it genuinely cannot be computed, set status to 'abstain' and put "
        "a short phrase in value naming the missing definition or data."
    )


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
def ensure_csv() -> dict[str, str]:
    CSV.mkdir(exist_ok=True)
    out = {}
    for name in TABLES:
        target = CSV / f"{name}.csv"
        if not target.exists():
            pd.read_parquet(CACHE / f"{name}.parquet").to_csv(target, index=False)
        out[name] = str(target)
    return out


def make_limits():
    """Sized for these tables. Defaults (1 MB / 1000 rows) truncate usage_logs,
    which makes the profile inexact and causes preflight to reject the package.
    """
    from fabric_rlm.knowledge_sources import ProfileLimits

    return ProfileLimits(
        max_input_bytes=256 * 1024 * 1024,
        max_records=700_000,
        read_chunk_bytes=4 * 1024 * 1024,
    )


def make_lm():
    import dspy

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return dspy.LM(
        model=f"openrouter/{MODEL}",
        api_base="https://openrouter.ai/api/v1",
        api_key=key,
        max_tokens=4096,
        temperature=1.0,
        cache=False,
    )


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------
_NP_KEYS = {"__type__", "__repr__", "__serializable__"}


def _unwrap(v):
    """Recover a number the runtime failed to serialise (np.int64(6642) etc)."""
    if isinstance(v, dict) and _NP_KEYS & set(v):
        import re

        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(v.get("__repr__", "")))
        if m:
            return float(m.group())
    return v


def _leaves(v, depth=0):
    """Every numeric value reachable in a nested answer payload."""
    out = []
    v = _unwrap(v)
    if isinstance(v, bool):
        return out
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, str):
        import re

        for m in re.finditer(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", v):
            try:
                out.append(float(m.group().replace(",", "")))
            except ValueError:
                pass
        return out
    if depth > 4:
        return out
    if isinstance(v, dict):
        for x in v.values():
            out.extend(_leaves(x, depth + 1))
    elif isinstance(v, (list, tuple)):
        for x in v:
            out.extend(_leaves(x, depth + 1))
    return out


def _strings(v, depth=0):
    out = []
    if isinstance(v, str):
        return [v]
    if depth > 4:
        return out
    if isinstance(v, dict):
        for x in v.values():
            out.extend(_strings(x, depth + 1))
    elif isinstance(v, (list, tuple)):
        for x in v:
            out.extend(_strings(x, depth + 1))
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(abs(b) * 0.005, 0.005)


def grade(answer, ref, hazard) -> dict:
    """Two axes, kept separate.

    contract  -- the bare value under key 'value' is right (what was asked for)
    analytic  -- the right number/entity appears anywhere in the answer
    """
    if not isinstance(answer, dict):
        answer = {"value": answer}

    raw = _unwrap(answer.get("value"))
    numeric_ref = isinstance(ref, (int, float)) and not isinstance(ref, bool)

    contract = False
    if numeric_ref:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            contract = _close(float(raw), float(ref))
        elif isinstance(raw, str):
            nums = _leaves(raw)
            contract = len(nums) == 1 and _close(nums[0], float(ref))
    else:
        contract = isinstance(raw, str) and str(ref).lower() in raw.lower()

    if numeric_ref:
        leaves = _leaves(answer)
        analytic = any(_close(x, float(ref)) for x in leaves)
        haz = any(_close(x, float(hazard)) for x in leaves) if isinstance(
            hazard, (int, float)) else False
    else:
        blob = " | ".join(_strings(answer)).lower()
        analytic = str(ref).lower() in blob
        haz = (str(hazard).lower() in blob) and not analytic

    status = str(answer.get("status", "")).lower()
    abstained = any(w in status for w in ("abstain", "insufficient", "unknown"))

    return {
        "contract_correct": bool(contract),
        "analytic_correct": bool(analytic),
        "hit_hazard": bool(haz),
        "abstained": bool(abstained),
        "machine_readable": isinstance(raw, (int, float, str)) and raw is not None,
        "value": raw if isinstance(raw, (int, float, str)) else repr(raw)[:120],
    }


# --------------------------------------------------------------------------
# Excel -- rewritten after every question
# --------------------------------------------------------------------------
def write_workbook(path: Path, refs: list[dict], results: list[dict],
                   profile: dict, learn_summary: dict) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    HDR_FILL = PatternFill("solid", fgColor="1F3864")
    HDR_FONT = Font(bold=True, color="FFFFFF", size=11)
    OK_FILL = PatternFill("solid", fgColor="C6EFCE")
    BAD_FILL = PatternFill("solid", fgColor="FFC7CE")
    WARN_FILL = PatternFill("solid", fgColor="FFEB9C")

    wb = Workbook()

    def style_header(ws, ncols):
        for c in range(1, ncols + 1):
            cell = ws.cell(row=1, column=c)
            cell.fill = HDR_FILL
            cell.font = HDR_FONT
            cell.alignment = Alignment(vertical="center", horizontal="left")
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

    def autosize(ws, widths):
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w

    # ---- Sheet 1: data used -------------------------------------------
    ws = wb.active
    ws.title = "1. Data Used"
    cols = ["Table", "Rows", "Columns", "Column names", "Declared grain",
            "Period column", "Units", "Definitions", "Notes", "OneLake source"]
    ws.append(cols)
    for name in TABLES:
        p = profile.get(name, {})
        colnames = ", ".join(c["name"] for c in p.get("columns", []))
        dec = DECLARED.get(name, {})
        defs = "; ".join(f"{k}: {v}" for k, v in (dec.get("definitions") or {}).items())
        units = "; ".join(f"{k}={v}" for k, v in (dec.get("units") or {}).items())
        ws.append([
            name,
            p.get("rows", ""),
            len(p.get("columns", [])),
            colnames,
            ", ".join(dec.get("grain", [])),
            dec.get("period_column", ""),
            units,
            defs,
            " ".join(dec.get("notes", [])),
            f"{LAKEHOUSE}/{name}",
        ])
    style_header(ws, len(cols))
    autosize(ws, [22, 10, 9, 54, 16, 15, 22, 66, 60, 70])
    for r in range(2, ws.max_row + 1):
        for c in (4, 8, 9):
            ws.cell(row=r, column=c).alignment = Alignment(wrap_text=True,
                                                            vertical="top")

    row = ws.max_row + 2
    ws.cell(row=row, column=1, value="Learned package (arm B)").font = Font(bold=True)
    for k, v in (learn_summary or {}).items():
        row += 1
        ws.cell(row=row, column=1, value=str(k))
        ws.cell(row=row, column=2, value=str(v))

    # ---- Sheet 2: answers ---------------------------------------------
    ws2 = wb.create_sheet("2. Questions & Answers")
    cols2 = ["Q", "Question", "Reference answer", "Hazard (wrong path)",
             "Arm A value", "A correct", "Arm B value", "B correct",
             "A turns", "B turns", "SQL used (arm B)", "Reasoning (arm B)",
             "Reference SQL", "Comments"]
    ws2.append(cols2)

    by_q = {}
    for r in results:
        by_q.setdefault(r["question_id"], {})[r["arm"]] = r

    for ref in refs:
        qid = ref["id"]
        a = by_q.get(qid, {}).get("A", {})
        b = by_q.get(qid, {}).get("B", {})
        ga, gb = a.get("grade", {}), b.get("grade", {})
        ans_b = b.get("answer") or {}

        comment = []
        if a and not a.get("ok"):
            comment.append(f"A error: {a.get('error', '')[:80]}")
        if b and not b.get("ok"):
            comment.append(f"B error: {b.get('error', '')[:80]}")
        if ga.get("hit_hazard"):
            comment.append("A hit the named hazard")
        if gb.get("hit_hazard"):
            comment.append("B hit the named hazard")
        if ga.get("analytic_correct") and not ga.get("contract_correct"):
            comment.append("A right but broke the output contract")
        if gb.get("analytic_correct") and not gb.get("contract_correct"):
            comment.append("B right but broke the output contract")
        if not comment:
            comment.append(ref["hazard_note"])

        ws2.append([
            qid,
            ref["question"],
            str(ref["reference"]),
            str(ref["hazard"]),
            str(ga.get("value", "")),
            "YES" if ga.get("analytic_correct") else ("" if not a else "no"),
            str(gb.get("value", "")),
            "YES" if gb.get("analytic_correct") else ("" if not b else "no"),
            a.get("turns", ""),
            b.get("turns", ""),
            str(ans_b.get("sql", ""))[:900] if isinstance(ans_b, dict) else "",
            str(ans_b.get("reasoning", ""))[:900] if isinstance(ans_b, dict) else "",
            ref["sql"],
            "; ".join(comment),
        ])

    style_header(ws2, len(cols2))
    autosize(ws2, [6, 62, 20, 20, 20, 10, 20, 10, 9, 9, 56, 56, 62, 46])
    for r in range(2, ws2.max_row + 1):
        for c in (2, 11, 12, 13, 14):
            ws2.cell(row=r, column=c).alignment = Alignment(wrap_text=True,
                                                            vertical="top")
        for c, gcell in ((6, 5), (8, 7)):
            v = ws2.cell(row=r, column=c).value
            if v == "YES":
                ws2.cell(row=r, column=c).fill = OK_FILL
            elif v == "no":
                ws2.cell(row=r, column=c).fill = BAD_FILL

    # ---- Sheet 3: evidence --------------------------------------------
    ws3 = wb.create_sheet("3. Evidence")
    cols3 = ["Q", "Arm", "Rep", "Status", "Submitted", "Turns",
             "Prompt tokens", "Completion tokens", "Wall seconds",
             "Answer status", "Grain reported", "Units reported",
             "Contract ok", "Analytic ok", "Hazard", "Abstained",
             "Full answer payload", "Error"]
    ws3.append(cols3)
    for r in results:
        g = r.get("grade", {})
        ans = r.get("answer") or {}
        ws3.append([
            r["question_id"], r["arm"], r.get("rep", 0),
            "ok" if r.get("ok") else "FAILED",
            str(r.get("submitted", "")),
            r.get("turns", ""), r.get("prompt_tokens", ""),
            r.get("completion_tokens", ""), r.get("wall_seconds", ""),
            str(ans.get("status", "")) if isinstance(ans, dict) else "",
            str(ans.get("grain", "")) if isinstance(ans, dict) else "",
            str(ans.get("units", "")) if isinstance(ans, dict) else "",
            str(g.get("contract_correct", "")),
            str(g.get("analytic_correct", "")),
            str(g.get("hit_hazard", "")),
            str(g.get("abstained", "")),
            json.dumps(ans, default=str)[:1500],
            str(r.get("error", ""))[:300],
        ])
    style_header(ws3, len(cols3))
    autosize(ws3, [6, 6, 6, 9, 11, 8, 14, 17, 13, 14, 30, 16, 12, 12, 9, 10, 90, 40])
    for r in range(2, ws3.max_row + 1):
        ws3.cell(row=r, column=17).alignment = Alignment(wrap_text=True,
                                                          vertical="top")
        if ws3.cell(row=r, column=4).value == "FAILED":
            ws3.cell(row=r, column=4).fill = BAD_FILL
        if ws3.cell(row=r, column=15).value == "True":
            ws3.cell(row=r, column=15).fill = WARN_FILL

    tmp = path.with_suffix(".tmp.xlsx")
    wb.save(tmp)
    tmp.replace(path)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 questions only")
    ap.add_argument("--only", default="")
    ap.add_argument("--arms", default="A,B")
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-live-calls", type=int, default=800)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", default="dbo_eval_results.json")
    ap.add_argument("--workbook", default="dbo_eval_report.xlsx")
    args = ap.parse_args()

    from fabric_rlm import RLM
    import ground_truth

    refs = ground_truth.build()
    bad = [r for r in refs if not r["engines_agree"] or not r["discriminating"]]
    if bad:
        raise SystemExit(f"ground truth not clean: {[r['id'] for r in bad]}")

    profile = json.loads((HERE / "profile.json").read_text(encoding="utf-8"))
    sources = ensure_csv()

    if args.smoke:
        refs = refs[:3]
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        refs = [r for r in refs if r["id"] in want]
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    wb_path = HERE / args.workbook
    out_path = HERE / args.output

    learn_summary = {}
    learned = None
    if "B" in arms or "C" in arms:
        t0 = time.perf_counter()
        learned = RLM.learn(sources=sources, declared=DECLARED, limits=make_limits())
        pkg = learned.package
        learn_summary = {
            "lessons_total": len(pkg.lessons),
            "lessons_active": sum(1 for x in pkg.lessons if x.status == "active"),
            "fingerprint": pkg.fingerprint,
            "learn_seconds": round(time.perf_counter() - t0, 1),
            "sources": len(sources),
        }
        print(f"[learn] {learn_summary}", flush=True)

    schedule = [
        {"question_id": r["id"], "arm": arm, "rep": rep}
        for r in refs for arm in arms for rep in range(args.repetitions)
    ]
    # Group by question so the workbook fills in question order, while arms
    # within a question stay adjacent. Shuffle only the arm order per question
    # so neither arm systematically runs first.
    rng = random.Random(args.seed)
    ordered = []
    for r in refs:
        block = [s for s in schedule if s["question_id"] == r["id"]]
        rng.shuffle(block)
        ordered.extend(block)

    by_id = {r["id"]: r for r in refs}
    results = []
    budget = args.max_live_calls

    for i, trial in enumerate(ordered, start=1):
        if budget <= 0:
            print("budget exhausted", flush=True)
            break
        ref = by_id[trial["question_id"]]
        # Arm A: sources only, no package.
        # Arm B: package only, no direct sources (the library's intended shape).
        # Arm C: BOTH -- the control that separates "learning misleads" from
        #        "arm B simply had less data access than arm A".
        knowledge = learned if trial["arm"] in ("B", "C") else None
        inputs = {} if trial["arm"] == "B" else dict(sources)
        rec = {**trial, "model": MODEL}
        started = time.perf_counter()
        try:
            rlm = RLM.from_task(
                task=ref["question"] + TASK_SUFFIX,
                inputs=inputs,
                outputs={"answer": dict},
                lm=make_lm(),
                knowledge=knowledge,
                max_turns=args.max_turns,
                timeout=args.timeout,
                capture_evidence=True,
                enable_skill_autoloading=False,
                skills=[],
                output_validator=require_value,
            )
            res = rlm.run()
            payload = res.payload
            ans = payload.get("answer") if isinstance(payload, dict) else payload
            if not isinstance(ans, dict):
                ans = {"value": ans}
            rec.update({
                "ok": True,
                "submitted": bool(res.submitted),
                "failure_reason": res.failure_reason,
                "turns": len(getattr(res.trajectory, "turns", []) or []),
                "prompt_tokens": res.total_prompt_tokens,
                "completion_tokens": res.total_completion_tokens,
                "wall_seconds": round(time.perf_counter() - started, 2),
                "answer": ans,
                "grade": grade(ans, ref["reference"], ref["hazard"]),
            })
            budget -= max(1, rec["turns"])
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            key = os.environ.get("OPENROUTER_API_KEY")
            if key:
                msg = msg.replace(key, "[REDACTED]")
            rec.update({"ok": False,
                        "error": f"{type(exc).__name__}: {msg[:300]}",
                        "wall_seconds": round(time.perf_counter() - started, 2)})
            budget -= 1

        results.append(rec)
        g = rec.get("grade", {})
        print(f"[{i}/{len(ordered)}] {trial['question_id']} arm={trial['arm']} "
              f"ok={rec.get('ok')} contract={g.get('contract_correct')} "
              f"analytic={g.get('analytic_correct')} hz={g.get('hit_hazard')} "
              f"turns={rec.get('turns')} val={g.get('value')} budget={budget}",
              flush=True)

        # Incremental: Q1 creates the workbook, every later question updates it.
        write_workbook(wb_path, refs, results, profile, learn_summary)
        out_path.write_text(json.dumps(
            {"model": MODEL, "learn": learn_summary, "trials": results},
            indent=2, default=str), encoding="utf-8")

    print(f"\nworkbook: {wb_path}")
    print(f"raw:      {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
