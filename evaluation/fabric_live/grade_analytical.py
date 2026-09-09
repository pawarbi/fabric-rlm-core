"""Grade RLM's analytical answers from the run log, independently of the workbook.

The workbook reset at q16 destroyed rows q01-q15, but the host recorded each
answer payload as it was returned. Grading those separates two distinct
failures: whether RLM ANALYSED correctly, and whether RLM MAINTAINED the
workbook correctly. Conflating them would misattribute the workbook defect to
analytical capability.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_gt = json.loads((HERE.parent / "dbo_eval" / "ground_truth.json").read_text())
GT = {r["id"]: r for r in _gt if r["id"] != "q23"}

_args = sys.argv[1:]
LOG_PATH = HERE / (_args[0] if _args else "run_log_cold_full.json")
OUT_PATH = HERE / (_args[1] if len(_args) > 1 else "analytical_grade.json")
_raw = LOG_PATH.read_text(encoding="utf-8", errors="replace")
LOG = json.loads(_raw[_raw.find("["):])

TOL = 0.005


def to_float(v):
    if isinstance(v, (int, float)):
        return float(v)
    if v is None:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", str(v).replace("$", ""))
    return float(m.group().replace(",", "")) if m else None


rows = []
for rec in LOG:
    qid = rec["id"]
    ref = GT[qid]["reference"]
    hazard = GT[qid].get("hazard")
    ans = rec.get("answer") or {}
    val = ans.get("value") if isinstance(ans, dict) else ans

    if isinstance(ref, str):
        text = str(val or "").strip().lower()
        verdict = "NO-ANSWER" if not text else (
            "PASS" if ref.strip().lower() in text else "FAIL")
        shown = str(val)[:26]
    else:
        got = to_float(val)
        if got is None:
            verdict, shown = "NO-ANSWER", str(val)[:26]
        elif abs(got - ref) / (abs(ref) or 1) <= TOL:
            verdict, shown = "PASS", f"{got:,.4f}"
        elif hazard is not None and isinstance(hazard, (int, float)) \
                and abs(got - hazard) / (abs(hazard) or 1) <= TOL:
            verdict, shown = "FAIL-HAZARD", f"{got:,.4f}"
        else:
            verdict, shown = "FAIL", f"{got:,.4f}"

    rows.append((qid, verdict, shown, ref, rec.get("turns"),
                 rec.get("prompt_tokens"), ans.get("workbook_rows_total")))

print(f"{'qid':5} {'verdict':12} {'rlm':>26} {'reference':>18} {'turns':>5} "
      f"{'prompt_tok':>10} {'rlm_rowcount':>12}")
print("-" * 94)
for qid, v, shown, ref, t, pt, wr in rows:
    r = f"{ref:,.4f}" if isinstance(ref, (int, float)) else str(ref)
    print(f"{qid:5} {v:12} {shown:>26} {r[:18]:>18} {str(t):>5} "
          f"{str(pt):>10} {str(wr):>12}")

n = len(rows)
p = sum(1 for r in rows if r[1] == "PASS")
haz = sum(1 for r in rows if r[1] == "FAIL-HAZARD")
na = sum(1 for r in rows if r[1] == "NO-ANSWER")
print(f"\nANALYTICAL accuracy (from run log): {p}/{n} = {100*p/n:.1f}%")
print(f"  fell for the documented fan-out hazard: {haz}")
print(f"  no machine-readable answer            : {na}")
print(f"  mean turns   : {sum(r[4] or 0 for r in rows)/n:.2f}")
print(f"  mean prompt  : {sum(r[5] or 0 for r in rows)/n:,.0f} tokens")
print("\nBaseline for reference only -- Phase 1 cold arm: 61/72 = 84.7%, "
      "5.04 turns, 9,160 prompt tokens.")
print("  NOT a valid comparison: different model (gpt-4.1-mini), different "
      "environment (local py3.11), and no workbook duty.")

# RLM's own claimed workbook row count vs reality -- did it notice the reset?
print("\n--- RLM's self-reported workbook_rows_total ---")
prev = 0
for qid, _, _, _, _, _, wr in rows:
    flag = ""
    if isinstance(wr, int):
        if wr < prev:
            flag = "  <-- RLM's own row count went BACKWARDS"
        prev = max(prev, wr)
    print(f"  {qid:5} {str(wr):>6}{flag}")

json.dump([{"qid": q, "verdict": v, "rlm": s, "reference": r,
            "turns": t, "prompt_tokens": pt, "rlm_rowcount": wr}
           for q, v, s, r, t, pt, wr in rows],
          open(OUT_PATH, "w"), indent=2)
print(f"\nwrote {OUT_PATH.name}  (source: {LOG_PATH.name})")
