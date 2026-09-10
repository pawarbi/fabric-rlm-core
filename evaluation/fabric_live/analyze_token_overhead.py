"""Is the learn-arm token overhead a UNIFORM per-question planner tax, or a
few tail events?

The rubber-duck review challenged my claim that +64% prompt tokens comes from
the per-task operation-selection LM call. I never captured
`operation_selection_prompt_tokens`, so I cannot attribute it directly. But
uniformity is testable with what I have: a per-question tax should appear on
EVERY question, including ones where the turn count is unchanged.

Confounders separated:
  - q15/q17/q24  an operation actually executed (knowledge_result injected)
  - q08/q21      gate-heavy, many extra turns (prompt grows per turn)
"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP_EXECUTED = {"q15", "q17", "q24"}
GATE_HEAVY = {"q08", "q21"}


def load(name):
    text = (HERE / name).read_text(encoding="utf-8")
    return {r["id"]: r for r in json.loads(text[text.find("["):])}


A = load("run_log_glm.json")
B = load("run_log_glm_learn.json")

print(f"{'qid':5}{'A_tok':>9}{'B_tok':>9}{'delta':>10}{'A_t':>5}{'B_t':>5}  note")
print("-" * 62)
rows = []
for q in sorted(set(A) & set(B)):
    a, b = A[q], B[q]
    if not a.get("prompt_tokens") or not b.get("prompt_tokens"):
        print(f"{q:5}{'-':>9}{'-':>9}{'-':>10}{'-':>5}{'-':>5}  no answer in one arm")
        continue
    delta = b["prompt_tokens"] - a["prompt_tokens"]
    dturns = b["turns"] - a["turns"]
    note = []
    if q in OP_EXECUTED:
        note.append("operation EXECUTED")
    if q in GATE_HEAVY:
        note.append("gate-heavy")
    rows.append((q, delta, dturns, q in OP_EXECUTED or q in GATE_HEAVY))
    print(f"{q:5}{a['prompt_tokens']:>9,}{b['prompt_tokens']:>9,}{delta:>+10,}"
          f"{a['turns']:>5}{b['turns']:>5}  {' '.join(note)}")

clean = [r for r in rows if not r[3]]
flat = [r for r in clean if r[2] == 0]

print(f"\n'clean' questions (no operation executed, not gate-heavy): {len(clean)}")
print(f"  median delta : {statistics.median(r[1] for r in clean):+,.0f}")
print(f"  mean   delta : {statistics.mean(r[1] for r in clean):+,.0f}")
print(f"  min / max    : {min(r[1] for r in clean):+,} / {max(r[1] for r in clean):+,}")
print(f"  negative     : {sum(1 for r in clean if r[1] < 0)} of {len(clean)}")

print(f"\nclean AND same turn count in both arms: {len(flat)}")
if flat:
    print(f"  deltas       : {[f'{r[1]:+,}' for r in flat]}")
    print(f"  median       : {statistics.median(r[1] for r in flat):+,.0f}")
    print("  A uniform per-question planner tax must show up HERE, where turn")
    print("  count is identical and no operation ran.")
