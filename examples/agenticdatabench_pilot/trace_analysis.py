"""What actually costs us on AgenticDataBench: mine the run's trajectories."""
import json, re, sys, statistics as st
from collections import Counter
from pathlib import Path

SCRATCH = Path(sys.argv[1])
OUT = SCRATCH / sys.argv[2]
TESTBED = SCRATCH / "AgenticDataBench" / "testbed"

tasks = {json.loads(l)["id"]: json.loads(l)
         for l in open(TESTBED / "tasks" / "dev.jsonl", encoding="utf-8")}
grades = {r["id"]: r["score"]
          for r in json.load(open(OUT / "grades.json", encoding="utf-8"))["results"]}

DISCOVERY = re.compile(
    r"\.head\(|\.columns|\.dtypes|\.info\(|\.shape|read_csv\([^)]*nrows|"
    r"\.sample\(|print\(df|glob\.|os\.listdir|\.describe\(", re.I)

rows = []
err_kinds = Counter()
for tid, task in tasks.items():
    p = OUT / tid / "trajectory.jsonl"
    if not p.exists():
        continue
    turns = [json.loads(l) for l in open(p, encoding="utf-8")]
    turns = [t for t in turns if "turn" in t]
    if not turns:
        continue
    n = len(turns)
    errs = [t for t in turns if t.get("error") or (t.get("stderr") or "").strip()]
    for t in errs:
        blob = str(t.get("error") or t.get("stderr") or "")
        m = re.search(r"(\w*Error|\w*Exception)", blob)
        err_kinds[m.group(1) if m else "other"] += 1
    # turns spent purely inspecting shape/schema before real computation
    disc = 0
    for t in turns:
        code = t.get("code") or ""
        if DISCOVERY.search(code) and not re.search(r"to_csv|to_json|savefig|SUBMIT", code):
            disc += 1
        else:
            break
    tin = sum((t.get("token_usage") or {}).get("prompt_tokens", 0) or 0 for t in turns)
    tout = sum((t.get("token_usage") or {}).get("completion_tokens", 0) or 0 for t in turns)
    rows.append(dict(id=tid, n=n, errs=len(errs), disc=disc,
                     tin=tin, tout=tout, score=grades.get(tid, 0.0),
                     submitted=any(t.get("submitted") for t in turns)))

n = len(rows)
print(f"trajectories analysed: {n}\n")

print("TURNS")
turn_counts = [r["n"] for r in rows]
print(f"  median {st.median(turn_counts):.0f}, mean {st.mean(turn_counts):.1f}, "
      f"at ceiling(25) {sum(1 for r in rows if r['n']>=25)} tasks")
ceil = [r for r in rows if r["n"] >= 25]
print(f"  tasks at ceiling scored {st.mean([r['score'] for r in ceil]):.3f} "
      f"vs {st.mean([r['score'] for r in rows if r['n']<25]):.3f} for the rest")

print("\nTOKENS (input dominates: context is resent every turn)")
tin = sum(r["tin"] for r in rows); tout = sum(r["tout"] for r in rows)
print(f"  input {tin/1e6:.1f}M  output {tout/1e6:.2f}M  -> input is "
      f"{100*tin/(tin+tout):.1f}% of spend-weighted volume")
print(f"  per task: {tin/n:,.0f} in / {tout/n:,.0f} out")

print("\nSCHEMA-DISCOVERY TURNS (leading turns that only inspect shape/columns)")
disc = [r["disc"] for r in rows]
print(f"  mean {st.mean(disc):.2f} turns/task, "
      f"{sum(1 for d in disc if d>=1)} tasks spend >=1, "
      f"{sum(1 for d in disc if d>=2)} spend >=2")
share = st.mean([r["disc"]/r["n"] for r in rows])
print(f"  ~{100*share:.0f}% of all turns are leading discovery")

print("\nERRORS")
print(f"  tasks with >=1 erroring turn: {sum(1 for r in rows if r['errs'])} ({100*sum(1 for r in rows if r['errs'])/n:.0f}%)")
print(f"  mean erroring turns/task: {st.mean([r['errs'] for r in rows]):.2f}")
print("  top error kinds:", dict(err_kinds.most_common(6)))
e = [r for r in rows if r["errs"]]; ne = [r for r in rows if not r["errs"]]
if e and ne:
    print(f"  score with errors {st.mean([r['score'] for r in e]):.3f} "
          f"vs without {st.mean([r['score'] for r in ne]):.3f}")

print("\nSUBMIT")
nosub = [r for r in rows if not r["submitted"]]
print(f"  never called SUBMIT: {len(nosub)} tasks, mean score {st.mean([r['score'] for r in nosub]) if nosub else 0:.3f}")
