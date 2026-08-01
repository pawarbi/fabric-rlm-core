"""Compare our runs against the paper's published results, like for like.

The published result files cover 344 tasks (246 public + 98 private). Our runs
cover the 246 public tasks only, so the headline numbers are not comparable as
printed. The public ids are an exact subset of theirs, so this recomputes every
published configuration restricted to those same 246 tasks.
"""
import glob, json, os, statistics as st, sys
from pathlib import Path

SCRATCH = Path(sys.argv[1])
TESTBED = SCRATCH / "AgenticDataBench" / "testbed"

public = {json.loads(l)["id"]
          for l in open(TESTBED / "tasks" / "dev.jsonl", encoding="utf-8")}

# OpenRouter list price per token, for cost of our own runs.
PRICES = {"minimax/minimax-m3": (0.30e-6, 1.20e-6),
          "moonshotai/kimi-k2.5": (0.57e-6, 2.85e-6)}

rows = []
for p in sorted(glob.glob(str(TESTBED / "results" / "*.json"))):
    d = json.load(open(p, encoding="utf-8"))
    name = os.path.basename(p).replace(".json", "")
    full = [r["total_score"] for r in d["results"]]
    pub = [r["total_score"] for r in d["results"] if r["id"] in public]
    tok = 0
    for r in d["results"]:
        if r["id"] in public:
            u = r.get("total_token_usage") or {}
            tok += (u.get("total_tokens") or 0)
    rows.append((name, len(full), 100*sum(full)/len(full),
                 len(pub), 100*sum(pub)/len(pub), tok))

ours = []
for label, outdir, model in [
        ("fabric-rlm + MiniMax M3", SCRATCH / "full_run2", "minimax/minimax-m3"),
        ("fabric-rlm + Kimi K2.5", SCRATCH / "full_run_kimi", "moonshotai/kimi-k2.5")]:
    gp = outdir / "grades.json"
    if not gp.exists():
        continue
    res = json.load(open(gp, encoding="utf-8"))["results"]
    res = [r for r in res if r["id"] in public]
    if not res:
        continue
    ours.append((label, len(res), 100*sum(r["score"] for r in res)/len(res)))

print("PUBLISHED RESULTS (paper's own runs)")
print(f"{'config':24s} {'n=344':>9s} {'n':>5s} {'public-246 only':>16s}")
for name, nf, sf, npub, spub, tok in sorted(rows, key=lambda r: -r[4]):
    print(f"  {name:22s} {sf:8.2f}% {npub:5d} {spub:15.2f}%")

print("\nOUR RUNS (same 246 public tasks, graded with their comparators)")
for label, n, s in ours:
    print(f"  {label:22s} {'':8s}  {n:5d} {s:15.2f}%")

if ours:
    best = max(r[4] for r in rows)
    med = st.median([r[4] for r in rows])
    print(f"\npublished on public-246: best {best:.2f}%, median {med:.2f}%")
    for label, n, s in ours:
        beat = sum(1 for r in rows if s > r[4])
        print(f"  {label}: {s:.2f}%  beats {beat}/{len(rows)} published configs")
