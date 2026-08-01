"""Prove grade_pilot.py agrees with the STOCK AgenticDataBench evaluator.

The stock evaluator has two Windows-only defects, both rooted in paths
containing backslashes:
  1. it substitutes a raw path into an eval_func string via re.sub, where
     the path is the replacement *template* ("bad escape \\s");
  2. the substituted path then sits inside a Python string literal that is
     eval()'d, so "\\U..." is parsed as a unicode escape.
Neither occurs on Linux, where paths use forward slashes.

To compare fairly, this script makes Windows look like Linux to the stock
code: os.path.join is swapped for posixpath.join inside the evaluation
module and the roots are passed with forward slashes. No scoring,
aggregation or comparator logic is touched -- the numbers are produced by
their code.
"""
import json, posixpath, sys
from pathlib import Path

TESTBED = Path(sys.argv[1]).resolve()
OUTDIR = Path(sys.argv[2]).resolve()
IDS = sys.argv[3].split(",")

sys.path.insert(0, str(TESTBED))
import da_agent.evaluators.evaluation as ev

ev.os.path.join = posixpath.join  # make joined paths POSIX, as on Linux

posix = lambda p: str(p).replace("\\", "/")

tasks = [json.loads(l) for l in open(TESTBED / "tasks" / "dev.jsonl", encoding="utf-8")]
sel = [t for t in tasks if t["id"] in set(IDS)]
tmp = TESTBED / "tasks" / "_equiv.jsonl"
with open(tmp, "w", encoding="utf-8") as fh:
    for t in sel:
        fh.write(json.dumps(t) + "\n")

evaluator = ev.Evaluator(output_dir=posix(OUTDIR), gold_dir=posix(TESTBED / "gold"),
                         timeout_seconds=300)
res = evaluator.evaluate(env_config=str(tmp))
stock = {r["id"]: round(r["total_score"], 6) for r in res}

mine = {r["id"]: round(r["score"], 6)
        for r in json.load(open(OUTDIR / "grades.json", encoding="utf-8"))["results"]}

print("\n%-22s %10s %10s  %s" % ("task", "stock", "mine", "match"))
ok = True
for tid in IDS:
    s, m = stock.get(tid), mine.get(tid)
    match = (s is not None and m is not None and abs(s - m) < 1e-6)
    ok &= match
    print("%-22s %10s %10s  %s" % (tid, s, m, "OK" if match else "MISMATCH"))
print("\nEQUIVALENT" if ok else "\nDIVERGENCE FOUND")
