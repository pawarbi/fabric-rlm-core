"""How often does compare_csv reject a value that is INSIDE its own tolerance?

With ignore_order=True (the library default), tolerance is applied by snapping
each value onto a shared grid, round(v/tol)*tol, and comparing buckets. Two
values closer together than tol land in different buckets whenever a grid line
sits between them, so the comparison is a false negative even though the
prediction is within the stated tolerance.

This sweeps offsets from 0 to tol and reports the rejection rate.
"""
import sys, tempfile
from pathlib import Path
import pandas as pd

TESTBED = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(TESTBED))
from da_agent.evaluators.metrics.table import compare_csv

TMP = Path(tempfile.mkdtemp())
TOL = 0.01  # compare_csv's default absolute tolerance

def one(offset, base, ignore_order):
    gold = pd.DataFrame({"v": base})
    pred = pd.DataFrame({"v": [b + offset for b in base]})
    p, g = TMP / "p.csv", TMP / "g.csv"
    pred.to_csv(p, index=False); gold.to_csv(g, index=False)
    r = compare_csv(str(p), str(g), ignore_order=ignore_order)
    return (r["score"] if isinstance(r, dict) else r) == 1.0

base = [0.10, 0.25, 0.42, 0.77, 1.35]
for ignore_order in (True, False):
    rejected = []
    steps = 40
    for i in range(steps + 1):
        off = TOL * i / steps          # 0 .. tol, all within tolerance
        if off >= TOL:
            continue
        if not one(off, base, ignore_order):
            rejected.append(round(off, 5))
    total = steps
    print(f"ignore_order={ignore_order}: "
          f"{len(rejected)}/{total} offsets strictly inside tolerance were "
          f"REJECTED ({100*len(rejected)/total:.0f}%)")
    if rejected:
        print(f"   e.g. offsets {rejected[:6]} (tolerance is {TOL})")
