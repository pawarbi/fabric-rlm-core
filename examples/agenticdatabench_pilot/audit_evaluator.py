"""Adversarial audit of AgenticDataBench's compare_csv.

Each case states what a correct grader should do, then reports what this one
actually does. Nothing here is patched -- the benchmark's own comparator is
called directly.
"""
import sys, tempfile
from pathlib import Path
import pandas as pd

TESTBED = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(TESTBED))
from da_agent.evaluators.metrics.table import compare_csv

TMP = Path(tempfile.mkdtemp())

def score(pred_df, gold_df, **opts):
    p, g = TMP / "p.csv", TMP / "g.csv"
    pred_df.to_csv(p, index=False)
    gold_df.to_csv(g, index=False)
    r = compare_csv(str(p), str(g), **opts)
    return round(r["score"] if isinstance(r, dict) else r, 4)

GOLD = pd.DataFrame({"name": ["a", "b", "c"],
                     "count": [10, 20, 30],
                     "rate": [0.10, 0.20, 0.30]})

cases = []
def case(desc, expected, got):
    cases.append((desc, expected, got))

# --- sanity
case("identical files", "1.0", score(GOLD.copy(), GOLD))

# --- naming / ordering: does the output contract actually matter?
p = GOLD.copy()[["rate", "count", "name"]]
case("prediction columns REORDERED", "1.0 if content-matched", score(p, GOLD))

p = GOLD.copy(); p.columns = ["zzz", "qqq", "www"]
case("prediction columns RENAMED", "1.0 if name-agnostic", score(p, GOLD))

# --- tolerance behaviour
p = GOLD.copy(); p["rate"] = [0.1049, 0.2049, 0.3049]
case("rate off by 0.0049 (abs tol 0.01)", "1.0 within tolerance", score(p, GOLD))
p = GOLD.copy(); p["rate"] = [0.1149, 0.2149, 0.3149]
case("rate off by 0.0149 (beyond tol)", "<1.0 outside tolerance", score(p, GOLD))
# bucket-boundary straddle: values within tol but on opposite sides of a grid line
p = GOLD.copy(); p["rate"] = [0.1050001, 0.2050001, 0.3050001]
case("rate straddling a tolerance bucket edge", "1.0 (fallback should catch)",
     score(p, GOLD))

# --- ignore_order semantics
p = pd.concat([GOLD.copy(), pd.DataFrame({"name": ["z"], "count": [999],
                                          "rate": [0.99]})])
case("prediction has EXTRA junk rows (ignore_order=True)", "<1.0 ideally",
     score(p, GOLD, ignore_order=True))

g2 = pd.DataFrame({"name": ["a", "a", "b"], "count": [10, 10, 20],
                   "rate": [0.1, 0.1, 0.2]})
p2 = pd.DataFrame({"name": ["a", "b"], "count": [10, 20], "rate": [0.1, 0.2]})
case("prediction DROPS a duplicate row (ignore_order=True)", "<1.0 ideally",
     score(p2, g2, ignore_order=True))

# --- cross-column false match
g3 = pd.DataFrame({"first": [1, 2, 3], "second": [1, 2, 3]})
p3 = pd.DataFrame({"only_one": [1, 2, 3]})
case("gold has 2 identical cols, prediction has 1", "<1.0 ideally",
     score(p3, g3))

# --- missing / empty
p4 = GOLD.copy().drop(columns=["rate"])
case("prediction MISSING a column", "<1.0", score(p4, GOLD))
case("prediction is a single junk column", "0.0",
     score(pd.DataFrame({"x": [0]}), GOLD))

# --- case/whitespace
p5 = GOLD.copy(); p5["name"] = ["A ", " B", "C"]
case("string case + whitespace differences", "1.0 (normalized)", score(p5, GOLD))

print(f"{'case':52s} {'expected':32s} {'actual'}")
print("-" * 100)
for d, e, g in cases:
    flag = ""
    if "ideally" in e and g == 1.0:
        flag = "   <-- LENIENT"
    print(f"{d:52s} {e:32s} {g}{flag}")
