"""Was arm B's data access equivalent to arm A's?

Arm C proved impossible: the library raises ValueError if you pass both a
knowledge package and inputs for the same aliases. So arm B is the only
learned configuration available, and the package itself must carry data
access. This checks whether it actually did.

If arm B executed real SQL against the real tables at a rate comparable to
arm A, then access was equivalent and the accuracy deficit is behavioural.
If arm B rarely reached the data, the deficit is access starvation.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

data = json.load(open("dbo_eval_results.json"))

# Signals that the run actually reached the underlying data rather than
# reciting a package-supplied aggregate.
REAL_SQL = re.compile(r"\b(from|join)\s+[\w./\\:\-]*"
                      r"(companies|invoices|payments|subscriptions|users|"
                      r"usage_logs|support_tickets|dim_date|features|industries)",
                      re.I)

stats = defaultdict(lambda: defaultdict(int))
for t in data["trials"]:
    if not t.get("ok"):
        continue
    arm = t["arm"]
    stats[arm]["n"] += 1
    ans = t.get("answer") or {}
    sql = str(ans.get("sql") or "")
    has_sql = bool(sql.strip()) and sql.strip().lower() not in ("none", "n/a", "null")
    hits_table = bool(REAL_SQL.search(sql))
    stats[arm]["emitted_sql"] += has_sql
    stats[arm]["sql_hits_real_table"] += hits_table
    correct = bool(t.get("grade", {}).get("analytic_correct"))
    if hits_table:
        stats[arm]["correct_when_real_sql"] += correct
        stats[arm]["n_real_sql"] += 1
    else:
        stats[arm]["n_no_real_sql"] += 1
        stats[arm]["correct_when_no_real_sql"] += correct

print("=" * 88)
print("DATA-ACCESS EQUIVALENCE CHECK")
print("=" * 88)
print(f'{"":<34}{"arm A":>14}{"arm B":>14}')
for k, lbl in (("n", "trials"),
               ("emitted_sql", "emitted any SQL"),
               ("sql_hits_real_table", "SQL references a real table")):
    a, b = stats["A"][k], stats["B"][k]
    print(f"{lbl:<34}{a:>14}{b:>14}")

print()
print("Accuracy conditioned on reaching the data:")
for arm in ("A", "B"):
    s = stats[arm]
    for tag, nk, ck in (("reached real table   ", "n_real_sql", "correct_when_real_sql"),
                        ("did NOT reach a table", "n_no_real_sql", "correct_when_no_real_sql")):
        n, c = s[nk], s[ck]
        pct = f"{100*c/n:.1f}%" if n else "  n/a"
        print(f"  arm {arm}  {tag}: {c:>3}/{n:<3} = {pct}")

print()
print("Interpretation:")
a_rate = stats["A"]["sql_hits_real_table"] / max(stats["A"]["n"], 1)
b_rate = stats["B"]["sql_hits_real_table"] / max(stats["B"]["n"], 1)
print(f"  arm A reached the data in {a_rate:.0%} of trials; arm B in {b_rate:.0%}.")
if b_rate >= a_rate - 0.10:
    print("  -> Access was EQUIVALENT. The accuracy deficit is behavioural,")
    print("     not starvation: arm B could reach the data and often chose not to.")
else:
    print("  -> Access was NOT equivalent. The deficit is confounded by the package")
    print("     failing to expose the sources, and must be reported as such.")
