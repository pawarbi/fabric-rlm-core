import json
import sys
from collections import defaultdict

data = json.load(open("dbo_eval_results.json"))
gt = {r["id"]: r for r in json.load(open("ground_truth.json"))}
want = sys.argv[1].split(",") if len(sys.argv) > 1 else ["q20", "q23"]

by = defaultdict(list)
for t in data["trials"]:
    by[t["question_id"]].append(t)

for q in want:
    r = gt[q]
    print("=" * 100)
    print(f'{q}  REF={r["reference"]!r}   HAZARD={r["hazard"]!r}')
    print(f'  Q : {r["question"]}')
    print(f'  SQL: {r["sql"][:200]}')
    print("-" * 100)
    for t in sorted(by[q], key=lambda x: (x["arm"], x["rep"])):
        g = t.get("grade", {})
        a = t.get("answer") or {}
        print(f'  arm {t["arm"]} rep{t["rep"]} turns={t.get("turns")} '
              f'analytic={g.get("analytic_correct")} val={g.get("value")!r}')
        if isinstance(a, dict):
            for k in ("status", "units", "grain"):
                if a.get(k):
                    print(f'        {k}: {str(a[k])[:110]}')
            if a.get("sql"):
                print(f'        sql: {str(a["sql"])[:220]}')
            if a.get("reasoning"):
                print(f'        why: {str(a["reasoning"])[:220]}')
    print()
