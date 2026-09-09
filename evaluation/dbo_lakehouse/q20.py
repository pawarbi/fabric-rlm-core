import json
import re

d = json.load(open("dbo_eval_results.json"))
UPPER = re.compile(r"=\s*'ACTIVE'")
for t in d["trials"]:
    if t["question_id"] != "q20":
        continue
    a = t.get("answer") or {}
    sql = str(a.get("sql") or "")
    print(f"--- arm {t['arm']} rep{t['rep']}  value={a.get('value')!r} "
          f"units={a.get('units')!r} status={a.get('status')!r} "
          f"uppercase_ACTIVE={bool(UPPER.search(sql))}")
    print("    reasoning:", str(a.get("reasoning"))[:200])
