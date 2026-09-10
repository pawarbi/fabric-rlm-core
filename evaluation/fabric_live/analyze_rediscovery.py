"""What did RLM have to rediscover on every question?

Motivating question (user): if `learn()` had actually explored the data —
"column x holds customer names as First Last", "sales_amount has negatives",
"this key fans out" — would that have cut turns?

This measures the *rediscovery* half of that question from the captured
trajectories of the GLM learn arm (the only arm with H6 trajectory capture).

Three buckets, kept strictly separate (conflating them inflates any savings
estimate):

  ORIENT  question-INDEPENDENT facts — catalog, schema, dtypes, cardinality,
          nulls, ranges, duplicates, grain, join fan-out, sample rows.
          These are cacheable into a knowledge package.
  ANALYZE question-SPECIFIC aggregation. Not cacheable.
  FRICTION retry / gate-rejection / syntax-error turns. Defect cost (F11),
          not knowledge cost.

Output is descriptive: what was re-derived and how often. It does NOT claim a
package would remove these turns — see the report for that caveat.
"""
import json
import re
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "run_log_glm_learn.json"

# ---- probe signatures for question-independent facts -----------------------
PROBES = {
    "catalog (list_sources)": r"list_sources\s*\(",
    "schema / dtypes":        r"\.schema\b|\bdtypes\b|\.columns\b|DESCRIBE\b",
    "sample rows":            r"\.head\s*\(|\.sample\s*\(|\bLIMIT\s+\d+",
    "row count":              r"\.shape\b|\blen\s*\(\s*\w+\s*\)|COUNT\s*\(\s*\*",
    "distinct / cardinality": r"\.nunique\s*\(|\.unique\s*\(|value_counts|DISTINCT",
    "null checks":            r"\.isna\s*\(|\.isnull\s*\(|\.notna\s*\(|IS\s+NULL|\.fillna\s*\(",
    "duplicate checks":       r"duplicated\s*\(|drop_duplicates\s*\(",
    "range / min-max":        r"\.describe\s*\(|\.min\s*\(|\.max\s*\(",
    "negative-value checks":  r"<\s*0\b|negative",
    "join fan-out / grain":   r"groupby\([^)]*\)\.(size|count)\s*\(|merge\(|\bJOIN\b",
}
FRICTION = r"Traceback|Error|error|not allowed|disallow|rejected|invalid|failed"
ANALYZE = r"\.sum\s*\(|\.mean\s*\(|\.agg\s*\(|GROUP\s+BY|\.pivot|weighted"


def split_turn(t):
    """Return (code, stdout) for a captured TurnRecord repr.

    The record exposes exactly three fields: turn, code, stdout. Friction must
    be judged from stdout, never from the whole record — matching 'error'
    against the full text produced a 91% false-positive rate on the first pass.
    """
    text = t if isinstance(t, str) else t.get("text", "")
    m = re.search(r"code=(['\"])(.*?)\1,\s*stdout=(['\"])(.*?)\3\s*\)?\s*$",
                  text, re.S)
    if m:
        return unesc(m.group(2)), unesc(m.group(4))
    m = re.search(r"code=(['\"])(.*?)\1,\s*stdout=", text, re.S)
    if m:
        return unesc(m.group(2)), unesc(text[m.end():])
    return "", unesc(text)


def unesc(s):
    try:
        return s.encode().decode("unicode_escape")
    except Exception:
        return s


# friction is judged on STDOUT only, and only on unambiguous markers
FRICTION_PAT = {
    "traceback": r"Traceback \(most recent call last\)",
    "exception": r"^\w*(Error|Exception):",
    "catalog gate": r"lakehouse\.query|not allowed|disallow",
}

rows = json.loads(LOG.read_text(encoding="utf-8")[
    LOG.read_text(encoding="utf-8").find("["):])
rows = [r for r in rows if r.get("trajectory")]

questions_with = Counter()
turns_with = Counter()
friction_kind = Counter()
per_q_bucket = {}
orient_stdout_bytes = 0

for r in rows:
    qid = r["id"]
    seen = set()
    orient = analyze = friction = 0
    first_analyze = None
    for i, t in enumerate(r["trajectory"]):
        code, out = split_turn(t)
        fk = [k for k, p in FRICTION_PAT.items()
              if re.search(p, out, re.M)]
        hit = [name for name, pat in PROBES.items() if re.search(pat, code)]
        is_analyze = bool(re.search(ANALYZE, code))
        for name in hit:
            turns_with[name] += 1
            seen.add(name)
        if fk:
            friction += 1
            for k in fk:
                friction_kind[k] += 1
        if hit and not is_analyze:
            orient += 1
            orient_stdout_bytes += len(out)
        elif is_analyze:
            analyze += 1
            if first_analyze is None:
                first_analyze = i
    for name in seen:
        questions_with[name] += 1
    per_q_bucket[qid] = (orient, analyze, friction, len(r["trajectory"]),
                         first_analyze)

n = len(rows)
print(f"GLM learn arm — {n} questions with captured trajectories\n")
print("QUESTION-INDEPENDENT FACTS RE-DERIVED (cacheable into a package)")
print(f"{'fact probed':26}{'questions':>11}{'   share':>9}{'turns':>7}")
print("-" * 55)
for name in sorted(PROBES, key=lambda k: -questions_with[k]):
    q = questions_with[name]
    print(f"{name:26}{q:>7}/{n:<3}{q/n:>8.0%}{turns_with[name]:>7}")

print("\nPER-QUESTION TURN BUDGET  (orientation and analysis are not exclusive")
print("of friction — a turn can both probe and fail)")
o = [v[0] for v in per_q_bucket.values()]
a = [v[1] for v in per_q_bucket.values()]
f = [v[2] for v in per_q_bucket.values()]
tot = sum(v[3] for v in per_q_bucket.values())
print(f"  orientation turns : {sum(o):4}  ({sum(o)/tot:.0%})  median {statistics.median(o):.0f}/question")
print(f"  analysis turns    : {sum(a):4}  ({sum(a)/tot:.0%})  median {statistics.median(a):.0f}/question")
print(f"  friction turns    : {sum(f):4}  ({sum(f)/tot:.0%})  median {statistics.median(f):.0f}/question")
print(f"  total captured    : {tot:4}")
print(f"  friction by kind  : {dict(friction_kind)}")
print(f"\n  stdout bytes consumed by orientation turns: {orient_stdout_bytes:,}")
print("  (that is the prompt cost a cached profile would REPLACE, not add)")

fa = [v[4] for v in per_q_bucket.values() if v[4] is not None]
if fa:
    print(f"\n  turns before the FIRST analytical turn: median {statistics.median(fa):.0f}, max {max(fa)}")
