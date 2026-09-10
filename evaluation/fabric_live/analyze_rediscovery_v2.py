"""Rediscovery analysis, v2 — SQL-aware and much stricter.

v1 was unsound and its numbers are withdrawn. Two defects, both found by
auditing what the regexes actually matched (`audit_probes.py`):

  * The probe vocabulary was written for pandas (`.shape`, `groupby().size()`,
    `.min()`), but the agent works almost entirely in SQL. Those patterns
    matched 0 times, producing false negatives — v1 claimed ranges were
    "never checked" while `MIN(`/`MAX(` appear 51 times.
  * `COUNT(*)` was counted as row-count *profiling*, but it is overwhelmingly
    part of an analytical query (`COUNT(*) AS invoices_with_payment`). v1's
    headline "133 row-count turns" was mostly analysis, not orientation.

v2 only counts a construct as ORIENTATION when it is *standalone* — a query
whose entire purpose is to inspect. Anything appearing alongside business
aggregation or grouping is treated as analysis. This is deliberately
conservative: it under-counts orientation rather than over-counting it.
"""
import json
import re
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
raw = (HERE / "run_log_glm_learn.json").read_text(encoding="utf-8")
rows = [r for r in json.loads(raw[raw.find("["):]) if r.get("trajectory")]


def unesc(s):
    try:
        return s.encode().decode("unicode_escape")
    except Exception:
        return s


def split_turn(t):
    text = t.get("text", "")
    m = re.search(r"code=(['\"])(.*?)\1,\s*stdout=", text, re.S)
    if not m:
        return "", unesc(text)
    return unesc(m.group(2)), unesc(text[m.end():])


SQL = re.compile(r"SELECT\b.*?(?=(?:\"\"\"|'''|\"|')\s*[\),])", re.S | re.I)
BUSINESS_AGG = re.compile(r"\bSUM\s*\(|\bAVG\s*\(|\bROUND\s*\(|GROUP\s+BY\b", re.I)


def statements(code):
    """Best-effort extraction of SQL statements from executed code."""
    out = []
    for m in re.finditer(r"SELECT\b", code, re.I):
        chunk = code[m.start():m.start() + 600]
        chunk = re.split(r"\"\"\"|'''|\n\s*\n", chunk)[0]
        out.append(" ".join(chunk.split()))
    return out


# --- strictly standalone inspection constructs ------------------------------
ORIENT_RULES = {
    # SELECT COUNT(*) FROM t  — nothing else in the select list, no grouping
    "bare row count":
        lambda s: re.match(r"SELECT\s+COUNT\(\*\)(\s+AS\s+\w+)?\s+FROM\s+[\w\.\"]+\s*$", s, re.I),
    # SELECT * FROM t LIMIT n — sampling rows to look at them
    "sample rows":
        lambda s: re.match(r"SELECT\s+\*\s+FROM\s+[\w\.\"]+(\s+LIMIT\s+\d+)?\s*$", s, re.I),
    # standalone distinct-cardinality probe, no business aggregation
    "distinct cardinality":
        lambda s: re.search(r"COUNT\s*\(\s*DISTINCT", s, re.I) and not BUSINESS_AGG.search(s),
    # standalone range probe
    "range (min/max)":
        lambda s: re.search(r"\b(MIN|MAX)\s*\(", s, re.I) and not BUSINESS_AGG.search(s),
    # standalone null probe
    "null probe":
        lambda s: re.search(r"IS\s+NULL|COUNT\s*\(\s*\w+\s*\)\s*[-<>]", s, re.I) and not BUSINESS_AGG.search(s),
}
CATALOG = re.compile(r"list_sources\s*\(")

q_with = Counter()
stmt_hits = Counter()
turn_kind = Counter()
per_q = {}

for r in rows:
    seen, orient_t, analysis_t, friction_t = set(), 0, 0, 0
    for t in r["trajectory"]:
        code, out = split_turn(t)
        friction = bool(re.search(r"Traceback \(most recent call last\)", out))
        kinds = set()
        if CATALOG.search(code):
            kinds.add("catalog listing")
        stmts = statements(code)
        analytical = any(BUSINESS_AGG.search(s) for s in stmts)
        for s in stmts:
            for name, rule in ORIENT_RULES.items():
                try:
                    if rule(s):
                        kinds.add(name)
                        stmt_hits[name] += 1
                except Exception:
                    pass
        seen |= kinds
        for k in kinds:
            turn_kind[k] += 1
        if friction:
            friction_t += 1
        elif analytical:
            analysis_t += 1
        elif kinds:
            orient_t += 1
    for k in seen:
        q_with[k] += 1
    per_q[r["id"]] = (orient_t, analysis_t, friction_t, len(r["trajectory"]))

n = len(rows)
print(f"GLM learn arm — {n} questions with captured trajectories")
print("v2: only STANDALONE inspection counts as orientation\n")
print(f"{'orientation construct':24}{'questions':>11}{'share':>8}{'turns':>7}{'stmts':>7}")
print("-" * 57)
for k in sorted(set(list(ORIENT_RULES) + ["catalog listing"]),
                key=lambda k: -q_with[k]):
    print(f"{k:24}{q_with[k]:>7}/{n:<3}{q_with[k]/n:>7.0%}{turn_kind[k]:>7}{stmt_hits[k]:>7}")

o = [v[0] for v in per_q.values()]
a = [v[1] for v in per_q.values()]
f = [v[2] for v in per_q.values()]
tot = sum(v[3] for v in per_q.values())
print(f"\nTURN BUDGET over {tot} captured turns (mutually exclusive)")
print(f"  pure orientation : {sum(o):4}  {sum(o)/tot:>5.0%}   median {statistics.median(o):.0f}/q")
print(f"  analytical       : {sum(a):4}  {sum(a)/tot:>5.0%}   median {statistics.median(a):.0f}/q")
print(f"  friction         : {sum(f):4}  {sum(f)/tot:>5.0%}   median {statistics.median(f):.0f}/q")
print(f"  unclassified     : {tot-sum(o)-sum(a)-sum(f):4}")
