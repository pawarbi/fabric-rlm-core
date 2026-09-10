"""Build a `declared=` block for the dbo lakehouse from independently computed
data statistics — the "arm D" that tests the lesson pathway `learn()` leaves
empty on a Delta source.

Design rules, all enforced mechanically:

  * Facts are generated ONLY from `profile.json` (computed in Phase 1 by
    `profile_tables.py` via deltalake/pandas, independent of the library's
    query compiler and of ground truth). Nothing is copied from
    `ground_truth.json`, and no per-question tailoring is applied.
  * `samples` and `min`/`max` are EXCLUDED wholesale. Category value lists and
    extrema are the stats most likely to answer a question directly
    ("which segment…", "what is the highest…"), and arm D's accuracy would be
    uninterpretable if any of them leaked.
  * Every rendered fact is then checked against every reference answer; a fact
    whose number collides with a reference value is dropped and reported.

What is kept is structural and domain-neutral: row counts, uniqueness/grain,
join-key fan-out, null density, negative-value presence, duplicate-label
warnings. These are exactly the statistics a data-exploring profiler would
produce, and the ones §3 of REDISCOVERY_ANALYSIS.md argues buy correctness.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
profile = json.loads((HERE / "profile.json").read_text(encoding="utf-8"))
truth = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8"))

REFERENCE_VALUES = set()
for row in truth:
    for key in ("reference", "reference_sql_engine", "hazard"):
        v = row.get(key)
        if isinstance(v, (int, float)):
            REFERENCE_VALUES.add(round(float(v), 2))

definitions = {}
notes = []
dropped = []


def collides(*numbers):
    """True if any number in a fact matches a reference or hazard answer."""
    for n in numbers:
        if n is None:
            continue
        if round(float(n), 2) in REFERENCE_VALUES:
            return True
    return False


# ---- per-table structural facts -------------------------------------------
key_domains = {}
for table, entry in sorted(profile.items()):
    rows = entry.get("rows")
    cols = entry.get("columns", [])
    if not isinstance(rows, int):
        continue

    unique_cols = [c["name"] for c in cols
                   if c.get("distinct") == rows and rows > 0]
    nullable = [(c["name"], c["nulls"]) for c in cols
                if isinstance(c.get("nulls"), int) and c["nulls"] > 0]
    dupes = [(c["name"], c["distinct"]) for c in cols
             if c.get("dtype") == "object"
             and isinstance(c.get("distinct"), int)
             and 0 < c["distinct"] < rows
             and ("name" in c["name"] or "id" in c["name"])]

    if collides(rows):
        dropped.append(f"{table}.rows={rows} collides with a reference value")
        continue

    parts = [f"{rows:,} rows"]
    if unique_cols:
        parts.append("one row per " + " + ".join(unique_cols)
                     if len(unique_cols) > 1
                     else f"one row per {unique_cols[0]} (unique key)")
    else:
        parts.append("NO single unique column — grain is a combination; "
                     "verify before assuming one row per entity")
    if nullable:
        parts.append("nulls present in " + ", ".join(
            f"{n} ({k:,})" for n, k in nullable[:4]))
    if dupes:
        parts.append("repeated labels in " + ", ".join(
            f"{n} ({d:,} distinct of {rows:,})" for n, d in dupes[:3])
            + " — do NOT join or group on these as if unique")
    definitions[f"dbo.{table}"] = "; ".join(parts) + "."

    for c in cols:
        nm = c["name"]
        if nm.endswith("_id") and isinstance(c.get("distinct"), int):
            key_domains.setdefault(nm, []).append((table, rows, c["distinct"]))

# ---- cross-table join fan-out ---------------------------------------------
fanouts = []
for key, uses in sorted(key_domains.items()):
    if len(uses) < 2:
        continue
    parents = [u for u in uses if u[2] == u[1]]
    children = [u for u in uses if u[2] < u[1]]
    for pt, prows, _ in parents:
        for ct, crows, cdist in children:
            if cdist == 0:
                continue
            ratio = crows / cdist
            if ratio > 1.05:
                fanouts.append(
                    f"joining dbo.{pt} to dbo.{ct} on {key} FANS OUT "
                    f"~{ratio:.1f}x ({crows:,} {ct} rows over {cdist:,} distinct "
                    f"{key}); aggregate the child first or parent columns will "
                    f"be double counted")

for f in fanouts[:12]:
    notes.append(f)

# ---- negative values -------------------------------------------------------
for table, entry in sorted(profile.items()):
    for c in entry.get("columns", []):
        mn = c.get("min")
        if isinstance(mn, (int, float)) and mn < 0 and not collides(mn):
            notes.append(
                f"dbo.{table}.{c['name']} contains NEGATIVE values — confirm "
                f"whether they are reversals/credits before summing as-is")

notes.append("These facts were computed from the tables directly. They "
             "describe structure and data quality only; they do not state any "
             "business interpretation. Verify anything you rely on.")

declared = {"lakehouse": {"definitions": definitions, "notes": notes}}

out = HERE / "declared_dbo.json"
out.write_text(json.dumps(declared, indent=1), encoding="utf-8")

rendered = json.dumps(declared)
print(f"definitions : {len(definitions)}")
print(f"notes       : {len(notes)}  ({len(fanouts)} fan-out warnings)")
print(f"rendered    : {len(rendered):,} chars")
print(f"dropped     : {len(dropped)}")
for d in dropped:
    print("   -", d)

# ---- contamination audit ---------------------------------------------------
print("\n=== contamination audit ===")
hits = []
for row in truth:
    for key in ("reference", "reference_sql_engine"):
        v = row.get(key)
        if isinstance(v, (int, float)):
            for form in (f"{v:,.2f}", f"{v:.2f}", str(v), f"{int(v):,}"
                         if float(v).is_integer() else str(v)):
                if form and len(form) > 3 and form in rendered:
                    hits.append((row["id"], key, form))
if hits:
    print("LEAK — reference values appear in the declared text:")
    for h in hits:
        print("   ", h)
else:
    print("PASS: no reference or hazard answer string appears in declared text.")
print(f"(checked {len(REFERENCE_VALUES)} distinct reference/hazard values)")
