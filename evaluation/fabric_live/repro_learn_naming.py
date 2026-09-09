"""F12 -- demonstrate that learn()'s only tabular lesson is gated on ENGLISH names.

Brief item 1 asks for domain dependencies with their behavioural effect shown,
and item 3 asks whether renaming that PRESERVES MEANING changes learned
lessons. This probe answers both without credentials and without a model.

It varies ONLY the column name. Type stays boolean, semantics stay "this row
belongs to the current reporting period", the table stays the same shape. If
learning were name-independent, every variant would yield the same lesson.

Run:  python repro_learn_naming.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fabric-rlm-core-pr75"))

from fabric_rlm.knowledge import SourceProfile
from fabric_rlm.knowledge_lessons import _tabular_structural_lessons

# Every one of these names denotes exactly the same thing: a boolean flag
# marking the rows in the current reporting period.
VARIANTS = [
    ("english, period word",      "is_current_quarter"),
    ("english, period word",      "current_period_flag"),
    ("english, no period word",   "is_current"),
    ("english, as-of form",       "as_of_date_flag"),
    ("abbreviation",              "cur_qtr_flg"),
    ("abbreviation",              "is_curr_per"),
    ("different convention",      "IsCurrentQuarter"),
    ("different convention",      "ACTIVE_PERIOD_IND"),
    ("domain word, same meaning", "in_reporting_window"),
    ("domain word, same meaning", "open_fiscal_period"),
    ("non-English name",          "periodo_actual"),
    ("non-English name",          "aktuelle_periode"),
]


def profile_with(column: str) -> SourceProfile:
    return SourceProfile(
        source_id="lakehouse",
        family="lakehouse",
        locator="abfss://example/Tables/dbo",
        snapshot_fingerprint="f" * 8,
        schema_fingerprint="s" * 8,
        schema={column: {"type": "boolean"}, "amount": {"type": "double"}},
        diagnostics={},
        sensitive_columns=(),
        role="numeric_evidence",
        status="active",
    )


print(f"{'naming style':28} {'column name':22} {'lesson?':8} {'status':10} conf")
print("-" * 82)
rows = []
for style, col in VARIANTS:
    lessons = _tabular_structural_lessons(profile_with(col))
    if lessons:
        L = lessons[0]
        rows.append((style, col, True, L.status, L.confidence))
        print(f"{style:28} {col:22} {'YES':8} {L.status:10} {L.confidence}")
    else:
        rows.append((style, col, False, "-", "-"))
        print(f"{style:28} {col:22} {'no':8} {'-':10} -")

got = sum(1 for r in rows if r[2])
print(f"\n{got}/{len(rows)} semantically IDENTICAL columns produced a lesson.")
print("\nConclusion: the only structural lesson a tabular/Lakehouse source can")
print("produce is gated on the English tokens in _CURRENT_PERIOD:")
print("  (is_)current | as_of | latest_(period|quarter|month|date|week)")
print("and its status is raised to 'active' only when the name ALSO matches")
print("_PERIOD_COLUMN: period|quarter|month|year|date|week|fiscal|day.")
print("\nRename-preserving-meaning therefore CHANGES what is learned. This is a")
print("naming-convention and English-keyword dependency in the learning path,")
print("not merely in examples or fixtures.")
