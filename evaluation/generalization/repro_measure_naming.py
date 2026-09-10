"""F15 repro: learned behaviour on a semantic model is gated by English names.

Pure-function test of the regex that decides which semantic-model lessons exist.
No Fabric, no LLM, no key, no network. Exits non-zero if the finding no longer
reproduces, so it doubles as a regression test for any future fix.

    python repro_measure_naming.py
"""
import sys

from fabric_rlm.knowledge_lessons import _PERIOD_COLUMN, _is_derived_measure

# A. English period-comparison measures. These SHOULD be nominated.
ENGLISH = [
    "Revenue PY", "Revenue YTD", "Orders YoY Pct", "Sales Growth",
    "Net Revenue Retention", "NRR", "GRR", "Churn Rate", "Revenue MoM",
    "TTM Revenue",
]

# B. The SAME measures in other languages. Same DAX semantics, same
# relationships, same definitions -- only the naming convention differs.
# de / es / fr / it / pt / nl / et.
NON_ENGLISH = [
    "Umsatz Vorjahr", "Ingresos Ano Anterior", "Chiffre affaires N-1",
    "Ricavi Anno Precedente", "Receita Ano Anterior", "Omzet Vorig Jaar",
    "Intaktid Eelmine Aasta",
]

# C. Period COLUMNS in other languages, against _PERIOD_COLUMN.
NON_ENGLISH_PERIOD = ["Datum", "Fecha", "Monat", "Trimestre", "Anno",
                      "Kuupaev", "Jahr"]

# D. English names that are NOT period comparisons. These should NOT fire.
# "Avg Delivery Variance" and "Scenario Variance" are REAL measures in the
# live ecommerce-dataset, and learn() really did nominate both.
FALSE_POSITIVES = [
    "Avg Delivery Variance", "Scenario Variance", "Delta Airlines Revenue",
    "Variance of Weight", "Statistical Variance", "Population Delta",
    "Price Change Approval Count", "Customer Retention Team Headcount",
]


def report(title, names, fn, expect):
    hits = [n for n in names if fn(n)]
    print(f"\n=== {title} ===")
    for n in names:
        print(f"  {str(fn(n)):5} {n}")
    print(f"  -> {len(hits)}/{len(names)} fire (expected {expect})")
    return len(hits)


a = report("A. English period-comparison measures (should fire)",
           ENGLISH, _is_derived_measure, "10/10")
b = report("B. Same measures, non-English names (equivalent semantics)",
           NON_ENGLISH, _is_derived_measure, "0/7 -- MISSED")
c = report("C. Period columns, non-English",
           NON_ENGLISH_PERIOD, lambda n: bool(_PERIOD_COLUMN.search(n)),
           "0/7 -- MISSED")
d = report("D. English names that are NOT period comparisons",
           FALSE_POSITIVES, _is_derived_measure, "8/8 -- FALSE POSITIVE")

print("\n" + "=" * 62)
print(f"English recall      : {a}/{len(ENGLISH)}")
print(f"Non-English recall  : {b}/{len(NON_ENGLISH)}   <- coverage gap")
print(f"Non-English periods : {c}/{len(NON_ENGLISH_PERIOD)}   <- coverage gap")
print(f"False positives     : {d}/{len(FALSE_POSITIVES)}   <- precision gap")

ok = (a == len(ENGLISH) and b == 0 and c == 0 and d == len(FALSE_POSITIVES))
print("\nF15 reproduces exactly." if ok else
      "\nF15 did NOT reproduce -- behaviour changed; re-read the finding.")
sys.exit(0 if ok else 1)
