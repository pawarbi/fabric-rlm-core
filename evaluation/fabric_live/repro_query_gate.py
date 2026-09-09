"""Reproduce the LakehouseSource.query rejection behaviour, no credentials needed.

_normalize_catalog_query is pure, so the first-stage gate can be exercised
directly. The question is not only WHAT it rejects but whether the error tells
the model enough to repair the query.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fabric-rlm-core-pr75"))

from fabric_rlm.lakehouse import _normalize_catalog_query  # noqa: E402

CASES = [
    ("plain aggregate",
     "SELECT SUM(amount_due) FROM invoices"),
    ("CTE",
     "WITH p AS (SELECT invoice_id FROM payments) SELECT COUNT(*) FROM p"),
    ("trailing line comment -- VERY COMMON IN LLM SQL",
     "SELECT SUM(amount_due) FROM invoices -- total outstanding"),
    ("leading line comment",
     "-- compute the total\nSELECT SUM(amount_due) FROM invoices"),
    ("block comment",
     "/* total */ SELECT SUM(amount_due) FROM invoices"),
    ("decrement operator inside expression",
     "SELECT a - -b FROM t"),
    ("negative literal after minus",
     "SELECT 5 - -3"),
    ("string containing a double dash",
     "SELECT * FROM t WHERE code = 'A--B'"),
    ("leading newline + SELECT",
     "\nSELECT 1"),
    ("EXPLAIN", "EXPLAIN SELECT 1"),
    ("DDL", "DROP TABLE invoices"),
]

print(f"{'case':52} {'accepted':9} error")
print("-" * 110)
for name, sql in CASES:
    try:
        _normalize_catalog_query(sql)
        print(f"{name:52} {'YES':9}")
    except ValueError as exc:
        print(f"{name:52} {'no':9} {exc}")

print("\nNote how many DISTINCT rejection causes share one identical message.")
