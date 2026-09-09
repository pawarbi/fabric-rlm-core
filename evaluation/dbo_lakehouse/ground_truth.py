"""Ground truth for the 25-question dbo evaluation.

Every reference answer is computed TWICE and independently:

  * once with pandas  (``pd_*`` functions)
  * once with DuckDB SQL over the same cached parquet

The two must agree to 1e-6 or the question is rejected before any live call.
The author of a question is also the author of its reference, so agreement
across two engines is the only cheap protection against a wrong reference.

Nothing in this module imports fabric_rlm. The references are therefore
independent of the library's query compiler, which is the point.

Each question also declares a ``hazard``: the value produced by the most
plausible WRONG path (usually join fan-out, null mishandling, or an
incomplete final period). A question whose hazard is within 0.5% of the
reference measures nothing and is dropped by check_degeneracy.py.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

CACHE = Path(__file__).parent / "cache"

# Tables in scope. The rlm_eval_* tables are deliberately excluded: they are
# 2-5 row fixtures left over from an earlier evaluation and are not part of
# this dataset.
TABLES = [
    "companies", "dim_date", "features", "industries", "invoices",
    "payments", "subscriptions", "support_tickets", "usage_logs", "users",
]


def load() -> dict[str, pd.DataFrame]:
    return {t: pd.read_parquet(CACHE / f"{t}.parquet") for t in TABLES}


def con(d: dict[str, pd.DataFrame]) -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    for name, df in d.items():
        c.register(name, df)
    return c


def _one(c: duckdb.DuckDBPyConnection, sql: str):
    r = c.execute(sql).fetchone()
    return None if r is None else r[0]


# --------------------------------------------------------------------------
# Q1  Billed vs collected -- classic invoice->payment fan-out
# --------------------------------------------------------------------------
Q1_SQL = """
SELECT ROUND(SUM(amount_due), 2)
FROM invoices
WHERE invoice_id IN (SELECT DISTINCT invoice_id FROM payments)
"""


def pd_q1(d):
    inv = d["invoices"]
    paid_ids = set(d["payments"]["invoice_id"].unique())
    return round(float(inv.loc[inv["invoice_id"].isin(paid_ids), "amount_due"].sum()), 2)


def hz_q1(d):
    """Fan-out: join then sum amount_due, double-counting multi-payment invoices."""
    m = d["invoices"].merge(d["payments"], on="invoice_id", how="inner")
    return round(float(m["amount_due"].sum()), 2)


# --------------------------------------------------------------------------
# Q2  Collection rate = collected / billed, over ALL invoices
# --------------------------------------------------------------------------
Q2_SQL = """
SELECT ROUND(100.0 * (SELECT SUM(amount) FROM payments)
                   / (SELECT SUM(amount_due) FROM invoices), 4)
"""


def pd_q2(d):
    return round(100.0 * float(d["payments"]["amount"].sum())
                 / float(d["invoices"]["amount_due"].sum()), 4)


def hz_q2(d):
    """Uses amount_paid instead of actual payments -- a different measure."""
    return round(100.0 * float(d["invoices"]["amount_paid"].sum())
                 / float(d["invoices"]["amount_due"].sum()), 4)


# --------------------------------------------------------------------------
# Q3  Mean days to first payment (invoice_date -> earliest payment_date)
# --------------------------------------------------------------------------
Q3_SQL = """
WITH firstpay AS (
  SELECT invoice_id, MIN(CAST(payment_date AS DATE)) AS d
  FROM payments GROUP BY invoice_id
)
SELECT ROUND(AVG(DATE_DIFF('day', CAST(i.invoice_date AS DATE), f.d)), 4)
FROM invoices i JOIN firstpay f ON f.invoice_id = i.invoice_id
"""


def pd_q3(d):
    p = d["payments"].copy()
    p["payment_date"] = pd.to_datetime(p["payment_date"])
    first = p.groupby("invoice_id", as_index=False)["payment_date"].min()
    i = d["invoices"].copy()
    i["invoice_date"] = pd.to_datetime(i["invoice_date"])
    m = i.merge(first, on="invoice_id", how="inner")
    return round(float((m["payment_date"] - m["invoice_date"]).dt.days.mean()), 4)


def hz_q3(d):
    """Averages over every payment, over-weighting invoices paid in instalments."""
    p = d["payments"].copy()
    p["payment_date"] = pd.to_datetime(p["payment_date"])
    i = d["invoices"].copy()
    i["invoice_date"] = pd.to_datetime(i["invoice_date"])
    m = i.merge(p, on="invoice_id", how="inner")
    return round(float((m["payment_date"] - m["invoice_date"]).dt.days.mean()), 4)


# --------------------------------------------------------------------------
# Q4  Mean tenure in days of ENDED subscriptions (end_date has 92 nulls)
# --------------------------------------------------------------------------
Q4_SQL = """
SELECT ROUND(AVG(DATE_DIFF('day', CAST(start_date AS DATE), CAST(end_date AS DATE))), 4)
FROM subscriptions WHERE end_date IS NOT NULL
"""


def pd_q4(d):
    s = d["subscriptions"].dropna(subset=["end_date"]).copy()
    return round(float((pd.to_datetime(s["end_date"])
                        - pd.to_datetime(s["start_date"])).dt.days.mean()), 4)


def hz_q4(d):
    """Treats still-active subs as ending today -- inflates tenure."""
    s = d["subscriptions"].copy()
    end = pd.to_datetime(s["end_date"]).fillna(pd.Timestamp("2025-03-31"))
    return round(float((end - pd.to_datetime(s["start_date"])).dt.days.mean()), 4)


# --------------------------------------------------------------------------
# Q5  Total MRR of currently active subscriptions
# --------------------------------------------------------------------------
Q5_SQL = "SELECT ROUND(SUM(mrr), 2) FROM subscriptions WHERE status = 'active'"


def pd_q5(d):
    s = d["subscriptions"]
    return round(float(s.loc[s["status"] == "active", "mrr"].sum()), 2)


def hz_q5(d):
    """Sums MRR across every subscription regardless of status."""
    return round(float(d["subscriptions"]["mrr"].sum()), 2)


# --------------------------------------------------------------------------
# Q6  Mean satisfaction score (678 nulls -- must not be counted as zero)
# --------------------------------------------------------------------------
Q6_SQL = "SELECT ROUND(AVG(satisfaction_score), 6) FROM support_tickets"


def pd_q6(d):
    return round(float(d["support_tickets"]["satisfaction_score"].mean()), 6)


def hz_q6(d):
    """Null-as-zero: divides by all 4,192 tickets instead of the 3,514 scored."""
    return round(float(d["support_tickets"]["satisfaction_score"].fillna(0).mean()), 6)


# --------------------------------------------------------------------------
# Q7  Share of tickets ever resolved
# --------------------------------------------------------------------------
Q7_SQL = """
SELECT ROUND(100.0 * COUNT(resolved_at) / COUNT(*), 4) FROM support_tickets
"""


def pd_q7(d):
    t = d["support_tickets"]
    return round(100.0 * float(t["resolved_at"].notna().sum()) / len(t), 4)


def hz_q7(d):
    """Counts distinct companies rather than tickets."""
    t = d["support_tickets"]
    return round(100.0 * float(t.dropna(subset=["resolved_at"])["company_id"].nunique())
                 / t["company_id"].nunique(), 4)


# --------------------------------------------------------------------------
# Q8  Mean hours to first response
# --------------------------------------------------------------------------
Q8_SQL = """
SELECT ROUND(AVG(DATE_DIFF('second', created_at, first_response_at)) / 3600.0, 6)
FROM support_tickets
"""


def pd_q8(d):
    t = d["support_tickets"]
    delta = (t["first_response_at"] - t["created_at"]).dt.total_seconds()
    return round(float(delta.mean()) / 3600.0, 6)


def hz_q8(d):
    """Restricts to resolved tickets only, though every ticket has a response."""
    t = d["support_tickets"].dropna(subset=["resolved_at"])
    delta = (t["first_response_at"] - t["created_at"]).dt.total_seconds()
    return round(float(delta.mean()) / 3600.0, 6)


# --------------------------------------------------------------------------
# Q9  Total API calls  (661,734 rows -- too large to paste into a prompt)
# --------------------------------------------------------------------------
Q9_SQL = "SELECT SUM(CAST(api_calls AS BIGINT)) FROM usage_logs"


def pd_q9(d):
    return int(d["usage_logs"]["api_calls"].sum())


def hz_q9(d):
    """Row count instead of the sum of the measure."""
    return int(len(d["usage_logs"]))


# --------------------------------------------------------------------------
# Q10 Feature with the highest total compute minutes -- name it
# --------------------------------------------------------------------------
Q10_SQL = """
SELECT f.display_name
FROM usage_logs u JOIN features f ON f.feature_id = u.feature_id
GROUP BY f.display_name ORDER BY SUM(u.compute_minutes) DESC LIMIT 1
"""


def pd_q10(d):
    m = d["usage_logs"].merge(d["features"], on="feature_id", how="inner")
    g = m.groupby("display_name")["compute_minutes"].sum()
    return str(g.idxmax())


def hz_q10(d):
    """Ranks by event count rather than by compute minutes."""
    m = d["usage_logs"].merge(d["features"], on="feature_id", how="inner")
    return str(m.groupby("display_name").size().idxmax())


# --------------------------------------------------------------------------
# Q11 Sector with the highest total api_calls (4-hop join)
# --------------------------------------------------------------------------
Q11_SQL = """
SELECT i.sector
FROM usage_logs u
JOIN users us   ON us.user_id     = u.user_id
JOIN companies c ON c.company_id  = us.company_id
JOIN industries i ON i.industry_id = c.industry_id
GROUP BY i.sector ORDER BY SUM(CAST(u.api_calls AS BIGINT)) DESC LIMIT 1
"""


def pd_q11(d):
    m = (d["usage_logs"]
         .merge(d["users"][["user_id", "company_id"]], on="user_id")
         .merge(d["companies"][["company_id", "industry_id"]], on="company_id")
         .merge(d["industries"][["industry_id", "sector"]], on="industry_id"))
    return str(m.groupby("sector")["api_calls"].sum().idxmax())


def hz_q11(d):
    """Ranks sectors by number of companies rather than by usage."""
    m = d["companies"].merge(d["industries"], on="industry_id")
    return str(m.groupby("sector").size().idxmax())


# --------------------------------------------------------------------------
# Q12 Premium-feature share of total compute minutes
# --------------------------------------------------------------------------
Q12_SQL = """
SELECT ROUND(100.0 * SUM(CASE WHEN f.is_premium = 1 THEN u.compute_minutes ELSE 0 END)
                   / SUM(u.compute_minutes), 4)
FROM usage_logs u JOIN features f ON f.feature_id = u.feature_id
"""


def pd_q12(d):
    m = d["usage_logs"].merge(d["features"], on="feature_id", how="inner")
    return round(100.0 * float(m.loc[m["is_premium"] == 1, "compute_minutes"].sum())
                 / float(m["compute_minutes"].sum()), 4)


def hz_q12(d):
    """Share of premium FEATURES (a catalogue property), not of usage."""
    f = d["features"]
    return round(100.0 * float((f["is_premium"] == 1).sum()) / len(f), 4)


# --------------------------------------------------------------------------
# Q13 Number of companies with no subscription at all
# --------------------------------------------------------------------------
Q13_SQL = """
SELECT COUNT(*) FROM companies
WHERE company_id NOT IN (SELECT DISTINCT company_id FROM subscriptions)
"""


def pd_q13(d):
    have = set(d["subscriptions"]["company_id"].unique())
    return int((~d["companies"]["company_id"].isin(have)).sum())


def hz_q13(d):
    """Companies minus subscriptions -- wrong because a company may have several."""
    return int(len(d["companies"]) - len(d["subscriptions"]))


# --------------------------------------------------------------------------
# Q14 Mean MRR per COMPANY (companies hold multiple subscriptions)
# --------------------------------------------------------------------------
Q14_SQL = """
SELECT ROUND(AVG(t), 6) FROM (
  SELECT company_id, SUM(mrr) AS t FROM subscriptions GROUP BY company_id
)
"""


def pd_q14(d):
    return round(float(d["subscriptions"].groupby("company_id")["mrr"].sum().mean()), 6)


def hz_q14(d):
    """Mean per subscription, not per company -- different grain."""
    return round(float(d["subscriptions"]["mrr"].mean()), 6)


# --------------------------------------------------------------------------
# Q15 Overdue invoice count
# --------------------------------------------------------------------------
Q15_SQL = "SELECT COUNT(*) FROM invoices WHERE status = 'overdue'"


def pd_q15(d):
    return int((d["invoices"]["status"] == "overdue").sum())


def hz_q15(d):
    """Anything not fully paid -- includes void and sent."""
    i = d["invoices"]
    return int((i["amount_paid"] < i["amount_due"]).sum())


# --------------------------------------------------------------------------
# Q16 Plan with the highest mean MRR
# --------------------------------------------------------------------------
Q16_SQL = """
SELECT plan_name FROM subscriptions GROUP BY plan_name
ORDER BY AVG(mrr) DESC LIMIT 1
"""


def pd_q16(d):
    return str(d["subscriptions"].groupby("plan_name")["mrr"].mean().idxmax())


def hz_q16(d):
    """Highest TOTAL mrr -- favours the most populous plan."""
    return str(d["subscriptions"].groupby("plan_name")["mrr"].sum().idxmax())


# --------------------------------------------------------------------------
# Q17 Active-user share
# --------------------------------------------------------------------------
Q17_SQL = "SELECT ROUND(100.0 * SUM(is_active) / COUNT(*), 4) FROM users"


def pd_q17(d):
    return round(100.0 * float(d["users"]["is_active"].sum()) / len(d["users"]), 4)


def hz_q17(d):
    """Share of users who ever appear in usage_logs -- a different definition."""
    seen = set(d["usage_logs"]["user_id"].unique())
    return round(100.0 * float(d["users"]["user_id"].isin(seen).sum())
                 / len(d["users"]), 4)


# --------------------------------------------------------------------------
# Q18 Last COMPLETE calendar month of invoicing, and its billed total
# --------------------------------------------------------------------------
Q18_SQL = """
SELECT ROUND(SUM(amount_due), 2) FROM invoices
WHERE STRFTIME(CAST(invoice_date AS DATE), '%Y-%m') = (
  SELECT MAX(STRFTIME(CAST(invoice_date AS DATE), '%Y-%m')) FROM invoices
)
"""


def pd_q18(d):
    i = d["invoices"].copy()
    ym = pd.to_datetime(i["invoice_date"]).dt.strftime("%Y-%m")
    return round(float(i.loc[ym == ym.max(), "amount_due"].sum()), 2)


def hz_q18(d):
    """Uses the payments calendar, which runs past the last invoice month."""
    p = d["payments"].copy()
    ym = pd.to_datetime(p["payment_date"]).dt.strftime("%Y-%m")
    return round(float(p.loc[ym == ym.max(), "amount"].sum()), 2)


# --------------------------------------------------------------------------
# Q19 Most common payment method by VALUE
# --------------------------------------------------------------------------
Q19_SQL = """
SELECT method FROM payments GROUP BY method ORDER BY SUM(amount) DESC LIMIT 1
"""


def pd_q19(d):
    return str(d["payments"].groupby("method")["amount"].sum().idxmax())


def hz_q19(d):
    """By transaction count rather than by value."""
    return str(d["payments"].groupby("method").size().idxmax())


# --------------------------------------------------------------------------
# Q20 Enterprise-company share of active MRR
# --------------------------------------------------------------------------
Q20_SQL = """
SELECT ROUND(100.0 * SUM(CASE WHEN c.company_size = 'enterprise' THEN s.mrr ELSE 0 END)
                   / SUM(s.mrr), 4)
FROM subscriptions s JOIN companies c ON c.company_id = s.company_id
WHERE s.status = 'active'
"""


def pd_q20(d):
    s = d["subscriptions"]
    s = s[s["status"] == "active"].merge(d["companies"], on="company_id", how="inner")
    return round(100.0 * float(s.loc[s["company_size"] == "enterprise", "mrr"].sum())
                 / float(s["mrr"].sum()), 4)


def hz_q20(d):
    """Ignores the active filter."""
    s = d["subscriptions"].merge(d["companies"], on="company_id", how="inner")
    return round(100.0 * float(s.loc[s["company_size"] == "enterprise", "mrr"].sum())
                 / float(s["mrr"].sum()), 4)


# --------------------------------------------------------------------------
# Q21 Region with the worst mean satisfaction
# --------------------------------------------------------------------------
Q21_SQL = """
SELECT c.region
FROM support_tickets t JOIN companies c ON c.company_id = t.company_id
WHERE t.satisfaction_score IS NOT NULL
GROUP BY c.region ORDER BY AVG(t.satisfaction_score) ASC LIMIT 1
"""


def pd_q21(d):
    m = d["support_tickets"].dropna(subset=["satisfaction_score"]).merge(
        d["companies"], on="company_id", how="inner")
    return str(m.groupby("region")["satisfaction_score"].mean().idxmin())


def hz_q21(d):
    """Null-as-zero changes the ranking."""
    m = d["support_tickets"].merge(d["companies"], on="company_id", how="inner")
    m = m.assign(s=m["satisfaction_score"].fillna(0))
    return str(m.groupby("region")["s"].mean().idxmin())


# --------------------------------------------------------------------------
# Q22 Companies with an active subscription AND a critical ticket
# --------------------------------------------------------------------------
Q22_SQL = """
SELECT COUNT(*) FROM (
  SELECT DISTINCT s.company_id FROM subscriptions s
  WHERE s.status = 'active'
    AND s.company_id IN (SELECT company_id FROM support_tickets WHERE priority='critical')
)
"""


def pd_q22(d):
    act = set(d["subscriptions"].query("status == 'active'")["company_id"])
    crit = set(d["support_tickets"].query("priority == 'critical'")["company_id"])
    return int(len(act & crit))


def hz_q22(d):
    """Counts ticket rows rather than distinct companies."""
    act = set(d["subscriptions"].query("status == 'active'")["company_id"])
    t = d["support_tickets"].query("priority == 'critical'")
    return int(t["company_id"].isin(act).sum())


# --------------------------------------------------------------------------
# Q23 Mean api_calls per ACTIVE user (denominator is a definition choice)
# --------------------------------------------------------------------------
Q23_SQL = """
SELECT ROUND(SUM(CAST(u.api_calls AS BIGINT)) * 1.0
             / (SELECT COUNT(*) FROM users WHERE is_active = 1), 6)
FROM usage_logs u
"""


def pd_q23(d):
    n = int(d["users"]["is_active"].sum())
    return round(float(d["usage_logs"]["api_calls"].sum()) / n, 6)


def hz_q23(d):
    """Divides by all users rather than active users."""
    return round(float(d["usage_logs"]["api_calls"].sum()) / len(d["users"]), 6)


# --------------------------------------------------------------------------
# Q24 Cancelled-subscription share
# --------------------------------------------------------------------------
Q24_SQL = """
SELECT ROUND(100.0 * SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END)
             / COUNT(*), 4) FROM subscriptions
"""


def pd_q24(d):
    s = d["subscriptions"]
    return round(100.0 * float((s["status"] == "cancelled").sum()) / len(s), 4)


def hz_q24(d):
    """Counts cancelled + expired as churn -- a different definition."""
    s = d["subscriptions"]
    return round(100.0 * float(s["status"].isin(["cancelled", "expired"]).sum())
                 / len(s), 4)


# --------------------------------------------------------------------------
# Q25 Mean invoices per subscription that was ever invoiced
# --------------------------------------------------------------------------
Q25_SQL = """
SELECT ROUND(AVG(n), 6) FROM (
  SELECT sub_id, COUNT(*) AS n FROM invoices GROUP BY sub_id
)
"""


def pd_q25(d):
    return round(float(d["invoices"].groupby("sub_id").size().mean()), 6)


def hz_q25(d):
    """Divides by all subscriptions, including those never invoiced."""
    return round(float(len(d["invoices"])) / len(d["subscriptions"]), 6)


# --------------------------------------------------------------------------

QUESTIONS = [
    ("q01", "Across invoices that have received at least one payment, what is the "
            "total amount_due? Report a single number rounded to 2 decimals.",
     pd_q1, Q1_SQL, hz_q1, "invoice->payment fan-out (9,204 payments / 7,584 invoices)"),
    ("q02", "What percentage of all billed amount_due has actually been collected, "
            "measured from the payments table? Round to 4 decimals.",
     pd_q2, Q2_SQL, hz_q2, "payments.amount vs invoices.amount_paid are different measures"),
    ("q03", "On average, how many days elapse between an invoice date and its FIRST "
            "payment? Consider only invoices that were paid. Round to 4 decimals.",
     pd_q3, Q3_SQL, hz_q3, "must collapse to first payment, not average over all payments"),
    ("q04", "For subscriptions that have ended, what is the mean tenure in days from "
            "start_date to end_date? Round to 4 decimals.",
     pd_q4, Q4_SQL, hz_q4, "end_date has 92 nulls for still-active subscriptions"),
    ("q05", "What is the total MRR of subscriptions whose status is 'active'? "
            "Round to 2 decimals.",
     pd_q5, Q5_SQL, hz_q5, "must filter on status"),
    ("q06", "What is the mean satisfaction_score across support tickets? "
            "Round to 6 decimals.",
     pd_q6, Q6_SQL, hz_q6, "678 nulls must be excluded, not treated as zero"),
    ("q07", "What percentage of support tickets have ever been resolved? "
            "Round to 4 decimals.",
     pd_q7, Q7_SQL, hz_q7, "grain: tickets, not companies"),
    ("q08", "What is the mean number of hours between ticket creation and first "
            "response? Round to 6 decimals.",
     pd_q8, Q8_SQL, hz_q8, "all tickets have a response; do not restrict to resolved"),
    ("q09", "What is the total number of api_calls recorded across all usage logs?",
     pd_q9, Q9_SQL, hz_q9, "661,734 rows: too large to inspect by hand; sum not count"),
    ("q10", "Which feature has the highest total compute_minutes? Give its display_name.",
     pd_q10, Q10_SQL, hz_q10, "rank by compute_minutes, not by event frequency"),
    ("q11", "Which industry sector accounts for the most api_calls? Give the sector name.",
     pd_q11, Q11_SQL, hz_q11, "4-hop join usage->users->companies->industries"),
    ("q12", "What percentage of total compute_minutes is consumed by premium features? "
            "Round to 4 decimals.",
     pd_q12, Q12_SQL, hz_q12, "share of usage, not share of the feature catalogue"),
    ("q13", "How many companies have no subscription record at all?",
     pd_q13, Q13_SQL, hz_q13, "a company may hold several subscriptions"),
    ("q14", "What is the mean total MRR per company, summing each company's "
            "subscriptions first? Round to 6 decimals.",
     pd_q14, Q14_SQL, hz_q14, "grain: per company, not per subscription"),
    ("q15", "How many invoices have status 'overdue'?",
     pd_q15, Q15_SQL, hz_q15, "status field, not an inferred underpayment test"),
    ("q16", "Which plan_name has the highest MEAN mrr?",
     pd_q16, Q16_SQL, hz_q16, "mean, not total"),
    ("q17", "What percentage of users are flagged is_active? Round to 4 decimals.",
     pd_q17, Q17_SQL, hz_q17, "the flag, not observed usage"),
    ("q18", "For the most recent month present in the invoices table, what is the "
            "total amount_due? Round to 2 decimals.",
     pd_q18, Q18_SQL, hz_q18, "invoices span 42 months; payments run later"),
    ("q19", "Which payment method accounts for the largest total payment value?",
     pd_q19, Q19_SQL, hz_q19, "by value, not by transaction count"),
    ("q20", "Among ACTIVE subscriptions, what percentage of MRR comes from companies "
            "whose company_size is 'enterprise'? Round to 4 decimals.",
     pd_q20, Q20_SQL, hz_q20, "active filter plus a join to company_size"),
    ("q21", "Which region has the LOWEST mean satisfaction_score? Give the region name.",
     pd_q21, Q21_SQL, hz_q21, "nulls must be excluded or the ranking flips"),
    ("q22", "How many distinct companies have both an active subscription and at "
            "least one critical-priority ticket?",
     pd_q22, Q22_SQL, hz_q22, "distinct companies, not ticket rows"),
    ("q23", "What is the mean number of api_calls per ACTIVE user? Round to 6 decimals.",
     pd_q23, Q23_SQL, hz_q23, "denominator is active users only"),
    ("q24", "What percentage of subscriptions have status 'cancelled'? "
            "Round to 4 decimals.",
     pd_q24, Q24_SQL, hz_q24, "'cancelled' only; 'expired' is a separate status"),
    ("q25", "Among subscriptions that have at least one invoice, what is the mean "
            "number of invoices per subscription? Round to 6 decimals.",
     pd_q25, Q25_SQL, hz_q25, "denominator is invoiced subscriptions only"),
]


def separation(ref, hz) -> float:
    """Relative distance between the correct answer and the plausible-wrong one.

    Returns 1.0 for categorical answers that differ, 0.0 when they coincide.
    A question with separation below MIN_SEPARATION cannot tell a competent
    run from an incompetent one and is rejected before any live call.
    """
    if isinstance(ref, (int, float)) and isinstance(hz, (int, float)):
        denom = max(abs(float(ref)), 1e-12)
        return abs(float(ref) - float(hz)) / denom
    return 0.0 if str(ref) == str(hz) else 1.0


MIN_SEPARATION = 0.02  # 2%


def build() -> list[dict]:
    from replacements import REPLACEMENTS

    d = load()
    c = con(d)

    questions = []
    for qid, text, pdf, sql, hzf, note in QUESTIONS:
        if qid in REPLACEMENTS:
            text, pdf, sql, hzf, note = REPLACEMENTS[qid]
        questions.append((qid, text, pdf, sql, hzf, note))

    out = []
    for qid, text, pdf, sql, hzf, note in questions:
        ref_pd = pdf(d)
        ref_sql = _one(c, sql)
        if isinstance(ref_sql, (int, float)) and isinstance(ref_pd, (int, float)):
            agree = abs(float(ref_pd) - float(ref_sql)) <= max(
                1e-6, abs(float(ref_pd)) * 1e-9)
        else:
            agree = str(ref_pd) == str(ref_sql)
        hz = hzf(d)
        out.append({
            "id": qid,
            "question": text,
            "reference": ref_pd,
            "reference_sql_engine": (float(ref_sql)
                                     if isinstance(ref_sql, (int, float))
                                     else str(ref_sql)),
            "engines_agree": bool(agree),
            "hazard": hz,
            "hazard_note": note,
            "separation": round(separation(ref_pd, hz), 6),
            "discriminating": separation(ref_pd, hz) >= MIN_SEPARATION,
            "sql": " ".join(sql.split()),
        })
    return out


if __name__ == "__main__":
    import json

    rows = build()
    bad = [r for r in rows if not r["engines_agree"]]
    weak = [r for r in rows if not r["discriminating"]]

    for r in rows:
        eng = "OK " if r["engines_agree"] else "ENGINE-MISMATCH"
        sep = r["separation"]
        mark = "  " if r["discriminating"] else "<< DEGENERATE"
        print(f'{eng} {r["id"]}  sep={sep:>10.4f}  ref={r["reference"]!r:<22} '
              f'hz={r["hazard"]!r:<22}{mark}')

    print(f"\nengine agreement : {len(rows) - len(bad)}/{len(rows)}")
    print(f"discriminating   : {len(rows) - len(weak)}/{len(rows)}")
    if weak:
        print("REJECTED:", ", ".join(r["id"] for r in weak))

    Path("ground_truth.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print("wrote ground_truth.json")
    raise SystemExit(1 if (bad or weak) else 0)
