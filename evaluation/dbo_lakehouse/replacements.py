"""Replacement questions for the nine that failed the degeneracy screen.

A question whose plausible-wrong path lands on the same number as the correct
path cannot distinguish a competent run from an incompetent one. Six of the
original 25 were exactly degenerate and three more were within 1%. These are
the replacements, screened before any live call is spent on them.
"""
from __future__ import annotations

import pandas as pd

# --- q02: outstanding value on OVERDUE invoices only -----------------------
Q02_SQL = """
SELECT ROUND(SUM(amount_due - amount_paid), 2)
FROM invoices WHERE status = 'overdue'
"""


def pd_q02(d):
    i = d["invoices"]
    i = i[i["status"] == "overdue"]
    return round(float((i["amount_due"] - i["amount_paid"]).sum()), 2)


def hz_q02(d):
    """Outstanding value across every invoice, ignoring the overdue filter."""
    i = d["invoices"]
    return round(float((i["amount_due"] - i["amount_paid"]).sum()), 2)


# --- q04: mean tenure of ended ANNUAL-CONTRACT subscriptions ---------------
Q04_SQL = """
SELECT ROUND(AVG(DATE_DIFF('day', CAST(start_date AS DATE), CAST(end_date AS DATE))), 4)
FROM subscriptions WHERE end_date IS NOT NULL AND annual_contract = 1
"""


def pd_q04(d):
    s = d["subscriptions"]
    s = s[s["end_date"].notna() & (s["annual_contract"] == 1)]
    return round(float((pd.to_datetime(s["end_date"])
                        - pd.to_datetime(s["start_date"])).dt.days.mean()), 4)


def hz_q04(d):
    """Ignores the annual_contract filter."""
    s = d["subscriptions"][d["subscriptions"]["end_date"].notna()]
    return round(float((pd.to_datetime(s["end_date"])
                        - pd.to_datetime(s["start_date"])).dt.days.mean()), 4)


# --- q13: companies holding more than one subscription ---------------------
Q13_SQL = """
SELECT COUNT(*) FROM (
  SELECT company_id FROM subscriptions GROUP BY company_id HAVING COUNT(*) > 1
)
"""


def pd_q13(d):
    n = d["subscriptions"].groupby("company_id").size()
    return int((n > 1).sum())


def hz_q13(d):
    """Subscriptions minus companies -- counts extra subs, not companies."""
    return int(len(d["subscriptions"]) - d["subscriptions"]["company_id"].nunique())


# --- q08: mean hours to RESOLUTION among resolved tickets ------------------
Q08_SQL = """
SELECT ROUND(AVG(DATE_DIFF('second', created_at, resolved_at)) / 3600.0, 6)
FROM support_tickets WHERE resolved_at IS NOT NULL
"""


def pd_q08(d):
    t = d["support_tickets"].dropna(subset=["resolved_at"])
    return round(float((t["resolved_at"] - t["created_at"])
                       .dt.total_seconds().mean()) / 3600.0, 6)


def hz_q08(d):
    """Unresolved tickets counted as zero hours, dragging the mean down."""
    t = d["support_tickets"].copy()
    sec = (t["resolved_at"] - t["created_at"]).dt.total_seconds().fillna(0)
    return round(float(sec.mean()) / 3600.0, 6)


# --- q10: module with the highest MEAN compute per event -------------------
Q10_SQL = """
SELECT f.module FROM usage_logs u JOIN features f ON f.feature_id = u.feature_id
GROUP BY f.module ORDER BY AVG(u.compute_minutes) DESC LIMIT 1
"""


def pd_q10(d):
    m = d["usage_logs"].merge(d["features"], on="feature_id", how="inner")
    return str(m.groupby("module")["compute_minutes"].mean().idxmax())


def hz_q10(d):
    """Highest TOTAL compute -- favours the busiest module, not the heaviest."""
    m = d["usage_logs"].merge(d["features"], on="feature_id", how="inner")
    return str(m.groupby("module")["compute_minutes"].sum().idxmax())


# --- q11: region with the highest MEAN api_calls per user ------------------
Q11_SQL = """
WITH per_user AS (
  SELECT us.user_id, c.region, SUM(CAST(u.api_calls AS BIGINT)) AS calls
  FROM usage_logs u
  JOIN users us    ON us.user_id    = u.user_id
  JOIN companies c ON c.company_id  = us.company_id
  GROUP BY us.user_id, c.region
)
SELECT region FROM per_user GROUP BY region ORDER BY AVG(calls) DESC LIMIT 1
"""


def pd_q11(d):
    m = (d["usage_logs"]
         .merge(d["users"][["user_id", "company_id"]], on="user_id")
         .merge(d["companies"][["company_id", "region"]], on="company_id"))
    per_user = m.groupby(["user_id", "region"], as_index=False)["api_calls"].sum()
    return str(per_user.groupby("region")["api_calls"].mean().idxmax())


def hz_q11(d):
    """Total api_calls per region -- dominated by whichever region has most users."""
    m = (d["usage_logs"]
         .merge(d["users"][["user_id", "company_id"]], on="user_id")
         .merge(d["companies"][["company_id", "region"]], on="company_id"))
    return str(m.groupby("region")["api_calls"].sum().idxmax())


# --- q12: premium share of api_calls (not compute) -------------------------
Q12_SQL = """
SELECT ROUND(100.0 * SUM(CASE WHEN f.is_premium=1 THEN CAST(u.api_calls AS BIGINT) ELSE 0 END)
                   / SUM(CAST(u.api_calls AS BIGINT)), 4)
FROM usage_logs u JOIN features f ON f.feature_id = u.feature_id
"""


def pd_q12(d):
    m = d["usage_logs"].merge(d["features"], on="feature_id", how="inner")
    return round(100.0 * float(m.loc[m["is_premium"] == 1, "api_calls"].sum())
                 / float(m["api_calls"].sum()), 4)


def hz_q12(d):
    """Share of premium usage EVENTS rather than of api_calls volume."""
    m = d["usage_logs"].merge(d["features"], on="feature_id", how="inner")
    return round(100.0 * float((m["is_premium"] == 1).sum()) / len(m), 4)


# --- q16: company_size with the highest MEAN mrr ---------------------------
Q16_SQL = """
SELECT c.company_size
FROM subscriptions s JOIN companies c ON c.company_id = s.company_id
GROUP BY c.company_size ORDER BY AVG(s.mrr) DESC LIMIT 1
"""


def pd_q16(d):
    m = d["subscriptions"].merge(d["companies"], on="company_id", how="inner")
    return str(m.groupby("company_size")["mrr"].mean().idxmax())


def hz_q16(d):
    """Ranks by subscription COUNT rather than by mean revenue."""
    m = d["subscriptions"].merge(d["companies"], on="company_id", how="inner")
    return str(m.groupby("company_size").size().idxmax())


# --- q19: payment method with the highest MEAN payment ---------------------
Q19_SQL = """
SELECT method FROM payments GROUP BY method ORDER BY AVG(amount) DESC LIMIT 1
"""


def pd_q19(d):
    return str(d["payments"].groupby("method")["amount"].mean().idxmax())


def hz_q19(d):
    """Highest TOTAL value -- the most-used method, not the highest-value one."""
    return str(d["payments"].groupby("method")["amount"].sum().idxmax())


# --- q25: mean payments per PAID invoice -----------------------------------
Q25_SQL = """
SELECT ROUND(AVG(n), 6) FROM (
  SELECT invoice_id, COUNT(*) AS n FROM payments GROUP BY invoice_id
)
"""


def pd_q25(d):
    return round(float(d["payments"].groupby("invoice_id").size().mean()), 6)


def hz_q25(d):
    """Divides by every invoice, including the 478 that were never paid."""
    return round(float(len(d["payments"])) / len(d["invoices"]), 6)


REPLACEMENTS = {
    "q02": ("What is the total outstanding value (amount_due minus amount_paid) "
            "across invoices whose status is 'overdue'? Round to 2 decimals.",
            pd_q02, Q02_SQL, hz_q02,
            "the overdue filter must be applied before summing"),
    "q04": ("For subscriptions that have ended AND were annual contracts "
            "(annual_contract = 1), what is the mean tenure in days? "
            "Round to 4 decimals.",
            pd_q04, Q04_SQL, hz_q04,
            "the annual_contract filter must be applied"),
    "q13": ("How many companies hold more than one subscription?",
            pd_q13, Q13_SQL, hz_q13,
            "count companies, not the surplus of subscriptions over companies"),
    "q08": ("Among tickets that were resolved, what is the mean number of hours from "
            "creation to resolution? Round to 6 decimals.",
            pd_q08, Q08_SQL, hz_q08,
            "678 unresolved tickets must be excluded, not counted as zero"),
    "q10": ("Which feature module has the highest MEAN compute_minutes per usage "
            "event? Give the module name.",
            pd_q10, Q10_SQL, hz_q10,
            "mean per event, not total -- total favours the busiest module"),
    "q11": ("Which region has the highest MEAN api_calls per user? Compute each "
            "user's total first. Give the region name.",
            pd_q11, Q11_SQL, hz_q11,
            "per-user mean, not regional total"),
    "q12": ("What percentage of total api_calls is attributable to premium features? "
            "Round to 4 decimals.",
            pd_q12, Q12_SQL, hz_q12,
            "share of call volume, not share of usage events"),
    "q16": ("Which company_size segment has the highest MEAN subscription mrr?",
            pd_q16, Q16_SQL, hz_q16,
            "mean revenue, not subscription count"),
    "q19": ("Which payment method has the highest MEAN payment amount?",
            pd_q19, Q19_SQL, hz_q19,
            "mean, not total"),
    "q25": ("Among invoices that received at least one payment, what is the mean "
            "number of payments per invoice? Round to 6 decimals.",
            pd_q25, Q25_SQL, hz_q25,
            "denominator is paid invoices (7,584), not all invoices (8,062)"),
}
