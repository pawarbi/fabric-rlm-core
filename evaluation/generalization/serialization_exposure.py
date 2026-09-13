"""Measure per-domain exposure to the scalar-serialization defect.

The defect is deterministic, so it does not need a language model to measure.
For each evaluation question this computes the answer the way an agent
plausibly would -- duckdb over the CSV fixtures, and pandas over the same
files -- and asks one question of each result:

    would the baseline ``freeze()`` have replaced this value with
    ``{"__serializable__": false}`` in the submitted answer?

The point is blast radius per domain and per engine. An aggregate that returns
``np.float64`` was always safe because ``np.float64`` subclasses ``float``; the
same aggregate returning ``np.int64`` was not, because ``np.int64`` subclasses
nothing. Which of those you get depends on the column's dtype, so exposure is a
property of the *data*, not of the question -- and that is exactly why it went
unnoticed.

Run:

    python -m evaluation.generalization.serialization_exposure \
        --fixtures evaluation/generalization/generated \
        --baseline 4878627 \
        --output evaluation/generalization/raw-results/serialization-exposure.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

BASELINE_DEFAULT = "4878627"


# ---------------------------------------------------------------------------
# loading the two freeze implementations
# ---------------------------------------------------------------------------
def load_baseline_freeze(repo: Path, rev: str) -> Callable[[Any], Any]:
    """Load ``freeze`` from ``serializers.py`` as it was at ``rev``.

    Loaded from a temporary file rather than imported, so the baseline and the
    current implementation can be held in memory at the same time without
    either shadowing ``fabric_rlm.serializers``.
    """
    src = subprocess.run(
        ["git", "show", f"{rev}:fabric_rlm/serializers.py"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp()) / "baseline_serializers.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_baseline_serializers", tmp)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.freeze


def leaked(frozen: Any) -> bool:
    """True when ``freeze`` produced an opaque marker instead of a value."""
    return isinstance(frozen, dict) and frozen.get("__serializable__") is False


# ---------------------------------------------------------------------------
# the computations
# ---------------------------------------------------------------------------
def _duckdb_cases(root: Path) -> list[tuple[str, str, Callable[[], Any]]]:
    """(question_id, engine, thunk) for duckdb over the CSV fixtures."""
    import duckdb

    def q(sql: str) -> Callable[[], Any]:
        def run() -> Any:
            con = duckdb.connect()
            try:
                return con.execute(sql).fetchone()[0]
            finally:
                con.close()
        return run

    inv = (root / "inventory" / "descriptive").as_posix()
    man = (root / "manufacturing" / "descriptive").as_posix()
    svc = (root / "service" / "descriptive").as_posix()

    return [
        ("inventory_available_units", "duckdb", q(f"""
            SELECT SUM(on_hand_units - allocated_units)
            FROM '{inv}/inventory_snapshots.csv'
            WHERE snapshot_date = (SELECT MAX(snapshot_date)
                                   FROM '{inv}/inventory_snapshots.csv')""")),
        ("inventory_open_order_units", "duckdb", q(f"""
            SELECT SUM(o.ordered_units) - COALESCE(SUM(s.shipped), 0)
            FROM '{inv}/order_lines.csv' o
            LEFT JOIN (SELECT order_id, line_id, SUM(shipped_units) shipped
                       FROM '{inv}/shipment_events.csv'
                       GROUP BY 1, 2) s
              ON o.order_id = s.order_id AND o.line_id = s.line_id""")),
        ("inventory_join_safe_value", "duckdb", q(f"""
            SELECT SUM(i.on_hand_units * p.unit_cost)
            FROM '{inv}/inventory_snapshots.csv' i
            JOIN '{inv}/products.csv' p ON i.product_id = p.product_id
            WHERE i.snapshot_date = (SELECT MAX(snapshot_date)
                                     FROM '{inv}/inventory_snapshots.csv')""")),
        ("inventory_fill_rate", "duckdb", q(f"""
            SELECT (SELECT SUM(shipped_units) FROM '{inv}/shipment_events.csv')
                   * 1.0
                   / (SELECT SUM(ordered_units) FROM '{inv}/order_lines.csv')""")),
        ("inventory_latest_snapshot_date", "duckdb", q(f"""
            SELECT MAX(CAST(snapshot_date AS DATE))
            FROM '{inv}/inventory_snapshots.csv'""")),
        ("inventory_value_decimal", "duckdb", q(f"""
            SELECT SUM(CAST(i.on_hand_units * p.unit_cost AS DECIMAL(18,2)))
            FROM '{inv}/inventory_snapshots.csv' i
            JOIN '{inv}/products.csv' p ON i.product_id = p.product_id""")),

        ("manufacturing_complete_units", "duckdb", q(f"""
            SELECT SUM(produced_units) FROM '{man}/production.csv'
            WHERE reporting_complete""")),
        ("manufacturing_weighted_defect_rate", "duckdb", q(f"""
            SELECT SUM(d.defect_units) * 1.0 / SUM(p.produced_units)
            FROM (SELECT reporting_period, SUM(produced_units) produced_units
                  FROM '{man}/production.csv' WHERE reporting_complete
                  GROUP BY 1) p
            JOIN (SELECT reporting_period, SUM(defect_units) defect_units
                  FROM '{man}/defects.csv' GROUP BY 1) d
              ON p.reporting_period = d.reporting_period""")),
        ("manufacturing_incomplete_excluded", "duckdb", q(f"""
            SELECT SUM(produced_units) FROM '{man}/production.csv'
            WHERE NOT reporting_complete""")),
        ("manufacturing_any_incomplete", "duckdb", q(f"""
            SELECT bool_or(NOT reporting_complete) FROM '{man}/production.csv'""")),

        ("service_reopened_tickets", "duckdb", q(f"""
            SELECT COUNT(DISTINCT ticket_id) FROM '{svc}/ticket_events.csv'
            WHERE event_type = 'reopen'""")),
        ("service_first_response_sla_rate", "duckdb", q(f"""
            WITH fr AS (SELECT ticket_id, MIN(event_at) first_at
                        FROM '{svc}/ticket_events.csv'
                        WHERE event_type = 'agent_response' GROUP BY 1)
            SELECT AVG(CASE WHEN date_diff('minute',
                            CAST(t.opened_at AS TIMESTAMP),
                            CAST(fr.first_at AS TIMESTAMP))
                            <= s.first_response_minutes THEN 1.0 ELSE 0.0 END)
            FROM '{svc}/tickets.csv' t
            JOIN fr ON fr.ticket_id = t.ticket_id
            JOIN '{svc}/sla_policies.csv' s ON s.policy_id = t.policy_id""")),
        ("service_first_response_dedup", "duckdb", q(f"""
            SELECT MIN(CAST(event_at AS TIMESTAMP))
                   - CAST((SELECT opened_at FROM '{svc}/tickets.csv'
                           WHERE ticket_id = 'T-001') AS TIMESTAMP)
            FROM '{svc}/ticket_events.csv'
            WHERE ticket_id = 'T-001' AND event_type = 'agent_response'""")),
        ("service_large_history_rows", "duckdb", q(f"""
            SELECT COUNT(*) FROM '{svc}/ticket_event_history_large.csv'""")),
    ]


def _pandas_cases(root: Path) -> list[tuple[str, str, Callable[[], Any]]]:
    """(question_id, engine, thunk) for pandas over the same fixtures."""
    import pandas as pd

    inv = root / "inventory" / "descriptive"
    man = root / "manufacturing" / "descriptive"
    svc = root / "service" / "descriptive"

    def available_units() -> Any:
        df = pd.read_csv(inv / "inventory_snapshots.csv")
        latest = df[df["snapshot_date"] == df["snapshot_date"].max()]
        return (latest["on_hand_units"] - latest["allocated_units"]).sum()

    def open_units() -> Any:
        o = pd.read_csv(inv / "order_lines.csv")
        s = pd.read_csv(inv / "shipment_events.csv")
        shipped = s.groupby(["order_id", "line_id"])["shipped_units"].sum()
        merged = o.join(shipped, on=["order_id", "line_id"])
        return o["ordered_units"].sum() - merged["shipped_units"].fillna(0).sum()

    def join_safe_value() -> Any:
        i = pd.read_csv(inv / "inventory_snapshots.csv")
        p = pd.read_csv(inv / "products.csv")
        latest = i[i["snapshot_date"] == i["snapshot_date"].max()]
        m = latest.merge(p, on="product_id")
        return (m["on_hand_units"] * m["unit_cost"]).sum()

    def fill_rate() -> Any:
        o = pd.read_csv(inv / "order_lines.csv")
        s = pd.read_csv(inv / "shipment_events.csv")
        return s["shipped_units"].sum() / o["ordered_units"].sum()

    def latest_snapshot_date() -> Any:
        df = pd.read_csv(inv / "inventory_snapshots.csv",
                         parse_dates=["snapshot_date"])
        return df["snapshot_date"].max()

    def complete_units() -> Any:
        df = pd.read_csv(man / "production.csv")
        return df.loc[df["reporting_complete"], "produced_units"].sum()

    def weighted_defect_rate() -> Any:
        p = pd.read_csv(man / "production.csv")
        d = pd.read_csv(man / "defects.csv")
        ok = set(p.loc[p["reporting_complete"], "reporting_period"])
        return (d[d["reporting_period"].isin(ok)]["defect_units"].sum()
                / p.loc[p["reporting_complete"], "produced_units"].sum())

    def incomplete_excluded() -> Any:
        df = pd.read_csv(man / "production.csv")
        return df.loc[~df["reporting_complete"], "produced_units"].sum()

    def any_incomplete() -> Any:
        df = pd.read_csv(man / "production.csv")
        return (~df["reporting_complete"]).any()

    def reopened() -> Any:
        e = pd.read_csv(svc / "ticket_events.csv")
        return e[e["event_type"] == "reopen"]["ticket_id"].nunique()

    def first_response_dedup() -> Any:
        t = pd.read_csv(svc / "tickets.csv", parse_dates=["opened_at"])
        e = pd.read_csv(svc / "ticket_events.csv", parse_dates=["event_at"])
        first = e[(e["ticket_id"] == "T-001")
                  & (e["event_type"] == "agent_response")]["event_at"].min()
        opened = t.loc[t["ticket_id"] == "T-001", "opened_at"].iloc[0]
        return first - opened

    def sla_rate() -> Any:
        t = pd.read_csv(svc / "tickets.csv", parse_dates=["opened_at"])
        e = pd.read_csv(svc / "ticket_events.csv", parse_dates=["event_at"])
        s = pd.read_csv(svc / "sla_policies.csv")
        first = (e[e["event_type"] == "agent_response"]
                 .groupby("ticket_id")["event_at"].min().rename("first_at"))
        m = t.join(first, on="ticket_id").merge(s, on="policy_id")
        mins = (m["first_at"] - m["opened_at"]).dt.total_seconds() / 60
        return (mins <= m["first_response_minutes"]).mean()

    def large_history_rows() -> Any:
        df = pd.read_csv(svc / "ticket_event_history_large.csv")
        return len(df["ticket_id"].unique())

    return [
        ("inventory_available_units", "pandas", available_units),
        ("inventory_open_order_units", "pandas", open_units),
        ("inventory_join_safe_value", "pandas", join_safe_value),
        ("inventory_fill_rate", "pandas", fill_rate),
        ("inventory_latest_snapshot_date", "pandas", latest_snapshot_date),
        ("manufacturing_complete_units", "pandas", complete_units),
        ("manufacturing_weighted_defect_rate", "pandas", weighted_defect_rate),
        ("manufacturing_incomplete_excluded", "pandas", incomplete_excluded),
        ("manufacturing_any_incomplete", "pandas", any_incomplete),
        ("service_reopened_tickets", "pandas", reopened),
        ("service_first_response_dedup", "pandas", first_response_dedup),
        ("service_first_response_sla_rate", "pandas", sla_rate),
        ("service_large_history_rows", "pandas", large_history_rows),
    ]


def domain_of(question_id: str) -> str:
    return question_id.split("_", 1)[0]


def run(fixtures: Path, repo: Path, baseline_rev: str) -> dict:
    from fabric_rlm.serializers import freeze as freeze_now

    freeze_before = load_baseline_freeze(repo, baseline_rev)

    cases = _duckdb_cases(fixtures) + _pandas_cases(fixtures)
    rows = []
    for question_id, engine, thunk in cases:
        try:
            value = thunk()
        except Exception as exc:  # noqa: BLE001
            rows.append({"question_id": question_id, "engine": engine,
                         "error": f"{type(exc).__name__}: {exc}"})
            continue
        before, after = freeze_before(value), freeze_now(value)
        rows.append({
            "question_id": question_id,
            "domain": domain_of(question_id),
            "engine": engine,
            "python_type": type(value).__name__,
            "repr": repr(value)[:80],
            "leaked_before": leaked(before),
            "leaked_after": leaked(after),
            "frozen_after": after if not leaked(after) else None,
        })

    ok = [r for r in rows if "error" not in r]
    by_domain: dict[str, dict] = {}
    for r in ok:
        d = by_domain.setdefault(r["domain"],
                                 {"cases": 0, "leaked_before": 0, "leaked_after": 0})
        d["cases"] += 1
        d["leaked_before"] += int(r["leaked_before"])
        d["leaked_after"] += int(r["leaked_after"])
    by_engine: dict[str, dict] = {}
    for r in ok:
        d = by_engine.setdefault(r["engine"],
                                 {"cases": 0, "leaked_before": 0, "leaked_after": 0})
        d["cases"] += 1
        d["leaked_before"] += int(r["leaked_before"])
        d["leaked_after"] += int(r["leaked_after"])

    return {
        "baseline_rev": baseline_rev,
        "cases": len(rows),
        "errors": [r for r in rows if "error" in r],
        "totals": {
            "cases": len(ok),
            "leaked_before": sum(int(r["leaked_before"]) for r in ok),
            "leaked_after": sum(int(r["leaked_after"]) for r in ok),
        },
        "by_domain": by_domain,
        "by_engine": by_engine,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path,
                    default=Path("evaluation/generalization/generated"))
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--baseline", default=BASELINE_DEFAULT)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    report = run(args.fixtures, args.repo, args.baseline)

    t = report["totals"]
    print(f"cases {t['cases']}   leaked before {t['leaked_before']}"
          f"   leaked after {t['leaked_after']}")
    print("\nby domain")
    for name, d in sorted(report["by_domain"].items()):
        print(f"  {name:15s} {d['leaked_before']:>2}/{d['cases']:<3} -> "
              f"{d['leaked_after']}/{d['cases']}")
    print("\nby engine")
    for name, d in sorted(report["by_engine"].items()):
        print(f"  {name:15s} {d['leaked_before']:>2}/{d['cases']:<3} -> "
              f"{d['leaked_after']}/{d['cases']}")
    print("\nleaked before the fix")
    for r in report["rows"]:
        if r.get("leaked_before"):
            print(f"  {r['engine']:7s} {r['question_id']:38s} "
                  f"{r['python_type']:18s} {r['repr'][:32]}")
    if report["errors"]:
        print("\nerrors")
        for r in report["errors"]:
            print(f"  {r['engine']:7s} {r['question_id']:38s} {r['error'][:90]}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str),
                               encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
