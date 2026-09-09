"""Profile every Delta table under da_agent_tests.Lakehouse/Tables/dbo.

Reads Delta directly from OneLake. Nothing here touches fabric_rlm, so the
resulting profile is independent of the library under test.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from deltalake import DeltaTable

ROOT = "abfss://sandeep_ws@onelake.dfs.fabric.microsoft.com/da_agent_tests.Lakehouse/Tables/dbo"
OUT = Path(__file__).parent / "profile.json"
CACHE = Path(__file__).parent / "cache"


def token() -> str:
    return subprocess.run(
        ["az", "account", "get-access-token", "--resource",
         "https://storage.azure.com", "--query", "accessToken", "-o", "tsv"],
        capture_output=True, text=True, check=True, shell=True,
    ).stdout.strip()


def storage_options() -> dict[str, str]:
    return {"bearer_token": token(), "use_fabric_endpoint": "true"}


TABLES = [
    "companies", "dim_date", "features", "industries", "invoices", "payments",
    "rlm_eval_defects", "rlm_eval_inventory_snapshots", "rlm_eval_order_lines",
    "rlm_eval_production", "rlm_eval_products", "rlm_eval_shipment_events",
    "rlm_eval_sla_policies", "rlm_eval_ticket_events", "rlm_eval_ticket_sla",
    "rlm_eval_tickets", "subscriptions", "support_tickets", "usage_logs", "users",
]


def main() -> int:
    CACHE.mkdir(exist_ok=True)
    opts = storage_options()
    profile: dict[str, object] = {}

    for name in TABLES:
        try:
            dt = DeltaTable(f"{ROOT}/{name}", storage_options=opts)
            df = dt.to_pandas()
        except Exception as exc:  # noqa: BLE001
            profile[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        # Cache locally so ground-truth computation never re-downloads.
        df.to_parquet(CACHE / f"{name}.parquet", index=False)

        cols = []
        for c in df.columns:
            s = df[c]
            info = {
                "name": c,
                "dtype": str(s.dtype),
                "nulls": int(s.isna().sum()),
                "distinct": int(s.nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                info["min"] = None if s.dropna().empty else float(s.min())
                info["max"] = None if s.dropna().empty else float(s.max())
            elif s.dtype == object or str(s.dtype).startswith("string"):
                info["samples"] = [str(v) for v in s.dropna().unique()[:5]]
            else:
                info["min"] = str(s.min()) if not s.dropna().empty else None
                info["max"] = str(s.max()) if not s.dropna().empty else None
            cols.append(info)

        profile[name] = {"rows": int(len(df)), "columns": cols}
        print(f"  OK   {name:34} {len(df):>9,} rows  {len(df.columns):>3} cols", flush=True)

    OUT.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
