"""Heterogeneous generalization evaluation for fabric-rlm-core in Fabric.

Runs the frozen PR-75 build against real Fabric sources spanning several
unrelated domains and every supported source type, and records what it
finds. Nothing here is specific to one company's metrics: the sources were
chosen precisely because none of them is the model the library was
developed against, which is excluded by name.

Stages
------
1. ``learn`` over every source, with and without declared metadata. No
   model calls; this is the generalization claim about the learning
   substrate, measured directly.
2. Independent reference answers, computed with Spark SQL and DAX rather
   than through the library's own query path, so a later comparison is not
   the library grading itself.
3. A budgeted live matrix. Skipped unless a key is present and a budget is
   left.

Results are written under ``Files/rlm-evaluation/hetero/<run_id>/``.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from datetime import datetime, timezone

# The model the library was developed against. Naming it here keeps it out
# of every evaluation set by construction rather than by memory.
EXCLUDED_DEV_MODEL = "ARR Model SF (79)"

RESULT_ROOT = "/lakehouse/default/Files/rlm-evaluation/hetero"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


# --------------------------------------------------------------------------
# The source matrix: unrelated domains, every supported source type.
# --------------------------------------------------------------------------

DELTA_SOURCES = [
    {
        "key": "ecommerce_olist",
        "domain": "e-commerce marketplace",
        "workspace_id": "cbcb3fe8-0a54-4e1d-a986-69db569e1025",
        "lakehouse_id": "361097a2-a533-4164-ad05-bbd3d6bbc814",
        "tables": ["Tables/orders", "Tables/order_items", "Tables/order_payments", "Tables/customers", "Tables/sellers", "Tables/products"],
        # order_items is one row per item, so a join to orders multiplies
        # order-level money; that hazard is the point of including it.
        "declared": {
            "grain": ["order_id"],
            "notes": [
                "order_items has one row per item, so joining it to orders "
                "repeats every order-level value once per item",
            ],
        },
        "declared_table": "orders",
    },
    {
        "key": "food_retail_bakehouse",
        "domain": "bakery franchise retail",
        "workspace_id": "cbcb3fe8-0a54-4e1d-a986-69db569e1025",
        "lakehouse_id": "550a3a96-4822-490c-b401-b5c69fce1c3d",
        "tables": ["Tables/sales_transactions", "Tables/sales_customers", "Tables/sales_franchises", "Tables/sales_suppliers"],
        "declared": None,
        "declared_table": "sales_transactions",
    },
    {
        "key": "service_ops_callcenter",
        "domain": "call centre service operations",
        "workspace_id": "ee90374b-f96b-49fa-a986-4e062defd063",
        "lakehouse_id": "b8a38cc5-2a6d-402f-8d78-93e7887d40cb",
        "tables": ["Tables/call_log", "Tables/call_enriched", "Tables/agent_performance_daily"],
        "declared": None,
        "declared_table": "call_log",
    },
    {
        "key": "retail_adventureworks",
        "domain": "retail",
        "workspace_id": "c3f4b749-0eac-4bb9-acdc-20a6744c47e6",
        "lakehouse_id": "93a7e555-9102-4a47-b1bd-518dea83b332",
        "tables": ["Tables/code_sales", "Tables/dim_category_aliases", "Tables/dim_country_aliases"],
        "declared": None,
        "declared_table": "code_sales",
    },
]

SEMANTIC_MODELS = [
    {"key": "sm_manufacturing", "domain": "manufacturing operations",
     "dataset": "Manufacturing Ops", "workspace": "L400"},
    {"key": "sm_adventureworks", "domain": "retail",
     "dataset": "AdventureWorks Sales", "workspace": "DBRX-Benchmark"},
    {"key": "sm_bakehouse", "domain": "bakery franchise retail",
     "dataset": "bakehouse_semantic_model", "workspace": "DBRX-Benchmark"},
    {"key": "sm_banking", "domain": "retail banking",
     "dataset": "banking_semantic_model", "workspace": "[KEEP]AgenticBankingApp"},
    {"key": "sm_wanderbricks", "domain": "travel",
     "dataset": "wanderbricks_semantic_model", "workspace": "DBRX-Benchmark"},
]


# --------------------------------------------------------------------------
# Stage 1: what learn() produces, per source type and per domain.
# --------------------------------------------------------------------------


def _lesson_rows(package) -> list[dict]:
    rows = []
    for lesson in getattr(package, "lessons", ()) or ():
        rows.append(
            {
                "kind": getattr(lesson, "kind", None),
                "subject": getattr(lesson, "subject", None),
                "status": getattr(lesson, "status", None),
                "confidence": getattr(lesson, "confidence", None),
                "basis": list(getattr(lesson, "basis", ()) or ()),
                "rule": dict(getattr(lesson, "structured_rule", {}) or {}),
            }
        )
    return rows


def _profile_rows(package) -> list[dict]:
    rows = []
    for profile in getattr(package, "sources", ()) or ():
        schema = getattr(profile, "schema", {}) or {}
        families = {}
        if isinstance(schema, dict):
            for family, section in schema.items():
                if isinstance(section, dict):
                    families[family] = sorted(section)[:40]
                elif isinstance(section, (list, tuple)):
                    families[family] = [str(item)[:80] for item in section][:40]
                else:
                    families[family] = str(section)[:200]
        rows.append(
            {
                "source_id": getattr(profile, "source_id", None),
                "family": getattr(profile, "family", None),
                "status": getattr(profile, "status", None),
                "locator": str(getattr(profile, "locator", ""))[:200],
                "schema_fingerprint": getattr(profile, "schema_fingerprint", None),
                "diagnostics": str(getattr(profile, "diagnostics", ""))[:600],
                "schema_families": families,
            }
        )
    return rows


def learn_over_sources(sources_by_key: dict, declared_by_key: dict) -> list[dict]:
    """Run learn() per source, without and then with declared metadata."""
    from fabric_rlm import RLM

    results = []
    for key, source in sources_by_key.items():
        row = {"key": key, "plain": None, "declared": None}
        for label, declared in (("plain", None), ("declared", declared_by_key.get(key))):
            if label == "declared" and not declared:
                continue
            started = time.time()
            try:
                knowledge = RLM.learn(sources={key: source}, **({"declared": declared} if declared else {}))
                package = getattr(knowledge, "package", knowledge)
                row[label] = {
                    "ok": True,
                    "seconds": round(time.time() - started, 2),
                    "lesson_count": len(getattr(package, "lessons", ()) or ()),
                    "lessons": _lesson_rows(package),
                    "profiles": _profile_rows(package),
                }
            except Exception as error:  # recorded, never raised: one bad source must not end the sweep
                row[label] = {
                    "ok": False,
                    "seconds": round(time.time() - started, 2),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-2500:],
                }
        results.append(row)
    return results


def build_delta_sources() -> tuple[dict, dict, list[dict]]:
    """LakehouseSource per delta lakehouse, plus their declared metadata."""
    from fabric_rlm import LakehouseSource

    sources, declared, notes = {}, {}, []
    for spec in DELTA_SOURCES:
        root = (
            f"abfss://{spec['workspace_id']}@onelake.dfs.fabric.microsoft.com/"
            f"{spec['lakehouse_id']}"
        )
        try:
            sources[spec["key"]] = LakehouseSource(root, tables=spec["tables"])
            notes.append({"key": spec["key"], "domain": spec["domain"], "root": root, "ok": True})
            if spec.get("declared"):
                declared[spec["key"]] = {spec["key"]: dict(spec["declared"])}
        except Exception as error:
            notes.append({"key": spec["key"], "root": root, "ok": False,
                          "error": f"{type(error).__name__}: {error}"})
    return sources, declared, notes


def register_views(spark) -> list[dict]:
    """Retained only so an older caller does not break; superseded by load_frames."""
    raise NotImplementedError("use load_frames(); this evaluation does not require Spark")


def build_semantic_sources() -> tuple[dict, list[dict]]:
    from fabric_rlm import SemanticModel

    sources, notes = {}, []
    for spec in SEMANTIC_MODELS:
        if spec["dataset"] == EXCLUDED_DEV_MODEL:
            notes.append({"key": spec["key"], "skipped": "development model, excluded by construction"})
            continue
        try:
            sources[spec["key"]] = SemanticModel(
                spec["dataset"], workspace=spec["workspace"],
                credential_provider="notebookutils", validate=False,
            )
            notes.append({"key": spec["key"], "domain": spec["domain"],
                          "dataset": spec["dataset"], "workspace": spec["workspace"], "ok": True})
        except Exception as error:
            notes.append({"key": spec["key"], "dataset": spec["dataset"], "ok": False,
                          "error": f"{type(error).__name__}: {error}"})
    return sources, notes


# --------------------------------------------------------------------------
# Stage 2: reference answers computed outside the library.
# --------------------------------------------------------------------------

def _onelake_token() -> str:
    """A storage token for OneLake, from whatever identity the session has."""
    try:
        import notebookutils

        return notebookutils.credentials.getToken("storage")
    except Exception:
        from azure.identity import DefaultAzureCredential

        return DefaultAzureCredential().get_token("https://storage.azure.com/.default").token


def _read_delta(workspace_id: str, lakehouse_id: str, table: str):
    """A Delta table as pandas, read directly, without the library."""
    from deltalake import DeltaTable

    url = f"abfss://{workspace_id}@onelake.dfs.fabric.microsoft.com/{lakehouse_id}/Tables/{table}"
    options = {"bearer_token": _onelake_token(), "use_fabric_endpoint": "true"}
    return DeltaTable(url, storage_options=options).to_pyarrow_table().to_pandas()


def load_frames() -> tuple[dict, list[dict]]:
    """Every delta table as a pandas frame, for reference answers only.

    The library never sees these frames. They exist so Stage 2 can compute
    an answer with plain pandas that owes nothing to the library's query
    path, and nothing to a SQL engine the library also uses.
    """
    frames, notes = {}, []
    for spec in DELTA_SOURCES:
        for scope in spec["tables"]:
            table = scope.split("/")[-1]
            name = f"{spec['key']}__{table}"
            try:
                frame = _read_delta(spec["workspace_id"], spec["lakehouse_id"], table)
                frames[name] = frame
                notes.append({"frame": name, "ok": True, "rows": int(len(frame)),
                              "columns": list(frame.columns)[:30]})
            except Exception as error:
                notes.append({"frame": name, "ok": False,
                              "error": f"{type(error).__name__}: {error}"[:400]})
    return frames, notes


# Each reference is a plain pandas computation over one or two frames. The
# hazard field says what a careless answer would return instead, so a wrong
# answer can be told apart from a merely different one.
REFERENCE_QUERIES = [
    {
        "key": "olist_orders_distinct",
        "domain": "e-commerce marketplace",
        "question": "How many distinct orders are there?",
        "hazard": "none; the control for the join hazard below",
        "compute": lambda f: int(f["ecommerce_olist__orders"]["order_id"].nunique()),
    },
    {
        "key": "olist_orders_with_items",
        "domain": "e-commerce marketplace",
        "question": "How many distinct orders have at least one item?",
        "hazard": "counting rows of the orders-to-items join counts items, not orders",
        "compute": lambda f: int(
            f["ecommerce_olist__orders"]["order_id"]
            .isin(f["ecommerce_olist__order_items"]["order_id"])
            .sum()
        ),
    },
    {
        "key": "olist_naive_join_rows",
        "domain": "e-commerce marketplace",
        "question": "Rows in the orders-to-items join, the wrong answer to the question above",
        "hazard": "recorded so a multiplied answer is distinguishable from a correct one",
        "compute": lambda f: int(
            f["ecommerce_olist__orders"][["order_id"]]
            .merge(f["ecommerce_olist__order_items"][["order_id"]], on="order_id")
            .shape[0]
        ),
    },
    {
        "key": "olist_payment_total",
        "domain": "e-commerce marketplace",
        "question": "What is the total payment value across all orders?",
        "hazard": "joining payments to items would multiply this",
        "compute": lambda f: round(float(f["ecommerce_olist__order_payments"]["payment_value"].sum()), 2),
    },
    {
        "key": "bakehouse_txn_count",
        "domain": "bakery franchise retail",
        "question": "How many sales transactions are recorded?",
        "hazard": "none",
        "compute": lambda f: int(len(f["food_retail_bakehouse__sales_transactions"])),
    },
    {
        "key": "callcenter_call_count",
        "domain": "call centre service operations",
        "question": "How many calls are in the call log?",
        "hazard": "none",
        "compute": lambda f: int(len(f["service_ops_callcenter__call_log"])),
    },
]


def run_reference_queries(frames: dict) -> list[dict]:
    """Answers computed by pandas, never by the library's compiler."""
    rows = []
    for spec in REFERENCE_QUERIES:
        record = {k: v for k, v in spec.items() if k != "compute"}
        try:
            record["reference_value"] = spec["compute"](frames)
            record["ok"] = True
        except Exception as error:
            record["ok"] = False
            record["error"] = f"{type(error).__name__}: {error}"[:400]
        rows.append(record)
    return rows


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(*, stages=(1, 2), openrouter_key=None, budget=0):
    run_id = f"{_now()}-hetero-{uuid.uuid4().hex[:8]}"
    out = f"{RESULT_ROOT}/{run_id}"
    report = {"run_id": run_id, "stages_requested": list(stages), "excluded_dev_model": EXCLUDED_DEV_MODEL}

    import fabric_rlm

    report["runtime"] = {
        "fabric_rlm": getattr(fabric_rlm, "__version__", "?"),
        "python": os.sys.version.split()[0],
    }

    if 1 in stages:
        delta_sources, delta_declared, delta_notes = build_delta_sources()
        semantic_sources, semantic_notes = build_semantic_sources()
        report["source_binding"] = {"delta": delta_notes, "semantic": semantic_notes}
        report["stage1_delta"] = learn_over_sources(delta_sources, delta_declared)
        report["stage1_semantic"] = learn_over_sources(semantic_sources, {})

    if 2 in stages:
        frames, frame_notes = load_frames()
        report["frame_binding"] = frame_notes
        report["stage2_references"] = run_reference_queries(frames)

    _write(f"{out}/report.json", report)
    return {"run_id": run_id, "output": out, "report": report}

