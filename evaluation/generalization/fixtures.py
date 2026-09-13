from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Mapping


SEED = 20260908
PROMPT_BUDGET_BYTES = 1_000_000
VARIANTS = ("descriptive", "abbreviated", "camel")

_TABLES: dict[str, dict[str, list[dict[str, object]]]] = {
    "inventory": {
        "inventory_snapshots": [
            {"snapshot_date": "2026-02-28", "warehouse_id": "W-01", "product_id": "P-001", "on_hand_units": 90, "allocated_units": 15},
            {"snapshot_date": "2026-03-31", "warehouse_id": "W-01", "product_id": "P-001", "on_hand_units": 100, "allocated_units": 20},
            {"snapshot_date": "2026-03-31", "warehouse_id": "W-01", "product_id": "P-002", "on_hand_units": 80, "allocated_units": 10},
            {"snapshot_date": "2026-03-31", "warehouse_id": "W-02", "product_id": "P-001", "on_hand_units": 50, "allocated_units": 14},
        ],
        "order_lines": [
            {"order_id": "O-001", "line_id": "1", "customer_id": "C-001", "customer_name": "Northwind", "product_id": "P-001", "product_name": "Widget", "ordered_units": 100},
            {"order_id": "O-002", "line_id": "1", "customer_id": "C-002", "customer_name": "Northwind", "product_id": "P-002", "product_name": "Widget", "ordered_units": 80},
            {"order_id": "O-003", "line_id": "1", "customer_id": "C-002", "customer_name": "Northwind", "product_id": "P-001", "product_name": "Widget Pro", "ordered_units": 100},
            {"order_id": "O-004", "line_id": "1", "customer_id": "C-003", "customer_name": "Contoso", "product_id": "P-002", "product_name": "Widget", "ordered_units": 30},
        ],
        "shipment_events": [
            {"shipment_id": "S-001", "order_id": "O-001", "line_id": "1", "event_date": "2026-03-05", "shipped_units": 60},
            {"shipment_id": "S-002", "order_id": "O-001", "line_id": "1", "event_date": "2026-03-12", "shipped_units": 20},
            {"shipment_id": "S-003", "order_id": "O-002", "line_id": "1", "event_date": "2026-03-09", "shipped_units": 80},
            {"shipment_id": "S-004", "order_id": "O-003", "line_id": "1", "event_date": "2026-03-15", "shipped_units": 90},
            {"shipment_id": "S-005", "order_id": "O-004", "line_id": "1", "event_date": "2026-03-18", "shipped_units": 12},
        ],
        "products": [
            {"product_id": "P-001", "product_name": "Widget", "unit_cost": 9.0},
            {"product_id": "P-002", "product_name": "Widget", "unit_cost": 11.25},
        ],
    },
    "manufacturing": {
        "production": [
            {"reporting_period": "2026-01", "line_id": "LN-A", "product_id": "SKU-1", "produced_units": 1000, "reporting_complete": True},
            {"reporting_period": "2026-01", "line_id": "LN-B", "product_id": "SKU-2", "produced_units": 800, "reporting_complete": True},
            {"reporting_period": "2026-02", "line_id": "LN-A", "product_id": "SKU-1", "produced_units": 200, "reporting_complete": False},
        ],
        "defects": [
            {"reporting_period": "2026-01", "line_id": "LN-A", "defect_type": "surface", "defect_units": 10},
            {"reporting_period": "2026-01", "line_id": "LN-A", "defect_type": "dimension", "defect_units": 5},
            {"reporting_period": "2026-01", "line_id": "LN-B", "defect_type": "surface", "defect_units": 18},
            {"reporting_period": "2026-01", "line_id": "LN-B", "defect_type": "dimension", "defect_units": 12},
            {"reporting_period": "2026-02", "line_id": "LN-A", "defect_type": "surface", "defect_units": 20},
        ],
    },
    "service": {
        "tickets": [
            {"ticket_id": "T-001", "customer_id": "C-101", "opened_at": "2026-03-01T08:00:00Z", "priority": "high", "policy_id": "P-H", "closed_at": "2026-03-01T12:00:00Z"},
            {"ticket_id": "T-002", "customer_id": "C-102", "opened_at": "2026-03-01T09:00:00Z", "priority": "normal", "policy_id": "P-N", "closed_at": "2026-03-02T10:00:00Z"},
            {"ticket_id": "T-003", "customer_id": "C-102", "opened_at": "2026-03-02T10:00:00Z", "priority": "normal", "policy_id": "P-N", "closed_at": "2026-03-03T10:00:00Z"},
        ],
        "ticket_events": [
            {"event_id": "E-001", "ticket_id": "T-001", "event_type": "agent_response", "event_at": "2026-03-01T08:20:00Z"},
            {"event_id": "E-002", "ticket_id": "T-001", "event_type": "agent_response", "event_at": "2026-03-01T08:35:00Z"},
            {"event_id": "E-003", "ticket_id": "T-002", "event_type": "agent_response", "event_at": "2026-03-01T11:30:00Z"},
            {"event_id": "E-004", "ticket_id": "T-002", "event_type": "reopened", "event_at": "2026-03-02T12:00:00Z"},
            {"event_id": "E-005", "ticket_id": "T-003", "event_type": "agent_response", "event_at": "2026-03-02T10:45:00Z"},
        ],
        "sla_policies": [
            {"policy_id": "P-H", "priority": "high", "first_response_minutes": 30},
            {"policy_id": "P-N", "priority": "normal", "first_response_minutes": 60},
        ],
    },
}

_ABBREVIATIONS = {
    "snapshot_date": "snap_dt", "warehouse_id": "wh_id", "product_id": "prd_id",
    "on_hand_units": "oh_qty", "allocated_units": "alloc_qty", "order_id": "ord_id",
    "line_id": "ln_id", "customer_id": "cust_id", "customer_name": "cust_nm",
    "product_name": "prd_nm", "ordered_units": "ord_qty", "shipment_id": "shp_id",
    "event_date": "evt_dt", "shipped_units": "shp_qty", "unit_cost": "u_cost",
    "reporting_period": "rpt_pd", "produced_units": "prod_qty",
    "reporting_complete": "rpt_ok", "defect_type": "dfct_cd",
    "defect_units": "dfct_qty", "ticket_id": "tkt_id", "opened_at": "opn_ts",
    "priority": "pri_cd", "policy_id": "pol_id", "closed_at": "cls_ts",
    "event_id": "evt_id", "event_type": "evt_cd", "event_at": "evt_ts",
    "first_response_minutes": "fr_mins",
}


def _camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part.capitalize() for part in tail)


def _field_name(name: str, variant: str) -> str:
    if variant == "descriptive":
        return name
    if variant == "abbreviated":
        return _ABBREVIATIONS.get(name, name)
    return _camel(name)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], variant: str) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(materialized[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[_field_name(name, variant) for name in fields])
        writer.writeheader()
        for row in materialized:
            writer.writerow({_field_name(name, variant): row[name] for name in fields})


def _questions() -> list[dict[str, object]]:
    return [
        {"question_id": "inventory_available_units", "domain": "inventory", "text": "How many units were available at the latest inventory snapshot?", "provided_definitions": ["available units", "latest inventory snapshot"], "expected_behavior": "answer"},
        {"question_id": "inventory_open_order_units", "domain": "inventory", "text": "How many ordered units remain unshipped after all partial shipments?", "provided_definitions": ["open ordered units", "partial shipments"], "expected_behavior": "answer"},
        {"question_id": "inventory_top_customer", "domain": "inventory", "text": "Which customer identity received the most shipped units?", "provided_definitions": ["customer identity", "shipped units"], "expected_behavior": "answer"},
        {"question_id": "inventory_join_safe_value", "domain": "inventory", "text": "What is the inventory value at the latest snapshot without multiplying values through joins?", "provided_definitions": ["inventory value", "latest inventory snapshot", "join grain"], "expected_behavior": "answer"},
        {"question_id": "inventory_fill_rate", "domain": "inventory", "text": "What share of ordered units has shipped?", "provided_definitions": ["fill rate"], "expected_behavior": "answer"},
        {"question_id": "manufacturing_complete_units", "domain": "manufacturing", "text": "How many units were produced in complete reporting periods?", "provided_definitions": ["complete reporting period"], "expected_behavior": "answer"},
        {"question_id": "manufacturing_weighted_defect_rate", "domain": "manufacturing", "text": "What is the weighted defect rate for complete periods?", "provided_definitions": ["weighted defect rate", "complete reporting period"], "expected_behavior": "answer"},
        {"question_id": "manufacturing_worst_line", "domain": "manufacturing", "text": "Which production line has the highest defect rate in complete periods?", "provided_definitions": ["line defect rate", "complete reporting period"], "expected_behavior": "answer"},
        {"question_id": "manufacturing_unweighted_trap", "domain": "manufacturing", "text": "What is the correct overall defect rate rather than the average of line rates?", "provided_definitions": ["overall defect rate"], "expected_behavior": "answer"},
        {"question_id": "manufacturing_incomplete_excluded", "domain": "manufacturing", "text": "How many reported units must be excluded because their period is incomplete?", "provided_definitions": ["complete reporting period"], "expected_behavior": "answer"},
        {"question_id": "service_first_response_sla_rate", "domain": "service", "text": "What share of tickets met their first-response SLA?", "provided_definitions": ["first response", "SLA met"], "expected_behavior": "answer"},
        {"question_id": "service_reopened_tickets", "domain": "service", "text": "How many distinct tickets were reopened?", "provided_definitions": ["reopened ticket"], "expected_behavior": "answer"},
        {"question_id": "service_breached_ticket", "domain": "service", "text": "Which ticket identities breached first-response SLA?", "provided_definitions": ["first response", "SLA breached"], "expected_behavior": "answer"},
        {"question_id": "service_first_response_dedup", "domain": "service", "text": "What is the first response time for T-001 despite repeated response events?", "provided_definitions": ["first response"], "expected_behavior": "answer"},
        {"question_id": "service_ambiguous_root_cause", "domain": "service", "text": "What root cause explains the SLA misses?", "provided_definitions": ["first response", "SLA breached"], "expected_behavior": "abstain_or_request_definition"},
    ]


def _definitions() -> dict[str, object]:
    canonical = {
        "inventory": {
            "available units": "At the latest snapshot only, sum on_hand_units minus allocated_units.",
            "open ordered units": "At order-line grain, ordered_units minus the sum of all shipment events, floored at zero.",
            "fill rate": "Total shipped units divided by total ordered units at order-line grain.",
            "inventory value": "At latest snapshot product grain, on_hand_units multiplied once by unit_cost.",
            "customer identity": "Use customer_id; customer_name is not unique.",
        },
        "manufacturing": {
            "complete reporting period": "Include only production rows where reporting_complete is true and matching defects.",
            "weighted defect rate": "Sum defect_units divided by sum produced_units; do not average row or line rates.",
            "line defect rate": "For each line, total defects divided by total produced units in complete periods.",
        },
        "service": {
            "first response": "Earliest agent_response event after ticket opened_at.",
            "SLA met": "First response minutes is less than or equal to the policy threshold.",
            "SLA breached": "First response is absent or exceeds the policy threshold.",
            "reopened ticket": "A distinct ticket with at least one reopened event.",
        },
    }
    variants: dict[str, object] = {}
    for variant in VARIANTS:
        variants[variant] = {}
        for domain, tables in _TABLES.items():
            fields = sorted({field for rows in tables.values() for field in rows[0]})
            variants[variant][domain] = {
                "field_mappings": {
                    _field_name(field, variant): field for field in fields
                },
                "definitions": canonical[domain],
            }
    return {"canonical_definitions": canonical, "variants": variants}


def _digest_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def generate_fixtures(
    output_dir: str | Path,
    *,
    seed: int = SEED,
    large_rows: int = 250_000,
) -> dict[str, object]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for domain, tables in _TABLES.items():
        for variant in VARIANTS:
            for table, rows in tables.items():
                _write_csv(root / domain / variant / f"{table}.csv", rows, variant)

    rng = random.Random(seed)
    large_path = root / "service" / "descriptive" / "ticket_event_history_large.csv"
    large_rows_data = (
        {
            "event_id": f"L-{index:08d}",
            "ticket_id": f"T-{1000 + index % 5000:05d}",
            "event_type": ("note", "status_change", "assignment")[index % 3],
            "event_at": f"2026-03-{1 + index % 28:02d}T{index % 24:02d}:{index % 60:02d}:00Z",
            "event_payload": f"seeded-{rng.randrange(10**12):012d}-" + ("x" * 48),
        }
        for index in range(large_rows)
    )
    _write_csv(large_path, large_rows_data, "descriptive")
    (root / "definitions.json").write_text(
        json.dumps(_definitions(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "questions.json").write_text(
        json.dumps(_questions(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "seed": seed,
        "variants": list(VARIANTS),
        "large_fixture": {
            "path": large_path.relative_to(root).as_posix(),
            "rows": large_rows,
            "prompt_budget_bytes": PROMPT_BUDGET_BYTES,
        },
        "content_sha256": _digest_tree(root),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["PROMPT_BUDGET_BYTES", "SEED", "VARIANTS", "generate_fixtures"]
