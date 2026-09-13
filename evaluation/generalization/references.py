from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .fixtures import VARIANTS


def _rows(root: Path, domain: str, variant: str, table: str, mappings: dict[str, str]) -> list[dict[str, str]]:
    with (root / domain / variant / f"{table}.csv").open(encoding="utf-8", newline="") as handle:
        return [
            {mappings.get(key, key): value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _answered(value: Any, units: str, grain: str, period: str, **extra: Any) -> dict[str, Any]:
    return {
        "expected_status": "answered",
        "value": value,
        "units": units,
        "grain": grain,
        "period": period,
        **extra,
    }


def _inventory(root: Path, variant: str, mappings: dict[str, str]) -> dict[str, dict[str, Any]]:
    snapshots = _rows(root, "inventory", variant, "inventory_snapshots", mappings)
    orders = _rows(root, "inventory", variant, "order_lines", mappings)
    shipments = _rows(root, "inventory", variant, "shipment_events", mappings)
    products = _rows(root, "inventory", variant, "products", mappings)
    latest = max(row["snapshot_date"] for row in snapshots)
    current = [row for row in snapshots if row["snapshot_date"] == latest]
    available = sum(int(row["on_hand_units"]) - int(row["allocated_units"]) for row in current)
    shipped_by_line: dict[tuple[str, str], int] = defaultdict(int)
    for row in shipments:
        shipped_by_line[(row["order_id"], row["line_id"])] += int(row["shipped_units"])
    open_units = sum(
        max(0, int(row["ordered_units"]) - shipped_by_line[(row["order_id"], row["line_id"])])
        for row in orders
    )
    shipped_by_customer: dict[str, int] = defaultdict(int)
    for row in orders:
        shipped_by_customer[row["customer_id"]] += shipped_by_line[(row["order_id"], row["line_id"])]
    top_customer = max(shipped_by_customer, key=lambda key: (shipped_by_customer[key], key))
    costs = {row["product_id"]: float(row["unit_cost"]) for row in products}
    inventory_value = sum(int(row["on_hand_units"]) * costs[row["product_id"]] for row in current)
    total_ordered = sum(int(row["ordered_units"]) for row in orders)
    total_shipped = sum(shipped_by_line.values())
    return {
        "inventory_available_units": _answered(available, "units", "latest warehouse-product snapshot rows", latest),
        "inventory_open_order_units": _answered(open_units, "units", "order line", "through 2026-03-18"),
        "inventory_top_customer": _answered(shipped_by_customer[top_customer], "shipped units", "customer_id", "through 2026-03-18", entity_id=top_customer),
        "inventory_join_safe_value": _answered(inventory_value, "currency units", "latest product snapshot rows", latest),
        "inventory_fill_rate": _answered(total_shipped / total_ordered, "ratio", "order line", "through 2026-03-18"),
    }


def _manufacturing(root: Path, variant: str, mappings: dict[str, str]) -> dict[str, dict[str, Any]]:
    production = _rows(root, "manufacturing", variant, "production", mappings)
    defects = _rows(root, "manufacturing", variant, "defects", mappings)
    complete = [row for row in production if row["reporting_complete"].lower() == "true"]
    complete_keys = {(row["reporting_period"], row["line_id"]) for row in complete}
    included_defects = [row for row in defects if (row["reporting_period"], row["line_id"]) in complete_keys]
    produced = sum(int(row["produced_units"]) for row in complete)
    defect_units = sum(int(row["defect_units"]) for row in included_defects)
    produced_by_line = {row["line_id"]: int(row["produced_units"]) for row in complete}
    defects_by_line: dict[str, int] = defaultdict(int)
    for row in included_defects:
        defects_by_line[row["line_id"]] += int(row["defect_units"])
    rates = {line: defects_by_line[line] / units for line, units in produced_by_line.items()}
    worst = max(rates, key=lambda key: (rates[key], key))
    excluded = sum(int(row["produced_units"]) for row in production if row not in complete)
    return {
        "manufacturing_complete_units": _answered(produced, "units", "complete production periods", "2026-01"),
        "manufacturing_weighted_defect_rate": _answered(defect_units / produced, "ratio", "complete production periods", "2026-01"),
        "manufacturing_worst_line": _answered(rates[worst], "ratio", "production line in complete periods", "2026-01", entity_id=worst),
        "manufacturing_unweighted_trap": _answered(defect_units / produced, "ratio", "complete production periods", "2026-01"),
        "manufacturing_incomplete_excluded": _answered(excluded, "units", "incomplete production periods", "2026-02"),
    }


def _minutes(start: str, end: str) -> float:
    return (datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds() / 60


def _service(root: Path, variant: str, mappings: dict[str, str]) -> dict[str, dict[str, Any]]:
    tickets = _rows(root, "service", variant, "tickets", mappings)
    events = _rows(root, "service", variant, "ticket_events", mappings)
    policies = _rows(root, "service", variant, "sla_policies", mappings)
    thresholds = {row["policy_id"]: int(row["first_response_minutes"]) for row in policies}
    first_response: dict[str, str] = {}
    reopened: set[str] = set()
    for row in events:
        if row["event_type"] == "agent_response":
            first_response[row["ticket_id"]] = min(first_response.get(row["ticket_id"], row["event_at"]), row["event_at"])
        elif row["event_type"] == "reopened":
            reopened.add(row["ticket_id"])
    elapsed = {
        row["ticket_id"]: _minutes(row["opened_at"], first_response[row["ticket_id"]])
        for row in tickets
        if row["ticket_id"] in first_response
    }
    met = {
        row["ticket_id"]
        for row in tickets
        if elapsed.get(row["ticket_id"], float("inf")) <= thresholds[row["policy_id"]]
    }
    breached = sorted({row["ticket_id"] for row in tickets} - met)
    return {
        "service_first_response_sla_rate": _answered(len(met) / len(tickets), "ratio", "ticket", "2026-03-01 through 2026-03-03"),
        "service_reopened_tickets": _answered(len(reopened), "tickets", "distinct ticket_id", "2026-03-01 through 2026-03-03"),
        "service_breached_ticket": _answered(breached, "ticket identities", "ticket", "2026-03-01 through 2026-03-03", entity_ids=breached),
        "service_first_response_dedup": _answered(elapsed["T-001"], "minutes", "ticket T-001 earliest agent response", "2026-03-01"),
        "service_ambiguous_root_cause": {
            "expected_status": "abstain",
            "reason": "No root-cause field or definition is provided; event timing cannot establish causality.",
            "units": "not applicable",
            "grain": "not defined",
            "period": "2026-03-01 through 2026-03-03",
        },
    }


def calculate_references(root: str | Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    fixture_root = Path(root)
    definitions = json.loads((fixture_root / "definitions.json").read_text(encoding="utf-8"))
    output: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    calculators = {
        "inventory": _inventory,
        "manufacturing": _manufacturing,
        "service": _service,
    }
    for domain, calculator in calculators.items():
        output[domain] = {}
        for variant in VARIANTS:
            mappings = definitions["variants"][variant][domain]["field_mappings"]
            output[domain][variant] = calculator(fixture_root, variant, mappings)
    return output


__all__ = ["calculate_references"]
