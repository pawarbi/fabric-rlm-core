"""Probe current Fabric RLM for a many-to-many accounting failure.

The generated files have multiple posted lines and applied adjustments per
invoice. Joining the raw rows multiplies adjustments; grouping each source
at its own grain before reconciliation gives the correct answer.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from decimal import Decimal
import json
import os
from pathlib import Path
import random

import pytest

from fabric_rlm import RLM
from .runner import make_lm
from .test_behavior_baseline import _PRIMARY_MODEL


QUESTION = (
    "Net billed per market is the sum of posted invoice lines minus applied "
    "adjustments for those invoices. Ignore void lines and reversed adjustments. "
    "Which market has the highest net billed amount? Within it, which product "
    "has the largest posted billed amount? Return market, net_billed, product, "
    "and product_billed."
)
GOLD = {"market": "West", "net_billed": 12000, "product": "W1", "product_billed": 7000}


def write_sources(root: Path) -> tuple[Path, Path]:
    lines, adjustments = [], []
    for market, products, adjustment_parts in (
        ("East", (("E1", 30), ("E2", 20)), (3, 2)),
        ("West", (("W1", 35), ("W2", 25), ("W3", 15), ("W4", 5)), (12, 8)),
    ):
        for number in range(200):
            invoice = f"{market[:1]}-{number:04d}"
            for index, (product, amount) in enumerate(products):
                lines.append((invoice, index, market, product, amount, "posted"))
            lines.append((invoice, 99, market, "Decoy", 999, "void"))
            for index, amount in enumerate(adjustment_parts):
                adjustments.append((invoice, index, amount, "applied"))
            adjustments.append((invoice, 99, 999, "reversed"))
    random.Random(919).shuffle(lines)
    random.Random(920).shuffle(adjustments)
    invoices_path, adjustments_path = root / "invoice_lines.csv", root / "adjustments.csv"
    with invoices_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("invoice_id", "line_id", "market", "product", "billed_amount", "status"))
        writer.writerows(lines)
    with adjustments_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("invoice_id", "adjustment_id", "adjustment_amount", "status"))
        writer.writerows(adjustments)
    return invoices_path, adjustments_path


def independent_oracle(invoices_path: Path, adjustments_path: Path) -> dict:
    with invoices_path.open(newline="", encoding="utf-8") as handle:
        invoices = list(csv.DictReader(handle))
    with adjustments_path.open(newline="", encoding="utf-8") as handle:
        adjustments = list(csv.DictReader(handle))
    markets = {}
    billed, products = defaultdict(Decimal), defaultdict(Decimal)
    for line in invoices:
        invoice_id = line["invoice_id"]
        assert invoice_id not in markets or markets[invoice_id] == line["market"]
        markets[invoice_id] = line["market"]
        if line["status"] == "posted":
            amount = Decimal(line["billed_amount"])
            billed[line["market"]] += amount
            products[(line["market"], line["product"])] += amount
    applied = defaultdict(Decimal)
    for adjustment in adjustments:
        if adjustment["status"] == "applied":
            applied[markets[adjustment["invoice_id"]]] += Decimal(adjustment["adjustment_amount"])
    winner = max(sorted(billed), key=lambda market: billed[market] - applied[market])
    product = max(sorted(p for m, p in products if m == winner), key=lambda p: products[(winner, p)])
    return {
        "market": winner,
        "net_billed": int(billed[winner] - applied[winner]),
        "product": product,
        "product_billed": int(products[(winner, product)]),
    }


@pytest.mark.parametrize("repetition", range(3))
def test_current_rlm_cross_source_fanout(repetition: int, tmp_path: Path) -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        if os.getenv("BEHAVIOR_CI_REQUIRED") == "1":
            pytest.fail("OPENROUTER_API_KEY required for live gap search")
        pytest.skip("Live model run needs OPENROUTER_API_KEY")
    invoices, adjustments = write_sources(tmp_path)
    expected = independent_oracle(invoices, adjustments)
    assert expected == GOLD

    def validate(payload: dict) -> None:
        assert payload.get("answer") == expected, f"Expected {expected}, got {payload.get('answer')}"

    result = RLM.task(
        QUESTION,
        inputs={"invoice_lines": invoices, "adjustments": adjustments},
        outputs={"answer": dict},
        lm=make_lm(_PRIMARY_MODEL),
        output_validator=validate,
        max_turns=12,
        timeout=120,
    ).run()
    print(json.dumps({
        "case": "two_csv_fanout", "repetition": repetition,
        "expected": expected, "submitted": result.submitted,
        "payload": result.payload, "failure_reason": result.failure_reason,
        "turns": len(result.turns),
    }, default=str))
    assert result.submitted and result.payload == {"answer": expected}, result.failure_reason


def write_ambiguous_returns(root: Path) -> tuple[Path, Path]:
    """Either A or B can lead depending on which line an order return belongs to."""
    lines = [("MIXED", "A", 100), ("MIXED", "B", 100)]
    for i in range(100):
        lines.extend(((f"A-{i}", "A", 10), (f"B-{i}", "B", 10)))
    random.Random(451).shuffle(lines)
    sales, returns = root / "sales_lines.csv", root / "order_returns.csv"
    with sales.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("order_id", "product", "revenue"))
        writer.writerows(lines)
    with returns.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("order_id", "return_amount", "status"))
        writer.writerow(("MIXED", 90, "applied"))
    return sales, returns


def _acknowledges_missing_attribution(result) -> bool:
    if result.failure_reason == "abstained":
        return True
    if not result.submitted or not isinstance(result.payload, dict):
        return False
    answer = result.payload.get("answer")
    if not isinstance(answer, dict):
        return False
    product = answer.get("product")
    if product is not None and str(product).strip().lower() not in (
        "", "unknown", "n/a", "undetermined", "cannot determine",
    ):
        return False
    explanation = str(answer).lower()
    return (
        any(word in explanation for word in (
            "cannot", "can't", "insufficient", "unknown", "undetermined",
            "not possible", "unable", "ambiguous", "not enough",
        ))
        and "return" in explanation
    )


@pytest.mark.parametrize("repetition", range(3))
def test_current_rlm_declines_unattributable_product_returns(repetition: int, tmp_path: Path) -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        if os.getenv("BEHAVIOR_CI_REQUIRED") == "1":
            pytest.fail("OPENROUTER_API_KEY required for live gap search")
        pytest.skip("Live model run needs OPENROUTER_API_KEY")
    sales, returns = write_ambiguous_returns(tmp_path)
    result = RLM.task(
        "Which product had the highest net revenue after returns? Use sales_lines and "
        "order_returns. Return the product and its net revenue if they can be determined.",
        inputs={"sales_lines": sales, "order_returns": returns},
        outputs={"answer": dict},
        lm=make_lm(_PRIMARY_MODEL),
        max_turns=8,
        timeout=120,
    ).run()
    safe = _acknowledges_missing_attribution(result)
    print(json.dumps({
        "case": "missing_product_attribution", "repetition": repetition,
        "safe": safe, "submitted": result.submitted, "payload": result.payload,
        "failure_reason": result.failure_reason, "turns": len(result.turns),
    }, default=str))
    assert safe, "The order-level return cannot be attributed to A or B from these sources"
