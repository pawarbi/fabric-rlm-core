from pathlib import Path

import csv

from behavior.test_gap_search_live import (
    GOLD, independent_oracle, write_ambiguous_returns, write_sources,
)


def test_fanout_fixture_has_expected_answer_and_join_trap(tmp_path: Path) -> None:
    invoices, adjustments = write_sources(tmp_path)
    assert independent_oracle(invoices, adjustments) == GOLD
    assert len(invoices.read_text(encoding="utf-8").splitlines()) == 1601
    assert len(adjustments.read_text(encoding="utf-8").splitlines()) == 1201
    # A direct join repeats each adjustment for every posted line. It picks East.
    assert 10000 - (1000 * 2) > 16000 - (4000 * 4)


def test_product_winner_changes_with_valid_return_allocation(tmp_path: Path) -> None:
    sales, returns = write_ambiguous_returns(tmp_path)
    with sales.open(newline="", encoding="utf-8") as handle:
        lines = list(csv.DictReader(handle))
    with returns.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"order_id": "MIXED", "return_amount": "90", "status": "applied"}]
    gross_a = sum(int(row["revenue"]) for row in lines if row["product"] == "A")
    gross_b = sum(int(row["revenue"]) for row in lines if row["product"] == "B")
    assert gross_a == gross_b == 1100
    assert gross_a - 90 < gross_b
    assert gross_b - 90 < gross_a
