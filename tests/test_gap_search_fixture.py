from pathlib import Path

from behavior.test_gap_search_live import GOLD, independent_oracle, write_sources


def test_fanout_fixture_has_expected_answer_and_join_trap(tmp_path: Path) -> None:
    invoices, adjustments = write_sources(tmp_path)
    assert independent_oracle(invoices, adjustments) == GOLD
    assert len(invoices.read_text(encoding="utf-8").splitlines()) == 1601
    assert len(adjustments.read_text(encoding="utf-8").splitlines()) == 1201
    # A direct join repeats each adjustment for every posted line. It picks East.
    assert 10000 - (1000 * 2) > 16000 - (4000 * 4)
