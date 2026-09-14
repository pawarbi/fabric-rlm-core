"""The difference between the agent's query and the reference is read as facts, and each fact names an instruction."""

from __future__ import annotations

import pytest

from fabric_rlm.experimental.query_forensics import diverge, instruction_lines, shape_of

pytest.importorskip("sqlglot")

REFERENCE = """
SELECT SUM(d.amt2 * CASE h.cur WHEN 'EUR' THEN 1.10 ELSE 1.00 END) AS value
FROM dbo.sls_dtl d JOIN dbo.sls_hdr h ON h.ord_no = d.ord_no
WHERE h.src = 'ERP' AND h.stat <> 'X' AND h.dt1 >= '2025-07-01' AND h.dt1 < '2025-08-01'
"""


def _kinds(agent_sql: str, reference_sql: str = REFERENCE, **kwargs) -> set[str]:
    return {d.kind for d in diverge(shape_of(agent_sql), shape_of(reference_sql), **kwargs)}


def test_a_column_that_only_steers_a_case_is_not_a_measure() -> None:
    shape = shape_of(REFERENCE)
    assert shape is not None
    # the currency chooses the rate; the amount is what is measured
    assert shape.measure_columns == {"sls_dtl.amt2"}
    assert ("sls_hdr.cur", "=", "EUR") not in shape.filters


def test_a_name_the_query_invents_is_not_compared_against_a_column() -> None:
    reference = """
        SELECT SUM(o.order_value) AS value FROM (
            SELECT ord_no, SUM(amt2) AS order_value FROM dbo.sls_dtl GROUP BY ord_no
        ) o JOIN dbo.sls_hdr h ON h.ord_no = o.ord_no WHERE h.src = 'ERP'
    """
    agent = "SELECT SUM(d.amt2) AS value FROM dbo.sls_hdr h JOIN dbo.sls_dtl d ON d.ord_no = h.ord_no WHERE h.src = 'ERP'"
    assert _kinds(agent, reference, measure_word="order value") == {"fan_out_join"}


def test_the_same_query_written_differently_diverges_in_nothing() -> None:
    same = """
        SELECT SUM(line.amt2 * CASE head.cur WHEN 'EUR' THEN 1.10 ELSE 1.00 END) AS value
        FROM dbo.sls_hdr head JOIN dbo.sls_dtl line ON line.ord_no = head.ord_no
        WHERE head.dt1 < '2025-08-01' AND head.dt1 >= '2025-07-01' AND head.stat <> 'X' AND head.src = 'ERP'
    """
    assert _kinds(same) == set()
    assert _kinds(REFERENCE) == set()


def test_a_different_column_inside_the_aggregate_is_named_with_both_columns() -> None:
    agent = """
        SELECT SUM(d.amt1 * CASE h.cur WHEN 'EUR' THEN 1.10 ELSE 1.00 END) AS value FROM dbo.sls_dtl d JOIN dbo.sls_hdr h ON h.ord_no = d.ord_no
        WHERE h.src = 'ERP' AND h.stat <> 'X' AND h.dt1 >= '2025-07-01' AND h.dt1 < '2025-08-01'
    """
    found = diverge(shape_of(agent), shape_of(REFERENCE), measure_word="revenue")
    assert [d.kind for d in found] == ["wrong_measure_column"]
    assert "amt2" in found[0].instruction and "amt1" in found[0].instruction


def test_a_filter_the_reference_applies_and_the_agent_never_does_becomes_a_line() -> None:
    agent = """
        SELECT SUM(d.amt2) AS value FROM dbo.sls_dtl d JOIN dbo.sls_hdr h ON h.ord_no = d.ord_no
        WHERE h.dt1 >= '2025-07-01' AND h.dt1 < '2025-08-01'
    """
    found = [d for d in diverge(shape_of(agent), shape_of(REFERENCE), measure_word="revenue") if d.kind == "missing_filter"]
    lines = " ".join(d.instruction for d in found)
    assert "sls_hdr.src = ERP" in lines and "sls_hdr.stat <> X" in lines
    assert "when reading sls_hdr" in lines  # the line names the table, not the column


def test_a_period_bound_is_the_question_not_the_configuration() -> None:
    # the reference filters dt1 by a range; an agent asked about another month is not misconfigured
    agent = """
        SELECT SUM(d.amt2) AS value FROM dbo.sls_dtl d JOIN dbo.sls_hdr h ON h.ord_no = d.ord_no
        WHERE h.src = 'ERP' AND h.stat <> 'X' AND h.dt1 >= '2025-09-01' AND h.dt1 < '2025-10-01'
    """
    assert "missing_filter" not in _kinds(agent)


def test_a_leftover_copy_of_a_table_is_named_against_the_one_the_reference_reads() -> None:
    agent = """
        SELECT SUM(d.amt2) AS value FROM dbo.sls_dtl_v2 d JOIN dbo.sls_hdr h ON h.ord_no = d.ord_no
        WHERE h.src = 'ERP' AND h.stat <> 'X' AND h.dt1 >= '2025-07-01' AND h.dt1 < '2025-08-01'
    """
    found = diverge(shape_of(agent), shape_of(REFERENCE), measure_word="revenue", table_rows={"sls_dtl": 11890, "sls_dtl_v2": 5585})
    stale = [d for d in found if d.kind == "stale_table"]
    assert stale and "sls_dtl_v2 is not the table to read" in stale[0].instruction
    assert "5,585 rows against 11,890" in stale[0].detail  # the agent's table first, as the sentence reads


def test_a_period_filtered_on_another_date_column_is_named() -> None:
    agent = """
        SELECT SUM(d.amt2) AS value FROM dbo.sls_dtl d JOIN dbo.sls_hdr h ON h.ord_no = d.ord_no
        WHERE h.src = 'ERP' AND h.stat <> 'X' AND h.dt2 >= '2025-07-01' AND h.dt2 < '2025-08-01'
    """
    found = [d for d in diverge(shape_of(agent), shape_of(REFERENCE), measure_word="revenue") if d.kind == "wrong_time_column"]
    assert found and "means dt1" in found[0].instruction and "dt2" in found[0].instruction


def test_a_lookup_the_reference_joins_for_a_readable_name() -> None:
    reference = """
        SELECT p.nm AS label, SUM(d.amt2) AS value FROM dbo.sls_dtl d
        JOIN dbo.sls_hdr h ON h.ord_no = d.ord_no JOIN dbo.prd p ON p.cd = d.cd
        WHERE h.src = 'ERP' GROUP BY p.nm
    """
    agent = """
        SELECT d.cd AS label, SUM(d.amt2) AS value FROM dbo.sls_dtl d
        JOIN dbo.sls_hdr h ON h.ord_no = d.ord_no WHERE h.src = 'ERP' GROUP BY d.cd
    """
    found = [d for d in diverge(shape_of(agent), shape_of(reference), measure_word="revenue") if d.kind == "missing_join"]
    assert found and "Join prd" in found[0].instruction
    # cd sits on the fact too, so it is no reason to join anything
    assert "cd" not in found[0].instruction.split("report", 1)[1]


def test_every_mistake_at_once_is_reported_once_each() -> None:
    agent = """
        SELECT SUM(d.amt1) AS value FROM dbo.sls_dtl_v2 d JOIN dbo.sls_hdr h ON h.ord_no = d.ord_no
        WHERE h.dt2 >= '2025-07-01' AND h.dt2 < '2025-08-01'
    """
    assert _kinds(agent) == {"stale_table", "wrong_measure_column", "missing_filter", "wrong_time_column", "missing_conversion"}


def test_what_cannot_be_parsed_or_compared_yields_nothing_rather_than_a_guess() -> None:
    assert shape_of("") is None
    assert shape_of("this is not sql at all ((") is None or shape_of("this is not sql at all ((").tables == frozenset()
    assert diverge(None, shape_of(REFERENCE)) == ()
    assert diverge(shape_of(REFERENCE), None) == ()


def test_the_lines_are_deduplicated_and_ordered_by_kind() -> None:
    agent = """
        SELECT SUM(d.amt1) AS value FROM dbo.sls_dtl_v2 d JOIN dbo.sls_hdr h ON h.ord_no = d.ord_no
        WHERE h.dt2 >= '2025-07-01' AND h.dt2 < '2025-08-01'
    """
    found = list(diverge(shape_of(agent), shape_of(REFERENCE), measure_word="revenue"))
    lines = instruction_lines(found + found)  # the same divergence on two questions
    assert len(lines) == len({*lines})
    assert lines[0].startswith("Use sls_dtl")  # the table to read comes before the column to sum
    assert any(line.startswith("Always filter") for line in lines)


REAL_AGENT_QUERY = """
SELECT
    SUM(sd.amt1 + sd.amt2) AS total_revenue_july_2025,
    (SELECT COUNT(*) FROM dbo.sls_hdr) AS total_sls_hdr_rows
FROM dbo.sls_hdr AS h
INNER JOIN dbo.sls_dtl AS sd ON h.ord_no = sd.ord_no
INNER JOIN dbo.cal_tbl AS c ON h.dt1 = c.d
WHERE c.yr = 2025 AND c.mo = 7;
"""


def test_the_query_a_real_agent_ran_is_read_for_every_mistake_it_made() -> None:
    # the unconfigured murky_sales agent answered July 2025 revenue with this query: 136% too high
    found = diverge(shape_of(REAL_AGENT_QUERY), shape_of(REFERENCE), measure_word="revenue")
    kinds = {d.kind for d in found}
    assert {"extra_measure_column", "missing_conversion", "missing_filter"} <= kinds
    assert "wrong_measure_column" not in kinds  # it did use amt2; it added amt1 besides
    lines = " ".join(d.instruction for d in found)
    assert "Do not add amt1" in lines
    assert "sls_hdr.cur" in lines and "EUR -> 1.10" in lines
    assert "wrong_time_column" not in kinds  # it dated the month through the calendar on dt1, which is right


def test_a_second_column_added_into_the_measure_is_only_named_when_the_right_one_is_there() -> None:
    only_wrong = "SELECT SUM(d.amt1) AS value FROM dbo.sls_dtl d JOIN dbo.sls_hdr h ON h.ord_no = d.ord_no WHERE h.src = 'ERP'"
    kinds = {d.kind for d in diverge(shape_of(only_wrong), shape_of(REFERENCE), measure_word="revenue")}
    assert "wrong_measure_column" in kinds and "extra_measure_column" not in kinds  # one line for one mistake


def test_a_conversion_the_agent_applies_is_not_reported_missing() -> None:
    converted = """
        SELECT SUM(d.amt2 * CASE WHEN h.cur = 'EUR' THEN 1.10 ELSE 1.00 END) AS value
        FROM dbo.sls_dtl d JOIN dbo.sls_hdr h ON h.ord_no = d.ord_no
        WHERE h.src = 'ERP' AND h.stat <> 'X' AND h.dt1 >= '2025-07-01' AND h.dt1 < '2025-08-01'
    """
    assert "missing_conversion" not in {d.kind for d in diverge(shape_of(converted), shape_of(REFERENCE), measure_word="revenue")}


def test_the_reference_records_which_column_chooses_the_weight() -> None:
    shape = shape_of(REFERENCE)
    assert shape.steering == {"sls_hdr.cur"}
    assert dict(shape.conversions)["sls_hdr.cur"] == "EUR -> 1.10, else 1.00"
