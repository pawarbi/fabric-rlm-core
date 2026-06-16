from __future__ import annotations

import pytest

openpyxl = pytest.importorskip("openpyxl")

from fabric_rlm.excel_artifacts import (
    ExcelTargetRange,
    iter_target_cells,
    parse_target_ranges,
    summarize_workbook_contract_context,
    summarize_workbook_structure_context,
    summarize_workbook_context,
    validate_target_range_sanity,
)


def test_parse_target_ranges_handles_comma_separated_sheet_ranges() -> None:
    parsed = parse_target_ranges("A1:A50,Sheet2!A1:E20,'Sheet, Three'!C3:D4")

    assert parsed == [
        ExcelTargetRange(sheet_name=None, cell_range="A1:A50"),
        ExcelTargetRange(sheet_name="Sheet2", cell_range="A1:E20"),
        ExcelTargetRange(sheet_name="Sheet, Three", cell_range="C3:D4"),
    ]


def test_parse_target_ranges_unescapes_quoted_sheet_apostrophes() -> None:
    parsed = parse_target_ranges("'O''Brien'!A1:B2")

    assert parsed == [
        ExcelTargetRange(sheet_name="O'Brien", cell_range="A1:B2"),
    ]


def test_excel_artifact_helpers_are_public_api() -> None:
    import fabric_rlm

    assert fabric_rlm.parse_target_ranges("A1") == [
        fabric_rlm.ExcelTargetRange(sheet_name=None, cell_range="A1")
    ]
    assert callable(fabric_rlm.summarize_workbook_contract_context)


def test_parse_target_ranges_normalizes_single_column_row_range() -> None:
    assert parse_target_ranges("H2:27") == [
        ExcelTargetRange(sheet_name=None, cell_range="H2:H27")
    ]


def test_parse_target_ranges_tolerates_quoted_whole_sheet_ranges() -> None:
    parsed = parse_target_ranges("'99250!A1:F9','99251!A1:F9','99252!A1:F8'")

    assert parsed == [
        ExcelTargetRange(sheet_name="99250", cell_range="A1:F9"),
        ExcelTargetRange(sheet_name="99251", cell_range="A1:F9"),
        ExcelTargetRange(sheet_name="99252", cell_range="A1:F8"),
    ]


def test_parse_target_ranges_tolerates_quote_between_sheet_and_range() -> None:
    parsed = parse_target_ranges("'Sheet1!'A1:A50,'Sheet2!'A1:E20,'Sheet3!'A1:A50")

    assert parsed == [
        ExcelTargetRange(sheet_name="Sheet1", cell_range="A1:A50"),
        ExcelTargetRange(sheet_name="Sheet2", cell_range="A1:E20"),
        ExcelTargetRange(sheet_name="Sheet3", cell_range="A1:A50"),
    ]


def test_parse_target_ranges_tolerates_missing_opening_quote_on_first_sheet() -> None:
    parsed = parse_target_ranges(
        "OUT CAS'!A2:C1529,'OUT CAS'!E2:G586,'OUT CAS'!I2:K13,'OUT CAS'!L2:O8"
    )

    assert parsed == [
        ExcelTargetRange(sheet_name="OUT CAS", cell_range="A2:C1529"),
        ExcelTargetRange(sheet_name="OUT CAS", cell_range="E2:G586"),
        ExcelTargetRange(sheet_name="OUT CAS", cell_range="I2:K13"),
        ExcelTargetRange(sheet_name="OUT CAS", cell_range="L2:O8"),
    ]


def test_summarize_workbook_structure_context_handles_multi_sheet_default(tmp_path) -> None:
    path = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    wb.active.title = "Consolidated Tracker"
    wb.active.append(["Task", "Owner", "Status", "Start", "End"])
    for sheet_name in ["Existing Task", "Additions", "Retired"]:
        ws = wb.create_sheet(sheet_name)
        ws.append(["Task", "Owner", "Status", "Start", "End"])
    wb.save(path)

    summary = summarize_workbook_structure_context(
        path,
        target_position="A3:E11",
        default_sheet="Consolidated Tracker,Existing Task,Additions,Retired",
    )

    assert "active_sheet: Consolidated Tracker" in summary
    assert (
        "target_ranges: Consolidated Tracker!A3:E11, Existing Task!A3:E11, "
        "Additions!A3:E11, Retired!A3:E11"
    ) in summary


def test_summarize_workbook_structure_context_handles_quoted_multi_sheet_ranges(tmp_path) -> None:
    path = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    wb.active.title = "Sheet1"
    wb.active.append(["A", "B", "C"])
    for sheet_name in ["Sheet2", "Sheet3", "Sheet4"]:
        ws = wb.create_sheet(sheet_name)
        ws.append(["A", "B", "C"])
    wb.save(path)

    summary = summarize_workbook_structure_context(
        path,
        target_position="Sheet3'!A:G,'Sheet4'!A:G",
        default_sheet="'Sheet3','Sheet4'",
    )

    assert "active_sheet: Sheet3" in summary
    assert "target_ranges: Sheet3!A:G, Sheet4!A:G" in summary


def test_iter_target_cells_resolves_default_and_explicit_sheets(tmp_path) -> None:
    path = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Main"
    ws["A1"] = "x"
    ws["A2"] = "y"
    ws2 = wb.create_sheet("Other")
    ws2["B1"] = 10
    ws2["B2"] = 20
    wb.save(path)

    out = list(iter_target_cells(path, "A1:A2,Other!B1:B2", default_sheet="Main"))

    assert [(c.sheet_name, c.coordinate, c.value) for c in out] == [
        ("Main", "A1", "x"),
        ("Main", "A2", "y"),
        ("Other", "B1", 10),
        ("Other", "B2", 20),
    ]


def test_validate_target_range_sanity_rejects_formula_error_and_prose(tmp_path) -> None:
    path = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A1"] = "=SUM(B1:B2)"
    ws["A2"] = "#N/A"
    ws["A3"] = "Macro: copy values"
    ws["A4"] = None
    wb.save(path)

    with pytest.raises(AssertionError, match="A1.*formula"):
        validate_target_range_sanity(path, "A1:A4")

    ws["A1"] = 3
    wb.save(path)
    with pytest.raises(AssertionError, match="A2.*Excel error"):
        validate_target_range_sanity(path, "A1:A4")

    ws["A2"] = None
    wb.save(path)
    with pytest.raises(AssertionError, match="A3.*code/prose"):
        validate_target_range_sanity(path, "A1:A4")

    ws["A3"] = "ok"
    wb.save(path)
    validate_target_range_sanity(path, "A1:A4")


def test_summarize_workbook_context_returns_compact_targeted_sheet_evidence(tmp_path) -> None:
    path = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet2"
    ws.append(["Date", "Employee Number", "Abs. Name", "Atts. Name", "Week", None])
    ws.append(["2021-04-20", 133049, None, None, 17, None])
    ws.append(["2021-04-21", 133049, "Sick Day", None, 17, None])
    ws.append(["2021-04-22", 133049, None, "Overtime", 17, None])
    ws["G2"] = "=SUM(E2:E4)"
    wb.create_sheet("Lookup")["A1"] = "code"
    wb.save(path)

    summary = summarize_workbook_context(
        path,
        target_position="F2:F92",
        default_sheet="Sheet2",
        max_sample_rows=3,
    )

    assert "WORKBOOK_CONTEXT" in summary
    assert "sheets: Sheet2, Lookup" in summary
    assert "target_ranges: Sheet2!F2:F92" in summary
    assert "active_sheet: Sheet2" in summary
    assert "dimensions: A1:G4 rows=4 cols=7" in summary
    assert "headers: A=Date | B=Employee Number | C=Abs. Name | D=Atts. Name | E=Week | F=<blank>" in summary
    assert "formulas: 1" in summary
    assert "row 2: A='2021-04-20' | B=133049 | C=<blank> | D=<blank> | E=17 | F=<blank>" in summary
    assert len(summary) < 2000


def test_summarize_workbook_structure_context_omits_sample_values(tmp_path) -> None:
    path = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet2"
    ws.append(["Date", "Employee Number", "Abs. Name", "Atts. Name", "Week", None])
    ws.append(["2021-04-20", 133049, None, None, 17, None])
    ws.append(["2021-04-21", 133049, "Sick Day", None, 17, None])
    wb.save(path)

    summary = summarize_workbook_structure_context(
        path,
        target_position="F2:F92",
        default_sheet="Sheet2",
    )

    assert "WORKBOOK_STRUCTURE_CONTEXT" in summary
    assert "target_ranges: Sheet2!F2:F92" in summary
    assert "headers: A=Date | B=Employee Number | C=Abs. Name | D=Atts. Name | E=Week | F=<blank>" in summary
    assert "sample_rows: omitted" in summary
    assert "row 2:" not in summary
    assert "Sick Day" not in summary


def test_summarize_workbook_contract_context_reports_target_shape_and_edges(tmp_path) -> None:
    path = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Item", "Amount", "Status"])
    ws.append(["A", 10, "open"])
    ws.append(["B", 20, "closed"])
    ws.append([None, None, None])
    ws.append(["TOTAL", "=SUM(B2:B3)", None])
    ws["E2"] = "source value that should not be sampled"
    wb.save(path)

    summary = summarize_workbook_contract_context(
        path,
        target_position="A2:C5",
        default_sheet="Summary",
    )

    assert "WORKBOOK_CONTRACT_CONTEXT" in summary
    assert "target_ranges: Summary!A2:C5" in summary
    assert "Summary!A2:C5 shape=4x3 cells=12" in summary
    assert "current_nonblank=7" in summary
    assert "current_formulas=1" in summary
    assert "edge_rows:" in summary
    assert "row 2: A='A' | B=10 | C='open'" in summary
    assert "row 5: A='TOTAL' | B='=SUM(B2:B3)' | C=<blank>" in summary
    assert "formula_cells: B5='=SUM(B2:B3)'" in summary
    assert "source value that should not be sampled" not in summary
    assert "preserve target range shape" in summary


def test_summarize_workbook_contract_context_handles_multi_sheet_targets(tmp_path) -> None:
    path = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    wb.active.title = "First"
    wb.active["A1"] = "left"
    second = wb.create_sheet("Second")
    second["B2"] = "right"
    wb.save(path)

    summary = summarize_workbook_contract_context(
        path,
        target_position="First!A1:A2,Second!B2:B3",
    )

    assert "target_ranges: First!A1:A2, Second!B2:B3" in summary
    assert "First!A1:A2 shape=2x1 cells=2" in summary
    assert "Second!B2:B3 shape=2x1 cells=2" in summary
