from __future__ import annotations

import pytest

openpyxl = pytest.importorskip("openpyxl")

from fabric_rlm.excel_artifacts import (
    ExcelTargetRange,
    iter_target_cells,
    parse_target_ranges,
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
