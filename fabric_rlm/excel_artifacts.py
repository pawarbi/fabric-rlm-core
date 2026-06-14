"""Excel artifact helpers for RLM validators and workbook-editing skills.

These utilities are intentionally benchmark-agnostic: they parse Excel target
ranges, iterate saved workbook cells, and validate common artifact mistakes
without knowing any golden answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class ExcelTargetRange:
    sheet_name: str | None
    cell_range: str


@dataclass(frozen=True)
class ExcelCellValue:
    sheet_name: str
    coordinate: str
    value: Any


ERROR_LITERALS: frozenset[str] = frozenset(
    {"#N/A", "#VALUE!", "#REF!", "#DIV/0!", "#NAME?", "#NULL!", "#NUM!"}
)
CODE_OR_PROSE_MARKERS: tuple[str, ...] = (
    "Sub ",
    "End Sub",
    "Power Query",
    "VBA",
    "Macro:",
    "let Source",
    "Application.",
    "ws.Range",
    "ws.Rows",
)
PLACEHOLDERS: frozenset[str] = frozenset({"-", "TBD", "N/A", "see notes"})


def parse_target_ranges(position: str) -> list[ExcelTargetRange]:
    """Parse a comma-separated Excel target-range expression.

    Supports bare ranges (``A1:B2``), sheet-qualified ranges
    (``Sheet2!A1:B2``), quoted sheet names with commas
    (``'Sheet, 3'!A1:B2``), and the common abbreviated single-column form
    (``H2:27`` -> ``H2:H27``).
    """

    parts = _split_range_list(position)
    if not parts:
        raise ValueError("target range is empty")
    return [_parse_one_range(part) for part in parts]


def iter_target_cells(
    workbook_path: str | Path,
    position: str,
    *,
    default_sheet: str | None = None,
    data_only: bool = True,
) -> Iterable[ExcelCellValue]:
    """Yield target cells from a saved workbook in row-major order."""

    openpyxl = _openpyxl()
    wb = openpyxl.load_workbook(workbook_path, data_only=data_only)
    for target in parse_target_ranges(position):
        sheet_name = _resolve_sheet(wb.sheetnames, target.sheet_name, default_sheet)
        ws = wb[sheet_name]
        for coord, value in _coord_values(ws[target.cell_range]):
            yield ExcelCellValue(sheet_name=sheet_name, coordinate=coord, value=value)


def validate_target_range_sanity(
    workbook_path: str | Path,
    position: str,
    *,
    default_sheet: str | None = None,
    error_literals: Iterable[str] = ERROR_LITERALS,
    code_or_prose_markers: Iterable[str] = CODE_OR_PROSE_MARKERS,
    placeholders: Iterable[str] = PLACEHOLDERS,
) -> None:
    """Reject common invalid Excel artifact states without golden answers.

    Intentional blanks are allowed. Formula strings, Excel error literals,
    placeholder text, and code/prose markers are rejected in the target cells.
    Raises ``AssertionError`` with a repair-friendly message on the first issue.
    """

    openpyxl = _openpyxl()
    wb_formula = openpyxl.load_workbook(workbook_path, data_only=False)
    wb_values = openpyxl.load_workbook(workbook_path, data_only=True)
    errors = set(error_literals)
    markers = tuple(code_or_prose_markers)
    placeholder_set = set(placeholders)

    for target in parse_target_ranges(position):
        sheet_name = _resolve_sheet(wb_formula.sheetnames, target.sheet_name, default_sheet)
        formula_cells = _coord_values(wb_formula[sheet_name][target.cell_range])
        value_cells = _coord_values(wb_values[sheet_name][target.cell_range])
        if len(formula_cells) != len(value_cells):
            raise AssertionError(
                f"{sheet_name}!{target.cell_range} has inconsistent cell counts "
                f"between formula/data_only views ({len(formula_cells)} vs {len(value_cells)})"
            )
        for (coord, formula_val), (_, data_val) in zip(formula_cells, value_cells):
            for val, mode in ((formula_val, "formula"), (data_val, "data_only")):
                if val in errors:
                    raise AssertionError(f"{sheet_name}!{coord} has Excel error {val!r} in {mode} view")
            if formula_val is None:
                continue
            if isinstance(formula_val, str):
                if formula_val.startswith("="):
                    raise AssertionError(f"{sheet_name}!{coord} still contains a formula string")
                if formula_val in placeholder_set:
                    raise AssertionError(f"{sheet_name}!{coord} contains placeholder {formula_val!r}")
                for marker in markers:
                    if marker in formula_val:
                        raise AssertionError(
                            f"{sheet_name}!{coord} contains code/prose marker {marker!r}"
                        )


def _split_range_list(position: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    in_quote = False
    for ch in position.strip():
        if ch == "'":
            in_quote = not in_quote
            current.append(ch)
            continue
        if ch == "," and not in_quote:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(ch)
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _parse_one_range(part: str) -> ExcelTargetRange:
    sheet_name: str | None = None
    cell_range = part.strip()
    if "!" in cell_range:
        sheet, cell_range = cell_range.rsplit("!", 1)
        sheet_name = _unquote_sheet_name(sheet.strip()) or None
    return ExcelTargetRange(sheet_name=sheet_name, cell_range=_normalize_range(cell_range.strip()))


def _unquote_sheet_name(sheet: str) -> str:
    if len(sheet) >= 2 and sheet[0] == "'" and sheet[-1] == "'":
        return sheet[1:-1].replace("''", "'")
    return sheet


def _normalize_range(cell_range: str) -> str:
    match = re.fullmatch(r"([A-Z]+)(\d+):(\d+)", cell_range, flags=re.IGNORECASE)
    if match:
        col, start, end = match.groups()
        return f"{col.upper()}{start}:{col.upper()}{end}"
    return cell_range


def _resolve_sheet(
    sheetnames: list[str], requested: str | None, default_sheet: str | None
) -> str:
    if requested:
        if requested not in sheetnames:
            raise AssertionError(f"target sheet {requested!r} not found; available sheets: {sheetnames}")
        return requested
    if default_sheet:
        if default_sheet not in sheetnames:
            raise AssertionError(
                f"default target sheet {default_sheet!r} not found; available sheets: {sheetnames}"
            )
        return default_sheet
    return sheetnames[0]


def _coord_values(range_obj: Any) -> list[tuple[str, Any]]:
    if hasattr(range_obj, "value"):
        return [(range_obj.coordinate, range_obj.value)]
    out: list[tuple[str, Any]] = []
    for item in range_obj:
        if hasattr(item, "value"):
            out.append((item.coordinate, item.value))
        else:
            out.extend((cell.coordinate, cell.value) for cell in item)
    return out


def _openpyxl() -> Any:
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Excel artifact helpers require openpyxl. Install it with `pip install openpyxl`."
        ) from exc
    return openpyxl
