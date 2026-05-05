---
applies_when:
  keywords:
    - xlsx
    - excel
    - workbook
    - openpyxl
    - sheet
    - worksheet
    - cell
    - cells
    - spreadsheet
    - .xlsx
    - column letter
    - cell range
output_fields: []
excludes: []
depends_on: []
specificity: domain
---
# excel_modify

Summary: You are MODIFYING an Excel `.xlsx` workbook in place using `openpyxl`. Open it, compute every required value in pure Python, write the **literal computed scalar** into each target cell, save back to the same path, and verify by reloading with `data_only=True`. This is NOT a JSONL / CSV / log-exploration task — do NOT try to `read_json_auto` or stream raw bytes; treat the input as a structured spreadsheet.

You never read the raw bytes back into the LM. Print only small summaries (headers, sample rows, the values you wrote) — never dump full sheets.

## Mandatory first-turn protocol

If you have not already inspected this exact workbook in a previous turn, your first code action MUST run the discovery below. It usually takes one turn and prevents wasted turns of column-name / sheet-name / merged-cell errors.

**Use the actual `WORKBOOK PATH` value from the prompt — do NOT copy the placeholder string below verbatim.** The placeholder is `<WORKBOOK_PATH_FROM_PROMPT>`; substitute the real path before running.

```python
import openpyxl
path = r"<WORKBOOK_PATH_FROM_PROMPT>"   # ← REPLACE with the real path from the prompt

# Load TWICE — once with formulas (for the writer), once with cached values (for reading inputs
# from cells that are formulas in the source). Many SpreadsheetBench inputs contain formulas.
wb = openpyxl.load_workbook(path)                       # for editing + saving
wb_vals = openpyxl.load_workbook(path, data_only=True)  # for reading computed values

print("SHEETS:", wb.sheetnames)
ws = wb.active                  # or wb["<sheet name>"] if the prompt names a sheet
ws_vals = wb_vals[ws.title]
print(f"ACTIVE SHEET: {ws.title}  dims={ws.dimensions}  max_row={ws.max_row}  max_col={ws.max_column}")

# Header row + a few sample rows. Print BOTH the formula view and the value view —
# if a cell shows '=D3+F3' in `ws` and a number in `ws_vals`, ALWAYS read inputs from `ws_vals`.
for r in range(1, min(6, ws.max_row + 1)):
    formula_row = [ws.cell(row=r, column=c).value for c in range(1, ws.max_column + 1)]
    value_row   = [ws_vals.cell(row=r, column=c).value for c in range(1, ws.max_column + 1)]
    print(f"row {r} formula:", formula_row)
    if formula_row != value_row:
        print(f"row {r} values: ", value_row)
```

After this prints you know the sheet name, header row, column order, whether any source cells are formulas, and roughly how many rows of data there are. Only THEN write the solution code.

## Anti-patterns (these caused real failures — do NOT repeat them)

- ❌ Writing Excel formulas (`ws['B3'] = "=SUM(A3:A10)"`). The grader uses `openpyxl.load_workbook(path, data_only=True)`, which does NOT evaluate formulas — it returns `None` for any cell whose cached value is missing. **Write computed scalar values only** (numbers, strings, dates).
- ❌ Forgetting `wb.save(path)`. Mutations to `ws` are in-memory only until you save. You must save back to the **same path** that was given to you.
- ❌ Saving to a different filename (`out.xlsx`, `result.xlsx`, etc.). The grader inspects the original WORKBOOK PATH — overwrite it.
- ❌ Treating the file like a JSONL / log / CSV stream. Do NOT call `open(path).read()`, `read_json_auto`, `pd.read_csv`, `json.loads`, or `for line in open(path)`. Use `openpyxl.load_workbook` (or `pandas.read_excel`).
- ❌ Assuming the sheet name. If the prompt does not name a sheet, default to `wb.active` and print `wb.sheetnames` first to confirm there is only one sheet.
- ❌ Writing to the wrong cell range. The grader inspects ONLY the TARGET CELL RANGE in the prompt. Iterate that exact range; don't spill into adjacent rows or columns.
- ❌ Ignoring header rows. Spreadsheet data usually starts on row 2 (row 1 is the header). Inspect first; do not start computing from row 1.
- ❌ Off-by-one between Excel column letters (A, B, C…) and openpyxl 1-based indices. `ws['B3']` is column 2, row 3. Use letter notation when the prompt gives letters; use `ws.cell(row=r, column=c)` only when iterating.
- ❌ Reading source cells with `ws.cell(...).value` when they are formulas — you get the literal string `'=D3+F3'`, not the computed number. **Read inputs from a `data_only=True` workbook** when the source contains formulas (see first-turn protocol).
- ❌ Pulling the whole sheet back into the LM context with `print(list(ws.values))`. Print 5 rows max for inspection.

## Reading source data

```python
# Read all data rows into Python objects so you can compute easily.
header = [c.value for c in ws[1]]                       # row 1 = headers
rows = [dict(zip(header, [c.value for c in r]))         # list[dict] keyed by header
        for r in ws.iter_rows(min_row=2, values_only=False)]
print(f"loaded {len(rows)} data rows; sample: {rows[0] if rows else None}")
```

If the workbook is large or has many sheets, prefer `pandas.read_excel(path, sheet_name=...)` for the read step (computation in pandas is concise) and `openpyxl` only for the targeted write step.

## Writing computed values

The TARGET CELL RANGE in the prompt is in Excel `A1:B10` notation. Compute the value for each cell in pure Python, then assign with **letter notation** to match the range exactly.

```python
# Example: target range is M2:M41, you've computed a list `marks` of 40 strings
assert len(marks) == 40, f"expected 40 values, got {len(marks)}"
for i, val in enumerate(marks, start=2):     # row 2..41
    ws[f"M{i}"] = val                        # literal scalar — NOT a formula

wb.save(path)                                # save back to the SAME path
```

For a single-row range like `B3:E3`, write four scalars:
```python
ws["B3"] = total_revenue_value
ws["C3"] = total_gp_value
ws["D3"] = ...
ws["E3"] = ...
wb.save(path)
```

Numeric types: prefer Python `int` / `float`. Dates: `datetime.datetime` (openpyxl writes as a real Excel date). Strings: plain `str`. Do NOT wrap any of these as `"=..."` strings.

## Mandatory verification step

After `wb.save(path)`, reload with `data_only=True` and print every cell in the TARGET CELL RANGE. Confirm:
1. None of the values is `None`.
2. None starts with `=` (which would mean you accidentally wrote a formula).
3. The values look reasonable (right magnitude, right type).

```python
wb2 = openpyxl.load_workbook(path, data_only=True)
ws2 = wb2[ws.title]
from openpyxl.utils.cell import range_boundaries
min_col, min_row, max_col, max_row = range_boundaries("M2:M41")   # use your TARGET CELL RANGE
for r in range(min_row, max_row + 1):
    for c in range(min_col, max_col + 1):
        v = ws2.cell(row=r, column=c).value
        print(ws2.cell(row=r, column=c).coordinate, "=", repr(v))
        assert v is not None, "wrote None — check your assignment"
        assert not (isinstance(v, str) and v.startswith("=")), f"wrote a formula at {ws2.cell(row=r, column=c).coordinate}"
print("VERIFY OK")
```

Only after the `VERIFY OK` line should you reply with your final answer.

## Final answer convention

The final `answer` field is just confirmation that the file was saved. Reply with the single word `done` (or a short status string). Do NOT put a formula, a list of values, or a long explanation in the answer — the grader inspects the saved workbook, not the answer string.

## Quick decision tree

- Prompt gives a `.xlsx` path + a TARGET CELL RANGE → use this skill.
- Prompt gives a `.jsonl` / `.csv` / log file path → use the `data_exploration` skill instead.
- Prompt gives both an `.xlsx` and an external data file → use this skill, but read the external file with the appropriate library (`pandas.read_csv`, etc.) before writing into the workbook.
