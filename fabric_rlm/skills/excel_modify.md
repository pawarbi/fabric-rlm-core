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

## Publishing a new workbook to OneLake

When the inputs include a `FileDestination`, do not pass its `abfss://` root to
openpyxl, pandas, notebookutils, or a storage SDK. Credentials intentionally
remain in the trusted parent process.

1. Create a local staged file with `destination.stage("report.xlsx")`.
2. Save and reopen that staged path with openpyxl.
3. After verification, call `destination.publish(staged)`.
4. Submit the returned manifest's `path` as the final workbook path.

```python
staged = destination.stage("regional_revenue.xlsx")
wb.save(staged.path)
verified = openpyxl.load_workbook(staged.path, data_only=True)
# Validate the saved workbook here.
artifact = destination.publish(staged)
SUBMIT(workbook_path=artifact["path"])
```

Never submit the staged local path as the final artifact. Publication must
succeed before claiming that the workbook exists in OneLake.

## READ THIS FIRST — the #1 failure mode

**The grader compares the SAVED CELL VALUES in the TARGET CELL RANGE against expected scalars.** It is a value-by-value comparison. It does NOT execute formulas, run macros, render Power Query, or interpret prose.

That means **none of the following will earn ANY credit**, even though they may look like they "describe" the right answer:

- Writing VBA code as a cell string: `ws["A1"] = "Sub DeleteEmptyRows()"`
- Writing Power Query M as a cell string: `ws["H2"] = "Power Query (M): let Source = ..."`
- Writing prose like `ws["A1"] = "Macro: delete row when col I has value and H is blank (executed below)."`
- Writing the *task description* into the worksheet
- Writing labels / headers / placeholder text (`"-"`, `"TBD"`, `"see notes"`) into target cells when actual values are required
- Writing Excel formula strings (`"=SUM(A1:A10)"`) — the grader uses `data_only=True` and reads `None`

**Your job is to actually perform the transformation in Python and write the resulting concrete values.** If the task is "delete rows where col I is non-empty and col H is blank", you must:

1. Iterate the rows with openpyxl, identify the matches
2. Use `ws.delete_rows(idx, amount=1)` to actually remove them (in reverse order!)
3. `wb.save(path)`
4. Reload and verify the surviving rows are the expected ones

You do NOT write the macro source code into a cell. You execute the equivalent operation and save the resulting workbook.

If you find yourself about to write a string into a cell that contains the words `"Sub "`, `"End Sub"`, `"For "`, `"Next "`, `"= IIf"`, `"Power Query"`, `"VBA"`, `"Macro:"`, `"Steps:"`, or `"let Source ="`, **STOP** — you are describing the solution instead of executing it. Re-read the task and produce the actual transformed data.

## Mandatory first-turn protocol

If you have not already inspected this exact workbook in a previous turn, your first code action MUST run the discovery below. It usually takes one turn and prevents wasted turns of column-name / sheet-name / merged-cell errors.

**Use the actual `WORKBOOK PATH` value from the prompt — do NOT copy the placeholder string below verbatim.** The placeholder is `<WORKBOOK_PATH_FROM_PROMPT>`; substitute the real path before running.

```python
import openpyxl
path = r"<WORKBOOK_PATH_FROM_PROMPT>"   # ← REPLACE with the real path from the prompt

wb = openpyxl.load_workbook(path)                       # for editing + saving
wb_vals = openpyxl.load_workbook(path, data_only=True)  # for reading computed values

print("SHEETS:", wb.sheetnames)
ws = wb.active                  # or wb["<sheet name>"] if the prompt names a sheet
ws_vals = wb_vals[ws.title]
print(f"ACTIVE SHEET: {ws.title}  dims={ws.dimensions}  max_row={ws.max_row}  max_col={ws.max_column}")

for r in range(1, min(6, ws.max_row + 1)):
    formula_row = [ws.cell(row=r, column=c).value for c in range(1, ws.max_column + 1)]
    value_row   = [ws_vals.cell(row=r, column=c).value for c in range(1, ws.max_column + 1)]
    print(f"row {r} formula:", formula_row)
    if formula_row != value_row:
        print(f"row {r} values: ", value_row)
```

After this prints you know the sheet name, header row, column order, whether any source cells are formulas, and roughly how many rows of data there are. Only THEN write the solution code.

## Restate the task before acting

After the first-turn discovery and BEFORE writing any cells, you must do the following in your reasoning (not in code):

1. Identify the **TARGET CELL RANGE** from the prompt (e.g. `F2:H10`, `A1:A15`, `M2:M41`).
2. Identify the **transformation** required (delete rows, sort, dedup, sum, classify, etc.).
3. Identify the **expected output type** for each target cell — number? string? date? Boolean?
4. Confirm the output is a **value**, not a description of how to compute the value.

If you cannot answer all four, re-read the prompt before writing code. Do NOT guess; do NOT pad with placeholder strings.

## Anti-patterns (these caused real failures — do NOT repeat them)

- **Writing the solution as text.** See the "READ THIS FIRST" section above. The grader gives 0 credit for VBA / M / prose in cells.
- Writing Excel formulas (`ws['B3'] = "=SUM(A3:A10)"`). The grader uses `openpyxl.load_workbook(path, data_only=True)`, which does NOT evaluate formulas — it returns `None` for any cell whose cached value is missing. **Write computed scalar values only**.
- Forgetting `wb.save(path)`. Mutations to `ws` are in-memory only until you save. You must save back to the **same path** that was given to you.
- Saving to a different filename (`out.xlsx`, `result.xlsx`, etc.). The grader inspects the original WORKBOOK PATH — overwrite it.
- Treating the file like a JSONL / log / CSV stream. Do NOT call `open(path).read()`, `read_json_auto`, `pd.read_csv`, `json.loads`, or `for line in open(path)`. Use `openpyxl.load_workbook` (or `pandas.read_excel`).
- Assuming the sheet name. If the prompt does not name a sheet, default to `wb.active` and print `wb.sheetnames` first to confirm there is only one sheet.
- **Writing to the wrong cell range.** The grader inspects ONLY the TARGET CELL RANGE in the prompt. Iterate that exact range; don't spill into adjacent rows or columns. **Off-by-one in the start row is the second-most-common failure** — if the target is `F2:H10`, write to rows 2..10 inclusive, not 1..9 or 3..11.
- Ignoring header rows. Spreadsheet data usually starts on row 2 (row 1 is the header). Inspect first; do not start computing from row 1.
- Off-by-one between Excel column letters (A, B, C…) and openpyxl 1-based indices. `ws['B3']` is column 2, row 3.
- Reading source cells with `ws.cell(...).value` when they are formulas — you get the literal string `'=D3+F3'`, not the computed number. **Read inputs from a `data_only=True` workbook** when the source contains formulas.
- Padding unfilled target cells with `"-"`, `""`, `"N/A"`, `"TBD"` or any placeholder. If you don't know what to write, the right answer is to recompute, not to fill with a sentinel.
- **Writing a value as prose or the wrong storage type.** If the expected output is a percentage, rate, currency, or count, store it as a NUMBER, not a sentence or a formatted string. Inspect how comparable cells in the workbook store such values (numeric with a number format vs text) and match that convention. `0.67` with a percent format and the text `"67%"` are different answers; a label like `"67% WIN RATE"` where a value belongs scores zero.
- **Verifying presence instead of correctness.** Reloading and confirming the target cells are filled and formula-free does not prove the values are right. Recompute a small sample of target cells by an independent method and compare; boundary rows are where near-miss failures concentrate.

## Reading source data

```python
header = [c.value for c in ws[1]]
rows = [dict(zip(header, [c.value for c in r]))
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
    ws[f"M{i}"] = val                        # literal scalar — NOT a formula, NOT prose

wb.save(path)                                # save back to the SAME path
```

## Large-range / sheet-level protocol

For sheet-level tasks or target ranges larger than a handful of cells, do not hand-edit scattered cells one at a time. Build a complete in-memory output table/matrix first, verify its dimensions, then write it in one pass.

1. Parse the target range with `openpyxl.utils.cell.range_boundaries`.
2. Compute the full output as `rows = [[...], [...]]`, including intentional blanks as `None`.
3. Assert `len(rows) == target_height` and every row has `target_width`.
4. Clear the target range before writing when the new output may be shorter than old content.
5. Write the matrix row-by-row, preserving blanks as `None`.
6. Reload and print summary counts: target cells, non-blank cells, formulas, Excel errors.

```python
from openpyxl.utils.cell import range_boundaries

TARGET = "H2:P8"  # use the actual prompt range
min_col, min_row, max_col, max_row = range_boundaries(TARGET)
height = max_row - min_row + 1
width = max_col - min_col + 1

# rows must be computed from the workbook data, not copied from this example
rows = [[None for _ in range(width)] for _ in range(height)]

assert len(rows) == height, (len(rows), height)
assert all(len(row) == width for row in rows), "output row width mismatch"

for r in range(min_row, max_row + 1):
    for c in range(min_col, max_col + 1):
        ws.cell(r, c).value = None

for i, row in enumerate(rows, start=min_row):
    for j, value in enumerate(row, start=min_col):
        ws.cell(i, j).value = value

wb.save(path)
```

For large sheet transforms (sort, filter, merge, delete rows, reshape tables), prefer workbook operations over per-cell guessing:

- Delete rows bottom-up with `ws.delete_rows`.
- Sort data in Python using row records, then rewrite the whole target table.
- For multi-sheet targets such as `A1:A50,Sheet2!A1:E20`, verify each range separately.
- If a filtered output has fewer rows than the target range, write the computed rows and leave the remaining target cells blank (`None`), not `#N/A`, `"-"`, or old values.

For row-deletion / row-insertion tasks, MUTATE THE STRUCTURE OF THE SHEET — do not write replacement text describing the deletion:

```python
# Task: "delete every row where col I is non-empty and col H is blank"
to_delete = []
for r in range(2, ws.max_row + 1):
    h = ws.cell(row=r, column=8).value
    i = ws.cell(row=r, column=9).value
    if i not in (None, "") and h in (None, ""):
        to_delete.append(r)
# Delete from the bottom up so indices stay valid
for r in reversed(to_delete):
    ws.delete_rows(r, amount=1)
wb.save(path)
```

For Power-Query / data-cleanup tasks, perform the equivalent transformation in Python and write the cleaned values:

```python
# Task: "in column A, replace each '#N/A' with the previous non-#N/A value"
import openpyxl
prev = None
for r in range(2, ws.max_row + 1):
    v = ws.cell(row=r, column=1).value
    if v == "#N/A" or v is None:
        ws.cell(row=r, column=1).value = prev
    else:
        prev = v
wb.save(path)
```

## Mandatory verification step

After `wb.save(path)`, reload with `data_only=True` and print every cell in the TARGET CELL RANGE. Confirm:

1. Every target cell has the value implied by the task. Blank cells are OK only when the correct output is blank, such as unused rows after extracting a shorter filtered list.
2. No target cell contains an unintended Excel error value like `#N/A`, `#VALUE!`, or `#REF!` unless the prompt explicitly asks for that literal error.
3. No non-blank value starts with `=` (which would mean you accidentally wrote a formula).
4. No non-blank value looks like VBA code, Power Query M, prose, or a placeholder (`"-"`, `"TBD"`).
5. The values look reasonable (right magnitude, right type, correct blank positions).
6. **Correctness spot-check**: recompute at least two target cells by an independent method (a different formula path, or a manual slice of the source rows) and assert they equal what you wrote. Always include the FIRST and LAST cells of each target range in this spot-check; off-by-one and boundary errors cluster there. If any spot-check disagrees, fix the computation and re-save before submitting.

```python
wb2 = openpyxl.load_workbook(path, data_only=True)
ws2 = wb2[ws.title]
from openpyxl.utils.cell import range_boundaries
TARGET = "M2:M41"   # ← use your actual TARGET CELL RANGE from the prompt
min_col, min_row, max_col, max_row = range_boundaries(TARGET)
for r in range(min_row, max_row + 1):
    for c in range(min_col, max_col + 1):
        v = ws2.cell(row=r, column=c).value
        coord = ws2.cell(row=r, column=c).coordinate
        print(coord, "=", repr(v))
        assert v not in ("#N/A", "#VALUE!", "#REF!", "#DIV/0!", "#NAME?"), (
            f"{coord} contains an Excel error; compute the intended value or leave blank only if blank is correct"
        )
        if v is None:
            continue
        if isinstance(v, str):
            assert not v.startswith("="), f"{coord} contains a formula string"
            forbidden = ("Sub ", "End Sub", "Power Query", "VBA", "Macro:", "let Source", "Application.", "ws.Range", "ws.Rows")
            for f in forbidden:
                assert f not in v, f"{coord} contains code/prose ({f!r}) instead of a value"
            assert v not in ("-", "TBD", "N/A", "see notes"), f"{coord} is a placeholder, recompute"
print("VERIFY OK")
```

If any assertion fails, fix the underlying bug (wrong range, wrong values, wrote prose) and re-save before submitting. Do NOT submit until VERIFY OK prints cleanly.

## Final answer convention

The final `answer` field is just confirmation that the file was saved. Reply with the single word `done` (or a short status string). Do NOT put a formula, a list of values, or a long explanation in the answer — the grader inspects the saved workbook, not the answer string.

## Quick decision tree

- Prompt gives a `.xlsx` path + a TARGET CELL RANGE → use this skill.
- Prompt asks "how would you do X?" or "describe the macro for X" → **NO**, that is not what SpreadsheetBench asks. Re-read the prompt — it is asking you to PERFORM X on the workbook.
- Prompt gives a `.jsonl` / `.csv` / log file path → use the `data_exploration` skill instead.
- Prompt gives both an `.xlsx` and an external data file → use this skill, but read the external file with the appropriate library (`pandas.read_csv`, etc.) before writing into the workbook.
