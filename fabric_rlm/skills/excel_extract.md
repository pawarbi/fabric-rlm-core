---
applies_when:
  keywords:
    - extract
    - extraction
    - parse
    - structured
    - schema
    - json
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
    - hidden
    - merged
    - comment
    - indent
    - number format
    - format string
output_fields: []
excludes: []
depends_on: []
specificity: domain
---
# excel_extract

Summary: You are READING a `.xlsx` workbook and emitting structured data (JSON, list of records, schema fields). The file may be MESSY — hidden rows/cols/sheets, cell comments holding values, custom `number_format` strings encoding meaning, indent-encoded hierarchy, merged headers, multiple tables on one sheet. Your first turn must scan **all** the metadata channels openpyxl exposes; only then classify rows and emit values.

Use this skill (not `excel_modify`) when the task is to *read and return* data, not to compute and write back into the workbook.

You never read raw bytes back into the LM. Print compact summaries (counts, samples, channel hits) — never dump full sheets.

## Purpose

Extract structured records from `.xlsx` workbooks where the truth may live in any of the structural channels markdown/text conversion destroys: hidden cells, cell comments, number-format strings, alignment indent, hidden sheets, merged ranges, defined names.

## Contract: output fields

This is a generic skill. The runtime/prompt defines the actual output fields per task. Whatever fields you emit, follow these field-level conventions:

- **Value fields** — *str | number | date*. Emit the *semantic* value, not just the displayed text. If a cell shows `"see note"` and has a comment with the real value, the comment text is the answer. If a cell displays `"PAID"` via a custom format string but `cell.value` is `1`, prefer `cell.value` for numeric reasoning and surface the displayed string separately when the schema asks for it.

- **Source/provenance fields** — *str enum*. When the schema includes a field like `<x>_source`, label one of: `cell_value | cell_comment | hidden_row | hidden_column | hidden_sheet | merged_header | indent_hierarchy | number_format | defined_name | external_ref | unknown`.

- **`hidden_in_source: bool`** — true if this record came from a row/column/sheet whose `hidden` flag was set. Surfacing the flag is preferred to silently dropping or including hidden data.

- **`indent_level: int`** — 0-based, sourced from `cell.alignment.indent` on the label column. Use the indent of the *label column*, not whatever happens to be column A.

- **Row references** — *int*. Always emit absolute 1-based openpyxl row numbers (`cell.row`), never enumerate indices.

## Required verifier

Before `SUBMIT`, run a self-check that the payload is structurally consistent. The check below is generic — extend it with task-specific invariants from the prompt schema.

```python
def verify(payload):
    """Generic structural verifier for extracted records."""
    assert payload is not None, "payload is None"
    # If the payload is a dict of per-section results, walk each.
    sections = payload.values() if isinstance(payload, dict) else [payload]
    for section in sections:
        if section is None:
            continue
        if isinstance(section, dict):
            for k, v in section.items():
                if k.endswith("_source"):
                    assert isinstance(v, str) and v, f"{k} must be a non-empty enum string"
                if k == "indent_level":
                    assert isinstance(v, int) and v >= 0, "indent_level must be int >= 0"
                if k == "hidden_in_source":
                    assert isinstance(v, bool), "hidden_in_source must be bool"
                if k == "row":
                    assert isinstance(v, int) and v >= 1, "row must be 1-based int >= 1"
        if isinstance(section, list):
            for rec in section:
                if not isinstance(rec, dict):
                    continue
                # Don't emit a record that is entirely None — silent failure.
                non_null = [v for v in rec.values() if v not in (None, "", [])]
                assert non_null, f"record has no non-null fields: {rec}"
```

Call `verify(payload)` and only `SUBMIT(...)` if it raises nothing.

## Tripwires

1. **Treating row 1 as the header without checking.** Many real workbooks have title banners, merged headers, blank rows, or metadata rows above the actual header. Scan candidate header rows and pick the one whose cells are mostly short labels (strings, no digits-only values).
2. **Returning the visible value when a hidden row has the truth.** If a row with `row.hidden == True` shares a key (SKU, ID, name) with a visible row that has a placeholder/decoy value, the hidden row is more likely to be the truth — at minimum, surface both with `hidden_in_source` flags.
3. **Dropping cell comments.** `cell.comment.text` often holds the real value behind a placeholder cell value like `"see note"`, `"TBD"`, `"(see master list)"`. Always probe `cell.comment is not None` over the populated area.
4. **Confusing display with value when `number_format` is custom.** A cell with `number_format == '"PAID";"PAID";"DUE"'` and `value == 1` displays `PAID`. The numeric `1` is the underlying truth; the format string is the encoding. For status flags / accounting / quarter labels, both are meaningful.
5. **Forgetting `data_only=True` for cached values.** Cells containing formulas show `'=A1+B1'` under `data_only=False`. Always load the workbook *twice* if any cell is a formula, and read inputs from the `data_only=True` view.
6. **Skipping hidden sheets.** Iterate `wb.sheetnames` (which includes hidden sheets in openpyxl) and check `ws.sheet_state` — `"hidden"` and `"veryHidden"` sheets often contain canonical totals or master lists.
7. **Misclassifying a data row as a header.** If your column-index detection (looking for "code"/"qty"/"description") fails on a candidate header row, that row is probably a *data* row, not a header — fall back to the previously detected header above, or scan upward.
8. **Emitting an empty list when the data region is populated.** If you produce an empty `[]` for a field whose source sheet has populated cells in its data region, your header/region classification is wrong. Stop, re-detect the header, or fall back to positional column mapping (column 1 → first schema field, column 2 → second, …) using the schema's own field order. Never silently submit `[]` — at minimum, surface a `region_detection_failed: true` diagnostic in the payload.
9. **Confusing a hidden column's *header label* with its *data value*.** A hidden column typically has a label cell at the header row (e.g. a metadata word) and one or more populated cells in the data rows below. When a schema field asks for a *value* (e.g. a name, a flag, an ID), you want the populated data cell whose row aligns with the entity row — never the header label, and never a placeholder string from a *different visible* column. The same logic applies to hidden rows: its leftmost cell may be a label, the right-side cells are the data.
10. **Treating one sheet as one table.** A sheet may contain several distinct table regions stacked vertically (separated by blank rows, banners, or merged titles), or arranged side-by-side. Detect each region's header *separately* before mapping columns, otherwise you will read column B as if it were column A of the next table.

## Invariants

- Every record emitted comes from at least one populated cell (no records made entirely of `None`).
- For every sheet in `wb.sheetnames`, you have inspected `sheet_state`, populated dimensions, hidden row/col counts, and merged range count at least once.
- For every populated cell whose `cell.comment` is not None, you have decided whether the comment carries data or commentary; if data, it appears in the output.
- Cells whose `cell.number_format` is non-`"General"` and is custom (contains `"`-quoted literal segments or `;` separators) have been classified — either the underlying `cell.value` or the displayed string is in the output (or both, per schema).
- Hidden rows and columns have been examined; the decision to include or drop them is recorded via a flag, not silent.
- Output row references (when emitted) are absolute 1-based and match `cell.row` / `cell.col_idx` from the workbook.

## Procedure

### Mandatory first-turn discovery

Run this once on the workbook before any per-record classification. It is cheap (a few thousand printed chars) and prevents the wrong-row-as-header / hidden-data-missed / comment-dropped failure modes that account for most extraction errors on real files.

**Use the actual workbook path from the prompt — do NOT copy the placeholder string below verbatim.**

```python
import openpyxl
from collections import Counter

path = r"<WORKBOOK_PATH_FROM_PROMPT>"   # ← REPLACE with the real path

wb   = openpyxl.load_workbook(path, data_only=False)   # formulas
wb_v = openpyxl.load_workbook(path, data_only=True)    # cached values

# 1) Sheet inventory — INCLUDING hidden sheets.
print("=== sheets ===")
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f"  {sn!r:25s} state={ws.sheet_state:10s} dims={ws.dimensions}  rows={ws.max_row}  cols={ws.max_column}")

# 2) Per-sheet structural skeleton.
def scan_sheet_meta(ws, ws_v, sample_limit=120):
    hidden_rows = [r for r,d in ws.row_dimensions.items() if d.hidden]
    hidden_cols = [c for c,d in ws.column_dimensions.items() if d.hidden]
    merged = [str(r) for r in ws.merged_cells.ranges]
    comments, custom_fmts, indents = [], Counter(), Counter()
    rmax = min(ws.max_row, sample_limit)
    cmax = min(ws.max_column, 20)
    for r in range(1, rmax + 1):
        for c in range(1, cmax + 1):
            cell = ws.cell(r, c)
            if cell.comment is not None:
                comments.append((cell.coordinate, cell.comment.text))
            nf = cell.number_format
            if nf and nf != "General" and ('"' in nf or ";" in nf):
                custom_fmts[nf] += 1
            if cell.alignment and cell.alignment.indent:
                indents[cell.alignment.indent] += 1
    return {
        "hidden_rows": hidden_rows,
        "hidden_cols": hidden_cols,
        "merged": merged[:8],
        "n_merged": len(merged),
        "comments": comments[:8],
        "n_comments": len(comments),
        "custom_number_formats": dict(custom_fmts.most_common(5)),
        "indent_levels": dict(indents),
    }

print("\n=== per-sheet metadata channels ===")
for sn in wb.sheetnames:
    ws = wb[sn]
    ws_v = wb_v[sn]
    meta = scan_sheet_meta(ws, ws_v)
    print(f"\n-- {sn} (state={ws.sheet_state})")
    for k, v in meta.items():
        print(f"   {k}: {v}")

# 3) For each VISIBLE sheet, print first 6 non-empty rows from cols 1..min(8,max_col)
#    showing BOTH the formula view and the value view if they differ.
print("\n=== sheet previews (visible only) ===")
for sn in wb.sheetnames:
    ws = wb[sn]
    if ws.sheet_state != "visible":
        continue
    ws_v = wb_v[sn]
    print(f"\n-- {sn}")
    shown = 0
    for r in range(1, ws.max_row + 1):
        cmax = min(8, ws.max_column)
        f_row = [ws.cell(r, c).value for c in range(1, cmax + 1)]
        if any(v not in (None, "") for v in f_row):
            v_row = [ws_v.cell(r, c).value for c in range(1, cmax + 1)]
            print(f"   r{r:03d} F:", f_row)
            if v_row != f_row:
                print(f"        V:", v_row)
            shown += 1
            if shown >= 6:
                break

# 4) Detect ALL candidate header rows per sheet (multi-table aware).
def looks_like_header(vals):
    nonempty = [v for v in vals if v not in (None, "")]
    if len(nonempty) < 2:
        return False
    str_share = sum(1 for v in nonempty if isinstance(v, str)) / len(nonempty)
    return str_share >= 0.7

print("\n=== candidate header rows per sheet (multi-table aware) ===")
for sn in wb.sheetnames:
    ws = wb[sn]
    if ws.sheet_state != "visible":
        continue
    hdrs = []
    for r in range(1, ws.max_row + 1):
        vals = [ws.cell(r, c).value for c in range(1, min(10, ws.max_column + 1))]
        if looks_like_header(vals):
            hdrs.append((r, [v for v in vals if v not in (None, "")][:6]))
    print(f"-- {sn}: {len(hdrs)} candidate header row(s)")
    for r, vals in hdrs[:8]:
        print(f"   r{r:03d}: {vals}")

# 5) Dump every populated cell in every hidden row, hidden column, hidden sheet.
#    These are the highest-signal places where canonical or audit data hides.
print("\n=== ALL populated hidden cells (rows, cols, sheets) ===")
from openpyxl.utils import get_column_letter, column_index_from_string
for sn in wb.sheetnames:
    ws = wb[sn]
    ws_v = wb_v[sn]
    h_rows = sorted(r for r, d in ws.row_dimensions.items() if d.hidden)
    h_cols = sorted(c for c, d in ws.column_dimensions.items() if d.hidden)
    if not h_rows and not h_cols and ws.sheet_state == "visible":
        continue
    print(f"\n-- {sn} (state={ws.sheet_state}, hidden_rows={h_rows}, hidden_cols={h_cols})")
    # Hidden rows
    for r in h_rows:
        vals = [ws_v.cell(r, c).value for c in range(1, ws.max_column + 1)]
        if any(v not in (None, "") for v in vals):
            print(f"   HIDDEN ROW r{r}: {vals[:8]}")
    # Hidden columns — print header + every populated value cell
    for cl in h_cols:
        ci = column_index_from_string(cl) if isinstance(cl, str) else cl
        col_data = []
        for r in range(1, ws.max_row + 1):
            v = ws_v.cell(r, ci).value
            if v not in (None, ""):
                col_data.append((r, v))
        if col_data:
            print(f"   HIDDEN COL {cl} (idx {ci}): {col_data[:10]}")
    # If sheet itself hidden, dump first 10 populated rows
    if ws.sheet_state != "visible":
        print(f"   HIDDEN SHEET — first 10 populated rows:")
        shown = 0
        for r in range(1, ws.max_row + 1):
            vals = [ws_v.cell(r, c).value for c in range(1, min(8, ws.max_column + 1))]
            if any(v not in (None, "") for v in vals):
                print(f"     r{r}: {vals}")
                shown += 1
                if shown >= 10:
                    break
```

After this prints, write a one-paragraph plan: which sheets are visible, what header rows you will use *for each sub-table region* on each sheet, which channels carry the data the prompt asks for (cell value? comment? number_format? indent? hidden row/col? hidden sheet?). For every populated hidden cell printed above, decide which schema field it belongs to (by row alignment) — do NOT skip any populated hidden value. Only then write extraction code.

### Extraction patterns

Choose patterns based on what the discovery surfaced.

**Hidden rows/columns**: probe `row_dimensions[r].hidden` / `column_dimensions[L].hidden` and emit `hidden_in_source: True` (or whatever flag the schema asks for) rather than silently merging or silently dropping.

```python
for r in range(header_row + 1, ws.max_row + 1):
    rec = {col: ws.cell(r, col_idx[col]).value for col in fields}
    rec["hidden_in_source"] = ws.row_dimensions[r].hidden
    records.append(rec)
```

**Cell comments**: read `cell.comment.text` (not `cell.comment` — that's an object) and decide if it carries a value. Strip a leading label like `"Confirmed phone:"` if the schema wants only the value.

```python
for row in ws.iter_rows():
    for cell in row:
        if cell.comment is not None:
            text = cell.comment.text  # e.g. "Confirmed phone: +353 1 555 9921"
            # Heuristic: if the cell value looks like a placeholder, prefer the comment.
```

**Custom `number_format`**: when the format contains `;`-separated quoted literals, the *displayed* string is the format choice for that value's sign/zero branch. **Iterate the populated data region** and emit one record per row where the format is custom — don't just sample a single cell.

```python
nf_records = []
for r in range(header_row + 1, ws.max_row + 1):
    for c in range(1, ws.max_column + 1):
        cell = ws.cell(r, c)
        nf = cell.number_format
        if not nf or nf == "General":
            continue
        if '"' not in nf and ";" not in nf:
            continue
        # Custom format with quoted literals → categorical encoding.
        # Resolve the displayed branch from the value sign:
        branches = nf.split(";")
        v = cell.value
        if isinstance(v, (int, float)):
            idx = 0 if v > 0 else (1 if v < 0 else (2 if len(branches) >= 3 else 0))
            display = branches[idx].strip().strip('"')
        else:
            display = None
        nf_records.append({
            "row": cell.row, "col": cell.column,
            "underlying": v, "display": display, "format": nf,
        })
# Emit nf_records to the payload (schema-shaped) — do NOT collapse to a single row.
```

**Indent-encoded hierarchy**: `cell.alignment.indent` returns 0, 1, 2... for the label column.

```python
items = []
for r in range(start_row, end_row + 1):
    label_cell = ws.cell(r, label_col)
    items.append({
        "label": (label_cell.value or "").strip(),
        "indent_level": (label_cell.alignment.indent or 0) if label_cell.alignment else 0,
    })
# is_parent: this row's indent is less than the next non-empty row's indent
for i, it in enumerate(items):
    nxt = next((items[j]["indent_level"] for j in range(i+1, len(items))), it["indent_level"])
    it["is_parent"] = nxt > it["indent_level"]
```

**Hidden sheets**: iterate `wb.sheetnames` (which already includes hidden sheets), check `ws.sheet_state in {"hidden", "veryHidden"}`, and read them like any other sheet.

```python
hidden_sheets = [sn for sn in wb.sheetnames if wb[sn].sheet_state != "visible"]
```

**Merged headers**: `ws.merged_cells.ranges` lists `MergedCellRange` objects. The value lives in the top-left cell only. When unmerging mentally, propagate that value across the spanned cells before classifying.

### Header-row detection (don't assume row 1)

```python
def looks_like_header(row_values):
    nonempty = [v for v in row_values if v not in (None, "")]
    if len(nonempty) < 2:
        return False
    # Mostly short strings, no all-numeric row.
    str_share = sum(1 for v in nonempty if isinstance(v, str)) / len(nonempty)
    return str_share >= 0.7

header_row = None
for r in range(1, min(20, ws.max_row + 1)):
    row = [ws.cell(r, c).value for c in range(1, min(10, ws.max_column + 1))]
    if looks_like_header(row):
        header_row = r
        break
```

If detection finds nothing in the first 20 rows, the sheet may be a key/value layout (label in col A, value in col B, repeated) — handle that explicitly by iterating and matching on the label column.

### Column mapping: header label → column index (mandatory)

**Once you have `header_row`, build a dict that maps schema field names to column indices by header label.** Never grab data with raw positional indices (`ws.cell(r, 1)`) without first checking what header sits above that column. Reading the wrong column is the #1 silent-failure mode on real workbooks.

```python
def build_field_to_col(ws, header_row, schema_field_aliases):
    """schema_field_aliases: {"sku": ["sku","material code","code","item","part"],
                              "description": ["description","desc","name","item name"],
                              "quantity": ["qty","quantity","count","amount"],
                              "uom": ["uom","unit","units","u/m"]} """
    headers = {}
    for c in range(1, ws.max_column + 1):
        v = ws.cell(header_row, c).value
        if isinstance(v, str) and v.strip():
            headers[v.strip().lower()] = c
    field_to_col = {}
    for field, aliases in schema_field_aliases.items():
        for a in aliases:
            if a in headers:
                field_to_col[field] = headers[a]
                break
    return field_to_col, headers

field_to_col, all_headers = build_field_to_col(ws, header_row, {
    "sku": ["sku","material code","code","item","part","material_code"],
    "description": ["description","desc","name","item name"],
    "quantity": ["qty","quantity","count","amount"],
    "uom": ["uom","unit","units","u/m"],
})
print("field_to_col:", field_to_col, " | headers seen:", all_headers)
# If a required field has NO column match, log it before iterating data rows —
# don't silently fall back to col 1.
```

Then iterate data rows using `field_to_col[field]`, never positional indices. After building records, **sanity-check** the first 3: do their `sku` values look like real SKUs (alphanumeric codes, not "Material" or "Type")? If not, you mapped to the wrong column — re-detect.

### Pre-submit reflection

Before `SUBMIT`, build a small **schema → channel** map and walk it:

```python
# Example shape — one row per output field.
channel_map = {
    # "field_name": "channel: cell.value | cell.comment | number_format | hidden_row | hidden_col | hidden_sheet | indent | merged",
}
# For every required output field, channel_map MUST have a non-empty entry.
# If you have a null/empty list/None for a field, ask: did I check ALL channels?
#   - cell.comment over the populated area
#   - row.hidden / column.hidden in the data region
#   - hidden sheets via wb.sheetnames + sheet_state
#   - cell.number_format with quoted literals
#   - cell.alignment.indent for hierarchy
# Only after that may you emit null for that field.
```

Then walk through:
1. Every output field — does `channel_map[field]` name a real channel? Did you actually open that channel in code?
2. Every record — is at least one field non-null? Empty `[]` arrays for non-empty data regions are forbidden (see Tripwire 8).
3. Every row reference emitted — is it 1-based and matches `cell.row`?
4. Did any sheet get skipped purely because `sheet_state != "visible"`? Was that intentional?
5. For every populated *hidden* row/col/sheet, is the data either represented in output or explicitly flagged as dropped? (Silent drops are bugs.)

Emit `SUBMIT(...)` only after `verify(payload)` raises nothing.

## Resolving a target column from a natural-language description

When the user describes the column they want using prose ("the headline X index", "the average price of Y", "annual rate for category Z") and the workbook has many similarly-named columns, treat picking the right column as its own subtask. **Do not pick the first substring match.** The dominant failure mode on wide schemas is grabbing a sub-component, contribution, or historical variant when the user asked for the canonical/headline series.

Apply these rules in order:

### 1. Use every qualifying token the user typed

Every modifier, parenthetical, unit, base-period, or numeric tag in the user's question is a disambiguator the user gave you on purpose. Examples of disambiguators (generalize, do not memorize):

- Quoted literals or values inside parentheses (e.g. "(2015=100)", "(% change)", "(Q1)")
- Units (`/kg`, `per capita`, `%`, `bps`)
- Modifiers (`headline`, `seasonally adjusted`, `excluding`, `total`, `gross`, `net`, `weighted`)
- Time qualifiers (`annual`, `monthly`, `YoY`, `cumulative`)
- Geographic / segment scopes (`UK`, `national`, `urban`, `Tier 1`)

When you build a candidate filter, include **all** of these tokens as required substrings. Dropping any one of them is the most common reason for picking the wrong column.

### 2. If your filter returns >3 candidates, that's ambiguity — do not guess

Print the candidate list. If you must pick, apply tie-breakers in this order:

1. **Parsimony.** Prefer the *shortest* matching title. Long titles usually carry extra qualifiers (`Contribution to...`, `Excluding...`, `Modelled...`, `Component of...`) that mark sub-series. The canonical series is usually the shortest title that still matches all required tokens.
2. **Reject sub-component markers.** Down-rank or exclude titles containing words like `Contribution`, `Component`, `Subseries`, `Excluding`, `Excl`, `Modelled`, `Imputed`, `Historical`, `Forecast`, `Adjustment`, `Rounding`, `Effect`. These are almost never what the user asked for unless they explicitly used those words.
3. **Match base/scale tokens exactly.** If the user mentioned a base year, base period, or scale (e.g. `2015=100`, `Jan 1987=100`, `index`, `level`, `rate`), require it in the title — do not accept a near-match (`1965=100` is not a substitute for `2015=100`).
4. **Confirm with the value.** After picking, print the matched title alongside a sample value so the verification step can catch a wrong pick.

If after all tie-breakers you still have multiple plausible matches, **return them as candidates in the payload** (with their column indices and a sample value each) rather than picking arbitrarily. Honest ambiguity beats confident wrong.

### 3. Required pattern for natural-language column lookup

```python
import re

def _required_tokens(question_text):
    """Pull tokens from a natural-language description that MUST appear in the matched column header.

    Includes: parenthetical hints, quoted literals, base-year tokens (YYYY=NN, NN=YYYY),
    units (per X, /X, %), and capitalized acronyms / category names.
    Generic — does not encode any domain.
    """
    toks = []
    # parentheticals
    toks += [m.group(1).strip() for m in re.finditer(r"\(([^)]+)\)", question_text)]
    # quoted strings
    toks += re.findall(r'"([^"]+)"', question_text)
    # base-year style "2015=100" / "100=2015"
    toks += re.findall(r"\b\d{2,4}\s*=\s*\d{2,4}\b", question_text)
    # units like "per kg", "/kg", "per capita", "% change"
    toks += re.findall(r"per\s+\w+|/\w+|%\s*\w*", question_text, flags=re.I)
    return [t for t in toks if t]


def find_column_candidates(titles, question_text, must_contain=None, exclude_markers=None):
    """Score and rank candidate columns for a natural-language question.

    titles: iterable of (col_idx, title_str)
    must_contain: list of explicit substrings (case-insensitive) the title MUST contain
    exclude_markers: list of substrings that disqualify a candidate (sub-series markers)
    """
    must = [s.lower() for s in (must_contain or [])]
    must += [t.lower() for t in _required_tokens(question_text)]
    exclude = [s.lower() for s in (exclude_markers or [
        "contribution", "component", "excluding", "excl ", "excl.",
        "modelled", "modeled", "imputed", "historical", "forecast",
        "adjustment", "rounding", "effect", "subseries"
    ])]

    candidates = []
    for col, title in titles:
        t = (title or "").lower()
        if must and not all(s in t for s in must):
            continue
        is_subseries = any(m in t for m in exclude)
        candidates.append({
            "col": col, "title": title,
            "len": len(title or ""),
            "is_subseries": is_subseries,
        })
    # Rank: non-subseries first, then shortest title (parsimony)
    candidates.sort(key=lambda x: (x["is_subseries"], x["len"]))
    return candidates


# Usage:
# titles = [(c, ws.cell(header_row, c).value) for c in range(1, ws.max_column+1)]
# cands  = find_column_candidates(titles, user_question, must_contain=["all items"])
# if not cands:
#     # filter too strict — relax must_contain by one token and retry
#     ...
# elif len(cands) == 1:
#     pick = cands[0]
# else:
#     # >1 plausible match — print top 5 and either pick the top-ranked one
#     # OR return all of them as candidates in the payload for user review.
#     for c in cands[:5]:
#         print(f"  col={c['col']}  len={c['len']}  sub={c['is_subseries']}  {c['title']!r}")
#     pick = cands[0]
```

### 4. Always verify the pick

After you select a column for a value lookup, print the picked title and the value before submitting:

```python
print(f"PICK: col={pick['col']}  title={pick['title']!r}  value={ws_v.cell(value_row, pick['col']).value}")
```

This one print line is your safety net. If a downstream check shows a suspect value, the trajectory will tell you exactly which column was chosen and why.

### 5. When multiple columns are equally plausible — return candidates, not a guess

For agent-style use cases, prefer this output shape over a single-value answer when ambiguity remains:

```json
{
  "answer": null,
  "candidates": [
    {"col": 756,  "title": "...short canonical title...",  "sample_value": 133.0},
    {"col": 2407, "title": "...other plausible match...",  "sample_value": 131.6}
  ],
  "reason": "multiple columns matched all required tokens; user disambiguation needed"
}
```

This converts a silent wrong answer into an answerable question.

## Quick decision tree

- Need to *read and return* structured data from a `.xlsx`? → use this skill.
- Need to *compute and write back* into the workbook (TARGET CELL RANGE prompt)? → use `excel_modify`.
- File has hidden cells / comments-as-data / custom number_format / indent hierarchy / hidden sheets? → this skill, definitely.
- File is a clean rectangular table and you just need columns? → still use this skill, but the discovery turn will be cheap and confirm there's nothing exotic.
- User describes the target column with prose and the schema is wide (>20 columns with overlapping names)? → use the *Resolving a target column from a natural-language description* section above.
