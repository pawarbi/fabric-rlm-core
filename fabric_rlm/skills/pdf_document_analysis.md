---
applies_when:
  keywords:
  - pdf
  - document
  - extract
  output_fields: []
excludes: []
depends_on:
- validation
specificity: domain
---
# pdf_document_analysis
Summary: Page-rendered, source-grounded playbook for long PDF and document-analysis tasks.
Dependencies: validation

## Gotchas

- Raw PDF text is useful for search and candidate discovery, but it can drop table
  structure, headers, footers, checkboxes, columns, and handwritten/embedded text.
- Never regex a whole flattened document for a value near a label. Section labels
  and stock phrases repeat, so the first match is usually from the wrong section.
  Locate the passage, print it, read it.
- Long-document failures often come from synthesizing too early. Inspect pages or
  page ranges first, then aggregate findings with page references.
- Dates and money appear in many equivalent forms. Normalize them before final
  submission, but keep original wording when ambiguity matters.
- A polished report can still score poorly if required sections, structured arrays,
  or final output keys are missing.

## PDF inspection pattern

1. Open every PDF with PyMuPDF (`fitz`) and record `page_count`.
2. Extract raw text for search/indexing only:
   - build page records like `{"page": n, "text": page.get_text("text")}`;
   - search this text for candidate dates, people, entities, amounts, headings, and
     requirement words;
   - do not treat raw text as the only evidence when layout matters.
3. Check interactive form widgets before assuming a visually filled form is blank.
   `page.get_text()` returns the template text but not widget values:

```python
form_values = {}
for page in doc:
    for widget in page.widgets() or []:
        if widget.field_name and widget.field_value not in (None, ""):
            form_values[widget.field_name] = widget.field_value
print(form_values)
```

4. Render candidate or all pages at about 200 DPI and pass page images to the
   model when the task requires faithful layout/table interpretation:

```python
import base64
import fitz

def render_page_data_uri(page, dpi=200):
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    data = base64.b64encode(pix.tobytes("png")).decode("ascii")
    return f"data:image/png;base64,{data}"
```

5. Chunk by page or small page ranges, not by arbitrary character windows. Keep the
   page number with every extracted fact.

## Pulling a specific fact out of a report

When the task needs one number, date, or named list rather than a whole-document
summary, locate the page first and read the passage. Searching the flattened
document for a value near a label is the most common way these tasks go wrong.
A term from the document's own title matches the cover page, and a stock phrase
like "revised upward by" or "subject to" matches whichever section comes first.

```python
import fitz

doc = fitz.open(path)
probe = "<phrase distinctive to the claim>"
for n, page in enumerate(doc, start=1):
    text = " ".join(page.get_text().split())   # collapse per page, never across
    i = text.find(probe)
    if i != -1:
        print(f"--- page {n} ---")
        print(text[max(0, i - 200):i + 600])   # read the surrounding sentences
```

Then take the value from the passage you just printed, not from a capture group
you never looked at.

- Probe on a phrase distinctive to the claim rather than the topic word. The
  topic word appears on every page; the claim is stated once.
- Print a window around every hit and confirm the context before using it. The
  same phrase on two pages is usually two different subjects.
- One sentence can carry several facts, and a clause can cover more than one
  entity. "by March in Spain, by June in France and Germany, and only in 2027 in
  Italy" names four countries, not three. Count entities, not clauses.
- Carry the sentence the value came from into the working table described under
  Source-grounded aggregation. A number with no source sentence cannot be
  audited or corrected.

## Tables that find_tables() cannot read

`page.find_tables()` handles ruled grids. On report-style tables with no cell
borders it returns header fragments or nothing at all, and `get_text()` flattens
the body to a bare run of numbers with no column meaning. Recover the columns
from geometry instead of guessing:

```python
rows = {}
for x0, y0, x1, y1, word, *_ in page.get_text("words"):
    rows.setdefault(round(y0), []).append((x0, word))
for y in sorted(rows):
    print([w for _, w in sorted(rows[y])])     # one printed line per table row
```

Read the header line to learn what each column means before mapping any value,
and check that the row label is the one you want. Documents routinely repeat a
row label, listing the same entity once under one measure and again under
another, so match on the section as well as the label.

## Page-level extraction

- Use **top-level `await asyncio.gather(...)`** to run independent page-level
  `predict()` calls in parallel — the interpreter executes your code inside an
  async event loop, so you can `await` directly. Fan out in **bounded batches**
  (e.g. 6-8 pages at a time) with `return_exceptions=True` so you don't trip
  provider rate limits and one bad page doesn't lose the batch:
  `results = await asyncio.gather(*[predict(...) for p in batch], return_exceptions=True)`.
  Ask for compact JSON-compatible facts per page: dates, entities, financial items,
  section summaries, and uncertainty notes.
- Prefer signatures with explicit outputs such as `dates`, `entities`,
  `financial_items`, and `evidence`. Include raw page text snippets for search
  context and rendered page data URIs when visual/layout evidence is needed.
- Deduplicate after extraction. Merge facts only when names, dates, amounts, and
  page evidence agree.

## Source-grounded aggregation

- Aggregate into a working table with columns for normalized value, original text,
  page number(s), category, confidence, and source snippet.
- Write the final report from the working table, not from memory. Cite pages in
  prose where helpful, especially for deadlines, obligations, and amounts.
- If evidence conflicts, report the conflict or choose the value supported by the
  clearest source and preserve the original wording.

## Date and value normalization

- Normalize dates to ISO `YYYY-MM-DD` when the year is known. Infer the year only
  from explicit document context; otherwise preserve the original text and mark
  ambiguity.
- Keep times and time zones in separate fields when requested.
- Preserve currency symbols, percentages, frequencies, deposits, penalties, and
  payment terms. Do not convert or round unless the task asks for it.
- Normalize entity names with common aliases, but keep contact names, roles,
  emails, phones, and addresses as separate fields when available.

## Required-section validation

Before `SUBMIT()`:

- Build the exact requested output object and verify every required key exists.
- Ensure every required report section is present with non-empty content.
- Check that structured lists contain dictionaries with the requested fields.
- Verify `page_count` matches the opened document(s), and that every important
  date/entity/financial item has source evidence.
- Run a small self-score: count missing required sections, empty arrays, unparsed
  dates, duplicate entities, unsupported claims, and missing file paths. Repair
  any failure before submitting.

## Pre-flight checklist

- [ ] The PDF was opened with PyMuPDF and all pages were enumerated.
- [ ] Interactive form widgets were checked for filled values.
- [ ] Raw text was used for search, while rendered pages were used for layout or
      visually important evidence.
- [ ] Every value taken from prose was read in its printed context, not captured
      by a regex over the whole document, and its source sentence was kept.
- [ ] Page-level extraction used independent `predict()` calls where it could
      improve coverage.
- [ ] Final facts were deduplicated and grounded to page numbers/snippets.
- [ ] Dates, entities, and financial values were normalized without losing the
      original source wording.
- [ ] Required sections, output keys, JSON-friendly types, and artifact paths were
      validated immediately before `SUBMIT()`.
