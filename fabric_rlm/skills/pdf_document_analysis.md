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
3. Render candidate or all pages at about 200 DPI and pass page images to the
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

4. Chunk by page or small page ranges, not by arbitrary character windows. Keep the
   page number with every extracted fact.

## Page-level extraction

- Use **top-level `await asyncio.gather(...)`** to run independent page-level
  `predict()` calls in parallel — the interpreter executes your code inside an
  async event loop, so you can `await` directly. Example:
  `results = await asyncio.gather(predict(...), predict(...), predict(...))`.
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
- [ ] Raw text was used for search, while rendered pages were used for layout or
      visually important evidence.
- [ ] Page-level extraction used independent `predict()` calls where it could
      improve coverage.
- [ ] Final facts were deduplicated and grounded to page numbers/snippets.
- [ ] Dates, entities, and financial values were normalized without losing the
      original source wording.
- [ ] Required sections, output keys, JSON-friendly types, and artifact paths were
      validated immediately before `SUBMIT()`.
