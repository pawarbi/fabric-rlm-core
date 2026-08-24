---
applies_when:
  keywords:
  - filing
  - 10-K
  - 10-Q
  - earnings release
  - balance sheet
  - income statement
  - cash flow statement
  output_fields: []
excludes: []
depends_on:
- pdf_document_analysis
specificity: domain
---
# financial_documents
Summary: Reporting conventions for filings and financial statements: parentheses as negative, scale headers, fiscal periods, adjacent period columns, subtotal and contra rows, restatements and non-GAAP measures.

Reporting conventions for filings and financial statements: 10-K, 10-Q, annual
reports, earnings releases. Use it alongside `pdf_document_analysis`, which covers
the mechanics of getting text and tables off a page. This playbook covers what the
numbers on that page mean.

These rules are specific to financial reporting and are wrong elsewhere. In a legal
document parentheses mark a subsection; in a scientific paper they mark a citation
or an uncertainty bound. Do not apply this playbook to a document that is not a
financial statement.

## Parentheses mean negative

`(1,577)` is -1,577 and `(0.6)%` is -0.6%. Flattened text keeps the brackets and
loses every other visual cue, so a figure that is a loss reads as a gain.

- Convert before you compare, sort or report. A largest-value comparison over a
  column holding `(4,690)` and `2,727` is wrong unless the bracket became a minus
  sign first.
- Carry the sign into the answer. "The effective tax rate was 14.7%" is a different
  claim from "-14.76%". The narrative around the table normally states the
  direction; check your number against that sentence before reporting it.
- A dash, an em dash or an empty cell means nil. It is not a zero you can compare.

## Scale sits in a header, not beside the number

A table states its unit once, in a line above the body: "(In millions)", "(In
thousands, except per share data)". The cells themselves are bare.

- Find the scale line before reporting any figure, and state the unit in the answer.
- One page can hold two tables at different scales. Per-share figures are almost
  always in whole dollars even when the table around them is in millions.

## Fiscal year is not calendar year

Many filers close outside December: Nike in May, Microsoft in June, Best Buy and
Ulta in late January or early February.

- "FY2023" means that company's fiscal 2023, which can sit mostly in calendar 2022.
- Take the period from the statement heading, not from the filing date on the cover.

## Periods sit side by side

A 10-Q prints three-month and nine-month columns next to each other; a 10-K prints
two or three years. The row label is identical across all of them.

- Read the column heading for every figure you take. A right row read under the
  wrong column is the most common error in these documents.
- Where a filing shows both, "the quarter" means the three-month column.

## Totals, subtotals and contra rows

Statements are arithmetic hierarchies, not lists of comparable items.

- A total equals the sum of its siblings. Test that arithmetically, because labels
  lie: a total can be called "Accounts payable and accrued expenses".
- Contra rows such as "Less: accumulated depreciation" reduce the section they sit
  in. They are not ordinary line items and normally should not compete in a
  comparison between line items.
- Segment tables carry a "Corporate and unallocated" plug row, which is not a
  segment. Exclude it when the question asks about segments.

## Restatements and non-GAAP measures

- A prior-year column may be restated, so the FY2021 figure printed in the FY2022
  filing can differ from the one printed in the FY2021 filing. Say which filing the
  figure came from.
- Adjusted, core, organic and similar measures are defined by the company, not by a
  standard. Use the company's own reconciliation table and name the measure you
  used.
