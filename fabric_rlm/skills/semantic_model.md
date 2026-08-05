---
applies_when:
  keywords:
    - semantic model
    - power bi
    - powerbi
    - dataset
    - dax
    - measure
    - measures
    - sempy
    - semantic link
    - tabular
    - star schema
    - fact table
    - dimension
    - report
    - kpi
    - yoy
    - ytd
    - by region
    - by category
    - by product
  output_fields: []
excludes: []
depends_on: []
specificity: domain
---
# semantic_model

Query a Power BI semantic model with `sempy`. Fabric notebooks only.

## Flow

```
list_tables / list_measures / list_relationships     <- always first
        |
   measure exists for the question?
     yes -> read its EXPRESSION, confirm it matches, use it
     no  -> write DAX over columns, say so in the answer
        |
   filtering on a column?  -> VALUES(col) first, match the real format
        |
   evaluate_measure (simple)  |  evaluate_dax + SUMMARIZECOLUMNS (grouped)
        |
   check: rows>0, parts~total, magnitude sane
        |
   SUBMIT
```

## Rules

| do | not |
| --- | --- |
| `fabric.evaluate_dax(ds, "EVALUATE ...")` | `read_table` on a fact table |
| `Sales[Year] = 2024` predicate | `FILTER('Sales', ...)` |
| `SUMMARIZECOLUMNS(col, "m", [M])` | `ADDCOLUMNS(SUMMARIZE(...))` |
| `CALCULATETABLE(SUMMARIZECOLUMNS(..), filt)` | filters inside SUMMARIZECOLUMNS |
| `SUMX(VALUES(t[c]), ..)` | `SUMX(t, ..)` over the whole table |
| `VAR x = [M]` reused | `[M]` three times |
| model's own measure | hand-rolled `SUM()` when a measure exists |
| ask for the grain needed | daily when monthly answers it |

## sempy works here

`import sempy.fabric` succeeds in this sandbox. Verified: import, `list_tables`,
`evaluate_dax` all return real data under the default security policy.

A failed call means **that call** failed. It never means sempy is unavailable.
Never answer "cannot be queried from this environment" - print the exception,
read it, try another query.

"Unanswerable" = the model has no such table/column/measure. It is never a
statement about the sandbox.

## Names lie

Measure names may not describe what they compute. Seen in real models:
`sls_amt_x` = sales for 2 categories only; `gm2_pct` = margin excluding 1
category; `po_ok_flagish` = on-time / high-value POs; `Day Yield Pct` = excludes
night shift. Each returns a plausible wrong number.

`list_measures` returns the DAX. Read it. Prefer the measure whose expression
has no extra filters.

Column values lie too: `US-West` vs `US West`, `2024` int vs `"2024"` str,
leading zeros, trailing spaces. A filter that matches nothing returns **blank,
not an error**. `VALUES(col)` before filtering.

## Check before SUBMIT

- 0 rows after a filter = the filter is wrong, not the answer
- grouped parts should ~ reconcile to the ungrouped total
  (`SUMMARIZECOLUMNS` drops blank keys, so a small gap can be real)
- magnitude plausible against a known quantity
- state the form: `0.0314` and `3.14%` are the same number
- one figure cross-checked a second way

## Python is available

DAX is not the only tool. `evaluate_dax` returns a DataFrame - reduce with DAX,
then use pandas for anything awkward to express in DAX (reshaping, string work,
joins against other inputs). Aggregate server-side first; never pull a fact
table to aggregate locally.
