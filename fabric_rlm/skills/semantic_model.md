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
    - drill
    - slice
    - by region
    - by category
    - by product
  output_fields: []
excludes: []
depends_on: []
specificity: domain
---
# semantic_model

Summary: query a Power BI semantic model with semantic link. Read the model's
metadata FIRST, let DAX do the aggregation server-side, and bring back only
small results. Never pull a fact table into the notebook.

This works only inside a Fabric notebook, where `sempy` ships in the runtime.

## sempy is available. Do not conclude otherwise.

`import sempy.fabric` works inside the execution sandbox. It has been verified
there with the default security policy active: import, `list_tables`,
`list_measures` and `evaluate_dax` all succeed and return real data.

This matters because the most common failure on this task is not a wrong
number, it is **giving up**. A model hits an unrelated sandbox message about
network egress, concludes the whole semantic-link path is unavailable, writes a
confident paragraph explaining why the question cannot be answered, and submits
that. Meanwhile the same call succeeds elsewhere in the same session.

So:

- If one call fails, that call failed. It says nothing about the others.
- A message about network egress refers to the specific call it names. It does
  not mean `sempy` is disabled.
- Never answer "this cannot be queried from this environment". If a query
  fails, print the exception, read it, and try a different query.
- Only say the question is unanswerable when the **model lacks the data** -
  no such table, no such column, no such measure. That is a statement about the
  semantic model, never about the sandbox.

## Mandatory first turn: read the model before querying it

Do this before writing any DAX. It costs one turn and saves several.

```python
import sempy.fabric as fabric

DATASET = "<the semantic model name>"

tables = fabric.list_tables(DATASET)
print(tables[["Name"]].to_string() if hasattr(tables, "columns") else tables)

measures = fabric.list_measures(DATASET)
# Name AND expression: the name alone is not enough - see the trap below.
cols = [c for c in ("Table Name", "Measure Name", "Measure Expression")
        if c in getattr(measures, "columns", [])]
print(measures[cols].to_string()[:4000])

rels = fabric.list_relationships(DATASET)
print(rels.to_string()[:2000])
```

Measure names are not guessable and column names are not either. Guessing costs
a turn per guess.

## Read the descriptions - they are free context

`list_tables`, `list_columns` and `list_measures` each return a **Description**
column. Model authors put the business meaning there: what a measure counts,
which grain a table sits at, what a coded column means. It is the cheapest
context available and it is already in the result you just fetched.

```python
m = fabric.list_measures(DATASET)
cols = [c for c in ("Table Name", "Measure Name", "Measure Expression",
                    "Description") if c in m.columns]
print(m[cols].to_string()[:4000])

t = fabric.list_tables(DATASET)
print(t[[c for c in ("Name", "Description") if c in t.columns]].to_string())
```

`displayFolder` on a measure groups it semantically - "Sales", "Quality",
"Inventory" - which hints at which measures belong to which question.

Descriptions are often empty. Empty means nobody filled them in, not that the
field is unimportant.

A model may also carry conventions that are written down nowhere in the schema:
a default time window, which of two similar measures the business means by a
term, how a ratio is normally expressed. If the answer depends on such a
convention, state the one you assumed in the answer, so a reader who uses a
different convention can see the difference immediately.

## Read a measure's expression, not just its name

Models frequently contain measures whose names do not describe what they
compute. Real examples seen in production models:

| name | what it actually computes |
| --- | --- |
| `sls_amt_x` | total sales **for two product categories only** |
| `gm2_pct` | margin **excluding one category** |
| `po_ok_flagish` | on-time POs divided by **high-value** POs, not all POs |
| `Day Yield Pct` | yield **excluding the night shift** |

Picking one of those for a general question returns a number that is plausible,
wrong, and very hard to notice. `list_measures` returns the DAX expression -
read it and confirm the measure computes what the question asked before using
it. When two measures could fit, prefer the one whose expression has no extra
filters.

## Prefer the model's own measures over recomputing

If a measure exists for what is being asked, use it. `[Total Sales]` already
encodes the right column, the right aggregation, and any business rules. A
hand-written `SUM(Sales[Amount])` may silently disagree with what the business
reports.

Recompute from columns only when no measure fits, and say so in the answer.

## Tool ranking

**1. `evaluate_measure`** - simple "measure by dimension, optionally filtered".
No DAX authoring, so no DAX syntax errors:

```python
df = fabric.evaluate_measure(
    DATASET, "Total Sales",
    groupby_columns=["Sales[region]"],
    filters={"Sales[Year]": ["2024"]},
)
```

**2. `evaluate_dax` with `SUMMARIZECOLUMNS`** - anything with real grouping,
several measures, TOPN, or time intelligence:

```python
df = fabric.evaluate_dax(DATASET, '''
EVALUATE
CALCULATETABLE(
    SUMMARIZECOLUMNS(
        Sales[region],
        "Total Sales", [Total Sales],
        "Total Profit", [Total Profit]
    ),
    Sales[Year] = 2024
)
''')
```

**3. `read_table`** - small dimension tables only, and always with `num_rows`.
Never for a fact table.

## DAX that the engine can optimise

- Filter with a direct column predicate, `Sales[Year] = 2024`, not
  `FILTER('Sales', 'Sales'[Year] = 2024)`. The predicate folds into a scan; the
  iterator does not.
- Apply filters **outside** with `CALCULATETABLE(SUMMARIZECOLUMNS(...), ...)`
  rather than inside `SUMMARIZECOLUMNS`.
- Group with `SUMMARIZECOLUMNS`, not `ADDCOLUMNS(SUMMARIZE(...))`.
- Iterate at the grain you need: `SUMX(VALUES(Table[Col]), ...)` rather than
  over the whole table.
- Cache a repeated measure reference in a `VAR` instead of evaluating it three
  times.
- Ask for the grain the question needs. Monthly is 12 rows; daily is 365.

## Anti-patterns

These produce right-looking answers slowly, or wrong answers confidently.

- **`read_table` on a fact table.** Materialises millions of rows into the
  notebook. If the answer needs aggregation, aggregate in DAX.
- **`EVALUATE 'Sales'`** with no filter or aggregation. Same problem.
- **Pulling detail rows to aggregate in pandas.** The storage engine is built
  for this and the data never has to move.
- **Using a measure because its name looked right.** Read the expression.
- **Answering "the sandbox blocks this".** See the first section.
- **Inventing a number when the model lacks the data.** Say what is missing.

## Result-size preflight

Before running a query you think might be large, count first:

```python
n = fabric.evaluate_dax(DATASET,
    'EVALUATE ROW("n", COUNTROWS(SUMMARIZECOLUMNS(Sales[region], Sales[Category])))')
print(n)
```

A grouped result should be tens or hundreds of rows. Thousands means the grain
is finer than the question needs.

## Before SUBMIT

- The number came from a query that ran, not from a guess or an estimate.
- If a measure was used, its expression matches what was asked.
- Grouped answers include every group returned, not just the top few, unless
  the question asked for a top-N.
- Percentages: state the form. `0.0314` and `3.14%` are the same number, and
  saying which one you mean prevents a correct answer being read as wrong.
- If the model genuinely lacks the data, the answer says what is missing rather
  than producing a figure.
