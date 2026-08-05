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

Summary: query a Power BI semantic model with semantic link. Read the metadata
first, aggregate in DAX, bring back small results.

Works inside a Fabric notebook, where `sempy` ships in the runtime.

## sempy is available. Do not conclude otherwise.

`import sempy.fabric` works in the execution sandbox, verified under the default
security policy. Bare `import fabric` is a different package and is blocked -
that block says nothing about sempy.

The most common failure here is not a wrong number, it is giving up. If one call
fails, that call failed. Print the exception and try a different query. Only say
a question is unanswerable when the **model lacks the data** - no such table,
column or measure. That is a statement about the model, never about the sandbox.

## First turn: read the model before querying it

Do this before writing any DAX. Names are not guessable and each guess costs a
turn.

```python
import sempy.fabric as fabric

DATASET = "<the semantic model name>"

print(fabric.list_tables(DATASET).to_string()[:2000])

m = fabric.list_measures(DATASET)
# Expression and Description too: names routinely misdescribe what a measure
# computes, and authors put the business meaning in Description.
cols = [c for c in ("Table Name", "Measure Name", "Measure Expression",
                    "Description") if c in m.columns]
print(m[cols].to_string()[:4000])

print(fabric.list_relationships(DATASET).to_string()[:2000])
```

## Querying

- `fabric.evaluate_measure(DATASET, "Total Sales", groupby_columns=[...],
  filters={...})` for "measure by dimension". No DAX to get wrong.
- `fabric.evaluate_dax(DATASET, "EVALUATE CALCULATETABLE(SUMMARIZECOLUMNS(...),
  <filters>)")` for grouping, several measures, TOPN or time intelligence.
- `fabric.read_table` only for small dimension tables, always with `num_rows`.
  Never a fact table - aggregate in DAX instead of pulling rows into pandas.

Prefer a measure that already exists over recomputing from columns, but read its
expression first and confirm it computes what was asked.

## Before SUBMIT

The number came from a query that ran. If a measure was used, its expression
matches the question. State whether a rate is `0.0314` or `3.14%`. If the model
genuinely lacks the data, say what is missing rather than producing a figure.
