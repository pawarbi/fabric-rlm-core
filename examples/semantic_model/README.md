# Semantic model as a data source

A Power BI semantic model is a data source like any other. Inside a Fabric
notebook, `sempy` is already in the runtime, so an RLM can read a model's
schema, run DAX against it, and combine the result with files sitting in a
lakehouse.

This directory holds the example that exercises that end to end: a semantic
model, a PDF, a CSV and a custom skill, combined into a formatted Excel
workbook written back to the lakehouse.

## What you need

- A Fabric workspace with a default lakehouse attached to the report notebook
  and an existing semantic model matching the contract below. The notebook
  identity needs read/Build access to that model and write access to the lakehouse.
- A Python 3.12 notebook runtime.
- Fabric AI services enabled for your tenant/capacity, with access to `gpt-5.1`.
  The recommended default is `FabricLM("gpt-5.1")`, which uses Fabric's built-in
  endpoint and notebook authentication. No OpenRouter key, plaintext credential
  file, or separate Azure OpenAI resource is needed. Live LM calls consume capacity.

Notebook generation and setup checks run offline in
`tests/test_semantic_example_setup.py`; live queries and LM runs need Fabric.

## Running it

From this directory, install the local generator dependency and build:

```bash
python -m pip install nbformat
python build_report_nb.py
fab import /YourWorkspace/multisource_report.Notebook -i multisource_report.Notebook -f
```

Alternatively, upload the generated `multisource_report.Notebook/notebook-content.ipynb`
through Fabric's notebook import UI. Attach your lakehouse as the **default**
lakehouse, then edit the single configuration/parameter cell near the top:
`WORKSPACE_ID` (the model's workspace), `MODEL_NAME`, and optionally `LM_MODEL`,
`ARM`, `MAX_TURNS`, and `TIMEOUT_S`. Run all cells. Both ground-truth queries and
the RLM model handle use that workspace explicitly. The install cell pins
`fabric-rlm[analytics]==0.6.4`.

The notebook generates `targets.csv`, `ops_memo.pdf`, and the context skill under
`/lakehouse/default/Files/multisource`. It writes the workbook to
`reports/ops_review.xlsx` there, and grading results to
`/lakehouse/default/Files/multisource_<ARM>.json`. These files are overwritten on
reruns. It does **not** generate, upload, or provision the semantic model.

## Required model contract

This is a manufacturing example, not a schema-independent report generator.
The supplied skill and the ground-truth DAX require:

| Object | Required meaning/use |
| --- | --- |
| `'Date'[Date]` | Date column; its maximum ends the inclusive trailing 30-day window. Relationships must propagate this filter to the KPI facts. |
| `[Total Sales]` | Currency sales total across all product categories. |
| `[Production Yield %]` | All-shift yield as a fraction, not a 0-100 percentage. |
| `[Downtime %]` | Downtime as a fraction. |
| `ProductionLog[Scrap]`, `ProductionLog[Qty]` | Numeric quantities; scrap rate is `DIVIDE(SUM(ProductionLog[Scrap]), SUM(ProductionLog[Qty]))`. |
| Plant grouping column | `list_columns` must expose a column whose name contains `plant` (case-insensitive). The grader selects the first match; it must group/filter production correctly. |

Totals and per-plant rates must be non-blank numeric values for that window.
For the original skill-ablation experiment, the model also exposed misleading
alternatives: `sls_amt_x` (Turbines and Pumps only) and `Day Yield Pct` (day shift
only). They are not required by the ground-truth queries, but without them the
experiment no longer tests those specific measure-selection mistakes.

For a model with different names or business definitions, adapt `CELL_TRUTH`
and `skills/report_context.md` together, including the date window and plant
grouping. Review the synthetic targets and memo for your scenario too. Changing
only `MODEL_NAME` does not make an arbitrary model compatible.

## RLM.learn comparison

The companion [`rlm_learn_semantic_model_value.py`](../notebooks/rlm_learn_semantic_model_value.py)
uses an existing business semantic model and needs no lakehouse attachment.
It pins `fabric-rlm==0.6.4` and also uses Fabric's built-in LM. Set the workspace
and model ID placeholders in its configuration cell. Map `MEASURE` to your
existing numeric scalar ARR measure (`ARR $` is only an example name), and keep
`QUESTION` aligned with its business meaning. No particular tables or columns
are queried directly; the measure and all its dependencies must already exist
and return one non-blank total. This source does not create a model or an ARR
schema. LM settings, turn limit, and knowledge-store path are in the same cell.
The saved knowledge package is overwritten when learning runs again.

## What the example does

The agent is asked for an operations review workbook with three sheets. Four
sources each carry facts none of the others can supply:

| source | supplies |
| --- | --- |
| semantic model | the actual KPI values |
| `targets.csv` | the target and owner for each KPI |
| `ops_memo.pdf` | three narrative risks, and the escalation threshold |
| `report_context` skill | which measure answers which KPI, the reporting window, which direction counts as good, the formatting conventions |

That split is the point. A miss is attributable: skip the CSV and the Target
column is empty, skip the PDF and the escalation column is wrong, skip the
skill and the agent picks `sls_amt_x` (which covers two product categories) or
`Day Yield Pct` (which drops the night shift), both of which return a number
that looks fine.

Grading is post-hoc, against ground truth computed in the same notebook with
sempy and pandas, plus openpyxl checks on the formatting. `output_validator`
enforces only that a readable workbook with the three named sheets exists. It
deliberately checks nothing about values or formatting, so it cannot manufacture
the result being measured.

## What it measured

Historical run on the original model with MiniMax M3, 44 checks (not a result
claim for the current FabricLM default or your model):

| arm | score | turns |
| --- | --- | --- |
| all four sources | 44/44 | 15 |
| same, minus the custom skill | 34/44 | 30 |

Without the skill the agent reported Q2 figures instead of the trailing 30 days
the house convention calls for. That reading is defensible, since the memo is
titled "Q2 Operations Review", but it makes every actual wrong, and the error
does not stay contained: the wrong downtime figure produces a 0.2571 point
variance, which clears the memo's 0.20 point escalation threshold, so a KPI is
escalated that should not have been. One wrong window moved a business decision
four columns downstream.

## A note on tolerances

The first version of the grader compared values at 1 percent relative
tolerance. Rates in this model sit near 0.98, and the wrong reporting window
moves them by about 0.1 percent, so a Production Yield of 0.976613 passed when
the right answer was 0.977740. It was a false pass on the exact error the test
exists to catch.

The tolerance is now 0.2 percent. If you adapt this for another model, check
that your tolerance is tighter than the difference between a right answer and
the most plausible wrong one, rather than picking a round number.

## Files

- `build_report_nb.py` builds the notebook and its local fixtures. The existing
  semantic model, lakehouse attachment, permissions, and Fabric AI access are
  prerequisites, not generated artifacts.
- `skills/report_context.md` is the custom skill. It is a reasonable template
  for writing your own: a glossary, the conventions that cannot be derived from
  the schema, and the measures whose names do not describe what they compute.

## On writing your own context skill

Two things worth knowing, both measured rather than assumed.

A skill reliably supplies **definitions**. House terms, which measure to use for
which KPI, what a coded column means: the model hits an unfamiliar term, has no
prior, and goes looking. This works well.

A skill supplies **ambient defaults** less reliably. A rule like "if no period
is named, use the trailing 30 days" has no trigger. Nothing in "what were total
sales" makes a model ask which window it should use, so its own prior wins. In
a separate four question test, the glossary entries landed and that default did
not.

The workaround is in this example: the request asks the agent to state the
reporting window in the workbook. That single instruction gives the convention
something to attach to, and the default then gets applied. If a rule has to hold
for every question regardless of wording, put its trigger in the task, not only
in the skill.
