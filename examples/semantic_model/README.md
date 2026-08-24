# Semantic model as a data source

A Power BI semantic model is a data source like any other. Inside a Fabric
notebook, `sempy` is already in the runtime, so an RLM can read a model's
schema, run DAX against it, and combine the result with files sitting in a
lakehouse.

This directory holds the example that exercises that end to end: a semantic
model, a PDF, a CSV and a custom skill, combined into a formatted Excel
workbook written back to the lakehouse.

## What you need

- A Fabric workspace with a lakehouse and a semantic model.
- A Python 3.12 notebook runtime.
- An LM. The example reads an OpenRouter key from a file in the lakehouse;
  point it at whatever you use.

None of this runs in CI, because all of it needs a live workspace.

## Running it

Edit `WORKSPACE_ID`, `LAKEHOUSE_ID` and `LAKEHOUSE_NAME` at the top of
`build_report_nb.py`, then:

```bash
python build_report_nb.py
fab import /YourWorkspace/multisource_report.Notebook -i multisource_report.Notebook -f
fab job run /YourWorkspace/multisource_report.Notebook -P ARM:string=full
```

`MODEL_NAME` in the parameter cell names the semantic model. The notebook
builds its own fixtures, so there is nothing to upload.

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

MiniMax M3, 44 checks:

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

- `build_report_nb.py` builds the notebook. Everything, including the fixtures,
  is generated, so the example is self contained.
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
