# Examples

What a user runs. Nothing here is needed by the package itself. The
benchmark harnesses live under `benchmarks/` and the test fixtures under
`tests/`.

## Fabric notebooks (`notebooks/`)

Ready to import into a Fabric workspace. Each installs the released
package at a pinned version and never carries a key in its source; the
test suite checks both.

Start here:

- `rlm_vs_plain_llm_imf_cpi.ipynb`: one task, tried two ways. Build an
  Excel report from 140 MB of IMF consumer price data with fabric-rlm and
  with a direct model call, and compare the two.
- `rlm_api_tour.ipynb`: a practical map of the stable public API: the two
  construction styles, typed outputs, artifacts, files, skills, validators,
  the security policy and timeout recovery.

Documents:

- `rlm_pdf_contract_comparison.ipynb`: download two sample contracts,
  compare them, display the payload. `rlm_minimal_contract_comparison.ipynb`
  is the shortest form of the same recipe.
- `rlm_pdf_document_analysis.ipynb`: analyse a sample RFP.
- `rlm_pdf_document_redaction.ipynb`: find the PII redaction targets in an
  employment agreement.
- `rlm_pdf_invoice_processing.ipynb`: extract structured data from two
  invoices.

Logs and data:

- `rlm_spark_log_root_cause.ipynb`: create a small Spark log and ask for
  the root cause.
- `rlm_what_moved.ipynb`: driver analysis over a lakehouse or a semantic
  model. Ask for a report in plain words and get a dashboard whose numbers
  come from the source's own engine.
- `rlm_data_agent_review.ipynb`: point it at a Fabric Data Agent and review
  what the agent uses and how it is instructed against the sources
  themselves.
- `rlm_learn_semantic_model_value.py`: see the value of `RLM.learn()` on a
  Fabric semantic model, cold against learned.

Some notebooks ship in two forms. The `.py` file is the notebook in
Fabric's own source format (it starts with `# Fabric notebook source` and
marks cells with `# CELL` and `# MARKDOWN`): it imports through Git
integration and diffs in a pull request. The `.ipynb` is the same notebook
for the Fabric UI import and for GitHub rendering. When both exist, the
`.py` is the source and the `.ipynb` is generated from it.

## `simple_math/`

The quickstart. `run_mock.py` runs the smallest possible RLM with a mock
model, and `task.json` is the task file for
`fabric-rlm run examples/simple_math/task.json`.

## `semantic_model/`

A semantic model, a PDF, a CSV and a custom skill combined into a
formatted Excel workbook written back to the lakehouse. Its README says
what you need and how each source contributes facts the others cannot.
