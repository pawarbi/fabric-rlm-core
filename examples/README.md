# Examples

What a user runs. Nothing here is needed by the package itself. The
benchmark harnesses live under `benchmarks/` and the test fixtures under
`tests/`.

## Fabric notebooks (`notebooks/`)

Ready to import into a Fabric workspace. Each installs the released
package at a pinned version and never carries a key in its source; the
test suite checks both.

Start here:

- `rlm_api_tour.ipynb`: start with small generated sales data, calculate
  regional revenue from a CSV, and check typed outputs and validators.
  Also covers construction styles, artifacts, skills, the security policy
  and timeout recovery. No uploads are needed.
- `rlm_spark_log_root_cause.ipynb`: generate a small Spark log, identify
  the failed job, stage and executor, and check the supporting log lines.
- `rlm_vs_plain_llm_imf_cpi.ipynb`: then try a larger task two ways. Build
  an Excel report from 140 MB of IMF consumer price data with fabric-rlm
  and with a direct model call, and compare the two.

Documents:

- `rlm_pdf_contract_comparison.ipynb`: download two sample contracts,
  compare them, display the payload. `rlm_minimal_contract_comparison.ipynb`
  is the shortest form of the same recipe.
- `rlm_pdf_document_analysis.ipynb`: analyse a sample RFP.
- `rlm_pdf_document_redaction.ipynb`: find the PII redaction targets in an
  employment agreement.
- `rlm_pdf_invoice_processing.ipynb`: extract structured data from two
  invoices.

Semantic models:

- `rlm_verified_deep_insights.ipynb`: investigate scrap-rate changes using
  governed semantic-model measures and the deep-discovery skill. Keep the
  structured evidence and display a readable Markdown report. Skill checks
  cover structure and some arithmetic, not automatic source or prose verification.
- `rlm_learn_semantic_model_value.py`: see the value of `RLM.learn()` on a
  Fabric semantic model, cold against learned.

Choose by task and prerequisites (run each notebook's pinned install first):

| Group | Task | Prerequisites |
|---|---|---|
| API tour and Spark log | Learn the API on generated CSV/log fixtures | Fabric notebook, writable default Lakehouse, Fabric AI model access; no uploads |
| IMF comparison | Build and compare large-data Excel reports | Writable default Lakehouse, Fabric AI model access, outbound access to the IMF API |
| PDF recipes | Compare, analyse, find redaction targets, or extract documents | Fabric AI model access and access to the sample download URLs; see each notebook's setup |
| Semantic-model learning | Compare cold and learned answers | Workspace/model IDs, a scalar measure and Fabric AI model access; no Lakehouse attachment required |
| Deep insights | Investigate a semantic model and render findings as Markdown | Workspace/model IDs, governed scrap/production measures and plant/category dimensions, Fabric AI model access; no Lakehouse attachment required |

Some notebooks ship in two forms. The `.py` file is the notebook in
Fabric's own source format (it starts with `# Fabric notebook source` and
marks cells with `# CELL` and `# MARKDOWN`): it imports through Git
integration and diffs in a pull request. The `.ipynb` is the same notebook
for the Fabric UI import and for GitHub rendering. When both exist, the
`.py` is the source and the `.ipynb` is generated from it.

## Confirm in Fabric

Local tests validate notebook syntax, pins, generated fixtures, source wiring,
saved-workbook checks, and report rendering. They do not establish live Fabric
authentication or model availability. After importing, run the install cell,
restart the session, then run the remaining cells in order.

1. **API tour:** attach a writable Lakehouse. Confirm the generated CSV totals
   and notebook assertions pass, and inspect one successful run.
2. **PDF recipes:** confirm downloads and assertions succeed. Inspect extracted
   values against the source; redaction discovery does not redact a file.
3. **IMF workbook:** allow the download and model calls to finish. Confirm every
   grading assertion passes and open the saved workbook in Excel. This is the
   larger evaluation, not the shortest getting-started notebook.
4. **Semantic-model examples:** configure the documented schema and IDs. Confirm
   source measure values independently; these examples do not create the model.

`/tmp` output is session-local. Use an existing Lakehouse Files directory when
you want durable artifacts. Surface errors rather than skipping failed cells;
report the notebook name, package version and redacted traceback when a check fails.

## `simple_math/`

The quickstart. `run_mock.py` runs the smallest possible RLM with a mock
model, and `task.json` is the task file for
`fabric-rlm run examples/simple_math/task.json`.

## `semantic_model/`

A semantic model, a PDF, a CSV and a custom skill combined into a
formatted Excel workbook written back to the lakehouse. Its README says
what you need and how each source contributes facts the others cannot.
