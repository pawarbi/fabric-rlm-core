# GLM-5.3-flash workbook arm (`glm-workbook-1`) — raw results

Run: 24 questions, cold, workbook duty ON, `z-ai/glm-5.3-flash` via OpenRouter,
Fabric python 3.12.12 / dspy 3.2.1 (in-pin), wheel install, isolated RUN_TAG paths.

## Headline

| Metric | gpt-4.1-mini (`cold-full-1`) | **GLM-5.3-flash (`glm-workbook-1`)** |
|---|---|---|
| Analytical accuracy (run log) | 8/24 = 33.3% | **22/24 = 91.7%** |
| Workbook accuracy (final .xlsx) | ungradable (15 rows destroyed) | **23/24 = 95.8%** |
| Rows lost to workbook reset | 15/24 | **0/24** |
| No machine-readable answer | 10/24 | **0/24** |
| Fan-out hazard failures | — | **0** |
| Mean turns | 13.88 | 7.88 |
| Mean prompt tokens | 54,145 | 40,314 |
| Final workbook bytes | 7,157 (collapsed from 11,083) | **23,144 (monotone)** |

Byte series was strictly monotone across all 24 questions: 7,488 → 23,144.
RLM's self-reported `workbook_rows_total` advanced 1 → 24 without going
backwards.

## The two run-log "failures" were grading artifacts, not RLM errors

q16 ("Which company_size segment has the highest MEAN subscription mrr?") and
q19 ("Which payment method has the highest MEAN payment amount?") are
categorical questions. RLM put the *metric* in `answer.value` and the
*category label* in `answer.reasoning`:

- q16 `value=3642.94`, reasoning: "enterprise=3642.94 (n=92) ... Enterprise has
  the highest mean MRR" — **correct**
- q19 `value=1193.8894`, reasoning: "ach=1193.8894 (n=2,237) > bank_transfer
  ... " — **correct**

In the **workbook**, RLM wrote `enterprise` for q16 (PASS) and the number for
q19 (FAIL). So the substantive analysis was right in 24/24; only the slot the
answer landed in varied.

**Cause: harness defect H5, not an RLM defect.** My `outputs={"answer": dict}`
contract names a single `value` field and never says what `value` must hold for
a categorical question. Recorded as a harness defect; the strict machine-readable
scores above are reported unchanged and NOT silently upgraded.

## Consequences for earlier findings

- **F9 (workbook reset) does not reproduce with GLM.** Same library, same
  prompt, same environment, same 24 questions — 0 resets vs 15 rows destroyed.
  F9 is **model-dependent behaviour**, not a deterministic library defect.
- **F11's causal role is now unsupported.** The run log stores answer payloads
  only, not trajectories: the "gate error" evidence is 1 self-report in the
  gpt-4.1-mini log and 1 in the control log — not a hit count. GLM's log
  contains **0** and GLM scored 91.7% in the same environment, so the
  environment imposes no 33% ceiling. F11 remains a **real, credential-free
  reproducible defect** (`_normalize_catalog_query`), but its role in the
  gpt-4.1-mini collapse is **not established**.

## Reproduce

```powershell
cd <session>\files\fabric_upload
python notebook.py execute --workspace-id 82ad2591-974a-4ad4-ace6-e24879274a4b `
  --notebook-id 2935d52f-4574-443d-a8d0-552936a7445a `
  --parameter "RUN_TAG=glm-workbook-1" --parameter "MODEL=z-ai/glm-5.3-flash" `
  --parameter "WORKBOOK_DUTY=True" --parameter "MAX_TURNS=22"

python read_files.py Files/rlm_excel_eval/run_log-glm-workbook-1.json --out run_log_glm.json
python grade_analytical.py run_log_glm.json analytical_grade_glm.json
python grade_rlm_workbook.py rlm_authored_glm.xlsx
```

Artifacts: `run_log_glm.json`, `analytical_grade_glm.json`,
`rlm_authored_glm.xlsx`, `rlm_authored_grade.json`,
lakehouse `Files/rlm_excel_eval/out/glm-workbook-1/rlm_answers.xlsx`.
