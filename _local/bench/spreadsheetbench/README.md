# SpreadsheetBench Verified-400 — fabric_rlm head-to-head

A reproducible head-to-head benchmark of three orchestration strategies on the
[SpreadsheetBench Verified-400](https://huggingface.co/datasets/KAKA22/SpreadsheetBench)
dataset, run on Microsoft Fabric notebooks against `fabric_rlm`.

## Strategies

| Code | Orchestration | Parent LM | Sub LM | Tool access |
|------|---------------|-----------|--------|-------------|
| **A** | `dspy.Predict` single-shot, code → subprocess exec | gpt-5 (`reasoning_effort=medium`) | — | python (one shot) |
| **F** | `fabric_rlm.RLM(engine='v6-custom')` with `excel_modify` skill | gpt-4.1-mini | — | python interpreter, multi-turn (max 14) |
| **S** | `fabric_rlm.RLM(engine='v6-custom')` with `sub_lm` worker | gpt-5 (planner) | gpt-4.1-mini (per-turn worker) | python interpreter, multi-turn (max 14) |

The hypothesis behind **S**: split frontier-grade planning (cheap because the planner
fires few tokens) from per-turn worker code generation (expensive in token volume but
not in reasoning depth) — pay frontier price only where it matters.

## Reproducing

### Prerequisites

- A Fabric workspace with a `diagnostic` Lakehouse and a `sandeep_ws` (or rename in
  `scripts/build_ssb_notebook.py`) workspace.
- A python env with `azure-identity`, `azure-storage-file-datalake`, `requests`.
- Workspace-level env var or KeyVault binding for `OPENAI_API_KEY` (read by `FabricLM`).
- The `fabric_rlm` wheel uploaded once to
  `Files/fabric_rlm_longcot/wheels/fabric_rlm-0.2.1.dev2+excelskill-py3-none-any.whl`.

### One-shot

```powershell
# 1. Build the three notebooks (one strategy each).
python scripts/build_ssb_notebook.py --strategy A --model gpt-5         --effort medium                       --run-id ssb-full400-A-<DATE>
python scripts/build_ssb_notebook.py --strategy F --model gpt-4.1-mini  --effort medium                       --run-id ssb-full400-F-<DATE>
python scripts/build_ssb_notebook.py --strategy S --model gpt-5         --sub-lm gpt-4.1-mini --effort medium --run-id ssb-full400-S-<DATE>

# 2. Upload + start them in parallel via the Fabric Jobs API.
python files/launch_ssb_smoke.py     # launcher; JOBS list is the 3 notebooks above

# 3. Poll until all three jobs finish (typical wall-clock 1.5-3 hr in parallel).
python files/poll_ssb_smoke.py

# 4. Download per-strategy results from the Lakehouse.
python files/dl_ssb_smoke.py

# 5. Three-way analysis + report.
python files/analyze_ssb_3way.py \
  --results-root files/ssb_smoke_results \
  --run-a ssb-full400-A-<DATE> \
  --run-f ssb-full400-F-<DATE> \
  --run-s ssb-full400-S-<DATE> \
  --out-md bench/spreadsheetbench/REPORT_ssb_full400_3way.md
```

### Self-bootstrapping dataset

The notebook downloads the SpreadsheetBench Verified-400 release tarball from
HuggingFace on first run, repackages it into a flat
`spreadsheets/<id>/{init,golden}.xlsx` layout + jsonl manifest, and caches both to
`Files/fabric_rlm_longcot/datasets/`. Subsequent runs hit the OneLake cache and
skip the download. **No local upload step needed.**

Bootstrap source: <https://huggingface.co/datasets/KAKA22/SpreadsheetBench/resolve/main/spreadsheetbench_verified_400.tar.gz>

## Grading

For each question the notebook:

1. Copies `<id>_init.xlsx` to a working path.
2. Strategy A: asks the LM for a complete python program (one shot), runs it in a
   subprocess (`python -c <code>`, 180 s timeout).
   Strategy F/S: hands the workbook path + instruction to a fabric_rlm RLM with the
   `excel_modify` skill (which teaches the model the load → discover → modify →
   verify-by-reload pattern). Up to 14 turns.
3. Loads the produced workbook with `openpyxl(data_only=True)` and compares it
   cell-by-cell against the golden xlsx within `answer_position` on
   `answer_sheet`. **Pass = every cell matches exactly** (numbers compared with a
   small tolerance, strings stripped).

The grader does NOT evaluate Excel formulas: any output cell that holds a formula
will be read as `None`. The instruction in strategy A's `dspy.Signature` and the
`excel_modify` skill both warn the model to write computed values, not formulas.

## Outputs

Per strategy `<L>` in `Files/fabric_rlm_adaptive_validation/spreadsheetbench/<run_id>/`:

- `results_<L>.jsonl` — one record per question (qid, passed, cells_matched/total,
  prompt_tokens, completion_tokens, elapsed_seconds, n_turns, error).
- `summary_<L>.json`  — totals (n, n_passed, pass_rate, total_seconds, model, sub_lm).
- `traces_<L>/trace_<qid>.json` — full RLM trajectory or generated code per question.

## Files in this repo

| Path | Purpose |
|------|---------|
| `scripts/build_ssb_notebook.py` | Generates one Fabric notebook per strategy. Self-bootstrapping dataset. |
| `fabric_rlm/skills/excel_modify.md` | Skill prompt for the F/S strategies. |
| `bench/spreadsheetbench/REPORT_ssb_h2h_50q.md` | Earlier 50Q pilot report. |
| `bench/spreadsheetbench/REPORT_ssb_full400_3way.md` | Full 400Q 3-way report (this run). |
| `files/build_ssb_full400.py` | Local-only helper to repack the dataset (notebook now does this itself). |
| `files/launch_ssb_smoke.py`, `poll_ssb_smoke.py`, `dl_ssb_smoke.py` | Job lifecycle (despite "smoke" in the name). |
| `files/analyze_ssb_3way.py` | 3-way analyzer (pairwise intersections + cost). |

## Caveats

- The leaderboard at <https://spreadsheetbench.github.io> uses the original
  912-question OJ-style eval with **multiple perturbed test instances per
  question** (the "online judge" variant). The Verified-400 release we use here is
  a single-instance subset annotated by the authors — it's **easier per question**
  than the OJ score and not directly comparable rank-for-rank.
- Cost estimates use public per-1M-token pricing (gpt-5: $1.25 in / $10.00 out;
  gpt-4.1-mini: $0.40 in / $1.60 out). For S the parent and sub_lm tokens are
  reported under the parent's RLM trajectory; the figure is an upper bound.
- gpt-5 / o-series models accept `reasoning_effort`; gpt-4.1-mini does not.
  `build_ssb_notebook.py` only passes `reasoning_effort` when the model name
  starts with `gpt-5`/`o3`/`o1`.
