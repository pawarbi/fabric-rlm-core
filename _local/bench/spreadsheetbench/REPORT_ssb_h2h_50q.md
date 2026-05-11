# SpreadsheetBench Verified-400 head-to-head (50Q subset)

Date: 2026-05-05
Wheel: `fabric_rlm-0.2.1.dev2+excelskill`
Subset: 50 cell-level questions sampled from SpreadsheetBench v1
(`ssb_subset_50.jsonl`, 1.3 MB workbook bundle in OneLake)

## Headlines

| Strategy | Pass | Pass% | Cells matched | Prompt tok | Completion tok | Wall-clock | Approx cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A** — gpt-5 single-shot (`dspy.Predict` + subprocess exec) | 23/50 | **46.0%** | 818/1085 (75.4%) | 25,660 | 217,959 | 34 min | **$2.21** |
| **F** — gpt-4.1-mini + RLM + Python interpreter + `excel_modify` skill | 21/50 | **42.0%** | 720/1085 (66.4%) | 1,029,008 | 60,623 | 15 min | **$0.51** |

- **F is within 4 points of A at 4.3× lower $ cost and 2.3× lower wall-clock.**
- Each strategy uniquely solves questions the other misses:
  - both pass: 15
  - only A: 8
  - only F: 6
  - neither: 21
  - **union (any-strategy): 29/50 = 58%**

## Comparison to public leaderboard (full 260-question Verified-400)

| Tool | Reported | Our 50Q subset |
|---|---:|---:|
| ChatGPT Agent (top of leaderboard) | 70.48% | — |
| Claude 3.7 Sonnet | ~57% | — |
| GPT-4o | ~50% | — |
| **A: gpt-5 single-shot (our run)** | — | 46.0% |
| **F: gpt-4.1-mini + RLM (our run)** | — | 42.0% |

Caveats: our 50Q subset is a stratified sample, not the full 260 Verified-400; the published leaderboard uses end-to-end agents with different scaffolding. The headline finding is the **A↔F delta and the cost ratio**, not absolute placement on the leaderboard.

## Why F doesn't beat A outright

1. **gpt-4.1-mini caps out** on multi-step Excel reasoning (e.g. SSB_55976: 1/4 cells; the model wrote correct logic but to the wrong cells). Even with the interpreter, raw reasoning quality matters.
2. **No bandit / no escalation.** gpt-4.1-mini does not accept `reasoning_effort`, so `EffortBanditPolicy` could not be used. F runs at fixed effort with `max_turns=14`.
3. **Single attempt per question.** Adding a second self-critique pass would likely close the 4-point gap but at the cost of doubling tokens.

## Why F still wins at "useful agent" criterion

- 6 of A's 27 failures are solved by F. These are mostly questions requiring per-cell verification (formulas in source workbook, multi-table sheets) where the interpreter reload step is decisive.
- For workloads where you'd run Excel manipulation **at scale**, the cost ratio matters more than the 4-point accuracy gap.

## Skill: `fabric_rlm/skills/excel_modify.md`

New skill shipped in `0.2.1.dev2+excelskill`. Triggers on `xlsx`/`workbook`/`openpyxl`/etc. keywords. Two key recipes baked in:

1. **Two-load discovery**: load the workbook twice (`data_only=False` for editing, `data_only=True` for reading source values), so cells containing formulas in the input return numbers, not the literal `'=D3+F3'` string. This single change fixed SSB_36097 (0/4 → 4/4 in smoke).
2. **Mandatory verify-by-reload**: after `wb.save(path)`, reload with `data_only=True` and assert no cell in TARGET RANGE is `None` or starts with `=`. Catches the formula-instead-of-value class of failure.

## Files

- Per-question results: `files/ssb_smoke_results/ssb-h2h-{A,F}-20260505/results_{A,F}.jsonl`
- Per-question traces: `files/ssb_smoke_results/ssb-h2h-{A,F}-20260505/traces_{A,F}/trace_<qid>.json`
- Subset metadata: `files/ssb_subset_50.jsonl`
- Workbook bundle (OneLake): `Files/fabric_rlm_longcot/datasets/ssb_subset_50.tar.gz`
- Notebooks: `notebooks/ssb_A_gpt_5.ipynb`, `notebooks/ssb_F_gpt_4_1_mini.ipynb`
- Skill: `fabric_rlm/skills/excel_modify.md`
