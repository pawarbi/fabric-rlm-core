# AgenticDataBench pilot

Adapter for running fabric-rlm against
[AgenticDataBench](https://github.com/AgenticDataBench/AgenticDataBench)
(arXiv 2607.01647): 246 public data-analysis tasks across 15 domains,
graded programmatically against gold files with no LLM judge.

## Layout

- `run_pilot.py` maps one benchmark task to one `RLM.task` call: each entry
  in the task's `data_sources` becomes a `File` input, and the model is
  instructed to write the required output file(s) into
  `outdir/<task_id>/`, the layout the benchmark's evaluator expects. It
  also writes the `dabench/result.json` marker without which the evaluator
  silently skips the task.
- `grade_pilot.py` grades with the benchmark's own metric functions
  (`da_agent.evaluators.metrics`). It exists because the stock
  `evaluate.py` breaks on Windows: it substitutes file paths into
  eval_func strings via `re.sub` with the raw path as the replacement
  template, and backslash sequences in Windows paths are parsed as regex
  escapes ("bad escape \s"). The scores themselves come from the
  benchmark's unmodified comparators.
- `claude_cli_lm.py` routes LM calls through a logged-in Claude Code CLI
  (`claude -p`), for machines with a CLI login but no API key.
- `stub_lm.py` replays scripted turns for plumbing tests only.

## Setup

```bash
git clone --depth 1 https://github.com/AgenticDataBench/AgenticDataBench.git
# On Windows, if checkout fails: git config core.longpaths true && git restore --source=HEAD :/
pip install jsonlines tqdm fuzzywuzzy python-Levenshtein
```

Download only the data files your task selection needs from
`https://huggingface.co/datasets/shawnzzzh/AgenticDataBench` into
`testbed/datasets/<domain>/`, preserving relative paths from each task's
`data_sources`. Sizes vary from under 1 MB to 368 MB per file; the full
dataset also carries a 1 GB embedding file no task needs.

## Run and grade

```bash
python run_pilot.py --testbed <...>/AgenticDataBench/testbed \
    --tasks agriculture_22,strategy_2,strategy_3 \
    --outdir <results dir> --lm openrouter/minimax/minimax-m3

python grade_pilot.py --testbed <...>/AgenticDataBench/testbed \
    --outdir <results dir> --tasks agriculture_22,strategy_2,strategy_3
```

## Results so far

MiniMax M3 via OpenRouter, `max_turns=12`, single seed, graded by
`grade_pilot.py` calling the benchmark's own `compare_csv`. Note that
`compare_csv` awards partial credit per matching column, so scores are
fractional rather than pass/fail.

    first 3 tasks (agriculture_22, strategy_2, strategy_3)   0.204
    10 held-out tasks, baseline                              0.551

Total spend across every run here was about $0.16, roughly half a cent per
task, at 23-80 seconds per task.

Raw-record adjudication of the first three (read the outputs against gold
before trusting the aggregate):

- `agriculture_22` grades per-class F1 of a cross-validated random forest
  with no seed given anywhere in the question. The run landed within about
  0.005 of every gold value and still scored 0.5. The task is not reliably
  passable by anyone and is worth reporting upstream.
- `strategy_2` was near perfect but swapped two columns: gold puts the bin
  *number* in `credit_bin` and the boundary *string* in `bin`. Awkward
  naming, but the statement settles it, so this is a model error.
- `strategy_3` counted all rows where the question's `total` meant
  outbound-call rows only (model error), and gold sorts descending while
  the statement never says so and grading uses `ignore_order=False`
  (benchmark under-specification).

A skill written against those failures was then tested on ten held-out
tasks and rejected; see `PREREGISTRATION.md` for the rule, the result and
the mechanism. Per-run grades are in `results/`.

## Task-audit notes (pilot scoping)

- 246 public tasks; 5 are single-CSV with `compare_csv` grading
  (`agriculture_22`, `strategy_2/3/5/7`), the smallest useful pilot set.
  `strategy_5`/`strategy_7` share a 368 MB input.
- 62 `financial_*` tasks carry no `data_sources` field; their questions
  name SQLite databases and companion markdown docs inline, so the adapter
  needs a resolver for those before the financial domain is runnable.
- `agriculture_22` grades per-class F1 of a cross-validated random forest
  with a 0.01 absolute tolerance and no pinned seed; whether that is
  reliably reproducible needs checking before trusting a fail there.
- `strategy_2` involves a Chinese-named label column and train-derived
  binning applied to a held-out split; per-column relative tolerances are
  specified in the task's eval_func.
