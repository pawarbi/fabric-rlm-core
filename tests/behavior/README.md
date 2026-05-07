# Behavior CI

A small, pinned set of LM evaluations that run on every PR via GitHub Actions
(see PR-2 — `.github/workflows/behavior-ci.yml`).  The job fails the build if
the model regresses against a committed baseline, catching the class of bug
that PR #12 missed (gpt-5.4-nano −20pp drop).

## What's here

| File | Purpose |
| --- | --- |
| `questions.py` | 5 pinned `Question` objects (2 compute, 2 messy, 1 self-correct) with locally computed ground truth — no binary fixtures. |
| `grader.py` | Pure comparator (`exact` / `near` / `string` / `set`) returning a `GradeResult`. |
| `runner.py` | Runs a `Question` via `fabric_rlm.RLM`, retries once on infra errors, classifies failures, exposes a `--calibrate` CLI. |
| `baseline_loader.py` | Loads & validates `baselines.json`; evaluates per-qid + aggregate gates. |
| `baselines.json` | Per-model calibration committed on `main` (created by the `--calibrate` CLI). |
| `test_grader.py`, `test_questions.py`, `test_baseline_loader.py`, `test_runner.py` | Offline unit tests; run on every CI build. |
| `test_behavior_baseline.py` | Online gate; runs only when `OPENROUTER_API_KEY` is present (and is required in GitHub Actions). |

## Two gates, both must pass

1. **Per-qid gate (primary).** Every qid that the baseline marks
   `expected_to_pass=True` must pass in the PR run.  This catches the case
   where one historically-stable question silently breaks.
2. **Aggregate floor (secondary).** `pr_passes >= baseline_passes - 1`.
   Belt-and-braces against multiple weak qids regressing together.

A qid that fails with an infra error (HTTP 429 / timeout / 5xx) is retried
once before being recorded as a wrong-answer fail.  Wrong answers are never
retried — that's the regression signal we want to detect.

## Local development

```powershell
# Offline tests (no API key needed) — runs on every CI build.
python -m pytest tests/behavior/ -q

# Run the full gate locally against the primary model.
$env:OPENROUTER_API_KEY = "sk-or-..."
python -m pytest tests/behavior/test_behavior_baseline.py -q -m primary
```

Without `OPENROUTER_API_KEY` the gate test **skips** locally.  Inside GitHub
Actions a missing key is a hard failure (so a misconfigured workflow cannot
silently pass).

## Recalibrating

You must recalibrate when:

* `questions.py` changes (the gate test will hard-fail with a sha-mismatch
  message until you do — the suite hash is stored in `baselines.json`).
* You want to change the primary or secondary model.
* The runtime / prompts change in a way that materially shifts pass rates.

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
$env:PYTHONPATH = "tests"
python -m behavior.runner --calibrate `
    --model openai/gpt-4.1-mini `
    --runs 5 `
    --out tests/behavior/baselines.json
```

Each candidate qid runs 5 times.  A qid is promoted to the blocking suite
(`expected_to_pass=True`) **iff its pass rate is 1.0** (i.e., 5/5).  A qid
that flaked even once is recorded but excluded from both gates -- a single PR
run is n=1, so any non-zero per-call failure rate translates directly into
noisy merge blockers.  The aggregate `min_passes` is set to
`baseline_passes - 1`.

If you want the looser 4/5 promotion threshold, pass
`--pass-rate-threshold 0.8` (not exposed via CLI yet -- call `calibrate(...)`
from Python directly, or accept that calibration spend goes up).

Re-running calibration for a different model **merges** into the existing
file (preserves other models), but refuses to merge if suite metadata
(`questions_sha256`, `max_turns`, `timeout_s`, `calibration_runs_per_qid`,
`suite_version`) disagrees.  Pass `--replace` to overwrite from scratch, or
`--force-merge` to override the guard (not recommended -- creates mixed-suite
baselines).

## Adding a question

1. Add the new `Question(...)` to `QUESTIONS` in `questions.py`.  Make sure
   the `expected` is computed locally (no external lookups, deterministic).
2. Recalibrate (see above).  Commit both `questions.py` and the regenerated
   `baselines.json` in the same PR.
3. The questions sha256 in `baselines.json` will update; subsequent PRs run
   against the new baseline.

## Fork PR caveat

PR runs from forks do not have access to repository secrets; the
`behavior-ci` workflow detects this and emits a warning step instead of
running the gate.  A maintainer must rebase the change onto an internal
branch (or push to a maintainer-controlled fork) before merge if the PR
touches `fabric_rlm/**` or `tests/behavior/**`.

## Cost

With `max_turns=8` and `timeout=120s` per question, a full primary-model PR
run costs roughly $0.01.  The free secondary run costs $0 but is rate-limited
and informational only.

## Escape hatch

If the primary job fails for non-regression reasons (provider outage, OpenRouter
routing issue), maintainers can re-run the workflow from the PR's "Checks"
tab.  Do **not** disable or skip the gate — the whole point is that
regressions block merge.
