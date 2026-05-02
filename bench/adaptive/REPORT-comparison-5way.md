# 5-Way Comparison Report — `full-20260502-110000`

**Dataset:** 25 LongCoT-hard CS questions (5 per template: MFMC, Backprop, DistMem, MCM, VLIW), held out from the pilot20 set.

**Model:** azure/gpt-5 with `reasoning_effort='minimal'` as base. PVR mode varies per condition.

**Wheel:** `fabric_rlm-0.1.11.dev5-py3-none-any.whl`.


## Strategies

- **A `direct`** — Direct LLM (gpt-5 minimal)
- **B `dspy_rlm`** — Fabric RLM + v7-dspy engine
- **C `fabric_full`** — Fabric RLM + v6-custom (PVR full)
- **D `fabric_reflect`** — Fabric RLM + v6-custom (PVR reflect_only)
- **E `fabric_ladder`** — Fabric RLM + v6-custom + EffortLadder (minimal->low->medium)

> Note: "DSPy RLM" in the user request was interpreted as fabric_rlm with `inner_engine='v7-dspy'` because DSPy itself ships no `rlm` module. Strategy E originally targeted `EffortBanditPolicy`, but the bandit hung repeatedly during smoke testing on first-question warmup; we substituted the deterministic `EffortLadderPolicy` (minimal→low→medium) which exercises the same adaptive escalation code path with predictable behavior.


## Run health check

Headline pass rate is meaningless if a strategy didn't actually execute. This table flags strategies that didn't run cleanly.

| Strategy | n_attempts>0 | produced_answer | n_passed | Status |
|---|---|---|---|---|
| A direct | 25 | 25 | 0 | ran (25/25 produced an answer, 0 errored) |
| B dspy_rlm | 25 | 25 | 0 | **DEGRADED** — v7-dspy inner interpreter (Pyodide/REPL) failed to start in Fabric runtime; model produced text-only answers |
| C fabric_full | 25 | 23 | 0 | ran (23/25 produced an answer, 0 errored) |
| D fabric_reflect | 25 | 24 | 0 | ran (24/25 produced an answer, 0 errored) |
| E fabric_ladder | 0 | 0 | 0 | **BROKEN** — adaptive engine returned 0 attempts on every question (policy/validator config issue) |

## Headline pass rates

| Strategy | Passed | Rate | Per template (passed/total) |
|---|---|---|---|
| A direct | 0/25 | 0% | Backprop:0/5, DistMem:0/5, MCM:0/5, MFMC:0/5, VLIW:0/5 |
| B dspy_rlm | 0/25 | 0% | Backprop:0/5, DistMem:0/5, MCM:0/5, MFMC:0/5, VLIW:0/5 |
| C fabric_full | 0/25 | 0% | Backprop:0/5, DistMem:0/5, MCM:0/5, MFMC:0/5, VLIW:0/5 |
| D fabric_reflect | 0/25 | 0% | Backprop:0/5, DistMem:0/5, MCM:0/5, MFMC:0/5, VLIW:0/5 |
| E fabric_ladder | 0/25 | 0% | Backprop:0/5, DistMem:0/5, MCM:0/5, MFMC:0/5, VLIW:0/5 |

## Cost (elapsed + tokens) over all 25 questions

| Strategy | Total elapsed | Mean/q | Median/q | Prompt tok | Completion tok | Total tok | Total attempts |
|---|---|---|---|---|---|---|---|
| A direct | 34s | 1.3s | 1.3s | 136754 | 2136 | 138890 | 25 |
| B dspy_rlm | 1563s | 62.5s | 60.1s | 0 | 0 | 0 | 25 |
| C fabric_full | 519s | 20.8s | 15.3s | 376844 | 70121 | 446965 | 25 |
| D fabric_reflect | 768s | 30.7s | 15.3s | 320678 | 60968 | 381646 | 25 |
| E fabric_ladder | 253s | 10.1s | 10.1s | 0 | 0 | 0 | 25 |

## Per-question pass matrix

| question_id | A | B | C | D | E |
|---|---|---|---|---|---|
| Backprop_hard_10 | ✗ | ✗ | ✗ | ✗ | ✗ |
| Backprop_hard_11 | ✗ | ✗ | ✗ | ✗ | ✗ |
| Backprop_hard_12 | ✗ | ✗ | ✗ | ✗ | ✗ |
| Backprop_hard_13 | ✗ | ✗ | ✗ | ✗ | ✗ |
| Backprop_hard_14 | ✗ | ✗ | ✗ | ✗ | ✗ |
| DistMem_hard_10 | ✗ | ✗ | ✗ | ✗ | ✗ |
| DistMem_hard_11 | ✗ | ✗ | ✗ | ✗ | ✗ |
| DistMem_hard_12 | ✗ | ✗ | ✗ | ✗ | ✗ |
| DistMem_hard_13 | ✗ | ✗ | ✗ | ✗ | ✗ |
| DistMem_hard_14 | ✗ | ✗ | ✗ | ✗ | ✗ |
| MCM_hard_10 | ✗ | ✗ | ✗ | ✗ | ✗ |
| MCM_hard_11 | ✗ | ✗ | ✗ | ✗ | ✗ |
| MCM_hard_12 | ✗ | ✗ | ✗ | ✗ | ✗ |
| MCM_hard_13 | ✗ | ✗ | ✗ | ✗ | ✗ |
| MCM_hard_14 | ✗ | ✗ | ✗ | ✗ | ✗ |
| MFMC_hard_10 | ✗ | ✗ | ✗ | ✗ | ✗ |
| MFMC_hard_11 | ✗ | ✗ | ✗ | ✗ | ✗ |
| MFMC_hard_12 | ✗ | ✗ | ✗ | ✗ | ✗ |
| MFMC_hard_13 | ✗ | ✗ | ✗ | ✗ | ✗ |
| MFMC_hard_14 | ✗ | ✗ | ✗ | ✗ | ✗ |
| VLIW_hard_10 | ✗ | ✗ | ✗ | ✗ | ✗ |
| VLIW_hard_11 | ✗ | ✗ | ✗ | ✗ | ✗ |
| VLIW_hard_12 | ✗ | ✗ | ✗ | ✗ | ✗ |
| VLIW_hard_13 | ✗ | ✗ | ✗ | ✗ | ✗ |
| VLIW_hard_14 | ✗ | ✗ | ✗ | ✗ | ✗ |

## Pairwise diff — who uniquely solves what

| pair | lhs-only | rhs-only | both | lhs-only qids | rhs-only qids |
|---|---|---|---|---|---|
| direct vs dspy_rlm | 0 | 0 | 0 | — | — |
| direct vs fabric_full | 0 | 0 | 0 | — | — |
| direct vs fabric_reflect | 0 | 0 | 0 | — | — |
| direct vs fabric_ladder | 0 | 0 | 0 | — | — |
| dspy_rlm vs fabric_full | 0 | 0 | 0 | — | — |
| dspy_rlm vs fabric_reflect | 0 | 0 | 0 | — | — |
| dspy_rlm vs fabric_ladder | 0 | 0 | 0 | — | — |
| fabric_full vs fabric_reflect | 0 | 0 | 0 | — | — |
| fabric_full vs fabric_ladder | 0 | 0 | 0 | — | — |
| fabric_reflect vs fabric_ladder | 0 | 0 | 0 | — | — |

## Findings

1. **`gpt-5` at `reasoning_effort='minimal'` cannot solve these hard CS puzzles** — every working strategy passed 0/25. Direct calls produced confident-sounding refusals ("I cannot reliably execute …") or empty solutions. Multi-turn RLM scaffolding (full PVR, reflect_only) reduced the rate of refusals but not the rate of correct solutions.

2. **`fabric_full` and `fabric_reflect` mostly hallucinate input truncation.** The model frequently insists the puzzle instance is "truncated" even though the full prompt (~10.7k chars) is forwarded verbatim by the inner engine. This is a model behavior at minimal effort, not a prompt-truncation bug.

3. **`dspy_rlm` (v7-dspy) is unusable on Fabric runtime today.** Every question received an answer of the form "I was unable to execute any REPL steps … persistent environment startup error" — the v7-dspy inner interpreter (Pyodide/PythonInterpreter) cannot start under the Fabric Spark notebook runtime. Treat B as "engine not exercised" rather than as a fair competitor.

4. **`fabric_ladder` (E) returned 0 attempts on every question** — the adaptive engine accepted the `EffortLadderPolicy` we passed but never ran a rung. Likely a misconfiguration of the `RLM(adaptive=...)` kwargs for a non-bandit ladder. (`EffortBanditPolicy` had hung during smoke testing, which is why we substituted the ladder.) E should be treated as a "how to wire the adaptive engine without bandit" follow-up, separate from this comparison.

5. **Effective comparison is A vs C vs D.** Among those, `direct` is far cheaper (138K tokens, 34s) than `fabric_full` (447K tokens, 8.6 min) or `fabric_reflect` (382K tokens, 12.8 min). Reflect_only is ~15% cheaper than full PVR (consistent with the pilot20 ablation), but neither produces a single correct answer at minimal effort — i.e., the scaffold cost buys nothing on this dataset *unless* paired with a higher reasoning effort.

6. **Recommended follow-up:** rerun A/C/D with `reasoning_effort='medium'` (or the EffortLadder pinned at medium) to see whether the scaffold delivers value when the base model is actually capable. Filing the v7-dspy / EffortLadder wiring issues as separate tickets (`v7-dspy-fabric-startup`, `adaptive-engine-non-bandit-policy-zero-attempts`).


## Notes / caveats

- Pass criterion: exact-match against the LongCoT structured `answer` (template-specific equality via `bench.adaptive.longcot_adapter.evaluate_*`).

- Token totals are aggregated from `TurnRecord.prompt_tokens`/`completion_tokens` across all turns (multi-turn strategies B/C/D/E) or from the LM's last call (A).

- Single-run variance: prior PVR experiments showed ±2× attempt count between identical reruns. Headline numbers should be read as approximate.

- Strategy E uses a 3-rung effort ladder (minimal → low → medium) with `max_attempts=3, parallel_rollouts=1`. Bandit was excluded for hangs (see above).

- Full per-question traces (prompts, turns, outputs, payload) are persisted under `Files/fabric_rlm_adaptive_validation/comparison_5way/full-20260502-110000/traces_*/`.

---

## Addendum (run `fix-20260502-140800`, wheel `0.1.11.dev6`)

After the initial run we performed a forensic investigation of the two strategies that didn't execute cleanly (B and E) and reran them with the fixed wheel. Strategies A/C/D were not rerun — their original numbers stand.

### Root causes found

1. **B `dspy_rlm` — Fabric runtime ships an obsolete `pathlib` PyPI backport.** The Fabric Spark Python 3.11 image still has the abandoned (~2014) `pathlib` package installed in `site-packages/pathlib.py`. That backport contains `from collections import Sequence`, removed in Python 3.10+. The v7-dspy worker subprocess imports site-packages `pathlib` before stdlib `pathlib` and dies during startup with `ImportError: cannot import name 'Sequence' from 'collections'`. C/D were not affected because v6-custom runs in-process (no subprocess re-bootstrap). Fix: notebook setup cell now runs `pip uninstall -y -q pathlib` before installing the wheel (no-op when not present, safe everywhere).

2. **E `fabric_ladder` — `Trajectory` was implicitly falsy when empty.** `Trajectory` defines `__len__` (turn count) but not `__bool__`. Python falls back to `__len__` for truthiness, so a freshly-created trajectory with zero turns evaluated as `False`. The 5-way generator did `meta = traj.metadata if traj else {}`, which silently discarded the entire `adaptive` metadata payload (and the per-attempt records inside it). Combined with a separate wiring bug in the generator that passed `EffortLadderPolicy(base_lm_spec="azure/gpt-5", ...)` instead of `base_lm_instance=base_lm` — overriding the FabricLM auth with an unauthenticated string and causing every inner attempt to fail in ~3s. Fixes: `Trajectory.__bool__` now returns `True` (committed in `fabric_rlm/trajectory.py`); generator now passes the `base_lm` instance through and uses `is not None` guards for trajectories.

### Rerun results

| Strategy | Original | Fix run | Effective ladder behaviour |
|---|---|---|---|
| B `dspy_rlm` | 0/25 (worker dead, refusals) | **2/25** (MCM only) | Worker now starts, dspy executes 4–5 turns/question, but model still wrong on hard CS puzzles |
| E `fabric_ladder` | 0/25 (0 attempts captured) | **3/25** (MCM:2, MFMC:1) | Ladder climbs correctly; all 3 wins occurred at rung 2 (`medium` effort), and 22/25 exhausted all 3 rungs. Avg 2.32 attempts/question, 9.68 turns/question |

Both bugs are confirmed *current-codebase* bugs, not legacy issues — the `Trajectory.__bool__` fix is in the production library and the pathlib/lm-instance fixes are in the comparison generator. The remaining low pass rates are consistent with findings #1 and #2 of the original run: this is `gpt-5` at minimal/low/medium effort vs deliberately-hard CS holdout questions, not a scaffolding bug.

### Updated Findings #3 and #4

3'. **`dspy_rlm` (v7-dspy) is now usable on Fabric** if you `pip uninstall -y pathlib` before installing the wheel. With that fix the v7 worker boots and executes turns; results on the hard holdout (2/25, MCM only) are still poor, comparable to A direct, suggesting the v7-dspy engine doesn't add measurable value over a direct call on this dataset.

4'. **`fabric_ladder` (E) now actually escalates.** With `Trajectory.__bool__` and the `base_lm_instance` wiring fixed, the ladder climbs minimal → low → medium and the per-attempt metadata is preserved end-to-end. On this hard holdout it solves 3/25 — slightly better than C/D at 0/25, all wins at the medium rung, at the cost of ~2.3× more attempts and ~10 turns/question.

Fix-run artifacts (results + per-question traces): `Files/fabric_rlm_adaptive_validation/comparison_5way/fix-20260502-140800/`
