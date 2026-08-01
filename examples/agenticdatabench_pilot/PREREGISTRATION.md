# Pre-registration: does `output_contract` improve AgenticDataBench scores?

Written before any held-out run. Fixed in advance so the result cannot be
rewritten after the fact.

## Why this exists

The `output_contract` skill was drafted after reading three pilot failures
(`agriculture_22`, `strategy_2`, `strategy_3`) against their gold files.
Rules written from observed failures and then measured on those same
failures produce an inflated result. Those three tasks are therefore burned
as a measurement set and appear nowhere below.

## What is explicitly not being done

No per-data-source hints, no per-task context, and nothing derived from
inspecting gold files goes into the skill or the prompt. Facts learned from
gold during the pilot ("`total` means outbound-call rows in strategy_3",
"gold sorts descending") are answer leakage; a skill carrying them would
measure gold-reading, not task-reading. The skill contains only
domain-independent rules that a careful reader could derive from a task
statement alone.

(Per-source context files are correct in production against a real
lakehouse, where house metric definitions are legitimate input. The
prohibition is specific to benchmark measurement.)

## Hypothesis

Two of three pilot failures were output-contract failures rather than
analysis failures: a positional metric-to-column mapping read by intuition
instead of by order, and a metric name whose implied row filter was
ignored. If that generalizes, a skill that forces contract extraction
before analysis and a reload-and-check before submit will raise the mean
score without materially raising cost.

## Sample

Ten tasks drawn by a rule fixed before inspection:

- graded solely by `compare_csv`
- has a `data_sources` field
- not one of the three pilot tasks
- question does not mention cross-validation or random forest (the
  `agriculture_22` audit showed unseeded model-training gold is not
  reliably passable by anyone)
- at most three input files (download budget)
- seeded shuffle (`random.seed(20260731)`), at most two per domain, first
  ten taken

Sample: `social_network_18`, `social_network_25`, `sports_20`,
`agriculture_12`, `energy_09`, `sports_25`, `tourism_32`, `tourism_25`,
`real_estate_22`, `agriculture_20`.

The sample is frozen. Tasks that fail for environmental reasons (missing
input file, unreadable format) are reported as excluded with the reason,
and excluded from both arms equally; they are not replaced with
alternatives chosen later.

## Design

Paired: every task runs twice with the same model
(`openrouter/minimax/minimax-m3`), same `max_turns`, same prompt builder.
Arm A is baseline (no `skills=`). Arm B adds `skills=["output_contract"]`
via a `SkillLoader` pointed at `contrib-skills/`. Grading is
`grade_pilot.py`, which calls AgenticDataBench's own `compare_csv`.

Single seed, temperature at library default. Two identical fabric-rlm runs
historically agree on about 84 percent of tasks, so run-to-run noise is
real and the decision rule below accounts for it.

## Decision rule (fixed in advance)

Let `d` be mean(arm B) minus mean(arm A) over the ten tasks, and let `n_up`
and `n_down` be the counts of tasks whose score rose or fell by more than
0.01.

- **Adopt** and document as measured if `d >= +0.10` and `n_down <= 1`.
- **Keep as unmeasured method documentation** (ship in contrib, no
  performance claim) if `-0.05 < d < +0.10`.
- **Reject** and delete the skill if `d <= -0.05` or `n_down >= 4`.

Ten tasks cannot establish significance. Whatever the outcome, the write-up
states the sample size and reports McNemar-style paired counts rather than
implying a general accuracy gain. No post-hoc subgroup ("it helped on the
tasks with explicit headers") will be used to rescue a null result; any
such observation becomes a new hypothesis for a later, larger run.

---

# Result: REJECTED (run 2026-07-31)

MiniMax M3 via OpenRouter, `max_turns=12`, single seed, both arms run from
the same commit. Graded by `grade_pilot.py` calling AgenticDataBench's
`compare_csv`. Total spend for both arms: about $0.15.

| task | baseline | +output_contract | delta |
|---|---|---|---|
| social_network_18 | 1.000 | 0.833 | -0.167 |
| social_network_25 | 1.000 | 0.444 | -0.556 |
| sports_20 | 0.583 | 0.583 | 0 |
| agriculture_12 | 0.000 | 0.000 | 0 |
| energy_09 | 1.000 | 0.600 | -0.400 |
| sports_25 | 1.000 | 1.000 | 0 |
| tourism_32 | 0.429 | 1.000 | +0.571 |
| tourism_25 | 0.500 | 0.500 | 0 |
| real_estate_22 | 0.000 | 0.000 | 0 |
| agriculture_20 | 0.000 | 0.000 | 0 |

Mean 0.551 baseline against 0.496 with the skill: **d = -0.055**, with
one task up, three down, six unchanged. `d <= -0.05` triggers the
pre-registered reject branch, so the skill is not shipped.

## The result is uninterpretable: the noise floor is larger than the effect

A third run was then done as a control: the **baseline configuration
repeated, identical in every respect**, no skill. It scored 0.450.

| task | base run 1 | base run 2 | +skill |
|---|---|---|---|
| social_network_18 | 1.000 | 1.000 | 0.833 |
| social_network_25 | 1.000 | **0.444** | 0.444 |
| sports_20 | 0.583 | 0.583 | 0.583 |
| agriculture_12 | 0.000 | 0.000 | 0.000 |
| energy_09 | 1.000 | **0.600** | 0.600 |
| sports_25 | 1.000 | **0.375** | 1.000 |
| tourism_32 | 0.429 | **1.000** | 1.000 |
| tourism_25 | 0.500 | 0.500 | 0.500 |
| real_estate_22 | 0.000 | 0.000 | 0.000 |
| agriculture_20 | 0.000 | 0.000 | 0.000 |
| **mean** | **0.551** | **0.450** | **0.496** |

Two identical baseline runs differ by **-0.101** and agree on only 6 of 10
tasks. The skill effect being tested was **-0.055**. The noise floor is
roughly twice the effect, so this design cannot resolve the question it
was built to answer, in either direction.

Worse for the earlier reading: the specific regressions that were
attributed to the skill reproduce exactly **without** it. `social_network_25`
drops to 0.444 and `energy_09` to 0.600 in the no-skill control, and
`tourism_32` rises to 1.000 there too. A first draft of this document
explained those three as skill mechanism -- a row-filter rule causing the
model to re-derive populations it already had right. That explanation was
fitted to noise and is withdrawn. It was plausible, internally consistent,
and wrong, which is exactly why the control run was necessary.

`sports_25` is the clearest single illustration: 1.000, then 0.375, then
1.000 again, with no configuration change of any kind between the first
and second.

## Corrected verdict

The pre-registered rule was applied honestly and returns REJECT, but the
rule itself was under-powered: it set a decision threshold of 0.05 against
a measurement whose noise floor is 0.10. **The correct scientific verdict
is not "harmful" but "no measurement was obtained."** The skill stays out
of the repository, because nothing here shows it helps and shipping
unmeasured guidance by default is what the contrib directory exists to
avoid; but nothing here shows it hurts either.

## Consequences for any future run

1. **Pin the temperature.** fabric-rlm defaults non-reasoning models to
   `temperature=1.0` (`_smart_defaults` in `fabric_rlm/lm.py`), and
   MiniMax M3 takes that path. That is the leading suspect for the
   variance. `run_pilot.py --temperature` now exists for measurement runs.
2. **Raise the turn ceiling.** `max_turns=12` bound on 4 of 10 tasks and
   truncated `agriculture_20` mid-write in both arms. The default is now
   25, with `--timeout` (per-turn execution, default 900s) exposed too.
3. **Replicate before comparing.** Any future arm comparison needs at
   least one same-config control run, and a decision threshold set above
   the observed noise floor rather than guessed in advance.
4. **Sample size.** At a per-task standard deviation this large, ten
   tasks is far too few. The full 246-task public set costs only a few
   dollars at roughly half a cent per task, so there is no good reason to
   measure on a small sample.
