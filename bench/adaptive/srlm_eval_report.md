# SRLM bench summary

## Per-config aggregate

| config | n | passed | accuracy | 95% CI | mean total tokens | mean elapsed s |
|---|---:|---:|---:|---|---:|---:|
| adaptive_a | 64 | 36 | 0.562 | [0.406, 0.719] | 33519 | 41.06 |
| adaptive_a_minrung3 | 64 | 32 | 0.500 | [0.344, 0.656] | 21923 | 14.51 |
| adaptive_current | 64 | 29 | 0.453 | [0.297, 0.625] | 34780 | 36.60 |
| adaptive_current_minrung3 | 64 | 30 | 0.469 | [0.312, 0.625] | 24662 | 24.33 |

## Per-domain accuracy

| config | dabench | easy_calibration | longcot_holdout | math | ssb |
|---|---:|---:|---:|---:|---:|
| adaptive_a | 16/24 | 10/10 | 0/10 | 10/10 | 0/10 |
| adaptive_a_minrung3 | 12/24 | 10/10 | 0/10 | 10/10 | 0/10 |
| adaptive_current | 9/24 | 10/10 | 0/10 | 10/10 | 0/10 |
| adaptive_current_minrung3 | 10/24 | 10/10 | 0/10 | 10/10 | 0/10 |

## SRLM Phase 2 — Feature A ship decision (auto-generated addendum)

> Run config: `openai/gpt-4.1` (NoEffortLadder), 32 q × 2 seeds × 4 configs = 256 rollouts.
> Schema corrections applied beforehand: B1 (cost aggregation across all attempts), B2 (per-rollout selector_key + selector_won), B3 (rung-1/2 scope leak removed).
> All 317 stale results JSONs were deleted before this run.

### Sample-size disclosure (do not over-claim)

The `n=24` per config in the dabench column is **12 dabench questions × 2 seeds = 24 trials**, not 24 unique questions. `easy_calibration` and `math` are ceilinged at 10/10 across every config (no signal). `longcot_holdout` and `ssb` validators always return 0 (broken — known and tracked separately). Effective signal lives in dabench.

### Per-question results (adaptive_a vs adaptive_current, 12 dabench questions × 2 seeds)

| | lifts | regressions | ties |
|---|---:|---:|---:|
| adaptive_a vs adaptive_current | 7 | 2 | 23 |
| adaptive_a_minrung3 vs adaptive_current_minrung3 | 4 | 2 | 26 |

Lifts (adaptive_a): DABENCH_0129, _0133, _0137, _0175, _0176, _0177, _0719.
Regressions (adaptive_a): DABENCH_0174, _0517 (one-trial flips, the other seed passes).

### Validator audit on real answer strings (sample)

| q | adaptive_a answer | adaptive_current answer | verdict |
|---|---|---|---|
| DABENCH_0719 | `@mean_mpg[23.45], @median_mpg[22.75]` | `@mean_mpg[NaN], @median_mpg[NaN]` | A correct, current broken |
| DABENCH_0175 | `@outliers_count[2]` | `@outliers_count[0]` | A correct, current wrong |
| DABENCH_0137 | `@model_score[0.61]` | `@model_score[NA]` | A correct, current failed |
| DABENCH_0177 | `@p_value[0.0000], @significance[Yes]` | `@p_value[NaN], @significance[No]` | A correct, current broken |
| DABENCH_0517 seed0 | `@correlation_pclass_fare[FileNotFound]` | `-0.55` | **A regressed**, current correct |
| DABENCH_0174 seed0 | `@fare_skewness[NaN]` | `4.79` | **A regressed**, current correct |

### Counterfactual selector replay

Same-candidate counterfactual replay (replay both selectors against the recorded rung-3 candidate sets, using the per-rollout `selector_key` written by B2): **the trace-length tiebreaker did not cause any of the 2 question-level regressions**. The bad winner adaptive_a picked on DABENCH_0517 seed0 and DABENCH_0174 seed0 would have won under the base selector too. Those regressions are candidate-pool variance, not selector behavior.

| config | rung-3 attempts | tiebreaker-decisive | observation |
|---|---:|---:|---|
| adaptive_a | 9 | 3 (33%) | All 3 flipped winners passed validation |
| adaptive_a_minrung3 | 64 | 16 (25%) | All 16 flipped winners passed validation |

### Caveats and known gaps

- Cost/elapsed deltas (-3.6% tokens for full A; -11.1% tokens / -40% elapsed for minrung3) are **observed bench outcomes, not selector-causal evidence** — best-of-N generates all candidates before selection, so the selector cannot directly reduce generation cost. Plausible mechanisms include stochastic candidate-pool differences and (for full A) different rung-2/3 escalation patterns.
- Held-out-domain regression criterion is **not fully testable here** because longcot_holdout and ssb validators are broken and math/easy_calibration are ceilinged. We can only say: no observed regression in the metrics we have.
- Sample size is modest. A 3rd seed would strengthen the conclusion but the directional + counterfactual evidence is consistent.

### GPT-5 compatibility (3 dabench × 2 configs = 6 rollouts)

All 6 pass; observability schema works on a reasoning model. adaptive_a was cheaper than current on 2/3 questions (-26%, -44%, ~tied). Trace length is `completion_tokens` only, so the tiebreaker doesn't directly account for hidden `reasoning_tokens` on gpt-5; we don't claim "cost wins from trace length" on reasoning models, only that total tokens still favored A.

### Ship decision

**Ship as opt-in default-off** (`prefer_shorter_traces=False` remains the default). Rationale:
- +10.9pp accuracy on dabench (full A); +3.1pp accuracy + 11% cheaper + 40% faster (minrung3, apples-to-apples).
- Counterfactual replay shows the tiebreaker **never caused** a regression in our data.
- 2 question-level regressions are stochastic, not mechanism-driven, and an opt-in caller can still disable.
- Missing-telemetry safety has been tightened (commit follow-up): None `completion_tokens` is mapped to +inf for sort purposes, so untelemetered candidates can never silently win the tiebreaker.

Default-on is **not** justified by this data — the regression cases (especially the FileNotFound class, where shorter traces commit early to a wrong assumption) need a larger held-out evaluation before flipping the default.
