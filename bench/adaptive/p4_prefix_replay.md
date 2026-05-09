# Prefix-replay simulator (Feature E feasibility)

**Question:** if we had stopped launching rung-3 candidates after K rollouts, how often would the predicate fire, how many completion tokens would we have saved, and would the selected winner / overall pass-fail have changed?

**Files scanned:** 256  
**Rung-3 rollouts replayed (N≥2):** 196  
**Source config filter:** `ALL CONFIGS`

## Predicates

- **all_pass** — every candidate in prefix passed validator. Provably no overall pass/fail change when validator IS grader (safe default).
- **all_fail_same** — every candidate failed AND they share `consensus_cluster_id`. Strict opt-in: suffix could rescue with a passing candidate (any pass-flip > 0 disqualifies it as a default).

## Per-config × per-domain × per-predicate aggregate

| config | domain | pred | n | fires | fire_rate | winner_flips | pass_flips | mean_tokens_saved (when fires) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| adaptive_a_minrung3 | dabench | all_pass | 24 | 5 | 21% | 1 | 0 | 1034 |
| adaptive_a_minrung3 | dabench | all_fail_same | 24 | 3 | 12% | 1 | 1 | 390 |
| adaptive_a_minrung3 | easy_calibration | all_pass | 10 | 10 | 100% | 6 | 0 | 153 |
| adaptive_a_minrung3 | easy_calibration | all_fail_same | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_a_minrung3 | longcot_holdout | all_pass | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_a_minrung3 | longcot_holdout | all_fail_same | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_a_minrung3 | math | all_pass | 10 | 10 | 100% | 6 | 0 | 561 |
| adaptive_a_minrung3 | math | all_fail_same | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_a_minrung3 | ssb | all_pass | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_a_minrung3 | ssb | all_fail_same | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_all | dabench | all_pass | 4 | 0 | 0% | 0 | 0 | 0 |
| adaptive_all | dabench | all_fail_same | 4 | 0 | 0% | 0 | 0 | 0 |
| adaptive_c_minrung3 | dabench | all_pass | 24 | 3 | 12% | 0 | 0 | 778 |
| adaptive_c_minrung3 | dabench | all_fail_same | 24 | 3 | 12% | 1 | 1 | 371 |
| adaptive_c_minrung3 | easy_calibration | all_pass | 10 | 10 | 100% | 0 | 0 | 161 |
| adaptive_c_minrung3 | easy_calibration | all_fail_same | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_c_minrung3 | longcot_holdout | all_pass | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_c_minrung3 | longcot_holdout | all_fail_same | 10 | 1 | 10% | 0 | 0 | 5076 |
| adaptive_c_minrung3 | math | all_pass | 10 | 9 | 90% | 0 | 0 | 396 |
| adaptive_c_minrung3 | math | all_fail_same | 10 | 1 | 10% | 0 | 0 | 671 |
| adaptive_c_minrung3 | ssb | all_pass | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_c_minrung3 | ssb | all_fail_same | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_current_minrung3 | dabench | all_pass | 24 | 3 | 12% | 0 | 0 | 1074 |
| adaptive_current_minrung3 | dabench | all_fail_same | 24 | 4 | 17% | 1 | 1 | 302 |
| adaptive_current_minrung3 | easy_calibration | all_pass | 10 | 10 | 100% | 0 | 0 | 146 |
| adaptive_current_minrung3 | easy_calibration | all_fail_same | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_current_minrung3 | longcot_holdout | all_pass | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_current_minrung3 | longcot_holdout | all_fail_same | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_current_minrung3 | math | all_pass | 10 | 9 | 90% | 0 | 0 | 527 |
| adaptive_current_minrung3 | math | all_fail_same | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_current_minrung3 | ssb | all_pass | 10 | 0 | 0% | 0 | 0 | 0 |
| adaptive_current_minrung3 | ssb | all_fail_same | 10 | 0 | 0% | 0 | 0 | 0 |

## Interpretation guide

- **all_pass with pass_flips=0 across all rows**: confirms the safety claim. Cost savings = `fires × mean_tokens_saved` per row.
- **all_fail_same with pass_flips > 0 in any row**: that row has rollouts where the suffix would have rescued a failing prefix. Cannot ship as default; ship behind a stricter opt-in flag and document the empirical risk per domain.
- **First-fire K distribution** (`fires_at_kN` keys in JSON): if most fires happen at K=1, an even more aggressive policy (skip rung 3 entirely after 1 passing rung-2 candidate) may be viable in a future phase.

