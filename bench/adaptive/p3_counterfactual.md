# Counterfactual selector replay

**Source config:** `adaptive_current_minrung3`  
**Source rollouts scanned:** 256  

## What this measures

For each rung-3 rollout produced by the source config, we replay three selector keys against the SAME captured candidate set:

- **baseline_key** — `(passed, score, conf, rf, tn, -idx)`
- **C_key**        — `(passed, score, conf, rf, tn, **cluster_size**, -idx)`
- **A_key**        — `(passed, score, conf, rf, tn, **-trace_len**, -idx)`

**winner_flip** = the alternative selector picked a DIFFERENT candidate than baseline. **pass_flip** = and that change ALSO changed the rollout's overall pass/fail. Pass-flips can only happen if the validator is imperfect within the candidate set (two candidates that the validator says PASS differ on the ground-truth grader). The bench validator IS the grader, so pass-flips are structurally near-zero — that's the duck's B1 argument made concrete.

## Per-domain results

| domain | rung3 | with_choice | cluster>1 | C_flip | C_pass_flip | A_flip | A_pass_flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| dabench | 24 | 24 | 8 | 0 | 0 | 0 | 0 |
| easy_calibration | 10 | 10 | 10 | 0 | 0 | 0 | 0 |
| longcot_holdout | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| math | 10 | 10 | 10 | 0 | 0 | 0 | 0 |
| ssb | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | **64** | **64** | **28** | **0** | **0** | **0** | **0** |

## Interpretation

- If `C_pass_flip` is ~0 on this same candidate set, then the +17pp DABench lift observed in the live A/B is sampling variance, not selector C's causal contribution.
- `C_flip` > 0 with `C_pass_flip` ≈ 0 means C does change WHICH candidate is reported (potentially affecting downstream cost / answer character) but does NOT change the bench-grader outcome.
- The right way to demonstrate a causal C lift is to either (a) replay against many rollouts with KNOWN ground truth that DIFFERS from the validator (impossible here — validator IS the grader), or (b) run a long-context bench where the validator is weaker and selector signal can dominate.

