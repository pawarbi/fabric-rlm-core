# SRLM bench summary

## Per-config aggregate

| config | n | passed | accuracy | 95% CI | mean total tokens | mean elapsed s |
|---|---:|---:|---:|---|---:|---:|
| adaptive_current | 2 | 2 | 1.000 | [1.000, 1.000] | 1354 | 2.08 |
| default | 2 | 2 | 1.000 | [1.000, 1.000] | 1384 | 8.56 |

## Per-domain accuracy

| config | easy_calibration |
|---|---:|
| adaptive_current | 2/2 |
| default | 2/2 |


---

## Feature A — Trace-Length Tiebreaker — Bench Result

- **Question set**: 32 questions across 5 domains (easy_calibration: 5, math: 5, dabench: 12, ssb: 5, longcot_holdout: 5)
- **Configs**: `adaptive_current` vs `adaptive_a` (`prefer_shorter_traces=True`)
- **Seeds**: 3 (n=96 rollouts per config; 192 total)
- **Model**: openai/gpt-4.1 via OpenRouter

### Per-config aggregate

| Config            | Acc (mean) | Acc 95% CI       | Mean tokens | Mean elapsed_s |
|---|---:|---|---:|---:|
| adaptive_current  | 0.302      | [0.156, 0.469]   | 6611        | 17.69          |
| adaptive_a        | 0.292      | [0.146, 0.458]   | 6861        | 19.74          |

### Per-domain accuracy

| Domain            | adaptive_current | adaptive_a | delta   |
|---|---:|---:|---:|
| easy_calibration  | 15/15            | 15/15      | +0.000  |
| math              | 14/15            | 13/15      | -0.067  |
| dabench           | 0/36             | 0/36       | +0.000  |
| ssb               | 0/15             | 0/15       | +0.000  |
| longcot_holdout   | 0/15             | 0/15       | +0.000  |

### Selection-path diagnostics

Critical observation from per-rollout instrumentation:

| Winner rung           | adaptive_current | adaptive_a |
|---|---:|---:|
| 0 (single attempt)    | 95               | 95         |
| 1 (single attempt)    | 1                | 1          |
| 3 (best-of-N parallel)| **0**            | **0**      |

**Tie-break candidates** (rung-3 selections with >=2 rollouts of differing trace length where Feature A's late-tier tie-break could have fired): **0 / 96**.

The tie-break code path was **never exercised** in this bench. The question difficulty profile and current adaptive policy resolved 99% of questions at rung 0 (single rollout), so `select_best_of_n` never had multiple equal-(passed,score,confidence,completeness) candidates to disambiguate.

### Pre-registered Feature A ship gates

- **Zero observed regressions at per-question level?** **NO** (1 regression: `AQUA_0003` seed 0, math). However: this regression occurred at `winner_rung=0` — a single-rollout terminal pass. The Feature A flag **cannot have caused it** (the tie-break code path is unreachable at rung 0). Attributed to LM non-determinism between separate API calls.
- **No domain regression?** Math: -0.067 (1/45 attempts); all other domains identical. Within seed-noise bounds; not statistically significant given 95% CIs overlap heavily.
- **Cost neutral or better?** Slight increase: +250 mean tokens (+3.8%), +2.05s elapsed (+11.6%). Within seed noise; not attributable to Feature A given the tie-break never fired.
- **5 calibration questions still 5/5?** **YES** — 15/15 across 3 seeds for both configs.
- **Tie-break ACTUALLY fired in any rollout?** **NO** — 0 of 96 adaptive_a rollouts reached rung-3 best-of-N with disambiguable candidates.

### Recommended ship tier: **experimental**

**Justification**: Feature A is correctly implemented (9 unit tests pass, including byte-identical-default regression and end-to-end plumbing), but this 32q bench provides **zero positive evidence** of behavioral benefit because the rung-3 best-of-N path was never triggered. The single observed accuracy delta (math 14/15 -> 13/15) is at rung 0 and therefore mechanically independent of the flag. Recommendation: keep the flag opt-in (`adaptive={"prefer_shorter_traces": True}`) and defer "recommended" tier until a bench is constructed that actually triggers rung-3 multi-rollout selection (e.g. higher `parallel_rollouts`, harder questions, or lower rung-0 pass rate). Default behavior is byte-identical, so risk of shipping the code is zero; the open question is utility, not safety.

### Surprising findings

1. **Validators dead on 3 of 5 domains** in this bench: dabench (0/36), ssb (0/15), longcot_holdout (0/15) all show 0 passes for **both** configs. Ship-decision signal is dominated by easy_calibration (saturated at 5/5) and math (5 questions x 3 seeds = 15 datapoints).
2. **Rung-3 never reached**: with the current adaptive policy and K=3 parallel rollouts, the best-of-N selection path is essentially dead code on this bench. Feature A (and any future selection-time tie-breaker) cannot be evaluated here without a bench refresh.
