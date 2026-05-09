# SRLM trajectory analysis

_Sources: bench\adaptive\results\srlm_eval_p3_  
_Total rollouts ingested: 256_

## 1. Cost / accuracy frontier

| config | n | accuracy | mean tokens | median tokens | mean elapsed (s) |
|---|---:|---:|---:|---:|---:|
| `adaptive_a_minrung3` | 64 | 0.516 | 23303 | 19622 | 26.33 |
| `adaptive_all` | 64 | 0.484 | 31786 | 19683 | 37.44 |
| `adaptive_c_minrung3` | 64 | 0.516 | 25496 | 20698 | 23.83 |
| `adaptive_current_minrung3` | 64 | 0.469 | 23053 | 15080 | 25.94 |

## 6. Per-domain pass rate × config

| domain | config | n | pass rate | mean tokens |
|---|---|---:|---:|---:|
| dabench | `adaptive_a_minrung3` | 24 | 0.542 | 30545 |
| dabench | `adaptive_all` | 24 | 0.458 | 38522 |
| dabench | `adaptive_c_minrung3` | 24 | 0.583 | 34505 |
| dabench | `adaptive_current_minrung3` | 24 | 0.417 | 33244 |
| easy_calibration | `adaptive_a_minrung3` | 10 | 1.000 | 4114 |
| easy_calibration | `adaptive_all` | 10 | 1.000 | 1371 |
| easy_calibration | `adaptive_c_minrung3` | 10 | 1.000 | 4137 |
| easy_calibration | `adaptive_current_minrung3` | 10 | 1.000 | 4096 |
| longcot_holdout | `adaptive_a_minrung3` | 10 | 0.000 | 54641 |
| longcot_holdout | `adaptive_all` | 10 | 0.000 | 95805 |
| longcot_holdout | `adaptive_c_minrung3` | 10 | 0.000 | 63011 |
| longcot_holdout | `adaptive_current_minrung3` | 10 | 0.000 | 51997 |
| math | `adaptive_a_minrung3` | 10 | 1.000 | 6982 |
| math | `adaptive_all` | 10 | 1.000 | 2561 |
| math | `adaptive_c_minrung3` | 10 | 0.900 | 7971 |
| math | `adaptive_current_minrung3` | 10 | 1.000 | 6635 |
| ssb | `adaptive_a_minrung3` | 10 | 0.000 | 10095 |
| ssb | `adaptive_all` | 10 | 0.000 | 11240 |
| ssb | `adaptive_c_minrung3` | 10 | 0.000 | 5247 |
| ssb | `adaptive_current_minrung3` | 10 | 0.000 | 5026 |

## 2. Rung-3 payoff (did extra rollouts buy us anything?)

_For rollouts that reached rung 3: was BoN choosing among meaningfully
different candidates, or rubber-stamping unanimous output?_

| config | rung3 rollouts | K=1 | all pass (no choice) | all fail (no rescue) | validator split (won) | validator split (lost) |
|---|---:|---:|---:|---:|---:|---:|
| `adaptive_a_minrung3` | 64 | 0 | 19 | 31 | 14 | 0 |
| `adaptive_all` | 4 | 0 | 0 | 0 | 4 | 0 |
| `adaptive_c_minrung3` | 64 | 0 | 18 | 31 | 15 | 0 |
| `adaptive_current_minrung3` | 64 | 0 | 17 | 34 | 13 | 0 |

**Read:** large `all pass` columns ⇒ rung-3 BoN burned tokens to confirm the obvious — early-exit candidate. Large `all fail` columns ⇒ rung-3 didn't rescue rung-1 wrongness — escalation policy needs work.

## 3. Wasted compute (was rung-3 unanimous?)

_How often did rung-3 BoN spend ~K× the tokens just to confirm a unanimous answer?_

| config | rung3 rollouts (K>1) | unanimous (waste) | split (real choice) | all singletons (model disagreed entirely) |
|---|---:|---:|---:|---:|
| `adaptive_a_minrung3` | 64 | 20 (31%) | 15 (23%) | 29 (45%) |
| `adaptive_all` | 4 | 0 (0%) | 3 (75%) | 1 (25%) |
| `adaptive_c_minrung3` | 64 | 21 (33%) | 11 (17%) | 32 (50%) |
| `adaptive_current_minrung3` | 64 | 18 (28%) | 10 (16%) | 36 (56%) |

**Read:** if `unanimous` is a high %, the novel-method opportunity is **early exit** — skip rung-3 BoN when rung-1's answer already looks confident.

## 4. Validator value-add (does score predict pass?)

_Among candidates that have a numeric score, split by score≥median and validator pass._

| config | candidates w/ score | high-score+pass | high-score+fail | low-score+pass | low-score+fail | validator–score agreement |
|---|---:|---:|---:|---:|---:|---:|
| `adaptive_a_minrung3` | 0 | 0 | 0 | 0 | 0 | 0% |
| `adaptive_all` | 0 | 0 | 0 | 0 | 0 | 0% |
| `adaptive_c_minrung3` | 0 | 0 | 0 | 0 | 0 | 0% |
| `adaptive_current_minrung3` | 0 | 0 | 0 | 0 | 0 | 0% |

**Read:** if `validator–score agreement` is high, score alone might suffice — validator could be a cheaper rubric. If low, validator is catching things score misses (keep it).

## 5. Consensus calibration (does cluster size predict correctness?)

_Across every candidate ever scored, group by cluster size, report pass rate._

### 5a. Pooled (CONFOUNDED — read 5b first)

| cluster size | candidates | pass rate |
|---:|---:|---:|
| 1 | 461 | 0.072 |
| 2 | 100 | 0.340 |
| 3 | 192 | 0.828 |

> ⚠️ **The pooled monotone trend is a Simpson's-paradox artifact.** Easy/math domains have near-100% pass AND high consensus, so any pooled view on a question set with easy domains will show size↑ → pass↑ even when consensus has zero signal on the hard domain that actually matters. Always read 5b.

### 5b. Stratified by (config × domain)

| config | domain | size 1 | size 2 | size 3+ |
|---|---|---|---|---|
| `adaptive_a_minrung3` | dabench | 8/46=0.17 | 12/20=0.60 | 0/6=0.00 |
| `adaptive_a_minrung3` | easy_calibration | — | — | 30/30=1.00 |
| `adaptive_a_minrung3` | longcot_holdout | 0/24=0.00 | 0/6=0.00 | — |
| `adaptive_a_minrung3` | math | 0/2=0.00 | 4/4=1.00 | 24/24=1.00 |
| `adaptive_a_minrung3` | ssb | 0/30=0.00 | — | — |
| `adaptive_all` | dabench | 1/46=0.02 | 6/22=0.27 | 0/9=0.00 |
| `adaptive_all` | longcot_holdout | 0/40=0.00 | 0/4=0.00 | 0/6=0.00 |
| `adaptive_all` | ssb | 0/48=0.00 | 0/2=0.00 | — |
| `adaptive_c_minrung3` | dabench | 12/49=0.24 | 6/20=0.30 | 0/3=0.00 |
| `adaptive_c_minrung3` | easy_calibration | — | — | 30/30=1.00 |
| `adaptive_c_minrung3` | longcot_holdout | 0/27=0.00 | — | 0/3=0.00 |
| `adaptive_c_minrung3` | math | 1/1=1.00 | 0/2=0.00 | 24/27=0.89 |
| `adaptive_c_minrung3` | ssb | 0/30=0.00 | — | — |
| `adaptive_current_minrung3` | dabench | 11/55=0.20 | 0/14=0.00 | 0/3=0.00 |
| `adaptive_current_minrung3` | easy_calibration | — | — | 30/30=1.00 |
| `adaptive_current_minrung3` | longcot_holdout | 0/30=0.00 | — | — |
| `adaptive_current_minrung3` | math | 0/3=0.00 | 6/6=1.00 | 21/21=1.00 |
| `adaptive_current_minrung3` | ssb | 0/30=0.00 | — | — |

**Read:** look for monotone size↑ → pass↑ within a SINGLE row (one config × one domain). If that fails on the domain we actually care about (DABench), self-consistency is not a useful signal there and Feature C should not be expected to help.
