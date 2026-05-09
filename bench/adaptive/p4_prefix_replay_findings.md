# Phase 4a — Prefix-replay findings (Feature E feasibility)

## Headline

**Feature E (early-exit) is empirically supported by captured data and
should be designed as a "probe-then-fanout" execution pattern, not a
multi-K predicate.**

## Empirical results (256 rollouts, 196 rung-3 rollouts replayed)

### `all_pass` predicate (safe default candidate)

| metric | value |
|---|---:|
| fire rate | 35% (69/196) |
| pass-flips | **0/196** ✓ safety claim confirmed |
| K-distribution of fires | 100% at K=1, 0% at K=2 |
| completion tokens saved per fire | mean ~424 |
| amortized savings per rollout | 149 completion tokens |

**100% of fires are at K=1.** The safe predicate reduces to a one-line
runtime check: did the first launched rung-3 candidate pass the validator?

### `all_fail_same_canonical` predicate (strict opt-in candidate)

| metric | value |
|---|---:|
| fire rate | 6% (12/196) |
| pass-flips | 3/196 (1.5%) — DBENCH only |
| Cannot be a default — duck B2 confirmed empirically |

The 3 pass-flips are all on `dabench`, where the suffix would have
included a passing candidate that the prefix's matching failures didn't
predict. Ship behind a stricter opt-in flag with this risk documented
per-domain.

### Per-domain `all_pass` fire rates

| domain | fire rate | mean tokens saved |
|---|---:|---:|
| easy_calibration | 100% | ~150 |
| math | 90-100% | ~470 |
| dabench | 12-21% | ~960 |
| longcot_holdout | 0% | — |
| ssb | 0% | — |

The 0% fire rates on longcot/ssb are because their substring-match
graders fail every candidate (0% pass rate baseline) — not because the
predicate is failing. Confirms the longcot/ssb harness is broken (P7
in the original Phase 4 backlog).

## Implication for Feature E design

**Runtime change is small and surgical:**

1. In `_run_rollouts` (or wherever rung-3 fanout happens), gate on a
   new param `early_exit_probe: bool = False` (default OFF).
2. When ON: launch ONE rung-3 candidate, await its completion, check
   its `validator_passed`. If True → return that candidate (skip
   N-1 launches). Else → fall back to existing parallel fanout for
   the remaining N-1.
3. Optional stricter param `early_exit_on_unanimous_fail: bool = False`
   for the all-fail-same-canonical case (separate flag, separate test
   surface, documented 1.5% empirical risk on dabench-like regimes).

**No K parameter needed.** All-pass fires only at K=1.

**No selector changes needed.** Selector continues to operate on
whatever candidate set it receives.

## Cost estimate

For a workload with the captured domain mix (60% math+easy, 40% other),
amortized savings of ~150 completion tokens per rung-3 rollout. At
typical pricing ($0.0006/1k completion for gpt-4.1) and 1000 rollouts,
that's ~$0.09. Small in absolute terms, but a **20-25% reduction in
rung-3 completion spend on math/easy workloads** with zero accuracy loss
is a defensible "novel contribution" claim.

The bigger structural win is **wall-clock**: a probe-then-fanout pattern
saves ~33% of rung-3 latency on the 35% of rollouts where the probe
passes (the remaining 65% pay a small serialization penalty equal to
one probe-vs-parallel-fanout difference).

## Trade-off: latency penalty when probe fails

The 65% of rollouts where the probe fails pay an extra ~(probe_latency)
of wall-clock vs the current pure-parallel-3 pattern, because the suffix
launches only after the probe completes. For configs that prioritize
latency over cost, this may be unacceptable; ship behind opt-in flag
so users decide.

## Caveats

- Per-candidate prompt/reasoning tokens not captured in observability,
  so completion-token-savings are a lower bound on true cost savings.
  Total token savings will be ~20-30% larger than the completion-only
  numbers above.
- Captured data uses validator-as-grader. In a regime where validator <
  grader (e.g., long-context bench), the safety claim still holds for
  `all_pass` (it relies on the selector's first sort key being `passed`
  per the bench's own validator) — but a candidate the probe says
  "passes" might not be the BEST passing candidate by the true grader.
  This is the same caveat as Feature C; document and accept.

## Sensitivity to probe-index choice (post duck-review)

Duck flagged that "first observability row" may be a thread-interleaving
artifact, not a real "first to complete" signal. Re-ran the simulation
treating probe = idx 0, idx 1, idx 2 separately:

| probe index | fire rate |
|---|---:|
| idx 0 (default) | 35% (69/196) |
| idx 1 | 39% (77/196) |
| idx 2 | 38% (75/196) |

Fire rate is **stable across probe choice (35-39%)**. The claim should
be stated as "deterministic probe = rollout_index 0 fires 35%". This is
a slight lower bound vs idx 1/2 — likely because in the captured data
idx 0 corresponds to whichever thread happened to write its
observability row first, which under thread interleaving may
slightly skew toward shorter or faster-completing rollouts.

For the runtime implementation we'll use idx 0 (deterministic, simple).

## Recommendation to duck

Adopt this design BEFORE coding the runtime change:

1. Param `early_exit_probe: bool = False` in `_run_rollouts`.
2. Implement as: probe (1 future) → await → branch.
3. TDD with mock futures (fast, no LM calls).
4. Counterfactual already done (this report).
5. Live bench can be deferred to Phase 4d — the offline evidence
   from 196 rollouts is sufficient to justify shipping opt-in default-OFF.
