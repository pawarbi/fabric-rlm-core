# PARKED: SRLM features (Phases 2 / 3 / 4)

**Status:** Parked 2026-05-09. Nothing on this branch is merged to `main`.
**Branch:** `feature/srlm-bench-fixes` (12 commits ahead of `main`, this branch).

> Read this file first if you (or future-me) come back to this work.

---

## What was tried

Three SRLM (Self-Rewarding Language Models) selector features for the
adaptive engine's rung-3 best-of-N path:

| Feature | Phase | What it does | Status |
|---|---|---|---|
| **A** Trace-length tiebreaker | 2 | Among rung-3 BoN winners that all pass the validator, prefer the shorter completion trace. | Code shipped on branch; opt-in default-OFF. **0 decision-flips on bench.** |
| **C** Self-consistency cluster tiebreaker | 3 | Among rung-3 BoN winners, prefer the answer that the most rollouts agree on. | Code shipped on branch; opt-in default-OFF. **0 decision-flips on bench.** |
| **E** Speculative Fanout (early-exit probe) | 4 | Run 1 rollout first; only fan out to K=3 if the probe fails the validator. | Code shipped on branch; opt-in default-OFF. **Saves ~30% tokens but 0 accuracy lift over single-call baseline.** |

Plus bench-harness scaffolding to even *measure* these:
- `LadderPolicy.start_rung` field (force the engine to start at rung N)
- `force_min_rung` config knob + `adaptive_*_minrung3` configs
- Per-domain validators (DABench, SSB, LongCoT) — substring grader was returning 0/N on 3 of 5 domains
- Bench cost summing across all rollouts (winner-only undercounted ~30%)
- `litellm.drop_params=True` in `make_lm` (non-reasoning models hard-crashed on `reasoning_effort`)

---

## Why we parked

**The bench tells the truth and it's not flattering.**

Honest 3-way smoke run, 27 questions, gpt-4.1 via OpenRouter, single seed:

```
default                          14/27 (52%)  179,452 tok  340.8s
adaptive_current_minrung3        14/27 (52%)  689,050 tok  396.0s   ← forced K=3 BoN, no tiebreaker
adaptive_e_minrung3              14/27 (52%)  575,849 tok  523.9s   ← Speculative Fanout
```

Per-domain breakdown was identical across all three configs (dabench 4/12,
math 5/5, longcot 0/5, easy 5/5).

**Conclusion:** Best-of-N — regardless of selector — adds **zero accuracy**
over a single-rollout baseline on this bench. Phase 4 (Speculative Fanout)
is therefore optimizing the cost of a config that gives zero accuracy uplift.

The SRLM paper's claimed gains live in a regime (long-context, weak validator)
that our current bench doesn't probe. Until we have that bench, we cannot
honestly market any of A / C / E.

A first practical demo (financial extraction, fanout vs forced K=3) showed
"28-36% savings" — but the comparison was a strawman. Adding a real `naive`
baseline (single `dspy.LM` call) made fanout look 10× more expensive and
3× slower for the same accuracy. The misleading demo was deleted before
shipping.

See `PHASE4-ship-decision.md` (in this dir) for the long version.

---

## What's on this branch

```
cf3b389 Phase 4b: wire early_exit_probe through bench harness
838d724 Phase 4b: Feature E (early-exit probe) opt-in default-OFF + 8 tests
fe2e315 Phase 4a: prefix-replay simulator + Feature E feasibility evidence
f60b4ea Phase 3: harden Feature C + ship-decision (post duck-review)
afd3d17 feat(srlm): add Feature C self-consistency cluster tiebreaker
35c9f4b feat(srlm): ship Feature A trace-length tiebreaker as opt-in default-off
40d513e fix(srlm): scope prefer_shorter_traces to rung-3 BoN; fix bench cost+observability
1c88589 Fix start_rung first-attempt path + drop unsupported LM params
a8b51a4 SRLM bench: domain validators + force_min_rung + minrung3 configs
803a7d8 Add LadderPolicy.start_rung for bench rung-isolation
ab6d63c Add Feature A: trace-length tiebreaker (Phase 2)         [also on feature/srlm-feat-a-tiebreaker]
dd3c22b Add SRLM bench harness                                    [also on feature/srlm-bench-harness]
```

All 135 tests pass on this branch.

---

## What is / isn't in main

**In main:**
- Engine consolidation (`engine='auto'` default, `default`/`dspy` aliases,
  v6/v7 deprecation) — `fa6b99c` PR #19
- adaptive engine `answer=None` fix — `f12a8f9`
- The base `bench/adaptive/run_bench.py` benchmark

**NOT in main (lives only on this parked branch):**
- The entire `bench/adaptive/_eval_lib.py` + `srlm_eval.py` SRLM bench harness
- Domain validators
- All three SRLM features (A, C, E)
- All bench harness bug fixes (drop_params, token-summing, observability)

> **Important:** We considered cherry-picking the harness bug fixes
> (`drop_params`, domain validators, token-summing) to `main`. We decided
> against it because the harness file they fix doesn't exist on `main`.
> Landing the fixes alone is impossible; landing them + the harness is
> "new infrastructure," not a bug fix. Duck reviewed and confirmed.

---

## Resume path

If reviving this:

1. Read `bench/adaptive/PHASE4-ship-decision.md` (honest "ship narrowly" doc).
2. Re-run the 3-way smoke:
   ```bash
   # uses your OPENROUTER_API_KEY
   python bench/adaptive/srlm_eval.py --config default --questions 27 --model openai/gpt-4.1
   python bench/adaptive/srlm_eval.py --config adaptive_current_minrung3 --questions 27 --model openai/gpt-4.1
   python bench/adaptive/srlm_eval.py --config adaptive_e_minrung3 --questions 27 --model openai/gpt-4.1
   python bench/adaptive/_three_way_smoke.py
   ```
3. Confirm the no-lift finding still holds (or doesn't) on whatever model
   you're using. If it still doesn't lift, **don't ship.**
4. To genuinely test best-of-N value, build a long-context / weak-validator
   bench — that's the SRLM paper's regime, not the smoke bench's.

---

## Most useful artifact this session produced

`bench/adaptive/_three_way_smoke.py` — a focused 3-way analyzer that loads
`{default, adaptive_current_minrung3, adaptive_e_minrung3}` per-question
result JSONs and prints honest pairwise deltas + per-domain tables.

This is the script that exposed best-of-N=no-op on our bench. Even if all
the SRLM feature code is eventually thrown out, this analyzer is reusable
for any future "is best-of-N actually helping me?" question.

(Note: `_three_way_smoke.py` is in `.gitignore` via the `_*` pattern. It's
force-added to this commit so it survives in repo history. The
`bench/adaptive/results/` dir it reads from is also gitignored — you'll
need to regenerate result JSONs locally per the resume path above.)
