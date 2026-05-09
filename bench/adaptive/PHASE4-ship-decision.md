# Phase 4 — Speculative Fanout (Feature E) — narrow ship decision

> **Reading order**: §1 (the honest TL;DR) → §2 (what changed since the
> first draft) → §3 (the hard finding) → §6 (what we ship and what we don't).

## 1. TL;DR (honest)

We implemented **Feature E — early-exit probe** (`AdaptiveRunner(early_exit_probe=True)`,
external label "Speculative Fanout"). The runtime change is sound and well-tested.

**However:** our pre-merge 3-way smoke bench shows that on the workloads we
have locally, **best-of-N (regardless of selector) adds zero accuracy
over a single-rollout `engine="default"` run.** Optimising the cost of a
config that doesn't lift accuracy is a microoptimization to a path no
Fabric customer should run on this workload in the first place.

We are shipping Feature E **narrow**: opt-in, default-OFF, with a ship doc
that explicitly says "this only matters if your workload genuinely lifts
under best-of-N — and ours, on this bench, does not." No marketing demo.
The artifact set is the runtime, the tests, the replay simulator, and the
3-way analyzer that produced this honest finding.

## 2. What changed since the first draft (and why)

The first draft (now deleted) framed Phase 4 as:

- "Speculative Fanout saves 16-36% tokens vs baseline with no accuracy loss"
- A practical financial-extraction demo with "29% savings, 8/8 correct"

Pre-merge, the user asked the right adversarial question: **"What is the
baseline? Could naive single-shot have solved it?"** The answer was yes:
naive solved 8/8 financial-extraction tasks at 1/10 the cost of the
"baseline" we'd been comparing against. The baseline was a forced
`start_rung=3, parallel_rollouts=3` config — a strawman.

Running the same `engine="default"` config on our 27Q bench smoke set
(same model, same seed, same questions used in the original ship-doc
evidence) revealed the broader problem:

| config                     | pass        | tokens   | elapsed |
|----------------------------|-------------|----------|---------|
| default (single rollout)   | 14/27 (52%) | 179,452  | 340.8s  |
| adaptive_current (K=3)     | 14/27 (52%) | 689,050  | 396.0s  |
| adaptive_e (Speculative Fanout) | 14/27 (52%) | 575,849  | 523.9s  |

**Best-of-N adds 0 accuracy on this bench.** Per-domain: dabench 4/12
across all three; math 5/5 across all three; longcot 0/5 across all three
(broken grader, known); easy_calibration 5/5 across all three. 8/27
questions flip per-question on dabench, but the aggregate is locked.

The "16.4% Speculative Fanout vs forced-K=3" savings number is real,
*but* the underlying K=3 config it optimizes is itself dominated by the
naive single-rollout config on every dimension that matters: accuracy,
tokens (3.8× higher), and wall time (16% higher). So the headline of any
honest pre-merge ship doc has to be: **best-of-N is a no-op on our bench
in the first place.**

Phase 1-3 features (A and C) hit the same wall: 0 decision-flips against
this bench because the validator IS the grader (causally there is no
selection signal best-of-N can exploit).

## 3. The hard finding

Three independent regimes triangulate the same conclusion:

**3.1 Counterfactual prefix-replay** (offline, 196 captured rung-3
rollouts): probe-only winners agree with full-fanout winners on 196/196.
*This proves Speculative Fanout is safe — it does not change outcomes.*
It does not prove best-of-N adds value, because the same finding is
consistent with "best-of-N is a no-op."

**3.2 Live smoke bench** (27Q, gpt-4.1, single seed): naive 14/27 ==
best-of-N 14/27 == Speculative Fanout 14/27. Token cost rank: naive ≪
Speculative Fanout < best-of-N. *Best-of-N adds 0 accuracy here.*

**3.3 Practical demo** (8 financial-extraction tasks): naive 8/8 grader-
correct at ~270 tokens/Q; baseline-K=3 8/8 at ~2,650 tokens/Q;
Speculative Fanout 8/8 at ~2,775 tokens/Q. *Best-of-N adds 0 accuracy
here too — and Speculative Fanout is more expensive than the K=3 baseline
on this trivially-easy workload because both must launch the probe and
both saturate validator immediately.*

The single regime where the SRLM/best-of-N premise was paper-claimed to
help — long-context with weak validators — does not exist as a working
bench harness in this repo, and we explicitly chose not to build it.

## 4. What we shipped (the code is good)

| Component | File | Status |
|---|---|---|
| Prefix-replay simulator | `bench/adaptive/_prefix_replay.py` | ✅ commit `b9804c7` |
| Replay tests (10) | `tests/test_prefix_replay.py` | ✅ |
| Replay findings | `bench/adaptive/p4_prefix_replay_findings.md` | ✅ |
| Runtime change | `fabric_rlm/experimental/adaptive_runner.py` (`early_exit_probe` flag, `_run_rollouts_with_probe`, distinct `stop_reason="early-exit: probe passed"`) | ✅ commit `838d724` |
| Runtime tests (8) | `tests/test_adaptive_early_exit.py` | ✅ |
| Bench harness wiring | `bench/adaptive/_eval_lib.py` (`adaptive_e_minrung3` config), `fabric_rlm/runtime.py` | ✅ commit `cf3b389` |
| 3-way smoke analyzer | `bench/adaptive/_three_way_smoke.py` | ✅ |
| Smoke results (3 configs × 27Q) | `bench/adaptive/results/srlm_eval_p4e_smoke/` | ✅ |

Targeted runtime tests: 8/8 pass. Full suite previously at 233/233.

## 5. The contract Speculative Fanout actually offers

Read carefully — this is the honest contract, not the first-draft contract:

What `early_exit_probe=True` **does**:
- On configs that already use `parallel_rollouts > 1`, replaces "launch all
  N in parallel" with "launch 1 probe; on validator pass, skip the
  remaining N-1; on validator fail, run the remaining N-1 in parallel."
- Emits `stop_reason="early-exit: probe passed"` when the probe path fires.
- Is byte-identical to today when `early_exit_probe=False` (the default).
- Saves N-1 candidates' worth of compute on every probe-pass.

What it **does not** do:
- Does not improve accuracy. It cannot — the worst case is the existing
  best-of-N, and the best case is identical (winner unchanged).
- Does not justify enabling `parallel_rollouts > 1` on workloads where
  K=1 already saturates accuracy. On *this* bench, that's all of them.
- Does not preserve per-question outcomes deterministically across runs
  (LM stochasticity at temperature > 0 produces ~10% per-question churn
  in any K>1 config — same as today's baseline).
- Does not improve latency. Failure path is `probe + suffix` (near-
  sequential), worse than today's all-parallel best-of-N.

## 6. Who this is actually for

Honestly: **probably no one on our current bench**. Specifically:

| If you... | Then... |
|---|---|
| Run `engine="default"` (single rollout) and have the accuracy you need | Don't enable Speculative Fanout. Don't enable best-of-N. |
| Run `parallel_rollouts > 1` because best-of-N **measurably** lifts accuracy on your workload (verify with naive K=1 control!) | `early_exit_probe=True` reclaims cost on probe-passes. Worth measuring. |
| Run `parallel_rollouts > 1` for variance reduction or downstream reproducibility | Speculative Fanout will increase per-question variance (suffix is skipped on probe-pass). Don't enable. |
| Are still exploring whether your workload lifts under best-of-N | Use the 3-way analyzer (`_three_way_smoke.py`) to decide *first*. Don't enable best-of-N speculatively. |

The default-OFF behavior protects everyone who isn't in the second row.

## 7. Naming

- **Internal / code / flag**: `early_exit_probe`. Precise, code-search-friendly.
- **Marketing / external label** (use sparingly): "Speculative Fanout."
  Riffs on speculative decoding; describes the probe-then-fanout schedule.

We are deliberately NOT marketing this as a headline feature. It is a
cost knob for users who already opted into best-of-N and have measured
that best-of-N actually pays off for them.

## 8. What we did NOT prove (and were honest about, second time around)

- **SRLM (Apple, 2024) selectors A/C: 0 accuracy lift on our bench.** Not
  refuting the paper — the regime where SRLM selectors should help (long
  context, weak validator, where multiple candidates can produce
  measurably different downstream outcomes) is not what our validator-as-
  grader bench measures. We just don't have evidence either way.
- **Best-of-N as a class: 0 accuracy lift on our bench.** Same caveat —
  this is the bench, not a universal claim about best-of-N.
- **Speculative Fanout as accuracy-positive:** it cannot be. It is
  accuracy-neutral by construction. We should never have implied otherwise.

## 9. What's next (parked, requires explicit re-approval)

- Long-context bench harness (RULER / ∞Bench / DABench-with-LLM-judge).
  This is the gating prerequisite to validating Features B and D, and to
  honestly answering "where does best-of-N actually help?" On the no-go
  list because the user explicitly chose to stop after E.
- Real `longcot` / `ssb` graders (currently substring stubs grading 0%).
- Sentinel-detection in `_canonicalize_answer` (small).
- Speculative Fanout sensitivity study on K=4, 5, 8.

These are NOT shipped, NOT promised, and require explicit user re-
approval before resuming.

## 10. Verdict

**Ship Feature E narrowly.** The runtime is sound, tests are solid, the
3-way analyzer is the most useful artifact this phase produced. Default-
OFF means we cause no harm to existing users.

The single most important honest sentence in this whole document, and it
should be on the README too:

> **Best-of-N (and therefore Speculative Fanout) does not lift accuracy
> on the bench in this repo. Enable Speculative Fanout only if you have
> independently verified that best-of-N pays off on your workload.**
