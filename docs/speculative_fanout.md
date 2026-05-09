# Speculative Fanout (`early_exit_probe`) — operator's guide

> **Default: OFF. You probably don't need this. Read §1 before enabling.**

`AdaptiveRunner(early_exit_probe=True)` changes the rung-3 best-of-N
schedule from "launch K parallel rollouts, pick the best" to "launch 1
probe rollout; if it passes the validator, skip the K-1 suffix; if it
fails, run the K-1 suffix in parallel and pick the best from {probe,
suffix}." External label for this behavior is **Speculative Fanout**.

## 1. Honest disclosure (read this first)

We benched this against the 27Q smoke set in this repo (model
`openai/gpt-4.1`, single seed) with three configs:

```
config                            pass        tokens   elapsed (sum)
default (single rollout)          14/27 (52%)  179,452  340.8s
adaptive_current (forced K=3)     14/27 (52%)  689,050  396.0s
adaptive_e (Speculative Fanout)   14/27 (52%)  575,849  523.9s
```

Two facts you must internalize:

1. **Best-of-N adds zero accuracy** on this bench. Same 14/27 across all
   three configs. Per-domain it's identical too (dabench 4/12, math 5/5,
   easy 5/5, longcot_holdout 0/5).
2. **Speculative Fanout cannot improve accuracy.** It can at best match
   the best-of-N config it replaces. When best-of-N is a no-op, so is
   Speculative Fanout — except it costs 3.2× more tokens and 1.5× more
   wall time than just using `engine="default"`.

So enabling Speculative Fanout is only a reasonable thing to do if you
have **independent evidence** that best-of-N already lifts accuracy on
your workload. If you do not have that evidence, **do not enable this**.

## 2. How to decide if your workload lifts under best-of-N

Use the 3-way analyzer in this repo:

```bash
# 1. Run all three configs on your question set
python bench/adaptive/srlm_eval.py \
  --model openai/gpt-4.1 \
  --configs default,adaptive_current_minrung3,adaptive_e_minrung3 \
  --seeds 1 \
  --results-dir bench/adaptive/results/your_run

# 2. Read the honest 3-way comparison
python bench/adaptive/_three_way_smoke.py
```

Look at the **`default` vs `adaptive_current_minrung3`** delta:

- If accuracy lift is < 5pp on your workload → best-of-N isn't paying off.
  Stay on `engine="default"`. Don't enable Speculative Fanout.
- If accuracy lift is ≥ 5pp and statistically meaningful → best-of-N is
  earning its keep. Then look at **`adaptive_current_minrung3` vs
  `adaptive_e_minrung3`** — if Speculative Fanout matches accuracy at
  lower token cost on your workload, enable it.

## 3. How to enable

```python
from fabric_rlm.experimental.adaptive_runner import AdaptiveRunner

runner = AdaptiveRunner(
    # ... your normal kwargs ...
    parallel_rollouts=3,
    early_exit_probe=True,    # opt-in
    validator=my_validator,   # required — probe needs a pass/fail signal
)
```

Equivalent path through the public `RLM` surface:

```python
rlm = RLM(
    signature="question -> answer",
    lm=lm,
    engine="adaptive",
    adaptive=dict(
        validator=my_validator,
        parallel_rollouts=3,
        early_exit_probe=True,
    ),
)
```

## 4. Contract

What `early_exit_probe=True` **does**:
- Replaces parallel best-of-N with probe-then-fanout when
  `parallel_rollouts > 1`.
- Skips the suffix on validator-pass; runs it on validator-fail.
- Emits `stop_reason="early-exit: probe passed"` for downstream
  observability.
- Is byte-identical to today when `early_exit_probe=False` (the default).

What it **does not** do:
- Does not improve accuracy. Best case is identical to all-parallel
  best-of-N; worst case is identical.
- Does not preserve per-question outcomes deterministically across runs
  (LM stochasticity at temperature > 0 produces ~10% per-question churn
  in any K>1 config — same as today's baseline).
- Does not improve latency. The failure path is `probe + suffix`, near-
  sequential, slower than today's all-parallel best-of-N.
- Does not justify enabling `parallel_rollouts > 1` on workloads where a
  single rollout already saturates accuracy.

## 5. When to use it

| If you... | Then... |
|---|---|
| Run `engine="default"` and have the accuracy you need | Don't enable Speculative Fanout. Don't enable best-of-N. |
| Run `parallel_rollouts > 1` and have **measured** that best-of-N lifts accuracy on your workload | `early_exit_probe=True` reclaims cost on probe-passes. Measure both configs first. |
| Run `parallel_rollouts > 1` for variance reduction / reproducibility | Speculative Fanout *increases* per-question variance (suffix is skipped on probe-pass). Don't enable. |
| Are exploring whether best-of-N pays off | Use the 3-way analyzer first. Don't enable best-of-N speculatively. |

## 6. Why it exists at all

Three regimes were measured during Phase 4:

- **Counterfactual prefix-replay** (`bench/adaptive/p4_prefix_replay_findings.md`):
  on 196 captured rung-3 rollouts, probe-only winners agreed with full-
  fanout winners on 196/196. Probe fired on ~35% of rollouts. *Proves
  Speculative Fanout is safe.* Does not prove best-of-N is useful.
- **Live smoke bench** (`bench/adaptive/results/srlm_eval_p4e_smoke/`):
  see §1. *Confirms safety. Refutes the value of best-of-N on this bench.*
- **Tests** (`tests/test_adaptive_early_exit.py`): 8 unit tests covering
  probe-pass, probe-fail-suffix, validator-raises, K=1 degeneracy,
  metadata `stop_reason` contract.

The runtime change is small, surgical, default-OFF. The artifact set
shipped includes the 3-way analyzer so users can answer the "is this
even worth enabling?" question themselves.

## 7. Pointers

- Implementation: `fabric_rlm/experimental/adaptive_runner.py`
  (`_run_rollouts_with_probe`, line ~395+)
- Tests: `tests/test_adaptive_early_exit.py`
- Replay study: `bench/adaptive/_prefix_replay.py` +
  `bench/adaptive/p4_prefix_replay_findings.md`
- Smoke bench: `bench/adaptive/results/srlm_eval_p4e_smoke/`
- 3-way analyzer: `bench/adaptive/_three_way_smoke.py`
- Ship decision (the full story): `bench/adaptive/PHASE4-ship-decision.md`
