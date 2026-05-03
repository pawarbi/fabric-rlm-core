# SPEC: Decompose-then-synthesize rung (rung 5)

Status: **Phase 1 (Specify)** per `.github/skills/spec-driven-development/SKILL.md`.
No code until this spec is reviewed and approved.

Branch: `feat/decompose-rung`
Parent research: `bench/adaptive/RESEARCH-lambda-rlm-comparison.md` §3.3
Borrowed from: λ-RLM `MULTI_HOP` plan (`SPLIT_δ → MAP(PEEK) → FILTER → MAP(M) → M`)

> **Speculative.** This is a research bet. Validate cheaply and be willing to
> shelve if the model can't emit useful decompositions on the prior-fail set.

## Assumptions I'm making

1. The decompose rung is **rung 5**, added on top of the existing 5-rung effort
   ladder (`minimal, low, medium, high, high+parallel`). Total cost weight at
   rung 5 ≈ rung 4 × 2 (two LLM phases: decomposition + synthesis).
2. The mechanic is a **two-phase forced workflow**, not a free-form prompt:
   - Phase A (decompose): single LLM call asking the model to emit ≤4 named
     sub-problems as structured output.
   - Phase B (solve sub-problems): each sub-problem solved by `sub_lm` at
     `medium` effort, in parallel.
   - Phase C (synthesize): single LLM call that gets all sub-answers + the
     original question, must produce SUBMIT.
3. The validator gates rung 5 the same way it gates every other rung — a wrong
   final answer escalates back into the bandit's posterior update.
4. **Depends on `feat/combinators-skill`** being merged first — Phase B uses
   `map_combinator(solve, sub_problems)` to express the parallel solve cleanly.
   If combinators aren't merged, Phase B falls back to a hand-coded
   `concurrent.futures.ThreadPoolExecutor`.
5. Validation set is **the prior-fail subset** of LongCoT CS-hard:
   `Backprop_hard_1`, `VLIW_hard_1`, `DistMem_hard_*` — questions where
   even rung 4 (high+parallel) failed. If rung 5 doesn't move any of these
   from fail → pass on N=5 trials, the rung is shelved.
6. → **Correct any of these now or I proceed.**

## Objective

Add a sixth rung that **structurally forces decomposition** before synthesis,
targeting questions where pure-effort scaling has hit a model-capability
ceiling. The hypothesis (untested): some hard puzzles fail not because the
model can't reason deeply but because it can't see the whole problem at once
and tries to solve it as one chunk.

**Success looks like**: rung 5 moves ≥1 of the 3 prior-fail templates
(`Backprop_hard_1`, `VLIW_hard_1`, one `DistMem_hard_*`) from 0/5 → ≥2/5 on a
5-trial micro-bench at the rung-5 cost ceiling. Bandit converges to picking
rung 5 over rung 4 on those templates within 10 attempts.

Failure looks like: rung 5 ≤ rung 4 on every prior-fail template, OR cost
overruns make it dominated by just running rung 4 twice. **Shelve in either
case** — write up the negative result in `REPORT-decompose-rung.md` so we
don't re-litigate it.

## Tech stack

- New module: `fabric_rlm/experimental/decompose_rung.py`
- Modifies: `fabric_rlm/experimental/effort_ladder_policy.py`
  (`_EFFORT_LADDER` extends to length 6; `_build_config` handles rung 5)
- Modifies: `fabric_rlm/experimental/bandit_policy.py`
  (`_EFFORT_RUNG_COST` gets a 6th entry ≈ 150)
- Uses `dspy.Signature` for the decompose call, `sub_lm` for the parallel
  solve, `dspy.Predict` for synthesis
- No new runtime deps

## Commands

```
Build:        python -m build
Test:         python -m pytest tests/test_decompose_rung.py
Lint:         ruff check fabric_rlm/experimental/decompose_rung.py
Micro-bench:  python bench/adaptive/run_decompose_microbench.py \
                --templates Backprop_hard_1,VLIW_hard_1,DistMem_hard_2 \
                --trials 5
Full-bench:   python scripts/build_comparison_5way_notebook.py --include-h
              # (Strategy H = F + decompose rung; only if micro-bench passes)
```

## Project structure

```
fabric_rlm/experimental/
    decompose_rung.py             ← new: DecomposeSig, decompose_then_synthesize()
    effort_ladder_policy.py       ← modified: rung 5 case
    bandit_policy.py              ← modified: _EFFORT_RUNG_COST length 6
bench/adaptive/
    run_decompose_microbench.py   ← new: 5-trial gate before full-bench
    REPORT-decompose-rung.md      ← new: validation OR negative-result report
tests/
    test_decompose_rung.py        ← new: phase A/B/C unit tests + integration
```

## Code style

```python
# fabric_rlm/experimental/decompose_rung.py
from __future__ import annotations
import dspy
from typing import Sequence


class _DecomposeSig(dspy.Signature):
    """Break the problem into 2-4 self-contained sub-problems whose solutions
    can be combined to answer the original. Each sub-problem must be solvable
    independently. Output as a numbered list."""
    question: str = dspy.InputField()
    sub_problems: list[str] = dspy.OutputField(desc="2-4 sub-problems")


class _SynthesizeSig(dspy.Signature):
    """Given the original question and the answers to its sub-problems,
    produce the final answer."""
    question: str = dspy.InputField()
    sub_answers: list[str] = dspy.InputField()
    final_answer: str = dspy.OutputField()


def decompose_then_synthesize(
    question: str, lm: dspy.LM, sub_lm: dspy.LM, *, max_subs: int = 4,
) -> str:
    """Phase A → B → C. Pure function of (question, lm, sub_lm)."""
    ...
```

Conventions:
- Two-phase structure is explicit; no recursive decompose (depth = 1 only).
- Synthesis call gets ALL sub-answers, not a streaming concat.
- Every phase logs to the existing `_RUN_LOG` so traces show decomposition.
- Failure of any sub-problem solve ⇒ rung 5 reports rung-failure to the
  bandit (no partial-credit semantics).

## Testing strategy

- **Unit** — Phase A: stub `lm` returns a fixed sub-problem list, assert parse;
  Phase B: stub `sub_lm.copy()` returns canned answers, assert parallel
  invocation count; Phase C: stub returns a fixed answer, assert pass-through.
- **Integration** — full `decompose_then_synthesize` against a mocked dspy
  pipeline, assert exactly 1 + N + 1 LLM calls.
- **Bandit integration** — `EffortBanditPolicy` with rung 5 enabled converges
  to rung 5 on a synthetic template where rung 4 returns wrong and rung 5
  returns right (Beta posteriors validate).
- **Micro-bench gate** (`run_decompose_microbench.py`) — 5 trials × 3 templates
  = 15 runs. Rung-5-only (no bandit). **If 0 templates show ≥2/5, STOP.** Write
  the negative result and shelve.
- **Full bench** — Strategy H = F + rung 5. Only run after micro-bench gate
  passes. Compare H vs F on the full 25-Q holdout.

Coverage target: ≥85% on `decompose_rung.py`.

## Boundaries

- **Always**: gate full-bench on micro-bench result; preserve backwards
  compat for existing 5-rung serialized BanditState (length-5 → pad with
  zeros for rung 5 entry, treat as never-tried); lint + test before commit.
- **Ask first**: making rung 5 the default top rung (it isn't until full-bench
  validates); changing the decompose prompt template post-validation;
  recursive decompose (depth > 1).
- **Never**: enable rung 5 by default in 0.x releases; ship without
  micro-bench result either way; let the synthesis step bypass the validator.

## Success criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | Rung 5 callable in isolation given (lm, sub_lm) | Unit + integration tests |
| 2 | Bandit picks rung 5 over rung 4 when posterior favours it | Synthetic-template bandit test |
| 3 | Existing 5-rung serialized state loads under length-6 schema | Persistence regression test |
| 4 | Micro-bench: ≥1 of 3 prior-fail templates moves 0/5 → ≥2/5 | `REPORT-decompose-rung.md` micro-bench table |
| 5 | (If #4 passes) Strategy H ≥ F + 1 point on 25-Q holdout | Full bench table in same report |
| 6 | (If #4 fails) Negative result documented and rung is unregistered | Same report; revert ladder length |

## Open questions

1. Should decomposition use `sub_lm` (cheap) or `lm` (expensive)? λ-RLM uses
   the expensive model for planning, cheap for leaves. We probably do too —
   the decomposition is the hardest call.
2. What if the model emits 1 sub-problem (refuses to decompose)? Treat as
   degenerate, fall back to rung 4 result, report rung-5 as no-op.
3. Should the synthesis step have access to the model's own scratch / chain
   of thought from each sub-solve? λ-RLM doesn't; we probably shouldn't
   either — keeps the synthesis call's context bounded.
4. **Coupling with `feat/task-type-classifier`**: when the classifier predicts
   `MULTI_HOP`, should it seed rung 5's prior directly? Yes — natural fit.
   Note this in the classifier SPEC's open-questions.
5. Cost ceiling: rung 5 ≈ 2× rung 4. If actual measured cost is >3× rung 4
   (decomposition + 4 sub-solves at medium > planned), reconsider the rung
   cost weight before declaring "dominated".

## Phases (after spec approval)

- **Phase 2 (Plan)**: dependency on `feat/combinators-skill` — confirm or
  decouple; sub_lm parallelization details; validator coupling.
- **Phase 3 (Tasks)**: decompose_rung impl → effort ladder extension →
  bandit cost vector → micro-bench harness → micro-bench run → gate.
- **Phase 4 (Implement, conditional on micro-bench)**: TDD per
  `.github/skills/test-driven-development/SKILL.md`. **STOP at micro-bench
  failure and write the negative-result report.**
