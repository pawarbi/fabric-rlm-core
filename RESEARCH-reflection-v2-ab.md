# Reflection v2 — A/B Plan & Decision Rule

## Context

Dev11 trace analysis (60 reflection turns / 77 attempts on 25-Q hard-CS bench)
showed the shipped reflection prompt is net-harmful:

| Branch    | Count | Pass | Pass-rate |
|-----------|-------|------|-----------|
| Confirmed | 17    | 3    | 18%       |
| Revised   | 43    | 6    | 14%       |

Revising hurt by 4 pp on the same questions, burned ~100–200K reasoning
tokens, and produced a concrete corruption case (`Backprop_hard_13`: model
rewrote a working `SUBMIT(answer="\n".join(out_lines))` as a hand-typed
inline string and never recovered).

Root cause was the prompt itself, not the concept: 5 of 6 steps push toward
revision, framing is adversarial ("ATTACK your own answer"), and the
approval signal (`REFLECTION_OK`) is buried at step 6.

## v2 design (committed, branch `experiment/reflection-v2`)

`build_reflection_prompt` rewritten:

- **Default direction is APPROVE.** The validator already accepted this
  SUBMIT — the reflection turn is a narrow gate, not a re-solver.
- **Two checks only**, both demonstrably useful in dev11:
  - (A) placeholder / clarification-request detection
  - (B) count mismatch on enumerated sub-questions (Q1..Qn)
- **Minimal-edit constraint** on revisions, targeting the dev11
  corruption mode: "make the SMALLEST possible change, reuse existing
  variables, do NOT rewrite a generated programmatic answer as a
  hand-typed inline string."
- **Executable approval signal**: `print("REFLECTION_OK")` in a code
  block, so the runtime's exec doesn't `NameError` on a bare token.
- Drops the adversarial "ATTACK" framing, the invariant enumeration step,
  and the "write Python that asserts each invariant" re-derivation step.

System-prompt teaser (line 26) updated to match the v2 contract.

## A/B design — 3 arms at medium + 1 sanity arm at high

Per rubber-duck critique, the critical comparison is **B vs C**, not B vs A.
v2 could beat v1 while still being worse than no-reflection.

**Effort tier choice — medium, not high.** Reflection's value proposition is
"catch mistakes the model would otherwise ship." At `gpt-5` high effort the
model already gets most hard-CS questions right, so there are very few wrong
SUBMITs for reflection to genuinely catch — over-revision noise dominates
the signal. Dev11 (60 reflection turns, 72% revise rate, no lift) is partly
a high-effort artifact: most revisions were unnecessary because the original
was already correct. Medium effort produces a lower base pass rate, more
genuine errors, and a cleaner read on whether (A) placeholder and (B) count
checks fire on real failures rather than phantom ones.

### Primary A/B (medium)

| Arm | Condition                     | Implementation                                         |
|-----|-------------------------------|--------------------------------------------------------|
| A   | v1 prompt (current shipped)   | revert `prompts.py` to pre-commit-0b45c09              |
| B   | v2 prompt                     | branch `experiment/reflection-v2` HEAD                 |
| C   | reflection OFF                | `RLM(..., enable_reflection=False)`                    |

All arms:
- Same wheel, same dataset (`bench/adaptive/longcot_cs_hard_holdout25.jsonl`),
  same `gpt-5` w/ `reasoning_effort='medium'`, same seed/run-id shape.
- Same trajectory capture (`FABRIC_RLM_CAPTURE_TURNS=1`).
- 25 questions × 3 arms = 75 runs.

### High-effort sanity arm (cheap add-on)

| Arm | Condition                     | Purpose                                                |
|-----|-------------------------------|--------------------------------------------------------|
| B-hi| v2 prompt @ high effort       | Confirm v2 doesn't worsen what strong-model case wins  |

- 25 runs.
- Compared against the dev11 v1-at-high baseline already on disk
  (`comparison_5way_local/accuracy-lift-dev11-20260504-023212/`).

`enable_reflection=False` already exists in `RLM.__init__`, so arm C needs
no code change — just a notebook flag.

## Pre-registered metrics (track all, decide on all)

| Metric                   | Definition                                                   |
|--------------------------|--------------------------------------------------------------|
| `pass@final`             | exact-match against gold (primary)                           |
| `revise_rate`            | % reflection turns that re-SUBMIT (vs print REFLECTION_OK)   |
| `harmful_revisions`      | original SUBMIT would have passed grader, revised one fails  |
| `beneficial_revisions`   | original SUBMIT would have failed, revised one passes        |
| `placeholder_catches`    | (A) check fired, original SUBMIT was a placeholder           |
| `count_catches`          | (B) check fired, original SUBMIT had wrong sub-question count|
| `reflection_tokens`      | total reasoning + completion tokens in reflection turns      |
| `total_tokens`           | overall                                                      |
| `cost_per_pass`          | `total_tokens / pass@final`                                  |

Harmful/beneficial require grading both the pre- and post-reflection
payload — already capturable from existing trace JSONs (turn before
reflection contains the SUBMIT, turn after contains the revised one).

## Decision rule (pre-registered, conservative)

Let `pass_X` = pass@final for arm X (medium-effort primary unless noted).

At medium effort the base pass rate is lower → more headroom → if reflection
adds value at all, it should show up here. The bar for "reflection earns
its keep" is therefore *stricter* than at high.

**Ship v2 (default `enable_reflection=True` with v2 prompt) iff ALL of:**

1. `pass_B > pass_C` (strictly beats off — at medium, ties go to off)
2. `pass_B >= pass_A` (no regression vs current shipped)
3. `harmful_revisions_B < harmful_revisions_A` (corruption mode reduced)
4. `placeholder_catches_B + count_catches_B >= 1` (the gate earned its keep)
5. `pass_B-hi >= pass_A-hi - 1` (high-effort sanity: v2 didn't break the
   strong-model case beyond a 1-question slack)

**Flip default OFF (`enable_reflection=False`) iff:**

- `pass_C >= pass_B` AND `pass_C >= pass_A` (off is at least as good as any
  reflection variant — even at the tier most favorable to reflection)

In the OFF case, mirror the decompose pattern in
`fabric_rlm/experimental/effort_ladder_policy.py:99-114` — keep the code
path, flip default, emit `DeprecationWarning` with dev11/AB evidence.

**Mixed result (v2 helps on some, hurts on others):**
Consider gated reflection — only run when `verifier_repair_history` is
non-empty, or placeholder regex pre-fires, or count heuristic pre-fires.
This is a follow-up branch, not part of the current scope.

## Execution

Local runner pattern: clone `scripts/run_comparison_5way_dspy_local.py`
into `scripts/run_reflection_ab_3arm.py`, sweep
`enable_reflection={True+v1, True+v2, False}`. v1 requires a stash of
the pre-commit prompts.py.

Fabric runner pattern: clone
`notebooks/comparison_5way_I_accuracy_lift_dev11.ipynb` into
`comparison_reflection_AB_3arm.ipynb`, three cells (one per arm), shared
preamble.

Cost estimate: 75 medium-effort runs + 25 high-effort sanity runs.
Medium runs are roughly 1/3 the reasoning-token cost of high, so total $
should land at ~50-60% of the dev11 run despite running more arms.
Wall time ~3-5 hours.

## Pre-existing test failures (unrelated, do not block A/B)

- `tests/test_halve_max_iter_param.py::test_no_halving_when_disabled`
- `tests/test_subprocess_interpreter.py::test_start_is_idempotent`
- `tests/test_verifier_wrapper.py::test_output_validator_rejects_then_succeeds_on_retry`

All Windows subprocess timeouts in `interpreter.py`. Verified unrelated
to reflection via `git stash` earlier this session.

## Status

- [x] v2 prompt designed + committed (0b45c09 on `experiment/reflection-v2`)
- [x] Tests updated + new v2-contract tests added (9/9 pass; full suite green)
- [x] A/B plan documented + decision rule pre-registered
- [x] Build 3-arm runner notebook (medium primary + high sanity)
- [ ] Execute A/B (likely user-side on Fabric due to cost)
- [ ] Analyze with `analyze_reflection*.py` (extend for 3-arm)
- [ ] Apply decision rule → merge v2 / flip default OFF / consider gating
- [ ] Commit + merge or revert
