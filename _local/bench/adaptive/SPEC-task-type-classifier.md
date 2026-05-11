# SPEC: Task-type classifier seeds the bandit prior

Status: **Phase 1 (Specify)** per `.github/skills/spec-driven-development/SKILL.md`.
No code until this spec is reviewed and approved.

Branch: `feat/task-type-classifier`
Parent research: `bench/adaptive/RESEARCH-lambda-rlm-comparison.md` §3.2
Borrowed from: λ-RLM (`nktkt/lambda-rlm`) up-front task-type detection

## Assumptions I'm making

1. The classifier runs **once per question** before the first `BanditPolicy.next()`
   call. Its output seeds the per-template `BanditState.posteriors` with informative
   `Beta(α, β)` priors instead of the current uniform `Beta(1, 1)`.
2. Classifier output is one of a fixed enum:
   `{LOOKUP, SEARCH, AGGREGATE, PAIRWISE, MULTI_HOP, CS_PUZZLE, CODE_GEN, UNKNOWN}`.
3. Classifier is implemented as a `dspy.Predict` against the **outer driver `lm`**
   (whatever the user passed to `RLM(...)`), with `reasoning_effort="minimal"`.
4. If the classifier returns `UNKNOWN` or fails, the bandit warms up with the
   current uniform prior — no behaviour change.
5. Prior seeding is a **one-shot nudge**, not a hard gate: Thompson-sampling can
   and will overwrite a wrong prior within ~3 attempts on the same template.
6. Validation benchmark is the existing **5-way comparison** generator —
   add a Strategy G = F + classifier and compare against F (6/25 baseline).
7. → **Correct any of these now or I proceed.**

## Objective

Replace the bandit's cold-start `Beta(1, 1)` priors with informative priors
derived from a one-call task-type classifier. Today every (model, template)
combination wastes ~3 attempts on rung 0 even when the question is obviously
hard. A 1-call classifier ($0.001) that shaves ≥1 escalation off the average
question pays for itself many times over.

**Success looks like**: Strategy G beats Strategy F by ≥2 points on the
LongCoT CS-hard 25-question holdout, at equal or lower total cost. Same wall
time within 10%.

Failure looks like: G ≤ F at higher cost (classifier overhead), or G regresses
because mis-classifications poison the prior more than they help.

## Tech stack

- `dspy.Predict` for the classifier signature (matches existing bench patterns)
- New module: `fabric_rlm/experimental/task_classifier.py`
- Modifies: `fabric_rlm/experimental/bandit_policy.py` (`BanditState` constructor
  accepts `prior_overrides: dict[str, tuple[float, float]] | None`)
- No new runtime deps

## Commands

```
Build:   python -m build
Test:    python -m pytest tests/test_task_classifier.py tests/test_bandit_policy.py
Lint:    ruff check fabric_rlm/experimental/task_classifier.py
Bench-G: python scripts/build_comparison_5way_notebook.py --include-g
Bench-F: python scripts/build_comparison_5way_notebook.py     # baseline
```

## Project structure

```
fabric_rlm/experimental/
    task_classifier.py            ← new: TaskClass enum + classify(question, lm) → TaskClass
    bandit_policy.py              ← modified: BanditState accepts prior_overrides
scripts/
    build_comparison_5way_notebook.py  ← modified: add Strategy G
tests/
    test_task_classifier.py       ← new
    test_bandit_prior_seeding.py  ← new — bandit picks rung 2 first when prior says HARD
bench/adaptive/
    REPORT-task-classifier.md     ← new: Strategy G vs F validation
```

## Code style

```python
# fabric_rlm/experimental/task_classifier.py
from __future__ import annotations
from enum import Enum
from typing import Optional
import dspy


class TaskClass(str, Enum):
    LOOKUP = "lookup"          # → seed rung 0 strongly
    SEARCH = "search"          # → favour SPLIT+FILTER plan; seed rung 0
    AGGREGATE = "aggregate"
    PAIRWISE = "pairwise"
    MULTI_HOP = "multi_hop"    # → seed rung 2; 2x max_turns
    CS_PUZZLE = "cs_puzzle"    # → seed rung 3 directly
    CODE_GEN = "code_gen"
    UNKNOWN = "unknown"        # → no prior override


class _ClassifySig(dspy.Signature):
    """Classify a question into one of {lookup, search, aggregate, pairwise,
    multi_hop, cs_puzzle, code_gen, unknown}. Pick the single best fit."""
    question: str = dspy.InputField()
    task_class: str = dspy.OutputField()


def classify(question: str, lm: dspy.LM) -> TaskClass:
    """One-shot classifier. Always returns; falls back to UNKNOWN on error."""
    ...
```

Conventions:
- Total functions: never raise; on any failure return `TaskClass.UNKNOWN`.
- Classifier prompt fits in ≤100 tokens.
- Prior overrides are pure data — no behavioural changes to BanditPolicy
  beyond accepting them.

## Testing strategy

- **Unit** (`tests/test_task_classifier.py`) — mock `dspy.LM`, assert each
  `TaskClass` is correctly parsed from sample model outputs; assert UNKNOWN
  fallback on parse failure / exception.
- **Unit** (`tests/test_bandit_prior_seeding.py`) — construct `BanditState`
  with `prior_overrides={"rung_3": (4.0, 1.0)}` and assert Thompson-sampling
  picks rung 3 with high probability on the first draw.
- **Integration** — add a fast smoke test that an `RLM` with classifier-seeded
  bandit makes one extra outer LLM call (the classifier) per question.
- **Bench** — Strategy G run on the 25-Q holdout, compared against F (6/25).
  Report includes per-template classification labels, per-question cost delta,
  and confusion matrix on a hand-labelled sample.

Coverage target: ≥90% on `task_classifier.py` and the new `BanditState` branch.

## Boundaries

- **Always**: keep classifier opt-in (default off so existing benchmarks don't
  shift); preserve uniform prior fallback path; lint + test before commit.
- **Ask first**: changing the `TaskClass` enum after first publication;
  changing `BanditState` serialization format (we persist to `state.json`);
  any change to `_RUNG_COST` or rung definitions.
- **Never**: hard-route based on classifier output (it seeds priors only);
  cache classifier results across runs (it's per-question state); ship without
  the Strategy G validation report.

## Success criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | `classify()` returns each enum value on appropriate inputs | Unit tests with stubbed `dspy.LM` |
| 2 | UNKNOWN fallback on any error | Unit test with raising mock |
| 3 | `BanditState(prior_overrides=...)` reproducibly picks the seeded rung | Statistical test: ≥80% rung-3 picks on first 10 draws |
| 4 | Strategy G ≥ F + 2 points on 25-Q holdout | `REPORT-task-classifier.md` table |
| 5 | Total cost(G) ≤ total cost(F) × 1.10 | Cost column in report |
| 6 | Bandit serialisation round-trip preserved | Existing `test_bandit_persistence.py` still passes |

## Open questions

1. Should the classifier output be persisted alongside `BanditState` so we can
   audit "what did the classifier think this question was?" in retrospectives?
   Probably yes — small, useful for debugging mis-classifications.
2. What's the right `α + β` mass for the seeded prior? Too small (1.5, 0.5) and
   the bandit overrides immediately; too large (10, 1) and it's a hard gate
   masquerading as a prior. Plan: start with `(α=4, β=1)` for confident
   classes, `(α=2, β=1)` for soft ones, tune in validation.
3. Does the classifier need a regex / keyword first-pass before the LLM call?
   Cuts cost on obvious cases. Probably yes once we have data.
4. How does this interact with `feat/decompose-rung`? The `MULTI_HOP` class
   should seed the decompose rung directly when that branch lands. Note in
   that SPEC's dependency section.

## Phases (after spec approval)

- **Phase 2 (Plan)**: dependency order; serialisation impact; classifier
  prompt iteration loop.
- **Phase 3 (Tasks)**: classifier impl → BanditState extension → Strategy G
  generator → validation run.
- **Phase 4 (Implement)**: TDD per `.github/skills/test-driven-development/SKILL.md`.

## Generalization

The classifier must operate on **arbitrary tasks**, not just LongCoT
CS-hard. Constraints:

- **Class taxonomy is universal, not benchmark-specific.** The 5 classes
  (LOCAL, GLOBAL_REDUCE, MULTI_HOP, FORMAT, REFUSAL) come from λ-RLM and
  describe reasoning shape, not domain. They must apply equally to a Spark
  log analysis (often GLOBAL_REDUCE), a multi-step math word problem
  (MULTI_HOP), an SQL question (FORMAT), and an ambiguous spec (REFUSAL).
  No CS-hard template names appear anywhere in the classifier code or
  prompts.
- **Classifier input is the prompt only.** No metadata fields like
  `template`, `domain`, `difficulty`, or `dataset`. If a deployment lacks
  those fields (every production user), the classifier still runs.
- **Beta-prior table is keyed by class, not by template.** The bandit's
  per-template state (`BanditState.template_state[tpl]`) stays as-is for
  backward compatibility, but the classifier seeds `class_priors[cls]`
  which works for any task that produces a class label, including tasks
  the bandit has never seen.
- **Cold-start fallback.** When the classifier returns UNKNOWN (low
  confidence, novel task), the bandit MUST fall through to its
  template-free prior — same behaviour as today's bandit on a brand-new
  template. No new failure mode introduced.
- **Validation is two-tier.** Headline metric (Strategy G ≥ F + 2 on
  CS-hard) proves the seeding helps on the studied family. A
  `tests/test_classifier_robustness.py` adds 20 hand-written prompts from
  unrelated domains (Spark, SQL, code-review, free-form) and asserts the
  classifier doesn't crash and returns a non-UNKNOWN class for ≥80% of
  them — proving generalisation.
- **Audit hook.** Classifier emits a `task_class` label into every
  trace's `metadata` so post-hoc analysis works on any future dataset
  without harness changes.

