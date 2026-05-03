# SPEC: `combinators` skill — typed primitives for the fabric_rlm REPL

Status: **Phase 1 (Specify)** per `.github/skills/spec-driven-development/SKILL.md`.
No code until this spec is reviewed and approved.

Branch: `feat/combinators-skill`
Parent research: `bench/adaptive/RESEARCH-lambda-rlm-comparison.md` §3.1
Borrowed from: λ-RLM (`nktkt/lambda-rlm`), arXiv:2603.20105

## Assumptions I'm making

1. The combinators ship as a new **skill** at `fabric_rlm/skills/combinators.md`,
   not as a runtime change. Skill loading already handles auto-import via the
   subprocess startup path.
2. The 7 primitives are pure Python (no extra deps); they live in a thin
   helper module the skill points the model to (`fabric_rlm/skills/_combinators.py`)
   and are auto-imported only when the skill is loaded.
3. `peek` is the only primitive that needs to call the cost tracker; all
   others are O(in-memory) and free.
4. Validation benchmark is **Spark-RCA** (`bench/adaptive/spark_generate.py`),
   NOT the LongCoT CS-hard set. CS-hard prompts forbid code execution, so
   combinators don't apply.
5. Skill is opt-in via `enable_skill_autoloading=True` or
   `skills=["combinators"]`. Default behaviour unchanged.
6. → **Correct any of these now or I proceed.**

## Objective

Give the model a deterministic, pre-verified library of combinators for
chunk/map/reduce-style work in the REPL, so it stops re-deriving boilerplate
each session and gains a cost-tracked partial-read primitive.

**Success looks like**: on the Spark-RCA benchmark, an `RLM(skills=["combinators"])`
run shows ≥1 of:

- ≥10% reduction in code-emission tokens per question (less boilerplate),
- ≥5% improvement in pass rate at equal cost,
- ≥10% reduction in subprocess wall time per question via cheaper `peek`
  vs full file reads.

Failure looks like: no measurable improvement on any axis, or a regression
on the existing CS-hard benchmark when the skill is loaded but unused.

## Tech stack

- Python 3.10+ (matches repo)
- No new runtime dependencies
- Existing `fabric_rlm` skill loader (no changes)
- Existing `bench/adaptive/run_bench.py` with a new `--skills combinators` flag

## Commands

```
Build:   python -m build
Test:    python -m pytest tests/
Lint:    ruff check fabric_rlm/skills/_combinators.py tests/test_combinators.py
Bench:   python bench/adaptive/run_bench.py --dataset spark --skills combinators
Compare: python bench/adaptive/run_bench.py --dataset spark   # baseline
```

## Project structure

```
fabric_rlm/skills/
    combinators.md            ← new: skill manifest + cookbook (front-matter required)
    _combinators.py           ← new: 7 pure-Python primitives, auto-imported
tests/
    test_combinators.py       ← new: unit tests per primitive (pure Python)
    test_combinators_skill.py ← new: integration test — RLM picks up skill
bench/adaptive/
    REPORT-combinators.md     ← new: validation report (post-implementation)
```

## Code style

```python
# fabric_rlm/skills/_combinators.py

from __future__ import annotations
from typing import Callable, Iterable, TypeVar

A = TypeVar("A")
B = TypeVar("B")


def split(text: str, k: int) -> list[str]:
    """Split text into k roughly-equal chunks (by char count, word-aware).

    >>> split("a b c d", 2)
    ['a b', 'c d']
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    ...


def peek(text: str, offset: int, n: int) -> str:
    """Return n chars starting at offset. Cost-tracked via subprocess hook."""
    ...
```

Conventions:
- Pure Python; no I/O inside primitives (model passes content in).
- Total functions: validate inputs, raise `ValueError` (not silent truncation).
- Type hints + docstring + ≥1 doctest per primitive.
- The skill `combinators.md` includes one worked example per task pattern.

## Testing strategy

- **Unit** (`tests/test_combinators.py`) — every primitive: edge cases (empty,
  k=1, k>len, non-ASCII), error conditions, doctests.
- **Integration** (`tests/test_combinators_skill.py`) — boot an `RLM(skills=
  ["combinators"])`, assert the names are importable inside the subprocess,
  assert cost tracker increments on `peek`.
- **Bench validation** (`bench/adaptive/run_bench.py --dataset spark`) —
  pre/post comparison on Spark-RCA, results in `REPORT-combinators.md`.
- **Regression** — re-run the 5-way comparison generator with skills enabled
  on Strategy F to confirm zero regression on CS-hard (expected: 6/25 ± noise).

Coverage target: 100% on `_combinators.py` (it's small and pure).

## Boundaries

- **Always**: type-hint every primitive, doctest every primitive, lint before
  commit, run `tests/test_combinators.py` before commit.
- **Ask first**: any change to the skill loader, any new dep beyond stdlib,
  any change to `_RUNG_COST` or bandit machinery (out of scope for this spike).
- **Never**: replace the REPL with combinators (this is additive only),
  enable the skill by default, ship without the bench validation report.

## Success criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | All 7 primitives implemented + tested | `pytest tests/test_combinators.py` 100% pass |
| 2 | Skill loads in `RLM(skills=["combinators"])` | `pytest tests/test_combinators_skill.py` |
| 3 | `peek` reports cost via subprocess tracker | Integration test asserts non-zero counter |
| 4 | Spark-RCA bench shows ≥1 of (token↓, pass↑, time↓) | `REPORT-combinators.md` with side-by-side table |
| 5 | CS-hard 5-way comparison F unchanged when skill loaded | Re-run F, score within ±1 of 6/25 |
| 6 | Skill front-matter follows existing PLAYBOOK_CONTRACT | Diff against `fabric_rlm/skills/PLAYBOOK_CONTRACT.md` |

## Open questions

1. Should `peek` count tokens or chars? λ-RLM uses chars; our cost tracker uses
   tokens. Probably tokens (for parity with the rest of the stack).
2. Where do we materialize the validation Spark-RCA dataset? `spark_generate.py`
   exists but the generated artifact may not be checked in.
3. Should `cross` be lazy (generator) or eager (list)? λ-RLM is eager. Eager
   is simpler; lazy is safer on large inputs. Default eager, document the cap.
4. Does the skill need a `depends_on:` entry for `core`? Probably yes — PLAN
   block before any combinator chain is good hygiene.

## Phases (after spec approval)

- **Phase 2 (Plan)**: dependency order, parallelizable test work, risk register.
- **Phase 3 (Tasks)**: ≤5-file tasks per `spec-driven-development` skill.
- **Phase 4 (Implement)**: one task at a time with TDD per
  `.github/skills/test-driven-development/SKILL.md`.

## Generalization

This skill must be **task-agnostic**. It is being scoped against Spark-RCA
because that's our cleanest validation surface, but the design constraints
below apply to every task family (CS-hard, Spark-RCA, free-form Q&A,
code-gen, multi-doc QA, agentic tool use):

- **No template-specific code.** The 7 primitives operate on generic Python
  values (sequences, predicates, callables). They take no `template=` kwarg,
  raise no template-specific errors, and import nothing from
  `bench/adaptive/longcot_adapter.py` or any other dataset module.
- **No prompt-shape assumptions.** `peek` truncates by token count of the
  string passed in; it does not parse for `<question>`, `solution =`, or
  any other marker. Callers (the model, via the skill playbook) decide what
  to peek at.
- **Deterministic, side-effect-free.** Every primitive is a pure function so
  it composes safely in any task's solver. The only "side effect" is `peek`
  reporting to the subprocess cost tracker — a piece of infra that already
  exists for every task.
- **Validation surface is dual.** Success criterion #4 (Spark-RCA bench)
  proves the primitives help on a structured-data task. Success criterion
  #5 (CS-hard zero regression) proves they don't *hurt* a task family they
  weren't designed for. A future task family added to `bench/` should
  inherit the same dual treatment automatically — the skill loader already
  works for any `RLM(skills=["combinators"])` regardless of task.
- **Documentation discipline.** The `combinators.md` playbook MUST include
  one worked example per task pattern (sequence-reduce, search, batch-map),
  not per benchmark name. If a future task family needs a new pattern, add
  it to the playbook — never branch on dataset name.

