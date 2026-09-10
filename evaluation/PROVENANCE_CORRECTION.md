# Provenance correction — the evaluated artifact was not the declared freeze base

Status: **correction to this evaluation's own record.** Supersedes the base-commit
claim in `AUDIT_INDEX.md` and `FIX_REGISTER.md`. Written after the user asked
whether the findings apply to PR #75.

## 1. What was claimed

Every prior document stated the evaluation ran against **`b5226712`**, and cited as
proof that `git diff b5226712 -- fabric_rlm` was empty when run from the eval
worktree after each commit.

## 2. What was actually evaluated

The wheel installed by every Fabric notebook was built from **`bd924bc`**, not
`b5226712`. Verified by normalised content comparison of the wheel's
`knowledge_api.py`, `lakehouse.py` and `runtime.py` against each candidate tree:

| file | matches |
|---|---|
| `knowledge_api.py` | `bd924bc`, `4466e9b`, pr75 working tree |
| `lakehouse.py` | `bd924bc`, `4466e9b`, pr75 working tree |
| `runtime.py` | **`bd924bc` and pr75 working tree only** — *not* `4466e9b` |

Independent confirmation: `declared=` (the arm D entry point) is **absent** from
`knowledge_api.py` at `b5226712` and present at `bd924bc`. Arm D ran and produced
32 declared lessons, so it cannot have run on `b5226712`.

## 3. Why the freeze check did not catch it

`git diff b5226712 -- fabric_rlm` was run **from the eval worktree**. That proves
the *eval branch* never modified core. The runtime was built from a *different
worktree* (`fabric-rlm-core-pr75`, branch `pr-75`), which the check never looked
at. The check was sound for what it measured and irrelevant to what was run.

**The freeze constraint was verified on the wrong artifact.**

## 4. The freeze constraint was also violated in fact

`4466e9b..pr-75` contains three commits absent from PR #75:

| commit | author | date | subject |
|---|---|---|---|
| `53e6cbb` | Sandeep Pawar | 2026-09-08 | fix: claim provenance reads signs and exponents; skills drop one domain's nouns |
| `a736098` | Sandeep Pawar | 2026-09-08 | docs: record the sign boundary the provenance screen does not cover |
| `bd924bc` | **Copilot** | 2026-09-09 | fix: make operation row-bound overflow recoverable instead of fatal |

`bd924bc` is **Copilot-authored and modifies core** — `runtime.py` (+17),
`knowledge_execution.py` (+19), plus tests. The user's instruction was *"Freeze the
core implementation and bundled skills during evaluation; report proposed fixes
separately."* A core fix was authored and built into the evaluated wheel. This is a
**violation of the stated constraint**, not merely a labelling error.

`53e6cbb` additionally modifies bundled skills (`analytical_integrity.md`,
`delta_lakehouse.md`, `semantic_model.md`), so the skills were not frozen either,
though those commits are the user's own.

## 5. What survives, and what does not

**Survives.** Findings F11, F15, F16 and the `nrr`/`grr` leak were read from the
`pr-75` tree, which is now confirmed to be the tree actually executed. Attribution
is therefore *correct*; only the base label was wrong. All four were additionally
verified to be **present at `4466e9b`**, the PR head.

**Does not survive unqualified.** Every quantitative result — cold 91.7%, learn
83.3%, arm D 87.5%, turn and token deltas — describes **`bd924bc`**. They are
approximately PR-75 evidence, but not evidence about `b5226712`, and not exactly
evidence about `4466e9b` either.

## 6. Open risk: arm comparability (unresolved)

All notebooks install one fixed, **mutable** lakehouse path,
`{EVAL}/fabric_rlm-0.6.0-py3-none-any.whl`, uploaded from `dist_eval`. Local wheel
builds exist at four distinct times:

```
09-08 14:46  wheel/
09-08 15:33  openrouter-notebook-artifacts/
09-08 22:42  fabric-rlm-core-pr75/dist_pr75/
09-09 14:34  fabric-rlm-core-pr75/dist_eval/   <- source of the uploaded wheel
```

Because the lakehouse path is overwritten in place, **arms executed at different
times may have run different bytes under the same filename.** The user explicitly
required "the same model/version, skills, execution limits, and sampling settings"
across arms A/B/D. That control is **not currently demonstrated**.

**Until this is resolved, the A/B/D comparison should not be treated as a fully
controlled experiment.**

### 6a. The F14 contradiction, resolved: the fix covers the wrong row-bound path

`bd924bc` (09-09) claims to make operation row-bound overflow *recoverable*, yet
arm D's q13 was recorded dying at `turns=None` with `ValueError: operation result
was truncated` (finding F14). The two are consistent, and the reason matters.

**Correction to an earlier draft of this section.** It stated that
`OperationResultTooLarge` "derives from `OperationPlanError`, not `ValueError`, so
the handler cannot catch the old raise." That reasoning is **wrong**:
`class OperationPlanError(ValueError)`, so `OperationResultTooLarge` *is* a
`ValueError`. The conclusion survives, but the mechanism is different and more
specific.

There are **two distinct row-bound overflow paths** in `knowledge_execution.py`:

| line | condition | raises | outcome |
|---|---|---|---|
| 309 | `len(rows) > max_output_rows` | `OperationResultTooLarge` | **recoverable** — `runtime.py:1524` falls back to ordinary execution |
| **284** | host returned `{"truncated": True}` | **bare `ValueError`** | **fatal** |

`bd924bc` fixed line 309 and left line 284 untouched. In `runtime.py` the handler
chain is `OperationResultTooLarge` → `OperationPlanError` → `except ValueError as
exc:` at line 1558, and that last handler records telemetry with
`reason="audit_failed"` and then **bare `raise`** — it re-raises, aborting the run.

A bare `ValueError` is not an `OperationResultTooLarge` nor an
`OperationPlanError`, so the `truncated` path falls through to line 1568 and kills
the task before the agent loop starts. That is precisely the observed `turns=None`
signature.

**Line 284 is a row bound by the code's own design.** The comment at line 677
states: *"The row bound stays an over-fetch so a period with more groups than the
operation may return is reported as truncated, never trimmed."* The Lakehouse
operation deliberately over-fetches by one row and signals overflow via the
`truncated` flag. So the `truncated` path is the *Lakehouse* expression of exactly
the condition line 309 handles for in-memory results — and by `bd924bc`'s own
stated rationale (*"rows follow from the model's plan choice, so they are
recoverable; columns are the host's contract, so they fail closed"*) it should be
recoverable too.

**Conclusion.** `bd924bc` is sound in intent and correct where it applies, but
**incomplete**: it makes the in-memory row bound recoverable while the Lakehouse
row bound — the one that actually fired on real Fabric Delta data — stays fatal.
F14 is therefore unfixed at `bd924bc`, and equally unfixed at `4466e9b`, which
lacks even the partial fix.

This also means q13's crash does **not** prove the arms ran different wheels — the
crash is explained at any wheel in the range. Arm comparability remains formally
undemonstrated (§6) but is no longer contradicted by this evidence.

## 7. Required corrections elsewhere

- `AUDIT_INDEX.md` and `FIX_REGISTER.md`: replace the `b5226712` base claim with
  `bd924bc`, and link this document.
- Any statement that core was frozen must be qualified by §4.
- The planned `fix/task-hardening` branch was cut from `b5226712`; it must be
  rebased onto the tree the evidence actually covers.

## 8. What this means for merging PR #75

PR #75 is **open**, head `4466e9b`, base `main`.

**Relationship to what was evaluated:** `4466e9b` is an *ancestor* of `bd924bc`.
The evaluated tree is PR #75 **plus** three unpushed local commits (§4). So the
evaluation is approximately PR-75 evidence — it is the closest evidence that
exists — but it tested a slightly *newer* tree than the PR contains.

**Merging `4466e9b` would land code lacking all three commits in §4**, including
the user's own `53e6cbb` (claim provenance reads signs and exponents; skills drop
one domain's nouns) and `a736098`. If those were intended to be part of this PR,
**the branch has not been pushed** and the PR is behind local work.

**Findings verified to be present at `4466e9b`** — merging does not introduce them
and does not fix them:

| finding | at `4466e9b` | path | severity |
|---|---|---|---|
| F11 comment-marker SQL gate rejects legitimate literals | present | `.task` | critical |
| F15 English-only measure-name regex gates learning | present | `.learn` | high |
| F16 clarification guard fails open for non-English | present | `.task` (opt-in) | critical when used |
| `nrr`/`grr` hard-coded in runtime core | present | `.learn` | the one genuine ARR leak |
| F14 fatal `ValueError` on truncated operation result | present | `.learn` | high |

**Recommendation — not a blanket yes or no.** The evaluation cannot certify
`4466e9b`, because it did not run `4466e9b`. What it supports:

1. The measured behaviour (cold 91.7% on 24 complex lakehouse questions) came from
   a tree that is PR #75 plus three commits. That is a reasonable, though not
   exact, signal that the PR's `.task` path works.
2. The `.learn` gate the user set — *accuracy ≥ cold in fewer turns* — **failed**
   on that tree (83.3% learn, 87.5% declared, vs 91.7% cold). PR #75 is titled
   "generalize learning"; **its central claim is the one the evaluation did not
   confirm.**
3. None of the five findings above are fixed by merging.

Before merging, the two cheap and decisive steps are: **push `pr-75` so the PR
contains the three local commits**, then **re-run at least the cold arm against the
exact merge target** so the numbers describe the code being merged.
