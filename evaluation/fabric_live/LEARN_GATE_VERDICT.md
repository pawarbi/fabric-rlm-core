# The `.learn` gate on the Fabric `dbo` lakehouse — VERDICT: FAIL

User's stated gate: *".learn accuracy >= cold in fewer turns or on
novel/complex questions."*

Both arms: `z-ai/glm-5.3-flash`, same 24 questions, same prompt, same source,
same `max_turns=22`, same environment (Fabric py3.12.12, dspy 3.2.1), workbook
duty on in both. **Only `knowledge=` differs.**

| Metric | A — cold (`glm-workbook-1`) | B — learn (`glm-learn-1`) | Δ |
|---|---|---|---|
| Analytical accuracy | **22/24 = 91.7%** | 20/24 = 83.3% | **−8.4 pts** |
| Workbook accuracy | **23/24 = 95.8%** | 20/24 = 83.3% | **−12.5 pts** |
| Mean turns | **7.88** | 9.50 | **+20.6%** |
| Mean prompt tokens | **40,314** | 66,320 | **+64.5%** |
| No machine-readable answer | **0** | 2 | worse |
| Hard crashes | **0** | 1 | worse |
| Lessons in the package | — | **0** | — |

**Gate fails on every axis simultaneously.** Learning did not improve accuracy,
did not reduce turns, and did not reduce tokens.

### The precise claim — one attributable regression, not two

"−8.4 points" would overstate the case, so state it exactly. On the **22
questions where neither new failure mode fired, the two arms score identically**:
20/22 each, missing the same two questions (q16, q19) for the same H5 grading
artifact. I detected **no difference in analytical quality** — though with
n = 1 per arm at temperature 1.0, that is "no difference detectable at this
power", not "no difference".

Two questions separate the arms. **Only one of them is attributable to
learning.**

- **q13 — attributable.** `ValueError: operation result was truncated`, raised
  inside `_prepare_registered_operation`. That frame is unreachable unless the
  package supplies registered operations, so this failure **cannot occur on the
  cold path**. Arm A answered q13 correctly.

- **q08 — NOT attributable; reclassified as sampling divergence.** I originally
  blamed the catalog gate (13 rejections, 20 of 22 turns). That claim does not
  survive scrutiny:
  1. Arm A's run log has **no `catalog_gate_rejections` field at all** — gate
     counting (H6) was added *after* the cold run. Arm A's "0" is **unmeasured,
     not zero**. I have no baseline to compare against.
  2. q08 shows **no `knowledge_result`**, so no operation executed on it.
  3. Both arms bind the same sources; at temperature 1.0 with n = 1, a
     20-turn excursion is well within observed variance.

So the defensible verdict is narrower than I first wrote:

> With a package containing **no lessons but a live operation set**, `.learn`
> showed **no detectable change in analytical quality**, cost **+21% turns and
> +64% prompt tokens in aggregate**, and introduced **one failure mode the cold
> path structurally cannot have** (F14), losing one question the cold arm
> answered correctly. It gained nothing on any axis of the user's gate.

## Why — the package had no lessons, but was **not** inert

```
LEARN package frozen: {'lessons_total': 0, 'lessons_active': 0,
                       'fingerprint': '5b56910466ced4d0...', 'learn_seconds': 4.9}
```

Identical fingerprint across two independent runs, so `learn()` is
deterministic. It profiled 10 Delta tables in ~5 s and produced **no lessons**
(F12 explains why: the only tabular lesson is gated on English column names).

**But zero lessons does not mean zero effect.** `KnowledgePackage` also carries
`operations` from `discover_registered_operations(profiles, sources)`, which for
a lakehouse are generated purely from `family == "lakehouse"` and
`diagnostics["snapshot_exact"] is True` — *independent of lessons*. The package
therefore had **0 lessons and a non-empty operation set**.

Evidence that operations were present and active:

- `_bind_knowledge_inputs` sets `knowledge_mode="registered_operations_available"`
  whenever any supported operation exists (`runtime.py:1224`).
- `knowledge_result` — the input alias injected **only** on successful operation
  execution — appears in exactly 3 of 24 trajectories: q15 (14 mentions),
  q17 (12), q24 (9), and 0 elsewhere.
- q13 crashed *inside* `_prepare_registered_operation`, a path that cannot be
  reached unless operations are available.

So the mechanism is: **learning contributed no knowledge, but switched on the
registered-operation machinery**, which fired usefully on 3 of 24 questions and
introduced a fatal failure mode on a 4th.

### What I can and cannot attribute

`_prepare_registered_operation` makes an **operation-selection LM call on every
task** before any parse or fallback
(`response_text, raw_response, selection_seconds = _call_lm_with_meta(self.outer_lm, messages)`,
recorded as `operation_selection_lm_calls: 1`). That is a real, unconditional
per-question cost, and it is confirmed by reading the code.

**I attempted to blame the +64% prompt tokens on it, and my own data does not
support that.** `analyze_token_overhead.py` compares the arms per question:

| subset | result |
|---|---|
| 18 "clean" questions (no operation executed, not gate-heavy) | median **+26,040**, but range **−28,725 to +80,667**, and **3 of 18 negative** |
| 4 questions with *identical turn counts* and no operation — where a uniform tax must show | +6,692, +20,670, +12,663, **−8,193** |
| 3 questions where an operation actually executed | **+5,655, +21,131, +607** — the *smallest* deltas |

A strict per-question additive tax cannot produce a −8,193 delta on a
same-turn-count question. At **temperature 1.0 with n = 1**, run-to-run sampling
variance dominates per-question token counts. And the operation-executed
questions being the *cheapest* is the opposite of what "the `knowledge_result`
packet inflates the prompt" would predict.

**Therefore:** the aggregate overhead (+64% tokens, +21% turns) is real and
reproducible in the totals, but its **decomposition is not established**. The
selection call certainly costs something; how much is unmeasured. Isolating it
needs `operation_selection_prompt_tokens`, which the framework already emits on
`trajectory.metadata` and which my harness failed to persist — recorded as
harness defect **H7**.

## Why — the package was empty of lessons

```
LEARN package frozen: {'lessons_total': 0, 'lessons_active': 0,
                       'fingerprint': '5b56910466ced4d0...', 'learn_seconds': 4.9}
```

Identical fingerprint across two independent runs, so `learn()` is
deterministic. It profiled 10 Delta tables in ~5 s and produced **nothing**.

This is documented behaviour, not a crash (`knowledge_api.py:223-226`): lessons
come only from `structural_lessons` + `declared_lessons`, and a Delta table with
no boolean English-named current-period flag yields neither. See F12.

**So arm B paid the cost of the knowledge path while carrying no lessons.** The
regression is overhead plus a new failure mode:

- an extra operation-selection LM call per question
  (`_prepare_registered_operation`) — confirmed in code, magnitude unmeasured,
- larger prompts in aggregate (+64%), decomposition **not** established,
- and a failure mode the cold path structurally cannot have (F14).

## The question arm B lost

- **q13 — F14, a hard crash.** `ValueError: operation result was truncated`
  (`knowledge_execution.py:284`), reaching the bare `except ValueError` at
  `runtime.py:1558` which re-raises at `:1568`, aborting the question after
  12 s. Arm A answered q13 correctly (334.0). **Learning turned a passing
  question into an exception.** This is the one clean causal finding of the arm.

- **q08 — no longer claimed.** See the reclassification above: arm A never
  measured gate rejections, so there is no baseline, and no operation executed
  on q08. Counted as an unexplained divergence, not a learning effect.

## F11 — what the gate counts do and do not show

Trajectory capture (harness defect H6, fixed before this run) gives real counts
instead of the model's self-reports, **for arm B only**:

| | value |
|---|---|
| Total catalog-gate rejections | **35** |
| Questions hitting the gate | **7 / 24** |
| Per question | q01:2, q02:2, **q08:13**, q15:2, q20:2, **q21:10**, q22:4 |

Within arm B, the two highest gate counts are the two highest turn counts
(q08 → 20, q21 → 17). That is a **correlation inside one arm**, and it is the
strongest form of the claim the evidence supports.

It is **not** evidence that learning causes gate rejections, because **arm A was
never instrumented** — H6 landed after the cold run, so I cannot say whether the
cold arm hit the gate 0 times or 35. Establishing that requires re-running arm A
with trajectory capture; until then the gate is a cost of the *system*, not
demonstrably a cost of *learning*.

What it definitely is not: the cause of the gpt-4.1-mini 33.3% collapse. GLM
absorbed 35 rejections and still scored 91.7% in arm A.

## Honest scope

- n = 1 per arm at temperature 1.0. Per-question token and turn deltas are
  dominated by sampling variance (three "clean" questions moved *negative*), so
  only aggregate directions and the structural F14 finding are load-bearing.
- The whole accuracy gap now rests on **one** question with an identified
  mechanism. That is thin; it is reported as such rather than inflated.
- The **lesson pathway was never exercised** — 0 lessons means this arm tested
  the *operations* pathway only. `.learn` with meaningful `declared=` metadata,
  or on a semantic model (the family with richer `structural_lessons`), is
  **untested and not claimed**.
- Both arms share the categorical-slot grading artifact (H5) on q16/q19, so it
  does not bias the comparison.
- Token figures are **raw prompt tokens, not cost**. The frozen operations JSON
  is cache-friendly and `knowledge_result` injection may bust the cached prefix;
  `total_cached_tokens` was not captured (H7).
- A `declared=` block would produce a non-empty package. That would test
  *author-supplied metadata*, which is worth doing, but it is not what
  `learn()` infers on its own.

## Reproduce

```powershell
cd <session>\files\fabric_upload
# arm A
python notebook.py execute --workspace-id 82ad2591-974a-4ad4-ace6-e24879274a4b `
  --notebook-id 2935d52f-4574-443d-a8d0-552936a7445a `
  --parameter "RUN_TAG=glm-workbook-1" --parameter "MODEL=z-ai/glm-5.3-flash" `
  --parameter "WORKBOOK_DUTY=True" --parameter "LEARN=False"
# arm B
python notebook.py execute --workspace-id 82ad2591-974a-4ad4-ace6-e24879274a4b `
  --notebook-id 2935d52f-4574-443d-a8d0-552936a7445a `
  --parameter "RUN_TAG=glm-learn-1" --parameter "MODEL=z-ai/glm-5.3-flash" `
  --parameter "WORKBOOK_DUTY=True" --parameter "LEARN=True"
# grade
python grade_analytical.py run_log_glm.json       analytical_grade_glm.json
python grade_analytical.py run_log_glm_learn.json analytical_grade_glm_learn.json
python repro_learn_naming.py    # why the package was empty, no credentials needed
```
