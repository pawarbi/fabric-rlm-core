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

### The precise claim — learning did not degrade *reasoning*, it added *failures*

"−8.4 points" would overstate the case, so state it exactly. On the **22
questions where neither new failure mode fired, the two arms are identical**:
20/22 each, missing the same two questions (q16, q19) for the same H5 grading
artifact. Learning changed analytical quality **not at all**.

The entire gap is two questions lost to two mechanisms that **only exist on the
knowledge path**:

- **q13** — F14 crash (`ValueError: operation result was truncated`)
- **q08** — F11 turn exhaustion (13 gate rejections, 20 of 22 turns)

So the defensible verdict is narrower and stronger than a quality regression:

> With an empty package, `.learn` left reasoning quality unchanged, cost **+21%
> turns and +64% prompt tokens**, and introduced **two failure modes the cold
> path does not have**, losing two questions the cold arm answered correctly.

## Why — the package was empty

```
LEARN package frozen: {'lessons_total': 0, 'lessons_active': 0,
                       'fingerprint': '5b56910466ced4d0...', 'learn_seconds': 4.9}
```

Identical fingerprint across two independent runs, so `learn()` is
deterministic. It profiled 10 Delta tables in ~5 s and produced **nothing**.

This is documented behaviour, not a crash (`knowledge_api.py:223-226`): lessons
come only from `structural_lessons` + `declared_lessons`, and a Delta table with
no boolean English-named current-period flag yields neither. See F12.

**So arm B paid the full cost of the knowledge path while carrying no
knowledge.** The regression is pure overhead:

- an extra planner round-trip per question (`_prepare_registered_operation`),
- a larger prompt,
- and a new failure mode that the cold path does not have (F14).

## The two questions arm B lost

- **q13 — F14, a hard crash.** `ValueError: operation result was truncated`
  (`knowledge_execution.py:284`), re-raised at `runtime.py:1568`, aborting the
  question after 12 s. Arm A answered q13 correctly (334.0). **Learning turned
  a passing question into an exception.**
- **q08 — F11, turn exhaustion.** 13 catalog-gate rejections drove it to 20 of
  22 turns and it never produced an answer. Arm A answered it correctly
  (120.4238) in 8 turns.

## F11 finally quantified

Trajectory capture (harness defect H6, fixed before this run) gives real counts
instead of the model's self-reports:

| | value |
|---|---|
| Total catalog-gate rejections | **35** |
| Questions hitting the gate | **7 / 24** |
| Per question | q01:2, q02:2, **q08:13**, q15:2, q20:2, **q21:10**, q22:4 |

The two highest gate counts are the two highest turn counts (q08 → 20 turns,
q21 → 17 turns) and q08 is the arm's only turn-exhaustion failure. That is
direct evidence the undiagnosable message costs turns and can cost the answer —
a **bounded** version of the causal claim I earlier withdrew for lack of
evidence. It is a real cost, but it is *not* what produced the gpt-4.1-mini
33.3%: GLM absorbed 35 rejections and still scored 91.7% in arm A.

## Honest scope

- n = 1 per arm; the whole difference rests on two questions, each attributable
  to a specific identified mechanism (F14, F11) rather than to noise.
- Both arms share the categorical-slot grading artifact (H5) on q16/q19, so it
  does not bias the comparison.
- This says nothing about `.learn` on a **semantic model**, the one family with
  richer `structural_lessons`. Untested, not claimed.
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
