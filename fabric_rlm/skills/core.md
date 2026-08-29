---
applies_when:
  keywords: []
  output_fields: []
excludes: []
depends_on: []
specificity: core
---
# core
Summary: Minimal always-on contract every RLM run obeys.

## Purpose

Define the non-negotiable behavior of every RLM turn: how to submit, when
to verify, and what never to echo back. This skill is always active.

## Contract: output fields

When the prompt declares output fields or provides a `solution = {...}`
literal, the SUBMIT payload MUST be a JSON object whose top-level keys
exactly match the declared schema names — no extras, no missing keys,
no renames, no nesting tricks.

## Required behavior

1. **PLAN before action.** Your **first turn** of every run MUST begin with a
   `## PLAN` block. **The block MUST be inside Python comments — every line
   starts with `#`.** Never emit raw markdown (e.g. a bare `## PLAN` line
   without a leading `#`) — the turn body is executed as Python and bare
   markdown will raise `SyntaxError`. Example:

   ```python
   # ## PLAN
   # Target: <one line>
   # Sub-problems: 1) ... 2) ...
   # Assumptions: ...
   # Approach: ...
   ```

   The plan states:
   - **Target.** What the prompt is asking you to produce, in your own words.
   - **Sub-problems.** The 1–5 sub-questions you need to answer.
   - **Assumptions.** Any interpretation you are committing to where the
     prompt is ambiguous. State them and proceed; do not refuse to act on
     under-specified prompts.
   - **Approach.** One sentence per sub-problem describing the strategy.

   The plan is for *you*, not the user. Keep it terse (≤15 lines total).
   Do not call any tools or write any computation in the same turn as the
   plan unless the task is so trivial that the plan is one line.

2. **VERIFY before SUBMIT.** The turn that contains your `SUBMIT(...)` call
   MUST be preceded (in the same turn) by a `## VERIFY` block. **The block
   MUST be inside Python comments — every line starts with `#`.** Never
   emit raw markdown headings or bullets; they will raise `SyntaxError`.
   Example:

   ```python
   # ## VERIFY
   # Target: <restated>
   # - constraint A: OK
   # - constraint B: OK
   SUBMIT(...)
   ```

   The block must:
   - Restate the target from your PLAN.
   - List each explicit constraint or output-shape requirement from the
     prompt and write `OK` or `FIX` next to it for your candidate value.
   - If any item is `FIX`, repair it and re-verify before emitting SUBMIT.

   If a domain skill provides a `verify(...)` function in its **Required
   verifier** section, call it inside the VERIFY block on your assembled
   payload; if it raises, that is a `FIX`. Only emit `SUBMIT(...)` once
   every item is `OK`.

3. **Single SUBMIT.** Emit exactly one `SUBMIT(...)` call with the computed
   value. Do not include extra prose around it. The runtime owns the strict
   `solution = ...` output contract; comply with whatever the active
   signature/playbook tells you to put inside `output`.

4. **Do not echo the prompt.** Never copy the question, the playbook
   text, or earlier scratch back into the SUBMIT payload.

5. **Skill bodies are advice, not output.** Skill prose may demonstrate
   formulas; only the verifier output and your computed values belong in
   the submission.

6. **Honor PRIOR_ATTEMPT_FEEDBACK if present.** If the prompt contains a
   `## PRIOR_ATTEMPT_FEEDBACK` block, read it first. It tells you what a
   previous attempt produced and why it failed. Use it to choose a
   different approach for this attempt — do not repeat the prior payload
   or the same line of reasoning.

7. **Do not refuse based on perceived truncation.** Ellipsis characters
   (`...` or `…`) inside set, sequence, or range notation — for example
   `{1, ..., n}`, `{1,...,5}`, `[a_1, ..., a_k]`, `r in {1,...,d}`, `i =
   1, 2, ..., N` — are **mathematical notation**, not signs that the
   prompt was cut off. The same applies to ellipses inside English
   enumerations like "for layers 1, 2, ..., L". Treat the prompt as
   complete and attempt the problem. Refuse only when an explicit data
   field is genuinely missing (e.g. an empty matrix literal, an
   unreplaced `<PLACEHOLDER>`, or text that ends mid-sentence with no
   closing punctuation). When in doubt, attempt and verify.
