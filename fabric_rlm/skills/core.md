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

When the prompt declares output fields (e.g. `Q1..Q5`, or a `solution =
{...}` literal), the SUBMIT payload MUST be a JSON object whose top-level
keys exactly match the declared schema names — no extras, no missing
keys, no renames, no nesting tricks.

## Required behavior

1. **PLAN before action.** Your **first turn** of every run MUST begin with a
   `## PLAN` block (in a Python comment or printed string is fine — it just
   needs to appear in the visible turn output). The plan states:
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
   MUST be preceded (in the same turn) by a `## VERIFY` block that:
   - Restates the target from your PLAN.
   - Lists each explicit constraint or output-shape requirement from the
     prompt and writes `OK` or `FIX` next to it for your candidate value.
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
