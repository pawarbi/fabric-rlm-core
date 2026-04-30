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

1. **Reflect-then-submit.** If a domain skill provides a `verify(...)`
   function in its **Required verifier** section, call it on your assembled
   payload. If it raises, repair the offending field and re-verify. Only
   emit `SUBMIT(...)` once `verify` returns silently. If no domain skill
   is active, still self-check the payload against any explicit invariants
   in the prompt before submitting.
2. **Single SUBMIT.** Emit exactly one `SUBMIT(...)` call with the computed
   value. Do not include extra prose around it. The runtime owns the strict
   `solution = ...` output contract; comply with whatever the active
   signature/playbook tells you to put inside `output`.
3. **Do not echo the prompt.** Never copy the question, the playbook
   text, or earlier scratch back into the SUBMIT payload.
4. **Skill bodies are advice, not output.** Skill prose may demonstrate
   formulas; only the verifier output and your computed values belong in
   the submission.
