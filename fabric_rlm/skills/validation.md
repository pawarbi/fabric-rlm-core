---
applies_when:
  keywords: []
  output_fields: []
excludes: []
depends_on:
- error_handling
specificity: utility
---
# validation
Summary: Verifier-first validation patterns for structured RLM tasks.
Dependencies: error_handling

## Gotchas

- Passing a single happy-path example is not enough; verifier failures often
  come from missing edge cases, duplicated rows, or silent type coercion.
- Output fields must match the requested names and shapes exactly. Do not
  rename fields, add explanatory wrappers, or return partial payloads.
- Off-by-one counts usually appear after filtering headers, footers, empty
  rows, or inclusive/exclusive ranges.

## Verified patterns

- Build a tiny local verifier before `SUBMIT()`: check required keys, lengths,
  numeric totals, and expected-vs-actual examples.
- Print concise diagnostics when validation fails, then fix the data path
  before submitting.
- Prefer explicit normalization steps for dates, money, IDs, and labels; keep
  the original value alongside the normalized value if ambiguity matters.

## Pre-flight checklist

- [ ] Required output fields are present with exact names.
- [ ] Counts, totals, and joins have been checked against source inputs.
- [ ] Empty, duplicate, and boundary cases were considered.
- [ ] The final `SUBMIT()` payload contains only the requested fields.

