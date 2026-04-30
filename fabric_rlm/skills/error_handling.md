---
applies_when:
  keywords: []
  output_fields: []
excludes: []
depends_on: []
specificity: utility
---
# error_handling
Summary: Recovery patterns for runtime errors without hiding real failures.
Dependencies: none

## Gotchas

- Broad `except Exception` blocks can hide the exact failure that should guide
  the next turn.
- Retrying the same code after a traceback usually wastes turns; inspect the
  failing value or shape first.
- `SUBMIT()` raises a sentinel outside normal `Exception` handling, so do not
  wrap final submission in broad recovery code.

## Verified patterns

- After a traceback, print the relevant type, shape, keys, or first few records
  before changing approach.
- Catch narrow exceptions only when the task explicitly needs recovery for that
  case, and re-raise unexpected errors.
- Keep recovery turns small: one diagnosis, one targeted fix, one verifier.

## Pre-flight checklist

- [ ] The last traceback was explained by a concrete value or shape mismatch.
- [ ] No broad catch is masking validation or submission errors.
- [ ] Recovery code changes the approach rather than repeating the same call.
- [ ] stderr warnings were reviewed if they affect correctness.

