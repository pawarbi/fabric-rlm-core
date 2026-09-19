---
applies_when:
  keywords: ["csvsummary", "csv summary"]
  output_fields: []
excludes: []
depends_on: []
specificity: domain
---
# csv_summary
Summary: Count CSV records and optionally sum a named numeric column into a structured summary.
Dependencies: none

This working example is documentation, not a bundled skill. To use it as a
custom skill, copy it to your own skills directory as `csv_summary.md` and pass
that directory through `SkillLoader(skill_dir=...)`. See
[Authoring Skills](authoring-skills.md) for configuration and a standalone
loader/verifier self-test. The section layout below is suggested, not required
by the runtime; only verifier extraction depends on its exact heading and fence.

## Purpose

Count records in a header-bearing CSV and optionally total one explicitly named
numeric column; do not infer a column, deduplicate records, or silently discard
invalid amounts.

## Contract: output fields

Configure the runtime with `outputs={"summary": dict}` and submit
`SUBMIT(summary=summary)`. The verifier receives the complete payload
`{"summary": summary}`, not the inner dictionary alone.

- `summary`: a dictionary with exactly `row_count` and `total`; the payload has
  exactly the outer key `summary`.
- `row_count`: a built-in `int`, excluding `bool`, greater than or equal to zero.
  Count records yielded by `csv.DictReader`, excluding the header. Do not count
  physical lines: quoted fields can contain newlines. Empty input or a header
  with no records yields zero records.
- `total`: a built-in `int` or finite `float`, excluding `bool`. If a column is
  requested, parse and sum that column for every record. Missing, blank,
  nonnumeric, or non-finite amounts require a stated limitation or clarification,
  not an invented zero. With no requested column, or no records, use `0`.

Parse CSV text using the standard-library `csv` module. Parse selected numeric
cells with `int` or `float` as appropriate, check finiteness, and check the sum
for overflow to a non-finite float. If values arrive already typed, reject
booleans before numeric checks. This generic example does not promise decimal
currency precision; use a task-specific decimal policy when required.

The verifier below checks structure, types, and finite numeric bounds only. It
does not have the source CSV, so it cannot prove that `row_count` or `total`
matches the data, or enforce the no-column total rule. Independently recompute
from the source when factual validation is required.

## Required verifier

Call `verify({"summary": summary})` on the computed result before submitting.
Repair any assertion failure and verify again. Only submit after it returns
silently. Keep this block self-contained; malformed inputs must raise
`AssertionError`, not an accidental indexing or conversion exception.

```python
def verify(payload):
    import math

    if not isinstance(payload, dict):
        raise AssertionError("payload must be a dictionary")
    if set(payload) != {"summary"}:
        raise AssertionError("payload must contain exactly summary; no unexpected keys")
    summary = payload["summary"]
    if not isinstance(summary, dict):
        raise AssertionError("summary must be a dictionary")
    if set(summary) != {"row_count", "total"}:
        raise AssertionError("summary must contain exactly row_count and total; no unexpected keys")
    row_count = summary["row_count"]
    if type(row_count) is not int or row_count < 0:
        raise AssertionError("summary.row_count must be an integer >= 0, not bool")
    total = summary["total"]
    if type(total) not in (int, float):
        raise AssertionError("summary.total must be int or float, not bool")
    if type(total) is float and not math.isfinite(total):
        raise AssertionError("summary.total must be finite")
```

## Tripwires

- Use `csv.DictReader`, not comma splitting or physical-line counting.
- Do not accept `True` as a count or `False` as a numeric total.
- Do not silently skip invalid cells, count the header, or deduplicate records.
- Do not mistake a passing structural verifier for proof of a correct CSV sum.
- Do not raise `ValueError` to reject a submission: runtime non-assertion verifier
  failures fail open and do not block submission on this check's behalf.

## Procedure

1. Identify the input CSV, encoding, delimiter, and optional numeric column from
   the task; ask for missing requirements instead of guessing.
2. Open the source with `newline=""` and the specified encoding. Parse records
   with `csv.DictReader`; count every yielded record and, if requested, parse
   and sum every selected cell under the contract above.
3. Build `summary = {"row_count": row_count, "total": total}`. Check the result
   against the source independently if the task requires factual verification.
4. Call `verify({"summary": summary})`, repair failures, then emit
   `SUBMIT(summary=summary)` in the runtime.

## Example Self-Test

After executing the verifier block above, run these independently of any CSV or
model. These are contract fixtures, not a hard-coded expected answer:

```python
verify({"summary": {"row_count": 0, "total": 0}})
verify({"summary": {"row_count": 2, "total": -1.5}})

try:
    verify({"summary": {"row_count": True, "total": 0}})
except AssertionError:
    pass
else:
    raise AssertionError("Verifier accepted a boolean row_count")
```

The guide's self-test loads `docs/skill-template.md` as `skill-template`, executes
its extracted `verifier_source` in a trusted namespace, and covers additional
missing keys, unexpected keys, malformed types, and non-finite numbers.
