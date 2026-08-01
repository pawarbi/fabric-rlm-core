# Audit of AgenticDataBench's evaluator

Run before committing to a full sweep. Every number below comes from calling
the benchmark's own `compare_csv` directly; nothing is patched. Reproduce with
`audit_evaluator.py`, `audit_tolerance.py` and `verify_grader_equivalence.py`.

## Our grader is faithful

`grade_pilot.py` was checked task-by-task against the stock `Evaluator` on two
full arms and matched every score exactly. To run the stock path on Windows at
all, `os.path.join` is swapped for `posixpath.join` and the roots are passed
with forward slashes; the scoring logic is untouched.

The stock evaluator has **two** Windows-only defects, both from backslashes in
paths, and both make it score every task 0 rather than fail loudly:

1. the eval_func path substitution uses the path as an `re.sub` *replacement
   template*, so `\s`, `\U` etc. raise "bad escape";
2. the substituted path then sits inside a Python string literal that is
   `eval()`d, where `\U` is parsed as a unicode escape.

Aggregation matches too: `total_score = sum(scores)/len(scores)` per task.

## How grading actually works

Columns are matched **by content**, not by name or by position. Every gold
column is searched for among *all* prediction columns. Consequences worth
knowing before writing any "output contract" guidance:

- renaming every column changes nothing (verified: score 1.0)
- reordering every column changes nothing (verified: score 1.0)
- `score_rule='divide'` gives partial credit: matched columns / total columns

So advice about exact headers and positional mapping is largely irrelevant to
the score on this benchmark. Row *order* does matter when
`ignore_order=False`.

## Defect 1 (harsh): tolerance is applied by grid-snapping

With `ignore_order=True`, values are normalized as `round(v/tol)*tol` and the
buckets compared. Two values closer together than `tol` land in different
buckets whenever a grid line falls between them, so a prediction inside the
stated tolerance is rejected.

Sweeping offsets from 0 to `tol`:

    ignore_order=True    20/40 offsets strictly inside tolerance REJECTED (50%)
    ignore_order=False    0/40 offsets rejected

Any offset at or above `tol/2` fails, so the effective tolerance is **half**
what the task declares. With `ignore_order=False` the fallback uses
`math.isclose` and behaves correctly.

**Reach:** 111 of 140 compare_csv-graded tasks (79%) use `ignore_order=True`.

**Impact on our runs:** approximately none. Re-checking every graded column
under a correct tolerance flagged one column on one task, and that task scored
1.000 anyway.

## Defect 2 (lenient): extra rows are free

`ignore_order=True` uses subset semantics: every gold value must appear in the
prediction. A prediction with extra junk rows still scores 1.0. There is a
row-count guard, but it only fires when the output is **too short**; too long
passes. Padding an answer cannot hurt and can help.

## Defect 3 (lenient): a missing column can still score

Because matching is content-based across all prediction columns, gold with two
identical columns is satisfied by a prediction containing one of them. Verified:
gold with two identical columns against a one-column prediction scores 1.0.

## Behaviour that is correct

identical files 1.0; wholly wrong prediction of the right shape 0.0; output
shorter than gold 0 with a clear message; string case and surrounding
whitespace normalized; per-column partial credit as documented.

## Corrections this forced to earlier task diagnoses

- `strategy_2` was **not** lost to swapped columns. Swapping the two columns
  back and re-scoring gives the identical 0.111, because matching is
  content-based. The loss is genuine computation differences in the binning.
- `strategy_3` **was** partly lost to sort order: reversing the rows to match
  gold's descending order raises it from 0.000 to 0.125. The task never states
  a sort order and grading uses `ignore_order=False`, so that part is
  benchmark under-specification. The rest is a genuine computation error.
- `agriculture_22` remains unpassable-by-construction: per-class F1 of an
  unseeded cross-validated random forest, and the halved effective tolerance
  makes it worse.

## Verdict

The grader is deterministic and usable, and its defects apply equally to every
system measured on it, so comparisons against published numbers stay
apples-to-apples. It is **not** precise enough to adjudicate small differences,
which compounds the separate run-to-run variance problem recorded in
`PREREGISTRATION.md`.
