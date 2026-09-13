# Why identical runs disagree

The noise floor says an unchanged configuration moves 12–16 accuracy points
against itself and flips 24–44% of its answers between identical runs. That
number invites one assumption — that the model is nondeterministic and the only
remedy is more repetitions.

That assumption is wrong here. **71% of the disagreement has a mechanical cause.**

Source: `evaluation/dbo_lakehouse/dbo_eval_results.json`, 150 trials
(2 arms × 25 questions × 3 reps), `openai/gpt-4.1-mini`. No new inference was
run to produce anything on this page.

## Reproduce

```bash
python -m evaluation.calibration evaluation/dbo_lakehouse/dbo_eval_results.json \
    --baseline A --candidate B
python -m evaluation.calibration evaluation/dbo_lakehouse/dbo_eval_results.json \
    --baseline B --candidate A
python -m pytest evaluation/calibration/tests -q
```

## The split

| cause | arm A | arm B | total | fixable? |
|---|---|---|---|---|
| `no_value` — submitted without an answer | 1 | 8 | **9 (53%)** | yes |
| `unserializable` — right number, lost in transport | 2 | 1 | **3 (18%)** | yes, fixed |
| `reasoning` — genuinely reached a different answer | 3 | 2 | **5 (29%)** | no |
| | 6 | 11 | 17 flipping questions | |

Mechanical share: **arm A 50%, arm B 82%, combined 71%.**

The arm with the *worse* floor has the *higher* mechanical share. Its extra noise
is not extra nondeterminism — it is extra breakage.

## Cause 1 — `unserializable` (fixed)

`np.float64` subclasses Python `float`; `np.int64` and `np.bool_` subclass
nothing. Both therefore missed the native-scalar branch of `freeze()` and were
emitted as `{"__serializable__": false}`. `df["qty"].sum()` on an integer column
— one of the most common expressions in the product — returned a correct number
in a form that read back as unusable.

Four trials, **all four graded wrong while carrying the right number**, twice
with the *identical* value that a sibling repetition got credit for:

```
A q09: np.int64(93257855) -> wrong     93257855 -> right
A q13: np.int64(334)      -> wrong     334      -> right
```

Fixed in core (PR #79) by unwrapping 0-dimensional array scalars, duck-typed on
`ndim`/`shape` so numpy stays an optional dependency.

**Classification: universal mechanism.** JSON serialization of numeric scalars;
no domain content.

## Cause 2 — `no_value` (not fixed; needs a decision)

Nine flipping questions involve a run that submitted no answer at all. Ten of the
eleven such trials are in arm B.

The failures cluster hard in long contexts:

| arm B trials | n | mean prompt tokens | mean turns | mean seconds |
|---|---|---|---|---|
| answered | 64 | 15,341 | 4.5 | 33.1 |
| no value | 10 | **22,987** | **6.8** | **47.0** |

Seven of the ten wrote SQL, filled in `units` and `grain`, and submitted with
`value: null` — they *described* the answer instead of computing it. Several said
so in the status field: `missing_execution`, `could_not_execute_prediction`.

**Seven of those ten were never flagged**: `failure_reason: None`,
`submitted: true`, `ok: true`. Two were honest abstentions and one was caught by
output validation; the rest were recorded as successful, completed tasks.

The status field also shows no contract at all — `ok`, `final`, `complete`,
`ABSTAIN`, `abstain`, and in one case `overdue`, which is the *filter condition*
from the question rather than a status.

### This is a harness gap, not a core defect

`fabric_rlm.validators.assert_keys` already rejects a null value:

```python
>>> assert_keys("value")({"value": None})
AssertionError: SUBMIT payload is missing required keys: ['value']
```

The dbo harness passed **no `output_validator` at all**. The safeguard shipped
with the library and went unused, so a null answer was never re-prompted.

**Proposed fix — harness.** Pass
`output_validator=assert_keys("value")` (chained with the existing analytical
validators). Expected effect: the seven silent nulls become bounded re-prompts
instead of recorded successes.

**Proposed fix — universal mechanism, separate and smaller.** A declared output
field submitted as null is currently indistinguishable from success in the run
record. Whatever the retry policy, it should not be reported with
`failure_reason: None`. This is an observability gap in core, not a domain rule.

**Not proposed:** any domain-specific core patch.

## What this changes about the A-vs-B conclusion

Arm B was read as "learning costs 12 accuracy points." Two corrections:

1. The −12 pp delta is inside arm A's own 12 pp floor — already overturned by
   calibration, and scorer-dependent (it survives at −16 pp under the strict
   `full` scorer).
2. The residue is **not** a reasoning regression. Arm B has *fewer* reasoning
   flips than arm A (2 vs 3). What learning changed is the completion rate: it
   inflated prompts ~60% (9,585 → 15,341 tokens on answered trials), and runs
   that fail do so at ~23k tokens.

So the honest statement is: **learning did not make the system reason worse; it
made it run out of room more often.** That points at context budget and the
missing null-answer guard, not at lesson quality.

## Effect of the serialization fix on the recorded data

Re-grading the four affected trials as correct (their numbers match the
reference):

| | observed | corrected |
|---|---|---|
| arm A accuracy per rep | 80 / 88 / 76 | 88 / 92 / 76 |
| arm A mean | 81.3 | 85.3 |
| arm A flip rate | 24% | 16% |
| arm B mean | 70.7 | 72.0 |
| arm B flip rate | 44% | 40% |

Stated honestly: this **raises accuracy and lowers the flip rate, but does not
lower the spread** — arm A's range widens from 12 to 16 pp because the
corrections land unevenly across repetitions. With three repetitions a range is
a very unstable estimator, so neither 12 nor 16 should be quoted as precise.
This is a correctness fix that removes one cause of disagreement, not a fix that
makes the measurement quiet.

## Consequence for the evaluation

Do not start the two blocked live runs until:

1. the harness sets an `output_validator` that rejects a null answer, and
2. the harness records ≥3 repetitions of an unchanged baseline, so every
   reported delta can be quoted against a floor measured in the same run.

Without (1), roughly 7% of trials are silent non-answers counted as completed
tasks. Without (2), any single-run delta under ~16 pp is unfalsifiable.
