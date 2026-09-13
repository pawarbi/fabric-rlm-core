# Noise-floor calibration — is it worth adopting?

**Question.** Prime Intellect calibrate their PR benchmarks by running `main`
against itself before comparing anything to it. Is that worth adding to this
evaluation harness?

**Answer: yes, and it changes a conclusion we already published.**

The test cost nothing to run. The existing `dbo_eval_results.json` is a
2 arms × 25 questions × 3 repetitions design, and the three repetitions *within
one arm* are already a same-versus-same measurement. The noise floor was sitting
in data we collected weeks ago and never looked at from this angle.

---

## 1. The measured noise floor

Arm A is the no-knowledge configuration. Nothing about it changes between
repetitions — same commit, same model, same questions, same limits.

| metric | rep 0 | rep 1 | rep 2 | noise floor |
|---|---|---|---|---|
| Accuracy | 80 pp | 88 pp | 76 pp | **12 pp** |
| Abstention rate | 0 pp | 0 pp | 0 pp | 0 pp |
| Turns / question | 4.76 | 4.88 | 5.64 | 0.88 |
| Latency / question | 20.78 s | 17.64 s | 22.45 s | 4.81 s |
| Tokens / question | 9,470 | 9,360 | 10,780 | 1,417 |

**An unchanged configuration swings 12 accuracy points on its own.**

Arm B is worse: 64 / 80 / 68 pp, a **16 pp** floor.

Per-question instability, which needs no threshold at all:

- Arm A: **6 of 25 questions (24%)** return a different verdict across identical runs
- Arm B: **11 of 25 questions (44%)**

## 2. What that does to the published comparison

```
  metric                    baseline  candidate     delta   point           calibrated
  ------------------------------------------------------------------------------------
  Accuracy                        80         68       -12   regression      no_clear_change   <-- OVERTURNED
  Abstention rate                  0          4        +4   no_clear_change no_clear_change
  Turns per question            4.88       4.92     +0.04   no_clear_change no_clear_change
  Latency per question         20.78      35.81    +15.03   regression      regression        <-- survives
  Tokens per question           9470  1.715e+04     +7685   regression      regression        <-- survives
```

`point` is what a report concludes from its headline numbers alone.
`calibrated` applies the floor measured above.

**The accuracy finding does not survive.** "Learning cost 12 accuracy points" is
the same size as the swing arm A produces against itself. It was never a result.

**The cost findings do survive, comfortably.** Latency +15.03 s against a 4.81 s
floor and tokens +7,685 against a 1,417 floor are roughly 3× and 5× the noise.
Learning genuinely makes runs slower and more expensive. That claim is now
*stronger* than before, because it is stated against a measured floor instead of
asserted from two numbers.

This is the property that makes the tool trustworthy: it does not simply
dissolve every finding.

## 3. The conclusion is scorer-dependent — and that is the point

| scorer | delta | floor | verdict |
|---|---|---|---|
| `analytic` (right number) | −12 pp | 12 pp | overturned |
| `contract` (right reporting basis) | −12 pp | 12 pp | overturned |
| `full` (number **and** basis, no hazard) | −16 pp | 12 pp | **survives** |

Under the strictest definition the accuracy regression is real. Under looser
definitions it is noise. A single headline number concealed that entirely.

The honest statement is: *learning does not measurably change whether the number
is right, but it does measurably degrade whether the whole answer is right, and
it reliably costs more time and tokens.* No point estimate could have said that.

## 4. What this implies for the evaluation record

Two claims in the existing record need to be re-stated:

1. **Cold accuracy "~90%, n=2" (22/24, 23/24).** With a 12 pp floor, n=2 cannot
   support a point estimate. The supportable claim is a range.
2. **Any A-vs-B accuracy conclusion** in `FIX_REGISTER.md` §0 must carry the
   floor, or be withdrawn.

The two blocked live runs should not be started until the harness records enough
repetitions to compute a floor, or their results will be unreadable in the same
way.

## 5. Cost

Zero additional inference. The floor came from repetitions we had already paid
for. The only ongoing requirement is that **every future run keeps at least three
repetitions of an unchanged baseline arm**, which the existing design already does.

Three repetitions is the minimum that yields a floor at all, and the module is
deliberately pessimistic there: below four samples it reports the full range
rather than an interquartile range, because a quartile of three points is false
precision.

---

## Running it

```bash
python -m evaluation.calibration evaluation/dbo_lakehouse/dbo_eval_results.json \
    --baseline-arm A --candidate-arm B --scorer analytic

# robustness checks
python -m evaluation.calibration evaluation/dbo_lakehouse/dbo_eval_results.json --baseline-arm B --candidate-arm A
python -m evaluation.calibration evaluation/dbo_lakehouse/dbo_eval_results.json --scorer full
```

Raw report: `evaluation/calibration/dbo_calibration.json`.

Tests: `python -m pytest evaluation/calibration/tests -q` (23 passing, pure
arithmetic, no model calls).

## Recommendation

**Adopt.** Add the calibrated floor to every arm-versus-arm claim, and publish
the flip rate alongside every accuracy number. Report an accuracy that flips on
24–44% of questions as a range, never as a point.

The one thing *not* to copy from prime-agent: their refinement stores
`evidence: proposal.rationale` — the model's own self-authored justification,
with no provenance or fingerprint. This harness's evidence model is stronger and
should stay as it is.
