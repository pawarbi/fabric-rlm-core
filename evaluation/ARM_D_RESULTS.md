# Arm D — `learn(declared=…)` on the `dbo` lakehouse: results

**Run:** `glm-declared-1`, 24 questions, `z-ai/glm-5.3-flash`, `skills=[]`,
`workbook_duty=True`, `MAX_TURNS=22` — identical to arms A and B.
**Package:** `lessons_total=32, lessons_active=32, lessons_declared=32`,
fingerprint `ddebb075…` (identical to the smoke run, so the package is
deterministic), frozen before question 1.
**Core frozen:** `git diff b5226712 -- fabric_rlm` empty.

---

## 0. The confound, stated first

Arm D is **not** evidence that `learn()` works. Arm B is `learn()` as shipped and
produced **0 lessons**. Arm D is `learn()` plus 32 facts I computed *outside the
library*, with pandas and deltalake, and injected through `declared=`. The
library cannot produce these facts itself — F12 established that `learn()` reads
zero rows.

So arm D measures **"what would learning be worth if the profiler read data?"**
It is evidence for a proposed fix, not for the current mechanism. The gate
verdict on `.learn()` as shipped is unchanged and remains **FAIL**.

## 1. Pre-registered result

The endpoints below were registered in `compare_arm_d.py` *before* the run
finished. Primary is **D vs B** — both arms run `learn()` and answer from a
frozen package, so they differ in one thing only.

**Correction applied:** q13 crashed in arm B (`turns=0`, `tokens=0`). A 0→9
"increase" is an artifact of the crash, not a measurement, so the pair is
excluded from the paired tests and reported separately.

### PRIMARY — paired sign test, 23 questions

| metric | D better | D worse | tie | p |
|---|---|---|---|---|
| turns | 11 | 10 | 2 | **1.0000** |
| tokens | 11 | 12 | 0 | **1.0000** |

Totals: turns 228 → 218 (**−4.4%**), tokens 1,777,338 → 1,845,445 (**+3.8%**).

**Null result on both.** The −4.4% turn total is carried by a few large swings
(q19 9→3, q24 8→3, q22 12→7) and cancelled by equal swings the other way
(q05 7→14, q25 7→12). The sign test is unambiguous: there is no systematic
turn or token effect.

Recall the earlier B-vs-A comparison was **20+/3−, p=0.0005** on tokens. That is
what a real effect looks like in this harness. This is not one.

### SECONDARY — accuracy (reported, not headlined)

| arm | PASS | rate | NO-ANSWER | FAIL (confidently wrong) |
|---|---|---|---|---|
| A cold | 22/24 | 91.7% | 0 | 2 (q16, q19) |
| B learn | 20/24 | 83.3% | **2** (q08, q13) | 2 (q16, q19) |
| D declared | 21/24 | 87.5% | **0** | 3 (q16, q19, **q21**) |

D recovers 1 point over B and remains 1 below cold. With n=1 per question at
temperature 1.0, this is not a distinguishable difference and is **not** claimed
as one.

## 2. The result that is not noise

Accuracy moved within noise; **task completion did not**.

> **Arm B abstained twice. Arm D abstained zero times. 24/24 questions
> produced an answer.**

The brief asks that abstentions and timeouts be counted as *incomplete tasks*
and that confident wrong answers be reported *separately*. Doing so:

- **incomplete tasks:** B = 2 → D = **0**
- **confidently wrong:** B = 2 → D = **3**

Both of B's abstentions were fixed, and one new wrong answer appeared. This is a
genuine trade, not a free win, and it is the honest headline: declared structural
facts converted two "I cannot answer this" outcomes into one right answer and one
wrong one.

**q13 specifically corroborates F14.** In arm B, q13 hit the truncated-result
branch at `knowledge_execution.py:284`, which raises a bare `ValueError` and
kills the task instead of falling back. Verbatim from
`progress-glm-learn-1.log`:

```
23:49:32  [13/24] q13 ok=False ... turns=None gate=None 12.0s
23:49:32     ERROR ValueError: operation result was truncated
23:49:32     TRACE runtime.py", line 1519, in _prepare_registered_operation
                 execution = execute_registered_operation(
               knowledge_execution.py", line 869, in execute_registered_operation
                 rows = _result_rows(raw_result, operation)
```

*Why this record shows `turns=None` and no trajectory* — worth stating, because
it reads at first glance like an infrastructure failure rather than an F14
death. The operation is prepared in `_prepare_registered_operation` **before the
agent loop begins**. `_result_rows` raises there, so the task dies during
preparation: no turn ever executes, no trajectory is recorded, no tokens are
billed, and the whole thing is over in 12 s. Zero turns is not evidence that
nothing analytical happened — it is the *signature* of F14, which kills at
prep time rather than degrading inside the loop. (`turns=None` is coerced to 0
when tabulating, which is why it must be excluded from the paired tests in §1.)

In arm D the agent had the
invoice→payment fan-out declared (9,204 payment rows over 7,584 distinct
`invoice_id`), wrote a pre-aggregated query, and never produced an oversized
result. F14 is a real bounded-recovery defect, and arm D routed around it — it
did not fix it. **The defect stands.**

## 3. Mechanism check — declared facts did *not* displace rediscovery

Registered mechanism test: if declared facts substitute for exploration, the
agent should stop calling `list_sources()`.

| arm | `list_sources()` in trajectory |
|---|---|
| B learn | 22/24 |
| D declared | 21/24 |

One question's difference. **The agent read the declared facts and then went and
looked anyway.** This explains the token result: declared context is *added* to
the prompt without *removing* any exploration, so tokens rise (+3.8%) while turns
stay flat.

This is the sharpest negative finding in arm D, and it is a finding about
retrieval, not about the facts themselves: 32 active, high-confidence,
source-declared lessons changed orientation behaviour by one question out of 24.

## 4. Contamination control

`build_declared.py` derives every fact from `profile.json` alone, and:

- excludes `samples` and `min`/`max` wholesale — value lists and extrema are the
  statistics most likely to answer a question outright;
- checks every rendered number against all 40 reference and hazard values and
  drops any that collide. **The filter fired**: `usage_logs.rows = 661734` is
  itself a reference answer and was removed automatically.

**Stated limitation, unprompted:** the declared block includes the
invoice→payment fan-out ratio, and fan-out is precisely the hazard that ground
truth flags for several questions. I computed it independently from the profile
and never copied it from the answer key, and a fan-out ratio is not an answer to
any question. But a reader may reasonably regard telling the agent *where the
trap is* as a softer form of help than telling it the answer. Arm D should be
read as an **upper bound** on what data profiling can buy.

## 5. Verdict against the user's gate

> Gate: `.learn` accuracy ≥ cold, in fewer turns or on novel/complex questions.

| | accuracy vs cold | turns | verdict |
|---|---|---|---|
| B — `learn()` as shipped | 83.3% vs 91.7% ✗ | +21% ✗ | **FAIL** |
| D — `learn()` + declared profile | 87.5% vs 91.7% ✗ | −4.4%, p=1.0 ✗ | **FAIL** |

Arm D narrows the gap but does not close it, and the turn improvement is not
statistically distinguishable from zero. **The gate fails in both configurations.**

What arm D *does* establish, and what should carry into the fix list:

1. Feeding real data statistics into `learn()` **eliminates abstention** (2 → 0)
   and fixes the F14 crash path in practice.
2. It does **not** reduce exploration (`list_sources()` 22/24 → 21/24), so it
   costs tokens without saving turns.
3. Therefore profiling buys **completeness, not efficiency** — consistent with
   the independent rediscovery-trace conclusion that friction (17% of turns),
   not orientation (11%), is where the turn budget goes.

## 6. Fix classification

| # | change | class |
|---|---|---|
| D-1 | `learn()` should profile rows — counts, grain, fan-out, null density — instead of reading only schema and fingerprints | **universal mechanism** |
| D-2 | Allow `grain`/`period_column`/`units` to be scoped per table; `_field_types` currently flattens all 21 lakehouse tables into one unqualified namespace, so only free-text `definitions`/`notes` scale to a multi-table alias | **universal mechanism** |
| D-3 | Retrieval should let a high-confidence declared fact **suppress** the corresponding rediscovery step, not merely accompany it | **universal mechanism** |
| D-4 | Fix F14: the truncated/column/byte branches raise bare `ValueError` and kill the task where the row branch falls back gracefully | **universal mechanism** |

D-3 is the one arm D newly justifies, and §3 is its evidence.

## 7. Reproduction

```bash
$env:RLM_OR_KEY = "<key>"
python launch_declared.py glm-declared-1          # ~45 min, 24 questions
python read_files.py "Files/rlm_excel_eval/run_log-glm-declared-1.json" \
    --out run_log_glm_declared.json
python grade_analytical.py run_log_glm_declared.json analytical_grade_glm_declared.json
python compare_arm_d.py run_log_glm_declared.json
```
