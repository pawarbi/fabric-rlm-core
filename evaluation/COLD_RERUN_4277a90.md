# Cold re-run against the PR #75 merge target

Artifact: **`4277a90`**, the current head of `feat/generalize-learning` (PR #75).
Build fingerprint **`1e40c289a9d954cfcdc099812779aff59ca1c6d9cbbcdfa8500d29b98ea32090`**,
printed by the notebook at import and matching the wheel built locally from that
commit. This is the first run in the evaluation whose results name the exact bytes
executed; see `PROVENANCE_CORRECTION.md` for why earlier runs did not.

Config: `MODEL=z-ai/glm-5.3-flash`, `LEARN=False` (arm A), `MAX_TURNS=22`,
`WORKBOOK_DUTY=True`, `SKILLS=` (library default), dbo schema of
`da_agent_tests.Lakehouse`, 24 questions, python 3.12.12, dspy 3.2.1.

## 1. Headline

| measure | result |
|---|---|
| tasks completed without error | **24/24** (`ok=True`, no crash, no timeout) |
| **reasoning correct** | **22/24 = 91.7%** |
| **correct in the delivered workbook** | **20/24 = 83.3%** |
| total turns | 192 |
| wall clock | 28 min |

The 2-question gap between reasoning and delivery is the important result, and it
is not a rounding artifact. **Two questions were answered correctly and then lost
by the workbook write path.**

## 2. The two genuine reasoning failures

| q | asked | reference | answered | nature |
|---|---|---|---|---|
| q12 | % of total `api_calls` attributable to premium features | 19.7798 | 30.5062 | numeric — wrong denominator or wrong feature set |
| q16 | **which** `company_size` segment has the highest mean `mrr` | `enterprise` | `3642.94` | **answer-shape** — returned the measure instead of the requested entity |

q16 is the more interesting of the two. The question asks *which segment*; the
agent computed the correct-looking statistic and returned the value rather than
the entity. Nothing in the run marked this as a mismatch, because the workbook
`Units` cell was filled in confidently as "USD (monthly recurring revenue per
subscription)". This is a **confidently wrong answer**, not an abstention.

## 3. The two delivery failures — a `.task`-path defect

Both questions were answered **correctly** and recorded `ok=True`,
`published=True` in the run log:

| q | run-log answer | reference | in workbook? |
|---|---|---|---|
| q13 | `334` companies | 334 | **row absent entirely** |
| q25 | `1.213608` | 1.2136 | present, but **`Question ID` written as the integer `25`, not `"q25"`** |

Two *distinct* defects in the same path:

**q13 — the row was written, staged, and never published.** This is verifiable
from the trajectory rather than inferred. On its final working turn the agent
appended the correct row:

```
a.append(["q13", "How many companies hold more than one subscription?", 334, "companies", ...])
...
stdout: saved staged file; answers rows: 13 | data rows: 20 | evidence rows: 43
```

Compare the same step on q14, which delivered:

```
stdout: published: /tmp/fabric-rlm-files-.../rlm_answers.xlsx | answer: 2616.79164 | answers rows: 13
```

q13 printed **`saved staged file`**; q14 printed **`published:`**. q13 then
submitted without ever promoting the staged file. The next question loads
`/tmp/workbook_in.xlsx`, which is the last *published* workbook — q12's — so
q13's row was never carried forward. That is exactly why the byte count does not
move: q12 `bytes=15497`, q13 `bytes=15497`, q14 `bytes=16159`.

**The run nevertheless recorded `ok=True` and `published=True` for q13.** The
status flag does not reflect whether the workbook was actually published. That is
the defect: not a lost write, but an unpublished one reported as published.

A likely contributing factor: q13 carried **`gate=11`**, the highest count of
catalog-gate rejections in the run (finding F11), and consumed 14 turns and 331 s
— by far the most expensive question. The turn budget went into rejected queries,
and the publish step was the casualty.

**q25 — the identifier lost its type.** The answer is correct and present, but the
key is the integer `25` where every other row holds the string `"q25"`. Any
consumer joining on question id drops the row.

The workbook has 23 data rows for 24 questions, and only 22 of them carry a
well-formed id.

**Why this matters more than the accuracy number.** The library's own reported
status was clean: 24/24 `ok=True`, `published=True` on every question. Nothing in
the run surfaced either defect. A user reading the run log would conclude all 24
questions were delivered. The failure is silent, and it corrupts the artifact the
task exists to produce.

## 4. Did the F14 repair cause any of this?

**No, and this is checkable rather than argued.** The repair changes
`_result_rows`, which is reachable only through `execute_registered_operation`
inside `_prepare_registered_operation`, which returns early when
`self._knowledge is None`. The cold arm binds no knowledge package.

Scanning the 840 KB run log for the entire registered-operation vocabulary:

```
knowledge_mode           0 occurrences
operation_execution      0 occurrences
registered_operation     0 occurrences
result_bound             0 occurrences
```

The code path the fix touches never executed. The fix is **provably inert for
arm A**; it targets the `.learn` path, where F14 actually fired.

## 5. Run-to-run variance — a caveat on every single-run number here

Same questions, same model, same settings, different commit but with the changed
code proven unreachable:

| | reasoning correct | delivered |
|---|---|---|
| prior cold run (`bd924bc`) | 23/24 | 23/24 |
| this cold run (`4277a90`) | 22/24 | 20/24 |

Five verdicts moved: q19 FAIL→PASS, q12 PASS→FAIL, q16 PASS→FAIL, and q13/q25
PASS→lost. Since the only code delta cannot execute in this arm, the reasoning
movement is **sampling variance**, and the delivery movement is an
**intermittent** workbook defect — it did not fire at all in the prior run.

Two consequences:

1. Single-run accuracy for this system should not be quoted to a tenth of a
   percent. The observed cold reasoning range is 22–23/24 across two runs; the
   honest statement is "roughly 90%, n=2 runs, not 91.7%".
2. The workbook defect did not fire in the prior run, so it is **not
   deterministic**. Two runs establish that it can fail; they establish nothing
   about how often. No rate should be inferred from n=2, and the prior run was on
   different bytes with a different question ordering. The practical consequence
   stands regardless: a defect that does not fire every time will pass a one-off
   acceptance test.

## 6. What this says about the merge

- **General execution capability holds.** 24/24 completed, no crashes, no
  timeouts, no bounded-recovery failures, on a real Fabric lakehouse over 24
  genuinely complex questions with a non-frontier model. The `.task` path is doing
  the analytical work.
- **The previously-made fixes are sound.** `53e6cbb` (signed and exponent-aware
  literals, AST negation handling, magnitude comparison) and `a736098` are
  universal mechanisms with no domain coupling. `bd924bc` was sound in intent but
  incomplete; `4277a90` completes it.
- **A new critical `.task` finding blocks a clean bill of health**: the deliverable
  loses correct answers silently, and self-reported status does not reveal it.
- The findings already recorded — F11 (comment-marker SQL gate), F15, F16, the
  `nrr`/`grr` leak — remain present and unaddressed at this head.

Nothing here argues against merging the `.task` work; it argues against merging it
with a "generalizes learning" claim attached, and against treating
`published=True` as evidence of delivery.

## 7. Reproduction

```
# build and stamp
cd fabric-rlm-core-pr75 && python -m build --wheel --outdir dist_eval
python fabric_upload/upload_assets.py

# deploy and run
python fabric_upload/make_notebook.py
python fabric_upload/notebook.py update --workspace-id 82ad2591-974a-4ad4-ace6-e24879274a4b \
    --notebook-id 2935d52f-4574-443d-a8d0-552936a7445a --from-file rlm_excel_full.ipynb
python fabric_upload/notebook.py execute --workspace-id 82ad2591-974a-4ad4-ace6-e24879274a4b \
    --notebook-id 2935d52f-4574-443d-a8d0-552936a7445a \
    --parameter RUN_TAG=cold-4277a90-full --parameter MODEL=z-ai/glm-5.3-flash \
    --parameter LEARN=False --parameter MAX_TURNS=22 --parameter WORKBOOK_DUTY=True \
    --parameter SKILLS= --parameter OPENROUTER_API_KEY=... --wait --timeout 5400

# grade (references never leave this machine)
python fabric_upload/fetch_binary.py \
    Files/rlm_excel_eval/out/cold-4277a90-full/rlm_answers.xlsx cold_4277a90.xlsx
python fabric_upload/grade_rlm_workbook.py cold_4277a90.xlsx
```

Note `fetch_binary.py`: the pre-existing `read_files.py` decodes to text and
silently corrupts an `.xlsx` (21321 "chars" against 22415 bytes; openpyxl rejects
it with "Bad offset for central directory"). Workbooks must be fetched as bytes.
