# Fix register — every proposed change, classified

**No code was changed.** Core was frozen for the whole evaluation
(`git diff b5226712 -- fabric_rlm` is empty across all 36 commits; the only
directories touched are `evaluation/` and `tests/`, and the 3 test files are
additions). Everything below is a **proposal with a reproduction**, not a patch.

Two independent classifications are given, because they answer different
questions:

- **Severity** — how much damage does it do?
- **Class** (required by the brief) — *universal mechanism* vs *source metadata*
  vs *optional domain skill*. This is the "don't put domain rules in core" test.

---

## 1. Critical — wrong or lost results that look successful

These are the ones where the system reports success while something is broken.

| id | defect | evidence | class |
|---|---|---|---|
| **F14** | Of the four result-bound checks in `_result_rows`, only the row bound raises the graceful `OperationResultTooLarge` and falls back. Truncation (`:284`), column bound (`:320`) and byte bound (`:329`) raise **bare `ValueError`**, which `runtime.py:1568` re-raises and the task dies. | Observed live: arm B q13, `ValueError: operation result was truncated`, killed inside `_prepare_registered_operation` **before the agent loop**, so `turns=None`, no trajectory, 12 s. | **universal mechanism** |
| **F16** | `assert_not_clarification_request` detects a deferral by matching English opener phrases. A non-English "please confirm / I need more information" matches nothing, so **no assert fires and the answer passes** — it fails *open*. The guard against a model dodging the question only works in English. | `_CLARIFICATION_OPENERS` (`validators.py:373-378`): 4/4 English detected, 0/4 German-Spanish-French-Italian detected. Docstring says "Universal … because clarification openers are domain-agnostic **English**." | **universal mechanism** |
| **F9** | A derived artifact (the Excel workbook) can be lost or not appended while the task still reports `ok=True`. | Phase 2; model-dependent — GLM maintained the workbook across 24 updates, gpt-4.1-mini did not. | **needs evidence first** — test the bundled `excel_modify` skill before any core change |

**F14 and F16 are the two I would fix first.** F14 turns a recoverable
size-limit into a dead task; F16 lets an evasion be scored as an answer.

---

## 2. High — learning is ineffective, and silently so

| id | defect | evidence | class |
|---|---|---|---|
| **F12** | `learn()` **reads zero rows**. `DeltaDirectoryAdapter` uses `DeltaTable(..., without_files=True)` by design; `LakehouseSourceAdapter` consumes only the catalog. So on a Delta lakehouse it produced **0 lessons**, and said nothing about it. | Source-verified at `knowledge_lakehouse_sources.py:241,403`; grep for `null_count\|distinct\|min_value\|skew\|row_count` finds only test fixtures. | (a) report coverage → **universal**  (b) column meaning → **source metadata** (`declared=`) |
| **F15** | Semantic-model lesson nomination is decided by one English regex, `_DERIVED_TIME_MEASURE`. English 10/10; the *same measures* in German/Spanish/French/Italian **0/7**; ordinary statistical English **8/8 false positives**. Two false positives are live (`Avg Delivery Variance`, `Scenario Variance` were told they require `period_context`). Scope: all 7 lessons were `candidate`, `active=0`, so this gates *what learning may propose*, not an observed answer. | `repro_measure_naming.py` — pure function, no Fabric/key/network, exits non-zero if it stops reproducing. | split — see §5 |
| **D-3** | Declared facts are **added to** the prompt without **removing** any exploration. 32 active, high-confidence, source-declared lessons moved `list_sources()` usage only 22/24 → 21/24. Hence +3.8% tokens for no turn saving. | Arm D mechanism check, pre-registered before the run. | **universal mechanism** |
| **OP-1** | 21 of 24 tasks paid an operation-selection LM call only to be declined. The 3 where an operation fired were correct 3/3 and nearly free. Memoise the decision. | Phase 2 split by operation-fired. | **universal mechanism** |

---

## 3. Medium — correctness-adjacent and ergonomics

| id | defect | class |
|---|---|---|
| **F11** | Query-gate rejections are indistinguishable across paths, and `--` is matched **inside string literals**, so a legitimate query containing `--` is rejected as a comment. | **universal mechanism** |
| **D-2** | `_field_types` flattens all 21 lakehouse tables into **one unqualified column namespace**, so `grain` / `period_column` / `units` cannot be scoped per table. Only free-text `definitions` / `notes` scale to a multi-table alias. | **universal mechanism** |
| **F15-4** | Silent zero-coverage: "no derived measures detected" is indistinguishable from "this model has none". A user cannot debug silence. | **universal mechanism** |
| **V-1** | `verify.py:95` `_LEADING_CONJUNCTION` strips `and` / `&` but not `und` / `y` / `et`, so `"und Contoso"` ≠ `"Contoso"` — a scoring artifact, not a real disagreement. | **universal mechanism** |

---

## 4. Rejected — deliberately not proposed

| proposal | why rejected |
|---|---|
| Add more English synonyms to `_CURRENT_PERIOD` / `_DERIVED_TIME_MEASURE` | Puts an unbounded naming vocabulary into core **and keeps the English assumption**. Treats the symptom. |
| Any domain-specific core patch | Out of scope by the brief, and none was made. |

---

## 5. The one genuine ARR leak, and how it splits

`_DERIVED_TIME_MEASURE` (`knowledge_lessons.py:54-58`) contains `nrr` and `grr`
— Net and Gross Revenue Retention — plus `retention` and `churn`. This is
subscription-business vocabulary in a runtime core regex that decides which
lessons exist. Per the brief, this is not "ARR in a comment"; it is runtime
behaviour.

| # | change | class |
|---|---|---|
| F15-1 | Move `nrr`, `grr`, `retention`, `churn` to an optional SaaS/subscription skill | **optional domain skill** |
| F15-2 | Let a source declare its derived measures / period columns | **source metadata** |
| F15-3 | Detect period-relative measures from the **DAX they reference** (`SAMEPERIODLASTYEAR`, `DATEADD`, `TOTALYTD`, `PARALLELPERIOD`) rather than from their names | **universal mechanism** |
| F15-4 | Emit a "no derived measures detected" signal | **universal mechanism** |

**F15-3 is the substantive fix.** DAX function names are invariant across
languages and already present in the model definition. On the live model it
would have produced **5 correct lessons instead of 5 correct and 2 wrong**, and
it fixes the non-English gap without adding a single vocabulary word to core.

---

## 6. Scoreboard

| class | count | ids |
|---|---|---|
| **universal mechanism** | 10 | F14, F16, F12a, D-2, D-3, OP-1, F11, F15-3, F15-4, V-1 |
| **source metadata** (`declared=`) | 2 | F12b, F15-2 |
| **optional domain skill** | 1 | F15-1 |
| **needs evidence before any change** | 1 | F9 |
| **rejected** | 2 | English-synonym expansion, domain core patches |

The distribution is the reassuring part: **almost every defect found is a
mechanism defect, not a domain defect.** Only one proposal (F15-1) requires
removing business vocabulary from core, and it is a four-word regex change.

---

## 7. What the fixes do *not* address

The `.learn` gate fails for a reason no item above repairs. Arm D supplied
`learn()` with real data statistics it cannot compute itself, and:

- accuracy 87.5% vs cold 91.7% — still behind;
- turns −4.4% at **p = 1.0**, tokens +3.8% at **p = 1.0** — null on both.

Abstentions did fall 2 → 0 while confidently-wrong rose 2 → 3. So profiling
buys **completeness, not efficiency**. Fixing F12 and D-3 is necessary for
learning to be worth its cost, but this evaluation does not show it is
sufficient.
