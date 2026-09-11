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

## 0. `.task` vs `.learn` — which path does each defect actually sit on?

Established by reading the guards, not by assumption. **This corrects the
severity ordering in §1**: the register originally ranked F14 as the top
critical item for core, and F14 turns out to be unreachable without a knowledge
package.

### Verified `.learn`-only — a plain `RLM.task()` never reaches these

| id | proof it cannot fire without a package |
|---|---|
| **F14** result bounds | `_prepare_registered_operation` (`runtime.py:1397`) opens with `if self._knowledge is None or metadata.get("knowledge_mode") != "registered_operations_available": return`. `_result_rows` is only reachable through `execute_registered_operation`, called at `runtime.py:1519` inside that guard. |
| **OP-1** operation-selection cost | Same function, same guard. |
| **F12** `learn()` reads zero rows | Only executes inside `RLM.learn()`. |
| **F15** measure-name gating | Lesson construction; `learn()` only. |
| **D-2 / D-3** declared scoping and retrieval | `declared=` is a `learn()` parameter. |

### Verified `.task` path — reachable with no knowledge package at all

| id | proof it is on the task path | severity |
|---|---|---|
| **F11** query gate | `_normalize_catalog_query` (`lakehouse.py:437`) is `LakehouseSource.query`, bound directly as a task input. No knowledge involved. | **critical** |
| **F9** ~~artifact loss~~ **WITHDRAWN** | `artifacts.py:140` `FileDestination` is indeed independent of learning, so the *reachability* claim was right. But the **defect claim was wrong**: `published=True` is a harness field (`make_notebook.py:453`, a OneLake file-existence probe), not a library status flag, and the library writes no workbook rows at all. Both q13 and q25 are model-authored. See §1 F9 and `COLD_RERUN_4277a90.md` §3/§3a. | **withdrawn — not a library defect** |
| **F16** clarification guard | `validators.py:382`, on the verification path — **but exported only**, never applied by default. Reachable only when a caller passes it to `validators=`. | critical *when used*, not default-path |
| **V-1** conjunction stripping | `verify.py:95`, used by the public `verified_task` / `answers_agree`. | **low** — see below |

### Three severity corrections I owe against my own earlier register

- **F9 is withdrawn entirely** (see §1). It was never a library defect — the
  `published` flag it rested on is the harness's own file-existence probe.
- **F14 is not a `.task` defect.** It is critical, but it belongs to `.learn`.
- **V-1 is low, not medium.** `answers_agree` documents that it *"errs toward
  disagreement: a false 'disagree' costs one reconciliation run, a false
  'agree' costs correctness"* (`verify.py:131`). Failing to strip `und`/`y`/`et`
  produces a false *disagreement*, which triggers an extra reconciliation run
  and costs latency — the safe direction, by design.

### A standing check this evaluation learned the hard way

Twice now a **measurement artifact** was filed as a library property: first the
freeze proof verified against the wrong tree (`PROVENANCE_CORRECTION.md`), then
F9's `published` flag. Before filing any finding as a library defect, confirm the
signal **originates in library code** — locate the emitting line in `fabric_rlm/`,
not in the harness, the notebook, or the model's own output.

### The result

**The `.task` critical list is one item: F11 — now fixed (PR #77).**

That is the substantive finding of this partition, and it is a good one: it is
consistent with cold `.task` scoring 91.7% on 24 unseen complex questions with
zero fan-out traps. Core execution is in better shape than the undifferentiated
register implied — most of what was found belongs to the learning path.

### F11, demonstrated

`_normalize_catalog_query` rejects on substring containment with no awareness of
string literals, and emits **one identical message for five different causes**:

```
PASS    SELECT * FROM t
REJECT  SELECT * FROM companies WHERE name = 'Smith--Jones'   <-- legitimate
REJECT  SELECT * FROM t WHERE code = 'A/*B'                   <-- legitimate
REJECT  SELECT * FROM t -- drop everything                    <-- correct
REJECT  ''                                                    <-- correct
REJECT  DELETE FROM t                                         <-- correct
all five: "LakehouseSource.query requires a read-only catalog query."
```

This is **not merely a turn-cost bug**. A row whose data contains `--` or `/*`
cannot be filtered on at all, and no rephrasing rescues it, so a legitimate
analytical question about such data is unanswerable. The identical message is
the second half of the defect: the agent cannot tell "your query contains a
comment marker" from "you used DELETE" from "your query is too long", which is
consistent with the 35 gate rejections observed across 7/24 questions.

#### F11, resolved — PR #77, branch `fix/catalog-query-literals`

Both halves fixed, off `main` at `77bd8b4`, TDD with the tests confirmed red
first. Classified **universal mechanism**: nothing in the fix encodes a dataset,
a naming convention or a business concept — it corrects how SQL text is read.

1. **Literal-aware scanning.** A new `_mask_quoted_spans()` blanks the inside of
   string literals and quoted identifiers, and the marker scan runs on that copy,
   so a marker is syntax only where it is syntax. Quoting the scanner does not
   parse — an unterminated quote, an `E'...'` escape string, a `$$...$$`
   dollar-quoted string — **fails closed**. The executed SQL is unchanged:
   `_normalize_catalog_query` still returns the original, and the mask is a
   separate string used only for scanning.
2. **Per-cause messages.** `_query_error(reason)` carries a cause at each of the
   eleven raise sites, including an unauthorized table that now lists the names
   the query may read. The `"read-only catalog query"` prefix is unchanged.

Verified after the fix:

```
PASS   SELECT * FROM t WHERE owner = 'Smith--Jones'
PASS   SELECT * FROM t WHERE sku   = 'XY--01'
PASS   SELECT * FROM t WHERE code  = 'A/*B'
PASS   SELECT "a--b" FROM t
BLOCK  SELECT * FROM t -- drop everything          : SQL comment markers are not allowed outside string literals
BLOCK  SELECT * FROM read_csv_auto/**/('C:/...')   : SQL comment markers are not allowed outside string literals
BLOCK  DELETE FROM t                               : a catalog query must begin with SELECT or WITH
BLOCK  ''                                          : the query is empty
BLOCK  SELECT * FROM t WHERE a = 'oops             : the query has an unterminated string literal
```

Security posture unchanged — every pre-existing rejection test still passes,
including the load-bearing `read_csv_auto/**/(...)` obfuscation, and the DuckDB
AST validation behind the gate is untouched. 17 new tests; the query file is 61
passed; the full suite excluding `tests/behavior` exits 0.

**Not re-measured.** The 35 rejections and the 331 s question were observed in
the cold run; whether the fix moves those numbers is a live-run question and is
recorded here as unmeasured, not as an improvement.

---

## 1. Critical — wrong or lost results that look successful

These are the ones where the system reports success while something is broken.

| id | defect | evidence | class |
|---|---|---|---|
| **F14** | Of the four result-bound checks in `_result_rows`, only the row bound raises the graceful `OperationResultTooLarge` and falls back. Truncation (`:284`), column bound (`:320`) and byte bound (`:329`) raise **bare `ValueError`**, which `runtime.py:1568` re-raises and the task dies. | Observed live: arm B q13, `ValueError: operation result was truncated`, killed inside `_prepare_registered_operation` **before the agent loop**, so `turns=None`, no trajectory, 12 s. | **universal mechanism** |
| **F16** | `assert_not_clarification_request` detects a deferral by matching English opener phrases. A non-English "please confirm / I need more information" matches nothing, so **no assert fires and the answer passes** — it fails *open*. The guard against a model dodging the question only works in English. | `_CLARIFICATION_OPENERS` (`validators.py:373-378`): 4/4 English detected, 0/4 German-Spanish-French-Italian detected. Docstring says "Universal … because clarification openers are domain-agnostic **English**." | **universal mechanism** |
| **F9** ~~artifact loss~~ | **WITHDRAWN 2026-09-10 — this was not a library defect.** The observation was real: on the cold re-run, q13 appended its correct row, printed `saved staged file`, and submitted without publishing, so the next question reloaded q12's workbook and the row was never carried; q25's `Question ID` was the integer `25`. The **attribution was wrong.** `published=True` is not a library status flag — it is the harness's own OneLake file-existence probe (`make_notebook.py:453` → `workbook_state()`), which reads `True` for every question because the workbook has existed since q01. And no library code writes workbook rows: `excel_artifacts.py` only ever calls `load_workbook` to read (`:71, :96-97, :146-147`), while `make_notebook.py` states "the parent harness deliberately does no Excel work at all." Both failures are **model-authored**; the harness's probe was simply too coarse to notice. | **withdrawn** — reclassified as (a) harness measurement weakness and (b) model prompt-adherence. See F9-P for the one salvageable idea. |
| **F9-P** *(proposal, not a finding)* | Nothing surfaces staged-but-never-published files at end of run. `FileDestination` already tracks `_staged_paths` (`artifacts.py:158`), so the run could report which staged files were never promoted. | Evidence **n=1** (q13). Deliberately **not implemented**: staging without publishing is legitimate (scratch files, superseded drafts), so it cannot be an error; and reporting it would not have prevented the loss. Needs a real frequency estimate first. | **proposed universal mechanism — unbuilt** |

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
| **F11** | ~~Query-gate rejections are indistinguishable across paths, and `--` is matched **inside string literals**, so a legitimate query containing `--` is rejected as a comment.~~ **Fixed — PR #77.** Severity was later raised to critical: the data is unfilterable, not merely awkward. | **universal mechanism** |
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
| **proposed, unbuilt** (evidence n=1) | 1 | F9-P |
| **withdrawn — not a library defect** | 1 | F9 |
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
