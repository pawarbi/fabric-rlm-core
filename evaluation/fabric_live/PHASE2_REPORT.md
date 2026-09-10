# Phase 2 — Can RLM author its own Excel deliverable in Fabric?

**Run tag:** `cold-full-1` · **Date:** 2026-09-09
**Environment:** Microsoft Fabric, Python 3.12.12, dspy 3.2.1 (in-pin), openpyxl 3.1.5
**Library:** `fabric_rlm` 0.6.0 wheel built from the frozen `pr-75` worktree
**Model:** `openai/gpt-4.1-mini` via OpenRouter, `max_turns=22`, `temperature=1.0`, `cache=False`
**Source:** `LakehouseSource(root=abfss://sandeep_ws@onelake.dfs.fabric.microsoft.com/da_agent_tests.Lakehouse)`, tables `Tables/dbo`
**Questions:** the same 24 used in Phase 1 (q23 withdrawn). Reference answers and grader code were never uploaded.

---

## 1. What was asked

Re-run the 25 questions **cold**, but this time have **RLM itself** create the
multi-sheet Excel and **update it after each question** — rather than my harness
authoring the workbook from RLM's answers, as in Phase 1.

## 2. Headline result

RLM **can** author, publish and incrementally maintain a formatted three-sheet
workbook in lakehouse Files — **with a capable model**. Every failure mode
originally attributed to the library turned out to be **model-dependent**.

The decisive comparison holds library, prompt, questions, environment, dspy
version, skills and limits constant and varies **only the model**:

| Metric | gpt-4.1-mini (`cold-full-1`) | **GLM-5.3-flash (`glm-workbook-1`)** |
|---|---|---|
| Analytical accuracy (run log, canonical) | 8/24 = 33.3% | **22/24 = 91.7%** |
| Workbook accuracy (final .xlsx, secondary) | ungradable — 15 rows destroyed | **23/24 = 95.8%** |
| Rows lost to workbook reset | 15/24 | **0/24** |
| No machine-readable answer | 10/24 | **0/24** |
| Fan-out hazard failures | — | **0** |
| Mean turns | 13.88 | 7.88 |
| Mean prompt tokens | 54,145 | 40,314 |
| Final workbook bytes | 7,157 (collapsed from 11,083) | **23,144, strictly monotone** |

**Two claims from the earlier draft are withdrawn:**

- "It cannot reliably maintain the workbook" — GLM maintained it across all 24
  questions with zero resets. F9 is a **model behaviour**, not a library defect.
- "The cause is `LakehouseSource.query` rejecting valid SQL" — **not
  established**. See F11: the gate is a genuine, independently reproduced
  defect, but the evidence for it *causing* the gpt-4.1-mini collapse is one
  model self-report, and GLM scored 91.7% in the same environment.

**Canonical accuracy metric:** the **run-log** score (`grade_analytical.py`).
It is available for every arm including no-workbook controls, so it is the only
figure comparable across all arms. The workbook score is **secondary** and
reported only for workbook-duty arms.

Note both GLM run-log misses (q16, q19) are **grading artifacts** — RLM named
the right category in `reasoning` while `value` held the metric. Cause is
harness defect **H5** (an under-specified output contract), **identified but not
fixed**: both GLM arms ran the same unfixed contract, so it cannot bias the
cold-vs-learn comparison. The strict scores above are reported unchanged and
deliberately **not** upgraded to 24/24.

## 3. Finding F9 — the workbook reset (NEW, high severity)

At **q16 the workbook was silently rebuilt from scratch**, discarding q01–q15.

```
q14  11,083 bytes    rlm_rowcount 14
q15  11,083 bytes    rlm_rowcount None
q16   7,157 bytes    rlm_rowcount 1     <-- reset
q17   7,646 bytes    rlm_rowcount 2
...
q25  10,446 bytes    rlm_rowcount 10
```

The final published workbook contains **only q16–q25 (9 answers)**. Fifteen
questions of completed, correct work were destroyed.

**Reproduction:** `progress-cold-full-1.log`, `run_log-cold-full-1.json`, and
`rlm_authored_full.xlsx`, all under
`Files/rlm_excel_eval/` and `out/cold-full-1/`. Re-run with
`python notebook.py execute --notebook-id 2935d52f-... --parameter RUN_TAG=<new>`.

**Expected:** each question appends one row per sheet; the workbook grows
monotonically; 24 rows at the end.
**Actual:** at q16 RLM received the 11,083-byte workbook (`workbook_in=True`),
ignored its contents, and wrote a fresh one containing only its own row.

**Why this matters more than the row loss:** every question reported
`ok=True published=True`, and RLM's own `workbook_rows_total` field reset from
14 to 1 without any error being raised. **The loss was invisible to every
success signal the library exposes.** A user running this unattended would
receive a workbook that looks well-formed and is missing 62% of the work.

**Affected behaviour:** not a library crash — a task-execution failure. The
model treats "produce the workbook" as satisfiable by producing *a* workbook.

### F9 severity DOWNGRADED — it is model-dependent

The GLM-5.3-flash arm re-ran this exact scenario — same library, prompt,
questions, environment, dspy version, limits and skills, varying **only the
model** — and produced **zero resets**: 24/24 rows retained, bytes strictly
monotone 7,488 → 23,144, `workbook_rows_total` advancing 1 → 24.

So F9 is **not a deterministic library defect**. It is a weak-model failure that
the library does not *detect*. That distinction changes the fix:

- **Not justified on this evidence:** a core append-or-fail patch in
  `artifacts.py`. One model out of two exhibits the failure; a core behavioural
  change on n = 1 model would be premature.
- **Justified:** the *detection* gap is real for every model. `published=True`
  and `ok=True` were returned while 62% of the work was destroyed. A cheap,
  domain-free improvement is to surface a warning when a published artifact is
  derived from an input artifact and is **smaller** than it, leaving the
  decision to the caller.
- **Test first:** the bundled `excel_modify` skill already teaches the exact
  discipline (§4c). If it eliminates F9 on gpt-4.1-mini, this is a **skill
  gap**, not a core defect at all.

**No core patch was made during this evaluation, per the standing constraint.**

## 4. Finding F10 — workbook duty costs turns and tokens, but NOT accuracy (REVISED)

**An earlier draft of this report claimed workbook duty caused the accuracy
drop. My own control refutes that.** The claim is withdrawn.

A control arm was run with the workbook duty removed and everything else held
constant — same model, sampling, source, turn budget, questions, and run-log
fields (`RUN_TAG=control-noworkbook-1`, `WORKBOOK_DUTY=False`).

| Metric | Phase 1 local | Fabric + workbook duty | Fabric control, no duty |
|---|---|---|---|
| Analytical accuracy | 84.7% | 33.3% | **33.3%** |
| No machine-readable answer | 1.4% | 37.5% | **41.7%** |
| Mean turns | 5.04 | 13.88 | 11.67 |
| Mean prompt tokens | 9,160 | 54,145 | 38,560 |

**Conclusion:** workbook duty costs **+19% turns and +40% prompt tokens for zero
accuracy change**. That is a real and worth-knowing overhead, but it is *not*
the cause of the accuracy collapse. The collapse tracks the **model**, not the
duty and not the environment — see §2 and F11.

This is the value of running the control: the intuitive explanation was wrong,
and only a held-constant comparison exposed it.

## 4b. Finding F11 — `LakehouseSource.query` rejects valid SQL with an
undiagnosable error (NEW, universal mechanism)

> **Severity and causal role revised.** An earlier draft called this "the actual
> cause of the Fabric accuracy collapse". **That causal claim is withdrawn.**
> The defect below is real and independently reproduced; its role in the
> gpt-4.1-mini collapse is **not established**. See "Why the causal claim was
> withdrawn" at the end of this section.

The defect surfaced in the model's own words in the gpt-4.1-mini runs:

- q05: *"**LakehouseSource.query method prohibits non-catalog queries**, thus
  unable to execute aggregate SQL to compute total MRR"*
- q18: *"Unable to execute any query on dbo.invoices due to **'read-only catalog
  query' errors from lakehouse.query**"*
- q08: *"however, **lakehouse.query disallows** the re[quired query]"*

**Reproduction — no credentials required.** `_normalize_catalog_query` is pure;
`repro_query_gate.py` exercises it directly. This is the load-bearing evidence
and it does not depend on any run:

| Query | Accepted | Message |
|---|---|---|
| `SELECT SUM(amount_due) FROM invoices` | yes | |
| `SELECT SUM(amount_due) FROM invoices -- total` | **no** | *requires a read-only catalog query.* |
| `-- compute the total\nSELECT ...` | **no** | *requires a read-only catalog query.* |
| `/* total */ SELECT ...` | **no** | *requires a read-only catalog query.* |
| `SELECT * FROM t WHERE code = 'A--B'` | **no (false positive)** | *requires a read-only catalog query.* |
| `EXPLAIN SELECT 1` | **no** | *requires a read-only catalog query.* |
| `DROP TABLE invoices` | **no (correct)** | *requires a read-only catalog query.* |

Two distinct defects:

1. **Undifferentiated error.** Seven different rejection paths
   (`lakehouse.py:446, 451, 492, 813`) raise byte-identical text. The model
   cannot repair what it cannot diagnose. This defect is proven by reading the
   code; its *behavioural* cost is discussed below.
2. **Over-broad comment detection.** `any(marker in normalized for marker in
   ("--", "/*", "*/"))` is a substring scan over the whole statement, so it
   rejects a `--` inside a **string literal**. Emitting SQL comments is one of
   the most common LLM SQL habits, making this a high-frequency trigger.

**Expected:** either accept SQL comments (they are inert once parsed), or reject
them with a message naming the violated rule.
**Actual:** silent-to-the-model rejection with no repair signal.

### Why the causal claim was withdrawn — and what replaced it

I had attributed the gpt-4.1-mini collapse (33.3%, 11.67 turns, 41.7%
abstentions) to this gate. Two things refuted that attribution:

1. **The evidence was one self-report, not a hit count.** The run log recorded
   the final answer payload only — **not** `RLMResult.trajectory`.
2. **GLM-5.3-flash scored 91.7% in the identical environment.**

I then fixed the harness (H6) to persist trajectories and count rejections
directly. The GLM learn arm gives the **measured** picture — **for that arm
only**:

| | value |
|---|---|
| Total catalog-gate rejections | **35** |
| Questions hitting the gate | **7 / 24** |
| Per question | q01:2, q02:2, **q08:13**, q15:2, q20:2, **q21:10**, q22:4 |

Within this arm, the two highest gate counts are the two highest turn counts
(q08 → 20 of 22 turns, q21 → 17). That is a **correlation inside a single arm**
and it is the strongest form the evidence supports.

**Cross-arm attribution is NOT available (harness defect H8).** H6 landed
*after* the cold arm ran, so arm A's run log has **no `catalog_gate_rejections`
field at all**. "0 vs 35" is unmeasured versus measured, not a comparison — I
cannot say whether the cold arm hit the gate 0 times or 40. Accordingly, the
earlier reading of **q08 as gate-caused turn exhaustion is withdrawn**; q08 is
reclassified as sampling divergence (see §10). The gate is a demonstrable cost
of the *system*; it is not demonstrably a cost of *learning*.

**The bounded, supported claim:** the undiagnosable message measurably costs
turns, and correlates with the worst-performing questions in the arm where it
was measured. **The unsupported claim, still withdrawn:** that it caused the
gpt-4.1-mini collapse — GLM absorbed 35 rejections and still scored 91.7%.

**Proposed fix — universal mechanism, NOT a domain rule.** Nothing here is
specific to ARR, invoices, or any business concept:

- Give each rejection path a distinct, actionable message, e.g.
  *"comment markers are not permitted in catalog queries — remove `--` and `/* */`"*,
  *"only SELECT and WITH statements are permitted; got EXPLAIN"*.
- Strip comments during normalization instead of rejecting, or detect them with
  a tokenizer that ignores string literals, removing the false positive.

**No core patch was made during this evaluation**, per the standing constraint.

## 4c. Finding F12 — `learn()` on a tabular/Lakehouse source is gated on
English column names (NEW, credential-free, goes to the core question)

This is the finding that most directly answers the original brief: *does
learned behaviour generalize, or is it specialized to a naming style?*

### What learning actually does

`knowledge_api.learn` builds lessons from exactly two places
(`knowledge_api.py:227-229`) and **nothing else** — there is no inference from
data values:

```python
lessons = {l.lesson_id: l for l in structural_lessons(package)}
for lesson in declared_lessons(package, declared):
    lessons[lesson.lesson_id] = lesson
```

The library's own comment is explicit and honest:

> *"Sources that declare nothing produce no lessons, and the package is then
> exactly what it was before learning existed."*

For a **non-semantic-model** source, `structural_lessons` delegates to
`_tabular_structural_lessons`, which can emit **exactly one kind of lesson**: a
`time_semantics` / "current period" lesson. It fires only when a column is
`boolean` **and** its name matches:

```
_CURRENT_PERIOD = (is_)?current | as_of | latest_(period|quarter|month|date|week)
```

and is promoted from `candidate` to `active` only when the name *also* matches
`_PERIOD_COLUMN = period|quarter|month|year|date|week|fiscal|day`.

### Demonstrated behavioural effect — `repro_learn_naming.py`

No credentials, no model, no network. Only the **column name** varies; the type
stays `boolean` and the meaning is identical throughout ("this row belongs to
the current reporting period"):

| naming style | column | lesson? | status |
|---|---|---|---|
| english, period word | `is_current_quarter` | **YES** | active |
| english, period word | `current_period_flag` | **YES** | active |
| english, no period word | `is_current` | **YES** | candidate |
| english, as-of form | `as_of_date_flag` | **YES** | active |
| abbreviation | `cur_qtr_flg` | **no** | — |
| abbreviation | `is_curr_per` | **no** | — |
| different convention | `IsCurrentQuarter` | **YES** | active |
| different convention | `ACTIVE_PERIOD_IND` | **no** | — |
| domain word, same meaning | `in_reporting_window` | **no** | — |
| domain word, same meaning | `open_fiscal_period` | **no** | — |
| non-English | `periodo_actual` | **no** | — |
| non-English | `aktuelle_periode` | **no** | — |

**5 of 12 semantically identical columns produce a lesson.** This satisfies the
brief's item 3 — *renaming that preserves meaning changes what is learned* —
and it is a runtime dependency, not a fixture or a comment.

**Expected:** either learn from something other than the name, or tell the
caller that nothing was learned and why.
**Actual:** silent empty package.

### Finding F13 — a corollary measured live: learning was a **no-op** here

The GLM learn smoke against the real `dbo` lakehouse:

```
LEARN package frozen: {'lessons_total': 0, 'lessons_active': 0,
                       'fingerprint': '5b56910466ced4d0...', 'learn_seconds': 5.2}
```

Zero lessons, in 5.2 s, over 10 Delta tables. This is **correct, documented
behaviour**, not a crash: the `dbo` tables declare no boolean current-period
flag, and I passed no `declared=`. But it means an arm-B "learned" run on this
lakehouse differs from cold **only** by dropping the direct source binding —
knowledge re-binds the source anyway (`runtime.py:1373`,
`bound.update(self._knowledge.bindings)`), which is why both smoke answers were
still exactly correct.

**Consequence for the evaluation:** on a plain Delta lakehouse with no declared
metadata, `RLM.learn()` has **nothing to contribute**, so an accuracy gate of
"learn ≥ cold" is measuring noise. Phase 1 saw non-empty packages only because I
supplied `DECLARED` by hand — i.e. Phase 1 measured *my* metadata, not the
library's inference.

### Fix classification — this is the brief's central distinction

| Fix | Classification | Rationale |
|---|---|---|
| "`is_current_quarter` means current period" | **source metadata** | A name cannot carry meaning reliably across conventions or languages. `declared=` already exists and is the right home. Hard-coding more synonyms into core would be adding a naming rule to core — exactly what the brief forbids. |
| **Report learning coverage**: surface `lessons_total == 0` and which sources contributed nothing | **universal mechanism** | Domain-free. Today a user on abbreviated or non-English names gets a silent no-op with no signal that `declared=` is required. |
| Extending `_CURRENT_PERIOD` with more English synonyms | **rejected** | Treats the symptom, keeps the English assumption, and moves an unbounded vocabulary into core. |

**No core patch was made during this evaluation.**

## 4d. Finding F14 — a truncated operation result aborts the task instead of
falling back to the cold path (NEW, learn-path only, universal mechanism)

**Reproduction:** `glm-learn-1`, q13. Observed once in 24 questions.

```
ERROR ValueError: operation result was truncated
TRACE  fabric_rlm/runtime.py, in _prepare_registered_operation
       raise ValueError("operation result was truncated")
```

Raised at `knowledge_execution.py:284` when a host operation returns
`truncated: True`. It reaches the bare `except ValueError` at `runtime.py:1558`,
which re-raises at `:1568`.

**The defect is larger than one condition.** `_result_rows` enforces four
sibling result bounds within ~45 lines of each other. **One** uses the graceful
exception class; the other three do not:

| line | condition | class raised | outcome |
|---|---|---|---|
| 309 | exceeds **row** bound | `OperationResultTooLarge` | **falls back** |
| 284 | result **truncated** | bare `ValueError` | **task dies** |
| 320 | exceeds **column** bound | bare `ValueError` | **task dies** |
| 329 | exceeds **byte** bound | bare `ValueError` | **task dies** |

These are the same class of condition — the host returned more data than the
contract allows — handled three different-but-identical ways and one correct
way. That asymmetry between adjacent lines in one function is the signature of
an oversight, not a design decision.

**Why the row bound survives and the others don't** comes down to handler order,
because `OperationPlanError` *is itself a `ValueError`*
(`class OperationPlanError(ValueError)`, `:64`; `class OperationResultTooLarge(OperationPlanError)`, `:68`):

```
runtime.py:1524   except OperationResultTooLarge  → fallback
runtime.py:1540   except OperationPlanError       → fallback
runtime.py:1558   except ValueError               → telemetry, then `raise` (:1568)
```

Anything raised as a subclass is caught early and degrades gracefully; anything
raised as a plain `ValueError` falls through to the re-raise. So every *other*
failure in `_prepare_registered_operation` — invalid planner response, planner
declines, unknown operation, bad parameter types — records an
`operation_fallback_reason`, returns `bound_inputs`, and lets the agent continue
on the raw sources. Only the three bare-`ValueError` bound violations kill the
task.

> Note for the record: an adversarial review of this finding asserted that
> `OperationResultTooLarge` "does not exist in the package". That is incorrect —
> it is defined at `knowledge_execution.py:68` and caught at `runtime.py:1524`.
> The review's underlying instinct (that this is a *family* of uncaught
> conditions, not a single one) was right, and the table above is the corrected,
> stronger version of the original finding.

The code's own comment states the intended invariant:

> *"The raw sources stay bound next to the packet: learning narrows the search,
> it never removes the cold path."*

These three paths remove the cold path.

**Expected:** fall back to the raw sources with an
`operation_fallback_reason`, exactly as the row bound does.
**Actual:** unhandled `ValueError` propagates and the question is lost.

**Measured cost:** q13 was answered **correctly (334.0)** by the cold arm and
crashed after 12 s in the learn arm. Enabling learning turned a passing question
into an exception. Only the *truncated* branch was observed live; the column and
byte branches are identified by inspection and **not** exercised in this
evaluation.

**Proposed fix (not applied — core is frozen):** raise all four bound violations
as `OperationResultTooLarge`, or narrow the re-raise so result-bound errors are
caught. One-line-per-site change in `knowledge_execution.py`.

**Classification: universal mechanism.** It concerns result-bound handling and
knows nothing about any business domain. **No core patch was made.**

**Classification: universal mechanism.** It concerns result-bound handling and
knows nothing about any business domain. **No core patch was made.**

## 4e. Does the bundled `excel_modify` skill prevent F9?

The library bundles skills including `excel_modify.md`, `excel_extract.md`,
`analytical_integrity.md`, `data_exploration.md` and `delta_lakehouse.md`.
**None load by default:** `RLM.__init__` defaults are `skills=None -> []` and
`enable_skill_autoloading=False` (`runtime.py:1018-1019`), and `RLM.task`
forwards kwargs unchanged. The evaluation therefore ran the stock default
configuration, not a handicapped one.

`excel_modify.md` teaches exactly the discipline whose absence produced F9:
*"You are MODIFYING an Excel `.xlsx` workbook in place ... Open it, ... save
back to the same path, and verify by reloading"*, and explicitly lists
*"Saving to a different filename"* as a mistake.

This is the **universal-mechanism vs optional-skill** distinction the brief asked
for. If enabling `skills=["excel_modify"]` removes the reset, **F9 belongs in a
skill and warrants no core patch.** The harness now supports this via the
`SKILLS` job parameter; the arm is specified and runnable but **has not been
run**, so no claim is made either way.

## 5. What is genuinely proven

- ✅ RLM **can** build a formatted, multi-sheet `.xlsx` — bold headers, fills,
  frozen panes — with no Excel helper code from me.
- ✅ RLM **can** publish it to lakehouse `Files/` via `FileDestination.publish()`.
- ✅ RLM **can** append to a workbook handed back to it: proven in the 2-question
  smoke (7,359 → 7,885 bytes, both rows present) and for 14 consecutive
  questions in the full run.
- ❌ RLM **cannot** be trusted to do so unattended over 24 questions.

## 6. Threats to validity — stated, not hidden

1. **Single repetition.** Phase 1 was 3 reps × 24 = 72 trials; Phase 2 is 1 rep ×
   24. The 33.3% figure has a wide interval and is **not** directly comparable to
   a 72-trial mean.
2. **Confounded comparison — the original one is now moot.** Phase 1 ran locally
   (Python 3.11.9); Phase 2 ran in Fabric (Python 3.12.12). That cross-phase
   comparison confounded environment with model and is **no longer used for any
   conclusion**. The load-bearing comparison is now *within* Phase 2 and within
   the same environment: gpt-4.1-mini vs GLM-5.3-flash, everything else held
   constant. The residual Python 3.11 vs 3.12 difference therefore cannot
   affect it.
3. **dspy version.** Both phases ran dspy 3.2.1, inside the `>=3.2.1,<3.3` pin.
   Earlier Fabric debug runs used 3.3.1 (out of pin); **none of those results are
   reported anywhere.**
4. **Disclosed scaffolding.** RLM cannot read `abfss://` into a local `File`
   (`File` is local-path only). The parent notebook therefore downloads the
   published workbook with `notebookutils.fs.cp` and re-injects it as
   `workbook_in`. **RLM authors all content and performs the append; the parent
   only moves bytes.** This is weaker than "RLM updates the Excel in place" and
   is stated as such.

## 7. Harness defects found and fixed (mine, not the library's)

| # | Defect | Effect | Fix |
|---|---|---|---|
| H1 | Library side-loaded as a zip with no `pyproject.toml` | dependency pins never enforced; `pip install dspy` took **3.3.1**, outside `>=3.2.1,<3.3` | build and install a wheel; hard-fail if the resolved version leaves the pin |
| H2 | Fixed output paths | cancelling a local shell does not cancel the Fabric job; a second run deleted the first run's workbook mid-flight | namespace all outputs and logs by `RUN_TAG` |
| H3 | Bare `except` around verification | hid a `NameError: WORKBOOK` across four runs, making RLM look like it never published | `except BaseException` + traceback |
| H4 | Client-side token expiry | poller died at ~15 min while the job continued | re-poll with a fresh token; never relaunch |
| H5 | Output contract does not say what `answer.value` holds for a **categorical** question | GLM's correct answers for q16/q19 landed in `reasoning` and were graded FAIL; cost 2 points of a real 24/24 | **Identified, NOT yet applied.** Both GLM arms ran the unfixed contract, so it affects them equally and does not bias the cold-vs-learn comparison. Scores are reported unchanged. |
| H6 | Run log stored the answer payload only, never `RLMResult.trajectory` | no gate-hit count existed, so F11's causal claim rested on 2 self-reports and had to be withdrawn | persist trajectory per question; count gate rejections directly |
| H7 | Trajectory capture stringified **turns** only, dropping `trajectory.metadata` — so `operation_selection_lm_calls`, `operation_selection_prompt_tokens` and `total_cached_tokens` were never persisted (0 occurrences in both run logs) | the +64% token overhead could not be decomposed; my attribution to the selection call had to be **withdrawn**, and raw tokens could not be converted to cost | persist `trajectory.metadata` and `RunResult.total_cached_tokens` per question |
| H8 | H6 landed **after** the cold arm ran, so arm A has no `catalog_gate_rejections` field | "0 vs 35 rejections" is unmeasured-vs-measured, not a comparison; q08's attribution to the gate had to be withdrawn | re-run arm A instrumented before any gate claim is made across arms |

## 8. Deployment blockers found (would affect any user)

1. A Fabric parameters cell needs the `# PARAMETERS CELL ****` marker or
   `executionData.parameters` is **silently ignored**.
2. RLM's code interpreter runs as a **subprocess** (`python -m fabric_rlm._worker`)
   which does not inherit `sys.path` — the library must be installed, or on
   `PYTHONPATH`.
3. `dspy` is **not preinstalled** in the Fabric Python 3.12 kernel (nor is
   `litellm`); `openpyxl`, `pandas`, `duckdb`, `deltalake`, `notebookutils` are.
4. Files written via the OneLake REST API are **not promptly visible** through the
   `/lakehouse/default/...` POSIX mount; `fs.exists()` needs a retry.
5. `File` accepts local paths only — it cannot read `abfss://`.
6. `knowledge=` and `inputs=` cannot both bind the same alias
   (`ValueError: task inputs conflict with knowledge source aliases`).

## 9. Conclusions

- **General execution capability:** confirmed, and stronger than the earlier
  draft claimed. With GLM-5.3-flash, RLM scored **91.7%** on 24 unseen complex
  questions against a real Fabric lakehouse, took **0** fan-out hazard traps,
  and returned a machine-readable answer every single time.
- **Artifact authoring:** confirmed end-to-end. RLM authored *and incrementally
  maintained* a formatted three-sheet workbook across 24 sequential updates with
  **zero** data loss (7,488 → 23,144 bytes, strictly monotone).
- **Portability across environments:** **no longer a failure.** The earlier
  "84.7% local vs 33.3% in Fabric" headline confounded environment with model.
  Holding the environment fixed and varying only the model gives 33.3% vs 91.7%,
  so the Fabric environment imposes no accuracy ceiling. **Model choice, not
  environment, dominated every metric measured.**
- **F9 and the accuracy collapse are model-dependent.** Same library, prompt,
  questions, dspy version and limits; only the model differed.
- **Cost of workbook duty:** +19% turns, +40% prompt tokens, **no** accuracy
  change (gpt-4.1-mini pair). Measured, not inferred.
- **F11 stands as a defect, not as a cause.** `_normalize_catalog_query` really
  does reject valid SQL and really does emit one undiagnosable message from
  seven paths — reproduced without credentials. Its contribution to the
  gpt-4.1-mini collapse is **unproven**; the run log stores answers, not
  trajectories, so there is no gate-hit count. Fixed for future arms.

### Unsupported claims remaining (explicitly not asserted)

- That 33.3% or 91.7% is stable across repetitions — **n = 1 per arm**.
- That `excel_modify` mitigates F9 — specified, **unmeasured**.
- That `.learn` helps or hurts **on this model**. On this lakehouse the
  question is close to meaningless: `learn()` produced **0 lessons** (F13), so
  arm B carries no knowledge to test. A meaningful learn gate needs either a
  semantic-model source or a hand-written `declared=` block.
- That the Python 3.11 → 3.12 difference contributes nothing.
- **That the +64% prompt-token overhead comes from the operation-selection LM
  call.** The call is real and unconditional (confirmed in source), but the
  attribution is **withdrawn**: `analyze_token_overhead.py` shows per-question
  deltas ranging −28,725 to +80,667, with **3 of 18** clean questions *negative*
  and the three questions where an operation actually executed showing the
  *smallest* deltas. At temperature 1.0 with n = 1, sampling variance dominates.
  The aggregate is real; the decomposition is unmeasured (harness defect H7).
- That the +64% represents **cost**. These are raw prompt tokens;
  `total_cached_tokens` was not captured, and the frozen operations JSON is
  cache-friendly.
- That the catalog gate costs the *learn* arm specifically — arm A has no
  `catalog_gate_rejections` field at all, so "0 vs 35" is unmeasured vs
  measured, not a comparison.

## 10. Answering the original brief directly

- **General execution capability:** demonstrated. 91.7% on 24 unseen complex
  questions over a real Fabric lakehouse, 0 hazard traps, 100% answer rate.
- **Generalization of *learned* behaviour:** **measured, and it fails on this
  configuration.** On the same model and source, `.learn` scored **83.3% vs
  91.7% cold**, took **+21% turns** and **+64% prompt tokens**. Stated
  precisely: on the 22 questions where no new failure mode fired the arms score
  **identically** (20/22 each), so I detected no reasoning difference — though
  at n = 1 and temperature 1.0 that is "none detectable at this power". Of the
  two remaining questions, **only q13 is attributable to learning** (F14 crashes
  in a frame reachable only when operations are registered). **q08 is not** —
  arm A was never instrumented for gate rejections, so there is no baseline, and
  no operation executed on q08; it is reclassified as sampling divergence.
  The package carried **0 lessons** but a **live operation set**, so what was
  actually tested is the *operations* pathway, not the lesson pathway. The
  explanation for the empty lesson set is F12: for tabular/Lakehouse
  sources learning emits at most one lesson type, gated on English column-name
  tokens — rename with the same meaning and it disappears (5/12 in the probe).
  The library is **not** ARR-specialized, but the learning path *is*
  **English-snake_case-specialized**. Full detail in `LEARN_GATE_VERDICT.md`.
- **Portability across sources:** Lakehouse/Delta confirmed live. Semantic
  model is the only source family with richer `structural_lessons`, and it was
  **not** exercised in Phase 2 — listed as untested, not claimed.
- **Remaining unsupported claims:** listed above, plus everything in §6.

## 11. Fix classification (as the brief requires)

| Finding | Universal mechanism | Source metadata | Optional domain skill |
|---|---|---|---|
| F11 distinct error messages per rejection path | **yes** | no | no |
| F11 comment handling (strip, or tokenize ignoring string literals) | **yes** | no | no |
| F12 report learning coverage / warn on an empty package | **yes** | no | no |
| F12 the meaning of a column name | no | **yes** — `declared=` | no |
| F12 more English synonyms in `_CURRENT_PERIOD` | **rejected** — puts a naming rule in core | no | no |
| F14 all four result-bound violations must fall back, not re-raise (3 of 4 currently fatal) | **yes** | no | no |
| Memoise the operation-selection decision — 21 of 24 tasks paid a selection LM call to be declined, while the 3 that converted were correct and nearly free | **yes** | no | no |
| F9 append-or-fail contract for derived artifacts | **weak** — model-dependent, so a core patch is not justified on this evidence | no | **candidate** — test `excel_modify` first |
| H5 output contract for categorical answers | harness-only | no | no |

None of these require knowledge of ARR, invoices, subscriptions, or any
business concept. **No core patches were made during this evaluation.**
