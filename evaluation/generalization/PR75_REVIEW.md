# Generalization audit — PR #75 review, findings and conclusions

Scope: does `fabric-rlm-core`, and `RLM.learn()` in particular, generalize
beyond the ARR models used during development — across domains, source types,
and naming conventions — in real Microsoft Fabric.

- Baseline: `b5226712a9aa41c3173d5f427e81244c333c0179`
- Under review: PR #75 head `4466e9b`, plus review fixes `53e6cbb`, `a736098`, `bd924bc`
- Live model: `openai/gpt-4.1-mini` via OpenRouter, seed 20260909
- Real-Fabric evidence: see `FABRIC_FINDINGS.md` (zero model calls)

**Library SHA per result set** — results are not comparable without this:

| Result set | Library SHA | File |
|---|---|---|
| Stage 3 full matrix, 24 trials | `a736098` (pre-fix) | `stage3_results.json` |
| Re-grade of the above | `a736098` | `stage3_regraded.json` |
| `q_ecom_avg_payment` arm B rerun | `bd924bc` (post-fix) | `stage3_postfix.json` |
| `q_ecom_order_count` arm B rerun | `bd924bc` (post-fix) | `stage3_postfix_ordercount.json` |
| F8 diagnostic, 6 reps | `bd924bc` | `stage3_f8_diag.json` |
| Stage 4 smoke, 8 new questions | `bd924bc` | `stage4_smoke.json` |
| **Stage 4 gate, 66 trials** | `bd924bc` | `stage4_gate.json` |

## Status of each requested measurement

| Area | Status |
|---|---|
| Domain-dependency audit | measured |
| Synthetic datasets, 3 domains, independent references | measured |
| Naming robustness | measured |
| Transfer across sources (file, Lakehouse, semantic model) | measured for binding, profiling, learning |
| **What learning adds to answers** | **measured — 66-trial gate, 11 questions, 3 domains** |
| Failure and change handling | measured; F1 and F7 are the significant results |
| Correctness vs efficiency | measured live, reported on two separate axes |

## Live matrix: measured

24 trials, 4 questions x 2 arms x 3 repetitions, on real Fabric Delta data
materialized locally. Arm A = no knowledge package. Arm B = `RLM.learn()`
with `declared=`.

Exact commands:

```powershell
$env:OPENROUTER_API_KEY = "<key>"
$env:PYTHONPATH = "<...>\fabric-rlm-core-pr75"
cd <...>\fabric-rlm-core-eval\evaluation\generalization
python stage3_fabric_live.py --smoke                          # 2 trials
python stage3_fabric_live.py --repetitions 3 --max-live-calls 400
python regrade_stage3.py                                      # offline, no model calls
# targeted post-fix reruns
python stage3_fabric_live.py --only q_ecom_avg_payment --arms B --repetitions 3
python stage3_fabric_live.py --only q_ecom_order_count --arms B --repetitions 3
```

### Grading correction

The first-pass grader read `answer["value"]` as a scalar and scored arm A wrong
when it returned a correct `numpy.int64` (`__serializable__: false`) or nested
the value in a dict. `regrade_stage3.py` separates **numeric correctness** from
**machine readability**. Both are reported; conflating them inverted the result.

`q_service_avg_talk` is **excluded from totals**: its reference (14.9936) and
its hazard value (14.994) fall within tolerance of each other, so it cannot
discriminate between arms. It was 3/3 in both.

### Results over the 3 discriminating questions (superseded — see the gate below)

| Question | Arm A (cold) | Arm B pre-fix `a736098` | Arm B post-fix `bd924bc` |
|---|---|---|---|
| `q_ecom_avg_payment` | 3/3 | 0/3 (1 hazard, **2 crashes**) | **2/3** (1 hazard, 0 crashes) |
| `q_ecom_order_count` | 3/3 | 2/3 (**1 crash**) | **3/3** |
| `q_retail_top_product` | 3/3 (all 3 unreadable) | 3/3 | 3/3 |
| **Total** | **9/9** | **5/9** | **8/9** |

**The pre-fix 5/9 was almost entirely a library crash, not a learning failure.**
Three of the four losses were the F7 row-bound abort.

This 3-question set is **too small to judge the gate**, and its grading was
still too strict. Both problems are corrected below.

---

## The gate: 66 trials, 11 questions, 3 domains

The user's gate: **learned accuracy >= cold, in fewer turns, or on novel and
complex questions.**

Question bank expanded from 3 discriminating questions to **11**, four per
domain, targeting denominator and roll-up traps: ratio-of-totals vs
mean-of-ratios, share of orders vs share of rows, per-customer vs
per-transaction, conditional denominators, and runner-up entities.

Every candidate was screened offline by `check_degeneracy.py` before costing a
live call. **Four were rejected** because their hazard value was within
tolerance of the correct answer and so could not discriminate:
`q_ecom_delivered_avg_items`, `q_retail_repeat_customer_share`,
`q_service_abandoned_rate`, and `q_service_avg_talk`.

```powershell
python check_degeneracy.py                       # offline screen
python stage3_fabric_live.py --only <11 ids> --arms A,B --repetitions 3 \
    --max-live-calls 600 --output stage4_gate.json
python analyze_gate.py stage4_gate.json          # offline
```

### Two axes, reported separately

- **analytic** — the agent computed the right number, anywhere in the answer.
- **contract** — the number was readable straight off `answer["value"]`.

The split is necessary. Six answers put a composite string in `value`
(`"Technical Support (30.9%)"`). That is an output-contract violation, not a
reasoning error, and scoring it as a wrong answer overstated the error rate
and penalised the cold arm hardest. See F9.

### Result, library `bd924bc`

| | Arm A (cold) | Arm B (learned) |
|---|---|---|
| Analytic correctness | 32/33 (97.0%) | **33/33 (100%)** |
| Contract compliance | 30/33 (90.9%) | **32/33 (97.0%)** |
| Named hazards hit | **0** | **0** |
| Abstentions | 1 | 0 |
| Errors / crashes | 0 | 0 |
| Not machine-readable | 9 | **3** |
| Mean turns | 4.52 | **3.21 (−29%)** |
| Mean prompt tokens | 6,883 | 6,580 (−4%) |

**Gate verdict: PASS.** Learned accuracy is not lower on either axis, in fewer
turns. **No question regressed under learning.** One question improved
(`q_service_top_type_share`, 2/3 → 3/3).

### Per domain

| Domain | Arm | Contract | Analytic | Turns |
|---|---|---|---|---|
| ecommerce | A | 12/12 | 12/12 | 3.67 |
| ecommerce | B | 12/12 | 12/12 | 3.00 |
| food_retail | A | 12/12 | 12/12 | 5.00 |
| food_retail | B | 12/12 | 12/12 | **2.58** |
| service_ops | A | 6/9 | 8/9 | 5.00 |
| service_ops | B | **8/9** | **9/9** | 4.33 |

### Honest limits on this result

- **Wilson 95% intervals overlap** (analytic: cold 84.7–99.5%, learned
  89.6–100%). The *accuracy* gap is not statistically established; only
  non-inferiority plus the efficiency gap are.
- Prompt-token savings largely vanished on the expanded set (−4%, against
  −33% on the original three). One question, `q_service_satisfied_rate_quality`,
  cost learning **more** (7.33 turns, 16,660 tokens vs 5.67 and 9,095). The
  turn reduction is robust; the token reduction is not.
- One model, one seed, three repetitions.

It runs 4 unseen questions over **real Fabric data** in 3 unrelated domains,
2 arms (cold vs declared-learned), 3 repetitions, shuffled under a fixed seed,
graded against pandas references that never enter the agent's inputs. Each
question carries a named *hazard* value so a wrong answer can be attributed to
the specific trap rather than merely scored wrong:

| question | reference | hazard | hazard meaning |
|---|---|---|---|
| distinct orders with an item | 98,666 | 112,650 | counted item rows, not orders |
| average payment per order | 160.99 | 154.10 | averaged payment rows, not orders |
| top product by revenue | 11,595.00 (Golden Gate Ginger) | 11,199.00 | returned the runner-up |
| overall average talk time | mean over calls | mean of per-agent means | unweighted average of averages |

---

## F1 — Default profiling limits make learned packages unusable on realistic data

**Severity: medium.** There *is* a supported path; it is undocumented, and the
error message misdescribes the cause.

### Reproduction (deterministic, zero model calls)

```python
k = RLM.learn(sources={"olist_orders": "olist_orders.csv"},      # 20 MB, real Fabric export
              declared={"olist_orders": {"grain": ["order_id"]}})
RLM.from_task(task="...", knowledge=k, lm=lm)
# ValueError: stale knowledge sources detected: olist_orders
```

Observed on the first run, immediately, on files that had just been written:

```
olist_orders        snapshot_exact=False  records_inspected=1000  records_truncated=True
DRIFT: {'olist_orders': 'inexact', 'olist_order_items': 'inexact', 'olist_order_payments': 'inexact'}
```

A 5 KB control file profiles with `snapshot_exact=True` and shows no drift.
The trigger is size, not content.

### The fix that exists

`ProfileLimits` defaults to `max_input_bytes=1 MB` and `max_records=1000`
(`knowledge_sources.py:51`). Raising them resolves the failure completely:

```python
limits = ProfileLimits(max_input_bytes=64*1024*1024, max_records=200_000,
                       read_chunk_bytes=1024*1024)
k = RLM.learn(sources=src, declared=..., limits=limits)
```

```
olist_order_items    snapshot_exact=True  records=112650  truncated=False
olist_order_payments snapshot_exact=True  records=103886  truncated=False
olist_orders         snapshot_exact=True  records= 99441  truncated=False
DRIFT: {}  is_current: True
lessons: Counter({('schema', 'active'): 8})
```

Learning all three tables took **3.4 seconds** and preflight another 3.4
seconds for roughly 41 MB. The default is therefore not buying much protection
on data of this size.

### Expected vs actual

- Expected: either the defaults accommodate ordinary analytical tables, or the
  failure explains itself and names the knob.
- Actual: `learn()` succeeds and promotes 8 active lessons; every subsequent
  run raises `ValueError: stale knowledge sources detected`. Nothing is stale —
  the files were never modified. The message describes a data-change condition
  when the real cause is a profiling budget, and it does not mention `limits=`,
  which appears in no user-facing documentation of `learn()`.

This is very likely the cause of the earlier observation that cold runs worked
while learned runs did not.

### Affected code

- `fabric_rlm/knowledge_sources.py:51` — `ProfileLimits` defaults.
- `fabric_rlm/knowledge_preflight.py:168` — `snapshot_exact is not True` → `"inexact"`.
- `fabric_rlm/runtime.py:1368`, `fabric_rlm/knowledge_execution.py:764` — any drift raises.

### Proposed fix — universal mechanism

1. Raise the defaults to something realistic for analytical tables, or size the
   record cap from the source rather than a fixed 1000.
2. Make the error actionable: distinguish "source changed" from "source could
   not be profiled exactly within the current limits", and name `limits=` in
   the latter.
3. Document `limits=` in the `RLM.learn` docstring alongside `declared=`.

The conservatism itself is justified and should stay.
`tests/test_knowledge_preflight.py:321` proves a middle-only mutation of a
truncated file yields an **identical** sampled fingerprint, so an inexact
profile genuinely cannot be trusted for staleness. Do not relax that check.

---

## F1b — Snapshot inexactness invalidates schema-scoped declared facts

Independent of F1, and still present whenever a source is genuinely too large
to profile exactly.

Declared facts record that they depend only on the schema:

```
semantic_fact | grain   | scope=schema | basis=('declared',) | status=active
semantic_fact | note 1  | scope=schema | basis=('declared',) | status=active
```

After preflight on unchanged, truncated files:

```
lesson status after preflight: Counter({('schema', 'stale'): 8})
```

**Schema-scoped, user-asserted facts are invalidated by a snapshot-level
inexactness.** `dependency_scope` exists precisely to scope invalidation, and
the gate ignores it. A user's statement that `order_items` is one row per item
is not evidence about row contents and cannot go stale because rows 1001+ were
not read.

### Proposed fix — universal mechanism, not a domain rule

1. Scope invalidation by dependency: `inexact` and `snapshot` drift should
   stale only knowledge whose `dependency_scope` includes the snapshot.
   `schema`-scoped knowledge should survive; `schema` drift should continue to
   stale everything. Relationships and operations genuinely depend on row
   content and should keep the current behaviour.
2. Degrade rather than abort at `runtime.py:1368`: drop staled knowledge, keep
   what survives, record the existing `*.stale` event, and reserve the hard
   raise for knowledge actually required by the run.

Pinning tests: a truncated but unchanged file must retain its declared
`schema`-scoped lessons; a file whose *schema* changes must still stale
everything; and `test_same_size_middle_only_large_file_mutation_is_inexact_not_current`
must keep passing for snapshot-scoped knowledge.

Not implemented here: this changes a safety boundary, and the instruction was
to report proposed fixes separately rather than land them during evaluation.

---

## F2 — Automatic lesson synthesis is gated on English name morphology

Fully evidenced in `FABRIC_FINDINGS.md` §3, against real Fabric semantic models
and Delta tables.

- Manufacturing Ops: 30 measures, exactly `Profit YTD` and `Sales YTD` match
  `_DERIVED_TIME_MEASURE`, and exactly 2 `context_requirement` lessons were
  produced.
- AdventureWorks Sales, bakehouse, wanderbricks: 0 matches, 0 lessons.
- 148 real Delta columns across 4 domains: 0 matches for `_CURRENT_PERIOD`,
  0 lessons.
- In the same manufacturing model, the abbreviated measures `gm2_pct`,
  `inv_rsk_u`, `prd_yld_day`, `sls_amt_x` produce nothing.

**This settles the ARR question.** The trigger is the token `ytd`, not a SaaS
concept; it fired in a manufacturing model and not in a retail one. There is no
ARR specialisation in executable code.

**Classification: source-metadata rule, not a core defect.** `declared=` is the
designed remedy and it works (§3, and F4 below). The residual English keyword
lists at `knowledge_retrieval.py:43` (`retention`, `nrr`, `grr`, `churn`) and
`knowledge_lessons.py:56` are SaaS-flavoured vocabulary in executable code;
they are harmless where they do not match, but they should be documented as a
built-in convenience heuristic, not as the mechanism users rely on.

---

## F3 — Bundled skills delivered domain-specific examples to the model (fixed)

`analytical_integrity.md` is delivered **verbatim** to the model (5,505
characters) and routes on generic keywords such as `ranking` and `materiality`.
A manufacturing task therefore received SaaS revenue examples in its prompt.
This was the only path by which "ARR" actually reached the model; the other 24
occurrences in `fabric_rlm/*.py` are comments and docstrings.

Fixed in `53e6cbb` / `a736098`: `analytical_integrity.md`, `semantic_model.md`,
and `delta_lakehouse.md` are now domain neutral, pinned by a new parametrized
guard `tests/test_domain_neutrality.py` covering all 13 bundled skills.

---

## F4 — `declared=` closes the naming gap, on real Fabric data

| `learn()` | lessons |
|---|---|
| manufacturing CSV, descriptive names | 1 |
| same data, abbreviated names (`ln`, `prd`, `up`, `icp`) | **0** |
| abbreviated + `declared=` | **6 active** |
| real OneLake Delta (olist), plain | **0** |
| real OneLake Delta (olist) + `declared=` | **2 active** |
| real Fabric CSV exports, 3 tables + `declared=` | **8 active** |

Declared facts reach every task, including unrelated ones, via
`retrieve_lessons` / `render_learned_guidance`.

The `declared=` contract is strict and **undocumented in the `learn` docstring**
(`knowledge_lessons.py:160-240`): `grain` is a list of columns, not prose;
`units` maps column to unit; `notes` is a list of strings. The PR body's
"one row per line x period" is the rendered form, not accepted input. Worth a
docstring, given this is now the primary generalization mechanism.

---

## F5 — Three defects found and fixed in PR #75

All by failing test first, all committed.

1. **Negative-literal bypass** (`analytical_integrity.py`). `collect()` had no
   `ast.UnaryOp` branch, so `-72800` was never collected as a literal. Every
   invented drop, loss, or decline passed the claim-provenance screen unread.
   Verified against the PR's own fixture: accepted before, flagged after.
2. **Exponent evidence.** `_TEXT_NUMBER` could not read `4.5e+06`, so a
   correctly computed `4500000` was reported as invented and burned a repair
   turn. Also fixed a pre-existing case where `abc123` matched `23`.
3. **Sign-vs-magnitude.** `_supported` now compares magnitudes, because code
   prints `abs drop = 1,500,000` while prose writes `Change: $-1,500,000`.

Forcing the screen on across the runtime suites yields the **same** 26 failures
before and after these changes, so they add no false positives.

---

## F6 — Review findings needing no code change

- **`conftest.py` disables the claim-provenance screen for most of the suite.**
  PR #75 adds an autouse fixture setting `FABRIC_RLM_CLAIM_PROVENANCE=0`
  everywhere except four modules. A green full suite therefore does not show
  the screen is safe in the runtime; it shows it is off. Forcing it on produces
  26 failures, because fake interpreters print nothing while scripting
  `SUBMIT(answer=1)`. Pre-existing, disclosed, and worth stating plainly.
- **`LakehouseSource` scope contract is undocumented and cost a full run.**
  `tables=["orders"]` fails validation; `tables=["Tables/orders"]` works
  (`lakehouse.py:190`). One docstring line would prevent this.
- **Fabric job runs report Python 3.10.20** despite `jupyter_kernel_name:
  python3.12`, which conflicts with the standing 3.12 preference.

---

## F7 — Row-bound overflow aborted the whole run (fixed, `bd924bc`)

Severity: **high**. This is the single largest correctness effect measured, and
it only reached the learned arm — knowledge made the agent fail *harder* than
no knowledge at all.

### Reproduction (deterministic, zero model calls)

`tests/test_knowledge_file_operations.py::test_group_by_high_cardinality_key_reports_a_bounded_plan_failure`

1. `declared={"grain": ["order_id"], ...}` promotes a grain lesson.
2. The lesson steers the model to `groupby=order_id`.
3. `order_id` is a **legal value in the operation's own `parameter_schema` enum** —
   the plan is valid by every check the planner applies.
4. The result frame is 99,440 rows, over `max_output_rows=100`.
5. `knowledge_execution.py:293` raised a bare `ValueError`.
6. `runtime.py` recorded telemetry and **re-raised**, aborting `RLM.run()`.

### Expected vs actual

- Expected: degrade to ordinary execution, as every sibling failure in this
  planner path already does (parse invalid, fallback, plan rejected, no
  compatible operation).
- Actual: the entire run raised. 3 of 24 live trials died this way.

### Affected code

- `fabric_rlm/knowledge_execution.py` — row bound
- `fabric_rlm/runtime.py` — `except ValueError: … raise`

### Fix — universal mechanism

New `OperationResultTooLarge(OperationPlanError)`, so the row bound is
recoverable and the runtime falls back, recording
`reason="result_bound_exceeded"`. This is a **universal mechanism**, not a
domain rule: it encodes "a bound the model's own plan can breach is a plan
error," with no reference to any column, metric, or business concept.

The **column bound deliberately stays a bare `ValueError`.** Rows follow from
the model's `groupby` choice (recoverable); columns are the host's own declared
contract, so an overflow there is a host bug and must fail closed. That
boundary is pinned by
`test_failed_host_audit_does_not_fall_back_to_ordinary_execution`.

### Verification on real Fabric data

`q_ecom_avg_payment` arm B went 0/3 with 2 crashes → **2/3 with 0 crashes**;
`q_ecom_order_count` arm B went 2/3 with 1 crash → **3/3**. No crashes remain.

---

## F8 — Withdrawn: the hazard did not survive a properly powered rerun

Severity: **withdrawn**. This was reported as the primary surviving
learning defect. A larger, correctly-graded run does not support that.

### What was originally claimed

One learned trial in three returned **154.1** on `q_ecom_avg_payment` — the
per-payment-row mean rather than the per-order average — despite an active
declared note warning about exactly that fan-out. Cold hit it 0/3. The
proposed fix was a universal grain assertion in the verifier.

### Why it was withdrawn

Re-running the same question against the fixed library did not reproduce it:

| batch | library | learned trials | hazards |
|---|---|---|---|
| initial 3-rep | `bd924bc` | 3 | 1 |
| diagnostic 6-rep | `bd924bc` | 6 | 0 |
| 66-trial gate | `bd924bc` | 33 | **0** |

**1 hazard in 42 learned trials (~2.4%)**, all of it in the first batch of
three. At n=3 a single stochastic event is indistinguishable from a defect.
Building a core mechanism on it would have been tuning on noise, and would
have violated the evaluation's own constraint against speculative core
patches.

### What the evidence actually shows

Across the 66-trial gate, **neither arm hit a single named hazard** — 0 of 33
cold, 0 of 33 learned. The declared notes are doing their job.

### Standing recommendation

No core change. If a grain assertion is ever added it should be justified by
a reproducible failure rate, not by one trial. The screening tool
(`check_degeneracy.py`) and the two-axis grader now make such a claim
falsifiable before it reaches a report.

---

## F9 — Answers violate the requested output contract while being right

Severity: **medium**. This is the largest *real* correctness effect left, and
it is a serialization and output-shape problem, not a reasoning one.

### Reproduction

`python analyze_gate.py stage4_gate.json`

Two distinct failure modes, both with the correct number present:

1. **Unserializable numpy scalar** — `value` becomes
   `{"__type__": "int64", "__repr__": "np.int64(6642)", "__serializable__": false}`.
   A caller reading `answer["value"]` gets a dict, not `6642`.
2. **Composite string** — `value` becomes `"Technical Support (30.9%)"` or
   `"Technical Support, 30.9"` when the task asked for a number in `value`
   and the entity in `entity`.

### Expected vs actual

- Expected: `answer["value"]` is the number.
- Actual: 9 of 33 cold answers and 3 of 33 learned answers are not readable
  that way, though nearly all are analytically correct.

### Measured effect

| axis | cold | learned |
|---|---|---|
| analytic correctness | 32/33 | **33/33** |
| contract compliance | 30/33 | **32/33** |
| not machine-readable | **9** | 3 |

Learning reduces the defect threefold, which is consistent with the package
normalizing how the agent reports results.

### Proposed fix — universal mechanism

Coerce numpy and other non-JSON scalars to plain Python numbers at the
output-validation boundary rather than emitting a placeholder dict. That is
a serializer change in `serializers.py` / output validation, with no
reference to any domain, metric, or column.

The composite-string mode is better addressed by output validation
rejecting a non-numeric `value` when the task declares a numeric output,
prompting one repair turn. Neither is a domain rule.

**Not fixed during this evaluation** — core is frozen, and F9 was found by
the harness rather than by the library's own tests.

---


## Conclusions

### General execution capability

**Established.** Over 66 trials on 11 questions across three unrelated domains
on real Fabric data, both arms were near-ceiling analytically (cold 32/33,
learned 33/33) with **zero crashes, zero named hazards, and one abstention**.
The execution-integrity machinery has real teeth once F5's bypasses are closed:
the claim-provenance screen now reads negative and exponent-formatted numbers.

Caveat: correct answers are not always **machine-readable**. 9 of 33 cold
answers and 3 of 33 learned answers put something other than a plain number in
`answer["value"]` — either an unserializable numpy placeholder or a composite
string. A caller reading `answer["value"]` would treat a correct answer as a
failure. This is F9, and it is the largest real defect still open.

### Generalization of learned behavior

**Two distinct answers, and conflating them would be misleading.**

- *Is it ARR-specific?* No. This is now settled causally: lesson synthesis
  fired in a manufacturing semantic model on `Sales YTD`, and did not fire in
  retail, bakery, or travel models. No executable ARR dependency exists. The
  only ARR content that reached the model was in bundled skill text, now fixed.
- *Is it automatic across arbitrary data?* No. Synthesis is gated on English
  derived-time and current-period name morphology. On 148 real Delta columns
  across 4 unrelated domains it produced nothing at all. Generalization is
  achieved through `declared=`, which is explicit source metadata the user must
  write — a reasonable design, but it should be described that way rather than
  as automatic learning.
- *Does learning help or hurt answers?* **Measured, and the gate passes.**
  Over 66 trials on 11 questions in 3 domains: learned 33/33 analytic and
  32/33 contract, cold 32/33 and 30/33, at 3.21 turns against 4.52. No
  question regressed; one improved. Neither arm hit a single named hazard.
  The confidence intervals overlap, so the honest claim is
  **non-inferiority plus a robust ~29% turn reduction**, not "learning is
  more accurate".
  Two earlier readings were wrong and are corrected here: the 5/9 that
  suggested "learning hurts" was mostly the F7 crash, and the residual F8
  hazard did not reproduce in 42 further trials.

### Portability across tested sources

Good, and genuinely demonstrated on real Fabric: 4 Lakehouses and 5 semantic
models across 8 unrelated domains all bound and profiled, with diagnostics
confirming 38-53 fields or 68-156 records actually inspected. Learning works
identically across file, Lakehouse, and semantic-model sources.

Tested with real integrations: Lakehouse/OneLake Delta, semantic models
(binding and profiling), local files. **Not tested:** Warehouse/SQL endpoints,
KQL databases, and mirrored databases — available in the tenant but not
exercised. No claim is made about them.

### Remaining unsupported claims

- That equivalent semantics across source types produce equivalent **answers**.
  Equivalent binding, profiling, and learning were shown across file, Lakehouse
  and semantic model; the live answer matrix ran against **locally materialized
  real Fabric data**, not against live `LakehouseSource` bindings, because local
  binding requires `notebookutils`. Answer-level cross-source equivalence
  remains untested.
- **Cost** was not measured; turns and prompt tokens were. End-to-end latency
  was not isolated from provider variance.
- Failure-handling breadth: staleness, row-bound overflow, and schema-scoped
  invalidation were exercised. **Timeouts, invalid joins, empty results, and
  failed/skipped verifier outcomes were not exercised live.**
- Configuration **C** from the original request — a package enriched from
  separate development runs — was **not built or tested**. Only A (no package)
  and B (`learn()` only) were compared.
- Only one model (`openai/gpt-4.1-mini`) was used, one seed, three repetitions.
  The efficiency gain and the residual F9 rate may both be model-dependent.
- The gate's **accuracy** gap is not statistically established: Wilson 95%
  intervals overlap (cold 84.7–99.5%, learned 89.6–100%). Non-inferiority and
  the ~29% turn reduction are supported; "learning is more accurate" is not.
- **Prompt-token savings did not replicate.** −33% on the original three
  questions became −4% on the eleven. One question cost learning nearly twice
  the tokens. Only the turn reduction is robust.
- Both arms sat near ceiling on 9 of 11 questions, so this set has limited
  power to separate them. A harder bank would be needed to show learning
  *ahead* rather than merely not behind.

