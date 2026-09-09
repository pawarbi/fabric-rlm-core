# Generalization audit — PR #75 review, findings and conclusions

Scope: does `fabric-rlm-core`, and `RLM.learn()` in particular, generalize
beyond the ARR models used during development — across domains, source types,
and naming conventions — in real Microsoft Fabric.

- Baseline: `b5226712a9aa41c3173d5f427e81244c333c0179`
- Under review: PR #75 head `4466e9b`, plus review fixes `53e6cbb`, `a736098`
- Live model: `openai/gpt-4.1-mini` via OpenRouter
- Real-Fabric evidence: see `FABRIC_FINDINGS.md` (zero model calls)

## Status of each requested measurement

| Area | Status |
|---|---|
| Domain-dependency audit | measured |
| Synthetic datasets, 3 domains, independent references | measured |
| Naming robustness | measured |
| Transfer across sources (file, Lakehouse, semantic model) | measured for binding, profiling, learning |
| **What learning adds to answers** | **UNMEASURED — credential expired; harness delivered** |
| Failure and change handling | partly measured; F1 is the significant result |
| Correctness vs efficiency | not measured live |

## Live matrix: blocked, harness delivered

The OpenRouter key supplied for this work is **expired**. The provider returns:

```
AuthenticationError: OpenrouterException - {"error":{"message":"API key expired.","code":401}}
```

Per the original instruction, those results are marked **unmeasured** rather
than estimated. The harness is complete and runnable:

```powershell
$env:PYTHONPATH = "<...>\fabric-rlm-core-pr75"
cd <...>\fabric-rlm-core-eval\evaluation\generalization
python stage3_fabric_live.py --smoke                       # 2 trials
python stage3_fabric_live.py --repetitions 3 --max-live-calls 400
```

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

## Conclusions

### General execution capability

Not established by this evaluation. The live matrix is unmeasured because the
credential expired. F1 would additionally have blocked the learned arm at the
default profiling limits; the harness now raises them explicitly. What is
established is that
the execution-integrity machinery has real teeth once F5's bypasses are closed:
the claim-provenance screen now reads negative and exponent-formatted numbers.

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

- That learning improves **answers**. Only lesson counts and package contents
  were measured. Nothing here shows a learned package changes correctness.
- That equivalent semantics across source types produce equivalent **answers**.
  Equivalent binding, profiling, and learning were shown; equivalent answers
  were not.
- Latency, token, and cost comparisons between arms.
- Failure-handling behaviours beyond staleness: timeouts, invalid joins, empty
  results, and verifier outcomes were not exercised live.

