# Fabric heterogeneous-source findings (PR #75)

Evidence gathered against **real Microsoft Fabric** artifacts — not the seeded
synthetic fixtures. Every number below comes from a run that made **zero model
calls**, so nothing here depends on a sampling seed or an LLM's behaviour.

- Library under test: PR #75 head `4466e9b` plus review fixes `53e6cbb`, `a736098`
- Notebook: `fabric_rlm_hetero_generalization_eval`
  (`3b7bac7d-b0af-445c-8dd3-90c16e13ee23`, workspace `sandeep_ws`)
- Run: `20260909T055947Z-hetero-e7733a0b`
- Raw report: `Files/rlm-evaluation/hetero/20260909T055947Z-hetero-e7733a0b/report.json`
- Harness: `fabric_hetero_eval.py`
- Development model deliberately excluded from the sample: `ARR Model SF (79)`

## 1. Sources exercised

Four unrelated business domains, four separate Fabric workspaces:

| key | domain | artifact | tables bound |
|---|---|---|---|
| `ecommerce_olist` | e-commerce marketplace | Lakehouse | 6 |
| `food_retail_bakehouse` | bakery franchise retail | Lakehouse | 4 |
| `service_ops_callcenter` | call-centre service operations | Lakehouse | 3 |
| `retail_adventureworks` | retail | Lakehouse | 3 |

Five semantic models, five unrelated domains: manufacturing operations, retail,
bakery franchise retail, retail banking, travel marketplace.

All 4 Lakehouse sources and all 5 semantic models bound and profiled
successfully. 16/16 Delta tables read from OneLake.

## 2. Profiling is source-type portable and domain neutral

Profiling succeeded on every source in every domain. Diagnostics prove real
inspection happened rather than a silent no-op:

| source | family | fields / records inspected | snapshot exact |
|---|---|---|---|
| `ecommerce_olist` | `lakehouse` | 38 fields, 6 delta entries | yes |
| `food_retail_bakehouse` | `lakehouse` | 42 fields, 4 delta entries | yes |
| `service_ops_callcenter` | `lakehouse` | 53 fields, 3 delta entries | yes |
| `sm_manufacturing` | `semantic_model` | 156 records | yes |
| `sm_adventureworks` | `semantic_model` | 78 records | yes |
| `sm_bakehouse` | `semantic_model` | 68 records | yes |

**Note, not a defect:** `SourceProfile.schema` is empty in the persisted
package. That is by design — the package retains fingerprints and diagnostics,
not the schema itself. Field counts in `diagnostics` are the evidence that
inspection occurred.

## 3. Automatic lesson synthesis is gated on English name morphology

This is the central finding, and it is now causal rather than circumstantial.

Measure names were extracted **independently of the library**, from each model's
TMDL definition via the Fabric `getDefinition` REST endpoint, then matched
against `knowledge_lessons._DERIVED_TIME_MEASURE`:

| semantic model | measures | regex matches | lessons produced |
|---|---|---|---|
| Manufacturing Ops | 30 | `Profit YTD`, `Sales YTD` | **2** (`context_requirement` on exactly those two) |
| AdventureWorks Sales | 1 | none | 0 |
| bakehouse_semantic_model | 6 | none | 0 |
| wanderbricks_semantic_model | 16 | none | 0 |

The correspondence is exact and one-to-one. Lesson synthesis fired in the
**manufacturing** domain, which settles the ARR question: the trigger is not a
SaaS concept, it is the token `ytd`. It did not fire in three other domains
because no measure there is named with an English derived-time token.

The same holds on the Lakehouse side. Across **148 real columns in 16 Delta
tables spanning 4 domains**, `_CURRENT_PERIOD` matched **zero** columns — and
all four sources produced 0 lessons. The zero-lesson outcome is therefore
**expected behaviour given the trigger contract**, not a failure. It is not
reported as a defect.

### Real-world naming-robustness data point

`Manufacturing Ops` happens to contain both descriptive and abbreviated
measures. The abbreviated ones — `gm2_pct`, `inv_rsk_u`, `po_ok_flagish`,
`prd_yld_day`, `sls_amt_x` — carry real meaning but produce no lessons, in the
same model where `Sales YTD` does. Naming sensitivity is reproducible on
production metadata, not just on constructed fixtures.

## 4. `declared=` closes the gap on real Fabric Delta

Same real Lakehouse, same run, only difference is source metadata:

| `learn()` on `ecommerce_olist` (real OneLake Delta) | lessons |
|---|---|
| plain | **0** |
| with `declared={... "grain": ["order_id"], "notes": [...]}` | **2 active `semantic_fact`** |

The promoted lessons are `grain` (`one_row_per_grain`, `["order_id"]`) and the
join-multiplication note, both `confidence: high`, `basis: ["declared"]`.

This reproduces on production data the result previously shown only on
synthetic CSVs, and it is the intended remedy for the naming sensitivity in §3.

## 5. Independent reference answers (real Fabric Delta)

Computed with pandas, entirely outside the library's query compiler:

| key | value |
|---|---|
| `olist_orders_distinct` | 99,441 |
| `olist_orders_with_items` | **98,666** (correct) |
| `olist_naive_join_rows` | **112,650** (the join-multiplication hazard) |
| `olist_payment_total` | 16,008,872.12 |
| `bakehouse_txn_count` | 3,333 |
| `callcenter_call_count` | 100,000 |

## 6. Operational findings

- **`LakehouseSource` scope contract.** `tables=` entries must be rooted at the
  scope kind: `Tables/orders`, not `orders` (`lakehouse.py:190`,
  `parts[0] != kind`). Bare table names fail validation with *"scopes must be
  safe relative paths"*. Omitting `tables=` uses the default `("Tables",)` and
  discovers the catalog. This cost a full run; it deserves a docstring.
- **Fabric Python-kernel notebooks have no `spark` session.** The harness is
  pure Python (`deltalake` + pandas) as a result.
- **Job runs reported Python 3.10.20** despite `jupyter_kernel_name:
  python3.12`. Unresolved.
- `notebookutils.session.restartPython()` breaks job-mode runs.

## 7. What this does and does not establish

Established:

- Profiling is portable across Lakehouse and semantic-model sources and is
  domain neutral.
- Lesson synthesis is domain neutral but **English-naming dependent**.
- `declared=` restores learning on sources whose names do not match the
  triggers, on real Fabric data.

Not yet established by this run — it makes zero model calls:

- Whether learned lessons change **answers**, not just metadata. That requires
  the budgeted live matrix.
