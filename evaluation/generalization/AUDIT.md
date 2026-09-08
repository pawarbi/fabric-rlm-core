# Generalization audit

**Baseline:** `b5226712a9aa41c3173d5f427e81244c333c0179`  
**Evaluation branch:** `eval/generalization-b522671`  
**Core/skill freeze:** no mismatches  
**Generated large fixture:** 250,000 rows, 30,000,049 bytes  
**Offline reliability suite:** 197 passed, 1 skipped  
**Live smoke:** OpenRouter authentication succeeded. The 15-call budget
completed six development calls and nine evaluation trials.
**Real Fabric verification:** Ten Delta tables, a dedicated Direct Lake model,
and both frozen Fabric adapters were exercised in `sandeep_ws`.

## Live smoke results

The smoke used `openai/gpt-4.1-mini`, descriptive names, one unseen question
per domain, one repetition, six turns, a 120-second timeout, disabled caching,
and the same settings for A/B/C.

| Metric | Result |
|---|---:|
| Fully correct | 0 / 9 |
| Reference value correct | 6 / 9 |
| Confident wrong | 7 / 9 |
| Incomplete | 2 / 9 |
| Evaluation-trial provider cost | $0.024820488 |
| Evaluation-trial prompt tokens | 84,118 |
| Evaluation-trial completion tokens | 6,568 |
| Evaluation-trial cached tokens reported by provider | 63,616 |
| Evaluation-trial wall time | 234.14 seconds |
| Mean evaluation-trial wall time | 26.02 seconds |

A averaged 19.93 seconds per evaluation trial, B 36.50 seconds, and C 21.62
seconds. Development-call tokens, cost, and latency were not retained by this
harness version and are excluded from those totals.

Six answers had the correct reference value but failed required reporting
metadata: inventory A/B did not match the declared grain, manufacturing B/C
did not identify the exact reporting period or expected grain, and service A/C
omitted the reporting period. Manufacturing A doubled production from 1,800 to
3,600 through a join. Service B exhausted six turns. Inventory C abstained
despite sufficient metadata, while A and B returned the correct value.

## Findings

| Finding | Reproduction | Expected | Actual | Affected code | Proposed fix |
|---|---|---|---|---|---|
| No executable ARR specialization found | `runner offline`; inspect `audit.arr_mentions` | Examples and comments must not count as runtime specialization | 12 ARR mentions were documentation/comments; 0 were executable | Documentation in `analytical_integrity.py`, `knowledge_lessons.py`, `semantic_model.py`, and `trajectory.py` | None. Preserve the executable/documentation distinction in future audits. |
| Live answer status aliases initially produced false incompletes | Regrade a correct answer with `status="success"` or `"ok"` | Successful structured answers should reach correctness checks | The grader initially accepted only `"answered"` | `evaluation/generalization/grader.py`; regression test in `tests/evaluation/test_generalization_fixtures.py` | **Evaluation mechanism:** canonicalize successful status aliases before grading. Original answers and metrics were preserved and regraded without additional model calls. |
| Semantic structural learning depends on English date naming patterns | The offline probe creates equivalent semantic profiles with `Period[IsCurrentQuarter]` and `prd[icq]` | Equivalent metadata mappings should yield equivalent lessons | Descriptive profile created 1 current-period lesson; abbreviated profile created 0 | `fabric_rlm/knowledge_lessons.py:49-59`, `:110-151`, `:375-380` | **Universal:** add declared semantic-role fields to the lesson mechanism. **Source metadata:** adapters should map columns/measures to roles such as current-period and period-dimension. **Optional skill:** keep English regexes only as fallback hints. |
| Lesson retrieval has English and business-vocabulary triggers | Offline score probe compares `customer` with unfamiliar `acct` | Equivalent mapped concepts should retrieve the same lesson | `customer` scored 0.7; `acct` scored 0 | `fabric_rlm/knowledge_retrieval.py:20-52`, `:124-152` | **Universal:** retrieve by stable semantic tags plus lexical overlap. **Source metadata:** provide concept aliases. **Optional language/domain skill:** supply English or domain trigger vocabulary; do not hard-code it in core. |
| Large files cannot use learned registered operations and are rejected on reuse | Generate the 30 MB CSV; run offline audit; bind the learned package | Large sources should retain an exact stable identity or explicitly use a source version | `snapshot_exact=false`, 0 operations; preflight classifies every later observation as `inexact`, and runtime refuses stale bindings | `fabric_rlm/knowledge_sources.py` bounded head/tail snapshot; `knowledge_operations.py:150-155`; `knowledge_preflight.py:29,107-108,168-169`; `runtime.py:1264` | **Universal:** stream a full digest without placing contents in prompts, or accept an adapter-supplied immutable version identity. **Source metadata:** remote adapters should expose ETag/version/table snapshot IDs. |
| `learn()` on ordinary files adds operations but no lessons | Learn the small inventory CSV in the offline audit | Configuration B should have a measurable learned-behavior delta or clearly disclose that it is only an operation catalog | 1 registered aggregate operation, 0 lessons | `fabric_rlm/knowledge_api.py:185+`; `fabric_rlm/knowledge_lessons.py:110` limits structural lessons to semantic models | **Universal:** capture typed telemetry for file operations and promote source-agnostic grain, failure, and strategy lessons. Domain definitions remain metadata/skills. |
| Registered operations cover only a narrow analytical algebra | Inspect discovered operations and run inventory/service questions requiring joins, deduplication, earliest-event logic, and weighted ratios | Supported tasks should execute equivalently across supported sources or fall back explicitly | Core supports semantic measure, sum/avg/count aggregates, Lakehouse aggregate, and Lakehouse pre-aggregate join; local multi-file joins and weighted-ratio/event semantics require unconstrained Python fallback | `fabric_rlm/knowledge_operations.py:113-198,204-393`; `knowledge_execution.py:140-147,416-840` | **Universal:** add typed relational operators for join cardinality, distinct entities, first/last event, ratio-of-sums, and snapshot selection. **Source metadata:** relationships, keys, event time, completeness flags. **Optional skill:** domain metric definitions. |
| Generic SQL is not a knowledge source | Inspect the registry and execution implementation set | SQL should be tested only if a supported adapter exists | Registry supports CSV, JSON, JSONL, Parquet, Delta, Lakehouse, semantic model, and opaque files; no generic SQL adapter exists | `knowledge_sources.py:137-140`; `knowledge_lakehouse_sources.py:496-510`; `knowledge_execution.py:140-147` | **Universal:** define a read-only SQL source contract with schema/version identity and bounded execution. **Source metadata:** dialect, endpoint, credentials, table contracts. Do not put business rules in the adapter. |
| Real Fabric access works, but local and Fabric-runtime credential paths differ | Create the ten `rlm_eval_*` Delta tables; run the explicit-catalog Lakehouse adapter locally; run the frozen wheel in `fabric_rlm_generalization_adapter_eval` | Supported handles should query real Fabric data with documented credentials and paths | The frozen Lakehouse adapter passed locally with an explicit catalog and a process-local storage-token shim. The frozen semantic adapter passed in Fabric with `credential_provider="notebookutils"`. Local automatic discovery still requires notebookutils, and local SemPy 0.14.1 failed through both its default provider and an injected Azure CLI credential/XMLA path | `fabric_rlm/lakehouse.py:321-419,704-813`; `fabric_rlm/semantic_model.py:532-667,785-817`; `knowledge_lakehouse_sources.py`; `knowledge_semantic_model.py` | **Universal:** accept typed injectable credentials and discovery providers for both adapters. **Source metadata:** explicit schema-aware catalogs and immutable item paths. **Runtime integration:** retain notebookutils as the supported in-Fabric provider, but do not make it the only real-auth path. |
| Schema-enabled Lakehouse paths require the schema segment | Run the explicit catalog first with `Tables/<table>`, then with `Tables/dbo/<table>` | The catalog path must address the physical Delta table | The first form returned `No files in log segment`; `Tables/dbo/<table>` returned 186, 1,800, and 2/3 | Caller-supplied `LakehouseSource.catalog`; schema discovery in `fabric_rlm/lakehouse.py:930-1135` | **Source metadata:** include schema in catalog identities and paths. **Universal:** improve the Delta-reader error so a missing schema path is reported as path/catalog mismatch rather than a generic Delta log failure. |
| A newly deployed Direct Lake definition requires initialization before DAX use | Deploy the TMDL model, query `INFO.VIEW.TABLES()` and `INFO.VIEW.MEASURES()`, trigger one dataset refresh, then repeat | A deployment harness should not claim readiness until metadata is queryable | Before refresh, both metadata views were empty and measure references failed. Refresh request `9048468c-cdff-4182-9748-9958e67f55ab` completed, after which all three measures returned their reference values | Evaluation deployment workflow; no frozen core code involved | **Evaluation mechanism:** trigger and poll initial refresh, then assert table/measure metadata and reference DAX before adapter evaluation. |
| Equivalent aggregate semantics pass across real Delta and Direct Lake adapters | Compare independent Python references, Lakehouse SQL, frozen `LakehouseSource.query`, Power BI DAX, and frozen `SemanticModel.aggregate` | Equivalent source semantics should produce equivalent answers | All five paths returned 186 available units, 1,800 complete produced units, and a 2/3 SLA rate. The semantic adapter reported 3 tables, 16 columns, 3 measures, and 3.768 seconds total query time | Real Fabric artifacts recorded in `evidence/fabric-probe.json` | No domain-specific core patch. Retain cross-source reference assertions as an integration gate. |
| Lifecycle and verifier safety behavior is covered by offline tests | Run `runner offline` | Stale evidence must stay stale; incompatible knowledge must be rejected; failed/skipped verifiers must not claim verification; failures must be bounded | Targeted suite completed with 197 passed and 1 skipped. It covers snapshot/schema drift, stale package rejection, invalid operation fallback, timeout capture, verifier repair, verifier crash logging, and disabled verifier behavior | `knowledge_preflight.py`, `knowledge_api.py`, `knowledge_operations.py`, `runtime.py`, `knowledge_evidence.py` | No domain patch. Retain these as universal invariants and add live end-to-end cases when credentials are available. |

## Dataset coverage

The fixture generator creates three independent domains and three equivalent
naming variants:

- **Inventory/fulfillment:** dated snapshots, partial shipments, repeated
  customer/product names, entity IDs, and value-multiplying join traps.
- **Manufacturing quality:** production counts, multiple defect rows, weighted
  defect rates, line rates, and incomplete reporting periods.
- **Service operations:** tickets, repeated response events, policy-specific
  SLAs, reopened events, and a deliberately undefined root-cause question.

Each domain has five evaluation questions. Reference calculations preserve
period, grain, units, and entity identity. The service root-cause case expects
abstention or a request for a missing definition.

## Conclusions

**General execution capability:** The frozen code has broad Python execution
and strong deterministic lifecycle safeguards. In the smoke, six of nine
answers reached the correct numeric value, but none preserved every required
period, grain, unit, and identity field. One unconstrained execution multiplied
production through a join, one run exhausted its turn budget, and one
knowledge-enabled run abstained despite sufficient metadata.

**Generalization of learned behavior:** Not established. File `learn()` produced
an operation catalog but no lessons, and all six B/C packages reported zero
lessons. B and C did not produce a fully correct answer. Inventory C regressed
from numerically correct A/B answers to an abstention. Semantic structural
lessons and retrieval remain observably sensitive to English, date, and
business vocabulary.

**Portability across tested sources:** Established for the three tested
aggregate semantics. Independent Python references, the Lakehouse SQL endpoint,
the frozen explicit-catalog Lakehouse adapter, Direct Lake DAX, and the frozen
semantic-model adapter all returned 186, 1,800, and 2/3. The semantic adapter
ran from an exact frozen wheel in Fabric with notebookutils credentials; the
Lakehouse adapter ran locally against real OneLake Delta files with an explicit
schema-aware catalog and a process-local credential shim. Local automatic
Lakehouse discovery and local SemPy/XMLA authentication remain unsupported.
Generic SQL remains unsupported as a knowledge source.

**Remaining unsupported claims:** The smoke measures only three descriptive-name
questions with one repetition. Full 15-question accuracy, abbreviated and
camel-case naming robustness, three-run reliability, stable A/B/C learning
gain, development-call efficiency, workbook correctness, and cross-source
answer equivalence across all 15 questions remain unmeasured. Cross-source
equivalence is proven only for three aggregate tasks, not joins, weighted
ratios, event deduplication, naming variants, insufficient metadata, or learned
lesson transfer. Live data refresh, schema-change, stale-evidence, and recovery
trials against the Fabric artifacts remain unmeasured.
