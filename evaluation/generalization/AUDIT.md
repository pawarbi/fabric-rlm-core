# Generalization audit

**Baseline:** `b5226712a9aa41c3173d5f427e81244c333c0179`  
**Evaluation branch:** `eval/generalization-b522671`  
**Core/skill freeze:** no mismatches  
**Generated large fixture:** 250,000 rows, 30,000,049 bytes  
**Offline reliability suite:** 197 passed, 1 skipped  
**Live A/B/C accuracy, speed, tokens, and cost:** unmeasured; OpenRouter
authentication returned 401 before a trial executed.

## Findings

| Finding | Reproduction | Expected | Actual | Affected code | Proposed fix |
|---|---|---|---|---|---|
| No executable ARR specialization found | `runner offline`; inspect `audit.arr_mentions` | Examples and comments must not count as runtime specialization | 12 ARR mentions were documentation/comments; 0 were executable | Documentation in `analytical_integrity.py`, `knowledge_lessons.py`, `semantic_model.py`, and `trajectory.py` | None. Preserve the executable/documentation distinction in future audits. |
| Semantic structural learning depends on English date naming patterns | The offline probe creates equivalent semantic profiles with `Period[IsCurrentQuarter]` and `prd[icq]` | Equivalent metadata mappings should yield equivalent lessons | Descriptive profile created 1 current-period lesson; abbreviated profile created 0 | `fabric_rlm/knowledge_lessons.py:49-59`, `:110-151`, `:375-380` | **Universal:** add declared semantic-role fields to the lesson mechanism. **Source metadata:** adapters should map columns/measures to roles such as current-period and period-dimension. **Optional skill:** keep English regexes only as fallback hints. |
| Lesson retrieval has English and business-vocabulary triggers | Offline score probe compares `customer` with unfamiliar `acct` | Equivalent mapped concepts should retrieve the same lesson | `customer` scored 0.7; `acct` scored 0 | `fabric_rlm/knowledge_retrieval.py:20-52`, `:124-152` | **Universal:** retrieve by stable semantic tags plus lexical overlap. **Source metadata:** provide concept aliases. **Optional language/domain skill:** supply English or domain trigger vocabulary; do not hard-code it in core. |
| Large files cannot use learned registered operations and are rejected on reuse | Generate the 30 MB CSV; run offline audit; bind the learned package | Large sources should retain an exact stable identity or explicitly use a source version | `snapshot_exact=false`, 0 operations; preflight classifies every later observation as `inexact`, and runtime refuses stale bindings | `fabric_rlm/knowledge_sources.py` bounded head/tail snapshot; `knowledge_operations.py:150-155`; `knowledge_preflight.py:29,107-108,168-169`; `runtime.py:1264` | **Universal:** stream a full digest without placing contents in prompts, or accept an adapter-supplied immutable version identity. **Source metadata:** remote adapters should expose ETag/version/table snapshot IDs. |
| `learn()` on ordinary files adds operations but no lessons | Learn the small inventory CSV in the offline audit | Configuration B should have a measurable learned-behavior delta or clearly disclose that it is only an operation catalog | 1 registered aggregate operation, 0 lessons | `fabric_rlm/knowledge_api.py:185+`; `fabric_rlm/knowledge_lessons.py:110` limits structural lessons to semantic models | **Universal:** capture typed telemetry for file operations and promote source-agnostic grain, failure, and strategy lessons. Domain definitions remain metadata/skills. |
| Registered operations cover only a narrow analytical algebra | Inspect discovered operations and run inventory/service questions requiring joins, deduplication, earliest-event logic, and weighted ratios | Supported tasks should execute equivalently across supported sources or fall back explicitly | Core supports semantic measure, sum/avg/count aggregates, Lakehouse aggregate, and Lakehouse pre-aggregate join; local multi-file joins and weighted-ratio/event semantics require unconstrained Python fallback | `fabric_rlm/knowledge_operations.py:113-198,204-393`; `knowledge_execution.py:140-147,416-840` | **Universal:** add typed relational operators for join cardinality, distinct entities, first/last event, ratio-of-sums, and snapshot selection. **Source metadata:** relationships, keys, event time, completeness flags. **Optional skill:** domain metric definitions. |
| Generic SQL is not a knowledge source | Inspect the registry and execution implementation set | SQL should be tested only if a supported adapter exists | Registry supports CSV, JSON, JSONL, Parquet, Delta, Lakehouse, semantic model, and opaque files; no generic SQL adapter exists | `knowledge_sources.py:137-140`; `knowledge_lakehouse_sources.py:496-510`; `knowledge_execution.py:140-147` | **Universal:** define a read-only SQL source contract with schema/version identity and bounded execution. **Source metadata:** dialect, endpoint, credentials, table contracts. Do not put business rules in the adapter. |
| Real Fabric access is environment-dependent beyond Azure CLI auth | `az login`; SQL endpoint discovery/query succeeded; run `fabric-probe.json` | A locally authenticated user should be able to profile supported Fabric handles, or the required runtime should be explicit | REST and `sqlcmd -G` worked. `LakehouseSource` automatic discovery failed outside notebookutils; SemPy failed without `FabricAnalyticsTokenCredentialProvider` | `fabric_rlm/lakehouse.py` discovery path; `fabric_rlm/semantic_model.py`; `knowledge_lakehouse_sources.py`; `knowledge_semantic_model.py` | **Universal:** accept injectable credential/discovery providers. **Source metadata:** explicit catalogs and immutable paths. Keep notebookutils/SemPy as optional Fabric-runtime integrations, not the only real-auth path. |
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
and strong deterministic lifecycle safeguards. The new non-ARR fixtures,
references, graders, freeze checks, and reliability probes run locally.
End-to-end model answer correctness and efficiency remain unmeasured because
the available OpenRouter credential was rejected before execution.

**Generalization of learned behavior:** Not established. File `learn()` produced
an operation catalog but no lessons; semantic structural lessons and retrieval
are observably sensitive to English/date/business vocabulary. No claim can be
made that configuration B or C improves unseen-domain answers, and the harness
will report per-question cases where learning hurts once live trials run.

**Portability across tested sources:** Real CSV profiling and a real Fabric SQL
endpoint query succeeded. Mocked Lakehouse/semantic-model adapter tests passed.
The library's real Lakehouse and semantic-model profiling did not run locally
despite Azure CLI authentication because those paths required Fabric
notebook-runtime credential/discovery providers. Generic SQL is unsupported as
a knowledge source. Equivalent synthetic semantics were therefore not proven
across files, Lakehouse, and semantic models.

**Remaining unsupported claims:** Accuracy, learning gain, naming robustness of
final answers, three-run reliability, tokens, provider cost, latency, workbook
correctness, and cross-source answer equivalence are not measured. Real
synthetic Lakehouse/semantic-model evaluation, data refresh/schema-change
end-to-end runs, and confident-wrong versus abstention rates remain pending a
working model credential and a Fabric execution path that can load the fixtures
without changing the frozen implementation.
