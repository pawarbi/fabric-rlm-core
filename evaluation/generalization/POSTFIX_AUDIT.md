# Post-serialization generalization evaluation

**Status: all 549 planned evaluation trials completed. General reliability and
positive learned-behavior transfer are not established.**
This report supersedes the narrative conclusions from the invalid BS/CS run,
not its preserved raw evidence. Core and bundled skills have not been edited
during this continuation.

## Final measured results

All 405 fixed-core naming trials, 135 unfixed descriptive trials and 9
counterfactual-ambiguity trials completed. Both matrices have zero core/skill
hash drift, zero fixture hash changes, and intact saved package hashes.
Provider account deltas including development runs were $1.116066600 and
$0.357881436 respectively; the ambiguity probe reported $0.031666536.
Total for these three runs: **approximately $1.506**, well below $25.
No trial-level exception was recorded in either matrix. That does not imply
completion: turn exhaustion, invalid output statuses and abstentions are
retained in the counts below.

Each fixed-core arm has 135 trials: 126 answerable cases and 9 unanswerable
root-cause cases. Correct abstention is reported separately, never as a
completed analytical answer.

| Fixed-core metric | A: no package | B: learn only | C: enriched |
|---|---:|---:|---:|
| Correct answer: value, units, period, required identity | 14/126 | 4/126 | 1/126 |
| Original strict answer score (includes grain wording) | 0/126 | 0/126 | 0/126 |
| Correct value and required identity | 76/126 | 28/126 | 33/126 |
| Incomplete analytical tasks (all abstentions included) | 31/135 | 77/135 | 73/135 |
| Successful-status wrong value or identity, answerable cases | 23 | 26 | 27 |
| Correctly recognized unanswerable root-cause cases | 2/9 | 2/9 | 4/9 |
| Serialization marker trials | 0 | 0 | 0 |
| Median evaluation wall seconds | 14.91 | 11.53 | 10.54 |
| Evaluation prompt tokens | 1,304,510 | 1,069,327 | 1,055,573 |
| Evaluation completion tokens | 96,156 | 76,913 | 69,361 |
| Provider-reported cached tokens | 908,672 | 633,088 | 673,152 |
| Evaluation provider cost | $0.3990 | $0.3573 | $0.3279 |

The shorter B/C times are not a success: many tasks stop with insufficient
information. Primary correctness remains sensitive to unit/period wording.
The value-and-identity results show that this is not merely a grading-vocabulary
problem. All nine B packages and all nine C packages contained **zero lessons**,
and zero lessons were injected. These measure the effects of operation catalogs
and the package execution path, not learned-lesson transfer.

### Domain and naming robustness

Correct value and identity, pooled across three repetitions. Inventory and
manufacturing have 15 answerable trials per cell; service has 12, plus 3
unanswerable cases excluded from this value metric.

| Domain / naming | A | B | C |
|---|---:|---:|---:|
| Inventory / descriptive | 10/15 | 2/15 | 2/15 |
| Inventory / abbreviated | 11/15 | 1/15 | 2/15 |
| Inventory / camel | 9/15 | 2/15 | 3/15 |
| Manufacturing / descriptive | 8/15 | 2/15 | 0/15 |
| Manufacturing / abbreviated | 8/15 | 2/15 | 2/15 |
| Manufacturing / camel | 9/15 | 2/15 | 3/15 |
| Service / descriptive | 9/12 | 5/12 | 6/12 |
| Service / abbreviated | 5/12 | 5/12 | 8/12 |
| Service / camel | 7/12 | 7/12 | 7/12 |

Mappings preserve meaning across column naming variants. Table names and
English task/definition language remain unchanged: this is **column-naming
robustness**, not fully opaque source naming or multilingual task robustness.
There is no consistent naming winner. Three repetitions are insufficient to
establish stable small differences; per-repetition counts are in the summary.

### Per-question results, including improvements and regressions

Correct value and required identity out of nine trials (three naming variants
times three repetitions); root-cause row instead counts appropriate abstention.
Full primary/strict scores and field-level checks are in
`evidence\final-naming-summary.json` and the raw results.

| Question | A | B | C |
|---|---:|---:|---:|
| Latest available inventory | 7 | 1 | 1 |
| Fill rate | 8 | 0 | 0 |
| Join-safe inventory value | 7 | 4 | 6 |
| Open ordered units | 6 | 0 | 0 |
| Top customer identity and shipped units | 2 | 0 | 0 |
| Complete-period produced units | 7 | 4 | 2 |
| Incomplete-period units excluded | 7 | 1 | 1 |
| Overall defect rate, not average of rates | 3 | 1 | 1 |
| Weighted defect rate | 2 | 0 | 0 |
| Worst production line and rate | 6 | 0 | 1 |
| Breached ticket identities | 7 | 2 | 4 |
| Deduplicated first response | 3 | 4 | 4 |
| First-response SLA rate | 5 | 4 | 7 |
| Distinct reopened tickets | 6 | 7 | 6 |
| Appropriate root-cause abstention | 2 | 2 | 4 |

Service contains observed improvements, so an overall average must not be read
as "packages always hurt." Conversely, repeated zero results on fill rate and
open-order quantities cannot be concealed by those improvements.

### Before/after serialization

The comparable descriptive subset has 45 trials per arm.

| Metric | Unfixed A/B/C | Fixed A/B/C |
|---|---|---|
| Serialization-marker trials | 12 / 0 / 0 | 0 / 0 / 0 |
| Correct value and identity | 17 / 14 / 11 | 27 / 9 / 8 |
| Primary correct answers | 3 / 3 / 0 | 4 / 1 / 0 |

The unfixed markers occurred in inventory (7) and manufacturing (5).
The fixed run has zero markers across all 405 naming trials. Together with
the deterministic scalar exposure probe, this supports the serialization fix,
**not a general accuracy fix**. B/C numerical scores regressed in this live
comparison; runs were sequential and packages were independently built, so
those differences cannot be causally attributed to serialization.

### Missing definitions and unsupported causal claims

The probe produced **zero explicit abstentions, eight substantive answers,
and one failed task**. Inventory assumed available means on-hand stock (230)
in all three repetitions; another consistent definition gives net stock (186).
Manufacturing assumed an unspecified quality KPI, often a yield rather than
the witness defect rate. Service supplied two causal-sounding answers despite
no cause field, with one other run failing.

The raw classifier labels these eight as `guessed_wrong` relative to witness A.
That label is **not proof of numerical wrongness**: inventory's 230 exactly
matches witness B. The defensible finding is failure to request the missing
convention or acknowledge that cause is unobserved. Private witness alternatives
and full answers are retained in the raw archive; no witness was agent input.

### Final conclusions and remaining gaps

**General execution capability:** Broad, but not reliably accurate. The fixed
system computes across unrelated domains and removes the observed scalar-loss
failure, yet wrong joins, missing answer metadata and incomplete tasks remain.

**Generalization of learned behavior:** Not demonstrated. File packages created
no lessons. A real, deterministically reproduced one-operation restriction
explains a subset of the B/C failures; semantic learning still has separately
demonstrated English/business/date-name dependencies. Do not recommend package
learning as a universal accuracy improvement based on this evaluation.

**Portability:** Established only for the historical three real Fabric
aggregate tasks and the tested local file paths. The new trials do not establish
all-question parity with SQL/Lakehouse or semantic models, nor learned transfer.

**Unsupported claims:** "Bulletproof," general positive learning gain, complete
live change/recovery coverage on Fabric, workbook correctness for these tasks,
independent claim-to-evidence coverage, and complete I/O counts. The large
dataset was profiled and exercised deterministically, not used in this live
question matrix. No findings were patched into core during evaluation.

## Earlier checkpoint details and reproduction

The sections below preserve the pre-run design, legacy observations, detailed
mechanism findings and source limitations. Any statement that final results
were pending refers to that earlier checkpoint; the completed results above
take precedence.

## Experimental identity

- Fixed-core evaluation starts at `7d7b6ee` on `eval/postfix-generalization`.
- Unfixed comparison uses core from `48786272ad5ca2cb8f329805400b56d799dd5b99`.
  Byte comparison confirms that only `fabric_rlm\serializers.py` and
  `fabric_rlm\lakehouse.py` differ. Both processes use the same runner, grader,
  fixture paths, model alias, turn limit, sampling settings and trial seed.
- The original historical baseline is `b5226712a9aa41c3173d5f427e81244c333c0179`;
  it is **not** the actual SHA of the new runs. Each new result records its
  actual SHA, core/skill hash manifest, harness hashes and fixture hashes.
- New outputs have private, run-specific artifact directories. B/C packages
  are saved before evaluation and their fingerprints checked before and after
  trials. Evaluation results never enter enrichment.
- A: sources without a package. B: `learn()` package with its source bindings.
  C: a separate package enriched from two development runs per domain/variant.
  B/C source bindings are automatic; adding duplicate aliases is invalid.
- Main matrix: 15 questions x 3 naming variants x 3 repetitions x A/B/C =
  **405 evaluation trials**, plus 18 development runs.
  Unfixed comparison: **135 evaluation trials**, plus 6 development runs.
  Ambiguity probe: 3 counterfactual-witness cases x 3 repetitions = **9 trials**.
- Model: `openai/gpt-4.1-mini`; DSPy cache disabled; temperature 1; six turns;
  library timeout 120 seconds. Provider prompt caching is observed and reported,
  not claimed disabled. Model alias resolution is not an immutable model
  version guarantee. Core-version runs are sequential, not randomized; do not
  interpret small latency differences as causal serialization effects.
- Each large matrix has a $5 account-usage stop with a $1 in-flight reserve,
  separately from the call cap. Together with the nine-case probe these stay
  below the user's remaining $25 ceiling at the previously observed prices.
  Account usage can include unrelated activity and provider billing delay.

## Confirmed findings and reproductions

| Finding | Expected versus observed | Reproduction / affected code | Proposed fix classification |
|---|---|---|---|
| Partial operation result removes needed sources | A fill-rate task needs shipped and ordered totals. A valid selected sum returns only `310` ordered units; synthesis receives only `knowledge_result`, not `shipment_events`. The original B run has 17 incomplete registered-operation trials; all 17 retained traces match their recorded final payload. This is **not** initial package-binding starvation. | `python -m evaluation.generalization.mechanism_probe --fixtures evaluation\generalization\generated --output evaluation\generalization\evidence\operation-source-restriction.json`. Real CSV profiling, preflight and host execution; only planner text is scripted. `runtime.py:1327-1334` selects one operation; `:1432-1438` removes original source aliases. Reference calculation is independently 262/310. | **Universal:** require task-complete operation coverage, or fall back / support bounded additional operations. **Source metadata:** join keys, grain, metric components. **Optional domain skill:** definitions only. No core fix applied. |
| Package effects are not evidence of lesson transfer | File B and C packages in the original descriptive run have zero lessons; injected-lesson count is zero. A/B/C differences cannot be attributed to helpful or harmful learned lessons. | `raw-results\postfix-descriptive.json`, package summaries; `knowledge_api.py:350-420` harvests typed evidence and promotes it; file paths need actual eligible observations. New runs preserve complete packages for inspection. | **Universal:** expose learning coverage and reasons evidence did not promote. Improve source-agnostic telemetry/promotion only with tests. **Metadata/optional skills:** domain meaning must be declared, not inferred from metric names in core. |
| Shape verification is not answer correctness | The runtime correctly records that `_require_value` ran and passed. That check only requires a non-null value, so an abstention or a wrong number can carry `verified=true`. This is not evidence a skipped semantic verifier passed. | Original B fill-rate trace: `verifier_execution.checks=[{check: output_validator, outcome: passed}]`; `runtime.py:2749-2771`. New metrics retain the check list and scope. | **Harness:** separate checker execution, independent reference grading and claim support. **Universal proposal:** typed verification scope would make this distinction harder to misread. No domain rule belongs in core. |
| Historical trace paths were overwritten | Different result files wrote identical IDs under one shared `traces` directory. All 45 original A trace links no longer match their recorded payloads; original B/C have 42 matching submissions each and 3 unsubmitted failures. Original answer records remain usable, but their trace chain is not fully recoverable. | `evidence\legacy-trace-audit.json`; old `runner.py` used `output.parent / "traces"`. | **Harness, fixed:** unique artifact directory and refusal to reuse result paths; persist package contents and hashes. Do not claim retrospective provenance was recovered. |
| Definition deletion alone does not prove ambiguity | Some removed definitions are already expressed by the question or schema. A correct answer without redundant text is not automatically an unsafe guess. | `insufficient_metadata.ambiguity_cases`: private witnesses show 186 versus 230 available units, pooled versus equal-line-weight quality rates, and distinct unobserved service causes, all consistent with visible data. | **Harness, fixed:** require counterfactual witnesses and keep them out of task inputs. Generic caveats/unsupported flags do not count as abstention. |
| Original headline scores mixed task completion with expected abstention | The agreed grader counts a correct request for a missing cause as behavioral success. User requested abstentions counted as incomplete analytical tasks. | `evidence_summary.summarize`, tested on nested strict results and wrong entity IDs. | **Harness:** retain original grades but report analytical completion and expected-abstention detection separately. Do not silently revise the scorer. |

## Original descriptive run, rechecked without new model calls

The denominator is 45 trials per arm (42 answerable tasks and 3 expected
root-cause abstentions). These are legacy observations, not the pending final
matrix. The revised task-completion accounting below treats every abstention
as incomplete, including appropriate abstention on the unanswerable question.

| Metric | A | B | C |
|---|---:|---:|---:|
| Answer correct under value/units/period/identity criterion | 4 | 2 | 1 |
| Original strict correct, excluding abstention | 0 | 0 | 0 |
| Value and required entity identity correct | 27 | 12 | 12 |
| Incomplete analytical tasks | 8 | 24 | 18 |
| Successful-status wrong value or identity | 9 | 8 | 14 |
| Right value/identity, framing mismatch | 23 | 10 | 11 |
| Expected abstention correctly detected | 1/3 | 1/3 | 1/3 |
| Known evaluation provider cost | $0.12573 | $0.11713 | $0.11682 |
| Median evaluation wall seconds | 13.42 | 10.11 | 11.22 |

`evidence\legacy-descriptive-summary.json` retains per-domain, per-question,
per-repetition and individual-dimension counts. Original grader outcomes are
retained alongside these categories. Faster incomplete tasks are not efficiency
gains. Unit wording such as "defects per produced unit" versus "ratio" remains
a scoring limitation; no units were dropped from the agreed primary criterion.
Grain token overlap is only a wording heuristic, not proof of correct grain.

## Reliability and source coverage

The runtime audit distinguishes documentation from behavior. Its literal
uppercase `ARR` scan finds comments/examples, not executable specialization;
that scan does **not** establish domain neutrality. Actual runtime dependencies
remain in `knowledge_lessons.py:49-59`: English current-period/date naming,
and measure-name patterns including `nrr`, `grr`, retention and churn.
The behavioral probe generates one structural lesson for
`Period[IsCurrentQuarter]` and zero for mapped `prd[icq]`; retrieval gives the
English `customer` trigger 0.7 and `acct` zero. See the prior F15 nomination
probe for multilingual and statistical-name counterexamples.
**Universal proposal:** semantic-role and capability-based nomination/retrieval.
**Source metadata:** stable concept/role mappings.
**Optional skills:** English/domain vocabulary, rather than more core synonyms.

The large fixture is 250,000 rows / 30,000,049 bytes, not prompt-sized.
Its bounded file fingerprint is inexact and it receives zero registered
operations. This is a **source-size boundary**, not an inventory or service
specialization. **Universal proposal:** streamed exact identity or trusted
source version IDs with bounded reading; **metadata:** immutable source versions.
No large-file learned-operation performance claim follows from generating it.

The targeted suite was rerun: **197 passed, 1 skipped** (Windows FIFO test).
The old-baseline freeze command exits 1 because exactly the two intentional
serialization files differ; its reliability tests exit 0. New run manifests
cover the actual fixed core and must show zero drift.
This suite covers source identity, refresh/schema drift, stale/incompatible
knowledge, invalid plans, bounded worker timeout and failed/skipped verifiers.
These are local deterministic tests and mocked external adapters, not new
live Fabric mutation results.

`evidence\fabric-probe.json` is **historical real-integration evidence** from
the original frozen core: three aggregate answers (186, 1800, 2/3) agree across
independent references, Lakehouse SQL, the explicit-catalog Lakehouse adapter,
Direct Lake DAX and the in-Fabric semantic adapter. This continuation has not
rerun those integrations or demonstrated all 15 tasks through each source.
Generic SQL is not a supported knowledge adapter. Do not call the scripted
planner reproduction a live model trial or a real Fabric adapter test.

## Limits and conclusions at this checkpoint

**General execution capability:** Cross-domain computation is demonstrated,
but reliable completion is not. The scalar fix has deterministic cross-domain
coverage and zero observed leaks in the valid original 135-trial fixed run.
The invalid 90 BS/CS trials do not contribute to that claim.

**Generalization of learned behavior:** Not established. Zero file lessons,
operation-selection regressions and the existing semantic naming/retrieval
dependencies prevent a stronger claim. Final naming results are pending.

**Portability across tested sources:** Historical equivalence is supported for
three aggregate semantics only. Rich joins, event semantics and learned lessons
are not thereby portable. New baseline/fixed file comparison is pending.

**Remaining unsupported claims:** "Bulletproof," a positive general learning
gain, complete cross-source parity, stable gain across model versions, independent
claim-to-evidence coverage, and live workbook correctness. Main file tasks do
not request workbooks. Source-call counts exclude direct pandas/filesystem
reads; answer `supported` flags are self-reports. Learn/profiling setup latency
is not included in per-trial wall time. A test can disprove universality with a
counterexample, but these finite synthetic trials cannot prove it.
