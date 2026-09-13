# Post-serialization generalization evaluation

**Status: live matrices running; no final accuracy or learning-gain conclusion yet.**
This report supersedes the narrative conclusions from the invalid BS/CS run,
not its preserved raw evidence. Core and bundled skills have not been edited
during this continuation.

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
