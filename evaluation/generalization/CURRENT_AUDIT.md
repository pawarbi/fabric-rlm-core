# Current-core knowledge nonregression audit

**All 81 corrected-candidate evaluation trials completed; all three source
nonregression gates failed. No efficiency or generalization benefit is
established.** Coverage and core/fixture integrity checks passed. This report
distinguishes that candidate from the earlier 27-trial smoke. Neither uses a
new blinded holdout.

## Experimental identity

| Stage | Evaluation SHA | Core | Input bindings | Measured scope |
|---|---|---|---|---|
| Initial current-main smoke | `f99c742f38fd33eba14364567caad4a25aa51fde` | `7347c278525c6bdebb586eaa78db541c2ed9d7d7` | CSV/Parquet path strings; local Delta handles | 27 evaluation + 18 separate development runs, one repetition |
| Binding-only checkpoint | `f78d8d3` | Same initial core | Explicit `File` handles | Offline source-equivalence checks; no live comparison |
| Corrected candidate | `d0306e93f573d54a5e624661dd2ceb34005161de` | `b75c60e7c7d89f943a7ee02ecae4477a0e64c93f` | Explicit `File` / local Delta handles | 81 evaluation + 18 separate development runs, three repetitions |

Implementation branch: `fix/knowledge-nonregression`, based on main at
`abb092456dd47bc71f56292fbda95ec07ac0db6c`. The evaluation branch is separate:
`eval/knowledge-nonregression`. The original checkout and user log files remain
untouched. No main merge, release, new dependency, or live database mutation
was performed.

Both smoke runs use `openai/gpt-4.1-mini` through OpenRouter, temperature 1,
six turns, a configured 120-second runtime timeout, empty skills and disabled
skill autoloading. **This is not default source-skill-enabled product behavior.**
The model name is an alias, not an immutable provider-version guarantee.
DSPy caching is disabled; provider and operating-system caches are not under
experimental control. Provider cached tokens are recorded.

Seeded source order is Parquet, local Delta, CSV. A/B/C order is randomized
within questions. A has sources and no package; B has `learn()` only; C is
enriched from separate development runs. B/C carry their own source bindings.
All packages are saved before evaluation and checked for mutation. Reference
answers and grader code are not model inputs, and evaluation evidence is never
enriched into later packages.

There are three distinct regression questions: latest net available inventory,
complete-period produced units, and first-response SLA rate. Repeating them
does not create new unseen questions. Equivalent data in three domains and
three column-naming variants is checked as complete row multisets, retaining
duplicates. Only descriptive names enter these live smoke runs. The separate
250,000-row / 30,000,049-byte fixture is retained, not used in live smoke tasks.

## Completed first smoke

Primary correctness retains value, units, reporting period and required entity
identity. Value/identity correctness is also reported independently. Abstentions,
missing values, serialization markers, and failed tasks remain incomplete.

| Source | Arm | Primary correct | Value/identity correct | Incomplete |
|---|---|---:|---:|---:|
| CSV | A | 1/3 | 1/3 | 2/3 |
| CSV | B | 0/3 | 2/3 | 1/3 |
| CSV | C | 0/3 | 2/3 | 1/3 |
| Parquet | A | 0/3 | 1/3 | 2/3 |
| Parquet | B | 0/3 | 2/3 | 1/3 |
| Parquet | C | 0/3 | 2/3 | 1/3 |
| Local Delta | A | 0/3 | 0/3 | 3/3 |
| Local Delta | B | 1/3 | 2/3 | 1/3 |
| Local Delta | C | 0/3 | 1/3 | 2/3 |

**CSV fails per-question parity:** A returns correct net available inventory
(`186`, with the required period); B/C fail that task. Gains on other questions
must not conceal this regression. Parquet and Delta relative passes are against
weak cold baselines, not acceptable absolute accuracy or a generalization win.
Two A submissions contain serialization markers.

All nine C packages have zero active enriched lessons. Development produced
22 query-execution observations, mostly insufficiently corroborated candidates.
The development protocol asks for one useful aggregation and one unsafe /
expensive probe per domain. It does not guarantee two independent successful
observations at the same grain. **Do not attribute B/C differences to activated
learning or lower promotion thresholds to manufacture a learning benefit.**

The provider account delta, including development, was **$0.144004**, under the
shared $2 ceiling with $1 reserved for in-flight work / billing delay. An account
delta can include unrelated account activity. One repetition and a shared
Windows host do not support a stable latency claim.

## Corrected candidate: gate still fails

Each arm has nine trials per source: three questions, repeated three times.
The following are unchanged historical grades, including their protocol
requirements and the limitations identified below.

| Source | Arm | Primary correct | Value/identity correct | Incomplete |
|---|---|---:|---:|---:|
| CSV | A | 3/9 | 7/9 | 2/9 |
| CSV | B | 3/9 | 9/9 | 0/9 |
| CSV | C | 2/9 | 8/9 | 1/9 |
| Parquet | A | 5/9 | 8/9 | 2/9 |
| Parquet | B | 2/9 | 8/9 | 1/9 |
| Parquet | C | 3/9 | 8/9 | 1/9 |
| Local Delta | A | 0/9 | 1/9 | 8/9 |
| Local Delta | B | 3/9 | 5/9 | 4/9 |
| Local Delta | C | 0/9 | 2/9 | 6/9 |

Per-question primary correctness, each out of three:

| Source / question | A | B | C |
|---|---:|---:|---:|
| CSV / available inventory | 3 | 3 | 2 |
| CSV / complete-period units | 0 | 0 | 0 |
| CSV / first-response SLA rate | 0 | 0 | 0 |
| Parquet / available inventory | 2 | 2 | 3 |
| Parquet / complete-period units | 3 | 0 | 0 |
| Parquet / first-response SLA rate | 0 | 0 | 0 |
| Delta / available inventory | 0 | 3 | 0 |
| Delta / complete-period units | 0 | 0 | 0 |
| Delta / first-response SLA rate | 0 | 0 | 0 |

**Do not equate every failed gate with arithmetic failure.** CSV C's inventory
miss computes `186` and the correct date, but returns `status="ready"`, which
the unchanged grader does not accept. The prompt did not enumerate successful
status values. The status check also prevents value checks from running in
that case; the reported value/identity count is therefore not an independent
raw-number-only score. This is an evaluation-contract limitation, not evidence
that the calculation failed. Preserve the result; clarify the status protocol
before a future frozen run, rather than adding a post-hoc accepted synonym.

Parquet's manufacturing regression is different: A reports `["2026-01"]`;
B/C calculate `1800` but omit the actual period or describe only the filter.
The host scalar packet can accelerate a calculation while losing required
answer context. All three Delta B SLA tasks are incomplete versus two A
incompletions; neither B nor C matches A's one correct SLA value. Broader
inventory gains must not hide those regressions.

Two successfully labelled answers have wrong numerical values (one Parquet B,
one Delta C). Two other answers retain opaque **native date**, not integer,
markers. One CSV A abstention submits `NaN`, triggering an evidence-capture
warning. Full answers, exact checker outcomes and field-level grades remain
in the raw archives.

All nine corrected C packages again contain **zero active lessons**, despite
28 development query observations. No lessons are injected. SQL eligibility
was fixed, but the development protocol still does not guarantee corroborating
successful runs at a shared grain. This remains an operation-package experiment,
not a demonstrated benefit from enriched lessons.

| Source | Median wall seconds A / B / C | Evaluation prompt tokens A / B / C |
|---|---|---|
| CSV | 13.32 / 14.35 / 16.79 | 102,544 / 109,150 / 88,660 |
| Parquet | 18.37 / 19.47 / 14.78 | 87,350 / 85,801 / 97,861 |
| Delta | 14.98 / 22.99 / 23.13 | 122,284 / 113,138 / 131,574 |

These include failed and incomplete trials, not post-selected successes.
Faster Parquet C loses primary correctness; lower CSV C prompt tokens coexist
with the protocol failure. Neither establishes the requested efficiency win.
The corrected account delta, including development, is **$0.329727024** under
the shared $2 ceiling. The two current-core smoke deltas total approximately **$0.474**;
that is not a claim about total historical session spending.

## Reproduced mechanisms and corrections

Current main already retained raw source bindings after host operations and
captured host-operation evidence. Those older evaluated-core shortcomings were
not patched a second time.

| Finding | Expected versus observed before correction | Affected code and reproduction | Classification / disposition |
|---|---|---|---|
| Incomplete parity reports could pass | Empty, missing, duplicate or out-of-range trials must not pass; a completion regression must not hide behind equal answer scores. | `knowledge_benchmark.py::cold_parity`; `test_parity_requires_exact_repetition_coverage`, `test_parity_rejects_completion_regression_hidden_by_equal_accuracy`. | **Universal**, fixed in `eefd228`. Harness gate also checks the complete planned manifest, including wholly omitted questions. |
| Oversized host output became successful learning evidence | A real 101-group result exceeds a 100-row bound and must not become successful evidence. It was previously harvested as successful. | `knowledge_evidence.py::_execution_status`; `test_over_bound_host_result_is_not_successful_learning_evidence`. | **Universal**, fixed in `31768fe`; rejected evidence remains rejected. |
| Host-only work appeared to use zero source calls | Count instrumented host work, without counting both the wrapper and its adapter telemetry. | `knowledge_evidence.py::_trajectory_source_call_summary`, runtime and benchmark consumers; `test_real_host_call_accounting_is_passive_across_sources`. | **Universal instrumentation**, fixed in `7347c27`; direct pandas/filesystem I/O is still outside coverage. |
| Correct integer aggregates became opaque markers | Real CSV sums and boolean aggregates must survive JSON submission. `np.int64(7)` was opaque while `np.float64(2.5)` survived. | `serializers.py::_as_scalar`; real-file submission regression plus scalar/container controls in `test_serializers.py`. | **Universal**, numeric-only PR #79 fix and guards in `2a39fbe`. Broader Decimal/date/time conversion policy is not included. |
| Planner selected incomplete / speculative operations | Net available inventory needs both on-hand and allocated quantities. The planner selected only an on-hand sum and literal `"max"`, yielding null; raw data remained available. | `runtime.py` operation-selector prompt; `test_planner_contract_requires_complete_grounded_operations`; initial CSV B/C inventory traces. | **Universal prompt mitigation**, `8d7c2d9`: optional, task-complete, grounded selection or fallback. Not a host-enforced semantic guarantee. The actual CSV date column was typed **string**; no date-name or `"max"` blacklist was added. |
| Complete grouped SQL did not promote | Two independent successful, untruncated grouped Delta SQL reads should establish an execution-grain fact. `lakehouse_sql` observations were excluded by the allowlist. | `knowledge_evidence.py::_is_aggregate_observation`, `knowledge_lessons.py::_valid_grain_lessons`; `test_grouped_lakehouse_sql_learns_only_from_complete_results`. | **Universal evidence eligibility**, fixed in `b75c60e`. Truncated results do not promote; raw query types, independent-run thresholds and verification requirements remain intact. |
| Harness paths were treated as inline CSV | Traces show `StringIO(path)` rather than opening the source, including cold runs. This was not source starvation. | `evaluation/generalization/representations.py::domain_sources`; typed-versus-path schema/fingerprint/operation equivalence tests. | **Harness correction**, `f78d8d3`: use public `File` bindings identically across A/B/C. Original artifacts remain separate. |
| Shape checks were mistaken for analytical proof | `_require_value` only checks a non-null value. A passed check does not prove units, period, joins, identity, or numerical correctness. | `runner.py::_require_value`, recorded verifier check lists and independent grader results. | **Reporting distinction**, preserved. **Universal proposal:** explicit verification scope and claim-to-evidence linkage. Domain formulas / definitions belong in **source metadata or optional skills**. |

Integrated offline validation on the candidate: **3229 passed, 11 skipped,
2 live tests deselected**; source and wheel builds succeeded. The grouped SQL
regression first failed on the complete-result case; its truncated control
already passed. No failing test or promotion threshold was removed.

## Newly exposed limits: proposals, not silently applied fixes

Reproduce the following by inspecting the named question / arm / repetition in
`typed-d0306e9-raw.zip`. `typed-failure-cases.json` resolves exact trace paths;
`typed-trace-slices.json` preserves the diagnostic code and error excerpts.

| Finding / reproduction | Expected versus actual | Affected surface / next correction |
|---|---|---|
| Parquet inventory A/r2 and B/r0 | Both compute `186`; `datetime.date(2026, 3, 31)` becomes an opaque period marker. | **Universal serialization boundary:** test and define native date/time encoding. The numeric-only fix did not claim to cover these types; broader conversion remains separate. |
| CSV SLA A/r2 | A non-finite answer should not invalidate all otherwise harvestable evidence. `NaN` passes the non-null shape check, and evidence capture logs that the value is not JSON-compatible. | **Universal:** finite JSON submission rules and evidence failure isolation, without converting undefined values to fabricated numbers. The run is an abstention, not a correct answer. |
| Delta manufacturing A/r0; inventory C/r0; SLA C/r0 | The model guesses `.schema()`, `.catalog_name`, `.to_pandas()`, omits required `sources=`, and confuses list rows with dict rows. It exhausts turns despite available data. | **Universal interface guidance / normalization:** expose the actual bounded source contract; separately measure source-skill-enabled execution. Do not relax source allowlists or invent convenience methods only for this dataset. |
| CSV SLA A/r2; Delta manufacturing A/r0 | Bare `.head()`, `.columns`, or query-result expressions execute but produce no stdout. The model concludes it cannot see available data. | **Universal REPL observability:** explicit print guidance or bounded expression feedback. Automatic echo changes execution behavior and needs its own compatibility tests; this is not source starvation. |
| Parquet manufacturing B/r0 | A complete answer needs the actual reporting period. The selected scalar supports the value, but synthesis submits `period=None`. | **Universal:** make operation reuse conservative about complete task context. Test an opt-in context-only path to isolate metadata/lesson value from host preplanning. **Metadata:** actual period definitions, not core date-name rules. |
| CSV inventory C/r2 | A correct computation is counted incomplete because `ready` is outside an unstated success vocabulary. | **Harness:** specify accepted statuses before the next study; retain original grades and distinguish protocol failures from mathematical errors. |
| C package preparation, all sources | A meaningful enrichment comparison needs supported, active learned behavior. One useful and one unsafe probe often produce unrelated / single-run candidates. | **Harness:** expand independent development-only coverage, report promotion readiness and setup cost, freeze before evaluation. **Core:** explain non-promotion; do not lower independence requirements. |

The proposed context-only experiment would retain source validation, bindings,
metadata and active lessons while skipping optional host-operation preplanning.
It would be opt-in, with existing behavior unchanged. It is not implemented or
measured here and would require an explicit public-interface decision.

## Remaining learning and portability limits

Current `knowledge_lessons.py` still contains executable English current-period /
date patterns and derived-measure name patterns including `nrr`, `grr`, retention
and churn. These nominate or activate particular lesson kinds; comments
containing "ARR" are not the evidence for this finding. Explicit source
declarations are already supported and must not be confused with name inference.
The historical naming/retrieval probes are not fresh live semantic-model tests.

**Universal next mechanisms:** expose why evidence did not promote; nominate and
retrieve from declared semantic roles / supported query structure rather than
adding more business keywords; separate execution facts from analytical truth;
and make optional operations preserve complete task coverage. **Source metadata:**
join keys, grain, units, period definitions, semantic-role mappings and stable
versions. **Optional domain skills:** domain vocabulary and specialized formulas.
No domain-specific rule was patched into core.

Real local CSV, Parquet and Delta execution is exercised here. Local
`LakehouseSource` SQL is not a live Fabric SQL endpoint. The prior published
three-task Fabric / semantic-model equivalence result remains historical;
this continuation does not establish its current-core learning, naming, or
mutation/recovery behavior. No unsupported source is counted as tested.

**General execution:** broad computational capability, not reliably accurate.
**Learned behavior:** benefit remains unestablished; C has no activated lessons.
**Portability:** real execution is measured across local representations, but
equivalent-answer reliability is not established.
**Unsupported claims:** bulletproof behavior, a new-domain holdout result,
default-skill product quality, complete live Fabric coverage, robust multilingual
or naming transfer, workbook correctness, exhaustive I/O accounting, independent
claim coverage, and a stable speed/token benefit.

## Deliverables and exact reproduction

`evidence/current-core/manifest.json` records archive hashes, every member hash
and the original roots needed to resolve absolute trace links after extraction:

| Artifact | Contents |
|---|---|
| `initial-f99c742-raw.zip` | 136 files: first smoke results, provider/trajectory traces, development records, frozen packages and manifests |
| `typed-d0306e9-raw.zip` | 244 files: corrected smoke with the same complete evidence categories |
| `seeded-fixtures.zip` | 140 files: all seeded naming variants, local representations, definitions, questions, private references and the large CSV |
| `initial-analysis.json`, `typed-analysis.json` | Derived per-source, per-question and per-arm scores, coverage, performance and learning readiness |
| `typed-failure-cases.json`, `typed-trace-slices.json` | Reproduction locators and actual failing code / output |
| `initial-numeric-probe.json`, `offline-tests.xml` | Deterministic initial-core scalar exposure and integrated candidate test results |

Raw bytes are preserved, including the observed non-standard `NaN` JSON value;
they were not rewritten into passing outputs. Archive readback hashes were
verified, and credential-pattern / active-key scans found no matches before
publication. The manifest is the portable locator map, not a new source identity.
The evidence directory disables Git text normalization so published member and
derived-file hashes continue to match on Windows and other platforms.

From an isolated checkout, reproduce the measured candidate without modifying
the original checkout:

```powershell
git worktree add --detach "..\knowledge-repro" d0306e93f573d54a5e624661dd2ceb34005161de
Set-Location "..\knowledge-repro"
$data = Join-Path $env:TEMP ('knowledge-data-' + [guid]::NewGuid())
$results = Join-Path $env:TEMP ('knowledge-smoke-' + [guid]::NewGuid())
python -m evaluation.generalization.runner prepare --output $data --representations --large-rows 250000
python -m pytest tests evaluation\generalization\tests -m "not primary and not secondary_free"
python -u -m evaluation.generalization.current_smoke --fixtures $data --output $results --max-cost-usd 2 --repetitions 3
foreach ($source in "csv", "parquet", "lakehouse") {
    $run = Get-Content -LiteralPath (Join-Path $results "$source.json") -Raw | ConvertFrom-Json
    if ($run.gate.passed -ne $true) { throw "Nonregression gate failed: $source" }
}
```

Use an environment-provided `OPENROUTER_API_KEY`, never a literal key in a
command. Batch exit code zero means execution completed, **not** that parity
passed. Archived packages retain their original absolute bindings; regenerate
fixtures / packages for a new machine instead of reusing stale bindings.
Logical fixtures and ordering are seeded; model sampling and provider alias
resolution are not deterministic.
