# Frozen execution-policy comparison

## Decision

**Do not treat context-only execution as a demonstrated optimization or promote
it to the default.** All 162 scheduled trials were recorded, but the
combined A/B/C nonregression gate failed for `context_only` on CSV, Parquet and
local Delta. These failures are in B; C meets the observed per-question counts
against A, but has no active lessons and therefore supplies no evidence that
enrichment caused its better counts.

Removing preplanning eliminated 54 selector calls, but B/C combined used only
0.39% fewer total tokens, took 25.76% more task wall time, and cost 2.14% more in
reported per-task provider cost. Successful-status wrong numeric answers rose
from 1 to 5. This is a failed acceptance result, not a claim that every use of
context-only execution is worse.

The implementation remains opt-in on `fix/knowledge-nonregression`; the existing
`auto` default and the original user checkout were not changed by this study.
This report supplements, rather than replaces, [CURRENT_AUDIT.md](CURRENT_AUDIT.md).

## What was held fixed

| Item | Recorded setting |
|---|---|
| Evaluation commit | `2814d10065ae2495b0ffa00b55cb3d990fcb789f` |
| Core implementation commit | `7d474de2fc32494e0c6465b369c452a8865e6ae9` |
| Context-only API commit | `a45ebbddb90e04f84daaa1b14dd66b1c3e70e368` |
| Model | `openai/gpt-4.1-mini` through OpenRouter |
| Sampling and limits | Temperature 1; 4,096 max output tokens per call; six turns; configured runtime timeout 120 seconds; DSPy cache disabled |
| Model identity evidence | All 850 recorded LM-history entries identify the same requested/response model aliases; no immutable provider model-version guarantee |
| Skills | `skills=[]`; autoload disabled; bundled skill files frozen |
| Fixture/schedule seed | `20260908`; not a provider sampling-seed guarantee |
| Tasks | One previously used question per domain, three repetitions per A/B/C arm, three representations, two policies: 162 tasks |
| Packages | The same 18 development-only B/C snapshots from the typed `d0306e9` study; no new learning, enrichment or development runs |
| Source/policy order | Parquet auto/context; local Delta context/auto; CSV auto/context; seeded A/B/C order within each run |
| Budget | Shared $2 ceiling with $1 reserve; account-usage delta **$0.788592420** |
| Environment | Windows, Python 3.11.9; DSPy 3.2.1, DuckDB 1.5.0, pandas 2.3.3, NumPy 2.4.0, PyArrow 23.0.1, delta-rs 1.5.0 |

The original and copied package hashes, source/fixture hashes, all core/skill
files and all 12 harness modules remained unchanged. Reference calculations,
answers and grading code were not added to agent inputs. Evaluation evidence
did not enter any subsequent package or trial.

This is a **regression study, not a new holdout**. The integrations are real
local CSV/Parquet reads and real local Delta queries, not live Fabric services.
Provider caching and OS caching were not controlled; cached-token counts were
recorded. Original package-development cost is historical, not zero: reused
packages record load time and `null` learning/enrichment times.

## Results

A = no package; B = learn-only; C = the previously enriched package.
Each A/B/C entry below is out of nine trials.

| Source | Policy | Correct A/B/C | Numeric check A/B/C | Incomplete A/B/C | B/C median seconds | Combined gate |
|---|---|---:|---:|---:|---:|---|
| CSV | auto | 3 / 3 / 3 | 7 / 6 / 8 | 1 / 3 / 1 | 18.40 / 15.13 | Fail |
| CSV | context_only | 3 / 5 / 4 | 7 / 8 / 7 | 2 / 1 / 0 | 15.13 / 19.08 | Fail |
| Parquet | auto | 4 / 0 / 3 | 7 / 7 / 9 | 1 / 3 / 0 | 11.63 / 12.06 | Fail |
| Parquet | context_only | 2 / 2 / 4 | 7 / 7 / 9 | 2 / 2 / 0 | 18.29 / 13.45 | Fail |
| Local Delta | auto | 0 / 1 / 0 | 0 / 4 / 2 | 9 / 5 / 7 | 14.73 / 19.82 | Pass against a zero-correct baseline |
| Local Delta | context_only | 2 / 0 / 3 | 2 / 1 / 4 | 7 / 8 / 4 | 28.34 / 31.71 | Fail |

Correct means the unchanged value/units/period criterion. Grain and the original
strict verdict remain separate in the raw and derived results. These three
questions do not require entity-ID outputs, so their numeric checks do not
demonstrate identity correctness. Numeric checks are taken from the existing
grader and remain status-gated, not an independent regrade.

Incomplete includes abstentions, exhausted/failed tasks and serialization
markers anywhere in the answer. It can overlap correct fields: CSV/context B
inventory r0 has the correct value and period but opaque Timestamps inside its
claims payload. No marker or status rule was loosened to make a gate pass.

### Per-question results

Each entry is A/B/C out of three. Inventory =
`inventory_available_units`; manufacturing = `manufacturing_complete_units`;
service = `service_first_response_sla_rate`.

| Source | Policy | Domain/question | Correct A/B/C | Numeric A/B/C | Incomplete A/B/C |
|---|---|---|---:|---:|---:|
| CSV | auto | Inventory | 3 / 3 / 3 | 3 / 3 / 3 | 0 / 0 / 0 |
| CSV | auto | Manufacturing | 0 / 0 / 0 | 2 / 3 / 3 | 0 / 0 / 0 |
| CSV | auto | Service | 0 / 0 / 0 | 2 / 0 / 2 | 1 / 3 / 1 |
| CSV | context_only | Inventory | 3 / 3 / 3 | 3 / 3 / 3 | 0 / 1 / 0 |
| CSV | context_only | Manufacturing | 0 / 2 / 1 | 2 / 2 / 2 | 1 / 0 / 0 |
| CSV | context_only | Service | 0 / 0 / 0 | 2 / 3 / 2 | 1 / 0 / 0 |
| Parquet | auto | Inventory | 3 / 0 / 3 | 3 / 2 / 3 | 0 / 3 / 0 |
| Parquet | auto | Manufacturing | 1 / 0 / 0 | 2 / 3 / 3 | 0 / 0 / 0 |
| Parquet | auto | Service | 0 / 0 / 0 | 2 / 2 / 3 | 1 / 0 / 0 |
| Parquet | context_only | Inventory | 2 / 2 / 3 | 3 / 3 / 3 | 1 / 1 / 0 |
| Parquet | context_only | Manufacturing | 0 / 0 / 1 | 3 / 2 / 3 | 0 / 1 / 0 |
| Parquet | context_only | Service | 0 / 0 / 0 | 1 / 2 / 3 | 1 / 0 / 0 |
| Local Delta | auto | Inventory | 0 / 1 / 0 | 0 / 1 / 0 | 3 / 2 / 3 |
| Local Delta | auto | Manufacturing | 0 / 0 / 0 | 0 / 3 / 2 | 3 / 0 / 1 |
| Local Delta | auto | Service | 0 / 0 / 0 | 0 / 0 / 0 | 3 / 3 / 3 |
| Local Delta | context_only | Inventory | 2 / 0 / 3 | 2 / 0 / 3 | 1 / 3 / 0 |
| Local Delta | context_only | Manufacturing | 0 / 0 / 0 | 0 / 1 / 1 | 3 / 2 / 1 |
| Local Delta | context_only | Service | 0 / 0 / 0 | 0 / 0 / 0 | 3 / 3 / 3 |

The context-only regressions are B inventory completion on CSV, B manufacturing
numeric/completion on Parquet, and B inventory correctness/numeric/completion
on Delta. Auto also regresses on CSV service and Parquet inventory/manufacturing.
The Delta auto pass is relative parity with A's **0/9 correct, 9/9 incomplete**;
it is not usable absolute accuracy. No service trial passes the full criterion.

### Efficiency, including unsuccessful tasks

These totals include all 54 B/C tasks per policy, not only successful survivors.
Per-arm, source, question and repetition metrics are in `analysis.json`.

| Metric | auto | context_only |
|---|---:|---:|
| Correct answers | 10 / 54 | 18 / 54 |
| Numeric checks passing | 36 / 54 | 36 / 54 |
| Incomplete tasks | 19 / 54 | 15 / 54 |
| Successful-status wrong numeric answers | 1 | 5 |
| Selector LM calls | 54 | 0 |
| Prompt tokens | 751,314 | 733,759 |
| Completion tokens | 70,119 | 84,443 |
| Total tokens | 821,433 | 818,202 |
| Cached tokens reported | 550,400 | 585,728 |
| Total task wall seconds | 1,037.66 | 1,304.99 |
| Median task wall seconds | 15.08 | 19.10 |
| Reported per-task provider cost | $0.245120040 | $0.250365060 |
| Instrumented adapter calls / failed calls | 70 / 28 | 39 / 15 |

The five context B/C numeric errors are substantive, not rounding: 0 rather
than 2/3 twice, 3,600 rather than 1,800 twice, and 52,882 rather than 1,800 once.
Separately, an A Parquet/context answer of 0.6667 fails the unchanged 1e-6
relative tolerance; that is a precision failure, not the same error as zero.

Task wall time includes LM setup, construction and execution, but excludes
package loading, persistence and budget polling. Adapter calls are not total
I/O: direct pandas/filesystem reads and some metadata operations are not
instrumented. Across all arms, runtime failure reasons are 37 `max_turns`, six
`abstained`, three `output_validation_failed`, and 116 with no failure reason.
No runtime failure was classified as a wall-clock timeout. A relative gate
does not impose an absolute accuracy floor or a separate confident-error cap.

## Findings and reproductions

Trace locators below use source/policy, question, arm, zero-based repetition.
For example, Delta/context manufacturing C/r1 is:
`lakehouse__context_only.artifacts\traces\manufacturing_complete_units__descriptive__r1__C.trajectory.jsonl`.
The raw archive contains every trace; `selected-cases.json` and
`trace-slices.json` provide the focused reproductions.

### F1. A named literal escapes the provenance screen

**Reproduction:** Delta/context manufacturing C/r1 has five unsuccessful turns,
then assigns `answer = {"status": "success", "value": 52882, ...}` and calls
`SUBMIT(answer=answer)`. It asserts a successful prior merge that never happened.
Expected: rejection or an explicit unsupported result. Actual: successful
submission, no integrity finding, and `verification_outcome="verified"`.
The recorded verifier scope is only `output_validator: passed`, not semantic
verification.

**Affected code:** `fabric_rlm\analytical_integrity.py:807-922` follows literal
SUBMIT arguments but not `ast.Name` bindings; `runtime.py:3204-3236` uses that
screen; the harness's `_require_value` only checks shape/non-null value.
`probes.json` reproduces one finding for a direct literal and none for the
equivalent assigned dictionary. Strict mode alone cannot make this screen
recognize a binding it does not inspect.

**Proposed universal fix:** conservatively follow literal-only bindings and
reassignments instead of treating a variable name as proof of computation;
test computed values, aliases and mutations to prevent false positives.
Expose verification scope distinctly from analytical correctness. No business
measure names or domain-specific value rules belong in this guard.

### F2. C still supplies no active learned behavior

**Reproduction:** all nine original C packages have zero active lessons;
inventory/manufacturing each have one candidate, service none. No lesson was
injected in any trial. `runtime.py:2124-2155` explicitly returns no guidance
without active lessons; promotion requires independent supporting runs
(`knowledge_lessons.py`, promotion policy).

**Expected/actual:** C was intended to test enriched guidance, but the development
protocol did not create qualifying repeated evidence. Reconstructed initial
system/user prefixes match in all 54 B/C pairs, all 27 A policy controls, and
all 27 context-only A/B pairs. Thus context-only B/C initially supplies the same
model-visible information as cold execution in this study. Its different
outcomes cannot be claimed as learned-guidance gains.

**Proposed experiment fix:** produce independent, successful, development-only
observations at the same supported grain, check that useful lessons activate
and are actually injected, then freeze new packages before a new holdout.
Do not lower promotion thresholds or reuse evaluation evidence. Grain,
completeness and SLA meanings remain **source/task metadata**, not core rules.

### F3. Source-interface friction dominates many Delta failures

**Reproduction:** Delta/context inventory B/r0 omits `sources=`, treats a returned
dictionary as a DataFrame, and mistakes unprinted expressions for missing data.
Manufacturing C/r1 also guesses `.name`, uses unsupported `DESCRIBE`, and joins
a second table absent from the first handle's catalog.

**Expected/actual:** use the declared query interface and interpret actual rows;
instead, bounded turns are spent guessing interfaces or recovering. Each Delta
handle in this harness intentionally contains only one table, so a cross-handle
SQL join is not a same-catalog join. This is not evidence that all Fabric
Lakehouses fail to join their own tables.

**Code/proposal:** `lakehouse.py:319-407` and
`evaluation\generalization\representations.py:60-82`. Publish bounded,
machine-readable method, argument, catalog-scope and result-shape contracts;
make stdout-only inspection explicit or provide bounded previews. A future
same-catalog test must bind a real shared catalog and create new packages,
not silently rebind these snapshots. These are **source capabilities/configuration**,
not inventory/manufacturing/service rules.

### F4. Date values still become opaque payload markers

**Reproduction:** Parquet/auto inventory B/r0 computes 186 but returns a native
`date` for the period. CSV/context inventory B/r0 returns the correct scalar
and period but opaque pandas Timestamps inside `claims.supported`. Five trials
contain markers overall.

**Expected/actual:** usable typed values throughout the required payload versus
opaque markers; correct answer fields and incomplete ancillary payloads must
not be conflated with arithmetic mistakes. **Affected code:** `serializers.py`,
`freeze`, and the harness's answer/claim contract.

**Proposed universal fix:** an explicit, tested date/datetime/Timestamp wire
policy and recursive output-contract validation. Preserve timezone semantics;
do not stringify arbitrary objects or invent dates. Calendar interpretation
belongs in **metadata**. The timestamp schema-descriptor fix below does not
fix date-value serialization.

### F5. A valid join doubles a parent measure

**Reproduction:** CSV/context manufacturing B/r2 merges complete production with
raw defect rows on period/line, then sums production: 3,600 rather than 1,800.
An independent CSV probe has two eligible production rows, four raw joined
rows, and two rows after an existence/semi-join; totals are 3,600 and 1,800.

**Expected/actual:** select production rows with matching defects without
multiplying production, versus successful execution of an inflated aggregation.
**Affected surface:** generated join/aggregation code and relationship/grain
contracts, not a hardcoded ARR measure.

**Proposed universal mechanism:** cardinality/row-count and measure-conservation
checks, or a validated semi-join/aggregate-before-join operation. Join keys,
intended multiplicities and completeness definitions belong in **source
metadata or an optional analytical skill**, not named-domain core branches.

### F6. Correct numbers are frequently delivered without their reporting period

**Reproduction:** CSV/context service B/r0 returns 2/3 with `period=null`;
CSV/auto manufacturing C/r0 returns 1,800 with only `"complete reporting periods"`.
Expected periods are March 1-3 and January 2026 respectively.

**Code/proposal:** the harness `_task_text` asks for period, but `_require_value`
deliberately does not enforce it so the omission remains measurable. In a
separate experiment, use the existing **generic output-contract mechanism**
to enforce task-required fields without giving the model oracle answers.
Actual reporting windows and calendar rules remain **metadata**. Do not
silently loosen the frozen grader or force every non-temporal task to have a date.

### F7. Persistence rejected a legitimate SQL timestamp type -- fixed before evaluation

**Reproduction:** loading the original service Delta package through the normal
store rejected `lakehouse_type="TIMESTAMP WITH TIME ZONE"`.
Expected: persist the profiled structural type unchanged; actual: validation
failure. **Fix:** `7d474de`, `knowledge_store.py` schema-descriptor validation,
recognizes the bounded WITH/WITHOUT TIME ZONE timestamp forms in type fields.
Round-trip fingerprints and restrictive free-text/diagnostic controls remain
intact. This is a **universal metadata-format fix**, not a general SQL type parser.

### F8. Windows environment restoration broke Git provenance -- fixed before evaluation

**Reproduction:** restoring an empty `GIT_CONFIG_VALUE_*` through Python's
Windows environment mapping leaves native inherited Git state without that
value. The first full suite failed the SHA lookup after 3,270 other passes.
Expected: retain caller Git configuration; actual: Git exit 128.
**Fix:** `2814d10`, `runner.py:_git_sha`, explicitly passes `os.environ.copy()`.
The real-Git regression test failed first; the final full offline run is
3,272 passed, 11 skipped and two live tests deselected. This is **harness
portability**, not a data-domain or model failure.

### F9. Saved LM histories are not immutable per-request message captures

**Reproduction:** 107 first history entries already contain later assistant
turns when serialized after the run. Comparing their whole message lists as
first requests would be incorrect. `probes.json` explicitly compares only
reconstructed initial system/user prefixes.

**Code/proposal:** `runner.py:write_trial_trace` captures the provider library's
history after execution. Snapshot messages at dispatch time for future
request-level audit guarantees; deep-copying only at the end is too late.
This is **universal observability**, not evidence of cross-trial input leakage.

## Evidence and commands

[evidence/policy-comparison/manifest.json](evidence/policy-comparison/manifest.json)
indexes **379 original files** in `policy-2814d10-raw.zip` plus nine derived/test
artifacts, with archive/member/file SHA-256 hashes. The archive SHA-256 is
`4e10b2eef50c320fd024136b691711961928ebb70f1a90f1bd844ed91d073be7`.
Credential-pattern and active-key scans found no matches. Byte-preserving
Git attributes prevent line-ending normalization from invalidating hashes.

The unchanged seeded fixtures and original development-only packages are in
the existing `current-core` archives, referenced by this manifest. Live reuse
requires their original source identities; offline inspection of extracted
results does not require live sources or credentials.

Run from an isolated checkout of `2814d10`; never switch or reset the original
user worktree. The recorded live arguments were:

```powershell
$p = 'C:\Users\sandeeppawar\.copilot\session-state\4fc1d5d0-25b1-4323-b111-f91ae7bc94cd\files'
Set-Location "$p\knowledge-current-eval"
$userKey = [Environment]::GetEnvironmentVariable('OPENROUTER_API_KEY', 'User')
if ($userKey) { $env:OPENROUTER_API_KEY = $userKey }
python -u -m evaluation.generalization.current_smoke --fixtures "$p\knowledge-current-data" --output "$p\knowledge-context-comparison" --frozen-runs "$p\knowledge-typed-smoke" --knowledge-executions auto context_only --repetitions 3 --max-cost-usd 2
```

The output path already exists and is intentionally non-overwritable: choose a
fresh output directory for any replay. Original package setup is not rerun.
For offline inspection, extract the raw archive to a new directory and run:

```powershell
python -m evaluation.generalization.evidence_summary --input 'C:\path\to\extracted\csv__context_only.json' --output 'C:\path\to\new-summary.json'
python -m pytest tests evaluation\generalization\tests -m 'not primary and not secondary_free' --disable-warnings --tb=short
```

## Separate conclusions

**General execution capability:** the runtime can execute across all three
local representations, but incomplete answers, join inflation, interface
mistakes and the named-literal provenance gap prevent a reliability claim.

**Generalization of learned behavior:** unestablished. C has no active lessons;
context-only A/B/C has matching reconstructed initial inputs. No enrichment
benefit or cross-domain learned transfer can be inferred from stochastic score
differences.

**Portability across tested sources:** real CSV, Parquet and local Delta were
measured, with materially different outcomes and explicitly scoped catalogs.
This is not a live Fabric SQL/semantic-model portability result.

**Remaining unsupported claims:** new holdouts with at least five questions
per domain, the full naming matrix on this candidate, genuinely active
development-enriched packages, live Fabric integrations and mutation/recovery
matrices, large-fixture live tasks, entity-identity tasks, and workbook
correctness remain unmeasured here. Independent claim-to-evidence coverage is
also unmeasured; `supported` flags are self-reported. The full suite/build and
successful archive checks are engineering controls, not substitutes for those
claims or for passing the accuracy/completion gate.
