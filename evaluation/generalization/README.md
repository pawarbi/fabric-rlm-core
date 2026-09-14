# Current-main knowledge regression evaluation

Core of the recorded 81-trial candidate: `b75c60e7c7d89f943a7ee02ecae4477a0e64c93f`, based on
`origin/main` at `abb092456dd47bc71f56292fbda95ec07ac0db6c`. The core fixes
are on `fix/knowledge-nonregression`; this separate evaluation branch
freezes that implementation and its bundled skills. Each run records the
actual evaluation checkout SHA and per-file hashes, not just this core SHA.

The current branch also contains the opt-in context-only API from
`a45ebbddb90e04f84daaa1b14dd66b1c3e70e368`; that mode was not present in the
recorded 81-trial study. Its live effect must be measured separately.
Package reuse additionally required `7d474de`: the standard store rejected the
legitimate `TIMESTAMP WITH TIME ZONE` source descriptor. The narrow persistence
fix preserves the original descriptor, fingerprints and privacy checks.

**Completed results:** the corrected 81-trial smoke at `d0306e9` failed all
three per-source nonregression gates. See [CURRENT_AUDIT.md](CURRENT_AUDIT.md)
for the per-question results, protocol limitations, proposed next mechanisms,
exact reproduction and independently preserved raw archives.

The fixture generator, independent Python references, grader and trace
controls are reused from `277cd3b782bec4290bff6034a586faacedd0dc86`.
These are **regression questions previously used in development/evaluation,
not a new blinded holdout**. Their old results are not measurements of this
core. No reference answers or evaluation-run evidence enter development or
subsequent model trials.

The first smoke at `f99c742f38fd33eba14364567caad4a25aa51fde` used plain
path strings for CSV/Parquet. Traces showed these being mistaken for inline
CSV content. This revision uses the public `File(...)` input contract for
both formats, identically in A/B/C. Tests confirm unchanged source schemas,
snapshots and operation catalogs; original raw results must remain separate.

The candidate also includes numeric-only scalar serialization, optional
complete-task/grounded-literal operation planning, and promotion of complete
grouped SQL observations. The planner change is guidance, not a semantic
correctness guarantee. SQL learning retains independent-run thresholds and
does not turn unverified execution into verified analytical evidence.

## Run in PowerShell

Use a checkout of `eval/knowledge-nonregression`. The evaluation environment
used DuckDB 1.5.0, PyArrow 23.0.1, delta-rs 1.5.0 and DSPy 3.2.1. The batch
also records Python, pandas and NumPy versions. No dependency changes are
required in the core package.

```powershell
$data = Join-Path $env:TEMP ('knowledge-data-' + [guid]::NewGuid())
$results = Join-Path $env:TEMP ('knowledge-smoke-' + [guid]::NewGuid())
python -m evaluation.generalization.runner prepare --output $data --representations
python -m pytest evaluation\generalization\tests
python -m evaluation.generalization.current_smoke --fixtures $data --output $results --max-cost-usd 2 --repetitions 1
```

Set `OPENROUTER_API_KEY` in the environment; never put it in a command or
artifact. The batch shares one monetary ceiling, including development,
and reserves $1 for in-flight work and delayed provider accounting. It
randomizes source order and A/B/C configuration order. DSPy caching is off;
provider caching is not controllable, and cached tokens are recorded.
`openai/gpt-4.1-mini` is an alias, not an immutable provider-version guarantee.

Each source representation has six separate development runs and nine
smoke trials at one repetition. `--repetitions 3` makes 27 evaluation
trials per representation, still only three distinct questions. For all
15 questions, omit `--smoke` when using `runner live` and supply at least
141 task calls for three repetitions:

```powershell
python -m evaluation.generalization.runner live --fixtures $data --output "$results\full-csv.json" --representation csv --variants descriptive --repetitions 3 --max-live-calls 141 --max-cost-usd 3
```

CSV, Parquet and local Delta through `LakehouseSource` are real
implementations, not mocked adapters. Conversion tests compare every
column and the complete multiset of rows across three domains and three
naming variants, retaining duplicates. Data is read as typed Arrow tables
and written without overwriting existing representations:
[DuckDB Arrow conversion](https://duckdb.org/docs/current/clients/python/conversion.html#apache-arrow),
[Arrow Parquet writer](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html),
[delta-rs writer](https://delta-io.github.io/delta-rs/usage/writing/).
The 250,000-row fixture is retained but excluded from live smoke prompts.

## Paired operation-policy comparison

The completed 162-task study is recorded in
[POLICY_COMPARISON.md](POLICY_COMPARISON.md), with raw evidence under
`evidence\policy-comparison`. Context-only eliminated selector calls but did not
pass the combined nonregression gate or establish an efficiency/enrichment win.

Compare `auto` and `context_only` with the **same previously frozen B/C
packages**, not packages independently regenerated between conditions:

```powershell
# $data is the unchanged fixture directory used by the previous frozen run.
# $frozenRuns is its batch directory, containing csv.artifacts\packages, etc.
$results = Join-Path $env:TEMP ('knowledge-policy-' + [guid]::NewGuid())
python -u -m evaluation.generalization.current_smoke --fixtures $data --output $results --frozen-runs $frozenRuns --knowledge-executions auto context_only --repetitions 3 --max-cost-usd 2
```

This runs 162 evaluation tasks and no new development tasks: three questions,
three repetitions, A/B/C, three source representations and two policies.
Source order is seeded; policy order alternates between source blocks. The
model, skills, task text, limits, fixture contents and B/C package fingerprints
are fixed across conditions. A is rerun as a no-package control in each policy.
Provider caching and alias resolution remain uncontrolled and recorded.

The previous run's B/C snapshots came from development only. This command reads
those package files, **not its evaluation answers or traces**. It checks all 18
original package hashes across the batch and validates current source identity
when loading. Both original snapshots and saved copies are checked during trials;
a changed/deleted snapshot stops the batch explicitly. No `learn()` or `enrich()`
call runs in reuse mode. Load time is measured; original learning/enrichment time
is `null`, not invented as zero cost. Historical development cost must be kept
separate from the marginal cost of this comparison.

The runner's audit snapshots are package DTOs rather than store envelopes.
Reuse converts them through the standard bounded, privacy-validating store in a
temporary directory, then uses the normal source-validation/rebinding API.
Fingerprints and original snapshot bytes are not rewritten.

Outputs are `csv__auto.json`, `csv__context_only.json`, and equivalent files for
Parquet / Lakehouse, each with a separate artifact directory and per-question
gate. Do not pool policies when judging regressions. This isolates optional host
preplanning; with the recorded packages' zero active C lessons it still cannot
demonstrate an enriched-lesson benefit. It is not a new-domain holdout or a live
Fabric integration test.

## Gate and interpretation

`batch.json` describes execution coverage and spending; each source JSON
contains raw answers, unchanged reference grades, actual verifier outcomes,
timings, tokens, frozen package references and the complete planned schedule.
Its sibling `.artifacts` directory contains provider/trajectory traces,
development metrics, package snapshots and code/fixture hashes. The live
runner refuses a mismatched imported library checkout.

`gate.passed` requires full planned-trial coverage, unchanged core/fixtures,
and no per-question loss in reference correctness, value/entity correctness,
or completion on answerable tasks. Expected abstention is graded separately;
all abstentions remain reported as incomplete. A parity pass is not an
absolute accuracy floor or statistical proof of generalization. Setup costs,
unknown measurements and regressions must remain visible; faster failures
are not wins.

The original strict grade remains alongside the value/units/period/entity
grade. Self-reported `supported` flags and a passed shape validator do not
establish independent analytical verification. Source-call metrics cover
instrumented calls/host operations, not every direct pandas or filesystem
operation. Workbooks are not requested in this smoke.

Live Fabric services, live semantic models, new held-out task families,
the full live naming matrix and live mutation/recovery are **not established
by this local smoke**. PR #79's NumPy scalar fix is not included in the
original `f99c742` baseline. Its numeric-only fix is included in this
candidate; the broader Decimal/date/time conversion policy is not.
Serialization markers are always reported rather than silently removed.
