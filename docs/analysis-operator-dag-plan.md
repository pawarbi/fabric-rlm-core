# Implementation Plan: Reproducible Deep-Insight Analysis DAG

## Overview

Implement the approved analysis-DAG specification as a sequence of small,
test-first vertical slices. The first milestone proves deterministic planning,
execution, evidence registration, exact KPI decomposition, and reproducible
local benchmarking. Later operator families remain blocked until their own
ground-truth and transfer gates pass.

## Architecture Decisions

- Keep the API under `fabric_rlm.experimental` until the transfer gate passes.
- Use immutable typed records and JSON-compatible serialization.
- Derive every child seed from a required run seed and stable identifiers.
- Separate planner validation, operator execution, evidence persistence, and
  benchmark scoring.
- Keep deterministic exact operators internal; add statistical dependencies
  only after approval and compatibility testing.
- Treat dataset preparation as a verified build step, never an implicit network
  action during tests or analysis.

## Phase 1: Contracts and Reproducibility

### Task 1: Define immutable analysis contracts

**Description:** Add experimental records for the analysis brief, DAG, nodes,
budgets, operator results, and evidence entries.

**Acceptance criteria:**
- Records reject missing IDs, invalid seeds, unknown execution modes, and
  non-JSON-compatible parameters.
- Records serialize to a stable canonical representation.
- No analytical dependency is added.

**Verification:**
- `python -m pytest tests\test_analysis_dag.py -q`

**Dependencies:** None

**Files likely touched:**
- `fabric_rlm\experimental\analysis_contracts.py`
- `tests\test_analysis_dag.py`
- `fabric_rlm\experimental\__init__.py`

**Estimated scope:** Medium

### Task 2: Derive seeds and fingerprints deterministically

**Description:** Add domain-separated seed derivation and canonical SHA-256
fingerprints for datasets, DAGs, nodes, split assignments, and results.

**Acceptance criteria:**
- Equal inputs produce equal seeds and fingerprints across process runs.
- Dataset, node, fold, or repetition changes produce independent child seeds.
- Ambient `random` or NumPy state cannot affect derivation.

**Verification:**
- `python -m pytest tests\test_analysis_dag.py -q -k "seed or fingerprint"`

**Dependencies:** Task 1

**Files likely touched:**
- `fabric_rlm\experimental\analysis_reproducibility.py`
- `tests\test_analysis_dag.py`

**Estimated scope:** Small

### Task 3: Validate typed DAGs before execution

**Description:** Validate registered operators, dependencies, acyclicity,
source authorization, seeds, parameters, and aggregate run budgets.

**Acceptance criteria:**
- Cycles, unknown operators, duplicate IDs, unauthorized sources, missing
  dependencies, unseeded nodes, and budget overflow are rejected.
- Validation returns a stable topological schedule with explicit parallel waves.
- Errors name the exact offending node and field.

**Verification:**
- `python -m pytest tests\test_analysis_dag.py -q -k "validate or schedule"`

**Dependencies:** Tasks 1-2

**Files likely touched:**
- `fabric_rlm\experimental\analysis_dag.py`
- `tests\test_analysis_dag.py`

**Estimated scope:** Small

## Checkpoint: Contract Foundation

- All analysis-DAG unit tests pass.
- Existing full suite passes.
- Canonical serialization, seeds, and schedules reproduce in a fresh process.
- Review contracts before operator implementation.

## Phase 2: Evidence and Exact Decomposition

### Task 4: Build the append-only evidence registry

**Description:** Record node lifecycle, structured failures, supersession, and
finding-to-evidence references without mutating completed evidence.

**Acceptance criteria:**
- Invalid lifecycle transitions and mutation of terminal evidence are rejected.
- Failed diagnostics cannot be referenced as action-ready evidence.
- Registry export has a deterministic fingerprint.

**Verification:**
- `python -m pytest tests\test_analysis_evidence.py -q`

**Dependencies:** Tasks 1-3

**Files likely touched:**
- `fabric_rlm\experimental\analysis_evidence.py`
- `tests\test_analysis_evidence.py`

**Estimated scope:** Small

### Task 5: Implement exact additive and rate decomposition

**Description:** Add deterministic sum, rate, and segment-contribution
decomposition with explicit reconciliation residuals.

**Acceptance criteria:**
- Components reconcile to observed KPI change within declared tolerance.
- Results are row-order independent and preserve expected scale/sign behavior.
- Zero denominators, missing segments, and invalid grain fail explicitly.

**Verification:**
- `python -m pytest tests\test_analysis_decomposition.py -q`

**Dependencies:** Tasks 1-4

**Files likely touched:**
- `fabric_rlm\experimental\analysis_operators.py`
- `tests\test_analysis_decomposition.py`

**Estimated scope:** Medium

### Task 6: Implement volume/rate/mix decomposition

**Description:** Add a mathematically explicit decomposition with method
identity, symmetric ordering treatment, and exact reconciliation.

**Acceptance criteria:**
- Known synthetic cases recover expected volume, rate, and mix effects.
- Swapping comparison periods has defined, tested sign behavior.
- Sparse and newly appearing/disappearing segments are handled explicitly.

**Verification:**
- `python -m pytest tests\test_analysis_decomposition.py -q -k "volume or mix"`

**Dependencies:** Task 5

**Files likely touched:**
- `fabric_rlm\experimental\analysis_operators.py`
- `tests\test_analysis_decomposition.py`

**Estimated scope:** Small

## Checkpoint: Exact Operator Slice

- Exact decomposition tests and metamorphic properties pass.
- Full suite and package build pass.
- Evidence records contain source, seed, method, diagnostics, and fingerprints.
- Review numerical definitions before benchmark expansion.

## Phase 3: Reproducible Benchmark Harness

### Task 7: Create seeded synthetic dataset manifests

**Description:** Generate the six approved local dataset families with known
truth, stable manifests, source hashes, and documented challenge variants.

**Acceptance criteria:**
- Same seed and generator version produce byte-identical manifests and data.
- Truth records remain separate from operator inputs.
- Dataset variants cover leakage, missingness, sign reversal, shift, and noise.

**Verification:**
- `python -m pytest tests\test_analysis_benchmarks.py -q -k "dataset or manifest"`

**Dependencies:** Task 2

**Files likely touched:**
- `fabric_rlm\experimental\analysis_benchmarks.py`
- `tests\test_analysis_benchmarks.py`

**Estimated scope:** Medium

### Task 8: Add leakage-safe split and resampling plans

**Description:** Implement persisted random, stratified, grouped, temporal, and
nested split plans with validation against entity and future leakage.

**Acceptance criteria:**
- Split assignments are deterministic and fingerprinted.
- Entity, temporal, duplicate-row, and target-derived leakage are rejected.
- Preprocessing-fit boundaries are represented in the evaluation contract.

**Verification:**
- `python -m pytest tests\test_analysis_validation.py -q`

**Dependencies:** Tasks 1-2, 7

**Files likely touched:**
- `fabric_rlm\experimental\analysis_validation.py`
- `tests\test_analysis_validation.py`

**Estimated scope:** Medium

### Task 9: Score per-case and aggregate benchmark metrics

**Description:** Record task-appropriate metrics, uncertainty, invariants,
runtime, failures, and slice-level regressions without allowing aggregate
scores to hide safety failures.

**Acceptance criteria:**
- Exact decomposition reports residual and attribution error.
- Metric aggregation preserves every failed invariant and dataset slice.
- Repeated runs with the same manifest produce equivalent results.

**Verification:**
- `python -m pytest tests\test_analysis_benchmarks.py -q -k "metric or score"`

**Dependencies:** Tasks 5-8

**Files likely touched:**
- `fabric_rlm\experimental\analysis_benchmarks.py`
- `tests\test_analysis_benchmarks.py`

**Estimated scope:** Medium

### Task 10: Add explicit dataset preparation and benchmark CLI

**Description:** Add commands for pinned public-dataset preparation and local
benchmark execution without network access during tests.

**Acceptance criteria:**
- Downloads require pinned URL, version, license, and SHA-256.
- Hash or license mismatch fails before extraction or use.
- CLI accepts root seed, datasets, tasks, repetitions, and output directory.

**Verification:**
- `python -m pytest tests\test_analysis_benchmark_cli.py -q`

**Dependencies:** Tasks 7-9

**Files likely touched:**
- `fabric_rlm\experimental\analysis_cli.py`
- `tests\test_analysis_benchmark_cli.py`
- `pyproject.toml`

**Estimated scope:** Medium

## Checkpoint: Local Transfer Gate

- Six synthetic/data families pass declared per-task criteria.
- Frozen Olist and CRM runs complete with preserved privacy and lineage.
- Same-seed reruns reproduce manifests, split assignments, DAGs, and results.
- Different-seed repetitions report stability distributions.
- Full tests, build, metadata checks, and five-axis review pass.

## Phase 4: Planner and Critic Integration

### Task 11: Validate model-planned DAG payloads

**Description:** Convert model output into the typed DAG only after strict schema,
authorization, budget, and operator-registry validation.

**Acceptance criteria:**
- Arbitrary Python, imports, unknown operators, and widened sources are rejected.
- User focus changes prioritization but cannot disable required diagnostics.
- Invalid plans return bounded repair feedback.

**Verification:**
- `python -m pytest tests\test_analysis_planner.py -q`

**Dependencies:** Tasks 1-10

**Files likely touched:**
- `fabric_rlm\experimental\analysis_planner.py`
- `tests\test_analysis_planner.py`
- `fabric_rlm\skills\deep_insight_discovery.md`

**Estimated scope:** Medium

### Task 12: Schedule critic-driven closure nodes

**Description:** Let the critic request bounded, typed follow-up nodes linked to
specific unresolved evidence gaps.

**Acceptance criteria:**
- Closure work is deduplicated, budgeted, acyclic, and source-authorized.
- Failed or unresolved closure keeps findings investigate-first.
- Completed closure preserves prior evidence and records supersession.

**Verification:**
- `python -m pytest tests\test_analysis_critic_closure.py -q`

**Dependencies:** Tasks 4, 11

**Files likely touched:**
- `fabric_rlm\experimental\analysis_planner.py`
- `tests\test_analysis_critic_closure.py`
- `fabric_rlm\skills\deep_insight_critic.md`

**Estimated scope:** Medium

## Checkpoint: First Milestone Complete

- Approved success criteria in the specification are satisfied.
- Experimental API and benchmark CLI are documented.
- Olist, CRM, and all six local dataset families pass the transfer gate.
- Real Fabric validation plan is reviewed before execution.

## Later Operator Increments

Each family receives its own specification amendment, tests, benchmark cases,
and transfer threshold before implementation:

1. trends, seasonality, and change points;
2. cohorts, retention, transitions, and funnels;
3. interactions, multivariate models, and driver ranking;
4. clustering and anomaly detection;
5. causal-design eligibility and separately approved estimators.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Benchmark overfits to generated data | High | Use six families, hidden truth variants, Olist/CRM transfer, and untouched holdouts |
| Leakage inflates quality | High | Persisted group/time-aware splits and fold-local preprocessing |
| Reproducibility differs by platform | High | Stable seed derivation, version capture, deterministic algorithms, declared tolerances |
| Planner bypasses safety controls | High | Typed operator registry and host-side validation before execution |
| Metrics reward shallow findings | Medium | Task-specific metric suites plus hard invariants and critic review |
| Dependency growth harms Fabric portability | Medium | No new dependency without approval, wheel/license/runtime checks |
| Multiple testing creates false discoveries | Medium | Predeclared hypotheses where possible and corrected exploratory inference |

## Dependency Order

```text
Contracts
  -> seeds/fingerprints
  -> DAG validation/scheduling
  -> evidence registry
  -> exact decomposition
  -> synthetic datasets
  -> split/validation plans
  -> benchmark metrics and CLI
  -> planner integration
  -> critic closure
```
