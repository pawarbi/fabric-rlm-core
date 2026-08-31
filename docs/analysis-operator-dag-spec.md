# Spec: Reproducible Deep-Insight Analysis DAG

## Objective

Build a source-agnostic analysis system in which an RLM plans a typed,
bounded DAG of analytical operators, deterministic host code executes the
operators, an evidence registry records every result, and a critic schedules
bounded follow-up work.

The system is intended for analysts who need decision-grade findings rather
than plausible narratives. It must support a user-supplied analysis brief such
as "focus on drivers", "find cohort changes", or "investigate anomalies"
without allowing that focus to bypass validation or evidence requirements.

The first delivery will establish the contracts, reproducibility controls,
local benchmark harness, and exact KPI decomposition. Trend, cohort, funnel,
interaction, multivariate, anomaly, clustering, and causal operators will be
added in later increments only after the earlier layer passes transfer tests.

## Assumptions

1. Planning may use an LM, but authoritative calculations run in deterministic,
   whitelisted host-side operators.
2. The base package remains lean. No new runtime dependency is added without
   explicit approval and a compatibility review.
3. Initial benchmarks use deterministic synthetic generators and redistributable
   local fixtures or user-supplied manifest bundles; downloaded datasets are
   pinned by URL, version, license, and SHA-256 digest.
4. Seeds control dataset generation, sampling, resampling, fold assignment,
   estimator initialization, and any worker-visible random source.
5. Causal estimation is out of the first implementation increment. It requires
   a separate design record and eligibility gate.

## Analysis Contract

### Analysis brief

The user can provide:

- business objective and decision to support;
- prioritized focus areas;
- target metrics and candidate outcomes;
- time window, comparison basis, and population;
- exclusions and privacy constraints;
- computational budget;
- acceptable descriptive, predictive, or causal interpretation level.

Missing fields receive explicit defaults recorded in the run manifest.

### Typed DAG

Each node declares:

- stable node ID and operator version;
- operator family and method;
- dependencies and execution mode (`sequential` or `parallel`);
- authorized source identities, columns, grain, filters, and joins;
- hypothesis or analytical question;
- validated parameters and random seed;
- resource budget and stopping conditions;
- required diagnostics, invariants, and output schema.

The planner may select and connect only registered operators. Cycles, unknown
operators, unauthorized sources, invalid parameters, and work beyond the run
budget are rejected before execution.

### Operator result

Every completed node emits:

- source and data fingerprints;
- code, operator, library, and environment versions;
- effective seed and split/fold assignments where applicable;
- sample size, missingness, exclusions, and coverage;
- point estimates and uncertainty;
- validation metrics and diagnostic results;
- reconciliation checks and invariants;
- limitations and permitted interpretation;
- structured failure state when execution or validation is not adequate;
- deterministic output fingerprint.

### Evidence registry

The registry is append-only within a run. It records planned, running,
completed, failed, superseded, and rejected nodes. Findings reference evidence
IDs rather than copying unverified values. A critic can request bounded closure
nodes, but cannot mutate prior evidence or promote a failed diagnostic.

## Reproducibility

A run has one required root seed. Child seeds are deterministically derived
from the root seed, dataset ID, operator ID, and repetition/fold ID. Operators
must not use ambient random state.

The run manifest records:

- root and derived seeds;
- dataset manifest, generator version, source hashes, and license;
- train/validation/test or resampling assignments;
- package versions and relevant runtime configuration;
- operator parameters and DAG fingerprint;
- input, evidence, and final-output fingerprints.

Re-running the same manifest in the same supported environment must reproduce
the DAG, split assignments, deterministic results, and fingerprints. Algorithms
with unavoidable platform variance must declare tolerances and the source of
non-determinism.

## Local Benchmark Matrix

The benchmark suite must cover structurally different datasets and questions,
not only the Olist and CRM examples:

| Dataset family | Required analytical tasks | Failure modes exercised |
| --- | --- | --- |
| Additive KPI synthetic data with known truth | exact sum, rate, volume/rate/mix, segment contribution | reconciliation errors, sign reversal, Simpson's paradox |
| Seasonal and changing time series | trend, seasonality, robust slope, change points, anomalies | leakage, edge effects, false alarms, missing periods |
| Customer/event panel | cohorts, retention, transitions, funnels | censoring, unequal exposure, duplicate events, cohort leakage |
| Correlated tabular data | regression, interactions, driver ranking | multicollinearity, missingness, nonlinearity, confounding proxies |
| Clustered and contaminated data | clustering and batch anomalies | unstable clusters, scale sensitivity, contamination sensitivity |
| Distribution-shift and missingness stress data | robustness, drift, subgroup metrics | covariate shift, missing-not-at-random sensitivity, small slices |
| Frozen real transfer bundles: Olist and CRM | end-to-end planning, execution, criticism | schema variation, joins, privacy, unsupported methods |

Synthetic cases must include known ground truth so detection power, false
positive rates, estimation error, and decomposition reconciliation can be
measured directly. Real-data cases evaluate robustness and usefulness without
pretending unknown truth is known.

## Validation Strategy

Validation must match the data-generating structure:

- random or stratified repeated holdout for independent observations;
- grouped cross-validation when entities repeat;
- rolling or expanding-window backtests for temporal data;
- nested cross-validation for model or hyperparameter selection;
- bootstrap or permutation inference when assumptions require it;
- untouched final holdout for benchmark-level claims.

Preprocessing, imputation, scaling, feature selection, and tuning are fitted
inside each training fold. Split assignments are persisted. The harness rejects
entity leakage, future leakage, duplicated rows across splits, target-derived
features, and evaluation on training data.

No single metric is sufficient. Depending on the task, reports include:

- estimation: bias, MAE/RMSE, interval coverage, interval width;
- classification/ranking: ROC AUC, PR AUC, log loss, Brier score, calibration;
- forecasting/change detection: MAE/MASE, precision/recall, detection delay,
  false alarms per period;
- clustering: stability across seeds/resamples, silhouette, adjusted Rand index
  when truth exists, cluster-size and interpretability checks;
- anomaly detection: precision/recall at budget, average precision, false alarm
  rate, detection delay;
- decomposition: exact reconciliation residual, attribution sign and magnitude
  error;
- all applicable tests: uncertainty, effect size, power/sensitivity, and
  multiple-testing correction.

Benchmark results are reported per dataset/task and in aggregate. Aggregate
scores cannot hide a failed safety invariant or a severe dataset-specific
regression.

## Commands

Initial development commands:

```text
Unit and integration tests:
python -m pytest tests\test_analysis_dag.py tests\test_analysis_benchmarks.py -q

Full suite:
python -m pytest -q

Package build:
python -m build

Package metadata:
python -m twine check dist\*
```

The benchmark CLI and exact command will be added with the benchmark harness.
It must accept a root seed, dataset selector, task selector, repetition count,
and output directory.

## Project Structure

```text
fabric_rlm\analysis\          typed contracts, planner validation, registry
fabric_rlm\analysis\operators deterministic operator implementations
fabric_rlm\analysis\benchmarks dataset generators and benchmark evaluation
tests\test_analysis_dag.py    contract, validation, scheduling, registry tests
tests\test_analysis_benchmarks.py reproducibility and metric tests
examples\                     later end-to-end examples, not core test logic
docs\analysis-operator-dag-spec.md living specification
```

Benchmark outputs and downloaded data remain outside version control. An
explicit preparation command may download a public dataset only when its URL,
version, license, and SHA-256 digest are pinned and verified. Small,
redistributable fixtures may be committed when their license and provenance
are documented.

## Code Style

Use immutable typed records and explicit validation. Operators return structured
results rather than printing or returning ad hoc dictionaries.

```python
@dataclass(frozen=True)
class OperatorRequest:
    node_id: str
    operator: str
    source_ids: tuple[str, ...]
    parameters: Mapping[str, object]
    seed: int
```

Public contracts use stable names and JSON-compatible values. Validation errors
identify the exact field and unsafe value. No broad exception handling or
success-shaped fallback is allowed.

## Testing Strategy

Development follows strict red-green-refactor TDD:

1. unit tests for contracts, seed derivation, DAG validation, metrics, and each
   deterministic operator;
2. property and metamorphic tests for invariants such as row-order independence,
   decomposition reconciliation, scale/translation behavior, and deterministic
   reruns;
3. integration tests for planner output to execution to evidence registration;
4. local benchmark tests across the matrix above;
5. frozen transfer regressions on Olist, CRM, and additional approved bundles;
6. real Fabric validation only after local transfer gates pass.

Benchmark comparisons use predeclared tolerances and baselines. A change fails
when it breaks an invariant, materially worsens a safety metric, or regresses
beyond the declared tolerance on a benchmark slice.

## Boundaries

- Always:
  - validate all planner output before execution;
  - use deterministic seed derivation and persist split assignments;
  - keep data preparation inside validation folds;
  - report uncertainty, diagnostics, missingness, and sample size;
  - correct for multiple comparisons where applicable;
  - preserve privacy and exact source lineage;
  - run targeted and full tests before commits.
- Ask first:
  - add or change package dependencies;
  - add a downloaded or third-party dataset to version control;
  - change a public API or packaged skill contract;
  - enable causal estimation or causal language.
- Never:
  - let model-generated code import arbitrary analytical libraries;
  - select a method using the final holdout;
  - use unseeded randomness;
  - claim causality from observational association;
  - silently drop failed diagnostics or unsupported data slices;
  - expose raw sensitive records to the model or benchmark artifacts.

## Success Criteria

The first increment is complete when:

1. typed analysis-brief, DAG, node, result, and evidence-registry contracts exist;
2. invalid, cyclic, unauthorized, over-budget, and unseeded plans are rejected;
3. seed derivation and run fingerprints reproduce exactly;
4. exact KPI decomposition reconciles to the observed change within a declared
   numeric tolerance;
5. deterministic synthetic benchmarks cover additive, rate, and
   volume/rate/mix ground-truth cases;
6. benchmark output records per-case metrics, invariants, versions, and hashes;
7. rerunning the same benchmark manifest produces identical assignments and
   equivalent results;
8. all repository quality gates pass.

Later increments add operator families only after their benchmark acceptance
criteria and interpretation limits are specified.

## Approved Scope Decisions

1. Public datasets may be acquired only through an explicit preparation
   command with pinned source, version, license, and SHA-256 verification.
2. The initial transfer gate requires six synthetic/data families plus the
   frozen Olist and CRM bundles.
3. The initial surface is an experimental Python API plus a benchmark CLI.
