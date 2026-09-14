# Benchmarks

Harnesses that measure fabric-rlm on public data sets, and the notebooks
that reproduce published figures. None of this is needed to use the
package; what a user runs is under `examples/`. Each harness has a test
under `tests/` that exercises it on a small local sample, so a change to
the package that breaks a harness fails the suite.

| File | What it does | Data it needs | Test |
|---|---|---|---|
| `ons_cpi_rlm_benchmark.py` | Runs fabric-rlm against the UK Office for National Statistics Consumer Price Inflation release, one large `mm23.xlsx` workbook, and scores the answers. | `mm23.xlsx` from ONS | `tests/test_ons_cpi_benchmark.py` |
| `olist_deep_insight_benchmark.py` | The local Olist deep-insight transfer benchmark: the RLM analyses caller-provided canonical Olist CSV files and every numeric evidence item is re-executed independently. | the Olist CSV files | `tests/test_olist_deep_insight_benchmark.py` |
| `olist_staged_deep_insight_benchmark.py` | The three-stage form: a measured research ledger first, then the contract scaffold and the insights, then host verification and numeric audit. | the Olist CSV files | `tests/test_olist_staged_deep_insight_benchmark.py` |
| `manifest_staged_deep_insight_benchmark.py` | The staged benchmark driven by a manifest instead of the Olist layout. | a manifest and its sources | `tests/test_manifest_staged_deep_insight_benchmark.py` |
| `olist_deep_insight_critic.py` | Checkpointed, source-agnostic adversarial review of discovery artifacts. | the artifacts of a run | `tests/test_olist_deep_insight_critic.py` |
| `critic_evidence_closure.py` | One source-aware evidence-closure cycle from critic challenges. | the artifacts of a run | `tests/test_critic_evidence_closure.py` |
| `action_readiness_synthesis.py` | Bounded program actions for approved, evidence-closed findings. | the artifacts of a run | used with the staged benchmark |

## Notebooks

- `notebooks/ssb400_minimax_m3_fabric_repro.ipynb` reproduces the
  SpreadsheetBench-400 figure reported in the README. It needs an
  OpenRouter key read from Key Vault or the environment and costs a few
  dollars.
- `notebooks/spreadsheetbench_400_openrouter_minimax_mlflow.ipynb` is the
  same run with MLflow tracking.
- `notebooks/rlm_knowledge_benchmark_matrix.py` (Fabric notebook source
  format) runs seeded, cache-disabled cold-versus-learned trials of the
  knowledge learner across source types.
- `notebooks/rlm_knowledge_value_fabric.ipynb` runs the same governed
  analytical question against a Fabric semantic model twice, cold and then
  with the learned knowledge package, and compares the two runs.

`tests/test_public_notebooks.py` checks these notebooks like every other
notebook in the repository for a key in the source and for the kernel. The
pinned-install check applies to the recipes and the release checks only: a
development notebook here may install the branch under test.
