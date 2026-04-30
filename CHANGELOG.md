# Changelog

## 0.1.9 — Slim core release

**Repository slimming.** This release ships `fabric-rlm-core`, a clean
distribution containing only the production runtime and the proven skills:

- **Kept:** runtime, subprocess interpreter (with the v0.1.8 asyncio fix),
  LM backends (OpenAI / Anthropic / FabricLM), skill loader & router,
  trajectory + replay, validators, and the skills `core`,
  `validation`, `error_handling`, `data_exploration`,
  `pdf_document_analysis`.
- **Removed:** `fabric_rlm.adaptive` (deprecation shim),
  `fabric_rlm.experimental.*` (AdaptiveOrchestrator),
  `fabric_rlm.skill_distiller`, the `benchmarks/` package,
  longcot signatures/schemas/skills, and all `_*` repo-level scratch.
- **API:** no breaking change for code that uses only the documented public
  API. `from fabric_rlm import AdaptiveOrchestrator` no longer works (it
  has been deprecated since 0.1.7 and only re-exported via a shim).
- **Docs:** new `README.md`, scrubbed `QUICKSTART.md` (no §9b Adaptive
  escalation, no longcot examples), `LICENSE` (MIT), `.gitignore`,
  `.gitattributes`.
- **Tests:** dropped longcot/adaptive/v6-skill-verifier suites; the kept
  ~33 tests cover runtime, interpreter, validators, serializers, replay,
  LM, skill loader/router, and the playbook contract.

## 0.1.8 — Asyncio fix in the subprocess worker

Fixed `_worker.py` calling `asyncio.run()` from inside an already-running
event loop in async-host environments (Fabric notebooks, Jupyter). Worker
now detects an existing loop and awaits in-place. Validated on the cc
(93%) and inv (97% RLM, 100% direct) Fabric runs.

## 0.1.7 — Universal validator + self-report contract

(Removed in 0.1.9 along with the rest of `experimental.adaptive`.)

## 0.1.6 — `data_exploration` skill hardening

- Skill cookbook annexed with chained-bracket gotcha, STRING-EQUALITY
  gotcha, Step 7 zero-result sanity check, universal placeholders.

## 0.1.5 — `data_exploration` skill: parsing fixes

Bug fixes around heterogeneous JSONL ingestion and downstream chained
bracket access.

## 0.1.4 — `data_exploration` skill: first iteration

Initial DuckDB + ripgrep + Python-streaming skill for files larger than
the LM context window.

## 0.1.3 — Reasoning-model handling

`FabricLM` / `OpenAILM` auto-handle reasoning models (e.g. `gpt-5`,
`o1`, `o3`).

## 0.1.2 — Skill text mentions pre-installed deps

`data_exploration` skill text now explicitly tells the LM that `duckdb`
and `polars` are pre-installed in the Fabric Python runtime.

## 0.1.1 — Large-file / log analysis (opt-in)

Added the opt-in `data_exploration` skill family for analyzing files
larger than the LM context window.

## 0.1.0 — Initial release

Public API for fabric-rlm: `RLM`, `RLMResult`, `FabricLM`, skills,
trajectory + replay, validators.
