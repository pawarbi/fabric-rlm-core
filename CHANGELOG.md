# Changelog

## 0.1.11 (unreleased) — PLAN / VERIFY / REFLECT (PVR) contract

The default `core` skill now ships with an explicit **PLAN / VERIFY /
REFLECT** contract, and the adaptive engine injects synthesized REFLECT
context on every failed attempt (not only validator rejections).

- **PLAN** — model decomposes the task before writing worker code.
- **VERIFY** — model self-checks the answer against task constraints
  before calling `SUBMIT(...)`.
- **REFLECT** — when an attempt fails (validator rejection, worker
  error, timeout), the next attempt receives a structured
  `PRIOR_ATTEMPT_FEEDBACK` block containing the failure reason and the
  prior answer to consider.

**Generalization ablation** (4 cases × 2 conditions, fresh bandit state):

| case | OFF pass | ON pass | OFF→ON attempts | OFF→ON tokens |
|---|---|---|---|---|
| easy-math, easy-csv | ✅ | ✅ | 1→1 | small overhead, no regression |
| Backprop_hard (solvable, multi-step) | ❌ ladder exhausted | ✅ rung 3 | 7→3 | 1.28M→413K (-68%) |
| VLIW_hard (capability ceiling) | ❌ | ❌ | 6→6 | 294K→242K (-18%) |

**OOD ablation** (structured extraction outside training distribution):

| case | OFF pass | ON pass | OFF→ON attempts | OFF→ON tokens | OFF→ON elapsed |
|---|---|---|---|---|---|
| rfp-extract (4 fields from RFP PDF text 100KB) | ✅ | ✅ | 2→1 | 18K→4.9K (-74%) | 60.6s→7.9s |
| spark-extract (5 fields from Spark log JSON 200KB) | ✅ | ✅ | 1→**3** | 6K→40K (+528%) | 10.1s→176s |

The Spark-log case revealed a new failure mode: PVR's VERIFY clause can spuriously
self-reject a correct first answer, amplifying retries on tasks the model would otherwise
nail cold. Correctness is preserved (always passes eventually), but the cost can be 10×+.

**Refined heuristic — when to enable PVR**:

| profile | PVR? |
|---|---|
| Easy single-step the model nails cold (Spark log triage, simple lookups) | optional/off — VERIFY can spuriously self-reject |
| Multi-field extraction with strict format (RFP) | **on** — PLAN/VERIFY enforces completeness |
| Multi-step reasoning, derivations, code synthesis | **on** — REFLECT prevents brute-force-and-fail |
| Capability ceiling | optional — fails marginally cheaper but doesn't rescue |

PVR is **on by default**. Disable with `FABRIC_RLM_PVR=0` for token-
sensitive batch workloads on known-trivial tasks.

## 0.1.10 — Experimental `engine="adaptive"`

**Adaptive escalation, opt-in.** When a validator rejects an attempt, an
outer meta-controller climbs a fixed ladder until either the validator
passes or a budget is exhausted:

```
rung 0 → baseline (cheap LM, default turns)
rung 1 → more_turns
rung 2 → more_effort         (medium reasoning_effort)
rung 3 → best_of_N           (parallel rollouts, same cheap LM)
rung 4 → strong_lm           (escalate to e.g. gpt-5, reasoning_effort=high)
```

Two surfaces:

- `RLM(engine="adaptive", adaptive={...})` — thin wrapper, headline ergonomics.
- `from fabric_rlm.experimental import AdaptiveRunner, LadderPolicy, Budget` —
  power user; gives you the per-attempt `AttemptRecord` log.

Notes:

- Inner engine defaults to `v6-custom`; pass `inner_engine="v7-dspy"` to switch.
- Emits a `UserWarning("experimental")` once at construction.
- Per-run summary attached to `result.trajectory.metadata["adaptive"]` with
  `winner_rung`, `attempts: [{rung, ...}]`, `stop_reason`, `elapsed_seconds`.
- Bench harness lives at `bench/adaptive/` (4 modes × 3 buckets, including a
  Spark RCA case). Baseline mechanics verified end-to-end on real LMs:
  `MFMC_hard_1` failed at gpt-4.1-mini, the ladder escalated through
  `[0,2,3,3,3,4,4,4]`, and gpt-5 at rung 4 solved it.

**Legacy / deprecated**: nothing removed; the only API surface added is
`engine="adaptive"` plus the `experimental` submodule. Nothing else is
behaviourally affected.

**Tests**: 53 passing — 32 policy + 8 runner + 7 runtime + 2 eval + the
existing 3 legacy + 1 spot-check.

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
