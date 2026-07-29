# Changelog

## Unreleased

### Fixed

- **Token totals no longer omit LM calls that produced no runnable code.** When a
  response arrives truncated mid-fence, or contains prose instead of a `python`
  block, the runtime retries it rather than executing it. Those attempts never
  became a `TurnRecord`, and since token totals were summed from the trajectory's
  turns, the provider's charge for them vanished from `total_prompt_tokens` and
  `total_completion_tokens`. Any run that hit either guard under-reported its
  spend; a run where *every* response hit one reported `n_turns=0` and no tokens
  at all while having taken a minute and cost real money. Those calls are now
  carried alongside the trajectory and included in the totals. The turn-exhaustion
  warning also reports attempts rather than recorded turns, so it no longer says
  "ran out of turns after 0 of 16".

  No behaviour change: the trajectory itself is unchanged, so the stuck-loop
  circuit breaker sees exactly what it saw before, and nothing about what the
  model is shown or generates is affected. Only the reported numbers move, and
  they move toward what the provider actually billed.

## 0.3.0 — 2026-07-28 — reasoning effort pinned, sharper document guidance

### Changed

- **Reasoning models now default to `reasoning_effort="medium"`.** Previously
  the effort was left unset, which meant the provider decided. Providers
  disagree: the same gpt-5 family model reached through one route arrived with
  reasoning on and through another with it off, using identical library code.
  A reasoning model that is not reasoning does not fail loudly, it fails
  plausibly. On a two-source document task, gpt-5.1 with the provider default
  scored 4.0 of 12 document checks and fabricated four correctly shaped rows of
  economies and dates that appear nowhere in the source; with the effort pinned
  it scored 12.0 of 12 across three consecutive runs. Pinning the value makes a
  run reproducible across routes instead of inheriting endpoint behaviour.

  This affects anyone on the gpt-5 or o1/o3/o4 family who did not set
  `reasoning_effort` explicitly: expect different cost, latency, and quality
  after upgrading. Chat models (`gpt-4o`, `gpt-5.1-chat`) and non-OpenAI
  backends are unchanged. Pass `reasoning_effort` to `FabricLM`, `OpenAILM`, or
  a dict LM spec to override, including back to a provider default.
- **`pdf_document_analysis` now teaches locate-then-read.** The skill covered
  page rendering and vision, but said nothing about pulling one figure out of a
  long report, and models reliably chose the worst method: flatten every page
  into one string and regex for a number near a label. Section labels and stock
  phrases repeat across a document, so the first match usually comes from the
  wrong section, and the answer that came back was confident and wrong. The
  skill now says to locate the passage, print it, read it, and keep the source
  sentence next to the value, and it documents an x-coordinate fallback for the
  borderless tables `find_tables()` cannot parse. Measured on a two-source
  extraction task with gpt-5-mini, three runs per variant: 3.0 of 12
  document-dependent checks before, 8.3 after.
- The free-tier behaviour gate targets `nvidia/nemotron-3-ultra-550b-a55b:free`.
  OpenRouter retired `openai/gpt-oss-120b` from its free tier, so the old slug
  404s on every question and the gate failed for anyone running the suite. The
  new model calibrates at 5/5 on all five questions. Override with
  `BEHAVIOR_SECONDARY_FREE_MODEL`; the old baseline is retained.
- The skill-authoring documents moved out of the installed package. `SkillLoader`
  listed `PLAYBOOK_CONTRACT` and `SKILL_TEMPLATE` as if they were loadable
  skills, so a model calling `list_skills()` saw two entries that are
  documentation about writing skills. They now live at
  `docs/authoring-skills.md` and `docs/skill-template.md`, and `list_skills()`
  returns the seven real skills.

## 0.2.9 — 2026-07-25 — installable on Windows and macOS

### Fixed

- **`pip install fabric-rlm` failed on Windows and macOS.** litellm, which
  arrives through dspy, stopped publishing Windows and macOS wheels at 1.92.0,
  so pip fell back to compiling its sdist and demanded a Rust toolchain. The
  dependency is now pinned to `litellm>=1.64,<1.92`, the last line with a
  universal wheel. Linux and Fabric notebooks were unaffected; every other
  platform could not install the package at all.

## 0.2.8 — 2026-07-23 — lossless SUBMIT payloads and release hardening

### Fixed

- **Lossless final `SUBMIT` payloads.** Final supported strings and collections
  are no longer truncated by the namespace snapshot limits. Iterative state
  remains bounded, while final payloads have a configurable 64 MiB default
  byte cap and fail explicitly when exceeded.
- **`dspy` dependency range.** Pinned to `dspy>=3.2.1,<3.4`. The subprocess
  interpreter mirrors dspy's kwargs-only tool dispatch (3.2.0) and JSON-RPC
  `CodeInterpreter` messaging (3.1.3), so the previous `>=3.1.2` floor sat below
  the protocol the runtime actually tracks. The ceiling guards against churn in
  dspy's experimental `RLM` / `CodeInterpreter` internals.
- **Example notebooks install from PyPI.** Every notebook now installs
  `fabric-rlm` from PyPI (tutorials unpinned, benchmark notebooks pinned to the
  release) instead of a TestPyPI pre-release index.

### Added

- Regression coverage and documentation for 10,000-character strings,
  500-row outputs, UTF-8 byte boundaries, invalid limits, bounded snapshot
  compatibility, and both interpreter surfaces.
- **New example notebooks** — a minimal contract-comparison walkthrough and a
  Spark-log root-cause analysis. The existing PDF notebooks were trimmed to
  focused, ready-to-import recipes.

### Changed

- `py.typed` is now explicitly declared as package data so type checkers pick up
  the inline types regardless of the build toolchain.
- Documentation and packaged skill playbooks were cleaned up for the public
  release (corrected built-in names, install commands, and cross-references),
  and the `excel_modify` playbook gained value-storage and
  verify-by-recomputation guidance.

## 0.2.7 — 2026-06-16 — opt-in Excel workbook structure context

### Added

- **Opt-in Excel workbook context.** Added
  `add_excel_workbook_context(...)`, a convenience wrapper that prepends a
  read-only workbook context block to an RLM task. The default mode is
  `mode="structure"`: sheet names, target ranges, dimensions, merged/formula
  counts, and headers only — no sample row values. `mode="full"` remains
  available when callers explicitly want compact data-only samples.

### Fixed

- **Workbook context for create-output-sheet tasks.** Structural context now
  reports missing target sheets and summarizes the existing workbook instead of
  failing before the model can create the requested sheet. Post-submit artifact
  validation remains strict.

## 0.2.6 — 2026-06-15 — Excel artifact validation and public workbook skill

### Added

- **Context-aware RLM output validation.** `RLM` now accepts
  `output_validator_context`, allowing validators to inspect runtime artifacts
  such as saved workbooks after `SUBMIT`.
- **Excel artifact helpers.** Added `fabric_rlm.excel_artifacts` with reusable
  target-range parsing, target-cell iteration, and artifact sanity validation
  for workbook-writing tasks.

### Changed

- **`excel_modify` is the public workbook-editing skill.** The improved Excel
  modify guidance, including literal-value guardrails and the large-range /
  sheet-level protocol, is now packaged as `excel_modify` instead of the
  experimental `excel_modify_gpt5` name.

## 0.2.5 — 2026-06-11 — task-constructor alias

### Added

- **`RLM.task(...)` ergonomic alias.** `RLM.task(task, inputs=..., outputs=..., **kwargs)`
  now constructs the same inline-task runtime as `RLM.from_task(...)`, preserving
  subclass dispatch, copied caller inputs/outputs, constructor kwargs, and
  legacy-engine deprecation warning behavior.

## 0.2.4 — 2026-06-10 — golden-trajectory replay + loop-robustness

### Added

- **`ReplayLM` golden-trajectory harness (`fabric_rlm.replay_lm`).** Re-run the
  *real* `RLM.run` loop from a recorded `Trajectory` with **zero API calls and
  zero subprocesses**. A recording already stores both sides of every turn — the
  raw LM response (`TurnRecord.response_text`) and the worker outcome
  (`stdout`/`error`/`submitted`/`state`/`submit_payload`) — which is everything
  needed to drive the loop again. Two in-memory fakes, `ReplayLM` (feeds recorded
  responses) and `ReplayInterpreter` (feeds reconstructed `ExecResult`s), plus a
  one-call `replay_trajectory(rlm, trajectory)` helper, give deterministic
  end-to-end regression coverage of feedback formatting, validation, repair
  routing, and stop conditions. Divergence is the signal: if a change makes the
  loop request more turns, execute different code, or stop earlier than the
  recording, `replay_trajectory` raises `DivergenceError`. New public exports:
  `ReplayLM`, `ReplayInterpreter`, `replay_trajectory`, `DivergenceError`.
  Frozen example recordings live in `examples/trajectories/`; see
  `examples/replay_golden_trajectories.py` for the how/why and the Lakehouse
  save/load path. Targets the default `engine="v6-custom"` loop; verifier-free
  recordings are the supported golden path.

### Changed

- **Robust code extraction.** A response with no `python` code fence is no longer
  shipped to the worker as bare prose (which burned a turn on a `SyntaxError`);
  the loop now short-circuits with a clean "resend one complete block" signal,
  the sibling of the existing truncated-fence guard. Extraction also now selects
  the **last** complete fenced block rather than the first, matching how models
  revise — a sketch in block 1, the corrected code in block 2.
- **Repair-turn diversity nudge (default `escalating`).** Validation / verifier
  repair feedback can append a line nudging the model to recompute a repeatedly
  failing field via a different method and cross-check before re-submitting —
  operationalizing the REFLECT principle *within* an attempt to break stuck
  loops. Modes via `FABRIC_RLM_REPAIR_NUDGE`: `off` / `static` / `escalating`
  (default; the line appears only after a field has failed at least twice).

### Fixed

- **PDF-skill parallel gather hardening.** The `pdf_document_analysis` skill's
  map-reduce snippet now bounds concurrency and uses `return_exceptions=True` so
  one bad chunk can't sink the whole gather, and documents the
  split → `predict()` per chunk → gather → synthesize idiom.

### Other

- **Opt-in failure-time truncation hint** (`FABRIC_RLM_TRUNCATION_HINT`, default
  **off**). When a turn's stdout overflows the feedback budget, an opt-in hint
  tells the model it is not seeing all output and should aggregate in Python or
  chunk + `predict()`. Shipped default-off as a tested safety net (A/B showed the
  dominant "sample-and-guess" failure produces small stdout that never triggers
  it, so no benefit is claimed by default).

## 0.2.3 — 2026-06-10 — tail-preserving feedback truncation + docs

### Changed

- **Tail-preserving feedback truncation.** When a turn's stdout, stderr, or
  error traceback overflows the feedback budget, the runtime now keeps both the
  **head and the tail** (with a `... (N chars omitted) ...` marker) instead of
  the head only. Python output is bottom-loaded — the final `print`, the last
  progress line, and a traceback's terminal `SomeError: ...` line all live at
  the end — so head-only truncation routinely dropped exactly the value the
  model needed. Same token budget, strictly more useful signal. Tail ratios are
  tunable via `FABRIC_RLM_STDOUT_TAIL_RATIO` (default 0.4),
  `FABRIC_RLM_STDERR_TAIL_RATIO` (0.5), and `FABRIC_RLM_ERROR_TAIL_RATIO` (0.7);
  set any to `0` to restore the previous head-only behavior. Error/traceback
  feedback now honors a dedicated `ERROR_FEEDBACK_LIMIT` instead of an inline
  2000-char head cut.

  A/B validation on overflowing-output tasks (OpenRouter, identical code, env
  flag flipped) showed combined task success rising from **25% → 87.5%** across
  `gpt-5.1` and `gpt-4.1`, with a clean causal chain: the answer token was
  visible in feedback ⇒ the model passed; truncated away ⇒ it failed. No
  regressions where output fit the budget (e.g. frame-collapsed tracebacks).

### Docs

- **QUICKSTART §3a (Fabric):** added a "Which models can I name?" note pointing
  to the Microsoft Learn Prebuilt AI models list (no hardcoded model names), so
  users pick model strings their workspace actually exposes.

## 0.2.2 — 2026-06-09 — engine consolidation + public-release hardening

### Fixed

- **Skill router: `from_task` blind spot.** Routing scores keywords against
  bound input values; when the user's question lives in `task=` and the
  inputs are just file paths, every run used to elect the same always-on
  bundle. Routing now falls back to the task text **only when the inputs
  carry zero keyword signal**, preserving the original menu-inflation
  protection for benchmark-style signatures. New trajectory metadata:
  `router_used_task_text_fallback`.
- **v7/dspy engine: token accounting.** `RLMResult.total_prompt_tokens` /
  `total_completion_tokens` / `total_cached_tokens` /
  `total_reasoning_tokens` are now populated for `engine="dspy"` runs by
  harvesting the dspy `lm.history` usage entries (previously always `None`,
  including for `engine="auto"` + `tools=` users).
- **Security rejections no longer wipe reported state.** A parent-side
  `SecurityPolicy` rejection fabricates a failed turn without consulting the
  worker; it previously carried `state={}`, erasing `final_state` and the
  turn's state snapshot. Such turns now carry the last real snapshot and a
  new `ExecResult.reached_worker=False` marker.
- **Legacy `Interpreter` stderr drain.** The v6 interpreter now pumps the
  worker's stderr on a background thread (ring-buffered, last 200 lines).
  Previously stderr was only read at exit, so chatty native libraries could
  fill the OS pipe buffer and deadlock the worker into a spurious
  `WorkerTimeout`.
- **CLI:** `--max-turns` / `--timeout` now override the task file even with
  falsy values; added `--version`, `--engine`, `--verbose`; `engine`,
  `verbose`, `enable_router`, `max_active_skills` are honored from the task
  JSON; unknown task-file keys warn instead of being silently dropped.
- README 30-second example used a non-existent `rlm.run(prompt=...)`
  signature; corrected to `RLM.from_task(...)`.
- **`SubprocessPythonInterpreter` startup timeout** raised 15s → 60s
  (override per-instance via `start_timeout=` or globally via
  `FABRIC_RLM_START_TIMEOUT`). Cold CPython spawns on loaded machines/CI
  runners legitimately exceed 15s; genuinely broken installs still fail
  fast because the dead worker closes stdout immediately.
- **Behavior CI gate: credential failures are no longer reported as model
  regressions.** 401/403 and `AuthenticationError`-shaped failures get a
  new `auth` error class and abort the gate immediately with a
  "fix OPENROUTER_API_KEY" message instead of failing every qid.

### Packaging / docs

- Version is now single-sourced from `fabric_rlm.__version__` (pyproject
  reads it via `[tool.setuptools.dynamic]`); README/QUICKSTART no longer
  hardcode wheel versions.
- Added PyPI metadata (`[project.urls]`, classifiers, keywords), and
  `CONTRIBUTING.md` / `SECURITY.md` (threat model + private reporting).
- Example notebooks no longer embed real workspace/lakehouse IDs
  (placeholders instead).
- Removed references to non-shipped design/eval documents from README and
  QUICKSTART; QUICKSTART troubleshooting now documents the real
  `failure_reason` values.
- CI: test matrix expanded to Python 3.10–3.13 on ubuntu + windows; added a
  packaging job (`python -m build`, `twine check`, wheel-content assertions
  for skills markdown and `py.typed`).

### Engine selection (consolidation)

- **`engine="auto"` is the new default** for `RLM(...)`. It picks `"dspy"`
  when a non-empty `tools=[...]` iterable is supplied, otherwise `"default"`.
  Existing code that passes no `tools=` (or an empty `tools=[]`) and didn't
  pass `engine=` keeps identical behavior (resolves to the same canonical
  engine as before).
- **New public aliases**: `engine="default"` (= legacy `"v6-custom"`) and
  `engine="dspy"` (= legacy `"v7-dspy"`).
- **Deprecated**: passing `engine="v6-custom"` or `engine="v7-dspy"`
  directly (or via `RLM.from_task(...)`) now emits a `DeprecationWarning`
  pointing at the user's call site. Behavior is unchanged — both still
  resolve to the same canonical engines. **Removal not before v0.3.**
- Migration: prefer `engine="auto"` (recommended), or explicit
  `engine="default"` / `engine="dspy"`. Adaptive (`engine="adaptive"`)
  is unaffected and remains experimental.
- Internal: `_normalize_engine_name` is pure (no side effects); the
  deprecation warning is emitted at public entry points (`__init__`,
  `from_task`) with correct stacklevel for both call paths. The adaptive
  inner-RLM factory translates canonical inner engines to public aliases
  before constructing inner attempts to avoid library self-warning.

## 0.2.1 — `excel_modify` skill + SpreadsheetBench head-to-head

### New

- **`fabric_rlm/skills/excel_modify.md`** — task-agnostic skill for in-place
  modification of `.xlsx` workbooks via openpyxl. Triggered by keywords
  `xlsx`, `workbook`, `openpyxl`, `sheet`, `cell range`, etc. Bakes in two
  recipes that fixed real benchmark failures:
  1. **Two-load discovery**: load the workbook with `data_only=False` for
     editing and `data_only=True` for reading source values, so cells whose
     source is a formula return numbers rather than the literal `'=D3+F3'`
     string.
  2. **Mandatory verify-by-reload**: after `wb.save()`, reload with
     `data_only=True` and assert no cell in the target range is `None` or
     starts with `=`. Catches the formula-instead-of-value failure class.

### Bench

- **SpreadsheetBench Verified-400 head-to-head** (50Q stratified subset),
  reproducible end-to-end on Fabric:
  - Strategy A (gpt-5 single-shot, dspy.Predict + subprocess exec):
    23/50 = 46.0%, $2.21
  - Strategy F (gpt-4.1-mini + RLM + Python interpreter + `excel_modify`):
    21/50 = 42.0%, $0.51 (4.3× cheaper, 2.3× faster wall-clock)
  - Union pass rate: 29/50 = 58.0%
  - Reproduce with the benchmark notebooks in `examples/notebooks/`
    (`spreadsheetbench_400_openrouter_minimax_mlflow.ipynb` and
    `ssb400_minimax_m3_fabric_repro.ipynb`).

## 0.1.11 — PLAN / VERIFY / REFLECT (PVR) contract

**Bug fix (dev6):** `Trajectory.__bool__` now explicitly returns `True`. Previously a `Trajectory` with zero turns evaluated as falsy because `__len__` was defined and Python falls back to it for truthiness, causing downstream `if traj: ...` guards (in benchmarks and result-collection helpers) to silently discard the trajectory's metadata — including the entire `adaptive` payload. Found while diagnosing a 5-way comparison where `EffortLadderPolicy` appeared to record 0 attempts on every question.

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
| easy-math, easy-csv | pass | pass | 1→1 | small overhead, no regression |
| Backprop_hard (solvable, multi-step) | fail ladder exhausted | pass rung 3 | 7→3 | 1.28M→413K (-68%) |
| VLIW_hard (capability ceiling) | fail | fail | 6→6 | 294K→242K (-18%) |

**OOD ablation** (structured extraction outside training distribution):

| case | OFF pass | ON pass | OFF→ON attempts | OFF→ON tokens | OFF→ON elapsed |
|---|---|---|---|---|---|
| rfp-extract (4 fields from RFP PDF text 100KB) | pass | pass | 2→1 | 18K→4.9K (-74%) | 60.6s→7.9s |
| spark-extract (5 fields from Spark log JSON 200KB) | pass | pass | 1→**3** | 6K→40K (+528%) | 10.1s→176s |

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

### 0.1.11.dev4 — Trajectory capture + diagnostic finding on PVR mechanism

Added opt-in turn capture (`FABRIC_RLM_CAPTURE_TURNS=1`) on
`AttemptRecord.to_summary()` so notebooks can persist per-turn
`response_text` / `code` / `stdout` for offline analysis. The OOD
ablation notebook now writes per-condition trajectory JSONL files into
the run directory.

**Diagnostic finding (important; informs how PVR is described):**
captured trajectories show the model emits **zero** `## PLAN` /
`## VERIFY` markers across every turn of every attempt at
`reasoning_effort='minimal'`, even when the skill prompt is delivered
verbatim. Strengthening the skill rules with explicit "MUST",
"contract violation" language, and worked code examples did **not**
make the model comply (verified on `pvr_ood_ablation` run
`20260502-150436-6b9f67`).

The measurable wins from PVR (Backprop -50% attempts / -16% tokens; RFP
-74% tokens) therefore come from the **inter-attempt REFLECT injection**
in `AdaptiveRunner._with_feedback` (the `[ADAPTIVE: prior attempt
rejected]` block fed into retries) and the runtime's post-SUBMIT
reflection turn — *not* from PLAN/VERIFY scaffolding in the skill
prompt. The PLAN/VERIFY rules in `core.md` are kept as-is for higher
reasoning-effort runs (where the model may still honor them) but the
operative contract today is REFLECT. A future change should either (a)
add a runtime check that re-prompts when `## PLAN` / `## VERIFY`
markers are missing, or (b) rename the documentation to "REFLECT
contract" and drop the PLAN/VERIFY claims.

Also fixed a `TypeError: unhashable type: 'slice'` crash in the OOD
notebook's `attempts_summary` cell when an attempt's `answer` payload
is a dict instead of a string.

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
