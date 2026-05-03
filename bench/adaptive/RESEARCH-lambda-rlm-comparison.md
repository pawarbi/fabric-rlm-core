# Research Note: Skills Audit + λ-RLM Comparison

Status: Discovery / planning — no code changes yet.
Outputs three follow-up spec branches: `feat/combinators-skill`, `feat/task-type-classifier`, `feat/decompose-rung`.
Companion to `REPORT-comparison-5way.md` (Addenda 1 & 2).

## 1. Skills audit — would loaded skills have helped the bandit hit > 6/25?

### Finding

**No.** Adding `enable_skill_autoloading=True` (or any explicit `skills=[...]`) to the
5-way comparison would have changed the F (`EffortBanditPolicy`) score by roughly zero
on the LongCoT CS-hard 25-question holdout.

### Evidence

1. **All six comparison strategies (A–F) ran with skills disabled.**
   - `scripts/build_comparison_5way_notebook.py` builds every `RLM(...)` with no
     `skills=` argument. `enable_skill_autoloading` defaults to `False`
     (`fabric_rlm/runtime.py:216`).
   - `bench/adaptive/run_bench.py` (the original 0.1.10 harness) explicitly sets
     `enable_skill_autoloading=False` on every mode (lines 353, 385, 450, 515).
   - This was a deliberate apples-to-apples choice to isolate the adaptive-loop
     comparison from the skill machinery.
   - Stale doc note: `build_comparison_5way_notebook.py:9` claims "C = ... + skills"
     but the C branch only flips a PVR env var. Fix in a follow-up.

2. **The CS-hard prompts forbid the only things skills teach.** First line of every
   MFMC / MCM / Backprop / DistMem / VLIW question (verified from
   `bench/adaptive/longcot_cs_hard_holdout25.jsonl`):
   > "You are not to use tools, write code, ask to use a solver, or ask any
   > clarifying questions. You must solve the puzzle in a single response."

3. **Per-skill applicability on this benchmark:**

   | Skill | Teaches | Applies here? |
   |---|---|---|
   | `core` | Always-on PLAN / VERIFY contract | Marginal — prompt mandates single-response anyway |
   | `error_handling` | try/except patterns when running code | ❌ no code allowed |
   | `validation` | verifier-first patterns when writing code | ❌ no code allowed |
   | `data_exploration` | DuckDB-on-JSONL cookbook | ❌ no datasets, no files; keyword gate doesn't even match |
   | `pdf_document_analysis` | page-rendered PDF playbook | ❌ no PDFs |

4. **`data_exploration` keyword gate** (`fabric_rlm/skills/data_exploration.md`
   front-matter) requires one of `[log, logs, csv, parquet, jsonl, json, dataset,
   rows, lines, file, filesystem]` in the prompt. The CS puzzles contain none of
   these, so even with autoloading enabled, only `core` would have loaded.

### Implication

Skills are a different lever for a different problem. To validate the skill path we
need a benchmark whose tasks actually allow / require code execution and data
access — e.g., the Spark-RCA dataset that already lives in
`bench/adaptive/spark_generate.py`, or a new long-document task pointed at
OneLake files.

The CS-hard ceiling on Backprop / DistMem / VLIW is a **model-capability ceiling**
at `gpt-5` medium-effort, not a skill / context / tool problem. The 5-rung effort
ladder validation (`checkpoint 011-012`) showed `high` and `high+parallel` rungs
move several templates from fail → pass but `Backprop_hard_1` and `VLIW_hard_1`
still fail at the top of the ladder.

## 2. λ-RLM (`nktkt/lambda-rlm`) — architectural comparison

λ-RLM is also called "RLM" but solves a different bottleneck.

| Axis | λ-RLM | fabric_rlm |
|---|---|---|
| Bottleneck targeted | Long-context overflow (10⁶+ tokens) | Reasoning depth + tool use at one prompt |
| LLM role | Leaf worker on bounded chunks | Driver of an open-ended REPL |
| Control flow | Deterministic combinators (SPLIT, MAP, FILTER, REDUCE, CROSS, CONCAT, PEEK) | Model-emitted Python in persistent CPython subprocess |
| Recursion | Explicit Y-combinator | Implicit `await predict(...)` via `sub_lm` |
| Plan | Pre-computed from auto-detected task type | Emergent per turn |
| Failure recovery | None — combinators are total | Validator + adaptive ladder + Thompson-sampling bandit |
| Compute escalation | Static — `k*`, `τ*` set once | Dynamic — rungs of effort / parallel / turns; learned per template |
| Cost bound | Closed-form `T(n) ≤ (nk*/τ*)·C(τ*)` | Empirical, per-rung |
| Best at | "Aggregate / search / classify across a 10M-token doc" | "Solve a hard puzzle in a 1k-token prompt with 16k reasoning tokens" |

### Where they overlap

Both reject "let the LLM freely generate code in a Turing-complete loop" as the
default when reliability matters — but take opposite paths:

- **λ-RLM**: constrain the loop to a small, total combinator algebra.
- **fabric_rlm**: keep the loop open-ended, wrap it with a meta-controller
  (validator + bandit).

They could plausibly **compose**: fabric_rlm's REPL could call λ-RLM combinators
for any sub-problem that needs to chew through long context. Neither replaces
the other.

## 3. Borrowable ideas — three follow-up spikes

In descending order of expected payoff. Each gets its own branch with a SPEC and
gated workflow per `.github/skills/spec-driven-development/SKILL.md` (no code in
SPEC phase).

### 3.1 Combinator library as a skill — **HIGH priority**

Branch: `feat/combinators-skill` · Spec: `bench/adaptive/SPEC-combinators-skill.md`

Ship λ-RLM's 7 primitives as a `combinators` skill, pre-imported in the
fabric_rlm subprocess:

```python
split(text, k=4)                # → [chunk, ...]
peek(text, offset=0, n=500)     # bounded view, cost-tracked
map_combinator(fn, chunks)
filter_combinator(pred, chunks)
reduce_combinator(op, vals)
concat(parts)
cross(a, b)
```

Removes the "coding tax" the model pays today re-deriving chunking each session.
Pairs naturally with `data_exploration` (DuckDB) and the existing PLAN/VERIFY
contract.

**Caveat**: irrelevant to the CS-puzzle benchmark; needs a long-context /
data-analysis benchmark to validate (Spark-RCA + OneLake jsonl).

### 3.2 Task-type classifier seeds the bandit prior — MEDIUM priority

Branch: `feat/task-type-classifier` · Spec: `bench/adaptive/SPEC-task-type-classifier.md`

Borrow λ-RLM's one-call up-front task classifier and use the predicted class
(`SEARCH`, `AGGREGATE`, `PAIRWISE`, `MULTI_HOP`, `CS_PUZZLE`, ...) to seed the
`BanditState` Beta posteriors with informative priors:

- Detected `LOOKUP` → favour rung 0 (minimal effort).
- Detected `MULTI_HOP` reasoning → start at rung 2+ with 2× max_turns.
- Detected `SEARCH` over long context → favour SPLIT+FILTER plan, skip parallel.

Today every (model, template) combination warms up cold — wasting the first ~3
attempts on rung 0 even when the template is obviously hard. A cheap classifier
pays for itself if it shaves ≥1 escalation per question on average.

**Caveat**: adds an LLM call per question (~$0.001). Need to verify the prior
seeding doesn't degrade Thompson-sampling convergence on templates the
classifier mis-labels.

### 3.3 Decompose-then-synthesize rung — SPECULATIVE

Branch: `feat/decompose-rung` · Spec: `bench/adaptive/SPEC-decompose-rung.md`

Borrow λ-RLM's `MULTI_HOP` plan
(`SPLIT_δ → MAP(PEEK) → FILTER → MAP(M) → M`) and turn it into rung 5 of the
effort ladder: model is forced to first emit sub-problems, solve each in
isolation (with `sub_lm`), then synthesize. This is structurally what humans do
on Backprop_hard_1 / VLIW_hard_1 — the templates the current bandit can't crack
at any pure-effort rung.

**Caveat**: real research bet. If the model can't emit a useful decomposition,
the rung is just expensive. Validate on the prior-fail set first
(`bench/adaptive/longcot_cs_hard_pilot20.jsonl` subset where rung 4 failed).

## 4. Not borrowing

- **Replacing the REPL with combinators** — kills the open-ended agentic
  capability that's fabric_rlm's reason to exist.
- **Y-combinator formalism** — Python's `await predict(...)` already gives
  recursive LLM calls; formalism adds no benefit.
- **Pre-computed deterministic plans** — directly conflicts with the bandit's
  "learn what works per template" premise. The classifier in §3.2 only seeds
  priors; the bandit still updates from outcomes.
- **Closed-form cost bound theorem** — would require constraining the loop the
  same way λ-RLM does. Explicit non-goal.

## 5. Open questions

1. **What benchmark validates the combinators skill?** Need a task class that
   actually rewards SPLIT/MAP/REDUCE — Spark-RCA is the closest existing
   candidate; alternatively synthesize a "count entities across N OneLake jsonl
   shards" task.
2. **What's the right classifier prompt for §3.2?** λ-RLM uses one LLM call;
   ours could be a small `dspy.Predict` or even a regex/keyword first-pass that
   only escalates to LLM on ambiguity.
3. **Does the §3.3 decompose rung need a verifier?** λ-RLM's `REDUCE` is total;
   our synthesis step is an LLM call that can be wrong. Probably needs the same
   validator gate every other rung uses.
4. **Should §3.1 land first as a no-op skill** (just imports + docs) so the
   model has the primitives available before §3.3 tries to depend on them?
   Probably yes — that's the dependency order in the SPECs.

## 6. Cross-references

- `REPORT-comparison-5way.md` — the parent 5-way comparison report
- `REPORT-comparison-5way.md` Addendum 2 — Strategy F bandit results (6/25)
- `REPORT-0.1.10.md` — original adaptive engine validation
- `fabric_rlm/skills/` — current skill inventory (core, data_exploration,
  error_handling, pdf_document_analysis, validation)
- `fabric_rlm/experimental/effort_ladder_policy.py` — current ladder
- `fabric_rlm/experimental/bandit_policy.py` — current bandit
- λ-RLM repo: https://github.com/nktkt/lambda-rlm
- λ-RLM paper: arXiv:2603.20105 (Roy et al. 2026)
