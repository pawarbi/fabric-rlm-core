# fabric-rlm — Test Drive Guide

A 5-minute getting-started guide for the `fabric_rlm` package.

> **What is it?** A portable Python-subprocess runtime for **Recursive Language
> Models** (RLMs). The model writes Python code, the code runs in a real CPython
> subprocess (not Pyodide/WASM), and the model iterates until it calls
> `SUBMIT(...)` with the answer.

---

## 1. Install

### 1a. Local dev

```powershell
# from the repo root
pip install -e .
```

### 1b. Inside a Fabric notebook (from a lakehouse-staged wheel)

```python
# Install the wheel WITHOUT --no-deps for dspy so its transitive deps
# (orjson, cloudpickle, litellm, pydantic, …) come along.
# DuckDB and polars are already in the Fabric Python runtime — don't reinstall.
%pip install -q --no-deps --force-reinstall \
    "/lakehouse/default/Files/fabric_rlm/wheels/fabric_rlm-0.1.9-py3-none-any.whl"
%pip install -q "dspy>=2.5"
```

> ⚠️ **Restart the Python session** after `%pip install` (Fabric ribbon →
> Restart session). `%pip` does not reload already-imported modules.
>
> ⚠️ Don't pass `--no-deps` to the `dspy` line — that's what causes
> `ModuleNotFoundError: No module named 'orjson'` at `from fabric_rlm import
> FabricLM`. Use `--no-deps` only on the `fabric_rlm` wheel itself (to
> protect Fabric's pinned numpy/pandas/duckdb/polars), and let dspy pull
> its own deps.

You also need an LLM API key. The examples below use OpenRouter (any model),
but `dspy.LM` works with OpenAI / Anthropic / Azure / local Ollama too.
**Inside Fabric you don't need a key** — `FabricLM(...)` uses the notebook's
identity (see §3a).

```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-…"
```

---

## 2. Smallest possible run (mock, no API)

This proves the install works and the code interpreter runs:

```powershell
python examples/simple_math/run_mock.py
# → {'answer': 5}
```

---

## 3. Real run with an LLM

### 3a. Inside Fabric (recommended) — use the built-in `FabricLM` helper

`FabricLM` wraps `synapse.ml.fabric.token_utils` + the workload OpenAI endpoint
into a one-liner. No keys, no URLs, no plumbing — uses your notebook's
identity automatically.

```python
from fabric_rlm import RLM, FabricLM

lm = FabricLM("gpt-5")   # defaults: temperature=1.0, max_tokens=16000

rlm = RLM.from_task(
    task="Compute the 30th Fibonacci number using a loop.",
    inputs={"n": 30},
    outputs=["fib"],
    lm=lm,
    max_turns=5,
)
print(rlm(n=30).payload)   # {'fib': 832040}
```

> 💡 **Reasoning-model handling (auto in v0.1.3+):** `FabricLM` /
> `OpenAILM` / `resolve_lm` now detect reasoning models (gpt-5,
> o1/o3/o4 family) and **omit `temperature` automatically** (it's
> rejected/ignored by the API). Pass `reasoning_effort="minimal"|"low"
> |"medium"|"high"` to control depth. Chat models (gpt-4.1,
> gpt-4o, …) keep `temperature` as a normal knob.
>
> ```python
> lm = FabricLM("gpt-5", reasoning_effort="high", max_tokens=32000)
> lm = FabricLM("gpt-4.1-mini", temperature=0.0)   # deterministic chat
> ```
>
> Note: dspy still requires `max_tokens >= 16000` for reasoning models
> (their invisible chain-of-thought counts against the budget). The
> helpers default to 16000; raise it for harder tasks.

Equivalent string-spec form (auto-routes through the same factory):

```python
rlm = RLM.from_task(..., lm="fabric/gpt-5")
```

Equivalent verbose form (if you need to pass extra dspy.LM kwargs the helper
doesn't expose — same code that lives in `fabric_rlm/lm.py:_fabric_factory`):

```python
import dspy
from synapse.ml.fabric.service_discovery import get_fabric_env_config
from synapse.ml.fabric.token_utils import TokenUtils

env = get_fabric_env_config().fabric_env_config
lm = dspy.LM(
    "azure/gpt-5",
    api_key="fabric-token",
    api_base=f"{env.ml_workload_endpoint}cognitive/openai",
    api_version="2025-04-01-preview",
    extra_headers={"Authorization": TokenUtils().get_openai_auth_header()},
    temperature=1.0,        # ignored by reasoning models, but dspy requires this exact value
    max_tokens=16000,       # dspy hard floor for reasoning models; raise for harder tasks
)
```

### 3b. Outside Fabric (e.g. local dev) — pass any `dspy.LM`

```python
import os, dspy
from fabric_rlm import RLM

lm = dspy.LM(
    model="openrouter/openai/gpt-5",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    temperature=1.0,
    max_tokens=16000,
)

rlm = RLM.from_task(task="Compute fib(30)", inputs={"n": 30},
                    outputs=["fib"], lm=lm, max_turns=5)
result = rlm(n=30)            # __call__ is sugar for .run()
print(result.payload)         # {'fib': 832040}
print(result.submitted)       # True
print(result.total_prompt_tokens, result.total_completion_tokens)
```

---

## 4. Two engines — pick one

```python
RLM(..., engine="v6-custom")   # default — our own loop, supports skills/router/reflection
RLM(..., engine="v7-dspy")     # delegates to dspy.predict.RLM, our subprocess as backend
```

| Engine | When to use |
|---|---|
| `v6-custom` | Default. You want skills, router, reflection, multi-turn verifier feedback. |
| `v7-dspy` | You want dspy-native behavior + composability with the rest of your dspy program. Same SUBMIT contract, same subprocess interpreter. |

Both write the SAME Python code through the SAME subprocess. Choose by what
you want around the loop, not for raw capability.

---

## 4b. `engine="adaptive"` — experimental escalation

**When**: hard problems where a single attempt with the cheap LM sometimes
fails but a stronger LM (or more attempts) would succeed. The adaptive engine
is a thin meta-controller around `RLM` that escalates compute when a
validator rejects an attempt:

| Rung | What it does | Typical use |
|---|---|---|
| 0 | baseline cheap LM, default turns | the easy majority |
| 1 | more_turns | "almost there, ran out of room" |
| 2 | more_effort (medium reasoning_effort) | needs a bit more thinking |
| 3 | best_of_N parallel rollouts (same cheap LM) | flaky / temperature-sensitive |
| 4 | strong_lm (e.g. gpt-5, reasoning_effort=high) | genuinely hard |

```python
from fabric_rlm import RLM, FabricLM

def my_validator(result) -> bool:
    return "expected_token" in (result.payload.get("answer") or "")

rlm = RLM(
    signature="question -> answer",
    lm=FabricLM("gpt-4.1-mini"),
    engine="adaptive",                 # outer wrapper
    inner_engine="v6-custom",           # what each attempt uses (default)
    adaptive=dict(
        strong_lm=FabricLM("gpt-5"),    # the rung-4 escalation LM
        validator=my_validator,         # gates pass/fail per attempt
        max_attempts=6,                 # ≥6 needed if parallel_rollouts=3
        parallel_rollouts=3,            # rollouts at rung 3
    ),
)

result = rlm.run({"question": "..."})
print(result.trajectory.metadata["adaptive"])
# {'winner_rung': 4, 'stop_reason': 'best-of-N rollout passed', ...}
```

The power-user surface is also available:

```python
from fabric_rlm.experimental import AdaptiveRunner, LadderPolicy, Budget
runner = AdaptiveRunner(rlm_factory=lambda cfg: RLM(...), policy=LadderPolicy(...), budget=Budget(...))
adaptive_result = runner.run({"question": "..."})
adaptive_result.attempts          # full per-attempt log
adaptive_result.winner.verdict    # validator's last call on the winner
```

This API is **experimental** — it emits a `UserWarning("experimental")` once
at construction. Behaviour, knob names, and metadata layout may change in
0.2.x. Validator is recommended; without one the runner cannot tell when to
escalate, and a `UserWarning` is emitted to flag that.

---

## 5. With a `dspy.Signature` (typed I/O)

Recommended for anything beyond a smoke test — you get free type validation:

```python
import dspy
from fabric_rlm import RLM, signature_validator, chain, assert_list_len

class CountWords(dspy.Signature):
    """Count words per sentence in the input text."""
    text: str  = dspy.InputField()
    counts: list[int] = dspy.OutputField(desc="one int per sentence")

rlm = RLM(
    signature=CountWords,
    lm=lm,
    output_validator=chain(
        signature_validator(CountWords),     # auto: list[int] check
        assert_list_len("counts", n=3),      # semantic: must be 3 sentences
    ),
    engine="v7-dspy",
    max_turns=10,
    halve_max_iter_on_retry=False,            # don't shrink budget on retry
)

result = rlm(text="Hi there. How are you. I am fine.")
print(result.payload)   # {'counts': [2, 3, 3]}
```

If the model SUBMITs the wrong shape, the validator raises `AssertionError`,
the runtime prepends `"VERIFIER FEEDBACK: …"` to the prompt and retries (up to
2x). No glue code needed.

---

## 6. The SUBMIT contract

Inside the model-generated Python code, the runtime injects a `SUBMIT(...)`
function. The model MUST call it with **keyword arguments matching the
signature output fields**:

```python
# Model writes:
result = sum(range(1, 101))
SUBMIT(answer=result)        # ✅ keyword arg
SUBMIT(result)               # ❌ positional — runtime raises a friendly TypeError
```

After SUBMIT, the loop terminates and `result.payload` holds the dict.

---

## 7. Tools (sub-LM, files, custom callbacks)

The model can call helper functions you expose. Two built-ins always available:

```python
# inside model's Python:
ans = llm_query("Summarize: …")     # makes a sub-LM call (uses sub_lm or main lm)
print(read_file("./data.csv"))      # reads from the host FS
```

You can also pass `sub_lm=` separately if you want a cheaper model for nested calls:

```python
RLM(..., lm=FabricLM("gpt-5"), sub_lm=FabricLM("gpt-5-mini"))   # in Fabric
RLM(..., lm=gpt5, sub_lm=dspy.LM("openrouter/openai/gpt-5-mini", ...))   # outside
```

---

## 8. Inspecting what happened — traces & trajectory

```python
result = rlm.run({"x": 42})

# Top-level summary
result.submitted              # bool
result.payload                # dict[str, Any] | None — the SUBMIT kwargs
result.failure_reason         # str | None — set on no-submit/error
result.total_prompt_tokens    # int | None — summed across all turns
result.total_completion_tokens
result.total_lm_seconds       # wall time in LM
result.total_worker_seconds   # wall time in subprocess
```

### Per-turn detail (the trajectory)

```python
for t in result.trajectory:                         # iterate TurnRecord objects
    print(f"--- Turn {t.turn}  type={t.turn_type}  "
          f"submitted={t.submitted}  err={'YES' if t.error else 'no'}")
    print("CODE:\n", t.code)
    print("STDOUT:\n", t.stdout)
    if t.error:
        print("ERROR:\n", t.error)
    print(f"tokens p={t.prompt_tokens} c={t.completion_tokens}  "
          f"lm={t.lm_call_seconds:.2f}s  worker={t.worker_execute_seconds:.2f}s")
```

`TurnRecord` fields: `turn`, `code`, `stdout`, `stderr`, `error`, `submitted`,
`state`, `response_text` (raw LM text), `prompt_tokens`, `completion_tokens`,
`total_tokens`, `lm_call_seconds`, `worker_execute_seconds`,
`validation_errors`, `turn_type` (`normal` / `verifier_repair` / etc.).

### Pretty-print as markdown

```python
print(result.trajectory.to_markdown())
```

### Save & replay later (best for sharing / debugging)

```python
result.trajectory.write_jsonl("/lakehouse/default/Files/traces/run1.jsonl")
```

```bash
python -m fabric_rlm.replay /lakehouse/default/Files/traces/run1.jsonl
python -m fabric_rlm.replay run1.jsonl --turns 3        # limit
python -m fabric_rlm.replay run1.jsonl --json | jq .    # machine-readable
```

### Raw LM prompts/responses (what the model actually saw)

The trajectory captures executed code; for the literal prompt strings, dspy
keeps a per-call log on the LM object:

```python
import json
print(json.dumps(lm.history[-1], indent=2, default=str))   # last call
for h in lm.history:
    print(h.get("prompt", "")[:500], "\n→", str(h.get("response", ""))[:500], "\n---")
```

---

## 9. Skills (optional — domain knowledge as markdown)

> **⚠️ Honest status:** *Explicit* skill preloading works (the markdown gets
> concatenated into the system prompt). The *automatic* router (`enable_router=True`)
> has a known activation bug — trajectory analysis showed it picks the same
> bundle of skills for every question regardless of input. Don't rely on it
> for production. The skills+router path was disabled in the 20Q parity
> bake-off; that's the config we have a green light on.

```python
# ✅ Works: explicit preload
rlm = RLM(..., skills=["pdf_document_analysis", "data_exploration"])

# ⚠️ Works but doesn't actually route — picks a fixed bundle.
# Tracked under todo `router-activation-bug` (CRITICAL, pending).
rlm = RLM(..., enable_router=True, max_active_skills=1)
```

Browse `fabric_rlm/skills/*.md` for examples (e.g. `pdf_document_analysis.md`,
**`data_exploration.md`** — opt-in, teaches DuckDB + ripgrep + Python streaming
for log/CSV/Parquet analysis on files larger than your context). Same caveat
applies to per-skill `verify()` blocks — wired but most skills don't have
meaningful verifier bodies yet (use `output_validator` + the validator
primitives in §10 instead).

### 9a. Large-file / log analysis (new in v0.1.1, opt-in)

> v0.1.2 note: skill text now explicitly tells the LM that `duckdb` and
> `polars` are pre-installed in the Fabric Python runtime, so it won't
> waste a turn trying to `%pip install` them.

For lakehouse logs / CSVs / Parquet too big to fit in context:

```python
from fabric_rlm import RLM, FabricLM

rlm = RLM(
    engine="v7-dspy",
    signature="question -> answer: str",
    lm=FabricLM("gpt-5", max_tokens=16000),   # in Fabric — uses notebook identity
    skills=["data_exploration"],              # opt-in; teaches load-once-then-query
    enable_router=False,
)
rlm(question="The Spark log is at /lakehouse/default/Files/spark.log (53 MB). "
             "Find: (1) failed_job_id, (2) top-3 slow tasks, (3) OOM count.")
```

Optional analytics deps for the DuckDB/polars path the skill recommends:

```bash
# Local dev only — Fabric Python runtime ALREADY ships duckdb + polars,
# so inside Fabric you don't need to install anything for this skill.
pip install fabric-rlm[analytics]   # adds duckdb>=1.1, polars>=1.0
```

Skill is opt-in only; the LM gracefully falls back to pure-Python streaming
if `import duckdb` fails.

---

## 10. Validators reference

All importable from `fabric_rlm`:

| Name | Purpose |
|---|---|
| `signature_validator(sig)` | Auto-derive Pydantic shape check from a `dspy.Signature` |
| `chain(*vs)` | Run validators in order, short-circuit on first failure |
| `assert_keys(*names)` | Required non-None keys |
| `assert_list_len(key, n, exact=True)` | Length check; resolves from `payload['solution']` JSON if needed |
| `assert_list_of(key, type)` | Every item is `isinstance(item, type)` |
| `assert_in_range(key, lo, hi)` | Numeric bounds (rejects `bool`) |
| `assert_matches_regex(key, pattern)` | `re.fullmatch` |
| `assert_predicate(fn, msg)` | Escape hatch for cross-field semantic checks |

See `_mfmc_validator_eval/VALIDATOR_DESIGN.md` for the design rationale.

---

## 11. Common knobs

| Param | Default | What |
|---|---|---|
| `max_turns` | 10 | Cap on tool-loop iterations |
| `timeout` | 300.0 | Subprocess timeout (seconds) |
| `enable_router` | False | Turn on skill router |
| `max_active_skills` | 2 | Max skills the router activates |
| `enable_reflection` | True | Self-reflection between turns (v6 only) |
| `enable_verifier` | True | Skill verifier blocks (v6 only) |
| `output_validator` | None | Your callable; raise `AssertionError` to reject |
| `halve_max_iter_on_retry` | True | If False, retries keep full `max_iterations` |
| `verbose` | False | Print live progress |
| `engine` | `"v6-custom"` | `v6-custom` or `v7-dspy` |

---

## 12. Things to try

Once the basic example runs:

1. **Long-log RCA** — `python _spark_log_eval/run_eval.py fabric` (synthetic 53 MB Spark log; subprocess greps it; LM never sees the raw bytes).
2. **20Q parity** — `python _run_local_20q.py` (CodeSearch templates, fab vs dspy side-by-side).
3. **Sub-LM verification** — `python _test_sub_lm_v7.py` (model calls `llm_query()` from inside its own code).

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OPENROUTER_API_KEY not set` | env var missing | `$env:OPENROUTER_API_KEY = "..."` |
| Subprocess hangs / timeouts | `max_turns` too low for task | bump `max_turns`, set `halve_max_iter_on_retry=False` |
| `_tool_stub() requires KEYWORD arguments only` | model called `SUBMIT(x)` instead of `SUBMIT(answer=x)` | the verifier feedback fixes this on retry; no action needed |
| `payload is None`, `failure_reason="no SUBMIT call"` | model gave up without calling SUBMIT | check `result.trajectory.turns`, increase `max_turns` |
| Repeated wrong shape | no `output_validator` | add `signature_validator(YourSig)` (catches type errors automatically) |

---

## Where to look next

- `fabric_rlm/runtime.py` — `RLM` class, full constructor signature
- `fabric_rlm/validators.py` — validator primitives + auto-generator
- `tests/test_rlm_facade.py` — minimal happy-path examples
- `tests/test_validators.py` — validator usage patterns
- `_mfmc_validator_eval/VALIDATOR_DESIGN.md` — why the validator API looks the way it does
- `_spark_log_eval/RESULTS.md` — long-log RCA writeup (the canonical "RLM wins" demo)
- `fabric_rlm_design.md` — original architecture / design doc
