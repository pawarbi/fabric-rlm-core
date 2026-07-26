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

### 1b. Inside a Fabric notebook

```python
%pip install -q fabric-rlm
# For the PDF notebooks, add the optional PyMuPDF extra:
# %pip install -q fabric-rlm[pdf]
```

> **Restart the Python session** after `%pip install` (Fabric ribbon →
> Restart session). `%pip` does not reload already-imported modules.

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

> **Which models can I name?** `FabricLM("...")` / `lm="fabric/..."` must
> reference a model that Fabric's prebuilt Foundry Tools host. The hosted set
> changes over time and varies by region (gpt-5 and gpt-4.1 were retired in
> June 2026; gpt-5.1 and gpt-5-mini are current) — check the authoritative
> list before picking a model:
> [Fabric AI services model list](https://learn.microsoft.com/en-us/fabric/data-science/ai-services/ai-services-overview#consumption-rate).
> For any model not on that list, use **Bring Your Own Key (BYOK)** or run
> outside Fabric (see §3b).

```python
from fabric_rlm import RLM, FabricLM

lm = FabricLM("gpt-5.1")   # reasoning model: temperature omitted automatically; max_tokens=16000

rlm = RLM.task(
    task="Compute the 30th Fibonacci number using a loop.",
    inputs={"n": 30},
    outputs=["fib"],
    lm=lm,
    max_turns=5,
)
print(rlm(n=30).payload)   # {'fib': 832040}
```

> **Reasoning-model handling:** `FabricLM` /
> `OpenAILM` / `resolve_lm` now detect reasoning models (gpt-5.x,
> o1/o3/o4 family) and **omit `temperature` automatically** (it's
> rejected/ignored by the API). Pass `reasoning_effort` (e.g. "low",
> "medium", "high"; allowed values vary by model generation) to control
> depth. Chat models keep `temperature` as a normal knob.
>
> ```python
> lm = FabricLM("gpt-5.1", reasoning_effort="high", max_tokens=32000)
> lm = FabricLM("gpt-4.1-mini", temperature=0.0)   # deterministic chat
> ```
>
> Note: dspy still requires `max_tokens >= 16000` for reasoning models
> (their invisible chain-of-thought counts against the budget). The
> helpers default to 16000; raise it for harder tasks.

Equivalent string-spec form (auto-routes through the same factory):

```python
rlm = RLM.task(..., lm="fabric/gpt-5.1")
```

Equivalent verbose form (if you need to pass extra dspy.LM kwargs the helper
doesn't expose — same code that lives in `fabric_rlm/lm.py:_fabric_factory`):

```python
import dspy
from synapse.ml.fabric.service_discovery import get_fabric_env_config
from synapse.ml.fabric.token_utils import TokenUtils

env = get_fabric_env_config().fabric_env_config
lm = dspy.LM(
    "azure/gpt-5.1",
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
    model="openrouter/openai/gpt-5.1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    temperature=1.0,
    max_tokens=16000,
)

rlm = RLM.task(task="Compute fib(30)", inputs={"n": 30},
               outputs=["fib"], lm=lm, max_turns=5)
result = rlm(n=30)            # __call__ is sugar for .run()
print(result.payload)         # {'fib': 832040}
print(result.submitted)       # True
print(result.total_prompt_tokens, result.total_completion_tokens)
```

---

## 4. Engines — `"auto"` is the default

```python
RLM(...)                       # engine="auto" — picks "dspy" if non-empty tools=[...] is supplied, else "default"
RLM(..., engine="default")     # custom loop, supports skills/router/reflection/verifier
RLM(..., engine="dspy")        # delegates to dspy.predict.RLM, our subprocess as backend
RLM(..., engine="dspy", tools=[my_tool, ...])  # tools= requires dspy
```

| Engine | When to use |
|---|---|
| `"auto"` (default) | You don't want to think about it. If you pass a non-empty `tools=[...]`, you get `"dspy"`; otherwise `"default"`. |
| `"default"` | You want skills, router, reflection, multi-turn verifier feedback. |
| `"dspy"` | You want dspy-native behavior + composability with the rest of your dspy program, or you need `tools=`. Same SUBMIT contract, same subprocess interpreter. |

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
| 4 | strong_lm (e.g. gpt-5.1, reasoning_effort=high) | genuinely hard |

```python
from fabric_rlm import RLM, FabricLM

def my_validator(result) -> bool:
    return "expected_token" in (result.payload.get("answer") or "")

rlm = RLM(
    signature="question -> answer",
    lm=FabricLM("gpt-4.1-mini"),
    engine="adaptive",                 # outer wrapper
    inner_engine="default",             # what each attempt uses (default)
    adaptive=dict(
        strong_lm=FabricLM("gpt-5.1"),    # the rung-4 escalation LM
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
future minor releases. Validator is recommended; without one the runner cannot tell when to
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
    engine="dspy",
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
function. Call it with **keyword arguments matching your declared `outputs`**
(recommended), or positional arguments in the same order:

```python
# Model writes (declared outputs=["answer"]):
result = sum(range(1, 101))
SUBMIT(answer=result)        # keyword — recommended, unambiguous
SUBMIT(result)               # also valid: positional maps to outputs in order
```

`SUBMIT` raises a `TypeError` only when you pass an unknown field name or more
positional arguments than there are output fields; the verifier surfaces that
and the model retries. After a valid SUBMIT the loop terminates and
`result.payload` holds the dict.

---

## 7. Tools (sub-LM, files, custom callbacks)

The model can call helper functions the runtime injects. The two you'll use most:

```python
# inside the model's Python:

# Recursive sub-LM call via a DSPy-style signature. Returns a result object
# whose attributes are the signature's output fields (uses sub_lm or the main lm).
fr = predict_sync("english -> french", english="Good morning")
print(fr.french)

# Files you bind as inputs arrive as File(...) handles, with .path, .name,
# .read_text(), .read_bytes(), and .exists():
#   RLM.task(..., inputs={"doc": File("/path/report.pdf")})
```

`predict(...)` is the async form of `predict_sync(...)`. The runtime also injects
`load_skill`, `activate_skill`, and `list_skills` for on-demand skill loading.

You can also pass `sub_lm=` separately if you want a cheaper model for nested calls:

```python
RLM(..., lm=FabricLM("gpt-5.1"), sub_lm=FabricLM("gpt-5-mini"))   # in Fabric
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

> **Honest status:** *Explicit* skill preloading is the recommended,
> battle-tested path (the markdown gets concatenated into the system prompt).
> The *automatic* router (`enable_router=True`) routes on keywords in the
> bound input values, **falling back to the `task=` text when the inputs
> carry no keyword signal**. It is still a keyword heuristic,
> not semantic matching — for production workloads with a known domain,
> prefer explicit `skills=[...]`. Check
> `result.trajectory.metadata["router_active"]` and
> `["router_used_task_text_fallback"]` to see what the router did.

```python
# Recommended: explicit preload
rlm = RLM(..., skills=["pdf_document_analysis", "data_exploration"])

# Keyword-heuristic routing; verify router_active in trajectory metadata.
rlm = RLM(..., enable_router=True, max_active_skills=2)
```

Browse `fabric_rlm/skills/*.md` for examples (e.g. `pdf_document_analysis.md`,
**`data_exploration.md`** — opt-in, teaches DuckDB + ripgrep + Python streaming
for log/CSV/Parquet analysis on files larger than your context). Same caveat
applies to per-skill `verify()` blocks — wired but most skills don't have
meaningful verifier bodies yet (use `output_validator` + the validator
primitives in §10 instead).

### 9a. Large-file / log analysis

For lakehouse logs / CSVs / Parquet too big to fit in context:

```python
from fabric_rlm import RLM, FabricLM

rlm = RLM(
    engine="dspy",
    signature="question -> answer: str",
    lm=FabricLM("gpt-5.1", max_tokens=16000),   # in Fabric — uses notebook identity
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

See `fabric_rlm/validators.py` for the implementation and
`tests/test_validators.py` for usage patterns.

---

## 11. Common knobs

| Param | Default | What |
|---|---|---|
| `max_turns` | 10 | Cap on tool-loop iterations |
| `timeout` | 300.0 | Subprocess timeout (seconds) |
| `enable_router` | False | Turn on skill router |
| `max_active_skills` | 2 | Max skills the router activates |
| `enable_verifier` | True | Enable skill verifier blocks when configured |
| `output_validator` | None | Your callable; raise `AssertionError` to reject |
| `halve_max_iter_on_retry` | True | If False, retries keep full `max_iterations` |
| `verbose` | False | Print live progress |
| `engine` | `"auto"` | `"auto"` (default), `"default"`, `"dspy"`, `"adaptive"` (experimental). |

---

## 12. Things to try

Once the basic example runs:

1. **CLI run** — `fabric-rlm run examples/simple_math/task.json` (swap the
   `lm` field in the JSON for your model), then
   `fabric-rlm trace inspect <trajectory.jsonl>` on the saved trace.
2. **Fabric notebook recipes** — `examples/notebooks/` ships minimal,
   ready-to-import Fabric notebooks for API basics, PDF workflows, and
   Spark-log root-cause analysis.
3. **Long-file analysis** — point the `data_exploration` skill at a CSV/log
   too big for context (§9a): the subprocess greps/queries it; the LM never
   sees the raw bytes.
4. **Sub-LM calls** — give the model a cheaper `sub_lm=` and a task that
   needs per-item summarization; watch it call `predict_sync()` from inside its
   own generated code (§7).

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OPENROUTER_API_KEY not set` | env var missing | `$env:OPENROUTER_API_KEY = "..."` |
| Subprocess hangs / timeouts | `max_turns` too low for task | bump `max_turns`, set `halve_max_iter_on_retry=False` |
| `SUBMIT() got unexpected keyword argument ...` / `takes at most N positional arguments` | model's SUBMIT call doesn't match the declared `outputs=[...]` | the verifier feedback fixes this on retry; make sure `outputs=[...]` matches what the model submits |
| `<tool>(...) requires KEYWORD arguments only, not positional` | a registered custom tool was called positionally | call registered tools with keyword arguments |
| `payload is None`, `failure_reason="max_turns"` | model never called SUBMIT before the budget ran out | check `result.trajectory.turns`, increase `max_turns` |
| `failure_reason="stuck_loop"` | model re-emitted identical failing code 3+ turns in a row | inspect the repeated error in the trajectory; rephrase the task or raise `stuck_loop_threshold` |
| `failure_reason="worker_timeout"` / `"worker_error"` | a code turn exceeded `timeout` / crashed the worker | raise `timeout=`, or check the recorded turn's `error` field |
| `failure_reason="output_validation_failed"` | final SUBMIT was missing required fields | declare realistic `outputs=[...]`; see the turn's `validation_errors` |
| Repeated wrong shape | no `output_validator` | add `signature_validator(YourSig)` (catches type errors automatically) |

---

## Where to look next

- `fabric_rlm/runtime.py` — `RLM` class, full constructor signature
- `fabric_rlm/validators.py` — validator primitives + auto-generator
- `fabric_rlm/security.py` — the module docstring is the authoritative
  statement of what the security baseline does and does not protect against
- `docs/authoring-skills.md` — how to author a skill
- `tests/test_rlm_facade.py` — minimal happy-path examples
- `tests/test_validators.py` — validator usage patterns
- `examples/notebooks/` — end-to-end Fabric recipes
