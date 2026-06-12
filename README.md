# fabric-rlm

A portable Python-subprocess runtime for **Recursive Language Models (RLMs)** —
the model writes Python, the code runs in a real CPython subprocess (not
Pyodide/WASM), and the model iterates until it calls `SUBMIT(...)` with the
answer. Runs anywhere CPython runs: your laptop, CI, a Fabric notebook, an
Azure Function.

This is the **slim core** distribution. It contains the runtime,
interpreter, skill loader/router, LM backends (OpenAI / Anthropic /
FabricLM), and a small set of always-useful skills:

- `pdf_document_analysis` — long-document analysis with `pymupdf`
- `data_exploration` — tabular EDA with pandas / polars / duckdb
- `excel_extract` / `excel_modify` — read and edit `.xlsx` workbooks via openpyxl
- `core`, `validation`, `error_handling` — always-on scaffolding

It also ships an **experimental adaptive engine** that escalates compute
(more turns → higher reasoning effort → best-of-N → strong LM) when a
validator rejects an attempt — opt-in via `engine="adaptive"`. See
`QUICKSTART.md` §4b for the API and `CHANGELOG.md` for the bench results.

### Engine selection

`RLM` ships with three stable engines (plus the experimental `adaptive`):

| Engine value | What it does | When to pick it |
|---|---|---|
| `"auto"` (default) | Picks `"dspy"` when a non-empty `tools=[...]` iterable is supplied, else `"default"` | You don't want to think about it. Recommended. |
| `"default"` | Custom loop with skills/router/reflection/verifier | You want skills + multi-turn verifier feedback. |
| `"dspy"` | Delegates to `dspy.predict.RLM` with our subprocess as backend | You want dspy-native composability or `tools=`. |

`engine="adaptive"` remains experimental (opt-in `UserWarning`).

The default `core` skill ships with a **PLAN / VERIFY / REFLECT (PVR)**
contract: PLAN forces decomposition before any worker code runs, VERIFY
is a self-check before SUBMIT, and REFLECT carries prior-attempt failure
context into retries (helpful when paired with the adaptive engine's
escalation policy). PVR is **on by default** and is universally safe on
easy single-step tasks (no outcome regression, ~30-65% extra prompt
tokens) and materially improves recovery on hard-but-solvable
multi-step tasks. Disable with the `FABRIC_RLM_PVR=0` environment
variable for token-sensitive batch workloads on known-trivial tasks.

## Install

```bash
pip install fabric-rlm
# or, from a locally built wheel
pip install dist/fabric_rlm-<version>-py3-none-any.whl
```

## 30-second example

```python
from fabric_rlm import RLM, OpenAILM

rlm = RLM.task(
    task="Summarize the parties, term, and termination clauses.",
    inputs={"pdf_path": "/path/to/contract.pdf"},
    outputs=["answer"],
    lm=OpenAILM("gpt-4o-mini"),
    skills=["pdf_document_analysis"],
)
result = rlm.run()
print(result.answer)
```

`RLM.from_task(...)` remains available as the explicit constructor form.

There is also a small CLI (installed as `fabric-rlm`):

```bash
fabric-rlm --version
fabric-rlm run examples/simple_math/task.json          # run a task from JSON
fabric-rlm trace inspect path/to/trajectory.jsonl      # summarize + diagnose a saved run
```

## Where to next

- **`QUICKSTART.md`** — step-by-step getting-started guide (install, first
  run, Fabric notebook usage, skill authoring).
- **`docs/fabric-runtime-deps.md`** — **read this if your Fabric notebook fails at
  import time** (`Sentinel`, `yarl.Query`, `aiohttp.ConnectionTimeoutError`).  Use the
  Python 3.12 `jupyter_python` kernel + `%pip` magic, not the Synapse PySpark kernel.
- **`CHANGELOG.md`** — release history.
- **`examples/notebooks/`** — proven Fabric notebook recipes
  (`rlm_pdf_contract_comparison`, `_invoice_processing`, `_document_analysis`,
  `_document_redaction`).
- **`fabric_rlm/skills/PLAYBOOK_CONTRACT.md`** — how skills are structured;
  copy `SKILL_TEMPLATE.md` to author a new one.

## Develop

```bash
git clone https://github.com/pawarbi/fabric-rlm-core.git
cd fabric-rlm-core
pip install -e ".[dev]"
pytest -q
```

See `CONTRIBUTING.md` for the development workflow and `SECURITY.md` for the
security model of the code-execution sandbox (read it before running
untrusted prompts against production data).

## License

MIT — see `LICENSE`.
