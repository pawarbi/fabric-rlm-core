# fabric-rlm

[![CI](https://github.com/pawarbi/fabric-rlm-core/actions/workflows/test.yml/badge.svg)](https://github.com/pawarbi/fabric-rlm-core/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/fabric-rlm.svg)](https://pypi.org/project/fabric-rlm/)
[![Python](https://img.shields.io/pypi/pyversions/fabric-rlm.svg)](https://pypi.org/project/fabric-rlm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Recursive Language Models (RLMs) for Microsoft Fabric notebooks. The model writes
Python, the code runs in a real CPython subprocess, and the model keeps iterating
(reading outputs, fixing its own errors) until it calls `SUBMIT(...)` with the
answer. It also runs anywhere else CPython runs: your laptop, CI, or an Azure
Function.

## Why not dspy's RLM directly in a notebook

dspy ships its own `RLM`, but it executes generated code in a Deno + Pyodide
(WASM) sandbox. That runtime is impractical to stand up in a Fabric notebook, and
because it is WASM-isolated it cannot import the notebook's installed Python
packages or reach the mounted Lakehouse. fabric-rlm keeps dspy's RLM loop and
replaces the interpreter with a CPython subprocess that runs inside the notebook
process. The model's code gets the full Python ecosystem (pandas, DuckDB, Polars,
openpyxl, PyMuPDF) and can read and write Lakehouse `Files` paths directly. The
raw bytes stay in the subprocess; only the model's summaries and computed results
go back through the LM, so it works on files far larger than any context window.

## What you get

- Built for Fabric notebooks: runs where dspy's Deno/Pyodide RLM can't, and reads
  and writes the mounted Lakehouse filesystem directly.
- No key and no resource to provision on Fabric: `FabricLM` uses the capacity's
  built-in Azure OpenAI endpoint (details below).
- Real subprocess execution: full CPython, native libraries, and real files.
- Reusable skills: Markdown playbooks for PDFs, tabular/log EDA, and Excel, with a
  keyword router that loads only what a task needs.
- A self-correcting loop: a PLAN / VERIFY / REFLECT contract and an output
  verifier feed failures back to the model so it fixes its own code.
- Three engines plus an experimental adaptive one, including native
  `dspy.predict.RLM` interop.
- A default security policy that scrubs secret-bearing environment variables and
  blocks destructive Lakehouse operations (`notebookutils.fs`, `mssparkutils`).
- Structured trajectories you can inspect, save to the Lakehouse, and replay
  offline.
- Runs locally too, and ships `py.typed`.

## Use it in a Fabric notebook

Install on the Python 3.12 `jupyter_python` kernel, then restart the session:

```python
%pip install fabric-rlm
# For the PDF skill and PDF notebooks, add the extra:
# %pip install fabric-rlm[pdf]
```

If imports fail on the Synapse PySpark kernel, see
[docs/fabric-runtime-deps.md](docs/fabric-runtime-deps.md).

On a paid Fabric capacity, `FabricLM` uses the capacity's built-in Azure OpenAI
endpoint. It discovers the endpoint through `synapse.ml.fabric` and authenticates
with the notebook's AAD token, refreshing it automatically on long runs. You do
not provision an Azure OpenAI resource and you do not manage an API key:

```python
from fabric_rlm import RLM, FabricLM, File

rlm = RLM.task(
    task="Find the root cause of the failure in this Spark log.",
    inputs={"log": File("/lakehouse/default/Files/logs/app.log")},
    outputs=["root_cause"],
    lm=FabricLM("gpt-5"),
)
print(rlm.run().root_cause)
```

Because the subprocess runs inside the notebook, the model's code reads and writes
mounted Lakehouse `Files` paths directly, and traces can be written back to the
Lakehouse. See [examples/notebooks/](examples/notebooks/) for ready-to-import
recipes; start with `rlm_api_tour.ipynb`.

## Run it locally

The same code runs on your laptop, in CI, or in an Azure Function. Use `OpenAILM`
or `AnthropicLM` with the matching API key instead of `FabricLM`:

```bash
pip install fabric-rlm
```

```python
from fabric_rlm import RLM, OpenAILM   # set OPENAI_API_KEY in your environment

rlm = RLM.task(
    task="Sum every integer from 1 to 1,000,000 that is divisible by 3 or 5.",
    outputs=["answer"],
    lm=OpenAILM("gpt-4o-mini"),
)
print(rlm.run().answer)
```

`RLM.task(...)` is the short constructor; `RLM.from_task(...)` is the explicit
form. `OpenAILM`, `AnthropicLM`, and `FabricLM` are thin wrappers over `dspy.LM`,
so any OpenAI, Anthropic, Azure, or local Ollama model works.

Requires Python 3.10 or newer. Optional extras:

| Extra | Adds | For |
|---|---|---|
| `fabric-rlm[pdf]` | PyMuPDF | the `pdf_document_analysis` skill and PDF notebooks |
| `fabric-rlm[analytics]` | DuckDB, Polars | large-file EDA with `data_exploration` |
| `fabric-rlm[fabric]` | SynapseML | Microsoft Fabric notebook integration |
| `fabric-rlm[dev]` | pytest | running the test suite |

## How it works

Each turn, the model gets the task (plus any bound inputs and active skills),
writes a block of Python, and the runtime runs it in the worker subprocess.
Standard output, errors, and a bounded snapshot of the namespace go back to the
model so it can correct course. The loop ends when the model calls `SUBMIT(...)`.

Inputs and files. Bind values, including large files, as inputs. Files arrive
inside the worker as `File(...)` handles with `.path`, `.read_text()`,
`.read_bytes()`, and `.exists()`, so a Lakehouse path or a local path is just a
file path.

The SUBMIT contract. The runtime injects `SUBMIT(...)`. Call it with keyword
arguments matching your declared `outputs`, or with positional arguments in the
same order. After a valid SUBMIT, `result.payload` holds the dict and each field
is also reachable as an attribute (`result.answer`).

Recursive sub-LM calls. Inside its Python, the model can call a nested LM with
`predict_sync("english -> french", english=phrase)` (or the async `predict`),
optionally routed to a cheaper `sub_lm=`.

## Engines

`RLM` ships with three stable engines, plus the experimental `adaptive`:

| Engine | What it does | When to pick it |
|---|---|---|
| `"auto"` (default) | Uses `"dspy"` when a non-empty `tools=[...]` is passed, otherwise `"default"` | You don't want to think about it. Recommended. |
| `"default"` | Custom loop with skills, router, reflection, and verifier | You want skills and multi-turn verifier feedback. |
| `"dspy"` | Delegates to `dspy.predict.RLM` with the subprocess as its backend | You want dspy-native composability or `tools=`. |
| `"adaptive"` | Escalates compute (more turns, then higher reasoning effort, then best-of-N, then a stronger LM) when a validator rejects an attempt | Hard, verifiable tasks. Experimental (opt-in `UserWarning`). |

The default `core` skill carries a PLAN / VERIFY / REFLECT contract: plan before
running code, self-check before SUBMIT, and carry prior-attempt failures into
retries. It is on by default. Set `FABRIC_RLM_PVR=0` to turn it off for
token-sensitive batch runs on trivial tasks.

## Bundled skills

Skills are Markdown playbooks the router loads on demand:

- `pdf_document_analysis`: long-document analysis with PyMuPDF
- `data_exploration`: tabular and log EDA with pandas, Polars, and DuckDB
- `excel_extract` / `excel_modify`: read and edit `.xlsx` workbooks via openpyxl
- `core`, `validation`, `error_handling`: default scaffolding

Write your own by copying
[fabric_rlm/skills/SKILL_TEMPLATE.md](fabric_rlm/skills/SKILL_TEMPLATE.md). The
structure is documented in
[PLAYBOOK_CONTRACT.md](fabric_rlm/skills/PLAYBOOK_CONTRACT.md).

## Security

This library runs model-generated code. The default `SecurityPolicy` provides
guardrails, not isolation: it scrubs secret-bearing environment variables from the
worker, screens generated code through a configurable policy before it runs, and
blocks destructive Lakehouse operations (`notebookutils.fs.rm` / `mv`,
`mssparkutils` aliases). Read [SECURITY.md](SECURITY.md) before running untrusted
prompts against sensitive data or credentials.

## CLI

```bash
fabric-rlm --version
fabric-rlm run examples/simple_math/task.json      # run a task from JSON
fabric-rlm trace inspect path/to/trajectory.jsonl  # summarize and diagnose a saved run
```

## Documentation

- [QUICKSTART.md](QUICKSTART.md): step-by-step guide covering install, first run,
  Fabric notebook usage, sub-LM calls, traces, and skill authoring.
- [docs/fabric-runtime-deps.md](docs/fabric-runtime-deps.md): read this if a
  Fabric notebook fails at import time (`Sentinel`, `yarl.Query`,
  `aiohttp.ConnectionTimeoutError`).
- [docs/lossless-submit-payloads.md](docs/lossless-submit-payloads.md): how final
  payloads avoid namespace-snapshot truncation.
- [examples/notebooks/](examples/notebooks/): ready-to-import Fabric recipes,
  including the PDF workflows, the Spark-log root-cause analysis, and the
  SpreadsheetBench benchmarks.
- [CHANGELOG.md](CHANGELOG.md): release history.

## Develop

```bash
git clone https://github.com/pawarbi/fabric-rlm-core.git
cd fabric-rlm-core
pip install -e ".[dev]"
pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## License

MIT. See [LICENSE](LICENSE).
