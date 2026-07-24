# fabric-rlm

[![CI](https://github.com/pawarbi/fabric-rlm-core/actions/workflows/test.yml/badge.svg)](https://github.com/pawarbi/fabric-rlm-core/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/fabric-rlm.svg)](https://pypi.org/project/fabric-rlm/)
[![Python](https://img.shields.io/pypi/pyversions/fabric-rlm.svg)](https://pypi.org/project/fabric-rlm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Recursive Language Models (RLMs) for real Python environments. The model writes
Python, the code runs in a real CPython subprocess (not Pyodide or WASM), and the
model keeps iterating (reading outputs, fixing its own errors) until it calls
`SUBMIT(...)` with the answer. It runs anywhere CPython runs: your laptop, CI, a
Microsoft Fabric notebook, or an Azure Function.

Because the code runs in a full CPython process, the model can use the whole
ecosystem (pandas, DuckDB, Polars, openpyxl, PyMuPDF) to work over files that are
far larger than any context window. The raw bytes stay in the subprocess; only
the model's summaries and computed results go back through the LM.

## What you get

- Real subprocess execution: full CPython, native libraries, and real files, not
  a WASM sandbox with a partial standard library.
- Reusable skills: Markdown playbooks for PDFs, tabular/log EDA, and Excel, with a
  keyword router that loads only what a task needs.
- A self-correcting loop: a PLAN / VERIFY / REFLECT contract and an output
  verifier feed failures back to the model so it fixes its own code.
- Three engines plus an experimental adaptive one, including native
  `dspy.predict.RLM` interop.
- A default security policy that scrubs secret-bearing environment variables from
  the worker and screens generated code before it runs.
- Structured trajectories you can inspect, save, and replay offline.
- Type hints, with a shipped `py.typed`.

## Install

```bash
pip install fabric-rlm
```

Optional extras:

| Extra | Adds | For |
|---|---|---|
| `fabric-rlm[pdf]` | PyMuPDF | the `pdf_document_analysis` skill and PDF notebooks |
| `fabric-rlm[analytics]` | DuckDB, Polars | large-file EDA with `data_exploration` |
| `fabric-rlm[fabric]` | SynapseML | Microsoft Fabric notebook integration |
| `fabric-rlm[dev]` | pytest | running the test suite |

Requires Python 3.10 or newer. Inside a Fabric notebook, run
`%pip install fabric-rlm` on the Python 3.12 `jupyter_python` kernel. See
[docs/fabric-runtime-deps.md](docs/fabric-runtime-deps.md) if imports fail.

## 30-second example

```python
from fabric_rlm import RLM, OpenAILM   # set OPENAI_API_KEY in your environment

# The model writes Python, runs it in a real CPython subprocess, and keeps
# iterating until it calls SUBMIT(...) with the answer.
rlm = RLM.task(
    task="Sum every integer from 1 to 1,000,000 that is divisible by 3 or 5.",
    outputs=["answer"],
    lm=OpenAILM("gpt-4o-mini"),
)
result = rlm.run()
print(result.answer)
```

`RLM.task(...)` is the short constructor; `RLM.from_task(...)` is the explicit
form. `OpenAILM`, `AnthropicLM`, and `FabricLM` are thin wrappers over `dspy.LM`,
so any OpenAI, Anthropic, Azure, or local Ollama model works too.

## How it works

Each turn, the model gets the task (plus any bound inputs and active skills),
writes a block of Python, and the runtime runs it in the worker subprocess.
Standard output, errors, and a bounded snapshot of the namespace go back to the
model so it can correct course. The loop ends when the model calls `SUBMIT(...)`.

Inputs and files. Bind values, including large files, as inputs. Files arrive
inside the worker as `File(...)` handles with `.path`, `.read_text()`,
`.read_bytes()`, and `.exists()`:

```python
from fabric_rlm import RLM, File, OpenAILM

rlm = RLM.task(
    task="Summarize the parties, term, and termination clauses.",
    inputs={"doc": File("/path/contract.pdf")},
    outputs=["summary"],
    lm=OpenAILM("gpt-4o-mini"),
    skills=["pdf_document_analysis"],   # needs the [pdf] extra
)
print(rlm.run().summary)
```

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

This library runs model-generated code on your machine. The default
`SecurityPolicy` provides guardrails, not isolation: it scrubs secret-bearing
environment variables from the worker and screens generated code through a
configurable policy before it runs. Read [SECURITY.md](SECURITY.md) before
running untrusted prompts against sensitive data or credentials.

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
- [examples/notebooks/](examples/notebooks/): ready-to-import Fabric recipes.
  Start with `rlm_api_tour.ipynb`, then the PDF workflows, the Spark-log
  root-cause analysis, and the SpreadsheetBench benchmarks.
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
