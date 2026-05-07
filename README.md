# fabric-rlm

A portable Python-subprocess runtime for **Recursive Language Models (RLMs)** —
the model writes Python, the code runs in a real CPython subprocess (not
Pyodide/WASM), and the model iterates until it calls `SUBMIT(...)` with the
answer. Runs anywhere CPython runs: your laptop, CI, a Fabric notebook, an
Azure Function.

This is the **slim core** distribution (v0.1.10). It contains the runtime,
interpreter, skill loader/router, LM backends (OpenAI / Anthropic /
FabricLM), and a small set of always-useful skills:

- `pdf_document_analysis` — long-document analysis with `pymupdf`
- `data_exploration` — tabular EDA with pandas / polars / duckdb
- `core`, `validation`, `error_handling` — always-on scaffolding

It also ships an **experimental adaptive engine** that escalates compute
(more turns → higher reasoning effort → best-of-N → strong LM) when a
validator rejects an attempt — opt-in via `engine="adaptive"`. See
`QUICKSTART.md` §4b for the API and §0.1.10 in `CHANGELOG.md` for the
bench results.

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
# or, from a wheel
pip install dist/fabric_rlm-0.1.10-py3-none-any.whl
```

## 30-second example

```python
from fabric_rlm import RLM, OpenAILM

rlm = RLM(lm=OpenAILM(model="gpt-4o-mini"), skills=["pdf_document_analysis"])
result = rlm.run(
    prompt="Summarize the parties, term, and termination clauses.",
    inputs={"pdf_path": "/path/to/contract.pdf"},
)
print(result.answer)
```

## Where to next

- **`QUICKSTART.md`** — step-by-step getting-started guide (install, first
  run, Fabric notebook usage, skill authoring).
- **`docs/fabric-runtime-deps.md`** — **read this if your Fabric notebook fails at
  import time** (`Sentinel`, `yarl.Query`, `aiohttp.ConnectionTimeoutError`).  Use the
  Python 3.12 `jupyter_python` kernel + `%pip` magic, not the Synapse PySpark kernel.
- **`fabric_rlm_design.md`** — design notes and the long-form story behind
  the runtime.
- **`CHANGELOG.md`** — release history.
- **`examples/notebooks/`** — proven Fabric notebook recipes
  (`rlm_pdf_contract_comparison`, `_invoice_processing`, `_document_analysis`,
  `_document_redaction`).
- **`fabric_rlm/skills/PLAYBOOK_CONTRACT.md`** — how skills are structured;
  copy `SKILL_TEMPLATE.md` to author a new one.

## Choosing a model

RLM gives the most lift when the model is **good enough to self-correct from
runtime errors but not so strong it one-shots the task**. Above and below that
band, the iterative protocol either adds overhead without benefit (strong
models) or burns the turn budget the model needs for the actual computation
(weak models).

| Model class                                              | Use RLM?                                                                  | Why                                                                                                                            |
| -------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Mid-tier instruction-followers** (e.g. `gpt-4.1`)      | ✅ **Sweet spot**                                                          | Reliably emits code blocks, follows the SUBMIT protocol, and reads tracebacks — but still benefits from a structured retry loop. |
| **Cheap instruction-followers** (e.g. `gpt-4.1-mini`)    | ✅ Good cost/perf                                                          | Modest but real gains on multi-step tasks. Cheapest production option.                                                          |
| **Strong reasoning models** (e.g. `gpt-5`)               | ✅ for genuinely iterative or exploratory tasks; ⚠️ skip for one-shot Q&A | Already self-corrects internally. RLM helps when the task needs verification or multi-pass refinement; otherwise it adds latency. |
| **Weak / quantized "nano" models**                       | ❌ Avoid                                                                   | Tight turn budgets + weak self-correction = the loop scaffolding consumes the budget that should go to the actual task.         |

In our internal SSB validation (Excel-modify, n=30 cells across the supported
tier), RLM took the bottom-tier instruction-follower from 0% raw → 40% with a
loop, with no regression on the stronger models. Use that as a rough sanity
check when picking a model for your workload.

## Develop

```bash
git clone <this-repo>
cd fabric-rlm-core
pip install -e ".[dev]"
pytest -q
```

## License

MIT — see `LICENSE`.
