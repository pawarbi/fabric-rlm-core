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
- **`fabric_rlm_design.md`** — design notes and the long-form story behind
  the runtime.
- **`CHANGELOG.md`** — release history.
- **`examples/notebooks/`** — proven Fabric notebook recipes
  (`rlm_pdf_contract_comparison`, `_invoice_processing`, `_document_analysis`,
  `_document_redaction`).
- **`fabric_rlm/skills/PLAYBOOK_CONTRACT.md`** — how skills are structured;
  copy `SKILL_TEMPLATE.md` to author a new one.

## Develop

```bash
git clone <this-repo>
cd fabric-rlm-core
pip install -e ".[dev]"
pytest -q
```

## License

MIT — see `LICENSE`.
