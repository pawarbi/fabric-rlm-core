# Fabric-RLM

[![CI](https://github.com/pawarbi/fabric-rlm-core/actions/workflows/test.yml/badge.svg)](https://github.com/pawarbi/fabric-rlm-core/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/fabric-rlm.svg)](https://pypi.org/project/fabric-rlm/)
[![Python](https://img.shields.io/pypi/pyversions/fabric-rlm.svg)](https://pypi.org/project/fabric-rlm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


Recursive Language Models (RLMs) for Microsoft Fabric notebooks. The model writes
Python, the code runs in a real CPython subprocess, and the model keeps iterating
(reading outputs, fixing its own errors) until it calls `SUBMIT(...)` with the
answer. It also runs anywhere else CPython runs: your laptop, CI, or an Azure
Function.

Recursive Language Models were introduced by Alex L. Zhang, Tim Kraska, and Omar
Khattab at MIT CSAIL, who define an RLM as a "general inference paradigm that
treats long prompts as part of an external environment". Instead of loading the
whole context into the model call, the model examines, decomposes, and recurses
over that context programmatically. That is what makes long-context, multi-step
work tractable when a single prompt cannot hold the data. Full citation in
[Acknowledgments](#acknowledgments).

**`fabric-rlm` is built and optimised for Fabric notebook environment**.

## Start here

[examples/notebooks/rlm_vs_plain_llm_imf_cpi.ipynb](examples/notebooks/rlm_vs_plain_llm_imf_cpi.ipynb)
is the notebook to run first. Import it into a Fabric workspace, attach a
Lakehouse, and it fetches its own data.

It ends by reasoning over two sources at once: a 140 MB CSV of IMF price data and
a 15-page PDF, the IMF's July 2026 World Economic Outlook Update. Together they
come to about 79 million tokens, roughly 200 times the 400,000-token context
window of gpt-5.1, so no prompt can hold them. The model computes a country
table from the CSV, reads figures out of the report's prose, and combines both
into one formatted Excel workbook, which the notebook then grades cell by cell.

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
- Plug and play : Point it to your files in the Lakehouse, define the task & contract
- It's primarily optimised and designed for data exploration, data analysis tasks

## When to use an RLM (and when not to)

An RLM earns its overhead when the task needs computation, iteration, or data
that will not fit in a context window. For a quick question or a conversational
task, a plain LLM call is cheaper and faster; the loop and the subprocess add
nothing.

Use an RLM when:

- The data is larger than any context window. Multi-gigabyte Spark logs, wide
  Excel workbooks, long JSONL streams: the model queries them with DuckDB,
  Polars, or openpyxl inside the subprocess, and only its aggregates enter the
  LM context. This is what the `data_exploration` skill was built for.
- The answer must be computed exactly or written to a file. An LLM cannot
  reliably do arithmetic over thousands of rows or edit an `.xlsx` in place; code
  can. On the full SpreadsheetBench Verified-400, MiniMax M3 through this RLM
  passes 82.5 percent of questions for about $2.60 of total model spend; see
  the benchmark section below.
- The task is multi-step and its output is checkable. With a validator attached,
  failed attempts feed structured reflection into retries: in the ablation, a
  hard multi-step task went from ladder-exhausted failure to passing, with 68%
  fewer tokens.
- You need strict multi-field extraction from long documents. The same ablation
  measured a 74% token reduction on a 100KB RFP extraction with the
  plan/verify contract on.

Skip the RLM when:

- It is a conversation, a rewrite, or a judgment call over text that fits in
  context. There is nothing to compute, so a single LM call wins.
- The model nails the task in one shot. The ablation caught this failure mode
  directly: on an easy log-extraction task the verify loop spuriously re-checked
  a correct first answer, inflating cost from 6K to 40K tokens and 10 seconds to
  176. Correctness held, but a plain call would have been 10x cheaper.
- The task exceeds the model's capability. The loop retries more cheaply, but it
  does not rescue tasks the model fundamentally cannot solve; use a stronger
  model instead.

### A worked example

One task, two attempts. The task: from a 140 MB IMF CPI pull (1.5 million rows,
194 countries, fetched live from the public SDMX API), build a formatted Excel
report: a pivot of the 10 highest-inflation countries by year, a merged title
cell, styled headers, and a second sheet listing every qualifying country.

The first attempt gives gpt-5.1 the question plus as much raw CSV as
fits in a prompt. The second gives gpt-5-mini, about 5x cheaper, the same
question through the RLM. Measured result:

| run | result | workbook | tokens | cost | seconds |
|---|---|---|---|---|---|
| plain call, gpt-5.1 | failed | none | 109,480 | $0.138 | 8.4 |
| RLM, gpt-5-mini | passed | correct, verified | 47,642 | $0.023 | 78.4 |

gpt-5.1 burned 109K tokens discovering the data was never in its context.
The mini model wrote DuckDB and openpyxl code in the subprocess, built the
workbook, and a deterministic ground-truth query verified every cell. The
failed call cost six times more than the successful one. The notebook then
pushes the same mini model through a harder task (finding inflation streaks
with tie-breaks, conditional formatting, and an embedded chart, cleared for
about two cents), runs an honest skill ablation, and finishes with the
two-source task described under [Start here](#start-here). Run it yourself:
[examples/notebooks/rlm_vs_plain_llm_imf_cpi.ipynb](examples/notebooks/rlm_vs_plain_llm_imf_cpi.ipynb).

### Benchmark: SpreadsheetBench Verified-400

[SpreadsheetBench](https://github.com/RUCKBReasoning/SpreadsheetBench-2) tests
whether an agent can carry out real spreadsheet-manipulation instructions,
graded cell-exactly against golden workbooks. On the full Version 1
Verified-400 set (all 400 questions, single attempt each, temperature 1.0,
fabric-rlm 0.2.8 with the `excel_modify` skill and workbook structure context,
model served from MiniMax's first-party endpoint):

| system | model | pass rate | model spend |
|---|---|---|---|
| fabric-rlm | MiniMax M3 (open weights, $0.30/M in, $1.20/M out) | **82.5%** (330/400) | $2.61 total, $0.0065 per question |

That 82.5 percent is the score reported by the benchmark's own
`evaluation.py`, run unmodified over our output workbooks, so it is measured the
same way as every figure below. It is self-reported in the sense that we ran the
script ourselves; an official submission is planned.

For context, the top of the public V1-Verified (400) leaderboard is held by
commercial spreadsheet products: Qingqiu Agent at 98.25, ByteDance's Data
Analysis Agent at 96.5, GPT for Excel at 92.5, WPS AI at 91.25. Ours comes from
an open-source library driving a cheap open-weight model, at a cost of well
under a cent per task.

```mermaid
xychart-beta
    title "SpreadsheetBench V1 Verified-400 pass rate (percent)"
    x-axis ["fabric-rlm + MiniMax M3", "WPS AI", "GPT for Excel", "Data Analysis Agent", "Qingqiu Agent"]
    y-axis "pass rate" 0 --> 100
    bar [82.5, 91.25, 92.5, 96.5, 98.25]
```

For scale, two vendors publish a result for Claude Opus 4.6 driven by a bare
prompt with no agent loop: 321/400 = 80.2 percent
([DealGlass results repo](https://github.com/arthursolwayne/spreadsheet-agents))
and 80.25 percent (Leni). Opus 4.6 tokens are priced roughly 17 times higher on
input and 21 times higher on output than MiniMax M3. So an open-source runtime
driving an open-weight model with a spreadsheet skill scores slightly above a
frontier model asked directly, at a small fraction of the per-token price. Two
caveats worth stating plainly: that comparison is against a single-shot prompt,
and the same Opus 4.6 inside a purpose-built scaffold reaches 95.2 percent, so the
honest reading is that scaffolding matters more than model choice, not that one
model beats another. All three figures are self-reported.

Reproduce it with
[examples/notebooks/ssb400_minimax_m3_fabric_repro.ipynb](examples/notebooks/ssb400_minimax_m3_fabric_repro.ipynb);
the run needs an OpenRouter key and costs a few dollars.

## Benchmark: AIDABench

SpreadsheetBench hands you one workbook and tells you which cells to fill.
[AIDABench](https://arxiv.org/abs/2603.15636) is closer to real analyst work: read
one or more source files, decide the shape of the answer yourself, and write a new
file. 41 percent of its file-generation tasks span multiple inputs, up to 13 in a
single task, and the target range is never given.

Same library, same MiniMax M3, no per-benchmark tuning:

| split | fabric-rlm + MiniMax M3 | best in the paper | cost per task |
|---|---|---|---|
| File generation (261 tasks) | ~42% | 49.4% (Claude Sonnet 4.5) | $0.023 vs $0.237 |
| Question answering (226 tasks) | 63.6% | 68.6% (Claude Sonnet 4.5) | $0.012 vs $0.122 |

![AIDABench accuracy vs cost per task](docs/assets/aida-benchmark.png)

That is five to seven points behind the leading model at roughly a tenth of the
cost on both splits, which puts it mid-table against the eleven models in the
paper. The cost figures price identical token usage at each model's published
rate, so they compare workloads rather than observed spend.

The third split, data visualization, was not run. Its deliverable is a chart image
graded on presentation rubrics, which needs a vision-capable judge and
chart-construction guidance this library does not ship.

File-generation scores were checked against AIDABench's own evaluator, run
unmodified, on a 62-task sample: 41.9 percent against our judge's 43.5 percent. QA
was graded with their `eval_QA.py` under two different grader models, which agreed
on 94.6 percent of answers.

The runners, both graders, every trajectory, the grader calibration, and an
account of the seven grading bugs found along the way are in
[pawarbi/fabric-rlm-benchmarks](https://github.com/pawarbi/fabric-rlm-benchmarks).
These are single-seed numbers. Two identical runs agreed on 84 percent of tasks,
so treat differences under about ten points as noise.

## How it works, in one picture

```mermaid
flowchart TD
    A["Task + inputs<br/>(values and File handles)"] --> B["RLM runtime<br/>routes skills, builds the prompt"]
    B --> C["Model writes Python"]
    C --> D["CPython subprocess executes it<br/>pandas, DuckDB, openpyxl, Lakehouse Files"]
    D --> E{"SUBMIT(...) called?"}
    E -- "no" --> F["stdout, errors, and a bounded<br/>namespace snapshot go back to the model"]
    F --> C
    E -- "yes" --> G{"Output verifier<br/>accepts?"}
    G -- "rejected, with feedback" --> C
    G -- "accepted" --> H["Result payload + trajectory<br/>(inspect, save, replay)"]
```

The raw bytes of your files never enter the LM context. The model reads
summaries, previews, and computed aggregates; the heavy lifting happens in the
subprocess.

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
    lm=FabricLM("gpt-5.1"),
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

## The worker API

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

## Skills

Skills are Markdown playbooks that tell the model how to do a kind of work
properly: which library to reach for, the traps to avoid, and what to check
before submitting. Seven ship with the package, so there is nothing to download
or configure. Name the ones a task needs and they are prepended to the prompt;
name none and the keyword router picks for you.

```python
from fabric_rlm import RLM, File, FabricLM

rlm = RLM.task(
    task="Rebuild the summary tab from the raw export and flag any variance over 5 percent.",
    inputs={"workbook": File("/lakehouse/default/Files/finance/q3.xlsx")},
    outputs=["answer"],
    lm=FabricLM("gpt-5.1"),
    skills=["excel_modify", "data_exploration"],   # load as many as the task needs
)
print(rlm.run().answer)
```

| Skill | What it covers |
|---|---|
| `excel_modify` | Editing `.xlsx` in place with openpyxl: writing computed values rather than formula strings, merged-cell anchors, target-range discipline, verifying by reloading |
| `excel_extract` | Reading workbooks: locating real header rows, multi-table sheets, formula versus cached value, pulling structured records out of messy layouts |
| `data_exploration` | Files too large for context: DuckDB and Polars over CSV, Parquet and JSONL, aggregating in code so raw rows never reach the prompt |
| `pdf_document_analysis` | Long documents with PyMuPDF: page enumeration, chunking, and per-chunk extraction |
| `core` | The PLAN / VERIFY / REFLECT contract applied to every run |
| `validation` | Checking an answer against the task's constraints before submitting |
| `error_handling` | What to do when a turn raises, so the next turn fixes rather than repeats |

The first four are keyword-routed, so `data_exploration` activates on a task
mentioning logs or CSVs even if you name no skills at all. The last three are
scaffolding and load by default.

### Writing your own

A skill is one Markdown file with a small frontmatter block. Put it anywhere the
notebook can read, including Lakehouse `Files`, and point a `SkillLoader` at that
folder:

```python
from fabric_rlm import RLM, SkillLoader, FabricLM

loader = SkillLoader(skill_dir="/lakehouse/default/Files/skills")
print(loader.list_skills())        # your skills plus the bundled ones

rlm = RLM.task(
    task="Extract the vendor totals from this invoice.",
    inputs={"doc": File("/lakehouse/default/Files/invoices/2026-07.pdf")},
    outputs=["totals"],
    lm=FabricLM("gpt-5.1"),
    skill_loader=loader,
    skills=["invoice_rules", "pdf_document_analysis"],
)
```

Your folder layers over the bundled skills rather than replacing them, so you can
mix your own with the shipped ones in the same `skills=[...]` list. Pass several
folders as a list if you keep them apart, and a file named after a bundled skill
overrides it. If you want only your own, pass `include_packaged=False`.

### Contributed skills

`contrib-skills/` in the repository holds playbooks that are not installed with the
package, either because they are narrower than the bundled ones or because the
measurements behind them are thinner than a default install should carry. Point a
loader at the folder to use one:

```python
loader = SkillLoader(skill_dir="contrib-skills")

rlm = RLM.task(
    task="Which segment had the largest operating loss in FY2022?",
    inputs={"filing": File("BOEING_2022_10K.pdf")},
    outputs=["answer"],
    skill_loader=loader,
    skills=["pdf_document_analysis", "financial_documents"],
)
```

`financial_documents` is the first of these: reporting conventions for 10-K, 10-Q
and earnings releases, covering parentheses as negative, scale stated in a header,
fiscal against calendar year, adjacent period columns and subtotal rows. It is
scoped to financial reporting on purpose, since parentheses mean something else in
legal and scientific documents. [docs/contrib-skills.md](docs/contrib-skills.md)
records what it was measured on and what the measurement does not support.

If a skill needs a library fabric-rlm does not depend on, install it in the
notebook (`%pip install python-docx`) before running. The sandbox blocks `pip`
and `subprocess`, so a skill cannot install its own dependencies. See
[docs/authoring-skills.md](docs/authoring-skills.md).

This is how house rules stop being tribal knowledge: your chart of accounts, the
naming conventions your reports use, the columns that are always dates. Write it
once, store it beside the data, and every run reads from the same copy.

Start from [docs/skill-template.md](docs/skill-template.md); the structure is
documented in [docs/authoring-skills.md](docs/authoring-skills.md).

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
- [examples/notebooks/](examples/notebooks/): ready-to-import Fabric recipes.
  Start with `rlm_vs_plain_llm_imf_cpi.ipynb` (the with-and-without comparison)
  and `rlm_api_tour.ipynb`, then the PDF workflows, the Spark-log root-cause
  analysis, and the SpreadsheetBench benchmarks.
- [CHANGELOG.md](CHANGELOG.md): release history.

## Develop

```bash
git clone https://github.com/pawarbi/fabric-rlm-core.git
cd fabric-rlm-core
pip install -e ".[dev]"
pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Acknowledgments

This project stands on the work of others, and it matters to say so:

- The Recursive Language Model paradigm comes from the paper
  [Recursive Language Models](https://arxiv.org/abs/2512.24601) by Alex L.
  Zhang, Tim Kraska, and Omar Khattab (MIT CSAIL), which showed that letting a
  model programmatically examine and recursively query its own prompt beats
  stuffing everything into context.
- [DSPy](https://github.com/stanfordnlp/dspy) provides the RLM predictor and
  the interpreter protocol this library plugs into, and `dspy.LM` powers every
  model backend here.
- [Predict-RLM](https://github.com/Trampoline-AI/predict-rlm) by Trampoline AI,
  a production-focused RLM runtime built on DSPy signatures, inspired the
  direction of this project.

## License

MIT. See [LICENSE](LICENSE).
