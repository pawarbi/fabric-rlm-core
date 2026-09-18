# Skills Guide

Skills are Markdown playbooks that give the model task-specific instructions: what to inspect, which mistakes to avoid, and how to check its answer. Some also contain a Python verifier that the runtime can execute against a submission. Loading a playbook does **not** install packages, execute its examples, grant source access, or guarantee that the model follows its procedure.

Start with one skill matched to the task. Add another only when it contributes a needed procedure or a compatible output contract. More skills can increase prompt size and introduce competing instructions; they are not a general quality switch.

## Defaults and Selection

No skills are active by default. `skills=None` becomes an empty selection, `enable_router=False`, and `enable_skill_autoloading=False`. `enable_verifier=True` enables checks for selected skills that actually contain a verifier; it does not select skills.

With the default router policy **and routing enabled**, `core` is the baseline because its specificity is `core`. With routing off, include `"core"` explicitly if you want its instructions. It is not unconditionally active.

Recommended starting selections, not library defaults:

| Task | Explicit selection | Practical boundary |
| --- | --- | --- |
| CSV, JSONL, Parquet, or log exploration | `skills=["data_exploration"]` | Discover the schema before querying; Delta directories require a Delta reader. |
| Read an Excel workbook into records | `skills=["excel_extract"]` | Inspect hidden cells, comments, formats, merged headers, and sheet structure. |
| Change an Excel workbook | `skills=["excel_modify"]` | Use only for authorized edits; verify the saved artifact, not just the final message. |
| Read a PDF with source references | `skills=["pdf_document_analysis"]` | Rendering and page evidence matter when text extraction loses layout. |
| Analyze Lakehouse Delta tables | `skills=["delta_lakehouse"]` | Read-only; prefer a bound `LakehouseSource` for remote OneLake access. |
| Query governed Power BI measures | `skills=["semantic_model"]` | Requires an accessible model and a working Fabric/SemPy or bound-handle setup. |
| Reconcile or rank evidence from several sources | Add `"analytical_integrity"` to the relevant source skill | It gives reasoning guidance, not an independent proof of the numbers. |

For a simple count or lookup, do not start with the two deep-insight skills. They require specialized, substantial payload contracts. Discovery and criticism usually belong in separate phases, with explicit host orchestration and independent source checks.

## Bundled Catalog

The catalog contains all 12 bundled skills. **Dependencies** means other skill names, not Python packages, and lists direct dependencies only. Composition normally includes them transitively: `pdf_document_analysis -> validation -> error_handling`.

**Required verifier** is `yes` only when `SkillLoader.load(name).verifier_present` is true. This means the loader found a Python block under the exact `## Required verifier` heading. It does not certify that the block is correct, sufficient, or successfully executed. A checklist or a differently named verification section does not count.

| Name | Purpose/use | When avoid/caveat | Dependencies | Required verifier |
| --- | --- | --- | --- | --- |
| `analytical_integrity` | Keep rankings, materiality, grain, candidate tuples, provenance, units, periods, and causal language faithful to evidence. | Not a source adapter or automatic source recomputation; distinct from the runtime flag with the same name. | none | no |
| `core` | Define output-field discipline, plan/verify/reflection behavior, and a single computed submission rather than echoed instructions. | Not automatically loaded with routing off; no executable required verifier. | none | no |
| `data_exploration` | Inspect file schemas, then use bounded summaries and load-once/query-many analysis for CSV, JSONL, Parquet, and logs. | Do not use ordinary Parquet globs on Delta tables; DuckDB-specific examples need DuckDB, with Python streaming as a fallback. | none | no |
| `deep_insight_critic` | Adversarially review discovery findings and produce a complete decision-disposition ledger and synthesis manifest. | Not a prose rewrite or discovery rerun; needs the exact source inventory/fingerprint and independent evidence, not invented challenges. | none | yes |
| `deep_insight_discovery` | Discover material KPIs, subgroups, trends, anomalies, and interactions with a candidate ledger and independently recomputed evidence. | Avoid for simple summaries; its versioned contract and portable checks do not replace host execution of source-verification expressions. | none | yes |
| `delta_lakehouse` | Read-only Delta analysis using transaction-log-aware readers, catalog discovery, and bounded queries. | Not for writes, merges, vacuum, or raw Parquet approximation; remote credentials belong in the trusted parent, not worker code. | none | no |
| `error_handling` | Diagnose runtime failures with narrow exceptions, concrete inspection, and targeted recovery. | Do not hide errors with broad catches or retry unchanged code; it supplies no submission verifier. | none | no |
| `excel_extract` | Read structured records from workbooks, preserving information in hidden cells, comments, formats, indentation, and merged regions. | Not for writing back; its generic structural verifier cannot prove extracted values match the workbook. | none | yes |
| `excel_modify` | Compute literal cell values or structural transformations, save an authorized workbook, and reload it for checking. | Not a read-only extraction skill or a formula-calculation engine; benchmark-style in-place instructions must match the user's requested destination. | none | no |
| `pdf_document_analysis` | Analyze long documents page by page, including forms and layout, and synthesize source-grounded findings. | Flattened text alone is insufficient for layout-sensitive evidence; PyMuPDF and any needed image-capable model must be available. | `validation` | no |
| `semantic_model` | Inspect semantic-model metadata, use governed measures, aggregate in DAX, and return small results. | Not a generic CSV reader; model access, authentication, and query context must work in your environment. | none | no |
| `validation` | Build task-specific checks for output keys, types, totals, joins, duplicates, and boundaries before submitting. | Guidance to write checks, not a bundled executable verifier or a source of expected answers. | `error_handling` | no |

The current specificity categories are `core` for `core`; `utility` for `analytical_integrity`, `error_handling`, and `validation`; and `domain` for the remaining eight. These are router metadata, not permission levels. Utility skills are **not** automatically shown as cards merely because they are utilities, and skills without verifiers can still be preloaded or routed.

### Contract-Specific Caveats

- `excel_extract` checks selected structural properties of a payload. Its success does not show that every sheet was inspected or that a value came from the right cell.
- `excel_modify` has a mandatory verification procedure but no loader-extracted required verifier. Check the saved workbook independently, especially the first and last target cells. Do not authorize an overwrite when the task only asks for an explanation.
- `deep_insight_discovery` uses `contract_version: 2` for baseline submissions and version `3` for evidence-closure runs. Its portable verifier checks structure, arithmetic, lineage declarations, and other constraints. The host must separately execute and compare the source-derived verification expressions; a plausible expression is not evidence that it ran.
- `deep_insight_critic` uses `critic_version: 1` and accepts source contract versions `2` and `3`. Its verifier checks inventory coverage, challenges, resolutions, and legal synthesis dispositions. It does not establish the semantic truth of a review or independently authenticate a supplied fingerprint.
- Skill prose includes environment-specific and benchmark-specific examples. Follow the actual task's fields, allowed operations, and source boundary rather than copying placeholder paths or benchmark answer conventions.

## Explicit Use

This Fabric notebook example uses a model name from the repository examples. Confirm that `gpt-5.1` is available to your capacity, or use your already configured LM. The input path must exist and be accessible to the worker.

```python
from fabric_rlm import FabricLM, File, RLM

rlm = RLM.from_task(
    task=(
        "Inspect the orders CSV schema. Count its data records, excluding the "
        "header, and return the column names in source order. Do not print "
        "the entire file."
    ),
    inputs={"orders": File("/lakehouse/default/Files/orders.csv")},
    outputs={"row_count": int, "columns": list},
    lm=FabricLM("gpt-5.1"),
    skills=["data_exploration"],
    enable_router=False,
    enable_skill_autoloading=False,
    max_turns=6,
)
result = rlm.run()
print(result.submitted, result.failure_reason)
print(result.payload)
print(result.trajectory.metadata.get("verifier_execution"))
```

`RLM.task(...)` is also available with the same task/input/output contract. The examples here use `RLM.from_task(...)`. With no `tools=`, the default `engine="auto"` resolves to the default Python-loop engine, also selectable as `engine="default"`.

Explicit selection avoids heuristic auto-election, but it is not a discovery or security allowlist. On the default engine, supplying `skills` also advertises the loader's index even when `enable_skill_autoloading=False`. Worker helpers remain available. If you need a restricted host catalog, configure a loader explicitly and retain the runtime's security/source controls.

`enable_skill_autoloading=True` advertises the available index for on-demand reading. It does not automatically preload all skills or activate every verifier. The library does not search arbitrary project folders for custom skills; provide those folders with `SkillLoader`.

## Inspect and Compose

These host-side operations do not call a model:

```python
from fabric_rlm import SkillLoader, compose_skills, list_skills, load_skill

print(list_skills())                    # Bundled names, sorted.
loader = SkillLoader()
print(loader.list_skills())             # This loader's names, sorted.
skill = loader.load("excel_extract")    # A Skill object, not Markdown.
print(skill.title, skill.summary)
print(skill.dependencies, skill.specificity, skill.verifier_present)
print(skill.applies_when_keywords, skill.applies_when_output_fields)

markdown = load_skill("excel_extract")  # Bundled Markdown string.
text = loader.load_text("pdf_document_analysis", include_dependencies=True)
combined = compose_skills(["validation"], loader=loader)
print(loader.format_index())
```

The exact discovery methods are the top-level `list_skills()` function and `SkillLoader.list_skills()`. There is no `SkillLoader.list()` method. `load_skill(...)` and `loader.load_text(...)` omit dependency bodies unless `include_dependencies=True`; `compose_skills(...)` includes dependencies by default, puts them before dependents, deduplicates, and rejects dependency cycles.

`Skill.summary` comes from the `Summary:` line after frontmatter; multi-line explanatory text remains in `Skill.content`. `verifier_source` contains extracted Python or `None`. These metadata fields describe the playbook, not runtime execution results.

## Load Versus Activate

Inside model-generated worker Python, the built-in `list_skills()` and `load_skill(name)` use a fresh default loader. `load_skill` returns Markdown only. Assigning that string to a variable does not make the model read it; printing the relevant text exposes it in execution feedback. It does not execute the examples or register a verifier.

```python
# Worker code, not a host RLM method.
print(load_skill("validation", include_dependencies=True))
```

The worker also has `activate_skill(name)`. It first loads bundled Markdown, prints an activation marker, and returns the body:

```python
# Worker code for a run intentionally using the router on the default engine.
print(activate_skill("excel_extract"))
```

The default engine processes activation markers **only when `enable_router=True`**. It registers the named skill's verifier if `enable_verifier=True`. With routing off, that marker does not activate a runtime verifier. Dynamic activation does not recursively register dependency verifiers or automatically inject all dependency bodies. Select required dependencies explicitly when their checks must run.

There is no public `rlm.activate_skill(...)` method. Do not import a private worker function into host code as a substitute. Initial routing also exists on the DSPy engine, but the default-engine marker-processing behavior must not be assumed there. `engine="auto"` switches to the DSPy engine when `tools=` is supplied; consult the [API reference](api-reference.md) before relying on engine-specific controls.

### Cards Are Presentation

The argument is **`skills_as_cards`**, and its default is **`False`**. Think of
a card as a short catalog entry, not the playbook itself: the skill name,
summary, and verifier indicator. A card does not summarize every procedure or
output constraint in the full skill.

| Setting | What the model receives initially | How it gets detailed instructions |
| --- | --- | --- |
| `skills_as_cards=False` (default) | The selected skill bodies, normally including dependencies | Already in the initial prompt |
| `skills_as_cards=True` | Short cards for explicitly selected skills instead of their full bodies | Worker code calls `load_skill(name)` and prints the relevant Markdown into feedback |

**Leave it off for most tasks.** Full instructions are usually the better starting
point when you know the skill is needed, especially for strict extraction/output
contracts or short runs. Consider cards when selected playbooks are large and
you want the model to read details on demand rather than preload all the bodies.
Cards can reduce the initial prompt, but loading later costs execution turns and
tokens; total cost, latency, and quality are not guaranteed to improve.

#### How to Use

This example uses a bundled skill that worker-side discovery can find. Supply
your configured `lm` and an accessible CSV file:

```python
from fabric_rlm import File, RLM

rlm = RLM.task(
    task=(
        "Read the data_exploration playbook with load_skill and print its "
        "relevant guidance before analyzing orders. Count the CSV data "
        "records, excluding the header, and return row_count."
    ),
    inputs={"orders": File("/lakehouse/default/Files/orders.csv")},
    outputs={"row_count": int},
    lm=lm,
    engine="default",
    skills=["data_exploration"],
    skills_as_cards=True,
    enable_router=False,
    max_turns=8,
)
result = rlm()
print(result.submitted, result.payload)
result.inspect()  # Check whether the model actually loaded the needed body.
```

The expected worker action is `print(load_skill("data_exploration"))`. Calling
`load_skill` only returns text; the model must expose it in feedback to read it.
Large printed bodies can be truncated by the feedback limit, so load/read the
relevant sections rather than assuming the whole document reached the model.
The task instruction above requests loading; the flag itself does not force it.

#### Important Interactions

On the default engine with routing off, `skills_as_cards=True` replaces explicitly selected full bodies with brief cards in the prompt. It **does not disable validators**: explicitly selected skills still register their required verifiers when `enable_verifier=True`, even if the model never loads the body. A verifier can therefore reject an output whose full contract the model has not read. Prefer full bodies for small, contract-heavy selections.

Router cards are different: they represent scored candidates that were not initially active. Their verifiers are not initially registered merely because the card appears. Activation can register a named verifier later under the engine/router conditions above. When routing is on, the router's active/card decision takes precedence over `skills_as_cards`. The DSPy skill-composition path does not implement the same card-only presentation mode.

Neither card presentation nor `enable_verifier=False` turns off independent output-schema checks or your configured host validators. Do not use cards as a validation or security policy.

- With no explicitly selected `skills`, this flag alone has no card bodies to replace.
- Use `engine="default"` when relying on this behavior. With `engine="auto"`,
  adding host `tools` selects DSPy, which does not implement this presentation mode.
- Leave cards off for host-only custom skills: your `SkillLoader` directories
  are not automatically available to worker `load_skill`. See Custom Skills below.
- On-demand `load_skill` does not include dependency bodies unless requested with
  `include_dependencies=True`. Cards are not a substitute for needed prerequisites.
- Test cards against full preloading on your workload. Inspect loading turns,
  validation errors, answer quality, and actual usage rather than assuming fewer
  initial prompt tokens means a cheaper or more reliable run.

## Optional Routing

Routing is a zero-model-call heuristic, not semantic task classification. Use explicit selection when you know the task. If you choose routing, inspect what it elected:

```python
from fabric_rlm import SkillLoader
from fabric_rlm.skill_router import SkillRouter

router = SkillRouter.from_loader(SkillLoader(), max_active_skills=1)
decision = router.route("Extract structured records from an Excel workbook")
print(decision.active)
print(decision.cards)
print(decision.scores, decision.reasons)
```

Current routing rules:

1. Each case-insensitive keyword substring match adds 2 to a skill's score. Substrings can overlap or match incidental words; no semantic guarantee follows.
2. Output-field matching adds 1, but the current token heuristic recognizes names shaped like `Q1`, `Q23`, or `A1` (pattern `[A-Z][A-Za-z]?\d+`), **not arbitrary output names**. It is not a general match against `outputs=`. Metadata such as `insights`, `reviewed_insights`, or `synthesis_manifest` does not auto-route via that heuristic.
3. Runtime routing uses a bounded textual representation of bound input values first. It falls back to task text only when the first decision has no scores. An incidental input match can suppress task fallback; it does not semantically combine all task evidence.
4. The baseline is all `specificity="core"` skills unless `router_baseline_skills` overrides it. Explicit `skills` are pinned in addition. Positive-score candidates fill the remaining ranked slots, with stable ties based on loader order.
5. `max_active_skills` limits ranked additions relative to the baseline, not the complete transitive active set. Explicit pins can exceed the cap; dependencies are added afterward when enabled.
6. Exclusions influence ranked candidate promotion, not a comprehensive conflict resolver for explicit pins and dependencies. Choose compatible contracts yourself.
7. Cards come from ranked-but-not-active candidates. Utilities with verifiers get card-order preference, but utility classification alone does not create a card. Zero-score unselected utilities such as `validation` and `error_handling` are not automatically cards.

For an optional runtime policy that uses router bookkeeping without electing extra ranked skills, pin the desired skill and make every selection control explicit:

```python
from fabric_rlm import FabricLM, File, RLM

rlm = RLM.from_task(
    "Count data records in the CSV, excluding its header.",
    inputs={"orders": File("/lakehouse/default/Files/orders.csv")},
    outputs={"row_count": int},
    lm=FabricLM("gpt-5.1"),
    engine="default",
    skills=["data_exploration"],
    enable_router=True,
    router_baseline_skills=["core"],
    router_candidate_specificities=[],
    router_include_dependencies=True,
    max_active_skills=0,
    enable_verifier=True,
)
```

This initially selects `core` plus the pinned `data_exploration`, not arbitrary ranked additions. It is still not an activation allowlist: the model can request worker activation of another bundled skill. To permit automatic domain ranking instead, deliberately set `router_candidate_specificities=["domain"]` and a small cap such as `max_active_skills=1`. `router_baseline_skills=[]` disables the automatic baseline, but does not prevent an explicitly selected or dependency skill from loading.

After a default-engine run, inspect `result.trajectory.metadata` keys `router_active`, `router_cards`, and `router_used_task_text_fallback`. These record initial routing; inspect worker stdout and `verifier_execution` for later activation and actual checks. The direct `RouteDecision` above additionally exposes scores and reasons.

## Custom Skills

Start with the [authoring guide](authoring-skills.md) and [skill template](skill-template.md). Use a safe filename such as `csv_row_count.md`: letters, digits, underscores, and hyphens, starting with a letter or digit. Loading by a traversal path is rejected. Keep a one-line `Summary:` near the title, and make the task boundary explicit.

Configure custom directories on the host:

```python
from fabric_rlm import SkillLoader

loader = SkillLoader("/lakehouse/default/Files/skills")
print(loader.list_skills())
custom = loader.load("csv_row_count")
print(custom.verifier_present)
```

Custom directories layer over bundled skills. With several directories, later ones win; a custom file with the same name overrides the packaged file. For a host-only catalog, pass a real custom directory and `include_packaged=False`, for example `SkillLoader("/lakehouse/default/Files/skills", include_packaged=False)`. Passing no custom directories falls back to packaged discovery even with that flag, so do not use `SkillLoader(include_packaged=False)` as an empty catalog.

**Host and worker discovery are separate.** Passing `skill_loader=loader` lets the parent compose custom bodies and execute their extracted verifiers, but the loader object and its directory configuration are not serialized to the worker. Worker `load_skill("csv_row_count")` cannot automatically find that host-only file, even if the directory happens to be mounted. A custom override can also differ from the bundled body returned by worker loading. Preload custom skills with explicit `skills=[...]` and `skills_as_cards=False`; do not depend on worker autoloading for them.

### Minimal CSV Playbook

The following is the complete content of a custom `csv_row_count.md`. Its verifier is optional in the loader format, but included here to demonstrate real rejection behavior. It uses only the standard library, has no skill dependencies, and checks shape and range, **not source correctness**.

````markdown
---
applies_when:
  keywords: ["csv row count"]
  output_fields: []
excludes: []
depends_on: []
specificity: domain
---
# csv_row_count
Summary: Count CSV data records without confusing quoted newlines with rows.
Dependencies: none

## Purpose

Count data records in a UTF-8 CSV file with one header record.

## Contract: output fields

The task supplies `csv_path` and asks: "Count CSV data records, excluding
the header, using csv.reader semantics; blank lines are skipped."
Return `result`, a dictionary with exactly `row_count`, a non-negative
integer. The header is the first non-empty record yielded by csv.reader. An empty
file has zero data records. A quoted multiline field is one record.

## Required verifier

```python
def verify(payload):
    if not isinstance(payload, dict) or set(payload) != {"result"}:
        raise AssertionError("payload must contain exactly result")
    result = payload["result"]
    if not isinstance(result, dict) or set(result) != {"row_count"}:
        raise AssertionError("result must contain exactly row_count")
    count = result["row_count"]
    if type(count) is not int or count < 0:
        raise AssertionError("row_count must be a non-negative integer")
```

Call verify(payload) before SUBMIT. These checks validate structure only;
the count must still be computed from the supplied file.

## Tripwires

- Do not count physical lines; quoted fields can contain newlines.
- Do not count the header as data.
- Do not turn a missing or unreadable file into a claimed zero count.

## Invariants

- The output has exactly result.row_count.
- row_count is an integer, not a Boolean, and is non-negative.

## Procedure

Open csv_path with encoding="utf-8" and newline="". Use
records = (row for row in csv.reader(source) if row), consume the header
with next(records, None), and compute count = sum(1 for row in records). Form
payload = {"result": {"row_count": count}}, run verify(payload),
then call SUBMIT(result=payload["result"]). Do not print the full file.
````

Once that playbook exists in the configured directory, use it with matching outputs:

```python
from fabric_rlm import FabricLM, RLM, SkillLoader

rlm = RLM.from_task(
    "Count CSV data records, excluding the header, using csv.reader semantics; "
    "blank lines are skipped.",
    inputs={"csv_path": "/lakehouse/default/Files/orders.csv"},
    outputs={"result": dict},
    lm=FabricLM("gpt-5.1"),
    skill_loader=SkillLoader("/lakehouse/default/Files/skills"),
    skills=["csv_row_count"],
    skills_as_cards=False,
    enable_router=False,
    enable_skill_autoloading=False,
    enable_verifier=True,
)
```

The required-verifier section must have that exact heading and a Python fence before any subsequent `##` heading. Define `verify(payload)` inside it. The loader extracts source text; it does not validate that a callable exists. Return normally on success and raise `AssertionError` with a precise message on a violated invariant. Returning `False` is not a rejection.

## Validation and Safety

On the default engine, the runtime executes applicable extracted verifiers against the submitted JSON-compatible payload in the worker. With routing off, explicitly selected skills register their verifiers; dependency bodies included through composition do not automatically register dependency verifiers. With routing on, initially active dependencies are included when `router_include_dependencies=True`. A skill without an extracted verifier supplies no check at this layer.

An `AssertionError` rejects the submission and requests repair within the available run budget. Other verifier exceptions, host execution errors, and verifier timeouts are logged and **fail open**: the runtime can accept the payload without that check succeeding. Non-JSON-serializable payloads can also cause skill checks to be skipped. This is graceful degradation, not a correctness guarantee.

Inspect `result.trajectory.metadata.get("verifier_execution")` on the default engine. Its `checks`, `passed`, `rejected`, and `degraded` fields describe actual execution; `verified` means at least one logged check passed and none in that summary rejected or degraded. It does not mean every business claim was proved. Review `verifier_repair_history`, turn errors, and warnings as well as `result.submitted`.

For source correctness, add a trusted `output_validator` or `output_validator_context` that independently recomputes task-specific expectations, then inspect its execution too. Do not treat a structural skill verifier as a substitute. See [verified tasks](verified-task.md) for additional orchestration; independent agreement also does not prove correctness.

The **`analytical_integrity` skill** is Markdown guidance about analytical reasoning. The **`analytical_integrity=` runtime flag** configures a separate runtime mechanism and defaults to `True`. Selecting the skill does not configure the flag; an empty skill selection does not turn off runtime integrity checks. Neither should be described as a universal source-truth validator.

| Do | Don't |
| --- | --- |
| Derive expected totals, populations, periods, and units from actual inputs or independently approved fixtures. | Invent gold answers, thresholds, fingerprints, or rejected candidates to make a check pass. |
| Test valid, invalid, empty, duplicate, malformed, and boundary cases against the stated contract. | Treat one happy-path example or a fluent report as validation. |
| Use source-independent recomputation where possible, and document what each check cannot establish. | Verify a model-supplied number by selecting the same number as a SQL constant. |
| Raise `AssertionError` deliberately after checking types and required keys. | Accidentally raise `KeyError`, `TypeError`, or `NotImplementedError` and assume the runtime rejected the answer. |
| Install required packages in the trusted notebook/environment before starting a run. | Ask generated worker code to execute pip through `subprocess`, `os.system`, or dynamic `exec`. |
| Keep remote credentials in trusted source adapters and restrict the allowed data scope. | Put tokens in skill Markdown or weaken security policy to make an example run. |
| Verify persisted artifacts and publication manifests before claiming success. | Assume a skill's instruction to save or publish proves that the operation happened. |

Skill dependencies are not package installers. For local file analytics, the repository provides the `analytics` extra; PDF work uses the `pdf` extra/PyMuPDF; Excel examples require `openpyxl`. Install what your selected task actually needs before running it. Fabric may already supply some packages; probe once and report an unavailable capability rather than fabricating a result.

The worker is a CPython subprocess with policy checks, **not an OS security sandbox**. Markdown is not a security boundary, and custom verifier code is executable code. Review custom playbooks and verifiers as trusted project inputs; do not run arbitrary downloaded verification blocks or bypass source/security controls.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| A custom skill is missing | Use `loader.list_skills()` on the host; check the exact filename, directory, and precedence. Worker discovery does not inherit that loader. |
| Loading a skill did not enable checking | Loading returns text. Use explicit selection, or supported router activation on the default engine, and confirm `verifier_present`. |
| An unexpected verifier rejects an answer | Check pinned skills, router decisions, and output-contract compatibility. Explicit cards still register verifiers. |
| An expected domain skill was not routed | Inspect keyword matches and task fallback; arbitrary `outputs=` names are not recognized by the field-token heuristic. Pin the skill instead. |
| A "verified" output has wrong numbers | Identify exactly which check passed. Structural validity, declared lineage, and source correctness are different claims. |
| A broken verifier did not reject the output | Look for fail-open errors/timeouts and `degraded` checks; use explicit assertion failures for contract violations. |
| A library or source call fails | Confirm host installation, worker availability, credentials, and source scope. Do not attempt a subprocess install or bypass the broker. |

## Validation Status

This guide is checked against the current loader, router, runtime, and all 12 bundled playbooks. The targeted suite passed **112 tests**, including catalog completeness, dependency/verifier metadata, the executable template, real CSV rejection/repair, multiline CSV counting, defaults, and load-versus-activate behavior. Live Fabric model calls, remote source access, and end-to-end analytical quality are not established by these local scripted tests.

Reproduce the targeted selection:

```text
python -m pytest tests/test_skills_guide.py tests/test_skill_lifecycle.py tests/test_api_arguments_skills.py tests/test_skill_frontmatter.py tests/test_skill_loader_layering.py tests/test_skill_router.py tests/test_runtime_router_integration.py tests/test_runtime_budget.py tests/test_api_reference.py
```

These tests verify instructions and runtime mechanics, not that loading a skill makes every answer correct. Only three bundled skills currently contain a required verifier, and their acceptance coverage is limited as documented above.

The full offline regression suite also passed: **3,692 passed, 13 skipped** (54 warnings). Provider credentials were removed from the test process; no paid model calls were made. Skipped environment/provider cases are not verified passes.

## References

- [Authoring skills](authoring-skills.md) and [starter template](skill-template.md).
- [API reference](api-reference.md), [argument test matrix](api-argument-tests.md), and [verified tasks](verified-task.md).
- [Skill loader](../fabric_rlm/skill_loader.py), [skill router](../fabric_rlm/skill_router.py), [runtime](../fabric_rlm/runtime.py), and [worker helpers](../fabric_rlm/_worker.py).
- [Bundled playbook sources](../fabric_rlm/skills/) and [contributed skills](contrib-skills.md).
- [Frontmatter tests](../tests/test_skill_frontmatter.py), [loader layering tests](../tests/test_skill_loader_layering.py), [router tests](../tests/test_skill_router.py), [runtime verifier tests](../tests/test_runtime_verifier.py), and [skill argument tests](../tests/test_api_arguments_skills.py).
- [Security model](../SECURITY.md) and [package dependencies](../pyproject.toml).
