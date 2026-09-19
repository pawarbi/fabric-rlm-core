# API Reference

This reference describes the checked-in [runtime](../fabric_rlm/runtime.py) and
[verified-task implementation](../fabric_rlm/verify.py), not a promise that every
engine implements every control identically. Defaults below are signature
defaults; a `None` default may select behavior during construction.

## Start here

For most tasks, use `RLM.task(...).run()` with a clear `task`, a configured `lm`,
named `inputs`, and explicit `outputs`. Leave engine, routing, adaptive, security,
and integrity settings at their defaults. Start with the default `max_turns=20`;
try a smaller budget such as `8` for a simple count, then inspect failures before
raising it. Neither setting is a total cost or end-to-end time limit.

This ordinary-task example assumes `lm` is already configured for your provider:

```python
from fabric_rlm import RLM

def validate_count(payload):
    if payload["answer"] < 0:
        raise AssertionError("Row count must be nonnegative")

result = RLM.task(
    "Count the records in rows. Return the count as answer.",
    lm=lm,
    inputs={"rows": [{"order_id": 1}, {"order_id": 2}]},
    outputs={"answer": int},
    max_turns=8,
    output_validator=validate_count,
).run()
if not result.submitted or result.failure_reason is not None:
    raise RuntimeError(result.failure_reason or "No submitted answer")
validate_count(result.payload)  # Check again at the application boundary.
print(result.payload["answer"])
```

Add `output_validator` for a business rule (the example checks range, not the
correct count); use `output_validator_context` when the rule needs original
inputs. Reject with `AssertionError`, not `return False`. Leave these hooks unset
when field/type checks suffice, but independently check correctness before acting
on an answer. Keep built-in verifier and integrity checks enabled.

**Advanced, not routine setup:** skill routing and prompt digests are for tuning
skill-heavy workflows; adaptive controls are experimental multi-attempt policy
work. Do not enable them just to get started. Use `verified_task` only when a
read-only, determinate answer merits two or three solves and the extra cost.

**Validation status:** the combined argument-level selection passed 169 tests; the
[API argument test coverage](api-argument-tests.md) will record coverage and gaps.
This source-based reference does not claim every argument/engine combination is
tested. Live authenticated Fabric execution has not been tested; provider/model
availability and authentication are separate from local runtime argument tests.

## RLM.task and RLM

`RLM.task(...)` and `RLM.from_task(...)` have the same contract and return an
**unexecuted** `RLM`. Call `.run()` to get an `RLMResult`. Both forward `**kwargs`
to the constructor; unknown keywords are errors, not LM-provider options.
There are **37 unique named parameters** below: 34 constructor parameters plus
the task-only `task`, `inputs`, and `outputs`. `knowledge` is shared, not counted
twice. `self`, `cls`, and variadic forwarding containers are not parameters in
this inventory.

```python
RLM.task(task, inputs=None, outputs=None, *, knowledge=None, **kwargs)
RLM.from_task(task, inputs=None, outputs=None, *, knowledge=None, **kwargs)

RLM(
    signature=None, *, lm, sub_lm=None, max_turns=20, timeout=300.0,
    verbose=False, skills=None, enable_skill_autoloading=False,
    skill_loader=None, enable_verifier=True, enable_router=False,
    max_active_skills=2, router_baseline_skills=None,
    router_candidate_specificities=None, router_include_dependencies=True,
    reserve_finalize_turns=0, recover_worker_timeouts=1, skills_as_cards=False,
    block_network=False, max_prompt_tokens=None, digest_after_turn=None,
    output_validator=None, output_validator_context=None,
    analytical_integrity=True, halve_max_iter_on_retry=True, engine="auto",
    adaptive=None, inner_engine="v6-custom", stuck_loop_threshold=3,
    tools=None, security=None, max_submit_bytes=67108864,
    knowledge=None, capture_evidence=False,
)
```

In the tables, **default engine** means `"default"` / `"v6-custom"`;
**DSPy engine** means `"dspy"` / `"v7-dspy"`. Unless narrowed explicitly, controls
apply to both ordinary engines. Adaptive attempts inherit ordinary-engine
limitations; see the experimental scope below.

### Task and Data

| Parameter | Default | Description | When to use / leave default |
| --- | --- | --- | --- |
| `task` | required | Task text (`str`), only on `task`/`from_task`. With no explicit signature it supplies the instructions; with a signature, the default engine appends it to signature instructions. DSPy prioritizes an explicit signature instead. | Always state the operation, input alias, and expected answer, e.g. "Count rows in source; return answer." There is no default task. |
| `inputs` | `None` | Task-factory-only dictionary of named worker inputs; copied to an initially empty binding dictionary. `.run(inputs={...})` merges overrides by name; `rlm(**inputs)` also runs. Use serializable values or supported handles such as `File`, `LakehouseSource`, and `SemanticModel`, not arbitrary live notebook objects. Names such as `source` and `source_frames` are application-chosen input aliases, not top-level runtime keywords. | Pass data explicitly, e.g. `{"source": File("/lakehouse/default/Files/orders.csv")}`. Leave unset only for tasks needing no bound data or bind it at `.run()`. |
| `outputs` | `None` | Task-factory-only list of field names or mapping of names to concrete Python classes. Names must be nonempty strings. `None` becomes no inline declarations (DSPy synthesizes `output` when needed). With an explicit signature, its output names take precedence; a typed mapping must match those names exactly as a set and count. See Output Contracts below for exact checks, serialization, and limitations. | Prefer `{"answer": int}` for a count or `{"answer": str}` for text. Omit only when a signature already declares the contract or undeclared output is intentional. |
| `signature` | `None` | Optional constructor positional argument: typically a DSPy Signature class or `"inputs -> outputs"` string. May also be forwarded through a task factory. Supplies output names and instructions; string annotations alone are not the typed-mapping runtime contract. Direct construction without one has no declared task/output contract on the default engine; DSPy requires a signature or task-factory state. | Leave unset with ordinary `RLM.task` calls. Supply one to reuse an established DSPy contract; keep task text and output declarations consistent with it. |
| `knowledge` | `None` | A `Knowledge` instance, not a path/dictionary. Binds learned source aliases, preflights source status/drift, retrieves active lessons, and can select bounded registered host operations. Explicit input alias collisions raise. Learning does not guarantee factual correctness or remove raw-source fallback. Adaptive knowledge/registered-operation parity is not guaranteed. | Pass an existing learned `Knowledge` instance for repeated source-aware analysis. Leave unset for one-off work with explicit inputs; avoid rebinding its aliases in `inputs`. |

### Models and Engines

| Parameter | Default | Description | When to use / leave default |
| --- | --- | --- | --- |
| `lm` | required | Required keyword-only outer LM: callable (including `FabricLM`/DSPy LM), model string, or DSPy/LiteLLM configuration dictionary. Strings use registered backend factories when matched, otherwise `dspy.LM`; dictionaries are resolved as DSPy configuration. `None` is not a usable LM. Provider credentials and generation settings belong here, not in RLM kwargs. | Always supply your configured provider LM. Set model effort, generation limits, and HTTP timeout on that LM, not on `RLM.task`. |
| `sub_lm` | `None` | Optional subordinate LM. Default engine sends its spec through worker JSON configuration, so use a worker-resolvable string or serializable dictionary, not a live LM object. When omitted, a string/dictionary `lm` is reused implicitly; a callable `lm` does not configure default-engine worker prediction helpers. DSPy resolves an explicit spec or falls back to the outer LM. Any explicit non-`None` value conflicts with `block_network=True` at construction. | Leave unset for ordinary Python/data tasks. Configure it only when worker prediction helpers need a model or should use a different one; verify worker credentials and network access separately. |
| `engine` | `"auto"` | Allowed: `"auto"`, `"default"`, `"dspy"`, `"adaptive"`, `"v6-custom"`, `"v7-dspy"`. Auto chooses DSPy for a nonempty materialized `tools` iterable, otherwise the default engine. Canonical `v6-custom`/`v7-dspy` spellings still work but emit `DeprecationWarning`; prefer public aliases. Adaptive emits an experimental warning. | Leave `"auto"` for normal use, including host tools. Pin `"default"` or `"dspy"` for reproducible engine-specific behavior; choose `"adaptive"` only for an intentional experiment. |
| `tools` | `None` | Iterable of host callables, including callable `dspy.Tool` objects; snapshotted once, so generators are consumed. Noncallables raise `TypeError`; nonempty tools with default/adaptive engines raise `NotImplementedError`. Names normally come from `__name__`; use `dspy.Tool(..., name=...)` for overrides. Host tool execution is not isolated by worker security. | Leave unset for worker-only analysis. Pass a narrow, trusted callable such as `tools=[lookup_order]` when host access is necessary; enforce authorization and side-effect limits inside the tool. |
| `adaptive` | `None` | Experimental configuration dictionary used only by `engine="adaptive"`; otherwise ignored. Selects policy, budget, validation, and optional hooks for multiple attempts. It is not a stable, exhaustively supported nested API; see Experimental Adaptive below. | Leave unset for normal use. Configure only when evaluating a multi-attempt policy with explicit acceptance criteria and budgets; first check current nested-option wiring. |
| `inner_engine` | `"v6-custom"` | Adaptive-only ordinary engine: `"v6-custom"`, `"v7-dspy"`, or aliases `"default"`, `"dspy"`. Neither `"auto"` nor `"adaptive"` is allowed here. Outside adaptive it is ignored, not validated, and replaced internally by the resolved outer engine. | Leave default unless an adaptive experiment specifically needs DSPy attempts (`"dspy"`). It is not how ordinary users select an engine; use `engine` instead. |

### Execution and Budgets

| Parameter | Default | Description | When to use / leave default |
| --- | --- | --- | --- |
| `max_turns` | `20` | Use a positive integer. Default-engine loop budget includes repair/truncation turns; not a global LM-call, token, or wall-time limit. DSPy uses it as initial `max_iterations`, with up to two wrapper retries and an extraction path, so total work can exceed it. Adaptive uses it as the base attempt budget. | Start at `20`, or try `8` for a simple count. Raise only when the trajectory shows useful unfinished work; fix repeated errors rather than buying more turns. |
| `timeout` | `300.0` | Worker-response timeout in seconds; use a positive number. Applies to execution/interpreter waits, with separate startup/control-plane allowances. Does not interrupt a blocked outer LM request or impose an end-to-end task deadline. Configure the LM's own HTTP `timeout` separately. | Leave default for ordinary execution. Increase for a known slow query, or reduce for fail-fast worker waits; changing this will not fix a hung provider request. |
| `reserve_finalize_turns` | `0` | Default engine only. Coerced with `max(0, int(value))`; when remaining turns reach this threshold, adds a submit-now prompt hint. Does not reserve extra turns, force submission, or guarantee finalization. | Leave `0` unless runs repeatedly compute an answer but exhaust turns before submitting. Try `2` to request earlier finalization, then inspect whether submissions improve. |
| `recover_worker_timeouts` | `1` | Default engine only. Coerced to a nonnegative integer. Maximum worker restart attempts after execution timeout or protocol/process death; original inputs are rebound but computed namespace state is lost. Recovery consumes loop budget and is not guaranteed to succeed. Not an LM retry setting. | Leave one recovery for read-only analysis. Set `0` when restart/re-execution is undesirable, especially around side effects; do not increase it to mask consistently slow code. |
| `stuck_loop_threshold` | `3` | Default-engine circuit breaker for consecutive failures with identical normalized code/error signatures. Integer at least `2`, or `None` to disable; booleans rejected. Returns failure reason `"stuck_loop"`. Validated for all engines, but no equivalent DSPy loop check is wired. | Keep `3` to stop unproductive repetition. Try `2` for faster failure; raise or disable only after confirming legitimate repeated failures are being stopped prematurely. |
| `halve_max_iter_on_retry` | `True` | DSPy wrapper only. On missing/invalid output or verifier rejection, reduce the next attempt to `max(1, (max_iter + 1) // 2)` iterations. `False` keeps the previous limit; neither choice creates a shared total budget. | Leave enabled to shorten DSPy repairs. Set `False` only when repairs genuinely need the full iteration allowance and the additional work is acceptable. |
| `max_prompt_tokens` | `None` | Default engine with router enabled only. If the **system message** estimate exceeds this threshold, attempts to replace active skill bodies with digests. Estimate is character-based, not provider tokenization. Not a whole-prompt/context cap, output cap, or hard total-token/spend budget; ignored without the router and by DSPy. | Leave unset for normal tasks. Tune only after measuring oversized router skill prompts; use provider/orchestration controls, not this knob, for hard budgets. |
| `digest_after_turn` | `None` | Default engine with router enabled only. Optional integer loop threshold: digest active skill bodies before a call once the turn counter reaches it. Actual code uses the run counter, not a separate lifetime per skill. Off by default; rewriting the system prefix can lose provider prompt-cache reuse and increase cost despite fewer tokens. | Leave unset unless long router runs justify a digest experiment. Compare answer quality and actual billed/cache usage before keeping a threshold such as `5`. |

### Skills and Routing

Explicit skills can help a known domain task. The discovery, routing, and digest
controls are advanced tuning, not a checklist of flags every user should enable.
See the [skills guide](skills-guide.md) for skill loading, presentation, and routing.

| Parameter | Default | Description | When to use / leave default |
| --- | --- | --- | --- |
| `skills` | `None` | List of installed/custom skill names; copied to an empty list when omitted. Without routing, composes their bodies with dependencies (unless default-engine cards are requested). With routing, pins these skills alongside the baseline. Pinning is not constrained by the ranked-candidate cap. Verifier availability is separate from instructional text. | Leave unset for a simple task. Add known installed skill names when their domain instructions/checks are needed; prefer a small explicit set before trying routing. |
| `skill_loader` | `None` | Host-side `SkillLoader`; otherwise constructs `SkillLoader()` using bundled skills. Controls host composition, routing, and verifier loading. A custom loader instance is not serialized to the worker's module-level `load_skill` helper, so custom host skill discovery does not guarantee matching worker discovery. | Leave default for bundled skills. Supply a loader for your custom skill collection, and separately verify worker discovery if the model must fetch custom bodies. |
| `enable_skill_autoloading` | `False` | Default-engine prompt discovery flag: advertises the skill index even without explicit skills. Does not automatically elect/load all skills or gate worker helper availability. An index is also shown when skills are explicit or routing is on. DSPy records the flag but does not implement this index/autoload behavior. | Leave off when you know the required skills or need none. Enable for default-engine tasks that should discover skills from the index without preselecting them. |
| `skills_as_cards` | `False` | Default engine, router disabled, explicit skills only: show short cards instead of preloading bodies; the model can fetch bodies with `load_skill`. Does not switch off already configured skill verifiers. Ignored by DSPy's skill composition and by router-controlled composition. | Leave off to provide full instructions immediately. Try `True` when explicit skill bodies are large and on-demand loading is worth extra turns; check that needed bodies are fetched. |
| `enable_router` | `False` | Enables zero-LM-cost skill selection by keywords and text tokens such as `Q1`/`A1` in both engines, not general matching against arbitrary `outputs` schema field names. Scores bound input text first, falling back to task text on zero signal. Default engine exposes cards and processes `activate_skill` signals; DSPy composes initially elected bodies, not the same dynamic card workflow. | Leave off for ordinary tasks and fixed skill sets. Enable for a varied workload with many installed skills, then inspect selections for missed or irrelevant skills. |
| `max_active_skills` | `2` | Router-only ranked-candidate allowance, coerced to a nonnegative integer. Baseline skills are additional; explicit pins outside the baseline consume allowance and reduce remaining ranked slots, but pins and dependency closure can exceed the allowance. Not a hard cap on all active skills or prompt size. | Keep `2` when first evaluating routing. Lower it for noisy candidates or raise it for tasks demonstrably needing more ranked skills; it will not shrink pinned/baseline skills. |
| `router_baseline_skills` | `None` | Router-only list of always-on names. `None` selects all skills with specificity `"core"`; `[]` explicitly selects no baseline. Unknown baseline names are skipped. | Keep the core baseline initially. Supply known names for a curated always-on set, or `[]` only when intentionally removing baseline instructions. |
| `router_candidate_specificities` | `None` | Router-only specificity filter on automatically ranked candidates. `None` means no filter; use values from `("core", "domain", "utility")`; `[]` elects no ranked candidates. Does not filter explicit pins, baseline, or dependency closure. | Leave unfiltered unless a workload needs a narrower candidate pool, e.g. `["domain"]`. Use `[]` to retain only baseline/pinned skills and their configured dependencies. |
| `router_include_dependencies` | `True` | Router-only boolean: adds transitive prerequisites of active skills outside the candidate allowance. `False` skips this expansion. Does not change non-router skill composition, which includes dependencies. | Keep enabled so selected skills receive prerequisites. Disable only for a controlled experiment or a curated set that already supplies everything required. |

### Validation and Safety

| Parameter | Default | Description | When to use / leave default |
| --- | --- | --- | --- |
| `enable_verifier` | `True` | Enables applicable loaded/activated skill verifiers, not all bundled verifiers. `False` does not disable required-field/type checks, either output-validator hook, or analytical integrity. Skill verifier assertions reject; crashes/timeouts may be logged and skipped. | Keep enabled. Disable only to diagnose a specific skill verifier or compare behavior in a controlled evaluation; inspect verifier execution metadata rather than assuming checks ran. |
| `output_validator` | `None` | Host callable `(payload) -> None`, after structural/skill checks. **Raise `AssertionError` to reject** and request repair. Return values, including `False`, are ignored; other `Exception` types are logged and fail open. Runs in both engines and every verified-task solve, subject to earlier checks passing. | Add for payload-only rules such as nonnegative counts or required report sections. Leave unset if structural checks suffice; use explicit `raise AssertionError(...)` for rejection. |
| `output_validator_context` | `None` | Host callable `(payload, context) -> None`, after payload validator passes. Same assertion-only rejection/fail-open contract. Default-engine context keys: `inputs`, `state`, `turn`, `trajectory`; DSPy keys: `inputs`, `attempt`, `prediction`, `trajectory`. State is a serialized snapshot, not live worker objects; DSPy trajectory is not yet populated with the current prediction's events at this hook. No built-in `source`/`source_frames` context keys: access your aliases through `context["inputs"]`. | Use when checking against original inputs, e.g. compare a count with `len(context["inputs"]["rows"])` for a supplied list. Leave unset for payload-only rules; branch explicitly for engine-specific context. |
| `analytical_integrity` | `True` | `True`/`"repair"` runs heuristic checks with at most two rejections, then may accept unresolved findings; `"strict"` keeps rejecting detected issues within the engine's retry budget; `False`/`"off"` disables. Normalization also accepts `None` as repair and case-insensitive boolean-like strings. `FABRIC_RLM_ANALYTICAL_INTEGRITY` can disable or force strict. Screens do not prove correctness, may fail open on internal exceptions, and have less trace evidence during DSPy validation. | Keep enabled; use `"strict"` when detected unresolved issues should keep triggering repair even at the risk of no answer. Disable only for an understood false positive or controlled comparison. |
| `security` | `None` | `SecurityPolicy` instance; `None` selects **`SecurityPolicy.default()`**, not disabled protection. Default-on AST restrictions and worker environment scrubbing; `SecurityPolicy.disabled()` opts out. Policy collection fields are tuples; violation modes are `"feedback"`/`"raise"`. This is defense in depth, not an OS sandbox, filesystem isolation, or a guarantee that an exception escapes every engine wrapper. Trusted host tools/validators remain privileged. | Leave `None` to retain protection. Customize a policy for a reviewed restriction/allowance; do not disable it merely to make generated code pass. Use OS isolation for stronger containment. |
| `block_network` | `False` | Opt-in Python socket-connect guard in the default execution worker and DSPy skill-verifier worker. The current DSPy execution interpreter does **not** receive this flag. Loopback remains allowed; parent LM calls/host tools are outside it. Explicit `sub_lm` plus `True` raises even on DSPy; implicitly inherited specs do not raise but default-worker remote calls can fail. Not an OS firewall: native extensions/direct low-level sockets and already-local data are not contained. | Enable as an extra guard for default-engine analysis of already-local data that needs no remote worker calls. Leave default when remote source/helper access is required; never treat it as task-wide network isolation. |
| `max_submit_bytes` | `67108864` | Signature uses `DEFAULT_MAX_SUBMIT_BYTES = 64 * 1024 * 1024` (64 MiB). Positive integer, booleans rejected. Limits final UTF-8 JSON payload bytes in both workers. Oversize submission fails explicitly rather than truncating. Does not cap intermediate output, memory use, or guarantee arbitrary Python objects survive serialization. | Leave default for small answers. Lower it for an application's response-size boundary, e.g. `1048576` for 1 MiB; prefer summaries over raising it to return entire datasets. |

### Observability

| Parameter | Default | Description | When to use / leave default |
| --- | --- | --- | --- |
| `verbose` | `False` | Prints default-engine turn/repair diagnostics. Not a full audit policy, secret-redaction setting, or equivalent DSPy verbosity control; use result trajectories for inspection. | Turn on while diagnosing default-engine failures or slow progress. Leave off for routine runs and review diagnostic output before sharing logs. |
| `capture_evidence` | `False` | Best-effort post-run harvesting of typed evidence from recorded telemetry into `result.evidence`. Observational: does not change prompts/answers, add validation, or prove provenance completeness. Harvest errors are logged without failing the result; absent engine telemetry cannot be reconstructed by this flag. | Enable when downstream inspection needs typed evidence records. Leave off if you only consume the answer; do not enable it expecting stronger validation or complete provenance. |

### Output Contracts

Both ordinary engines validate declared fields: missing values, `None`, blank
strings, and blank bytes fail. Empty containers fail for names `output`, `answer`,
`result`, and `report` (case-insensitive); specific fields such as `items` may
legitimately be empty. Name-only outputs do not enforce types or business rules.

Typed mappings accept concrete classes, not parameterized generics such as
`list[str]` or union annotations. For `bool`, `bytes`, `dict`, `float`, `int`,
`list`, `set`, `str`, and `tuple`, checking is **exact type identity**, not
`isinstance`: `True` fails an `int` contract, `1` fails a `float` contract, and
subclasses fail a built-in contract. Other classes use `isinstance`.
Checks apply to the received payload: serialization can turn tuples/sets into
lists, bytes into hex strings, and custom objects into markers. Declaring a class
does not add transport support or recursively validate container members.

Assertion rejection produces repair feedback, not guaranteed repair. Inspect
`result.submitted`, `result.failure_reason`, `result.payload`, and
`result.trajectory.metadata` (including `verifier_execution` and unresolved
integrity findings). A configured verifier is not evidence that it ran. For a
fail-closed application boundary, independently validate the returned result
before publishing or taking side effects.

### Experimental Adaptive

The nested `adaptive` dictionary is separate from the 37-name public inventory.
Current wiring recognizes policy/budget objects and options such as `strong_lm`,
`parallel_rollouts`, `feedback_injection`, `skip_more_turns_when_submitted`,
`max_attempts`, `max_total_turns`, `max_parallel`, `max_wall_seconds`, `validator`,
`on_attempt`, and optional task-classifier settings. This is **not an exhaustive
supported nested-parameter reference**; consult [`_run_adaptive`](../fabric_rlm/runtime.py),
[`adaptive_policy.py`](../fabric_rlm/experimental/adaptive_policy.py), and
[`adaptive_runner.py`](../fabric_rlm/experimental/adaptive_runner.py) for current
defaults and enforcement. Unknown dictionary keys need not raise. Without an
explicit adaptive validator, acceptance is structural, not semantic. Multiple
attempts may repeat side effects; wall/time/turn accounting is not a universal
preemptive LM-request or spend limit.

## verified_task

For read-only, determinate analytical answers. Unlike `RLM.task`, it
**executes immediately** and returns `VerifiedResult`.

```python
verified_task(
    task, *, outputs, inputs=None, field_name=None,
    reconcile_guidance=_RECONCILE_GUIDANCE, reconcile_lm=None,
    reconcile_max_turns=None, agree=None, **rlm_kwargs,
)
```

| Parameter | Default | Description | When to use / leave default |
| --- | --- | --- | --- |
| `task` | required | Original task text, used unchanged for both initial solves. Third solve receives this text plus reconciliation guidance and both selected candidate answer strings, not full traces, rejected payloads, or private reasoning. | Ask a read-only question with a determinate answer, e.g. a row count over a fixed source. Use ordinary `RLM.task` for creative prose or mutation. |
| `outputs` | required | Nonempty list of names or typed mapping, retained for every solve's output validation. Same runtime type/serialization rules as above. Empty outputs raise `ValueError` before solving. | Always declare a compact answer contract, e.g. `{"answer": int}`. Keep explanations in a separate field so prose differences do not dominate answer comparison. |
| `inputs` | `None` | Shared input configuration forwarded to each fresh RLM. Fresh solve contexts do not mean immutable source snapshots, independent provider caches, or deep-copied host objects. Avoid side-effecting tasks/tools because execution repeats. | Supply the same fixed/read-only data for all solves; snapshot changing sources outside the wrapper. Leave unset only when the task needs no bound data. |
| `field_name` | `None` | Selects the single output field compared; omission uses the first declared name in list/mapping order. An undeclared name raises `ValueError`. Values are converted to strings (`None` becomes `""`); other output fields are not compared. | Leave unset for a single answer field. Set `field_name="answer"` for multi-field outputs so ordering changes cannot accidentally compare an explanation instead. |
| `reconcile_guidance` | `_RECONCILE_GUIDANCE` | Default multiline text in [verify.py](../fabric_rlm/verify.py): re-derive from data, find the divergence, execute/print at least one check, answer first, and explain the losing answer briefly. Supply a string to replace it (`""` removes it). These are prompt instructions, not an independently enforced executed-check requirement. | Keep default initially. Replace it when reconciliation needs a domain-specific check, such as recounting distinct IDs; preserve the output contract and enforce critical rules with validators. |
| `reconcile_lm` | `None` | Override `lm` for the third solve only; same supported LM forms as RLM. `None` inherits the initial LM unchanged. Does not override an explicit `sub_lm`, ensure a stronger model, or bypass engine-specific LM constraints. | Leave unset for the same-model baseline. Supply a separately configured stronger LM when difficult disagreements justify its cost; verify provider availability independently. |
| `reconcile_max_turns` | `None` | Override `max_turns` for the third solve only. If supplied, must be a positive integer, not `bool`, else `ValueError`. `None` inherits the initial setting, including runtime default `20`. Budgets apply per solve, not across the ensemble. | Leave inherited unless re-derivation needs more room. For example, pair initial `max_turns=8` with `reconcile_max_turns=12`; account for two initial solves plus a possible third. |
| `agree` | `None` | Callable `(a: str, b: str) -> bool`; `None` (or another falsy value) selects `answers_agree`. Called only when both initial selected answers are usable. Default comparator checks signed number strings, normalized equality, list item sets, then whole-word containment/token overlap. It is heuristic agreement, not numerical-tolerance or semantic proof; callback exceptions propagate. | Leave default only if heuristic string agreement suits the answer. Use `lambda a, b: a.strip() == b.strip()` for exact text, or a guarded numeric parser with a domain tolerance for numbers. |

`**rlm_kwargs` forwards constructor options such as `lm` (still required),
`engine`, `skills`, `knowledge`, validators, and `max_turns` through
`RLM.from_task` for every solve. Only the two reconciliation overrides above
replace settings for the third solve. No separate trigger-policy keyword exists.

### Selection and Failure

1. Run A and B in fresh contexts. An answer is usable only if `submitted` is true, `failure_reason is None`, and its selected-field string is nonblank.
2. If both are usable and agree, skip reconciliation and return verdict `"agree"`. Prefer the candidate with clean `integrity_ok`; otherwise choose A.
3. Disagreement **or any unavailable initial answer** triggers C. `reconciliation_reason` is respectively `"disagreement"` or `"unavailable_answer"`; on agreement it is `None`.
4. Select usable C with verdict `"reconciled"`. If C is unusable, prefer a usable initial candidate, then clean integrity, then A. This fallback retains `"reconciled"` for compatibility; it does not mean reconciliation succeeded.
5. If none is usable, return C with verdict `"failed"`, preserving its failure reason.

These steps describe returned attempts. `verified_task` does not catch provider,
construction, or comparator exceptions: an exception that escapes RLM bubbles
out and can prevent later solves or a `VerifiedResult`. Default-engine outer-LM
errors can escape; DSPy catches errors inside its execution wrapper and may
instead return an unsubmitted result, which triggers the unavailable-answer path.

`VerifiedResult.result` is the selected `RLMResult`; `answer_a`/`answer_b` are the
initial selected-field strings. `attempts` retains two or three results.
`selected_attempt_index` is zero-based (or `None` for a manually constructed
result not found by identity). `reconciliation_attempted` means three attempts;
`reconciliation_succeeded` means verdict `"reconciled"` with index `2`;
`fallback_used` means three attempts but index `0` or `1`. `ok` means verdict is
not `"failed"`, **not verified correctness**. `total_prompt_tokens` and
`total_completion_tokens` sum attempt totals, treating missing counts as zero;
these are reported usage, not a cost cap.

Agreement can reproduce a shared mistake; a usable reconciler is not checked
against either candidate for correctness. Avoid prose-generation tasks and
questions with many valid answers if structural comparison has no useful signal.
Use ordinary execution for file publication or mutation, not this wrapper.

## Examples and Models

Illustrative Fabric example, not a live-tested result. It requires an authenticated
Fabric notebook, available models, and the source file already in place:

```python
from fabric_rlm import FabricLM, File, RLM, verified_task

inputs = {"source": File("/lakehouse/default/Files/orders.csv")}
lm = FabricLM("gpt-5-mini", reasoning_effort="medium", timeout=600)
task = "Count all data rows in source. Print the count; return it as answer."

result = RLM.task(task, inputs=inputs, outputs={"answer": int}, lm=lm).run()
if result.submitted:
    print(result.payload["answer"])

vr = verified_task(
    task, inputs=inputs, outputs={"answer": int}, lm=lm,
    max_turns=8, reconcile_lm=FabricLM("gpt-5.1"), reconcile_max_turns=12,
)
print(vr.verdict, vr.reconciliation_succeeded, vr.fallback_used)
if vr.ok:
    print(vr.result.payload["answer"])
```

`FabricLM(model, **kwargs)` uses Fabric service discovery and notebook-identity
authentication, with a one-time token-refresh retry for recognized auth-expiry
errors. RLM runtime knobs control execution; `reasoning_effort`, `temperature`,
`max_tokens`, and provider HTTP `timeout` belong to `FabricLM` or an LM spec.
Passing `reasoning_effort` directly to `RLM.task` is an unknown-constructor-keyword
error. In [lm.py](../fabric_rlm/lm.py), recognized reasoning models default to
`reasoning_effort="medium"`, `max_tokens=16000`, HTTP `timeout=600`, and no
temperature; other models default to `temperature=1.0`, `max_tokens=16000`, and
`timeout=600`. Reasoning tokens share the provider completion budget. Allowed
effort values and generation limits vary by model/provider; FabricLM drops
non-`1.0`/non-`None` temperature overrides for recognized reasoning models.

Microsoft's [Fabric model documentation](https://learn.microsoft.com/en-us/fabric/data-science/ai-services/ai-services-overview#consumption-rate-for-openai-language-models)
lists `gpt-5.1` and `gpt-5-mini` when checked on 2026-09-18. Availability depends
on current Fabric region/capacity configuration. These names and authentication
behavior were checked against documentation/source, **not a live authenticated
Fabric trial**. See also the [public quick start](../README.md#quick-start-in-fabric)
and [reconciliation guide](verified-task.md).
