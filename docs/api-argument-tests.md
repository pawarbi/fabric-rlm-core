# API Argument Test Coverage

This is an evidence map for the [API reference](api-reference.md), not a claim that every argument, accepted value, engine combination, or model has been exhaustively validated. The ordinary API inventory is **37 unique named arguments**: 34 constructor arguments plus `task`, `inputs`, and `outputs`; shared `knowledge` is counted once. The eight named `verified_task` arguments are mapped separately. Forwarding containers such as `**rlm_kwargs` are not additional arguments.

## Results and Scope

**Combined targeted validation: 169 passed, no skips, 16 dependency/deprecation warnings.** This includes **64 new data-backed cases** (24 execution, 14 skills, 14 engines, 10 verified-task, 2 CSV-halving), six API-reference cases, and supporting generalization, knowledge/evidence, network, reconciliation, and retry-halving tests. Do not add the standalone counts again.

**Clean release-checkout offline suite: 3,447 passed, 13 skipped, 55 warnings.** This excludes the uncommitted research harnesses/tests used during development. Provider credentials were removed from the test process and LiteLLM metadata configured locally. Skips are not passes and do not establish live Fabric/provider support. No paid model calls were made for this argument-validation work.

The new modules script the outer LM but retain real execution engines and subprocess workers. Data is synthetic: the skill and verified-task fixtures are actual CSV files containing `17, 25, 8` (total 50); the execution fixture filters sales rows to total 50 instead of 950; the engine fixture varies an active amount between 19 and 47 and excludes a cancelled 900 row. Engine and CSV-halving tests parse CSV text rather than an on-disk CSV. Verified-task tests compute 42 by omitting the final row and 50 by including it; CSV-halving tests recompute 50 on each attempt while submitting 51, 51, then 50. These are runtime-contract tests, not real-world datasets or model-quality evaluations. The context-validator test really saves, reopens, rejects, repairs, and rechecks an XLSX; it skips without `openpyxl`.

Classification used below:

- **SP**: real subprocess execution with scripted outer LM and actual CSV computation. Host prompt, routing, validator, or metadata assertions are identified in the scenario; they do not mean the worker was mocked.
- **LOCAL**: real subprocess/client transport to a local fake endpoint; no remote provider or authenticated service is tested.
- **HOST+SP**: real host operation/tool plus real worker, with scripted LM.
- **NUM+SP**: real DSPy execution with scripted numeric submissions and a constructor spy delegating to the real constructor; no CSV/data-derived answer.
- **MOCK**: fake interpreter or stubbed solve results; host orchestration/prompt/metadata checks only, not actual code execution.
- **GAP**: no direct test identified for the stated behavior.

Test-node prefixes below expand to these exact paths. A function node selects all its parameterized cases unless a bracketed case is shown; links open the corresponding source definition.

| Prefix | File |
| --- | --- |
| E | [`tests/test_api_arguments_execution.py`](../tests/test_api_arguments_execution.py) |
| S | [`tests/test_api_arguments_skills.py`](../tests/test_api_arguments_skills.py) |
| N | [`tests/test_api_arguments_engines.py`](../tests/test_api_arguments_engines.py) |
| W | [`tests/test_api_arguments_verified.py`](../tests/test_api_arguments_verified.py) |
| C | [`tests/test_api_arguments_halving.py`](../tests/test_api_arguments_halving.py) |
| H | [`tests/test_halve_max_iter_param.py`](../tests/test_halve_max_iter_param.py) |
| G | [`tests/test_generalization.py`](../tests/test_generalization.py) |
| K | [`tests/test_knowledge_learning_runtime.py`](../tests/test_knowledge_learning_runtime.py) |
| V | [`tests/test_verify_ensemble.py`](../tests/test_verify_ensemble.py) |
| R | [`tests/test_api_reference.py`](../tests/test_api_reference.py) |

## API Reference Checks

The six cases in `tests/test_api_reference.py` check documentation inventory, signature/default alignment, starting guidance, public links, and unexecuted task construction. They are documentation/host checks, not CSV execution coverage, and are separate from the 64 reported standalone runtime cases.

| Exact test node | Cases | Checked boundary |
| --- | --- | --- |
| [R::test_api_parameter_tables_match_signatures_and_defaults](../tests/test_api_reference.py#L29) | 2 | Ordinary 37 and verified-task 8 named arguments appear exactly once with no extras; documented defaults match inspected signatures, including required outputs and symbolic reconciliation guidance. |
| [R::test_api_reference_links_to_sources_and_gives_starting_guidance](../tests/test_api_reference.py#L85) | 1 | Source links and starting guidance mention `RLM.task(...).run()`, `max_turns=20`, and `output_validator`. This checks those guidance markers, not every argument's when-to-use advice. |
| [R::test_public_guides_link_to_api_reference](../tests/test_api_reference.py#L97) | 2 | README and QUICKSTART link to the reference. |
| [R::test_task_construction_keeps_documented_ordinary_defaults](../tests/test_api_reference.py#L102) | 1 | Task construction does not invoke the LM; selected constructor defaults, 20 turns, repair integrity, resolved engine, and empty inline inputs/outputs are checked. |

## Ordinary API: 37 Arguments

| Field | Scenario and tested boundary | Exact test node(s), using prefixes above | Class |
| --- | --- | --- | --- |
| `task` | Task text reaches prompt; CSV total/count survives missing-field repair. | [E::test_task_inputs_outputs_signature_and_lm_repair](../tests/test_api_arguments_execution.py#L63) | SP + host prompt |
| `inputs` | Bound CSV remains available after column error; timeout restart rebinds original inputs, not computed state. | [E::test_worker_csv_error_is_repaired_without_rebinding_inputs](../tests/test_api_arguments_execution.py#L89); [E::test_timeout_budget_restarts_and_rebinds_csv_input](../tests/test_api_arguments_execution.py#L124) | SP |
| `outputs` | Typed total/count declaration rejects missing count, then accepts computed fields; not every supported type. | [E::test_task_inputs_outputs_signature_and_lm_repair](../tests/test_api_arguments_execution.py#L63) | SP |
| `signature` | Omitted/explicit string on default path; omitted/explicit DSPy signature with host tool. | [E::test_task_inputs_outputs_signature_and_lm_repair](../tests/test_api_arguments_execution.py#L63); [N::test_host_tool_csv_total](../tests/test_api_arguments_engines.py#L78) | SP; HOST+SP |
| `lm` | Scripted callable drives actual computation and receives repair feedback; scripted DSPy LM also used. Not every provider/configuration form. | [E::test_task_inputs_outputs_signature_and_lm_repair](../tests/test_api_arguments_execution.py#L63); [N::test_host_tool_csv_total](../tests/test_api_arguments_engines.py#L78) | SP; HOST+SP |
| `sub_lm` | Serializable dict config reaches worker `predict`; real DSPy/OpenAI HTTP client sends selected CSV subset and fake bearer key to loopback server computing subtotal. | [N::test_worker_sub_lm_roundtrip](../tests/test_api_arguments_engines.py#L195) | LOCAL |
| `engine` | Default execution, explicit DSPy and auto-with-tools, adaptive escalation; no complete alias/feature cross-product. | [E::test_max_submit_bytes_rejects_oversize_and_allows_csv_repair](../tests/test_api_arguments_execution.py#L315); [N::test_host_tool_csv_total](../tests/test_api_arguments_engines.py#L78); [N::test_adaptive_csv_escalation](../tests/test_api_arguments_engines.py#L245) | SP; HOST+SP |
| `tools` | Worker calls real host regional-total callable; host PID, filtered sum, and combined worker result checked. | [N::test_host_tool_csv_total](../tests/test_api_arguments_engines.py#L78) | HOST+SP |
| `adaptive` | Real ladder/validator/callback: one-attempt stop versus fifth-rung strong scripted LM; cancelled-row error remains a failure despite submission. | [N::test_adaptive_csv_escalation](../tests/test_api_arguments_engines.py#L245) | SP + host policy |
| `inner_engine` | Adaptive CSV attempts execute with `default` and `dspy`; selected result and attempt summaries checked. | [N::test_adaptive_csv_escalation](../tests/test_api_arguments_engines.py#L245) | SP + host metadata |
| `max_turns` | One turn computes but cannot submit; two turns submit correct CSV total. | [E::test_max_turns_bounds_csv_work_not_just_configuration](../tests/test_api_arguments_execution.py#L107) | SP |
| `timeout` | Five-second worker wait interrupts scripted 30-second sleep after CSV work; not an outer-LM deadline. | [E::test_timeout_budget_restarts_and_rebinds_csv_input](../tests/test_api_arguments_execution.py#L124) | SP |
| `recover_worker_timeouts` | Zero returns worker-timeout failure; one restart loses computed namespace and recomputes from rebound CSV. | [E::test_timeout_budget_restarts_and_rebinds_csv_input](../tests/test_api_arguments_execution.py#L124) | SP |
| `stuck_loop_threshold` | Two repeated CSV column errors stop at threshold 2; `None` permits subsequent repair. | [E::test_stuck_loop_threshold_stops_repeated_csv_column_errors](../tests/test_api_arguments_execution.py#L151) | SP |
| `halve_max_iter_on_retry` | Real DSPy recomputes CSV total 50 on all three attempts; validator rejects 51 twice, then accepts 50. Enabled budgets are 8/4/2; disabled budgets are 8/8/8; both record two repairs. No remaining CSV-backed halving gap. Older numeric tests supplement this with omitted/default behavior and a separate 10/10 repair. | [C::test_halve_max_iter_on_retry_with_csv_total[halving-enabled]](../tests/test_api_arguments_halving.py#L17); [C::test_halve_max_iter_on_retry_with_csv_total[halving-disabled]](../tests/test_api_arguments_halving.py#L17); [H::test_halving_default_behavior](../tests/test_halve_max_iter_param.py#L53); [H::test_no_halving_when_disabled](../tests/test_halve_max_iter_param.py#L77); [H::test_no_halving_recovers_when_default_would_starve](../tests/test_halve_max_iter_param.py#L100) | SP + host validation/constructor spy; supplemental NUM+SP |
| `reserve_finalize_turns` | Reserve 0/1 changes final-turn budget hint during three-turn CSV computation; does not prove guaranteed finalization. | [S::test_reserve_finalize_turns](../tests/test_api_arguments_skills.py#L242) | SP + host prompt |
| `max_prompt_tokens` | Threshold 1 versus 1,000,000 digests large synthetic skill body; checks shorter system message, not exact tokens or a hard context cap. | [S::test_prompt_budget[max_prompt_tokens]](../tests/test_api_arguments_skills.py#L220) | SP + host prompt |
| `digest_after_turn` | `None` versus 2 preserves/replaces skill body on expected calls in a three-turn CSV run. | [S::test_prompt_budget[digest_after_turn]](../tests/test_api_arguments_skills.py#L220) | SP + host prompt |
| `skills` | Empty list versus explicit checker changes prompt body and actual verifier execution. | [S::test_skills](../tests/test_api_arguments_skills.py#L115) | SP + host prompt/metadata |
| `skill_loader` | Two custom host loaders supply different bodies for the same checker; verifier still executes. Worker discovery parity is not tested. | [S::test_skill_loader](../tests/test_api_arguments_skills.py#L126) | SP + host prompt/metadata |
| `enable_skill_autoloading` | Off/on hides/shows available-skill index without preloading checker body. | [S::test_enable_skill_autoloading](../tests/test_api_arguments_skills.py#L196) | SP + host prompt |
| `skills_as_cards` | Body versus short card; cards indicate bodies are not preloaded, while the explicit checker verifier executes in both modes. | [S::test_skills_as_cards](../tests/test_api_arguments_skills.py#L206) | SP + host prompt/metadata |
| `enable_router` | Off/on changes elected bodies and active metadata while CSV result stays correct. | [S::test_router_arguments[enable_router]](../tests/test_api_arguments_skills.py#L178) | SP + host routing |
| `max_active_skills` | Candidate allowance 1/2 changes active bodies and remaining cards; not a hard cap on all skills. | [S::test_router_arguments[max_active_skills]](../tests/test_api_arguments_skills.py#L178) | SP + host routing |
| `router_baseline_skills` | Empty baseline versus explicit core baseline changes active set and prompt. | [S::test_router_arguments[router_baseline_skills]](../tests/test_api_arguments_skills.py#L178) | SP + host routing |
| `router_candidate_specificities` | Domain versus utility filter elects different synthetic skills. | [S::test_router_arguments[router_candidate_specificities]](../tests/test_api_arguments_skills.py#L178) | SP + host routing |
| `router_include_dependencies` | Off/on omits/adds helper dependency outside ranked candidates. | [S::test_router_arguments[router_include_dependencies]](../tests/test_api_arguments_skills.py#L178) | SP + host routing |
| `enable_verifier` | Wrong computed 42 repaired to 50; skill checker rejects when enabled, host validator still rejects when disabled. | [S::test_enable_verifier](../tests/test_api_arguments_skills.py#L143) | SP + host validation |
| `output_validator` | None, normal return, assertion rejection, and ignored `False`; computed wrong 51 can pass provenance without assertion-based truth check. | [E::test_output_validator_known_source_contract_and_false_caveat](../tests/test_api_arguments_execution.py#L170) | SP + host validation |
| `output_validator_context` | Uses original inputs/state/trajectory; reopens actual XLSX, rejects cell 51, accepts repaired 50 and row count 3. | [E::test_context_validator_reopens_saved_xlsx_and_repairs_cells](../tests/test_api_arguments_execution.py#L207) | SP + host artifact validation |
| `analytical_integrity` | Strict rejects unsupported literal 999 after printing computed total; off accepts it. This is a heuristic, not a correctness oracle. | [E::test_analytical_integrity_checks_actual_emitted_numeric_literal](../tests/test_api_arguments_execution.py#L342) | SP |
| `security` | Default/disabled policy scrubs/preserves a synthetic secret-key environment probe while CSV total remains correct. Not comprehensive isolation testing. | [E::test_security_scrubs_only_fake_secret_probe_at_task_boundary](../tests/test_api_arguments_execution.py#L256) | SP |
| `block_network` | On/off preserves real loopback CSV access and checks worker flag; separate remote-connect rejection test has no CSV. | [E::test_block_network_preserves_loopback_csv_access](../tests/test_api_arguments_execution.py#L278); [tests/test_block_network.py::test_remote_connect_refused_in_worker](../tests/test_block_network.py#L124) | LOCAL; real worker guard |
| `max_submit_bytes` | 256 rejects oversized detail and permits repair; 4096 accepts first payload on default and DSPy engines. | [E::test_max_submit_bytes_rejects_oversize_and_allows_csv_repair](../tests/test_api_arguments_execution.py#L315) | SP |
| `knowledge` | Real learned manufacturing CSV host aggregate cross-checked against worker raw CSV read; source binding remains available. | [G::test_the_packet_introduces_itself_and_the_raw_source_stays_bound](../tests/test_generalization.py#L213) | HOST+SP |
| `capture_evidence` | Real CSV host group-by records execution/grain evidence; separate on/off test checks identical prompts/answer but only submits filename, not a CSV aggregate. | [G::test_host_operations_are_evidence_and_a_file_source_learns_its_grain](../tests/test_generalization.py#L238); [K::test_capture_on_and_off_produce_the_same_prompts_and_answer](../tests/test_knowledge_learning_runtime.py#L312) | HOST+SP; real worker filename/metadata |
| `verbose` | Quiet/loud host output differs during the same real CSV computation. | [S::test_verbose](../tests/test_api_arguments_skills.py#L254) | SP + host diagnostics |

## Verified Task: Eight Arguments

The 10 cases in `test_api_arguments_verified.py` run real default workers against an on-disk CSV; only model responses are scripted, and constructor spies delegate to the real factory. The table maps wrapper-level data computation, agreement, reconciliation, validation repair, and fallback. The older `test_verify_ensemble.py` alone uses `SequencedInterpreter` payloads or stubbed solves rather than CSV execution; it remains supplemental unit coverage, not the basis for the data-backed claims below.

| Field | Scenario / remaining gap | Exact test node(s) | Class |
| --- | --- | --- | --- |
| `task` | Original task reaches independent A/B prompts; C receives task plus custom guidance and computed candidate answers 42/50, without private worker state in the prompt. | [W::test_typed_outputs_default_to_first_field_and_skip_stronger_model](../tests/test_api_arguments_verified.py#L73); [W::test_computed_disagreement_preserves_inputs_callbacks_and_custom_guidance](../tests/test_api_arguments_verified.py#L125) | SP + host prompt |
| `outputs` | Typed CSV total/note mapping defaults to first field; string total is rejected and repaired to integer before comparison. Untyped list accepts numeric/string totals. | [W::test_typed_outputs_default_to_first_field_and_skip_stronger_model](../tests/test_api_arguments_verified.py#L73); [W::test_typed_outputs_repair_csv_total_before_comparing](../tests/test_api_arguments_verified.py#L106); [W::test_default_agreement_compares_numeric_and_string_outputs_as_strings](../tests/test_api_arguments_verified.py#L211) | SP + host validation |
| `inputs` | A/B/C reopen the bound CSV in isolated workers; worker-local mutation does not leak. Original inputs/settings/outputs remain unchanged; validators receive original inputs and worker state, and callbacks are preserved. | [W::test_computed_disagreement_preserves_inputs_callbacks_and_custom_guidance](../tests/test_api_arguments_verified.py#L125); [W::test_no_overrides_reuses_model_and_full_default_budget_without_mutating_arguments](../tests/test_api_arguments_verified.py#L312) | SP + host callbacks/constructor spy |
| `field_name` | Default first typed field agrees on 50 despite different notes; explicit `total` detects computed 42/50 disagreement despite a matching first-field note and selects C's 50. | [W::test_typed_outputs_default_to_first_field_and_skip_stronger_model](../tests/test_api_arguments_verified.py#L73); [W::test_explicit_field_selects_total_instead_of_matching_first_field](../tests/test_api_arguments_verified.py#L195) | SP + host selection |
| `reconcile_guidance` | Custom CSV-check guidance appears only in C's prompt alongside computed candidates; omitted guidance also supports real CSV reconciliation. Empty guidance is not directly asserted. | [W::test_computed_disagreement_preserves_inputs_callbacks_and_custom_guidance](../tests/test_api_arguments_verified.py#L125); [W::test_no_overrides_reuses_model_and_full_default_budget_without_mutating_arguments](../tests/test_api_arguments_verified.py#L312) | SP + host prompt |
| `reconcile_lm` | Separate scripted LM executes only C after computed disagreement; agreement skips it. Without overrides the same model drives all three real solves. | [W::test_computed_disagreement_preserves_inputs_callbacks_and_custom_guidance](../tests/test_api_arguments_verified.py#L125); [W::test_typed_outputs_default_to_first_field_and_skip_stronger_model](../tests/test_api_arguments_verified.py#L73); [W::test_no_overrides_reuses_model_and_full_default_budget_without_mutating_arguments](../tests/test_api_arguments_verified.py#L312) | SP + host routing |
| `reconcile_max_turns` | Inherited one-turn budget leaves C's CSV-derived 42 rejected and falls back to valid B=50; override 2 repairs C to 50. With no overrides C spends the full default 20 turns doing CSV work. | [W::test_reconcile_budget_inherits_or_repairs_rejected_csv_answer](../tests/test_api_arguments_verified.py#L258); [W::test_no_overrides_reuses_model_and_full_default_budget_without_mutating_arguments](../tests/test_api_arguments_verified.py#L312) | SP + host budget/validation |
| `agree` | Default compares numeric/string 50 as strings but rejects 50 versus computed 50.01; custom tolerance accepts those strings and skips C. Validator-rejected CSV answers bypass even an always-true comparator. | [W::test_default_agreement_compares_numeric_and_string_outputs_as_strings](../tests/test_api_arguments_verified.py#L211); [W::test_exact_default_comparison_and_custom_tolerance](../tests/test_api_arguments_verified.py#L229); [W::test_reconcile_budget_inherits_or_repairs_rejected_csv_answer](../tests/test_api_arguments_verified.py#L258) | SP + host comparator/validation |

Supplemental older unit tests retain distinct boundary evidence; none of these are CSV-worker tests:

| Boundary | Exact test node(s) | Class |
| --- | --- | --- |
| Task/input forwarding and stronger-model routing; agreement skips C | [V::test_strong_reconciler_overrides_only_third_solve](../tests/test_verify_ensemble.py#L138); [V::test_agreement_never_calls_strong_reconciler](../tests/test_verify_ensemble.py#L125) | MOCK |
| Typed first/explicit field, wrong-type repair, empty outputs and undeclared/empty field rejection | [V::test_typed_mapping_selects_first_field_or_explicit_field](../tests/test_verify_ensemble.py#L202); [V::test_typed_output_default_still_repairs_wrong_type](../tests/test_verify_ensemble.py#L235); [V::test_empty_outputs_fail_before_solving](../tests/test_verify_ensemble.py#L219); [V::test_unknown_field_fails_before_solving](../tests/test_verify_ensemble.py#L228) | MOCK |
| Default reconciliation, budget/fallback and invalid-budget rejection | [V::test_disagreement_runs_reconciler_and_wins](../tests/test_verify_ensemble.py#L96); [V::test_reconciler_turn_budget_and_fallback](../tests/test_verify_ensemble.py#L183); [V::test_invalid_reconcile_turn_budget_fails_before_solving](../tests/test_verify_ensemble.py#L245) | MOCK |
| Unavailable candidate cannot be rescued by custom agreement; exact numeric comparison | [V::test_failed_solve_is_never_a_candidate_even_with_custom_agree](../tests/test_verify_ensemble.py#L400); [V::test_agree_numbers_must_match_exactly](../tests/test_verify_ensemble.py#L253) | MOCK; host comparator unit |

## Interpretation and Gaps

- Every ordinary argument and all eight verified-task arguments have mapped scenarios. Real CSV-backed retry-halving and wrapper reconciliation are now covered, but this is not exhaustive coverage of all accepted values, combinations, engine/feature cross-products, or pairwise interactions.
- The separate [generalization tests](../tests/test_generalization.py) use more realistic synthetic manufacturing CSV, inventory Parquet, and local service-ticket Delta scenarios. In particular, `G::test_dependent_code_blocks_run_in_order_and_fabricated_output_is_named` computes a real filtered CSV total; the two G nodes in the matrix exercise learned source operations and evidence. They still script models and do not establish real-world analytical quality.
- Context-budget tests are small, three-turn runs with one deliberately large skill body. They check prompt rewriting/hints, not long-context reliability, exact tokenizer accounting, cache savings, or hard cost caps.
- Adaptive coverage checks real serial ladder escalation and the `max_attempts` stop. `max_total_turns=10`, `max_parallel=1`, and `max_wall_seconds=120` are supplied, but their independent exhaustion boundaries and parallel rollouts are not established. DSPy's host-side `llm_query` sub-LM path is not exercised by the local worker `predict` test.
- The prompt header is "Skill cards (bodies not preloaded; use `load_skill(name)` to read)". `S::test_skills_as_cards` records the presentation/verifier distinction: explicit skill verifiers execute even when bodies are not preloaded; do not infer verifier inactivity from instructional-body visibility.
- [Suite provenance setup](../tests/conftest.py) disables claim provenance for most tests. The new execution, skill, and verified-task modules explicitly restore it; verified-task also clears the analytical-integrity override. The engine and retry-halving modules do not restore provenance. Passing those modules alone therefore does not prove production-default provenance behavior across engines.
- Security and socket-guard checks are defense-in-depth examples, not OS sandbox, filesystem isolation, native-code containment, or task-wide network guarantees. Host tools/validators remain privileged. No live authenticated Fabric/provider execution or model quality is claimed.

## Reproduce

Run from this worktree root with the project's test dependencies installed. XLSX needs `openpyxl`; broader data tests may need pandas, PyArrow, DuckDB, and Delta Lake. Review skips rather than counting them as passes. No real model credentials are needed for the targeted selection; the sub-LM endpoint is loopback with a fake key.

Standalone commands corresponding to the reported counts (experimental cases are intentionally included):

```powershell
python -m pytest -q tests/test_api_arguments_execution.py
python -m pytest -q tests/test_api_arguments_skills.py
python -m pytest -q tests/test_api_arguments_engines.py
python -m pytest -q tests/test_api_arguments_verified.py
python -m pytest -q tests/test_api_arguments_halving.py
python -m pytest -q tests/test_halve_max_iter_param.py
```

Combined runtime subset and supporting evidence tests:

```powershell
python -m pytest -q tests/test_api_arguments_execution.py tests/test_api_arguments_skills.py tests/test_api_arguments_engines.py tests/test_api_arguments_verified.py tests/test_api_arguments_halving.py
python -m pytest -q tests/test_api_reference.py tests/test_halve_max_iter_param.py tests/test_verify_ensemble.py tests/test_generalization.py tests/test_knowledge_learning_runtime.py tests/test_block_network.py
```

Broader offline-oriented regression selection, excluding live behavior gates and Fabric mount smoke tests even if credentials/mounts exist:

```powershell
$env:LITELLM_LOCAL_MODEL_COST_MAP = "True"
$env:LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS = "True"
python -m pytest -q tests --ignore=tests/behavior/test_behavior_baseline.py --ignore=tests/test_smoke_fabric.py -m "not primary and not secondary_free"
```

Those environment settings avoid LiteLLM metadata downloads; they do not enforce network isolation. This selection includes experimental/local integration tests and is not the live behavior suite. Use an externally controlled no-egress environment if strict offline execution is required. A bare `pytest` may select live behavior tests when credentials are present; the commands above do not claim a full-suite pass.
