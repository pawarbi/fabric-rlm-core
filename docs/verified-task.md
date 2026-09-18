# Independent solves and reconciliation

`verified_task` runs two blind solves. If their selected answers disagree or one
is unavailable, it runs a third solve against the original task and inputs plus
the two candidate answers. Agreement is evidence, not proof of correctness.

```python
from fabric_rlm import FabricLM, verified_task

vr = verified_task(
    task=task_text,
    inputs=inputs,
    outputs={"answer": str},
    lm=FabricLM("gpt-5-mini"),
    reconcile_lm=FabricLM("gpt-5.1"),
    max_turns=12,
    reconcile_max_turns=16,
)
print(vr.verdict)
print(vr.selected_attempt_index)  # 0/1: initial solve; 2: reconciler
print(vr.reconciliation_reason)  # None, disagreement, unavailable_answer
print(vr.reconciliation_succeeded, vr.fallback_used)
if vr.result.submitted:
    print(vr.result.payload)
vr.result.inspect()
```

Use models available in your Fabric region, or pass any supported LM provider.

## Defaults

- `reconcile_lm=None` inherits `lm`; it does not change the two initial solves.
- `reconcile_max_turns=None` inherits `max_turns`, including the runtime's default
  of 20 when neither is specified. Budgets apply **per solve**, not to the ensemble.
- Omitted `field_name` selects the first declared output name for either
  `outputs=["answer"]` or `outputs={"answer": str}`. The typed mapping is retained
  for validation; no `field_name="answer"` workaround is needed (issue #86).
- Explicit `field_name` must exist in outputs. Empty outputs fail before any call.
- `agree(a, b)` can override answer comparison. It receives strings and is called
  only for two usable answers. Only the selected field is compared, not all fields.

Keep explanations, timing and code separate from the concise comparison field;
incidental differences can otherwise trigger unnecessary reconciliation.

## Trigger and outcome

Both initial solves run. Reconciliation is skipped only if both have submitted,
nonblank selected answers with no terminal failure and the comparator agrees.
One or two unavailable answers also trigger it. There is no separate trigger
policy argument. Output validators apply to every solve, including reconciliation.

Existing verdict strings remain `agree`, `reconciled`, `failed` for compatibility.
If reconciliation yields no usable answer but an initial candidate is usable,
that initial answer is still returned with verdict `reconciled`. Use the explicit
provenance properties to distinguish that fallback:

| Outcome | reconciliation_succeeded | fallback_used |
| --- | --- | --- |
| Initial answers agree | False | False |
| Reconciler supplies selected answer | True | False |
| Reconciler fails; earlier answer selected | False | True |
| No usable answer | False | False |

`reconciliation_attempted` identifies whether a third attempt exists. `ok` means
an answer was available, not that it is correct. `selected_attempt_index` is
zero-based; for a manually constructed result with no matching attempt it is None.
All attempts remain in `vr.attempts`, with their trajectories and failure reasons.

## Scope

This is for read-only analytical tasks with determinate answers. Do not run file
publication, workbook mutation, or other side effects two or three times through
this wrapper. Generate artifacts separately after accepting the analysis.

The reconciler receives task/inputs, candidate answer fields and
`reconcile_guidance`. It does not automatically receive failed traces, rejected
payloads, or private model reasoning. Failed initial solves appear as no answer.
No new requirements, host-review, planning, or judge API is necessary: use existing
`output_validator`/`output_validator_context` for application acceptance checks.
