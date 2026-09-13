# Cross-domain generalization evaluation

## Post-serialization evaluation

The `eval/postfix-generalization` branch evaluates the scalar fixes separately
from the original baseline below. New live runs record the actual Git SHA,
hash every core/skill file and fixture, persist B/C package contents before
evaluation, check package fingerprints for mutation, and use a unique
`<output-stem>.artifacts` directory. Existing outputs are refused, not replaced.
Older results predate these protections: their shared trace paths can have
been overwritten and their package fingerprints are not full saved packages.

Use the live command below with a **new output filename** and
`--max-cost-usd 5`. This checks account usage between trials, reserving one
dollar for in-flight billing; the call cap remains a separate bound.
Provider prompt caching can still occur despite DSPy `cache=False`; inspect
`cached_tokens`. The model identifier is a provider alias, not an immutable
model-weight version.

Keep the current value/units/period/identity score and original strict score.
Report value-only and individual fields separately, without changing the
criterion after seeing results. A passed `output_validator` checks answer
shape only; `verification_outcome="verified"` does **not** mean the reference
answer is correct. The retained `verifier_execution` identifies which checks ran.
Source-call telemetry counts instrumented adapter calls, not direct pandas
reads. Claims marked `supported` by the answer are self-reports, not independently
verified evidence coverage.

The current audit and limitations are in `POSTFIX_AUDIT.md`. To reproduce the
final fixed-core matrix from this branch (use fresh output paths):

```powershell
$fixtures = (Resolve-Path evaluation\generalization\generated).Path
python -m evaluation.generalization.runner live `
  --fixtures $fixtures --output evaluation\generalization\raw-results\final-naming.json `
  --variants descriptive,abbreviated,camel --repetitions 3 `
  --max-live-calls 423 --max-cost-usd 5
python -m evaluation.generalization.insufficient_metadata `
  --fixtures $fixtures --output evaluation\generalization\raw-results\final-ambiguity.json `
  --repetitions 3 --max-live-calls 9
python -m evaluation.generalization.evidence_summary `
  --input evaluation\generalization\raw-results\final-naming.json `
  --output evaluation\generalization\evidence\final-naming-summary.json
```

For the unfixed comparison, use a separate worktree at `4878627`, copy this
branch's `evaluation\generalization\*.py` harness into that worktree (not core),
and run the same `runner live` command with `--variants descriptive`,
`--max-live-calls 141`, and a distinct `final-baseline.json` output. Pass the
same absolute `$fixtures` path. Verify that only `fabric_rlm\serializers.py`
and `fabric_rlm\lakehouse.py` differ between core directories. Keep provider
settings unchanged; do not merge evaluation evidence into any tested package.

This evaluation is pinned to baseline commit
`b5226712a9aa41c3173d5f427e81244c333c0179`. The freeze manifest covers every
file below `fabric_rlm/`, including bundled skills. Evaluation code, generated
fixtures, references, graders, and results live outside that frozen scope.

## Reproduce

Run from the repository root on branch `eval/generalization-b522671`.

```powershell
# Generate all three domains, three naming variants, private references,
# and the 250,000-row prompt-impractical fixture.
python -m evaluation.generalization.runner prepare `
  --output evaluation\generalization\generated `
  --seed 20260908 `
  --large-rows 250000

# Verify the frozen core, run the dependency audit, and execute the targeted
# reliability/change-handling suite.
python -m evaluation.generalization.runner offline `
  --repo . `
  --fixtures evaluation\generalization\generated `
  --output evaluation\generalization\raw-results

# Budget-capped smoke: one unseen question per domain, A/B/C, plus separate
# development runs used only to build C. All calls count against the cap.
python -m evaluation.generalization.runner live `
  --fixtures evaluation\generalization\generated `
  --output evaluation\generalization\raw-results\live-smoke.json `
  --model openai/gpt-4.1-mini `
  --repetitions 1 `
  --variants descriptive `
  --max-live-calls 15 `
  --max-turns 6 `
  --timeout 120 `
  --smoke

# Required descriptive-name matrix: 15 questions x 3 repetitions x 3 arms,
# plus 6 separate development calls = 141 maximum live calls.
python -m evaluation.generalization.runner live `
  --fixtures evaluation\generalization\generated `
  --output evaluation\generalization\raw-results\live-descriptive-full.json `
  --model openai/gpt-4.1-mini `
  --repetitions 3 `
  --variants descriptive `
  --max-live-calls 141 `
  --max-turns 6 `
  --timeout 120

# Full naming matrix: 45 question/variant cases x 3 repetitions x 3 arms,
# plus 18 separate development calls = 423 maximum live calls.
python -m evaluation.generalization.runner live `
  --fixtures evaluation\generalization\generated `
  --output evaluation\generalization\raw-results\live-all-names-full.json `
  --model openai/gpt-4.1-mini `
  --repetitions 3 `
  --variants descriptive,abbreviated,camel `
  --max-live-calls 423 `
  --max-turns 6 `
  --timeout 120
```

`OPENROUTER_API_KEY` must be set in the process environment. The runner uses
`cache=False`, a fixed model, fixed sampling settings, seeded trial and arm
order, fixed execution limits, and counts development calls against the live
budget. Reference answers and grader logic are loaded in the supervising
harness and are never passed to the agent.

## Artifacts

- `fixtures.py`: deterministic fixture generator.
- `references.py`: independent reference calculations using Python CSV logic,
  not the library query compiler.
- `grader.py`: separates correct, confident-wrong, incomplete, and unsupported
  claim outcomes.
- `audit.py`: executable-vs-documentation ARR scan and behavioral dependency
  probes.
- `frozen-baseline.json`: SHA-256 freeze manifest for core code and skills.
- `evidence/`: captured offline, live-smoke, and Fabric adapter results. The
  live smoke preserves the original answers and metrics and records the
  grading revision applied after the first live run exposed status aliases.
- `generated/`: ignored generated fixtures and private references.

The generated definitions document is `generated\definitions.json`. Private
reference answers are in `generated\private\references.json`; the runner never
binds that path or its contents into an RLM task.

## Real Fabric evidence

`evidence\fabric-probe.json` records the live workspace, Lakehouse, notebook,
semantic-model, refresh, query, and adapter results. The evaluation created
temporary artifacts named `fabric_rlm_generalization_*` and `rlm_eval_*` in
workspace `sandeep_ws`; they remain in place for reproduction.

The real integration verified the same three independently calculated answers
through the Lakehouse SQL endpoint, frozen `LakehouseSource.query`, Direct Lake
DAX, and frozen `SemanticModel.aggregate`: 186 latest-snapshot available units,
1,800 complete-period produced units, and a 2/3 first-response SLA rate.

Important runtime requirements discovered by the test:

- Schema-enabled Lakehouse Delta paths use `Tables/dbo/<table>`.
- Outside Fabric, `LakehouseSource` needs an explicit catalog and an injectable
  storage credential; automatic discovery requires notebookutils.
- `SemanticModel` works in Fabric with
  `credential_provider="notebookutils"`. Local SemPy/XMLA authentication did
  not work with the Azure CLI identity even though Power BI REST did.
- A newly deployed Direct Lake model must be refreshed and its metadata queried
  successfully before adapter evaluation.
