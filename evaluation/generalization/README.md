# Cross-domain generalization evaluation

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
budget. Reference answers and grader logic are loaded only after each run and
are never passed to the agent.

## Artifacts

- `fixtures.py`: deterministic fixture generator.
- `references.py`: independent reference calculations using Python CSV logic,
  not the library query compiler.
- `grader.py`: separates correct, confident-wrong, incomplete, and unsupported
  claim outcomes.
- `audit.py`: executable-vs-documentation ARR scan and behavioral dependency
  probes.
- `frozen-baseline.json`: SHA-256 freeze manifest for core code and skills.
- `evidence/`: captured raw offline, live-smoke, and Fabric adapter results.
- `generated/`: ignored generated fixtures and private references.

The generated definitions document is `generated\definitions.json`. Private
reference answers are in `generated\private\references.json`; the runner never
binds that path or its contents into an RLM task.
