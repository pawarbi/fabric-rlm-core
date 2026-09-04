# Contributing to fabric-rlm

Thanks for your interest! This is a small project — the process is light.

## Development setup

```bash
git clone https://github.com/pawarbi/fabric-rlm-core.git
cd fabric-rlm-core
pip install -e ".[dev]"
pytest -q
```

Python 3.10 to 3.12 (3.13 is not supported yet). The test suite spawns real CPython subprocesses (the worker),
so expect it to take a few minutes; no API keys are required for the unit
suite.

## Tests

- **Unit tests** (`tests/*.py`) must pass offline. Add a test for every
  behavior change — this repo leans DAMP: one test names one behavior.
- **Behavior gate** (`tests/behavior/`) runs pinned LM evaluations against a
  committed baseline. It only runs when `OPENROUTER_API_KEY` is set and is
  exercised in CI on pull requests. If your change intentionally shifts LM
  behavior, recalibrate with `python tests/behavior/runner.py --calibrate`
  and commit the updated `baselines.json` with an explanation.
- Run the full suite before opening a PR: `pytest -q`.

## Skills

New skills are markdown playbooks in `fabric_rlm/skills/`. Follow
`docs/authoring-skills.md` and copy `docs/skill-template.md` to
start. Include frontmatter (`applies_when.keywords`, `specificity`) so the
router can score the skill, and a `## Required verifier` block when the
output shape is checkable.

## Pull requests

- Keep PRs focused; separate mechanical changes from behavior changes.
- Update `CHANGELOG.md` under an `Unreleased` heading.
- Public API changes (anything exported from `fabric_rlm/__init__.py`) need
  a docstring and a QUICKSTART mention.
- Deprecations: keep the old spelling working with a `DeprecationWarning`
  for at least one minor version.

## Security

Code-execution / sandbox issues: see [SECURITY.md](SECURITY.md). Do not open
public issues for exploitable reports.
