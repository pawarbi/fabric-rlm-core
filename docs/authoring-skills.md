# Authoring Skills

A skill is a Markdown playbook loaded by `fabric_rlm.skill_loader.SkillLoader`.
Write concise task guidance, define the expected outputs, and optionally add a
verifier for checks you can actually justify. The loader does not require a
fixed list or order of prose sections. A skill without a verifier can still be
loaded and activated; it simply contributes no submission checks.

Start with [skill-template.md](skill-template.md), a working `csv_summary`
example, not an unfinished stub. It lives under `docs/`, so it is **not a
bundled skill**. For your own application, copy it into a custom skills directory
and rename it `csv_summary.md` (or choose your own filename and update the title,
metadata, contract, and tests). Keep unrelated documentation out of that directory:
the loader discovers every immediate `*.md` file with a safe name, not just files
that follow this suggested layout.

## Names and Loading

The filename stem is the skill's identity, not its Markdown title. Safe names
start with an ASCII letter or digit and contain only ASCII letters, digits,
underscores, and hyphens. `load("csv_summary")` and `load("csv_summary.md")` both
work; paths such as `../csv_summary` do not.

Pass your custom loader to the runtime explicitly. For example, after placing
your authored file in `my_skills/csv_summary.md`, use this configuration with
your configured `lm` and input CSV:

```python
from fabric_rlm import RLM
from fabric_rlm.skill_loader import SkillLoader

loader = SkillLoader(skill_dir="my_skills")
rlm = RLM.from_task(
    "Count CSV records in csv_path and sum the amount column using csv_summary.",
    outputs={"summary": dict},
    lm=lm,
    skills=["csv_summary"],
    skill_loader=loader,
    enable_router=False,
    enable_verifier=True,
)
result = rlm.run(inputs={"csv_path": "data/amounts.csv"})
```

Custom directories layer over packaged skills by default. A custom file wins
over a packaged file of the same name; with multiple directories, the last
directory wins. Use `include_packaged=False` with a custom directory for an
isolated collection. Dependencies are included by `compose_skills` by default,
or by `load_text(..., include_dependencies=True)`; `load()` alone does not expand
them. Composition detects dependency cycles.

This example directly selects the skill so its extracted verifier runs on
submission with verification enabled. With routing enabled, verification also
depends on activation. Do not assume that worker-side `load_skill()` or
`list_skills()` discovers your custom directory: those convenience functions
construct a default, packaged loader. Nor does merely reading Markdown register
a verifier with the runtime.

## Metadata That Is Parsed

Put optional YAML frontmatter at the very start of the file, between `---`
lines. Follow it immediately with a title, an explicit `Summary:`, and, if
needed, a legacy `Dependencies:` line. The loader examines only the first 20
body lines for these metadata lines; prose is not used to invent a summary.

```yaml
---
applies_when:
  keywords: ["csvsummary", "csv summary"]
  output_fields: []
excludes: []
depends_on: []
specificity: domain
---
```

- `applies_when.keywords`: case-insensitive question substrings; each match
  contributes 2 points to automatic routing.
- `applies_when.output_fields`: each exact match contributes 1 point, but the
  question extractor only recognizes tokens matching `[A-Z][A-Za-z]?\d+`, such
  as `Q1`, `Q23`, or `A1`. It is not a general output-schema parser; a field like
  `summary` does not score through this heuristic. Use keywords for this example.
- `depends_on`: a **nonempty normalized** frontmatter value takes precedence
  over legacy `Dependencies:`. Empty, absent, or unusable values fall back to the
  legacy line, even when frontmatter exists. To declare no dependencies, use
  `depends_on: []` and omit the legacy line or set it to `Dependencies: none`.
- `excludes`: advisory, directional exclusions during automatic ranking, not
  a safety gate. Already-active skills' exclusions block later ranked candidates;
  a candidate's own exclusions do not retroactively remove earlier selections.
  Baseline selection, explicit selection, dependency expansion, and direct
  composition do not enforce a universal mutual-exclusion rule.
- `specificity`: optional, defaults to `domain`; recognized values are `core`,
  `domain`, and `utility`. Unknown values warn and fall back to `domain`. The
  router defaults to a `core` baseline unless its baseline is explicitly
  overridden; candidate-specificity filters can further restrict ranking.
  Utilities are not inherently card-only. Cards are ranked but inactive
  candidates, with utilities having extracted verifiers ordered first.
- `verifier_present`: derived from nonempty extracted source, not an author-set
  flag and not proof that the source compiles or defines a working `verify`.

List-like metadata accepts lists/tuples or comma-separated strings; use YAML
lists for clarity. If YAML parsing fails, produces a non-mapping, or PyYAML is
unavailable, recognized frontmatter is removed from the body but its metadata
is ignored. Legacy body metadata still applies. Without routing metadata, a
skill can still be explicitly selected or loaded as a dependency; there is no
automatic alphabetical fallback activation. Score ties preserve candidate order
(the loader's discovery order is sorted).

## Suggested Layout

These are authoring suggestions, not runtime-mandatory headings:

- `## Purpose`: state the task class and when not to use the skill.
- `## Contract: output fields`: name each output, its type, meaning, formula,
  and source. Quote exact task wording when implementing a specific task; do not
  invent a benchmark contract for a generic skill.
- `## Required verifier`: optional executable checks, using the extraction
  convention below if you include them.
- `## Tripwires`: concrete mistakes to avoid, not invented incident histories.
- `## Procedure`: short parse, compute, verify, and submit instructions where
  useful. Add examples or invariants only when they clarify the contract.

For verifier extraction, use the exact heading `## Required verifier` on its own
line followed by a fenced block opened with exactly three backticks and
`python`. The loader extracts the first such block after the heading, provided
there is no intervening `## ` heading. Keep all imports and helpers required by
`verify(payload)` inside that block. Other prose headings have no equivalent
mandatory extraction contract.

## Verifier Do and Don't

- **Do** define `verify(payload)` to return silently on success and raise
  `AssertionError` with a field-specific repair message on a contract violation.
- **Do** check container types and required/unexpected keys before indexing,
  then check value types before arithmetic. Reject `bool` explicitly where an
  integer or float is required: Python treats booleans as integers.
- **Do** test empty inputs, zero, negative totals where allowed, non-finite
  numbers, missing fields, extra fields, and malformed types independently.
- **Do** distinguish structural checks from factual validation. The CSV example
  checks shape and numeric bounds; it does not reopen the CSV or prove the
  reported count and sum are true. Use an independent source-aware check when
  that guarantee is needed.
- **Don't** reject a structurally defined answer using an invented exact
  formula or hard-coded expected total. Check only justified bounds unless
  independent evidence supports equality.
- **Don't** use `ValueError`, `KeyError`, `TypeError`, or a `NotImplementedError`
  stub as the rejection mechanism. Non-assertion errors, timeouts, and other
  verifier failures fail open: the runtime logs/skips the broken check rather
  than rejecting the payload on its behalf. A returned `False` is not rejection
  either. Catch expected parsing errors and explicitly raise `AssertionError`
  when parsing is part of the verifier's contract.
- **Don't** treat an extracted verifier, an activated skill, or even a passing
  structural check as proof of task correctness. Also test known-correct answers
  and deliberately wrong ones against the checks you actually claim to enforce.

## Run the Example Locally

Run this Python self-test from the repository root in your project environment.
It needs no model, worker, CSV file, or custom-directory copy. The loader name
is `skill-template` here because the file still lives under its documentation
filename. Execute only trusted authored verifier code; `exec` here is ordinary
Python execution, not a sandbox.

```python
from pathlib import Path
from fabric_rlm.skill_loader import SkillLoader

loader = SkillLoader(skill_dir=Path("docs"), include_packaged=False)
skill = loader.load("skill-template")
assert skill.title == "csv_summary"
assert skill.summary
assert skill.dependencies == ()
assert skill.verifier_source is not None
namespace = {}
exec(skill.verifier_source, namespace)
verify = namespace["verify"]

valid = [
    {"summary": {"row_count": 0, "total": 0}},
    {"summary": {"row_count": 3, "total": 50}},
    {"summary": {"row_count": 2, "total": -1.5}},
]
invalid = [
    None,
    {},
    {"summary": []},
    {"summary": {"row_count": 1}},
    {"summary": {"row_count": 1, "total": 0, "extra": 1}},
    {"summary": {"row_count": 1, "total": 0}, "extra": 1},
    {"summary": {"row_count": -1, "total": 0}},
    {"summary": {"row_count": 1.0, "total": 0}},
    {"summary": {"row_count": True, "total": 0}},
    {"summary": {"row_count": 1, "total": False}},
    {"summary": {"row_count": 1, "total": "50"}},
    {"summary": {"row_count": 1, "total": float("nan")}},
    {"summary": {"row_count": 1, "total": float("inf")}},
    {"summary": {"row_count": 1, "total": float("-inf")}},
]
for payload in valid:
    verify(payload)
for payload in invalid:
    try:
        verify(payload)
    except AssertionError:
        pass
    else:
        raise AssertionError(f"Invalid payload was accepted: {payload!r}")
print(f"Passed {len(valid)} valid and {len(invalid)} invalid cases")
```

Unexpected exception types deliberately escape this test: a `ValueError` is a
broken verifier, not a successful rejection. These fixture values test the
contract, not a particular CSV's truth, and the verifier does not pin a total.

The companion regression command is `python -m pytest tests/test_skills_guide.py`.
That test file is being added separately; until it is available, run the
self-test above. Existing loader/router coverage can also be run with
`python -m pytest tests/test_skill_loader_layering.py tests/test_skill_frontmatter.py tests/test_skill_router.py`.

## Libraries and Permissions

Prefer standard-library procedures when sufficient, as in the CSV example.
If a skill needs a third-party library, document installation in the host or
notebook environment before the run and an explicit missing-library fallback
or limitation. Do not instruct worker code to install dependencies: default
security policy blocks shell/subprocess installation. Do not weaken that policy
to accommodate a playbook. Keep verifier code small and self-contained, and
avoid relying on undeclared worker variables or network access.
