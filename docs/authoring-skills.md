# Playbook Contract

> **Authoring a new skill?** Start from `SKILL_TEMPLATE.md`.

> Skills MUST follow this contract. The fabric-rlm runtime expects every skill
> to include the sections defined here, in this order, using exactly these
> heading names. The reflect-before-submit turn relies on each playbook
> exposing concrete invariants and a runnable verifier — generic checklists
> are not enough to catch wrong math, only field-level definitions and
> invariants are.

Every skill file under `fabric_rlm/skills/` (excluding this contract document
itself) MUST contain the six top-level sections shown below in the order
listed. The contract test (`tests/test_playbook_contract.py`) checks the
headings are present.

## Frontmatter (Track 5″)

Each skill SHOULD begin with a YAML frontmatter block delimited by `---`
lines. The block is consumed by the SkillRouter (see `skill_router.py`) to
decide whether the skill is preloaded into the system prompt, presented as
a card, or omitted entirely. Defaults are applied when fields are missing,
so the block is technically optional — but unannotated skills will only ever
be matched alphabetically as a fallback.

```yaml
---
applies_when:
  keywords: ["FLOW GAUNTLET", "max-flow", "min-cut"]
  output_fields: ["final_capacities", "saturated_edges"]
excludes: []
depends_on: ["longcot_core"]
specificity: domain        # one of: core | domain | utility
---
```

- `applies_when.keywords` — lowercased substrings the router looks for in
  the question text. Each match contributes weight 2 to the skill's score.
- `applies_when.output_fields` — output keys that, when present in the
  question, contribute weight 1 to the skill's score.
- `excludes` — list of skill names that conflict with this one and must not
  be co-loaded with it.
- `depends_on` — list of skill names whose bodies should also be loaded
  whenever this skill is. Falls back to the legacy `Dependencies:` line if
  the frontmatter is absent.
- `specificity` — `core` skills are always-on (e.g. `core.md`), `utility`
  skills (e.g. `validation`, `error_handling`) are presented as cards by
  default, `domain` skills (the bulk) participate in keyword routing.
- `verifier_present` is **auto-derived**: true when `## Required verifier`
  contains a runnable Python `verify(payload)` block. Authors do not set it.

## Required sections

1. `## Purpose`
   1 sentence (was: 1–3). When does this skill apply? What task class does
   it solve? No procedural content here — just scope.

2. `## Contract: output fields`
   The operational definition of every required output key (was named
   `## Output fields`). Nothing here may be left to model interpretation.
   For each field provide:
   - **name** (e.g. `Q2`)
   - **type** (e.g. `int`, `str`)
   - **exact definition**, citing the source (question prompt line, benchmark
     spec, paper). Quote the prompt verbatim where possible.
   - **canonical formula or procedure** for computing it.

3. `## Required verifier`
   A fenced ` ```python ` block (≈5–20 lines) defining
   `verify(payload) -> None`. The function MUST raise `AssertionError(...)`
   with a clear message on each invariant violation. The playbook MUST
   instruct the model to call `verify(payload)` on its computed solution and
   only emit `SUBMIT(...)` if it passes silently.

   **A skill without a runnable verifier is a *hint*, not a skill** — the
   runtime will present it as a card only and will not reject SUBMITs on
   its behalf.

4. `## Tripwires`
   3–5 bullets (was: 2–3 under `## Common failure modes`), each one sentence,
   naming historically observed failures. Reference specific instance IDs /
   runs where possible — concrete failures teach better than abstract
   warnings.

5. `## Invariants`
   A bullet list (or table) of every invariant the output must satisfy. Cover
   ranges, signs, monotonicity, cross-field consistency, and format
   constraints. Be concrete: "`Q2 >= 0`", not "Q2 should be reasonable". The
   verifier from §3 should encode each of these.

6. `## Procedure` *(optional but recommended-terse)*
   The actual playbook steps (parse → solve → verify → submit). **Only fill
   if the model genuinely needs procedural guidance** — leave terse, or omit
   entirely if §2 + §3 + §4 are self-explanatory. Step-by-step tutorial prose
   is the bloat the contract-first format is trying to remove. When present,
   steps must reference the verifier from §3 explicitly.

## Notes for authors

- Anything outside the required sections (additional `###` subsections
  inside them, `## Example self-test`, `## Strict final format`, etc.) is
  allowed but is not part of the contract. Add it only if it earns its
  keep — duplication is the enemy of precision.
- Keep the YAML-like metadata header (`# <name>`, `Summary:`, `Dependencies:`)
  intact at the top of the file, **after** the frontmatter block — the skill
  loader still parses those lines for back-compat. Frontmatter `depends_on`
  takes precedence over the legacy `Dependencies:` line when both are
  present.
- If a definition is genuinely ambiguous (e.g. the prompt is under-specified),
  state the assumption in §2 and make the §3 verifier *permissive* on that
  field rather than wrong. A loose-but-correct invariant beats a tight wrong
  one.

## Third-party libraries

Skills routinely need a library the package does not depend on — `python-docx`
for Word, `python-pptx` for PowerPoint, `markitdown` for document conversion.
Two things follow.

**The sandbox cannot install anything.** `subprocess` and `pip` are blocked by
the default `SecurityPolicy`, so a skill that opens with a runtime install
wastes a turn and returns:

```
SecurityPolicyViolation: call to 'subprocess.run' is disabled because
shell/subprocess execution bypasses the network and filesystem guardrails.
```

Install in the notebook environment instead, before the RLM runs, and say so in
the skill so the model does not try:

```python
%pip install python-docx
```

**Degrade explicitly rather than assuming.** Have the skill probe once and
branch, so a missing library produces a stated limitation instead of an empty
result that reads like a real answer:

```python
try:
    from docx import Document
    HAVE_DOCX = True
except ImportError:
    HAVE_DOCX = False
```

Where a standard-library path exists, give it. A `.docx` is a zip of XML, so
reading one needs no dependency at all — only writing does. A skill that offers
the fallback keeps working in environments where the install never happened.

Do not relax `SecurityPolicy.forbidden_calls` to permit the install. That
reopens network and filesystem access for LM-emitted code, which is a poor
trade for a file-format reader.

### Authoring rules (learned the hard way)

> **Rule: quote question text verbatim.**
> When defining output fields in §2, quote the question's exact wording in
> a `>` blockquote. Do **not** paraphrase. Paraphrasing introduces drift
> between what the playbook says and what the question asks. The B' MCM
> playbook had this bug on `Q4` ("longest matched-paren span" instead of
> the prompt's "longest distance between an opening and closing parentheses
> in number of matrices"), causing `MCM_hard_3` to regress in the v2
> signal pilot (`20260427-signal-pilot-mcm3-v2`).

> **Rule: don't pin a value where the question doesn't.**
> If a field's definition is structural (it depends on the shape of another
> output, not a closed form over the inputs), the §3 verifier should check
> **bounds** (range, type, sign) but NOT pin an exact formula. Reserve hard
> equality invariants for fields the question explicitly defines as a
> formula. Example: MCM `Q5 = (Q4 - Q3) * Q2` is question-pinned and safe
> to assert as `==`. MCM `Q4` is structural (depends on `Q1`'s parse tree),
> so the verifier asserts only `0 <= Q4 <= n`. A wrong tight invariant
> rejects correct answers; a loose correct invariant lets the model
> through to score normally.

> **Rule: ground-truth must pass.**
> Every `verify` function shipped in §3 MUST accept every labeled answer
> in the project's benchmark dataset. The
> `tests/test_skill_verifiers_against_groundtruth.py` regression test
> enforces this automatically — adding a new invariant that rejects any
> ground-truth row will fail CI before any Fabric run.
