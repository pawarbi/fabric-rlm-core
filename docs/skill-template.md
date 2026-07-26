---
applies_when:
  keywords: []          # TODO: words/phrases that, when present in the question, should activate this skill.
  output_fields: []     # TODO: output keys this skill knows about (e.g. ["Q1","Q2"]).
excludes: []            # TODO: skills that conflict with this one and must not be co-loaded.
depends_on: []          # TODO: skills whose bodies this one builds on; may be empty.
specificity: domain     # one of: core | domain | utility
---
# <skill_name>
Summary: STARTER template — copy this file, rename, and fill in every TODO.
Dependencies: none
> **This is a STARTER.** Copy this file to `fabric_rlm/skills/<new_skill>.md`,
> rename the title above, fill in the YAML frontmatter, and fill in every
> TODO. After authoring, run
> `pytest tests/test_playbook_contract.py`
> to verify the contract. See `PLAYBOOK_CONTRACT.md` for the full contract and
> `pdf_document_analysis.md` for a worked example.

## Purpose

<!-- TODO: ONE sentence. When does this skill apply? What task class? -->

## Contract: output fields

<!--
TODO: For each required output key, provide a block of the following shape.
Quote the question's exact wording in a `>` blockquote — do NOT paraphrase.
Then state the operational definition with the canonical formula or procedure.
-->

- **`<field_name>`** — *<type>*. <!-- TODO: one-line definition -->.

  > <!-- TODO: verbatim question text quoted from the prompt / benchmark spec -->

  Canonical formula or procedure: <!-- TODO: closed-form formula, or step-by-step
  computation. Cite the source (prompt line, paper, benchmark spec). -->

<!-- TODO: repeat the block above for each required output field. -->

## Required verifier

The model MUST run this verifier on the computed `solution` payload before
emitting `SUBMIT(...)`. If `verify` raises, fix the offending field and
re-verify. Only submit when `verify` returns silently.

> A skill without a runnable `verify(payload) -> None` function is a *hint*,
> not a skill — the runtime will only present it as a card and will never
> reject SUBMITs on its behalf.

<!-- See `pdf_document_analysis.md` for a worked example. -->

```python
def verify(payload):
    """Raise AssertionError on any invariant violation.

    TODO: Replace this stub with concrete invariant checks. Each `assert`
    should carry a clear message naming the offending field. Prefer
    bounds-only checks for structural fields; reserve strict equality for
    fields the question pins to a closed form.
    """
    raise NotImplementedError(
        "TODO: implement verify() — see PLAYBOOK_CONTRACT.md and pdf_document_analysis.md"
    )
```

## Tripwires

<!-- TODO: 3-5 bullets, each one sentence, naming historically observed
failures. Reference specific instance IDs / runs where possible —
concrete failures teach better than abstract warnings. -->

- <!-- TODO: tripwire 1 -->
- <!-- TODO: tripwire 2 -->
- <!-- TODO: tripwire 3 -->

## Invariants

<!-- TODO: bullet list of every invariant the output must satisfy.
Cover ranges, signs, types, monotonicity, cross-field consistency, format.
Be concrete: "`Q2 >= 0`", not "Q2 should be reasonable". The verifier above
should encode each of these. -->

- <!-- TODO: invariant 1 -->
- <!-- TODO: invariant 2 -->
- <!-- TODO: invariant 3 -->

## Procedure

> **Tutorial: only fill if the model genuinely needs procedural guidance.**
> Leave terse — three to five bullets — or omit entirely if the verifier and
> contract above are self-explanatory. Long step-by-step prose is the bloat
> we are trying to avoid.

1. <!-- TODO: parse / extract inputs from the prompt -->
2. <!-- TODO: solve / compute each output field -->
3. **Verify**: call `verify(solution)` from the Required verifier section. If
   it raises, repair the offending field and call `verify` again.
4. <!-- TODO: emit final output via `SUBMIT(...)` in the strict format
   required by the benchmark -->
