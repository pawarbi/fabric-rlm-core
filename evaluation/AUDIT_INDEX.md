# fabric-rlm-core generalization audit — index and coverage

Base commit under test: **`b5226712`**. Evaluation branch:
`eval/generalization-b522671`. **Core frozen throughout** — verify with:

```bash
git diff b5226712 -- fabric_rlm    # must be empty
```

No core patches were made during this evaluation. Proposed fixes are described
and classified, never applied.

---

## Where to read

| Document | Covers |
|---|---|
| `evaluation/generalization/` + its report | Synthetic domains, naming robustness, source transfer (brief items 2–4) |
| `evaluation/dbo_lakehouse/DBO_EVAL_REPORT.md` | Phase 1: 25 questions on a real lakehouse, cold vs learned, run locally |
| `evaluation/fabric_live/PHASE2_REPORT.md` | Phase 2: the same questions run **inside Fabric** with RLM authoring its own Excel; findings F9–F14 |
| `evaluation/fabric_live/LEARN_GATE_VERDICT.md` | The `.learn` gate, measured within one model |
| `evaluation/fabric_live/GLM_ARM_RESULTS.md` | Raw model-comparison numbers |

---

## Coverage against the brief

### Measured

| Brief item | Status | Evidence |
|---|---|---|
| 1. Audit domain dependencies | **measured** | F12: the only tabular lesson is gated on English name tokens — `repro_learn_naming.py`, no credentials. Verified representative: the Delta profiler emits exactly `"boolean"`. |
| 3. Naming robustness | **measured for the learning path** | 5/12 semantically identical columns yield a lesson; abbreviations and non-English names yield none. |
| 5. What learning adds (A vs B) | **measured** | Arms A/B differ only by `knowledge=`: 91.7% vs 83.3%, +21% turns, +64% tokens (paired sign test p=0.0005 tokens / 0.0075 turns), package had **0 lessons but a live operation set**. Gate **FAILS**. Where an operation actually fired (3/24) learn was **3/3 and nearly free**; the regression is entirely on the 21 questions that paid a selection call and were declined. |
| 6. Failure handling | **partially measured** | F14 (3 of 4 result-bound violations abort the task instead of falling back), F11 (35 gate rejections over 7/24 questions **in the learn arm only** — the cold arm was uninstrumented, H8, so no cross-arm claim), F9 (silent artifact loss with `ok=True`, model-dependent). |
| 7. Correctness vs efficiency, separately | **measured** | Accuracy, turns, prompt tokens, gate rejections, per-question, per-arm. Abstentions and crashes counted as incomplete, not averaged away. |
| General execution capability | **measured** | 91.7% on 24 unseen complex questions against real Fabric Delta tables, 0 fan-out hazard traps. |

### Partially measured

| Item | What is missing |
|---|---|
| 4. Transfer across sources | Lakehouse/Delta exercised **live**. Files exercised in Phase 1. **Semantic model and SQL endpoint not exercised in Fabric.** |
| 5. Arm C (package enriched from separate development runs) | **Impossible as specified**: `knowledge=` and `inputs=` cannot bind the same alias (`ValueError`). Documented, not worked around. |
| 5. Three repetitions per configuration | Phase 1 has 3 reps. **Phase 2 arms are n = 1.** |
| F9 mitigation via the bundled `excel_modify` skill | Harness supports it (`--parameter SKILLS=excel_modify`); **arm not run**. |
| GLM no-workbook control | Run for gpt-4.1-mini only; **not run for GLM**. |

### Not tested — explicitly not claimed

- **SQL / Warehouse endpoint** as a distinct source type in Fabric.
- The three synthetic domains (inventory, manufacturing, service) **at Fabric
  scale** — they were exercised locally, not deployed as Delta tables.
- Stability across models beyond the two tested.

### Since closed

- ~~**Semantic-model sources**~~ — now tested. `RLM.learn()` on the live
  `ecommerce-dataset` yields 7 lessons, **all `candidate`, `active=0`**, all
  from one English regex. See `F15_SEMANTIC_NAMING.md`.
- ~~**`declared=`-supplied metadata**~~ — now tested as arm D. 32 active
  lessons; abstentions 2→0; turns and tokens **null** (sign test p=1.0). See
  `ARM_D_RESULTS.md`.

---

## Conclusions, separately as the brief requires

**1. General execution capability — demonstrated.**
22/24 (91.7%) on unseen complex questions over real Fabric Delta tables, zero
fan-out hazard traps, a machine-readable answer every time, plus a correctly
formatted three-sheet Excel deliverable maintained across 24 incremental
updates.

**2. Generalization of learned behaviour — not demonstrated; mechanism
identified.**
`learn()` on tabular/Lakehouse sources can emit exactly one lesson type, gated
on English column-name tokens, so meaning-preserving renames change what is
learned. On the real lakehouse it produced **0 lessons**, and the learn arm was
strictly worse: same reasoning quality, two extra failure modes, +64% tokens.
On a **semantic model** it produces 7 lessons, but all `candidate` / `active=0`,
and all from a single English regex that misses non-English names entirely
(0/7) while false-positiving on ordinary statistical English (8/8) — two of
those false positives occurred on the live model.

The library is **not broadly ARR-specialized** — no measure name, business
concept or source type from the development domain drives a runtime decision —
with one concrete exception: `nrr` and `grr` (Net/Gross Revenue Retention) are
subscription-metric tokens hard-coded in a runtime core regex. The dominant
coupling is **naming-convention and English-language specialization**, which is
broader and more mundane than ARR coupling.

**3. Portability across tested sources — confirmed where tested.**
Delta/Lakehouse in real Fabric with real credentials, and local files. An
earlier claim that portability "breaks in Fabric" is **withdrawn**: it compared
local against Fabric while also changing the model. Holding environment fixed
and varying only the model gives 33.3% vs 91.7% — **model choice dominated every
metric measured.**

**4. Remaining unsupported claims.**
Everything in "Partially measured" and "Not tested" above, plus: n = 1 for all
Phase-2 arms; the Python 3.11→3.12 difference is uncontrolled (though it cannot
affect the within-Phase-2 comparisons, which are the load-bearing ones); and no
claim is made that `excel_modify` mitigates F9.

---

## Fix classification summary

Universal mechanisms (no domain knowledge, safe for core):

- **F11** — distinct actionable message per rejection path; stop matching `--`
  inside string literals.
- **F12** — report learning coverage so an empty package is not silent.
- **F14** — a truncated operation result should fall back to the raw sources,
  as the equivalent `OperationResultTooLarge` already does.

Belongs in **source metadata** (`declared=`), not core:

- The meaning of any particular column name.

**Rejected** as a core change:

- Adding more English synonyms to `_CURRENT_PERIOD` — that puts an unbounded
  naming vocabulary into core and keeps the English assumption.

Needs evidence before any core change:

- **F9** — model-dependent; test the bundled `excel_modify` skill first.
