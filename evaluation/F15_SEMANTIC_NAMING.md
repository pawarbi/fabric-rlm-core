# F15 — Learned behaviour on a semantic model is gated by English measure names

**Status:** confirmed, live + local repro
**Arm:** no-LLM probe (`RLM.learn()` only; profiling makes no model calls)
**Source:** `ecommerce-dataset` semantic model, workspace `82ad2591-…`
**Core frozen:** yes — this is an observation, no core change proposed here

---

## 1. What the probe answered

The report previously listed semantic-model `learn()` as *untested, not claimed*.
It is now tested. The probe binds the real semantic model and calls bare
`RLM.learn()` with no `declared=`, no key, and no LLM.

```
SemanticModel built: SemanticModel('ecommerce-dataset')
LESSONS total=7 active=0
by_basis={'name_pattern': 7}
```

Two facts follow immediately.

**(a) Semantic models do produce lessons where Delta produced none.** Arm B on
the `dbo` lakehouse produced `lessons_total: 0`. Here `learn()` produces 7. This
is consistent with F12: the lakehouse profiler reads no rows and a Delta table
declares no measures, so there is nothing to pattern-match on. A semantic model
carries measure names, and measure names are the entire input.

**(b) All 7 are `candidate`, none `active`.** So bare `learn()` still contributes
**nothing that reaches an agent** — active lessons are what get retrieved.
The gate verdict is unchanged by this probe. What changed is the *reason*:
on Delta there was nothing to learn from; on a semantic model there is, but it
is nominated at low confidence and parked pending evidence.

## 2. Every lesson came from one regex

All 7 lessons are `kind='context_requirement'`, `basis=('name_pattern',)`,
`confidence='low'`, produced at `knowledge_lessons.py:320`:

```python
derived = [str(name) for name in measures
           if _DERIVED_TIME_MEASURE.search(_leaf(str(name)))]
```

`_DERIVED_TIME_MEASURE` (`knowledge_lessons.py:54-58`):

```
previous | prior | py | pp | yoy | qoq | mom | growth |
nrr | grr | retention | churn | change |
delta | variance | vs | last (year|quarter|month|period) |
ttm | ltm | ytd | qtd | mtd
```

The measures it fired on in the live model: `Revenue PY`, `Revenue YTD`,
`Revenue YoY Pct`, `Orders PY`, `Orders YoY Pct`, `Avg Delivery Variance`,
`Scenario Variance`.

## 3. Behavioural effect — demonstrated, not asserted

**Scope this claim before reading the numbers.** All 7 lessons were
`status='candidate'`, and `active=0`. Candidates are not retrieved, so **no
lesson in this probe was observed influencing any answer.** What the regex
decides is *which candidates exist* — a gate on what learning is even capable of
proposing for a semantic model. That is the effect measured below. It is a real
dependency in runtime core, and it is not a demonstration that an answer changed.

`_is_derived_measure` is a pure function, so the dependency is directly
measurable. Repro: `evaluation/generalization/repro_measure_naming.py`.

| set | n | fires | meaning |
|---|---|---|---|
| A. English period-comparison measures | 10 | **10/10** | correct |
| B. Same measures, non-English names | 7 | **0/7** | **missed** |
| C. Period columns, non-English (`_PERIOD_COLUMN`) | 7 | **0/7** | **missed** |
| D. English names that are *not* period comparisons | 8 | **8/8** | **false positive** |

Set B is not a trick: `Umsatz Vorjahr`, `Ingresos Año Anterior`,
`Chiffre d'affaires N-1`, `Ricavi Anno Precedente` are the *same measures* with
the same DAX semantics and the same relationships. Meaning is preserved; only
the naming convention changed. Learned behaviour drops to zero.

Set D is not hypothetical either — **two of its members were observed live**:
`Avg Delivery Variance` and `Scenario Variance` are in `ecommerce-dataset`, and
`learn()` asserted of both that they `require: ('period_context',)` with
`fallback_strategy: 'explicit_period_base_measure'`. A delivery-time variance
is a dispersion statistic, not a period comparison. **2 of 7 live lessons
(29%) are wrong**, and they are wrong for a reason the mechanism cannot detect:
`variance` and `delta` are ordinary statistical English.

## 4. `nrr` and `grr` are ARR vocabulary in runtime core

The audit brief said finding "ARR" in a comment is not evidence. This is not a
comment. `nrr` (Net Revenue Retention) and `grr` (Gross Revenue Retention) are
subscription/SaaS metrics, hard-coded into a runtime regex in core that decides
which lessons exist. `retention` and `churn` are the same family.

This is the clearest instance found of the development domain leaking into
runtime behaviour. It is narrow — per §3 it changes which *candidates* are
nominated and no candidate was observed reaching an agent in this probe — but it
is real, it is in core, and it is exactly the class of rule the brief says
belongs elsewhere.

## 5. Honest counterweight

The library does not oversell this. `basis=('name_pattern',)` renders to the
agent as **"name pattern only"** (`knowledge_retrieval.py:106`), confidence is
`low`, and status is `candidate` — so a name-derived guess is never presented as
established, and the false positives in §3 never reached an agent in this probe.
The docstring at `:270` states plainly that the role "was read from the name".

So the defect is not dishonesty. It is **coverage**: a model named in any
language other than English learns nothing at all, and the failure is silent —
indistinguishable from a model that genuinely has no derived measures.

## 6. Proposed fix — classification

Per the brief, each fix is classified as universal mechanism, source metadata,
or optional domain skill. **No core patch is made during this evaluation.**

| # | change | class | rationale |
|---|---|---|---|
| F15-1 | Move `nrr`, `grr`, `retention`, `churn` out of core into an optional SaaS/subscription domain skill | **optional domain skill** | Business vocabulary of one industry. Nothing universal about them. |
| F15-2 | Let a source declare its derived measures / period columns explicitly, so nomination does not depend on language | **source metadata** | The owner knows which measures are period-relative. This is the same channel arm D uses, and it is language-neutral. |
| F15-3 | Prefer structural signals over names — a measure whose DAX references a time-intelligence function (`SAMEPERIODLASTYEAR`, `DATEADD`, `TOTALYTD`, `PARALLELPERIOD`) is period-relative regardless of its name | **universal mechanism** | DAX function names are invariant across languages and are already available in the model definition. This subsumes most of set A, fixes all of set B and C, and eliminates set D — `Avg Delivery Variance` references no time-intelligence function. |
| F15-4 | Report a "no derived measures detected" signal so silence is distinguishable from absence | **universal mechanism** | Silent zero-coverage is the part a user cannot debug. |

F15-3 is the substantive one: it replaces a language-dependent heuristic with a
language-independent structural one, and it is strictly more accurate on the
live model — it would have produced 5 correct lessons instead of 5 correct and
2 wrong.

## 7. Reproduction

```bash
# live probe (Fabric; no LLM key required)
python notebook.py update  --workspace-id 82ad2591-974a-4ad4-ace6-e24879274a4b \
    --notebook-id 483aad00-f008-4280-a184-7aba03066f00 \
    --from-file probe_semantic_learn.ipynb --name "Env Probe 312"
python notebook.py execute --workspace-id 82ad2591-974a-4ad4-ace6-e24879274a4b \
    --notebook-id 483aad00-f008-4280-a184-7aba03066f00 --wait
python read_files.py "Files/rlm_excel_eval/probe-semantic-learn.log"

# local regex repro (no Fabric, no key)
python evaluation/generalization/repro_measure_naming.py
```

Expected: `LESSONS total=7 active=0`, `by_basis={'name_pattern': 7}`;
table in §3 reproduced exactly.
