# Lakehouse `dbo` Evaluation — 25 Complex Questions, Cold vs. Learned

**Library under test:** `fabric-rlm-core` @ `bd924bc` (PR #75 branch), core frozen — no patches applied during this evaluation.
**Source:** `abfss://sandeep_ws@onelake.dfs.fabric.microsoft.com/da_agent_tests.Lakehouse/Tables/dbo` (real Fabric OneLake, live read).
**Model:** `openai/gpt-4.1-mini`, temperature 1.0, `cache=False`, `max_turns=12`, `timeout=300`, `skills=[]`, autoloading off — identical across arms.
**Design:** 25 questions × 2 arms × 3 repetitions = **150 trials, 0 crashes, 0 harness errors.**

---

## 0. Read this first — the domain caveat

This lakehouse is a **SaaS subscriptions / MRR / usage** dataset. That is the library's *development domain*.

**This evaluation therefore measures home-turf competence and `.learn` lift. It is not evidence of cross-domain generalization.** The cross-domain evidence lives in the separate three-domain study (inventory, manufacturing, service ops). Nothing below should be quoted as a generalization result.

---

## 1. Headline: the learning gate FAILS on this dataset

The pre-declared gate was: **`.learn` accuracy ≥ cold, in fewer turns or on novel/complex questions.**

Excluding q23 (defective reference — see §3), 24 questions, 144 trials:

| Metric | Arm A (cold) | Arm B (learned) | |
|---|---|---|---|
| **Analytic correct** | **61/72 (84.7%)** [75–91] | 53/72 (73.6%) [62–82] | **FAIL** |
| Contract correct | **61/72 (84.7%)** | 53/72 (73.6%) | FAIL |
| Hit the named hazard | 3/72 | 2/72 | — |
| Abstained | 0/72 | 2/72 | — |
| **Not machine-readable** | 1/72 | **10/72** | **10× worse** |
| Errors / crashes | 0 | 0 | — |
| Mean turns | 5.04 | **4.83 (−4.1%)** | PASS |
| **Mean prompt tokens** | 9,160 | **16,410 (+79.1%)** | **regression** |
| Mean wall seconds | 19.4 | 35.6 | regression |

**GATE: FAIL.** Accuracy fails; the fewer-turns half passes but does not rescue it.

Wilson intervals overlap (75–91 vs 62–82), so the accuracy drop is **suggestive, not statistically conclusive** at n=72 per arm. The mechanism evidence in §2 is what makes it credible, not the interval.

**9 regressions** (q01, q02, q03, q07, q08, q12, q17, q18, q25) vs **4 improvements** (q09, q11, q21, q24). Reported per-question so the average cannot hide them.

### This contradicts the earlier gate, which PASSED

The earlier generalization gate passed with 3–5 sources. This package has **46 lessons across 10 sources**. The most likely difference is **scale of the knowledge package**, not domain. Reported as measured correlation; §2 supplies the causal mechanism.

---

## 2. Mechanism: package-supplied aggregates short-circuit reasoning

This is the finding of the evaluation, and it is a **universal mechanism, not a domain rule.**

### 2.1 The confound was ruled out first

The obvious objection: arm B ran with `inputs={}`, so maybe it just had less data access. Two checks kill that.

**(a) The library forbids the control arm.** I attempted arm C (package **and** sources). All 18 trials failed identically:

```
ValueError: task inputs conflict with knowledge source aliases:
  companies, dim_date, features, industries, invoices, payments,
  subscriptions, support_tickets, usage_logs, users
```

You **cannot** pass a knowledge package and inputs for the same aliases. Arm B is therefore not a handicap I imposed — it is **the only learned configuration the library permits**. The package *is* the data-access path.

**(b) Access was measurably equivalent.**

| | Arm A | Arm B |
|---|---|---|
| Emitted any SQL | 75/75 | 72/75 |
| SQL references a real table | 65/75 (**87%**) | 69/75 (**92%**) |
| Correct *when it reached the data* | 51/65 (**78.5%**) | 51/69 (**73.9%**) |

Arm B reached the data **more** often and was still worse. The deficit survives conditioning on access. **It is behavioural.**

### 2.2 The behaviour

**7 of arm B's 22 wrong answers explicitly cite a precomputed aggregate from the knowledge package. Arm A does this 0 times** (it has no package to cite). Verbatim:

- **q17** — `"Used the host-provided aggregate avg(is_active) from the users source"` → returned `0.7824`. Reference `78.2413`. The package supplied the proportion; the model returned it as the answer and **skipped the ×100 the question asked for**, while labelling `units: "percentage"`. Cold got it right 3/3.
- **q02** — `"The sum of amount_due for overdue invoices is known to be 71379.94 from the provided knowledge_result"` → returned the package's `amount_due` total instead of `amount_due − amount_paid`. **The package answered a simpler question and the model accepted it.**
- **q18** — `"Precomputed knowledge_result amount_due was None, unable to retrieve total amount_due from invoices table without DB access."` The model **believed it had no data access** because the package was present. It did. It never tried.
- **q23** ×3 — used the package's `sum(api_calls)` against a hand-built denominator.

**The pattern:** the package presents a profile-derived aggregate that is *close to* what was asked. The model substitutes it for the final answer and stops reasoning about grain, units, or the remaining transformation.

**Why this is universal, not domain-specific:** these aggregates come from *profiling*, not from my `declared` metadata. Any source, any domain, produces them. Nothing about SaaS or MRR is involved.

**Proposed fix — universal mechanism (not a core domain patch):** profile-derived aggregates should be labelled as *inputs to a computation*, not as candidate answers, and the answer contract should require a unit/grain reconciliation step against the question before a package-supplied scalar may be emitted. This belongs in the answer contract, **not** in source metadata and **not** in a domain skill.

### 2.3 Contract-formatting regression is F9 recurring

`not machine-readable`: **1/72 (A) → 10/72 (B)**. Same defect as the previously-filed F9, at **10× the rate under learning**. Not a new finding — an amplification of a known one. Longer prompts appear to degrade contract adherence.

Corroborating: `status` came back as `success`, `ok`, `OK`, `SUCCESS` across trials, and `units` as both `percentage` and `percent`. The contract is not being normalised.

---

## 3. Reference defects — full disclosure

I audited every question for the signature that exposed q23: **all trials in both arms converging on the same non-reference value.**

**q23 — WITHDRAWN. My reference was wrong; the library was right.**
"Mean api_calls per ACTIVE user." My SQL divided *all* `usage_logs.api_calls` by *active-user count* — **mismatched numerator and denominator populations** (ref 11,173.96). The model consistently returned usage *of active users* ÷ active users, the better reading. Excluded from all scoring above.

> **Dual-engine agreement did not protect me.** pandas and DuckDB agreed because *I authored both encodings of the same misreading*. Cross-engine validation catches implementation bugs, not specification errors. **0/6 across both arms is the tell** — when both arms fail every repetition identically, suspect the reference first.

**Audit result: no other reference defects.** Only q20 and q23 were 0/6; the other 23 references were reproduced by at least one trial, so they are demonstrably reachable.

---

## 4. q20 — three separate findings, not one accuracy loss

q20 is 0/6, but **my reference (20.8027) is correct**. It fails for three unrelated reasons that must not be pooled into the learning comparison:

**(a) Empty result reported as a confident zero — 4/6 trials, BOTH arms.** `WHERE s.status = 'ACTIVE'` (uppercase) against lowercase data → zero rows → returned `0` with `status: "success"`. Arm B rep1 states it outright: *"There are no active subscriptions, so the percentage of MRR from enterprise companies is 0."*

This is exactly the **empty-results failure mode from the original brief**: an empty result surfaced as a confident answer instead of an abstention. It is **arm-independent** — a genuine library finding, unrelated to learning.
*Proposed fix (universal):* an aggregate over zero input rows must not be emitted as a scalar result; it must raise an explicit empty-result condition. Case-insensitive matching is **not** the fix — the fix is not answering confidently from nothing.

**(b) Scale error — arm A rep1.** Exactly correct SQL, returned `0.208` while declaring `units: "percentage"`. The answer **contradicts its own units field** — mechanically detectable, and the same defect class as q17. Same for q24 rep2 (`0.383` vs `38.2957`).

**(c) Silent proxy substitution — arm A rep0.** *"Assuming MRR proportional to active subscription counts"* — answered a **different question** (subscription-count share, not MRR share) and reported `status: "ok"` with no uncertainty flag. An unsupported assumption presented as a verified result.

---

## 5. What learning did help with

Not everything regressed. **4 improvements: q09, q11, q21, q24.** And turns fell 4.1%. The package genuinely helps when the requested quantity *matches* a lesson's grain and units — the harm is concentrated in questions needing a **transformation on top of** a package aggregate.

That is the actionable shape of the result: **learning helps on lookup-shaped questions and hurts on transform-shaped ones.**

---

## 6. Honest status

| Claim | Status |
|---|---|
| Library executes reliably against real Fabric OneLake Delta | **Demonstrated** — 150/150 trials completed, 0 crashes |
| Cold accuracy on 24 screened complex questions | **Measured** — 84.7% [75–91] |
| `.learn` improves accuracy | **Refuted on this dataset** — 73.6% [62–82], gate FAIL |
| `.learn` reduces turns | **Supported** — −4.1% |
| `.learn` reduces tokens | **Refuted** — +79.1% prompt tokens |
| Package aggregates cause substitution errors | **Demonstrated** — 7 verbatim citations, 0 in cold |
| Access asymmetry explains the deficit | **Ruled out** — arm B reached data more often (92% vs 87%) |
| This shows cross-domain generalization | **NO** — development domain; see §0 |
| Configuration C (enriched from dev runs) | **Never built** — not measured |

---

## 7. Reproduction

```powershell
$env:OPENROUTER_API_KEY = "<key>"
$env:PYTHONPATH = "<...>\fabric-rlm-core-pr75"
cd <...>\files\dbo_eval

python profile_tables.py                 # read 20 Delta tables from OneLake, cache parquet
python ground_truth.py                   # 25 questions; exits non-zero unless 25/25 agree + discriminate
python run_dbo_eval.py --smoke           # 6 trials
python run_dbo_eval.py --arms A,B --repetitions 3 --max-live-calls 400
python analyze.py dbo_eval_results.json  # gate verdict + per-question table
python audit_references.py               # reference-defect audit
python access_check.py                   # access-equivalence check
python diagnose.py                       # scale-error + package-citation anatomy
```

**Deliverables:** `dbo_eval_report.xlsx` (3 sheets: data used / question-answer-SQL-reasoning / per-question evidence, written incrementally after every question), `dbo_eval_results.json` (150 raw trials), `ground_truth.json`, `armC_control.json` (the ValueError proof).

---

## 8. Conclusions

**General execution capability.** Solid. 150/150 trials completed against live OneLake Delta with no crashes, 84.7% cold accuracy on questions specifically screened to be non-degenerate and to have a plausible wrong path. The library reads real Fabric data and computes correctly most of the time.

**Generalization of learned behavior.** **Not established — and on this dataset, learning is net harmful.** With 46 lessons over 10 sources, accuracy fell 11 points, prompt tokens rose 79%, and contract violations rose 10×. The cause is identified and is a **universal mechanism**: profile-derived aggregates are presented in a form that invites substitution as the final answer. The earlier passing gate had a far smaller package, which suggests this is a **scaling property of knowledge packages**.

**Portability across tested sources.** Only one source type was exercised here (OneLake Delta via cached parquet). **No cross-source claim is made from this run.**

**Remaining unsupported claims.** Cross-domain generalization (wrong dataset for it); configuration C; any claim that `.learn` reduces cost.

**Fix classification.** Every proposed fix above is a **universal mechanism** in the answer contract — aggregate-provenance labelling, unit/grain reconciliation, empty-result handling, status/units normalisation. **None** is a domain rule, and **none** was applied during this evaluation. Core remains frozen.
