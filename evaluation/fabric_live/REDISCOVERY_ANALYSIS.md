# Would a data-exploring `learn()` have cut turns?

**Question (user):** what if `learn()` actually explored the data and built
knowledge the agent can't otherwise have — "column x holds customer names as
First Last", "it has duplicates", "sales_amount has negatives, skewed right"?
What did RLM have to rediscover on every single question?

**Short answer:** the premise is correct and stronger than assumed. `learn()`
does not merely under-use the data — **it never reads a single row.**

But the trace evidence corrects the second half of the intuition. RLM does *not*
burn many turns rediscovering data facts; it goes almost straight to analytical
SQL. What it re-derives every time is the **catalog** (21/23). What it mostly
**never checks at all** is data quality — cardinality 6/23, ranges 3/23, nulls
2/23, no row sampling. So a data-exploring `learn()` would buy **correctness
more than speed**.

---

## 1. `learn()` reads no data — by construction

Both adapters that can serve a Fabric lakehouse profile from **metadata only**.

**`DeltaDirectoryAdapter`** (`knowledge_lakehouse_sources.py:241`) — the
docstring states the intent outright:

> *"Profile a local or mounted Delta directory from transaction metadata only.
> `DeltaTable(..., without_files=True)` intentionally avoids loading the active
> data-file set; only version, table metadata, and schema are used."*

**`LakehouseSourceAdapter`** (`:403`), the one used for
`abfss://…/da_agent_tests.Lakehouse/Tables`, consumes only `resolved.catalog` —
table names, column names, column types, and snapshot fingerprints. It never
opens a table.

And nothing anywhere computes a data statistic. Searching the whole package for
`null_count|distinct|nunique|min_value|max_value|skew|duplicate|histogram|quantile|row_count`
returns only:

- `_benchmark_manifest.py` — test-fixture tooling, not the runtime profiler
- `knowledge_execution.py:750` — `"row_count": len(self.rows)`, the size of an
  *operation result* at execution time, not a profile
- `knowledge_execution.py:232-252` — "must not contain duplicates" validation of
  *plan parameters*, not of data

So a `SourceProfile` carries `schema`, `diagnostics`, fingerprints and
`sensitive_columns` — and no distribution, cardinality, null, range, or format
information. **`learn()` structurally cannot know that `amount` has negatives or
that a join key fans out**, because it never looks.

This also explains F12 from the other direction. The one tabular lesson is
gated on a column being `type == "boolean"` with an English current-period
*name* — a rule expressible from schema alone. The lesson vocabulary is thin
because **the profile it is derived from is thin.**

## 2. What the agent re-derived on every question

> **v1 of this section was unsound and its numbers are withdrawn.** The probe
> vocabulary was written for pandas (`.shape`, `groupby().size()`, `.min()`) but
> the agent works almost entirely in **SQL**, so those patterns matched 0 times
> and produced false negatives. Worse, v1 counted every `COUNT(*)` as row-count
> *profiling* when it is overwhelmingly part of an analytical query
> (`COUNT(*) AS invoices_with_payment`). v1's headline — "row count 23/23, 133
> turns, the single clearest waste" — was **wrong**. Audit in
> `audit_probes.py`; corrected measurement in `analyze_rediscovery_v2.py`.

v2 counts a construct as orientation **only when it is standalone** — a
statement whose entire purpose is to inspect, with no `SUM`/`AVG`/`ROUND`/
`GROUP BY` alongside it. This deliberately under-counts rather than over-counts.

From the 23 captured trajectories of the GLM learn arm (the cold arm predates
trajectory capture, H8):

| standalone orientation construct | questions | share | turns |
|---|---|---|---|
| **catalog listing (`list_sources()`)** | **21 / 23** | **91%** | 21 |
| distinct cardinality probe | 6 / 23 | 26% | 11 |
| range (`MIN`/`MAX`) probe | 3 / 23 | 13% | 3 |
| null probe | 2 / 23 | 9% | 3 |
| bare row count (`SELECT COUNT(*) FROM t`) | **1 / 23** | 4% | 1 |
| sample rows (`SELECT * … LIMIT`) | 0 / 23 | 0% | 0 |

Turn budget over 228 captured turns:

| bucket | turns | share |
|---|---|---|
| pure orientation | 25 | **11%** |
| analytical | 86 | 38% |
| friction (traceback) | 38 | 17% |
| unclassified | 79 | 35% |

**The corrected finding is the opposite of the intuitive one.** RLM does *not*
burn many turns rediscovering data facts. It goes almost straight to analytical
SQL. The one thing it genuinely re-derives on nearly every question is the
**catalog listing — 21 of 23** — and six tables (`invoices`, `companies`,
`dim_date`, `features`, `industries`, `payments`) appear in **23/23**
trajectories because the catalog is re-listed and re-printed from scratch each
time. That is the only well-supported rediscovery cost, and it is schema-level,
which `learn()` *already* captures and evidently fails to hand over usefully.

Note also that **friction (17%) exceeds pure orientation (11%)**. On this
evidence, defects (F11/F14) cost more turns than orientation does.

## 3. The user's examples split into two very different cases

**(a) Turn savings — small, and mostly one thing.** Only the catalog listing
recurs broadly. A profile that eliminated it addresses ~21 turns and the bulk of
the repeated stdout, but the ceiling here is **11% of turns**, not a third.

**(b) Correctness, not speed — the larger prize.** Standalone data-quality
probes are *rare*: cardinality 6/23, ranges 3/23, nulls 2/23, and no sampling at
all. The agent largely **does not check** for fan-out, negatives, duplicates or
skew — it writes the aggregate and reports the number.

For this half the value proposition inverts. Caching those facts saves almost no
turns, because the agent wasn't spending turns on them. It would instead surface
hazards the agent is currently **silently not checking** — an unflagged fan-out
or an unnoticed negative is a *wrong answer*, not a slow one. On a lakehouse
where the brief explicitly asked about "joins that can multiply values", the arm
still produced confident answers with almost no distribution checking. **This is
where a data-exploring `learn()` would earn its keep.**

## 4. What this does and does not license

**Measured:** that the profiler reads no rows (source, §1); that the catalog is
re-listed in 21/23 questions; that standalone data-quality probing is rare.

**NOT measured — a hypothesis, stated as a ceiling.** I cannot claim a richer
profile removes those 25 orientation turns without running that arm. It would
recover less than all of them because a profile **enters the prompt** (it must
be smaller than the ~21 catalog dumps it displaces), because the agent may
re-verify anyway (correct behaviour against a stale profile), and because 17%
friction is untouched by any profile.

**And the honest headline: turn savings are not the case for data profiling
here — correctness is.**

## 5. Fix classification (per the brief)

| proposed knowledge | classification | rationale |
|---|---|---|
| row counts, null counts, distinct counts, min/max, join-key fan-out ratio, table grain | **universal mechanism** | pure statistics; no business concept. The profiler already visits every source — it just declines to read them. |
| "`sales_amount` is right-skewed with negative values present" | **universal mechanism** (the statistic) | detecting negatives/skew needs no domain knowledge |
| "negative `sales_amount` means a **return/credit**" | **optional domain skill** | that is an interpretation, not a measurement |
| "`customer_name` holds `First Last`" | **source metadata** (`declared=`) | a format assertion the author knows and the data only suggests |
| "column `xyz` is about …" | **source metadata** (`declared=`) | semantics belong to the author |
| widening `_CURRENT_PERIOD` with more English synonyms | **rejected** | puts a naming rule in core; this is the F12 mistake, not its fix |

The important line: **statistics are universal, meanings are metadata.** A
profiler that reports "`order_id` has 4,812 distinct values over 19,344 rows,
so joining on it fans out ~4×" is domain-neutral and belongs in core. One that
says "this looks like an invoice line" does not.

## 6. Reproduce

```bash
python audit_probes.py              # shows what each regex really matched
python analyze_rediscovery_v2.py    # the corrected, SQL-aware measurement
python analyze_rediscovery.py       # v1 — RETAINED ONLY as the retracted version
```

All three need `run_log_glm_learn.json` alongside. **Use v2.** v1 is kept in the
tree solely so the retraction is auditable, and its output should not be cited.

Source claims verified against the `pr-75` worktree at `bd924bc`:
`knowledge_lakehouse_sources.py:241,244,279,403,413`; `knowledge.py:507`
(`SourceProfile` fields).

## 7. The measurement that would settle it

A `declared=` + statistics arm: pre-compute the section-2 facts once, supply
them, hold model/questions/limits fixed, and compare turns and accuracy against
the 91.7% cold baseline. That directly tests both halves — whether orientation
turns fall, and whether the (b) facts prevent any wrong answers. **Not run**;
proposed, not claimed.
