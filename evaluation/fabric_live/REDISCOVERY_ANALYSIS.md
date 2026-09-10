# Would a data-exploring `learn()` have cut turns?

**Question (user):** what if `learn()` actually explored the data and built
knowledge the agent can't otherwise have — "column x holds customer names as
First Last", "it has duplicates", "sales_amount has negatives, skewed right"?
What did RLM have to rediscover on every single question?

**Short answer:** the premise is correct and stronger than assumed. `learn()`
does not merely under-use the data — **it never reads a single row.** And the
traces show the agent re-deriving the same question-independent facts on
essentially every question.

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

From the 23 captured trajectories of the GLM learn arm
(`analyze_rediscovery.py`; the cold arm predates trajectory capture, H8).
Probes are matched against the executed **code**; friction against **stdout**.

| question-independent fact | questions | share | turns spent |
|---|---|---|---|
| **row count** | **23 / 23** | **100%** | **133** |
| **catalog (`list_sources()`)** | **21 / 23** | **91%** | 21 |
| join fan-out / grain | 11 / 23 | 48% | **48** |
| distinct / cardinality | 10 / 23 | 43% | 33 |
| null checks | 8 / 23 | 35% | 12 |
| sample rows | 5 / 23 | 22% | 7 |
| schema / dtypes | 4 / 23 | 17% | 5 |
| negative-value checks | 3 / 23 | 13% | 9 |
| duplicate checks | **0 / 23** | 0% | 0 |
| range / min-max | **0 / 23** | 0% | 0 |

Turn budget across 228 captured turns:

| bucket | turns | share |
|---|---|---|
| orientation (question-independent) | 79 | **35%** |
| analysis (question-specific) | 87 | 38% |
| friction (traceback / exception / gate) | 40 | 18% |

Orientation consumed **249,643 bytes of stdout**. Six tables — `invoices`,
`companies`, `dim_date`, `features`, `industries`, `payments` — appear in
**23/23** questions, because the catalog is re-listed and re-inspected from
scratch every time.

**The single clearest waste: 133 row-count turns across 23 questions to
establish a fact that is identical every time and could have been computed once.**

## 3. The user's examples split into two very different cases

This distinction matters more than the totals, and the data forces it.

**(a) Facts the agent *does* re-derive → a profile would buy turns.**
Row counts (100%), catalog/schema (91%), join fan-out (48%), cardinality (43%),
nulls (35%). These are recomputed constantly and are question-independent.

**(b) Facts the agent *never* checks → a profile would buy correctness, not
speed.** Duplicate checks and min/max ranges were probed in **0 of 23**
questions. Negative values in only 3 of 23 — on a lakehouse where the brief
explicitly asked about "joins that can multiply values" and returns/credits.

For (b) the value proposition inverts: caching them saves no turns, because the
agent wasn't spending turns on them. It would instead surface a hazard the agent
is currently **silently not checking** — an unflagged fan-out or an unnoticed
negative is a *wrong answer*, not a slow one. Given the arm still produced
confident answers on those questions, (b) is arguably the higher-value half.

## 4. What this does and does not license

**Measured:** what was re-derived, how often, at what turn and byte cost, and
that the profiler cannot supply any of it.

**NOT measured — this is a hypothesis, stated as an upper bound.** I cannot
claim a richer profile removes those 79 orientation turns without running that
arm. Three reasons it would recover less than 100%:

1. A profile **enters the prompt**. It replaces ~250 KB of orientation stdout,
   but only if it is smaller than what it displaces — an unbounded profile over
   21 tables could easily cost more than it saves.
2. The agent may re-verify anyway. Nothing forces it to trust a supplied fact,
   and for a *stale* profile re-verification is correct behaviour.
3. 18% of turns are friction (F11/F14), which a profile does not address.

The honest framing: **35% of turns went to facts a profile could in principle
supply**, and that is the ceiling, not the expected gain.

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
python analyze_rediscovery.py     # needs run_log_glm_learn.json alongside
```

Source claims verified against the `pr-75` worktree at `bd924bc`:
`knowledge_lakehouse_sources.py:241,244,279,403,413`; `knowledge.py:507`
(`SourceProfile` fields).

## 7. The measurement that would settle it

A `declared=` + statistics arm: pre-compute the section-2 facts once, supply
them, hold model/questions/limits fixed, and compare turns and accuracy against
the 91.7% cold baseline. That directly tests both halves — whether orientation
turns fall, and whether the (b) facts prevent any wrong answers. **Not run**;
proposed, not claimed.
