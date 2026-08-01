---
applies_when:
  keywords:
  - why did
  - why are
  - why is
  - root cause
  - variance
  - declined
  - dropped
  - went down
  - what is driving
  - driving the
  - churn
  - kpi
  - scrap rate
  - defect rate
  - explain the change
  - explain the decline
  - driver analysis
  output_fields:
  - report
  - drivers
  - findings
excludes: []
depends_on:
- data_exploration
specificity: domain
---
# driver_analysis
Summary: Diagnosing why a metric moved: premise validation before explanation, an explicit gap against a stated baseline, four decomposition lenses (dimension, rate x volume, mix, population), quantified contributions that reconcile to the gap, and a report contract with a mandatory residual and a mandatory "not checked" section.

A method for answering "why did this metric move?" with numbers instead of
narrative. Domain generic: revenue, scrap rate, churn, conversion, margin,
on-time delivery and cycle time all decompose the same way, only the column
names change. Load it alongside `data_exploration`, which covers querying
files too large for context. This playbook covers what to compute and in
what order.

The deliverable is a report in which every named driver carries a signed,
quantified contribution, the contributions plus an explicit residual sum to
the total change, and every figure states the computation that produced it.
An unquantified driver is a hypothesis and must be labeled as one.

## Phase 0: prove the move is real

Do this before any decomposition. Data manufactures false moves in five
recurring ways, and finding one ends the task with a complete answer:

- Partial period. A month that is 20 days old is not down 30 percent, it is
  unfinished. Compare complete periods or equal days elapsed.
- Late-arriving data. Check the latest partitions landed: row counts by load
  date, max timestamp per source.
- Definition or pipeline change. Renamed codes, a new upstream filter, unit
  or currency changes, duplicate or missed loads. Compare totals across the
  boundary of the change.
- Calendar effects. Trading days, holiday shifts, leap days, fiscal versus
  calendar periods. Normalize per available day when periods differ.
- Normal variation. Compute historical period-to-period swings first. A move
  inside ordinary noise or the usual seasonal dip is the finding itself.

If the premise fails, SUBMIT a report saying the metric did not really
move, with the evidence. Do not invent drivers for an artifact.

## Phase 1: pin the metric

State source table, formula, filters, grain and unit before computing. If a
metric definitions source exists among the inputs (a semantic model export,
an internal metrics file), use its definition and cite it. Recompute the
headline number once from the raw grain; never inherit it from an aggregate
you cannot verify.

## Phase 2: frame the gap

Name the baseline (prior period, same period last year, plan) and why.
Gap = actual - baseline, signed. Every later number allocates this one.

For a trending or seasonal metric, a naive prior-period baseline misstates
the gap. Build the expectation from the pattern: trend continuation, or
same period last year adjusted for the recent growth rate. A metric that
grew 10 percent a year and is now flat has a real gap at zero
period-over-period change; a business that dips every July has no July gap
to explain. State in the report how the expectation was built.

## Phase 2b: locate the move in time

Compute the metric as a series at a finer grain than the comparison (daily
or weekly under a monthly gap) and classify the shape before decomposing,
because the shape says what kind of cause to look for:

- Step change: find the breakpoint (the split maximizing the difference in
  window means is enough) and hunt for events dated at it. A step with no
  matching event is a Phase 0 signal worth rechecking.
- Gradual drift: no single break. Points at accumulating causes (mix
  shift, churn outpacing acquisition, price erosion, wear). Hunting a
  dated event on a drift misleads.
- Single-period dip or spike: outage, stockout, one-off order, or a data
  artifact. Check persistence before treating it as a trend.
- In line with the metric's own trend or seasonality: a Phase 0 finding,
  not a gap to decompose.

Carry two timing checks into the drill-down: simultaneity (segments
breaking together implies a common cause; staggered breaks imply a rollout
or spread) and persistence (state whether the latest data is worsening,
stable, or recovering).

## Phase 3: decompose through four lenses

1. Dimensional contribution. Per segment of each candidate dimension:
   contribution = actual - baseline. For additive metrics the contributions
   sum to the gap; assert this in code. Scan several dimensions and keep the
   ones where the gap concentrates.
2. Rate x volume. For product metrics (price x quantity, rate x uptime,
   conversion x traffic): delta_M = delta_rate * volume_base +
   delta_volume * rate_base + delta_rate * delta_volume. Report the
   interaction term separately.
3. Mix versus within. A ratio can move with no segment changing because the
   weights shifted. Split delta_R into sum(delta_w_i * r_i_base) plus
   sum(w_i_actual * delta_r_i). Always check mix before attributing a ratio
   move to performance.
4. Population. Split entities into retained, new and lost. Like-for-like
   change plus new entrants minus lost baseline equals the gap. Losing two
   accounts is a different problem from every account shrinking.

Then rank segments by absolute contribution, report how much of the gap the
top few cover, drill concentrated segments one level deeper, and look for a
dated coincident event. Date alignment is association; call it association
unless the mechanism is confirmed, and say what would confirm it. Report
offsetting movements: a -100 gap built from -150 and +50 has two findings.

## Phase 4: reconcile

Named drivers plus an explicit residual must sum to the gap. Report the
residual as its own line. A residual above roughly 20 percent of the gap
means the analysis is incomplete; decompose further or say so plainly.
Never stretch driver estimates to make the total tidy.

## Computation rules

- Never average ratios across segments; recompute from numerator and
  denominator at the target level.
- Identical filters and definitions on actual and baseline. A filter that
  moved between periods is itself a Phase 0 finding.
- All contributions against the same baseline; mixed baselines cannot
  reconcile.
- Signs stay consistent: drivers of a decline are negative, offsets
  positive.
- Reconcile on unrounded values; round only for presentation.

## Semantic model sources

When the metric lives in a Power BI or Fabric semantic model (queried via
sempy or a Fabric IQ tool) the method holds but the mechanics shift. Check
the model's last refresh before anything: a stale refresh manufactures
declines. Reproduce the filter context the user saw, including report and
visual filters and row-level security, and state it beside every number.
The metric definition is the measure's DAX; quote it rather than
re-deriving from the fact table. Decompose by evaluating the measure
grouped by dimension columns from the model's own dimension tables, not by
exporting fact rows. Test additivity before reconciling: ratios, distinct
counts and semi-additive balances do not sum across segments, so decompose
their numerator and denominator measures separately or work like-for-like.

## Report contract

The submitted report must contain these sections in order: Premise check,
Metric definition, The gap (including how the baseline was built and the
shape and start of the move), Drivers (a ranked table with signed
contribution, percent of gap, and the computation behind each row),
Offsetting factors, Residual, Caveats and data quality, Not checked. The
"Not checked" section is mandatory: a driver report that hides its blind
spots reads as more complete than it is.

## Required verifier

Run this against the payload before SUBMIT and only submit if it passes
silently. It checks structure, not truth: sections present and the report
quantified. It stays permissive on shape so tasks with non-report outputs
pass through untouched.

```python
import re

def verify(payload) -> None:
    text = None
    for key in ("report", "answer", "findings", "analysis"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, str) and len(value) > 300:
            text = value
            break
    if text is None:
        return
    low = text.lower()
    for section in ("premise", "definition", "gap", "driver",
                    "residual", "not checked"):
        assert section in low, f"report is missing a '{section}' section"
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    assert len(numbers) >= 8, (
        "driver report is under-quantified: ranked contributions, a gap "
        "and a residual need more than a handful of numbers")
```

## Tripwires

- Explaining a partial period: the single most common failure. The current
  month trails the prior one because it is not over.
- Averaging segment ratios instead of recomputing: produces a "decline"
  no segment experienced, or hides one every segment experienced.
- Attributing a mix shift to performance: every segment improved, the
  total fell, and the report blames execution.
- Contributions computed against different baselines in the same table:
  the drivers cannot sum to the gap and the reconciliation hides it.
- Causal language on date coincidence: a promotion ended the same week the
  metric fell, and the report states cause without checking mechanism.
- A trending metric judged against a flat prior period: a growth slowdown
  reads as "no change", a seasonal dip reads as a decline, and the real
  gap against expectation goes unmeasured.
- A non-additive measure reconciled by summing segments: distinct
  customer counts and ratio measures produce contributions that cannot
  sum to the gap, and forcing them to fabricates drivers.
