---
name: driver-analysis
description: >
  Diagnose why a metric moved and deliver an evidence-backed driver report.
  Use this whenever the user asks why a number changed or differs from
  expectation: sales are down, churn is up, scrap rate rose, margin
  compressed, conversion fell, costs overran, cycle time grew, yield
  slipped. Also use it when they say "find the drivers", "root cause this
  number", "explain the variance", "break down the change", or point at
  tables and ask what is going on with a KPI, even if they never say the
  words driver analysis. Applies to any domain with measurable outcomes:
  sales, marketing, manufacturing, operations, finance, supply chain,
  customer analytics. Applies to any tabular source, including Power BI
  and Fabric semantic models queried through Fabric IQ, where the metric
  is a DAX measure and questions sound like "why is this measure down"
  or "explain this KPI on my report".
---

# Driver analysis

A method for answering "why did this metric move?" with numbers instead of
narrative. It is domain generic. Revenue, scrap rate, churn, conversion,
on-time delivery, gross margin and cycle time all decompose the same way;
only the column names change.

The output is a report in which every named driver carries a quantified
contribution, the contributions reconcile to the total change, and every
figure can be reproduced by a stated computation. An unquantified driver is
a hypothesis, and it must be labeled as one.

## When this applies

Use it when a measurable metric changed against some baseline (last period,
same period last year, plan, a control group) and the task is to explain
the change from data. Do not use it for forecasting, for one-off anecdotes
with no baseline, or for questions that are qualitative from the start.

## Terms

- **Metric**: the number that moved, with its exact formula and filters.
- **Grain**: the row level the metric is computed from (order line, machine
  cycle, customer month).
- **Baseline**: what the metric is being compared against.
- **Gap**: actual minus baseline. The single number the whole analysis must
  explain. Positive or negative, keep its sign everywhere.
- **Segment**: one member of a dimension (a region, a product line, a
  machine, a customer cohort).
- **Contribution**: the share of the gap attributable to one segment or
  factor, in the metric's own units. Contributions are signed and must sum
  to the gap.

## Phase 0: prove the move is real

Most "why is the metric down" investigations end here, so do this first
and do it honestly. Before explaining a change, rule out the ways the data
manufactures one:

1. **Partial period.** The most common false alarm. A month that is 20 days
   old is not down 30 percent; it is unfinished. Compare complete periods,
   or the same days elapsed in each period.
2. **Late-arriving data.** Check whether the latest partitions are loaded.
   Row counts by load date or a max timestamp per source usually settles it.
3. **Definition or pipeline change.** A renamed status code, a new filter in
   an upstream view, a currency or unit change, a duplicate or missed load.
   Compare row counts and totals across the boundary of the change.
4. **Calendar effects.** Trading days, holiday shifts, week numbering, a leap
   day, fiscal calendars. Normalize per available day when periods differ.
5. **Normal variation.** Compute the metric's historical period-to-period
   swings. If the current move is inside its ordinary noise or its usual
   seasonal dip, say so; that can be the entire finding.

If the premise fails, the report says the metric did not really move, shows
why, and stops. That is a complete and valuable answer. Do not proceed to
invent drivers for an artifact.

## Phase 1: pin the metric definition

Ambiguity here silently invalidates everything downstream. Before
computing, state: source table, formula, filters, grain, and the unit. If
the organization keeps metric definitions (a semantic model, a metrics
file, house documentation), use that definition and cite it. If several
definitions are plausible, pick one, state it in the report, and note the
alternative. Recompute the headline number yourself from the raw grain at
least once; do not inherit it from a dashboard aggregate you cannot verify.

## Phase 2: frame the gap

State the comparison explicitly: which period or group is "actual", which
is "baseline", and why that baseline (prior period, same period last year,
plan, trend). Compute the gap. Everything that follows allocates this one
number; write it down before decomposing.

For a metric with a visible trend or seasonal pattern, a naive prior-period
baseline misstates the gap. Build the expectation from the pattern instead:
trend continuation, or same period last year adjusted for the recent growth
rate. A metric that grew 10 percent a year and is now flat has a real gap
even though the period-over-period change is zero, and a seasonal business
that dips every July has no July gap to explain. The gap worth explaining
is actual minus expected, and the report must say how expected was built.

## Phase 2b: locate the move in time

Before decomposing, compute the metric as a time series at a finer grain
than the comparison (daily or weekly under a monthly gap) and characterize
the shape of the move. The shape tells you what kind of cause to look for:

- **Step change.** The series breaks at a date. Find the breakpoint (the
  split that maximizes the difference in means between the windows before
  and after is usually enough) and hunt for events dated at it: a price
  change, a release, a policy, a pipeline deploy. A step in the data with
  no matching event is also a Phase 0 signal worth rechecking.
- **Gradual drift.** No single break, a slope that worsened. Points at slow
  accumulating causes: mix shifting toward weaker segments, churn
  outpacing acquisition, price erosion, equipment wear. Hunting for a
  single dated event on a drift misleads.
- **Single-period dip or spike.** One period out of line and a return to
  trend afterwards. Usually an outage, a stockout, a one-off order, or a
  data artifact. Check persistence before treating it as a trend.
- **In line with pattern.** The move matches the metric's own seasonality
  or established trend. That is a Phase 0 finding, not a gap to decompose.

Two timing checks carry into the drill-down. Simultaneity: if the top
segments all broke in the same period, look for one common cause; if the
breaks stagger across segments, look for something rolling out or
spreading. Persistence: state whether the most recent data is worsening,
stable, or already recovering, because that changes both the diagnosis and
the urgency of the report.

## Phase 3: decompose the gap

Work through four lenses. Not every lens applies to every metric, but check
each one deliberately rather than defaulting to the first groupby.

### Lens 1: dimensional contribution

For each candidate dimension (region, product, plant, line, channel, shift,
customer segment), compute per-segment actual, baseline, and contribution =
actual minus baseline. For an additive metric the contributions sum to the
gap exactly; verify that they do. Scan several dimensions and keep the ones
where the gap concentrates instead of stopping at the first split.

### Lens 2: rate times volume

Most business metrics are a product: revenue = price x quantity, output =
rate x uptime, conversions = rate x traffic, defects = defect rate x
volume. Split the change with the standard decomposition:

    delta_M = delta_rate * volume_base
            + delta_volume * rate_base
            + delta_rate * delta_volume

Report the interaction term separately rather than silently folding it into
one side. "Volume held, price fell" and "price held, volume fell" are
different diagnoses with different owners.

### Lens 3: mix versus within-segment change

A ratio metric can move with no segment changing at all, purely because the
weights shifted toward weaker segments. For R = sum(w_i * r_i) with weights
w_i summing to 1, split delta_R into a mix part, sum(delta_w_i * r_i_base),
and a rate part, sum(w_i_actual * delta_r_i). Always check mix before
attributing a ratio move to performance. This is Simpson's paradox in
production form: every region's conversion can improve while the total
falls.

### Lens 4: population change

Split entities into retained, new, and lost (customers, SKUs, machines,
suppliers, employees). The gap splits into: change among retained entities
(like-for-like), plus contribution of new entrants, minus the baseline
contribution of the lost. "Sales are down" caused by losing two accounts is
a different problem from every account shrinking three percent.

## Phase 4: concentrate and drill

Ask whether the gap is concentrated or broad. Rank segments by absolute
contribution and report how much of the gap the top three to five cover. If
concentrated, drill those segments one more level (that region by product,
that machine by shift) and look for a dated, coincident event: a price
change, a stockout, a promotion ending, a machine intervention, a departed
salesperson, a policy change. Date alignment between an event and an
inflection is evidence of association. Call it an association. Do not
promote it to a cause unless the mechanism is confirmed, and say what would
confirm it. If the gap is broad-based across every dimension, say that too;
it points at systemic causes (market, season, pricing) rather than local
ones.

Also report offsetting movements. If the gap is -100 built from segments
worth -150 and +50, the +50 is a finding, and "what went right" often
matters as much as what went wrong.

## Phase 5: reconcile

Named drivers plus an explicit residual must sum to the gap. Report the
residual as its own line. If the residual exceeds roughly 20 percent of the
gap, the analysis is incomplete; either decompose further or say plainly
that a material share is unexplained. Never stretch driver estimates to
make the total tidy.

## Computation rules

- Never average ratios across segments. Recompute every ratio from its
  numerator and denominator at the target level.
- Apply identical filters and definitions to actual and baseline. A filter
  that moved between periods is itself a Phase 0 finding.
- Compute all contributions against the same baseline. Mixing
  prior-period and prior-year baselines in one table produces numbers that
  cannot reconcile.
- Keep signs consistent: for a declining metric, drivers of the decline
  carry negative contributions and offsets carry positive ones.
- Round for presentation only at the end; reconcile on unrounded values.

## When the source is a semantic model (Power BI / Fabric IQ)

The method is unchanged; the mechanics shift. Adjust each phase:

- **Phase 0, freshness first.** Check the model's last refresh time before
  anything else. A stale or failed refresh is the semantic-model form of
  late-arriving data, and it manufactures declines constantly. Also
  reproduce the filter context the user is looking at: "sales are down"
  usually means "down on a specific report visual", and that visual
  carries report, page and visual-level filters plus possible row-level
  security. State the filter context every number was computed under; two
  users can legitimately see different values of the same measure.
- **Phase 1 is retrieval, not derivation.** The metric definition already
  exists as a DAX measure. Read the measure's expression and quote it in
  the report. Note what is inside it: CALCULATE filter modifiers, time
  intelligence, currency conversion. Do not re-derive the metric from the
  fact table unless you are validating the measure itself.
- **Phase 2b uses the model's date table.** Query the measure over the
  date dimension at a finer grain to get the series. Month-to-date and
  quarter-to-date measures are partial by construction; compare them only
  to equally partial baselines.
- **Phase 3 queries, never exports.** Decompose by evaluating the measure
  grouped by dimension columns (SUMMARIZECOLUMNS or the equivalent
  through the tool you have). Candidate dimensions are the model's
  related dimension tables and hierarchies; enumerate them from the model
  metadata rather than guessing.
- **Test additivity before reconciling.** Contributions summing to the gap
  holds only for additive measures. Ratios, distinct counts and
  semi-additive balances (inventory, account balances) do not sum across
  segments. Test it: compare the sum of per-segment values to the total.
  For a non-additive measure, decompose its numerator and denominator
  measures separately, or work like-for-like; never allocate a distinct
  count by summing segment values.

## Report template

Structure the deliverable exactly like this:

    # Why <metric> <moved>: driver analysis
    ## Premise check
    Is the move real? Artifacts ruled out, size of move vs normal variation.
    ## Metric definition
    Formula, source, filters, grain, unit. Baseline and why.
    ## The gap
    Actual, baseline and how it was built (naive or trend and
    seasonality adjusted), gap (signed), period covered, shape of the
    move (step, drift, one-period dip) and when it started.
    ## Drivers
    | rank | driver | contribution | % of gap | evidence |
    Each row: signed contribution in metric units, and the computation or
    query that produced it.
    ## Offsetting factors
    Segments or factors that moved against the gap.
    ## Residual
    Unexplained remainder, as a value and a percent of the gap.
    ## Caveats and data quality
    Anything that limits confidence, including Phase 0 observations.
    ## Not checked
    Dimensions, lenses or hypotheses not examined, and why.

The "Not checked" section is mandatory. A driver report that hides its
blind spots reads as more complete than it is.

## Final checks before delivering

- Contributions plus residual equal the gap within rounding tolerance.
- Every quantitative claim carries its number and its computation.
- Trending or seasonal metrics were compared to a pattern-based
  expectation, not to a naive prior period.
- The report states the shape and start of the move, and whether the
  latest data shows it persisting or recovering.
- Ratio metrics were recomputed, never averaged.
- Mix was checked before any ratio move was attributed to performance.
- The premise check appears in the report even when the move is real.
- Causal language is only used where a mechanism was confirmed; otherwise
  the report says "associated" and names what would confirm it.
