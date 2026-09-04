---
applies_when:
  keywords:
    - rank by
    - rank the
    - rank them
    - ranked by
    - ranking
    - business impact
    - deteriorat
    - materiality
    - materially
    - multiple sources
    - cross-source
    - reconcile
    - contradict
    - provenance
  output_fields: []
excludes: []
depends_on: []
specificity: utility
---
# analytical_integrity

Summary: keep the analysis faithful to the request and to the evidence,
whatever produced the evidence. Source adapters decide how a number is
retrieved; these rules decide whether the reasoning over it holds.

The rules activate by analytical operation, not by input type. A CSV, a
Lakehouse query, and a semantic-model measure are held to the same checks.

## Materiality

Do not describe a value as increasing, decreasing, improving, or
deteriorating solely because one floating-point value is larger than another.
`926400.0000001` versus `926400.0` is not a decline. Decide the materiality
rule for the task (an absolute amount, a relative fraction, or both), state
it, and test with `is_material_change(current, baseline,
absolute_tolerance=..., relative_tolerance=..., direction=...)`. Segments
whose movement is inside the rule are flat; exclude them from lists of
deteriorating or growing items.

## Requested ranking

When the task asks to rank by a concept (impact, risk, opportunity,
deterioration, growth, severity, contribution), follow the chain:

requested concept -> operational definition -> numeric metric -> calculation
-> sort -> result.

Define the metric explicitly, compute it, sort by it, and show it in the
answer (a column such as `Estimated impact`, or a line "ranked by ..."). Do
not substitute current size or another convenient field unless you state that
proxy and why it stands in for the concept.

## Candidate identity

If an earlier step selects multidimensional combinations (Product x Region x
Customer Group), later steps must keep those combinations as tuples. Do not
replace them with independent per-dimension lists and filter by
`isin(...) & isin(...) & isin(...)`; that admits every cross combination.
Use `restrict_to_candidate_tuples(frame, candidates, keys=[...])`.

## Grain

The final grain must be the requested or deliberately chosen grain. Do not
return Product x Region when Product x Region x Customer Group was requested,
or drop to individual customers, without saying so and why.

## Derived metrics and time

Recompute rates, shares, deltas, and averages from their verified components
rather than trusting a model-supplied figure. "Current", "latest", and
"this quarter" must use the source's business time (a current-period flag,
an as-of date) when one exists, not the maximum date in the data.

## Several sources contributing to one finding

- Provenance: every material figure stays attributed to the input that
  supports it. A synthesis may combine them, but the reader must be able to
  tell which number came from which source.
- Entities: prefer explicit shared keys (customer_id, subscription_id,
  product_id). Name-based or fuzzy matching is inferred evidence; say so, give
  its ambiguity, and do not rest a consequential calculation on it.
- Metrics: a similar name does not make two metrics equivalent. Before
  comparing "Active Customers #" from a semantic model with
  `COUNT(DISTINCT customer_id)` from a file, check population, aggregation,
  and time basis. Prefer the governed measure; do not silently substitute a
  raw reconstruction.
- Periods: compare business as-of time, data availability, reporting period,
  and the requested window. Align to a common valid period or state the
  mismatch. Sources are not contemporaneous by default.
- Units: reconcile unit, currency, scale, and aggregation semantics before
  comparing (USD vs EUR, dollars vs thousands, percent vs decimal, monthly vs
  quarterly). Conversions must be explicit.
- Contradictions: when sources materially disagree (ARR down, usage up,
  commentary positive), report the conflict as a conflict. Do not force one
  story.

## Fact, derived, interpretation, cause

Keep the four levels distinguishable: observed ("ARR fell from 4.2M to
3.9M"), derived ("ARR declined 7.1%"), interpretation ("this indicates
weakening revenue"), causal ("lower adoption caused the decline"). Causal
wording needs causal evidence; descriptive or associational evidence does not
become a cause in the write-up.

## Before SUBMIT

Check, from what is already computed and without another source query:
ranking matches the request; directional words match the numbers under the
stated materiality rule; grain is the requested one; candidate tuples were
preserved; derived metrics reconcile to components; requested fields are all
present; each material claim traces to evidence; across sources, entities,
periods, definitions, and units were reconciled and contradictions surfaced.
`validate_analysis_integrity(...)` runs these checks on the inputs you give it
and returns a report with `problems`; fix them before submitting.
