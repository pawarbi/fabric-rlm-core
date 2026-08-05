---
applies_when:
  keywords:
    - report
    - workbook
    - excel
    - xlsx
    - ops review
    - kpi
    - scorecard
    - target
    - variance
    - plant
    - scrap
    - yield
    - downtime
  output_fields: []
excludes: []
depends_on: []
specificity: domain
---
# report_context

House conventions for Manufacturing Ops reporting. These are business rules, not
metadata: none can be derived from the semantic model, the CSV, or the memo.
Where a convention here disagrees with what a measure name suggests, this wins.

## Which measure answers which KPI

| KPI | use |
| --- | --- |
| Total Sales | `[Total Sales]`. Not `sls_amt_x`, which covers Turbines and Pumps only. |
| Production Yield | `[Production Yield %]`, all shifts. Not `Day Yield Pct`, which drops the night shift. |
| Downtime | `[Downtime %]` |
| Scrap Rate | not a stored measure. `DIVIDE(SUM(ProductionLog[Scrap]), SUM(ProductionLog[Qty]))` |

## Reporting window

Every figure in an ops review covers the **trailing 30 days** ending at
`MAX('Date'[Date])` in the semantic model, unless the request names another
period. Put the window in the workbook so a reader knows what they are looking
at.

## Variance and status

Variance is **actual minus target**, in the same units as the KPI.

For Downtime and Scrap Rate, lower is better, so a *negative* variance is good.
For Total Sales and Production Yield, higher is better. Status must reflect the
direction of the KPI, not the sign of the number.

## Workbook conventions

- Header row: bold, white text on a dark fill, and the pane frozen below it.
- Currency: number format `$#,##0`.
- Rates: stored as a fraction and displayed with number format `0.00%`. Never
  write the string "3.14%" into a cell - write `0.0314` and format it.
- Column widths set so no value shows as `####`.
- One sheet per section, named exactly as the request names it.
