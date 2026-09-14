# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "jupyter",
# META     "jupyter_kernel_name": "python3.12"
# META   },
# META   "dependencies": {}
# META }

# MARKDOWN ********************

# # What moved: driver analysis over a lakehouse or a semantic model
#
# Point this notebook at a data source and ask for a report in plain words.
# The source's own query engine measures every figure (SQL over OneLake for
# a lakehouse, DAX for a semantic model), every reported figure is recomputed
# by an independent query, and the page renders as a dashboard in IBCS
# notation: the message as the title, this period dark and the period
# compared with grey, a rise green and a fall red with the sign and a hatch
# as a second cue, the bridge of drivers, a variance chart (every group
# before and after, its change and its change in percent), the driver
# scatter, and a Pareto view for groupings with many members. Movements on
# incomplete periods sit collapsed at the bottom; an "About this page" block
# records when the page was made, from what, and with which request. No
# language model touches the numbers.
#
# Tables with no amount (cases, tickets, sessions, events) are measured by
# their row count; any numeric column that is not a key is a measure; the
# tables, joins and definitions in `INSTRUCTIONS` are read the way a Data
# Agent reads its instructions.
#
# Four report kinds, chosen from the request:
#
# - **trend**: `"trend of revenue by product category"`
# - **root cause**: `"why did reseller sales fall in December 2013 by product and territory"`
# - **recap**: `"weekly recap"` or `"what moved"` (every fact with a time axis, four by default)
# - **top movers**: `"top 10 movers by customer year over year"`
#
# Ask in your own words. Beyond the kind, a question can name:
#
# - a period: `"what changed in July 2025"` (against June 2025 and July 2024), `"July 2025 vs June 2025"`, `"in 2024"`
# - a filter on any value the source holds, looked up before it is applied: `"for tiktok"`,
#   `"where customer state = SP"`, `"for the Night shift"`, `"for desktop and safari"`
# - a grouping without "by": `"which platform to focus on. whats moving them"`, `"revenue per region"`
# - a check against the trend: `"what changed in July 2025 for product xyz and was it in line with the trend"`,
#   `"was August 2025 normal for revenue"`
#
# And the **Monday Morning Brief**: last week for the metrics you name, in
# context (the week before, the same week last year, the recent averages,
# the seasonal expectation), with level shifts, drivers, the volume and rate
# split, the day-of-week pattern, what moved together, and KPIs built from
# the structure of the data (new, active and churned entities, ratios,
# crossings, the share of the top groups).
#
# The page says how the request was read; a wrong reading is fixed by
# rewording, or by passing a `ReportSpec` with the exact columns.

# CELL ********************

%pip install -q "fabric-rlm==0.6.1"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

import os

from fabric_rlm import LakehouseSource, SemanticModel
from fabric_rlm.brief import brief
from fabric_rlm.reports import ReportSpec, report
from fabric_rlm.sweep import what_moved

# quiet the Delta reader
os.environ.setdefault("RUST_LOG", "delta_kernel=error,deltalake=error")

# --- the source (pick one) ----------------------------------------------------
LAKEHOUSE_ROOT = "abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>.Lakehouse"
LAKEHOUSE_TABLES = "Tables"           # or "Tables/dbo", or a list such as ["Tables/dbo/factinternetsales", "Tables/dbo/dimproduct"]
SEMANTIC_MODEL = None                 # a model name or id, for example "AdventureWorks Sales"; None uses the lakehouse

# --- what the source is, in words ------------------------------------------------
# joins the catalog cannot see, definitions, which tables matter: "sales.custName = customers.custName; Revenue = SUM(Amount)"
INSTRUCTIONS = ""

# --- the report -------------------------------------------------------------------
REQUEST = "what moved"                # plain words; see the list above
YEARS = None                          # None discovers the complete years from the data; or [2012, 2013]
BUDGET = 80                           # queries the report may run (a few seconds each inside Fabric)
REPORT_PATH = "/lakehouse/default/Files/what_moved.html"   # or None to skip saving

# --- the Monday Morning Brief (leave METRICS and KPIS empty to skip) --------------
METRICS = []                          # for example ["revenue by region", "orders"]; a table name alone counts its rows
KPIS = []                             # for example ["new customers", "churned customers over 4 weeks", "active customers",
#                                       "average order value = sales amount / order quantity", "top 3 share of revenue by product",
#                                       "orders where channel = Internet vs orders where channel = Reseller"]
WEEK = None                           # None takes the latest complete week with real coverage; or "2024-12-23"
BRIEF_PATH = "/lakehouse/default/Files/monday_morning_brief.html"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Bind the source
#
# A lakehouse resolves its catalog once (every query reuses it); a semantic
# model is bound by name or id in this or another workspace.

# CELL ********************

if SEMANTIC_MODEL:
    source = SemanticModel(SEMANTIC_MODEL)
    print(source.schema(max_chars=1500))
else:
    source = LakehouseSource(LAKEHOUSE_ROOT, tables=LAKEHOUSE_TABLES).resolve()
    print(f"{len(source.catalog or ())} tables:", ", ".join(e["name"] for e in (source.catalog or ())[:20]))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Build the report
#
# The request is read against the source's vocabulary. Every figure comes
# from the source and is recomputed; the badge at the top of the page says
# how many figures were checked and whether any failed.

# CELL ********************

result = report(source, REQUEST, years=YEARS, instructions=INSTRUCTIONS, budget=BUDGET)   # a Report, or a Brief for a brief request
print(result.title)
for line in getattr(getattr(result, "spec", None), "reading", ()):
    print("  " + line)
print(result.summary())
for line in result.lines()[:12]:
    print("  " + line)
for note in result.notes:
    print("note:", note)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## The dashboard
#
# Three things to know, the picture, the cards, the trends, and for every
# finding the bridge, the variance chart, the scatter and the Pareto view,
# with the tables and the queries behind the figures. Set-aside movements
# and the reference block are collapsed at the bottom.

# CELL ********************

displayHTML(result.to_html())  # noqa: F821 - provided by the Fabric notebook runtime

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

if REPORT_PATH:
    print("saved", result.save(REPORT_PATH))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## The Monday Morning Brief
#
# Name the metrics and the KPIs to track; the entity behind "new" and
# "churned" (customers, devices, accounts, whatever the data has) is
# discovered from the structure and its choice is explained on the page.
# Schedule this notebook for Monday mornings and read the saved page.

# CELL ********************

if METRICS or KPIS:
    monday = brief(source, METRICS, kpis=KPIS, week=WEEK, instructions=INSTRUCTIONS, budget=max(BUDGET, 120))
    print(monday.summary(), "| week:", monday.week_label)
    print(monday.narrative())
    for note in monday.notes:
        print("note:", note)
    displayHTML(monday.to_html())  # noqa: F821
    if BRIEF_PATH:
        print("saved", monday.save(BRIEF_PATH))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Without a request
#
# `what_moved` is the recap without the reading step. A `ReportSpec` names
# exactly what to compute when the plain-words reading is not what you
# meant.

# CELL ********************

# swept = what_moved(source, years=YEARS, instructions=INSTRUCTIONS, budget=BUDGET)
# displayHTML(swept.to_html())
#
# exact = report(source, ReportSpec(kind="root_cause", facts=("factresellersales",), measures=("SalesAmount",), groupings=("EnglishProductCategoryName", "SalesTerritoryRegion"), period={"year": 2013, "month": 12}, against={"year": 2012, "month": 12}), years=YEARS, instructions=INSTRUCTIONS, budget=BUDGET)
# displayHTML(exact.to_html())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
