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
# by an independent query, and the page renders as a dashboard with trend,
# waterfall and driver scatter charts. No language model touches the numbers.
#
# Four report kinds, chosen from the request:
#
# - **trend**: `"trend of revenue by product category"`
# - **root cause**: `"why did reseller sales fall in December 2013 by product and territory"`
# - **recap**: `"weekly recap"` or `"what moved"`
# - **top movers**: `"top 10 movers by customer year over year"`
#
# The page says how the request was read; a wrong reading is fixed by
# rewording, or by passing a `ReportSpec` with the exact columns.

# CELL ********************

%pip install -q fabric-rlm  # pin a version in a scheduled notebook

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

import os

from fabric_rlm import LakehouseSource, SemanticModel
from fabric_rlm.reports import ReportSpec, report
from fabric_rlm.sweep import what_moved

# quiet the Delta reader
os.environ.setdefault("RUST_LOG", "delta_kernel=error,deltalake=error")

# --- the source (pick one) ----------------------------------------------------
LAKEHOUSE_ROOT = "abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>.Lakehouse"
LAKEHOUSE_TABLES = "Tables"           # or "Tables/dbo", or a list such as ["Tables/dbo/factinternetsales", "Tables/dbo/dimproduct"]
SEMANTIC_MODEL = None                 # a model name or id, for example "AdventureWorks Sales"; None uses the lakehouse

# --- the request ----------------------------------------------------------------
REQUEST = "what moved"                # plain words; see the list above
YEARS = None                          # None discovers the complete years from the data; or [2012, 2013]
INSTRUCTIONS = ""                     # anything a Data Agent's instructions would say: joins, definitions, which tables matter
BUDGET = 40                           # queries the report may run (about 30 s each over OneLake from outside Fabric, faster inside)
REPORT_PATH = "/lakehouse/default/Files/what_moved.html"   # or None to skip saving

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

result = report(source, REQUEST, years=YEARS, instructions=INSTRUCTIONS, budget=BUDGET)
for line in result.spec.reading:
    print("  " + line)
print(result.sweep.summary())
for line in result.lines()[:12]:
    print("  " + line)
for note in result.sweep.notes:
    print("note:", note)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## The dashboard
#
# Headline cards, trends, and for every finding the waterfall of drivers,
# the driver scatter (above the diagonal a group moved more than its size),
# the tables and the queries behind the figures.

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

# ## Without a request
#
# `what_moved` is the recap without the reading step: every fact, every
# measure, the comparisons the time axis supports, decomposed by every
# grouping until the budget runs out. A `ReportSpec` names exactly what to
# compute when the plain-words reading is not what you meant.

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
