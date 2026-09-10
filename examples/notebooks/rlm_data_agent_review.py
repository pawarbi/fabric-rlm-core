# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "jupyter",
# META     "jupyter_kernel_name": "python3.12"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse_name": "",
# META       "default_lakehouse_workspace_id": ""
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Review a Fabric Data Agent with fabric-rlm
#
# Point this notebook at a Data Agent. It reads what the agent uses and how it
# is instructed, profiles the same sources through fabric-rlm, and then:
#
# 1. **Diagnoses the setup** against the documented guidance: schema names in
#    agent-level instructions that belong with the data source, references to
#    tables the source does not have, conflicting definitions, missing
#    descriptions (the first routing signal), length past the truncation limit,
#    few-shot counts, schema size.
# 2. **Builds ground truth without ground truth**: questions generated from the
#    schemas, each answered by a query executed against the source itself
#    (SQL for a lakehouse, a bounded aggregate for a semantic model).
# 3. **Evaluates the agent** on the same questions, grading the query it
#    executed where the run steps expose it and its prose otherwise, and
#    classifying failures by cause.
# 4. **Suggests changes** (agent instructions, data-source instructions,
#    descriptions, few-shots built from executed references) and writes a
#    Markdown report. Applying the changes to the draft stage is a separate,
#    explicit step at the end.
#
# Python notebook, runtime 3.12. No Spark session is needed. The review module ships with the fabric-rlm release the install cell pins; to run it before that release, install the build from the pull request branch instead.

# CELL ********************

%pip install -q "fabric-rlm==0.6.1" "fabric-data-agent-sdk==0.1.28a0"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# --- configuration ---------------------------------------------------------
AGENT_NAME = "Sales Agent RLM"        # the Data Agent to review (name or id)
WORKSPACE_NAME = None                 # None = this notebook's workspace
STAGE = "staging"                     # "staging" reviews the draft configuration, "published" the live one
REPETITIONS = 1                       # 3 before believing a delta
QUESTIONS_PER_SOURCE = 8
TOP_N = 10
APPLY_SUGGESTIONS = False             # True writes to the agent's DRAFT stage (never published)
REPORT_PATH = "/lakehouse/default/Files/data_agent_review.md"  # or None to skip saving

# --- what you know about the agent (optional) -------------------------------
# Used to scope the questions, to order them, to declare definitions to the
# RLM, and as context for every RLM task (the second opinion below).
SCOPE = ""            # e.g. "Adventure Works internet and reseller sales, 2010 to 2013, by product, territory and customer"
PRIORITIES = []       # e.g. ["factresellersales", "product category", "gross margin"]; matching questions come first
DEFINITIONS = {}      # e.g. {"gross margin": "SUM(SalesAmount - TotalProductCost)"}; override the agent's own
KNOWN_QUESTIONS = []  # your own ground truth, graded first: [("Total internet sales in 2013?", "SELECT SUM(SalesAmount) FROM dbo.factinternetsales f JOIN dbo.dimdate d ON f.OrderDateKey = d.DateKey WHERE d.CalendarYear = 2013")]
NOTES = ""            # e.g. "2014 is partial; customer names are PII"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 1. Read the agent: sources, instructions, few-shots

# CELL ********************

from fabric.dataagent.client import FabricDataAgentManagement

from fabric_rlm.data_agent_review import SdkAgentReader

management = FabricDataAgentManagement(AGENT_NAME, WORKSPACE_NAME) if WORKSPACE_NAME else FabricDataAgentManagement(AGENT_NAME)
snapshot = SdkAgentReader(management, stage=STAGE).snapshot()

print(f"{snapshot.name}: {len(snapshot.instructions):,} characters of agent instructions, {len(snapshot.datasources)} source(s)")
for source in snapshot.datasources:
    print(f"- {source.name or source.id} ({source.kind}): {len(source.instructions):,} chars of instructions, {len(source.fewshots)} few-shots, {len(source.selected_tables) or 'an unknown number of'} tables selected, description {'present' if source.description.strip() else 'missing'}")

if not any(source.selected_tables for source in snapshot.datasources):
    # the selection could not be read; show what the elements endpoint returns so the reader can be adjusted
    for handle in management.list_datasources(stage=STAGE):
        try:
            page = handle.get_elements(stage=STAGE)
            elements = page.get("value") or []
            print(f"elements of {handle._id}: {len(elements)} root element(s); first: {elements[0] if elements else None}")
        except Exception as exc:
            print(f"elements of {handle._id} unavailable: {type(exc).__name__}: {exc}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 2. Bind the same sources through fabric-rlm
#
# Each data source becomes a fabric-rlm handle: a `LakehouseSource` for
# lakehouse tables, a `SemanticModel` for a model. `RLM.learn` profiles them
# (schema, fingerprints, registered operations, declared facts from the
# agent's own definitions), and the review reads the schemas from those
# profiles. Executors run the reference queries against the handles.
#
# A lakehouse handle is scoped to the tables the agent has selected, so the
# profile and the questions cover what the agent sees; when the selection
# cannot be read, the whole lakehouse is profiled. The profile limits are
# raised because a lakehouse catalog is many tables, not one file. Lakehouse
# queries read each table's Parquet files, resolved from the Delta log, so
# tables the Delta readers reject (Spark `void` columns) still answer.

# CELL ********************

import time

import sempy.fabric as fabric

from fabric_rlm import RLM, LakehouseSource, SemanticModel
from fabric_rlm.data_agent_review import (
    LakehouseExecutor,
    ReviewContext,
    SemanticModelExecutor,
    declared_from_snapshot,
    schema_from_profile,
)
from fabric_rlm.knowledge_sources import ProfileLimits

PROFILE_LIMITS = ProfileLimits(max_fields=4096, max_diagnostic_bytes=4 * 1024 * 1024)  # a lakehouse catalog, not one file
context = ReviewContext(scope=SCOPE, priorities=tuple(PRIORITIES), definitions=dict(DEFINITIONS), questions=tuple(KNOWN_QUESTIONS), notes=NOTES)

workspace_id = fabric.resolve_workspace_id(WORKSPACE_NAME) if WORKSPACE_NAME else fabric.get_workspace_id()
items_by_workspace = {}

sources, handles = {}, {}
for source in snapshot.datasources:
    # a source can live in another workspace than the agent; bind it where it is
    source_workspace = source.workspace_id or workspace_id
    if source_workspace not in items_by_workspace:
        items_by_workspace[source_workspace] = fabric.list_items(workspace=source_workspace)
    items = items_by_workspace[source_workspace]
    row = items[items["Id"] == source.item_id].head(1)
    item_name = row["Display Name"].iloc[0] if len(row) else source.name
    if source.kind == "lakehouse":
        # OneLake by ids: no name resolution, no spaces, no friendly-name suffix
        root = f"abfss://{source_workspace}@onelake.dfs.fabric.microsoft.com/{source.item_id}" if source.item_id else f"abfss://{fabric.resolve_workspace_name(source_workspace)}@onelake.dfs.fabric.microsoft.com/{item_name}.Lakehouse"
        # only the tables the agent has selected; the whole lakehouse when the selection is unknown
        scopes = [f"Tables/{table}" for table in source.selected_tables] or None
        t0 = time.time()
        handle = LakehouseSource(root, tables=scopes).resolve()   # discover the catalog once; every query reuses it
        print(f"{item_name}: {len(handle.catalog or ())} tables discovered in {round(time.time() - t0)} s")
    elif source.kind == "semantic_model":
        handle = SemanticModel(item_name, workspace=source_workspace)
    else:
        print(f"source kind {source.kind} is not reviewed yet: {item_name}")
        continue
    sources[source.id] = handle
    handles[source.id] = (source, handle)

knowledge = RLM.learn(sources=sources, limits=PROFILE_LIMITS)
schemas = [schema_from_profile(profile, source_id=profile.source_id) for profile in knowledge.package.sources]
declared = declared_from_snapshot(snapshot, schemas, context=context)   # the agent's definitions plus yours
if declared:
    knowledge = RLM.learn(sources=sources, declared=declared, limits=PROFILE_LIMITS)   # the agent's stated definitions become facts the RLM knows

executors = {}
for source_id, (source, handle) in handles.items():
    schema = next(s for s in schemas if s.source_id == source_id)
    if source.kind == "lakehouse":
        executors[source_id] = LakehouseExecutor(handle.query, schema.tables)
    else:
        executors[source_id] = SemanticModelExecutor(handle.aggregate)

for schema in schemas:
    print(f"{schema.source_id}: {len(schema.tables)} tables, {len(schema.measures)} measures, {len(schema.relationships)} relationships")

# the references need this path; show the reason when it does not work
for source_id, (source, handle) in handles.items():
    if source.kind != "lakehouse":
        continue
    schema = next(s for s in schemas if s.source_id == source_id)
    table = next(iter(schema.tables), None)
    if table is None:
        continue
    t0 = time.time()
    try:
        rows = executors[source_id].run({"sql": f"SELECT COUNT(*) AS n FROM {table}"})
        print(f"query check on {source.name or source_id}.{table}: {rows[0]['n']:,} rows in {round(time.time() - t0, 1)} s")
    except Exception as exc:
        print(f"query check on {source.name or source_id}.{table} FAILED after {round(time.time() - t0, 1)} s: {type(exc).__name__}: {exc}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 3. Diagnose, generate questions, build references, ask the agent, grade
#
# The agent is asked through its Responses endpoint (SDK 0.1.28a0 and later;
# the Assistants endpoint on older SDKs). The response carries the run steps:
# the functions the agent called and their outputs, including the query it
# executed and the source it routed to. That query is re-run against the
# source and graded against the reference rows; prose is graded only when no
# query was captured. On a multi-source agent a failed answer that came from
# another source is graded `misrouted`.

# CELL ********************

from fabric_rlm.data_agent_review import AssistantsAgentAsker, ResponsesAgentAsker, review_agent

# "sandbox" asks the draft configuration, "production" the published one
agent_stage = "sandbox" if STAGE == "staging" else "production"
try:
    from fabric.dataagent.client import FabricOpenAIResponses

    asker = ResponsesAgentAsker(FabricOpenAIResponses(artifact_name=AGENT_NAME, workspace_name=WORKSPACE_NAME, ai_skill_stage=agent_stage))
except ImportError:
    from fabric.dataagent.client import FabricOpenAI

    asker = AssistantsAgentAsker(client=FabricOpenAI(artifact_name=AGENT_NAME, workspace_name=WORKSPACE_NAME, ai_skill_stage=agent_stage))
report = review_agent(
    snapshot,
    schemas,
    executors,
    asker,
    top=TOP_N,
    limit_per_source=QUESTIONS_PER_SOURCE,
    repetitions=REPETITIONS,
    context=context,
)
print("outcomes:", report.score())
for note in report.notes:
    print("note:", note)
for finding in report.findings:
    print(f"[{finding.severity}] {finding.code}: {finding.message[:120]}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 4. The report

# CELL ********************

from IPython.display import Markdown, display

markdown = report.to_markdown()
display(Markdown(markdown))
if REPORT_PATH:
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print("saved", REPORT_PATH)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 4b. Run steps
#
# What the agent did for each question: the functions it called, their
# arguments and outputs, the query it executed. Failed questions come first.

# CELL ********************

import pandas as pd

rows = []
for question in report.questions:
    graded = next(g for g in report.graded if g.question_id == question.id)
    for attempt, answer in enumerate(report.answers.get(question.id, ()), start=1):
        for number, step in enumerate(answer.steps, start=1):
            rows.append({
                "question": question.id, "outcome": graded.outcome, "cause": graded.cause, "attempt": attempt, "step": number,
                "kind": step.kind, "function": step.name, "arguments": step.arguments[:300], "output": step.output[:300],
                "routed_to": answer.datasource, "language": answer.language,
            })
steps = pd.DataFrame(rows)
if len(steps):
    display(steps.sort_values(["outcome", "question", "attempt", "step"]).reset_index(drop=True))
else:
    print("the endpoint returned no run steps")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 5. Second opinion on a free-form question (optional)
#
# For a question the templates do not cover, fabric-rlm can derive a
# reference itself: two blind sandboxed solves that must agree, reconciled
# on disagreement, with the integrity screens and claims tracing on. Set a
# model and a question to use it.

# CELL ********************

FREEFORM_QUESTION = None   # e.g. "Which product subcategories had the highest gross margin in 2013?"
LM = None                  # e.g. {"model": "openrouter/z-ai/glm-5.3-flash", "cache": False}

if FREEFORM_QUESTION and LM:
    from fabric_rlm import verified_task

    task = FREEFORM_QUESTION + ("\n\n" + context.as_prompt() if context.as_prompt() else "")   # the RLM gets your scope and notes
    reference = verified_task(
        task,
        inputs={source_id: handle for source_id, (_, handle) in handles.items()},
        outputs=["answer"],
        knowledge=knowledge,
        lm=LM,
        max_turns=8,
        timeout=300,
    )
    print("verdict:", reference.verdict)
    print("reference answer:", reference.result.payload)
    agent_answer = asker(FREEFORM_QUESTION)
    print("agent answer:", agent_answer.text[:1000])
    print("agent query:", agent_answer.query)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 6. Apply the suggestions to the draft stage (explicit)
#
# Nothing below runs unless `APPLY_SUGGESTIONS` is `True`. Changes go to the
# agent's draft (staging) stage; publish from the portal after re-running the
# evaluation and comparing the report.

# CELL ********************

from fabric_rlm.data_agent_review import SdkAgentWriter, apply_suggestions

if APPLY_SUGGESTIONS:
    applied = apply_suggestions(SdkAgentWriter(management), report.suggestions)
    print("applied to the draft stage:", applied)
    print("re-run section 3 against the draft stage, then compare the two reports before publishing.")
else:
    print("dry run: set APPLY_SUGGESTIONS = True to write the suggestions to the draft stage")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
