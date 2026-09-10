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
# Python notebook, runtime 3.12. No Spark session is needed.

# CELL ********************

%pip install -q "git+https://github.com/pawarbi/fabric-rlm-core@feat/data-agent-review" fabric-data-agent-sdk  # switch to @main once the branch is merged

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# --- configuration ---------------------------------------------------------
AGENT_NAME = "Sales Agent RLM"        # the Data Agent to review (name or id)
WORKSPACE_NAME = None                 # None = this notebook's workspace
REPETITIONS = 1                       # 3 before believing a delta
QUESTIONS_PER_SOURCE = 8
TOP_N = 10
APPLY_SUGGESTIONS = False             # True writes to the agent's DRAFT stage (never published)
REPORT_PATH = "/lakehouse/default/Files/data_agent_review.md"  # or None to skip saving

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 1. Read the agent: sources, instructions, few-shots

# CELL ********************

from fabric.dataagent.client import FabricDataAgentManagement, FabricOpenAI

from fabric_rlm.data_agent_review import SdkAgentReader

management = FabricDataAgentManagement(AGENT_NAME, WORKSPACE_NAME) if WORKSPACE_NAME else FabricDataAgentManagement(AGENT_NAME)
snapshot = SdkAgentReader(management).snapshot()

print(f"{snapshot.name}: {len(snapshot.instructions):,} characters of agent instructions, {len(snapshot.datasources)} source(s)")
for source in snapshot.datasources:
    print(f"- {source.name or source.id} ({source.kind}): {len(source.instructions):,} chars of instructions, {len(source.fewshots)} few-shots, description {'present' if source.description.strip() else 'missing'}")

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

# CELL ********************

import sempy.fabric as fabric

from fabric_rlm import RLM, LakehouseSource, SemanticModel
from fabric_rlm.data_agent_review import (
    LakehouseExecutor,
    SemanticModelExecutor,
    declared_from_snapshot,
    schema_from_profile,
)

workspace_id = fabric.resolve_workspace_id(WORKSPACE_NAME) if WORKSPACE_NAME else fabric.get_workspace_id()
workspace_name = fabric.resolve_workspace_name(workspace_id)
items = fabric.list_items(workspace=workspace_id)

sources, handles = {}, {}
for source in snapshot.datasources:
    row = items[items["Id"] == source.item_id].head(1)
    item_name = row["Display Name"].iloc[0] if len(row) else source.name
    item_workspace = workspace_name if not source.workspace_id or source.workspace_id == workspace_id else fabric.resolve_workspace_name(source.workspace_id)
    if source.kind == "lakehouse":
        handle = LakehouseSource(f"abfss://{item_workspace}@onelake.dfs.fabric.microsoft.com/{item_name}.Lakehouse")
    elif source.kind == "semantic_model":
        handle = SemanticModel(item_name, workspace=item_workspace)
    else:
        print(f"source kind {source.kind} is not reviewed yet: {item_name}")
        continue
    sources[source.id] = handle
    handles[source.id] = (source, handle)

knowledge = RLM.learn(sources=sources)
schemas = [schema_from_profile(profile, source_id=profile.source_id) for profile in knowledge.package.sources]
declared = declared_from_snapshot(snapshot, schemas)
if declared:
    knowledge = RLM.learn(sources=sources, declared=declared)   # the agent's stated definitions become facts the RLM knows

executors = {}
for source_id, (source, handle) in handles.items():
    schema = next(s for s in schemas if s.source_id == source_id)
    if source.kind == "lakehouse":
        executors[source_id] = LakehouseExecutor(handle.query, schema.tables)
    else:
        executors[source_id] = SemanticModelExecutor(handle.aggregate)

for schema in schemas:
    print(f"{schema.source_id}: {len(schema.tables)} tables, {len(schema.measures)} measures, {len(schema.relationships)} relationships")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 3. Diagnose, generate questions, build references, ask the agent, grade
#
# The agent is asked through its Assistants endpoint so the run steps expose
# the query it executed; that query is re-run against the source and graded
# against the reference rows. Prose is graded only when no query was captured.

# CELL ********************

from fabric_rlm.data_agent_review import AssistantsAgentAsker, review_agent

asker = AssistantsAgentAsker(client=FabricOpenAI(artifact_name=AGENT_NAME, workspace_name=WORKSPACE_NAME))
report = review_agent(
    snapshot,
    schemas,
    executors,
    asker,
    top=TOP_N,
    limit_per_source=QUESTIONS_PER_SOURCE,
    repetitions=REPETITIONS,
)
print("outcomes:", report.score())
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

    reference = verified_task(
        FREEFORM_QUESTION,
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
