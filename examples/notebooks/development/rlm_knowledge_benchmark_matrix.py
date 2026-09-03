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

# # Registered knowledge operation benchmark matrix
#
# This development notebook compares cold RLM analysis with learned, typed,
# host-owned operations across a Fabric semantic model, a Lakehouse Delta
# table, exact CSV and Parquet snapshots, and a fan-out-prone two-fact case.
#
# The benchmark questions are natural business questions. Operation IDs,
# source columns, expected values, and validators stay in hidden task
# configuration. Every trial disables the LM cache and task/arm order is
# randomized from a fixed seed.

# CELL ********************

import json
import traceback
from datetime import datetime, timezone

import notebookutils

_BOOTSTRAP_WORKSPACE_ID = "2680c303-be42-4d4a-b230-281d2cedf17b"
_BOOTSTRAP_LAKEHOUSE_ID = "54511b33-e765-469b-8d04-84df03d623bf"
RUN_LOG_PATH = (
    f"abfss://{_BOOTSTRAP_WORKSPACE_ID}@onelake.dfs.fabric.microsoft.com/"
    f"{_BOOTSTRAP_LAKEHOUSE_ID}/Files/knowledge-demo/benchmark-matrix/"
    "benchmark-run-log.json"
)
RUN_LOG = {"events": []}

def persist_run_log(status, **details):
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        **details,
    }
    RUN_LOG["status"] = status
    RUN_LOG["events"].append(event)
    notebookutils.fs.put(
        RUN_LOG_PATH,
        json.dumps(RUN_LOG, indent=2, default=str),
        True,
    )

def persist_uncaught_exception(shell, etype, evalue, tb, tb_offset=None):
    persist_run_log(
        "failed",
        error_type=etype.__name__,
        error_message=str(evalue),
        traceback="".join(traceback.format_exception(etype, evalue, tb)),
    )
    return shell.InteractiveTB.structured_traceback(
        etype,
        evalue,
        tb,
        tb_offset=tb_offset,
    )

def install_uncaught_exception_logger():
    get_ipython().set_custom_exc(
        (Exception,),
        persist_uncaught_exception,
    )

install_uncaught_exception_logger()
persist_run_log("running", phase="bootstrap")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

%pip install -q "git+https://github.com/pawarbi/fabric-rlm-core.git@feature/knowledge-opportunistic-fallback" "duckdb>=1.1" "deltalake>=1.0"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Configure Fabric sources and explicit OneLake output
#
# No Lakehouse is attached to this notebook. Source discovery and persisted
# artifacts use explicit IDs and ABFSS paths.

# CELL ********************

import copy
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake
import notebookutils

import fabric_rlm
from fabric_rlm import (
    FabricLM,
    File,
    LakehouseSource,
    RLM,
    SemanticModel,
    load_knowledge,
)
from fabric_rlm.knowledge_benchmark import (
    KnowledgeBenchmarkTask,
    run_knowledge_benchmark,
)

WORKSPACE_ID = "2680c303-be42-4d4a-b230-281d2cedf17b"
MODEL_ID = "f76244f0-6352-4947-bbaf-98ad3f76f96c"
LAKEHOUSE_ID = "54511b33-e765-469b-8d04-84df03d623bf"
ROOT_SEED = 20260319
REPETITIONS = 2

LAKEHOUSE_ROOT = (
    f"abfss://{WORKSPACE_ID}@onelake.dfs.fabric.microsoft.com/"
    f"{LAKEHOUSE_ID}"
)
ARTIFACT_ROOT = f"{LAKEHOUSE_ROOT}/Files/knowledge-demo/benchmark-matrix"
RUN_LOG_PATH = f"{ARTIFACT_ROOT}/benchmark-run-log.json"
LOCAL_ROOT = Path("/tmp/fabric_rlm_knowledge_benchmark")
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

RUN_LOG = {"events": []}

def persist_run_log(status, **details):
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        **details,
    }
    RUN_LOG["status"] = status
    RUN_LOG["events"].append(event)
    notebookutils.fs.put(
        RUN_LOG_PATH,
        json.dumps(RUN_LOG, indent=2, default=str),
        True,
    )

def persist_uncaught_exception(shell, etype, evalue, tb, tb_offset=None):
    persist_run_log(
        "failed",
        error_type=etype.__name__,
        error_message=str(evalue),
        traceback="".join(traceback.format_exception(etype, evalue, tb)),
    )
    return shell.InteractiveTB.structured_traceback(
        etype,
        evalue,
        tb,
        tb_offset=tb_offset,
    )

def install_uncaught_exception_logger():
    get_ipython().set_custom_exc(
        (Exception,),
        persist_uncaught_exception,
    )

install_uncaught_exception_logger()
persist_run_log(
    "running",
    phase="configured",
    fabric_rlm_version=fabric_rlm.__version__,
)

print(
    {
        "fabric_rlm_version": fabric_rlm.__version__,
        "workspace_id": WORKSPACE_ID,
        "semantic_model_id": MODEL_ID,
        "lakehouse_id": LAKEHOUSE_ID,
        "artifact_root": ARTIFACT_ROOT,
        "repetitions": REPETITIONS,
        "seed": ROOT_SEED,
    }
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Learn the governed semantic model and derive independent expectations
#
# The fixed scalar expectation was independently validated for 2026/Q2.
# Grouped expectations are queried directly from the semantic model only for
# benchmark scoring; they are never included in the RLM task prompt.

# CELL ********************

model = SemanticModel(
    MODEL_ID,
    workspace=WORKSPACE_ID,
    credential_provider="notebookutils",
)
semantic_store = f"{ARTIFACT_ROOT}/semantic-model.json"
semantic_knowledge = RLM.learn(
    sources={"arr_model": model},
    store=semantic_store,
    overwrite=True,
)
semantic_operation = next(
    operation
    for operation in semantic_knowledge.package.operations
    if operation.operation == "semantic_model.measure"
)
semantic_operation_id = semantic_operation.operation_id

allowed_groups = semantic_operation.parameter_schema["groupby"]["enum"]
region_column = next(
    name
    for name in allowed_groups
    if name.casefold().endswith("[owner region]")
)
product_family_column = next(
    name
    for name in allowed_groups
    if "product" in name.casefold() and "family" in name.casefold()
)
quarter_filters = {
    "Period[YearQuarter]": ["2026/Q2"],
    "ARR Data[IS_QUARTER]": ["1"],
}

arr_by_region = model.measure(
    "ARR $",
    groupby=[region_column],
    filters=quarter_filters,
)
arr_by_region_product = model.measure(
    "ARR $",
    groupby=[region_column, product_family_column],
    filters=quarter_filters,
)
arr_for_canada = model.measure(
    "ARR $",
    filters={
        **quarter_filters,
        region_column: ["Americas - Canada"],
    },
)

EXPECTED_TOTAL_ARR = 237576169.6
EXPECTED_CANADA_ARR = float(arr_for_canada["ARR $"].sum())

def result_dimension_column(frame, qualified_name):
    leaf_name = qualified_name.rsplit("[", 1)[-1].rstrip("]")
    return next(
        column
        for column in frame.columns
        if str(column).casefold()
        in {qualified_name.casefold(), leaf_name.casefold()}
    )

region_value_column = result_dimension_column(
    arr_by_region,
    region_column,
)
top_region_value = float(arr_by_region["ARR $"].max())
EXPECTED_TOP_REGIONS = {
    str(row[region_value_column])
    for _, row in arr_by_region.iterrows()
    if abs(float(row["ARR $"]) - top_region_value) <= 1.0
}

combination_columns = [
    result_dimension_column(arr_by_region_product, region_column),
    result_dimension_column(arr_by_region_product, product_family_column),
]
top_combination_value = float(arr_by_region_product["ARR $"].max())
EXPECTED_TOP_COMBINATIONS = {
    tuple(str(row[column]) for column in combination_columns)
    for _, row in arr_by_region_product.iterrows()
    if abs(float(row["ARR $"]) - top_combination_value) <= 1.0
}

print(
    {
        "semantic_operation_id": semantic_operation_id,
        "region_dimension": region_column,
        "product_family_dimension": product_family_column,
        "expected_total_arr": EXPECTED_TOTAL_ARR,
        "expected_canada_arr": EXPECTED_CANADA_ARR,
        "expected_top_regions": sorted(EXPECTED_TOP_REGIONS),
        "expected_top_region_value": top_region_value,
        "expected_top_combinations": sorted(EXPECTED_TOP_COMBINATIONS),
        "expected_top_combination_value": top_combination_value,
    }
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Prepare exact Delta, CSV, and Parquet benchmark sources
#
# The Lakehouse table remains a governed Delta source. Exact local CSV and
# Parquet snapshots are materialized from the same small validation table so
# the three source-family paths can be compared against identical values.

# CELL ********************

orders_lakehouse = LakehouseSource(
    f"{LAKEHOUSE_ROOT}/Tables/dbo/knowledge_validation_orders"
).resolve()
orders_entry = orders_lakehouse.catalog[0]
orders_packet = orders_lakehouse.query(
    "SELECT * FROM orders",
    sources={"orders": str(orders_entry["name"])},
    max_rows=100,
)
orders_frame = pd.DataFrame(
    orders_packet["rows"],
    columns=orders_packet["columns"],
)

def pick_column(columns, *tokens):
    return next(
        column
        for column in columns
        if all(token in column.casefold() for token in tokens)
    )

amount_column = pick_column(orders_frame.columns, "amount")
region_column_orders = pick_column(orders_frame.columns, "region")
order_id_column = pick_column(orders_frame.columns, "order")
expected_order_value = float(orders_frame[amount_column].sum())
expected_order_count = int(orders_frame[order_id_column].nunique())
expected_region_values = {
    str(region): float(value)
    for region, value in (
        orders_frame.groupby(region_column_orders)[amount_column].sum().items()
    )
}

csv_path = LOCAL_ROOT / "knowledge_validation_orders.csv"
parquet_path = LOCAL_ROOT / "knowledge_validation_orders.parquet"
orders_frame.to_csv(csv_path, index=False)
orders_frame.to_parquet(parquet_path, index=False)

lakehouse_store = f"{ARTIFACT_ROOT}/lakehouse-orders.json"
csv_store = f"{ARTIFACT_ROOT}/csv-orders.json"
parquet_store = f"{ARTIFACT_ROOT}/parquet-orders.json"

lakehouse_knowledge = RLM.learn(
    sources={"orders_lakehouse": orders_lakehouse},
    store=lakehouse_store,
    overwrite=True,
)
csv_knowledge = RLM.learn(
    sources={"orders_csv": File(csv_path)},
    store=csv_store,
    overwrite=True,
)
parquet_knowledge = RLM.learn(
    sources={"orders_parquet": File(parquet_path)},
    store=parquet_store,
    overwrite=True,
)

lakehouse_operation = next(
    operation
    for operation in lakehouse_knowledge.package.operations
    if operation.operation == "lakehouse.aggregate"
    and operation.parameter_schema["catalog_source"]["enum"]
    == (str(orders_entry["name"]),)
)
csv_operation = next(
    operation
    for operation in csv_knowledge.package.operations
    if operation.operation == "tabular.aggregate"
)
parquet_operation = next(
    operation
    for operation in parquet_knowledge.package.operations
    if operation.operation == "tabular.aggregate"
)

print(
    {
        "orders_table": orders_entry["name"],
        "expected_order_value": expected_order_value,
        "expected_order_count": expected_order_count,
        "lakehouse_operation": lakehouse_operation.operation_id,
        "csv_operation": csv_operation.operation_id,
        "parquet_operation": parquet_operation.operation_id,
    }
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Seed a reproducible fan-out-prone two-fact case
#
# Multiple indoor rows and multiple outdoor rows exist for the same month.
# A raw month join would multiply both facts. The learned operation must
# aggregate each fact independently before joining.

# CELL ********************

indoor_path = LOCAL_ROOT / "tourism_indoor"
outdoor_path = LOCAL_ROOT / "tourism_outdoor"

write_deltalake(
    str(indoor_path),
    pa.table(
        {
            "month": ["2016-07", "2016-08", "2016-08"],
            "visits": [4000000, 2000000, 2448818],
        }
    ),
    mode="overwrite",
)
write_deltalake(
    str(outdoor_path),
    pa.table(
        {
            "month": ["2016-07", "2016-08", "2016-08", "2016-08"],
            "visits": [10000000, 4000000, 4000000, 3798702],
        }
    ),
    mode="overwrite",
)

indoor_table = DeltaTable(str(indoor_path), without_files=True)
outdoor_table = DeltaTable(str(outdoor_path), without_files=True)
tourism_lakehouse = LakehouseSource(
    "file:///tmp/fabric_rlm_knowledge_benchmark/tourism",
    catalog=[
        {
            "kind": "delta",
            "name": "indoor",
            "path": str(indoor_path),
            "version": indoor_table.version(),
            "table_id": indoor_table.metadata().id,
            "columns": [["month", "VARCHAR"], ["visits", "BIGINT"]],
        },
        {
            "kind": "delta",
            "name": "outdoor",
            "path": str(outdoor_path),
            "version": outdoor_table.version(),
            "table_id": outdoor_table.metadata().id,
            "columns": [["month", "VARCHAR"], ["visits", "BIGINT"]],
        },
    ],
)
tourism_store = f"{ARTIFACT_ROOT}/tourism-fanout.json"
tourism_knowledge = RLM.learn(
    sources={"tourism": tourism_lakehouse},
    store=tourism_store,
    overwrite=True,
)
tourism_operation = next(
    operation
    for operation in tourism_knowledge.package.operations
    if operation.operation == "lakehouse.preaggregate_join"
)

EXPECTED_INDOOR_VISITS = 4448818
EXPECTED_OUTDOOR_VISITS = 11798702
raw_join_indoor = EXPECTED_INDOOR_VISITS * 3
raw_join_outdoor = EXPECTED_OUTDOOR_VISITS * 2

print(
    {
        "fanout_operation": tourism_operation.operation_id,
        "audited_indoor_total": EXPECTED_INDOOR_VISITS,
        "audited_outdoor_total": EXPECTED_OUTDOOR_VISITS,
        "raw_join_indoor_wrong": raw_join_indoor,
        "raw_join_outdoor_wrong": raw_join_outdoor,
    }
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Define the natural-language task matrix
#
# Validators and operation IDs are deliberately separate from the questions.
# The questions do not expose implementation details or physical schema.

# CELL ********************

def close_to(field, expected, relative=0.0001):
    def validate(payload):
        if not payload or payload.get(field) is None:
            return False
        actual = float(payload[field])
        return abs(actual - expected) <= max(1.0, abs(expected) * relative)
    return validate

def two_values(first_field, first_expected, second_field, second_expected):
    return lambda payload: bool(
        payload
        and close_to(first_field, first_expected)(payload)
        and close_to(second_field, second_expected)(payload)
    )

def grouped_semantic_correct(payload):
    return bool(
        payload
        and close_to("total_value", EXPECTED_TOTAL_ARR)(payload)
        and str(payload.get("top_region")) in EXPECTED_TOP_REGIONS
        and close_to("top_region_value", top_region_value)(payload)
    )

def two_dimension_semantic_correct(payload):
    claimed = (
        str(payload.get("top_region")),
        str(payload.get("top_product_family")),
    ) if payload else None
    return bool(
        payload
        and close_to("total_value", EXPECTED_TOTAL_ARR)(payload)
        and claimed in EXPECTED_TOP_COMBINATIONS
        and close_to(
            "top_combination_value",
            top_combination_value,
        )(payload)
    )

def grouped_file_correct(payload):
    if not payload or not close_to(
        "total_value",
        expected_order_value,
    )(payload):
        return False
    actual = payload.get("region_values")
    if not isinstance(actual, dict) or set(actual) != set(expected_region_values):
        return False
    return all(
        abs(float(actual[region]) - expected) <= max(
            0.01,
            abs(expected) * 0.0001,
        )
        for region, expected in expected_region_values.items()
    )

def fanout_correct(payload):
    return bool(
        payload
        and payload.get("period") == "2016-08"
        and two_values(
            "indoor_visits",
            EXPECTED_INDOOR_VISITS,
            "outdoor_visits",
            EXPECTED_OUTDOOR_VISITS,
        )(payload)
    )

TASK_CONFIG = {
    "semantic_scalar": {
        "question": (
            "What was total annual recurring revenue in the second quarter "
            "of 2026?"
        ),
        "outputs": {"value": float, "analysis": str},
        "expected_operation_id": semantic_operation_id,
        "validator": close_to("value", EXPECTED_TOTAL_ARR),
        "knowledge": semantic_knowledge,
        "inputs": {"arr_model": model},
    },
    "semantic_grouped": {
        "question": (
            "Which regions had the highest annual recurring revenue in the "
            "second quarter of 2026, and what was the total?"
        ),
        "outputs": {
            "total_value": float,
            "top_region": str,
            "top_region_value": float,
            "analysis": str,
        },
        "expected_operation_id": semantic_operation_id,
        "validator": grouped_semantic_correct,
        "knowledge": semantic_knowledge,
        "inputs": {"arr_model": model},
    },
    "semantic_filtered": {
        "question": (
            "For Americas - Canada, what was annual recurring revenue in the "
            "second quarter of 2026?"
        ),
        "outputs": {"value": float, "analysis": str},
        "expected_operation_id": semantic_operation_id,
        "validator": close_to("value", EXPECTED_CANADA_ARR),
        "knowledge": semantic_knowledge,
        "inputs": {"arr_model": model},
    },
    "semantic_two_dimensions": {
        "question": (
            "Which regional product-family combinations contributed the most "
            "annual recurring revenue in the second quarter of 2026, and what "
            "was the overall total?"
        ),
        "outputs": {
            "total_value": float,
            "top_region": str,
            "top_product_family": str,
            "top_combination_value": float,
            "analysis": str,
        },
        "expected_operation_id": semantic_operation_id,
        "validator": two_dimension_semantic_correct,
        "knowledge": semantic_knowledge,
        "inputs": {"arr_model": model},
    },
    "lakehouse_delta_scalar": {
        "question": "What is the total recorded order value?",
        "outputs": {"value": float, "analysis": str},
        "expected_operation_id": lakehouse_operation.operation_id,
        "validator": close_to("value", expected_order_value),
        "knowledge": lakehouse_knowledge,
        "inputs": {"orders_lakehouse": orders_lakehouse},
    },
    "csv_grouped": {
        "question": "How is recorded order value distributed across regions?",
        "outputs": {
            "total_value": float,
            "region_values": dict,
            "analysis": str,
        },
        "expected_operation_id": csv_operation.operation_id,
        "validator": grouped_file_correct,
        "knowledge": csv_knowledge,
        "inputs": {"orders_csv": File(csv_path)},
    },
    "parquet_scalar": {
        "question": "What is the total order value in this snapshot?",
        "outputs": {"value": float, "analysis": str},
        "expected_operation_id": parquet_operation.operation_id,
        "validator": close_to("value", expected_order_value),
        "knowledge": parquet_knowledge,
        "inputs": {"orders_parquet": File(parquet_path)},
    },
    "fanout_two_fact": {
        "question": (
            "In the latest month covered by both indoor and outdoor "
            "attractions, how many visits were recorded in each setting?"
        ),
        "outputs": {
            "indoor_visits": float,
            "outdoor_visits": float,
            "period": str,
            "analysis": str,
        },
        "expected_operation_id": tourism_operation.operation_id,
        "validator": fanout_correct,
        "knowledge": tourism_knowledge,
        "inputs": {"tourism": tourism_lakehouse},
    },
}

TASKS = tuple(
    KnowledgeBenchmarkTask(
        task_id=task_id,
        question=config["question"],
        expected_operation_id=config["expected_operation_id"],
        is_correct=config["validator"],
    )
    for task_id, config in TASK_CONFIG.items()
)

for task in TASKS:
    print(task.task_id, "->", task.question)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Run randomized cold-versus-learned trials
#
# Each task runs once per arm per repetition. `cache=False` is passed to every
# newly constructed Fabric LM and verified by the benchmark harness.

# CELL ********************

def make_lm(**_trial):
    return FabricLM(
        "gpt-5.1",
        reasoning_effort="medium",
        cache=False,
    )

def run_task(task, arm, lm):
    config = TASK_CONFIG[task.task_id]
    persist_run_log(
        "running",
        phase="trial",
        task_id=task.task_id,
        arm=arm,
    )
    source_argument = (
        {"knowledge": config["knowledge"]}
        if arm == "learned"
        else {"inputs": config["inputs"]}
    )
    result = RLM.task(
        task.question,
        outputs=config["outputs"],
        lm=lm,
        max_turns=10,
        **source_argument,
    ).run()
    persist_run_log(
        "running",
        phase="trial_completed",
        task_id=task.task_id,
        arm=arm,
        submitted=result.submitted,
        outputs=result.outputs,
    )
    return result

persist_run_log("running", phase="benchmark")
report = run_knowledge_benchmark(
    tasks=TASKS,
    repetitions=REPETITIONS,
    seed=ROOT_SEED,
    make_lm=make_lm,
    run=run_task,
)

trial_rows = [trial.to_dict() for trial in report.trials]
summary = report.summary()
persist_run_log(
    "running",
    phase="drift_checks",
    completed_trials=len(trial_rows),
)
display(pd.DataFrame(trial_rows))
display(pd.DataFrame(summary).T)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Verify fail-closed drift behavior
#
# These checks alter rebound metadata or a copied file only. They do not modify
# the Fabric semantic model or Lakehouse table.

# CELL ********************

class DriftedSemanticModel(SemanticModel):
    def measures(self):
        frame = super().measures().copy()
        expression_column = next(
            column
            for column in frame.columns
            if "expression" in str(column).casefold()
        )
        frame.loc[frame.index[0], expression_column] = (
            str(frame.loc[frame.index[0], expression_column])
            + " /* benchmark drift */"
        )
        return frame

drift_checks = {}

try:
    load_knowledge(
        semantic_store,
        sources={
            "arr_model": DriftedSemanticModel(
                MODEL_ID,
                workspace=WORKSPACE_ID,
                credential_provider="notebookutils",
            )
        },
    )
    drift_checks["semantic_measure_expression"] = "unexpectedly accepted"
except ValueError as exc:
    drift_checks["semantic_measure_expression"] = f"rejected: {exc}"

drifted_catalog = copy.deepcopy(list(orders_lakehouse.catalog or ()))
orders_catalog_entry = next(
    entry
    for entry in drifted_catalog
    if entry["name"] == orders_entry["name"]
)
orders_catalog_entry["columns"].append(["unexpected_column", "VARCHAR"])
try:
    load_knowledge(
        lakehouse_store,
        sources={
            "orders_lakehouse": LakehouseSource(
                LAKEHOUSE_ROOT,
                catalog=drifted_catalog,
            )
        },
    )
    drift_checks["lakehouse_catalog_schema"] = "unexpectedly accepted"
except ValueError as exc:
    drift_checks["lakehouse_catalog_schema"] = f"rejected: {exc}"

csv_path.write_text(
    csv_path.read_text(encoding="utf-8") + "\n999,Nowhere,999.0",
    encoding="utf-8",
)
try:
    load_knowledge(
        csv_store,
        sources={"orders_csv": File(csv_path)},
    )
    drift_checks["csv_snapshot"] = "unexpectedly accepted"
except ValueError as exc:
    drift_checks["csv_snapshot"] = f"rejected: {exc}"

assert all(value.startswith("rejected:") for value in drift_checks.values())
print(json.dumps(drift_checks, indent=2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Persist the benchmark evidence to OneLake
#
# Results include correctness, operation-selection accuracy, audit pass rate,
# turns, prompt/completion tokens, LM/worker/host/wall time, and provenance.

# CELL ********************

results_path = f"{ARTIFACT_ROOT}/benchmark-results.json"
results_document = {
    "seed": report.seed,
    "repetitions": report.repetitions,
    "summary": summary,
    "trials": trial_rows,
    "drift_checks": drift_checks,
}
notebookutils.fs.put(
    results_path,
    json.dumps(results_document, indent=2, default=str),
    True,
)
persist_run_log(
    "completed",
    phase="persisted",
    results_path=results_path,
    completed_trials=len(trial_rows),
)

print(results_path)
print(json.dumps(summary, indent=2, default=str))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
