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

# # See the value of `RLM.learn()` with a Fabric semantic model
#
# This notebook asks one business question twice:
#
# 1. **Cold:** the RLM receives the live semantic-model handle.
# 2. **Learned:** `RLM.learn()` creates a portable knowledge package containing
#    an allowlisted, typed semantic-model operation. The operation executes in
#    the trusted notebook process and gives the RLM a bounded audited result.
#
# The final table compares correctness, turns, tokens, time, and provenance.
# No Lakehouse attachment is required.

# CELL ********************

%pip install -q "git+https://github.com/pawarbi/fabric-rlm-core.git@feature/knowledge-opportunistic-fallback"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 1. Configure the demo
#
# Edit only these values when using another workspace or semantic model.
# Choose a scalar measure that is meaningful to your users.

# CELL ********************

import time
from pathlib import Path

import pandas as pd

import fabric_rlm
from fabric_rlm import FabricLM, RLM, SemanticModel

WORKSPACE_ID = "2680c303-be42-4d4a-b230-281d2cedf17b"
MODEL_ID = "f76244f0-6352-4947-bbaf-98ad3f76f96c"
MEASURE = "ARR $"
QUESTION = "What is the total annual recurring revenue across the model?"
KNOWLEDGE_STORE = Path("/tmp/rlm_learn_semantic_model_value.json")

print(
    {
        "fabric_rlm_version": fabric_rlm.__version__,
        "workspace_id": WORKSPACE_ID,
        "semantic_model_id": MODEL_ID,
        "measure": MEASURE,
        "question": QUESTION,
    }
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 2. Establish the trusted answer
#
# The notebook evaluates the selected measure directly in the parent process.
# This value is used only for scoring the two RLM answers; it is not included
# in either prompt.

# CELL ********************

model = SemanticModel(
    MODEL_ID,
    workspace=WORKSPACE_ID,
    credential_provider="notebookutils",
)

expected_frame = model.measure(MEASURE)
expected_value = float(expected_frame[MEASURE].sum())

display(expected_frame)
print(f"Trusted expected value: {expected_value:,.2f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 3. Ask the question cold
#
# A cold worker receives the semantic-model handle but not a learned operation.
# Fabric credentials remain in the trusted parent process, so direct worker
# access can be unavailable or require additional exploration.

# CELL ********************

cold_started = time.perf_counter()
cold_result = RLM.task(
    QUESTION,
    inputs={"business_model": model},
    outputs={"value": float, "analysis": str},
    lm=FabricLM(
        "gpt-5.1",
        reasoning_effort="medium",
        cache=False,
    ),
    max_turns=6,
).run()
cold_wall_seconds = time.perf_counter() - cold_started

print("Cold answer")
print(cold_result.outputs)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 4. Learn the governed operation
#
# `RLM.learn()` inspects safe semantic-model metadata in the parent process and
# registers a typed `semantic_model.measure` operation. Runtime credentials,
# raw model data, and physical endpoints are not stored in the package.

# CELL ********************

learn_started = time.perf_counter()
knowledge = RLM.learn(
    sources={"business_model": model},
    store=KNOWLEDGE_STORE,
    overwrite=True,
)
learn_wall_seconds = time.perf_counter() - learn_started

operations = [
    {
        "operation_id": operation.operation_id,
        "operation": operation.operation,
        "max_output_rows": operation.max_output_rows,
        "max_output_columns": operation.max_output_columns,
        "status": operation.status,
    }
    for operation in knowledge.package.operations
]

print(f"Knowledge fingerprint: {knowledge.package.fingerprint}")
print(f"Learning time: {learn_wall_seconds:.2f}s")
display(pd.DataFrame(operations))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 5. Ask the identical question with learned knowledge
#
# The RLM selects from the registered operation contract. The notebook executes
# the selected operation with the live model credential, validates and bounds
# the result, fingerprints it, and gives only the audited result to the RLM.

# CELL ********************

learned_started = time.perf_counter()
learned_result = RLM.task(
    QUESTION,
    knowledge=knowledge,
    outputs={"value": float, "analysis": str},
    lm=FabricLM(
        "gpt-5.1",
        reasoning_effort="medium",
        cache=False,
    ),
    max_turns=6,
).run()
learned_wall_seconds = time.perf_counter() - learned_started

print("Learned answer")
print(learned_result.outputs)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 6. Compare value and trust
#
# A correct learned run should match the trusted measure and show
# `registered_operation` with an audit status of `passed`.

# CELL ********************

def answer_value(result):
    value = result.outputs.get("value")
    return float(value) if value is not None else None

def is_correct(result):
    value = answer_value(result)
    if value is None:
        return False
    return abs(value - expected_value) <= max(1.0, abs(expected_value) * 0.0001)

def comparison_row(label, result, wall_seconds):
    metadata = result.trajectory.metadata
    return {
        "run": label,
        "answer": answer_value(result),
        "correct": is_correct(result),
        "submitted": result.submitted,
        "turns": result.n_turns,
        "prompt_tokens": result.total_prompt_tokens,
        "completion_tokens": result.total_completion_tokens,
        "lm_seconds": result.total_lm_seconds,
        "worker_seconds": result.total_worker_seconds,
        "host_operation_seconds": metadata.get("operation_host_seconds"),
        "wall_seconds": wall_seconds,
        "knowledge_mode": metadata.get("knowledge_mode"),
        "operation_id": metadata.get("operation_id"),
        "audit_status": metadata.get("operation_audit_status"),
        "operation_result_fingerprint": metadata.get(
            "operation_result_fingerprint"
        ),
    }

comparison = pd.DataFrame(
    [
        comparison_row("Cold", cold_result, cold_wall_seconds),
        comparison_row("With RLM.learn()", learned_result, learned_wall_seconds),
    ]
)
display(comparison)

learned_metadata = learned_result.trajectory.metadata
assert cold_result.submitted, "The cold SemanticModel task did not submit."
assert answer_value(cold_result) != 0.0, "The cold task returned a placeholder zero."
assert is_correct(cold_result), "The cold answer did not match the trusted value."
assert is_correct(learned_result), "The learned answer did not match the trusted value."
assert learned_metadata.get("knowledge_mode") == "registered_operation"
assert learned_metadata.get("operation_audit_status") == "passed"
assert learned_metadata.get("operation_result_fingerprint")

print(
    {
        "trusted_value": expected_value,
        "cold_correct": is_correct(cold_result),
        "learned_correct": is_correct(learned_result),
        "learned_operation": learned_metadata.get("operation_id"),
        "learned_audit": learned_metadata.get("operation_audit_status"),
        "knowledge_fingerprint": knowledge.package.fingerprint,
    }
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## What `RLM.learn()` added
#
# - **Credential-safe execution:** the semantic-model query ran in the trusted
#   notebook process instead of inside the isolated worker.
# - **A reusable operation:** the package records a typed, allowlisted measure
#   contract rather than model-generated DAX or Python.
# - **Fail-closed validation:** source drift, invalid parameters, oversized
#   results, and audit failures are rejected.
# - **Portable provenance:** package, operation, source, and result
#   fingerprints make the learned answer traceable.
# - **Measurable comparison:** correctness and execution metrics are shown
#   beside the cold run using the same question and cache-disabled LMs.
