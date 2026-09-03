# Fabric notebook setup — installing fabric_rlm + dspy reliably

If your Fabric notebook fails at import time with errors like:
- `ImportError: cannot import name 'Sentinel' from 'typing_extensions'`
- `ImportError: cannot import name 'Query' from 'yarl'`
- `ModuleNotFoundError: aiohttp.ConnectionTimeoutError`

…you may be running on the **Synapse PySpark** kernel (Python 3.11), whose pre-installed
`cluster-env` ships old versions of `pydantic`, `aiohttp`, `yarl`, and `typing_extensions`.
A fresh `pip install dspy>=3.x` can pull newer transitive dependencies that conflict with
that frozen environment before the first user cell runs. There is **no reliable way** to
overwrite the cluster-env packages from a notebook subprocess: `pip --target=…` either
silently no-ops or hot-swaps a partially-loaded module and corrupts the kernel.

## Supported Fabric notebook runtime — Python 3.12

Fabric's **Python 3.12 / `jupyter_python`** runtime is the supported notebook
environment for `fabric-rlm`. Python 3.11 remains a Fabric notebook option, but
the Synapse PySpark environment is outside the supported `fabric-rlm`
configuration because its shared managed dependency stack can conflict with
preinstalled Semantic Link, pandas, NumPy, and telemetry packages even when an
install command appears to complete.

### 1. Notebook metadata (top of `.ipynb`)

```json
"metadata": {
    "kernelspec":  {"display_name": "Jupyter", "language": "python", "name": "jupyter"},
    "language_info": {"name": "python"},
    "kernel_info":   {"name": "jupyter", "jupyter_kernel_name": "python3.12"},
    "microsoft": {"language": "python", "language_group": "jupyter_python"},
    "dependencies": {"lakehouse": {"default_lakehouse": "<LH_ID>",
                                   "default_lakehouse_name": "<LH_NAME>",
                                   "default_lakehouse_workspace_id": "<WS_ID>"}}
}
```

### 2. Per-cell metadata (every cell)

```json
{"language": "python", "language_group": "jupyter_python"}
```

### 3. First code cell — install via `%pip` magic (NOT `subprocess.pip`)

```python
%pip install -q "git+https://github.com/pawarbi/fabric-rlm-core.git@feature/knowledge-opportunistic-fallback"
```

For an immutable experiment, replace the branch name with the validated commit
SHA in a fresh Python 3.12 session:

```python
%pip install -q "git+https://github.com/pawarbi/fabric-rlm-core.git@<commit-sha>"
```

Restart the session after either command before importing `fabric_rlm`.

### 4. Use `engine="dspy"` for data-analysis tasks

`engine="default"` does not provide pandas/duckdb-aware skills out of the box. For any
task that needs to read CSV/parquet, use `engine="dspy"` and pass
`skills=["data_exploration"]`:

```python
from fabric_rlm import RLM, FabricLM

base_lm = FabricLM("gpt-5.1", reasoning_effort="medium", max_tokens=16000)
rlm = RLM(
    signature="question -> answer",
    lm=base_lm,
    engine="dspy",
    skills=["data_exploration"],
    max_turns=8,
    timeout=300.0,
)
result = rlm.run({"question": "..."})
```

> Note: `engine="auto"` (the default since 0.2.x) routes to `"dspy"` only
> when a non-empty `tools=[...]` iterable is supplied. Passing
> `skills=["data_exploration"]` alone keeps you on `"default"`, so set
> `engine="dspy"` explicitly here.

### Reference notebook

`examples/notebooks/rlm_spark_log_root_cause.ipynb` is a ready-to-import working
example: it runs an RLM over a large Spark log to find a failure's root cause
without any of the dependency cascades above.

## Don't do this

- Do not use Python 3.11 for a supported `fabric-rlm` notebook setup. It
  remains a Fabric option, but its shared Synapse PySpark environment can report
  resolver conflicts such as incompatible `typeguard` or `importlib-metadata`
  versions after an otherwise successful installation.
- Avoid: `pip install --target=<cluster-env site-packages>` to "patch" a single dep — works
  for one package but the next transitive dep (aiohttp → yarl → multidict → ...) hits the
  same wall. Whack-a-mole.
- Avoid: `cache=False` on `FabricLM` — known to interact badly with the DSPy engine on the
  jupyter_python kernel; omit it.

## Reproducible benchmark notebooks

Use the metadata above when creating or converting a Fabric notebook. The
Python 3.12 flagship example is
`examples/notebooks/rlm_vs_plain_llm_imf_cpi.ipynb`; adjust its model and
dataset cells for your run.
