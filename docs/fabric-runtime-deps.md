# Fabric notebook setup — installing fabric_rlm + dspy reliably

If your Fabric notebook fails at import time with errors like:
- `ImportError: cannot import name 'Sentinel' from 'typing_extensions'`
- `ImportError: cannot import name 'Query' from 'yarl'`
- `ModuleNotFoundError: aiohttp.ConnectionTimeoutError`

…you are running on the **Synapse PySpark** kernel (Python 3.11) whose pre-installed
`cluster-env` ships old versions of `pydantic`, `aiohttp`, `yarl`, and `typing_extensions`.
A fresh `pip install dspy>=3.x` pulls newer transitive deps that conflict with that frozen
environment, and the kernel crashes before the first user cell ever runs. There is **no
reliable way** to overwrite the cluster-env packages from a notebook subprocess — `pip
--target=…` either silently no-ops or hot-swaps a partially-loaded module and corrupts the
kernel.

## Recipe — use the Python 3.12 (`jupyter_python`) kernel

Fabric exposes a second kernel — **Python 3.12 / `jupyter_python`** — whose pip is clean
and behaves like a normal Python environment. Switch your notebook to it and the entire
cascade goes away.

### 1. Notebook metadata (top of `.ipynb`)

```json
"metadata": {
    "kernelspec":  {"display_name": "Python 3.12", "language": "python", "name": "python3.12"},
    "language_info": {"name": "python", "version": "3.12"},
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
%pip uninstall -y pathlib 2>/dev/null || true
%pip install -q --no-deps --force-reinstall "/lakehouse/default/Files/.../fabric_rlm-X.Y.Z-py3-none-any.whl"
%pip install -q "dspy>=3.0.4"
```

`%pip` runs in the kernel's own pip, so the new packages are picked up immediately. No
`sys.modules.pop` + reload trickery is needed.

### 4. Use `engine="dspy"` for data-analysis tasks

`engine="default"` does not provide pandas/duckdb-aware skills out of the box. For any
task that needs to read CSV/parquet, use `engine="dspy"` and pass
`skills=["data_exploration"]`:

```python
from fabric_rlm import RLM, FabricLM

base_lm = FabricLM("gpt-5", reasoning_effort="medium", max_tokens=16000)
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

- Avoid: `subprocess.check_call(["pip","install","dspy>=3.0.4"])` from inside a Synapse PySpark
  cell — the install pulls newer pydantic-core that needs `typing_extensions.Sentinel`
  which the Synapse cluster-env doesn't have.
- Avoid: `pip install --target=<cluster-env site-packages>` to "patch" a single dep — works
  for one package but the next transitive dep (aiohttp → yarl → multidict → ...) hits the
  same wall. Whack-a-mole.
- Avoid: `cache=False` on `FabricLM` — known to interact badly with the DSPy engine on the
  jupyter_python kernel; omit it.

## Reproducible benchmark notebooks

The `examples/notebooks/` directory ships ready-to-import Fabric notebooks that
follow the recipe above — including the SpreadsheetBench runs
(`spreadsheetbench_400_openrouter_minimax_mlflow.ipynb` and
`ssb400_minimax_m3_fabric_repro.ipynb`). Import one and adjust the model and
dataset cells for your run.
