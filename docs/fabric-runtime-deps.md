# Fabric notebook setup and dependency troubleshooting

Import errors involving `typing_extensions.Sentinel`, `yarl.Query`, or
`aiohttp.ConnectionTimeoutError` can stem from incompatible dependency versions,
mixed package locations, or modules loaded before an installation. They are not
proof that a particular kernel is running or that the kernel is broken.

## 1. Select the runtime in Fabric

Python 3.12 is recommended for the current documented notebook workflow. Select
the Python runtime through the Fabric notebook UI, subject to the options in
your workspace. This is not a guarantee of a clean dependency environment or
compatibility with every Fabric runtime; the package's supported Python versions
and dependency constraints are declared in [pyproject.toml](../pyproject.toml).

Imported notebooks may contain hardcoded kernel or runtime metadata. Check the
selected runtime in the UI after import rather than assuming that metadata
selects an available kernel. No manual notebook or per-cell metadata editing is
needed for this setup. Attach a default lakehouse through the UI if using the
lakehouse paths below.

## 2. Install with normal dependency resolution

Run this in a notebook cell before importing the package:

```python
%pip install "fabric-rlm[analytics]"
```

Alternatively, upload a release wheel to your lakehouse and install it with its
analytics extra. Replace the example path and version with your actual wheel:

```python
%pip install "/lakehouse/default/Files/wheels/fabric_rlm-X.Y.Z-py3-none-any.whl[analytics]"
```

Use one installation route, not both. Let pip resolve the package's declared
dependencies, including the DSPy compatibility bounds. The source of truth is
[pyproject.toml](../pyproject.toml), with release-specific constraints carried in
the installed wheel's metadata; do not separately install an unconstrained DSPy
version or bypass dependency resolution. The analytics extra supplies the
data-analysis libraries used by the example.

**Restart the Python session after installation, before running the remaining
cells.** A restart is mandatory if affected modules were already imported:
`%pip` does not reload them. Use Fabric's session restart control, then rerun
imports and setup cells, not the installation cell. Do not try to repair a
running session by removing entries from `sys.modules`, hot-reloading dependency
chains, or overwriting managed runtime package directories.

If imports still fail after a restart, inspect the full traceback, installed
versions, and package locations. `%pip check` can identify declared dependency
conflicts, but a successful check does not establish that imports or live model
calls work. Resolve conflicts in the notebook's configured environment rather
than blindly uninstalling managed packages or repeatedly forcing reinstalls.

## 3. Run a data-analysis task

Both `engine="default"` and `engine="dspy"` can analyze CSV and Parquet files
through the Python subprocess when the required libraries and files are
available. Start with `engine="auto"`: it selects `"dspy"` when a non-empty
`tools=[...]` is supplied and `"default"` otherwise. Passing
`skills=["data_exploration"]` alone does not require the DSPy engine.

Upload a CSV named `sales.csv` to the default lakehouse's `Files` area, then run
the following cell after restarting. Change `data_path` for another CSV or
Parquet file accessible to the notebook and worker.

```python
from fabric_rlm import FabricLM, File, RLM

data_path = "/lakehouse/default/Files/sales.csv"
lm = FabricLM("gpt-5.1", reasoning_effort="medium", max_tokens=16000)
rlm = RLM.task(
    task=(
        "Inspect the supplied data file. Report its row count, column names, "
        "missing-value counts, and a concise summary grounded in the data."
    ),
    inputs={"data_file": File(data_path)},
    outputs=["answer"],
    lm=lm,
    engine="auto",
    skills=["data_exploration"],
    max_turns=8,
    timeout=300.0,
)
result = rlm()
print(result.payload)
```

`FabricLM` uses the notebook's Fabric identity and requires access to Fabric's
AI endpoint. Confirm model availability for your region and workspace against
the [Fabric AI services model list](https://learn.microsoft.com/en-us/fabric/data-science/ai-services/ai-services-overview#consumption-rate).
The explicit reasoning effort and token budget above are a starting point for
this model, not settings supported by every provider or model. Adjust them when
changing models. `cache=False` is an optional LM setting, not a prohibited Fabric
kernel configuration or a dependency fix.

## Examples and benchmark reproduction

- [Spark log root-cause notebook](../examples/notebooks/rlm_spark_log_root_cause.ipynb): a log-analysis example. Review its setup, file paths, and model before running; importing it does not guarantee dependency compatibility.
- [SpreadsheetBench 400 OpenRouter/MLflow notebook](../benchmarks/notebooks/spreadsheetbench_400_openrouter_minimax_mlflow.ipynb): configure its provider credentials, dataset, and logging settings for your run.
- [SpreadsheetBench MiniMax M3 Fabric reproduction](../benchmarks/notebooks/ssb400_minimax_m3_fabric_repro.ipynb): review the notebook's installation and reproduction prerequisites before execution.

General examples live under `examples/notebooks/*.ipynb`; benchmark reproduction
notebooks live under `benchmarks/notebooks/*.ipynb`. These tracked notebooks are
starting points, not guarantees of runtime compatibility, model availability,
or identical benchmark results.

See the [Test Drive Guide](../QUICKSTART.md) for general usage and provider setup.
Use this page for Fabric installation and restart guidance; neither guide
guarantees that a particular managed environment or live provider will work.
