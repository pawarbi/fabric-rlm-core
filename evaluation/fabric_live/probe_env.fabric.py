# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "jupyter",
# META     "jupyter_kernel_name": "python3.12"
# META   }
# META }

# CELL ********************


import sys, importlib.util, json, notebookutils
out = {"python": sys.version, "executable": sys.executable, "path_head": sys.path[:6], "mods": {}}
for m in ["dspy","litellm","openai","pydantic","openpyxl","pandas","duckdb","deltalake","pyarrow"]:
    s = importlib.util.find_spec(m)
    v = None
    if s:
        try:
            v = getattr(__import__(m), "__version__", "?")
        except Exception as e:
            v = f"IMPORT-FAILED {type(e).__name__}: {e}"
    out["mods"][m] = {"found": s is not None, "version": v,
                      "origin": getattr(s, "origin", None) if s else None}
dest = "abfss://sandeep_ws@onelake.dfs.fabric.microsoft.com/da_agent_tests.Lakehouse/Files/rlm_excel_eval/env_probe.json"
notebookutils.fs.put(dest, json.dumps(out, indent=2), True)
print("written")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
