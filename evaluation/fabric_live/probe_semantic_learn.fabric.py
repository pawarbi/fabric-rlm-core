# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "jupyter",
# META     "jupyter_kernel_name": "python3.12"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "f13a07da-7f1d-4bc8-9b8c-de25a4961b74",
# META       "default_lakehouse_name": "da_agent_tests",
# META       "default_lakehouse_workspace_id": "82ad2591-974a-4ad4-ace6-e24879274a4b",
# META       "known_lakehouses": [
# META         {
# META           "id": "f13a07da-7f1d-4bc8-9b8c-de25a4961b74"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# No-LLM probe. Answers one question: does RLM.learn() produce a non-empty
# knowledge package for a SEMANTIC MODEL, where the same call on a Delta
# lakehouse produced zero lessons? Profiling makes no model calls, so this
# needs no OpenRouter key and costs nothing.
DATASET = "ecommerce-dataset"
WORKSPACE = ""
OUT_NAME = "semantic_learn_probe.json"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

import json
import subprocess
import sys
import traceback
from pathlib import Path

EVAL = "/lakehouse/default/Files/rlm_excel_eval"
LOG = Path(f"{EVAL}/probe-semantic-learn.log")
LOG.parent.mkdir(parents=True, exist_ok=True)
LOG.write_text("", encoding="utf-8")


def note(msg):
    line = str(msg)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


rc = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     f"{EVAL}/fabric_rlm-0.6.0-py3-none-any.whl"],
    capture_output=True, text=True)
note(f"pip rc={rc.returncode}")

from fabric_rlm import RLM, SemanticModel  # noqa: E402
import fabric_rlm  # noqa: E402

note(f"fabric_rlm from {fabric_rlm.__file__}")

result = {"dataset": DATASET, "workspace": WORKSPACE or "<attached>"}


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

try:
    sm = SemanticModel(dataset=DATASET, workspace=WORKSPACE or None)
    note(f"SemanticModel built: {sm!r}")

    learned = RLM.learn(sources={"model": sm})
    pkg = learned.package
    lessons = list(pkg.lessons)

    by_basis = {}
    by_status = {}
    for x in lessons:
        for b in (getattr(x, "basis", ()) or ("<none>",)):
            by_basis[b] = by_basis.get(b, 0) + 1
        by_status[x.status] = by_status.get(x.status, 0) + 1

    result.update({
        "ok": True,
        "lessons_total": len(lessons),
        "lessons_active": sum(1 for x in lessons if x.status == "active"),
        "by_basis": by_basis,
        "by_status": by_status,
        "fingerprint": getattr(pkg, "fingerprint", None),
        # Full text so the comparison against the lakehouse arm is auditable,
        # not just a count.
        "lessons": [
            {
                "status": x.status,
                "basis": list(getattr(x, "basis", ()) or []),
                "kind": getattr(x, "kind", None),
                "text": str(getattr(x, "text", x))[:600],
            }
            for x in lessons
        ],
    })
    note(f"LESSONS total={len(lessons)} active={result['lessons_active']}")
    note(f"by_basis={by_basis}")
    for i, x in enumerate(result["lessons"][:60], 1):
        note(f"  [{i}] ({x['status']}/{x['basis']}) {x['text'][:300]}")
except Exception as exc:
    result.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                   "trace": traceback.format_exc()[-3000:]})
    note(f"FAILED {result['error']}")
    note(result["trace"])

Path(f"{EVAL}/{OUT_NAME}").write_text(
    json.dumps(result, indent=2, default=str), encoding="utf-8")
note(f"wrote {EVAL}/{OUT_NAME}")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
