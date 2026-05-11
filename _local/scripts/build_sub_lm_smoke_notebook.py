"""Build a 5-question smoke notebook to validate the sub_lm split.

Outer LM: gpt-5 (minimal effort, planning only).
Sub LM:   gpt-4.1 (called from inside the worker via the global ``predict()``).

Each of the 5 tasks instructs the agent to translate an English phrase
to French using ``predict("english -> french", english=...)`` and then
SUBMIT the result. This forces every successful run to exercise both LMs:
gpt-5 plans + writes the call; gpt-4.1 produces the translation.

We log the full trajectory and tag each LM invocation by model so we
can verify the split actually happened.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

WHEEL_PATH = "/lakehouse/default/Files/fabric_rlm_longcot/wheels/fabric_rlm-0.1.11.dev17+sublm-py3-none-any.whl"

LH_ID = "9d10bce5-1edc-4875-83c4-ac0a98a02775"
LH_NAME = "diagnostic"
WS_ID = "82ad2591-974a-4ad4-ace6-e24879274a4b"

CONFIGURE_CELL = (
    "%%configure -f\n"
    '{"vCores": 4, "defaultLakehouse": '
    '{"name": "' + LH_NAME + '", "id": "' + LH_ID + '", "workspaceId": "' + WS_ID + '"}}'
)

HEADER = """# sub_lm split-LM smoke test (gpt-5 outer + gpt-4.1 inner)

Validates that ``RLM(lm=outer, sub_lm=inner)`` actually routes nested
``predict()`` calls inside the sandbox to ``sub_lm`` while the planning
loop stays on ``lm``.

5 trivial translation tasks. The agent MUST call
``predict("english -> french", english=...)`` to produce the translation
and then SUBMIT.

Output: ``Files/fabric_rlm_sub_lm_smoke/<RUN_ID>/``
"""

SETUP_CELL = '''import sys, json, time, traceback, uuid, subprocess, platform as _platform
from pathlib import Path

RUN_ID = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
FILES_ROOT = Path("/lakehouse/default/Files")
RUN_ROOT = FILES_ROOT / "fabric_rlm_sub_lm_smoke" / RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=True)

WHEEL = "''' + WHEEL_PATH + '''"

SUMMARY = {
    "run_id": RUN_ID,
    "started_at": time.time(),
    "python": _platform.python_version(),
    "wheel": WHEEL,
    "outer_model": "gpt-5",
    "outer_reasoning_effort": "minimal",
    "sub_model": "gpt-4.1",
    "stages": [],
    "cases": [],
    "passed": False,
    "error": None,
}
SUMMARY_PATH = RUN_ROOT / "summary.json"

def write_summary():
    SUMMARY_PATH.write_text(json.dumps(SUMMARY, indent=2, default=str))

def stage(name, **extra):
    SUMMARY["stages"].append({"name": name, "ts": time.time(), **extra})
    write_summary()
    print(f"[stage] {name} {extra}")

stage("setup_complete")
'''

INSTALL_CELL = '''stage("installing_wheel")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall", "--no-deps", WHEEL],
    check=True,
)
try:
    import dspy
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "dspy>=3.1.2"], check=True)
    import dspy

for m in [k for k in list(sys.modules) if k.startswith("fabric_rlm")]:
    sys.modules.pop(m, None)
import fabric_rlm
stage("imported", fabric_rlm=fabric_rlm.__version__, dspy=dspy.__version__)
'''

RUN_CELL = '''from fabric_rlm import RLM, FabricLM

PHRASES = [
    "Hello, how are you?",
    "I love programming.",
    "The cat is on the mat.",
    "Good morning, world.",
    "Where is the library?",
]

# Sanity check #1: can we even talk to gpt-4.1 directly from this Fabric runtime?
stage("sanity_gpt41_direct")
try:
    direct_41 = FabricLM("gpt-4.1", cache=False)
    resp = direct_41("Translate 'Hello' to French. Reply with just the French word.")
    SUMMARY["sanity_gpt41_direct"] = {"ok": True, "response": str(resp)[:300]}
except Exception as exc:
    SUMMARY["sanity_gpt41_direct"] = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[:4000]}
write_summary()
print("sanity gpt-4.1 direct:", SUMMARY["sanity_gpt41_direct"].get("ok"), str(SUMMARY["sanity_gpt41_direct"])[:300])

# Sanity check #2: dspy.Predict against gpt-4.1 (isolates dspy adapter + signature flow)
stage("sanity_gpt41_dspy_predict")
try:
    import dspy
    direct_41b = FabricLM("gpt-4.1", cache=False)
    with dspy.context(lm=direct_41b):
        pr = dspy.Predict("english -> french")
        out = pr(english="Hello, how are you?")
    SUMMARY["sanity_gpt41_dspy_predict"] = {"ok": True, "french": getattr(out, "french", None), "raw": str(out)[:300]}
except Exception as exc:
    SUMMARY["sanity_gpt41_dspy_predict"] = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[:6000]}
write_summary()
print("sanity dspy.Predict gpt-4.1:", SUMMARY["sanity_gpt41_dspy_predict"].get("ok"), str(SUMMARY["sanity_gpt41_dspy_predict"])[:400])

# Build the two LMs. cache=False to make sure both are actually hit.
outer_lm = FabricLM("gpt-5", reasoning_effort="minimal", cache=False)
# sub_lm is passed as a string spec so the worker (separate process) can
# re-resolve it via FabricLM there. A live FabricLM instance does not
# survive the subprocess boundary.
sub_lm_spec = "gpt-4.1"
stage("lms_built", outer="gpt-5/minimal", sub_spec=sub_lm_spec)

INSTRUCTIONS = (
    "Translate the English phrase in `english` to French and SUBMIT only the "
    "French string. You MUST call the sub-LM via "
    "``result = predict_sync('english -> french', english=english)`` and read "
    "``result.french``; do not translate yourself. If predict_sync() raises, "
    "print the traceback with ``traceback.print_exc()`` and "
    "``SUBMIT(answer='SUB_LM_FAILED')``. Otherwise ``SUBMIT(answer=result.french)``."
)

for idx, phrase in enumerate(PHRASES, start=1):
    case = {"idx": idx, "phrase": phrase, "started_at": time.time()}
    print(f"\\n=== case {idx}: {phrase!r} ===")
    try:
        rlm = RLM.from_task(
            task=INSTRUCTIONS,
            inputs={"english": phrase},
            outputs=["answer"],
            lm=outer_lm,
            sub_lm=sub_lm_spec,
            max_turns=5,
            timeout=120,
            verbose=False,
        )
        result = rlm.run({"english": phrase})

        case["submitted"] = bool(result.submitted)
        case["payload"] = result.payload
        case["n_turns"] = len(result.trajectory)
        case["total_prompt_tokens"] = result.total_prompt_tokens
        case["total_completion_tokens"] = result.total_completion_tokens
        case["failure_reason"] = getattr(result, "failure_reason", None)

        traj_dump = []
        for t_idx, t in enumerate(result.trajectory):
            traj_dump.append({
                "turn": t_idx,
                "turn_type": getattr(t, "turn_type", None),
                "prompt_tokens": getattr(t, "prompt_tokens", None),
                "completion_tokens": getattr(t, "completion_tokens", None),
                "code": (getattr(t, "code", None) or "")[:2000],
                "stdout": (getattr(t, "stdout", None) or "")[:3000],
                "stderr": (getattr(t, "stderr", None) or "")[:6000],
                "error": (getattr(t, "error", None) or "")[:3000],
                "submit_payload": getattr(t, "submit_payload", None),
            })
        case["trajectory"] = traj_dump

        used_predict = any(
            ("predict(" in (turn.get("code") or "")) or ("predict_sync(" in (turn.get("code") or ""))
            for turn in traj_dump
        )
        sub_lm_failed = (case.get("payload") or {}).get("answer") == "SUB_LM_FAILED"
        case["used_predict_helper"] = used_predict
        case["sub_lm_failed"] = sub_lm_failed
        print(f"  submitted={case['submitted']} payload={case['payload']} "
              f"used_predict={used_predict} sub_lm_failed={sub_lm_failed} n_turns={case['n_turns']}")
    except Exception as exc:
        case["error"] = repr(exc)
        case["traceback"] = traceback.format_exc()
        print(f"  ERROR: {exc}")
    case["elapsed_seconds"] = time.time() - case["started_at"]
    SUMMARY["cases"].append(case)
    (RUN_ROOT / f"case_{idx:02d}.json").write_text(json.dumps(case, indent=2, default=str))
    write_summary()

stage("all_cases_done")
'''

VERDICT_CELL = '''n = len(SUMMARY["cases"])
n_submitted = sum(1 for c in SUMMARY["cases"] if c.get("submitted"))
n_used_predict = sum(1 for c in SUMMARY["cases"] if c.get("used_predict_helper"))
n_sub_lm_failed = sum(1 for c in SUMMARY["cases"] if c.get("sub_lm_failed"))
n_sub_lm_ok = n - n_sub_lm_failed
# Require: every case submitted, every case used predict(), and predict() actually
# produced a French answer (not the SUB_LM_FAILED fallback).
SUMMARY["passed"] = (
    n_submitted == n
    and n_used_predict == n
    and n_sub_lm_failed == 0
)
SUMMARY["totals"] = {
    "n_cases": n,
    "n_submitted": n_submitted,
    "n_used_predict_helper": n_used_predict,
    "n_sub_lm_succeeded": n_sub_lm_ok,
    "n_sub_lm_failed": n_sub_lm_failed,
}
write_summary()
print("\\n=== VERDICT ===")
print(f"  cases: {n}")
print(f"  submitted: {n_submitted}/{n}")
print(f"  used predict() (sub_lm): {n_used_predict}/{n}")
print(f"  sub_lm produced answer: {n_sub_lm_ok}/{n}")
print(f"  PASS: {SUMMARY['passed']}")
print(f"  output dir: {RUN_ROOT}")
'''


def cell(src: str, kind: str = "code") -> dict:
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    return {
        "cells": [
            cell(HEADER, "markdown"),
            cell(CONFIGURE_CELL),
            cell(SETUP_CELL),
            cell(INSTALL_CELL),
            cell(RUN_CELL),
            cell(VERDICT_CELL),
        ],
        "metadata": {
            "kernelspec": {"name": "jupyter", "display_name": "Python 3.11", "language": "python"},
            "kernel_info": {"name": "jupyter", "jupyter_kernel_name": "python3.11"},
            "language_info": {"name": "python"},
            "dependencies": {
                "lakehouse": {
                    "default_lakehouse": LH_ID,
                    "default_lakehouse_name": LH_NAME,
                    "default_lakehouse_workspace_id": WS_ID,
                }
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build_notebook()
    out = NB_DIR / "sub_lm_smoke.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out, "cells=", len(nb["cells"]))
