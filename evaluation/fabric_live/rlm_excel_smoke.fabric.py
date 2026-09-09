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

# MARKDOWN ********************

# # RLM authors the workbook — smoke (2 questions)
#
# Cold run (no knowledge package). RLM answers each question **and writes the
# Excel itself**, updating it after every question.
#
# Output: `abfss://sandeep_ws@onelake.dfs.fabric.microsoft.com/da_agent_tests.Lakehouse/Files/rlm_excel_eval/out/rlm_answers.xlsx`

# PARAMETERS CELL ********************

# PARAMETERS -- values are injected by the job scheduler at run time.
# The key is deliberately empty here so it is never persisted in the
# notebook definition stored in the workspace.
OPENROUTER_API_KEY = ""
MAX_QUESTIONS = 2
MODEL = "openai/gpt-4.1-mini"
MAX_TURNS = 22
# Isolates this run's outputs. Two concurrent runs sharing one output directory
# will delete and overwrite each other's workbook and progress log.
RUN_TAG = ""
# Control switch. False = same questions, model, sampling, source and turn
# budget, but NO workbook duty: the agent is asked only for the answer. This
# isolates the cost of artifact authoring from the execution environment.
WORKBOOK_DUTY = True
# Comma-separated bundled skills to load, e.g. "excel_modify,analytical_integrity".
# Empty string = the library default (no skills, no autoloading).
SKILLS = ""
# Arm selector. False = arm A (cold, sources bound directly, no knowledge
# package). True = arm B: RLM.learn() runs ONCE up front, the package is frozen,
# and every question is answered from the package instead of the raw source.
# knowledge= and inputs= cannot bind the same alias, so arm B must drop the
# "lakehouse" input -- that is the library's intended shape, not a handicap.
LEARN = False

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

import os, sys, json, zipfile, time, traceback
from pathlib import Path

if not RUN_TAG:
    RUN_TAG = time.strftime("%Y%m%d-%H%M%S")

# Defensive: a job parameter delivered as the string "False" is truthy in
# Python and would silently invert the control arm.
WORKBOOK_DUTY = str(WORKBOOK_DUTY).strip().lower() not in ("false", "0", "no", "")
LEARN = str(LEARN).strip().lower() not in ("false", "0", "no", "")
SKILL_LIST = [s.strip() for s in str(SKILLS).split(",") if s.strip()]

EVAL = "/lakehouse/default/Files/rlm_excel_eval"
PROGRESS = Path(f"{EVAL}/progress-{RUN_TAG}.log")

def note(msg):
    """Write progress to Files immediately so a crashed run is still diagnosable."""
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line)
    try:
        with open(PROGRESS, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass

note("notebook started")
note(f"python {sys.version.split()[0]}")
note(f"OPENROUTER_API_KEY supplied: {bool(OPENROUTER_API_KEY)} "
     f"(len={len(OPENROUTER_API_KEY)})")
note(f"RUN_TAG={RUN_TAG}  WORKBOOK_DUTY={WORKBOOK_DUTY}  "
     f"ARM={'B-learn' if LEARN else 'A-cold'}/"
     f"{'workbook' if WORKBOOK_DUTY else 'control-no-workbook'}  "
     f"MODEL={MODEL}  SKILLS={SKILL_LIST or 'none (library default)'}")
if not OPENROUTER_API_KEY:
    note("FATAL: parameter was not injected -- is the cell tagged as a "
         "PARAMETERS CELL in the Fabric source format?")
    raise SystemExit("OPENROUTER_API_KEY parameter was not supplied")
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

LIB = "/tmp/rlm_lib"
Path(LIB).mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(f"{EVAL}/fabric_rlm_pr75.zip") as z:
    z.extractall(LIB)

# Install the wheel rather than side-loading the zip onto sys.path. The zip
# carries only the package directory -- no pyproject.toml -- so side-loading
# gives you the module without its dependency graph and nothing enforces the
# declared pins. That is how an earlier run silently ended up on dspy 3.3.1
# when the project requires >=3.2.1,<3.3. pip resolves the pins from the wheel
# metadata, which is also how a real user installs the library.
import subprocess
WHEEL = f"{EVAL}/fabric_rlm-0.6.0-py3-none-any.whl"
note("pip install of the wheel (resolves pinned dependencies)")
proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", WHEEL],
                      capture_output=True, text=True)
note(f"pip rc={proc.returncode}")
if proc.returncode != 0:
    note("pip stderr:\n" + proc.stderr[-3000:])
    raise SystemExit("library install failed")

import fabric_rlm
from fabric_rlm import RLM, LakehouseSource, File, FileDestination
note(f"fabric_rlm loaded from {fabric_rlm.__file__}")

import openpyxl
note(f"openpyxl {openpyxl.__version__}")

import dspy
DSPY_VERSION = getattr(dspy, "__version__", "unknown")
note(f"dspy {DSPY_VERSION}")
# Fail loudly if the resolved version is outside the supported window: results
# produced against an unsupported dspy are not evidence about this library.
_mm = tuple(int(p) for p in DSPY_VERSION.split(".")[:2])
if not ((3, 2) <= _mm < (3, 3)):
    note(f"FATAL: dspy {DSPY_VERSION} violates the pin >=3.2.1,<3.3")
    raise SystemExit(f"unsupported dspy {DSPY_VERSION}")

import notebookutils
note("notebookutils available")

questions = json.loads(Path(f"{EVAL}/questions.json").read_text())
questions = questions[:MAX_QUESTIONS]
note(f"{len(questions)} questions loaded (no reference answers present)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

TABLES_ROOT = "abfss://sandeep_ws@onelake.dfs.fabric.microsoft.com/da_agent_tests.Lakehouse/Tables/dbo"
OUT_ROOT    = "abfss://sandeep_ws@onelake.dfs.fabric.microsoft.com/da_agent_tests.Lakehouse/Files/rlm_excel_eval/out/" + RUN_TAG
WORKBOOK    = "rlm_answers.xlsx"
MOUNTED_WB  = f"/lakehouse/default/Files/rlm_excel_eval/out/{RUN_TAG}/rlm_answers.xlsx"

try:
    source = LakehouseSource(root=TABLES_ROOT)
    note(f"LakehouseSource built: root={source.root} tables={source.tables}")
except Exception as exc:
    note(f"FATAL LakehouseSource: {type(exc).__name__}: {exc}")
    raise

try:
    dest = FileDestination(root=OUT_ROOT)
    note(f"FileDestination built: root={dest.root}")
except Exception as exc:
    note(f"FATAL FileDestination: {type(exc).__name__}: {exc}")
    raise

# Arm B: learn ONCE, freeze, then answer every question from the frozen
# package. Learning happens before any evaluation question is seen, so no
# evaluation answer or evaluation-run evidence can enter the package.
LEARNED = None
LEARN_SUMMARY = {}
if LEARN:
    import time as _t
    _t0 = _t.perf_counter()
    LEARNED = RLM.learn(sources={"lakehouse": source})
    _pkg = LEARNED.package
    LEARN_SUMMARY = {
        "lessons_total": len(_pkg.lessons),
        "lessons_active": sum(1 for x in _pkg.lessons if x.status == "active"),
        "fingerprint": getattr(_pkg, "fingerprint", None),
        "learn_seconds": round(_t.perf_counter() - _t0, 1),
    }
    note(f"LEARN package frozen: {LEARN_SUMMARY}")

# Start clean so the incremental behaviour is unambiguous.
try:
    target = f"{OUT_ROOT}/rlm_answers.xlsx"
    if notebookutils.fs.exists(target):
        notebookutils.fs.rm(target)
        note("removed previous workbook")
    else:
        note("no previous workbook present")
except Exception as exc:
    note(f"workbook cleanup skipped: {type(exc).__name__}: {exc}")

note(f"tables={TABLES_ROOT}")
note(f"output={OUT_ROOT}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

INSTRUCTIONS = """
You are answering one analytical question and recording it in a shared Excel
workbook that persists across questions.

STEP 1 -- ANSWER THE QUESTION
Compute the answer from the lakehouse tables. Be careful with grain: several
tables join one-to-many, so aggregate before joining when a join would multiply
a measure. State units explicitly. If the result set is empty, say so and mark
the answer uncertain -- do NOT report an empty aggregate as a confident zero.

STEP 2 -- RECORD IT IN THE WORKBOOK
The workbook has exactly three sheets, in this order:

  "1. Data Used"  One row per source table you actually used for THIS question.
                  Columns: Question ID | Table | Rows scanned | Columns used |
                  Why this table was needed.

  "2. Answers"    Exactly ONE row for THIS question.
                  Columns: Question ID | Question | Answer | Units | Grain |
                  SQL used | Reasoning / comments.

  "3. Evidence"   One row per piece of evidence supporting THIS question's
                  answer. Columns: Question ID | Check performed | Observed
                  value | How it was obtained.

RULES FOR UPDATING
- If `workbook_in` is given, it is the workbook containing every PREVIOUS
  question. You MUST load it with openpyxl.load_workbook and APPEND your rows
  to the existing sheets. Every earlier row must still be present when you
  save. Do not recreate the file from scratch and do not reorder sheets.
- If `workbook_in` is not given, you are the first question: create the
  workbook with the three sheets and a header row on each.
- Format it properly: bold white-on-dark header row, freeze the header row,
  set column widths so text is readable, and wrap long text cells.

STEP 3 -- PUBLISH
Use EXACTLY this filename -- do not invent a temporary name, and do not
publish more than one file:

  f = workbook_out.stage("rlm_answers.xlsx")
  ... save the workbook to f.path with openpyxl ...
  workbook_out.publish(f, overwrite=True)

Submit `answer` as a dict with these keys:
  value                -> the numeric answer
  units                -> e.g. "USD", "days", "percentage"
  grain                -> what one row of your result represents
  sql                  -> the SQL or code you used
  reasoning            -> short explanation, including any assumption made
  workbook_rows_total  -> total data rows now in sheet "2. Answers"
  workbook_published   -> True once publish() returned successfully
"""

CONTROL_INSTRUCTIONS = """
You are answering one analytical question.

ANSWER THE QUESTION
Compute the answer from the lakehouse tables. Be careful with grain: several
tables join one-to-many, so aggregate before joining when a join would multiply
a measure. State units explicitly. If the result set is empty, say so and mark
the answer uncertain -- do NOT report an empty aggregate as a confident zero.

Submit `answer` as a dict with these keys:
  value                -> the numeric answer
  units                -> e.g. "USD", "days", "percentage"
  grain                -> what one row of your result represents
  sql                  -> the SQL or code you used
  reasoning            -> short explanation, including any assumption made
"""

def make_lm():
    return dspy.LM(
        model=f"openrouter/{MODEL}",
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_tokens=4096,
        temperature=1.0,
        cache=False,
    )

results = []

def workbook_state(retries=6, delay=2.0):
    """(exists, size) for the published workbook, retrying past write lag.

    publish() lands the file through the OneLake REST API; fs.exists can
    briefly report False straight afterwards, so a single check understates
    what RLM actually produced.
    """
    remote = f"{OUT_ROOT}/{WORKBOOK}"
    for attempt in range(retries):
        try:
            if notebookutils.fs.exists(remote):
                sizes = [f.size for f in notebookutils.fs.ls(OUT_ROOT)
                         if f.name == WORKBOOK]
                return True, (sizes[0] if sizes else 0)
        except Exception as exc:
            note(f"   workbook_state attempt {attempt}: "
                 f"{type(exc).__name__}: {exc}")
        time.sleep(delay)
    return False, 0

def fetch_published_workbook():
    """Local copy of RLM's published workbook, or None.

    HARNESS SCAFFOLDING -- disclosed deliberately. RLM publishes through the
    OneLake REST API, and `File` can only read local paths, so the parent
    copies the workbook down and hands it back as the next `workbook_in`.
    RLM still authors all content and performs the append itself; the parent
    only moves bytes across a boundary RLM cannot cross.
    """
    exists, _ = workbook_state(retries=3, delay=2.0)
    if not exists:
        return None
    local = "/tmp/workbook_in.xlsx"
    try:
        if Path(local).exists():
            Path(local).unlink()
        notebookutils.fs.cp(f"{OUT_ROOT}/{WORKBOOK}", f"file://{local}")
        if Path(local).exists() and Path(local).stat().st_size > 0:
            return local
        note("   workbook copy produced no local file")
    except Exception as exc:
        note(f"   workbook fetch failed: {type(exc).__name__}: {exc}")
    return None

try:
    for i, q in enumerate(questions):
        qid, text = q["id"], q["question"]
        # Arm B binds the frozen package instead of the source; binding both
        # under the same alias raises ValueError by design.
        inputs = {} if LEARN else {"lakehouse": source}
        if WORKBOOK_DUTY:
            inputs["workbook_out"] = dest

        prior = (fetch_published_workbook()
                 if (WORKBOOK_DUTY and i > 0) else None)
        if prior:
            inputs["workbook_in"] = File(prior)
            note(f"   handing back workbook ({Path(prior).stat().st_size} bytes)")

        if WORKBOOK_DUTY:
            task = f"Question ID: {qid}\nQuestion: {text}\n{INSTRUCTIONS}"
        else:
            task = (f"Question ID: {qid}\nQuestion: {text}\n{CONTROL_INSTRUCTIONS}")
        t0 = time.perf_counter()
        row = {"id": qid, "question": text, "has_workbook_in": bool(prior),
               "model": MODEL, "workbook_duty": WORKBOOK_DUTY,
               "skills": SKILL_LIST, "run_tag": RUN_TAG,
               "arm": "B-learn" if LEARN else "A-cold",
               "learn_summary": LEARN_SUMMARY}
        try:
            rlm = RLM.task(
                task,
                inputs=inputs,
                outputs={"answer": dict},
                knowledge=LEARNED,
                lm=make_lm(),
                max_turns=MAX_TURNS,
                timeout=900,
                enable_skill_autoloading=False,
                skills=SKILL_LIST,
            )
            res = rlm.run()
            payload = res.payload
            ans = payload.get("answer") if isinstance(payload, dict) else payload
            turns_obj = list(getattr(res.trajectory, "turns", []) or [])
            # H6: the answer payload alone cannot tell us WHY a model struggled.
            # Persist a bounded textual view of every turn plus explicit counts
            # of known library rejection messages, so error frequency is
            # measurable rather than inferred from the model's self-reports.
            traj, gate_hits = [], 0
            try:
                for ti, t in enumerate(turns_obj):
                    txt = str(t)
                    gate_hits += txt.count("read-only catalog query")
                    traj.append({"i": ti, "text": txt[:4000]})
            except Exception as exc:
                traj = [{"i": -1, "text": f"trajectory capture failed: {exc}"}]
            row.update({
                "ok": True,
                "submitted": res.submitted,
                "answer": ans,
                "turns": len(turns_obj),
                "trajectory": traj,
                "catalog_gate_rejections": gate_hits,
                "prompt_tokens": res.total_prompt_tokens,
                "completion_tokens": res.total_completion_tokens,
                "failure_reason": res.failure_reason,
            })
        except Exception as exc:
            row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "trace": traceback.format_exc()[-2000:]})
        row["seconds"] = round(time.perf_counter() - t0, 1)

        published, size = workbook_state() if WORKBOOK_DUTY else (False, 0)
        row["published"] = published
        row["published_bytes"] = size
        results.append(row)

        note(f"[{i+1}/{len(questions)}] {qid} ok={row.get('ok')} "
             f"workbook_in={row['has_workbook_in']} published={published} "
             f"bytes={size} turns={row.get('turns')} "
             f"gate={row.get('catalog_gate_rejections')} {row['seconds']}s")
        if not row.get("ok"):
            note(f"   ERROR {row.get('error')}")
            note(f"   TRACE {row.get('trace','')[-700:]}")

        try:
            Path(f"{EVAL}/run_log-{RUN_TAG}.json").write_text(
                json.dumps(results, indent=2, default=str))
        except Exception as exc:
            note(f"   run_log write failed: {type(exc).__name__}: {exc}")
except BaseException as exc:
    note(f"LOOP ABORTED: {type(exc).__name__}: {exc}")
    note(traceback.format_exc()[-2500:])
    raise
note("loop finished")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Verify what RLM actually produced -- read the published workbook back.
import openpyxl
from pathlib import Path

ok = Path(MOUNTED_WB).exists()
print("workbook published:", ok, "->", MOUNTED_WB)
if ok:
    wb = openpyxl.load_workbook(MOUNTED_WB)
    print("sheets:", wb.sheetnames)
    for ws in wb.worksheets:
        print(f"  {ws.title:<16} rows={ws.max_row:<4} cols={ws.max_column}")
    ans = None
    for cand in wb.sheetnames:
        if cand.strip().lower().endswith("answers"):
            ans = wb[cand]
            break
    if ans is not None:
        ids = [ans.cell(r, 1).value for r in range(2, ans.max_row + 1)]
        print("question ids recorded in Answers sheet:", ids)
        print("INCREMENTAL OK:" , len([x for x in ids if x]) == len(questions))

Path(f"/lakehouse/default/Files/rlm_excel_eval/run_log-{RUN_TAG}.json").write_text(
    json.dumps(results, indent=2, default=str))
print("\nrun log written for RUN_TAG =", RUN_TAG)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
