"""Generate Fabric notebooks for the 5-way comparison.

Usage:
    python scripts/build_comparison_5way_notebook.py <strategy> [--smoke N]

strategy: A, B, C, D, E, or F
  A = direct LLM via dspy.Predict
  B = fabric_rlm v7-dspy engine
  C = fabric_rlm v6-custom + skills + full PVR
  D = fabric_rlm v6-custom + reflect_only
  E = fabric_rlm v6-custom + EffortLadder (deterministic ladder)
  F = fabric_rlm v6-custom + EffortBandit (Thompson-sampled ladder)

--smoke N : only run first N questions (default = all 25)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WHEEL = "fabric_rlm-0.2.0-py3-none-any.whl"
WHEEL_PATH = f"/lakehouse/default/Files/fabric_rlm_longcot/wheels/{WHEEL}"
DATASETS = {
    "cs": "/lakehouse/default/Files/fabric_rlm_longcot/datasets/longcot_cs_hard_regression25.jsonl",
    "aqua": "/lakehouse/default/Files/fabric_rlm_longcot/datasets/aqua_rat_15.jsonl",
    "dabench": "/lakehouse/default/Files/fabric_rlm_longcot/datasets/dabench_15.jsonl",
    "dabench_duckdb": "/lakehouse/default/Files/fabric_rlm_longcot/datasets/dabench_15_duckdb.jsonl",
}

WS_ID = "82ad2591-974a-4ad4-ace6-e24879274a4b"
LH_ID = "9d10bce5-1edc-4875-83c4-ac0a98a02775"
LH_NAME = "diagnostic"

STRATEGY_INFO = {
    "A": {"label": "direct", "title": "A — Direct LLM (dspy.Predict)"},
    "B": {"label": "dspy_rlm", "title": "B — DSPy RLM (engine=v7-dspy)"},
    "C": {"label": "fabric_full", "title": "C — Fabric RLM (v6-custom, full PVR)"},
    "D": {"label": "fabric_reflect", "title": "D — Fabric RLM (v6-custom, reflect_only)"},
    "E": {"label": "fabric_ladder", "title": "E — Fabric RLM + EffortLadder (v6-custom, adaptive, deterministic minimal->low->medium)"},
    "F": {"label": "fabric_bandit", "title": "F — Fabric RLM + EffortBandit (v6-custom, adaptive, Thompson-sampled minimal->low->medium->high with per-template Beta posteriors, 4-rung ladder)"},
    "S": {"label": "fabric_sublm", "title": "S — Fabric RLM (v6-custom, full PVR) + sub_lm=gpt-4.1 (worker LM split)"},
}


CELL_META = {"language": "python", "language_group": "jupyter_python"}


def cell_code(src: str) -> dict:
    return {"cell_type": "code", "metadata": dict(CELL_META), "execution_count": None,
            "outputs": [], "source": [line + "\n" for line in src.splitlines()]}


def cell_md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": dict(CELL_META),
            "source": [line + "\n" for line in src.splitlines()]}


NB_KERNELSPEC = {
    "kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python3.12"},
    "language_info": {"name": "python"},
    "kernel_info": {"name": "jupyter", "jupyter_kernel_name": "python3.12"},
    "microsoft": {"language": "python", "language_group": "jupyter_python"},
}


def build(strategy: str, smoke_n: int | None, run_id: str, dataset: str = "cs", base_effort: str = "minimal") -> dict:
    info = STRATEGY_INFO[strategy]
    label = info["label"]
    smoke_suffix = f" — SMOKE n={smoke_n}" if smoke_n else ""
    dataset_path = DATASETS[dataset]

    cells = []
    cells.append(cell_md(
        f"# 5-way comparison · Strategy {strategy} · {info['title']}{smoke_suffix}\n"
        f"\n"
        f"Wheel: `{WHEEL}` · Dataset: `{dataset_path.rsplit('/', 1)[-1]}`\n"
        f"\n"
        f"Shared `RUN_ID`: `{run_id}` (joins across A/C/D/E + local B for analysis).\n"
        f"\n"
        f"**Kernel:** Python 3.12 (`jupyter_python`) — same as `rlm_massive_log_demo_v5`.\n"
        f"Lakehouse is pinned via notebook metadata (`dependencies.lakehouse`)."
    ))

    # %pip install cell -- same recipe as the production rlm_massive_log_demo_v5
    # notebook.  We install dspy unconditionally because (a) several strategies
    # need it and (b) the python3.12 jupyter kernel handles the install cleanly
    # (no cluster-env cascade like the synapse_pyspark kernel).
    cells.append(cell_code(
        '%pip uninstall -y pathlib 2>/dev/null || true\n'
        f'%pip install -q --no-deps --force-reinstall "{WHEEL_PATH}"\n'
        '%pip install -q "dspy>=3.0.4"'
    ))

    # Setup cell ----------------------------------------------------------
    setup = f'''import os, sys, json, time, traceback, uuid, platform as _plat, subprocess, re
from pathlib import Path
WHEEL_PATH = "{WHEEL_PATH}"
DATASET_PATH = "{dataset_path}"
DATASET_KIND = "{dataset}"
STRATEGY = "{strategy}"
STRATEGY_LABEL = "{label}"
RUN_ID = "{run_id}"
SMOKE_N = {smoke_n if smoke_n else 'None'}
TIER = "comparison_5way"
FILES_ROOT = Path("/lakehouse/default/Files")
RUN_ROOT = FILES_ROOT / "fabric_rlm_adaptive_validation" / TIER / RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = RUN_ROOT / f"summary_{{STRATEGY_LABEL}}_{{DATASET_KIND}}.json"
RESULTS_PATH = RUN_ROOT / f"results_{{STRATEGY_LABEL}}_{{DATASET_KIND}}.jsonl"
TRACES_DIR = RUN_ROOT / f"traces_{{STRATEGY_LABEL}}_{{DATASET_KIND}}"
TRACES_DIR.mkdir(parents=True, exist_ok=True)

summary = {{"tier": TIER, "run_id": RUN_ID, "strategy": STRATEGY,
           "strategy_label": STRATEGY_LABEL, "started_at": time.time(),
           "wheel": WHEEL_PATH, "smoke_n": SMOKE_N,
           "python": _plat.python_version(), "stages": [], "results_summary": {{}}}}

def write_summary():
    summary["elapsed_seconds"] = time.time() - summary["started_at"]
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))

def stage(name, **info):
    summary["stages"].append({{"stage": name, "t": round(time.time()-summary["started_at"],1), **info}})
    write_summary(); print("[stage]", name, info)

stage("setup", run_root=str(RUN_ROOT))
# Wheel + dspy installed by %pip cell above (kernel-level pip).  Just import.
try:
    import fabric_rlm
    stage("imported", fabric_rlm=fabric_rlm.__version__)
except Exception as _ie:
    stage("import_failed", error=repr(_ie), traceback=traceback.format_exc())
    raise
if STRATEGY in ("A", "B"):
    try:
        import dspy
        stage("dspy_imported", dspy=dspy.__version__)
    except Exception as _ie:
        stage("dspy_import_failed", error=repr(_ie), traceback=traceback.format_exc())
        raise
'''
    cells.append(cell_code(setup))

    # Dataset + validator cell -------------------------------------------
    dv = '''rows = []
for line in Path(DATASET_PATH).read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line: rows.append(json.loads(line))
if SMOKE_N: rows = rows[:SMOKE_N]
stage("dataset_loaded", n=len(rows))

CS_JSON_OBJECT_TEMPLATES = {"HM","MFMC","Scheduling","TM","MCM","LLVM"}
CS_INTEGER_TEMPLATES = {"VLIW","CodeTrace"}
CS_INTEGER_LIST_TEMPLATES = {"Backprop","DistMem"}
INT_RE = re.compile(r"-?\\d+")
INT_CSV_RE = re.compile(r"-?\\d+(?:\\s*,\\s*-?\\d+)+")
# AQUA: extract last A-E letter on a line that looks like an answer
AQUA_ANS_RE = re.compile(r"(?:^|\\b)(?:answer|final\\s*answer|ans|final)\\s*[:=]?\\s*\\(?([A-Ea-e])\\)?", re.IGNORECASE | re.MULTILINE)
AQUA_LETTER_RE = re.compile(r"\\(?([A-Ea-e])\\)?")
# DABENCH: extract @key[value] tokens; value may be number, string, or list
DABENCH_TOKEN_RE = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)\\s*\\[([^\\[\\]]*)\\]")
DABENCH_FLOAT_RE = re.compile(r"-?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?")

def _resp_text(resp):
    if resp is None: return ""
    if isinstance(resp, str): return resp
    return str(resp)

def _extract_solution(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip() or None

def _extract_last_json_object(text):
    if not text: return None
    last = None
    depth = 0; start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                last = text[start:i+1]
                start = -1
    if not last: return None
    try: return json.loads(last)
    except Exception:
        try: return json.loads(last.replace("'", '"'))
        except Exception: return None

def _parse_int_list(text):
    if not text: return None
    m = INT_CSV_RE.search(text)
    if m:
        return [int(x.strip()) for x in m.group(0).split(",")]
    nums = INT_RE.findall(text)
    return [int(n) for n in nums] if nums else None

def _dabench_extract_tokens(text):
    # Last occurrence of each key wins (matches DABench reference grader)
    out = {}
    for m in DABENCH_TOKEN_RE.finditer(text or ""):
        out[m.group(1).strip().lower()] = m.group(2).strip()
    return out

def _dabench_value_match(pred_val, gold_val):
    if pred_val is None: return False
    pv = str(pred_val).strip()
    gv = str(gold_val).strip()
    # numeric compare with tolerance
    pm = DABENCH_FLOAT_RE.fullmatch(pv) or DABENCH_FLOAT_RE.search(pv)
    gm = DABENCH_FLOAT_RE.fullmatch(gv) or DABENCH_FLOAT_RE.search(gv)
    if pm and gm:
        try:
            pf = float(pm.group(0)); gf = float(gm.group(0))
            return abs(pf - gf) <= max(0.01, 0.02 * abs(gf))
        except Exception:
            pass
    return pv.lower() == gv.lower()

def grade(template, gold_answer, response_text):
    text = _resp_text(response_text)
    sol = _extract_solution(text) or text
    expected = gold_answer
    if isinstance(expected, str):
        try: expected = json.loads(expected)
        except Exception: pass
    if template == "DABENCH":
        # gold = list of [key, value] pairs; ALL must match
        if not isinstance(expected, list) or not expected:
            return False
        # search both the post-think solution AND the full text
        pred_tokens = _dabench_extract_tokens(sol)
        if not pred_tokens:
            pred_tokens = _dabench_extract_tokens(text)
        for pair in expected:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                return False
            k, gv = pair[0], pair[1]
            pv = pred_tokens.get(str(k).strip().lower())
            if not _dabench_value_match(pv, gv):
                return False
        return True
    if template == "AQUA":
        # gold_answer is single uppercase letter A-E
        gold = str(gold_answer).strip().upper()
        m = AQUA_ANS_RE.search(sol) or AQUA_ANS_RE.search(text)
        if m:
            return m.group(1).upper() == gold
        # fallback: scan the LAST line for a lone A-E letter
        for line in reversed([l.strip() for l in (sol or text).splitlines() if l.strip()]):
            m2 = AQUA_LETTER_RE.fullmatch(line)
            if m2:
                return m2.group(1).upper() == gold
            # also accept "A)" or "A." at the start of the last meaningful line
            m3 = AQUA_LETTER_RE.match(line)
            if m3 and len(line) <= 4:
                return m3.group(1).upper() == gold
            break
        return False
    if template in CS_JSON_OBJECT_TEMPLATES:
        cand = _extract_last_json_object(sol) or _extract_last_json_object(text)
        return cand == expected
    if template in CS_INTEGER_TEMPLATES:
        m = INT_RE.search(sol) or INT_RE.search(text)
        if m is None: return False
        try: return int(m.group(0)) == int(str(expected).strip())
        except Exception: return False
    if template in CS_INTEGER_LIST_TEMPLATES:
        if isinstance(expected, list):
            exp_list = [int(x) for x in expected]
        else:
            exp_list = _parse_int_list(str(expected))
        pred = _parse_int_list(sol) or _parse_int_list(text)
        return pred == exp_list
    return False

stage("validator_ready")
'''
    cells.append(cell_code(dv))

    # LM cell -------------------------------------------------------------
    lm_cell = f'''from fabric_rlm import RLM, FabricLM
os.environ["FABRIC_RLM_CAPTURE_TURNS"] = "1"
try:
    base_lm = FabricLM("gpt-5", reasoning_effort="{base_effort}", max_tokens=16000)
    stage("lm_built", model="gpt-5", effort="{base_effort}")
except Exception as _le:
    stage("lm_build_failed", error=repr(_le), traceback=traceback.format_exc())
    raise
'''
    cells.append(cell_code(lm_cell))

    # Per-strategy run cell -----------------------------------------------
    if strategy == "A":
        run_cell = '''dspy.configure(lm=base_lm)
predict = dspy.Predict("question -> answer")

with RESULTS_PATH.open("w", encoding="utf-8") as out_fh:
    for idx, row in enumerate(rows):
        qid = row["question_id"]; tpl = row["template"]; gold = row.get("answer")
        rec = {"strategy": STRATEGY_LABEL, "question_id": qid, "template": tpl,
               "started_at": time.time()}
        try:
            t0 = time.perf_counter()
            pred = predict(question=row["prompt"])
            elapsed = time.perf_counter() - t0
            ans = getattr(pred, "answer", None)
            history = getattr(base_lm, "history", []) or []
            last = history[-1] if history else {}
            usage = (last.get("usage") or {}) if isinstance(last, dict) else {}
            prompt_tok = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            completion_tok = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            passed = grade(tpl, gold, ans) if ans is not None else False
            rec.update({
                "passed": bool(passed), "elapsed_seconds": elapsed,
                "prompt_tokens": prompt_tok, "completion_tokens": completion_tok,
                "answer_preview": (str(ans)[:1000] if ans is not None else None),
                "n_attempts": 1, "n_turns": 1,
            })
            trace = {"strategy": STRATEGY_LABEL, "question_id": qid, "template": tpl,
                     "prompt": row["prompt"], "answer": str(ans) if ans is not None else None,
                     "history_last": last, "passed": rec["passed"]}
            (TRACES_DIR / f"trace_{qid}.json").write_text(json.dumps(trace, default=str, indent=2), encoding="utf-8")
        except Exception as exc:
            rec.update({"passed": False, "error": repr(exc),
                        "traceback": traceback.format_exc()})
        out_fh.write(json.dumps(rec, default=str) + "\\n"); out_fh.flush()
        stage("q_done", idx=idx+1, qid=qid, passed=rec.get("passed"),
              elapsed=round(rec.get("elapsed_seconds") or 0, 1),
              tokens=(rec.get("prompt_tokens",0) or 0)+(rec.get("completion_tokens",0) or 0))
'''
    elif strategy in ("B", "C", "D", "S"):
        if strategy == "D":
            mode_setup = 'os.environ["FABRIC_RLM_PVR_MODE"] = "reflect_only"\nos.environ.pop("FABRIC_RLM_PVR", None)'
        else:
            mode_setup = 'os.environ["FABRIC_RLM_PVR_MODE"] = "full"\nos.environ.pop("FABRIC_RLM_PVR", None)'
        engine_name = "v7-dspy" if (strategy == "B" or dataset.startswith("dabench")) else "v6-custom"
        sub_lm_kwarg = ', sub_lm="gpt-4.1"' if strategy == "S" else ""
        run_cell = f'''{mode_setup}
stage("pvr_mode_set", mode=os.environ["FABRIC_RLM_PVR_MODE"])

with RESULTS_PATH.open("w", encoding="utf-8") as out_fh:
    for idx, row in enumerate(rows):
        qid = row["question_id"]; tpl = row["template"]; gold = row.get("answer")
        rec = {{"strategy": STRATEGY_LABEL, "question_id": qid, "template": tpl,
               "started_at": time.time()}}
        try:
            _skills = ["data_exploration"] if DATASET_KIND.startswith("dabench") else None
            rlm = RLM(signature="question -> answer", lm=base_lm,
                      engine="{engine_name}", max_turns=8{sub_lm_kwarg},
                      skills=_skills, timeout=300.0)
            t0 = time.perf_counter()
            result = rlm.run({{"question": row["prompt"]}})
            elapsed = time.perf_counter() - t0
            ans = (result.payload or {{}}).get("answer") if result.payload else None
            traj = result.trajectory
            turn_records = list(getattr(traj, "turns", []) or []) if traj is not None else []
            turns = [t.to_dict() if hasattr(t, "to_dict") else t for t in turn_records]
            prompt_tok = sum((getattr(t, "prompt_tokens", None) or 0) for t in turn_records)
            completion_tok = sum((getattr(t, "completion_tokens", None) or 0) for t in turn_records)
            passed = bool(result.submitted) and grade(tpl, gold, ans) if ans is not None else False
            rec.update({{
                "passed": bool(passed), "submitted": result.submitted,
                "elapsed_seconds": elapsed,
                "prompt_tokens": prompt_tok, "completion_tokens": completion_tok,
                "n_attempts": 1, "n_turns": len(turns) if turns else None,
                "answer_preview": (str(ans)[:1000] if ans is not None else None),
            }})
            trace = {{"strategy": STRATEGY_LABEL, "question_id": qid, "template": tpl,
                     "prompt": row["prompt"], "answer": str(ans) if ans is not None else None,
                     "submitted": result.submitted, "passed": rec["passed"],
                     "turns": turns, "metadata": traj.metadata if traj is not None else None}}
            (TRACES_DIR / f"trace_{{qid}}.json").write_text(json.dumps(trace, default=str, indent=2), encoding="utf-8")
        except Exception as exc:
            rec.update({{"passed": False, "error": repr(exc),
                        "traceback": traceback.format_exc()}})
        out_fh.write(json.dumps(rec, default=str) + "\\n"); out_fh.flush()
        stage("q_done", idx=idx+1, qid=qid, passed=rec.get("passed"),
              elapsed=round(rec.get("elapsed_seconds") or 0, 1),
              tokens=(rec.get("prompt_tokens",0) or 0)+(rec.get("completion_tokens",0) or 0))
'''
    elif strategy == "E":
        run_cell = '''os.environ["FABRIC_RLM_PVR_MODE"] = "full"
os.environ.pop("FABRIC_RLM_PVR", None)
from fabric_rlm.experimental import EffortLadderPolicy
stage("pvr_mode_set", mode=os.environ["FABRIC_RLM_PVR_MODE"])

with RESULTS_PATH.open("w", encoding="utf-8") as out_fh:
    for idx, row in enumerate(rows):
        qid = row["question_id"]; tpl = row["template"]; gold = row.get("answer")
        rec = {"strategy": STRATEGY_LABEL, "question_id": qid, "template": tpl,
               "started_at": time.time()}
        try:
            def _validator(result, _gold=gold, _tpl=tpl):
                if not result.submitted or not result.payload: return False
                ans = result.payload.get("answer")
                return ans is not None and grade(_tpl, _gold, ans)
            policy = EffortLadderPolicy(
                base_lm_instance=base_lm,
                base_reasoning_effort="minimal",
                parallel_rollouts=1,
                effort_ladder=("minimal", "low", "medium"),
            )
            rlm = RLM(signature="question -> answer", lm=base_lm,
                      engine="adaptive",
                      adaptive=dict(policy=policy, validator=_validator,
                                    max_attempts=3, parallel_rollouts=1))
            t0 = time.perf_counter()
            result = rlm.run({"question": row["prompt"]})
            elapsed = time.perf_counter() - t0
            ans = (result.payload or {}).get("answer") if result.payload else None
            traj = result.trajectory
            meta = (traj.metadata or {}).get("adaptive", {}) if traj is not None else {}
            attempts = meta.get("attempts", [])
            passed = bool(result.submitted) and grade(tpl, gold, ans) if ans is not None else False
            rec.update({
                "passed": bool(passed), "submitted": result.submitted,
                "elapsed_seconds": elapsed,
                "starting_rung": attempts[0].get("rung") if attempts else None,
                "winner_rung": meta.get("winner_rung"),
                "stop_reason": meta.get("stop_reason"),
                "n_attempts": len(attempts),
                "n_turns": sum(a.get("turns_used") or 0 for a in attempts),
                "prompt_tokens": sum(a.get("prompt_tokens") or 0 for a in attempts),
                "completion_tokens": sum(a.get("completion_tokens") or 0 for a in attempts),
                "answer_preview": (str(ans)[:1000] if ans is not None else None),
            })
            trace = {"strategy": STRATEGY_LABEL, "question_id": qid, "template": tpl,
                     "prompt": row["prompt"], "answer": str(ans) if ans is not None else None,
                     "submitted": result.submitted, "passed": rec["passed"],
                     "winner_rung": rec["winner_rung"], "starting_rung": rec["starting_rung"],
                     "attempts": [
                         {"rung": a.get("rung"), "passed": a.get("passed"),
                          "submitted": a.get("submitted"),
                          "turns_used": a.get("turns_used"),
                          "failure_reason": a.get("failure_reason"),
                          "feedback": a.get("feedback"),
                          "answer_preview": str(((a.get("payload_preview") or {}).get("answer")) or "")[:500],
                          "turns": a.get("turns", [])}
                         for a in attempts]}
            (TRACES_DIR / f"trace_{qid}.json").write_text(json.dumps(trace, default=str, indent=2), encoding="utf-8")
        except Exception as exc:
            rec.update({"passed": False, "error": repr(exc),
                        "traceback": traceback.format_exc()})
        out_fh.write(json.dumps(rec, default=str) + "\\n"); out_fh.flush()
        stage("q_done", idx=idx+1, qid=qid, passed=rec.get("passed"),
              elapsed=round(rec.get("elapsed_seconds") or 0, 1),
              tokens=(rec.get("prompt_tokens",0) or 0)+(rec.get("completion_tokens",0) or 0))
'''
    elif strategy == "F":
        run_cell = '''os.environ["FABRIC_RLM_PVR_MODE"] = "full"
os.environ.pop("FABRIC_RLM_PVR", None)
from fabric_rlm.experimental import EffortBanditPolicy, BanditState, EFFORT_RUNG_COST
stage("pvr_mode_set", mode=os.environ["FABRIC_RLM_PVR_MODE"])

# Persistent bandit state for the run; per-template (task_key=template) so the
# bandit accumulates Beta(alpha,beta) posteriors per template across the 25-question sweep.
BANDIT_STATE_PATH = RUN_ROOT / f"bandit_state_{DATASET_KIND}.json"
bandit_state = BanditState.from_path(BANDIT_STATE_PATH)
stage("bandit_state_loaded", path=str(BANDIT_STATE_PATH),
      n_keys=len(bandit_state.priors))

with RESULTS_PATH.open("w", encoding="utf-8") as out_fh:
    for idx, row in enumerate(rows):
        qid = row["question_id"]; tpl = row["template"]; gold = row.get("answer")
        rec = {"strategy": STRATEGY_LABEL, "question_id": qid, "template": tpl,
               "started_at": time.time()}
        try:
            def _validator(result, _gold=gold, _tpl=tpl):
                if not result.submitted or not result.payload: return False
                ans = result.payload.get("answer")
                return ans is not None and grade(_tpl, _gold, ans)
            policy = EffortBanditPolicy(
                base_lm_instance=base_lm,
                base_reasoning_effort="minimal",
                parallel_rollouts=1,
                effort_ladder=("minimal", "low", "medium", "high"),
                state=bandit_state,
                task_key=tpl,
                warmup=2,
                rung_cost=EFFORT_RUNG_COST,
            )
            # Capture the bandit's pre-decision posterior snapshot for this template
            pre_obs = bandit_state.total_observations(tpl)
            pre_betas = {r: bandit_state.beta_for(tpl, r) for r in (0, 1, 2, 3)}
            rlm = RLM(signature="question -> answer", lm=base_lm,
                      engine="adaptive",
                      adaptive=dict(policy=policy, validator=_validator,
                                    max_attempts=4, parallel_rollouts=1))
            t0 = time.perf_counter()
            result = rlm.run({"question": row["prompt"]})
            elapsed = time.perf_counter() - t0
            ans = (result.payload or {}).get("answer") if result.payload else None
            traj = result.trajectory
            meta = (traj.metadata or {}).get("adaptive", {}) if traj is not None else {}
            attempts = meta.get("attempts", [])
            passed = bool(result.submitted) and grade(tpl, gold, ans) if ans is not None else False
            # Record outcomes back into the bandit state so the next question
            # for the same template benefits from the signal.
            for a in attempts:
                rung_i = a.get("rung")
                if rung_i is None:
                    continue
                bandit_state.record(tpl, int(rung_i), bool(a.get("passed")))
            try:
                bandit_state.save()
            except Exception as _se:
                stage("bandit_save_warn", err=repr(_se))
            post_betas = {r: bandit_state.beta_for(tpl, r) for r in (0, 1, 2, 3)}
            rec.update({
                "passed": bool(passed), "submitted": result.submitted,
                "elapsed_seconds": elapsed,
                "starting_rung": attempts[0].get("rung") if attempts else None,
                "winner_rung": meta.get("winner_rung"),
                "stop_reason": meta.get("stop_reason"),
                "n_attempts": len(attempts),
                "n_turns": sum(a.get("turns_used") or 0 for a in attempts),
                "prompt_tokens": sum(a.get("prompt_tokens") or 0 for a in attempts),
                "completion_tokens": sum(a.get("completion_tokens") or 0 for a in attempts),
                "answer_preview": (str(ans)[:1000] if ans is not None else None),
                "bandit_pre_observations": pre_obs,
                "bandit_pre_betas": pre_betas,
                "bandit_post_betas": post_betas,
                "bandit_warmup_active": pre_obs < 2,
            })
            trace = {"strategy": STRATEGY_LABEL, "question_id": qid, "template": tpl,
                     "prompt": row["prompt"], "answer": str(ans) if ans is not None else None,
                     "submitted": result.submitted, "passed": rec["passed"],
                     "winner_rung": rec["winner_rung"], "starting_rung": rec["starting_rung"],
                     "bandit_pre_observations": pre_obs,
                     "bandit_pre_betas": pre_betas, "bandit_post_betas": post_betas,
                     "attempts": [
                         {"rung": a.get("rung"), "passed": a.get("passed"),
                          "submitted": a.get("submitted"),
                          "turns_used": a.get("turns_used"),
                          "failure_reason": a.get("failure_reason"),
                          "feedback": a.get("feedback"),
                          "answer_preview": str(((a.get("payload_preview") or {}).get("answer")) or "")[:500],
                          "turns": a.get("turns", [])}
                         for a in attempts]}
            (TRACES_DIR / f"trace_{qid}.json").write_text(json.dumps(trace, default=str, indent=2), encoding="utf-8")
        except Exception as exc:
            rec.update({"passed": False, "error": repr(exc),
                        "traceback": traceback.format_exc()})
        out_fh.write(json.dumps(rec, default=str) + "\\n"); out_fh.flush()
        stage("q_done", idx=idx+1, qid=qid, passed=rec.get("passed"),
              elapsed=round(rec.get("elapsed_seconds") or 0, 1),
              tokens=(rec.get("prompt_tokens",0) or 0)+(rec.get("completion_tokens",0) or 0))
'''
    else:
        raise SystemExit(f"unknown strategy {strategy}")

    cells.append(cell_code(run_cell))

    # Wrap-up cell --------------------------------------------------------
    wrap = '''by_template = {}
all_rows = [json.loads(l) for l in RESULTS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
total = len(all_rows); passed = sum(1 for r in all_rows if r.get("passed"))
total_prompt = sum(r.get("prompt_tokens",0) or 0 for r in all_rows)
total_completion = sum(r.get("completion_tokens",0) or 0 for r in all_rows)
total_elapsed = sum(r.get("elapsed_seconds",0) or 0 for r in all_rows)
for r in all_rows:
    t = r.get("template","?")
    bt = by_template.setdefault(t, {"n":0, "passed":0})
    bt["n"] += 1
    if r.get("passed"): bt["passed"] += 1
summary["results_summary"] = {
    "n": total, "n_passed": passed,
    "pass_rate": (passed/total) if total else 0.0,
    "total_prompt_tokens": total_prompt,
    "total_completion_tokens": total_completion,
    "total_tokens": total_prompt + total_completion,
    "total_elapsed_seconds": total_elapsed,
    "by_template": by_template,
}
write_summary()
stage("done")
print(json.dumps(summary["results_summary"], indent=2, default=str))
'''
    cells.append(cell_code(wrap))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python3.12"},
            "language_info": {"name": "python", "version": "3.12"},
            "kernel_info": {"name": "jupyter", "jupyter_kernel_name": "python3.12"},
            "microsoft": {"language": "python", "language_group": "jupyter_python"},
            "dependencies": {"lakehouse": {"default_lakehouse": LH_ID,
                                            "default_lakehouse_name": LH_NAME,
                                            "default_lakehouse_workspace_id": WS_ID}},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("strategy", choices=sorted(STRATEGY_INFO))
    p.add_argument("--smoke", type=int, default=None)
    p.add_argument("--run-id", required=True, help="shared RUN_ID for joining across strategies")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="cs",
                   help="cs = LongCoT cs/hard regression25; aqua = AQuA-RAT 15")
    p.add_argument("--out-dir", default="notebooks")
    p.add_argument("--effort", default="minimal", choices=["minimal","low","medium","high"],
                   help="reasoning_effort for base FabricLM")
    args = p.parse_args()

    nb = build(args.strategy, args.smoke, args.run_id, dataset=args.dataset, base_effort=args.effort)
    label = STRATEGY_INFO[args.strategy]["label"]
    suffix = f"_smoke{args.smoke}" if args.smoke else ""
    out = Path(args.out_dir) / f"comparison_5way_{args.strategy}_{label}_{args.dataset}{suffix}.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}  cells={len(nb['cells'])}  run_id={args.run_id}  dataset={args.dataset}  smoke={args.smoke}")


if __name__ == "__main__":
    main()
