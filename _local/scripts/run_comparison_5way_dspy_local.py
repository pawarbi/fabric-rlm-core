"""Local runner for strategy B (DSPy RLM = fabric_rlm engine='v7-dspy').

Runs strategy B locally on the same 25 holdout questions, mirroring the
Fabric notebook output layout so analyze_5way.py can join all 5 strategies.

Usage:
    python scripts/run_comparison_5way_dspy_local.py --run-id 20260502-XXX [--smoke N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "bench" / "adaptive" / "longcot_cs_hard_holdout25.jsonl"

# session-state files dir mirrors lakehouse layout
SESSION_FILES = Path(
    r"C:\Users\sandeeppawar\.copilot\session-state\83674b05-6393-422f-bd1f-ee20b1f0502a\files"
)
LOCAL_BASE = SESSION_FILES / "comparison_5way_local"

CS_JSON_OBJECT_TEMPLATES = {"HM", "MFMC", "Scheduling", "TM", "MCM", "LLVM"}
CS_INTEGER_TEMPLATES = {"VLIW", "CodeTrace"}
CS_INTEGER_LIST_TEMPLATES = {"Backprop", "DistMem"}
INT_RE = re.compile(r"-?\d+")
INT_CSV_RE = re.compile(r"-?\d+(?:\s*,\s*-?\d+)+")


def _resp_text(resp) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    return str(resp)


def _extract_solution(text: str) -> str | None:
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip() or None


def _extract_last_json_object(text: str):
    if not text:
        return None
    last = None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                last = text[start:i + 1]
                start = -1
    if not last:
        return None
    try:
        return json.loads(last)
    except Exception:
        try:
            return json.loads(last.replace("'", '"'))
        except Exception:
            return None


def _parse_int_list(text):
    if not text:
        return None
    m = INT_CSV_RE.search(text)
    if m:
        return [int(x.strip()) for x in m.group(0).split(",")]
    nums = INT_RE.findall(text)
    return [int(n) for n in nums] if nums else None


def grade(template, gold_answer, response_text) -> bool:
    text = _resp_text(response_text)
    sol = _extract_solution(text) or text
    expected = gold_answer
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except Exception:
            pass
    if template in CS_JSON_OBJECT_TEMPLATES:
        cand = _extract_last_json_object(sol) or _extract_last_json_object(text)
        return cand == expected
    if template in CS_INTEGER_TEMPLATES:
        m = INT_RE.search(sol) or INT_RE.search(text)
        if m is None:
            return False
        try:
            return int(m.group(0)) == int(str(expected).strip())
        except Exception:
            return False
    if template in CS_INTEGER_LIST_TEMPLATES:
        if isinstance(expected, list):
            exp_list = [int(x) for x in expected]
        else:
            exp_list = _parse_int_list(str(expected))
        pred = _parse_int_list(sol) or _parse_int_list(text)
        return pred == exp_list
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--smoke", type=int, default=None)
    args = p.parse_args()

    label = "dspy"
    strategy = "B"

    run_root = LOCAL_BASE / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / f"summary_{label}.json"
    results_path = run_root / f"results_{label}.jsonl"
    traces_dir = run_root / f"traces_{label}"
    traces_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.smoke:
        rows = rows[: args.smoke]

    os.environ["FABRIC_RLM_CAPTURE_TURNS"] = "1"

    # try v7-dspy locally; if it fails to set up, abort early with a clear note.
    summary = {
        "tier": "comparison_5way",
        "run_id": args.run_id,
        "strategy": strategy,
        "strategy_label": label,
        "started_at": time.time(),
        "smoke_n": args.smoke,
        "n": len(rows),
        "engine": "v7-dspy (local)",
        "stages": [],
        "results_summary": {},
    }

    def write_summary():
        summary["elapsed_seconds"] = time.time() - summary["started_at"]
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

    def stage(name, **info):
        summary["stages"].append({"stage": name,
                                  "t": round(time.time() - summary["started_at"], 1),
                                  **info})
        write_summary()
        print("[stage]", name, info)

    try:
        import dspy
        import fabric_rlm
        from fabric_rlm import RLM, FabricLM
        stage("imported", dspy=dspy.__version__, fabric_rlm=fabric_rlm.__version__)
    except Exception as exc:
        stage("import_failed", error=repr(exc))
        raise

    try:
        base_lm = FabricLM("gpt-5", reasoning_effort="minimal", cache=False)
        stage("lm_built", model="gpt-5")
    except Exception as exc:
        stage("lm_failed", error=repr(exc), traceback=traceback.format_exc())
        raise

    with results_path.open("w", encoding="utf-8") as out_fh:
        for idx, row in enumerate(rows):
            qid = row["question_id"]
            tpl = row["template"]
            gold = row.get("answer")
            rec = {"strategy": label, "question_id": qid, "template": tpl,
                   "started_at": time.time()}
            try:
                rlm = RLM(signature="question -> answer", lm=base_lm,
                          engine="v7-dspy", max_turns=8)
                t0 = time.perf_counter()
                result = rlm.run({"question": row["prompt"]})
                elapsed = time.perf_counter() - t0
                ans = (result.payload or {}).get("answer") if result.payload else None
                traj = result.trajectory
                meta = traj.metadata if traj else {}
                turns = (meta or {}).get("turns")
                usage = (meta or {}).get("usage") or {}
                passed = bool(result.submitted) and (
                    grade(tpl, gold, ans) if ans is not None else False
                )
                rec.update({
                    "passed": bool(passed),
                    "submitted": result.submitted,
                    "elapsed_seconds": elapsed,
                    "prompt_tokens": usage.get("prompt_tokens") or 0,
                    "completion_tokens": usage.get("completion_tokens") or 0,
                    "n_attempts": 1,
                    "n_turns": len(turns) if turns else None,
                    "answer_preview": (str(ans)[:1000] if ans is not None else None),
                })
                trace = {
                    "strategy": label, "question_id": qid, "template": tpl,
                    "prompt": row["prompt"],
                    "answer": str(ans) if ans is not None else None,
                    "submitted": result.submitted, "passed": rec["passed"],
                    "turns": turns, "metadata": meta,
                }
                (traces_dir / f"trace_{qid}.json").write_text(
                    json.dumps(trace, default=str, indent=2), encoding="utf-8"
                )
            except Exception as exc:
                rec.update({"passed": False, "error": repr(exc),
                           "traceback": traceback.format_exc()})
            out_fh.write(json.dumps(rec, default=str) + "\n")
            out_fh.flush()
            stage("q_done", idx=idx + 1, qid=qid, passed=rec.get("passed"),
                  elapsed=round(rec.get("elapsed_seconds") or 0, 1),
                  tokens=(rec.get("prompt_tokens", 0) or 0) +
                         (rec.get("completion_tokens", 0) or 0))

    by_template: dict = {}
    all_rows = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(all_rows)
    n_pass = sum(1 for r in all_rows if r.get("passed"))
    for r in all_rows:
        bt = by_template.setdefault(r.get("template", "?"), {"n": 0, "passed": 0})
        bt["n"] += 1
        if r.get("passed"):
            bt["passed"] += 1
    summary["results_summary"] = {
        "n": total, "n_passed": n_pass,
        "pass_rate": (n_pass / total) if total else 0.0,
        "total_prompt_tokens": sum(r.get("prompt_tokens", 0) or 0 for r in all_rows),
        "total_completion_tokens": sum(r.get("completion_tokens", 0) or 0 for r in all_rows),
        "total_elapsed_seconds": sum(r.get("elapsed_seconds", 0) or 0 for r in all_rows),
        "by_template": by_template,
    }
    write_summary()
    stage("done")
    print(json.dumps(summary["results_summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
