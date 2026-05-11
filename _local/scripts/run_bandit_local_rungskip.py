"""Local Strategy-F (EffortBanditPolicy) runner — proves rung-skip fix on real bench.

Mirrors notebooks/comparison_5way_F_fabric_bandit.ipynb but runs locally via OpenRouter
so we can exercise the rung-skip code path on real holdout questions without Fabric.

Key difference vs. prior Fabric notebook: uses a 4-rung ladder
("minimal","low","medium","high") to actually exercise rungs >= 1 with the
bandit warm-state, which is the path that crashed pre-fix (KeyError max_tokens
when EffortLadderPolicy.build_config emitted lm_instance=None for rung>=1).

Usage:
    python scripts/run_bandit_local_rungskip.py --run-id RUNID --smoke 5
    python scripts/run_bandit_local_rungskip.py --run-id RUNID            # full 25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "bench" / "adaptive" / "longcot_cs_hard_holdout25.jsonl"
SESSION_FILES = Path(
    r"C:\Users\sandeeppawar\.copilot\session-state\83674b05-6393-422f-bd1f-ee20b1f0502a\files"
)
LOCAL_BASE = SESSION_FILES / "comparison_5way_local"

# Reuse the grader from the existing local runner
from scripts.run_comparison_5way_dspy_local import grade  # noqa: E402


def build_openrouter_lm(model: str = "openai/gpt-4o-mini"):
    """Build a dspy.LM pointed at OpenRouter."""
    import dspy
    import litellm
    # OpenRouter routes to upstreams that may not accept reasoning_effort
    # (e.g. plain gpt-4o-mini). Drop unsupported params so calls succeed —
    # the rung-skip fix we're validating is independent of reasoning_effort
    # actually being honored upstream.
    litellm.drop_params = True
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return dspy.LM(
        model=f"openrouter/{model}",
        api_key=key,
        api_base="https://openrouter.ai/api/v1",
        cache=False,
        max_tokens=16000,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--smoke", type=int, default=None,
                   help="run only first N questions (1 per template if --by-template)")
    p.add_argument("--by-template", action="store_true",
                   help="when --smoke N, take 1 question per template until N templates")
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--ladder", default="minimal,low,medium,high",
                   help="comma-separated effort levels for the ladder")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max-attempts", type=int, default=4,
                   help="adaptive max_attempts; must be >= len(ladder) to walk all rungs")
    p.add_argument("--preseed-rung", type=int, default=None,
                   help="if set, pre-seed bandit_state for every template "
                   "with N successes at this rung (forces warm-state hop)")
    p.add_argument("--preseed-n", type=int, default=3)
    args = p.parse_args()

    label = "fabric_bandit_local_rungskip"
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
        if args.by_template:
            seen = set()
            picked = []
            for r in rows:
                t = r["template"]
                if t in seen:
                    continue
                seen.add(t)
                picked.append(r)
                if len(picked) >= args.smoke:
                    break
            rows = picked
        else:
            rows = rows[: args.smoke]

    os.environ["FABRIC_RLM_CAPTURE_TURNS"] = "1"
    ladder = tuple(s.strip() for s in args.ladder.split(",") if s.strip())

    summary = {
        "tier": "comparison_5way_local",
        "run_id": args.run_id,
        "strategy": "F",
        "strategy_label": label,
        "engine": "adaptive (EffortBanditPolicy)",
        "model": args.model,
        "ladder": list(ladder),
        "warmup": args.warmup,
        "started_at": time.time(),
        "n": len(rows),
        "stages": [],
        "results_summary": {},
    }

    def write_summary():
        summary["elapsed_seconds"] = time.time() - summary["started_at"]
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

    def stage(name, **info):
        entry = {"stage": name, "t": round(time.time() - summary["started_at"], 1), **info}
        summary["stages"].append(entry)
        write_summary()
        print("[stage]", name, info, flush=True)

    try:
        import fabric_rlm
        from fabric_rlm import RLM
        from fabric_rlm.experimental import (
            EffortBanditPolicy, BanditState, EFFORT_RUNG_COST,
        )
        stage("imported", fabric_rlm=fabric_rlm.__version__)
    except Exception as exc:
        stage("import_failed", error=repr(exc), tb=traceback.format_exc())
        raise

    try:
        base_lm = build_openrouter_lm(args.model)
        stage("lm_built", model=args.model)
    except Exception as exc:
        stage("lm_failed", error=repr(exc))
        raise

    bandit_state_path = run_root / f"bandit_state_{label}.json"
    bandit_state = BanditState.from_path(bandit_state_path)
    stage("bandit_state_loaded", path=str(bandit_state_path),
          n_keys=len(bandit_state.priors))

    # Optional pre-seed: simulate a warm bandit so it hops directly to a
    # high rung. Pre-fix this would crash with KeyError(max_tokens). Post-fix
    # it succeeds. This is the targeted test of the rung-skip fix.
    if args.preseed_rung is not None:
        templates = sorted({r["template"] for r in rows})
        for tpl in templates:
            for _ in range(args.preseed_n):
                bandit_state.record(tpl, args.preseed_rung, True)
        bandit_state.save()
        stage("bandit_preseeded", rung=args.preseed_rung, n=args.preseed_n,
              templates=templates)

    n_pass = 0
    n_total = 0
    rung_hist = {}
    err_count = 0
    rungskip_triggered = 0  # count of attempts where starting_rung >= 1

    with results_path.open("w", encoding="utf-8") as out_fh:
        for idx, row in enumerate(rows):
            qid = row["question_id"]
            tpl = row["template"]
            gold = row.get("answer")
            rec = {"strategy": label, "question_id": qid, "template": tpl,
                   "started_at": time.time()}
            try:
                def _validator(result, _gold=gold, _tpl=tpl):
                    if not result.submitted or not result.payload:
                        return False
                    ans = result.payload.get("answer")
                    return ans is not None and grade(_tpl, _gold, ans)

                policy = EffortBanditPolicy(
                    base_lm_instance=base_lm,
                    base_reasoning_effort="minimal",
                    parallel_rollouts=1,
                    effort_ladder=ladder,
                    state=bandit_state,
                    task_key=tpl,
                    warmup=args.warmup,
                    rung_cost=EFFORT_RUNG_COST,
                )
                pre_obs = bandit_state.total_observations(tpl)
                pre_betas = {r: bandit_state.beta_for(tpl, r) for r in range(len(ladder))}

                rlm = RLM(signature="question -> answer", lm=base_lm,
                          engine="adaptive",
                          adaptive=dict(policy=policy, validator=_validator,
                                        max_attempts=args.max_attempts, parallel_rollouts=1))
                t0 = time.perf_counter()
                result = rlm.run({"question": row["prompt"]})
                elapsed = time.perf_counter() - t0
                ans = (result.payload or {}).get("answer") if result.payload else None
                traj = result.trajectory
                meta = (traj.metadata or {}).get("adaptive", {}) if traj is not None else {}
                attempts = meta.get("attempts", [])
                passed = bool(result.submitted) and grade(tpl, gold, ans) if ans is not None else False
                for a in attempts:
                    rung_i = a.get("rung")
                    if rung_i is None:
                        continue
                    bandit_state.record(tpl, int(rung_i), bool(a.get("passed")))
                try:
                    bandit_state.save()
                except Exception as _se:
                    stage("bandit_save_warn", err=repr(_se))
                post_betas = {r: bandit_state.beta_for(tpl, r) for r in range(len(ladder))}
                start_r = attempts[0].get("rung") if attempts else None
                win_r = meta.get("winner_rung")
                if isinstance(start_r, int) and start_r >= 1:
                    rungskip_triggered += 1
                if win_r is not None:
                    rung_hist[win_r] = rung_hist.get(win_r, 0) + 1

                rec.update({
                    "passed": bool(passed), "submitted": result.submitted,
                    "elapsed_seconds": elapsed,
                    "starting_rung": start_r,
                    "winner_rung": win_r,
                    "stop_reason": meta.get("stop_reason"),
                    "n_attempts": len(attempts),
                    "n_turns": sum(a.get("turns_used") or 0 for a in attempts),
                    "answer_preview": (str(ans)[:1000] if ans is not None else None),
                    "bandit_pre_observations": pre_obs,
                    "bandit_pre_betas": pre_betas,
                    "bandit_post_betas": post_betas,
                })
                trace = {"strategy": label, "question_id": qid, "template": tpl,
                         "prompt": row["prompt"], "gold": gold,
                         "answer": str(ans) if ans is not None else None,
                         "submitted": result.submitted, "passed": rec["passed"],
                         "winner_rung": rec["winner_rung"], "starting_rung": rec["starting_rung"],
                         "stop_reason": rec["stop_reason"],
                         "bandit_pre_observations": pre_obs,
                         "bandit_pre_betas": pre_betas, "bandit_post_betas": post_betas,
                         "attempts": [
                             {"rung": a.get("rung"), "passed": a.get("passed"),
                              "submitted": a.get("submitted"),
                              "turns_used": a.get("turns_used"),
                              "failure_reason": a.get("failure_reason"),
                              "answer_preview": str(((a.get("payload_preview") or {}).get("answer")) or "")[:500]}
                             for a in attempts]}
                (traces_dir / f"trace_{qid}.json").write_text(
                    json.dumps(trace, default=str, indent=2), encoding="utf-8"
                )
                n_total += 1
                if passed:
                    n_pass += 1
            except Exception as exc:
                err_count += 1
                rec.update({"passed": False, "error": repr(exc),
                            "traceback": traceback.format_exc()})
                # Save error trace
                (traces_dir / f"trace_{qid}.json").write_text(
                    json.dumps({"strategy": label, "question_id": qid, "template": tpl,
                                "error": repr(exc), "traceback": traceback.format_exc()},
                               indent=2),
                    encoding="utf-8",
                )
            out_fh.write(json.dumps(rec, default=str) + "\n")
            out_fh.flush()
            stage("q_done", idx=idx + 1, qid=qid, tpl=tpl,
                  passed=rec.get("passed"), winner_rung=rec.get("winner_rung"),
                  starting_rung=rec.get("starting_rung"),
                  elapsed=round(rec.get("elapsed_seconds") or 0, 1))

    summary["results_summary"] = {
        "n_total": n_total,
        "n_pass": n_pass,
        "pass_rate": (n_pass / n_total) if n_total else None,
        "winner_rung_histogram": rung_hist,
        "rungskip_triggered": rungskip_triggered,
        "errors": err_count,
    }
    write_summary()
    print(f"\nFINAL: pass={n_pass}/{n_total}  rung_hist={rung_hist}  rungskip={rungskip_triggered}  errors={err_count}")


if __name__ == "__main__":
    main()
