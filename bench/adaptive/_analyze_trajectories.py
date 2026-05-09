"""
Trajectory analyzer for SRLM bench results.

Ingests every result JSON under bench/adaptive/results/<run-dir>/ and produces
five cuts that speak to the *meta* question: "is SRLM worth it, and do the
trajectories teach us something novel we could turn into a method?"

Cuts:
  1. Cost/accuracy frontier (per config)
  2. Rung-3 payoff (did the extra rollouts flip a rung-1 wrong answer to right?)
  3. Wasted compute (rung-3 BoN where every candidate already agreed)
  4. Validator value-add (rollouts where validator pass/fail split the field)
  5. Cluster discrimination (per cluster_size, what fraction passed?)

Output: markdown to stdout (or --out file). Pure stdlib, no plotting.

Run:
    python bench/adaptive/_analyze_trajectories.py \
        --results-dirs bench/adaptive/results/srlm_eval bench/adaptive/results/srlm_eval_p3 \
        --out bench/adaptive/trajectory_analysis.md
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def _load_results(dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in dirs:
        if not d.exists():
            continue
        for f in d.rglob("*.json"):
            try:
                obj = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(obj, dict):
                continue
            if "config_name" not in obj or "passed" not in obj:
                continue
            obj["__source__"] = str(f)
            rows.append(obj)
    return rows


# --------------------------------------------------------------------------- #
# selector_key shape helpers
#
# selector_key layout (from select_best_of_n):
#   [0] passed              (1/0)
#   [1] score               (-inf if absent)
#   [2] conf                (-inf if absent)
#   [3] required_filled     (count)
#   [4] total_non_blank     (count)
#   [5] cluster_size        (when consensus enabled, else -rollout_idx)
#   [6] -trace_length OR -rollout_idx
#   [7] -rollout_idx        (only if both A and C enabled)
# --------------------------------------------------------------------------- #


def _passed_flag(key: list[Any]) -> int:
    if not key:
        return 0
    try:
        return int(key[0])
    except Exception:  # noqa: BLE001
        return 0


# --------------------------------------------------------------------------- #
# Cut 1: cost/accuracy frontier
# --------------------------------------------------------------------------- #

def _frontier(rows: list[dict[str, Any]]) -> str:
    by_cfg: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"pass": [], "tokens": [], "elapsed": []}
    )
    for r in rows:
        cfg = r["config_name"]
        by_cfg[cfg]["pass"].append(1.0 if r.get("passed") else 0.0)
        by_cfg[cfg]["tokens"].append(float(r.get("total_cost_tokens") or 0))
        by_cfg[cfg]["elapsed"].append(float(r.get("elapsed_s") or 0))

    lines = ["## 1. Cost / accuracy frontier", ""]
    lines.append("| config | n | accuracy | mean tokens | median tokens | mean elapsed (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cfg in sorted(by_cfg):
        s = by_cfg[cfg]
        n = len(s["pass"])
        acc = mean(s["pass"]) if s["pass"] else 0.0
        lines.append(
            f"| `{cfg}` | {n} | {acc:.3f} | "
            f"{mean(s['tokens']):.0f} | {median(s['tokens']):.0f} | "
            f"{mean(s['elapsed']):.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Cut 2: rung-3 payoff — did extra rollouts flip a wrong rung-1 answer?
# We approximate via observability: at the WINNING rung, look at the spread
# of the `passed` flag across rollouts. If some rollouts pass and others don't,
# BoN selection had real choice; if all-pass or all-fail, BoN was tautological.
# --------------------------------------------------------------------------- #

def _rung_payoff(rows: list[dict[str, Any]]) -> str:
    by_cfg: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "rung3_total": 0,
            "rung3_K1": 0,             # only one candidate at winning rung
            "rung3_all_pass": 0,       # K>1 and all candidates pass validator
            "rung3_all_fail": 0,       # K>1 and all candidates fail
            "rung3_split_won": 0,      # K>1, validator split, AND we landed on a passing candidate
            "rung3_split_lost": 0,     # K>1, validator split, AND we landed on a failing candidate
        }
    )

    for r in rows:
        cfg = r["config_name"]
        winning_rung = r.get("winner_rung")
        if winning_rung != 3:
            continue
        obs = r.get("observability") or []
        if not obs:
            continue

        s = by_cfg[cfg]
        s["rung3_total"] += 1

        if len(obs) == 1:
            s["rung3_K1"] += 1
            continue

        passes = [_passed_flag(o.get("selector_key") or []) for o in obs]
        if all(p == 1 for p in passes):
            s["rung3_all_pass"] += 1
        elif all(p == 0 for p in passes):
            s["rung3_all_fail"] += 1
        else:
            # validator split → BoN had real choice
            if r.get("passed"):
                s["rung3_split_won"] += 1
            else:
                s["rung3_split_lost"] += 1

    lines = [
        "## 2. Rung-3 payoff (did extra rollouts buy us anything?)",
        "",
        "_For rollouts that reached rung 3: was BoN choosing among meaningfully",
        "different candidates, or rubber-stamping unanimous output?_",
        "",
        "| config | rung3 rollouts | K=1 | all pass (no choice) | all fail (no rescue) | validator split (won) | validator split (lost) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg in sorted(by_cfg):
        s = by_cfg[cfg]
        lines.append(
            f"| `{cfg}` | {s['rung3_total']} | {s['rung3_K1']} | "
            f"{s['rung3_all_pass']} | {s['rung3_all_fail']} | "
            f"{s['rung3_split_won']} | {s['rung3_split_lost']} |"
        )
    lines.append("")
    lines.append(
        "**Read:** large `all pass` columns ⇒ rung-3 BoN burned tokens to confirm "
        "the obvious — early-exit candidate. Large `all fail` columns ⇒ rung-3 "
        "didn't rescue rung-1 wrongness — escalation policy needs work."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Cut 3: wasted compute via cluster size
# At the winning rung, if every candidate is in the SAME cluster (all
# `consensus_cluster_size == K`), the model already agreed before we picked.
# --------------------------------------------------------------------------- #

def _wasted_compute(rows: list[dict[str, Any]]) -> str:
    by_cfg: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rung3": 0, "unanimous": 0, "split": 0, "all_singletons": 0}
    )

    for r in rows:
        if r.get("winner_rung") != 3:
            continue
        obs = r.get("observability") or []
        if len(obs) < 2:
            continue
        cfg = r["config_name"]
        s = by_cfg[cfg]
        s["rung3"] += 1
        sizes = [int(o.get("consensus_cluster_size") or 0) for o in obs]
        K = len(sizes)
        if all(sz == K for sz in sizes):
            s["unanimous"] += 1
        elif all(sz <= 1 for sz in sizes):
            s["all_singletons"] += 1
        else:
            s["split"] += 1

    lines = [
        "## 3. Wasted compute (was rung-3 unanimous?)",
        "",
        "_How often did rung-3 BoN spend ~K× the tokens just to confirm a unanimous answer?_",
        "",
        "| config | rung3 rollouts (K>1) | unanimous (waste) | split (real choice) | all singletons (model disagreed entirely) |",
        "|---|---:|---:|---:|---:|",
    ]
    for cfg in sorted(by_cfg):
        s = by_cfg[cfg]
        n = s["rung3"]
        lines.append(
            f"| `{cfg}` | {n} | {s['unanimous']} ({s['unanimous']/max(n,1):.0%}) | "
            f"{s['split']} ({s['split']/max(n,1):.0%}) | "
            f"{s['all_singletons']} ({s['all_singletons']/max(n,1):.0%}) |"
        )
    lines.append("")
    lines.append(
        "**Read:** if `unanimous` is a high %, the novel-method opportunity is "
        "**early exit** — skip rung-3 BoN when rung-1's answer already looks confident."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Cut 4: validator value-add — score vs validator agreement
# When score is present (selector_key[1] != -inf), do high-score candidates
# also tend to pass the validator? If not, score and validator disagree, and
# the validator is doing real work.
# --------------------------------------------------------------------------- #

def _validator_value_add(rows: list[dict[str, Any]]) -> str:
    # Aggregate per (cfg) over all candidates seen at any rung's BoN.
    by_cfg: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "n_candidates": 0,
            "with_score": 0,
            "score_ok_pass": 0,
            "score_ok_fail": 0,
            "score_low_pass": 0,
            "score_low_fail": 0,
        }
    )

    for r in rows:
        cfg = r["config_name"]
        obs = r.get("observability") or []
        # collect scores for this rollout to derive a relative threshold
        scores = []
        for o in obs:
            key = o.get("selector_key") or []
            if len(key) >= 2 and isinstance(key[1], (int, float)) and not math.isinf(key[1]):
                scores.append(float(key[1]))
        if not scores:
            for o in obs:
                by_cfg[cfg]["n_candidates"] += 1
            continue
        thr = median(scores)
        for o in obs:
            by_cfg[cfg]["n_candidates"] += 1
            key = o.get("selector_key") or []
            if not key:
                continue
            passed = _passed_flag(key) == 1
            score = key[1] if len(key) >= 2 else float("-inf")
            if isinstance(score, (int, float)) and not math.isinf(score):
                by_cfg[cfg]["with_score"] += 1
                high = score >= thr
                if high and passed:
                    by_cfg[cfg]["score_ok_pass"] += 1
                elif high and not passed:
                    by_cfg[cfg]["score_ok_fail"] += 1
                elif not high and passed:
                    by_cfg[cfg]["score_low_pass"] += 1
                else:
                    by_cfg[cfg]["score_low_fail"] += 1

    lines = [
        "## 4. Validator value-add (does score predict pass?)",
        "",
        "_Among candidates that have a numeric score, split by score≥median and validator pass._",
        "",
        "| config | candidates w/ score | high-score+pass | high-score+fail | low-score+pass | low-score+fail | validator–score agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg in sorted(by_cfg):
        s = by_cfg[cfg]
        n = s["with_score"]
        agree = s["score_ok_pass"] + s["score_low_fail"]
        agree_pct = agree / n if n else 0.0
        lines.append(
            f"| `{cfg}` | {n} | {s['score_ok_pass']} | {s['score_ok_fail']} | "
            f"{s['score_low_pass']} | {s['score_low_fail']} | {agree_pct:.0%} |"
        )
    lines.append("")
    lines.append(
        "**Read:** if `validator–score agreement` is high, score alone might "
        "suffice — validator could be a cheaper rubric. If low, validator is "
        "catching things score misses (keep it)."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Cut 5: cluster size → pass rate (post-hoc consensus calibration)
# Across all observability rows, group by cluster_size; what fraction of
# candidates in size-N clusters actually pass the validator? This tells us
# whether self-consistency is itself a good signal.
# --------------------------------------------------------------------------- #

def _cluster_calibration(rows: list[dict[str, Any]]) -> str:
    # Pooled view (kept for back-compat / quick read).
    pooled: dict[int, dict[str, int]] = defaultdict(lambda: {"n": 0, "pass": 0})
    # Stratified: per (config, domain) — duck B2 fix. Pooled monotonicity
    # is dominated by easy/math (always-pass, always-cluster) and hides
    # the picture on harder domains.
    strata: dict[tuple[str, str], dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "pass": 0})
    )
    for r in rows:
        cfg = r["config_name"]
        dom = r.get("domain") or "?"
        for o in r.get("observability") or []:
            sz_raw = o.get("consensus_cluster_size")
            if sz_raw is None:
                continue
            sz = int(sz_raw)
            passed = _passed_flag(o.get("selector_key") or []) == 1
            pooled[sz]["n"] += 1
            strata[(cfg, dom)][sz]["n"] += 1
            if passed:
                pooled[sz]["pass"] += 1
                strata[(cfg, dom)][sz]["pass"] += 1

    lines = [
        "## 5. Consensus calibration (does cluster size predict correctness?)",
        "",
        "_Across every candidate ever scored, group by cluster size, report pass rate._",
        "",
        "### 5a. Pooled (CONFOUNDED — read 5b first)",
        "",
        "| cluster size | candidates | pass rate |",
        "|---:|---:|---:|",
    ]
    for sz in sorted(pooled):
        s = pooled[sz]
        lines.append(f"| {sz} | {s['n']} | {s['pass']/max(s['n'],1):.3f} |")
    lines.append("")
    lines.append(
        "> ⚠️ **The pooled monotone trend is a Simpson's-paradox artifact.** "
        "Easy/math domains have near-100% pass AND high consensus, so any "
        "pooled view on a question set with easy domains will show "
        "size↑ → pass↑ even when consensus has zero signal on the hard "
        "domain that actually matters. Always read 5b."
    )
    lines.append("")
    lines.append("### 5b. Stratified by (config × domain)")
    lines.append("")
    lines.append("| config | domain | size 1 | size 2 | size 3+ |")
    lines.append("|---|---|---|---|---|")
    for cfg, dom in sorted(strata):
        s = strata[(cfg, dom)]
        s1 = s.get(1, {"n": 0, "pass": 0})
        s2 = s.get(2, {"n": 0, "pass": 0})
        s3 = {"n": 0, "pass": 0}
        for sz, b in s.items():
            if sz >= 3:
                s3["n"] += b["n"]
                s3["pass"] += b["pass"]

        def _fmt(b: dict[str, int]) -> str:
            if b["n"] == 0:
                return "—"
            return f"{b['pass']}/{b['n']}={b['pass']/b['n']:.2f}"

        lines.append(
            f"| `{cfg}` | {dom} | {_fmt(s1)} | {_fmt(s2)} | {_fmt(s3)} |"
        )
    lines.append("")
    lines.append(
        "**Read:** look for monotone size↑ → pass↑ within a SINGLE row "
        "(one config × one domain). If that fails on the domain we "
        "actually care about (DABench), self-consistency is not a useful "
        "signal there and Feature C should not be expected to help."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Cut 6: per-domain pass rate (sanity)
# --------------------------------------------------------------------------- #

def _per_domain(rows: list[dict[str, Any]]) -> str:
    by: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "pass": 0, "tokens": 0}
    )
    for r in rows:
        dom = r.get("domain") or "?"
        cfg = r["config_name"]
        b = by[(dom, cfg)]
        b["n"] += 1
        b["pass"] += 1 if r.get("passed") else 0
        b["tokens"] += int(r.get("total_cost_tokens") or 0)

    lines = ["## 6. Per-domain pass rate × config", ""]
    lines.append("| domain | config | n | pass rate | mean tokens |")
    lines.append("|---|---|---:|---:|---:|")
    for (dom, cfg) in sorted(by):
        b = by[(dom, cfg)]
        lines.append(
            f"| {dom} | `{cfg}` | {b['n']} | {b['pass']/max(b['n'],1):.3f} | "
            f"{b['tokens']/max(b['n'],1):.0f} |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="analyze_trajectories")
    p.add_argument(
        "--results-dirs",
        nargs="+",
        required=True,
        help="One or more directories under bench/adaptive/results/",
    )
    p.add_argument("--out", default="-", help="output markdown file (default stdout)")
    args = p.parse_args(argv)

    dirs = [Path(d) for d in args.results_dirs]
    rows = _load_results(dirs)
    if not rows:
        print("# No result rows found.", end="\n")
        return 1

    parts = [
        f"# SRLM trajectory analysis\n",
        f"_Sources: {', '.join(str(d) for d in dirs)}_  ",
        f"_Total rollouts ingested: {len(rows)}_",
        "",
        _frontier(rows),
        _per_domain(rows),
        _rung_payoff(rows),
        _wasted_compute(rows),
        _validator_value_add(rows),
        _cluster_calibration(rows),
    ]
    out = "\n".join(parts)

    if args.out == "-":
        print(out)
    else:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
