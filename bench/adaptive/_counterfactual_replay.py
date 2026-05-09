"""Counterfactual selector replay — isolates Feature C's CAUSAL effect.

Background
----------
The Phase 3 A/B bench compared four configs (baseline, +A, +C, +A+C) and
observed +17pp DABench lift for ``adaptive_c_minrung3``. Duck review (B1)
flagged that this comparison conflates two effects:

1. **Selection effect** (what we wanted to measure) — does the C selector
   pick a better candidate from a fixed candidate set?
2. **Sampling effect** (confound) — different configs run different
   rollouts, so they sample DIFFERENT candidate sets from the LM. Even
   with identical seeds, thread interleaving and any flag-driven prompt
   differences shift which 5 candidates appear in rung 3.

Critically, the bench validator is the *first* sort key in
``select_best_of_n``. So:

* If the candidate set has ≥1 passing candidate → all configs pick a
  passing candidate → final pass/fail identical.
* If the candidate set has 0 passing candidates → all configs pick a
  failing candidate → final pass/fail identical.
* Selector C can only differentiate **which** passing (or which failing)
  candidate is picked — not whether the rollout passes overall.

Therefore the +17pp lift CANNOT be selector C's doing. It is sampling
variance: ``adaptive_c_minrung3`` happened to draw more candidate sets
that contained a passing candidate.

What this analyzer does
-----------------------
Take a single config's rollouts (we use ``adaptive_current_minrung3``,
the baseline). For each rung-3 rollout, take the candidate observability
rows AS-IS, and compute who would have won under three selector keys:

* **baseline_key** — ``(passed, score, conf, rf, tn, -idx)``
* **C_key**        — ``(passed, score, conf, rf, tn, cluster_size, -idx)``
* **A_key**        — ``(passed, score, conf, rf, tn, -trace_len, -idx)``

We then check: did the chosen winner CHANGE? Did the **pass/fail**
outcome change? (It can only change if the validator is imperfect, e.g.
two passing candidates have different correctness — which the bench
validator can't know about. So pass/fail flips are essentially
impossible. The interesting metric is winner-flip rate.)

Output: per-domain table of (n rollouts, winner flips, pass/fail flips,
flip-direction) for each alternative selector vs baseline.

This is the closest thing to a causal estimate of Feature C we can
obtain WITHOUT a full new bench. If C's flip rate is ~0% on the same
candidate sets, the +17pp lift is provably noise.

Usage::

    python bench/adaptive/_counterfactual_replay.py \\
        --results-dir bench/adaptive/results/srlm_eval_p3 \\
        --source-config adaptive_current_minrung3 \\
        --output bench/adaptive/p3_counterfactual.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from glob import glob
from typing import Any


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        # Treat -inf / inf the way the live selector does
        return f
    return f


def _key_baseline(o: dict[str, Any], idx: int) -> tuple:
    sk = o.get("selector_key") or []
    # Original key (defensive default if missing)
    if len(sk) >= 6:
        return tuple(sk[:6])
    # Reconstruct from observability fields
    passed = 1 if o.get("validator_passed") else 0
    score = _safe_float(o.get("score"), -math.inf)
    conf = _safe_float(o.get("confidence"), -math.inf)
    rf = int(o.get("required_filled") or 0)
    tn = int(o.get("total_nonblank") or 0)
    return (passed, score, conf, rf, tn, -idx)


def _key_c(o: dict[str, Any], idx: int) -> tuple:
    sk = o.get("selector_key") or []
    passed = 1 if o.get("validator_passed") else (sk[0] if sk else 0)
    score = _safe_float(o.get("score"), -math.inf) if not sk else _safe_float(sk[1], -math.inf)
    conf = _safe_float(o.get("confidence"), -math.inf) if not sk else _safe_float(sk[2], -math.inf)
    rf = int(o.get("required_filled") or 0) if not sk else int(sk[3] or 0)
    tn = int(o.get("total_nonblank") or 0) if not sk else int(sk[4] or 0)
    cs = int(o.get("consensus_cluster_size") or 1)
    return (passed, score, conf, rf, tn, cs, -idx)


def _key_a(o: dict[str, Any], idx: int) -> tuple:
    sk = o.get("selector_key") or []
    passed = 1 if o.get("validator_passed") else (sk[0] if sk else 0)
    score = _safe_float(o.get("score"), -math.inf) if not sk else _safe_float(sk[1], -math.inf)
    conf = _safe_float(o.get("confidence"), -math.inf) if not sk else _safe_float(sk[2], -math.inf)
    rf = int(o.get("required_filled") or 0) if not sk else int(sk[3] or 0)
    tn = int(o.get("total_nonblank") or 0) if not sk else int(sk[4] or 0)
    # Feature A prefers SHORTER traces → negate completion_tokens.
    ct = _safe_float(o.get("completion_tokens"), 0.0)
    return (passed, score, conf, rf, tn, -ct, -idx)


def _winner(obs_rows: list[dict[str, Any]], keyfn) -> tuple[int, dict[str, Any]] | None:
    if not obs_rows:
        return None
    scored = [(keyfn(o, i), i, o) for i, o in enumerate(obs_rows)]
    scored.sort(key=lambda x: x[0], reverse=True)
    _, idx, win = scored[0]
    return idx, win


def _winner_passed(o: dict[str, Any]) -> bool:
    """Did the selected candidate pass the bench validator?"""
    sk = o.get("selector_key") or []
    if sk:
        return bool(sk[0])
    return bool(o.get("validator_passed"))


def replay(results_dir: str, source_config: str) -> dict[str, Any]:
    files = sorted(glob(os.path.join(results_dir, "*.json")))
    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: {
        "n_rung3": 0,
        "n_with_choice": 0,        # >1 candidate (replay is meaningful)
        "c_winner_flip": 0,        # C key picks different candidate than baseline
        "c_pass_flip": 0,          # ...AND that flip changes pass/fail
        "a_winner_flip": 0,
        "a_pass_flip": 0,
        "n_with_cluster_gt1": 0,   # cluster info actually says something
    })

    flip_examples: list[dict[str, Any]] = []

    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if d.get("config_name") != source_config:
            continue
        if d.get("winner_rung") != 3:
            # Counterfactual at rung 3 only (where BoN actually runs).
            continue
        obs = d.get("observability") or []
        # Only rung-3 observability rows (each row = one rung-3 candidate).
        rung3 = [o for o in obs if o.get("rung") == 3]
        if not rung3:
            # Fallback: assume all obs are rung-3 candidates if "rung" not stamped.
            rung3 = obs
        if not rung3:
            continue
        dom = d.get("domain", "?")
        bd = by_domain[dom]
        bd["n_rung3"] += 1
        if len(rung3) <= 1:
            continue
        bd["n_with_choice"] += 1

        sizes = [int(o.get("consensus_cluster_size") or 1) for o in rung3]
        if max(sizes) > 1:
            bd["n_with_cluster_gt1"] += 1

        base = _winner(rung3, _key_baseline)
        cwin = _winner(rung3, _key_c)
        awin = _winner(rung3, _key_a)
        if base is None:
            continue
        b_idx, b_obs = base
        c_idx, c_obs = cwin if cwin else base
        a_idx, a_obs = awin if awin else base

        if c_idx != b_idx:
            bd["c_winner_flip"] += 1
            cp_pass = _winner_passed(c_obs)
            bp_pass = _winner_passed(b_obs)
            if cp_pass != bp_pass:
                bd["c_pass_flip"] += 1
            if len(flip_examples) < 8:
                flip_examples.append({
                    "domain": dom,
                    "case": os.path.basename(f),
                    "selector": "C",
                    "baseline_pick": b_obs.get("candidate_answer_preview"),
                    "C_pick": c_obs.get("candidate_answer_preview"),
                    "baseline_pass": bp_pass,
                    "C_pass": cp_pass,
                    "cluster_sizes": sizes,
                })
        if a_idx != b_idx:
            bd["a_winner_flip"] += 1
            ap_pass = _winner_passed(a_obs)
            bp_pass = _winner_passed(b_obs)
            if ap_pass != bp_pass:
                bd["a_pass_flip"] += 1

    return {"by_domain": dict(by_domain), "flip_examples": flip_examples,
            "source_config": source_config, "n_files": len(files)}


def render_markdown(report: dict[str, Any]) -> str:
    out: list[str] = []
    out.append("# Counterfactual selector replay")
    out.append("")
    out.append(f"**Source config:** `{report['source_config']}`  ")
    out.append(f"**Source rollouts scanned:** {report['n_files']}  ")
    out.append("")
    out.append("## What this measures")
    out.append("")
    out.append(
        "For each rung-3 rollout produced by the source config, we replay "
        "three selector keys against the SAME captured candidate set:"
    )
    out.append("")
    out.append("- **baseline_key** — `(passed, score, conf, rf, tn, -idx)`")
    out.append("- **C_key**        — `(passed, score, conf, rf, tn, **cluster_size**, -idx)`")
    out.append("- **A_key**        — `(passed, score, conf, rf, tn, **-trace_len**, -idx)`")
    out.append("")
    out.append(
        "**winner_flip** = the alternative selector picked a DIFFERENT "
        "candidate than baseline. **pass_flip** = and that change ALSO "
        "changed the rollout's overall pass/fail. Pass-flips can only "
        "happen if the validator is imperfect within the candidate set "
        "(two candidates that the validator says PASS differ on the "
        "ground-truth grader). The bench validator IS the grader, so "
        "pass-flips are structurally near-zero — that's the duck's B1 "
        "argument made concrete."
    )
    out.append("")
    out.append("## Per-domain results")
    out.append("")
    out.append("| domain | rung3 | with_choice | cluster>1 | C_flip | C_pass_flip | A_flip | A_pass_flip |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    totals = {"n_rung3": 0, "n_with_choice": 0, "c_winner_flip": 0, "c_pass_flip": 0,
              "a_winner_flip": 0, "a_pass_flip": 0, "n_with_cluster_gt1": 0}
    for dom in sorted(report["by_domain"]):
        bd = report["by_domain"][dom]
        for k in totals:
            totals[k] += bd[k]
        out.append(
            f"| {dom} | {bd['n_rung3']} | {bd['n_with_choice']} | "
            f"{bd['n_with_cluster_gt1']} | {bd['c_winner_flip']} | "
            f"{bd['c_pass_flip']} | {bd['a_winner_flip']} | {bd['a_pass_flip']} |"
        )
    out.append(
        f"| **TOTAL** | **{totals['n_rung3']}** | **{totals['n_with_choice']}** | "
        f"**{totals['n_with_cluster_gt1']}** | **{totals['c_winner_flip']}** | "
        f"**{totals['c_pass_flip']}** | **{totals['a_winner_flip']}** | "
        f"**{totals['a_pass_flip']}** |"
    )
    out.append("")
    out.append("## Interpretation")
    out.append("")
    out.append(
        "- If `C_pass_flip` is ~0 on this same candidate set, then the "
        "+17pp DABench lift observed in the live A/B is sampling variance, "
        "not selector C's causal contribution.")
    out.append(
        "- `C_flip` > 0 with `C_pass_flip` ≈ 0 means C does change WHICH "
        "candidate is reported (potentially affecting downstream cost / "
        "answer character) but does NOT change the bench-grader outcome.")
    out.append(
        "- The right way to demonstrate a causal C lift is to either "
        "(a) replay against many rollouts with KNOWN ground truth that "
        "DIFFERS from the validator (impossible here — validator IS the "
        "grader), or (b) run a long-context bench where the validator is "
        "weaker and selector signal can dominate.")
    out.append("")
    if report["flip_examples"]:
        out.append("## Flip examples (first 8)")
        out.append("")
        for ex in report["flip_examples"]:
            out.append(
                f"- **{ex['domain']}** `{ex['case']}` "
                f"({ex['selector']}): cluster_sizes={ex['cluster_sizes']}; "
                f"baseline→`{(ex['baseline_pick'] or '')[:60]!r}` "
                f"(pass={ex['baseline_pass']}); "
                f"C→`{(ex['C_pick'] or '')[:60]!r}` (pass={ex['C_pass']})"
            )
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--source-config", default="adaptive_current_minrung3")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = replay(args.results_dir, args.source_config)
    md = render_markdown(report)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
