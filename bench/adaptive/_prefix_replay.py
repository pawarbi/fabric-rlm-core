"""Prefix-replay simulator for Feature E (early-exit policy).

Background
----------
Phase 3 trajectory analysis showed that ~80% of rung-3 spend buys no
decision: either all candidates pass the validator (decision was already
made at K=1) or all fail (additional rollouts can't rescue). A naive
"early-exit" reading of that is: stop launching rollouts once the prefix
tells us the outcome.

The duck flagged that the cost claim ONLY holds if the runtime stops
LAUNCHING the suffix — cancelling already-running futures doesn't
reliably reclaim tokens. Before we change `_run_rollouts` to a
batched fanout, we want to verify:

1. How often would the early-exit predicate fire on captured data?
2. What's the distribution of saved completion tokens (lower bound;
   we have per-candidate ``trace_length_completion`` but not per-candidate
   prompt/reasoning, so true savings are slightly higher)?
3. Does the safe-default predicate (``all_pass``) ever flip the
   selected winner or the rollout's overall pass/fail outcome?
4. Does the strict predicate (``all_fail_same_canonical``) flip outcomes
   often enough to disqualify it as a default?

This module is the offline simulator. It walks captured observability
rows in execution order (``selector_key[-1] = -idx``) and reports per-
rollout: did the predicate fire, at what K, and what would have flipped.

Aggregation lives in ``replay_dataset`` and renders a per-domain × per-K
markdown report — analogous to the Phase 3 ``_counterfactual_replay.py``
output, but for prefix-stop semantics rather than alternative selector
keys.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from typing import Any, Callable, Sequence


# ----- predicates -----


def _passed(o: dict[str, Any]) -> bool:
    sk = o.get("selector_key") or []
    if sk:
        return bool(sk[0])
    return bool(o.get("validator_passed"))


def all_pass(prefix: Sequence[dict[str, Any]]) -> bool:
    """Every candidate in the prefix passed the validator.

    Provably no accuracy loss when the validator is the grader, because
    the BoN selector's first sort key is `passed` — the full-set winner
    will also be a passing candidate. The selector might pick a
    DIFFERENT passing candidate from the suffix (winner-flip), but the
    rollout's overall pass/fail is preserved.
    """
    if not prefix:
        return False
    return all(_passed(o) for o in prefix)


def all_fail_same_canonical(prefix: Sequence[dict[str, Any]]) -> bool:
    """Every candidate failed AND they share the same canonical answer.

    Strict opt-in: a suffix candidate could pass the validator with a
    different answer, in which case early-exiting loses accuracy. Use
    only when prefix-replay shows the rescue rate is acceptable.
    """
    if len(prefix) < 2:
        return False
    if any(_passed(o) for o in prefix):
        return False
    cluster_ids = {o.get("consensus_cluster_id") for o in prefix}
    # None means "couldn't canonicalize" — never claim "same canonical".
    if None in cluster_ids:
        return False
    return len(cluster_ids) == 1


# ----- replay -----


@dataclass(frozen=True)
class PrefixReplayResult:
    """Outcome of one rollout's worth of prefix-replay.

    Attributes:
        fired: predicate became True at some K < N.
        first_fire_k: smallest K (≥1) at which the predicate fired,
            or None if it never did.
        winner_flip: prefix-winner ≠ full-set-winner (different obs row).
        pass_flip: prefix-winner's pass/fail differs from full-set
            winner's pass/fail. Implies an accuracy change.
        completion_tokens_saved: sum of trace_length_completion of the
            suffix that we would have skipped (lower bound on savings —
            doesn't include prompt/reasoning tokens).
        n_candidates: total candidates in this rollout (full set size).
    """
    fired: bool
    first_fire_k: int | None
    winner_flip: bool
    pass_flip: bool
    completion_tokens_saved: int
    n_candidates: int


def _execution_order(obs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort observability rows by execution order.

    selector_key is `(passed, score, conf, rf, tn, [cluster_size if C],
    [-trace_len if A], -idx)`. The LAST element is always -idx, so
    sorting by `-selector_key[-1]` recovers the launch order (idx
    ascending).
    """
    def order_key(o: dict[str, Any]) -> int:
        sk = o.get("selector_key") or []
        if not sk:
            return 0
        # selector_key[-1] = -idx; idx ascending = -(-idx) ascending
        return -int(sk[-1])
    return sorted(obs, key=order_key)


def _full_set_winner(obs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pick the full-set winner via the LIVE selector_key (already captured).

    By using the rollout's actual selector_key we make a config-faithful
    comparison: prefix-replay vs the same selector applied to the full
    set. No need to reconstruct the key.
    """
    def sort_key(o: dict[str, Any]) -> tuple:
        sk = o.get("selector_key") or []
        # Replace any non-finite floats with -inf so tuples sort.
        cleaned: list[Any] = []
        for v in sk:
            if isinstance(v, float) and not math.isfinite(v):
                cleaned.append(-math.inf if v < 0 else math.inf)
            else:
                cleaned.append(v)
        return tuple(cleaned)
    return max(obs, key=sort_key)


def _completion(o: dict[str, Any]) -> int:
    v = o.get("trace_length_completion")
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def replay_rollout(
    obs: Sequence[dict[str, Any]],
    *,
    predicate: Callable[[Sequence[dict[str, Any]]], bool],
) -> PrefixReplayResult:
    """Simulate stop-at-K early-exit for one rollout.

    Walks the candidates in execution order, and for K = 1 .. N-1 checks
    if `predicate(prefix)` fires. Reports the outcome at the FIRST fire.
    """
    rows = _execution_order(obs)
    n = len(rows)
    if n <= 1:
        return PrefixReplayResult(
            fired=False, first_fire_k=None, winner_flip=False,
            pass_flip=False, completion_tokens_saved=0, n_candidates=n,
        )

    full_winner = _full_set_winner(rows)
    full_winner_passed = _passed(full_winner)

    for k in range(1, n):
        prefix = rows[:k]
        if not predicate(prefix):
            continue
        prefix_winner = _full_set_winner(prefix)
        suffix_completion = sum(_completion(o) for o in rows[k:])
        return PrefixReplayResult(
            fired=True,
            first_fire_k=k,
            winner_flip=(prefix_winner is not full_winner),
            pass_flip=(_passed(prefix_winner) != full_winner_passed),
            completion_tokens_saved=suffix_completion,
            n_candidates=n,
        )

    return PrefixReplayResult(
        fired=False, first_fire_k=None, winner_flip=False,
        pass_flip=False, completion_tokens_saved=0, n_candidates=n,
    )


# ----- dataset-level aggregation -----


def replay_dataset(
    results_dir: str,
    *,
    source_config: str | None = None,
) -> dict[str, Any]:
    """Aggregate prefix-replay over a directory of saved bench JSONs.

    If `source_config` is None, replay across every config (each rollout
    is attributed to its own config). Otherwise filter to that config.
    """
    files = sorted(glob(os.path.join(results_dir, "*.json")))
    # nested: by_config -> by_domain -> by_predicate -> aggregate
    out: dict[str, dict[str, dict[str, dict[str, int | float]]]] = (
        defaultdict(lambda: defaultdict(lambda: {
            "all_pass": _zero_agg(),
            "all_fail_same": _zero_agg(),
        }))
    )
    n_files_scanned = 0
    n_rung3_replayed = 0

    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        n_files_scanned += 1
        cfg = d.get("config_name", "?")
        if source_config and cfg != source_config:
            continue
        if d.get("winner_rung") != 3:
            continue
        obs_all = d.get("observability") or []
        rung3 = [o for o in obs_all if o.get("rung") == 3] or obs_all
        if len(rung3) < 2:
            continue
        n_rung3_replayed += 1
        dom = d.get("domain", "?")
        for pred_name, pred in (("all_pass", all_pass),
                                ("all_fail_same", all_fail_same_canonical)):
            res = replay_rollout(rung3, predicate=pred)
            agg = out[cfg][dom][pred_name]
            agg["n"] += 1
            if res.fired:
                agg["fires"] += 1
                agg["tokens_saved_sum"] += res.completion_tokens_saved
                if res.winner_flip:
                    agg["winner_flips"] += 1
                if res.pass_flip:
                    agg["pass_flips"] += 1
                k = res.first_fire_k or 0
                agg[f"fires_at_k{k}"] = agg.get(f"fires_at_k{k}", 0) + 1

    return {
        "n_files_scanned": n_files_scanned,
        "n_rung3_replayed": n_rung3_replayed,
        "source_config": source_config,
        "by_config": {c: dict(d) for c, d in out.items()},
    }


def _zero_agg() -> dict[str, int | float]:
    return {
        "n": 0,
        "fires": 0,
        "winner_flips": 0,
        "pass_flips": 0,
        "tokens_saved_sum": 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    out: list[str] = []
    out.append("# Prefix-replay simulator (Feature E feasibility)")
    out.append("")
    out.append(
        "**Question:** if we had stopped launching rung-3 candidates after "
        "K rollouts, how often would the predicate fire, how many "
        "completion tokens would we have saved, and would the selected "
        "winner / overall pass-fail have changed?"
    )
    out.append("")
    out.append(
        f"**Files scanned:** {report['n_files_scanned']}  \n"
        f"**Rung-3 rollouts replayed (N≥2):** {report['n_rung3_replayed']}  \n"
        f"**Source config filter:** "
        f"`{report['source_config'] or 'ALL CONFIGS'}`"
    )
    out.append("")
    out.append("## Predicates")
    out.append("")
    out.append(
        "- **all_pass** — every candidate in prefix passed validator. "
        "Provably no overall pass/fail change when validator IS grader "
        "(safe default).")
    out.append(
        "- **all_fail_same** — every candidate failed AND they share "
        "`consensus_cluster_id`. Strict opt-in: suffix could rescue "
        "with a passing candidate (any pass-flip > 0 disqualifies it "
        "as a default).")
    out.append("")
    out.append("## Per-config × per-domain × per-predicate aggregate")
    out.append("")
    out.append(
        "| config | domain | pred | n | fires | fire_rate | "
        "winner_flips | pass_flips | mean_tokens_saved (when fires) |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    by_config = report["by_config"]
    for cfg in sorted(by_config):
        for dom in sorted(by_config[cfg]):
            for pred in ("all_pass", "all_fail_same"):
                a = by_config[cfg][dom][pred]
                if a["n"] == 0:
                    continue
                fr = a["fires"] / a["n"] if a["n"] else 0
                mts = (a["tokens_saved_sum"] / a["fires"]
                       if a["fires"] else 0)
                out.append(
                    f"| {cfg} | {dom} | {pred} | {a['n']} | {a['fires']} | "
                    f"{fr:.0%} | {a['winner_flips']} | {a['pass_flips']} | "
                    f"{mts:.0f} |"
                )
    out.append("")
    out.append("## Interpretation guide")
    out.append("")
    out.append(
        "- **all_pass with pass_flips=0 across all rows**: confirms the "
        "safety claim. Cost savings = `fires × mean_tokens_saved` per row.")
    out.append(
        "- **all_fail_same with pass_flips > 0 in any row**: that row has "
        "rollouts where the suffix would have rescued a failing prefix. "
        "Cannot ship as default; ship behind a stricter opt-in flag and "
        "document the empirical risk per domain.")
    out.append(
        "- **First-fire K distribution** (`fires_at_kN` keys in JSON): "
        "if most fires happen at K=1, an even more aggressive policy "
        "(skip rung 3 entirely after 1 passing rung-2 candidate) may be "
        "viable in a future phase.")
    out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--source-config", default=None,
                   help="Filter to one config; default is ALL.")
    p.add_argument("--output", required=True)
    p.add_argument("--json-output", default=None,
                   help="Optional: dump raw aggregate JSON alongside MD.")
    args = p.parse_args()
    report = replay_dataset(args.results_dir, source_config=args.source_config)
    md = render_markdown(report)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {args.output}")
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.json_output}")


if __name__ == "__main__":
    main()
