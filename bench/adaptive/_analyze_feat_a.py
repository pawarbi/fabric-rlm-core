"""Quick analysis of Feature A bench results."""
import json
import glob
import collections
import sys

results_dir = sys.argv[1] if len(sys.argv) > 1 else "bench/adaptive/results/feat_a"
files = glob.glob(f"{results_dir}/*.json")

by_q = collections.defaultdict(dict)
rung3_multi = {"adaptive_current": 0, "adaptive_a": 0}
tie_break_candidates_a = 0  # rollouts where flag could fire

for f in files:
    d = json.load(open(f))
    by_q[(d["question_id"], d["seed"])][d["config_name"]] = d
    obs = d.get("observability") or []
    cfg = d["config_name"]
    if len(obs) > 1:
        rung3_multi[cfg] += 1
        # Selector key on the winner (only one entry has it).
        keys = [tuple(o["selector_key"]) for o in obs if o.get("selector_key")]
        if cfg == "adaptive_a" and keys:
            wkey = keys[0]
            # Check whether ANY non-winner rollout could share (passed, score,
            # confidence, required_filled, total_non_blank) with the winner.
            # We don't have those for losers in the JSON; use trace_length as
            # proxy: tie-break "actually mattered" iff len(set(trace lengths)) > 1
            # AND the winner's -trace_length slot was non-default (i.e., len>2).
            tlens = [o.get("trace_length_completion") for o in obs]
            if len(wkey) == 7 and len(set(tlens)) > 1:
                tie_break_candidates_a += 1

# Divergence
divergences = []
for (qid, seed), cfgs in sorted(by_q.items()):
    a = cfgs.get("adaptive_a", {})
    c = cfgs.get("adaptive_current", {})
    if a.get("passed") != c.get("passed"):
        divergences.append({
            "qid": qid, "seed": seed, "domain": a.get("domain"),
            "current_passed": c.get("passed"), "a_passed": a.get("passed"),
            "current_tokens": c.get("total_cost_tokens"),
            "a_tokens": a.get("total_cost_tokens"),
            "current_winner_rung": c.get("winner_rung"),
            "a_winner_rung": a.get("winner_rung"),
        })

# Same-answer different selection
same_pass_diff_winner = 0
for (qid, seed), cfgs in by_q.items():
    a = cfgs.get("adaptive_a", {}); c = cfgs.get("adaptive_current", {})
    if a.get("passed") and c.get("passed") and a.get("answer") != c.get("answer"):
        same_pass_diff_winner += 1

print(f"Multi-rollout (rung-3) selection events:")
print(f"  adaptive_current: {rung3_multi['adaptive_current']}")
print(f"  adaptive_a:       {rung3_multi['adaptive_a']}")
print(f"Tie-break candidates in adaptive_a (rung-3 with >1 distinct trace lengths): {tie_break_candidates_a}")
print()
print(f"Per-(question,seed) divergences (passed differs): {len(divergences)}")
for d in divergences:
    print(f"  {d['qid']} seed={d['seed']} domain={d['domain']}: "
          f"current(passed={d['current_passed']},rung={d['current_winner_rung']},tok={d['current_tokens']}) "
          f"vs a(passed={d['a_passed']},rung={d['a_winner_rung']},tok={d['a_tokens']})")
print()
print(f"Both passed but different answer chosen: {same_pass_diff_winner}")

# Per-domain delta
by_domain = collections.defaultdict(lambda: {"current": [0,0], "a": [0,0]})
for f in files:
    d = json.load(open(f))
    cfg = "current" if d["config_name"] == "adaptive_current" else "a"
    by_domain[d["domain"]][cfg][1] += 1
    if d["passed"]:
        by_domain[d["domain"]][cfg][0] += 1
print("Per-domain (passed/total):")
for dom, v in sorted(by_domain.items()):
    cp, ct = v["current"]; ap, at = v["a"]
    delta = (ap/at if at else 0) - (cp/ct if ct else 0)
    print(f"  {dom:20s}: current {cp}/{ct}  a {ap}/{at}  delta={delta:+.3f}")
