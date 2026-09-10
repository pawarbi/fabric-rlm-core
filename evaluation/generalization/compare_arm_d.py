"""Pre-registered arm D vs arm B comparison.

Registered BEFORE arm D finished, so the endpoints cannot be chosen to suit
the numbers:

  PRIMARY   paired sign test on turns and total tokens, D vs B, same 24 qs.
            B is the right control: both arms run learn() and answer from a
            frozen package. A (cold) differs in two ways at once.
  SECONDARY accuracy -- REPORTED, NOT HEADLINED. Cold was 22/24; there are two
            points of headroom, n=1 per question at temperature 1.0. An earlier
            causal claim in this evaluation was withdrawn to exactly this
            variance.
  MECHANISM does declared metadata DISPLACE rediscovery, or merely add context?
            Baseline: the learn arm called list_sources() in 21 of 23
            trajectories. A drop is the only evidence of displacement.

    python compare_arm_d.py run_log_glm_declared.json
"""
import json
import sys
from pathlib import Path


def load(p):
    rows = json.loads(Path(p).read_text(encoding="utf-8"))
    return {r["id"]: r for r in rows}


def toks(r):
    return (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)


def sign_test(pairs, label):
    """Two-sided exact binomial sign test, ties dropped."""
    from math import comb
    pos = sum(1 for a, b in pairs if b > a)
    neg = sum(1 for a, b in pairs if b < a)
    n = pos + neg
    if n == 0:
        print(f"  {label}: all ties")
        return
    k = min(pos, neg)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    print(f"  {label}: {pos} up / {neg} down / {len(pairs)-n} tie   p={p:.4f}")


def uses_list_sources(r):
    return "list_sources" in json.dumps(r.get("trajectory") or "")


B = load("run_log_glm_learn.json")
D = load(sys.argv[1] if len(sys.argv) > 1 else "run_log_glm_declared.json")
A = load("run_log_glm.json")
ids = sorted(set(B) & set(D))
print(f"paired on {len(ids)} questions\n")

print("PRIMARY -- paired sign test, D vs B (B first, D second)")
sign_test([(B[i].get("turns") or 0, D[i].get("turns") or 0) for i in ids], "turns ")
sign_test([(toks(B[i]), toks(D[i])) for i in ids], "tokens")

bt = sum(B[i].get("turns") or 0 for i in ids)
dt = sum(D[i].get("turns") or 0 for i in ids)
bk = sum(toks(B[i]) for i in ids)
dk = sum(toks(D[i]) for i in ids)
print(f"\n  totals  turns  B={bt} D={dt} ({dt-bt:+d}, {100*(dt-bt)/max(bt,1):+.1f}%)")
print(f"          tokens B={bk:,} D={dk:,} ({dk-bk:+,}, {100*(dk-bk)/max(bk,1):+.1f}%)")

print("\nSECONDARY -- completion (reported, not headlined)")
for nm, arm in (("A cold ", A), ("B learn", B), ("D decl ", D)):
    got = [i for i in ids if i in arm]
    print(f"  {nm}: ok {sum(1 for i in got if arm[i].get('ok'))}/{len(got)}")

print("\nMECHANISM -- did declared facts displace rediscovery?")
for nm, arm in (("B learn", B), ("D decl ", D)):
    n = sum(1 for i in ids if uses_list_sources(arm[i]))
    print(f"  {nm}: list_sources in {n}/{len(ids)} trajectories")

print("\nPER-QUESTION (turns / tokens / ok)")
print(f"  {'id':6} {'B turns':>8}{'D turns':>9}{'B tok':>10}{'D tok':>10}  B/D ok")
for i in ids:
    print(f"  {i:6} {B[i].get('turns') or 0:8}{D[i].get('turns') or 0:9}"
          f"{toks(B[i]):10,}{toks(D[i]):10,}  {str(B[i].get('ok'))[0]}/{str(D[i].get('ok'))[0]}")
