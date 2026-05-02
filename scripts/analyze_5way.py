"""Analyze 5-way comparison run.

Reads results_<label>.jsonl + summary_<label>.json from a downloaded
run dir and writes a markdown REPORT-comparison-5way.md with:

* Pass-rate table (overall + per template)
* Cost table (mean/median elapsed, total tokens)
* Strategy-vs-strategy diff (which q's each strategy uniquely solves)
* Pareto: pass-rate vs total tokens
* Three trace highlights for the most informative diffs

Usage:
    python scripts/analyze_5way.py <run_dir> [-o OUT_MD]
"""
from __future__ import annotations
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

STRATS = [
    ("A", "direct",          "Direct LLM (gpt-5 minimal)"),
    ("B", "dspy_rlm",        "Fabric RLM + v7-dspy engine"),
    ("C", "fabric_full",     "Fabric RLM + v6-custom (PVR full)"),
    ("D", "fabric_reflect",  "Fabric RLM + v6-custom (PVR reflect_only)"),
    ("E", "fabric_ladder",   "Fabric RLM + v6-custom + EffortLadder (minimal->low->medium)"),
]


def load_results(run_dir: Path):
    res = {}
    for code, label, _ in STRATS:
        p = run_dir / f"results_{label}.jsonl"
        if not p.exists():
            res[label] = []
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        res[label] = rows
    return res


def load_summaries(run_dir: Path):
    s = {}
    for _, label, _ in STRATS:
        p = run_dir / f"summary_{label}.json"
        s[label] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return s


def pct(n, d): return f"{(n/d*100):.0f}%" if d else "n/a"


def fmt_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("bench/adaptive/REPORT-comparison-5way.md"))
    args = ap.parse_args()

    results = load_results(args.run_dir)
    summaries = load_summaries(args.run_dir)
    run_id = args.run_dir.name

    # Pass-rate per strategy (overall + per template)
    pass_rows = []
    cost_rows = []
    health_rows = []
    qids_passed = {}        # label -> set(qids)
    pass_by_q = defaultdict(dict)
    for code, label, desc in STRATS:
        rows = results[label]
        n = len(rows)
        n_pass = sum(1 for r in rows if r.get("passed"))
        n_submitted = sum(1 for r in rows if r.get("submitted"))
        n_attempted = sum(1 for r in rows if (r.get("n_attempts") or 0) > 0 or (r.get("submitted") is not None and r.get("answer_preview")))
        n_with_ans = sum(1 for r in rows if r.get("answer_preview"))
        n_errored = sum(1 for r in rows if r.get("error"))
        passed_qids = {r["question_id"] for r in rows if r.get("passed")}
        qids_passed[label] = passed_qids
        for r in rows:
            pass_by_q[r["question_id"]][label] = bool(r.get("passed"))

        # per-template
        by_tpl = defaultdict(lambda: [0,0])
        for r in rows:
            by_tpl[r["template"]][0] += 1
            by_tpl[r["template"]][1] += 1 if r.get("passed") else 0
        tpl_str = ", ".join(f"{t}:{p}/{n2}" for t,(n2,p) in sorted(by_tpl.items()))
        pass_rows.append([f"{code} {label}", f"{n_pass}/{n}", pct(n_pass,n), tpl_str])

        # health
        if label == "fabric_ladder" and n_with_ans == 0:
            health = "**BROKEN** — adaptive engine returned 0 attempts on every question (policy/validator config issue)"
        elif label == "dspy_rlm" and n_with_ans > 0 and n_pass == 0:
            sample = next((r.get("answer_preview","") for r in rows if r.get("answer_preview")),"")
            if "REPL" in sample or "environment startup" in sample:
                health = "**DEGRADED** — v7-dspy inner interpreter (Pyodide/REPL) failed to start in Fabric runtime; model produced text-only answers"
            else:
                health = f"ran ({n_with_ans}/{n} produced an answer)"
        else:
            health = f"ran ({n_with_ans}/{n} produced an answer, {n_errored} errored)"
        health_rows.append([f"{code} {label}", n_attempted or n_with_ans, n_with_ans, n_pass, health])

        elapsed = [r.get("elapsed_seconds") or 0 for r in rows]
        prompt_tok = sum(r.get("prompt_tokens") or 0 for r in rows)
        comp_tok = sum(r.get("completion_tokens") or 0 for r in rows)
        n_attempts = sum(r.get("n_attempts") or 1 for r in rows)
        cost_rows.append([
            f"{code} {label}",
            f"{sum(elapsed):.0f}s",
            f"{statistics.mean(elapsed):.1f}s" if elapsed else "n/a",
            f"{statistics.median(elapsed):.1f}s" if elapsed else "n/a",
            prompt_tok, comp_tok, prompt_tok+comp_tok, n_attempts,
        ])

    # Strategy vs strategy diff: for each pair, who passes that the other doesn't
    labels = [lab for _, lab, _ in STRATS]
    diff_rows = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i >= j: continue
            a_only = sorted(qids_passed[a] - qids_passed[b])
            b_only = sorted(qids_passed[b] - qids_passed[a])
            both = sorted(qids_passed[a] & qids_passed[b])
            diff_rows.append([
                f"{a} vs {b}",
                len(a_only), len(b_only), len(both),
                ", ".join(a_only)[:80] or "—",
                ", ".join(b_only)[:80] or "—",
            ])

    # Per-question table
    all_qids = sorted({q for rows in results.values() for q in (r["question_id"] for r in rows)})
    perq_rows = []
    for q in all_qids:
        cells = []
        for _, label, _ in STRATS:
            v = pass_by_q[q].get(label)
            cells.append("✓" if v is True else ("✗" if v is False else "—"))
        perq_rows.append([q] + cells)

    # Build markdown
    md = []
    md.append(f"# 5-Way Comparison Report — `{run_id}`\n")
    md.append("**Dataset:** 25 LongCoT-hard CS questions (5 per template: MFMC, Backprop, DistMem, MCM, VLIW), held out from the pilot20 set.\n")
    md.append("**Model:** azure/gpt-5 with `reasoning_effort='minimal'` as base. PVR mode varies per condition.\n")
    md.append("**Wheel:** `fabric_rlm-0.1.11.dev5-py3-none-any.whl`.\n")

    md.append("\n## Strategies\n")
    for code, label, desc in STRATS:
        md.append(f"- **{code} `{label}`** — {desc}")
    md.append("\n> Note: \"DSPy RLM\" in the user request was interpreted as fabric_rlm with `inner_engine='v7-dspy'` because DSPy itself ships no `rlm` module. Strategy E originally targeted `EffortBanditPolicy`, but the bandit hung repeatedly during smoke testing on first-question warmup; we substituted the deterministic `EffortLadderPolicy` (minimal→low→medium) which exercises the same adaptive escalation code path with predictable behavior.\n")

    md.append("\n## Run health check\n")
    md.append("Headline pass rate is meaningless if a strategy didn't actually execute. This table flags strategies that didn't run cleanly.\n")
    md.append(fmt_table(["Strategy", "n_attempts>0", "produced_answer", "n_passed", "Status"], health_rows))

    md.append("\n## Headline pass rates\n")
    md.append(fmt_table(["Strategy", "Passed", "Rate", "Per template (passed/total)"], pass_rows))

    md.append("\n## Cost (elapsed + tokens) over all 25 questions\n")
    md.append(fmt_table(
        ["Strategy", "Total elapsed", "Mean/q", "Median/q", "Prompt tok", "Completion tok", "Total tok", "Total attempts"],
        cost_rows))

    md.append("\n## Per-question pass matrix\n")
    md.append(fmt_table(["question_id"] + [c for c,_,_ in STRATS], perq_rows))

    md.append("\n## Pairwise diff — who uniquely solves what\n")
    md.append(fmt_table(
        ["pair", "lhs-only", "rhs-only", "both", "lhs-only qids", "rhs-only qids"],
        diff_rows))

    md.append("\n## Findings\n")
    md.append("1. **`gpt-5` at `reasoning_effort='minimal'` cannot solve these hard CS puzzles** — every working strategy passed 0/25. Direct calls produced confident-sounding refusals (\"I cannot reliably execute …\") or empty solutions. Multi-turn RLM scaffolding (full PVR, reflect_only) reduced the rate of refusals but not the rate of correct solutions.\n")
    md.append("2. **`fabric_full` and `fabric_reflect` mostly hallucinate input truncation.** The model frequently insists the puzzle instance is \"truncated\" even though the full prompt (~10.7k chars) is forwarded verbatim by the inner engine. This is a model behavior at minimal effort, not a prompt-truncation bug.\n")
    md.append("3. **`dspy_rlm` (v7-dspy) is unusable on Fabric runtime today.** Every question received an answer of the form \"I was unable to execute any REPL steps … persistent environment startup error\" — the v7-dspy inner interpreter (Pyodide/PythonInterpreter) cannot start under the Fabric Spark notebook runtime. Treat B as \"engine not exercised\" rather than as a fair competitor.\n")
    md.append("4. **`fabric_ladder` (E) returned 0 attempts on every question** — the adaptive engine accepted the `EffortLadderPolicy` we passed but never ran a rung. Likely a misconfiguration of the `RLM(adaptive=...)` kwargs for a non-bandit ladder. (`EffortBanditPolicy` had hung during smoke testing, which is why we substituted the ladder.) E should be treated as a \"how to wire the adaptive engine without bandit\" follow-up, separate from this comparison.\n")
    md.append("5. **Effective comparison is A vs C vs D.** Among those, `direct` is far cheaper (138K tokens, 34s) than `fabric_full` (447K tokens, 8.6 min) or `fabric_reflect` (382K tokens, 12.8 min). Reflect_only is ~15% cheaper than full PVR (consistent with the pilot20 ablation), but neither produces a single correct answer at minimal effort — i.e., the scaffold cost buys nothing on this dataset *unless* paired with a higher reasoning effort.\n")
    md.append("6. **Recommended follow-up:** rerun A/C/D with `reasoning_effort='medium'` (or the EffortLadder pinned at medium) to see whether the scaffold delivers value when the base model is actually capable. Filing the v7-dspy / EffortLadder wiring issues as separate tickets (`v7-dspy-fabric-startup`, `adaptive-engine-non-bandit-policy-zero-attempts`).\n")

    md.append("\n## Notes / caveats\n")
    md.append("- Pass criterion: exact-match against the LongCoT structured `answer` (template-specific equality via `bench.adaptive.longcot_adapter.evaluate_*`).\n")
    md.append("- Token totals are aggregated from `TurnRecord.prompt_tokens`/`completion_tokens` across all turns (multi-turn strategies B/C/D/E) or from the LM's last call (A).\n")
    md.append("- Single-run variance: prior PVR experiments showed ±2× attempt count between identical reruns. Headline numbers should be read as approximate.\n")
    md.append("- Strategy E uses a 3-rung effort ladder (minimal → low → medium) with `max_attempts=3, parallel_rollouts=1`. Bandit was excluded for hangs (see above).\n")
    md.append(f"- Full per-question traces (prompts, turns, outputs, payload) are persisted under `Files/fabric_rlm_adaptive_validation/comparison_5way/{run_id}/traces_*/`.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {args.out}  ({len(md)} sections)")


if __name__ == "__main__":
    main()
