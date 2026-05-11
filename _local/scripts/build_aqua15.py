"""Build a 15-question quantitative/data-analysis bench from AQuA-RAT.

AQuA-RAT is multi-choice quantitative reasoning (percentages, rates,
algebra, ratios) — a clean stand-in for "data analysis" questions when
real PDF parsing isn't worth setting up in Fabric.

Output schema mirrors the LongCoT rows so the comparison harness can
treat the two datasets uniformly:
  question_id, template, prompt, answer, metadata

Output: bench/adaptive/aqua_rat_15.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "bench" / "adaptive" / "aqua_rat_15.jsonl"
N = 15

INSTRUCTIONS = (
    "You are given a multiple-choice quantitative reasoning problem.\n"
    "Pick the single best answer from the listed options.\n"
    "Show your reasoning briefly, then on the final line output exactly:\n"
    "    Answer: X\n"
    "where X is one of A, B, C, D, or E.\n\n"
)


def main() -> None:
    from datasets import load_dataset  # local import — only needed at build time

    ds = load_dataset("aqua_rat", split="test", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if len(rows) >= N:
            break
        # Skip degenerate rows
        if not ex.get("question") or not ex.get("options") or len(ex.get("options") or []) < 4:
            continue
        if not ex.get("correct") or ex["correct"] not in {"A", "B", "C", "D", "E"}:
            continue
        opts = "\n".join(ex["options"])
        prompt = INSTRUCTIONS + "Question:\n" + ex["question"].strip() + "\n\nOptions:\n" + opts
        rows.append({
            "question_id": f"AQUA_{i:04d}",
            "domain": "quant",
            "difficulty": "medium",
            "template": "AQUA",
            "prompt": prompt,
            "answer": ex["correct"],
            "metadata": {
                "options": ex["options"],
                "rationale_preview": (ex.get("rationale") or "")[:400],
            },
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
