"""Build a 25-question REGRESSION set of LongCoT cs/hard questions.

Excludes pilot20 (used by adaptive bench) and holdout25 (used by 5-way comp).
Picks the next 5 per template by question_id.

Output: bench/adaptive/longcot_cs_hard_regression25.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.adaptive.longcot_adapter import (  # noqa: E402
    example_to_row,
    filter_examples,
    load_longcot_dataset,
)

EXCLUDE_PATHS = [
    ROOT / "bench" / "adaptive" / "longcot_cs_hard_pilot20.jsonl",
    ROOT / "bench" / "adaptive" / "longcot_cs_hard_holdout25.jsonl",
]
OUT_PATH = ROOT / "bench" / "adaptive" / "longcot_cs_hard_regression25.jsonl"
TEMPLATES = ("MFMC", "Backprop", "DistMem", "MCM", "VLIW")
PER_TEMPLATE = 5


def main() -> None:
    used: set[str] = set()
    for p in EXCLUDE_PATHS:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                used.add(json.loads(line)["question_id"])
    print(f"excluded {len(used)} prior question_ids")

    cs_hard = filter_examples(
        load_longcot_dataset(split="hard"), domains="cs", difficulties="hard"
    )
    print(f"cs/hard pool: {len(cs_hard)}")

    selected: list = []
    counts: dict[str, int] = {}
    for tpl in TEMPLATES:
        bucket = sorted(
            (ex for ex in cs_hard if ex.template == tpl and ex.question_id not in used),
            key=lambda ex: ex.question_id,
        )
        if len(bucket) < PER_TEMPLATE:
            raise SystemExit(f"{tpl}: only {len(bucket)} candidates after exclusion")
        chosen = bucket[:PER_TEMPLATE]
        selected.extend(chosen)
        counts[tpl] = len(chosen)

    print(f"counts: {counts}; total: {len(selected)}")

    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for ex in selected:
            fh.write(json.dumps(example_to_row(ex, include_answer=True), ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
