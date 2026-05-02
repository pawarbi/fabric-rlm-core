"""Build the 25-question holdout for the 5-way comparison.

Loads LongCoT from HF, filters cs/hard, excludes pilot20 question_ids,
stratifies 5 per template across MFMC/Backprop/DistMem/MCM/VLIW.

Output: bench/adaptive/longcot_cs_hard_holdout25.jsonl
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

PILOT_PATH= ROOT / "bench" / "adaptive" / "longcot_cs_hard_pilot20.jsonl"
OUT_PATH = ROOT / "bench" / "adaptive" / "longcot_cs_hard_holdout25.jsonl"
TEMPLATES = ("MFMC", "Backprop", "DistMem", "MCM", "VLIW")
PER_TEMPLATE = 5


def main() -> None:
    pilot_ids = {
        json.loads(line)["question_id"]
        for line in PILOT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    print(f"pilot20 ids: {len(pilot_ids)}")

    examples = load_longcot_dataset(split="hard")
    print(f"loaded {len(examples)} total examples")

    cs_hard = filter_examples(examples, domains="cs", difficulties="hard")
    print(f"cs/hard: {len(cs_hard)}")

    selected: list = []
    counts: dict[str, int] = {}
    for tpl in TEMPLATES:
        bucket = [
            ex
            for ex in cs_hard
            if ex.template == tpl and ex.question_id not in pilot_ids
        ]
        bucket.sort(key=lambda ex: ex.question_id)
        if len(bucket) < PER_TEMPLATE:
            raise SystemExit(
                f"template {tpl}: only {len(bucket)} candidates after exclusion"
            )
        chosen = bucket[:PER_TEMPLATE]
        selected.extend(chosen)
        counts[tpl] = len(chosen)

    print(f"counts: {counts}")
    print(f"total selected: {len(selected)}")

    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for ex in selected:
            row = example_to_row(ex, include_answer=True)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
