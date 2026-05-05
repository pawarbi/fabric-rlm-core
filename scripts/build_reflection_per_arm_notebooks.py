"""Split reflection_v2_AB_3arm.ipynb into 4 per-arm notebooks for parallel Fabric execution.

All 4 share the same RUN_ID (passed in) so analysis joins them.
Each per-arm notebook keeps cells 0-5 (setup) + only that arm's pair, drops decision cell.
The first markdown header is rewritten to be arm-specific so the notebook is self-documenting in Fabric.
"""
import json, sys, time, copy
from pathlib import Path

SRC = Path("notebooks/reflection_v2_AB_3arm.ipynb")
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else f"reflection-ab-3arm-{time.strftime('%Y%m%d-%H%M%S')}"

ARM_DOCS = {
    "A": {
        "label": "A_medium_v1",
        "title": "Reflection v2 A/B \u2014 Arm A (control: medium effort, v1 reflection prompt ON)",
        "effort": "medium",
        "reflection": "ON",
        "prompt": "v1 (frozen pre-commit-0b45c09 build_reflection_prompt)",
        "purpose": "Control arm. Reproduces the dev11 baseline behaviour where v1's 'attack your own answer + re-derive' prompt drove a 72% revise rate without accuracy lift. Used as the comparison ceiling for v2 and the delta vs reflection-OFF.",
    },
    "B": {
        "label": "B_medium_v2",
        "title": "Reflection v2 A/B \u2014 Arm B (treatment: medium effort, v2 reflection prompt ON)",
        "effort": "medium",
        "reflection": "ON",
        "prompt": "v2 (current `fabric_rlm.prompts.build_reflection_prompt`: default-approve, two narrow checks, minimal-edit)",
        "purpose": "Primary treatment arm. Tests whether v2's default-approve + targeted-checks design preserves wins while suppressing the harmful over-revisions seen in dev11.",
    },
    "C": {
        "label": "C_medium_off",
        "title": "Reflection v2 A/B \u2014 Arm C (baseline: medium effort, reflection OFF)",
        "effort": "medium",
        "reflection": "OFF (`enable_reflection=False`)",
        "prompt": "n/a",
        "purpose": "Reflection-off baseline. The decision rule's FLIP_OFF branch fires if pass_C >= max(pass_A, pass_B), in which case reflection adds no value at medium effort and gets advisory-deprecated.",
    },
    "Bhi": {
        "label": "B_high_v2_sanity",
        "title": "Reflection v2 A/B \u2014 Arm B-hi (sanity: high effort, v2 reflection prompt ON)",
        "effort": "high",
        "reflection": "ON",
        "prompt": "v2",
        "purpose": "High-effort sanity arm. Confirms v2 doesn't regress the strong-model case vs the dev11 high-effort baseline (6/25). Decision rule requires pass_B-hi >= dev11_baseline - 1.",
    },
}

# (arm_short, [arm-specific cell indices: markdown_header, code_run])
ARM_CELLS = {
    "A":   [6, 7],
    "B":   [8, 9],
    "C":   [10, 11],
    "Bhi": [12, 13],
}

src = json.loads(SRC.read_text(encoding="utf-8"))
SETUP_INDICES = [0, 1, 2, 3, 4, 5]


def arm_header(short: str) -> dict:
    d = ARM_DOCS[short]
    body = (
        f"# {d['title']}\n\n"
        f"**Branch:** `experiment/reflection-v2`  \n"
        f"**Run ID (shared across all 4 arms):** `{RUN_ID}`  \n"
        f"**Arm label:** `{d['label']}`\n\n"
        f"## Configuration\n"
        f"- **Effort tier:** {d['effort']}\n"
        f"- **Reflection:** {d['reflection']}\n"
        f"- **Reflection prompt:** {d['prompt']}\n"
        f"- **Dataset:** `longcot_cs_hard_holdout25.jsonl` (n=25)\n"
        f"- **Wheel:** `fabric_rlm-0.1.11.dev13+reflectionv2-py3-none-any.whl`\n"
        f"- **Engine:** plain `RLM` (no bandit / no decompose) so reflection is the only varying knob\n\n"
        f"## Purpose\n"
        f"{d['purpose']}\n\n"
        f"## Output location\n"
        f"`abfss://sandeep_ws@onelake.dfs.fabric.microsoft.com/diagnostic.Lakehouse/Files/fabric_rlm_adaptive_validation/reflection_ab/{RUN_ID}/{d['label']}/`  \n"
        f"\u2514\u2500 `results.jsonl`, `arm_summary.json`, `traces/trace_*.json`\n\n"
        f"## Sibling arms (parallel runs, same RUN_ID)\n"
        + "\n".join(
            f"- `{ARM_DOCS[s]['label']}` (effort={ARM_DOCS[s]['effort']}, reflection={ARM_DOCS[s]['reflection']})"
            for s in ("A", "B", "C", "Bhi") if s != short
        )
        + "\n\n"
        f"After all 4 arms complete, run the decision-rule cell from `reflection_v2_AB_3arm.ipynb` (or the local equivalent) against the shared RUN_ID directory to apply the SHIP_V2 / FLIP_OFF rule from `RESEARCH-reflection-v2-ab.md`.\n"
    )
    return {"cell_type": "markdown", "metadata": {}, "source": body.splitlines(keepends=True)}


for arm_short, arm_cells in ARM_CELLS.items():
    nb = copy.deepcopy(src)
    keep = SETUP_INDICES + arm_cells
    nb["cells"] = [src["cells"][i] for i in keep]
    nb["cells"][1] = arm_header(arm_short)  # replace the generic 3-arm header with arm-specific doc
    setup_cell = nb["cells"][2]
    new_src = []
    for line in setup_cell["source"]:
        if line.startswith("RUN_ID = "):
            new_src.append(f'RUN_ID = "{RUN_ID}"\n')
        else:
            new_src.append(line)
    setup_cell["source"] = new_src
    out = SRC.parent / f"reflection_v2_AB_arm_{arm_short}.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} cells={len(nb['cells'])}")

print(f"\nRUN_ID={RUN_ID}")

