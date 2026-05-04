"""Split reflection_v2_AB_3arm.ipynb into 4 per-arm notebooks for parallel Fabric execution.

All 4 share the same RUN_ID (passed in) so analysis joins them.
Each per-arm notebook keeps cells 0-5 (setup) + only that arm's pair, drops decision cell.
"""
import json, sys, time, copy
from pathlib import Path

SRC = Path("notebooks/reflection_v2_AB_3arm.ipynb")
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else f"reflection-ab-3arm-{time.strftime('%Y%m%d-%H%M%S')}"

# (arm_label, [arm-specific cell indices: markdown_header, code_run])
ARMS = [
    ("A", [6, 7]),
    ("B", [8, 9]),
    ("C", [10, 11]),
    ("Bhi", [12, 13]),
]

src = json.loads(SRC.read_text(encoding="utf-8"))
SETUP_INDICES = [0, 1, 2, 3, 4, 5]

for arm_short, arm_cells in ARMS:
    nb = copy.deepcopy(src)
    keep = SETUP_INDICES + arm_cells
    nb["cells"] = [src["cells"][i] for i in keep]
    # Pin shared RUN_ID by replacing the RUN_ID = f"..." line in cell 2
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
