"""A ranking trap with distractor measures and a misleading largest row."""

from __future__ import annotations

import csv
from pathlib import Path
import random


QUESTION = (
    "Which site has the highest total net units? Within that site, "
    "which station contributes the most net units?"
)
ANSWER = {"group": "Harbor", "group_total": 4300, "detail": "H2", "detail_total": 2000}


def write_hard_csv(path: Path) -> None:
    rows = []
    patterns = [
        ("Harbor", "H1", 60, 25, 5),
        ("Harbor", "H2", 80, 25, 5),
        ("Harbor", "H3", 40, 20, 5),
        ("Mesa", "M9", 10, 350, 1000),
        ("Mesa", "M10", 10, 10, 1000),
        ("Ridge", "R1", 100, 30, 5),
        ("Ridge", "R2", 20, 10, 5),
    ]
    for site, station, count, net, gross_extra in patterns:
        rows.extend((site, station, net, net + gross_extra, gross_extra) for _ in range(count))
    random.Random(9182).shuffle(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("site", "station", "net_units", "gross_units", "returns_count"))
        writer.writerows(rows)
