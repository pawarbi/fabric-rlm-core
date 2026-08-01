"""Select the runnable task set for the full run and split it into shards."""
import json, sys
from pathlib import Path

TESTBED = Path(sys.argv[1])
NSHARDS = int(sys.argv[2])
OUT = Path(sys.argv[3])

tasks = [json.loads(l) for l in open(TESTBED / "tasks" / "dev.jsonl",
                                     encoding="utf-8")]

runnable, skipped = [], []
for t in tasks:
    dom = t["domain"]
    if t.get("data_sources"):
        missing = []
        root = TESTBED / "datasets" / dom
        for src in t["data_sources"]:
            rel = src
            for p in (f"datasets/{dom}/", f"{dom}/"):
                if rel.startswith(p):
                    rel = rel[len(p):]
            if (root / rel).exists():
                continue
            # Bare filenames sometimes name a file nested in a subdirectory.
            if root.exists() and any(c.is_file()
                                     for c in root.rglob(Path(rel).name)):
                continue
            missing.append(src)
        if missing:
            skipped.append((t["id"], f"missing input: {missing[0][:50]}"))
            continue
    else:
        if not (TESTBED / "datasets" / dom).exists():
            skipped.append((t["id"], "domain dir absent"))
            continue
    runnable.append(t["id"])

OUT.mkdir(parents=True, exist_ok=True)
shards = [runnable[i::NSHARDS] for i in range(NSHARDS)]
for i, s in enumerate(shards):
    (OUT / f"shard{i}.txt").write_text(",".join(s), encoding="utf-8")
(OUT / "runnable.json").write_text(json.dumps(runnable), encoding="utf-8")
(OUT / "skipped.json").write_text(json.dumps(skipped, indent=1), encoding="utf-8")

print(f"runnable: {len(runnable)} / {len(tasks)}")
print(f"skipped : {len(skipped)}")
from collections import Counter
print("skip reasons:", dict(Counter(r for _, r in skipped).most_common(6)))
print("shard sizes:", [len(s) for s in shards])
