#!/bin/bash
# Full AgenticDataBench run, sharded for parallelism and resumable.
# Re-scans for newly downloaded data between waves, so it can start before
# the download finishes and pick up the rest afterwards. Tasks already
# holding an output file are skipped.
SCRATCH="$1"
OUTDIR="$SCRATCH/full_run2"
TESTBED="$SCRATCH/AgenticDataBench/testbed"
NSHARDS=6
export PYTHONIOENCODING=utf-8

mkdir -p "$OUTDIR"

for wave in 1 2 3; do
  echo "=== wave $wave: re-scanning available data ==="
  python "$SCRATCH/make_shards.py" "$TESTBED" $NSHARDS "$SCRATCH/shards"

  # Drop tasks that already produced an output file in a previous wave.
  python - "$SCRATCH" $NSHARDS <<'PY'
import json, sys
from pathlib import Path
scratch = Path(sys.argv[1]); n = int(sys.argv[2])
testbed = scratch / "AgenticDataBench" / "testbed"
outdir = scratch / "full_run2"
tasks = {json.loads(l)["id"]: json.loads(l)
         for l in open(testbed / "tasks" / "dev.jsonl", encoding="utf-8")}
todo = []
for tid in json.load(open(scratch / "shards" / "runnable.json")):
    outs = tasks[tid]["output_file_name"]
    if all((outdir / tid / o).exists() for o in outs):
        continue
    todo.append(tid)
for i in range(n):
    (scratch / "shards" / f"shard{i}.txt").write_text(
        ",".join(todo[i::n]), encoding="utf-8")
print(f"wave todo: {len(todo)} tasks")
PY

  pids=()
  for i in $(seq 0 $((NSHARDS-1))); do
    IDS=$(cat "$SCRATCH/shards/shard$i.txt")
    if [ -z "$IDS" ]; then continue; fi
    ( cd "$SCRATCH/runner" && python run_pilot.py \
        --testbed "$TESTBED" --tasks "$IDS" --outdir "$OUTDIR" \
        --lm openrouter/minimax/minimax-m3 --max-turns 25 --timeout 900 \
        --temperature 0 > "$SCRATCH/full_shard$i.wave$wave.log" 2>&1 ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "=== wave $wave complete ==="
  ls "$OUTDIR" | wc -l
done
echo "FULL RUN COMPLETE"
