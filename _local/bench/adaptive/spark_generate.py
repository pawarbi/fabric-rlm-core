"""Generate a large synthetic Spark log with embedded patterns to find.

Embeds:
- 1 failed job at the end (job_id 17, stage_id 42, root cause = OOM in executor 7, task 88001)
- ~50 OOM occurrences spread out (executor 7 dominant)
- ~10 slow tasks (>120s) — top-3 are 88002 (312s), 88001 (285s), 88003 (247s)
- background INFO/WARN noise filling out the bulk
"""
import random, sys, os, time

random.seed(42)

OUT_PATH = os.path.join(os.path.dirname(__file__), "spark_app.log")
TARGET_LINES = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000

EXECUTORS = list(range(1, 17))
HOSTS = [f"worker-{i:02d}.fabric.local" for i in range(1, 17)]

def ts(i):
    return time.strftime("%y/%m/%d %H:%M:%S", time.gmtime(1700000000 + i // 10))

def info_line(i):
    e = random.choice(EXECUTORS)
    h = random.choice(HOSTS)
    task = random.randint(1, 100000)
    stage = random.randint(1, 60)
    dur = random.randint(50, 8000)
    bytesn = random.randint(1024, 1024 * 1024 * 50)
    msgs = [
        f"INFO TaskSetManager: Finished task {task}.0 in stage {stage}.0 (TID {task}) in {dur} ms on {h} (executor {e})",
        f"INFO BlockManagerInfo: Added rdd_{stage}_{task} in memory on {h}:{34000 + e} (size: {bytesn // 1024} KB, free: 12.3 GB)",
        f"INFO Executor: Running task {task}.0 in stage {stage}.0 (TID {task})",
        f"INFO ShuffleBlockFetcherIterator: Getting {random.randint(10, 500)} non-empty blocks",
        f"INFO MemoryStore: Block broadcast_{random.randint(1,200)} stored as values in memory (estimated size {random.randint(1, 50)} MB, free {random.randint(1, 12)} GB)",
    ]
    return f"{ts(i)} {random.choice(msgs)}"

def warn_line(i):
    e = random.choice(EXECUTORS)
    h = random.choice(HOSTS)
    msgs = [
        f"WARN HeartbeatReceiver: Removing executor {e} with no recent heartbeats: {random.randint(120, 300)} ms exceeds timeout 120000 ms",
        f"WARN TaskSetManager: Lost task {random.randint(1, 100000)}.0 in stage {random.randint(1, 60)}.0 (TID {random.randint(1, 100000)}) on {h}, executor {e}: java.io.IOException (Connection reset by peer)",
        f"WARN BlockManager: Block rdd_{random.randint(1,60)}_{random.randint(1,100000)} could not be removed",
    ]
    return f"{ts(i)} {random.choice(msgs)}"

def write_log():
    print(f"Writing {TARGET_LINES} lines to {OUT_PATH}...", flush=True)
    slow_tasks = [
        ("88001", 42, 285_000),
        ("88002", 43, 312_000),
        ("88003", 42, 247_000),
        ("70010", 30, 138_000),
        ("70011", 31, 142_000),
        ("70012", 32, 155_000),
        ("70013", 33, 161_000),
        ("70014", 34, 124_000),
        ("70015", 35, 131_000),
        ("70016", 36, 129_000),
    ]
    oom_executors = [7] * 35 + random.choices([3, 5, 11, 13, 15], k=15)
    oom_lines = []
    for ex in oom_executors:
        tid = random.randint(10000, 99999)
        st = random.choice([42, 43, 50, 51])
        oom_lines.append(
            f"ERROR Executor: Exception in task {tid}.0 in stage {st}.0 (TID {tid})\n"
            f"java.lang.OutOfMemoryError: Java heap space\n"
            f"\tat org.apache.spark.sql.execution.aggregate.HashAggregateExec.doExecute(HashAggregateExec.scala:104)\n"
            f"WARN TaskSetManager: Task {tid}.0 in stage {st}.0 (TID {tid}) failed on worker-{ex:02d}.fabric.local, executor {ex}: java.lang.OutOfMemoryError"
        )

    final_failure = (
        "ERROR YarnScheduler: Lost executor 7 on worker-07.fabric.local: Container marked as failed: "
        "container_1700000000000_0001_01_000008. Exit status: 143. "
        "Diagnostics: Container killed on request. Exit code is 143\n"
        "ERROR TaskSetManager: Task 88001 in stage 42.0 failed 4 times; aborting job\n"
        "ERROR DAGScheduler: Job 17 failed: collect at SparkApp.scala:142, took 1247.583921 s\n"
        "INFO DAGScheduler: ResultStage 43 (collect at SparkApp.scala:142) failed in 1247.583 s due to "
        "Job aborted due to stage failure: Task 88001 in stage 42.0 failed 4 times, "
        "most recent failure: Lost task 88001.3 in stage 42.0 (TID 88001) (worker-07.fabric.local executor 7): "
        "java.lang.OutOfMemoryError: Java heap space"
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        slow_indices = sorted(random.sample(range(TARGET_LINES // 2), len(slow_tasks)))
        oom_indices = sorted(random.sample(range(TARGET_LINES // 4, TARGET_LINES - 100), len(oom_lines)))
        slow_iter = iter(zip(slow_indices, slow_tasks))
        oom_iter = iter(zip(oom_indices, oom_lines))
        next_slow = next(slow_iter, None)
        next_oom = next(oom_iter, None)

        for i in range(TARGET_LINES - 5):
            if next_slow and i == next_slow[0]:
                tid, stg, dur = next_slow[1]
                h = "worker-07.fabric.local" if stg in (42, 43) else random.choice(HOSTS)
                ex = 7 if stg in (42, 43) else random.choice(EXECUTORS)
                f.write(
                    f"{ts(i)} INFO TaskSetManager: Finished task {tid}.0 in stage {stg}.0 (TID {tid}) "
                    f"in {dur} ms on {h} (executor {ex})\n"
                )
                next_slow = next(slow_iter, None)
            elif next_oom and i == next_oom[0]:
                f.write(next_oom[1] + "\n")
                next_oom = next(oom_iter, None)
            elif random.random() < 0.08:
                f.write(warn_line(i) + "\n")
            else:
                f.write(info_line(i) + "\n")
            if i % 50000 == 0 and i > 0:
                print(f"  {i}/{TARGET_LINES}", flush=True)
        f.write(final_failure + "\n")

    sz = os.path.getsize(OUT_PATH)
    print(f"Done. {OUT_PATH} = {sz/1024/1024:.1f} MB")
    print(f"Ground truth:")
    print(f"  failed_job_id: 17")
    print(f"  failed_stage_id: 42")
    print(f"  root_cause: OutOfMemoryError on executor 7")
    print(f"  top_3_slow_tasks: 88002 (312s), 88001 (285s), 88003 (247s)")
    print(f"  oom_count: 50 (executor 7 dominant: 35)")

if __name__ == "__main__":
    write_log()
