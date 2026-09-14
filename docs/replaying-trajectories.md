# Replaying trajectories

The RLM loop is where the interesting logic lives: feedback formatting,
output validation, repair routing, the stuck-loop and max-turns stop
conditions. That logic changes often, and exercising it normally costs a
real model call, which is slow, flaky and billed. `replay_trajectory`
removes the model from the equation.

A recorded `Trajectory` stores both sides of every turn: the raw model
response and the worker outcome (stdout, error, submitted, state, submit
payload). That is everything needed to drive the loop again.
`replay_trajectory` feeds the recording back through the real `RLM.run`
with two in-memory fakes, `ReplayLM` and `ReplayInterpreter`:

- zero API calls, zero subprocesses, fully deterministic;
- end-to-end coverage of the orchestration layer;
- a reproducible bug report is one `.jsonl` file a user can send you.

Divergence is the signal. If a change makes the loop ask for more turns
than were recorded, `ReplayLM` raises `DivergenceError`; if the loop stops
earlier, `replay_trajectory` raises after the run. "The loop behaves
differently now" becomes a red test instead of a surprise in production.
It works whatever model produced the recording, so frozen files keep
protecting every later refactor as newer models are adopted.

## Recording

Run a task with a real model and keep the trajectory. The signature is
needed to replay, so store it in the metadata:

```python
from fabric_rlm import RLM

rlm = RLM("question -> answer", lm="openai/gpt-4.1-mini")
result = rlm.run({"question": "What is 17 * 23 + 5?"})
trajectory = result.trajectory
trajectory.metadata["signature"] = "question -> answer"
trajectory.write_jsonl("trajectories/arithmetic.jsonl")
```

## Replaying

```python
from fabric_rlm import RLM, Trajectory, replay_trajectory

def offline_lm(*args, **kwargs):
    return "offline"  # never called: replay swaps in a ReplayLM

trajectory = Trajectory.from_jsonl("trajectories/arithmetic.jsonl")
rlm = RLM(trajectory.metadata["signature"], lm=offline_lm, enable_verifier=False, max_turns=8)
result = replay_trajectory(rlm, trajectory)
print(result.submitted, len(result.trajectory.turns), result.payload)
```

## Persisting to a Fabric Lakehouse

A mounted Files path is a normal file:

```python
trajectory.write_jsonl("/lakehouse/default/Files/rlm_traces/arithmetic.jsonl")
```

Reload it from anywhere through the abfss URI (read through
`notebookutils.fs`, no extra dependency), or from a Spark DataFrame of the
JSONL:

```python
loaded = Trajectory.from_jsonl(
    "abfss://<workspace>@onelake.dfs.fabric.microsoft.com/"
    "<lakehouse>.Lakehouse/Files/rlm_traces/arithmetic.jsonl"
)
df = spark.read.json("Files/rlm_traces/arithmetic.jsonl")
loaded = Trajectory.from_dicts(df.collect())
```

The repository keeps its own frozen recordings under
`tests/fixtures/trajectories/`, replayed by
`tests/test_golden_trajectories.py` on every run of the suite.
