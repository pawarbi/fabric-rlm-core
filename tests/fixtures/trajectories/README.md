# Golden trajectories

Each `.jsonl` here is a real, frozen recording of the RLM loop solving a
representative task. `tests/test_golden_trajectories.py` replays every one
through the current loop with `replay_trajectory`, which swaps the model
and the worker for in-memory fakes fed from the recording: no API call, no
subprocess, fully deterministic. If a change to feedback formatting, output
validation, repair routing or the stop conditions makes the loop behave
differently on a recording, that test goes red and names the trajectory.

The recordings work whatever model produced them, so they keep protecting
the loop as newer models are adopted.

To add one, run a task with a real model, set
`trajectory.metadata["signature"]` to the signature the run used, write it
with `trajectory.write_jsonl(...)` into this directory, and the test picks
it up. `docs/replaying-trajectories.md` covers recording, persisting to a
Lakehouse and replaying in your own code.
