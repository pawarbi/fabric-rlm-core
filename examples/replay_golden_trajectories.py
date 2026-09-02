"""Golden-trajectory testing with ReplayLM -- how and why.

WHY
---
The RLM loop is where the interesting logic lives: feedback formatting, output
validation, repair routing, stuck-loop / max-turns stop conditions. That logic
changes often, and exercising it normally costs a real LM call (slow, flaky,
and billed). ReplayLM removes the model from the equation.

A recorded ``Trajectory`` already stores *both* sides of every turn -- the raw
LM response (``TurnRecord.response_text``) and the worker outcome
(``stdout``/``error``/``submitted``/``state``/``submit_payload``). That is
everything needed to drive the loop again. ``replay_trajectory`` feeds those
recordings back through the **real** ``RLM.run`` with two in-memory fakes
(``ReplayLM`` + ``ReplayInterpreter``):

  * zero API calls, zero subprocesses, fully deterministic;
  * end-to-end coverage of the orchestration layer;
  * a reproducible bug report is just a ``.jsonl`` file a user can ship you.

Divergence is the signal: if a change makes the loop ask for more turns than
were recorded, ReplayLM raises ``DivergenceError``; if it stops earlier,
``replay_trajectory`` raises after the run. "The loop behaves differently now"
becomes a red test instead of a silent production surprise.

Note: it works regardless of which model produced the recording -- the golden
files are frozen, so they protect every future refactor even as you adopt newer
models.

HOW
---
Run this file from the repo root::

    python examples/replay_golden_trajectories.py

It replays the frozen recordings in ``examples/trajectories/`` offline.
"""

from __future__ import annotations

from pathlib import Path

from fabric_rlm import RLM, Trajectory, replay_trajectory

TRAJECTORY_DIR = Path(__file__).resolve().parent / "trajectories"


def _offline_lm(*args, **kwargs) -> str:
    # Never actually called: replay_trajectory swaps in a ReplayLM. Present only
    # because RLM requires an lm at construction time.
    return "offline"


def replay_all() -> None:
    for path in sorted(TRAJECTORY_DIR.glob("*.jsonl")):
        traj = Trajectory.from_jsonl(path)
        signature = traj.metadata["signature"]
        rlm = RLM(signature, lm=_offline_lm, enable_verifier=False, max_turns=8)

        result = replay_trajectory(rlm, traj)  # no API calls, no subprocess

        print(
            f"{path.stem:22s} submitted={result.submitted!s:5s} "
            f"turns={len(result.trajectory.turns)} payload={result.payload}"
        )


# ---------------------------------------------------------------------------
# Recording and persisting trajectories (including to a Fabric Lakehouse)
# ---------------------------------------------------------------------------


def record_and_save_example() -> None:
    """Illustrative only -- needs a real LM. Shows where recordings come from
    and how to persist them locally or to a Lakehouse for later replay."""
    # 1. Run a task with a real LM and capture the trajectory.
    #
    #    rlm = RLM("question -> answer", lm="openai/gpt-4.1-mini")
    #    result = rlm.run({"question": "What is 17 * 23 + 5?"})
    #    traj = result.trajectory
    #    traj.metadata["signature"] = "question -> answer"  # needed to replay
    #
    # 2a. Persist locally and replay later with zero API calls:
    #
    #    traj.write_jsonl("trajectories/arithmetic.jsonl")
    #    loaded = Trajectory.from_jsonl("trajectories/arithmetic.jsonl")
    #    replay_trajectory(rlm, loaded)
    #
    # 2b. Persist to a Fabric Lakehouse. A mounted Files path is a normal file:
    #
    #    traj.write_jsonl("/lakehouse/default/Files/rlm_traces/arithmetic.jsonl")
    #
    #     ...and reload it from anywhere via the abfss:// URI (read through
    #     notebookutils.fs automatically, no extra deps):
    #
    #    loaded = Trajectory.from_jsonl(
    #        "abfss://<workspace>@onelake.dfs.fabric.microsoft.com/"
    #        "<lakehouse>.Lakehouse/Files/rlm_traces/arithmetic.jsonl"
    #    )
    #    replay_trajectory(rlm, loaded)
    #
    #     You can also load straight from a Spark DataFrame of the JSONL:
    #
    #    df = spark.read.json("Files/rlm_traces/arithmetic.jsonl")
    #    loaded = Trajectory.from_dicts(df.collect())
    raise SystemExit(
        "record_and_save_example() is documentation-only; see the comments."
    )


if __name__ == "__main__":
    replay_all()
