"""Golden-trajectory regression tests.

Each ``.jsonl`` under ``examples/trajectories/`` is a real, frozen recording of
the RLM loop solving a representative task. We replay each one through the
*current* loop with :func:`replay_trajectory` (zero API calls) and assert the
recorded outcome reproduces exactly.

If a future change to feedback formatting, validation, repair routing, or the
stop conditions alters the loop's behavior on these recordings, one of these
tests goes red -- pointing at the exact trajectory that diverged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fabric_rlm import RLM, Trajectory, replay_trajectory

TRAJECTORY_DIR = Path(__file__).resolve().parents[1] / "examples" / "trajectories"
GOLDEN_FILES = sorted(TRAJECTORY_DIR.glob("*.jsonl"))


def _noop_lm(*args, **kwargs) -> str:  # replaced by ReplayLM inside replay
    return "noop"


@pytest.mark.skipif(not GOLDEN_FILES, reason="no recorded golden trajectories")
@pytest.mark.parametrize("path", GOLDEN_FILES, ids=lambda p: p.stem)
def test_golden_trajectory_replays_to_recorded_outcome(path: Path) -> None:
    traj = Trajectory.from_jsonl(path)
    signature = traj.metadata.get("signature")
    assert signature, f"{path.name} is missing the 'signature' metadata key"

    max_turns = int(traj.metadata.get("max_turns", len(traj.turns)))
    rlm = RLM(signature, lm=_noop_lm, enable_verifier=False, max_turns=max_turns)

    result = replay_trajectory(rlm, traj)

    # The recorded run submitted; replay must reproduce both the submit flag
    # and the exact payload, and consume every recorded turn (strict=True would
    # have raised on any over/under-consumption).
    last = traj.turns[-1]
    assert result.submitted is bool(traj.metadata.get("recorded_submitted"))
    assert result.payload == last.submit_payload
    assert len(result.trajectory.turns) == len(traj.turns)
