"""Unit tests for the ReplayLM golden-trajectory harness.

These tests are fully hermetic: no subprocess, no network, no real LM. They
drive the *real* ``RLM.run`` loop with a recorded trajectory so that feedback
formatting, validation, repair flow, and stop conditions are exercised end to
end with zero API calls.
"""

from __future__ import annotations

import pytest

from fabric_rlm import (
    RLM,
    Trajectory,
    TurnRecord,
)
from fabric_rlm.replay_lm import (
    DivergenceError,
    ReplayInterpreter,
    ReplayLM,
    replay_trajectory,
)


def _submit_response(payload_repr: str) -> str:
    return f"```python\nSUBMIT({payload_repr})\n```"


def _turn(
    *,
    turn: int,
    code: str = "x = 1",
    stdout: str = "",
    stderr: str = "",
    error: str | None = None,
    submitted: bool = False,
    state: dict | None = None,
    response_text: str = "",
    submit_payload: dict | None = None,
) -> TurnRecord:
    return TurnRecord(
        turn=turn,
        code=code,
        stdout=stdout,
        stderr=stderr,
        error=error,
        submitted=submitted,
        state=state or {},
        response_text=response_text,
        submit_payload=submit_payload,
    )


def _dummy_rlm(**kwargs):
    # ``lm`` is required by RLM but is swapped out by replay_trajectory, so a
    # trivial callable is fine. signature uses a single ``answer`` output.
    return RLM(
        "question -> answer",
        lm=lambda *a, **k: "noop",
        enable_verifier=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# ReplayLM
# ---------------------------------------------------------------------------


def test_replay_lm_returns_recorded_responses_in_order() -> None:
    lm = ReplayLM(["first", "second"])
    assert lm(messages=[{"role": "user", "content": "x"}]) == "first"
    assert lm(messages=[{"role": "user", "content": "y"}]) == "second"


def test_replay_lm_records_calls_for_assertions() -> None:
    lm = ReplayLM(["only"])
    lm(messages=[{"role": "user", "content": "hi"}])
    assert len(lm.calls) == 1
    assert lm.calls[0][-1]["content"] == "hi"


def test_replay_lm_raises_divergence_when_exhausted() -> None:
    lm = ReplayLM(["one"])
    lm(messages=[{"role": "user", "content": "x"}])
    with pytest.raises(DivergenceError):
        lm(messages=[{"role": "user", "content": "x"}])


def test_replay_lm_from_trajectory_uses_response_text() -> None:
    traj = Trajectory(
        turns=[
            _turn(turn=1, response_text="r1"),
            _turn(turn=2, response_text="r2"),
        ]
    )
    lm = ReplayLM.from_trajectory(traj)
    assert lm(messages=[]) == "r1"
    assert lm(messages=[]) == "r2"


# ---------------------------------------------------------------------------
# ReplayInterpreter
# ---------------------------------------------------------------------------


def test_replay_interpreter_reconstructs_exec_result() -> None:
    traj = Trajectory(
        turns=[
            _turn(
                turn=1,
                stdout="hello",
                state={"x": 1},
                submitted=True,
                submit_payload={"answer": 42},
            )
        ]
    )
    interp = ReplayInterpreter.from_trajectory(traj)
    with interp:
        result = interp.execute("anything")
    assert result.ok is True
    assert result.submitted is True
    assert result.stdout == "hello"
    assert result.state == {"x": 1}
    assert result.submit_payload == {"answer": 42}


def test_replay_interpreter_error_turn_is_not_ok() -> None:
    traj = Trajectory(turns=[_turn(turn=1, error="NameError: boom")])
    interp = ReplayInterpreter.from_trajectory(traj)
    with interp:
        result = interp.execute("bad")
    assert result.ok is False
    assert result.error == "NameError: boom"


def test_replay_interpreter_configure_and_set_inputs_are_noops() -> None:
    interp = ReplayInterpreter.from_trajectory(Trajectory(turns=[_turn(turn=1)]))
    with interp:
        assert interp.configure_lm({"model": "x"}) == {}
        assert interp.set_inputs({"a": 1}) is None


def test_replay_interpreter_detects_code_mismatch() -> None:
    traj = Trajectory(turns=[_turn(turn=1, code="x = 1")])
    interp = ReplayInterpreter.from_trajectory(traj)
    with interp:
        interp.execute("y = 2")  # different code than recorded
    assert interp.divergence_error is not None


def test_replay_interpreter_strict_false_ignores_code_mismatch() -> None:
    traj = Trajectory(turns=[_turn(turn=1, code="x = 1")])
    interp = ReplayInterpreter.from_trajectory(traj, strict=False)
    with interp:
        interp.execute("y = 2")
    assert interp.divergence_error is None


def test_replay_interpreter_over_consumption_raises() -> None:
    traj = Trajectory(turns=[_turn(turn=1, code="x = 1")])
    interp = ReplayInterpreter.from_trajectory(traj)
    with interp:
        interp.execute("x = 1")
        with pytest.raises(DivergenceError):
            interp.execute("z = 3")


# ---------------------------------------------------------------------------
# replay_trajectory: end-to-end through the real loop
# ---------------------------------------------------------------------------


def test_replay_trajectory_reproduces_single_turn_submit() -> None:
    payload = {"answer": "Paris"}
    traj = Trajectory(
        turns=[
            _turn(
                turn=1,
                code="SUBMIT({'answer': 'Paris'})",
                response_text=_submit_response("{'answer': 'Paris'}"),
                submitted=True,
                submit_payload=payload,
            )
        ]
    )
    rlm = _dummy_rlm()
    result = replay_trajectory(rlm, traj)
    assert result.submitted is True
    assert result.payload == payload
    assert len(result.trajectory.turns) == 1


def test_replay_trajectory_multi_turn_then_submit() -> None:
    traj = Trajectory(
        turns=[
            _turn(
                turn=1,
                code="print('thinking')",
                stdout="thinking\n",
                response_text="```python\nprint('thinking')\n```",
            ),
            _turn(
                turn=2,
                code="SUBMIT({'answer': '7'})",
                response_text=_submit_response("{'answer': '7'}"),
                submitted=True,
                submit_payload={"answer": "7"},
            ),
        ]
    )
    rlm = _dummy_rlm()
    result = replay_trajectory(rlm, traj)
    assert result.submitted is True
    assert result.payload == {"answer": "7"}
    assert len(result.trajectory.turns) == 2


def test_replay_trajectory_validation_repair_flow() -> None:
    # First submit fails the output validator; second submit passes. The real
    # loop must route the validation-repair feedback and accept the retry.
    def validator(payload):
        if payload.get("answer") != "correct":
            raise AssertionError("answer must be 'correct'")

    traj = Trajectory(
        turns=[
            _turn(
                turn=1,
                code="SUBMIT({'answer': 'wrong'})",
                response_text=_submit_response("{'answer': 'wrong'}"),
                submitted=True,
                submit_payload={"answer": "wrong"},
            ),
            _turn(
                turn=2,
                code="SUBMIT({'answer': 'correct'})",
                response_text=_submit_response("{'answer': 'correct'}"),
                submitted=True,
                submit_payload={"answer": "correct"},
            ),
        ]
    )
    rlm = _dummy_rlm(output_validator=validator)
    result = replay_trajectory(rlm, traj)
    assert result.submitted is True
    assert result.payload == {"answer": "correct"}
    assert len(result.trajectory.turns) == 2
    # The output-validator rejection routes a repair turn: the retry is tagged
    # 'verifier_repair', proving the repair feedback was formatted and accepted.
    assert result.trajectory.turns[1].turn_type == "verifier_repair"


def test_replay_trajectory_detects_divergence_when_loop_wants_more_turns() -> None:
    # Recording has 1 non-submit turn but max_turns=5: the live loop will ask
    # for a 2nd LM response the recording does not have -> DivergenceError.
    traj = Trajectory(
        turns=[
            _turn(
                turn=1,
                code="print('hi')",
                stdout="hi\n",
                response_text="```python\nprint('hi')\n```",
            )
        ]
    )
    rlm = _dummy_rlm(max_turns=5)
    with pytest.raises(DivergenceError):
        replay_trajectory(rlm, traj)


def test_replay_trajectory_detects_underconsumption() -> None:
    # Recording has 2 turns but the first one already submits, so the loop
    # stops after turn 1 leaving a recorded response unused -> DivergenceError.
    traj = Trajectory(
        turns=[
            _turn(
                turn=1,
                code="SUBMIT({'answer': 'a'})",
                response_text=_submit_response("{'answer': 'a'}"),
                submitted=True,
                submit_payload={"answer": "a"},
            ),
            _turn(
                turn=2,
                code="print('never')",
                response_text="```python\nprint('never')\n```",
            ),
        ]
    )
    rlm = _dummy_rlm()
    with pytest.raises(DivergenceError):
        replay_trajectory(rlm, traj)


def test_replay_trajectory_round_trips_through_jsonl(tmp_path) -> None:
    traj = Trajectory(
        turns=[
            _turn(
                turn=1,
                code="SUBMIT({'answer': 'ok'})",
                response_text=_submit_response("{'answer': 'ok'}"),
                submitted=True,
                submit_payload={"answer": "ok"},
            )
        ],
        metadata={"task": "demo"},
    )
    path = tmp_path / "t.jsonl"
    traj.write_jsonl(path)
    loaded = Trajectory.from_jsonl(path)

    result = replay_trajectory(_dummy_rlm(), loaded)
    assert result.submitted is True
    assert result.payload == {"answer": "ok"}


def test_replay_trajectory_detects_code_mismatch_end_to_end() -> None:
    # The recorded response contains a real SUBMIT block, but the recorded
    # ``code`` was tampered with. On replay the loop extracts the real block,
    # which won't match the recorded code -> DivergenceError.
    traj = Trajectory(
        turns=[
            _turn(
                turn=1,
                code="THIS IS NOT THE REAL CODE",
                response_text=_submit_response("{'answer': 'a'}"),
                submitted=True,
                submit_payload={"answer": "a"},
            )
        ]
    )
    with pytest.raises(DivergenceError):
        replay_trajectory(_dummy_rlm(), traj)


def test_replay_trajectory_rejects_unsupported_engine() -> None:
    rlm = _dummy_rlm()
    rlm.engine = "v7-dspy"  # pretend a non-default engine
    traj = Trajectory(turns=[_turn(turn=1, response_text="x")])
    with pytest.raises(ValueError):
        replay_trajectory(rlm, traj)


def test_replay_is_hermetic_real_interpreter_never_constructed(monkeypatch) -> None:
    # If replay ever fell back to the real subprocess Interpreter, this sentinel
    # would fire. Replay must use only the in-memory ReplayInterpreter.
    import fabric_rlm.runtime as rt

    def explode(*a, **k):  # pragma: no cover - should never run
        raise AssertionError("real Interpreter was constructed during replay")

    monkeypatch.setattr(rt, "Interpreter", explode)
    traj = Trajectory(
        turns=[
            _turn(
                turn=1,
                code="SUBMIT({'answer': 'z'})",
                response_text=_submit_response("{'answer': 'z'}"),
                submitted=True,
                submit_payload={"answer": "z"},
            )
        ]
    )
    result = replay_trajectory(_dummy_rlm(), traj)
    assert result.submitted is True
