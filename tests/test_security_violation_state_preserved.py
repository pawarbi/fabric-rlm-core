"""A parent-side security rejection must not wipe the run's reported state.

Pre-fix bug: the fabricated ``ExecResult`` for a SecurityPolicy violation
carried ``state={}``; the v6 loop then did ``final_state = result.state``
unconditionally, so one rejected turn erased the reported namespace snapshot
even though the worker's real namespace was untouched.
"""

from __future__ import annotations

from fabric_rlm import RLM
from fabric_rlm.interpreter import ExecResult

from test_runtime_verifier import ScriptedLM, _install_fake_interpreter, _submit


def _ok_turn_with_state(state: dict) -> ExecResult:
    return ExecResult(
        ok=True, submitted=False, stdout="set up", stderr="", state=dict(state)
    )


def _security_rejection() -> ExecResult:
    msg = "SecurityPolicyViolation: call to 'os.remove' is disabled"
    return ExecResult(
        ok=False,
        submitted=False,
        stdout="",
        stderr=msg,
        state={},
        error=msg,
        reached_worker=False,
    )


def test_security_rejection_preserves_final_state(monkeypatch) -> None:
    good_state = {"df_rows": 42, "answer_draft": "x"}
    _install_fake_interpreter(
        monkeypatch,
        results=[
            _ok_turn_with_state(good_state),
            _security_rejection(),
            _submit({"output": "done"}),
        ],
    )
    lm = ScriptedLM(
        [
            "```python\ndf_rows = 42\n```",
            "```python\nimport os\nos.remove('x')\n```",
            "```python\nSUBMIT(output='done')\n```",
        ]
    )
    rlm = RLM.from_task(
        "task",
        outputs=["output"],
        lm=lm,
        max_turns=5,
        timeout=5,
        enable_verifier=False,
    )
    result = rlm.run()

    rejected_turn = result.trajectory.turns[1]
    assert rejected_turn.error and "SecurityPolicyViolation" in rejected_turn.error
    # The rejected turn's recorded state mirrors the last real snapshot
    # instead of a misleading empty namespace.
    assert rejected_turn.state == good_state


def test_security_rejection_final_state_when_run_ends_on_violation(monkeypatch) -> None:
    """Run that exhausts turns right after a rejection still reports the
    last real snapshot as final_state."""
    good_state = {"k": 1}
    _install_fake_interpreter(
        monkeypatch,
        results=[
            _ok_turn_with_state(good_state),
            _security_rejection(),
        ],
    )
    lm = ScriptedLM(
        [
            "```python\nk = 1\n```",
            "```python\nimport os\nos.remove('x')\n```",
        ]
    )
    rlm = RLM.from_task(
        "task",
        outputs=["output"],
        lm=lm,
        max_turns=2,
        timeout=5,
        enable_verifier=False,
    )
    result = rlm.run()

    assert result.submitted is False
    assert result.final_state == good_state
