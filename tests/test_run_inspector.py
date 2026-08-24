from __future__ import annotations

from pathlib import Path

import pytest

from fabric_rlm import RunInspector
from fabric_rlm.runtime import RLMResult
from fabric_rlm.trajectory import Trajectory, TurnRecord


def _turn(
    number: int,
    *,
    code: str = "print('ok')",
    stdout: str = "ok",
    error: str | None = None,
    submitted: bool = False,
    turn_type: str = "normal",
    duration_s: float = 1.0,
) -> TurnRecord:
    return TurnRecord(
        turn=number,
        code=code,
        stdout=stdout,
        stderr="",
        error=error,
        submitted=submitted,
        state={},
        response_text=f"response {number}",
        duration_s=duration_s,
        turn_type=turn_type,
        prompt_tokens=100,
        completion_tokens=20,
        reasoning_tokens=5,
        cached_tokens=80,
        lm_call_seconds=0.75,
        worker_execute_seconds=duration_s,
        submit_payload={"answer": "done"} if submitted else None,
    )


def _result(*turns: TurnRecord) -> RLMResult:
    return RLMResult(
        submitted=any(turn.submitted for turn in turns),
        payload={"answer": "done"} if any(turn.submitted for turn in turns) else None,
        trajectory=Trajectory(turns=list(turns)),
        final_state={},
        total_prompt_tokens=sum(turn.prompt_tokens or 0 for turn in turns),
        total_completion_tokens=sum(turn.completion_tokens or 0 for turn in turns),
        total_cached_tokens=sum(turn.cached_tokens or 0 for turn in turns),
        total_reasoning_tokens=sum(turn.reasoning_tokens or 0 for turn in turns),
        total_lm_seconds=sum(turn.lm_call_seconds or 0 for turn in turns),
        total_worker_seconds=sum(turn.worker_execute_seconds or 0 for turn in turns),
        max_turns=10,
    )


def test_result_inspect_returns_notebook_renderable_timeline() -> None:
    result = _result(
        _turn(1, error="ValueError: bad query"),
        _turn(2, submitted=True, turn_type="validation_repair", duration_s=12.0),
    )

    inspector = result.inspect(slow_turn_seconds=10)
    html = inspector._repr_html_()

    assert isinstance(inspector, RunInspector)
    assert "RLM run inspector" in html
    assert "Turn 1" in html
    assert "Error" in html
    assert "Turn 2" in html
    assert "Slow" in html
    assert "Repair" in html
    assert "SUBMITTED" in html
    assert "<details" in html


def test_inspector_marks_a_successful_turn_after_an_error_as_recovered() -> None:
    result = _result(
        _turn(1, error="ValueError: bad query"),
        _turn(2),
    )

    html = result.inspect().to_html()

    assert "Recovered" in html


def test_inspector_escapes_untrusted_trajectory_content() -> None:
    result = _result(
        _turn(
            1,
            code="<script>alert('code')</script>",
            stdout="<img src=x onerror=alert('stdout')>",
            error="</style><script>alert('error')</script>",
        )
    )

    html = result.inspect().to_html()

    assert "<script>" not in html
    assert "<img src=x" not in html
    assert "</style><script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img src=x" in html


def test_inspector_truncates_large_sections_with_an_explicit_marker() -> None:
    result = _result(_turn(1, stdout="x" * 100))

    html = result.inspect(max_chars=20).to_html()

    assert "truncated" in html
    assert "x" * 30 not in html


def test_inspector_handles_a_run_with_no_turns() -> None:
    result = _result()

    html = result.inspect().to_html()

    assert "No executable turns were recorded" in html
    assert "NOT SUBMITTED" in html


def test_inspector_can_save_a_standalone_html_document(tmp_path: Path) -> None:
    destination = tmp_path / "run.html"

    returned = _result(_turn(1, submitted=True)).inspect().save_html(destination)

    assert returned == destination
    saved = destination.read_text(encoding="utf-8")
    assert saved.startswith("<!doctype html>")
    assert "<title>RLM run inspector</title>" in saved
    assert "Turn 1" in saved


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_chars": 0}, "max_chars"),
        ({"max_chars": -1}, "max_chars"),
        ({"max_chars": True}, "max_chars"),
        ({"max_chars": "20"}, "max_chars"),
        ({"slow_turn_seconds": 0}, "slow_turn_seconds"),
        ({"slow_turn_seconds": -1.0}, "slow_turn_seconds"),
        ({"slow_turn_seconds": False}, "slow_turn_seconds"),
        ({"slow_turn_seconds": "10"}, "slow_turn_seconds"),
    ],
)
def test_inspector_validates_display_limits(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _result(_turn(1)).inspect(**kwargs)
