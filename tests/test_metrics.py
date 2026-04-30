import pytest

from fabric_rlm.metrics import (
    ValidationReport,
    assert_report_passed,
    classify_failure,
    summarize_trajectory,
)
from fabric_rlm.trajectory import Trajectory, TurnRecord


def test_validation_report_passes_only_when_all_checks_pass() -> None:
    report = ValidationReport(run_id="unit")
    report.add_check("imports", True)
    report.add_check("worker", False, failure_type="runtime")

    assert not report.passed
    assert report.to_dict()["checks"][1]["failure_type"] == "runtime"
    with pytest.raises(AssertionError):
        assert_report_passed(report)


def test_classify_failure() -> None:
    assert classify_failure("SyntaxError: invalid syntax") == "syntax"
    assert classify_failure("worker timed out") == "timeout"
    assert classify_failure("solution = missing") == "formatting"


def test_summarize_trajectory() -> None:
    trajectory = Trajectory()
    trajectory.append(
        TurnRecord(
            turn=1,
            code="x = 1",
            stdout="",
            stderr="",
            error=None,
            submitted=False,
            state={"x": 1},
            duration_s=0.1,
        )
    )
    trajectory.append(
        TurnRecord(
            turn=2,
            code="SUBMIT(answer=x)",
            stdout="",
            stderr="",
            error=None,
            submitted=True,
            state={"x": 1},
            duration_s=0.2,
        )
    )

    summary = summarize_trajectory(trajectory)

    assert summary["turns"] == 2
    assert summary["submitted"] is True
    assert summary["wall_time_s"] == pytest.approx(0.3)

