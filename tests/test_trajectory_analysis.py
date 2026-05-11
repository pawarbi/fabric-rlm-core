"""Tests for Trajectory loading + summary + diagnose helpers."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fabric_rlm import Issue, Trajectory, TurnRecord


FIXTURES = Path(__file__).parent / "fixtures"
BUG_TRACE = FIXTURES / "trajectory_with_md_bug.jsonl"
CLEAN_TRACE = FIXTURES / "trajectory_clean.jsonl"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _make_traj() -> Trajectory:
    traj = Trajectory(metadata={"skills": ["core"]})
    traj.append(
        TurnRecord(
            turn=1,
            code="x = 1\nprint(x)",
            stdout="1\n",
            stderr="",
            error=None,
            submitted=False,
            state={"x": 1},
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            lm_call_seconds=1.5,
        )
    )
    traj.append(
        TurnRecord(
            turn=2,
            code='SUBMIT({"answer": 42})',
            stdout="",
            stderr="",
            error=None,
            submitted=True,
            state={"x": 1},
            prompt_tokens=120,
            completion_tokens=15,
            total_tokens=135,
            submit_payload={"answer": 42},
        )
    )
    return traj


def test_round_trip_via_jsonl(tmp_path: Path) -> None:
    original = _make_traj()
    p = tmp_path / "t.jsonl"
    original.write_jsonl(p)
    loaded = Trajectory.from_jsonl(p)
    assert loaded.metadata == original.metadata
    assert len(loaded) == len(original)
    for o, l in zip(original.turns, loaded.turns):
        assert o.turn == l.turn
        assert o.code == l.code
        assert o.submitted == l.submitted
        assert o.total_tokens == l.total_tokens


def test_from_jsonl_accepts_file_like(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    _make_traj().write_jsonl(p)
    with open(p, encoding="utf-8") as fh:
        loaded = Trajectory.from_jsonl(fh)
    assert len(loaded) == 2


def test_from_jsonl_accepts_iterable_of_lines(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    _make_traj().write_jsonl(p)
    lines = p.read_text(encoding="utf-8").splitlines()
    loaded = Trajectory.from_jsonl(iter(lines))
    assert len(loaded) == 2


def test_from_jsonl_accepts_stringio() -> None:
    payload = io.StringIO(
        json.dumps({"metadata": {"skills": []}})
        + "\n"
        + json.dumps(
            {
                "turn": 1,
                "code": "pass",
                "stdout": "",
                "stderr": "",
                "error": None,
                "submitted": False,
                "state": {},
            }
        )
        + "\n"
    )
    loaded = Trajectory.from_jsonl(payload)
    assert len(loaded) == 1
    assert loaded.metadata == {"skills": []}


def test_from_dicts_forward_compat_ignores_unknown_fields() -> None:
    records = [
        {"metadata": {}},
        {
            "turn": 1,
            "code": "pass",
            "stdout": "",
            "stderr": "",
            "error": None,
            "submitted": False,
            "state": {},
            "future_field_added_later": "ignored cleanly",
        },
    ]
    loaded = Trajectory.from_dicts(records)
    assert len(loaded) == 1
    assert loaded.turns[0].turn == 1


def test_from_dicts_backward_compat_fills_missing_optionals() -> None:
    # Old trajectory missing reasoning_tokens / cached_tokens / submit_payload.
    records = [
        {"metadata": {}},
        {"turn": 1, "code": "pass"},
    ]
    loaded = Trajectory.from_dicts(records)
    assert loaded.turns[0].cached_tokens is None
    assert loaded.turns[0].reasoning_tokens is None
    assert loaded.turns[0].submit_payload is None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_basic_shape() -> None:
    s = _make_traj().summary()
    assert s["turns"] == 2
    assert s["submitted"] is True
    assert s["submit_turn"] == 2
    assert s["errors"] == 0
    assert s["error_kinds"] == {}
    assert s["prompt_tokens"] == 220
    assert s["completion_tokens"] == 35
    assert s["total_tokens"] == 255
    assert s["lm_seconds"] == 1.5


def test_summary_counts_error_kinds() -> None:
    traj = Trajectory()
    traj.append(
        TurnRecord(
            turn=1,
            code="x =",
            stdout="",
            stderr="",
            error='File "<x>", line 1\n  x =\nSyntaxError: invalid syntax',
            submitted=False,
            state={},
        )
    )
    traj.append(
        TurnRecord(
            turn=2,
            code="undefined_name",
            stdout="",
            stderr="",
            error="Traceback ...\nNameError: name 'undefined_name' is not defined",
            submitted=False,
            state={},
        )
    )
    s = traj.summary()
    assert s["errors"] == 2
    assert s["error_kinds"] == {"SyntaxError": 1, "NameError": 1}


def test_summary_on_real_clean_fixture() -> None:
    s = Trajectory.from_jsonl(CLEAN_TRACE).summary()
    assert s["turns"] >= 1
    assert s["total_tokens"] > 0
    assert s["lm_seconds"] > 0


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------


def test_diagnose_clean_trajectory_has_no_issues() -> None:
    issues = _make_traj().diagnose()
    assert issues == []


def test_diagnose_finds_markdown_in_code_on_real_bug_fixture() -> None:
    """The mode2_extract_customer fixture contains the actual ## VERIFY bug."""
    issues = Trajectory.from_jsonl(BUG_TRACE).diagnose()
    md_issues = [i for i in issues if i.kind == "markdown_in_code"]
    assert md_issues, "Expected markdown_in_code detector to fire on bug fixture"
    # The buggy turn has a 'Target:' label and '- Output...' bullet
    assert any("Target" in i.message or "Output" in i.message or "-" in i.message for i in md_issues)


def test_diagnose_python_comment_heading_is_not_flagged() -> None:
    traj = Trajectory()
    traj.append(
        TurnRecord(
            turn=1,
            code="# ## PLAN\n# Step 1: read file\nprint('ok')",
            stdout="ok\n",
            stderr="",
            error=None,
            submitted=False,
            state={},
        )
    )
    assert [i for i in traj.diagnose() if i.kind == "markdown_in_code"] == []


def test_diagnose_finds_repeated_error() -> None:
    traj = Trajectory()
    for i in range(1, 4):
        traj.append(
            TurnRecord(
                turn=i,
                code="x =",
                stdout="",
                stderr="",
                error="SyntaxError: invalid syntax",
                submitted=False,
                state={},
            )
        )
    repeated = [i for i in traj.diagnose() if i.kind == "repeated_error"]
    assert repeated, "Expected repeated_error detector to fire"


def test_diagnose_finds_noop_turn() -> None:
    traj = Trajectory()
    traj.append(
        TurnRecord(
            turn=1, code="", stdout="", stderr="", error=None, submitted=False, state={}
        )
    )
    noops = [i for i in traj.diagnose() if i.kind == "noop_turn"]
    assert len(noops) == 1


def test_diagnose_finds_token_cliff() -> None:
    traj = Trajectory()
    for i in range(1, 5):
        traj.append(
            TurnRecord(
                turn=i,
                code="pass",
                stdout="",
                stderr="",
                error=None,
                submitted=False,
                state={},
                prompt_tokens=1000,
            )
        )
    traj.append(
        TurnRecord(
            turn=5,
            code="pass",
            stdout="",
            stderr="",
            error=None,
            submitted=False,
            state={},
            prompt_tokens=50000,
        )
    )
    cliffs = [i for i in traj.diagnose() if i.kind == "token_cliff"]
    assert len(cliffs) == 1
    assert cliffs[0].turn == 5


def test_issue_is_serializable() -> None:
    issue = Issue(turn=3, kind="repeated_error", message="loop")
    assert issue.to_dict() == {"turn": 3, "kind": "repeated_error", "message": "loop"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_trace_inspect_json_output(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "fabric_rlm.cli", "trace", "inspect", str(BUG_TRACE), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "summary" in payload and "issues" in payload
    assert payload["summary"]["turns"] >= 1
    assert any(i["kind"] == "markdown_in_code" for i in payload["issues"])


def test_cli_trace_inspect_human_output() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "fabric_rlm.cli", "trace", "inspect", str(BUG_TRACE)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Trajectory:" in result.stdout
    assert "turns" in result.stdout
    assert "Diagnostics:" in result.stdout


def test_cli_trace_inspect_no_diagnose_skips_issues() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "fabric_rlm.cli", "trace", "inspect", str(BUG_TRACE), "--no-diagnose", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["issues"] == []
