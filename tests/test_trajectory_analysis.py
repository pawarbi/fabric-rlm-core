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
    s = _make_traj()
    s = s.summary()
    assert s["turns"] == 2
    assert s["submitted"] is True
    assert s["submit_turn"] == 2
    assert s["errors"] == 0
    assert s["error_kinds"] == {}
    assert s["prompt_tokens"] == 220
    assert s["completion_tokens"] == 35
    assert s["total_tokens"] == 255
    assert s["lm_seconds"] == 1.5


def test_summary_returns_none_for_uninstrumented_tokens() -> None:
    """When no turn carries token data, summary() returns None — not 0 — so
    dashboards can distinguish "not instrumented" from "zero usage"."""
    traj = Trajectory()
    traj.append(
        TurnRecord(
            turn=1, code="pass", stdout="", stderr="", error=None, submitted=False, state={}
        )
    )
    s = traj.summary()
    assert s["prompt_tokens"] is None
    assert s["completion_tokens"] is None
    assert s["total_tokens"] is None
    assert s["cached_tokens"] is None
    assert s["reasoning_tokens"] is None
    assert s["lm_seconds"] is None


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
    """Detector requires 3 consecutive same-class errors to fire (avoids
    false-positives during normal recovery cycles)."""
    traj = Trajectory()
    for i in range(1, 5):
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
    assert len(repeated) == 1
    msg = repeated[0].message
    assert "consecutive" in msg
    assert "SyntaxError" in msg
    # Streak length and end turn should be reported so user sees how long the loop ran.
    assert "4" in msg


def test_diagnose_does_not_fire_repeated_error_on_two() -> None:
    traj = Trajectory()
    for i in range(1, 3):
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
    assert [i for i in traj.diagnose() if i.kind == "repeated_error"] == []


def test_diagnose_does_not_flag_python_type_annotations() -> None:
    """Real Python like ``Result: dict = {}`` must NOT trip markdown_in_code."""
    traj = Trajectory()
    traj.append(
        TurnRecord(
            turn=1,
            code='Result: dict[str, int] = {}\nResponse: str = "ok"\nFinalAnswer: list = []',
            stdout="",
            stderr="",
            error=None,
            submitted=False,
            state={},
        )
    )
    md = [i for i in traj.diagnose() if i.kind == "markdown_in_code"]
    assert md == [], f"False positive on type annotations: {md}"


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


def test_diagnose_token_cliff_handles_two_cliffs() -> None:
    """Baseline excludes the cliff turn itself, so two true cliffs both fire
    even though their presence inflates the naive overall mean."""
    traj = Trajectory()
    for i in [1, 2, 3, 4]:
        traj.append(
            TurnRecord(
                turn=i,
                code="pass",
                stdout="",
                stderr="",
                error=None,
                submitted=False,
                state={},
                prompt_tokens=500,
            )
        )
    traj.append(
        TurnRecord(
            turn=5, code="pass", stdout="", stderr="", error=None, submitted=False,
            state={}, prompt_tokens=80000,
        )
    )
    traj.append(
        TurnRecord(
            turn=6, code="pass", stdout="", stderr="", error=None, submitted=False,
            state={}, prompt_tokens=80000,
        )
    )
    cliffs = [i for i in traj.diagnose() if i.kind == "token_cliff"]
    assert {c.turn for c in cliffs} == {5, 6}


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


def test_cli_trace_inspect_accepts_stdin() -> None:
    """``-`` reads from stdin so users can pipe Spark/notebookutils output in."""
    payload = BUG_TRACE.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "fabric_rlm.cli", "trace", "inspect", "-", "--json"],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(result.stdout)
    assert out["summary"]["turns"] >= 1
    assert any(i["kind"] == "markdown_in_code" for i in out["issues"])


def test_cli_trace_inspect_fail_on_issues_exits_nonzero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "fabric_rlm.cli", "trace", "inspect", str(BUG_TRACE), "--fail-on-issues", "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["issues"]


def test_cli_trace_inspect_fail_on_issues_clean_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "fabric_rlm.cli", "trace", "inspect", str(CLEAN_TRACE), "--fail-on-issues", "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Remote URI handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri",
    [
        "abfss://container@account.dfs.core.windows.net/traces/x.jsonl",
        "s3://bucket/traces/x.jsonl",
        "https://example.com/x.jsonl",
        "gs://bucket/x.jsonl",
    ],
)
def test_from_jsonl_rejects_remote_uri_with_actionable_message(uri: str) -> None:
    """Library is dependency-free; users must pre-read remote files with their
    own client (notebookutils / fsspec / mlflow.artifacts / Spark) and pass a
    file-like, line iterable, or parsed dicts. The error message must say so."""
    with pytest.raises(ValueError) as exc:
        Trajectory.from_jsonl(uri)
    msg = str(exc.value).lower()
    assert "not supported" in msg or "uri" in msg
    # At least one of the suggested clients should be mentioned.
    assert any(hint in msg for hint in ("notebookutils", "fsspec", "mlflow", "spark"))


# ---------------------------------------------------------------------------
# Spark Row duck-typing
# ---------------------------------------------------------------------------


class _FakeSparkRow:
    """Minimal stub mimicking pyspark.sql.Row's .asDict(recursive=...) API."""

    def __init__(self, **kw: Any) -> None:
        self._d = kw

    def asDict(self, recursive: bool = False) -> dict[str, Any]:
        return dict(self._d)


def test_from_dicts_accepts_spark_like_rows() -> None:
    rows = [
        _FakeSparkRow(metadata={"skills": ["core"]}),
        _FakeSparkRow(turn=1, code="pass", stdout="", stderr="", error=None, submitted=False, state={}),
        _FakeSparkRow(turn=2, code='SUBMIT({"a": 1})', stdout="", stderr="", error=None, submitted=True, state={}),
    ]
    traj = Trajectory.from_dicts(rows)
    assert traj.metadata == {"skills": ["core"]}
    assert len(traj) == 2
    assert traj.turns[1].submitted is True


def test_from_dicts_rejects_non_dict_non_row() -> None:
    with pytest.raises(TypeError):
        Trajectory.from_dicts([42])
