"""Tests for Trajectory loading + summary + diagnose helpers."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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
        "s3://bucket/traces/x.jsonl",
        "https://example.com/x.jsonl",
        "gs://bucket/x.jsonl",
    ],
)
def test_from_jsonl_rejects_non_azure_remote_uri_with_actionable_message(uri: str) -> None:
    """Non-Azure remote URIs aren't natively supported. The error must tell
    the user to read the bytes with their own client and pass file-like /
    iterable / dicts to the loader."""
    with pytest.raises(ValueError) as exc:
        Trajectory.from_jsonl(uri)
    msg = str(exc.value).lower()
    assert "not supported" in msg or "uri" in msg
    assert any(hint in msg for hint in ("fsspec", "mlflow", "spark", "requests"))


def test_from_jsonl_lakehouse_fuse_path_reads_as_local_file(tmp_path: Path) -> None:
    """``/lakehouse/default/Files/...`` is a regular FUSE-mounted path on
    Fabric Spark; from a code path point of view it's just an absolute file
    path. We simulate it with a temp file and confirm no special handling
    is needed and no scheme-rejection happens."""
    fake_lakehouse_file = tmp_path / "lakehouse_like.jsonl"
    fake_lakehouse_file.write_text(
        BUG_TRACE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    traj = Trajectory.from_jsonl(str(fake_lakehouse_file))
    assert len(traj) > 0
    assert traj.metadata  # envelope round-tripped


# ---------------------------------------------------------------------------
# Azure storage URI loading (Fabric / Synapse)
# ---------------------------------------------------------------------------


class _FakeFs:
    """Mimics notebookutils.fs.cp(src, "file:/tmp/...") by writing local file."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def cp(self, src: str, dst: str) -> None:
        self.calls.append((src, dst))
        # Strip the ``file:`` prefix Fabric uses for the destination.
        local = dst[len("file:") :] if dst.startswith("file:") else dst
        Path(local).write_text(self.payload, encoding="utf-8")


def _install_fake_notebookutils(
    monkeypatch: pytest.MonkeyPatch, payload: str
) -> _FakeFs:
    """Inject a fake ``notebookutils`` module into ``sys.modules`` so the
    lazy-import path inside trajectory.py picks it up. We also clear any
    previously-imported ``mssparkutils`` to make the test order-independent."""
    import types

    fs = _FakeFs(payload)
    fake = types.ModuleType("notebookutils")
    fake.fs = fs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "notebookutils", fake)
    monkeypatch.setitem(sys.modules, "mssparkutils", fake)
    return fs


@pytest.mark.parametrize(
    "uri",
    [
        "abfss://container@account.dfs.core.windows.net/traces/x.jsonl",
        "abfs://container@account.dfs.core.windows.net/x.jsonl",
        "wasbs://container@account.blob.core.windows.net/x.jsonl",
        "wasb://container@account.blob.core.windows.net/x.jsonl",
    ],
)
def test_from_jsonl_loads_azure_uri_via_notebookutils(
    monkeypatch: pytest.MonkeyPatch, uri: str
) -> None:
    payload = BUG_TRACE.read_text(encoding="utf-8")
    fs = _install_fake_notebookutils(monkeypatch, payload)
    traj = Trajectory.from_jsonl(uri)
    assert len(traj) > 0
    # cp() was called exactly once with the supplied URI and a file: dest.
    assert len(fs.calls) == 1
    src, dst = fs.calls[0]
    assert src == uri
    assert dst.startswith("file:")
    # Diagnose still works on the loaded trajectory.
    assert any(i.kind == "markdown_in_code" for i in traj.diagnose())


def test_from_jsonl_uses_synapse_compat_fs_when_top_level_fs_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some Synapse runtimes ship ``notebookutils`` without a top-level
    ``.fs`` attribute; the usable API is ``notebookutils.mssparkutils.fs``.
    The reader must walk the candidate chain instead of giving up."""
    import types

    payload = CLEAN_TRACE.read_text(encoding="utf-8")
    fs = _FakeFs(payload)

    shim = types.ModuleType("notebookutils.mssparkutils")
    shim.fs = fs  # type: ignore[attr-defined]
    fake_nu = types.ModuleType("notebookutils")
    fake_nu.mssparkutils = shim  # type: ignore[attr-defined]
    # Deliberately NO `.fs` on the top-level notebookutils module.

    monkeypatch.setitem(sys.modules, "notebookutils", fake_nu)
    monkeypatch.delitem(sys.modules, "mssparkutils", raising=False)

    traj = Trajectory.from_jsonl("abfss://account/x.jsonl")
    assert len(traj) > 0
    assert len(fs.calls) == 1


def test_from_jsonl_azure_uri_falls_back_to_fsspec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When notebookutils/mssparkutils are absent, fsspec should be tried."""
    import types

    monkeypatch.delitem(sys.modules, "notebookutils", raising=False)
    monkeypatch.delitem(sys.modules, "mssparkutils", raising=False)

    # Block real notebookutils/mssparkutils imports; allow everything else.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__  # type: ignore[index]

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"notebookutils", "mssparkutils"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    payload = CLEAN_TRACE.read_text(encoding="utf-8")
    fake_fsspec = types.ModuleType("fsspec")

    class _Ctx:
        def __init__(self, text: str) -> None:
            self._text = text

        def __enter__(self) -> "_Ctx":
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def read(self) -> str:
            return self._text

    fake_fsspec.open = lambda uri, *a, **kw: _Ctx(payload)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fsspec", fake_fsspec)

    traj = Trajectory.from_jsonl(
        "abfss://container@account.dfs.core.windows.net/traces/x.jsonl"
    )
    assert len(traj) > 0


def test_from_jsonl_azure_uri_without_any_client_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "notebookutils", raising=False)
    monkeypatch.delitem(sys.modules, "mssparkutils", raising=False)
    monkeypatch.delitem(sys.modules, "fsspec", raising=False)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__  # type: ignore[index]

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"notebookutils", "mssparkutils", "fsspec"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError) as exc:
        Trajectory.from_jsonl("abfss://container@account.dfs.core.windows.net/x.jsonl")
    msg = str(exc.value).lower()
    assert "fabric" in msg or "synapse" in msg or "fsspec" in msg


def test_from_jsonl_does_not_fall_back_when_first_reader_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If an available reader (notebookutils.fs.cp) raises, we abort with a
    RuntimeError instead of silently retrying via fsspec — masking
    permission/auth/not-found errors would be a footgun."""
    import types

    class _BoomFs:
        def cp(self, src: str, dst: str) -> None:
            raise PermissionError("AAD says no")

    fake_nu = types.ModuleType("notebookutils")
    fake_nu.fs = _BoomFs()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "notebookutils", fake_nu)
    # If we accidentally fell back to fsspec we'd see this payload — but we shouldn't.
    fake_fsspec = types.ModuleType("fsspec")
    fake_fsspec.open = lambda *a, **kw: io.StringIO("{}\n")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fsspec", fake_fsspec)

    with pytest.raises(RuntimeError) as exc:
        Trajectory.from_jsonl("abfss://account/x.jsonl")
    assert "AAD" in str(exc.value)


def test_from_jsonl_azure_uri_error_redacts_sas_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SAS tokens in query strings must not appear in raised exception text."""
    import types

    class _BoomFs:
        def cp(self, src: str, dst: str) -> None:
            raise RuntimeError("backend exploded")

    fake_nu = types.ModuleType("notebookutils")
    fake_nu.fs = _BoomFs()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "notebookutils", fake_nu)

    sas = "sv=2024-11-04&sig=SECRETSIGNATURE&se=2025-01-01"
    uri = f"abfss://c@account.dfs.core.windows.net/x.jsonl?{sas}"
    with pytest.raises(RuntimeError) as exc:
        Trajectory.from_jsonl(uri)
    msg = str(exc.value)
    assert "SECRETSIGNATURE" not in msg
    assert "sig=" not in msg


def test_from_jsonl_azure_uri_cleans_up_temp_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TemporaryDirectory used as the cp() destination must be removed."""
    import tempfile as _tempfile

    captured_dirs: list[str] = []
    real_tmpdir_cls = _tempfile.TemporaryDirectory

    class _TrackingTemporaryDirectory(real_tmpdir_cls):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            captured_dirs.append(self.name)

    monkeypatch.setattr(
        "fabric_rlm.trajectory.tempfile.TemporaryDirectory",
        _TrackingTemporaryDirectory,
    )
    _install_fake_notebookutils(monkeypatch, CLEAN_TRACE.read_text(encoding="utf-8"))

    Trajectory.from_jsonl("abfss://account/x.jsonl")
    assert captured_dirs, "expected TemporaryDirectory to be used"
    for d in captured_dirs:
        assert not Path(d).exists(), f"temp dir {d} was not cleaned up"


def test_from_jsonl_accepts_file_uri_for_local_path(tmp_path: Path) -> None:
    """``file:///abs/path`` and ``file://localhost/abs/path`` are local."""
    fake = tmp_path / "x.jsonl"
    fake.write_text(CLEAN_TRACE.read_text(encoding="utf-8"), encoding="utf-8")
    # Build a file:// URI that round-trips through urlsplit on Windows + POSIX.
    uri = fake.resolve().as_uri()
    traj = Trajectory.from_jsonl(uri)
    assert len(traj) > 0


def test_from_jsonl_rejects_file_uri_with_remote_host() -> None:
    with pytest.raises(ValueError) as exc:
        Trajectory.from_jsonl("file://otherhost/some/path.jsonl")
    assert "not supported" in str(exc.value).lower()


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
