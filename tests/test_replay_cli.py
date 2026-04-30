"""End-to-end tests for the trajectory replay CLI."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest

from fabric_rlm.replay import main


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


@pytest.fixture
def basic_trajectory(tmp_path: Path) -> Path:
    records = [
        {"metadata": {"max_turns": 3}},
        {
            "turn": 0,
            "code": "print('hello world')",
            "stdout": "hello world\n",
            "stderr": "",
            "error": None,
            "submitted": False,
            "duration_s": 0.05,
            "turn_type": "normal",
            "validation_errors": [],
        },
        {
            "turn": 1,
            "code": "raise ValueError('boom')",
            "stdout": "",
            "stderr": "Traceback ...\nValueError: boom",
            "error": "ValueError: boom",
            "submitted": False,
            "duration_s": 0.10,
            "turn_type": "normal",
            "validation_errors": [],
        },
        {
            "turn": 2,
            "code": "submit(answer=42)",
            "stdout": "submitted\n",
            "stderr": "",
            "error": None,
            "submitted": True,
            "duration_s": 0.01,
            "turn_type": "submit",
            "validation_errors": [],
        },
    ]
    path = tmp_path / "traj.jsonl"
    _write_jsonl(path, records)
    return path


def test_basic_render(basic_trajectory: Path) -> None:
    rc, out = _run([str(basic_trajectory), "--no-color"])
    assert rc == 0
    assert "Turn 0" in out
    assert "Turn 1" in out
    assert "Turn 2" in out
    assert "=== Turn" in out
    assert "print('hello world')" in out
    assert "hello world" in out
    assert "ValueError: boom" in out
    assert "submit(answer=42)" in out
    assert "submitted=True" in out


def test_json_mode(basic_trajectory: Path) -> None:
    rc, out = _run([str(basic_trajectory), "--json"])
    assert rc == 0
    payload = json.loads(out)
    assert payload["metadata"] == {"max_turns": 3}
    assert len(payload["turns"]) == 3
    assert payload["turns"][0]["code"] == "print('hello world')"
    assert payload["turns"][2]["submitted"] is True
    # Round-trip: every turn carries the canonical fields
    for t in payload["turns"]:
        for key in ("turn_index", "turn_type", "code", "stdout", "stderr",
                    "error", "submitted", "duration_s", "validation_errors"):
            assert key in t


def test_turns_limit(tmp_path: Path) -> None:
    records = [{"metadata": {}}]
    for i in range(5):
        records.append({
            "turn": i,
            "code": f"print({i})",
            "stdout": f"{i}\n",
            "stderr": "",
            "error": None,
            "submitted": False,
            "duration_s": 0.0,
        })
    path = tmp_path / "five.jsonl"
    _write_jsonl(path, records)

    rc, out = _run([str(path), "--no-color", "--turns", "2"])
    assert rc == 0
    assert "Turn 0" in out
    assert "Turn 1" in out
    assert "Turn 2" not in out
    assert "Turn 3" not in out


def test_handles_metadata_nesting(tmp_path: Path) -> None:
    # Schema variant: canonical fields are nested under "metadata".
    records = [
        {
            "metadata": {
                "turn_index": 0,
                "code": "x = 1 + 1",
                "stdout": "result=2\n",
                "stderr": "",
                "error": None,
                "submitted": False,
                "duration_s": 0.02,
                "turn_type": "normal",
            },
        },
    ]
    path = tmp_path / "nested.jsonl"
    _write_jsonl(path, records)

    rc, out = _run([str(path), "--no-color"])
    assert rc == 0
    assert "Turn 0" in out
    assert "x = 1 + 1" in out
    assert "result=2" in out


def test_full_disables_truncation(tmp_path: Path) -> None:
    big = "A" * 10_000
    records = [
        {"metadata": {}},
        {
            "turn": 0,
            "code": "print('x')",
            "stdout": big,
            "stderr": "",
            "error": None,
            "submitted": False,
            "duration_s": 0.0,
        },
    ]
    path = tmp_path / "big.jsonl"
    _write_jsonl(path, records)

    rc_default, out_default = _run([str(path), "--no-color"])
    assert rc_default == 0
    assert "truncated" in out_default
    # The middle of the giant blob must not all be present.
    assert big not in out_default

    rc_full, out_full = _run([str(path), "--no-color", "--full"])
    assert rc_full == 0
    assert "truncated" not in out_full
    assert big in out_full
