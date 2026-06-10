"""CLI flag handling: --version, explicit-zero overrides, engine passthrough."""

from __future__ import annotations

import json

import pytest

import fabric_rlm
import fabric_rlm.cli as cli_mod


def test_version_flag_prints_version_and_exits(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert fabric_rlm.__version__ in out


class _FakeResult:
    submitted = True
    payload = {"output": "ok"}
    failure_reason = None
    final_state: dict = {}

    class trajectory:  # noqa: N801 - duck-typed attribute
        @staticmethod
        def write_jsonl(path):
            pass


def _capture_from_task(captured):
    def fake_from_task(task, inputs=None, outputs=None, **kwargs):
        captured.update(kwargs, task=task)

        class _FakeRLM:
            def run(self):
                return _FakeResult()

        return _FakeRLM()

    return fake_from_task


def _write_task_file(tmp_path, **extra):
    config = {"task": "t", "lm": "mock/model", "inputs": {}, "outputs": ["output"]}
    config.update(extra)
    path = tmp_path / "task.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


def test_cli_flag_overrides_config_max_turns(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(cli_mod.RLM, "from_task", _capture_from_task(captured))
    task_file = _write_task_file(tmp_path, max_turns=9)
    cli_mod.main(["run", task_file, "--max-turns", "3"])
    assert captured["max_turns"] == 3


def test_config_max_turns_used_when_no_flag(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(cli_mod.RLM, "from_task", _capture_from_task(captured))
    task_file = _write_task_file(tmp_path, max_turns=9)
    cli_mod.main(["run", task_file])
    assert captured["max_turns"] == 9


def test_engine_passthrough_from_config_and_flag(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(cli_mod.RLM, "from_task", _capture_from_task(captured))
    task_file = _write_task_file(tmp_path, engine="dspy")
    cli_mod.main(["run", task_file])
    assert captured["engine"] == "dspy"

    cli_mod.main(["run", task_file, "--engine", "default"])
    assert captured["engine"] == "default"


def test_unknown_config_keys_warn_but_run(tmp_path, monkeypatch, capsys) -> None:
    captured: dict = {}
    monkeypatch.setattr(cli_mod.RLM, "from_task", _capture_from_task(captured))
    task_file = _write_task_file(tmp_path, not_a_real_key=1)
    rc = cli_mod.main(["run", task_file])
    assert rc == 0
    err = capsys.readouterr().err
    assert "not_a_real_key" in err
