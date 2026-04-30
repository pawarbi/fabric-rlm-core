from pathlib import Path

import pytest

from fabric_rlm import File, Interpreter, WorkerTimeout


def test_interpreter_persists_state() -> None:
    with Interpreter(timeout=5) as interp:
        first = interp.execute("x = 10")
        second = interp.execute("x += 5\nprint(x)")

    assert first.ok
    assert second.ok
    assert second.stdout.strip() == "15"
    assert second.state["x"] == 15


def test_interpreter_survives_syntax_and_runtime_errors() -> None:
    with Interpreter(timeout=5) as interp:
        syntax = interp.execute("if True print('bad')")
        runtime = interp.execute("1 / 0")
        recovery = interp.execute("answer = 42")

    assert not syntax.ok
    assert "SyntaxError" in syntax.error
    assert not runtime.ok
    assert "ZeroDivisionError" in runtime.error
    assert recovery.ok
    assert recovery.state["answer"] == 42


def test_interpreter_survives_system_exit() -> None:
    with Interpreter(timeout=5) as interp:
        system_exit = interp.execute("raise SystemExit('stop requested')")
        recovery = interp.execute("answer = 42")

    assert not system_exit.ok
    assert "SystemExit" in system_exit.error
    assert recovery.ok
    assert recovery.state["answer"] == 42


def test_submit_is_first_class_response() -> None:
    with Interpreter(timeout=5) as interp:
        result = interp.execute("SUBMIT(answer=42, status='done')")

    assert result.ok
    assert result.submitted
    assert result.submit_payload == {"answer": 42, "status": "done"}


def test_top_level_await() -> None:
    with Interpreter(timeout=5) as interp:
        result = interp.execute("import asyncio\nawait asyncio.sleep(0)\nvalue = 123")

    assert result.ok
    assert result.state["value"] == 123


def test_reset_clears_user_state() -> None:
    with Interpreter(timeout=5) as interp:
        interp.execute("x = 1")
        reset = interp.reset()

    assert reset["ok"]
    assert "x" not in reset["state"]


def test_set_inputs_decodes_file(tmp_path: Path) -> None:
    input_file = File(tmp_path / "input.txt")
    input_file.write_text("fabric")

    with Interpreter(timeout=5) as interp:
        interp.set_inputs({"input_file": input_file})
        result = interp.execute("text = input_file.read_text()")

    assert result.ok
    assert result.state["text"] == "fabric"


def test_timeout_kills_worker() -> None:
    interp = Interpreter(timeout=0.2).start()
    try:
        with pytest.raises(WorkerTimeout):
            interp.execute("import time\ntime.sleep(5)")
        assert not interp.is_running
    finally:
        interp.shutdown()

