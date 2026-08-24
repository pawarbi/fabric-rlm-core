"""A worker hard-crash names its likely cause when the code was threading.

Two benchmark trials died with a GIL fatal error inside ThreadPoolExecutor
around native calls; both blind solves crashed the same way, the ensemble
agreed on empty, and the trial was lost. The model never saw why, so its
retry repeated the pattern. The hint turns the fatal into a repairable turn.
"""

from __future__ import annotations

import queue

import pytest

from fabric_rlm.interpreter import (
    Interpreter,
    SubprocessPythonInterpreter,
    WorkerProtocolError,
    concurrency_death_hint,
)

THREADED = (
    "from concurrent.futures import ThreadPoolExecutor\n"
    "with ThreadPoolExecutor(8) as ex: list(ex.map(f, rows))"
)
SERIAL = "for r in rows: f(r)"


def test_threaded_code_gets_a_hint():
    hint = concurrency_death_hint(THREADED)
    assert "ThreadPoolExecutor" in hint
    assert "serial loop" in hint


def test_serial_code_gets_no_hint():
    assert concurrency_death_hint(SERIAL) == ""


def test_no_code_gets_no_hint():
    assert concurrency_death_hint(None) == ""
    assert concurrency_death_hint("") == ""


def test_legacy_death_message_carries_the_hint():
    """Simulate the death branch without spawning a worker: a closed stdout
    queue (None sentinel) after threaded code was recorded."""
    it = Interpreter.__new__(Interpreter)
    it._stderr_buf = ["Fatal Python error: PyEval_SaveThread"]
    it._stdout_queue = queue.Queue()
    it._stdout_queue.put(None)
    it.timeout = 1
    it._last_exec_code = THREADED
    with pytest.raises(WorkerProtocolError) as err:
        it._recv()
    msg = str(err.value)
    assert "Worker exited without response" in msg
    assert "ThreadPoolExecutor" in msg


def test_legacy_death_without_threading_is_unchanged():
    it = Interpreter.__new__(Interpreter)
    it._stderr_buf = []
    it._stdout_queue = queue.Queue()
    it._stdout_queue.put(None)
    it.timeout = 1
    it._last_exec_code = SERIAL
    with pytest.raises(WorkerProtocolError) as err:
        it._recv()
    assert "NOTE" not in str(err.value)


def test_v7_death_message_carries_the_hint():
    v7 = SubprocessPythonInterpreter.__new__(SubprocessPythonInterpreter)
    v7._stderr_buf = ["Fatal Python error: PyEval_SaveThread"]
    v7._stdout_queue = queue.Queue()
    v7._stdout_queue.put(None)
    v7._last_exec_code = THREADED
    with pytest.raises(Exception) as err:
        v7._read_response_line(timeout=1, context="awaiting execute response")
    msg = str(err.value)
    assert "Worker exited unexpectedly" in msg
    assert "ThreadPoolExecutor" in msg
