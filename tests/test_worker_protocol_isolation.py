"""Writes to the worker's fd 1 from outside Python must not corrupt the protocol.

``contextlib.redirect_stdout`` only sees writes that go through ``sys.stdout``.
A .NET runtime loaded through pythonnet, a C extension or a child process
writes to file descriptor 1 directly. SemPy's XMLA client does this for
variant-typed measures, which used to end every such run as ``worker_died``.
"""

from __future__ import annotations

import queue

import pytest

from fabric_rlm import Interpreter
from fabric_rlm.interpreter import (
    SubprocessPythonInterpreter,
    WorkerProtocolError,
    WorkerTimeout,
    _parse_protocol_line,
)

NATIVE_WRITE = (
    "import os\n"
    "os.write(1, b\"2026-09-20 00:08:09.050: '[__m0]' ('System.Object'/'Object') "
    "is serialized as string.\\n\")\n"
    "os.write(1, b'42\\n')\n"          # valid JSON, still not a frame
    "os.write(1, b'no newline here ')\n"
    "print('python-level print')\n"
    "value = 7\n"
)


@pytest.mark.parametrize("isolate", ["1", "0"])
def test_native_stdout_writes_do_not_break_the_default_engine(
    monkeypatch: pytest.MonkeyPatch, isolate: str
) -> None:
    # "1": the worker moves its protocol off fd 1. "0": an older worker that
    # shares the stream, where the parent has to skip the stray lines.
    monkeypatch.setenv("FABRIC_RLM_ISOLATE_PROTOCOL", isolate)
    with Interpreter(timeout=30) as interp:
        first = interp.execute(NATIVE_WRITE)
        second = interp.execute("print(value + 1)")

    assert first.ok, first
    assert "python-level print" in first.stdout
    assert first.state["value"] == 7
    assert second.ok and second.stdout.strip() == "8"


def test_isolated_worker_sends_native_writes_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FABRIC_RLM_ISOLATE_PROTOCOL", "1")
    with Interpreter(timeout=30) as interp:
        assert interp.execute("import os\nos.write(1, b'native marker line\\n')").ok
        assert interp.execute("x = 1").ok
        captured = "".join(interp._stderr_buf)
    assert "native marker line" in captured
    assert "[worker stdout]" not in captured  # it never reached the protocol stream


def test_native_stdout_writes_do_not_break_the_dspy_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("dspy")
    monkeypatch.setenv("FABRIC_RLM_ISOLATE_PROTOCOL", "0")
    interp = SubprocessPythonInterpreter(timeout=30)
    try:
        interp.start()
        output = interp.execute(NATIVE_WRITE + "print(value)")
    finally:
        interp.shutdown()
    assert "7" in str(output)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('{"ok": true, "stdout": ""}\n', {"ok": True, "stdout": ""}),
        ("'[__m0]' ('System.Object'/'Object') is serialized as string.\n", None),
        ("42\n", None),
        ('"a json string"\n', None),
        ("[1, 2]\n", None),
        ("\n", None),
        ('stray text {"ok": true}\n', {"ok": True}),
        ("stray text {not json}\n", None),
    ],
)
def test_parse_protocol_line(line: str, expected: dict | None) -> None:
    assert _parse_protocol_line(line) == expected


def test_recv_skips_noise_under_one_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    interp = Interpreter(timeout=5)
    interp._stdout_queue = queue.Queue()
    for line in ("noise one\n", "7\n", '{"ok": true}\n'):
        interp._stdout_queue.put(line)
    assert interp._recv() == {"ok": True}
    assert any("noise one" in entry for entry in interp._stderr_buf)


def test_recv_times_out_when_only_noise_arrives(monkeypatch: pytest.MonkeyPatch) -> None:
    interp = Interpreter(timeout=5)
    interp._stdout_queue = queue.Queue()
    monkeypatch.setattr(interp, "kill", lambda: None)
    interp._stdout_queue.put("noise\n")
    with pytest.raises(WorkerTimeout):
        interp._recv(timeout=0.2)


def test_recv_gives_up_on_an_endless_stream_of_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fabric_rlm.interpreter._MAX_SKIPPED_PROTOCOL_LINES", 5)
    interp = Interpreter(timeout=5)
    interp._stdout_queue = queue.Queue()
    for _ in range(10):
        interp._stdout_queue.put("noise\n")
    with pytest.raises(WorkerProtocolError, match="non-protocol lines"):
        interp._recv()
