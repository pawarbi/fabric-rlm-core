"""End-to-end check that the JSON-RPC subprocess path returns a clean error envelope.

Spawns ``python -m fabric_rlm._worker`` as the host would, drives one ``execute``
request whose user code raises, and asserts the JSON-RPC error envelope carries
only the user-facing exception ``type`` + ``message`` — with no asyncio /
nest_asyncio / _worker.py framework noise leaking into the wire payload.

This documents the BUG-LIB-1 contract for the subprocess code path. (The bloated
tracebacks observed in SSB came from the *in-process* ``_execute()`` path used
by the Fabric runtime, not from this subprocess path. Subprocess clients receive
the exception type + message and re-raise locally with their own traceback.)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_worker(code: str, timeout: float = 15.0) -> dict:
    """Drive the worker via JSON-RPC, return the single ``execute`` result dict."""
    request = {
        "jsonrpc": "2.0",
        "method": "execute",
        "params": {"code": code},
        "id": 1,
    }
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "fabric_rlm._worker"],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )
    # Worker writes one JSON line per response; pick the first parseable line.
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(envelope, dict) and envelope.get("id") == 1:
            return envelope
    raise AssertionError(
        f"No JSON-RPC response with id=1 found.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def _err(envelope: dict) -> dict:
    """Worker contract: user errors come back as JSON-RPC error envelopes."""
    assert "error" in envelope, f"Expected error envelope, got {envelope}"
    return envelope["error"]


@pytest.mark.timeout(30)
def test_subprocess_user_error_envelope_shape_is_clean():
    """The JSON-RPC ``execute`` error envelope must carry the exception class and
    message only — no asyncio / nest_asyncio / _worker.py noise leaking via the
    message field. (BUG-LIB-1: prior raw format_exc() risked bloated payloads
    if the worker ever included tracebacks here.)"""
    envelope = _run_worker("a = 1 / 0\n")
    err = _err(envelope)
    assert err["data"]["type"] == "ZeroDivisionError"
    assert "division by zero" in err["message"]
    # No framework frame names anywhere in the wire payload
    payload = json.dumps(err)
    assert "nest_asyncio" not in payload, f"Framework frame leaked:\n{payload}"
    assert "_run_code" not in payload, f"Framework frame leaked:\n{payload}"
    assert "run_until_complete" not in payload, f"Framework frame leaked:\n{payload}"


@pytest.mark.timeout(30)
def test_subprocess_assertion_message_preserved():
    envelope = _run_worker("assert False, 'mismatch in column B'\n")
    err = _err(envelope)
    assert err["data"]["type"] == "AssertionError"
    assert "mismatch in column B" in err["message"]
    payload = json.dumps(err)
    assert "nest_asyncio" not in payload
    assert "_worker.py" not in payload


@pytest.mark.timeout(30)
def test_subprocess_module_not_found_envelope_clean():
    envelope = _run_worker("import this_module_does_not_exist_xyz\n")
    err = _err(envelope)
    assert err["data"]["type"] == "ModuleNotFoundError"
    assert "this_module_does_not_exist_xyz" in err["message"]
    payload = json.dumps(err)
    assert "nest_asyncio" not in payload
    assert "asyncio" not in payload
