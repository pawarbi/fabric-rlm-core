"""Unit tests for the default-on security baseline (``fabric_rlm.security``).

Goals
-----
1. The default policy rejects destructive / network / dynamic-dispatch
   patterns that an adversarial-or-curious LM would use.
2. The default policy *accepts* every code shape we see in real saved
   trajectories (no regression in behaviour).
3. ``scrub_env`` removes secret-bearing keys, preserves runtime essentials,
   and is case-insensitive on both sides.
4. ``Interpreter`` and ``SubprocessPythonInterpreter`` honour the policy:
   ``Interpreter.execute`` returns a fake-failure ``ExecResult`` (so the
   RLM loop can recover via its normal error path), and
   ``SubprocessPythonInterpreter.execute`` raises ``CodeInterpreterError``.
5. ``on_violation="raise"`` re-raises ``SecurityViolation`` instead of
   returning a fake-failure.
6. ``RLM`` accepts a ``security=`` kwarg, defaults to ``SecurityPolicy.default()``,
   and threads the policy down to the spawned interpreter.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from fabric_rlm.security import SecurityPolicy, SecurityViolation


# ---------------------------------------------------------------------------
# 1. validate_code: must REJECT
# ---------------------------------------------------------------------------

_REJECT_CASES: list[tuple[str, str]] = [
    ("os.remove", "import os\nos.remove('/etc/passwd')"),
    ("os.unlink", "import os\nos.unlink('x')"),
    ("os.rmdir", "import os\nos.rmdir('x')"),
    ("shutil.rmtree", "import shutil\nshutil.rmtree('x')"),
    ("Path.unlink", "from pathlib import Path\nPath('x').unlink()"),
    ("Path.rmdir", "from pathlib import Path\nPath('x').rmdir()"),
    ("subprocess.run", "import subprocess\nsubprocess.run(['rm','-rf','/'])"),
    ("subprocess.check_output", "import subprocess\nsubprocess.check_output(['ls'])"),
    ("subprocess.Popen", "import subprocess\nsubprocess.Popen(['x'])"),
    ("os.system", "import os\nos.system('rm -rf /')"),
    ("os.popen", "import os\nos.popen('ls')"),
    ("os.execv", "import os\nos.execv('/bin/sh', ['sh'])"),
    ("socket connect", "import socket\ns = socket.socket()\ns.connect(('1.1.1.1', 53))"),
    ("urllib request", "import urllib.request\nurllib.request.urlopen('http://x')"),
    ("requests.get", "import requests\nrequests.get('http://x')"),
    ("httpx.get", "import httpx\nhttpx.get('http://x')"),
    # __import__("forbidden_module") is rejected via the targeted decoder
    # even though plain __import__() is no longer in forbidden_builtins.
    ("__import__ subprocess", "__import__('subprocess').run(['ls'])"),
    ("importlib", "import importlib\nimportlib.import_module('subprocess')"),
    # Module-alias bypass: import subprocess as sp -> sp.run
    ("module alias", "import subprocess as sp\nsp.run(['ls'])"),
    # from-alias bypass: from subprocess import run as r -> r(['ls'])
    ("from-alias", "from subprocess import run as r\nr(['ls'])"),
    # Value-alias bypass: rm = os.remove -> rm('x')
    ("value alias", "import os\nrm = os.remove\nrm('x')"),
    # getattr() dynamic dispatch bypass on the forbidden module
    ("getattr dispatch", "import os\ngetattr(os, 'remove')('/x')"),
    ("from subprocess import + call", "from subprocess import run\nrun(['ls'])"),
    ("os.replace dotted", "import os\nos.replace('a', 'b')"),
]

_RETIRED_NOTEBOOK_ALIAS = "ms" + "sparkutils"
_REJECT_CASES.extend(
    [
        (
            "retired notebook utility direct destructive call",
            f"from notebookutils import {_RETIRED_NOTEBOOK_ALIAS}\n"
            f"{_RETIRED_NOTEBOOK_ALIAS}.fs.rm('x', True)",
        ),
        (
            "retired notebook utility nested destructive call",
            "import notebookutils\n"
            f"notebookutils.{_RETIRED_NOTEBOOK_ALIAS}.fs.mv('x', 'y')",
        ),
    ]
)


@pytest.mark.parametrize("label,code", _REJECT_CASES, ids=[c[0] for c in _REJECT_CASES])
def test_validate_code_rejects(label: str, code: str) -> None:
    msg = SecurityPolicy.default().validate_code(code)
    assert msg, f"expected rejection for {label}, got allow"
    assert "SecurityPolicyViolation" in msg


# ---------------------------------------------------------------------------
# 2. validate_code: must ALLOW (must not regress benign code)
# ---------------------------------------------------------------------------

_ALLOW_CASES: list[tuple[str, str]] = [
    ("pandas read", "import pandas as pd\ndf = pd.read_excel('x.xlsx')\nprint(df.head())"),
    ("openpyxl", "from openpyxl import load_workbook\nwb = load_workbook('x.xlsx', data_only=False)"),
    ("json", "import json\nprint(json.dumps({'a': 1}))"),
    ("re", "import re\nprint(re.findall(r'\\d+', 'a1b2'))"),
    ("collections", "from collections import Counter\nprint(Counter('aaab'))"),
    ("duckdb", "import duckdb\ncon = duckdb.connect()\nprint(con.execute('SELECT 1').fetchall())"),
    ("pathlib read", "from pathlib import Path\nprint(Path('x.txt').read_text())"),
    ("getattr safe", "class C:\n    x = 1\nprint(getattr(C, 'x'))"),
    ("isinstance", "assert isinstance({'a': 1}, dict)"),
    ("verifier shape", (
        "import json as _j\n"
        "_p = _j.loads('{\"output\": {\"k\": 1}}')\n"
        "def verify(p):\n"
        "    assert isinstance(p, dict)\n"
        "    assert 'output' in p\n"
        "verify(_p)\n"
    )),
    ("os.path", "import os.path\nprint(os.path.exists('x'))"),
    ("write file (NOT blocked)", "open('x.txt', 'w').write('hi')"),  # intentional: v1 does not block writes
    # Real corpus patterns that MUST continue to work after the no-regression
    # check (these caused regressions during initial deployment and are
    # explicitly part of the default-policy contract).
    ("sandboxed eval", "expr='1+1'; print(eval(expr, {'__builtins__': None}, {'abs': abs}))"),
    ("re.compile", "import re\npat = re.compile(r'\\d+')\nprint(pat.findall('a1b2'))"),
    ("__import__ benign", "print(__import__('re').findall(r'\\d+', 'a1b2'))"),
    ("compile()/exec() bare", "src='x=1'\nexec(compile(src, '<s>', 'exec'))"),
]


@pytest.mark.parametrize("label,code", _ALLOW_CASES, ids=[c[0] for c in _ALLOW_CASES])
def test_validate_code_allows(label: str, code: str) -> None:
    msg = SecurityPolicy.default().validate_code(code)
    assert msg is None, f"unexpected rejection for {label}: {msg}"


def test_disabled_policy_allows_everything() -> None:
    pol = SecurityPolicy.disabled()
    # Even an obviously bad line is allowed
    assert pol.validate_code("import os; os.remove('/etc/passwd')") is None


def test_empty_code_allowed() -> None:
    assert SecurityPolicy.default().validate_code("") is None
    assert SecurityPolicy.default().validate_code("   \n\t") is None


def test_syntax_error_defers_to_worker() -> None:
    # Syntax errors should NOT be rejected pre-flight (the worker produces
    # the canonical traceback).
    assert SecurityPolicy.default().validate_code("def (") is None


# ---------------------------------------------------------------------------
# 3. scrub_env
# ---------------------------------------------------------------------------


def test_scrub_env_strips_connection_strings() -> None:
    """Connection-string env vars do not contain KEY/TOKEN/SECRET/etc. in their
    names but routinely carry credentials. The default policy must scrub them."""
    src = {
        "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=...",
        "SQLCONNSTR_DefaultConnection": "Server=...;Password=x",
        "CUSTOMCONNSTR_redis": "redis://:secret@host:6379",
        "DATABASE_URL": "postgres://user:pwd@host/db",
        "POSTGRES_URL": "postgres://u:p@h/db",
        "REDIS_URL": "redis://x:y@h:6379",
        "MY_DB_DSN": "host=x port=5432 password=y",
        "BLOB_SAS_URL": "https://x.blob.core.windows.net/?sv=...&sig=...",
        "AZURE_TENANT_ID": "00000000-0000-0000-0000-000000000000",
    }
    out = SecurityPolicy.default().scrub_env(src)
    for k in src:
        assert k not in out, f"{k} should be stripped (connection-string class)"



    src = {
        "OPENAI_API_KEY": "sk-x",
        "AZURE_OPENAI_KEY": "sk-x",
        "GITHUB_TOKEN": "ghp_x",
        "AWS_SECRET_ACCESS_KEY": "x",
        "MY_SECRET": "x",
        "MY_PASSWORD": "x",
        "MY_TOKEN": "x",
        "MY_CREDENTIAL": "x",
    }
    out = SecurityPolicy.default().scrub_env(src)
    for k in src:
        assert k not in out, f"{k} should be stripped"


def test_scrub_env_preserves_runtime() -> None:
    src = {"PATH": "/usr/bin", "PYTHONPATH": "/site", "TMP": "/tmp", "OPENAI_API_KEY": "sk"}
    out = SecurityPolicy.default().scrub_env(src)
    assert out["PATH"] == "/usr/bin"
    assert out["PYTHONPATH"] == "/site"
    assert out["TMP"] == "/tmp"
    assert "OPENAI_API_KEY" not in out


def test_scrub_env_case_insensitive() -> None:
    src = {"openai_api_key": "sk", "Path": "/x"}
    out = SecurityPolicy.default().scrub_env(src)
    assert "openai_api_key" not in out
    assert out["Path"] == "/x"


def test_scrub_env_disabled_passthrough() -> None:
    src = {"OPENAI_API_KEY": "sk"}
    assert SecurityPolicy.disabled().scrub_env(src) == src


# ---------------------------------------------------------------------------
# 4. Interpreter wiring (legacy + subprocess)
# ---------------------------------------------------------------------------


def test_legacy_interpreter_returns_fake_failure_on_violation() -> None:
    from fabric_rlm.interpreter import Interpreter

    interp = Interpreter(security=SecurityPolicy.default())
    result = interp.execute("import os\nos.remove('/etc/passwd')")
    assert result.ok is False
    assert result.submitted is False
    assert "SecurityPolicyViolation" in (result.error or "")
    # Worker must NEVER have been spawned — proves pre-flight gating works
    assert interp.proc is None


def test_legacy_interpreter_no_policy_unchanged() -> None:
    from fabric_rlm.interpreter import Interpreter

    interp = Interpreter()  # no policy
    assert interp.security is None


def test_subprocess_interpreter_raises_on_violation() -> None:
    from fabric_rlm.interpreter import SubprocessPythonInterpreter, _import_dspy_code_interpreter
    _, CodeInterpreterError = _import_dspy_code_interpreter()

    interp = SubprocessPythonInterpreter(security=SecurityPolicy.default())
    with pytest.raises(CodeInterpreterError) as exc_info:
        interp.execute("import subprocess\nsubprocess.run(['ls'])")
    assert "SecurityPolicyViolation" in str(exc_info.value)


def test_on_violation_raise_path() -> None:
    from fabric_rlm.interpreter import Interpreter

    pol = SecurityPolicy(on_violation="raise")
    interp = Interpreter(security=pol)
    with pytest.raises(SecurityViolation):
        interp.execute("import os\nos.remove('/x')")


# ---------------------------------------------------------------------------
# 5. RLM constructor accepts and defaults policy
# ---------------------------------------------------------------------------


def test_rlm_defaults_to_default_policy() -> None:
    from fabric_rlm.runtime import RLM
    import dspy

    lm = dspy.LM("openrouter/openai/gpt-4.1")
    rlm = RLM(lm=lm, skills=[])
    assert isinstance(rlm._security, SecurityPolicy)
    assert rlm._security.enabled is True


def test_rlm_accepts_disabled_policy() -> None:
    from fabric_rlm.runtime import RLM
    import dspy

    lm = dspy.LM("openrouter/openai/gpt-4.1")
    rlm = RLM(lm=lm, skills=[], security=SecurityPolicy.disabled())
    assert rlm._security.enabled is False


# ---------------------------------------------------------------------------
# 6. End-to-end: env-scrubbed worker actually doesn't see the secret
# ---------------------------------------------------------------------------


def test_env_scrub_propagates_to_real_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spawn a real worker with the default policy and verify a known-secret
    env var is NOT visible inside the child process."""
    from fabric_rlm.interpreter import Interpreter

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-be-stripped")
    monkeypatch.setenv("MY_BENIGN_VAR", "kept")  # not on any strip glob

    with Interpreter(timeout=30, security=SecurityPolicy.default()) as interp:
        result = interp.execute(
            "import os\n"
            "print('OPENAI=', os.environ.get('OPENAI_API_KEY', '<missing>'))\n"
            "print('BENIGN=', os.environ.get('MY_BENIGN_VAR', '<missing>'))\n"
        )
    assert result.ok, f"expected success, got {result.error}"
    out = result.stdout
    assert "OPENAI= <missing>" in out, f"OPENAI_API_KEY leaked into worker: {out}"
    assert "BENIGN= kept" in out, f"benign var unexpectedly stripped: {out}"


def test_env_scrub_disabled_lets_secret_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity check: with policy disabled the worker DOES see the env var
    (proves the strip is what removes it, not some other layer)."""
    from fabric_rlm.interpreter import Interpreter

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-leak-ok")

    with Interpreter(timeout=30, security=SecurityPolicy.disabled()) as interp:
        result = interp.execute(
            "import os\nprint('OPENAI=', os.environ.get('OPENAI_API_KEY', '<missing>'))\n"
        )
    assert result.ok
    assert "OPENAI= sk-test-leak-ok" in result.stdout


@pytest.mark.parametrize("block_network", [False, True])
def test_interpreter_scrubs_secrets_by_default(
    monkeypatch: pytest.MonkeyPatch,
    block_network: bool,
) -> None:
    from fabric_rlm.interpreter import Interpreter

    monkeypatch.setenv("OPENAI_API_KEY", "sk-provider-secret")
    monkeypatch.setenv("PROBE_FAKE_KEY", "sk-probe-secret")
    monkeypatch.setenv("MY_BENIGN_VAR", "kept")

    with Interpreter(timeout=30, block_network=block_network) as interp:
        result = interp.execute(
            "import os\n"
            "print('OPENAI=', os.environ.get('OPENAI_API_KEY', '<missing>'))\n"
            "print('PROBE=', os.environ.get('PROBE_FAKE_KEY', '<missing>'))\n"
            "print('BENIGN=', os.environ.get('MY_BENIGN_VAR', '<missing>'))\n"
        )

    assert result.ok, result.error
    assert "OPENAI= <missing>" in result.stdout
    assert "PROBE= <missing>" in result.stdout
    assert "BENIGN= kept" in result.stdout


def test_subprocess_interpreter_scrubs_secrets_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fabric_rlm.interpreter import SubprocessPythonInterpreter

    monkeypatch.setenv("OPENAI_API_KEY", "sk-provider-secret")
    monkeypatch.setenv("PROBE_FAKE_KEY", "sk-probe-secret")
    monkeypatch.setenv("MY_BENIGN_VAR", "kept")

    with SubprocessPythonInterpreter(timeout=30) as interp:
        result = interp.execute(
            "import os\n"
            "print('OPENAI=', os.environ.get('OPENAI_API_KEY', '<missing>'))\n"
            "print('PROBE=', os.environ.get('PROBE_FAKE_KEY', '<missing>'))\n"
            "print('BENIGN=', os.environ.get('MY_BENIGN_VAR', '<missing>'))\n"
        )

    assert "OPENAI= <missing>" in result
    assert "PROBE= <missing>" in result
    assert "BENIGN= kept" in result
