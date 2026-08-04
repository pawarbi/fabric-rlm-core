"""block_network: the worker may not connect off-machine when asked.

The interesting cases are the ones that bit while building this by hand against
DataAgentBench, so they are all here as tests rather than as folklore:

* denying socket construction outright kills the worker before turn one, because
  asyncio's Windows event loop builds its self-pipe with socket.socketpair()
* the leak vector is a *library* call, so a source-level denylist never sees it
* blocking the network still leaves a populated on-disk cache readable
"""

from __future__ import annotations

import os

import pytest

from fabric_rlm import RLM
from fabric_rlm.interpreter import Interpreter
from fabric_rlm.security import SecurityPolicy
from fabric_rlm.netguard import ENV_FLAG, NetworkEgressBlocked, install, is_loopback


class ScriptedLM:
    def __init__(self, turns):
        self.turns = list(turns)
        self.i = 0

    def __call__(self, messages=None, prompt=None, **kwargs):
        if self.i >= len(self.turns):
            return ["SUBMIT(answer='done')"]
        t = self.turns[self.i]
        self.i += 1
        return [t]


def code(body: str) -> str:
    return f"```python\n{body}\n```"


# --- the address predicate ---------------------------------------------------

@pytest.mark.parametrize("addr", [
    ("127.0.0.1", 8080),
    ("127.5.4.3", 1),
    ("::1", 443),
    ("localhost", 80),
    "/tmp/some.sock",          # AF_UNIX never leaves the box
    b"/tmp/other.sock",
    ("", 0),
])
def test_local_addresses_allowed(addr):
    assert is_loopback(addr) is True


@pytest.mark.parametrize("addr", [
    ("huggingface.co", 443),
    ("8.8.8.8", 53),
    ("192.168.1.10", 22),      # LAN is still off-machine
    ("2001:4860:4860::8888", 443),
])
def test_remote_addresses_refused(addr):
    assert is_loopback(addr) is False


def test_hostname_fails_closed():
    """An unresolvable name must be refused, not resolved to decide."""
    assert is_loopback(("not-a-real-host.invalid", 443)) is False


# --- in-process behaviour ----------------------------------------------------

def test_install_is_idempotent():
    first = install()
    second = install()
    assert second is False, "second install should be a no-op"
    assert first in (True, False)


# --- the worker --------------------------------------------------------------

def _run(body: str, **kwargs):
    """Run one turn of *body* in the worker.

    security is disabled on purpose. SecurityPolicy's static denylist already
    rejects socket.create_connection and urlopen at the source level, so with it
    on these tests would pass without netguard doing anything at all - they
    would be testing the wrong layer. The point of netguard is exactly the case
    the static check cannot see, so it has to be measured on its own.
    """
    lm = ScriptedLM([code(body), code("SUBMIT(answer='done')")])
    return RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=4,
                    timeout=90, security=SecurityPolicy.disabled(), **kwargs).run()


def test_worker_starts_and_works_with_block_network():
    """The regression that mattered: denying sockets outright killed startup.

    asyncio's ProactorEventLoop calls socket.socketpair() for its self-pipe and
    nest_asyncio builds a loop at import, so a blanket deny meant every task
    died in about a second having done nothing.
    """
    r = _run("print(1 + 1)", block_network=True)
    assert r.submitted is True
    assert r.payload["answer"] == "done"


def test_loopback_still_allowed_in_worker():
    body = (
        "import socket\n"
        "s = socket.socket()\n"
        "s.bind(('127.0.0.1', 0))\n"
        "s.listen(1)\n"
        "port = s.getsockname()[1]\n"
        "c = socket.create_connection(('127.0.0.1', port), timeout=5)\n"
        "print('loopback ok')\n"
        "c.close(); s.close()\n"
    )
    r = _run(body, block_network=True)
    turn = r.trajectory.turns[0]
    assert "loopback ok" in (turn.stdout or ""), turn.error


def test_remote_connect_refused_in_worker():
    body = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('93.184.216.34', 80), timeout=5)\n"
        "    print('NOT BLOCKED')\n"
        "except OSError as e:\n"
        "    print('blocked:', 'egress is blocked' in str(e))\n"
    )
    r = _run(body, block_network=True)
    out = r.trajectory.turns[0].stdout or ""
    assert "blocked: True" in out, out


def test_library_wrapped_fetch_is_refused():
    """The actual DAB vector: no denied symbol appears in the model's code."""
    body = (
        "import urllib.request as u\n"
        "try:\n"
        "    u.urlopen('https://huggingface.co', timeout=5)\n"
        "    print('NOT BLOCKED')\n"
        "except Exception as e:\n"
        "    print('blocked:', 'egress is blocked' in str(e))\n"
    )
    r = _run(body, block_network=True)
    out = r.trajectory.turns[0].stdout or ""
    assert "blocked: True" in out, out


def test_off_by_default():
    """No guard, and no flag in the child env, unless asked for."""
    body = (
        "import os\n"
        f"print('flag=', os.environ.get({ENV_FLAG!r}))\n"
    )
    r = _run(body)
    out = r.trajectory.turns[0].stdout or ""
    assert "flag= None" in out, out


def test_interpreter_sets_the_flag_only_when_asked():
    assert Interpreter().block_network is False
    assert Interpreter(block_network=True).block_network is True


def test_explicit_sub_lm_is_rejected():
    with pytest.raises(ValueError, match="block_network=True cannot be combined"):
        RLM.task(task="t", outputs=["a"], lm=ScriptedLM([]),
                 block_network=True, sub_lm={"model": "x"})


def test_lm_spec_alone_does_not_trip_the_guard():
    """sub_lm_spec is set implicitly from lm; that must stay usable."""
    rlm = RLM.task(task="t", outputs=["a"], lm={"model": "openai/gpt-4o-mini"},
                   block_network=True)
    assert rlm.block_network is True


def test_a_cache_still_reads_with_the_network_blocked(tmp_path):
    """Stated so nobody mistakes this for data isolation.

    The DAB rerun would have leaked from ~/.cache/huggingface with every socket
    refused, because the dataset was already on disk from the run that leaked.
    """
    cached = tmp_path / "already_here.csv"
    cached.write_text("label\nWorld\n", encoding="utf-8")
    body = (
        "from pathlib import Path\n"
        f"print('read:', Path(r'{cached}').read_text().strip().splitlines()[-1])\n"
    )
    r = _run(body, block_network=True)
    out = r.trajectory.turns[0].stdout or ""
    assert "read: World" in out, out
