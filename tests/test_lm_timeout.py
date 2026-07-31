"""A dead connection must raise within the request timeout, not hang forever.

Regression for the batch-run incident where five concurrent workers froze for
35+ minutes with zero CPU and zero spend: each was blocked inside an LM HTTP
call whose connection the provider had dropped. The task-level timeout never
fired because it is checked between turns, and a blocked read never returns to
let it run. The fix gives every resolved LM a default per-request timeout so
the call errors and the normal retry/turn machinery takes over.
"""
import http.server
import socketserver
import threading
import time

import pytest

from fabric_rlm.lm import resolve_lm


class _BlackHole(http.server.BaseHTTPRequestHandler):
    """Accepts the request and never responds."""

    def do_POST(self):  # noqa: N802
        time.sleep(3600)

    def log_message(self, *a):  # silence
        pass


class _Server(socketserver.ThreadingTCPServer):
    # Handler threads sleep for an hour by design; they must be daemons or the
    # test process cannot exit while one is mid-black-hole.
    daemon_threads = True
    allow_reuse_address = True


@pytest.fixture()
def black_hole_server():
    srv = _Server(("127.0.0.1", 0), _BlackHole)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_dict_spec_gets_default_timeout():
    lm = resolve_lm({"model": "openai/gpt-4.1-mini", "api_key": "sk-test"})
    assert lm.kwargs.get("timeout"), "dict spec must carry a default request timeout"


def test_string_spec_gets_default_timeout():
    lm = resolve_lm("openai/gpt-4.1-mini")
    assert lm.kwargs.get("timeout"), "string spec must carry a default request timeout"


def test_caller_override_wins():
    lm = resolve_lm({"model": "openai/gpt-4.1-mini", "api_key": "sk-test", "timeout": 42})
    assert lm.kwargs["timeout"] == 42


def test_blocked_call_raises_within_bound(black_hole_server):
    lm = resolve_lm({
        "model": "openai/gpt-4.1-mini",
        "api_key": "sk-test",
        "api_base": black_hole_server,
        "timeout": 4,
        "max_retries": 0,
        "num_retries": 0,
    })
    t0 = time.time()
    with pytest.raises(Exception):
        lm(messages=[{"role": "user", "content": "hi"}])
    elapsed = time.time() - t0
    assert elapsed < 30, f"blocked call took {elapsed:.0f}s; timeout did not bound it"
