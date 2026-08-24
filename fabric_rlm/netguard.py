"""Refuse outbound connections inside the worker process.

Off unless asked for: pass ``block_network=True`` to ``RLM``. Plenty of notebook
code legitimately calls an API from worker code, so this must never be the
default.

Why the source-level denylist is not enough. :class:`fabric_rlm.security.
SecurityPolicy` already rejects ``requests``, ``httpx``, ``aiohttp``,
``urllib.request.urlopen`` and ``socket.socket``, but it inspects the *model's
own code*. Anything that wraps the request slips through untouched, because the
model's code never names a denied symbol::

    from datasets import load_dataset      # no denied name appears
    load_dataset("ag_news")                # ...and it fetches anyway

    pd.read_csv("https://example.com/x.csv")

That is not hypothetical. On DataAgentBench, trials solved a classification task
by downloading the public dataset and reading its gold label column, which the
static check could not see. The block therefore has to sit at the socket layer,
at runtime, where it does not matter which library issued the call.

Why the guard is on *connect* and not on socket construction. Denying
``socket.socket``/``socketpair`` outright kills the worker before it serves a
single turn: asyncio's Windows ProactorEventLoop builds its self-pipe with
``socket.socketpair()``, and ``nest_asyncio`` creates a loop at import. That
pair is two already-connected loopback sockets used for local signalling and
cannot reach a remote host. So sockets may be created freely, and only
connections to non-loopback addresses are refused.

What this does not do:

* It does not guard ``_socket.socket`` used directly, because a C type's methods
  cannot be replaced from Python. Deliberate abuse of that path is out of scope;
  the goal is to stop a model reaching for a convenient download.
* It proves nothing about a C extension issuing raw syscalls. Only an OS-level
  control - a firewall rule on the worker binary, a container with no network,
  a network namespace - can support a claim that the process *cannot* reach the
  network. This makes such a claim much more credible, not proven.
* **It is not contamination prevention.** Blocking egress does nothing about
  data already on the machine, and redirecting ``HF_HOME`` does not fix that
  either - it only changes where the *library* looks by default. Measured on a
  real DataAgentBench trial: with the guard on and ``HF_HOME`` redirected, the
  model was refused at ``urlopen``, went looking for cache directories, then
  read ``os.path.expanduser("~/.cache/huggingface")`` by absolute path and
  loaded all 127,600 labelled rows from disk. It never touched the network.

  If a dataset must be unreachable, remove it from the filesystem. A network
  block cannot substitute for that, and a post-hoc audit of what the model
  actually ran is what catches it when prevention fails.
"""

from __future__ import annotations

import ipaddress

#: Set to ``"1"`` in the worker's environment to install the guard at startup.
ENV_FLAG = "FABRIC_RLM_BLOCK_NETWORK"

_MSG = (
    "network egress is blocked in this sandbox (block_network=True). Work with "
    "the data already available locally instead of downloading anything."
)


class NetworkEgressBlocked(OSError):
    """Raised instead of connecting to a non-loopback address."""


def is_loopback(address: object) -> bool:
    """True when *address* stays on this machine.

    Non-tuple addresses (AF_UNIX paths, bytes) never leave the host and pass. A
    hostname is refused rather than resolved: resolving it to decide would mean
    a DNS lookup for every connect, and "unknown" should fail closed.
    """
    if not isinstance(address, tuple) or not address:
        return True
    host = address[0]
    if host in ("", None):
        return True
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    if host in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


_installed = False


def install() -> bool:
    """Install the connect guard. Idempotent; returns True if it did work now."""
    global _installed
    if _installed:
        return False

    import socket

    real_socket = socket.socket
    real_create_connection = socket.create_connection

    def _check(address: object) -> None:
        if not is_loopback(address):
            raise NetworkEgressBlocked(f"{_MSG} (refused connect to {address!r})")

    class _GuardedSocket(real_socket):  # type: ignore[misc, valid-type]
        """An ordinary socket that refuses to connect off-machine."""

        def connect(self, address):
            _check(address)
            return super().connect(address)

        def connect_ex(self, address):
            _check(address)
            return super().connect_ex(address)

    def _guarded_create_connection(address, *args, **kwargs):
        _check(address)
        return real_create_connection(address, *args, **kwargs)

    socket.socket = _GuardedSocket
    socket.create_connection = _guarded_create_connection
    _installed = True
    return True
