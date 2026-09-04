"""Parent-side wrappers for the persistent fabric-rlm worker subprocess.

Two wrappers live here:

- ``Interpreter`` (legacy ``{op: ...}`` envelope, returns ``ExecResult``) — the
  v6.5 surface kept for back-compat with notebooks and tests that already use it.

- ``SubprocessPythonInterpreter`` (JSON-RPC 2.0, satisfies the dspy
  ``CodeInterpreter`` Protocol) — the v7 surface used to delegate the RLM loop
  to ``dspy.predict.RLM`` while keeping our CPython subprocess as the code
  execution backend.

Both wrappers spawn the same ``python -m fabric_rlm._worker`` process; the
worker dispatches per-message based on envelope shape (see ``_worker.main``).
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from . import netguard
from .artifacts import FileDestination, encode_for_worker, publish_file
from .lakehouse import LakehouseSource, execute_lakehouse_query
from .security import SecurityPolicy
from .serializers import DEFAULT_MAX_SUBMIT_BYTES, validate_max_submit_bytes


class WorkerTimeout(TimeoutError):
    """Raised when the worker does not respond within the configured timeout."""


_CONTROL_PLANE_TIMEOUT_FLOOR_SECONDS = 10.0


_CONCURRENCY_MARKERS = ("ThreadPoolExecutor", "ProcessPoolExecutor",
                        "import threading", "threading.Thread",
                        "import multiprocessing", "multiprocessing.",
                        "asyncio.gather", "asyncio.run")


def _worker_env(
    security: SecurityPolicy | None,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a child environment that is secret-safe unless explicitly disabled."""

    policy = security if security is not None else SecurityPolicy.default()
    return policy.scrub_env(env)


def concurrency_death_hint(code) -> str:
    """Why did the worker die? If the code was threading, say so.

    A worker hard-crash (GIL fatal, native segfault) costs the whole solve
    and the model never learns why: it sees a bare protocol error and its
    retry repeats the pattern. Observed twice in one benchmark run - both
    times ThreadPoolExecutor around native calls. Naming the cause turns a
    fatal into a repairable turn.
    """
    if not code:
        return ""
    hits = sorted({m for m in _CONCURRENCY_MARKERS if m in code})
    if not hits:
        return ""
    return (" NOTE: this code used " + ", ".join(hits) + ". The worker is "
            "not thread-safe for native calls (duckdb, database drivers, "
            "predict_sync) and concurrency can crash it fatally, losing all "
            "session state. Rewrite the work as a plain serial loop.")


class WorkerProtocolError(RuntimeError):
    """Raised when the worker exits or returns invalid protocol data."""


_LAKEHOUSE_QUERY_TOOL = "__fabric_rlm_lakehouse_query__"
_FILE_PUBLISH_TOOL = "__fabric_rlm_file_publish__"


def _collect_lakehouse_sources(value: Any) -> list[LakehouseSource]:
    if isinstance(value, LakehouseSource):
        return [value]
    if isinstance(value, dict):
        return [
            source
            for item in value.values()
            for source in _collect_lakehouse_sources(item)
        ]
    if isinstance(value, (list, tuple)):
        return [
            source
            for item in value
            for source in _collect_lakehouse_sources(item)
        ]
    return []


def _collect_file_destinations(value: Any) -> list[FileDestination]:
    if isinstance(value, FileDestination):
        return [value]
    if isinstance(value, dict):
        return [
            destination
            for item in value.values()
            for destination in _collect_file_destinations(item)
        ]
    if isinstance(value, (list, tuple)):
        return [
            destination
            for item in value
            for destination in _collect_file_destinations(item)
        ]
    return []


def _execute_bound_lakehouse_query(
    bound_sources: list[LakehouseSource],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    root = kwargs.get("root")
    catalog = kwargs.get("catalog")
    source = next(
        (
            candidate
            for candidate in bound_sources
            if candidate.root == root and list(candidate.catalog or ()) == catalog
        ),
        None,
    )
    if source is None:
        raise PermissionError(
            "LakehouseSource is not bound to this worker or its catalog was modified."
        )
    return execute_lakehouse_query(
        source,
        sql=kwargs.get("sql", ""),
        sources=kwargs.get("sources", {}),
        max_rows=kwargs.get("max_rows", 1_000),
    )


def _execute_bound_file_publish(
    bound_destinations: list[FileDestination],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    root = kwargs.get("root")
    staging_root = kwargs.get("staging_root")
    staging_identity = kwargs.get("staging_identity")
    max_bytes = kwargs.get("max_bytes")
    destination = next(
        (
            candidate
            for candidate in bound_destinations
            if candidate.root == root
            and candidate.staging_root == staging_root
            and list(candidate._staging_identity) == staging_identity
            and candidate.max_bytes == max_bytes
        ),
        None,
    )
    if destination is None:
        raise PermissionError(
            "FileDestination is not bound to this worker or was modified."
        )
    return publish_file(
        destination,
        local_path=kwargs.get("local_path", ""),
        relative_path=kwargs.get("relative_path", ""),
        overwrite=kwargs.get("overwrite", False),
    )


def _close_worker_resources(
    proc: subprocess.Popen[str],
    threads: tuple[threading.Thread | None, ...],
) -> None:
    """Close Popen pipes after worker exit and let pump threads finish."""

    if proc.stdin is not None and not proc.stdin.closed:
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass
    for thread in threads:
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)
    for stream, thread in zip((proc.stdout, proc.stderr), threads):
        # TextIOWrapper.close() can block on the read lock held by a live
        # pump thread. A normally exited worker produces EOF and lets the
        # thread finish before this point; leave abnormal live-thread cleanup
        # to process teardown rather than deadlocking shutdown.
        if stream is not None and not stream.closed and not (thread and thread.is_alive()):
            try:
                stream.close()
            except (OSError, ValueError):
                pass


@dataclass
class ExecResult:
    ok: bool
    submitted: bool
    stdout: str
    stderr: str
    state: dict[str, Any]
    error: str | None = None
    submit_payload: dict[str, Any] | None = None
    # False when the turn never reached the worker (e.g. rejected by the
    # parent-side SecurityPolicy). Such results carry an empty ``state``
    # snapshot that does NOT reflect the worker's real namespace — callers
    # tracking state across turns must not overwrite their last good
    # snapshot with it.
    reached_worker: bool = True
    # Typed records of the source calls the turn made (worker-side semantic
    # model telemetry plus parent-side Lakehouse queries). ``None`` when the
    # turn reported none.
    source_calls: list[dict[str, Any]] | None = None

    @classmethod
    def from_response(cls, raw: dict[str, Any]) -> "ExecResult":
        reported = raw.get("source_calls")
        return cls(
            ok=bool(raw.get("ok")),
            submitted=bool(raw.get("submitted", False)),
            stdout=str(raw.get("stdout", "")),
            stderr=str(raw.get("stderr", "")),
            state=dict(raw.get("state", {})),
            error=raw.get("error"),
            submit_payload=raw.get("submit_payload"),
            source_calls=(
                [dict(item) for item in reported if isinstance(item, dict)]
                if isinstance(reported, list)
                else None
            ),
        )


class Interpreter:
    """Persistent Python subprocess with JSON-line protocol."""

    def __init__(
        self,
        timeout: float = 300.0,
        python: str | None = None,
        cwd: str | None = None,
        security: SecurityPolicy | None = None,
        max_submit_bytes: int = DEFAULT_MAX_SUBMIT_BYTES,
        block_network: bool = False,
    ):
        self.timeout = timeout
        self.python = python or sys.executable
        self.cwd = cwd
        self.block_network = bool(block_network)
        # ``security`` is opt-in at this layer (None = no enforcement). The
        # public RLM facade is responsible for passing a default policy in
        # for LM-facing executions; verifier code paths intentionally leave
        # this None because verifier code is trusted, internal, and may need
        # APIs the policy would otherwise reject.
        self.security: SecurityPolicy | None = security
        self.max_submit_bytes = validate_max_submit_bytes(max_submit_bytes)
        self.proc: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_buf: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._lakehouse_sources: list[LakehouseSource] = []
        self._file_destinations: list[FileDestination] = []

    @property
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> "Interpreter":
        if self.is_running:
            raise RuntimeError("Worker is already running")

        kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "cwd": self.cwd,
            "env": _worker_env(self.security),
        }
        if self.block_network:
            # The worker installs the guard itself when it sees this.
            env = kwargs["env"]
            env[netguard.ENV_FLAG] = "1"
        if os.name == "nt":
            # CREATE_NO_WINDOW: without it, every worker spawned from a
            # detached parent opens its own console window - a long benchmark
            # run put dozens of empty terminals on the user's desktop.
            kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                       | subprocess.CREATE_NO_WINDOW)
        else:
            kwargs["start_new_session"] = True

        self._stdout_queue = queue.Queue()
        self._stderr_buf = []
        self.proc = subprocess.Popen(
            [self.python, "-u", "-m", "fabric_rlm._worker"],
            **kwargs,
        )
        self._stdout_thread = threading.Thread(target=self._pump_stdout, daemon=True)
        self._stdout_thread.start()
        # Drain stderr continuously. Without this, anything the worker writes
        # to its real stderr outside the redirected execute window (import-time
        # warnings from dspy/litellm, native-library chatter from pymupdf or
        # duckdb) accumulates in the OS pipe buffer; once full (~64KB) the
        # worker blocks on the write and the parent reports a spurious
        # WorkerTimeout. Ring-buffered to the last 200 lines for diagnostics.
        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_thread.start()
        return self

    def execute(self, code: str) -> ExecResult:
        # Apply the security policy parent-side. On rejection, fabricate an
        # ExecResult that looks like a failed turn so the RLM loop's existing
        # error-recovery path kicks in (model sees stderr, retries with a
        # different approach). on_violation="raise" surfaces the violation
        # as an exception instead — used by tests and strict callers.
        if self.security is not None and self.security.enabled:
            violation = self.security.validate_code(code)
            if violation is not None:
                if self.security.on_violation == "raise":
                    from .security import SecurityViolation

                    raise SecurityViolation(violation)
                # Default: feedback to the loop. ``reached_worker=False``
                # tells the caller this empty ``state`` is a placeholder, not
                # a real namespace snapshot — the worker was never consulted.
                return ExecResult(
                    ok=False,
                    submitted=False,
                    stdout="",
                    stderr=violation,
                    state={},
                    error=violation,
                    reached_worker=False,
                )
        self._last_exec_code = code
        return self._execute_code(code, timeout=self.timeout)

    def warmup(self) -> None:
        """Exercise the replacement worker's exec path before user code."""

        result = self._execute_code(
            "",
            timeout=max(self.timeout, _CONTROL_PLANE_TIMEOUT_FLOOR_SECONDS),
        )
        if not result.ok:
            raise WorkerProtocolError(
                "replacement worker warmup failed: "
                + (result.error or result.stderr or "unknown worker error")
            )

    def _execute_code(self, code: str, *, timeout: float) -> ExecResult:
        self._send(
            {"op": "exec", "code": code, "max_submit_bytes": self.max_submit_bytes}
        )
        self._pending_source_calls: list[dict[str, Any]] = []
        while True:
            raw = self._recv(timeout=timeout)
            if raw.get("method") == "tool_call":
                self._handle_internal_tool_call(raw)
                continue
            result = ExecResult.from_response(raw)
            # Parent-side calls (Lakehouse SQL) join the worker-reported ones
            # so a turn's source calls are one list wherever they ran.
            pending = self._pending_source_calls
            self._pending_source_calls = []
            if pending:
                result.source_calls = list(result.source_calls or []) + pending
            return result

    def configure_lm(self, spec: Any) -> dict[str, Any]:
        return self._request({"op": "configure_lm", "spec": encode_for_worker(spec)})

    def set_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self._lakehouse_sources = _collect_lakehouse_sources(inputs)
        self._file_destinations = _collect_file_destinations(inputs)
        encoded = {name: encode_for_worker(value) for name, value in inputs.items()}
        return self._request({"op": "set_inputs", "inputs": encoded})

    def _timed_lakehouse_query(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Run a bound Lakehouse query and record its typed telemetry.

        The record names the source by its root and carries the SQL only as
        a fingerprint and a length: enough to recognise a repeated query
        shape, nothing that could carry data or a path into durable
        knowledge.
        """
        from hashlib import sha256

        sql = str(kwargs.get("sql", ""))
        started = time.monotonic()
        record: dict[str, Any] = {
            "query_type": "lakehouse_sql",
            "source_root": str(kwargs.get("root", "")),
            "query_fingerprint": sha256(sql.encode("utf-8")).hexdigest()[:16],
            "query_chars": len(sql),
            "source_count": len(kwargs.get("sources", {}) or {}),
            "max_rows": kwargs.get("max_rows", 1_000),
            "executed": True,
        }
        try:
            result = _execute_bound_lakehouse_query(self._lakehouse_sources, kwargs)
        except Exception as exc:
            record.update(
                execution_seconds=round(time.monotonic() - started, 3),
                reason="execution_error",
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
            self._pending_source_calls.append(record)
            raise
        rows = result.get("rows") if isinstance(result, dict) else None
        record.update(
            execution_seconds=round(time.monotonic() - started, 3),
            returned_rows=len(rows) if isinstance(rows, list) else None,
            truncated=bool(result.get("truncated")) if isinstance(result, dict) else None,
            total_seconds=round(time.monotonic() - started, 3),
        )
        self._pending_source_calls.append(record)
        return result

    def _handle_internal_tool_call(self, request: dict[str, Any]) -> None:
        request_id = request.get("id")
        params = request.get("params", {}) or {}
        name = params.get("name")
        kwargs = params.get("kwargs", {}) or {}
        try:
            if name == _LAKEHOUSE_QUERY_TOOL:
                result = self._timed_lakehouse_query(kwargs)
            elif name == _FILE_PUBLISH_TOOL:
                result = _execute_bound_file_publish(
                    self._file_destinations,
                    kwargs,
                )
            else:
                raise WorkerProtocolError(f"Unknown internal worker tool: {name}")
            response = {
                "jsonrpc": "2.0",
                "result": {
                    "value": json.dumps(result, ensure_ascii=False),
                    "type": "json",
                },
                "id": request_id,
            }
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": _JSONRPC_APP_ERRORS.get(
                        type(exc).__name__,
                        _JSONRPC_APP_ERRORS["Unknown"],
                    ),
                    "message": str(exc),
                    "data": {"type": type(exc).__name__},
                },
                "id": request_id,
            }
        self._send(response)

    def reset(self) -> dict[str, Any]:
        return self._request({"op": "reset"})

    def ping(self) -> dict[str, Any]:
        return self._request({"op": "ping"})

    def shutdown(self) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                self._send({"op": "shutdown"})
                proc.wait(timeout=5)
        except Exception:
            try:
                if proc.poll() is None:
                    if os.name == "nt":
                        proc.kill()
                    else:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=5)
            except Exception:
                pass
        finally:
            _close_worker_resources(proc, (self._stdout_thread, self._stderr_thread))
            if self.proc is proc:
                self.proc = None

    def kill(self) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
        finally:
            _close_worker_resources(proc, (self._stdout_thread, self._stderr_thread))
            if self.proc is proc:
                self.proc = None

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        self._send(message)
        return self._recv(
            timeout=max(self.timeout, _CONTROL_PLANE_TIMEOUT_FLOOR_SECONDS)
        )

    def _send(self, message: dict[str, Any]) -> None:
        if not self.is_running or self.proc is None or self.proc.stdin is None:
            raise WorkerProtocolError("Worker is not running")
        try:
            self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise WorkerProtocolError(self._format_worker_exit("Worker pipe closed")) from exc

    def _recv(self, *, timeout: float | None = None) -> dict[str, Any]:
        active_timeout = self.timeout if timeout is None else timeout
        try:
            line = self._stdout_queue.get(timeout=active_timeout)
        except queue.Empty as exc:
            self.kill()
            raise WorkerTimeout(
                f"Worker timed out after {active_timeout}s"
            ) from exc

        if line is None:
            raise WorkerProtocolError(
                self._format_worker_exit("Worker exited without response")
                + concurrency_death_hint(getattr(self, "_last_exec_code", None)))

        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkerProtocolError(f"Invalid worker JSON response: {line!r}") from exc

    def _pump_stdout(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            self._stdout_queue.put(line)
        self._stdout_queue.put(None)

    def _pump_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self._stderr_buf.append(line)
            if len(self._stderr_buf) > 200:
                self._stderr_buf.pop(0)

    def _format_worker_exit(self, prefix: str) -> str:
        # Read from the ring buffer maintained by _pump_stderr — the pipe is
        # already being drained by that thread, so a direct .read() here would
        # race it (and could block if the thread hadn't started).
        stderr = "".join(self._stderr_buf).strip()
        return f"{prefix}. Worker stderr:\n{stderr}".rstrip()

    def __enter__(self) -> "Interpreter":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.shutdown()


# =====================================================================
# v7 SubprocessPythonInterpreter — satisfies the dspy CodeInterpreter
# Protocol so that ``dspy.predict.RLM`` can drive our worker.
# =====================================================================
#
# Source contracts:
# - dspy.primitives.code_interpreter.CodeInterpreter (lines 39-148)
# - dspy.primitives.python_interpreter.PythonInterpreter.execute (lines 484-563)
# - dspy.primitives.python_interpreter.PythonInterpreter._handle_tool_call (lines 264-315)
#
# This class deliberately mirrors the wire protocol of dspy's Deno-backed
# PythonInterpreter (JSON-RPC 2.0) so that v7 introduces zero new behaviour
# in the dspy.RLM loop — only a different execution backend (CPython
# subprocess instead of Deno+Pyodide).


def _import_dspy_code_interpreter() -> tuple[type, type]:
    from dspy.primitives.code_interpreter import (
        CodeInterpreterError as _CIE,
        FinalOutput as _FO,
    )
    return _FO, _CIE


# Mirrors dspy.primitives.python_interpreter.JSONRPC_APP_ERRORS so we map
# error codes back to types correctly in execute().
_JSONRPC_APP_ERRORS = {
    "SyntaxError": -32000,
    "NameError": -32001,
    "TypeError": -32002,
    "ValueError": -32003,
    "AttributeError": -32004,
    "IndexError": -32005,
    "KeyError": -32006,
    "RuntimeError": -32007,
    "CodeInterpreterError": -32008,
    "Unknown": -32099,
}


class SubprocessPythonInterpreter:
    """CPython-subprocess interpreter satisfying the dspy CodeInterpreter Protocol.

    Spawns ``python -m fabric_rlm._worker`` with explicit ``PYTHONPATH`` and
    ``sys.executable`` (the v7 fix for the Fabric ``No module named
    fabric_rlm._worker`` crash class).

    Lifecycle (per dspy CodeInterpreter Protocol):
        1. ``__init__`` only configures; no subprocess spawned.
        2. ``start()`` spawns and runs the startup self-test.
        3. ``execute(code, variables=None)`` lazily starts and lazily registers
           tools; returns ``FinalOutput | str | None``.
        4. ``shutdown()`` is idempotent.

    Tools dict is mutable so ``dspy.predict.RLM`` can mutate it after construction
    via ``interpreter.tools.update(...)`` (see dspy/predict/rlm.py).
    """

    def __init__(
        self,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        timeout: float = 300.0,
        start_timeout: float | None = None,
        python: str | None = None,
        cwd: str | None = None,
        security: SecurityPolicy | None = None,
        max_submit_bytes: int = DEFAULT_MAX_SUBMIT_BYTES,
    ) -> None:
        self.tools: dict[str, Callable[..., Any]] = dict(tools) if tools else {}
        self.output_fields = output_fields
        self.timeout = timeout
        # Startup self-test timeout. Defaults generously (60s) because a cold
        # CPython spawn on a loaded machine (AV scan, CI runner contention)
        # can legitimately take tens of seconds; a genuinely broken install
        # still fails fast because the dead worker closes stdout immediately
        # ("Worker exited unexpectedly"), without waiting out this timeout.
        # Override per-instance via start_timeout= or globally via the
        # FABRIC_RLM_START_TIMEOUT environment variable.
        if start_timeout is None:
            try:
                start_timeout = float(os.environ.get("FABRIC_RLM_START_TIMEOUT", "60"))
            except ValueError:
                start_timeout = 60.0
        self.start_timeout = start_timeout
        self.python = python or sys.executable
        self.cwd = cwd
        # See ``Interpreter.__init__`` — opt-in at this layer; ``RLM`` is the
        # public surface that defaults this on for LM-facing code.
        self.security: SecurityPolicy | None = security
        self.max_submit_bytes = validate_max_submit_bytes(max_submit_bytes)

        self._proc: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_buf: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._tools_registered = False
        self._request_id = 0
        self._lakehouse_sources: list[LakehouseSource] = []
        self._file_destinations: list[FileDestination] = []

        # Diagnostics populated by start():
        self._spawn_cmd: list[str] | None = None
        self._spawn_env: dict[str, str] | None = None
        self._worker_self_test: dict[str, Any] | None = None

        # Test hooks (used by Prove-It regression tests):
        self._extra_python_args: list[str] | None = None
        self._force_pythonpath: str | None = None

    # ----- Lifecycle ---------------------------------------------------------

    def _compute_pythonpath(self) -> str:
        """Build the PYTHONPATH that lets the worker import ``fabric_rlm``.

        The fix for the Fabric crash class: spawn with an env that explicitly
        includes the parent dir of the imported ``fabric_rlm`` package, so the
        subprocess does not depend on inherited sys.path from a possibly
        isolated lakehouse-deps directory.
        """
        if self._force_pythonpath is not None:
            return self._force_pythonpath
        import fabric_rlm

        parent = os.path.dirname(os.path.dirname(os.path.abspath(fabric_rlm.__file__)))
        existing = os.environ.get("PYTHONPATH", "")
        parts = [parent] + ([existing] if existing else [])
        return os.pathsep.join(parts)

    def start(self) -> None:
        """Spawn the worker (idempotent) and run the startup self-test."""
        if self._proc is not None and self._proc.poll() is None:
            return

        FinalOutput, CodeInterpreterError = _import_dspy_code_interpreter()

        cmd: list[str] = [self.python, "-u"]
        if self._extra_python_args:
            cmd.extend(self._extra_python_args)
        cmd.extend(["-m", "fabric_rlm._worker"])

        env = _worker_env(
            self.security,
            {**os.environ, "PYTHONPATH": self._compute_pythonpath()},
        )

        kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "cwd": self.cwd,
            "env": env,
        }
        if os.name == "nt":
            # CREATE_NO_WINDOW: without it, every worker spawned from a
            # detached parent opens its own console window - a long benchmark
            # run put dozens of empty terminals on the user's desktop.
            kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                       | subprocess.CREATE_NO_WINDOW)
        else:
            kwargs["start_new_session"] = True

        self._spawn_cmd = cmd
        self._spawn_env = env
        self._stdout_queue = queue.Queue()
        self._stderr_buf = []
        self._tools_registered = False

        self._proc = subprocess.Popen(cmd, **kwargs)
        self._stdout_thread = threading.Thread(target=self._pump_stdout, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_thread.start()

        # Startup self-test: confirm we are talking to OUR worker, not an
        # arbitrary process with the same name. Fails fast if the subprocess
        # cannot import fabric_rlm._worker (the Fabric crash class).
        try:
            response = self._send_jsonrpc(
                "_self_test", {}, timeout=self.start_timeout, context="during startup self-test"
            )
        except Exception as exc:
            self.shutdown()
            stderr = "\n".join(self._stderr_buf).strip()
            raise CodeInterpreterError(
                f"fabric_rlm worker failed to start: {exc}. Worker stderr:\n{stderr}"
            ) from exc

        self._worker_self_test = response

    def shutdown(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "shutdown"}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
            if proc.poll() is None:
                proc.wait(timeout=5)
        except Exception:
            try:
                if proc.poll() is None:
                    if os.name == "nt":
                        proc.kill()
                    else:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=5)
            except Exception:
                pass
        finally:
            _close_worker_resources(proc, (self._stdout_thread, self._stderr_thread))

    def __enter__(self) -> "SubprocessPythonInterpreter":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)

    # ----- execute -----------------------------------------------------------

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        FinalOutput, CodeInterpreterError = _import_dspy_code_interpreter()

        # Parent-side security policy. On v7 we raise CodeInterpreterError
        # (the dspy contract) rather than fabricating a FinalOutput — the
        # dspy.RLM loop reads CodeInterpreterError as a recoverable turn
        # error and feeds the message back to the LM. This matches how
        # syntax/runtime errors are surfaced today.
        if self.security is not None and self.security.enabled:
            violation = self.security.validate_code(code)
            if violation is not None:
                if self.security.on_violation == "raise":
                    from .security import SecurityViolation

                    raise SecurityViolation(violation)
                raise CodeInterpreterError(violation)

        if self._proc is None or self._proc.poll() is not None:
            self.start()
        if not self._tools_registered:
            self._register_tools()

        if variables:
            bound_variables = {
                name: value
                for name, value in variables.items()
                if _collect_lakehouse_sources(value)
                or _collect_file_destinations(value)
            }
            ordinary_variables = {
                name: value
                for name, value in variables.items()
                if name not in bound_variables
            }
            if bound_variables:
                self._lakehouse_sources = _collect_lakehouse_sources(
                    bound_variables
                )
                self._file_destinations = _collect_file_destinations(
                    bound_variables
                )
                self._send_jsonrpc(
                    "set_inputs",
                    {
                        "inputs": {
                            name: encode_for_worker(value)
                            for name, value in bound_variables.items()
                        }
                    },
                    timeout=self.timeout,
                    context="binding parent-backed inputs",
                )
            if ordinary_variables:
                code = self._inject_variables(code, ordinary_variables) + "\n" + code

        # Send execute request. Then loop reading frames: tool_call requests
        # from worker get dispatched and replied to inline; the matching
        # execute response terminates the loop.
        self._last_exec_code = code
        self._request_id += 1
        request_id = self._request_id
        self._write_jsonrpc(
            {
                "jsonrpc": "2.0",
                "method": "execute",
                "params": {
                    "code": code,
                    "max_submit_bytes": self.max_submit_bytes,
                },
                "id": request_id,
            }
        )

        while True:
            line = self._read_response_line(self.timeout, "during execute")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Skip non-JSON lines defensively (matches dspy's behaviour).
                continue

            # Worker -> host tool callback.
            if msg.get("method") == "tool_call":
                self._handle_tool_call(msg)
                continue

            if msg.get("id") != request_id:
                # Out-of-order or unrelated frame; skip.
                continue

            if "result" in msg:
                result = msg["result"]
                if isinstance(result, dict) and "final" in result:
                    return FinalOutput(result["final"])
                if isinstance(result, dict) and "output" in result:
                    out = result["output"]
                    return out if out != "" else ""
                return result

            if "error" in msg:
                err = msg["error"] or {}
                code_int = err.get("code", _JSONRPC_APP_ERRORS["Unknown"])
                message = err.get("message", "Unknown error")
                data = err.get("data") or {}
                err_type = data.get("type", "Error")
                if code_int == _JSONRPC_APP_ERRORS["SyntaxError"]:
                    raise SyntaxError(f"Invalid Python syntax: {message}")
                raise CodeInterpreterError(f"{err_type}: {message}")

            raise CodeInterpreterError(f"Unexpected JSON-RPC frame: {msg!r}")

    def _inject_variables(self, code: str, variables: dict[str, Any]) -> str:
        """Prepend ``name = <repr>`` lines for each variable.

        Slice 1 keeps this simple — JSON-serialisable literal injection only.
        Large-var filesystem injection (>100MB) is deferred to a later slice
        if needed (dspy uses 100MB threshold).
        """
        lines = []
        for name, value in variables.items():
            if not name.isidentifier():
                FinalOutput, CodeInterpreterError = _import_dspy_code_interpreter()
                raise CodeInterpreterError(f"Invalid variable name: {name!r}")
            lines.append(f"{name} = {value!r}")
        return "\n".join(lines)

    # ----- tool registration + callback dispatch -----------------------------

    def _register_tools(self) -> None:
        """One-time per process: tell worker which tool names to expose to user code."""
        params: dict[str, Any] = {}
        if self.tools:
            params["tools"] = [{"name": name} for name in self.tools]
        if self.output_fields:
            params["outputs"] = self.output_fields
        if not params:
            self._tools_registered = True
            return
        self._send_jsonrpc("register", params, timeout=self.timeout, context="registering tools")
        self._tools_registered = True

    def _handle_tool_call(self, request: dict[str, Any]) -> None:
        """Dispatch a worker->host ``tool_call`` and write the JSON-RPC reply.

        Mirrors dspy.PythonInterpreter._handle_tool_call (lines 292-314).
        """
        FinalOutput, CodeInterpreterError = _import_dspy_code_interpreter()
        request_id = request.get("id")
        params = request.get("params", {}) or {}
        name = params.get("name")
        kwargs = params.get("kwargs", {}) or {}

        try:
            if name == _LAKEHOUSE_QUERY_TOOL:
                result = _execute_bound_lakehouse_query(
                    self._lakehouse_sources,
                    kwargs,
                )
            elif name == _FILE_PUBLISH_TOOL:
                result = _execute_bound_file_publish(
                    self._file_destinations,
                    kwargs,
                )
            elif name not in self.tools:
                raise CodeInterpreterError(f"Unknown tool: {name}")
            else:
                result = self.tools[name](**kwargs)
            is_json = isinstance(result, (list, dict))
            value = (
                json.dumps(result, ensure_ascii=False)
                if is_json
                else (str(result) if result is not None else "")
            )
            response = {
                "jsonrpc": "2.0",
                "result": {"value": value, "type": "json" if is_json else "string"},
                "id": request_id,
            }
        except Exception as exc:
            err_type = type(exc).__name__
            err_code = _JSONRPC_APP_ERRORS.get(err_type, _JSONRPC_APP_ERRORS["Unknown"])
            response = {
                "jsonrpc": "2.0",
                "error": {"code": err_code, "message": str(exc), "data": {"type": err_type}},
                "id": request_id,
            }
        self._write_jsonrpc(response)

    # ----- low-level IO ------------------------------------------------------

    def _send_jsonrpc(
        self, method: str, params: dict[str, Any], *, timeout: float, context: str
    ) -> Any:
        """Send a JSON-RPC request and read back its matching response."""
        self._request_id += 1
        request_id = self._request_id
        self._write_jsonrpc(
            {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}
        )
        while True:
            line = self._read_response_line(timeout, context)
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("method") == "tool_call":
                self._handle_tool_call(msg)
                continue
            if msg.get("id") != request_id:
                continue
            if "error" in msg:
                err = msg["error"] or {}
                raise RuntimeError(f"JSON-RPC error {context}: {err.get('message', err)}")
            return msg.get("result")

    def _write_jsonrpc(self, payload: dict[str, Any]) -> None:
        FinalOutput, CodeInterpreterError = _import_dspy_code_interpreter()
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            stderr = "\n".join(self._stderr_buf).strip()
            raise CodeInterpreterError(
                f"Worker process is not running. Worker stderr:\n{stderr}"
            )
        try:
            proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            stderr = "\n".join(self._stderr_buf).strip()
            raise CodeInterpreterError(
                f"Failed to write to worker: {exc}. Worker stderr:\n{stderr}"
            ) from exc

    def _read_response_line(self, timeout: float, context: str) -> str:
        FinalOutput, CodeInterpreterError = _import_dspy_code_interpreter()
        try:
            line = self._stdout_queue.get(timeout=timeout)
        except queue.Empty as exc:
            stderr = "\n".join(self._stderr_buf).strip()
            raise CodeInterpreterError(
                f"Worker did not respond within {timeout}s {context}. "
                f"Worker stderr:\n{stderr}"
            ) from exc
        if line is None:
            stderr = "\n".join(self._stderr_buf).strip()
            raise CodeInterpreterError(
                f"Worker exited unexpectedly {context}. Worker stderr:\n{stderr}"
                + concurrency_death_hint(getattr(self, "_last_exec_code", None))
            )
        return line.strip()

    def _pump_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._stdout_queue.put(line)
        self._stdout_queue.put(None)

    def _pump_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            # Cap memory: keep last 200 lines.
            self._stderr_buf.append(line)
            if len(self._stderr_buf) > 200:
                self._stderr_buf.pop(0)
