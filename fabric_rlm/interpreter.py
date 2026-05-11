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
from dataclasses import dataclass
from typing import Any, Callable

from .artifacts import encode_for_worker
from .security import SecurityPolicy


class WorkerTimeout(TimeoutError):
    """Raised when the worker does not respond within the configured timeout."""


class WorkerProtocolError(RuntimeError):
    """Raised when the worker exits or returns invalid protocol data."""


@dataclass
class ExecResult:
    ok: bool
    submitted: bool
    stdout: str
    stderr: str
    state: dict[str, Any]
    error: str | None = None
    submit_payload: dict[str, Any] | None = None

    @classmethod
    def from_response(cls, raw: dict[str, Any]) -> "ExecResult":
        return cls(
            ok=bool(raw.get("ok")),
            submitted=bool(raw.get("submitted", False)),
            stdout=str(raw.get("stdout", "")),
            stderr=str(raw.get("stderr", "")),
            state=dict(raw.get("state", {})),
            error=raw.get("error"),
            submit_payload=raw.get("submit_payload"),
        )


class Interpreter:
    """Persistent Python subprocess with JSON-line protocol."""

    def __init__(
        self,
        timeout: float = 300.0,
        python: str | None = None,
        cwd: str | None = None,
        security: SecurityPolicy | None = None,
    ):
        self.timeout = timeout
        self.python = python or sys.executable
        self.cwd = cwd
        # ``security`` is opt-in at this layer (None = no enforcement). The
        # public RLM facade is responsible for passing a default policy in
        # for LM-facing executions; verifier code paths intentionally leave
        # this None because verifier code is trusted, internal, and may need
        # APIs the policy would otherwise reject.
        self.security: SecurityPolicy | None = security
        self.proc: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None

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
        }
        # Scrub secret-bearing env vars from the worker if a policy is set.
        # Without this branch the child inherits the parent env wholesale.
        if self.security is not None and self.security.enabled:
            kwargs["env"] = self.security.scrub_env(dict(os.environ))
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        self._stdout_queue = queue.Queue()
        self.proc = subprocess.Popen(
            [self.python, "-u", "-m", "fabric_rlm._worker"],
            **kwargs,
        )
        self._stdout_thread = threading.Thread(target=self._pump_stdout, daemon=True)
        self._stdout_thread.start()
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
                # Default: feedback to the loop.
                return ExecResult(
                    ok=False,
                    submitted=False,
                    stdout="",
                    stderr=violation,
                    state={},
                    error=violation,
                )
        raw = self._request({"op": "exec", "code": code})
        return ExecResult.from_response(raw)

    def configure_lm(self, spec: Any) -> dict[str, Any]:
        return self._request({"op": "configure_lm", "spec": encode_for_worker(spec)})

    def set_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        encoded = {name: encode_for_worker(value) for name, value in inputs.items()}
        return self._request({"op": "set_inputs", "inputs": encoded})

    def reset(self) -> dict[str, Any]:
        return self._request({"op": "reset"})

    def ping(self) -> dict[str, Any]:
        return self._request({"op": "ping"})

    def shutdown(self) -> None:
        if not self.is_running:
            return
        try:
            self._send({"op": "shutdown"})
            assert self.proc is not None
            self.proc.wait(timeout=5)
        except Exception:
            self.kill()

    def kill(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        if os.name == "nt":
            self.proc.kill()
        else:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        self.proc.wait(timeout=5)

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        self._send(message)
        return self._recv()

    def _send(self, message: dict[str, Any]) -> None:
        if not self.is_running or self.proc is None or self.proc.stdin is None:
            raise WorkerProtocolError("Worker is not running")
        try:
            self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise WorkerProtocolError(self._format_worker_exit("Worker pipe closed")) from exc

    def _recv(self) -> dict[str, Any]:
        try:
            line = self._stdout_queue.get(timeout=self.timeout)
        except queue.Empty as exc:
            self.kill()
            raise WorkerTimeout(f"Worker timed out after {self.timeout}s") from exc

        if line is None:
            raise WorkerProtocolError(self._format_worker_exit("Worker exited without response"))

        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkerProtocolError(f"Invalid worker JSON response: {line!r}") from exc

    def _pump_stdout(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            self._stdout_queue.put(line)
        self._stdout_queue.put(None)

    def _format_worker_exit(self, prefix: str) -> str:
        stderr = ""
        if self.proc is not None and self.proc.stderr is not None:
            try:
                stderr = self.proc.stderr.read()
            except Exception:
                stderr = ""
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
        start_timeout: float = 15.0,
        python: str | None = None,
        cwd: str | None = None,
        security: SecurityPolicy | None = None,
    ) -> None:
        self.tools: dict[str, Callable[..., Any]] = dict(tools) if tools else {}
        self.output_fields = output_fields
        self.timeout = timeout
        self.start_timeout = start_timeout
        self.python = python or sys.executable
        self.cwd = cwd
        # See ``Interpreter.__init__`` — opt-in at this layer; ``RLM`` is the
        # public surface that defaults this on for LM-facing code.
        self.security: SecurityPolicy | None = security

        self._proc: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_buf: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._tools_registered = False
        self._request_id = 0

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

        env = {**os.environ, "PYTHONPATH": self._compute_pythonpath()}
        if self.security is not None and self.security.enabled:
            # Scrub secrets from the worker env. PYTHONPATH was set above and
            # is on the policy's default keep list, so it survives the scrub.
            env = self.security.scrub_env(env)

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
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
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
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "shutdown"}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:
            try:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
            except Exception:
                pass

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
            code = self._inject_variables(code, variables) + "\n" + code

        # Send execute request. Then loop reading frames: tool_call requests
        # from worker get dispatched and replied to inline; the matching
        # execute response terminates the loop.
        self._request_id += 1
        request_id = self._request_id
        self._write_jsonrpc(
            {"jsonrpc": "2.0", "method": "execute", "params": {"code": code}, "id": request_id}
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
            if name not in self.tools:
                raise CodeInterpreterError(f"Unknown tool: {name}")
            result = self.tools[name](**kwargs)
            is_json = isinstance(result, (list, dict))
            value = (
                json.dumps(result)
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

