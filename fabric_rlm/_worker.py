"""Subprocess worker for fabric-rlm.

Speaks two protocols on the same stdin/stdout channel, dispatched per-message
based on envelope shape:

1. Legacy ``{"op": ...}`` envelope used by ``fabric_rlm.interpreter.Interpreter``.
2. JSON-RPC 2.0 envelope used by ``fabric_rlm.interpreter.SubprocessPythonInterpreter``,
   which mirrors ``dspy.primitives.python_interpreter.PythonInterpreter`` so that
   ``dspy.predict.RLM`` can drive our worker as a drop-in replacement for its
   Deno/Pyodide sandbox.

User code stdout/stderr are captured during execute so the protocol channel
stays clean. Tool callbacks (worker -> host) use the JSON-RPC ``tool_call``
method, blocking on the host's response on stdin.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib

# Allow user code that uses the canonical `asyncio.run(...)` /
# `loop.run_until_complete(...)` idioms even though the worker is already
# inside a running event loop (see _execute / _jsonrpc_execute).
# Without this, every model that follows the standard Python async pattern
# crashes with "asyncio.run() cannot be called from a running event loop"
# and burns turns figuring out the correct top-level-await idiom.
try:
    import nest_asyncio  # type: ignore[import-not-found]

    nest_asyncio.apply()
except ImportError:  # pragma: no cover - dep missing in stripped envs
    pass
import dataclasses
import inspect
import io
import json
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, get_args, get_origin

from .artifacts import File, decode_from_worker_wire
from .serializers import DEFAULT_INJECTED_NAMES, freeze, snapshot
from .skill_loader import list_skills, load_skill


# Capture the real stdio handles BEFORE any user code can redirect them.
# Tool-callback stubs and protocol writes go through these to avoid being
# captured by `contextlib.redirect_stdout` during _execute.
#
# Force UTF-8 on the wire so non-ASCII characters captured from user code
# (e.g., smart quotes in PDF text) don't crash the worker on hosts whose
# default stdout encoding is not UTF-8 (Windows cp1252). The parent-side
# subprocess.Popen is configured with encoding="utf-8", so this matches.
_REAL_STDIN = sys.stdin
_REAL_STDOUT = sys.stdout
for _stream in (_REAL_STDIN, _REAL_STDOUT):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - best-effort on exotic streams
            pass


_ACTIVATE_MARKER = "[FABRIC_RLM_ACTIVATE]"


def activate_skill(name: str) -> str:
    """Activate a skill: load its body and signal the host to enable its verifier.

    The marker line printed below is parsed by the RLM driver to add ``name`` to
    the active-verifier set. The skill body string is also returned so the model
    can inspect it inline.
    """

    body = load_skill(name)
    print(f"{_ACTIVATE_MARKER}:{name}")
    return body

_namespace: dict[str, Any] = {}
_lm_spec: Any = None
_lm_instance: Any = None


class _SubmitSignal(BaseException):
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload


# Output-field names registered by the host (dspy.RLM passes
# self._get_output_fields_info()). Empty == default kwargs-only SUBMIT.
_registered_output_fields: list[str] = []


def SUBMIT(*args: Any, **kwargs: Any) -> None:
    """End execution and return a final structured payload to the driver.

    Mirrors dspy's runner.js dynamic SUBMIT signature:
    if the host registered output_fields ``[a, b]``, the model can call
    ``SUBMIT(a_val, b_val)`` (positional, in registered order),
    ``SUBMIT(a=a_val, b=b_val)`` (keyword), or any mix.
    Without registered output_fields, falls back to kwargs-only.
    """
    if not _registered_output_fields:
        if args:
            raise TypeError(
                "SUBMIT() takes 0 positional arguments but "
                f"{len(args)} were given (no output fields registered)"
            )
        raise _SubmitSignal(dict(kwargs))

    names = _registered_output_fields
    if len(args) > len(names):
        raise TypeError(
            f"SUBMIT() takes at most {len(names)} positional arguments "
            f"({', '.join(names)}); got {len(args)}"
        )
    payload: dict[str, Any] = {}
    for i, val in enumerate(args):
        payload[names[i]] = val
    for k, v in kwargs.items():
        if k in payload:
            raise TypeError(f"SUBMIT() got multiple values for argument '{k}'")
        if k not in names:
            raise TypeError(
                f"SUBMIT() got unexpected keyword argument '{k}' "
                f"(expected: {', '.join(names)})"
            )
        payload[k] = v
    raise _SubmitSignal(payload)


def _install_runtime_api() -> None:
    _namespace.clear()
    _namespace.update(
        {
            "File": File,
            "SUBMIT": SUBMIT,
            "predict": predict,
            "predict_sync": predict_sync,
            "load_skill": load_skill,
            "activate_skill": activate_skill,
            "list_skills": list_skills,
        }
    )
    _install_sandbox_shim()


# Public names exposed by the synthetic ``sandbox`` module. Kept in sync
# with _install_runtime_api so user code can use either eval-namespace
# globals (``SUBMIT(...)``) or the conventional import form
# (``from sandbox import SUBMIT``). Without this shim, gpt-5 wastes ~7-8%
# of turns on ``ModuleNotFoundError: No module named 'sandbox'``.
_SANDBOX_PUBLIC_NAMES: tuple[str, ...] = (
    "File",
    "SUBMIT",
    "predict",
    "predict_sync",
    "load_skill",
    "activate_skill",
    "list_skills",
)

# Private sentinel used to mark our shim module so re-installs don't clobber
# a real `sandbox` package the user happened to import. Object identity is
# more collision-resistant than a boolean flag (a real module that happened
# to set ``__fabric_rlm_shim__ = True`` would otherwise be misidentified).
_SANDBOX_SHIM_SENTINEL: object = object()


def _install_sandbox_shim() -> None:
    """Register a synthetic ``sandbox`` module in sys.modules.

    Attributes mirror the runtime API names above and resolve to the live
    module-level functions, so users can write either::

        SUBMIT(answer="x")            # eval-namespace form
        from sandbox import SUBMIT    # import form (this shim)
        import sandbox; sandbox.SUBMIT(answer="x")

    OWNERSHIP POLICY
    ----------------
    The fabric_rlm worker subprocess intentionally OWNS the ``sandbox``
    module name in its sys.modules. If a real package named ``sandbox``
    is already present at install time, it is replaced. Rationale:

    1. The worker subprocess is a dedicated execution sandbox; user code
       running inside it expects ``sandbox`` to mean the harness API.
    2. The 105/1354 (7.7%) gpt-5 turn loss measured in benchmarks comes
       from the model assuming this convention (see PR #4).
    3. If a real ``sandbox`` package is needed, users should fork the
       worker entrypoint -- the worker is a single-purpose process.

    The shim is reinstalled on every ``_install_runtime_api()`` call (i.e.
    after JSON-RPC ``reset``) so attribute references stay live even if a
    test or framework reloads ``fabric_rlm._worker``.
    """
    import sys
    import types

    module = sys.modules.get("sandbox")
    is_our_shim = (
        module is not None
        and getattr(module, "__fabric_rlm_shim__", None) is _SANDBOX_SHIM_SENTINEL
    )
    if not is_our_shim:
        module = types.ModuleType("sandbox")
        module.__doc__ = (
            "Synthetic shim exposing the fabric_rlm worker runtime API "
            "(SUBMIT, predict, File, ...) under the conventional `sandbox` "
            "module name. Owned by fabric_rlm._worker -- replaced on each "
            "_install_runtime_api() call."
        )
        module.__fabric_rlm_shim__ = _SANDBOX_SHIM_SENTINEL  # type: ignore[attr-defined]
        sys.modules["sandbox"] = module

    current = globals()
    for name in _SANDBOX_PUBLIC_NAMES:
        setattr(module, name, current[name])
    # Drop any attributes from a previous install that aren't part of the
    # curated surface (defensive: keeps `dir(sandbox)` clean across reinstalls).
    # Skips dunders / underscore names so we don't disturb __spec__, __loader__,
    # __fabric_rlm_shim__, etc. that the import system or our sentinel rely on.
    for attr in list(vars(module)):
        if attr.startswith("_"):
            continue
        if attr not in _SANDBOX_PUBLIC_NAMES:
            delattr(module, attr)


def _set_lm_spec(spec: Any) -> None:
    global _lm_spec, _lm_instance
    _lm_spec = spec
    _lm_instance = None


def _get_lm() -> Any:
    global _lm_instance
    if _lm_spec is None:
        raise RuntimeError("Sub-LM is not configured. Call Interpreter.configure_lm() first.")
    if _lm_instance is None:
        from .lm import resolve_lm

        _lm_instance = resolve_lm(_lm_spec)
    return _lm_instance


async def predict(
    signature: Any,
    instructions: str | None = None,
    pydantic_schemas: dict[str, type] | None = None,
    **kwargs: Any,
) -> Any:
    """Call the configured sub-LM from user code."""

    import dspy

    lm = _get_lm()
    sig = _build_dspy_signature(dspy, signature, instructions, pydantic_schemas)
    prepared_kwargs = _prepare_predict_inputs(dspy, sig, kwargs)
    try:
        predictor = dspy.Predict(sig)
    except Exception as exc:
        raise RuntimeError(f"Could not create DSPy predictor for signature {signature!r}: {exc}") from exc

    with dspy.context(lm=lm):
        if hasattr(predictor, "acall"):
            result = await predictor.acall(**prepared_kwargs)
            return _serialize_predict_result(dspy, result)
        result = predictor(**prepared_kwargs)
        if inspect.isawaitable(result):
            result = await result
        return _serialize_predict_result(dspy, result)


def predict_sync(
    signature: Any,
    instructions: str | None = None,
    pydantic_schemas: dict[str, type] | None = None,
    **kwargs: Any,
) -> Any:
    """Synchronous wrapper around :func:`predict`.

    Lets users call the sub-LM without writing ``asyncio.run(...)`` boilerplate
    inside every turn::

        result = predict_sync('english -> french', english=phrase)
        SUBMIT(answer=result.french)

    Safe to call even though the worker is already inside a running event
    loop because ``nest_asyncio.apply()`` runs at module import (see top of
    this file). If ``nest_asyncio`` is unavailable, falls back to creating
    a fresh loop.
    """

    coro = predict(
        signature,
        instructions=instructions,
        pydantic_schemas=pydantic_schemas,
        **kwargs,
    )
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.run_until_complete(coro)
    except RuntimeError:
        pass
    return asyncio.run(coro)


def _build_dspy_signature(
    dspy: Any,
    signature: Any,
    instructions: str | None,
    pydantic_schemas: Mapping[str, type] | None,
) -> Any:
    if not isinstance(signature, str):
        if pydantic_schemas:
            raise ValueError("pydantic_schemas can only be used with string DSPy signatures.")
        if instructions is not None:
            if hasattr(signature, "with_instructions"):
                return signature.with_instructions(instructions)
            raise ValueError("instructions can only be used with string signatures or Signature.with_instructions().")
        return signature

    custom_types = _collect_signature_custom_types(dspy, pydantic_schemas)
    try:
        return _call_dspy_signature(dspy, signature, instructions, custom_types)
    except Exception as exc:
        hint = (
            " If the signature uses custom or Pydantic output types, pass "
            "pydantic_schemas={'TypeName': TypeName}."
        )
        raise ValueError(f"Could not build DSPy signature {signature!r}: {exc}.{hint}") from exc


def _call_dspy_signature(
    dspy: Any,
    signature: str,
    instructions: str | None,
    custom_types: dict[str, type] | None,
) -> Any:
    if custom_types:
        try:
            if instructions is not None:
                return dspy.Signature(signature, instructions=instructions, custom_types=custom_types)
            return dspy.Signature(signature, custom_types=custom_types)
        except TypeError as exc:
            if "custom_types" not in str(exc) and "unexpected keyword" not in str(exc):
                raise

    if instructions is not None:
        try:
            return dspy.Signature(signature, instructions=instructions)
        except TypeError:
            return dspy.Signature(signature, instructions)
    return dspy.Signature(signature)


def _collect_signature_custom_types(
    dspy: Any,
    pydantic_schemas: Mapping[str, type] | None,
) -> dict[str, type] | None:
    if pydantic_schemas is not None and not isinstance(pydantic_schemas, Mapping):
        raise TypeError("pydantic_schemas must be a mapping of type names to Python classes.")

    custom_types: dict[str, type] = {}
    if hasattr(dspy, "Image"):
        custom_types["dspy.Image"] = dspy.Image

    for name, value in _namespace.items():
        if isinstance(value, type):
            custom_types.setdefault(name, value)

    for name, value in (pydantic_schemas or {}).items():
        if not isinstance(value, type):
            raise TypeError(f"pydantic_schemas[{name!r}] must be a Python class, got {type(value).__name__}.")
        custom_types[str(name)] = value

    return custom_types or None


def _prepare_predict_inputs(dspy: Any, sig: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    input_fields = getattr(sig, "input_fields", None)
    if not isinstance(input_fields, Mapping):
        return kwargs

    prepared = dict(kwargs)
    for name, field in input_fields.items():
        if name not in prepared:
            continue
        annotation = getattr(field, "annotation", None)
        if _annotation_contains_dspy_image(dspy, annotation):
            prepared[name] = _wrap_image_value(dspy, name, prepared[name], annotation)
    return prepared


def _annotation_contains_dspy_image(dspy: Any, annotation: Any) -> bool:
    image_type = getattr(dspy, "Image", None)
    if image_type is None or annotation is None:
        return False
    if annotation is image_type:
        return True
    origin = get_origin(annotation)
    if origin is not None:
        return any(_annotation_contains_dspy_image(dspy, arg) for arg in get_args(annotation))
    return False


def _wrap_image_value(dspy: Any, field_name: str, value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (list, tuple, set, frozenset) and args:
        if not isinstance(value, (list, tuple, set, frozenset)):
            raise TypeError(f"Input field {field_name!r} expects image values in a collection.")
        wrapped = [_wrap_image_value(dspy, field_name, item, args[0]) for item in value]
        if origin is tuple:
            return tuple(wrapped)
        if origin in (set, frozenset):
            return origin(wrapped)
        return wrapped
    return _wrap_single_image(dspy, field_name, value)


def _wrap_single_image(dspy: Any, field_name: str, value: Any) -> Any:
    image_type = getattr(dspy, "Image", None)
    if image_type is None or value is None or isinstance(value, image_type):
        return value
    source = value.path if isinstance(value, File) else str(value) if isinstance(value, Path) else value
    try:
        return image_type(source)
    except Exception as exc:
        raise TypeError(
            f"Could not convert input field {field_name!r} ({type(value).__name__}) to dspy.Image: {exc}"
        ) from exc


def _serialize_predict_result(dspy: Any, result: Any) -> Any:
    prediction_type = getattr(dspy, "Prediction", None)
    if prediction_type is not None and isinstance(result, prediction_type):
        return prediction_type(**freeze(result.toDict()))
    if hasattr(result, "toDict"):
        return freeze(result)
    if hasattr(result, "model_dump") or (
        dataclasses.is_dataclass(result) and not isinstance(result, type)
    ):
        return freeze(result)
    if isinstance(result, (Mapping, list, tuple, set)):
        return freeze(result)
    return result


async def _run_code(code: str) -> Any:
    compiled = compile(code, "<fabric_rlm_worker>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    result = eval(compiled, _namespace, _namespace)
    if inspect.isawaitable(result):
        return await result
    return result


def _state() -> dict[str, Any]:
    return snapshot(_namespace, injected_names=set(DEFAULT_INJECTED_NAMES))


def _execute(code: str) -> dict[str, Any]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            asyncio.run(_run_code(code))
        return {
            "ok": True,
            "submitted": False,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "state": _state(),
        }
    except _SubmitSignal as submit:
        return {
            "ok": True,
            "submitted": True,
            "submit_payload": freeze(submit.payload),
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "state": _state(),
        }
    except BaseException:
        return {
            "ok": False,
            "submitted": False,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "error": traceback.format_exc(),
            "state": _state(),
        }


def _set_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    decoded = {name: decode_from_worker_wire(value) for name, value in inputs.items()}
    _namespace.update(decoded)
    return {"ok": True, "state": _state()}


def _handle_message(message: dict[str, Any]) -> dict[str, Any] | None:
    op = message.get("op")
    if op == "exec":
        return _execute(message["code"])
    if op == "set_inputs":
        return _set_inputs(message.get("inputs", {}))
    if op == "configure_lm":
        _set_lm_spec(message.get("spec"))
        return {"ok": True}
    if op == "reset":
        _install_runtime_api()
        return {"ok": True, "state": _state()}
    if op == "ping":
        return {"ok": True, "state": _state()}
    if op == "shutdown":
        return None
    return {"ok": False, "error": f"Unknown op: {op!r}", "state": _state()}


def _write_response(response: dict[str, Any]) -> None:
    _REAL_STDOUT.write(json.dumps(response, ensure_ascii=False) + "\n")
    _REAL_STDOUT.flush()


# =====================================================================
# JSON-RPC 2.0 protocol (dspy CodeInterpreter contract)
# =====================================================================
#
# Mirrors `dspy.primitives.python_interpreter` (lines 30-72 helpers,
# 264-315 register/tool_call, 484-563 execute loop) so that a host
# implementing the dspy CodeInterpreter Protocol can drive this worker
# without changes. We re-implement, not import, because dspy's helpers
# live next to its Deno-specific PythonInterpreter and we want the
# worker module to stay independent of dspy at startup.

# Application-level JSON-RPC error codes. SyntaxError MUST stay -32000
# because dspy.PythonInterpreter.execute (line 555) special-cases that
# code to re-raise SyntaxError UNwrapped.
JSONRPC_APP_ERRORS: dict[str, int] = {
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


_registered_tools: dict[str, dict] = {}
_jsonrpc_outbound_id: int = 0


def _jsonrpc_send(payload: dict[str, Any]) -> None:
    """Write a JSON-RPC frame as a single line to the real stdout."""
    _REAL_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _REAL_STDOUT.flush()


def _jsonrpc_read_line() -> str:
    """Block reading one line of JSON-RPC input from the real stdin."""
    line = _REAL_STDIN.readline()
    if not line:
        raise RuntimeError("Host closed stdin while worker was awaiting a JSON-RPC reply.")
    return line


def _next_outbound_id() -> int:
    global _jsonrpc_outbound_id
    _jsonrpc_outbound_id += 1
    return _jsonrpc_outbound_id


def _make_tool_stub(name: str) -> Callable[..., Any]:
    """Build a user-code-callable stub that round-trips a tool call to the host.

    The stub is synchronous: it sends a ``tool_call`` request, blocks reading
    the next stdin line (which by protocol is the matching response), and
    returns the value to user code (or raises if the host returned an error).

    Mirrors the dspy contract:
    - For ``type=="string"`` results the value string is returned as-is.
    - For ``type=="json"`` results the JSON string is returned as-is so user
      code can call ``json.loads`` (matches dspy's runner.js behaviour).
    """

    def _tool_stub(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError(
                f"{name}(...) requires KEYWORD arguments only, not positional. "
                f"Call as {name}(<param>=<value>, ...). See the tool's docstring for parameter names."
            )
        request_id = _next_outbound_id()
        request = {
            "jsonrpc": "2.0",
            "method": "tool_call",
            "params": {"name": name, "kwargs": kwargs},
            "id": request_id,
        }
        _jsonrpc_send(request)

        # Block waiting for the matching response. Skip JSON-RPC notifications
        # (none expected in this direction) and unrelated lines defensively.
        while True:
            line = _jsonrpc_read_line()
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" not in msg or msg.get("id") != request_id:
                continue
            if "error" in msg:
                err = msg["error"] or {}
                data = err.get("data") or {}
                err_type = data.get("type", "RuntimeError")
                raise RuntimeError(f"{err_type}: {err.get('message', 'tool call failed')}")
            result = msg.get("result") or {}
            return result.get("value", "")

    _tool_stub.__name__ = name
    return _tool_stub


def _install_registered_tool_stubs() -> None:
    for name in _registered_tools:
        _namespace[name] = _make_tool_stub(name)


def _jsonrpc_register(params: dict[str, Any]) -> dict[str, Any]:
    """Register host-side tools (and optional output_fields) callable from user code.

    When ``outputs`` is provided (dspy.RLM injects it via
    ``interpreter.output_fields = self._get_output_fields_info()``), update the
    SUBMIT positional-arg ordering so the model's ``SUBMIT(value)`` call
    succeeds. Without this, dspy's prompt tells the model
    ``SUBMIT(field1, field2, ...)`` but our worker raises TypeError → the dspy
    loop never terminates.
    """
    global _registered_output_fields
    _registered_tools.clear()
    for tool in params.get("tools") or []:
        if "name" not in tool:
            raise ValueError("register: each tool must have a 'name'")
        _registered_tools[tool["name"]] = tool
    _install_registered_tool_stubs()

    outputs = params.get("outputs") or []
    _registered_output_fields = [o["name"] for o in outputs if "name" in o]

    return {
        "ok": True,
        "registered": list(_registered_tools.keys()),
        "output_fields": list(_registered_output_fields),
    }


def _jsonrpc_self_test() -> dict[str, Any]:
    """Startup-only diagnostic so the host can confirm it spawned the right worker."""
    info: dict[str, Any] = {
        "worker_file": __file__,
        "sys_executable": sys.executable,
        "sys_version": sys.version,
        "registered_tools": list(_registered_tools.keys()),
    }
    try:
        import dspy as _dspy

        info["dspy_version"] = getattr(_dspy, "__version__", "unknown")
    except Exception:
        info["dspy_version"] = None
    return info


def _jsonrpc_execute(code: str) -> dict[str, Any]:
    """Run user code and shape the result into the dspy CodeInterpreter contract.

    Returns a result dict that the caller wraps in ``{"jsonrpc":"2.0","result":...}``.
    On failure raises (caller wraps as JSON-RPC error frame).
    """
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            asyncio.run(_run_code(code))
    except _SubmitSignal as submit:
        # Mirrors dspy.PythonInterpreter.execute lines 538-540: SUBMIT becomes
        # a successful result with a "final" key carrying the output payload.
        return {"final": freeze(submit.payload), "stdout": stdout.getvalue()}
    except SyntaxError as exc:
        # Re-raise with code -32000 so the host re-raises as SyntaxError UNWRAPPED
        # (dspy.PythonInterpreter.execute line 555).
        raise _JsonRpcAppError("SyntaxError", str(exc)) from exc
    except BaseException as exc:
        err_type = type(exc).__name__
        raise _JsonRpcAppError(err_type, str(exc)) from exc
    # Normal completion: stdout becomes the result string.
    return {"output": stdout.getvalue()}


class _JsonRpcAppError(Exception):
    """Internal carrier so _handle_jsonrpc can build the correct error frame."""

    def __init__(self, type_name: str, message: str):
        super().__init__(message)
        self.type_name = type_name
        self.message = message


def _handle_jsonrpc(message: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatch one JSON-RPC request/notification. Returns response dict or None.

    Returning None signals shutdown to the main loop.
    """
    method = message.get("method")
    msg_id = message.get("id")  # may be missing for notifications
    params = message.get("params") or {}

    # Shutdown notification: no response.
    if method == "shutdown" and msg_id is None:
        return None

    if msg_id is None:
        # Unknown notification — silently ignore per JSON-RPC 2.0.
        return {"_skip": True}

    try:
        if method == "_self_test":
            result = _jsonrpc_self_test()
        elif method == "register":
            result = _jsonrpc_register(params)
        elif method == "execute":
            code = params.get("code")
            if not isinstance(code, str):
                raise _JsonRpcAppError("ValueError", "execute: 'code' must be a string")
            result = _jsonrpc_execute(code)
        elif method == "ping":
            result = {"pong": True}
        else:
            raise _JsonRpcAppError("ValueError", f"Unknown JSON-RPC method: {method!r}")
        return {"jsonrpc": "2.0", "result": result, "id": msg_id}
    except _JsonRpcAppError as exc:
        code = JSONRPC_APP_ERRORS.get(exc.type_name, JSONRPC_APP_ERRORS["Unknown"])
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": exc.message,
                "data": {"type": exc.type_name},
            },
            "id": msg_id,
        }
    except Exception as exc:  # pragma: no cover -- defensive
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": JSONRPC_APP_ERRORS["Unknown"],
                "message": str(exc) or "internal worker error",
                "data": {"type": type(exc).__name__, "trace": traceback.format_exc()},
            },
            "id": msg_id,
        }


def main() -> None:
    _install_runtime_api()
    while True:
        line = _REAL_STDIN.readline()
        if not line:  # EOF
            break
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _write_response({"ok": False, "error": f"Invalid JSON: {line!r}"})
            continue

        # Envelope discrimination: JSON-RPC 2.0 messages carry "jsonrpc",
        # legacy fabric_rlm.Interpreter messages carry "op".
        if isinstance(message, dict) and "jsonrpc" in message:
            response = _handle_jsonrpc(message)
            if response is None:
                break
            if response.get("_skip"):
                continue
            _jsonrpc_send(response)
            continue

        try:
            response = _handle_message(message)
            if response is None:
                break
        except Exception:
            response = {"ok": False, "error": traceback.format_exc(), "state": _state()}
        _write_response(response)


if __name__ == "__main__":
    main()

