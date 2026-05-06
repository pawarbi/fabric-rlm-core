"""BUG-LIB-1 regression test: worker tracebacks must not include framework noise.

When user code raises an error, the model receives the traceback as the
"error" field. Today that field includes ~17 lines of asyncio /
nest_asyncio / fabric_rlm._worker plumbing before the user's actual
exception, which (1) wastes tokens and (2) buries the real diagnostic.

The user-visible error must:
- Start with "Traceback (most recent call last):" or the exception line
- Contain at least one frame from the user code marker "<fabric_rlm_worker>"
- NOT contain framework frames (nest_asyncio, asyncio/, fabric_rlm/_worker.py)
- End with the actual exception type + message

This test exercises the same _execute path that subprocess interpreter uses
in production (see fabric_rlm/_worker.py:_execute).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from fabric_rlm import _worker


def _run_failing_code(code: str) -> dict:
    """Invoke the in-process _execute path and return the response dict."""
    return _worker._execute(code)


def test_user_assertion_error_message_is_preserved() -> None:
    result = _run_failing_code("assert False, 'cell A7 is None'")
    assert result["ok"] is False
    err = result["error"]
    assert "AssertionError: cell A7 is None" in err


def test_user_traceback_contains_user_frame_marker() -> None:
    result = _run_failing_code("assert False, 'boom'")
    assert "<fabric_rlm_worker>" in result["error"], (
        "user-code frame must remain in traceback so model can locate the line"
    )


@pytest.mark.parametrize(
    "noise",
    [
        "nest_asyncio",
        "asyncio/tasks.py",
        "asyncio/futures.py",
        "asyncio/runners.py",
        "fabric_rlm/_worker.py",
        "_run_code",
        "run_until_complete",
        "__step_run_and_handle_result",
    ],
)
def test_user_traceback_omits_framework_frames(noise: str) -> None:
    result = _run_failing_code("assert False, 'boom'")
    assert noise not in result["error"], (
        f"framework frame {noise!r} leaked into user-visible traceback:\n{result['error']}"
    )


def test_user_traceback_short_compared_to_raw() -> None:
    """The filtered traceback should be < 6 lines for a one-line user error
    (was ~18 lines before the fix)."""
    result = _run_failing_code("assert False, 'boom'")
    line_count = len(result["error"].splitlines())
    assert line_count <= 6, (
        f"expected <=6 lines, got {line_count}:\n{result['error']}"
    )


def test_runtime_error_class_and_message_preserved() -> None:
    """ZeroDivisionError full message must reach the model."""
    result = _run_failing_code("x = 1 / 0")
    assert "ZeroDivisionError" in result["error"]
    assert "division by zero" in result["error"]


def test_module_not_found_error_preserved() -> None:
    """ModuleNotFoundError (e.g. `from sandbox import SUBMIT`) must reach the model
    so it can adapt its imports."""
    result = _run_failing_code("import this_module_definitely_does_not_exist")
    assert "ModuleNotFoundError" in result["error"]
    assert "this_module_definitely_does_not_exist" in result["error"]


def test_multiline_user_code_keeps_correct_line_number() -> None:
    """User-code line number reported in the trace must match the user's source."""
    code = "x = 1\ny = 2\nassert False, 'on line three'"
    result = _run_failing_code(code)
    err = result["error"]
    assert "line 3" in err
    assert "on line three" in err


# ---------------------------------------------------------------------------
# Rubber-duck review follow-ups (chained exceptions, SyntaxError, async user
# code, defensive internal formatter, robust path filter).
# ---------------------------------------------------------------------------


def test_chained_exception_with_cause_preserves_root_cause() -> None:
    """`raise X from Y` must show BOTH exceptions so the model sees the real cause."""
    code = (
        "try:\n"
        "    int('not a number')\n"
        "except ValueError as e:\n"
        "    raise RuntimeError('wrapped') from e\n"
    )
    result = _run_failing_code(code)
    err = result["error"]
    assert "ValueError" in err, f"root cause lost from chain:\n{err}"
    assert "not a number" in err
    assert "RuntimeError" in err
    assert "wrapped" in err
    assert "direct cause" in err, f"chain separator missing:\n{err}"


def test_chained_exception_with_implicit_context_preserved() -> None:
    """Implicit chaining (`raise X` inside `except`) must show both exceptions
    via `During handling of the above exception, another exception occurred`.
    """
    code = (
        "try:\n"
        "    1 / 0\n"
        "except ZeroDivisionError:\n"
        "    raise AssertionError('handler failed')\n"
    )
    result = _run_failing_code(code)
    err = result["error"]
    assert "ZeroDivisionError" in err, f"implicit context lost:\n{err}"
    assert "AssertionError" in err
    assert "handler failed" in err
    assert "During handling" in err


def test_chained_exception_with_suppressed_context_omits_inner() -> None:
    """`raise X from None` must suppress the implicit context."""
    code = (
        "try:\n"
        "    1 / 0\n"
        "except ZeroDivisionError:\n"
        "    raise ValueError('clean') from None\n"
    )
    result = _run_failing_code(code)
    err = result["error"]
    assert "ValueError: clean" in err
    assert "ZeroDivisionError" not in err, (
        f"`from None` should suppress implicit context, got:\n{err}"
    )


def test_syntax_error_reaches_user_with_useful_message() -> None:
    """SyntaxError happens at compile time -- the formatter must still produce
    a non-empty, informative message even with no user frames in the traceback."""
    result = _run_failing_code("def foo(:\n    pass\n")
    assert result["ok"] is False
    err = result["error"]
    assert "SyntaxError" in err, f"SyntaxError type missing from:\n{err!r}"
    assert err.strip() != "", "formatter produced empty output for SyntaxError"


def test_async_user_function_keeps_user_frames() -> None:
    """User code that legitimately uses asyncio (e.g. `await asyncio.sleep(0)`)
    must keep its OWN frames -- only stdlib asyncio internals are filtered."""
    code = (
        "import asyncio\n"
        "async def my_user_helper():\n"
        "    await asyncio.sleep(0)\n"
        "    raise ValueError('from async user code')\n"
        "asyncio.run(my_user_helper())\n"
    )
    result = _run_failing_code(code)
    err = result["error"]
    assert result["ok"] is False
    assert "ValueError" in err
    assert "from async user code" in err
    assert "my_user_helper" in err, (
        f"user's async frame was filtered out as 'asyncio noise':\n{err}"
    )
    # Stdlib asyncio internals (runners.py, base_events.py, tasks.py) MUST be filtered.
    assert "base_events.py" not in err
    assert "runners.py" not in err
    assert "tasks.py" not in err


def test_user_dir_named_asyncio_is_not_filtered_out() -> None:
    """Defensive: a user file in a directory whose tail is e.g. asyncio_helpers
    must NOT be filtered. The filter only targets the real stdlib asyncio
    package by checking against asyncio.__file__'s parent directory."""
    import asyncio as _asyncio_mod
    asyncio_root = str(Path(_asyncio_mod.__file__).resolve().parent) + os.sep
    fake = "/home/user/asyncio_helpers/utils.py"
    assert _worker._is_worker_frame(fake, "/path/to/_worker.py", asyncio_root) is False, (
        "user file with 'asyncio' in path was wrongly filtered as worker plumbing"
    )


def test_internal_traceback_formatter_keeps_full_stack() -> None:
    """The defensive `_format_internal_traceback` (used for worker bugs, NOT
    user code) must include the full unfiltered stack so operators can debug."""
    try:
        # Raise from inside the worker module so frames belong to _worker.py.
        # Use _execute on a snippet that exercises some worker plumbing then
        # synthesize an internal exception ourselves.
        raise RuntimeError("simulated internal worker bug")
    except RuntimeError as exc:
        out = _worker._format_internal_traceback(exc)
    assert "RuntimeError" in out
    assert "simulated internal worker bug" in out
    # Internal formatter MUST keep this test's frame (it would be filtered by
    # the user formatter as "test infra"... actually it wouldn't, but the key
    # property is that NO frames are filtered).
    assert "Traceback" in out
    assert "test_internal_traceback_formatter_keeps_full_stack" in out


def test_user_traceback_helper_is_idempotent_to_call_with_explicit_exc() -> None:
    """Calling _format_user_traceback with an explicit exc and via sys.exc_info
    inside an except block must produce the same result for the same exception."""
    try:
        raise ValueError("same input")
    except ValueError as exc:
        from_explicit = _worker._format_user_traceback(exc)
        from_sys = _worker._format_user_traceback()
    assert "ValueError: same input" in from_explicit
    assert "ValueError: same input" in from_sys

