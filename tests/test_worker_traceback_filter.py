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
