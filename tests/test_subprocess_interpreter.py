"""Slice 1 tests: SubprocessPythonInterpreter satisfies dspy.CodeInterpreter Protocol.

DAMP > DRY: each test names exactly one behavior. One assertion per concept.

Source contracts being verified:
- dspy/primitives/code_interpreter.py lines 39-148 (Protocol, FinalOutput, CodeInterpreterError)
- dspy/primitives/python_interpreter.py lines 264-315 (register_tools + handle_tool_call)
- dspy/primitives/python_interpreter.py lines 484-563 (execute loop + error mapping)
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from dspy.primitives.code_interpreter import (
    CodeInterpreter,
    CodeInterpreterError,
    FinalOutput,
)

from fabric_rlm.interpreter import SubprocessPythonInterpreter


# ----- Protocol satisfaction ---------------------------------------------------


def test_satisfies_dspy_code_interpreter_protocol() -> None:
    interp = SubprocessPythonInterpreter()
    assert isinstance(interp, CodeInterpreter)


def test_tools_property_is_mutable_dict() -> None:
    interp = SubprocessPythonInterpreter()
    assert interp.tools == {}
    interp.tools["echo"] = lambda msg: msg
    assert "echo" in interp.tools


# ----- Basic execution ---------------------------------------------------------


def test_simple_print_returns_stdout_string() -> None:
    with SubprocessPythonInterpreter() as interp:
        result = interp.execute("print('hello world')")
    assert result == "hello world\n"


def test_no_output_returns_empty_string() -> None:
    with SubprocessPythonInterpreter() as interp:
        result = interp.execute("x = 1 + 1")
    assert result == ""


def test_state_persists_across_execute_calls() -> None:
    with SubprocessPythonInterpreter() as interp:
        interp.execute("x = 42")
        interp.execute("y = x * 2")
        result = interp.execute("print(y)")
    assert result.strip() == "84"


# ----- SUBMIT semantics --------------------------------------------------------


def test_submit_returns_FinalOutput_with_kwargs() -> None:
    with SubprocessPythonInterpreter() as interp:
        result = interp.execute("SUBMIT(answer=42, status='done')")
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": 42, "status": "done"}


def test_submit_after_state_setup() -> None:
    with SubprocessPythonInterpreter() as interp:
        interp.execute("computed = sum(range(10))")
        result = interp.execute("SUBMIT(answer=computed)")
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": 45}


def test_submit_accepts_positional_args_when_output_fields_registered() -> None:
    """Regression for the v7 SUBMIT-positional bug.

    dspy.RLM tells the model the SUBMIT signature is positional (e.g.
    ``SUBMIT(solution)``) per ``runner.js``. dspy injects
    ``interpreter.output_fields`` then resets ``_tools_registered=False`` so the
    next ``execute()`` re-registers with our worker. Without dynamic SUBMIT
    rebuilding, ``SUBMIT(solution_dict)`` raises TypeError → dspy loop never
    terminates → 0/3 accuracy in the local 3Q smoke. This test pins the fix.
    """
    with SubprocessPythonInterpreter() as interp:
        # Simulate what dspy.RLM._inject_execution_context does:
        interp.output_fields = [{"name": "solution", "type": "str"}]
        interp._tools_registered = False
        result = interp.execute("SUBMIT({'Q1': 1, 'Q2': 2})")
    assert isinstance(result, FinalOutput)
    assert result.output == {"solution": {"Q1": 1, "Q2": 2}}


def test_submit_positional_then_keyword_mix_with_output_fields() -> None:
    """Multi-field SUBMIT supports positional, keyword, and mixed calls."""
    with SubprocessPythonInterpreter() as interp:
        interp.output_fields = [
            {"name": "answer", "type": "int"},
            {"name": "label", "type": "str"},
        ]
        interp._tools_registered = False
        # All-positional, in registered order
        r1 = interp.execute("SUBMIT(7, 'lucky')")
    assert isinstance(r1, FinalOutput)
    assert r1.output == {"answer": 7, "label": "lucky"}

    with SubprocessPythonInterpreter() as interp:
        interp.output_fields = [
            {"name": "answer", "type": "int"},
            {"name": "label", "type": "str"},
        ]
        interp._tools_registered = False
        # Mixed positional + keyword
        r2 = interp.execute("SUBMIT(7, label='lucky')")
    assert isinstance(r2, FinalOutput)
    assert r2.output == {"answer": 7, "label": "lucky"}


# ----- Error handling (mirrors dspy contract) ---------------------------------


def test_syntax_error_raised_unwrapped() -> None:
    """Per dspy contract: SyntaxError is NOT wrapped in CodeInterpreterError."""
    with SubprocessPythonInterpreter() as interp:
        with pytest.raises(SyntaxError):
            interp.execute("def bad(:")


def test_runtime_error_wrapped_in_CodeInterpreterError() -> None:
    with SubprocessPythonInterpreter() as interp:
        with pytest.raises(CodeInterpreterError) as exc_info:
            interp.execute("undefined_variable")
    assert "NameError" in str(exc_info.value)


def test_division_by_zero_wrapped() -> None:
    with SubprocessPythonInterpreter() as interp:
        with pytest.raises(CodeInterpreterError) as exc_info:
            interp.execute("1 / 0")
    assert "ZeroDivisionError" in str(exc_info.value) or "division" in str(exc_info.value).lower()


def test_state_survives_runtime_error() -> None:
    """Worker stays alive after user-code error so subsequent execute() works."""
    with SubprocessPythonInterpreter() as interp:
        interp.execute("kept = 99")
        with pytest.raises(CodeInterpreterError):
            interp.execute("undefined_var")
        result = interp.execute("print(kept)")
    assert result.strip() == "99"


# ----- Tool callbacks (worker -> host JSON-RPC tool_call) ---------------------


def test_tool_callback_invoked_from_user_code() -> None:
    seen_kwargs: dict = {}

    def echo(msg: str) -> str:
        seen_kwargs["msg"] = msg
        return f"echoed:{msg}"

    with SubprocessPythonInterpreter(tools={"echo": echo}) as interp:
        result = interp.execute("print(echo(msg='hi'))")

    assert seen_kwargs == {"msg": "hi"}
    assert "echoed:hi" in result


def test_tool_callback_returning_dict_is_json_serialized() -> None:
    """Per dspy contract: dict/list returns are JSON-serialized so user code can json.loads them."""
    def get_data(key: str) -> dict:
        return {"k": key, "v": 42}

    with SubprocessPythonInterpreter(tools={"get_data": get_data}) as interp:
        result = interp.execute(
            "import json\n"
            "raw = get_data(key='abc')\n"
            "obj = json.loads(raw)\n"
            "print(obj['v'])"
        )
    assert result.strip() == "42"


def test_tool_callback_can_be_added_after_construction() -> None:
    """dspy.RLM does interpreter.tools.update({...}) after construction."""
    interp = SubprocessPythonInterpreter()
    interp.tools["greet"] = lambda name: f"hi {name}"
    try:
        interp.start()
        result = interp.execute("print(greet(name='alice'))")
        assert "hi alice" in result
    finally:
        interp.shutdown()


def test_tool_error_propagates_to_user_code() -> None:
    """A failing tool surfaces as a runtime error in user code."""
    def failing_tool() -> str:
        raise ValueError("tool boom")

    with SubprocessPythonInterpreter(tools={"failing_tool": failing_tool}) as interp:
        with pytest.raises(CodeInterpreterError) as exc_info:
            interp.execute("failing_tool()")
    assert "boom" in str(exc_info.value).lower()


# ----- Lifecycle ---------------------------------------------------------------


def test_start_is_idempotent() -> None:
    interp = SubprocessPythonInterpreter()
    try:
        interp.start()
        interp.start()  # second call must not raise
        result = interp.execute("print('alive')")
        assert "alive" in result
    finally:
        interp.shutdown()


def test_shutdown_is_idempotent() -> None:
    interp = SubprocessPythonInterpreter()
    interp.start()
    interp.shutdown()
    interp.shutdown()  # second call must not raise


def test_execute_lazily_starts_subprocess() -> None:
    interp = SubprocessPythonInterpreter()
    try:
        result = interp.execute("print('lazy')")
        assert "lazy" in result
    finally:
        interp.shutdown()


# ----- Hardened spawn (Prove-It regression test for Fabric crash class) -------


def test_spawn_uses_explicit_PYTHONPATH_to_fabric_rlm_parent() -> None:
    """Regression test for the Fabric `No module named fabric_rlm._worker` crash.

    Root cause: subprocess inherited a sys.path that didn't include our package dir.
    Fix: spawn with explicit env={'PYTHONPATH': <fabric_rlm parent>}.

    This test asserts the env wiring without depending on Fabric.
    """
    import fabric_rlm

    expected_parent = os.path.dirname(os.path.dirname(os.path.abspath(fabric_rlm.__file__)))

    interp = SubprocessPythonInterpreter()
    try:
        interp.start()
        env = interp._spawn_env  # exposed for test introspection
    finally:
        interp.shutdown()

    pythonpath = env.get("PYTHONPATH", "")
    parts = pythonpath.split(os.pathsep)
    assert expected_parent in parts, (
        f"Expected fabric_rlm parent dir {expected_parent!r} in PYTHONPATH, "
        f"got: {pythonpath!r}"
    )


def test_spawn_uses_sys_executable() -> None:
    interp = SubprocessPythonInterpreter()
    try:
        interp.start()
        cmd = interp._spawn_cmd  # exposed for test introspection
    finally:
        interp.shutdown()
    assert cmd[0] == sys.executable


def test_startup_self_test_records_worker_file() -> None:
    """After start(), host has verified the worker is the expected fabric_rlm._worker."""
    import fabric_rlm._worker as worker_mod

    interp = SubprocessPythonInterpreter()
    try:
        interp.start()
        worker_file = interp._worker_self_test["worker_file"]
    finally:
        interp.shutdown()

    assert os.path.normpath(worker_file) == os.path.normpath(worker_mod.__file__)


def test_broken_pythonpath_fails_fast_at_start(tmp_path) -> None:
    """Prove-It: with a busted Python that can't find fabric_rlm, start() raises clearly.

    Strategy: spawn with -E -S (no env vars, no site-packages, no .pth files) AND
    a cwd that has no fabric_rlm AND empty PYTHONPATH. The worker import must fail;
    start() must surface a clear CodeInterpreterError, not hang.
    """
    interp = SubprocessPythonInterpreter(cwd=str(tmp_path), start_timeout=10.0)
    interp._extra_python_args = ["-E", "-S"]
    interp._force_pythonpath = ""
    with pytest.raises(CodeInterpreterError) as exc_info:
        interp.start()
    msg = str(exc_info.value).lower()
    assert "worker" in msg or "fabric_rlm" in msg or "module" in msg


def test_nonexistent_python_fails_fast_at_start() -> None:
    """Second Prove-It: a totally invalid Python executable surfaces clearly at start()."""
    interp = SubprocessPythonInterpreter(
        python="/nonexistent/path/to/python_xyz_xyz", start_timeout=5.0
    )
    with pytest.raises((CodeInterpreterError, FileNotFoundError, OSError)):
        interp.start()


# ----- dspy.RLM end-to-end smoke (real dspy loop driving our interpreter) -----


def test_drives_dspy_RLM_end_to_end_with_mock_lm() -> None:
    """Acceptance test: dspy.predict.RLM accepts our interpreter and runs to SUBMIT."""
    import dspy
    from dspy.predict import RLM

    class StubLM(dspy.LM):
        def __init__(self) -> None:
            super().__init__(model="stub", model_type="chat")
            self._calls = 0

        def __call__(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
            self._calls += 1
            return [
                "[[ ## reasoning ## ]]\nLet me solve.\n\n"
                "[[ ## code ## ]]\n```python\nSUBMIT(answer=42)\n```\n\n"
                "[[ ## completed ## ]]\n"
            ]

    class MySig(dspy.Signature):
        """Solve the thing."""

        question: str = dspy.InputField()
        answer: int = dspy.OutputField()

    stub = StubLM()
    interp = SubprocessPythonInterpreter()
    try:
        with dspy.context(lm=stub):
            rlm = RLM(signature=MySig, sub_lm=stub, interpreter=interp, max_iterations=3)
            prediction = rlm(question="what?")
    finally:
        interp.shutdown()

    assert prediction.answer == 42
