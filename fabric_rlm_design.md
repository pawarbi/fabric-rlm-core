# fabric-rlm — Design Document

**Status:** Validated prototype. Substrate proven end-to-end in Microsoft Fabric
Python notebook against multimodal invoice extraction with multi-turn error
recovery. Ready to extract into a reusable library.

**Audience:** A developer (or coding agent like Claude Code) building this into
a pip-installable package. This doc contains the full reference implementation,
all design decisions, the lessons learned from validation, and the open
questions that remain.

---

## 1. What this library does

`fabric-rlm` lets you run **Recursive Language Models** ([Zhang, Kraska,
Khattab 2025](https://arxiv.org/abs/2512.24601v1)) inside any Python
environment that can spawn a subprocess — including Microsoft Fabric notebooks,
Databricks notebooks, vanilla Jupyter, and plain scripts.

An RLM is an LM that solves a task by **writing Python code in a REPL loop**
rather than by being orchestrated through a fixed pipeline. The model itself
decides what code to run next, sees the output, and iterates until it calls
`SUBMIT()` with its final answer. The model can call a sub-LM via `predict()`
for sub-tasks.

This is a port of the architecture from
[`predict-rlm`](https://github.com/Trampoline-AI/predict-rlm) which uses a
Deno+Pyodide WASM sandbox. WASM doesn't run cleanly in restricted notebook
environments like Fabric, so this library replaces it with a persistent Python
subprocess. Same API surface (`predict`, `SUBMIT`, `File`), different sandbox.

### Why a separate library

`predict-rlm` is excellent in environments where you control the runtime. In
managed analytics platforms — Fabric, Databricks, Snowpark, SageMaker
notebooks — you don't. Network egress, subprocess restrictions, lack of Deno,
and event-loop quirks all prevent direct use. Every team using these platforms
hits the same wall. This library is the wall removed.

---

## 2. Public API

Three layers, increasing in flexibility.

### Layer 1: Quick start (DSPy signature)

```python
import dspy
from fabric_rlm import RLM, File

class ExtractInvoices(dspy.Signature):
    """Extract structured data from invoice images. For each image, use
    predict() to get vendor, line items, and totals. Validate that line
    items sum to the invoice total. SUBMIT once all are processed."""
    images: list[File] = dspy.InputField()
    extractions: list[dict] = dspy.OutputField()
    validation_summary: dict = dspy.OutputField()

rlm = RLM(
    ExtractInvoices,
    lm="azure/gpt-4o",       # outer LM — writes code
    sub_lm="azure/gpt-4o-mini", # inner LM — called via predict()
)

result = rlm(images=[File("invoice1.png"), File("invoice2.png")])
print(result.extractions)
print(result.trajectory)  # full turn-by-turn replay
```

### Layer 2: Inline task (no Signature class)

```python
from fabric_rlm import RLM, File

rlm = RLM.from_task(
    task="Read each invoice image, extract totals, return as a list of dicts.",
    inputs={"images": [File("inv1.png"), File("inv2.png")]},
    outputs=["extractions", "validation_summary"],
    lm="azure/gpt-4o",
    sub_lm="azure/gpt-4o-mini",
)
result = rlm.run()
```

### Layer 3: Direct interpreter (no agent loop)

For users who want the sandbox without the LM-driven REPL — useful for
running untrusted user code in a notebook, or for testing the worker alone.

```python
from fabric_rlm import Interpreter

with Interpreter() as interp:
    interp.execute("x = 1 + 1")
    out = interp.execute("print(x); y = x * 10")
    print(out.stdout)        # "2\n"
    print(out.state)         # {"x": 2, "y": 20}
```

### Configuration

Three LM backends ship in v1, registered as plugins:

```python
from fabric_rlm.lm import FabricLM, OpenAILM, AnthropicLM

lm = FabricLM("gpt-5")                        # auto-discovers Fabric endpoint
lm = OpenAILM("gpt-4o", api_key=os.environ[...])
lm = AnthropicLM("claude-sonnet-4-5")
```

`RLM` accepts either a string (resolved through the default backend) or any
`LM` instance.

---

## 3. Architecture

```
┌──────────────────────────────────────────────┐
│ User notebook / script                        │
│   rlm = RLM(MySignature, lm=...)              │
│   result = rlm(inputs)                        │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│ RLM driver (Python, in user's process)        │
│   - Builds system prompt from Signature       │
│   - Loop: outer LM → extract code → execute   │
│           → format feedback → next turn       │
│   - Records trajectory                        │
│   - Returns dspy.Prediction-like result       │
└─────────────────┬────────────────────────────┘
                  │ JSON over stdin/stdout
                  ▼
┌──────────────────────────────────────────────┐
│ Worker (separate Python subprocess)           │
│   - Persistent namespace dict (`ns`)          │
│   - Injected: File, predict, SUBMIT, SKILLs   │
│   - Top-level await via                       │
│     ast.PyCF_ALLOW_TOP_LEVEL_AWAIT            │
│   - SUBMIT raises BaseException sentinel      │
│   - Snapshots namespace per turn (JSON-safe)  │
└──────────────────────────────────────────────┘
```

### SKILL playbooks

`RLM(..., skills=["validation"], enable_skill_autoloading=True)` adds a
task-generic SKILL index and optional preloaded playbooks to the system prompt.
Worker code can also call `list_skills()` and `load_skill("validation")`.
Bundled SKILLs live under `fabric_rlm.skills` and cover validation,
error-handling, and long-context multiple-choice reasoning. For post-run
improvement loops, `python -m fabric_rlm.skill_distiller trajectory.jsonl`
creates a local Markdown report proposing SKILL updates from failure evidence.

### Why subprocess (and not the alternatives)

| Approach | Verdict | Reason |
|---|---|---|
| `exec()` in driver process | **Rejected** | Code can touch driver's secrets, Spark session, notebookutils. Real production risk. |
| Pyodide WASM (predict-rlm) | **Rejected** | Doesn't run in Fabric. Big install. Pyodide-incompatible packages won't work. |
| Fabric User Data Functions | **Rejected** | Stateless across turns; namespace must round-trip as JSON. Forces all values to be JSON-serializable on every turn. |
| `notebookutils.notebook.run` | **Rejected** | Inherits parent execution context including identity and secrets. No real isolation. |
| `jupyter_client` second kernel | **Rejected** | Heavier dependency, more moving parts, doesn't gain anything over subprocess. |
| **Subprocess + stdin/stdout** | **Accepted** | OS-level isolation, persistent namespace, inherits installed packages, works anywhere Python works. |

### Why JSON namespace snapshots (and not pickle)

Pickle would let any Python object survive turn-to-turn. JSON forces the
model to keep the namespace introspectable. Two reasons this matters:

1. **Debuggability.** You can `cat` the snapshot file and read what the
   model was thinking. With pickle you get a binary blob.
2. **Forces good behavior.** If the model wants a `pd.DataFrame` across
   turns, it writes it to disk and keeps the path string in namespace.
   This is closer to how a careful human writes notebook code.

The escape hatch: non-serializable values are recorded as
`{"__type__": ..., "__repr__": ..., "__serializable__": false}` so the
model still knows the variable exists but the value is opaque.

### Why `BaseException` for SUBMIT

User code commonly contains `try: ... except Exception: ...`. If `SUBMIT`
raised a normal `Exception`, that try block would catch it and the loop
would never end. `BaseException` bypasses normal exception handling — the
only thing that catches it is the worker's main loop.

### Why `ast.PyCF_ALLOW_TOP_LEVEL_AWAIT`

The model writes code like `result = await predict(...)`. Without this
flag, that's a syntax error outside an `async def`. With this flag, the
code compiles and we wrap the resulting coroutine in `asyncio.run()`.

This is much cleaner than the alternative of wrapping every code block
in `async def __turn__():` — the model's code compiles exactly as written.

---

## 4. Reference implementation

This section contains the complete validated worker and driver. A coding
agent can use these as the starting point and refactor into proper modules.

### 4.1 Worker (`fabric_rlm/_worker.py`)

This file is the subprocess. The driver writes it to disk and launches it
with `python -u _worker.py`. It speaks JSON over stdin/stdout.

```python
"""
Worker subprocess for fabric-rlm.

Protocol (one JSON object per line on stdin/stdout):

  → {"op": "exec", "code": "..."}
  ← {"ok": true, "submitted": false, "stdout": "...", "stderr": "...",
     "state": {...}}

  → {"op": "exec", "code": "SUBMIT(answer=42)"}
  ← {"ok": true, "submitted": true, "submit_payload": {"answer": 42},
     "stdout": "", "stderr": "", "state": {...}}

  → {"op": "exec", "code": "1/0"}
  ← {"ok": false, "submitted": false, "stdout": "", "stderr": "",
     "error": "Traceback...", "state": {...}}

  → {"op": "shutdown"}
  ← (process exits)
"""

import sys, json, traceback, ast, inspect, asyncio, io, contextlib, types, base64
from pathlib import Path

# ------------------------------------------------------------------
# Namespace and built-in API exposed to user code
# ------------------------------------------------------------------

ns = {}
_LM_FACTORY = None  # set by configure_lm()


class File:
    """Wraps a file path. Available in user code as `File(path)`."""
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.name = Path(self.path).name

    def read_bytes(self):
        return Path(self.path).read_bytes()

    def read_text(self, encoding="utf-8"):
        return Path(self.path).read_text(encoding=encoding)

    def as_data_uri(self, mime="image/png"):
        b64 = base64.b64encode(self.read_bytes()).decode()
        return f"data:{mime};base64,{b64}"

    def toDict(self):
        return {"name": self.name, "path": self.path}

    def __repr__(self):
        return f"File({self.path})"


class _SubmitSignal(BaseException):
    """Raised by SUBMIT() — bypasses user try/except blocks."""
    def __init__(self, payload):
        self.payload = payload


def SUBMIT(**kwargs):
    """Finish the run with the given output fields."""
    raise _SubmitSignal(kwargs)


async def predict(signature, **kwargs):
    """Call the sub-LM. Available in user code as `await predict(...)`."""
    import dspy
    if _LM_FACTORY is None:
        raise RuntimeError("Sub-LM not configured. Driver must call configure_lm first.")
    lm = _LM_FACTORY()
    sig = dspy.Signature(signature) if isinstance(signature, str) else signature
    predictor = dspy.Predict(sig)
    with dspy.context(lm=lm):
        return await predictor.acall(**kwargs)


def configure_lm(factory_fn):
    """Driver calls this once via a special op to install the sub-LM factory."""
    global _LM_FACTORY
    _LM_FACTORY = factory_fn


ns["File"] = File
ns["SUBMIT"] = SUBMIT
ns["predict"] = predict


# ------------------------------------------------------------------
# Namespace serialization
# ------------------------------------------------------------------

_INJECTED_NAMES = {"File", "SUBMIT", "predict"}


def freeze(v):
    """Recursively convert a value to a JSON-safe form."""
    if hasattr(v, "toDict"):
        return freeze(v.toDict())
    if hasattr(v, "model_dump"):
        return freeze(v.model_dump())
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [freeze(x) for x in v]
    if isinstance(v, dict):
        return {str(k): freeze(x) for k, x in v.items()}
    return {
        "__type__": type(v).__name__,
        "__repr__": repr(v)[:300],
        "__serializable__": False,
    }


def snapshot(truncate_strings_at=200):
    """Build a JSON-safe view of the user namespace for the driver."""
    out = {}
    for k, v in ns.items():
        if k.startswith("_") or k in _INJECTED_NAMES:
            continue
        if callable(v) or isinstance(v, (types.ModuleType, type)):
            continue
        try:
            frozen = freeze(v)
            if isinstance(frozen, dict) and frozen.get("__serializable__") is False:
                out[k] = frozen
                continue
            json.dumps(frozen)
            if isinstance(frozen, str) and len(frozen) > truncate_strings_at:
                out[k] = (frozen[:truncate_strings_at]
                          + f"...<truncated, total {len(frozen)} chars>")
            else:
                out[k] = frozen
        except Exception as e:
            out[k] = {"__error__": str(e), "__serializable__": False}
    return out


# ------------------------------------------------------------------
# Code execution
# ------------------------------------------------------------------

async def run_code(code):
    compiled = compile(
        code, "<rlm_worker>", "exec",
        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
    )
    result = eval(compiled, ns, ns)
    if inspect.isawaitable(result):
        return await result
    return result


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main():
    for line in sys.stdin:
        msg = json.loads(line)
        op = msg.get("op")

        if op == "exec":
            stdout, stderr = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    asyncio.run(run_code(msg["code"]))
                response = {
                    "ok": True, "submitted": False,
                    "stdout": stdout.getvalue(), "stderr": stderr.getvalue(),
                    "state": snapshot(),
                }
            except _SubmitSignal as sig:
                response = {
                    "ok": True, "submitted": True,
                    "submit_payload": freeze(sig.payload),
                    "stdout": stdout.getvalue(), "stderr": stderr.getvalue(),
                    "state": snapshot(),
                }
            except Exception:
                response = {
                    "ok": False, "submitted": False,
                    "stdout": stdout.getvalue(), "stderr": stderr.getvalue(),
                    "error": traceback.format_exc(),
                    "state": snapshot(),
                }
            print(json.dumps(response), flush=True)

        elif op == "configure_lm":
            # Driver sends pickled factory bytes, base64-encoded.
            # See driver-side `Interpreter.configure_lm` for how this is sent.
            import pickle
            factory = pickle.loads(base64.b64decode(msg["factory_b64"]))
            configure_lm(factory)
            print(json.dumps({"ok": True}), flush=True)

        elif op == "shutdown":
            break


if __name__ == "__main__":
    main()
```

### 4.2 Interpreter (`fabric_rlm/interpreter.py`)

Thin wrapper around the subprocess. This is also the public Layer 3 API.

```python
import subprocess, sys, json, os, signal, select, base64, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

WORKER_PATH = Path(__file__).parent / "_worker.py"


@dataclass
class ExecResult:
    ok: bool
    submitted: bool
    stdout: str
    stderr: str
    state: dict
    error: Optional[str] = None
    submit_payload: Optional[dict] = None


class Interpreter:
    """A persistent Python subprocess sandbox.

    Use as a context manager:

        with Interpreter() as interp:
            r = interp.execute("x = 1 + 1")
            print(r.state)  # {"x": 2}
    """

    def __init__(self, timeout: float = 300.0, python: Optional[str] = None):
        self.timeout = timeout
        self.python = python or sys.executable
        self.proc: Optional[subprocess.Popen] = None

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("Already started")
        self.proc = subprocess.Popen(
            [self.python, "-u", str(WORKER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        return self

    def execute(self, code: str) -> ExecResult:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("Worker is not running")
        self._send({"op": "exec", "code": code})
        raw = self._recv()
        return ExecResult(
            ok=raw["ok"],
            submitted=raw.get("submitted", False),
            stdout=raw.get("stdout", ""),
            stderr=raw.get("stderr", ""),
            state=raw.get("state", {}),
            error=raw.get("error"),
            submit_payload=raw.get("submit_payload"),
        )

    def configure_lm(self, factory_fn):
        """Install a sub-LM factory in the worker.

        factory_fn must be picklable — typically a top-level function or a
        class with __call__ that constructs and returns a dspy.LM instance.
        """
        b64 = base64.b64encode(pickle.dumps(factory_fn)).decode()
        self._send({"op": "configure_lm", "factory_b64": b64})
        return self._recv()

    def shutdown(self):
        if self.proc and self.proc.poll() is None:
            try:
                self._send({"op": "shutdown"})
                self.proc.wait(timeout=5)
            except Exception:
                self.kill()

    def kill(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            self.proc.wait()

    def _send(self, msg: dict):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _recv(self) -> dict:
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout)
        if not ready:
            self.kill()
            raise TimeoutError(f"Worker timed out after {self.timeout}s")
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"Worker exited without response:\n{stderr}")
        return json.loads(line)

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.shutdown()
```

### 4.3 Driver (`fabric_rlm/runtime.py`)

The agent loop. This is the Layer 1/2 API.

```python
import inspect
from dataclasses import dataclass, field
from typing import Optional, Union, Any

from .interpreter import Interpreter, ExecResult
from .prompts import build_system_prompt, build_initial_user_message
from .lm import resolve_lm


@dataclass
class TurnRecord:
    turn: int
    code: str
    stdout: str
    stderr: str
    error: Optional[str]
    submitted: bool
    state_keys: list


@dataclass
class RLMResult:
    submitted: bool
    payload: Optional[dict]
    trajectory: list  # of TurnRecord
    final_state: dict

    def __getattr__(self, name):
        if self.payload and name in self.payload:
            return self.payload[name]
        raise AttributeError(name)


class RLM:
    def __init__(
        self,
        signature=None,
        *,
        lm,
        sub_lm=None,
        max_turns: int = 10,
        verbose: bool = True,
    ):
        self.signature = signature
        self.outer_lm = resolve_lm(lm)
        self.sub_lm_spec = sub_lm or lm   # picklable spec, not the LM instance
        self.max_turns = max_turns
        self.verbose = verbose

    @classmethod
    def from_task(cls, task: str, inputs: dict, outputs: list[str], **kwargs):
        """Build an inline RLM from a plain task description."""
        instance = cls.__new__(cls)
        instance.signature = None
        instance._inline_task = task
        instance._inline_outputs = outputs
        instance._inline_inputs = inputs
        instance.outer_lm = resolve_lm(kwargs["lm"])
        instance.sub_lm_spec = kwargs.get("sub_lm", kwargs["lm"])
        instance.max_turns = kwargs.get("max_turns", 10)
        instance.verbose = kwargs.get("verbose", True)
        return instance

    def __call__(self, **inputs) -> RLMResult:
        return self.run(inputs)

    def run(self, inputs: Optional[dict] = None) -> RLMResult:
        inputs = inputs or getattr(self, "_inline_inputs", {})

        with Interpreter() as interp:
            interp.configure_lm(_make_lm_factory(self.sub_lm_spec))

            # Bind inputs into the worker namespace
            input_setup = self._build_input_setup_code(inputs)
            interp.execute(input_setup)

            sys_prompt = build_system_prompt(
                signature=self.signature,
                inline_task=getattr(self, "_inline_task", None),
                inline_outputs=getattr(self, "_inline_outputs", None),
                inputs=inputs,
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": build_initial_user_message(inputs)},
            ]

            trajectory = []
            payload = None

            for turn in range(1, self.max_turns + 1):
                self._log(f"\n=== Turn {turn}/{self.max_turns} ===")

                response = self.outer_lm(messages=messages)
                response_text = response[0] if isinstance(response, list) else response
                code = _extract_code(response_text)
                self._log(f"[code]\n{code[:1000]}")

                if _looks_truncated(response_text):
                    feedback = (
                        "Your previous response was truncated mid-code. "
                        "Rewrite that turn in under 30 lines. "
                        "Skip helper functions; call predict() directly."
                    )
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": feedback})
                    continue

                result = interp.execute(code)
                self._log(f"[stdout]\n{result.stdout[:1000]}")

                trajectory.append(TurnRecord(
                    turn=turn, code=code,
                    stdout=result.stdout, stderr=result.stderr,
                    error=result.error, submitted=result.submitted,
                    state_keys=list(result.state.keys()),
                ))

                if result.submitted:
                    payload = result.submit_payload
                    self._log(f"[SUBMITTED] {list(payload.keys())}")
                    break

                feedback = self._format_feedback(result, turn)
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": feedback})

            return RLMResult(
                submitted=payload is not None,
                payload=payload,
                trajectory=trajectory,
                final_state=trajectory[-1].state_keys if trajectory else [],
            )

    def _build_input_setup_code(self, inputs: dict) -> str:
        """Generate code that re-creates the input values inside the worker."""
        # Simple approach: pickle inputs, base64, unpickle in worker.
        # For File objects, reconstruct via path.
        import pickle, base64
        b64 = base64.b64encode(pickle.dumps(inputs)).decode()
        return (
            "import pickle, base64\n"
            f"_inputs = pickle.loads(base64.b64decode({b64!r}))\n"
            "globals().update(_inputs)\n"
        )

    def _format_feedback(self, result: ExecResult, turn: int) -> str:
        parts = [f"REPL output from turn {turn}:\n```\n{result.stdout[:3000]}\n```"]
        if not result.ok:
            parts.append(f"\nERROR:\n```\n{result.error[:1500]}\n```\nWrite a recovery turn.")
        elif result.stderr:
            parts.append(f"\nstderr:\n```\n{result.stderr[:500]}\n```")
        parts.append(f"\nState keys: {', '.join(result.state.keys())}")
        parts.append("\nContinue. Write the next code block, or call SUBMIT() if done.")
        return "".join(parts)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


def _extract_code(text: str) -> str:
    if "```python" in text:
        s = text.index("```python") + len("```python")
        rest = text[s:]
        if "```" in rest:
            return rest[:rest.index("```")].strip()
    if "```" in text:
        s = text.index("```") + 3
        rest = text[s:]
        if "```" in rest:
            return rest[:rest.index("```")].strip()
    return text.strip()


def _looks_truncated(text: str) -> bool:
    """Detect when LM output cut off mid-code (only opening fence)."""
    return "```" in text and text.count("```") < 2


def _make_lm_factory(spec):
    """Build a picklable factory function from an LM spec."""
    # spec is either a string ("azure/gpt-4o") or a dict of kwargs.
    # The factory is what gets pickled into the worker.
    def factory():
        from .lm import resolve_lm
        return resolve_lm(spec)
    return factory
```

### 4.4 Prompts (`fabric_rlm/prompts.py`)

This is the **single most important file in the library**. The validation
test showed the model will write 700-line preambles if you let it. The
prompt is what stops that.

```python
SYSTEM_PROMPT_TEMPLATE = """You are an RLM (Recursive Language Model) running in a Python REPL.

You solve the task by writing Python code. Each block you write is executed
in a persistent namespace, then stdout is returned to you. Variables persist
across turns. Build your answer incrementally.

## Sandbox API

`File(path)` — wraps a file path. Useful methods:
    File("foo.png").as_data_uri()        # for multimodal predict()
    File("foo.txt").read_text()
    File("foo.bin").read_bytes()

`await predict(signature, instructions=None, pydantic_schemas=None, **kwargs)` — call the sub-LM. Examples:

    # Plain text
    r = await predict(
        "question: str -> answer: str",
        instructions="Answer in one sentence.",
        question="...",
    )
    print(r.answer)

    # Typed output
    r = await predict(
        "text: str -> record: Record",
        pydantic_schemas={"Record": Record},
        text="...",
    )
    print(r.record["name"])

    # Multimodal
    import dspy
    r = await predict("image: dspy.Image, q: str -> a: str", image=File("img.png"), q="...")
    print(r.a)

`SUBMIT(**fields)` — finish with your final answer. Calling SUBMIT ends
execution. You MUST call this once your answer is ready.

## Code style — CRITICAL

- Keep each turn under 40 lines of code.
- Do NOT define helper libraries, JSON parsers, or validators.
- predict() returns a structured object — access fields with .field_name.
- Inline simple operations. One turn = one focused action.
- ALWAYS close your ```python ... ``` fence.
- Avoid putting triple-backtick characters in string literals (it confuses
  the response parser). Use chr(96)*3 if you absolutely need them.
- Use print() liberally — it is your only window into what your code did.

## State management

- Variables persist across turns. Reuse them.
- The driver shows you the namespace keys after each turn.
- Non-JSON values (DataFrames, custom objects) appear as opaque markers
  — they still exist in namespace, you just can't see their contents.

## Recovery

- If a turn fails or returns wrong data, write a recovery turn.
- Don't repeat the same approach. Diagnose, then change.

## Task

{task_description}

## Inputs available in namespace

{input_listing}

## Required output fields for SUBMIT()

{output_listing}

Begin. Write your first code block.
"""


def build_system_prompt(signature, inline_task, inline_outputs, inputs):
    if signature is not None:
        task_description = inspect.getdoc(signature) or "(no description)"
        outputs = list(signature.output_fields.keys())
    else:
        task_description = inline_task
        outputs = inline_outputs

    input_listing = "\n".join(f"  {k}: {_describe_value(v)}" for k, v in inputs.items())
    output_listing = "\n".join(f"  - {o}" for o in outputs)

    return SYSTEM_PROMPT_TEMPLATE.format(
        task_description=task_description,
        input_listing=input_listing,
        output_listing=output_listing,
    )


def build_initial_user_message(inputs):
    return (
        "Begin. The inputs are bound as variables in your namespace. "
        "Write your first code block."
    )


def _describe_value(v):
    import inspect
    if isinstance(v, list):
        return f"list of {len(v)} items"
    if isinstance(v, dict):
        return f"dict with keys: {list(v.keys())}"
    return type(v).__name__
```

### 4.5 LM backends (`fabric_rlm/lm.py`)

```python
"""LM backend resolution. Adding a backend = adding a registration."""

import os
from typing import Union

_BACKENDS = {}


def register_backend(prefix: str, factory):
    """Register a model-prefix → factory function."""
    _BACKENDS[prefix] = factory


def resolve_lm(spec: Union[str, dict, "dspy.LM"]):
    """Turn a spec into a configured dspy.LM instance."""
    import dspy
    if isinstance(spec, dspy.LM):
        return spec
    if isinstance(spec, dict):
        return dspy.LM(**spec)
    if isinstance(spec, str):
        for prefix, factory in _BACKENDS.items():
            if spec.startswith(prefix):
                return factory(spec)
        # Default: pass through to dspy.LM directly
        return dspy.LM(spec, temperature=1.0, max_tokens=16000)
    raise TypeError(f"Unsupported LM spec: {type(spec)}")


# ----- Fabric backend -----

def _fabric_factory(model_name: str):
    """Auto-discover Fabric's built-in OpenAI endpoint."""
    import dspy
    from synapse.ml.fabric.service_discovery import get_fabric_env_config
    from synapse.ml.fabric.token_utils import TokenUtils

    env = get_fabric_env_config().fabric_env_config
    auth = TokenUtils().get_openai_auth_header()
    base = f"{env.ml_workload_endpoint}cognitive/openai"

    return dspy.LM(
        f"azure/{model_name.split('/', 1)[-1]}",
        api_key="fabric-token",
        api_base=base,
        api_version="2025-04-01-preview",
        extra_headers={"Authorization": auth},
        temperature=1.0,
        max_tokens=16000,
    )


register_backend("fabric/", _fabric_factory)


# ----- OpenAI / Anthropic via DSPy defaults -----

def OpenAILM(model: str, **kwargs):
    import dspy
    return dspy.LM(f"openai/{model}",
                   api_key=kwargs.pop("api_key", os.environ.get("OPENAI_API_KEY")),
                   temperature=1.0, max_tokens=16000, **kwargs)


def AnthropicLM(model: str, **kwargs):
    import dspy
    return dspy.LM(f"anthropic/{model}",
                   api_key=kwargs.pop("api_key", os.environ.get("ANTHROPIC_API_KEY")),
                   temperature=1.0, max_tokens=16000, **kwargs)


def FabricLM(model: str, **kwargs):
    return _fabric_factory(model)
```

---

## 5. Lessons from the validation run

These are bugs/gotchas the test surfaced. Bake them into the library.

1. **LM over-engineers by default.** First validation run produced ~700-line
   preambles defining JSON parsers, decimal converters, helper utilities. The
   prompt now explicitly forbids this. Without that prompt section the
   library is unusable.

2. **Triple-backtick collisions.** The model wrote regex patterns containing
   `` ``` `` which collided with markdown fences in its own output, causing
   responses to truncate. The prompt now warns about this and suggests
   `chr(96)*3` as the workaround. The driver also detects truncation
   (`text.count("```") < 2`) and asks for a rewrite.

3. **Recovery happens in code, not in stdout.** Don't try to detect
   "recovery" by scanning stdout for words like "fail" or "retry." It's a
   structural property: errored turn followed by successful turn = recovery.

4. **`SUBMIT` must use `BaseException`.** User code routinely contains
   broad except blocks. Plain `Exception` gets swallowed.

5. **Reasoning models enforce defaults.** `gpt-5`-class models on Azure
   require `temperature=1.0` and `max_tokens >= 16000`. The library should
   set these by default and warn if the user overrides them.

6. **Embedded raw strings break notebook cells.** When the worker source
   was passed as `r"""..."""` inline in a Jupyter cell, parsers occasionally
   failed on edge cases. The worker is now a separate file, loaded by path.

7. **Inputs need to round-trip into the worker.** The driver pickles inputs,
   base64-encodes, and the worker unpickles into globals. This works for
   `File`, primitives, lists, dicts. Complex custom classes need pickle
   support or won't survive.

8. **Fabric endpoint discovery is a one-liner.** `synapse.ml.fabric` exposes
   `get_fabric_env_config()` and `TokenUtils()` — no Key Vault round-trip
   needed.

---

## 6. Module structure

```
fabric_rlm/
├── __init__.py          # exports RLM, Interpreter, File, FabricLM, OpenAILM, AnthropicLM
├── _worker.py           # subprocess entry point (kept private)
├── interpreter.py       # Interpreter class (Layer 3 API)
├── runtime.py           # RLM driver (Layer 1/2 API)
├── prompts.py           # System prompt templates
├── lm.py                # Backend registry + Fabric/OpenAI/Anthropic
├── types.py             # File, ExecResult, TurnRecord, RLMResult dataclasses
├── cli.py               # `fabric-rlm run task.yaml` entry point
└── py.typed             # marker for type checkers

tests/
├── test_interpreter.py  # subprocess sandbox tests (no LM calls)
├── test_prompts.py      # prompt construction tests
├── test_runtime.py      # full RLM tests with mocked LM
├── test_smoke_fabric.py # gated smoke test against real Fabric LM
└── conftest.py

examples/
├── invoice_extraction/  # the validation case
├── pdf_qa/              # document Q&A
├── data_exploration/    # RLM over a Polars DataFrame
└── README.md
```

`pyproject.toml` essentials:

```toml
[project]
name = "fabric-rlm"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "dspy>=3.1.2",
]

[project.optional-dependencies]
fabric = []   # synapse.ml.fabric is pre-installed in Fabric runtime
openai = []   # dspy handles it
anthropic = []
dev = ["pytest", "pytest-asyncio"]

[project.scripts]
fabric-rlm = "fabric_rlm.cli:main"
```

---

## 7. CLI

For non-notebook use, ship a CLI that takes a YAML task spec:

```yaml
# task.yaml
task: |
  Read each invoice image and extract vendor, line items, and totals.
  Validate that line items sum to the invoice total. SUBMIT once done.

inputs:
  images:
    - !file ./invoices/inv1.png
    - !file ./invoices/inv2.png

outputs: [extractions, validation_summary]

lm: fabric/gpt-5
sub_lm: fabric/gpt-5
max_turns: 10
```

```bash
fabric-rlm run task.yaml --output result.json --trajectory trajectory.jsonl
```

Implementation in `cli.py` is straightforward — parse YAML, build `RLM`,
call `.run()`, dump outputs.

---

## 8. Testing strategy

Three layers, each independently runnable:

**Unit tests** (no network, no LM):
- `Interpreter` round-trips JSON correctly
- `freeze` handles primitives, dataclasses, Pydantic, opaque objects
- `_extract_code` handles `python`, no-language, and missing fences
- `_looks_truncated` catches single-fence outputs
- Prompt construction produces expected templates

**Mock LM integration tests:**
- A canned LM that returns predetermined responses for each turn
- Verifies the agent loop progresses, SUBMIT exits cleanly,
  truncation triggers a rewrite, errors flow back as feedback

**Smoke tests** (gated on env var, real LM calls):
- Run the invoice example end-to-end against a real Fabric LM
- Run the PDF QA example
- Assert SUBMIT called within max_turns and payload has expected keys

---

## 9. Open questions for v2

These are deliberately deferred. v1 ships without solving them.

1. **Tools beyond `predict()`.** predict-rlm has a `Skill` system that
   bundles instructions + PyPI deps + tools. For v2 we'd add a
   `tools=[func1, func2]` parameter that injects user-defined Python
   callables into the worker namespace.

2. **Concurrency inside a single turn.** Currently each turn runs one
   `asyncio.run`. The model can `asyncio.gather()` inside that, which
   works. But long-horizon tasks would benefit from multiple workers
   running in parallel. v2 could expose a `concurrency` parameter.

3. **DSPy optimizer integration.** Trajectories are structured logs.
   GEPA and MIPROv2 should be able to consume them as training signal
   for prompt optimization. Out of scope for v1.

4. **Pickle alternatives for inputs.** Some objects don't pickle
   (database connections, file handles). v2 could expose a hook for
   custom input serialization.

5. **Restricted execution mode.** Optional AST-based whitelist of
   allowed builtins for higher-trust contexts. Right now the
   subprocess provides OS-level isolation but the code inside it can
   do anything Python can do.

---

## 10. What this library is and isn't

**Is:**
- A working RLM implementation for environments where Pyodide doesn't run
- The validated subprocess + JSON-namespace + persistent-worker pattern
- A drop-in replacement for predict-rlm's API surface in those environments

**Isn't:**
- A WASM sandbox. The subprocess is OS-isolated but not WASM-isolated.
- Faster than predict-rlm. Subprocess startup + JSON round-trips add
  overhead. Acceptable for prototyping; benchmark before claiming production.
- A general-purpose code execution sandbox. It's tuned for LM-written
  code in trusted contexts.

---

## 11. Build prompt for a coding agent

If you're handing this to Claude Code or similar:

> Build the `fabric-rlm` Python package as specified in this design doc.
> Start with `_worker.py` and `interpreter.py` — verify the round-trip
> works with a hand-written `interp.execute("x = 1; print(x)")` test.
> Then add `runtime.py` and `prompts.py` and verify with a mocked LM that
> just returns `"```python\nSUBMIT(answer=42)\n```"`. Then add `lm.py`
> backends and the CLI. Write tests as you go. Use `pytest` and
> `pytest-asyncio`. Target Python 3.10+.

End of design document.
