# Lossless `SUBMIT` payloads

## Problem

Before this change, the worker serialized both iterative namespace snapshots
and final `SUBMIT(...)` payloads through the same bounded `freeze()` call.
That default intentionally truncates strings after 2,000 characters and
collections after 200 items so model feedback stays small.

Those limits are correct for state snapshots but incorrect for final answers.
A valid large CSV or table could therefore be silently shortened before the
parent runtime received it.

Minimal reproduction on an affected version:

```python
from fabric_rlm import Interpreter

with Interpreter() as interpreter:
    result = interpreter.execute(
        "rows = [[i, f'value-{i}'] for i in range(500)]\n"
        "SUBMIT(csv='header\\n' + 'x' * 10000, rows=rows)"
    )

assert len(result.submit_payload["csv"]) == 10007
assert len(result.submit_payload["rows"]) == 500
```

The first assertion previously observed a truncation marker after 2,000
characters. The second observed only 200 rows plus a synthetic truncation
record.

## Design

`freeze()` accepts `None` for either bound:

```python
freeze(value, max_string_length=None, max_collection_items=None)
```

The worker uses this lossless mode only for `_SubmitSignal.payload`, then
measures the standalone payload's UTF-8 JSON representation. Namespace
snapshots continue using the existing 2,000-character and 200-item defaults.

Final payloads default to a 64 MiB byte limit. Configure it on the public
facade or either interpreter:

```python
from fabric_rlm import Interpreter, RLM, SubprocessPythonInterpreter

rlm = RLM(
    signature="question -> answer",
    lm=my_lm,
    max_submit_bytes=128 * 1024 * 1024,
)

legacy = Interpreter(max_submit_bytes=128 * 1024 * 1024)
dspy_interpreter = SubprocessPythonInterpreter(
    max_submit_bytes=128 * 1024 * 1024
)
```

The limit must be a positive integer. Encoding stops as soon as a payload
crosses the configured limit; the payload is never partially returned or
marked as submitted.

This separation preserves both requirements:

- iterative state remains bounded and prompt-cache friendly;
- declared final outputs cross the worker boundary without data loss;
- unexpectedly large outputs cannot consume unbounded protocol memory.

Unsupported Python objects still become opaque JSON-safe markers. Lossless
means no size truncation for supported values; it does not make arbitrary
objects serializable.

The byte cap applies to the final payload itself, not the surrounding protocol
envelope, state snapshot, stdout, or stderr. File-backed transport is not
enabled automatically: it introduces path authorization, cleanup, lifetime,
and cross-host portability concerns that require a separate artifact contract.
The current conversion still materializes the JSON-safe Python object before
measuring it, so this cap bounds protocol output size rather than worst-case
worker peak memory. Container/process memory limits remain the hard backstop
for hostile in-memory objects.

## Recommended table submission

Prefer structured tables rather than pre-rendered CSV:

```python
SUBMIT(
    prediction={
        "columns": ["id", "amount"],
        "rows": [[1, 12.5], [2, 18.0]],
    }
)
```

Structured transport avoids a second CSV parse, preserves scalar types, and
prevents quoting or newline errors. Render CSV only at the final filesystem
boundary.

String output remains supported and is now lossless:

```python
SUBMIT(prediction="id,amount\r\n1,12.5\r\n2,18.0\r\n")
```

## Compatibility

- `freeze()` and `snapshot()` retain their existing bounded defaults.
- Existing small `SUBMIT` payloads are unchanged.
- Existing CSV-string submissions remain valid.
- Final supported strings and collections are no longer truncated.
- Payloads above the default 64 MiB limit fail explicitly rather than risking
  unbounded memory and IPC usage.

## Regression tests

Run the focused tests:

```bash
pytest -q \
  tests/test_serializers.py \
  tests/test_interpreter.py \
  tests/test_subprocess_interpreter.py
```

The coverage includes:

- bounded snapshot behavior remains unchanged;
- explicit unbounded `freeze()` behavior;
- exact UTF-8 JSON byte-boundary enforcement, including non-ASCII values;
- invalid and non-positive limit rejection;
- 10,000-character final strings;
- 500-row final collections;
- legacy and DSPy-compatible interpreter surfaces.

Run the complete suite before release:

```bash
pytest -q
```
