"""Reproduce and verify lossless final-output transport."""

from fabric_rlm import Interpreter


with Interpreter() as interpreter:
    result = interpreter.execute(
        "rows = [[i, f'value-{i}'] for i in range(500)]\n"
        "csv_text = 'header\\n' + ('x' * 10000)\n"
        "SUBMIT(prediction=csv_text, rows=rows)"
    )

assert result.submit_payload is not None
assert len(result.submit_payload["prediction"]) == 10_007
assert len(result.submit_payload["rows"]) == 500
print("Lossless SUBMIT verified.")
