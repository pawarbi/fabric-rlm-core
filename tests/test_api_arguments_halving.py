"""Exercise retry budgets with real DSPy execution of CSV-derived totals."""

from __future__ import annotations

import pytest

from fabric_rlm import RLM
from test_halve_max_iter_param import _patched_rlm_factory
from test_verifier_wrapper import _ScriptedLM


@pytest.mark.parametrize(
    ("halve_max_iter_on_retry", "expected_max_iterations"),
    [(True, [8, 4, 2]), (False, [8, 8, 8])],
    ids=["halving-enabled", "halving-disabled"],
)
def test_halve_max_iter_on_retry_with_csv_total(
    monkeypatch, halve_max_iter_on_retry, expected_max_iterations
) -> None:
    captured: list[int] = []
    _patched_rlm_factory(monkeypatch, captured)
    validated: list[dict] = []

    def validate_total(payload: dict) -> None:
        validated.append(dict(payload))
        assert payload["answer"] == 50, "answer must be the CSV total, 50"

    lm = _ScriptedLM(
        [
            "import csv\n"
            "import io\n"
            "total = sum(int(row['amount']) "
            "for row in csv.DictReader(io.StringIO(csv_text)))\n"
            f"SUBMIT(answer=total + {offset}, csv_total=total)"
            for offset in (1, 1, 0)
        ]
    )
    rlm = RLM(
        signature="question, csv_text -> answer: int, csv_total: int",
        lm=lm,
        engine="v7-dspy",
        max_turns=8,
        halve_max_iter_on_retry=halve_max_iter_on_retry,
        output_validator=validate_total,
    )
    # Feedback is prepended to question, leaving the CSV input intact on retries.
    result = rlm(question="Sum the amount column.", csv_text="amount\n17\n25\n8\n")

    assert captured == expected_max_iterations
    assert lm.calls == 3
    assert validated == [
        {"answer": 51, "csv_total": 50},
        {"answer": 51, "csv_total": 50},
        {"answer": 50, "csv_total": 50},
    ]
    assert result.submitted is True, result.failure_reason
    assert result.failure_reason is None
    assert result.payload == {"answer": 50, "csv_total": 50}
    assert len(result.trajectory.metadata["verifier_repair_history"]) == 2
