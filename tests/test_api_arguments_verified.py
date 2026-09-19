"""verified_task argument contracts over CSV data and real default workers.

Only model responses are scripted; constructor spies delegate to the real RLM.
No authored skills, interpreter replacements, or model API calls are needed.
"""

from __future__ import annotations

import csv
import os
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fabric_rlm import RLM, verified_task


TASK = "Sum every amount in the supplied CSV, including the last row."
READ_ROWS = """import csv, os
assert 'worker_marker' not in globals(), 'Worker state leaked between solves'
assert options == {'note': 'original input'}, 'Input mutation leaked between solves'
worker_marker = 'private worker state'
worker_pid = os.getpid()
options['note'] = 'worker-local mutation'
with open(source, newline='', encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))
"""


class ScriptedModel:
    def __init__(self, codes):
        self.codes = list(codes)
        self.messages = []

    def __call__(self, *, messages):
        self.messages.append(deepcopy(messages))
        assert self.codes, "Unexpected model call: script exhausted"
        return SimpleNamespace(
            content=f"```python\n{self.codes.pop(0)}\n```",
            usage={"prompt_tokens": 100, "completion_tokens": 10},
        )


def solve_code(subset="rows", submit="SUBMIT(amount=total, note='all rows')"):
    return (
        READ_ROWS
        + f"total = sum(int(row['amount']) for row in {subset})\n"
        + "print('computed amount', total)\n"
        + submit
    )


@pytest.fixture
def csv_inputs(tmp_path):
    source = tmp_path / "amounts.csv"
    source.write_text("item,amount\na,17\nb,25\nc,8\n", encoding="utf-8")
    with source.open(newline="", encoding="utf-8") as stream:
        amounts = [int(row["amount"]) for row in csv.DictReader(stream)]
    assert amounts == [17, 25, 8]
    assert sum(amounts) == 50 and sum(amounts[:-1]) == 42
    return {"source": str(source), "options": {"note": "original input"}}


@pytest.fixture(autouse=True)
def production_guards(monkeypatch, _claim_provenance_default):
    # Undo the suite's fake-interpreter opt-out, not the production checks.
    monkeypatch.delenv("FABRIC_RLM_CLAIM_PROVENANCE", raising=False)
    monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)


def test_typed_outputs_default_to_first_field_and_skip_stronger_model(csv_inputs):
    outputs = {"amount": int, "note": str}
    lm = ScriptedModel([
        solve_code(submit="SUBMIT(amount=total, note='alpha')"),
        solve_code(submit="SUBMIT(amount=total, note='omega')"),
    ])
    strong = ScriptedModel([])
    result = verified_task(
        task=TASK, inputs=csv_inputs, outputs=outputs, lm=lm,
        reconcile_lm=strong, reconcile_max_turns=2, max_turns=1, timeout=30,
    )

    assert result.verdict == "agree" and result.ok
    assert result.answer_a == result.answer_b == "50"
    assert result.result.payload == {"amount": 50, "note": "alpha"}
    assert len(result.attempts) == len(lm.messages) == 2
    assert not strong.messages
    assert result.selected_attempt_index == 0
    assert result.result is result.attempts[0]
    assert not result.reconciliation_attempted
    assert not result.reconciliation_succeeded and not result.fallback_used
    assert result.reconciliation_reason is None
    assert result.total_prompt_tokens == 200
    assert result.total_completion_tokens == 20
    for attempt, messages in zip(result.attempts, lm.messages):
        assert attempt.submitted and attempt.failure_reason is None
        assert attempt.turns[0].stdout.strip() == "computed amount 50"
        assert attempt.final_state["worker_pid"] != os.getpid()
        prompt = "\n".join(message["content"] for message in messages)
        assert TASK in prompt
        assert "Analyst 1 answered:" not in prompt


def test_typed_outputs_repair_csv_total_before_comparing(csv_inputs):
    lm = ScriptedModel([
        solve_code(submit="SUBMIT(amount=str(total), note='all rows')"),
        "SUBMIT(amount=total, note='all rows')",
        solve_code(),
    ])
    result = verified_task(
        TASK, inputs=csv_inputs, outputs={"amount": int, "note": str},
        lm=lm, max_turns=2, timeout=30,
    )

    assert result.verdict == "agree"
    assert result.result.payload["amount"] == 50
    assert [len(attempt.turns) for attempt in result.attempts] == [2, 1]
    assert len(lm.messages) == 3 and not lm.codes
    assert "amount" in " ".join(result.attempts[0].turns[0].validation_errors)
    assert result.attempts[0].turns[1].turn_type == "validation_repair"


def test_computed_disagreement_preserves_inputs_callbacks_and_custom_guidance(csv_inputs):
    outputs = {"amount": int, "note": str}
    original_inputs = deepcopy(csv_inputs)
    lm = ScriptedModel([solve_code("rows[:-1]"), solve_code()])
    strong = ScriptedModel([solve_code()])
    guidance = "\nCUSTOM CHECK: reopen the CSV and include its final row.\n"
    validated = []
    contexts = []

    def validate(payload):
        # Type-only intentionally accepts the computed 42, allowing disagreement.
        assert type(payload["amount"]) is int
        validated.append(payload["amount"])

    def validate_context(payload, context):
        assert context["inputs"] == original_inputs
        assert context["state"]["worker_pid"] != os.getpid()
        contexts.append(payload["amount"])

    settings = dict(
        lm=lm, max_turns=1, timeout=30,
        output_validator=validate, output_validator_context=validate_context,
    )
    original_settings = dict(settings)
    with patch.object(RLM, "from_task", wraps=RLM.from_task) as factory:
        result = verified_task(
            TASK, inputs=csv_inputs, outputs=outputs, reconcile_lm=strong,
            reconcile_max_turns=2, reconcile_guidance=guidance, **settings,
        )

    assert result.verdict == "reconciled" and result.ok
    assert (result.answer_a, result.answer_b) == ("42", "50")
    assert result.result.payload["amount"] == 50
    assert result.result is result.attempts[2]
    assert result.selected_attempt_index == 2
    assert result.reconciliation_attempted and result.reconciliation_succeeded
    assert not result.fallback_used
    assert result.reconciliation_reason == "disagreement"
    assert validated == contexts == [42, 50, 50]
    assert [attempt.max_turns for attempt in result.attempts] == [1, 1, 2]
    assert len(lm.messages) == 2 and len(strong.messages) == 1
    assert result.total_prompt_tokens == 300
    assert result.total_completion_tokens == 30
    assert csv_inputs == original_inputs and settings == original_settings
    assert outputs == {"amount": int, "note": str}
    assert factory.call_count == 3
    for index, call in enumerate(factory.call_args_list):
        assert call.kwargs["inputs"] is csv_inputs
        assert call.kwargs["outputs"] is outputs
        assert call.kwargs["output_validator"] is validate
        assert call.kwargs["output_validator_context"] is validate_context
        assert call.kwargs["timeout"] == 30
        assert call.kwargs["lm"] is (strong if index == 2 else lm)
        if index < 2:
            assert call.args == (TASK,)
    for messages in lm.messages:
        prompt = "\n".join(message["content"] for message in messages)
        assert guidance not in prompt and "Analyst 1 answered:" not in prompt
        assert "private worker state" not in prompt
    third_prompt = "\n".join(message["content"] for message in strong.messages[0])
    assert TASK + guidance in third_prompt
    assert "Analyst 1 answered: 42" in third_prompt
    assert "Analyst 2 answered: 50" in third_prompt
    assert "private worker state" not in third_prompt
    assert [attempt.turns[0].stdout.strip() for attempt in result.attempts] == [
        "computed amount 42", "computed amount 50", "computed amount 50",
    ]
    assert all(attempt.submitted and attempt.failure_reason is None for attempt in result.attempts)


def test_explicit_field_selects_total_instead_of_matching_first_field(csv_inputs):
    submit = "SUBMIT(note='same note', total=total)"
    lm = ScriptedModel([solve_code("rows[:-1]", submit), solve_code(submit=submit)])
    strong = ScriptedModel([solve_code(submit=submit)])
    result = verified_task(
        TASK, inputs=csv_inputs, outputs={"note": str, "total": int},
        field_name="total", lm=lm, reconcile_lm=strong, max_turns=1, timeout=30,
    )

    assert result.verdict == "reconciled"
    assert (result.answer_a, result.answer_b) == ("42", "50")
    assert result.result.payload == {"note": "same note", "total": 50}
    assert result.selected_attempt_index == 2 and result.reconciliation_succeeded
    assert len(strong.messages) == 1


def test_default_agreement_compares_numeric_and_string_outputs_as_strings(csv_inputs):
    lm = ScriptedModel([
        solve_code(submit="SUBMIT(amount=total)"),
        solve_code(submit="SUBMIT(amount=str(total))"),
    ])
    strong = ScriptedModel([])
    result = verified_task(
        TASK, inputs=csv_inputs, outputs=["amount"], lm=lm,
        reconcile_lm=strong, max_turns=1, timeout=30,
    )

    assert [attempt.payload["amount"] for attempt in result.attempts] == [50, "50"]
    assert result.answer_a == result.answer_b == "50"
    assert result.verdict == "agree" and not strong.messages
    assert not result.reconciliation_attempted


@pytest.mark.parametrize("custom_tolerance", [False, True])
def test_exact_default_comparison_and_custom_tolerance(csv_inputs, custom_tolerance):
    lm = ScriptedModel([
        solve_code(submit="SUBMIT(amount=total)"),
        solve_code(submit="rounded = total + 0.01\nprint(rounded)\nSUBMIT(amount=rounded)"),
    ])
    strong = ScriptedModel([solve_code(submit="SUBMIT(amount=total)")])
    compared = []

    def within_tolerance(a, b):
        assert isinstance(a, str) and isinstance(b, str)
        compared.append((a, b))
        return abs(float(a) - float(b)) < 0.02

    result = verified_task(
        TASK, inputs=csv_inputs, outputs=["amount"], lm=lm,
        agree=within_tolerance if custom_tolerance else None,
        reconcile_lm=strong, max_turns=1, timeout=30,
    )

    assert (result.answer_a, result.answer_b) == ("50", "50.01")
    assert result.result.payload["amount"] == 50
    assert result.verdict == ("agree" if custom_tolerance else "reconciled")
    assert compared == ([("50", "50.01")] if custom_tolerance else [])
    assert len(strong.messages) == (0 if custom_tolerance else 1)
    assert result.reconciliation_succeeded is (not custom_tolerance)
    assert result.selected_attempt_index == (0 if custom_tolerance else 2)


@pytest.mark.parametrize("reconcile_turns", [None, 2])
def test_reconcile_budget_inherits_or_repairs_rejected_csv_answer(csv_inputs, reconcile_turns):
    lm = ScriptedModel([solve_code("rows[:-1]"), solve_code()])
    strong = ScriptedModel([
        solve_code("rows[:-1]"),
        "total = sum(int(row['amount']) for row in rows)\n"
        "print('repaired amount', total)\nSUBMIT(amount=total, note='all rows')",
    ])
    validated = []
    compared = []

    def source_truth(payload):
        with open(csv_inputs["source"], newline="", encoding="utf-8") as stream:
            expected = sum(int(row["amount"]) for row in csv.DictReader(stream))
        validated.append(payload["amount"])
        assert payload["amount"] == expected, "CSV total must include every row"

    def agree(a, b):
        compared.append((a, b))
        return True

    result = verified_task(
        TASK, inputs=csv_inputs, outputs={"amount": int, "note": str},
        lm=lm, max_turns=1, timeout=30, output_validator=source_truth,
        agree=agree, reconcile_lm=strong, reconcile_max_turns=reconcile_turns,
    )

    budget = reconcile_turns or 1
    assert result.verdict == "reconciled" and result.ok
    assert result.result.payload["amount"] == 50
    assert result.answer_a == "" and result.answer_b == "50"
    assert not compared, "An unavailable answer must bypass even a permissive comparator"
    assert result.reconciliation_reason == "unavailable_answer"
    assert result.reconciliation_attempted
    assert [len(attempt.turns) for attempt in result.attempts] == [1, 1, budget]
    assert [attempt.max_turns for attempt in result.attempts] == [1, 1, budget]
    assert len(lm.messages) == 2 and len(strong.messages) == budget
    assert validated == ([42, 50, 42, 50] if reconcile_turns else [42, 50, 42])
    assert not result.attempts[0].submitted
    assert result.attempts[0].failure_reason == "output_validation_failed"
    assert result.attempts[0].trajectory.metadata["verifier_repair_history"][0]["skill"] == "output_validator"
    if reconcile_turns:
        assert result.reconciliation_succeeded and not result.fallback_used
        assert result.result is result.attempts[2] and result.selected_attempt_index == 2
        assert result.result.turns[1].turn_type == "verifier_repair"
        assert "CSV total must include every row" in strong.messages[1][-1]["content"]
        assert not strong.codes
    else:
        assert result.fallback_used and not result.reconciliation_succeeded
        assert result.result is result.attempts[1] and result.selected_attempt_index == 1
        assert result.attempts[2].failure_reason == "output_validation_failed"
        assert not result.attempts[2].submitted
        assert len(strong.codes) == 1, "Repair must not run past the inherited budget"


def test_no_overrides_reuses_model_and_full_default_budget_without_mutating_arguments(csv_inputs):
    # Spend the entire default budget doing real CSV work, not a constructor-only check.
    read_again = (
        "with open(source, newline='', encoding='utf-8') as stream:\n"
        "    rows = list(csv.DictReader(stream))\n"
        "total = sum(int(row['amount']) for row in rows)\nprint(total)\n"
    )
    lm = ScriptedModel([
        solve_code("rows[:-1]"), solve_code(), solve_code(submit=""),
        *[read_again for _ in range(18)],
        read_again + "SUBMIT(amount=total, note='all rows')",
    ])
    outputs = {"amount": int, "note": str}
    inputs_before = deepcopy(csv_inputs)
    settings = {"lm": lm, "timeout": 30}
    with patch.object(RLM, "from_task", wraps=RLM.from_task) as factory:
        result = verified_task(TASK, inputs=csv_inputs, outputs=outputs, **settings)

    assert result.verdict == "reconciled" and result.reconciliation_succeeded
    assert result.result.payload == {"amount": 50, "note": "all rows"}
    assert [attempt.max_turns for attempt in result.attempts] == [20, 20, 20]
    assert [len(attempt.turns) for attempt in result.attempts] == [1, 1, 20]
    assert len(lm.messages) == 22 and not lm.codes
    assert all(turn.error is None for attempt in result.attempts for turn in attempt.turns)
    assert result.total_prompt_tokens == 2200 and result.total_completion_tokens == 220
    assert result.selected_attempt_index == 2 and not result.fallback_used
    assert csv_inputs == inputs_before
    assert outputs == {"amount": int, "note": str}
    assert settings == {"lm": lm, "timeout": 30}
    assert factory.call_count == 3
    for call in factory.call_args_list:
        assert call.kwargs["lm"] is lm
        assert call.kwargs["inputs"] is csv_inputs
        assert call.kwargs["outputs"] is outputs
        assert "max_turns" not in call.kwargs and "engine" not in call.kwargs
