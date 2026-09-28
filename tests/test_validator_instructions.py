"""A validator's own ``instructions`` reach the model with the task (issue #122).

Live, with a rule stated only inside the validator, the first answer was right in
0 of 7 runs; stated in the task, 2 of 3. A validator can carry the rule it checks
as an ``instructions`` string and the runtime shows it under the task, in both
engines, without repeating text the task already contains.
"""

from __future__ import annotations

import pytest

from fabric_rlm import RLM
from fabric_rlm.prompts import build_system_prompt

RULE = "Exclude canceled orders from every revenue figure."
SUBMIT = "```python\nSUBMIT(revenue=10)\n```"


class SystemPromptLM:
    def __init__(self):
        self.system: list[str] = []

    def __call__(self, *, messages):
        self.system.append(str(messages[0].get("content", "")))
        return SUBMIT


def check(payload):
    assert payload["revenue"] == 10, "revenue must exclude canceled orders"


class Checks:
    instructions = RULE

    def __call__(self, payload):
        return None


def run(task="Give revenue.", **kwargs):
    lm = SystemPromptLM()
    result = RLM.task(task, outputs={"revenue": int}, lm=lm, max_turns=2, timeout=60, **kwargs).run()
    assert result.submitted
    return lm.system[0]


def test_function_attribute_reaches_the_prompt_under_the_task():
    check.instructions = RULE
    try:
        prompt = run(output_validator=check)
    finally:
        del check.instructions
    assert "checked after SUBMIT against these rules" in prompt
    assert prompt.index("Give revenue.") < prompt.index(RULE) < prompt.index("## Required output fields")


def test_object_attribute_and_context_validator():
    def with_context(payload, context):
        return None

    with_context.instructions = "Report revenue in BRL."
    prompt = run(output_validator=Checks(), output_validator_context=with_context)
    assert RULE in prompt and "Report revenue in BRL." in prompt


def test_text_already_in_the_task_is_not_repeated():
    prompt = run(task="Give revenue.\n\n" + RULE.replace(" every ", "  every\n"), output_validator=Checks())
    assert "checked after SUBMIT" not in prompt


def test_the_same_rule_on_both_validators_appears_once():
    def with_context(payload, context):
        return None

    with_context.instructions = RULE
    prompt = run(output_validator=Checks(), output_validator_context=with_context)
    assert prompt.count(RULE) == 1


@pytest.mark.parametrize("value", [None, "", "   ", 42])
def test_no_usable_instructions_leaves_the_prompt_unchanged(value):
    class V:
        instructions = value

        def __call__(self, payload):
            return None

    assert run(output_validator=V()) == run()


def test_prompt_builder_is_unchanged_without_rules():
    base = build_system_prompt(inline_task="t", inline_outputs=["x"])
    assert build_system_prompt(inline_task="t", inline_outputs=["x"], validator_rules=None) == base
    assert build_system_prompt(inline_task="t", inline_outputs=["x"], validator_rules=" ") == base


def test_v7_engine_appends_the_rules_to_the_signature_instructions():
    pytest.importorskip("dspy")
    rlm = RLM.task("Give revenue.", outputs={"revenue": int}, lm=SystemPromptLM(), output_validator=Checks())
    signature = rlm._build_dspy_signature(("revenue",), extra_instructions=rlm._validator_rules_text())
    assert RULE in signature.instructions and "Give revenue." in signature.instructions
