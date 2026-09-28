"""The output contract rejects empty containers and checks list/dict element types (issue #120).

A semantic-model run that fought column names for ten turns submitted {} for
sales_by_region and came back submitted=True. An empty container now fails any
required field unless the caller allows it (allow_empty), and dict[str, float]
or list[dict] can be declared and are checked element by element.
"""

from __future__ import annotations

import logging

import pytest

from fabric_rlm import RLM
from fabric_rlm.prompts import build_system_prompt
from fabric_rlm.runtime import validate_submit_payload


class ScriptedLM:
    def __init__(self, turns):
        self.turns = list(turns)
        self.prompts: list[str] = []

    def __call__(self, *, messages):
        self.prompts.append(str(messages[-1].get("content", "")) if messages else "")
        return self.turns.pop(0) if self.turns else self.last

    last = "```python\nSUBMIT(sales_by_region={'North': 10.0})\n```"


def code(text):
    return f"```python\n{text}\n```"


# -- the rules -----------------------------------------------------------------------------------------

def test_an_empty_container_fails_any_required_field_by_default():
    for value in ({}, [], ()):
        errors = validate_submit_payload({"sales_by_region": value}, ["sales_by_region"]).errors
        assert errors and "empty" in errors[0]


def test_allow_empty_by_name_or_for_every_non_core_field():
    assert validate_submit_payload({"anomalies": []}, ["anomalies"], allow_empty={"anomalies"}).ok
    assert validate_submit_payload({"anomalies": []}, ["anomalies"], allow_empty=True).ok
    # core final-output names stay strict under True, unless named
    assert not validate_submit_payload({"answer": []}, ["answer"], allow_empty=True).ok
    assert validate_submit_payload({"answer": []}, ["answer"], allow_empty={"answer"}).ok


def test_element_types_are_checked_and_the_first_bad_one_is_named():
    fields, types = ["by_region"], {"by_region": dict[str, float]}
    assert validate_submit_payload({"by_region": {"North": 1.5}}, fields, types).ok
    (error,) = validate_submit_payload({"by_region": {"North": 1.5, "South": "n/a"}}, fields, types).errors
    assert "dict[str, float]" in error and "value for key 'South' got str" in error
    (error,) = validate_submit_payload({"rows": [{"a": 1}, 3]}, ["rows"], {"rows": list[dict]}).errors
    assert "item 1 got int" in error
    assert validate_submit_payload({"names": ["a", "b"]}, ["names"], {"names": list[str]}).ok


@pytest.mark.parametrize("bad", [int | str, tuple[int, ...], set[str], dict[str], list])
def test_unsupported_types_are_refused_at_construction(bad):
    if bad is list:   # a bare class is fine
        RLM.task("x", inputs={}, outputs={"a": bad}, lm=ScriptedLM([]))
        return
    with pytest.raises(TypeError, match="list\\[...\\]"):
        RLM.task("x", inputs={}, outputs={"a": bad}, lm=ScriptedLM([]))


def test_the_prompt_shows_the_parameterized_type():
    prompt = build_system_prompt(inline_task="t", inline_outputs=["by_region"],
                                 inline_output_types={"by_region": dict[str, float]})
    assert "by_region: dict[str, float]" in prompt


# -- in a run -----------------------------------------------------------------------------------------

def test_an_empty_submission_is_sent_back_and_the_filled_one_accepted(caplog):
    lm = ScriptedLM([code("SUBMIT(sales_by_region={})"), code("SUBMIT(sales_by_region={'North': 10.0})")])
    with caplog.at_level(logging.WARNING, logger="fabric_rlm.runtime"):
        result = RLM.task("Sales by region.", inputs={}, outputs={"sales_by_region": dict[str, float]},
                          lm=lm, max_turns=3, timeout=60).run()
    assert result.submitted and result.payload == {"sales_by_region": {"North": 10.0}}
    assert any("is an empty dict" in prompt for prompt in lm.prompts)
    assert any("allow_empty" in record.message for record in caplog.records)


def test_allow_empty_accepts_a_valid_empty_answer():
    lm = ScriptedLM([code("SUBMIT(anomalies=[])")])
    result = RLM.task("List anomalies.", inputs={}, outputs={"anomalies": list}, lm=lm, max_turns=2,
                      timeout=60, allow_empty={"anomalies"}).run()
    assert result.submitted and result.payload == {"anomalies": []}


def test_a_typo_in_allow_empty_is_an_error_not_a_silent_no_op():
    lm = ScriptedLM([code("SUBMIT(anomalies=[])")])
    rlm = RLM.task("List anomalies.", inputs={}, outputs={"anomalies": list}, lm=lm, allow_empty={"anomaly"})
    with pytest.raises(ValueError, match="not output fields"):
        rlm.run()


def test_a_wrongly_typed_element_is_sent_back():
    lm = ScriptedLM([code("SUBMIT(sales_by_region={'North': '10'})"), code("SUBMIT(sales_by_region={'North': 10.0})")])
    result = RLM.task("Sales by region.", inputs={}, outputs={"sales_by_region": dict[str, float]},
                      lm=lm, max_turns=3, timeout=60).run()
    assert result.submitted and result.payload == {"sales_by_region": {"North": 10.0}}
    assert any("value for key 'North' got str" in prompt for prompt in lm.prompts)


def test_the_hint_is_given_again_in_a_later_run(caplog):
    for _ in range(2):
        lm = ScriptedLM([code("SUBMIT(sales_by_region={})"), code("SUBMIT(sales_by_region={'North': 10.0})")])
        with caplog.at_level(logging.WARNING, logger="fabric_rlm.runtime"):
            caplog.clear()
            RLM.task("Sales by region.", inputs={}, outputs={"sales_by_region": dict}, lm=lm, max_turns=3, timeout=60).run()
        assert sum("allow_empty" in record.message for record in caplog.records) == 1
