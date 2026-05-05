from fabric_rlm import RLM
from fabric_rlm.runtime import validate_submit_payload


class ScriptedLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.messages = []

    def __call__(self, *, messages):
        self.messages.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("No scripted responses left")
        return self.responses.pop(0)


def test_rlm_recovers_after_worker_error_and_submits() -> None:
    lm = ScriptedLM(
        [
            "```python\nvalue = 1 / 0\n```",
            "```python\nvalue = 42\nSUBMIT(answer=value)\n```",
        ]
    )
    rlm = RLM.from_task("Return 42.", outputs=["answer"], lm=lm, max_turns=3, timeout=5)

    result = rlm.run()

    assert result.submitted
    assert result.answer == 42
    assert len(result.trajectory) == 2
    assert result.trajectory[0].error and "ZeroDivisionError" in result.trajectory[0].error
    assert "Write a recovery turn" in lm.messages[1][-1]["content"]


def test_rlm_requests_rewrite_for_truncated_code_fence() -> None:
    lm = ScriptedLM(
        [
            "```python\nx = 1",
            "```python\nSUBMIT(answer=1)\n```",
        ]
    )
    rlm = RLM.from_task("Return one.", outputs=["answer"], lm=lm, max_turns=3, timeout=5)

    result = rlm.run()

    assert result.submitted
    assert result.answer == 1
    assert len(result.trajectory) == 1
    assert "truncated" in lm.messages[1][-1]["content"]


def test_rlm_returns_structured_max_turn_failure() -> None:
    lm = ScriptedLM(["```python\nx = 1\n```", "```python\nx += 1\n```"])
    rlm = RLM.from_task("Never submit.", outputs=["answer"], lm=lm, max_turns=2, timeout=5)

    result = rlm.run()

    assert not result.submitted
    assert result.failure_reason == "max_turns"
    assert result.final_state["x"] == 2


def test_submit_validation_repairs_blank_required_answer() -> None:
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(answer='   ')\n```",
            "```python\nSUBMIT(answer='repaired')\n```",
        ]
    )
    rlm = RLM.from_task("Return a non-blank answer.", outputs=["answer"], lm=lm, max_turns=2, timeout=5)

    result = rlm.run()

    assert result.submitted
    assert result.answer == "repaired"
    assert len(result.trajectory) == 2
    assert result.trajectory[0].submitted
    assert "blank string" in result.trajectory[0].validation_errors[0]
    assert "failed output validation" in lm.messages[1][-1]["content"]


def test_submit_validation_rejects_missing_required_field_at_max_turns() -> None:
    lm = ScriptedLM(["```python\nSUBMIT(other=1)\n```"])
    rlm = RLM.from_task("Return an answer.", outputs=["answer"], lm=lm, max_turns=1, timeout=5)

    result = rlm.run()

    assert not result.submitted
    assert result.failure_reason == "output_validation_failed"
    assert result.trajectory[0].validation_errors == ["Missing required output field 'answer'."]


def test_submit_validation_allows_empty_specific_collection_outputs() -> None:
    lm = ScriptedLM(["```python\nSUBMIT(citations=[])\n```"])
    rlm = RLM.from_task("Return citations, possibly empty.", outputs=["citations"], lm=lm, max_turns=1, timeout=5)

    result = rlm.run()

    assert result.submitted
    assert result.payload == {"citations": []}


def test_validate_submit_payload_rejects_empty_core_output_collections() -> None:
    validation = validate_submit_payload({"answer": []}, ["answer"])

    assert not validation.ok
    assert validation.errors == ("Required core output field 'answer' is an empty list.",)

