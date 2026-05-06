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


# ---------------------------------------------------------------------------
# input_previews — see LIB-NEW-1 in validation_10/LIBRARY_BUGS.md
# ---------------------------------------------------------------------------


def test_input_previews_callable_renders_into_initial_user_message() -> None:
    """A callable in input_previews is evaluated against the bound value
    and its returned string is injected into the first user message."""
    lm = ScriptedLM(["```python\nSUBMIT(answer='ok')\n```"])
    captured = {}

    def preview(value):
        captured["value"] = value
        return f"PREVIEW_TEXT_FOR={value!r}"

    rlm = RLM.from_task(
        "Return ok.",
        outputs=["answer"],
        lm=lm,
        max_turns=2,
        timeout=5,
        input_previews={"x": preview},
    )

    result = rlm.run({"x": 42})

    assert result.submitted
    initial_user = lm.messages[0][1]["content"]
    assert "INPUT PREVIEWS" in initial_user
    assert "PREVIEW_TEXT_FOR=42" in initial_user
    assert captured["value"] == 42


def test_input_previews_static_string_works() -> None:
    """A non-callable string in input_previews is used verbatim."""
    lm = ScriptedLM(["```python\nSUBMIT(answer='ok')\n```"])
    rlm = RLM.from_task(
        "Return ok.",
        outputs=["answer"],
        lm=lm,
        max_turns=2,
        timeout=5,
        input_previews={"x": "STATIC_PREVIEW"},
    )

    rlm.run({"x": 1})

    initial_user = lm.messages[0][1]["content"]
    assert "STATIC_PREVIEW" in initial_user


def test_input_previews_callable_raising_falls_back_gracefully() -> None:
    """A callable that raises must not crash the run; the offending input
    is omitted from the preview block (other previews still rendered)."""
    lm = ScriptedLM(["```python\nSUBMIT(answer='ok')\n```"])

    def bad(_value):
        raise RuntimeError("boom")

    rlm = RLM.from_task(
        "Return ok.",
        outputs=["answer"],
        lm=lm,
        max_turns=2,
        timeout=5,
        input_previews={"a": bad, "b": lambda v: f"B={v}"},
    )

    result = rlm.run({"a": 1, "b": 2})

    assert result.submitted
    initial_user = lm.messages[0][1]["content"]
    assert "B=2" in initial_user
    assert "boom" not in initial_user


def test_input_previews_unset_keeps_legacy_message() -> None:
    """Back-compat: omitting input_previews entirely yields the
    pre-existing initial user message with no preview block."""
    lm = ScriptedLM(["```python\nSUBMIT(answer='ok')\n```"])
    rlm = RLM.from_task(
        "Return ok.",
        outputs=["answer"],
        lm=lm,
        max_turns=2,
        timeout=5,
    )

    rlm.run({"x": 1})

    initial_user = lm.messages[0][1]["content"]
    assert "INPUT PREVIEWS" not in initial_user


def test_input_previews_empty_dict_keeps_legacy_message() -> None:
    """Back-compat: an empty input_previews dict is equivalent to None."""
    lm = ScriptedLM(["```python\nSUBMIT(answer='ok')\n```"])
    rlm = RLM.from_task(
        "Return ok.",
        outputs=["answer"],
        lm=lm,
        max_turns=2,
        timeout=5,
        input_previews={},
    )

    rlm.run({"x": 1})

    initial_user = lm.messages[0][1]["content"]
    assert "INPUT PREVIEWS" not in initial_user


def test_input_previews_for_unbound_input_is_silently_dropped() -> None:
    """A preview keyed on a name not in bound inputs must not appear."""
    lm = ScriptedLM(["```python\nSUBMIT(answer='ok')\n```"])
    rlm = RLM.from_task(
        "Return ok.",
        outputs=["answer"],
        lm=lm,
        max_turns=2,
        timeout=5,
        input_previews={"not_bound": "SHOULD_NOT_APPEAR"},
    )

    rlm.run({"x": 1})

    initial_user = lm.messages[0][1]["content"]
    assert "SHOULD_NOT_APPEAR" not in initial_user
    assert "INPUT PREVIEWS" not in initial_user


def test_input_previews_callable_returning_non_string_is_dropped() -> None:
    """Callable returning a non-string or whitespace-only value is
    treated as no preview — the section must not render."""
    lm = ScriptedLM(["```python\nSUBMIT(answer='ok')\n```"])
    rlm = RLM.from_task(
        "Return ok.",
        outputs=["answer"],
        lm=lm,
        max_turns=2,
        timeout=5,
        input_previews={"x": lambda _: 123, "y": lambda _: "   "},
    )

    rlm.run({"x": 1, "y": 2})

    initial_user = lm.messages[0][1]["content"]
    assert "INPUT PREVIEWS" not in initial_user
    assert "123" not in initial_user

