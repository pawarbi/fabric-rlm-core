"""The model is told about predict() only when it can work, and sees the real error when it cannot.

Found in Fabric (#104): every shipped example passes a live ``FabricLM(...)``
object, which cannot cross into the worker, so no sub-LM is configured. The
system prompt advertised ``predict`` anyway and the PDF skill recommends it.
In eight traced PDF runs the model called it first and lost that turn, and in
two more it lost a second turn because ``predict_sync`` replaced the real
error with "cannot reuse already awaited coroutine".
"""

from __future__ import annotations

import asyncio

import pytest

from fabric_rlm import RLM, _worker
from fabric_rlm.prompts import build_system_prompt

from test_api_arguments_engines import _ScriptedLM


def _prompt(**kwargs):
    return build_system_prompt(inline_task="Count the rows.", inline_outputs=["n"], inputs={"rows": [1, 2]}, **kwargs)


def test_the_helpers_are_advertised_by_default_and_when_a_sub_lm_exists():
    assert _prompt() == _prompt(sub_lm_available=True)
    assert "calls the configured sub-LM" in _prompt()


def test_the_helpers_are_not_advertised_when_no_sub_lm_is_configured():
    prompt = _prompt(sub_lm_available=False)
    assert "calls the configured sub-LM" not in prompt and "predict_sync(signature" not in prompt
    assert "`predict` and `predict_sync` are not available in this run" in prompt
    assert "SUBMIT(**fields)" in prompt and "File(path)" in prompt           # the rest of the API is still there
    assert len(prompt) < len(_prompt())                                        # and the prompt got shorter, not longer


def _system_prompt_of_a_run(**kwargs):
    lm = _ScriptedLM(["SUBMIT(n=2)"])
    result = RLM.task("Count the rows.", inputs={"rows": [1, 2]}, outputs={"n": int}, lm=lm, max_turns=2, **kwargs).run()
    assert result.submitted
    messages = lm.calls[0]["messages"]
    return next(m["content"] for m in messages if m["role"] == "system")


def test_a_run_with_a_live_lm_object_does_not_offer_predict():
    assert "not available in this run" in _system_prompt_of_a_run()


def test_a_run_with_a_sub_lm_spec_still_offers_predict():
    # The spec is resolved only when generated code calls predict, so no model is contacted here.
    assert "calls the configured sub-LM" in _system_prompt_of_a_run(sub_lm="openai/not-called-in-this-test")


def test_predict_sync_surfaces_the_real_error_inside_a_running_loop(monkeypatch):
    monkeypatch.setattr(_worker, "_lm_spec", None)
    monkeypatch.setattr(_worker, "_lm_instance", None)

    async def generated_code():
        return _worker.predict_sync("text -> label", text="hello")     # what a model writes inside the worker's loop

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(generated_code())
    assert "not available in this run" in str(raised.value)
    assert "cannot reuse already awaited coroutine" not in str(raised.value)
