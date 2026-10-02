"""ask_each(): opt-in, host-run questions over many items with a validated output schema.

The LMs here are plain callables (the same contract as ``RLM(lm=...)``), so
these run with no network. The end-to-end cases drive a real worker
subprocess with a scripted main LM.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pytest

from fabric_rlm import RLM, AskEach, DecisionLM, File
from fabric_rlm import ask_each as ask_each_module
from fabric_rlm.ask_each import AskEachError, is_throttle_error, run_ask_each
from fabric_rlm.interpreter import SubprocessPythonInterpreter
from fabric_rlm.prompts import build_system_prompt

THEMES = ["Brakes", "Steering", "Other"]
OUTPUT = {"theme": {"choices": THEMES}, "safety_critical": "bool"}


def classify(text: str) -> dict:
    theme = "brakes" if "brake" in text else ("Steering" if "steer" in text else "Other")
    return {"theme": theme, "safety_critical": "yes" if "crash" in text else "false"}


def item_of(messages) -> str:
    user = messages[-1]["content"]
    return user.split("Item:\n", 1)[1].split("\n\nYour previous answer", 1)[0] if "Item:\n" in user else user


class CallableLM:
    """A text LM: answers each prompt through ``respond(item_text, attempt, messages)``."""

    def __init__(self, respond):
        self.respond = respond
        self.lock = threading.Lock()
        self.prompts: list[list[dict]] = []
        self.attempts: dict[str, int] = {}
        self.active = 0
        self.peak = 0

    def __call__(self, *, messages):
        text = item_of(messages)
        with self.lock:
            attempt = self.attempts.get(text, 0)
            self.attempts[text] = attempt + 1
            self.prompts.append(messages)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            answer = self.respond(text, attempt, messages)
        finally:
            with self.lock:
                self.active -= 1
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, str) else json.dumps(answer)


def request(items, output=OUTPUT, **kwargs):
    return {"items": items, "question": "Classify the complaint.", "output": output, **kwargs}


# ----------------------------------------------------------------- host engine
def test_results_align_with_items_and_are_coerced_under_concurrency():
    items = [f"complaint {i}: " + ("brake failure crash" if i % 3 == 0 else "steering drift") for i in range(40)]
    lm = CallableLM(lambda text, attempt, m: classify(text))
    result, record = run_ask_each(lm, request(items, concurrency=8))
    assert [r["theme"] for r in result["results"]] == ["Brakes" if i % 3 == 0 else "Steering" for i in range(40)]
    assert result["results"][0]["safety_critical"] is True and result["results"][1]["safety_critical"] is False
    assert result["stats"]["ok"] == 40 and result["stats"]["failed"] == 0 and result["stats"]["unfinished"] == 0
    assert record["query_type"] == "ask_each" and "brake" not in json.dumps(record)   # no item text in telemetry


def test_invalid_answer_is_retried_with_the_validation_error_in_the_prompt():
    lm = CallableLM(lambda text, attempt, m: {"theme": "Wheels", "safety_critical": "no"} if attempt == 0 else classify(text))
    result, _ = run_ask_each(lm, request(["brake noise"]))
    assert result["results"][0]["theme"] == "Brakes" and result["stats"]["retried"] == 1
    assert "must be exactly one of" in lm.prompts[1][-1]["content"]


def test_an_item_that_never_validates_is_none_with_its_error_not_a_guess():
    lm = CallableLM(lambda text, attempt, m: {"theme": "Wheels", "safety_critical": "maybe"})
    result, _ = run_ask_each(lm, request(["brake noise"], retries=1))
    assert result["results"] == [None]
    assert "exactly one of" in result["errors"][0]["error"] and result["stats"]["failed"] == 1


def test_prose_around_the_json_is_tolerated_but_no_json_is_an_error():
    lm = CallableLM(lambda text, attempt, m: 'Sure. ```json\n{"n": 3}\n```' if "a" in text else "three")
    result, _ = run_ask_each(lm, request(["a", "b"], output={"n": "int"}, retries=0))
    assert result["results"][0] == {"n": 3} and result["results"][1] is None
    assert "not a JSON object" in result["errors"][0]["error"]


def test_empty_text_is_a_valid_answer_but_an_empty_choice_or_number_is_not():
    lm = CallableLM(lambda text, attempt, m: {"quote": "", "theme": "", "amount": ""})
    result, _ = run_ask_each(lm, request(["x"], output={"quote": "str", "theme": {"choices": THEMES}, "amount": "float"}, retries=0))
    assert result["results"] == [None]
    assert "`theme` is missing" in result["errors"][0]["error"] and "`amount` is missing" in result["errors"][0]["error"]
    ok, _ = run_ask_each(CallableLM(lambda t, a, m: {"quote": ""}), request(["x"], output={"quote": "str"}))
    assert ok["results"] == [{"quote": ""}]


def test_an_lm_exception_fails_only_that_item():
    lm = CallableLM(lambda text, attempt, m: RuntimeError("boom") if text == "bad" else classify(text))
    result, _ = run_ask_each(lm, request(["brake", "bad", "steer"], retries=1))
    assert result["results"][1] is None and result["results"][0]["theme"] == "Brakes"
    assert "RuntimeError: boom" in result["errors"][0]["error"] and result["stats"]["lm_errors"] == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"items": []}, "no items"),
        ({"items": "abc"}, "must be a list"),
        ({"question": " "}, "needs a question"),
        ({"output": {}}, "needs output"),
        ({"output": {"x": "date"}}, 'must be "str", "int", "float", "bool", or \\{"choices"'),
        ({"output": {"_x": "str"}}, "plain identifiers"),
        ({"output": {"x": {"choices": ["A", "a"]}}}, "duplicate choices"),
        ({"retries": 9}, "retries must be an integer from 0 to 5"),
        ({"batch_size": 0}, "batch_size must be an integer"),
        ({"max_seconds": -1}, "max_seconds must be a positive number"),
        ({"items": ["x" * 50_001]}, "Trim or chunk"),
        ({"items": ["x" * 40_000] * 2_600}, "add up to more than"),
    ],
)
def test_bad_requests_are_refused_with_a_usable_message(kwargs, message):
    with pytest.raises(AskEachError, match=message):
        run_ask_each(CallableLM(lambda t, a, m: {}), {**request(["x"]), **kwargs})


def test_the_host_caps_items_concurrency_and_time():
    lm = CallableLM(lambda text, attempt, m: (time.sleep(0.02), classify(text))[1])
    config = AskEach(max_concurrency=2, max_seconds=60, max_items=5)
    with pytest.raises(AskEachError, match="the limit is 5"):
        run_ask_each(lm, request(["brake"] * 6), config)
    result, _ = run_ask_each(lm, request(["brake"] * 5, concurrency=32, max_seconds=10_000), config)
    assert result["stats"]["concurrency"] == 2 and lm.peak <= 2
    assert result["stats"]["ok"] == 5


def test_items_left_when_the_time_limit_runs_out_are_reported_not_guessed():
    lm = CallableLM(lambda text, attempt, m: (time.sleep(0.2), classify(text))[1])
    started = time.monotonic()
    result, _ = run_ask_each(lm, request([f"brake {i}" for i in range(30)], concurrency=1, max_seconds=0.5))
    assert time.monotonic() - started < 3
    stats = result["stats"]
    assert stats["unfinished"] > 0 and stats["ok"] + stats["unfinished"] == 30 and stats["failed"] == 0
    unfinished = [e for e in result["errors"] if e["error"].startswith("not answered")]
    assert len(unfinished) == stats["unfinished"] and all(result["results"][e["index"]] is None for e in unfinished)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_a_stalled_call_does_not_hold_the_map_past_its_time_limit(batch_size):
    release = threading.Event()

    def respond(text, attempt, messages):
        if "stall" in messages[-1]["content"]:
            release.wait(30)
        return classify(text)

    lm = batch_lm(lambda numbered: (release.wait(30) if any("stall" in t for _, t in numbered) else None,
                                    json.dumps([{"n": n, **classify(t)} for n, t in numbered]))[1])
    if batch_size == 1:
        lm = CallableLM(respond)
    items = ["brake a", "steer b", "brake c", "stall d", "brake e", "steer f"]
    started = time.monotonic()
    try:
        result, _ = run_ask_each(lm, request(items, concurrency=4, max_seconds=0.5, batch_size=batch_size))
    finally:
        release.set()
    assert time.monotonic() - started < 3
    assert result["results"][3] is None and result["stats"]["unfinished"] >= 1
    assert any(e["index"] == 3 and e["error"].startswith("not answered") for e in result["errors"])
    assert result["stats"]["abandoned_calls"] == 1 and result["stats"]["usage_complete"] is False
    _wait_idle(lm)


def _wait_idle(lm, seconds=5.0):
    """Let abandoned calls finish so their threads do not outlive the test."""
    end = time.monotonic() + seconds
    while lm.active and time.monotonic() < end:
        time.sleep(0.01)
    assert lm.active == 0


def test_tokens_and_cost_are_marked_a_floor_when_a_call_is_abandoned():
    release = threading.Event()

    class PricedLM(CallableLM):
        def __init__(self):
            super().__init__(lambda text, attempt, m: (release.wait(30) if "stall" in text else None, classify(text))[1])
            self.history = []

        def __call__(self, *, messages):
            out = super().__call__(messages=messages)
            with self.lock:
                self.history.append({"usage": {"prompt_tokens": 1000, "completion_tokens": 10}, "cost": 0.01})
            return out

    lm = PricedLM()
    try:
        result, record = run_ask_each(lm, request(["brake", "stall"], concurrency=2, max_seconds=0.3))
    finally:
        release.set()
    stats = result["stats"]
    assert stats["calls"] == 2 and stats["abandoned_calls"] == 1 and stats["usage_complete"] is False
    assert stats["prompt_tokens"] == 1000                     # the abandoned call's usage never arrived
    assert record["abandoned_calls"] == 1
    summary = ask_each_module.summarize_records([record])
    assert summary["abandoned_calls"] == 1 and summary["usage_complete"] is False
    _wait_idle(lm)
    done, _ = run_ask_each(lm, request(["brake"]))
    assert done["stats"]["abandoned_calls"] == 0 and done["stats"]["usage_complete"] is True


def test_a_stalled_decision_call_does_not_hold_the_map_past_its_time_limit():
    release = threading.Event()
    finished = threading.Event()

    class Stalling(FakeDecision):
        def decide(self, state, questions):
            if "stall" in state:
                release.wait(30)
                finished.set()
            return super().decide(state, questions)

    started = time.monotonic()
    try:
        result, _ = run_ask_each(Stalling(), request(["brake", "stall"], output={"theme": {"choices": THEMES}},
                                                     max_seconds=0.5))
    finally:
        release.set()
    assert time.monotonic() - started < 3
    assert result["results"][0]["theme"] == "Brakes" and result["results"][1] is None
    assert result["stats"]["unfinished"] == 1 and result["stats"]["abandoned_calls"] == 1
    assert finished.wait(5)


def test_throttling_waits_and_slows_down_without_spending_retries(monkeypatch):
    monkeypatch.setattr(ask_each_module, "_BACKOFF_FIRST", 0.01)
    state = {"refused": 0}
    lock = threading.Lock()

    def respond(text, attempt, messages):
        with lock:
            if state["refused"] < 6:
                state["refused"] += 1
                return RuntimeError("Error code: 429 - Too Many Requests")
        return classify(text)

    lm = CallableLM(respond)
    result, record = run_ask_each(lm, request([f"brake {i}" for i in range(20)], concurrency=8, retries=0))
    assert result["stats"]["ok"] == 20 and result["stats"]["failed"] == 0
    assert result["stats"]["throttled"] == 6 and result["stats"]["retried"] == 0
    assert result["stats"]["lowest_concurrency"] < 8 and record["throttled"] == 6


@pytest.mark.parametrize("text", ["429 Too Many Requests", "RateLimitError: rate limit reached",
                                  "CapacityLimitExceeded", "The server is overloaded"])
def test_throttle_errors_are_recognised(text):
    assert is_throttle_error(RuntimeError(text))
    assert not is_throttle_error(ValueError("invalid api key"))


def test_usage_and_cost_come_from_the_lm_history_or_the_responses():
    class HistoryLM(CallableLM):
        def __init__(self):
            super().__init__(lambda text, attempt, m: classify(text))
            self.history = [{"usage": {"prompt_tokens": 999}}]   # before the map: not counted
            self.model = "openrouter/some-model"

        def __call__(self, *, messages):
            out = super().__call__(messages=messages)
            with self.lock:
                self.history.append({"usage": {"prompt_tokens": 10, "completion_tokens": 3}, "cost": 0.001})
            return [out]

    result, _ = run_ask_each(HistoryLM(), request(["brake", "steer", "x"]))
    stats = result["stats"]
    assert (stats["prompt_tokens"], stats["completion_tokens"], stats["cost"]) == (30, 9, 0.003)
    assert stats["model"] == "openrouter/some-model"

    def dict_lm(*, messages):
        return {"content": json.dumps(classify(item_of(messages))), "usage": {"input_tokens": 7, "output_tokens": 2}}

    result, _ = run_ask_each(dict_lm, request(["brake", "steer"]))
    assert (result["stats"]["prompt_tokens"], result["stats"]["completion_tokens"]) == (14, 4)


# ----------------------------------------------------------------- batching
def batch_lm(batch_answer):
    """A batching LM that reads each item's label from the prompt, so it never assumes the numbering."""
    def respond(text, attempt, messages):
        user = messages[-1]["content"]
        if "Items:\n" in user:
            parts = re.split(r"(?m)^\[(\d+)\]\n", user.split("Items:\n", 1)[1])[1:]
            numbered = [(int(parts[i]), parts[i + 1].strip()) for i in range(0, len(parts), 2)]
            return batch_answer(numbered)
        return classify(text)
    return CallableLM(respond)


def test_batches_answer_several_items_per_call_and_stay_aligned():
    lm = batch_lm(lambda numbered: json.dumps([{"n": n, **classify(t)} for n, t in numbered]))
    items = [("brake " if i % 2 else "steer ") + str(i) for i in range(25)]
    result, _ = run_ask_each(lm, request(items, batch_size=10))
    assert [r["theme"] for r in result["results"]] == ["Brakes" if i % 2 else "Steering" for i in range(25)]
    assert result["stats"]["batches"] == 3 and result["stats"]["calls"] == 3


def test_batch_items_are_numbered_from_one():
    seen = []
    lm = batch_lm(lambda numbered: (seen.append([n for n, _ in numbered]),
                                    json.dumps([{"n": n, **classify(t)} for n, t in numbered]))[1])
    run_ask_each(lm, request(["brake", "steer", "x"], batch_size=3))
    assert seen == [[1, 2, 3]]


@pytest.mark.parametrize("shift", [-1, 1])
def test_a_batch_numbered_off_by_one_is_never_used_and_each_item_is_asked_alone(shift):
    # The model numbers its answers from 0 (or 2) while the items are numbered from 1:
    # trusting the numbers would give every item its neighbour's answer.
    lm = batch_lm(lambda numbered: json.dumps([{"n": n + shift, **classify(t)} for n, t in numbered]))
    items = ["brake 0", "steer 1", "brake 2", "steer 3"]
    result, _ = run_ask_each(lm, request(items, batch_size=4))
    assert [r["theme"] for r in result["results"]] == ["Brakes", "Steering", "Brakes", "Steering"]
    assert result["stats"]["batch_fallbacks"] == 4 and result["stats"]["calls"] == 5
    singles = [m for m in lm.prompts if "Items:\n" not in m[-1]["content"]]
    assert singles and all("rejected" not in m[-1]["content"] for m in singles)


def test_an_off_by_one_batch_with_no_retries_is_reported_not_guessed():
    lm = batch_lm(lambda numbered: json.dumps([{"n": n - 1, **classify(t)} for n, t in numbered]))
    result, _ = run_ask_each(lm, request(["brake", "steer"], batch_size=2, retries=0))
    assert result["results"] == [None, None] and result["stats"]["ok"] == 0
    assert all("item numbers did not match" in e["error"] for e in result["errors"])


@pytest.mark.parametrize("corrupt", ["missing", "invalid"])
def test_a_bad_batch_row_falls_back_to_a_single_call_for_that_item_only(corrupt):
    def answer(numbered):
        rows = [{"n": n, **classify(t)} for n, t in numbered]
        if corrupt == "missing":
            rows.pop(1)
        else:
            rows[1]["theme"] = "Wheels"
        return json.dumps(rows)

    result, _ = run_ask_each(batch_lm(answer), request(["brake a", "steer b", "brake c"], batch_size=3))
    assert [r["theme"] for r in result["results"]] == ["Brakes", "Steering", "Brakes"]
    assert result["stats"]["batch_fallbacks"] == 1 and result["stats"]["calls"] == 2


def test_a_repeated_batch_number_distrusts_the_whole_batch():
    def answer(numbered):
        rows = [{"n": n, **classify(t)} for n, t in numbered]
        return json.dumps(rows + [dict(rows[1])])

    result, _ = run_ask_each(batch_lm(answer), request(["brake a", "steer b", "brake c"], batch_size=3))
    assert [r["theme"] for r in result["results"]] == ["Brakes", "Steering", "Brakes"]
    assert result["stats"]["batch_fallbacks"] == 3 and result["stats"]["calls"] == 4


def test_unparseable_batch_answer_with_no_retries_reports_every_item():
    result, _ = run_ask_each(batch_lm(lambda numbered: "no json here"), request(["brake", "steer"], batch_size=2, retries=0))
    assert result["results"] == [None, None]
    assert all("not a JSON array" in e["error"] for e in result["errors"])


# ----------------------------------------------------------------- decision models
class FakeDecision:
    is_decision_model = True
    model = "typesafe/jev-fake"
    bool_threshold = 0.5

    def __init__(self):
        self.history = []

    def decide(self, state, questions):
        self.history.append({"cost": 0.00003, "usage": {"prompt_tokens": 5}})
        answers = {}
        for name, q in questions.items():
            if q["type"] == "choice":
                answers[name] = {"choice": "Brakes" if "brake" in state else "Other", "confidence": 0.9}
            else:
                answers[name] = {"noul": 0.8 if "crash" in state else 0.1}
        return {"answers": answers}


def test_decision_model_maps_choices_and_bools_and_returns_probabilities():
    result, _ = run_ask_each(FakeDecision(), request(["brake crash", "radio"]))
    first, second = result["results"]
    assert first == {"theme": "Brakes", "theme_confidence": 0.9, "safety_critical": True, "safety_critical_p": 0.8}
    assert second["safety_critical"] is False and result["stats"]["decision_model"] is True
    assert result["stats"]["cost"] == 0.00006


def test_decision_model_refuses_text_and_number_fields_with_a_workaround():
    with pytest.raises(AskEachError, match="bucket a number into ranges"):
        run_ask_each(FakeDecision(), request(["x"], output={"amount": "float"}))


def test_decision_model_keys_only_go_to_their_own_host(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    assert DecisionLM("typesafe/jev-1.13").api_key == "or-key"
    assert DecisionLM("jev-latest", api_base="https://api.typesafe.ai/v1/systemone").api_key == "ts-key"
    assert DecisionLM("x", api_base="https://decisions.example.com/v1").api_key is None
    assert DecisionLM("x", api_base="https://decisions.example.com/v1", api_key="mine").api_key == "mine"
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert DecisionLM("typesafe/jev-1.13").api_key is None   # the TypeSafe key is never sent to OpenRouter


# ----------------------------------------------------------------- prompt and configuration
def test_off_by_default_and_the_prompt_is_unchanged():
    lm = CallableLM(lambda t, a, m: {})
    rlm = RLM.task("t", outputs={"x": int}, lm=lm)
    assert rlm.ask_each is None and rlm._ask_each_prompt({}) == ""
    assert "ask_each" not in build_system_prompt(inline_task="t", inline_outputs=["x"])


@pytest.mark.parametrize("value", [True, "openrouter/openai/gpt-5-mini", AskEach(max_seconds=60)])
def test_turning_it_on_accepts_true_an_lm_or_a_config(value):
    rlm = RLM.task("t", outputs={"x": int}, lm=CallableLM(lambda t, a, m: {}), ask_each=value)
    assert isinstance(rlm.ask_each, AskEach)
    text = rlm._ask_each_prompt({})
    assert "ask_each(items, question, output" in text and "Finding the pages" not in text


def test_document_guidance_only_with_a_document_input_and_decision_notes_only_with_a_decision_model(tmp_path: Path):
    pdf = tmp_path / "policy.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    csv = tmp_path / "rows.csv"
    csv.write_text("a\n1\n")
    rlm = RLM.task("t", outputs={"x": int}, lm=CallableLM(lambda t, a, m: {}), ask_each=True)
    assert "Finding the pages" in rlm._ask_each_prompt({"doc": File(pdf)})
    assert "Finding the pages" not in rlm._ask_each_prompt({"rows": File(csv)})
    decision = RLM.task("t", outputs={"x": int}, lm=CallableLM(lambda t, a, m: {}), ask_each=FakeDecision())
    assert "backed by a decision model" in decision._ask_each_prompt({})
    assert "backed by a decision model" not in rlm._ask_each_prompt({})


def test_document_guidance_states_the_configured_output_limit(tmp_path: Path, monkeypatch):
    import fabric_rlm.runtime as runtime_module

    pdf = tmp_path / "policy.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    rlm = RLM.task("t", outputs={"x": int}, lm=CallableLM(lambda t, a, m: {}), ask_each=True)
    monkeypatch.setattr(runtime_module, "STDOUT_FEEDBACK_LIMIT", 12345)
    text = rlm._ask_each_prompt({"doc": File(pdf)})
    assert "at most 12,345 characters of output" in text
    assert "{output_limit}" not in text
    assert '"sets_refund_window": bool' in text  # literal braces in the example survive


# ----------------------------------------------------------------- second, text model (AskEach(text_lm=...))
def _host(text_lm=None):
    from types import SimpleNamespace

    return SimpleNamespace(ask_each_lm=FakeDecision(), ask_each_config=AskEach(text_lm=text_lm), ask_each_records=[])


def test_model_text_routes_to_the_text_lm():
    from fabric_rlm.interpreter import _run_host_ask_each

    text = CallableLM(lambda t, a, m: {"quote": t.upper()})
    host = _host(text)
    result, record = _run_host_ask_each(host, {"items": ["a rule", "b rule"], "question": "Quote it.",
                                               "output": {"quote": "str"}, "model": "text"})
    assert [r["quote"] for r in result["results"]] == ["A RULE", "B RULE"]
    assert record["model_choice"] == "text" and len(text.prompts) == 2
    assert host.ask_each_lm.history == []  # the decision model was not called


def test_text_field_on_the_decision_model_points_to_model_text():
    from fabric_rlm.interpreter import _run_host_ask_each

    with pytest.raises(AskEachError, match='pass model="text"'):
        _run_host_ask_each(_host(CallableLM(lambda t, a, m: {})), {"items": ["x"], "question": "q", "output": {"quote": "str"}})
    with pytest.raises(AskEachError) as info:
        _run_host_ask_each(_host(None), {"items": ["x"], "question": "q", "output": {"quote": "str"}})
    assert 'model="text"' not in str(info.value)


@pytest.mark.parametrize("model, match", [("text", "not available"), ("fast", 'must be "default" or "text"')])
def test_model_choice_is_checked(model, match):
    from fabric_rlm.interpreter import _run_host_ask_each

    with pytest.raises(AskEachError, match=match):
        _run_host_ask_each(_host(None), {"items": ["x"], "question": "q", "output": {"ok": "bool"}, "model": model})


def test_text_lm_switches_document_reading_to_verified_quotes(tmp_path: Path):
    pdf = tmp_path / "policy.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    pages = RLM.task("t", outputs={"x": int}, lm=CallableLM(lambda t, a, m: {}),
                     ask_each=AskEach(lm=FakeDecision()))._ask_each_prompt({"doc": File(pdf)})
    quotes = RLM.task("t", outputs={"x": int}, lm=CallableLM(lambda t, a, m: {}),
                      ask_each=AskEach(lm=FakeDecision(), text_lm="openrouter/openai/gpt-5-mini"))._ask_each_prompt({"doc": File(pdf)})
    assert "read the full text of EVERY page" in pages and 'model="text"' not in pages
    assert 'model="text"' in quotes and "Separate passages with ' || '" in quotes
    assert "read the full text of EVERY page" not in quotes and "so read every flagged page" not in quotes


@pytest.mark.parametrize("bad", [dict(max_seconds=0), dict(max_concurrency=0), dict(max_items=0), dict(max_concurrency=True)])
def test_config_limits_are_checked(bad):
    with pytest.raises(ValueError):
        AskEach(**bad)


# ----------------------------------------------------------------- end to end (worker subprocess)
class ScriptedMainLM:
    def __init__(self, code: str):
        self.code = code
        self.systems: list[str] = []

    def __call__(self, *, messages):
        self.systems.append(messages[0]["content"])
        return f"```python\n{self.code}\n```"


PRESENT = "try:\n    ask_each\n    present = 1\nexcept NameError:\n    present = 0\nSUBMIT(present=present)"


def test_end_to_end_off_means_the_worker_has_no_ask_each():
    main = ScriptedMainLM(PRESENT)
    result = RLM.task("Is the helper defined?", outputs={"present": int}, lm=main, max_turns=2, timeout=60).run()
    assert result.payload == {"present": 0} and "ask_each" not in main.systems[0]


@pytest.mark.parametrize("block_network", [False, True])
def test_end_to_end_one_call_in_the_worker_with_telemetry_and_run_totals(block_network):
    code = ("r = ask_each(['brake crash', 'steer', 'radio'], 'Classify the complaint.', "
            "{'theme': ['Brakes', 'Steering', 'Other'], 'safety_critical': bool})\n"
            "SUBMIT(themes=[x['theme'] for x in r], ok=r.stats['ok'])")
    main = ScriptedMainLM(code)
    mapper = CallableLM(lambda text, attempt, m: classify(text))
    result = RLM.task("Classify.", outputs={"themes": list, "ok": int}, lm=main, ask_each=mapper,
                      max_turns=2, timeout=60, block_network=block_network).run()
    assert result.payload == {"themes": ["Brakes", "Steering", "Other"], "ok": 3}
    assert "ask_each(items, question, output" in main.systems[0]
    records = [c for t in result.turns for c in (t.source_calls or []) if c.get("query_type") == "ask_each"]
    assert len(records) == 1 and records[0]["items"] == 3
    summary = result.trajectory.metadata["ask_each"]
    assert summary["calls"] == 1 and summary["items"] == 3 and summary["ok"] == 3


def test_end_to_end_dataframe_columns_and_to_frame():
    code = ("import pandas as pd\n"
            "df = pd.DataFrame({'summary': ['brake crash', 'steer'], 'id': [7, 8]})\n"
            "r = ask_each(df, 'Classify.', {'theme': ['Brakes', 'Steering', 'Other']}, columns=['summary'])\n"
            "out = r.to_frame(df)\n"
            "SUBMIT(rows=out[['id', 'theme']].to_dict(orient='records'))")
    seen = []
    mapper = CallableLM(lambda text, attempt, m: (seen.append(text), classify(text))[1])
    result = RLM.task("Classify.", outputs={"rows": list}, lm=ScriptedMainLM(code), ask_each=mapper,
                      max_turns=2, timeout=60).run()
    assert result.payload == {"rows": [{"id": 7, "theme": "Brakes"}, {"id": 8, "theme": "Steering"}]}
    assert all("id" not in json.loads(t) for t in seen)   # only the chosen columns were sent


def test_end_to_end_a_bad_output_spec_is_an_error_the_model_sees():
    code = ("try:\n    ask_each(['a'], 'q', {'x': dict})\n    msg = ''\n"
            "except TypeError as exc:\n    msg = str(exc)\nSUBMIT(explains='must be str, int, float, bool' in msg)")
    result = RLM.task("t", outputs={"explains": bool}, lm=ScriptedMainLM(code), ask_each=CallableLM(lambda t, a, m: {}),
                      max_turns=2, timeout=60).run()
    assert result.payload == {"explains": True}


def test_second_engine_interpreter_exposes_ask_each_only_when_turned_on():
    mapper = CallableLM(lambda text, attempt, m: classify(text))
    records: list = []
    with SubprocessPythonInterpreter(timeout=60) as interp:
        interp.ask_each_lm, interp.ask_each_config, interp.ask_each_records = mapper, AskEach(), records
        out = interp.execute("r = ask_each(['brake'], 'q', {'theme': ['Brakes', 'Other']})\nprint(r[0]['theme'])")
    assert "Brakes" in str(out) and len(records) == 1
    with SubprocessPythonInterpreter(timeout=60) as interp:
        out = interp.execute("print('ask_each' in globals())")
    assert "False" in str(out)


def test_second_engine_records_ask_each_as_a_source_call_of_the_code_that_made_it():
    mapper = CallableLM(lambda text, attempt, m: classify(text))
    code = "r = ask_each(['brake', 'steer'], 'q', {'theme': ['Brakes', 'Steering', 'Other']})\nprint(r.stats['ok'])"
    with SubprocessPythonInterpreter(timeout=60) as interp:
        interp.ask_each_lm, interp.ask_each_config, interp.ask_each_records = mapper, AskEach(), []
        interp.execute("x = 1")
        interp.execute(code)
        log = interp.source_call_log
    assert len(log) == 1 and log[0][0] == code
    assert [c["query_type"] for c in log[0][1]] == ["ask_each"] and log[0][1][0]["items"] == 2


@pytest.mark.parametrize("call, message", [
    ("ask_each(['x' * 50_001], 'q', {'a': str})", "Trim or chunk"),
    ("ask_each(['x'] * 6, 'q', {'a': str})", "the limit is 5"),
])
def test_oversized_requests_are_refused_in_the_worker_before_reaching_the_host(call, message):
    mapper = CallableLM(lambda text, attempt, m: {"a": "y"})
    records: list = []
    code = f"try:\n    {call}\n    print('SENT')\nexcept ValueError as exc:\n    print('REFUSED', exc)"
    with SubprocessPythonInterpreter(timeout=60) as interp:
        interp.ask_each_lm, interp.ask_each_config, interp.ask_each_records = mapper, AskEach(max_items=5), records
        out = str(interp.execute(code))
    assert "REFUSED" in out and message in out
    assert records == [] and interp.source_call_log == [] and mapper.prompts == []


def test_too_many_rows_are_refused_before_the_frame_is_converted():
    mapper = CallableLM(lambda text, attempt, m: {"a": "y"})
    code = ("import pandas as pd\n"
            "df = pd.DataFrame({'s': list('abcdef')})\n"
            "def _boom(*a, **k):\n    raise RuntimeError('CONVERTED')\n"
            "pd.DataFrame.to_dict = _boom\n"
            "try:\n    ask_each(df, 'q', {'a': str})\nexcept Exception as exc:\n    print('REFUSED', exc)")
    with SubprocessPythonInterpreter(timeout=60) as interp:
        interp.ask_each_lm, interp.ask_each_config, interp.ask_each_records = mapper, AskEach(max_items=5), []
        out = str(interp.execute(code))
    assert "the limit is 5" in out and "CONVERTED" not in out


def test_a_failed_run_each_is_recorded_as_a_source_call():
    code = "try:\n    run_each([], 'task', {'a': str})\nexcept Exception as exc:\n    print('FAILED', exc)"
    with SubprocessPythonInterpreter(timeout=60) as interp:
        interp.ask_each_lm, interp.ask_each_config, interp.ask_each_records = (
            CallableLM(lambda t, a, m: {}), AskEach(sub_runs=True), [])
        interp.run_child = lambda inputs, task, outputs: None
        out = str(interp.execute(code))
        log = interp.source_call_log
    assert "FAILED" in out and "no items" in out
    assert len(log) == 1 and log[0][1][0]["query_type"] == "run_each" and log[0][1][0]["executed"] is False


def test_missing_values_reach_the_model_as_empty_text():
    seen = []
    mapper = CallableLM(lambda text, attempt, m: (seen.append(text), {"a": "y"})[1])
    code = ("import pandas as pd, numpy as np\n"
            "df = pd.DataFrame({'s': ['brake', None, 'x'], 'v': [1.0, 2.0, np.nan]})\n"
            "ask_each(df, 'q', {'a': str})\n"
            "ask_each(pd.Series(['a', None, np.nan]), 'q', {'a': str})\n"
            "ask_each([None, {'k': None}], 'q', {'a': str})")
    with SubprocessPythonInterpreter(timeout=60) as interp:
        interp.ask_each_lm, interp.ask_each_config, interp.ask_each_records = mapper, AskEach(), []
        interp.execute(code)
    assert len(seen) == 8
    assert not any("null" in t or "NaN" in t for t in seen)
    assert '{"s": "", "v": 2.0}' in seen and '{"s": "x", "v": ""}' in seen and seen.count("") == 3


# ----------------------------------------------------------------- SemanticModel.find_measures
class _Answers(list):
    stats: dict = {}


def _model_with_measures(n_filler=300):
    import pandas as pd
    from fabric_rlm.semantic_model import SemanticModel

    rows = [{"Table Name": "M", "Measure Name": f"Filler {i}", "Measure Expression": f"SUM(T[c{i}])",
             "Measure Description": ""} for i in range(n_filler)]
    rows += [
        {"Table Name": "M", "Measure Name": "Avg Unit Retail Price", "Measure Expression": "DIVIDE([POS $ Sales],[POS Unit Sales])",
         "Measure Description": "POS dollars per unit"},
        {"Table Name": "M", "Measure Name": "Avg Unit Retail Price YA", "Measure Expression": "CALCULATE([Avg Unit Retail Price], SAMEPERIODLASTYEAR(D[Date]))",
         "Measure Description": "a year ago"},
    ]
    model = SemanticModel("m", validate=False)
    object.__setattr__(model, "measures", lambda: pd.DataFrame(rows))
    return model


def _fake_ask(calls):
    def ask(items, question, output, columns=None, **kwargs):
        calls.append((len(items), question, output, columns, kwargs))
        out = _Answers()
        for _, row in items.iterrows():
            name = row["measure_name"]
            fit = "exact" if name == "Avg Unit Retail Price" else ("close" if name.startswith("Avg Unit Retail Price") else "no")
            out.append({"fit": fit})
        out.stats = {"failed": 0, "unfinished": 0}
        return out
    return ask


def test_find_measures_screens_every_measure_and_ranks_exact_first():
    calls = []
    result = _model_with_measures().find_measures("average price per unit at the register", ask=_fake_ask(calls))
    assert calls[0][0] == 302                          # every measure was screened, none filtered first
    assert "average price per unit" in calls[0][1] and calls[0][2] == {"fit": ["exact", "close", "no"]}
    assert "measure_expression" in calls[0][3] and calls[0][4] == {"batch_size": 25, "concurrency": 16}
    assert list(result["measure_name"]) == ["Avg Unit Retail Price", "Avg Unit Retail Price YA"]
    assert list(result["fit"]) == ["exact", "close"] and "DIVIDE" in result["measure_expression"].iloc[0]
    assert result.attrs["screened"] == {"measures": 302, "flagged": 2, "returned": 2, "failed": 0, "unfinished": 0}


def test_find_measures_orders_by_decision_model_confidence_and_caps_the_list():
    def ask(items, question, output, columns=None, **kwargs):
        out = _Answers({"fit": "close", "fit_confidence": (i % 7) / 10} for i in range(len(items)))
        out.stats = {}
        return out

    result = _model_with_measures(20).find_measures("q", ask=ask, top=5)
    assert len(result) == 5 and list(result["confidence"]) == sorted(result["confidence"], reverse=True)


def test_find_measures_without_ask_each_says_what_to_do():
    with pytest.raises(RuntimeError, match="not turned on"):
        _model_with_measures(3).find_measures("q")


def test_find_measures_refuses_a_measure_list_it_cannot_read():
    import pandas as pd

    model = _model_with_measures(3)
    object.__setattr__(model, "measures", lambda: pd.DataFrame([{"Foo": 1, "Bar": "x"}] * 4))
    calls = []
    with pytest.raises(ValueError, match="could not read the measure list"):
        model.find_measures("q", ask=_fake_ask(calls))
    assert calls == []


def test_find_measures_puts_answers_without_a_confidence_last():
    def ask(items, question, output, columns=None, **kwargs):
        confidences = [None, 0.6, None, 0.9]
        out = _Answers({"fit": "close", **({"fit_confidence": c} if c is not None else {})} for c in confidences)
        out.stats = {}
        return out

    result = _model_with_measures(2).find_measures("q", ask=ask)
    assert list(result["confidence"])[:2] == [0.9, 0.6] and result["confidence"].iloc[2:].isna().all()


def test_find_measures_is_mentioned_only_with_a_semantic_model_input():
    from fabric_rlm import SemanticModel
    from fabric_rlm.prompts import ask_each_section
    from fabric_rlm.runtime import _has_semantic_model_input

    assert "find_measures" not in ask_each_section()
    assert "find_measures" in ask_each_section(semantic_model=True)
    model = SemanticModel("m", validate=False)
    assert _has_semantic_model_input({"m": model}) and _has_semantic_model_input({"ms": [model]})
    assert not _has_semantic_model_input({"x": 1, "f": [File("a.pdf")]})


def test_result_and_error_types_are_exported():
    import fabric_rlm

    assert fabric_rlm.AskEachResult is ask_each_module.AskEachResult
    assert fabric_rlm.AskEachError is AskEachError
    r = fabric_rlm.AskEachResult([{"a": 1}, None], errors=[{"index": 1, "error": "x"}], stats={"output_fields": ["a"]})
    assert r.errors[0]["index"] == 1 and list(r.to_frame()["a"].iloc[:1]) == [1]


def test_worker_publishes_ask_each_for_helpers_only_when_turned_on():
    mapper = CallableLM(lambda text, attempt, m: classify(text))
    code = "import fabric_rlm.ask_each as a\nprint('HOOK', a.WORKER_ASK_EACH is not None)"
    with SubprocessPythonInterpreter(timeout=60) as interp:
        interp.ask_each_lm, interp.ask_each_config, interp.ask_each_records = mapper, AskEach(), []
        assert "HOOK True" in str(interp.execute(code))
    with SubprocessPythonInterpreter(timeout=60) as interp:
        assert "HOOK False" in str(interp.execute(code))


def test_schema_says_when_the_measure_list_is_cut_and_where_to_look(monkeypatch):
    import fabric_rlm.ask_each as ask_each_mod

    model = _model_with_measures(300)
    text = model.schema()
    assert "of 302 measures; the listing is cut here" in text and "search model.measures()" in text
    monkeypatch.setattr(ask_each_mod, "WORKER_ASK_EACH", lambda *a, **k: None)
    assert "call model.find_measures(" in model.schema()
    small = _model_with_measures(3).schema()
    assert "listing is cut" not in small and "Avg Unit Retail Price" in small
