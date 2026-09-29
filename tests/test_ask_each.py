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
        ({"output": {"x": "date"}}, "must be str, int, float, bool"),
        ({"output": {"_x": "str"}}, "plain identifiers"),
        ({"output": {"x": {"choices": ["A", "a"]}}}, "duplicate choices"),
        ({"retries": 9}, "retries must be an integer from 0 to 5"),
        ({"batch_size": 0}, "batch_size must be an integer"),
        ({"max_seconds": -1}, "max_seconds must be a positive number"),
        ({"items": ["x" * 50_001]}, "Trim or chunk"),
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
    def respond(text, attempt, messages):
        user = messages[-1]["content"]
        if "Items:\n" in user:
            texts = [t.strip() for t in re.split(r"(?m)^\[\d+\]\n", user.split("Items:\n", 1)[1])[1:]]
            return batch_answer(texts)
        return classify(text)
    return CallableLM(respond)


def test_batches_answer_several_items_per_call_and_stay_aligned():
    lm = batch_lm(lambda texts: json.dumps([{"n": n, **classify(t)} for n, t in enumerate(texts)]))
    items = [("brake " if i % 2 else "steer ") + str(i) for i in range(25)]
    result, _ = run_ask_each(lm, request(items, batch_size=10))
    assert [r["theme"] for r in result["results"]] == ["Brakes" if i % 2 else "Steering" for i in range(25)]
    assert result["stats"]["batches"] == 3 and result["stats"]["calls"] == 3


@pytest.mark.parametrize("corrupt", ["missing", "duplicate", "invalid"])
def test_a_bad_batch_row_falls_back_to_a_single_call_for_that_item_only(corrupt):
    def answer(texts):
        rows = [{"n": n, **classify(t)} for n, t in enumerate(texts)]
        if corrupt == "missing":
            rows.pop(1)
        elif corrupt == "duplicate":
            rows.append(dict(rows[1]))
        else:
            rows[1]["theme"] = "Wheels"
        return json.dumps(rows)

    result, _ = run_ask_each(batch_lm(answer), request(["brake a", "steer b", "brake c"], batch_size=3))
    assert [r["theme"] for r in result["results"]] == ["Brakes", "Steering", "Brakes"]
    assert result["stats"]["batch_fallbacks"] == 1 and result["stats"]["calls"] == 2


def test_unparseable_batch_answer_with_no_retries_reports_every_item():
    result, _ = run_ask_each(batch_lm(lambda texts: "no json here"), request(["brake", "steer"], batch_size=2, retries=0))
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
    def ask(items, question, output, columns=None):
        calls.append((len(items), question, output, columns))
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
    assert "measure_expression" in calls[0][3]
    assert list(result["measure_name"]) == ["Avg Unit Retail Price", "Avg Unit Retail Price YA"]
    assert list(result["fit"]) == ["exact", "close"] and "DIVIDE" in result["measure_expression"].iloc[0]
    assert result.attrs["screened"] == {"measures": 302, "flagged": 2, "returned": 2, "failed": 0, "unfinished": 0}


def test_find_measures_orders_by_decision_model_confidence_and_caps_the_list():
    def ask(items, question, output, columns=None):
        out = _Answers({"fit": "close", "fit_confidence": (i % 7) / 10} for i in range(len(items)))
        out.stats = {}
        return out

    result = _model_with_measures(20).find_measures("q", ask=ask, top=5)
    assert len(result) == 5 and list(result["confidence"]) == sorted(result["confidence"], reverse=True)


def test_find_measures_without_ask_each_says_what_to_do():
    with pytest.raises(RuntimeError, match="not turned on"):
        _model_with_measures(3).find_measures("q")


def test_worker_publishes_ask_each_for_helpers_only_when_turned_on():
    mapper = CallableLM(lambda text, attempt, m: classify(text))
    code = "import fabric_rlm.ask_each as a\nprint('HOOK', a.WORKER_ASK_EACH is not None)"
    with SubprocessPythonInterpreter(timeout=60) as interp:
        interp.ask_each_lm, interp.ask_each_config, interp.ask_each_records = mapper, AskEach(), []
        assert "HOOK True" in str(interp.execute(code))
    with SubprocessPythonInterpreter(timeout=60) as interp:
        assert "HOOK False" in str(interp.execute(code))
