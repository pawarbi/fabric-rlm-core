"""llm_map(): a host-run parallel map over items with a validated output schema.

The map LM is faked with a DSPy DummyLM subclass so these run with no network.
The end-to-end cases drive a real worker subprocess with a scripted main LM.
"""

from __future__ import annotations

import re
import threading

import pytest
from dspy.utils.dummies import DummyLM, dotdict

from fabric_rlm import RLM
from fabric_rlm.interpreter import _run_host_llm_map
from fabric_rlm.llm_map import LLMMapError, run_llm_map
from fabric_rlm.prompts import build_system_prompt

THEMES = ["Brakes", "Steering", "Other"]
OUTPUT = {"theme": {"choices": THEMES}, "safety_critical": "bool"}


class RespondingLM(DummyLM):
    """A map LM whose answer is a function of the item text and the attempt number."""

    def __init__(self, respond):
        super().__init__(answers={})
        self.respond = respond
        self.lock = threading.Lock()
        self.calls: list[str] = []
        self.systems: list[str] = []

    def forward(self, prompt=None, messages=None, **kwargs):
        user = messages[-1]["content"]
        system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        item = re.search(r"\[\[ ## item ## \]\]\s*(.*?)\s*(\n\n|$)", user, re.S)
        text = item.group(1) if item else user
        with self.lock:
            attempt = sum(1 for c in self.calls if c == text)
            self.calls.append(text)
            self.systems.append(system)
        answer = self.respond(text, attempt)
        if isinstance(answer, Exception):
            raise answer
        content = self._format_answer_fields(answer)
        message = dotdict(content=content, tool_calls=None)
        return dotdict(choices=[dotdict(message=message, finish_reason="stop")],
                       usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0), model="dummy")


def classify(text, attempt):
    theme = "brakes" if "brake" in text else ("Steering" if "steer" in text else "Other")
    return {"theme": theme, "safety_critical": "yes" if "crash" in text else "false"}


class BatchAwareLM(RespondingLM):
    """Answers batch prompts through ``batch`` (list of item texts -> answers text); single prompts via ``respond``."""

    def __init__(self, batch, respond=classify):
        super().__init__(respond)
        self.batch = batch
        self.batch_sizes: list[int] = []

    def forward(self, prompt=None, messages=None, **kwargs):
        user = messages[-1]["content"]
        block = re.search(r"\[\[ ## items ## \]\]\s*(.*?)(?=\n\n\[\[ ##|\Z)", user, re.S)
        if not block:
            return super().forward(prompt=prompt, messages=messages, **kwargs)
        texts = [t.strip() for t in re.split(r"(?m)^\[\d+\]\n", block.group(1))[1:]]
        with self.lock:
            self.batch_sizes.append(len(texts))
        content = self._format_answer_fields({"answers": self.batch(texts)})
        return dotdict(choices=[dotdict(message=dotdict(content=content, tool_calls=None), finish_reason="stop")],
                       usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0), model="dummy")


def good_batch(texts):
    import json as _json
    return _json.dumps([{"n": n, **classify(t, 0)} for n, t in enumerate(texts)])


# ----------------------------------------------------------------- host engine
def test_results_align_with_items_and_are_coerced_under_concurrency():
    items = [f"#{i} the brake failed" if i % 3 == 0 else (f"#{i} steering stiff crash" if i % 3 == 1 else f"#{i} radio")
             for i in range(60)]
    result, record = run_llm_map(RespondingLM(classify), {"items": items, "instructions": "Classify.",
                                                          "output": OUTPUT, "concurrency": 8})
    assert [r["theme"] for r in result["results"]] == [["Brakes", "Steering", "Other"][i % 3] for i in range(60)]
    assert [r["safety_critical"] for r in result["results"]] == [i % 3 == 1 for i in range(60)]
    assert result["stats"]["ok"] == 60 and result["stats"]["failed"] == 0 and result["errors"] == []
    assert record["query_type"] == "llm_map" and record["items"] == 60 and record["output_fields"] == ["theme", "safety_critical"]


def test_invalid_answer_is_retried_with_the_validation_error_in_the_prompt():
    def first_wrong(text, attempt):
        return {"theme": "Suspension", "safety_critical": "false"} if attempt == 0 else {"theme": "Brakes", "safety_critical": "true"}

    lm = RespondingLM(first_wrong)
    result, _ = run_llm_map(lm, {"items": ["brake pedal sank"], "instructions": "Classify.", "output": OUTPUT})
    assert result["results"] == [{"theme": "Brakes", "safety_critical": True}]
    assert result["stats"]["retried"] == 1 and result["stats"]["calls"] == 2
    assert "rejected" in lm.systems[1] and "Suspension" in lm.systems[1] and "Brakes" in lm.systems[1]


def test_an_item_that_never_validates_is_none_with_its_error_not_a_guess():
    result, _ = run_llm_map(RespondingLM(lambda t, a: {"theme": "Wheels", "safety_critical": "maybe"}),
                            {"items": ["a", "brake"], "instructions": "Classify.", "output": OUTPUT, "retries": 1})
    assert result["results"] == [None, None]
    assert result["stats"]["failed"] == 2 and result["stats"]["calls"] == 4
    assert result["errors"][0]["index"] == 0 and "must be exactly one of" in result["errors"][0]["error"]
    assert "true or false" in result["errors"][0]["error"]


def test_empty_text_is_a_valid_answer_but_an_empty_choice_or_number_is_not():
    spec = {"relevant": "bool", "summary": "str", "n": "int"}
    lm = RespondingLM(lambda t, a: {"relevant": "false", "summary": "", "n": "3"})
    result, _ = run_llm_map(lm, {"items": ["page 1"], "instructions": "x", "output": spec})
    assert result["results"] == [{"relevant": False, "summary": "", "n": 3}] and result["stats"]["retried"] == 0
    result, _ = run_llm_map(RespondingLM(lambda t, a: {"relevant": "", "summary": "", "n": ""}),
                            {"items": ["page 1"], "instructions": "x", "output": spec, "retries": 0})
    assert result["results"] == [None] and "`relevant` is missing" in result["errors"][0]["error"]


def test_an_lm_exception_fails_only_that_item():
    def flaky(text, attempt):
        return RuntimeError("provider 500") if "bad" in text else classify(text, attempt)

    result, _ = run_llm_map(RespondingLM(flaky), {"items": ["brake", "bad item", "steer"], "instructions": "x",
                                                  "output": OUTPUT, "retries": 1})
    assert result["results"][0]["theme"] == "Brakes" and result["results"][2]["theme"] == "Steering"
    assert result["results"][1] is None and result["stats"]["lm_errors"] == 2
    assert "provider 500" in result["errors"][0]["error"]


@pytest.mark.parametrize("kwargs, message", [
    ({"items": [], "instructions": "x", "output": OUTPUT}, "no items"),
    ({"items": "abc", "instructions": "x", "output": OUTPUT}, "must be a list"),
    ({"items": ["a"], "instructions": " ", "output": OUTPUT}, "needs instructions"),
    ({"items": ["a"], "instructions": "x", "output": {}}, "needs output"),
    ({"items": ["a"], "instructions": "x", "output": {"theme": "list"}}, "must be str, int"),
    ({"items": ["a"], "instructions": "x", "output": {"theme": {"choices": ["A", "a"]}}}, "duplicate"),
    ({"items": ["a"], "instructions": "x", "output": OUTPUT, "concurrency": 0}, "concurrency"),
    ({"items": ["a"], "instructions": "x", "output": OUTPUT, "batch_size": 0}, "batch_size"),
    ({"items": ["a"], "instructions": "x", "output": OUTPUT, "batch_size": 101}, "batch_size"),
    ({"items": ["x" * 60_000], "instructions": "x", "output": OUTPUT}, "characters"),
])
def test_bad_requests_are_refused_with_a_usable_message(kwargs, message):
    with pytest.raises(LLMMapError, match=message):
        run_llm_map(RespondingLM(classify), kwargs)


def test_no_host_lm_gives_a_message_the_model_can_act_on():
    with pytest.raises(RuntimeError, match="llm_map is not available"):
        _run_host_llm_map(None, {"items": ["a"], "instructions": "x", "output": OUTPUT})


# ----------------------------------------------------------------- batching
ITEMS_25 = [f"#{i} the brake failed" if i % 3 == 0 else (f"#{i} steering stiff crash" if i % 3 == 1 else f"#{i} radio")
            for i in range(25)]
EXPECTED_25 = [["Brakes", "Steering", "Other"][i % 3] for i in range(25)]


def test_batches_answer_several_items_per_call_and_stay_aligned():
    lm = BatchAwareLM(good_batch)
    result, record = run_llm_map(lm, {"items": ITEMS_25, "instructions": "Classify.", "output": OUTPUT,
                                      "batch_size": 10, "concurrency": 4})
    assert [r["theme"] for r in result["results"]] == EXPECTED_25
    assert sorted(lm.batch_sizes) == [5, 10, 10] and lm.calls == []  # no per-item calls at all
    assert result["stats"]["calls"] == 3 and result["stats"]["batches"] == 3 and result["stats"]["batch_fallbacks"] == 0
    assert record["batch_size"] == 10


@pytest.mark.parametrize("corrupt, reason", [
    (lambda rows: [r for r in rows if r["n"] != 2], "missing"),
    (lambda rows: rows + [dict(rows[2], theme="Other")], "duplicated"),
    (lambda rows: [dict(r, theme="Wheels") if r["n"] == 2 else r for r in rows], "must be exactly one of"),
])
def test_a_bad_batch_row_falls_back_to_a_single_call_for_that_item_only(corrupt, reason):
    import json as _json
    lm = BatchAwareLM(lambda texts: _json.dumps(corrupt([{"n": n, **classify(t, 0)} for n, t in enumerate(texts)])))
    result, _ = run_llm_map(lm, {"items": ITEMS_25[:5], "instructions": "Classify.", "output": OUTPUT, "batch_size": 5})
    assert [r["theme"] for r in result["results"]] == EXPECTED_25[:5]
    assert result["stats"]["batch_fallbacks"] == 1 and lm.calls == [ITEMS_25[2]]
    assert reason in lm.systems[0]  # the batch's complaint about the item is the single call's feedback


def test_unparseable_batch_answer_falls_back_for_every_item_and_retries_zero_reports_them():
    lm = BatchAwareLM(lambda texts: "sorry, here you go: {not json")
    result, _ = run_llm_map(lm, {"items": ITEMS_25[:4], "instructions": "Classify.", "output": OUTPUT, "batch_size": 4})
    assert [r["theme"] for r in result["results"]] == EXPECTED_25[:4] and result["stats"]["batch_fallbacks"] == 4
    result, _ = run_llm_map(BatchAwareLM(lambda texts: "nope"),
                            {"items": ITEMS_25[:4], "instructions": "x", "output": OUTPUT, "batch_size": 4, "retries": 0})
    assert result["results"] == [None] * 4 and "not a JSON array" in result["errors"][0]["error"]


def test_a_batch_is_cut_by_characters_so_long_items_do_not_pile_up():
    lm = BatchAwareLM(good_batch)
    items = ["brake " + "x" * 25_000 for _ in range(5)]
    result, _ = run_llm_map(lm, {"items": items, "instructions": "x", "output": OUTPUT, "batch_size": 50})
    assert result["stats"]["ok"] == 5 and max(lm.batch_sizes) <= 2


def test_prompt_advertises_llm_map_only_when_available():
    on = build_system_prompt(inline_task="t", inline_outputs=["a"], llm_map_available=True)
    off = build_system_prompt(inline_task="t", inline_outputs=["a"])
    assert "llm_map(items, instructions, output" in on and "llm_map(" not in off


# ----------------------------------------------------------------- decision-model backend
class FakeDecisionLM:
    is_decision_model = True
    model = "fake-decider"
    bool_threshold = 0.5

    def __init__(self, fail_on=None):
        self.requests = []
        self.fail_on = fail_on
        self.lock = threading.Lock()

    def decide(self, state, questions):
        with self.lock:
            self.requests.append((state, questions))
        if self.fail_on and self.fail_on in state:
            raise RuntimeError("HTTP 502")
        theme = "Brakes" if "brake" in state else ("Steering" if "steer" in state else "Other")
        return {"answers": {"theme": {"type": "choice", "choice": theme, "confidence": 0.9 if theme != "Other" else 0.4},
                            "safety_critical": {"type": "noul", "noul": 0.97 if "crash" in state else 0.1}}}


def test_decision_model_maps_choices_and_bools_and_returns_confidence():
    lm = FakeDecisionLM(fail_on="bad")
    result, record = run_llm_map(lm, {"items": ["brake crash", "steer", "radio", "bad"], "instructions": "Classify.",
                                      "output": OUTPUT, "batch_size": 10})
    assert result["results"][0] == {"theme": "Brakes", "theme_confidence": 0.9, "safety_critical": True, "safety_critical_p": 0.97}
    assert result["results"][2]["theme"] == "Other" and result["results"][2]["theme_confidence"] == 0.4
    assert result["results"][3] is None and "502" in result["errors"][0]["error"]
    assert record["decision_model"] and result["stats"]["calls"] == 4  # batch_size is ignored: one call per item
    state, questions = lm.requests[0]
    assert questions["theme"]["type"] == "choice" and set(questions["theme"]["criteria"]) == set(THEMES)
    assert questions["safety_critical"]["type"] == "noul" and "Classify." in questions["theme"]["instructions"]


def test_decision_model_refuses_text_and_number_fields_with_a_workaround():
    with pytest.raises(LLMMapError, match="bucket a number into ranges"):
        run_llm_map(FakeDecisionLM(), {"items": ["a"], "instructions": "x", "output": {"summary": "str", "theme": {"choices": THEMES}}})


def test_prompt_tells_the_planner_what_a_decision_model_can_answer():
    on = build_system_prompt(inline_task="t", inline_outputs=["a"], llm_map_available=True, llm_map_decision_model=True)
    assert "backed by a decision model" in on and "_confidence" in on
    assert "backed by a decision model" not in build_system_prompt(inline_task="t", inline_outputs=["a"], llm_map_available=True)


def test_end_to_end_decision_model_confidence_reaches_the_worker():
    turn = _code(
        "out = llm_map(['brake crash', 'radio'], 'Classify.', {'theme': ['Brakes', 'Steering', 'Other'], 'safety_critical': bool})\n"
        "df = out.to_frame()\n"
        "print(df.to_dict('records'))\n"
        "SUBMIT(conf=[round(x, 2) for x in df['theme_confidence']], themes=list(df['theme']))"
    )
    result = RLM.task(task="t", outputs={"conf": list, "themes": list}, lm=ScriptedLM([turn]), map_lm=FakeDecisionLM(),
                      block_network=True, max_turns=2).run()
    assert result.payload == {"conf": [0.9, 0.4], "themes": ["Brakes", "Other"]}


# ----------------------------------------------------------------- end to end through the worker
class ScriptedLM:
    def __init__(self, turns):
        self.turns = list(turns)

    def __call__(self, *, messages):
        return self.turns.pop(0)


def _code(body):
    return f"```python\n{body}\n```"


MAP_AND_SUBMIT = _code(
    "items = ['#1 brake failed, crash', '#2 steering stiff', '#3 radio', '#4 brake noise']\n"
    "out = llm_map(items, 'Classify the complaint.', {'theme': ['Brakes', 'Steering', 'Other'], 'safety_critical': bool})\n"
    "print(out.stats['ok'], out.stats['failed'])\n"
    "SUBMIT(brakes=sum(1 for r in out if r and r['theme'] == 'Brakes'), critical=sum(1 for r in out if r and r['safety_critical']))"
)


@pytest.mark.parametrize("block_network", [False, True])
def test_end_to_end_one_call_in_the_worker_and_telemetry_on_the_turn(block_network):
    map_lm = RespondingLM(classify)
    result = RLM.task(task="Count brake complaints.", outputs={"brakes": int, "critical": int},
                      lm=ScriptedLM([MAP_AND_SUBMIT]), map_lm=map_lm, block_network=block_network, max_turns=2).run()
    assert result.submitted and result.payload == {"brakes": 2, "critical": 1}
    calls = [c for t in result.trajectory.turns for c in t.source_calls]
    record = next(c for c in calls if c.get("query_type") == "llm_map")
    assert record["items"] == 4 and record["ok"] == 4 and record["failed"] == 0
    assert "brake" not in str(record).lower()  # counts and field names only, never item text
    assert len(map_lm.calls) == 4


def test_end_to_end_dataframe_columns_and_to_frame():
    turn = _code(
        "import pandas as pd\n"
        "df = pd.DataFrame({'id': [10, 11, 12], 'summary': ['brake crash', 'steer', 'radio'], 'secret': ['x', 'y', 'z']})\n"
        "out = llm_map(df, 'Classify.', {'theme': ['Brakes', 'Steering', 'Other'], 'safety_critical': bool}, columns=['summary'])\n"
        "joined = out.to_frame(df)\n"
        "print(joined.to_dict('records'))\n"
        "SUBMIT(themes=list(joined['theme']), ids=list(joined['id']))"
    )
    map_lm = RespondingLM(classify)
    result = RLM.task(task="t", outputs={"themes": list, "ids": list}, lm=ScriptedLM([turn]), map_lm=map_lm, max_turns=2).run()
    assert result.payload == {"themes": ["Brakes", "Steering", "Other"], "ids": [10, 11, 12]}
    assert all("secret" not in c and '"id"' not in c for c in map_lm.calls)


def test_end_to_end_bad_output_spec_is_a_normal_error_the_model_sees():
    bad = _code("out = llm_map(['a'], 'x', {'theme': dict})")
    fix = _code("SUBMIT(n=1)")
    result = RLM.task(task="t", outputs={"n": int}, lm=ScriptedLM([bad, fix]), map_lm=RespondingLM(classify), max_turns=3).run()
    assert result.submitted
    assert "llm_map output 'theme' must be" in (result.trajectory.turns[0].error or result.trajectory.turns[0].stderr or "")
