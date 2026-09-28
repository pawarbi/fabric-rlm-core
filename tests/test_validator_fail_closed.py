"""User validators fail closed (issue #119) and say what they rejected (issue #122).

A validator that cannot check an answer has not checked it: a crash, a False
return or a timeout rejects the answer unless the caller opts into
validator_errors="accept". The truth in every case is net sales of 700; the
scripted model submits gross sales (1000) first.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import pytest

from fabric_rlm import RLM, File

TRUTH = 700
GROSS = "```python\nimport pandas as pd\nnet_sales = int(pd.read_csv(orders)['amount'].sum())\nSUBMIT(net_sales=net_sales)\n```"
NET = ("```python\nimport pandas as pd\ndf = pd.read_csv(orders)\nnet_sales = int(df.loc[~df['returned'], 'amount'].sum())\n"
       "SUBMIT(net_sales=net_sales)\n```")


class ScriptedLM:
    def __init__(self, turns):
        self.turns = list(turns)
        self.prompts: list[str] = []

    def __call__(self, *, messages):
        self.prompts.append(str(messages[-1].get("content", "")) if messages else "")
        return self.turns.pop(0) if self.turns else NET


@pytest.fixture()
def orders(tmp_path: Path) -> File:
    path = tmp_path / "orders.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["amount", "returned"])
        writer.writerows([[100, False], [200, True], [150, False], [250, False], [100, True], [200, False]])
    return File(path)


def run(orders, turns=(GROSS, NET), max_turns=4, **kwargs):
    lm = ScriptedLM(turns)
    result = RLM.task(
        "Net sales excluding returned orders, as an integer.",
        inputs={"orders": orders},
        outputs={"net_sales": int},
        lm=lm,
        max_turns=max_turns,
        timeout=60,
        **kwargs,
    ).run()
    return result, lm


def assumes_nested(payload):
    assert payload["net_sales"]["value"] == TRUTH   # TypeError on an int


def returns_false(payload):
    return payload["net_sales"] == TRUTH


def reference_query_fails(payload, context):
    raise RuntimeError("reference query failed")


def is_net(payload):
    assert payload["net_sales"] == TRUTH, "net_sales must exclude returned orders"


def test_a_false_return_rejects_and_the_right_answer_is_verified(orders):
    result, lm = run(orders, output_validator=returns_false)
    assert result.submitted and result.payload == {"net_sales": TRUTH}
    assert result.verified is True
    assert any("returned False" in prompt for prompt in lm.prompts)


def test_an_asserting_validator_rejects_with_neutral_wording_naming_the_field(orders):
    result, lm = run(orders, output_validator=is_net)
    assert result.submitted and result.payload == {"net_sales": TRUTH} and result.verified
    rejection = next(prompt for prompt in lm.prompts if "rejected" in prompt)
    assert "rejected by the output validator" in rejection
    assert "output-format" not in rejection and "`output` field" not in rejection
    assert "Fix `net_sales` as described above" in rejection


def test_a_crashing_validator_never_lets_an_answer_through(orders):
    result, lm = run(orders, turns=(GROSS, GROSS, GROSS, GROSS), output_validator=assumes_nested)
    assert result.submitted is False and result.payload is None
    assert result.verified is False
    assert result.trajectory.metadata["stopped_reason"] == "validator_error"
    assert "TypeError" in result.trajectory.metadata["validator_error"]
    assert len(result.turns) == 2   # stopped after two failures, not after max_turns


def test_a_crashing_context_validator_never_lets_an_answer_through(orders):
    result, _ = run(orders, output_validator_context=reference_query_fails)
    assert result.submitted is False
    assert result.trajectory.metadata["stopped_reason"] == "validator_error"
    assert "reference query failed" in result.trajectory.metadata["validator_error"]


def test_one_failure_then_a_working_check_is_not_a_stop(orders):
    calls = {"n": 0}

    def flaky(payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("reference query timed out")
        assert payload["net_sales"] == TRUTH

    result, _ = run(orders, output_validator=flaky)
    assert result.submitted and result.payload == {"net_sales": TRUTH} and result.verified


def test_accept_restores_the_old_behaviour_and_says_it_is_unverified(orders, caplog):
    with caplog.at_level(logging.WARNING, logger="fabric_rlm.runtime"):
        result, _ = run(orders, output_validator=assumes_nested, validator_errors="accept")
    assert result.submitted and result.payload == {"net_sales": 1000}
    assert result.verified is False
    assert result.trajectory.metadata["verifier_execution"]["degraded"] == ["output_validator"]
    assert any("result.verified is False" in record.message for record in caplog.records)


def test_a_slow_validator_times_out_and_counts_as_a_failed_check(orders):
    def hangs(payload):
        time.sleep(30)

    started = time.monotonic()
    result, _ = run(orders, output_validator=hangs, validator_timeout=0.5)
    assert time.monotonic() - started < 25
    assert result.submitted is False
    assert "timed out after 0.5 s" in result.trajectory.metadata["validator_error"]


def test_no_validator_means_not_verified(orders):
    result, _ = run(orders, turns=(NET,))
    assert result.submitted and result.verified is False


@pytest.mark.parametrize("bad", [{"validator_errors": "ignore"}, {"validator_timeout": 0}, {"validator_timeout": True}])
def test_settings_are_checked(orders, bad):
    with pytest.raises(ValueError):
        RLM.task("x", inputs={"orders": orders}, outputs={"net_sales": int}, lm=ScriptedLM([]), **bad)


def test_the_inspector_shows_the_rejection_under_its_turn(orders):
    result, _ = run(orders, output_validator=is_net)
    page = result.inspect().to_html()
    assert "Rejected by output_validator" in page and "net_sales must exclude returned orders" in page
    assert ">Rejected<" in page and "Verified" in page
