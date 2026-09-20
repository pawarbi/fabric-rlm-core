"""A zero or empty answer submitted from the step that computed it gets one confirming step.

Found in Fabric (#105). Output reaches the model only after a step ends, so a
value computed and submitted in one block was never looked at. Replayed over
125 logged runs, the check would have sent back six, and all six were wrong
answers: a total of 0.0 after ``except: continue`` skipped every row, 0 of
2,000 files loaded, and four staffing answers that were empty where the truth
was one more adjuster in two regions. No correct run was flagged.
"""

from __future__ import annotations

import pytest

from fabric_rlm import RLM
from fabric_rlm.analytical_integrity import check_blind_empty_submit

from test_api_arguments_engines import _ScriptedLM

SWALLOWED = (
    "monthly = [0.0] * 12\n"
    "for row in rows:\n"
    "    try:\n"
    "        monthly[int(row['month']) - 1] += float(row['amount'])\n"
    "    except Exception:\n"
    "        continue\n"
    "total = round(sum(monthly), 2)\n"
    "SUBMIT(total=total)"
)


@pytest.mark.parametrize(
    ("code", "payload", "named"),
    [
        (SWALLOWED, {"total": 0.0}, "total = 0.0"),
        ("n_loaded = len(orders)\nby_currency = {}\nSUBMIT(n_files=2000, n_loaded=n_loaded, value=by_currency)",
         {"n_files": 2000, "n_loaded": 0, "value": {}}, "n_loaded = 0, value = {}"),
        ("extra = {region: max(0, need[region] - have[region]) for region in need}\nSUBMIT(extra=extra)",
         {"extra": {"South": 0, "West": 0}}, "extra = {'South': 0, 'West': 0}"),
        ("rate = float('nan') if not rows else 1.0\nSUBMIT(rate=rate)", {"rate": float("nan")}, "rate = nan"),
    ],
)
def test_an_empty_answer_computed_and_submitted_in_one_step_is_named(code, payload, named):
    problems = check_blind_empty_submit(code, payload)
    assert len(problems) == 1 and problems[0].startswith(named)
    assert "never looked at" in problems[0] and "SUBMIT in the next step" in problems[0]


@pytest.mark.parametrize(
    ("code", "payload"),
    [
        ("total = sum(r['amount'] for r in rows)\nSUBMIT(total=total)", {"total": 1175.0}),      # not empty
        ("SUBMIT(total=total)", {"total": 0.0}),                        # computed in an earlier step, which printed it
        ("SUBMIT(total=sum(numbers))", {"total": 0}),                   # built from an input, nothing assigned here
        ("SUBMIT(violations=0, note='none found')", {"violations": 0, "note": "none found"}),    # a literal: the literal check owns it
        ("note = ''\nok = False\nSUBMIT(note=note, ok=ok)", {"note": "", "ok": False}),         # strings and booleans are not judged
        ("payload = {'total': t}\nSUBMIT(**payload)", {"total": 0.0}),                           # cannot tell which name fed it
        ("total = (\nSUBMIT(total=total)", {"total": 0.0}),                                      # code that does not parse
        ("total = 0.0\nSUBMIT(total=total)", None),
        (None, {"total": 0.0}),
    ],
)
def test_anything_else_is_left_alone(code, payload):
    assert check_blind_empty_submit(code, payload) == []


def _run(*codes, max_turns=4, **kwargs):
    rows = [{"month": "2025-01", "amount": "10.5"}, {"month": "2025-02", "amount": "4.5"}]
    lm = _ScriptedLM(list(codes))
    result = RLM.task("Total amount.", inputs={"rows": rows}, outputs={"total": float}, lm=lm, max_turns=max_turns, **kwargs).run()
    return result, lm


def test_a_run_is_sent_back_once_and_then_gives_the_right_answer():
    # int('2025-01') raises, the bare except skips every row, the total is 0.0: the run seen in Fabric.
    fixed = "total = sum(float(row['amount']) for row in rows)\nprint(len(rows), total)"
    result, lm = _run(SWALLOWED, fixed, "SUBMIT(total=total)")
    assert result.submitted and result.outputs == {"total": 15.0}
    assert [turn.turn_type for turn in result.turns] == ["normal", "verifier_repair", "normal"]
    said = [str(message["content"]) for message in lm.calls[-1]["messages"]]
    assert any("total = 0.0 was submitted from the same step that computed it" in text for text in said)
    assert result.integrity_ok


def test_a_true_zero_costs_one_confirming_step():
    code = "matches = [row for row in rows if float(row['amount']) > 1000]\ntotal = float(sum(float(m['amount']) for m in matches))"
    result, _ = _run(code + "\nSUBMIT(total=total)", "print(len(rows), len(matches), total)", "SUBMIT(total=total)")
    assert result.submitted and result.outputs == {"total": 0.0} and result.integrity_ok
    assert len(result.turns) == 3


def test_the_check_is_silent_on_the_last_turn_and_when_the_screen_is_off():
    last, _ = _run(SWALLOWED, max_turns=1)
    assert last.submitted and last.outputs == {"total": 0.0}           # no turn left in which to confirm
    off, _ = _run(SWALLOWED, analytical_integrity=False)
    assert off.submitted and len(off.turns) == 1
