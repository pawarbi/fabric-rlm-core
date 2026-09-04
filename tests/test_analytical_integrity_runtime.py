"""Analytical integrity at the runtime boundary: validators, trajectory
detectors, and the pre-SUBMIT screen that drives a repair turn.

The regression scenario mirrors the live ARR trajectory: the task asked for
a ranking by business impact, the code sorted by current ARR, candidates
chosen at Product x Region x Group were filtered by three independent lists,
and the prose called a 1e-7 difference a decline.
"""

from __future__ import annotations

import pytest

import fabric_rlm.runtime as runtime_mod
from fabric_rlm import RLM
from fabric_rlm.analytical_integrity import infer_requested_ranking
from fabric_rlm.interpreter import ExecResult
from fabric_rlm.trajectory import Trajectory, TurnRecord, detect_ranking_drift, extract_plan
from fabric_rlm.validators import (
    assert_directional_claims_consistent,
    assert_grain_preserved,
    assert_ranking_disclosed,
    chain,
)

# -- validators -------------------------------------------------------------


def test_directional_validator_rejects_prose_that_contradicts_its_numbers():
    v = assert_directional_claims_consistent("analysis")
    v({"analysis": "ARR fell from 430,083 to 400,713."})
    with pytest.raises(AssertionError, match="effectively equal"):
        v({"analysis": "ARR declined from 926,400.00 to 926,400.00."})
    with pytest.raises(AssertionError, match="show a increase"):
        v({"analysis": "ARR dropped from 3.9M to 4.2M."})


def test_directional_validator_scans_nested_payloads_when_no_key_is_given():
    v = assert_directional_claims_consistent()
    with pytest.raises(AssertionError):
        v({"report": {"sections": ["Usage declined from 10 to 10."]}})


def test_ranking_validator_is_a_no_op_without_a_ranking_request():
    assert assert_ranking_disclosed("analysis", "What is total ARR?") is None
    v = chain(assert_ranking_disclosed("analysis", "What is total ARR?"))
    v({"analysis": "Total ARR is 4.2M."})


def test_ranking_validator_requires_the_requested_metric_to_be_shown():
    v = assert_ranking_disclosed("analysis", "Rank the segments by business impact of deterioration.")
    assert v is not None
    v({"analysis": "Rank | Segment | Estimated impact\n1 | Cloud APAC | 500,000"})
    with pytest.raises(AssertionError, match="never mentions"):
        v({"analysis": "Top segments: DefensePro AMERICA (22.0M), Alteon EMEA (0.9M)."})
    with pytest.raises(AssertionError, match="ranked by current ARR"):
        v({"analysis": "Segments ranked by current ARR; impact could not be computed."})


def test_grain_validator_on_records_and_on_text():
    v = assert_grain_preserved("rows", ["product", "region", "customer group"])
    v({"rows": [{"product": "A", "region": "US", "customer_group": "Enterprise", "arr": 1}]})
    with pytest.raises(AssertionError, match="changed silently"):
        v({"rows": [{"product": "A", "region": "US", "arr": 1}]})
    v({"rows": [{"product": "A", "region": "US", "arr": 1}], "grain_note": "customer group dropped: blank measures"})

    t = assert_grain_preserved("analysis", ["product", "region", "customer group"])
    t({"analysis": "By product, region and customer group: ..."})
    with pytest.raises(AssertionError, match="requested grain"):
        t({"analysis": "By product and region: ..."})
    t({"analysis": "By product and region (grain coarsened because groups were blank)."})


# -- trajectory detectors ----------------------------------------------------------

CARTESIAN_CODE = '''
products = candidates["product"].unique()
regions = candidates["region"].unique()
groups = candidates["group"].unique()
history = seg[seg["product"].isin(products) & seg["region"].isin(regions) & seg["group"].isin(groups)]
'''

TUPLE_CODE = '''
history = restrict_to_candidate_tuples(seg, candidates, keys=["product", "region", "group"])
'''


def _turn(n: int, code: str, submitted: bool = False) -> TurnRecord:
    return TurnRecord(turn=n, code=code, stdout="", stderr="", error=None, submitted=submitted, state={})


def test_cartesian_candidate_filter_is_detected():
    issues = Trajectory(turns=[_turn(1, "candidates = wide[wide.change < 0]"), _turn(2, CARTESIAN_CODE)]).diagnose()
    kinds = [i.kind for i in issues]
    assert "cartesian_candidate_filter" in kinds
    issue = next(i for i in issues if i.kind == "cartesian_candidate_filter")
    assert issue.turn == 2
    assert "restrict_to_candidate_tuples" in issue.message and "['product', 'region', 'group']" in issue.message


def test_single_or_unrelated_isin_filters_are_not_flagged():
    single = "regions = candidates['region'].unique()\nout = seg[seg['region'].isin(regions)]"
    unrelated = (
        "regions = candidates['region'].unique()\nyears = calendar['year'].unique()\n"
        "out = seg[seg['region'].isin(regions) & seg['year'].isin(years)]"
    )
    for code in (single, unrelated, TUPLE_CODE):
        assert [i for i in Trajectory(turns=[_turn(1, code)]).diagnose() if i.kind == "cartesian_candidate_filter"] == []


def test_plan_block_is_extracted_from_the_first_turn():
    code = (
        "# ## PLAN\n# Target: rank segments by business impact\n# Approach: three quarters\n"
        "# ## VERIFY\n# later\nimport pandas as pd\n"
    )
    assert extract_plan([_turn(1, code)]) == "Target: rank segments by business impact\nApproach: three quarters"
    assert extract_plan([_turn(1, "x = 1")]) is None


def test_ranking_drift_between_plan_and_sort_field():
    turns = [
        _turn(1, "# ## PLAN\n# Rank segments by business impact of deterioration\nimport pandas as pd"),
        _turn(2, 'summary = summary.sort_values("latest_arr", ascending=False)'),
    ]
    issues = Trajectory(turns=turns).diagnose()
    drift = [i for i in issues if i.kind == "ranking_drift"]
    assert drift and drift[0].turn == 2 and "latest_arr" in drift[0].message

    fixed = turns + [_turn(3, 'summary["impact"] = summary.prior - summary.latest\nsummary = summary.sort_values("impact")')]
    assert [i for i in Trajectory(turns=fixed).diagnose() if i.kind == "ranking_drift"] == []


def test_detect_ranking_drift_with_an_explicit_request():
    request = infer_requested_ranking("rank by churn risk")
    assert detect_ranking_drift([_turn(1, "df.sort_values('revenue')")], request) is not None
    assert detect_ranking_drift([_turn(1, "df.nlargest(5, 'risk_score')")], request) is None
    assert detect_ranking_drift([_turn(1, "print(1)")], request) is None
    assert detect_ranking_drift([_turn(1, "df.sort_values('revenue')")], None) is None


# -- runtime integration -----------------------------------------------------------------


class ScriptedLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def __call__(self, *, messages):
        if not self.responses:
            raise AssertionError("No scripted LM responses left")
        return self.responses.pop(0)


class FakeInterpreter:
    def __init__(self, results):
        self._results = list(results)
        self.executed: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def configure_lm(self, spec):
        return {}

    def set_inputs(self, inputs):
        return None

    def execute(self, code):
        if "_fabric_rlm_payload" in code:
            return ExecResult(ok=True, submitted=False, stdout="", stderr="", state={})
        self.executed.append(code)
        return self._results.pop(0)


def _ran():
    return ExecResult(ok=True, submitted=False, stdout="", stderr="", state={})


def _submit(payload):
    return ExecResult(ok=True, submitted=True, stdout="", stderr="", state={}, submit_payload=payload)


TASK = (
    "Find the product x region x customer group segments whose ARR deteriorated over the "
    "last three quarters and rank them by business impact of the deterioration."
)
BAD_ANALYSIS = (
    "Segments ranked by current ARR. DefensePro in AMERICA (CARRIER) declined from "
    "926,400.00 to 926,400.00 over the period."
)
GOOD_ANALYSIS = (
    "Rank | Segment | ARR 2025/Q4 | ARR 2026/Q2 | Estimated impact\n"
    "1 | Cloud APAC TELCO | 5,000,000 | 4,500,000 | 500,000\n"
    "ARR fell from 5,000,000 to 4,500,000; impact is the ARR lost over the window."
)


def _fence(code: str) -> str:
    return f"```python\n{code}\n```"


def test_regression_bad_submission_is_sent_back_then_accepted(monkeypatch):
    fake = FakeInterpreter([_ran(), _submit({"analysis": BAD_ANALYSIS}), _ran(), _submit({"analysis": GOOD_ANALYSIS})])
    monkeypatch.setattr(runtime_mod, "Interpreter", lambda **kwargs: fake)
    lm = ScriptedLM(
        [
            _fence(CARTESIAN_CODE + '\nsummary = history.sort_values("latest_arr", ascending=False)'),
            _fence(f"SUBMIT(analysis={BAD_ANALYSIS!r})"),
            _fence(
                'summary["impact"] = summary["prior"] - summary["latest"]\n'
                'summary = summary.sort_values("impact", ascending=False)\n'
                + TUPLE_CODE
            ),
            _fence(f"SUBMIT(analysis={GOOD_ANALYSIS!r})"),
        ]
    )
    rlm = RLM.from_task(TASK, outputs=["analysis"], lm=lm, max_turns=6, timeout=5)

    result = rlm.run()

    assert result.submitted is True
    assert result.payload == {"analysis": GOOD_ANALYSIS}
    types_ = [t.turn_type for t in result.trajectory]
    assert "verifier_repair" in types_
    history = result.trajectory.metadata["verifier_repair_history"]
    assert history[0]["skill"] == "analytical_integrity"
    rejection = history[0]["assertion"]
    assert "effectively equal" in rejection
    assert "ranked by current ARR" in rejection
    assert "sorted by" in rejection and "latest_arr" in rejection
    assert "restrict_to_candidate_tuples" in rejection
    assert "analytical_integrity_unresolved" not in result.trajectory.metadata


def test_screen_can_be_disabled_by_argument_and_by_environment(monkeypatch):
    for disable in ("argument", "environment"):
        fake = FakeInterpreter([_submit({"analysis": BAD_ANALYSIS})])
        monkeypatch.setattr(runtime_mod, "Interpreter", lambda **kwargs: fake)
        monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)
        kwargs = {}
        if disable == "argument":
            kwargs["analytical_integrity"] = False
        else:
            monkeypatch.setenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", "0")
        lm = ScriptedLM([_fence(f"SUBMIT(analysis={BAD_ANALYSIS!r})")])
        result = RLM.from_task(TASK, outputs=["analysis"], lm=lm, max_turns=3, timeout=5, **kwargs).run()
        assert result.submitted and result.payload == {"analysis": BAD_ANALYSIS}
        assert [t.turn_type for t in result.trajectory] == ["normal"]


def test_screen_gives_up_after_two_rejections_and_records_the_findings(monkeypatch):
    fake = FakeInterpreter([_submit({"analysis": BAD_ANALYSIS})] * 3)
    monkeypatch.setattr(runtime_mod, "Interpreter", lambda **kwargs: fake)
    monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)
    lm = ScriptedLM([_fence(f"SUBMIT(analysis={BAD_ANALYSIS!r})")] * 3)

    result = RLM.from_task(TASK, outputs=["analysis"], lm=lm, max_turns=6, timeout=5).run()

    assert result.submitted is True
    assert [t.turn_type for t in result.trajectory] == ["normal", "verifier_repair", "verifier_repair"]
    unresolved = result.trajectory.metadata["analytical_integrity_unresolved"]
    assert any("effectively equal" in p for p in unresolved)


def test_a_clean_answer_to_a_plain_question_is_not_touched(monkeypatch):
    fake = FakeInterpreter([_submit({"answer": "Total ARR is 4.2M, up from 3.9M a year ago."})])
    monkeypatch.setattr(runtime_mod, "Interpreter", lambda **kwargs: fake)
    monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)
    lm = ScriptedLM([_fence("SUBMIT(answer='Total ARR is 4.2M, up from 3.9M a year ago.')")])

    result = RLM.from_task("What is total ARR?", outputs=["answer"], lm=lm, max_turns=3, timeout=5).run()

    assert result.submitted and [t.turn_type for t in result.trajectory] == ["normal"]
    assert "verifier_repair_history" not in result.trajectory.metadata
