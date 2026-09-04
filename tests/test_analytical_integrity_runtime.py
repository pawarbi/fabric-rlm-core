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

LIVE_REPAIR_CODE = '''
seg_3q = restrict_to_candidate_tuples(seg_3q, seg_latest, keys=["product", "region", "customer_group"])
'''

# The live pattern, verbatim in shape: independent lists from the candidate
# frame fed together to aggregate(filters=...) on a semantic model.
LIVE_AGGREGATE_CODE = '''
prod_vals = seg_latest["product"].unique().tolist()
reg_vals = seg_latest["region"].unique().tolist()
cust_vals = seg_latest["customer_group"].unique().tolist()

seg_3q = business_model.aggregate(
    measures=["ARR $ Basic", "Active Customers #"],
    groupby=["Products[Product]", "Sold To[Region]", "Sold To[Customer Group]", "Period[YearQuarter]"],
    filters={
        "Products[Product]": prod_vals,
        "Sold To[Region]": reg_vals,
        "Sold To[Customer Group]": cust_vals,
    },
)
'''


def _turn(n: int, code: str, submitted: bool = False) -> TurnRecord:
    return TurnRecord(turn=n, code=code, stdout="", stderr="", error=None, submitted=submitted, state={})


def test_cartesian_candidate_filter_is_detected():
    issues = Trajectory(turns=[_turn(1, "candidates = wide[wide.change < 0]"), _turn(2, CARTESIAN_CODE)]).diagnose()
    kinds = [i.kind for i in issues]
    assert "cartesian_candidate_filter" in kinds
    issue = next(i for i in issues if i.kind == "cartesian_candidate_filter")
    assert issue.turn == 2
    assert "restrict_to_candidate_tuples" in issue.message
    for col in ("product", "region", "group"):
        assert col in issue.message


def test_the_live_aggregate_filters_pattern_is_detected():
    """Regression for the trajectory that motivated the feature."""
    issues = Trajectory(turns=[_turn(1, "seg_latest = wide[wide.deteriorating]"), _turn(2, LIVE_AGGREGATE_CODE)]).diagnose()
    flagged = [i for i in issues if i.kind == "cartesian_candidate_filter"]
    assert flagged and flagged[0].turn == 2
    assert "aggregate" in flagged[0].message and "seg_latest" in flagged[0].message
    for name in ("prod_vals", "reg_vals", "cust_vals"):
        assert name in flagged[0].message


def test_lists_consumed_by_other_operations_are_also_detected():
    frame_rebuild = (
        "products = c['product'].unique()\nregions = c['region'].unique()\n"
        "pairs = pd.DataFrame({'product': products, 'region': regions})"
    )
    query = (
        "products = c['product'].unique().tolist()\nregions = c['region'].unique().tolist()\n"
        "rows = source.query(products=products, regions=regions)"
    )
    for code in (frame_rebuild, query):
        assert [i for i in Trajectory(turns=[_turn(1, code)]).diagnose() if i.kind == "cartesian_candidate_filter"]


def test_looking_at_lists_or_using_one_is_not_flagged():
    looked_at = "products = c['product'].unique()\nregions = c['region'].unique()\nprint(products, regions)\nn = len(regions)"
    one_list = "products = c['product'].unique().tolist()\nregions = c['region'].unique().tolist()\nout = model.aggregate(['x'], filters={'P': products})"
    same_column_twice = "a = c['product'].unique()\nb = c['product'].unique()\nout = f(a, b)"
    for code in (looked_at, one_list, same_column_twice):
        assert [i for i in Trajectory(turns=[_turn(1, code)]).diagnose() if i.kind == "cartesian_candidate_filter"] == []


def test_same_turn_tuple_restriction_is_a_repair():
    """Retrieve a bounded superset, then restore exact identity at once."""
    code = LIVE_AGGREGATE_CODE + LIVE_REPAIR_CODE
    assert [i for i in Trajectory(turns=[_turn(1, code)]).diagnose() if i.kind == "cartesian_candidate_filter"] == []
    inline = (
        "prod_vals = c['product'].unique().tolist()\nreg_vals = c['region'].unique().tolist()\n"
        "history = restrict_to_candidate_tuples(model.aggregate(['x'], filters={'P': prod_vals, 'R': reg_vals}), c, keys=['product', 'region'])"
    )
    assert [i for i in Trajectory(turns=[_turn(1, inline)]).diagnose() if i.kind == "cartesian_candidate_filter"] == []
    merged = LIVE_AGGREGATE_CODE + "\nseg_3q = seg_3q.merge(seg_latest[['product', 'region', 'customer_group']].drop_duplicates(), on=['product', 'region', 'customer_group'])"
    assert [i for i in Trajectory(turns=[_turn(1, merged)]).diagnose() if i.kind == "cartesian_candidate_filter"] == []


def test_an_unrelated_merge_or_partial_restriction_does_not_repair():
    unrelated = [_turn(1, LIVE_AGGREGATE_CODE), _turn(2, "lookup = accounts.merge(owners, on='owner_id')")]
    assert [i for i in Trajectory(turns=unrelated).diagnose() if i.kind == "cartesian_candidate_filter"]
    partial = [_turn(1, LIVE_AGGREGATE_CODE), _turn(2, "seg_3q = restrict_to_candidate_tuples(seg_3q, seg_latest, keys=['product', 'region'])")]
    assert [i for i in Trajectory(turns=partial).diagnose() if i.kind == "cartesian_candidate_filter"]
    before = [_turn(1, "seg_3q = restrict_to_candidate_tuples(seg_3q, seg_latest, keys=['product', 'region', 'customer_group'])"), _turn(2, LIVE_AGGREGATE_CODE)]
    assert [i for i in Trajectory(turns=before).diagnose() if i.kind == "cartesian_candidate_filter"]


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


def test_defining_the_metric_is_not_ranking_by_it():
    """The review's false negative: an impact column exists, the sort ignores it."""
    request = infer_requested_ranking("rank by business impact")
    turns = [_turn(1, 'summary["impact"] = summary["prior"] - summary["current"]\nsummary = summary.sort_values("latest_arr", ascending=False)')]
    issue = detect_ranking_drift(turns, request)
    assert issue is not None and issue.kind == "ranking_drift"
    assert "latest_arr" in issue.message and "'impact' was defined but the ranking that reaches the answer did not use it" in issue.message


def test_the_sort_that_reaches_the_answer_is_the_ranking():
    """Lineage from SUBMIT decides which sort is the ranking, not the clock."""
    request = infer_requested_ranking("rank by business impact")
    overwritten = [
        _turn(1, 'summary["impact"] = summary.prior - summary.current\nsummary = summary.sort_values("impact")'),
        _turn(2, 'summary = summary.sort_values("latest_arr", ascending=False)'),
        _turn(3, 'SUBMIT(analysis=summary.to_string())', submitted=True),
    ]
    assert detect_ranking_drift(overwritten, request) is not None
    resorted_by_derived = [
        _turn(1, 'summary["impact"] = summary.prior - summary.current'),
        _turn(2, 'summary["rank"] = summary["impact"].rank(ascending=False)\nsummary = summary.sort_values("rank")'),
        _turn(3, 'SUBMIT(analysis=summary.to_string())', submitted=True),
    ]
    assert detect_ranking_drift(resorted_by_derived, request) is None


def test_a_display_sort_of_supporting_detail_is_not_the_ranking():
    """The review's false positive: a later quarter sort feeds the answer too."""
    request = infer_requested_ranking("rank by business impact")
    turns = [
        _turn(1, 'ranked = summary.sort_values("impact", ascending=False).head(5)'),
        _turn(2, 'detail = history.sort_values("quarter")\nlines = []\nfor _, r in ranked.iterrows():\n    lines.append(str(r))'),
        _turn(3, 'SUBMIT(analysis=build_answer(ranked, detail, lines))', submitted=True),
    ]
    assert detect_ranking_drift(turns, request) is None


def test_without_lineage_any_concept_sort_satisfies_the_check():
    request = infer_requested_ranking("rank by business impact")
    turns = [
        _turn(1, 'summary = summary.sort_values("impact")'),
        _turn(2, 'summary = summary.sort_values("latest_arr")'),
    ]
    assert detect_ranking_drift(turns, request) is None


def test_polars_sorted_and_sql_order_by_are_read():
    request = infer_requested_ranking("rank by churn risk")
    assert detect_ranking_drift([_turn(1, 'out = frame.sort("revenue", descending=True)')], request) is not None
    assert detect_ranking_drift([_turn(1, 'out = frame.sort("risk_score", descending=True)')], request) is None
    assert detect_ranking_drift([_turn(1, 'rows = sorted(rows, key=lambda r: r["revenue"], reverse=True)')], request) is not None
    assert detect_ranking_drift([_turn(1, 'rows = sorted(rows, key=lambda r: r.risk_score, reverse=True)')], request) is None
    assert detect_ranking_drift([_turn(1, 'df = con.sql("SELECT * FROM t ORDER BY revenue DESC LIMIT 5").df()')], request) is not None
    assert detect_ranking_drift([_turn(1, 'df = con.sql("SELECT * FROM t ORDER BY t.risk_score DESC")')], request) is None


def test_a_metric_the_answer_declares_as_the_ranking_is_accepted():
    """From the live CSV run: the code sorted by abs_loss and a history sort
    by period_order fed the answer; the answer said 'Ranking metric:
    absolute ARR loss (USD)'. That is a disclosed ranking, not drift."""
    request = infer_requested_ranking("rank them by business impact of the deterioration")
    turns = [
        _turn(1, 'ranked = candidates.sort_values(["abs_loss", "pct_loss"], ascending=[False, False])\nlines = []\nfor i, r in ranked.iterrows():\n    seg_hist = hist[hist.product == r.product].sort_values("period_order")\n    lines.append(str(r))'),
        _turn(2, 'analysis_text = "\\n".join(lines)\nSUBMIT(analysis=analysis_text)', submitted=True),
    ]
    undeclared = "Rank 1: Cloud DDoS. Business impact was assessed."
    assert detect_ranking_drift(turns, request, answer_text=undeclared) is not None
    declared = "Ranking metric: absolute ARR loss (USD) from 2025/Q4 to 2026/Q2. Rank 1: Cloud DDoS, business impact $1,500,000."
    assert detect_ranking_drift(turns, request, answer_text=declared) is None
    parquet_style = "Methodology: ranked by absolute USD loss (arr_2025Q4 - arr_2026Q2); business impact is the ARR lost."
    parquet_turns = [_turn(1, 'candidates = candidates.sort_values("abs_loss_usd", ascending=False)'), _turn(2, 'SUBMIT(analysis=str(candidates))', submitted=True)]
    assert detect_ranking_drift(parquet_turns, request, answer_text=parquet_style) is None
    assert detect_ranking_drift(parquet_turns, request, answer_text="Segments ranked by current ARR.") is not None


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
# No ranking metric declared, the concept never mentioned, a decline claimed
# over equal values: every screen has something to say about this one.
BAD_ANALYSIS = (
    "Segments listed by current ARR. DefensePro in AMERICA (CARRIER) declined from "
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
    monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)
    lm = ScriptedLM(
        [
            _fence(
                LIVE_AGGREGATE_CODE
                + '\nsummary["impact"] = seg_3q.prior - seg_3q.latest'
                + '\nsummary = summary.sort_values("latest_arr", ascending=False)'
            ),
            _fence(f"SUBMIT(analysis={BAD_ANALYSIS!r})"),
            _fence(
                'summary["impact"] = summary["prior"] - summary["latest"]\n'
                'summary = summary.sort_values("impact", ascending=False)\n'
                + LIVE_REPAIR_CODE
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
    assert "never mentions" in rejection
    assert "sorted by" in rejection and "latest_arr" in rejection
    assert "'impact' was defined but the ranking that reaches the answer did not use it" in rejection
    assert "aggregate consumes independent lists" in rejection
    assert "restrict_to_candidate_tuples" in rejection
    assert "analytical_integrity_unresolved" not in result.trajectory.metadata
    assert result.integrity_ok is True and result.integrity_problems == []


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
    assert result.integrity_ok is False
    assert result.integrity_problems == unresolved


def test_strict_mode_never_accepts_a_submission_with_findings(monkeypatch):
    fake = FakeInterpreter([_submit({"analysis": BAD_ANALYSIS})] * 4)
    monkeypatch.setattr(runtime_mod, "Interpreter", lambda **kwargs: fake)
    monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)
    lm = ScriptedLM([_fence(f"SUBMIT(analysis={BAD_ANALYSIS!r})")] * 4)

    result = RLM.from_task(TASK, outputs=["analysis"], lm=lm, max_turns=4, timeout=5, analytical_integrity="strict").run()

    assert result.submitted is False
    assert result.integrity_ok is False
    assert any("effectively equal" in p for p in result.integrity_problems)
    assert all(t.turn_type == "verifier_repair" for t in result.trajectory.turns[1:])


def test_integrity_mode_values_are_validated():
    with pytest.raises(ValueError, match="analytical_integrity"):
        RLM.from_task("x", outputs=["a"], lm=ScriptedLM([]), analytical_integrity="lenient")


def test_leaked_object_repr_is_sent_back(monkeypatch):
    leaked = "Ranked by impact.\n1. Product='<bound method Series.prod of product Cloud DDoS>'"
    fixed = "Ranked by impact.\n1. Cloud DDoS | APAC | TELCO: impact $1,500,000."
    fake = FakeInterpreter([_submit({"analysis": leaked}), _submit({"analysis": fixed})])
    monkeypatch.setattr(runtime_mod, "Interpreter", lambda **kwargs: fake)
    monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)
    lm = ScriptedLM([_fence(f"SUBMIT(analysis={leaked!r})"), _fence(f"SUBMIT(analysis={fixed!r})")])

    result = RLM.from_task("Rank segments by impact.", outputs=["analysis"], lm=lm, max_turns=4, timeout=5).run()

    assert result.payload == {"analysis": fixed}
    history = result.trajectory.metadata["verifier_repair_history"]
    assert "object representation" in history[0]["assertion"]


def test_defined_metric_note_names_only_head_related_columns():
    request = infer_requested_ranking("rank by business impact of the deterioration")
    turns = [
        _turn(1, 'rank_df["deterioration_timing"] = "inside"\nrank_df = rank_df.sort_values("latest_arr")'),
        _turn(2, 'SUBMIT(analysis=str(rank_df))', submitted=True),
    ]
    issue = detect_ranking_drift(turns, request)
    assert issue is not None
    assert "deterioration_timing" not in issue.message
    assert "No metric for" in issue.message


def test_zero_change_item_is_sent_back_when_the_task_is_about_change(monkeypatch):
    listed = "Deteriorated segments:\n1. Alteon | EMEA | ENTERPRISE\n   Change: $-0 (-0.0%)   Ranked-by: abs drop = $0\n"
    fixed = "Deteriorated segments (materiality: >= $1,000):\n1. Cloud DDoS | APAC | TELCO\n   Change: $-1,500,000   Ranked-by: abs drop = $1,500,000\n"
    fake = FakeInterpreter([_submit({"analysis": listed}), _submit({"analysis": fixed})])
    monkeypatch.setattr(runtime_mod, "Interpreter", lambda **kwargs: fake)
    monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)
    lm = ScriptedLM([_fence(f"SUBMIT(analysis={listed!r})"), _fence(f"SUBMIT(analysis={fixed!r})")])

    result = RLM.from_task("List the segments whose ARR deteriorated.", outputs=["analysis"], lm=lm, max_turns=4, timeout=5).run()

    assert result.payload == {"analysis": fixed}
    assert "zero" in result.trajectory.metadata["verifier_repair_history"][0]["assertion"]


def test_zero_change_is_not_flagged_when_the_task_is_not_about_change(monkeypatch):
    listed = "Inventory by segment:\n1. Alteon | EMEA | ENTERPRISE\n   Change: $0\n"
    fake = FakeInterpreter([_submit({"analysis": listed})])
    monkeypatch.setattr(runtime_mod, "Interpreter", lambda **kwargs: fake)
    monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)
    lm = ScriptedLM([_fence(f"SUBMIT(analysis={listed!r})")])

    result = RLM.from_task("List ARR by segment.", outputs=["analysis"], lm=lm, max_turns=3, timeout=5).run()

    assert result.payload == {"analysis": listed}
    assert "verifier_repair_history" not in result.trajectory.metadata


def test_a_clean_answer_to_a_plain_question_is_not_touched(monkeypatch):
    fake = FakeInterpreter([_submit({"answer": "Total ARR is 4.2M, up from 3.9M a year ago."})])
    monkeypatch.setattr(runtime_mod, "Interpreter", lambda **kwargs: fake)
    monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)
    lm = ScriptedLM([_fence("SUBMIT(answer='Total ARR is 4.2M, up from 3.9M a year ago.')")])

    result = RLM.from_task("What is total ARR?", outputs=["answer"], lm=lm, max_turns=3, timeout=5).run()

    assert result.submitted and [t.turn_type for t in result.trajectory] == ["normal"]
    assert "verifier_repair_history" not in result.trajectory.metadata


# -- mixed-input prompt checklist -------------------------------------------------------


def test_prompt_adds_cross_source_checklist_only_with_several_evidence_inputs(tmp_path):
    from fabric_rlm import File, LakehouseSource, SemanticModel
    from fabric_rlm.prompts import build_system_prompt

    csv = tmp_path / "usage.csv"
    csv.write_text("customer_id,usage\n1,10\n", encoding="utf-8")
    pdf = tmp_path / "contract.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    single = build_system_prompt(inline_task="t", inputs={"arr": SemanticModel("ARR", validate=False)}, inline_outputs=["a"])
    assert "Several evidence inputs are bound" not in single

    mixed = build_system_prompt(
        inline_task="t",
        inputs={"arr": SemanticModel("ARR", validate=False), "usage": File(str(csv)), "contract": File(str(pdf))},
        inline_outputs=["a"],
    )
    assert "Several evidence inputs are bound (arr, usage, contract)" in mixed
    assert "explicit shared key" in mixed and "report the disagreement" in mixed

    files_only = build_system_prompt(
        inline_task="t", inputs={"usage": File(str(csv)), "contract": File(str(pdf))}, inline_outputs=["a"]
    )
    assert "Several evidence inputs are bound" in files_only, "activation is by count, not by input class"

    nested_list = build_system_prompt(
        inline_task="t", inputs={"sources": [File(str(csv)), File(str(pdf))]}, inline_outputs=["a"]
    )
    assert "Several evidence inputs are bound (sources[0], sources[1])" in nested_list
    nested_dict = build_system_prompt(
        inline_task="t",
        inputs={"customer": {"arr": SemanticModel("ARR", validate=False), "usage": File(str(csv))}},
        inline_outputs=["a"],
    )
    assert "Several evidence inputs are bound (customer.arr, customer.usage)" in nested_dict


def test_evidence_sources_are_recognised_by_marker_not_by_class(tmp_path):
    from fabric_rlm import FileDestination
    from fabric_rlm.prompts import evidence_leaves, is_evidence_source

    class FutureSource:
        __rlm_evidence_source__ = True

    class NotEvidence:
        pass

    assert is_evidence_source(FutureSource())
    assert not is_evidence_source(NotEvidence())
    assert not is_evidence_source(object.__new__(FileDestination))
    assert evidence_leaves({"a": FutureSource(), "b": {"c": FutureSource()}, "d": NotEvidence()}) == ["a", "b.c"]
