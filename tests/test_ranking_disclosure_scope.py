"""The ranking-disclosure check is for narrative answers, not for the sort order of a sheet.

Found in Fabric: the check fired in 21 logged runs on tasks that say how a
table is to be sorted ("sorted descending by that average", "sorted by
department name", "revenue descending"). It read the sort key as an analytical
ranking concept and asked the answer to name it. A payload of numbers and a
file path has no prose in which to do that, so each run lost one or two turns
and a correct result came back with ``integrity_ok`` false. All four runs of
the IMF notebook were hit, and one of them damaged its workbook while
"repairing" a submission that was right.
"""

from __future__ import annotations

import pytest

from fabric_rlm import RLM
from fabric_rlm.analytical_integrity import check_ranking_disclosure, infer_requested_ranking

from test_api_arguments_engines import _ScriptedLM


@pytest.mark.parametrize(
    ("task", "concept"),
    [
        ("Rows 3 to 17: the top 15 countries, sorted by streak length descending, then start month ascending.", "streak length"),
        ("One row per region sorted by amount descending, number format #,##0.00.", "amount"),
        ("List the rows ordered by revenue desc.", "revenue"),
        ("Rank the accounts by churn risk, highest first.", "churn risk"),
    ],
)
def test_the_sort_direction_is_not_part_of_the_concept(task, concept):
    assert infer_requested_ranking(task).concept == concept


def test_a_back_reference_is_not_a_concept():
    # "that average" points at something the task defined earlier; an answer cannot be asked to name it.
    task = ("Rows 3 to 12: the 10 qualifying countries with the highest five-year average of their monthly YoY values, "
            "sorted descending by that average (ties by country code ascending).")
    assert infer_requested_ranking(task) is None
    later = task + " One row per qualifying country, sorted descending by the five-year average."
    assert infer_requested_ranking(later).concept == "five-year average"


def test_a_narrative_answer_is_still_held_to_the_concept():
    request = infer_requested_ranking("Rank segments by business impact")
    assert check_ranking_disclosure("Top segments: A (20M), B (5M).", request)
    assert check_ranking_disclosure("Ranked by business impact (ARR at risk): A (20M), B (5M).", request) == []


@pytest.mark.parametrize(
    ("task", "explicit"),
    [
        ("Rank the segments by business impact.", True),
        ("Prioritize the accounts by churn risk.", True),
        ("One row per region sorted by amount descending.", False),
        ("Rows 3 to 17: the top 15 countries, sorted by streak length descending.", False),
        ("List the rows ordered by revenue.", False),
    ],
)
def test_only_rank_and_prioritize_are_explicit_requests(task, explicit):
    assert infer_requested_ranking(task).explicit is explicit


def _run(task, *codes, outputs, **kwargs):
    lm = _ScriptedLM(list(codes))
    result = RLM.task(task, inputs={"rows": [["North", 2.5], ["South", 10.0]]}, outputs=outputs, lm=lm, max_turns=4, **kwargs).run()
    return result


def test_a_payload_with_no_prose_is_not_asked_to_name_the_sort_key():
    result = _run("Build the table sorted by amount descending and return the total.",
                  "total = sum(amount for _, amount in rows)\nprint(total)", "SUBMIT(total=total)", outputs={"total": float})
    assert result.submitted and result.integrity_ok and len(result.turns) == 2
    assert [turn.turn_type for turn in result.turns] == ["normal", "normal"]


def test_a_written_answer_that_hides_the_metric_is_still_sent_back():
    hidden = "answer = 'The regions that matter most are South and then North, which leads the other one clearly.'\nprint(answer)"
    shown = "answer = 'Ranked by amount: South (10.0), then North (2.5), so South leads the other region clearly.'\nprint(answer)"
    result = _run("Rank the regions by amount and explain.", hidden, "SUBMIT(answer=answer)", shown, "SUBMIT(answer=answer)",
                  outputs={"answer": str})
    assert result.submitted and "Ranked by amount" in result.outputs["answer"]
    assert "verifier_repair" in [turn.turn_type for turn in result.turns]


STREAKS = (
    "streaks = sorted(rows, key=lambda r: r[0])\n"          # an intermediate sort, as any streak computation needs
    "ranked = sorted(streaks, key=lambda r: -r[1])\n"
    "longest = ranked[0][0]\nprint(ranked, longest)"
)


def test_a_table_task_is_not_screened_for_the_names_its_code_sorted_by():
    # Seen in Fabric after the disclosure fix: "sorted by ['TIME_PERIOD']" and "['COUNTRY', 'avg5']" were called drift.
    task = "Rows 3 to 17: the top 15 countries, sorted by streak length descending. Return longest_country."
    result = _run(task, STREAKS, "SUBMIT(longest_country=longest)", outputs={"longest_country": str})
    assert result.submitted and result.integrity_ok
    assert [turn.turn_type for turn in result.turns] == ["normal", "normal"]


def test_an_explicit_ranking_request_is_still_screened_without_prose():
    # "Rank ... by impact" with a typed list and no prose: the code that reaches the answer sorted by something else.
    code = ("import pandas as pd\n"
            "frame = pd.DataFrame(rows, columns=['region', 'latest_arr'])\n"
            "ranked = frame.sort_values('latest_arr', ascending=False)\n"
            "top = ranked.region.tolist()\nprint(top)")
    fixed = ("frame['impact'] = frame.latest_arr * 2\nranked = frame.sort_values('impact', ascending=False)\n"
             "top = ranked.region.tolist()\nprint(top)")
    result = _run("Rank the regions by business impact.", code, "SUBMIT(top=top)", fixed, "SUBMIT(top=top)", outputs={"top": list})
    assert result.submitted
    assert "verifier_repair" in [turn.turn_type for turn in result.turns]
