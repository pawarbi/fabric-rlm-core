"""Source-agnostic analytical integrity helpers.

Every test here runs the same checks over evidence that could have come from
a CSV File, a Lakehouse query result, or a SemanticModel aggregate. The
source is fixed per test only so the fixture is concrete; nothing in the
assertions depends on it.
"""

from __future__ import annotations

import math
import sys
import types

import pandas as pd
import pytest

from fabric_rlm import File, SemanticModel
from fabric_rlm.analytical_integrity import (
    AnalyticalIntegrityError,
    change_direction,
    check_directional_claims,
    check_ranking_disclosure,
    classify_claim_level,
    infer_requested_ranking,
    is_material_change,
    parse_directional_claims,
    restrict_to_candidate_tuples,
    validate_analysis_integrity,
    validate_evidence_lineage,
    validate_grain,
    validate_ranking,
)

# -- materiality ------------------------------------------------------------


def test_float_noise_is_not_a_material_decline():
    assert not is_material_change(
        current=926400.0,
        baseline=926400.0000001,
        absolute_tolerance=1.0,
        direction="decrease",
    )
    assert not is_material_change(926400.0, 926400.0000001)
    assert change_direction(926400.0, 926400.0000001) == "flat"


def test_real_movement_is_a_material_decline():
    assert is_material_change(current=400713, baseline=430083, direction="decrease")
    assert not is_material_change(current=400713, baseline=430083, direction="increase")
    assert change_direction(400713, 430083) == "decrease"


def test_absolute_and_relative_tolerances_are_the_callers_rule():
    assert not is_material_change(1000, 999, absolute_tolerance=5)
    assert is_material_change(1000, 990, absolute_tolerance=5)
    assert not is_material_change(1005, 1000, relative_tolerance=0.01)
    assert is_material_change(1020, 1000, relative_tolerance=0.01)
    assert change_direction(1005, 1000, relative_tolerance=0.01) == "flat"


def test_zero_baseline_uses_only_the_absolute_rule():
    assert is_material_change(5, 0, relative_tolerance=0.5)
    assert not is_material_change(5, 0, absolute_tolerance=10)


def test_missing_and_non_numeric_values_are_never_a_change():
    assert not is_material_change(None, 10)
    assert not is_material_change(10, float("nan"))
    assert not is_material_change("n/a", 10)
    assert change_direction(None, 10) == "unknown"


def test_formatted_strings_parse_like_the_answer_prints_them():
    assert is_material_change("$3.9M", "$4.2M", direction="decrease")
    assert is_material_change("18%", "15%", direction="increase")
    assert not is_material_change("926,400.00", "926,400.00")


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        is_material_change(1, 2, direction="sideways")
    with pytest.raises(ValueError):
        is_material_change(1, 2, absolute_tolerance=-1)


# -- candidate tuples ---------------------------------------------------------

CANDIDATES = [("Cloud", "US", "Enterprise"), ("ADC", "EMEA", "Telco")]
HISTORY = [
    ("Cloud", "US", "Enterprise"),
    ("ADC", "EMEA", "Telco"),
    ("Cloud", "EMEA", "Telco"),
    ("ADC", "US", "Enterprise"),
]
KEYS = ["product", "region", "group"]


def test_tuple_restriction_excludes_cross_combinations_on_dataframes():
    history = pd.DataFrame(HISTORY, columns=KEYS).assign(arr=[1, 2, 3, 4])
    candidates = pd.DataFrame(CANDIDATES, columns=KEYS)

    kept = restrict_to_candidate_tuples(history, candidates, keys=KEYS)

    assert sorted(map(tuple, kept[KEYS].itertuples(index=False, name=None))) == sorted(CANDIDATES)
    assert ("Cloud", "EMEA", "Telco") not in set(kept[KEYS].itertuples(index=False, name=None))
    assert list(kept.columns) == KEYS + ["arr"]


def test_tuple_restriction_accepts_tuple_and_dict_candidates_and_row_lists():
    history = pd.DataFrame(HISTORY, columns=KEYS)
    by_tuples = restrict_to_candidate_tuples(history, CANDIDATES, keys=KEYS)
    by_dicts = restrict_to_candidate_tuples(
        history, [dict(zip(KEYS, c)) for c in CANDIDATES], keys=KEYS
    )
    assert len(by_tuples) == len(by_dicts) == 2

    rows = [dict(zip(KEYS, h)) for h in HISTORY]
    kept = restrict_to_candidate_tuples(rows, CANDIDATES, keys=KEYS)
    assert [tuple(r[k] for k in KEYS) for r in kept] == CANDIDATES


def test_independent_per_dimension_filters_are_what_the_helper_prevents():
    """The failure mode, spelled out: three isin filters admit four rows."""
    history = pd.DataFrame(HISTORY, columns=KEYS)
    candidates = pd.DataFrame(CANDIDATES, columns=KEYS)
    products = candidates["product"].unique()
    regions = candidates["region"].unique()
    groups = candidates["group"].unique()
    cartesian = history[
        history["product"].isin(products)
        & history["region"].isin(regions)
        & history["group"].isin(groups)
    ]
    assert len(cartesian) == 4
    assert len(restrict_to_candidate_tuples(history, candidates, keys=KEYS)) == 2


def test_tuple_restriction_reports_missing_keys():
    history = pd.DataFrame(HISTORY, columns=KEYS)
    with pytest.raises(AnalyticalIntegrityError, match="lacks the key columns"):
        restrict_to_candidate_tuples(history, CANDIDATES, keys=["product", "colour"])
    with pytest.raises(ValueError):
        restrict_to_candidate_tuples(history, CANDIDATES, keys=[])


# -- ranking ------------------------------------------------------------------

SEGMENTS = pd.DataFrame(
    {
        "segment": ["A", "B"],
        "current_size": [20_000_000, 5_000_000],
        "impact": [50_000, 500_000],
    }
)


def test_ranking_by_impact_puts_the_small_high_impact_segment_first():
    ranked = SEGMENTS.sort_values("impact", ascending=False)
    summary = validate_ranking(
        requested_concept="business impact",
        operational_definition="impact is the ARR at risk: prior-quarter ARR minus current ARR",
        ranking_metric="impact",
        ranking_values=ranked["impact"].tolist(),
    )
    assert ranked["segment"].tolist() == ["B", "A"]
    assert summary["metric"] == "impact" and summary["proxy"] is False


def test_ranking_by_current_size_does_not_answer_an_impact_request():
    ranked = SEGMENTS.sort_values("current_size", ascending=False)
    with pytest.raises(AnalyticalIntegrityError, match="rank by 'business impact'"):
        validate_ranking(
            requested_concept="business impact",
            operational_definition="segments sorted by their current size",
            ranking_metric="current_size",
            ranking_values=ranked["current_size"].tolist(),
        )


def test_a_stated_and_justified_proxy_is_accepted():
    summary = validate_ranking(
        requested_concept="business impact",
        operational_definition=(
            "current_size is used as the proxy for business impact because the "
            "driver measures are blank at this grain"
        ),
        ranking_metric="current_size",
        ranking_values=[20_000_000, 5_000_000],
    )
    assert summary["proxy"] is True


def test_ranking_needs_a_definition_and_a_sorted_result():
    with pytest.raises(AnalyticalIntegrityError, match="operational definition"):
        validate_ranking(
            requested_concept="risk",
            operational_definition="",
            ranking_metric="risk_score",
        )
    with pytest.raises(AnalyticalIntegrityError, match="not sorted"):
        validate_ranking(
            requested_concept="risk",
            operational_definition="risk_score is churn probability times ARR",
            ranking_metric="risk_score",
            ranking_values=[1, 3, 2],
        )


def test_requested_ranking_is_found_in_task_text():
    request = infer_requested_ranking(
        "Identify the five segments with the largest ARR deterioration and rank "
        "them by business impact of the deterioration, using the last three quarters."
    )
    assert request is not None
    assert request.concept.startswith("business impact of the deterioration")
    assert "impact" in request.tokens

    assert infer_requested_ranking("What was total revenue last month?") is None
    top = infer_requested_ranking("List the top 5 customers by churn risk and explain.")
    assert top is not None and top.concept == "churn risk"


def test_ranking_disclosure_requires_the_metric_to_be_visible():
    request = infer_requested_ranking("Rank segments by business impact")
    assert check_ranking_disclosure(
        "Rank | Segment | Current ARR | Change | Estimated impact\n1 | A | ...", request
    ) == []
    problems = check_ranking_disclosure("Top segments: A (20M), B (5M).", request)
    assert problems and "never mentions" in problems[0]
    problems = check_ranking_disclosure(
        "Segments ranked by current ARR. Impact was not computed.", request
    )
    assert problems and "ranked by current ARR" in problems[0]
    assert check_ranking_disclosure(
        "Segments ranked by current ARR as a proxy for impact.", request
    ) == []


# -- grain ------------------------------------------------------------------------


def test_silent_grain_substitution_is_rejected():
    with pytest.raises(AnalyticalIntegrityError, match="changed silently"):
        validate_grain(
            requested=["Product", "Region", "Customer Group"],
            actual=["Product", "Region"],
        )


def test_explained_grain_change_and_label_normalisation_pass():
    summary = validate_grain(
        requested=["Products[Line Of Business]", "Sold To[Sold_To Region]"],
        actual=["line_of_business", "sold_to_region"],
    )
    assert summary["missing"] == [] and summary["extra"] == []
    summary = validate_grain(
        requested=["Product", "Region", "Customer Group"],
        actual=["Product", "Region"],
        explanation="Customer Group was dropped because the driver measures are blank at that grain",
    )
    assert summary["missing"] == ["Customer Group"] and summary["explained"]


# -- directional prose ----------------------------------------------------------


def test_prose_decline_over_equal_values_is_flagged():
    problems = check_directional_claims(
        "DefensePro in EMEA declined from 926,400.00 in 2025/Q4 to 926,400.00 in 2026/Q2."
    )
    assert len(problems) == 1 and "effectively equal" in problems[0]


def test_prose_direction_must_match_the_numbers():
    problems = check_directional_claims("ARR fell from $3.9M to $4.2M this quarter.")
    assert len(problems) == 1 and "show a increase" in problems[0]
    assert check_directional_claims("ARR grew from 100 to 120.") == []
    assert check_directional_claims("ARR moved from 100 to 90.") == []


def test_materiality_rule_applies_to_prose_claims():
    text = "Usage declined from 1,000 to 999 users."
    assert check_directional_claims(text) == []
    flagged = check_directional_claims(text, absolute_tolerance=5)
    assert flagged and "effectively equal" in flagged[0]
    claim = parse_directional_claims(text)[0]
    assert (claim.baseline, claim.current, claim.claimed) == (1000.0, 999.0, "decrease")


# -- claim levels and lineage -------------------------------------------------------


def test_claim_levels_are_distinguishable():
    assert classify_claim_level("ARR fell from 4.2M to 3.9M.") == "observed"
    assert classify_claim_level("ARR declined 7.1%.") == "derived"
    assert classify_claim_level("This indicates weakening revenue performance.") == "interpretation"
    assert classify_claim_level("Lower product adoption caused the ARR decline.") == "causal"


def test_material_claims_need_a_source_and_causal_claims_need_causal_evidence():
    report = validate_evidence_lineage(
        [
            {"claim": "ARR is 4.2M", "value": 4.2e6, "metric": "ARR"},
            {"claim": "Lower adoption caused the decline", "source": "notes"},
        ],
        sources=["arr_model", "notes"],
    )
    assert any("has no source" in p for p in report.problems)
    assert any("causal language" in p for p in report.problems)
    ok = validate_evidence_lineage(
        [{"claim": "ARR is 4.2M", "value": 4.2e6, "source": "arr_model"}],
        sources=["arr_model"],
    )
    assert ok.ok and "cross_source" not in ok.checks


def test_unknown_sources_are_rejected():
    report = validate_evidence_lineage(
        [{"claim": "x", "value": 1, "source": "spreadsheet"}], sources=["arr_model"]
    )
    assert report.problems and "unknown source" in report.problems[0]


# -- source-agnostic scenarios ----------------------------------------------------------

SEGMENT_HISTORY = [
    # product, region, group, quarter, arr
    ("DefensePro", "AMERICA", "CARRIER", "2025/Q4", 430083.0),
    ("DefensePro", "AMERICA", "CARRIER", "2026/Q2", 400713.0),
    ("Alteon", "EMEA", "ENTERPRISE", "2025/Q4", 926400.0000001),
    ("Alteon", "EMEA", "ENTERPRISE", "2026/Q2", 926400.0),
    ("Cloud", "APAC", "TELCO", "2025/Q4", 5_000_000.0),
    ("Cloud", "APAC", "TELCO", "2026/Q2", 4_500_000.0),
    ("DefensePro", "EMEA", "ENTERPRISE", "2025/Q4", 20_000_000.0),
    ("DefensePro", "EMEA", "ENTERPRISE", "2026/Q2", 19_950_000.0),
    # Flat, never a candidate, but admitted by independent per-dimension lists.
    ("Cloud", "EMEA", "ENTERPRISE", "2025/Q4", 100_000.0),
    ("Cloud", "EMEA", "ENTERPRISE", "2026/Q2", 100_000.0),
]
GRAIN = ["product", "region", "group"]


def _analyse(frame: pd.DataFrame) -> dict:
    """The corrected analysis, written once; every source runs it unchanged."""
    wide = frame.pivot_table(index=GRAIN, columns="quarter", values="arr").reset_index()
    wide["change"] = wide["2026/Q2"] - wide["2025/Q4"]
    materiality = {"absolute_tolerance": 1_000.0}
    wide["deteriorating"] = [
        is_material_change(cur, base, direction="decrease", **materiality)
        for cur, base in zip(wide["2026/Q2"], wide["2025/Q4"])
    ]
    candidates = wide[wide["deteriorating"]].copy()
    candidates["impact"] = candidates["2025/Q4"] - candidates["2026/Q2"]
    ranked = candidates.sort_values("impact", ascending=False)
    restricted = restrict_to_candidate_tuples(frame, ranked, keys=GRAIN)
    report = validate_analysis_integrity(
        requested_grain=GRAIN,
        actual_grain=list(ranked.columns[:3]),
        ranking={
            "concept": "business impact of deterioration",
            "definition": "impact is the ARR lost between 2025/Q4 and 2026/Q2",
            "metric": "impact",
            "values": ranked["impact"].tolist(),
        },
        candidate_keys=GRAIN,
        candidates=ranked,
        selected=restricted,
        materiality=materiality,
        directional_claims=[
            {"label": tuple(r[GRAIN]), "current": r["2026/Q2"], "baseline": r["2025/Q4"], "claimed": "decrease"}
            for _, r in ranked.iterrows()
        ],
    )
    return {"ranked": ranked, "restricted": restricted, "report": report}


def _assert_integrity(result: dict) -> None:
    ranked, restricted, report = result["ranked"], result["restricted"], result["report"]
    assert report.ok, report.problems
    assert ("Alteon", "EMEA", "ENTERPRISE") not in set(ranked[GRAIN].itertuples(index=False, name=None)), (
        "the noise-only segment must be excluded"
    )
    assert ("DefensePro", "EMEA", "ENTERPRISE") in set(ranked[GRAIN].itertuples(index=False, name=None)), (
        "a 50k move is outside the 1k rule, so it stays"
    )
    assert ("Cloud", "EMEA", "ENTERPRISE") not in set(ranked[GRAIN].itertuples(index=False, name=None))
    assert ranked["impact"].tolist() == sorted(ranked["impact"].tolist(), reverse=True)
    assert ranked.iloc[0]["product"] == "Cloud", "impact, not current size, orders the list"
    assert set(restricted[GRAIN].itertuples(index=False, name=None)) == set(
        ranked[GRAIN].itertuples(index=False, name=None)
    )


def test_csv_file_analysis_obeys_the_same_checks(tmp_path):
    path = tmp_path / "segments.csv"
    pd.DataFrame(SEGMENT_HISTORY, columns=GRAIN + ["quarter", "arr"]).to_csv(path, index=False)
    handle = File(str(path))

    frame = pd.read_csv(handle.path)
    _assert_integrity(_analyse(frame))


def test_lakehouse_query_rows_obey_the_same_checks():
    """A bounded Lakehouse query returns rows; the checks work on rows too."""
    rows = [dict(zip(GRAIN + ["quarter", "arr"], r)) for r in SEGMENT_HISTORY]
    frame = pd.DataFrame(rows)
    result = _analyse(frame)
    _assert_integrity(result)

    kept_rows = restrict_to_candidate_tuples(rows, result["ranked"], keys=GRAIN)
    assert {tuple(r[k] for k in GRAIN) for r in kept_rows} == set(
        result["ranked"][GRAIN].itertuples(index=False, name=None)
    )


@pytest.fixture
def semantic_model(monkeypatch):
    columns = pd.DataFrame(
        [
            {"Table Name": "Products", "Column Name": "Product", "Data Type": "String", "Description": ""},
            {"Table Name": "Sold To", "Column Name": "Region", "Data Type": "String", "Description": ""},
            {"Table Name": "Sold To", "Column Name": "Group", "Data Type": "String", "Description": ""},
            {"Table Name": "Period", "Column Name": "Quarter", "Data Type": "String", "Description": ""},
        ]
    )
    measures = pd.DataFrame(
        [{"Table Name": "M", "Measure Name": "ARR $", "Measure Expression": "x",
          "Measure Description": "", "Measure Display Folder": ""}]
    )
    result = pd.DataFrame(
        [
            {"Products[Product]": p, "Sold To[Region]": r, "Sold To[Group]": g,
             "Period[Quarter]": q, "[__m0]": a}
            for p, r, g, q, a in SEGMENT_HISTORY
        ]
    )

    def evaluate_dax(dataset, query, **kwargs):
        if "group_count" in query:
            return pd.DataFrame({"[group_count]": [8]})
        return result.copy()

    fabric = types.ModuleType("sempy.fabric")
    fabric.list_columns = lambda *a, **k: columns
    fabric.list_measures = lambda *a, **k: measures
    fabric.list_tables = lambda *a, **k: pd.DataFrame([{"Name": "Products"}])
    fabric.list_relationships = lambda *a, **k: pd.DataFrame()
    fabric.evaluate_dax = evaluate_dax
    sempy = types.ModuleType("sempy")
    sempy.fabric = fabric
    monkeypatch.setitem(sys.modules, "sempy", sempy)
    monkeypatch.setitem(sys.modules, "sempy.fabric", fabric)
    return SemanticModel("ARR", validate=False)


def test_semantic_model_aggregate_obeys_the_same_checks(semantic_model):
    """The live regression case: aggregate() feeds the identical analysis."""
    frame = semantic_model.aggregate(
        ["ARR $"],
        groupby=["Products[Product]", "Sold To[Region]", "Sold To[Group]", "Period[Quarter]"],
    )
    frame = frame.rename(
        columns={
            "products_product": "product",
            "sold_to_region": "region",
            "sold_to_group": "group",
            "period_quarter": "quarter",
            "arr": "arr",
        }
    )
    result = _analyse(frame)
    _assert_integrity(result)

    prose = "\n".join(
        f"{r['product']} in {r['region']} ({r['group']}): ARR fell from "
        f"{r['2025/Q4']:,.2f} to {r['2026/Q2']:,.2f}; estimated impact {r['impact']:,.0f}."
        for _, r in result["ranked"].iterrows()
    )
    final = validate_analysis_integrity(
        answer_text=prose, requested_ranking="business impact", materiality={"absolute_tolerance": 1000}
    )
    assert final.ok, final.problems


def test_the_old_analysis_fails_the_same_checks_on_any_source():
    frame = pd.DataFrame(SEGMENT_HISTORY, columns=GRAIN + ["quarter", "arr"])
    wide = frame.pivot_table(index=GRAIN, columns="quarter", values="arr").reset_index()
    # 1. raw `<` calls the noise segment a decline
    naive = wide[wide["2026/Q2"] < wide["2025/Q4"]]
    assert ("Alteon", "EMEA", "ENTERPRISE") in set(naive[GRAIN].itertuples(index=False, name=None))
    # 2. ranking by current ARR instead of impact
    by_size = naive.sort_values("2026/Q2", ascending=False)
    # 3. independent per-dimension lists
    cartesian = frame[
        frame["product"].isin(naive["product"].unique())
        & frame["region"].isin(naive["region"].unique())
        & frame["group"].isin(naive["group"].unique())
    ]
    report = validate_analysis_integrity(
        ranking={
            "concept": "business impact of deterioration",
            "definition": "sorted by current ARR",
            "metric": "2026/Q2",
            "values": by_size["2026/Q2"].tolist(),
        },
        candidate_keys=GRAIN,
        candidates=naive,
        selected=cartesian,
        materiality={"absolute_tolerance": 1000},
        directional_claims=[
            {"label": tuple(r[GRAIN]), "current": r["2026/Q2"], "baseline": r["2025/Q4"], "claimed": "decrease"}
            for _, r in naive.iterrows()
        ],
        answer_text="Alteon in EMEA declined from 926,400.00 to 926,400.00. Segments ranked by current ARR.",
        requested_ranking="business impact",
    )
    text = "\n".join(report.problems)
    assert "rank by 'business impact of deterioration'" in text
    assert "never candidates" in text
    assert "effectively equal" in text
    assert "described as 'decrease'" in text
    assert "ranked by current ARR" in text
    with pytest.raises(AnalyticalIntegrityError):
        report.raise_for_problems()


def test_pdf_evidence_keeps_provenance_and_does_not_become_causal():
    """Facts extracted from a PDF stay attributed to it and stay observations."""
    claims = [
        {"claim": "Renewal date is 2026-12-31", "source": "contract_pdf", "level": "observed"},
        {"claim": "Contract value is $1.2M", "source": "contract_pdf", "value": 1.2e6, "unit": "USD"},
        {"claim": "The price increase caused the churn", "source": "contract_pdf"},
    ]
    report = validate_evidence_lineage(claims, sources=["contract_pdf"])
    assert [p for p in report.problems if "causal language" in p]
    assert not [p for p in report.problems if "no source" in p]
    assert "ranking" not in report.checks and "cross_source" not in report.checks

    numeric = validate_analysis_integrity(
        answer_text="Per the contract PDF, seats increased from 100 to 100 this year."
    )
    assert numeric.problems and "effectively equal" in numeric.problems[0]


def test_semantic_model_plus_lakehouse_requires_a_shared_key_and_aligned_periods():
    claims = [
        {"claim": "ARR is 4.2M", "source": "arr_model", "metric": "ARR", "value": 4.2e6,
         "time_basis": "2026/Q2", "entity": "Contoso"},
        {"claim": "90-day usage change is -18%", "source": "usage_lakehouse", "metric": "usage change",
         "value": -18, "time_basis": "2026-05-11", "entity": "Contoso"},
    ]
    unreconciled = validate_evidence_lineage(
        claims,
        sources=["arr_model", "usage_lakehouse"],
        joins=[{"sources": ["arr_model", "usage_lakehouse"], "match": "inferred"}],
    )
    text = "\n".join(unreconciled.problems)
    assert "no shared key" in text and "different periods" in text

    reconciled = validate_evidence_lineage(
        claims,
        sources=["arr_model", "usage_lakehouse"],
        joins=[{"sources": ["arr_model", "usage_lakehouse"], "key": "customer_id", "match": "explicit"}],
        disclosures={"period_alignment": "usage window ends one day before the ARR as-of date"},
    )
    assert reconciled.ok, reconciled.problems
    assert "cross_source" in reconciled.checks
    assert {s for c in claims for s in [c["source"]]} == {"arr_model", "usage_lakehouse"}


def test_semantic_model_plus_pdf_keeps_measured_and_commentary_apart():
    claims = [
        {"claim": "ARR declined from 4.2M to 3.9M", "source": "arr_model", "metric": "ARR",
         "level": "observed", "entity": "Contoso", "direction": "decrease", "time_basis": "2026/Q2"},
        {"claim": "Management describes the account as healthy and growing", "source": "commentary_pdf",
         "level": "interpretation", "entity": "Contoso", "direction": "increase", "time_basis": "2026/Q1"},
    ]
    hidden = validate_evidence_lineage(claims, sources=["arr_model", "commentary_pdf"])
    text = "\n".join(hidden.problems)
    assert "disagree about 'Contoso'" in text and "different periods" in text

    surfaced = validate_evidence_lineage(
        claims,
        sources=["arr_model", "commentary_pdf"],
        disclosures={
            "contradiction_surfaced": True,
            "period_alignment": "commentary predates the ARR quarter",
        },
    )
    assert surfaced.ok, surfaced.problems
    assert claims[1]["level"] == "interpretation", "commentary is not a measured fact"


def test_three_sources_synthesise_without_losing_provenance():
    claims = [
        {"claim": "ARR is 4.2M", "source": "arr_model", "metric": "ARR", "value": 4.2e6, "unit": "USD",
         "time_basis": "2026/Q2", "entity": "Contoso", "direction": "decrease"},
        {"claim": "Usage fell 18% over 90 days", "source": "usage_lakehouse", "metric": "usage",
         "value": -18, "unit": "percent", "time_basis": "2026/Q2", "entity": "Contoso", "direction": "decrease"},
        {"claim": "Renewal is due 2026-12-31", "source": "contract_pdf", "level": "observed",
         "time_basis": "2026/Q2"},
        {"claim": "Active customers is 120", "source": "arr_model", "metric": "active customers",
         "value": 120, "definition": "governed measure Active Customers #"},
        {"claim": "Active customers is 131", "source": "usage_lakehouse", "metric": "active customers",
         "value": 131, "definition": "COUNT(DISTINCT customer_id) with any usage"},
    ]
    joins = [
        {"sources": ["arr_model", "usage_lakehouse"], "key": "customer_id", "match": "explicit"},
        {"sources": ["usage_lakehouse", "contract_pdf"], "key": "customer name", "match": "inferred", "confidence": "medium"},
    ]
    unreconciled = validate_evidence_lineage(claims, sources=["arr_model", "usage_lakehouse", "contract_pdf"], joins=joins)
    assert any("different definitions" in p for p in unreconciled.problems)

    report = validate_evidence_lineage(
        claims,
        sources=["arr_model", "usage_lakehouse", "contract_pdf"],
        joins=joins,
        disclosures={"metric_definitions_reconciled": "kept as two separate counts"},
    )
    assert report.ok, report.problems
    assert any("inferred" in n for n in report.notes)


def test_units_must_be_reconciled_before_comparison():
    claims = [
        {"claim": "Revenue 1,200", "source": "ledger_csv", "metric": "revenue", "value": 1200, "unit": "thousands of USD"},
        {"claim": "Revenue 1.1M", "source": "arr_model", "metric": "revenue", "value": 1.1e6, "unit": "USD"},
    ]
    report = validate_evidence_lineage(claims, sources=["ledger_csv", "arr_model"])
    assert report.problems and "different units" in report.problems[0]
    assert validate_evidence_lineage(
        claims, sources=["ledger_csv", "arr_model"], disclosures={"unit_conversion": "thousands x 1000"}
    ).ok


def test_entry_point_activates_only_the_checks_with_inputs():
    empty = validate_analysis_integrity()
    assert empty.ok and empty.checks == []
    only_grain = validate_analysis_integrity(requested_grain=["a", "b"], actual_grain=["a", "b"])
    assert only_grain.checks == ["grain"] and only_grain.ok
    with pytest.raises(AnalyticalIntegrityError):
        validate_analysis_integrity(requested_grain=["a", "b"], actual_grain=["a"], strict=True)
    assert math.isfinite(1.0)
