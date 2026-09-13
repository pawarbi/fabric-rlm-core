"""Tests for the generalization grader.

Every case below is either taken from an observed live trial or constructed to
pin a specific way the grader could mislead. The two failure modes that matter
are opposite and both damaging:

- scoring a correct answer as ``confident_wrong`` (what the verbatim grader
  did, on 6 of 6 answered smoke trials), and
- loosening the comparison until a genuinely wrong answer scores correct.

The second is the more dangerous one, so most of these tests exist to pin
things the grader must *keep* rejecting.
"""
from __future__ import annotations

import pytest

from evaluation.generalization.grader import grade_answer


# ---------------------------------------------------------------------------
# observed live trials: right value, rejected on free-text framing
# ---------------------------------------------------------------------------
def test_observed_inventory_trial_is_correct() -> None:
    """Arm A, inventory_available_units. Value, units and period all matched;
    only the free-text grain wording differed, and it was scored wrong."""
    answer = {
        "status": "success",
        "value": 186,
        "units": "units",
        "period": "2026-03-31",
        "grain": "latest inventory snapshot product grain",
    }
    expected = {
        "expected_status": "answered",
        "value": 186,
        "units": "units",
        "period": "2026-03-31",
        "grain": "latest warehouse-product snapshot rows",
    }
    result = grade_answer(answer, expected)
    assert result["outcome"] == "correct"
    assert result["correct"] is True


def test_strict_score_is_reported_alongside() -> None:
    """The old verbatim result must stay visible so the looser grader cannot
    quietly inflate a number without the change being auditable."""
    answer = {
        "status": "success",
        "value": 186,
        "units": "units",
        "period": "2026-03-31",
        "grain": "latest inventory snapshot product grain",
    }
    expected = {
        "expected_status": "answered",
        "value": 186,
        "units": "units",
        "period": "2026-03-31",
        "grain": "latest warehouse-product snapshot rows",
    }
    result = grade_answer(answer, expected)
    assert result["strict"]["correct"] is False
    assert result["correct"] is True


def test_grain_is_scored_but_does_not_decide_correctness() -> None:
    answer = {
        "status": "success",
        "value": 186,
        "units": "units",
        "period": "2026-03-31",
        "grain": "latest inventory snapshot product grain",
    }
    expected = {
        "expected_status": "answered",
        "value": 186,
        "units": "units",
        "period": "2026-03-31",
        "grain": "latest warehouse-product snapshot rows",
    }
    result = grade_answer(answer, expected)
    assert result["grain"]["match"] in {"exact", "equivalent"}
    assert "grain" not in result["checks"]


# ---------------------------------------------------------------------------
# period: the real weakness the verbatim grader was hiding
# ---------------------------------------------------------------------------
def test_period_naming_a_filter_instead_of_a_period_is_wrong() -> None:
    """Observed twice: the model put its filter condition in the period field.
    Same shape as the dbo run submitting status 'overdue'. This is a real
    defect and must keep failing."""
    answer = {"status": "success", "value": 1800, "units": "units",
              "period": "complete reporting periods", "grain": "production rows"}
    expected = {"expected_status": "answered", "value": 1800, "units": "units",
                "period": "2026-01", "grain": "complete production periods"}
    result = grade_answer(answer, expected)
    assert result["correct"] is False
    assert result["checks"]["period"] is False
    assert result["checks"]["value"] is True


def test_missing_period_is_wrong_not_merely_unreported() -> None:
    answer = {"status": "success", "value": 0.6666666666666666,
              "units": "ratio", "period": None, "grain": "ticket"}
    expected = {"expected_status": "answered", "value": 0.6666666666666666,
                "units": "ratio", "period": "2026-03-01 through 2026-03-03",
                "grain": "ticket"}
    result = grade_answer(answer, expected)
    assert result["correct"] is False
    assert result["checks"]["period"] is False


def test_period_with_extra_precision_matches() -> None:
    """A month reference answered as that month's date range is not an error."""
    answer = {"status": "success", "value": 1800, "units": "units",
              "period": "2026-01-01 through 2026-01-31", "grain": "period"}
    expected = {"expected_status": "answered", "value": 1800, "units": "units",
                "period": "2026-01", "grain": "period"}
    assert grade_answer(answer, expected)["checks"]["period"] is True


def test_period_superset_matches_open_ended_reference() -> None:
    answer = {"status": "success", "value": 1, "units": "units",
              "period": "2026-03-01 through 2026-03-18", "grain": "x"}
    expected = {"expected_status": "answered", "value": 1, "units": "units",
                "period": "through 2026-03-18", "grain": "x"}
    assert grade_answer(answer, expected)["checks"]["period"] is True


def test_wrong_period_is_rejected() -> None:
    answer = {"status": "success", "value": 1800, "units": "units",
              "period": "2026-02", "grain": "x"}
    expected = {"expected_status": "answered", "value": 1800, "units": "units",
                "period": "2026-01", "grain": "x"}
    assert grade_answer(answer, expected)["checks"]["period"] is False


# ---------------------------------------------------------------------------
# value: tolerant of representation, never of magnitude
# ---------------------------------------------------------------------------
def test_numeric_string_value_matches() -> None:
    answer = {"status": "success", "value": "186", "units": "units",
              "period": "2026-03-31", "grain": "x"}
    expected = {"expected_status": "answered", "value": 186, "units": "units",
                "period": "2026-03-31", "grain": "x"}
    assert grade_answer(answer, expected)["checks"]["value"] is True


def test_thousands_separator_value_matches() -> None:
    answer = {"status": "success", "value": "1,800", "units": "units",
              "period": "2026-01", "grain": "x"}
    expected = {"expected_status": "answered", "value": 1800, "units": "units",
                "period": "2026-01", "grain": "x"}
    assert grade_answer(answer, expected)["checks"]["value"] is True


def test_float_and_int_forms_match() -> None:
    answer = {"status": "success", "value": 1800.0, "units": "units",
              "period": "2026-01", "grain": "x"}
    expected = {"expected_status": "answered", "value": 1800, "units": "units",
                "period": "2026-01", "grain": "x"}
    assert grade_answer(answer, expected)["checks"]["value"] is True


def test_off_by_one_value_is_confidently_wrong() -> None:
    answer = {"status": "success", "value": 187, "units": "units",
              "period": "2026-03-31", "grain": "x"}
    expected = {"expected_status": "answered", "value": 186, "units": "units",
                "period": "2026-03-31", "grain": "x"}
    result = grade_answer(answer, expected)
    assert result["outcome"] == "confident_wrong"
    assert result["checks"]["value"] is False


def test_percent_form_is_not_silently_converted_to_a_ratio() -> None:
    """66.67 is not 0.6667. Converting would mask a real units error, and the
    units field is graded separately precisely so this stays visible."""
    answer = {"status": "success", "value": 66.67, "units": "percent",
              "period": "2026-03-01 through 2026-03-03", "grain": "ticket"}
    expected = {"expected_status": "answered", "value": 0.6666666666666666,
                "units": "ratio", "period": "2026-03-01 through 2026-03-03",
                "grain": "ticket"}
    result = grade_answer(answer, expected)
    assert result["correct"] is False
    assert result["checks"]["value"] is False


def test_serialization_marker_is_never_correct() -> None:
    """The defect this evaluation exists to catch: an opaque marker in the
    value slot is an incomplete answer, not a correct one."""
    answer = {"status": "success",
              "value": {"__type__": "int64", "__serializable__": False},
              "units": "units", "period": "2026-03-31", "grain": "x"}
    expected = {"expected_status": "answered", "value": 186, "units": "units",
                "period": "2026-03-31", "grain": "x"}
    result = grade_answer(answer, expected)
    assert result["correct"] is False
    assert result["outcome"] == "incomplete"
    assert result["unserializable"] is True


# ---------------------------------------------------------------------------
# units: alias-tolerant, not meaning-blind
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("got,ref", [
    ("ratio", "proportion"),
    ("units", "unit"),
    ("Units", "units"),
    ("  units ", "units"),
    ("tickets", "ticket"),
])
def test_equivalent_units_match(got: str, ref: str) -> None:
    answer = {"status": "success", "value": 1, "units": got,
              "period": "2026-01", "grain": "x"}
    expected = {"expected_status": "answered", "value": 1, "units": ref,
                "period": "2026-01", "grain": "x"}
    assert grade_answer(answer, expected)["checks"]["units"] is True


@pytest.mark.parametrize("got,ref", [
    ("percent", "ratio"),
    ("minutes", "hours"),
    ("usd", "units"),
])
def test_different_units_are_rejected(got: str, ref: str) -> None:
    answer = {"status": "success", "value": 1, "units": got,
              "period": "2026-01", "grain": "x"}
    expected = {"expected_status": "answered", "value": 1, "units": ref,
                "period": "2026-01", "grain": "x"}
    assert grade_answer(answer, expected)["checks"]["units"] is False


# ---------------------------------------------------------------------------
# grain scoring
# ---------------------------------------------------------------------------
def test_grain_trivial_variation_is_equivalent() -> None:
    answer = {"status": "success", "value": 1, "units": "units",
              "period": "2026-01", "grain": "ticket_id"}
    expected = {"expected_status": "answered", "value": 1, "units": "units",
                "period": "2026-01", "grain": "ticket"}
    assert grade_answer(answer, expected)["grain"]["match"] in {"exact", "equivalent"}


def test_grain_losing_a_qualifier_is_a_mismatch() -> None:
    """'production rows' drops 'complete', which is the whole point of the
    question. That is a real loss of meaning and must be recorded as one."""
    answer = {"status": "success", "value": 1800, "units": "units",
              "period": "2026-01", "grain": "production rows"}
    expected = {"expected_status": "answered", "value": 1800, "units": "units",
                "period": "2026-01", "grain": "complete production periods"}
    assert grade_answer(answer, expected)["grain"]["match"] == "mismatch"


def test_missing_grain_is_recorded() -> None:
    answer = {"status": "success", "value": 1800, "units": "units",
              "period": "2026-01", "grain": None}
    expected = {"expected_status": "answered", "value": 1800, "units": "units",
                "period": "2026-01", "grain": "complete production periods"}
    assert grade_answer(answer, expected)["grain"]["match"] == "missing"


# ---------------------------------------------------------------------------
# behaviour preserved from the original grader
# ---------------------------------------------------------------------------
def test_null_value_is_incomplete() -> None:
    answer = {"status": "success", "value": None, "units": "units",
              "period": None, "grain": None}
    expected = {"expected_status": "answered", "value": 186, "units": "units",
                "period": "2026-03-31", "grain": "x"}
    result = grade_answer(answer, expected)
    assert result["outcome"] == "incomplete"
    assert result["correct"] is False


def test_abstention_expected_and_given() -> None:
    answer = {"status": "abstain", "value": "no root-cause field is provided"}
    expected = {"expected_status": "abstain"}
    result = grade_answer(answer, expected)
    assert result["outcome"] == "correct"
    assert result["correct"] is True


def test_answering_when_abstention_was_required_is_confidently_wrong() -> None:
    answer = {"status": "success", "value": "network latency", "units": "n/a",
              "period": "2026-03", "grain": "ticket"}
    expected = {"expected_status": "abstain"}
    result = grade_answer(answer, expected)
    assert result["outcome"] == "confident_wrong"
    assert result["correct"] is False


def test_timeout_status_is_incomplete() -> None:
    answer = {"status": "timeout"}
    expected = {"expected_status": "answered", "value": 1}
    assert grade_answer(answer, expected)["outcome"] == "incomplete"


def test_entity_identity_is_still_checked() -> None:
    answer = {"status": "success", "value": "Northwind", "units": "customer",
              "period": "2026-03", "grain": "customer",
              "entity_id": "C-002"}
    expected = {"expected_status": "answered", "value": "Northwind",
                "units": "customer", "period": "2026-03", "grain": "customer",
                "entity_id": "C-001"}
    result = grade_answer(answer, expected)
    assert result["checks"]["entity_id"] is False
    assert result["correct"] is False


def test_entity_name_case_and_space_variation_matches() -> None:
    answer = {"status": "success", "value": " northwind ", "units": "customer",
              "period": "2026-03", "grain": "customer"}
    expected = {"expected_status": "answered", "value": "Northwind",
                "units": "customer", "period": "2026-03", "grain": "customer"}
    assert grade_answer(answer, expected)["checks"]["value"] is True


def test_unsupported_claim_still_surfaces() -> None:
    answer = {"status": "success", "value": 186, "units": "units",
              "period": "2026-03-31", "grain": "x",
              "claims": [{"text": "growth is accelerating", "supported": False}]}
    expected = {"expected_status": "answered", "value": 186, "units": "units",
                "period": "2026-03-31", "grain": "x"}
    result = grade_answer(answer, expected)
    assert result["outcome"] == "unsupported_claim"
    assert result["unsupported_claims"]
