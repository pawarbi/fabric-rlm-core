"""Which word decides the direction a "from A to B" sentence claims.

Found in Fabric: the shipped contract-comparison notebook was sent back twice,
and ran out of turns, over "The contract version number increases from 2.0 to
3.1". The screen read the noun "contract" as the verb "contracted" and took it
over "increases" because it came first. A screen that rejects a correct
sentence costs turns and can fail a run, so doubtful wording must pass.
"""

from __future__ import annotations

import pytest

from fabric_rlm.analytical_integrity import check_directional_claims, parse_directional_claims


@pytest.mark.parametrize(
    "sentence",
    [
        "ARR fell from $3.9M to $4.2M this quarter.",
        "Churn rose from 5% to 3%.",
        "Margin improved sharply from 40% to 35%.",
        "Revenue contracted from 80 to 95.",
        "Growth rate rose from 5% to 3%.",
        "Headcount moved from 100 to 90, an increase.",
        "Revenue declined from 926,400.00 to 926,400.00.",
    ],
)
def test_a_wrong_direction_is_still_flagged(sentence):
    assert check_directional_claims(sentence), sentence


@pytest.mark.parametrize(
    "sentence",
    [
        # the noun next to a verb of the other direction: the verb beside "from" decides
        "The contract version number increases from 2.0 to 3.1.",
        "Contract value increased from 5M to 8M.",
        "Growth rate fell from 5% to 3%.",
        "Dropout rate rose from 5% to 7%.",
        "Lower tier customers grew from 100 to 120.",
        "The higher priced plan fell from 80 to 60 seats.",
        "Fall enrollment grew from 100 to 120.",
        "Gross margin improvement slowed and margin fell from 42% to 40%.",
        # no verb at all: nothing is claimed
        "Total contract value from 5M to 8M.",
        # both directions and neither beside "from": too doubtful to reject
        "Revenue declined in the growth segment from 5 to 3.",
        "Headcount moved from 100 to 90, a decline after earlier growth.",
        # unchanged behavior
        "ARR grew from 100 to 120.",
        "ARR moved from 100 to 90.",
        "ARR dropped from 5,000,000.00 USD in 2025/Q4 to 3,500,000.00 USD in 2026/Q2, a decrease of 1,500,000.00 USD.",
    ],
)
def test_a_correct_or_doubtful_sentence_is_not_flagged(sentence):
    assert check_directional_claims(sentence) == [], sentence


def test_the_word_beside_from_is_the_claim():
    claim = parse_directional_claims("Growth rate fell sharply from 5% to 3%.")[0]
    assert (claim.claimed, claim.actual) == ("decrease", "decrease")
