from __future__ import annotations

import math

import pytest

from fabric_rlm.experimental import (
    additive_decomposition,
    rate_decomposition,
)


def test_additive_decomposition_reconciles_segment_contributions() -> None:
    result = additive_decomposition(
        node_id="revenue-change",
        before={"north": 100.0, "south": 80.0, "closed": 20.0},
        after={"north": 130.0, "south": 70.0, "new": 15.0},
        seed=17,
    )

    assert result.values["before_total"] == 200.0
    assert result.values["after_total"] == 215.0
    assert result.values["observed_change"] == 15.0
    assert result.values["components"] == (
        {
            "segment": "closed",
            "before": 20.0,
            "after": 0.0,
            "contribution": -20.0,
        },
        {
            "segment": "new",
            "before": 0.0,
            "after": 15.0,
            "contribution": 15.0,
        },
        {
            "segment": "north",
            "before": 100.0,
            "after": 130.0,
            "contribution": 30.0,
        },
        {
            "segment": "south",
            "before": 80.0,
            "after": 70.0,
            "contribution": -10.0,
        },
    )
    assert result.diagnostics["reconciliation"]["passed"] is True
    assert result.diagnostics["reconciliation"]["residual"] == 0.0


def test_additive_decomposition_is_order_independent_and_scale_equivariant() -> None:
    first = additive_decomposition(
        node_id="ordered",
        before={"a": 10.0, "b": 20.0},
        after={"a": 12.0, "b": 16.0},
        seed=1,
    )
    reordered = additive_decomposition(
        node_id="reordered",
        before={"b": 20.0, "a": 10.0},
        after={"b": 16.0, "a": 12.0},
        seed=1,
    )
    scaled = additive_decomposition(
        node_id="scaled",
        before={"a": 100.0, "b": 200.0},
        after={"a": 120.0, "b": 160.0},
        seed=1,
    )

    assert first.values == reordered.values
    assert (
        first.diagnostics["input_fingerprint"]
        == reordered.diagnostics["input_fingerprint"]
    )
    changed = additive_decomposition(
        node_id="changed",
        before={"a": 10.0, "b": 20.0},
        after={"a": 12.0, "b": 17.0},
        seed=1,
    )
    assert (
        first.diagnostics["input_fingerprint"]
        != changed.diagnostics["input_fingerprint"]
    )
    assert scaled.values["observed_change"] == pytest.approx(
        first.values["observed_change"] * 10
    )
    assert tuple(
        component["contribution"] for component in scaled.values["components"]
    ) == pytest.approx(
        tuple(
            component["contribution"] * 10
            for component in first.values["components"]
        )
    )


def test_rate_decomposition_exactly_separates_numerator_and_denominator_effects() -> None:
    result = rate_decomposition(
        node_id="conversion-change",
        before_numerator=20.0,
        before_denominator=100.0,
        after_numerator=30.0,
        after_denominator=120.0,
        seed=23,
    )

    assert result.values["before_rate"] == pytest.approx(0.20)
    assert result.values["after_rate"] == pytest.approx(0.25)
    assert result.values["observed_change"] == pytest.approx(0.05)
    assert result.values["numerator_effect"] == pytest.approx(
        0.5 * ((30 / 100 - 20 / 100) + (30 / 120 - 20 / 120))
    )
    assert result.values["denominator_effect"] == pytest.approx(
        0.5 * ((20 / 120 - 20 / 100) + (30 / 120 - 30 / 100))
    )
    assert result.diagnostics["reconciliation"]["passed"] is True
    assert abs(result.diagnostics["reconciliation"]["residual"]) <= 1e-12


def test_rate_decomposition_reverses_effect_signs_when_periods_swap() -> None:
    forward = rate_decomposition(
        node_id="forward",
        before_numerator=20,
        before_denominator=100,
        after_numerator=30,
        after_denominator=120,
        seed=1,
    )
    reverse = rate_decomposition(
        node_id="reverse",
        before_numerator=30,
        before_denominator=120,
        after_numerator=20,
        after_denominator=100,
        seed=1,
    )

    for metric in (
        "observed_change",
        "numerator_effect",
        "denominator_effect",
    ):
        assert reverse.values[metric] == pytest.approx(-forward.values[metric])


def test_decomposition_fingerprints_normalize_signed_zero() -> None:
    positive_zero = additive_decomposition(
        node_id="positive-zero",
        before={"a": 0.0},
        after={"a": 1.0},
        seed=1,
    )
    negative_zero = additive_decomposition(
        node_id="negative-zero",
        before={"a": -0.0},
        after={"a": 1.0},
        seed=1,
    )

    assert (
        positive_zero.diagnostics["input_fingerprint"]
        == negative_zero.diagnostics["input_fingerprint"]
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "before": {"a": math.nan},
                "after": {"a": 1.0},
            },
            "before.a",
        ),
        (
            {
                "before": {"": 1.0},
                "after": {"a": 1.0},
            },
            "segment",
        ),
        (
            {
                "before": {"a": 1.0},
                "after": {"a": 2.0},
                "tolerance": 0,
            },
            "tolerance",
        ),
    ],
)
def test_additive_decomposition_rejects_invalid_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        additive_decomposition(node_id="invalid", seed=1, **kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("before_denominator", 0),
        ("after_denominator", 0),
        ("before_denominator", -1),
        ("after_denominator", math.inf),
    ],
)
def test_rate_decomposition_rejects_invalid_denominators(
    field: str,
    value: float,
) -> None:
    kwargs = {
        "before_numerator": 20,
        "before_denominator": 100,
        "after_numerator": 30,
        "after_denominator": 120,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        rate_decomposition(node_id="invalid", seed=1, **kwargs)
