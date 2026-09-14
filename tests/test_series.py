"""The series arithmetic against planted truths: a known season, a known step, a known slope, and a flat series that must say nothing."""

from __future__ import annotations

import math
from types import SimpleNamespace

from fabric_rlm.series import Month, analyse, change_points, comovement, decompose, moving_average, usable_months, yoy


def _months(values, *, start=(2022, 1), rows=100):
    year, month = start
    out = []
    for v in values:
        out.append(SimpleNamespace(year=year, month=month, value=float(v), rows=rows))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


SEASON = [0.80, 0.85, 0.95, 1.05, 1.20, 1.30, 1.25, 1.10, 1.00, 0.90, 0.80, 0.80]


def test_a_planted_season_and_slope_come_back_as_planted():
    values = [1000 * (1 + 0.10 * i / 12) * SEASON[i % 12] for i in range(48)]  # +10% a year, a June peak, a January trough
    story = analyse(_months(values), name="revenue")
    d = story.decomposition
    assert d is not None and d.multiplicative and d.explained > 0.9
    assert max(d.seasonal, key=lambda k: d.seasonal[k]) == 6 and min(d.seasonal, key=lambda k: d.seasonal[k]) in (1, 11, 12)
    assert abs(d.seasonal[6] - 1.30 / (sum(SEASON) / 12)) < 0.03
    assert story.trend_per_year is not None and 0.07 <= story.trend_per_year <= 0.13
    assert story.window == 12 and story.average[23] is not None and story.average[0] is None
    assert any("Seasonality explains" in s and "June runs" in s for s in story.sentences)
    assert any("The trend runs +" in s and "a year" in s for s in story.sentences)
    assert not any("new level" in s for s in story.sentences)  # a smooth slope is not a step


def test_a_planted_step_is_found_at_its_month_and_a_flat_series_says_nothing():
    values = [1000.0] * 14 + [1300.0] * 10
    story = analyse(_months(values), name="orders")
    assert [p.start for p in story.shifts] == ["2023-03-01"] and story.shifts[0].before == 1000.0 and story.shifts[0].after == 1300.0
    assert story.sentences[0] == "Orders has been running at a new level since Mar 2023: 1,000 to 1,300 a month (+30%)."
    flat = analyse(_months([500.0 + (i % 2) for i in range(30)]), name="orders")
    assert flat.shifts == () and flat.sentences == () and (flat.trend_per_year is None or abs(flat.trend_per_year) < 0.03)


def test_year_over_year_and_the_moving_average_are_plain_arithmetic():
    months = usable_months(_months([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 110, 120, 130]))
    changes = dict((m.label, v) for m, v in yoy(months))
    assert changes["Jan 2022"] is None and changes["Jan 2023"] == 0.10 and abs(changes["Mar 2023"] - 0.30) < 1e-9
    assert moving_average([1, 2, 3, 4, 5], 3) == [None, 2.0, 3.0, 4.0, None]
    even = moving_average(list(range(1, 15)), 12)  # the 2 x 12 average sits on a month, not between two
    assert even[6] == 7.0 and even[0] is None and even[13] is None


def test_a_trailing_stub_month_is_dropped_before_any_fit():
    points = _months([100.0] * 26)
    points[-1] = SimpleNamespace(year=points[-1].year, month=points[-1].month, value=3.0, rows=2)  # the last month has not arrived
    months = usable_months(points)
    assert len(months) == 25 and all(m.value == 100.0 for m in months)
    assert decompose(months) is not None and change_points(months) == []


def test_comovement_names_the_pair_and_who_led():
    base = [1000 * (1 + 0.02 * math.sin(i / 2.0)) * SEASON[i % 12] for i in range(40)]
    a = _months(base)
    b = _months([v * 0.5 for v in base])  # the same shape at half the size
    c = _months([1000.0] * 40)  # nothing to correlate with
    lines = comovement({"revenue": a, "orders": b, "flat": c})
    assert lines and lines[0].startswith("Revenue and orders moved together year over year") and "(r = 1.00)" in lines[0]
    assert lines[-1].startswith("These are associations")
    assert not any("flat" in line for line in lines[:-1])


def test_month_labels_and_indexes():
    m = Month(2024, 3, 12.5, 40)
    assert m.start == "2024-03-01" and m.label == "Mar 2024" and m.index == 2024 * 12 + 2
