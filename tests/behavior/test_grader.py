"""Offline tests for the behavior-CI grader."""

from __future__ import annotations

from .grader import grade


# ---------------------------------------------------------------------------
# exact
# ---------------------------------------------------------------------------

class TestExactComparator:
    def test_int_equal_passes(self) -> None:
        r = grade(42, 42, cmp="exact")
        assert r.passed
        assert r.reason == ""

    def test_int_unequal_fails(self) -> None:
        r = grade(42, 43, cmp="exact")
        assert not r.passed
        assert "42" in r.reason and "43" in r.reason

    def test_str_int_coerces_and_passes(self) -> None:
        r = grade("42", 42, cmp="exact")
        assert r.passed, r.reason

    def test_int_str_int_coerces_and_passes(self) -> None:
        r = grade(42, "42", cmp="exact")
        assert r.passed, r.reason

    def test_str_with_spaces_strips(self) -> None:
        r = grade("hello", "  hello\n", cmp="exact")
        assert r.passed, r.reason

    def test_str_unequal_fails(self) -> None:
        r = grade("hello", "world", cmp="exact")
        assert not r.passed

    def test_float_equal_to_int_passes(self) -> None:
        r = grade(3.0, 3, cmp="exact")
        assert r.passed, r.reason

    def test_none_actual_fails(self) -> None:
        r = grade(None, 42, cmp="exact")
        assert not r.passed
        assert "None" in r.reason or "no answer" in r.reason


# ---------------------------------------------------------------------------
# near
# ---------------------------------------------------------------------------

class TestNearComparator:
    def test_close_floats_pass(self) -> None:
        r = grade(3.1415926, 3.141592, cmp="near")
        assert r.passed, r.reason

    def test_far_floats_fail(self) -> None:
        r = grade(3.14, 6.28, cmp="near")
        assert not r.passed

    def test_string_float_coerces(self) -> None:
        r = grade("12.5", 12.5, cmp="near")
        assert r.passed, r.reason

    def test_non_numeric_fails(self) -> None:
        r = grade("foo", 12.5, cmp="near")
        assert not r.passed
        assert "non-numeric" in r.reason


# ---------------------------------------------------------------------------
# string
# ---------------------------------------------------------------------------

class TestStringComparator:
    def test_list_order_matters(self) -> None:
        r = grade([3, 1, 4], [3, 1, 4], cmp="string")
        assert r.passed, r.reason

    def test_list_different_order_fails(self) -> None:
        r = grade([1, 3, 4], [3, 1, 4], cmp="string")
        assert not r.passed

    def test_list_vs_tuple_fails(self) -> None:
        r = grade([1, 2], (1, 2), cmp="string")
        assert not r.passed

    def test_string_strip_passes(self) -> None:
        r = grade("  bob ", "bob", cmp="string")
        assert r.passed, r.reason


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------

class TestSetComparator:
    def test_same_elements_different_order_passes(self) -> None:
        r = grade([1, 2, 3], [3, 2, 1], cmp="set")
        assert r.passed, r.reason

    def test_missing_element_fails(self) -> None:
        r = grade([1, 2], [1, 2, 3], cmp="set")
        assert not r.passed
        assert "missing" in r.reason

    def test_extra_element_fails(self) -> None:
        r = grade([1, 2, 3, 4], [1, 2, 3], cmp="set")
        assert not r.passed
        assert "extra" in r.reason

    def test_unhashable_fails_gracefully(self) -> None:
        r = grade([[1, 2]], [[1, 2]], cmp="set")
        assert not r.passed
        assert "set-convertible" in r.reason


# ---------------------------------------------------------------------------
# unknown comparator
# ---------------------------------------------------------------------------

def test_unknown_comparator_fails_with_clear_message() -> None:
    r = grade(1, 1, cmp="bogus")
    assert not r.passed
    assert "unknown comparator" in r.reason
