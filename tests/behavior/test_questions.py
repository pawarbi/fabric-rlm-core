"""Tests for the pinned behavior-CI question set."""

from __future__ import annotations

from .questions import QUESTIONS, get_question, questions_sha256


def test_exactly_five_questions() -> None:
    assert len(QUESTIONS) == 5


def test_unique_qids() -> None:
    qids = [q.qid for q in QUESTIONS]
    assert len(qids) == len(set(qids))


def test_categories_match_locked_scope() -> None:
    by_cat: dict[str, int] = {}
    for q in QUESTIONS:
        by_cat[q.category] = by_cat.get(q.category, 0) + 1
    # Locked scope: 2 compute + 2 messy + 1 selfcorrect.
    assert by_cat == {"compute": 2, "messy": 2, "selfcorrect": 1}, by_cat


def test_get_question_returns_matching() -> None:
    q = get_question("C1_sum_squares_mod")
    assert q.qid == "C1_sum_squares_mod"
    assert q.category == "compute"


def test_get_question_raises_on_unknown() -> None:
    import pytest

    with pytest.raises(KeyError):
        get_question("NONESUCH")


def test_questions_are_reproducible_across_imports() -> None:
    # Re-import the module, rebuild the question list, compare expected values.
    import importlib

    from . import questions as mod

    importlib.reload(mod)
    rebuilt = mod.QUESTIONS
    for a, b in zip(QUESTIONS, rebuilt):
        assert a.qid == b.qid
        assert a.expected == b.expected, f"non-deterministic ground truth for {a.qid}"


def test_questions_sha256_is_stable() -> None:
    h1 = questions_sha256()
    h2 = questions_sha256()
    assert h1 == h2
    assert len(h1) == 64  # hex SHA-256


def test_every_question_has_nonempty_task_and_expected() -> None:
    for q in QUESTIONS:
        assert q.task.strip(), f"{q.qid}: empty task"
        assert q.expected is not None, f"{q.qid}: expected is None"
