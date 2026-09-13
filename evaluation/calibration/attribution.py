"""Attribute run-to-run disagreement to its cause.

A noise floor says *how much* an unchanged configuration disagrees with itself.
It does not say *why*, and the answer matters: noise caused by a serialization
bug or by a run that produced no value at all is fixable engineering, whereas
noise caused by the model reaching a different conclusion is not — it can only
be averaged down with more repetitions.

Reporting a single floor invites the assumption that all of it is model
nondeterminism. This module forces the split to be stated.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .trials import ANSWERED, NO_VALUE, UNSERIALIZABLE, Trial, TrialSet

__all__ = [
    "REASONING",
    "flip_causes",
    "mechanical_share",
]

# A question whose grade changed between identical runs while every repetition
# still produced a real, serializable value: the system genuinely disagreed
# with itself about the answer.
REASONING = "reasoning"


def _cause(trials: list[Trial]) -> str:
    """Name the cause of one question's disagreement across repetitions."""
    states = {t.answer_state for t in trials} - {ANSWERED}
    if not states:
        return REASONING
    return "+".join(sorted(states))


def flip_causes(trial_set: TrialSet, arm: str) -> Counter[str]:
    """Count, by cause, the questions whose grade flips between identical runs.

    Only questions that actually flip are counted; a question that is
    consistently wrong is a correctness problem, not a noise problem.
    """
    by_question: dict[str, list[Trial]] = defaultdict(list)
    for trial in sorted(trial_set.for_arm(arm), key=lambda t: t.rep):
        by_question[trial.question_id].append(trial)

    causes: Counter[str] = Counter()
    for trials in by_question.values():
        if len({t.correct for t in trials}) < 2:
            continue
        causes[_cause(trials)] += 1
    return causes


def mechanical_share(causes: Counter[str]) -> float:
    """Fraction of flips with a mechanical cause, in 0..1.

    This is the share that engineering can remove. The remainder needs more
    repetitions, a different model, or a tighter question.
    """
    total = sum(causes.values())
    if not total:
        return 0.0
    return (total - causes.get(REASONING, 0)) / total
